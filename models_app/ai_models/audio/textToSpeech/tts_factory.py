"""
Factory for Text-to-Speech services.

This module provides a factory class for creating and managing different TTS services,
offering a unified interface for text-to-speech conversions regardless of the underlying engine.
"""

import logging
import os
import tempfile
import functools
from typing import Dict, Optional, List, Union, Any, Tuple, BinaryIO, Callable, Generator
from enum import Enum

# Import base TTS service 
from .base_tts_service import BaseTTSService, handle_tts_errors, synthesis_operation, handle_audio_processing_errors

# Import service implementations
from .spark_tts import SparkTTSService
from .coqui_tts import CoquiTTSService
from .mozilla_tts import MozillaTTSService

# Import audio processing utilities
from models_app.ai_models.utils.audio.processing import (
    convert_audio_format,
    normalize_audio,
    trim_silence,
    assess_audio_quality,
    get_audio_duration,
    change_speech_rate,
    change_pitch,
    mix_audio_files,
    generate_audio_chunks
)

# Import text processing utilities
from models_app.ai_models.utils.audio.text_processing import (
    preprocess_text_for_tts,
    is_ssml,
    validate_ssml,
    detect_language,
    segment_long_text,
    generate_ssml
)

# Import caching system
from models_app.ai_models.utils.audio.caching import (
    get_tts_cache
)

# Import common utilities for error handling
from models_app.ai_models.utils.common.errors import (
    AudioModelError,
    TTSError,
    AudioProcessingError
)

logger = logging.getLogger(__name__)


class TTSEngine(str, Enum):
    """Enum representing available TTS engines."""
    SPARK = "spark"
    COQUI = "coqui"
    MOZILLA = "mozilla"


class TTSFactory:
    """
    Factory class for Text-to-Speech services.
    
    This class provides a unified interface to different TTS engines and allows
    for easy switching between them.
    """
    
    def __init__(self, default_engine: Union[TTSEngine, str] = TTSEngine.SPARK, use_cache: bool = True):
        """
        Initialize the TTS factory.
        
        Args:
            default_engine: The default TTS engine to use
            use_cache: Whether to use the TTS cache
        """
        self.default_engine = TTSEngine(default_engine) if isinstance(default_engine, str) else default_engine
        self._services: Dict[TTSEngine, BaseTTSService] = {}
        self.use_cache = use_cache
        
    def get_service(self, engine: Optional[Union[TTSEngine, str]] = None) -> BaseTTSService:
        """
        Get or create a TTS service for the specified engine.
        
        Args:
            engine: The TTS engine to use. If None, uses the default engine.
            
        Returns:
            A TTS service instance
            
        Raises:
            ValueError: If an invalid engine is specified
        """
        engine = TTSEngine(engine) if isinstance(engine, str) and engine else self.default_engine
        
        # Create service if it doesn't exist
        if engine not in self._services:
            if engine == TTSEngine.SPARK:
                self._services[engine] = SparkTTSService()
            elif engine == TTSEngine.COQUI:
                self._services[engine] = CoquiTTSService()
            elif engine == TTSEngine.MOZILLA:
                self._services[engine] = MozillaTTSService()
            else:
                raise ValueError(f"Invalid TTS engine: {engine}")
                
        return self._services[engine]
    
    def synthesize_speech(self, text: str, output_path: Optional[str] = None, 
                        engine: Optional[Union[TTSEngine, str]] = None, 
                        apply_post_processing: bool = True,
                        normalize_volume: bool = True,
                        remove_silence: bool = True,
                        target_format: str = "wav",
                        sample_rate: int = 24000,
                        speech_rate: float = 1.0,
                        pitch_adjustment: float = 0.0,
                        preprocess_text: bool = True,
                        use_cache: Optional[bool] = None,
                        **kwargs) -> Optional[str]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the audio file. If None, generates a temporary file.
            engine: The TTS engine to use. If None, uses the default engine.
            apply_post_processing: Whether to apply post-processing to the audio
            normalize_volume: Whether to normalize the audio volume
            remove_silence: Whether to trim silence from the audio
            target_format: Target audio format (wav, mp3, etc.)
            sample_rate: Target sample rate
            speech_rate: Rate of speech (0.5-2.0, where 1.0 is normal)
            pitch_adjustment: Pitch adjustment in semitones (-10 to 10)
            preprocess_text: Whether to preprocess the text for better pronunciation
            use_cache: Whether to use the TTS cache, overrides instance setting
            **kwargs: Additional parameters to pass to the specific TTS service
            
        Returns:
            Path to the generated audio file or None if synthesis failed
        """
        # Determine whether to use cache
        should_use_cache = self.use_cache if use_cache is None else use_cache
        
        # Get engine instance
        engine_enum = TTSEngine(engine) if isinstance(engine, str) and engine else self.default_engine
        engine_name = engine_enum.value
        
        # Get voice ID if provided
        voice_id = kwargs.get('voice_id')
        
        # Check if we have a reference audio for voice cloning
        # If so, we can't use cache as voice cloning is dynamic
        reference_audio = kwargs.get('reference_audio')
        should_use_cache = should_use_cache and reference_audio is None
        
        # Preprocess text if needed (unless it's SSML)
        original_text = text
        if preprocess_text and not is_ssml(text):
            # Detect language if not specified
            language = kwargs.get('language', detect_language(text))
            
            # Preprocess text for better pronunciation
            text = preprocess_text_for_tts(text, language=language)
        elif is_ssml(text):
            # Validate SSML and fix if needed
            valid, fixed_ssml, error = validate_ssml(text)
            if not valid and fixed_ssml:
                text = fixed_ssml
                logger.info(f"Fixed SSML: {error}")
            elif not valid:
                logger.warning(f"Invalid SSML: {error}")
        
        # Check cache first if enabled
        if should_use_cache:
            cache = get_tts_cache()
            
            # Create options dict for cache key
            cache_options = {
                'format': target_format,
                'sample_rate': sample_rate,
                'speech_rate': speech_rate,
                'pitch_adjustment': pitch_adjustment
            }
            
            # Add relevant kwargs to cache options
            for key in ['language', 'speaker', 'style', 'emotion']:
                if key in kwargs:
                    cache_options[key] = kwargs[key]
            
            # Check if we have a cached version
            cached_path = cache.get_cached_audio(text, engine_name, voice_id, cache_options)
            if cached_path:
                logger.info(f"Using cached TTS output for: {text[:50]}...")
                
                # If output path is specified, copy the cached file
                if output_path:
                    import shutil
                    shutil.copy2(cached_path, output_path)
                    return output_path
                
                return cached_path
        
        # Get the appropriate service
        service = self.get_service(engine_enum)
        
        # For long text, consider splitting into segments
        if len(text) > 2000 and not is_ssml(text):
            return self._synthesize_long_text(
                text, output_path, engine_enum, 
                apply_post_processing, normalize_volume, remove_silence,
                target_format, sample_rate, speech_rate, pitch_adjustment,
                **kwargs
            )
        
        # Generate raw speech without post-processing
        raw_output_path = service.synthesize_speech(text, output_path, **kwargs)
        
        if not raw_output_path or not os.path.exists(raw_output_path):
            logger.error(f"Speech synthesis failed for text: {text[:50]}...")
            return None
        
        # If no post-processing needed and no rate/pitch adjustment, return the raw output
        if not apply_post_processing and speech_rate == 1.0 and pitch_adjustment == 0.0:
            # Cache if enabled
            if should_use_cache:
                cache_options = {
                    'format': os.path.splitext(raw_output_path)[1].lstrip('.'),
                    'speech_rate': 1.0,
                    'pitch_adjustment': 0.0
                }
                for key in ['language', 'speaker', 'style', 'emotion']:
                    if key in kwargs:
                        cache_options[key] = kwargs[key]
                        
                cache.cache_audio(text, raw_output_path, engine_name, voice_id, cache_options)
            
            return raw_output_path
        
        # Apply post-processing steps
        processed_path = raw_output_path
        
        try:
            # Apply speech rate adjustment if needed
            if speech_rate != 1.0:
                processed_path = change_speech_rate(processed_path, speech_rate)
                logger.debug(f"Adjusted speech rate: {processed_path}")
            
            # Apply pitch adjustment if needed
            if pitch_adjustment != 0.0:
                processed_path = change_pitch(processed_path, pitch_adjustment)
                logger.debug(f"Adjusted pitch: {processed_path}")
            
            # Normalize volume if requested
            if normalize_volume:
                processed_path = normalize_audio(processed_path)
                logger.debug(f"Normalized audio volume: {processed_path}")
            
            # Trim silence if requested
            if remove_silence:
                processed_path = trim_silence(processed_path)
                logger.debug(f"Trimmed silence: {processed_path}")
            
            # Convert to target format if different from current format
            current_format = os.path.splitext(processed_path)[1].lstrip('.')
            if current_format != target_format:
                processed_path = convert_audio_format(processed_path, target_format, sample_rate)
                logger.debug(f"Converted to {target_format}: {processed_path}")
            
            # Check quality of the final output
            quality = self.assess_speech_quality(processed_path)
            if not quality.get('is_suitable_for_tts', True):
                logger.warning(f"Generated speech might have quality issues: {quality}")
            
            # Cache the processed output if enabled
            if should_use_cache:
                cache_options = {
                    'format': target_format,
                    'sample_rate': sample_rate,
                    'speech_rate': speech_rate,
                    'pitch_adjustment': pitch_adjustment
                }
                for key in ['language', 'speaker', 'style', 'emotion']:
                    if key in kwargs:
                        cache_options[key] = kwargs[key]
                
                cache.cache_audio(text, processed_path, engine_name, voice_id, cache_options, {
                    'quality': quality,
                    'original_text': original_text
                })
            
            return processed_path
            
        except Exception as e:
            logger.error(f"Error in audio post-processing: {str(e)}")
            # Return the original output if post-processing fails
            return raw_output_path
    
    def _synthesize_long_text(self, text: str, output_path: Optional[str] = None, 
                            engine: Optional[Union[TTSEngine, str]] = None,
                            apply_post_processing: bool = True,
                            normalize_volume: bool = True,
                            remove_silence: bool = True,
                            target_format: str = "wav",
                            sample_rate: int = 24000,
                            speech_rate: float = 1.0,
                            pitch_adjustment: float = 0.0,
                            **kwargs) -> Optional[str]:
        """
        Synthesize long text by splitting it into segments and combining the results.
        
        Args:
            (Same as synthesize_speech)
            
        Returns:
            Path to the combined audio file
        """
        # Split text into segments
        segments = segment_long_text(text, max_length=1000)
        logger.info(f"Split long text into {len(segments)} segments")
        
        # Synthesize each segment
        segment_paths = []
        for i, segment in enumerate(segments):
            logger.info(f"Synthesizing segment {i+1}/{len(segments)}")
            
            # Create temporary path for this segment
            with tempfile.NamedTemporaryFile(suffix=f'.{target_format}', delete=False) as tmp:
                segment_output_path = tmp.name
            
            # Synthesize segment
            segment_path = self.synthesize_speech(
                segment, 
                segment_output_path,
                engine=engine,
                apply_post_processing=apply_post_processing,
                normalize_volume=normalize_volume,
                remove_silence=remove_silence,
                target_format=target_format,
                sample_rate=sample_rate,
                speech_rate=speech_rate,
                pitch_adjustment=pitch_adjustment,
                use_cache=True,  # Always try to use cache for segments
                **kwargs
            )
            
            if segment_path:
                segment_paths.append(segment_path)
            else:
                logger.error(f"Failed to synthesize segment {i+1}")
        
        if not segment_paths:
            logger.error("Failed to synthesize any segments")
            return None
        
        # Combine the segments
        if len(segment_paths) == 1:
            # Only one segment was successfully synthesized
            if output_path:
                import shutil
                shutil.copy2(segment_paths[0], output_path)
                os.unlink(segment_paths[0])
                return output_path
            return segment_paths[0]
        
        # Mix the segments with equal weight
        if output_path is None:
            # Create a temporary output path
            with tempfile.NamedTemporaryFile(suffix=f'.{target_format}', delete=False) as tmp:
                output_path = tmp.name
        
        # Mix the segments
        combined_path = mix_audio_files(segment_paths)
        
        # Convert to target format if needed
        current_format = os.path.splitext(combined_path)[1].lstrip('.')
        if current_format != target_format:
            combined_path = convert_audio_format(combined_path, target_format, sample_rate)
        
        # Copy to output path if different
        if output_path != combined_path:
            import shutil
            shutil.copy2(combined_path, output_path)
            os.unlink(combined_path)
        
        # Clean up segment files
        for path in segment_paths:
            try:
                os.unlink(path)
            except:
                pass
        
        return output_path
    
    def get_speech_to_audio_buffer(self, text: str, engine: Optional[Union[TTSEngine, str]] = None, 
                                apply_post_processing: bool = True,
                                speech_rate: float = 1.0,
                                pitch_adjustment: float = 0.0,
                                use_cache: Optional[bool] = None,
                                **kwargs) -> Optional[bytes]:
        """
        Convert text to speech and return as an audio buffer.
        
        Args:
            text: Text to convert to speech
            engine: The TTS engine to use. If None, uses the default engine.
            apply_post_processing: Whether to apply post-processing
            speech_rate: Rate of speech (0.5-2.0, where 1.0 is normal)
            pitch_adjustment: Pitch adjustment in semitones (-10 to 10)
            use_cache: Whether to use the TTS cache, overrides instance setting
            **kwargs: Additional parameters to pass to the specific TTS service
            
        Returns:
            Audio buffer as bytes or None if conversion failed
        """
        # Generate a temporary file first
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Synthesize to temporary file
            processed_path = self.synthesize_speech(
                text, 
                temp_path, 
                engine, 
                apply_post_processing=apply_post_processing,
                speech_rate=speech_rate,
                pitch_adjustment=pitch_adjustment,
                use_cache=use_cache,
                **kwargs
            )
            
            if not processed_path:
                return None
            
            # Read the file into a buffer
            with open(processed_path, 'rb') as audio_file:
                audio_buffer = audio_file.read()
            
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            # Clean up the processed file if different from temp_path
            if processed_path != temp_path and os.path.exists(processed_path):
                os.unlink(processed_path)
                
            return audio_buffer
            
        except Exception as e:
            logger.error(f"Error converting speech to audio buffer: {str(e)}")
            
            # Clean up temporary files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
            return None
    
    def stream_synthesized_speech(self, text: str, chunk_size_ms: int = 1000, 
                               engine: Optional[Union[TTSEngine, str]] = None,
                               apply_post_processing: bool = True,
                               **kwargs):
        """
        Stream synthesized speech in chunks.
        
        Args:
            text: Text to synthesize
            chunk_size_ms: Size of each chunk in milliseconds
            engine: The TTS engine to use
            apply_post_processing: Whether to apply post-processing
            **kwargs: Additional parameters for synthesis
            
        Yields:
            Audio chunks as bytes
        """
        # Synthesize speech to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            processed_path = self.synthesize_speech(
                text, 
                temp_path, 
                engine, 
                apply_post_processing=apply_post_processing,
                **kwargs
            )
            
            if not processed_path:
                return
            
            # Stream chunks from the file
            for chunk in generate_audio_chunks(processed_path, chunk_size_ms):
                yield chunk
            
            # Clean up temporary files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            if processed_path != temp_path and os.path.exists(processed_path):
                os.unlink(processed_path)
                
        except Exception as e:
            logger.error(f"Error streaming synthesized speech: {str(e)}")
            
            # Clean up temporary files
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def assess_speech_quality(self, audio_path: str) -> Dict[str, Any]:
        """
        Assess the quality of generated speech.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_metrics = assess_audio_quality(audio_path)
            
            # Add TTS-specific metrics
            duration = get_audio_duration(audio_path)
            quality_metrics['speech_rate'] = len(quality_metrics.get('text', '')) / max(duration, 0.1)
            
            return quality_metrics
        except Exception as e:
            logger.error(f"Error assessing speech quality: {str(e)}")
            return {
                'error': str(e),
                'is_suitable_for_tts': False
            }
    
    def optimize_audio_for_playback(self, audio_path: str, target_format: str = "mp3", 
                                  normalize: bool = True) -> Optional[str]:
        """
        Optimize an audio file for web playback.
        
        Args:
            audio_path: Path to the audio file
            target_format: Target format for web playback (mp3, ogg, etc.)
            normalize: Whether to normalize the audio volume
            
        Returns:
            Path to the optimized audio file
        """
        try:
            processed_path = audio_path
            
            # Normalize if requested
            if normalize:
                processed_path = normalize_audio(processed_path)
            
            # Convert to web-friendly format
            processed_path = convert_audio_format(processed_path, target_format)
            
            return processed_path
        except Exception as e:
            logger.error(f"Error optimizing audio for playback: {str(e)}")
            return None
    
    def generate_ssml(self, text: str, voice: Optional[str] = None, 
                    rate: Optional[float] = None, pitch: Optional[float] = None,
                    volume: Optional[float] = None, language: Optional[str] = None) -> str:
        """
        Generate SSML markup from plain text.
        
        Args:
            text: Plain text to convert to SSML
            voice: Voice name/ID to use
            rate: Speech rate (0.5-2.0, where 1.0 is normal)
            pitch: Voice pitch (-10 to 10, where 0 is normal)
            volume: Volume (0-100, where 100 is normal)
            language: Language code
            
        Returns:
            Text with SSML markup
        """
        return generate_ssml(text, voice, rate, pitch, volume, language)
    
    def batch_synthesize_speech(self, texts: List[str], output_dir: str, 
                               engine: Optional[Union[TTSEngine, str]] = None,
                               apply_post_processing: bool = True, **kwargs) -> List[str]:
        """
        Synthesize multiple texts in batch.
        
        Args:
            texts: List of texts to synthesize
            output_dir: Directory to save output files
            engine: TTS engine to use
            apply_post_processing: Whether to apply post-processing
            **kwargs: Additional parameters for synthesis
            
        Returns:
            List of paths to generated audio files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each text
        output_paths = []
        for i, text in enumerate(texts):
            # Create output path
            output_path = os.path.join(output_dir, f"speech_{i+1}.wav")
            
            # Synthesize speech
            path = self.synthesize_speech(
                text, 
                output_path,
                engine=engine,
                apply_post_processing=apply_post_processing,
                **kwargs
            )
            
            if path:
                output_paths.append(path)
            else:
                logger.error(f"Failed to synthesize text {i+1}: {text[:50]}...")
        
        return output_paths
    
    def list_available_engines(self) -> List[TTSEngine]:
        """
        List all available TTS engines.
        
        Returns:
            List of available TTS engines
        """
        return [engine for engine in TTSEngine]
    
    def list_available_voices(self, engine: Optional[Union[TTSEngine, str]] = None) -> List[Dict[str, str]]:
        """
        List available voices/speakers for the specified engine.
        
        Args:
            engine: The TTS engine to use. If None, uses the default engine.
            
        Returns:
            List of available voice/speaker dictionaries with name and ID
        """
        service = self.get_service(engine)
        
        if isinstance(service, SparkTTSService):
            voices = service.list_voices()
            return [{'id': voice, 'name': voice} for voice in voices]
        elif isinstance(service, CoquiTTSService):
            speakers = service.list_speakers()
            return [{'id': speaker, 'name': speaker} for speaker in speakers]
        elif isinstance(service, MozillaTTSService):
            speakers = service.list_speakers()
            return [{'id': speaker, 'name': speaker} for speaker in speakers]
        
        return []
    
    def list_available_languages(self, engine: Optional[Union[TTSEngine, str]] = None) -> List[Dict[str, str]]:
        """
        List available languages for the specified engine.
        
        Args:
            engine: The TTS engine to use. If None, uses the default engine.
            
        Returns:
            List of available language dictionaries with code and name
        """
        service = self.get_service(engine)
        
        if hasattr(service, "list_languages"):
            languages = service.list_languages()
            
            # Create language name mapping
            language_names = {
                'en': 'English',
                'fr': 'French',
                'de': 'German',
                'es': 'Spanish',
                'it': 'Italian',
                'ja': 'Japanese',
                'ko': 'Korean',
                'pt': 'Portuguese',
                'ru': 'Russian',
                'zh': 'Chinese',
                # Add more as needed
            }
            
            # Create dictionaries with name if available
            return [
                {'code': lang, 'name': language_names.get(lang, lang)}
                for lang in languages
            ]
        
        return [] 