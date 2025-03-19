"""
Factory for Speech-to-Text services.

This module provides a factory class for creating and managing different STT services,
offering a unified interface for transcribing speech regardless of the underlying engine.
"""

import logging
import os
import tempfile
import functools
from typing import Dict, Optional, List, Union, Any, Tuple, BinaryIO, Callable, Generator
from enum import Enum

# Import common utilities for error handling
from models_app.ai_models.utils.common.errors import (
    AudioModelError,
    STTError,
    AudioProcessingError
)

# Import base STT service and decorators
from models_app.ai_models.audio.speechToText.base_stt_service import (
    BaseSTTService,
    handle_stt_errors,
    handle_audio_processing_errors,
    transcription_operation
)

# Import handler decorators
from error_handlers.common_handlers import (
    handle_errors,
    measure_time,
    retry
)

# Import audio processing utilities
from models_app.ai_models.utils.audio.processing import (
    convert_audio_format,
    normalize_audio,
    convert_to_mono,
    detect_speech_segments_with_transcription_timestamps,
    remove_background_noise,
    extract_audio_from_video,
    assess_audio_quality,
    trim_silence,
    generate_audio_chunks,
    get_audio_duration
)

# Import text processing utilities
from models_app.ai_models.utils.audio.text_processing import (
    extract_spoken_phrases,
    detect_language,
    detect_language_parts,
    verify_text_matches_audio,
    normalize_text,
    expand_abbreviations,
    normalize_numbers,
    normalize_dates
)

# Import caching system
from models_app.ai_models.utils.audio.caching import (
    get_stt_cache
)

logger = logging.getLogger(__name__)

# Define custom error handling decorators for STT
def handle_stt_errors(func: Callable) -> Callable:
    """Decorator to handle STT errors with proper logging and formatting."""
    @functools.wraps(func)
    @handle_errors(error_type=STTError)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"STT error in {func.__name__}: {str(e)}")
            if not isinstance(e, STTError):
                raise STTError(
                    message=f"Error in speech-to-text conversion: {str(e)}",
                    details={"original_error": str(e), "function": func.__name__}
                ) from e
            raise
    return wrapper

def handle_audio_processing_errors(func: Callable) -> Callable:
    """Decorator to handle audio processing errors with retry logic."""
    @functools.wraps(func)
    @retry(max_attempts=2, exceptions=(AudioProcessingError,))
    @handle_errors(error_type=AudioProcessingError)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if not isinstance(e, AudioProcessingError):
                logger.error(f"Audio processing error in {func.__name__}: {str(e)}")
                raise AudioProcessingError(
                    message=f"Error processing audio data: {str(e)}",
                    details={"original_error": str(e), "function": func.__name__}
                ) from e
            raise
    return wrapper


class STTEngine(str, Enum):
    """Enum representing available STT engines."""
    WHISPER_INSANELY_FAST = "whisper_insanely_fast"
    WHISPER_FASTER = "whisper_faster"
    WHISPER_X = "whisper_x"


class STTFactory:
    """
    Factory class for Speech-to-Text services.
    
    This class provides a unified interface to different STT engines and allows
    for easy switching between them.
    """
    
    def __init__(self, default_engine: Union[STTEngine, str] = STTEngine.WHISPER_INSANELY_FAST, 
                use_cache: bool = True, 
                cache_models: bool = True):
        """
        Initialize the STT factory.
        
        Args:
            default_engine: The default STT engine to use
            use_cache: Whether to use the STT cache for transcription results
            cache_models: Whether to cache loaded models in memory
        """
        self.default_engine = STTEngine(default_engine) if isinstance(default_engine, str) else default_engine
        self._services: Dict[STTEngine, BaseSTTService] = {}
        self.use_cache = use_cache
        self.cache_models = cache_models
    
    @handle_stt_errors
    def get_service(self, engine: Optional[Union[STTEngine, str]] = None, model_size: str = "base") -> BaseSTTService:
        """
        Get or create an STT service for the specified engine.
        
        Args:
            engine: The STT engine to use. If None, uses the default engine.
            model_size: The model size to use (tiny, base, small, medium, large-v2)
            
        Returns:
            An STT service instance
            
        Raises:
            ValueError: If an invalid engine is specified
        """
        engine = STTEngine(engine) if isinstance(engine, str) and engine else self.default_engine
        
        # Create service if it doesn't exist
        if engine not in self._services:
            if engine == STTEngine.WHISPER_INSANELY_FAST:
                from .whisper_insanely_fast.service import WhisperInsanelyFastService
                self._services[engine] = WhisperInsanelyFastService(model_size=model_size)
            elif engine == STTEngine.WHISPER_FASTER:
                from .whisper_faster.service import WhisperFasterService
                self._services[engine] = WhisperFasterService(model_size=model_size)
            elif engine == STTEngine.WHISPER_X:
                from .whisper_x.service import WhisperXService
                self._services[engine] = WhisperXService(model_size=model_size)
            else:
                raise ValueError(f"Invalid STT engine: {engine}")
                
        return self._services[engine]
    
    @transcription_operation
    def transcribe_audio(self, audio_path: str, 
                        engine: Optional[Union[STTEngine, str]] = None,
                        language: Optional[str] = None, 
                        apply_preprocessing: bool = True,
                        remove_noise: bool = False,
                        detect_speakers: bool = False,
                        extract_phrases: bool = True,
                        use_cache: Optional[bool] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to the audio file
            engine: The STT engine to use. If None, uses the default engine.
            language: Language code to use for transcription. If None, auto-detects.
            apply_preprocessing: Whether to apply preprocessing to the audio
            remove_noise: Whether to remove background noise
            detect_speakers: Whether to detect different speakers
            extract_phrases: Whether to extract individual phrases
            use_cache: Whether to use the STT cache, overrides instance setting
            **kwargs: Additional parameters to pass to the specific STT service
            
        Returns:
            Dictionary with transcription results
        """
        # Determine whether to use cache
        should_use_cache = self.use_cache if use_cache is None else use_cache
        
        # Get engine instance
        engine_enum = STTEngine(engine) if isinstance(engine, str) and engine else self.default_engine
        engine_name = engine_enum.value
        
        # Get model size if provided
        model_size = kwargs.get('model_size', 'base')
        
        # Handle file formats - check if it's a video
        if os.path.splitext(audio_path)[1].lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            logger.info(f"Extracting audio from video file: {audio_path}")
            audio_path = extract_audio_from_video(audio_path)
        
        # Check cache first if enabled
        if should_use_cache:
            cache = get_stt_cache()
            
            # Create options dict for cache key
            cache_options = {
                'language': language,
                'detect_speakers': detect_speakers,
                'extract_phrases': extract_phrases
            }
            
            # Add relevant kwargs to cache options
            for key in ['model_size', 'word_timestamps', 'initial_prompt']:
                if key in kwargs:
                    cache_options[key] = kwargs[key]
            
            # Check if we have a cached version
            cached_result = cache.get_cached_transcription(audio_path, engine_name, cache_options)
            if cached_result:
                logger.info(f"Using cached STT result for: {audio_path}")
                return cached_result
        
        # Preprocess audio if needed
        if apply_preprocessing:
            processed_path = self._preprocess_audio(audio_path, remove_noise)
        else:
            processed_path = audio_path
        
        # Log audio information for debugging
        duration = get_audio_duration(processed_path)
        logger.debug(f"Processing audio file with duration: {duration:.2f} seconds")
        
        # Get the appropriate service
        service = self.get_service(engine_enum, model_size)
        
        # Transcribe audio
        try:
            transcription = service.transcribe(
                processed_path, 
                language=language,
                **kwargs
            )
            
            # Add metadata and process results
            result = self._process_transcription_result(
                transcription, 
                detect_speakers=detect_speakers,
                extract_phrases=extract_phrases
            )
            
            # Add additional metrics if needed
            if kwargs.get('verify_accuracy', False) and result.get('text'):
                # If ground truth is provided, verify accuracy
                ground_truth = kwargs.get('ground_truth')
                if ground_truth:
                    result['accuracy_metrics'] = verify_text_matches_audio(
                        ground_truth, 
                        result['text']
                    )
            
            # Cache the result if enabled
            if should_use_cache:
                cache.cache_transcription(audio_path, result, engine_name, cache_options)
            
            # Clean up temporary file if needed
            if processed_path != audio_path and os.path.exists(processed_path):
                try:
                    os.unlink(processed_path)
                except:
                    pass
            
            # Clean up model cache if not needed
            if not self.cache_models:
                self._cleanup_service(engine_enum)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            
            # Clean up temporary file if needed
            if processed_path != audio_path and os.path.exists(processed_path):
                try:
                    os.unlink(processed_path)
                except:
                    pass
            
            raise STTError(
                message=f"Transcription failed: {str(e)}",
                details={"engine": engine_name, "audio_path": audio_path}
            ) from e
    
    @handle_audio_processing_errors
    def _preprocess_audio(self, audio_path: str, remove_noise: bool = False) -> str:
        """
        Preprocess audio file for better transcription.
        
        Args:
            audio_path: Path to the audio file
            remove_noise: Whether to remove background noise
            
        Returns:
            Path to the preprocessed audio file
        """
        # Check audio quality
        quality = assess_audio_quality(audio_path)
        
        # Skip preprocessing if audio already suitable for STT
        if quality.get('is_suitable_for_stt', False) and not remove_noise:
            return audio_path
        
        try:
            processed_path = audio_path
            
            # Convert to mono if needed
            if quality.get('channels', 1) > 1:
                processed_path = convert_to_mono(processed_path)
                logger.debug(f"Converted audio to mono: {processed_path}")
            
            # Normalize audio
            processed_path = normalize_audio(processed_path)
            logger.debug(f"Normalized audio: {processed_path}")
            
            # Remove background noise if requested
            if remove_noise:
                processed_path = remove_background_noise(processed_path)
                logger.debug(f"Removed background noise: {processed_path}")
            
            # Trim silence from beginning and end
            processed_path = trim_silence(processed_path)
            logger.debug(f"Trimmed silence from audio: {processed_path}")
            
            # Convert to 16kHz WAV format if needed
            if quality.get('sample_rate', 16000) != 16000:
                processed_path = convert_audio_format(processed_path, 'wav', 16000)
                logger.debug(f"Converted audio to 16kHz WAV: {processed_path}")
            
            return processed_path
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            raise AudioProcessingError(
                message=f"Failed to preprocess audio: {str(e)}",
                details={"audio_path": audio_path}
            ) from e
    
    def _process_transcription_result(self, transcription: Dict[str, Any], 
                                    detect_speakers: bool = False,
                                    extract_phrases: bool = True) -> Dict[str, Any]:
        """
        Process raw transcription result to add metadata and structure.
        
        Args:
            transcription: Raw transcription result
            detect_speakers: Whether to detect different speakers
            extract_phrases: Whether to extract individual phrases
            
        Returns:
            Processed transcription result
        """
        result = transcription.copy()
        
        # Add metadata if missing
        if 'language' not in result:
            text = result.get('text', '')
            result['language'] = detect_language(text)
        
        # Enhanced text normalization
        if result.get('text'):
            # Get the detected language
            lang = result.get('language', 'en')
            
            # Basic text normalization
            normalized = normalize_text(result['text'])
            
            # Enhanced normalization
            normalized = expand_abbreviations(normalized)
            normalized = normalize_numbers(normalized, language=lang)
            normalized = normalize_dates(normalized, language=lang)
            
            result['normalized_text'] = normalized
            
            # Check for multiple languages in the text
            language_parts = detect_language_parts(result['text'])
            if len(language_parts) > 1:
                # Multiple languages detected
                result['multi_language'] = True
                result['language_parts'] = language_parts
        
        # Extract phrases if requested
        if extract_phrases and result.get('text'):
            result['phrases'] = extract_spoken_phrases(result['text'])
        
        # Try to detect speakers if requested and not already done
        if detect_speakers and 'speakers' not in result:
            # Some engines natively detect speakers, if not we attempt it
            if 'segments' in result:
                # Simple heuristic - if we have segments with clear pauses,
                # we can try to guess speaker changes
                segments = result['segments']
                prev_end = 0
                speaker_id = 0
                
                for segment in segments:
                    # If there's a significant pause, might be speaker change
                    if 'start' in segment and (segment['start'] - prev_end) > 1.5:
                        speaker_id = 1 - speaker_id  # Toggle between 0 and 1
                    
                    segment['speaker'] = f"SPEAKER_{speaker_id}"
                    
                    if 'end' in segment:
                        prev_end = segment['end']
            
            # Group text by detected speakers
            if 'segments' in result:
                speakers = {}
                for segment in result['segments']:
                    speaker = segment.get('speaker', 'UNKNOWN')
                    if speaker not in speakers:
                        speakers[speaker] = []
                    
                    speakers[speaker].append({
                        'text': segment.get('text', ''),
                        'start': segment.get('start'),
                        'end': segment.get('end')
                    })
                
                result['speakers'] = speakers
        
        return result
    
    @handle_stt_errors
    def transcribe_audio_buffer(self, audio_buffer: bytes, 
                              file_format: str = 'wav',
                              engine: Optional[Union[STTEngine, str]] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio from a buffer.
        
        Args:
            audio_buffer: Audio data as bytes
            file_format: Format of the audio buffer
            engine: The STT engine to use. If None, uses the default engine.
            **kwargs: Additional parameters for transcription
            
        Returns:
            Dictionary with transcription results
        """
        # Save buffer to temporary file
        with tempfile.NamedTemporaryFile(suffix=f'.{file_format}', delete=False) as tmp:
            tmp.write(audio_buffer)
            audio_path = tmp.name
        
        try:
            # Transcribe the audio file
            result = self.transcribe_audio(
                audio_path,
                engine=engine,
                **kwargs
            )
            
            # Clean up the temporary file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio buffer: {str(e)}")
            
            # Clean up the temporary file
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            raise STTError(
                message=f"Failed to transcribe audio buffer: {str(e)}",
                details={"format": file_format}
            ) from e
    
    @handle_stt_errors
    def transcribe_continuous(self, audio_stream: BinaryIO, 
                           engine: Optional[Union[STTEngine, str]] = None,
                           chunk_size_ms: int = 5000,
                           **kwargs):
        """
        Perform continuous transcription from an audio stream.
        
        Args:
            audio_stream: Audio data stream
            engine: The STT engine to use
            chunk_size_ms: Size of each chunk in milliseconds
            **kwargs: Additional parameters for transcription
            
        Yields:
            Dictionaries with partial transcription results
        """
        # Get the appropriate service
        engine_enum = STTEngine(engine) if isinstance(engine, str) and engine else self.default_engine
        model_size = kwargs.get('model_size', 'base')
        service = self.get_service(engine_enum, model_size)
        
        # Check if the service supports streaming
        if not hasattr(service, 'transcribe_stream'):
            logger.error(f"Engine {engine_enum.value} does not support streaming transcription")
            yield {
                'error': 'Streaming not supported by this engine',
                'text': '',
                'final': True
            }
            return
        
        try:
            # Use the audio chunk generator for streaming if supported
            chunks = generate_audio_chunks(audio_stream, chunk_size_ms)
            
            # Set up streaming transcription
            for result in service.transcribe_stream(audio_stream, chunk_size_ms=chunk_size_ms, **kwargs):
                # Normalize text in streaming results for consistency
                if 'text' in result and result['text']:
                    result['normalized_text'] = normalize_text(result['text'])
                    
                yield result
                
            # Clean up model cache if not needed
            if not self.cache_models:
                self._cleanup_service(engine_enum)
                
        except Exception as e:
            logger.error(f"Error in continuous transcription: {str(e)}")
            yield {
                'error': str(e),
                'text': '',
                'final': True
            }
    
    def _cleanup_service(self, engine: STTEngine):
        """
        Clean up resources for a specific service.
        
        Args:
            engine: The STT engine to clean up
        """
        if engine in self._services:
            service = self._services[engine]
            if hasattr(service, 'cleanup'):
                try:
                    service.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up service {engine.value}: {e}")
    
    @handle_stt_errors
    def cleanup(self):
        """
        Clean up all services and release resources.
        """
        for engine in list(self._services.keys()):
            self._cleanup_service(engine)
            del self._services[engine]
    
    def list_available_engines(self) -> List[STTEngine]:
        """
        List all available STT engines.
        
        Returns:
            List of available STT engines
        """
        return [engine for engine in STTEngine]
    
    @handle_stt_errors
    def list_available_models(self, engine: Optional[Union[STTEngine, str]] = None) -> List[Dict[str, str]]:
        """
        List available models for the specified engine.
        
        Args:
            engine: The STT engine to use. If None, uses the default engine.
            
        Returns:
            List of available model dictionaries with id and name
        """
        model_size = 'base'  # Default model size to initialize with
        service = self.get_service(engine, model_size)
        
        if hasattr(service, "list_models"):
            models = service.list_models()
            return [{'id': model_id, 'name': name} for model_id, name in models.items()]
        
        return []
    
    @handle_stt_errors
    def list_available_languages(self, engine: Optional[Union[STTEngine, str]] = None) -> List[Dict[str, str]]:
        """
        List available languages for the specified engine.
        
        Args:
            engine: The STT engine to use. If None, uses the default engine.
            
        Returns:
            List of available language dictionaries with code and name
        """
        model_size = 'base'  # Default model size to initialize with
        service = self.get_service(engine, model_size)
        
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