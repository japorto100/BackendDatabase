"""
Faster Whisper service implementation.

This module provides a service for transcribing audio using the Faster Whisper model,
which is a memory-efficient implementation of OpenAI's Whisper model.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Any, BinaryIO, Generator, Union
import numpy as np

# Import the base STT service
from models_app.ai_models.audio.speechToText.base_stt_service import (
    BaseSTTService,
    transcription_operation,
    handle_stt_errors,
    handle_audio_processing_errors
)

# Import common utilities for error handling
from models_app.ai_models.utils.common.errors import (
    AudioModelError,
    STTError,
    AudioProcessingError
)

from .model_manager import WhisperFasterModelManager

logger = logging.getLogger(__name__)

class WhisperFasterService(BaseSTTService):
    """
    Service for transcribing audio using the Faster Whisper model.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None, 
                compute_type: str = "float16", cache_dir: Optional[str] = None):
        """
        Initialize the Faster Whisper service.
        
        Args:
            model_size: Size of the model to use ('tiny', 'base', 'small', 'medium', 'large-v2')
            device: Device to use for inference ('cpu', 'cuda', or specific CUDA device)
            compute_type: Compute type for inference ('float32', 'float16', 'int8')
            cache_dir: Directory to cache models
        """
        # Initialize the base class
        super().__init__(model_size=model_size, device=device, cache_dir=cache_dir)
        
        self.compute_type = compute_type
        
        # Initialize model manager
        self.model_manager = WhisperFasterModelManager(
            cache_dir=cache_dir,
            device=device
        )
        
        self._model = None
        
    @handle_stt_errors
    def _load_model(self):
        """Load the Faster Whisper model if not already loaded."""
        if self._model is not None:
            return
        
        try:
            # Load model through the model manager
            self._model = self.model_manager.load_model(
                model_size=self.model_size,
                compute_type=self.compute_type
            )
            
            logger.info(f"Loaded Faster Whisper model {self.model_size} on {self.model_manager.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            raise ImportError(
                "Faster Whisper requires the faster-whisper package. "
                "Install with: pip install faster-whisper"
            )
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise STTError(
                message=f"Failed to load Faster Whisper model: {str(e)}", 
                details={"model_size": self.model_size, "device": self.model_manager.device}
            ) from e
    
    @transcription_operation
    def transcribe(self, audio_path: str, language: Optional[str] = None, 
                 task: str = "transcribe", word_timestamps: bool = False,
                 beam_size: int = 5, vad_filter: bool = True,
                 initial_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'fr', 'de')
            task: Task to perform ('transcribe' or 'translate')
            word_timestamps: Whether to include timestamps for each word
            beam_size: Beam size for decoding
            vad_filter: Whether to use voice activity detection to filter out non-speech
            initial_prompt: Optional prompt to guide the transcription
            
        Returns:
            Dictionary with transcription results
        """
        # Call the parent method to update statistics
        super().transcribe(audio_path, language, **kwargs)
        
        self._load_model()
        
        try:
            # Set up parameters
            options = {
                "language": language,
                "task": task,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "word_timestamps": word_timestamps,
                "initial_prompt": initial_prompt
            }
            
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}
            
            # Run inference
            segments, info = self._model.transcribe(audio_path, **options)
            
            # Process the segments into a standardized format
            result = {
                "text": "",
                "segments": [],
                "language": info.language,
                "language_probability": info.language_probability
            }
            
            # Collect segments
            segments_list = list(segments)  # Convert generator to list
            
            # Combine all segment texts for full transcript
            all_texts = [segment.text for segment in segments_list]
            result["text"] = " ".join(all_texts)
            
            # Format segments
            for i, segment in enumerate(segments_list):
                formatted_segment = {
                    "id": i,
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "confidence": segment.avg_logprob
                }
                
                # Add word timestamps if available
                if word_timestamps and hasattr(segment, "words") and segment.words:
                    formatted_segment["words"] = [
                        {
                            "text": word.word,
                            "start": word.start,
                            "end": word.end,
                            "confidence": word.probability
                        }
                        for word in segment.words
                    ]
                
                result["segments"].append(formatted_segment)
            
            # Add confidence score
            result["confidence"] = sum(s.get("confidence", 0) for s in result["segments"]) / max(len(result["segments"]), 1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise STTError(
                message=f"Failed to transcribe with Faster Whisper: {str(e)}", 
                details={"audio_path": audio_path, "language": language, "task": task}
            ) from e
    
    @handle_audio_processing_errors
    def transcribe_stream(self, audio_stream: BinaryIO, chunk_size_ms: int = 5000, 
                        language: Optional[str] = None, **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Perform streaming transcription from an audio stream.
        
        Args:
            audio_stream: Audio data stream
            chunk_size_ms: Size of each chunk in milliseconds
            language: Language code for transcription
            **kwargs: Additional parameters for transcription
            
        Yields:
            Dictionaries with partial transcription results
        """
        # Call the parent method to update statistics
        super().transcribe_stream(audio_stream, chunk_size_ms, language, **kwargs)
        
        self._load_model()
        
        try:
            import soundfile as sf
            import librosa
            import numpy as np
            from io import BytesIO
            
            # Create a temporary file for the stream
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write the stream to a temporary file
                audio_stream.seek(0)
                temp_file.write(audio_stream.read())
                temp_file.flush()
            
            # Read the audio file
            audio_array, sampling_rate = librosa.load(temp_path, sr=16000)
            
            # Calculate chunk size in samples
            chunk_size_samples = int((chunk_size_ms / 1000) * sampling_rate)
            
            # Process audio in chunks
            for i in range(0, len(audio_array), chunk_size_samples):
                chunk = audio_array[i:i + chunk_size_samples]
                
                # Skip silent chunks
                if np.abs(chunk).max() < 0.01:
                    continue
                
                # Save chunk to a temporary file
                chunk_path = f"{temp_path}_chunk_{i}.wav"
                sf.write(chunk_path, chunk, sampling_rate)
                
                try:
                    # Transcribe the chunk
                    result = self.transcribe(
                        chunk_path,
                        language=language,
                        vad_filter=True,  # Use VAD for streaming to filter noise
                        **kwargs
                    )
                    
                    # Add chunk information
                    result["is_partial"] = True
                    result["chunk_id"] = i // chunk_size_samples
                    
                    # Calculate timing based on chunk position
                    chunk_start_time = (i / sampling_rate)
                    for segment in result.get("segments", []):
                        segment["start"] += chunk_start_time
                        segment["end"] += chunk_start_time
                        for word in segment.get("words", []):
                            word["start"] += chunk_start_time
                            word["end"] += chunk_start_time
                    
                    yield result
                    
                finally:
                    # Clean up temporary chunk file
                    if os.path.exists(chunk_path):
                        os.unlink(chunk_path)
            
            # Final cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Error in streaming transcription: {e}")
            raise STTError(
                message=f"Failed to perform streaming transcription: {str(e)}", 
                details={"language": language}
            ) from e
    
    @handle_stt_errors
    def list_models(self) -> Dict[str, str]:
        """
        List available Whisper models.
        
        Returns:
            Dictionary mapping model IDs to descriptive names
        """
        return {
            "tiny": "Tiny (39M parameters)",
            "base": "Base (74M parameters)",
            "small": "Small (244M parameters)",
            "medium": "Medium (769M parameters)",
            "large-v1": "Large v1 (1550M parameters)",
            "large-v2": "Large v2 (1550M parameters)",
            "large-v3": "Large v3 (1550M parameters)"
        }
    
    @handle_stt_errors
    def list_languages(self) -> List[str]:
        """
        List supported languages.
        
        Returns:
            List of supported language codes
        """
        # Whisper supports many languages, here are some common ones
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", 
            "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", 
            "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", 
            "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", 
            "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", 
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", 
            "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", 
            "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", 
            "ba", "jw", "su"
        ]
    
    @handle_stt_errors
    def cleanup(self):
        """Release resources and clear model cache."""
        # Release model resources
        self._model = None
        
        # Clear the cache
        if self.model_manager:
            self.model_manager.clear_cache()
            
        logger.info("Cleaned up Faster Whisper resources") 