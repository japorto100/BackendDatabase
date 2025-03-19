"""
Insanely Fast Whisper service implementation.

This module provides a service for transcribing audio using the Insanely Fast Whisper model,
which is an optimized implementation of OpenAI's Whisper model.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Any, BinaryIO, Generator, Union
import numpy as np
import torch

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

from .model_manager import WhisperInsanelyFastModelManager

logger = logging.getLogger(__name__)

class WhisperInsanelyFastService(BaseSTTService):
    """
    Service for transcribing audio using the Insanely Fast Whisper model.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None, 
                compute_type: str = "float16", cache_dir: Optional[str] = None):
        """
        Initialize the Insanely Fast Whisper service.
        
        Args:
            model_size: Size of the model to use ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
            device: Device to use for inference ('cpu', 'cuda', 'mps', or specific CUDA device)
            compute_type: Compute type for inference ('float32', 'float16', 'int8')
            cache_dir: Directory to cache models
        """
        # Initialize the base class
        super().__init__(model_size=model_size, device=device, cache_dir=cache_dir)
        
        self.compute_type = compute_type
        
        # Initialize model manager
        self.model_manager = WhisperInsanelyFastModelManager(
            cache_dir=cache_dir,
            device=device
        )
        
        self._model = None
        self._processor = None
        
    @handle_stt_errors
    def _load_model(self):
        """Load the Whisper model and processor if not already loaded."""
        if self._model is not None and self._processor is not None:
            return
        
        try:
            # Load model through the model manager
            self._model, self._processor = self.model_manager.load_model(
                model_size=self.model_size,
                compute_type=self.compute_type
            )
            
            logger.info(f"Loaded Whisper model {self.model_size} on {self.model_manager.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            raise ImportError(
                "Insanely Fast Whisper requires transformers and optimum-bettertransformer. "
                "Install with: pip install transformers optimum[bettertransformer] torch"
            )
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise STTError(
                message=f"Failed to load Insanely Fast Whisper model: {str(e)}", 
                details={"model_size": self.model_size, "device": self.model_manager.device}
            ) from e
    
    @transcription_operation
    def transcribe(self, audio_path: str, language: Optional[str] = None, 
                 task: str = "transcribe", word_timestamps: bool = False,
                 return_timestamps: bool = True, chunk_size_sec: Optional[int] = None,
                 batch_size: int = 16, initial_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'fr', 'de')
            task: Task to perform ('transcribe' or 'translate')
            word_timestamps: Whether to include timestamps for each word
            return_timestamps: Whether to include timestamps for segments
            chunk_size_sec: Size of audio chunks in seconds (None for no chunking)
            batch_size: Batch size for processing chunks
            initial_prompt: Optional prompt to guide the transcription
            
        Returns:
            Dictionary with transcription results
        """
        # Call the parent method to update statistics
        super().transcribe(audio_path, language, **kwargs)
        
        self._load_model()
        
        try:
            import librosa
            from transformers import pipeline
            
            # Load audio
            audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
            
            # Create pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model=self._model,
                tokenizer=self._processor.tokenizer,
                feature_extractor=self._processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=chunk_size_sec,
                batch_size=batch_size,
                return_timestamps=True if word_timestamps else return_timestamps,
                torch_dtype=torch.float16 if self.compute_type == "float16" else torch.float32,
                device=self.model_manager.device
            )
            
            # Set up parameters
            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language
            if initial_prompt:
                generate_kwargs["prompt"] = initial_prompt
                
            generate_kwargs["task"] = task
            
            # Run inference
            result = pipe(
                audio_array,
                generate_kwargs=generate_kwargs,
            )
            
            # Process the result into a standardized format
            return self._format_result(result, word_timestamps=word_timestamps)
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise STTError(
                message=f"Failed to transcribe with Insanely Fast Whisper: {str(e)}", 
                details={"audio_path": audio_path, "language": language, "task": task}
            ) from e
    
    def _format_result(self, result: Dict[str, Any], word_timestamps: bool = False) -> Dict[str, Any]:
        """
        Format the raw model output into a standardized format.
        
        Args:
            result: Raw output from the pipeline
            word_timestamps: Whether word-level timestamps were requested
            
        Returns:
            Formatted result dictionary
        """
        formatted = {
            "text": result.get("text", ""),
            "language": result.get("language", None),
        }
        
        # Process chunks/segments
        if "chunks" in result:
            segments = result["chunks"]
        elif "segments" in result:
            segments = result["segments"]
        else:
            # Create a single segment if no segments are provided
            segments = [{"text": formatted["text"], "start": 0.0, "end": 0.0}]
        
        # Format segments
        formatted_segments = []
        for i, segment in enumerate(segments):
            formatted_segment = {
                "id": i,
                "text": segment.get("text", ""),
                "start": segment.get("timestamp", (0.0, 0.0))[0] if isinstance(segment.get("timestamp"), tuple) else segment.get("start", 0.0),
                "end": segment.get("timestamp", (0.0, 0.0))[1] if isinstance(segment.get("timestamp"), tuple) else segment.get("end", 0.0),
            }
            
            # Add word timestamps if available
            if word_timestamps and "words" in segment:
                formatted_segment["words"] = [
                    {
                        "text": word.get("text", ""),
                        "start": word.get("timestamp", (0.0, 0.0))[0] if isinstance(word.get("timestamp"), tuple) else word.get("start", 0.0),
                        "end": word.get("timestamp", (0.0, 0.0))[1] if isinstance(word.get("timestamp"), tuple) else word.get("end", 0.0),
                    }
                    for word in segment["words"]
                ]
            
            formatted_segments.append(formatted_segment)
        
        formatted["segments"] = formatted_segments
        
        # Add confidence if available
        if "confidence" in result:
            formatted["confidence"] = result["confidence"]
        else:
            # Estimate confidence based on model size (better models generally have higher confidence)
            confidence_map = {
                "tiny": 0.6,
                "base": 0.7,
                "small": 0.8,
                "medium": 0.85,
                "large-v2": 0.9,
                "large-v3": 0.95
            }
            formatted["confidence"] = confidence_map.get(self.model_size, 0.75)
        
        return formatted
    
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
        
        # Update with your streaming implementation
        # For now, we'll implement a basic chunking approach
        try:
            import librosa
            import soundfile as sf
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
                
                # Transcribe the chunk
                try:
                    result = self.transcribe(
                        chunk_path, 
                        language=language,
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
        self._processor = None
        
        # Clear the cache
        if self.model_manager:
            self.model_manager.clear_cache()
            
        logger.info("Cleaned up Insanely Fast Whisper resources") 