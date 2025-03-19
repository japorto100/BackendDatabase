"""
WhisperX service implementation.

This module provides a service for transcribing audio using the WhisperX model,
which extends OpenAI's Whisper with speaker diarization and improved alignment.
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

from .model_manager import WhisperXModelManager

logger = logging.getLogger(__name__)

class WhisperXService(BaseSTTService):
    """
    Service for transcribing audio using the WhisperX model with speaker diarization.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None, 
                compute_type: str = "float16", cache_dir: Optional[str] = None,
                diarize: bool = False, num_speakers: Optional[int] = None):
        """
        Initialize the WhisperX service.
        
        Args:
            model_size: Size of the Whisper model to use
            device: Device to use for inference ('cpu', 'cuda', or specific CUDA device)
            compute_type: Compute type for inference ('float32', 'float16', 'int8')
            cache_dir: Directory to cache models
            diarize: Whether to enable speaker diarization by default
            num_speakers: Number of speakers for diarization (if None, auto-detected)
        """
        # Initialize the base class
        super().__init__(model_size=model_size, device=device, cache_dir=cache_dir)
        
        self.compute_type = compute_type
        self.diarize = diarize
        self.num_speakers = num_speakers
        
        # Initialize model manager
        self.model_manager = WhisperXModelManager(
            cache_dir=cache_dir,
            device=device
        )
        
        self._model = None
        self._alignment_model = None
        self._diarization_model = None
        
    @handle_stt_errors
    def _load_model(self):
        """Load the WhisperX model if not already loaded."""
        if self._model is not None:
            return
        
        try:
            # Load ASR model through the model manager
            self._model = self.model_manager.load_asr_model(
                model_size=self.model_size,
                compute_type=self.compute_type
            )
            
            logger.info(f"Loaded WhisperX model {self.model_size} on {self.model_manager.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            raise ImportError(
                "WhisperX requires the whisperx package. "
                "Install with: pip install git+https://github.com/m-bain/whisperx.git"
            )
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            raise STTError(
                message=f"Failed to load WhisperX model: {str(e)}", 
                details={"model_size": self.model_size, "device": self.model_manager.device}
            ) from e
    
    @handle_stt_errors
    def _load_diarization_model(self):
        """Load the speaker diarization model if not already loaded."""
        if self._diarization_model is not None:
            return
        
        try:
            # Load diarization model through the model manager
            self._diarization_model = self.model_manager.load_diarization_model()
            
            logger.info(f"Loaded diarization model on {self.model_manager.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import required packages for diarization: {e}")
            raise ImportError(
                "Speaker diarization in WhisperX requires additional dependencies. "
                "Install with: pip install pyannote.audio"
            )
        except Exception as e:
            logger.error(f"Failed to load diarization model: {e}")
            raise STTError(
                message=f"Failed to load diarization model: {str(e)}", 
                details={"model_size": self.model_size, "device": self.model_manager.device}
            ) from e
    
    @transcription_operation
    def transcribe(self, audio_path: str, language: Optional[str] = None, 
                 task: str = "transcribe", word_timestamps: bool = True,
                 diarize: Optional[bool] = None, num_speakers: Optional[int] = None,
                 min_speakers: int = 1, max_speakers: int = 8,
                 batch_size: int = 16, initial_prompt: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio to text with optional speaker diarization.
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'fr', 'de')
            task: Task to perform ('transcribe' or 'translate')
            word_timestamps: Whether to include timestamps for each word
            diarize: Whether to perform speaker diarization
            num_speakers: Number of speakers for diarization (if None, auto-detected)
            min_speakers: Minimum number of speakers to consider for auto-detection
            max_speakers: Maximum number of speakers to consider for auto-detection
            batch_size: Batch size for processing
            initial_prompt: Optional prompt to guide the transcription
            
        Returns:
            Dictionary with transcription results including speaker information if diarized
        """
        # Call the parent method to update statistics
        super().transcribe(audio_path, language, **kwargs)
        
        self._load_model()
        
        # Determine whether to apply diarization
        should_diarize = self.diarize if diarize is None else diarize
        
        try:
            import whisperx
            
            # Set up parameters
            asr_options = {
                "language": language,
                "task": task,
                "batch_size": batch_size,
                "initial_prompt": initial_prompt
            }
            
            # Remove None values
            asr_options = {k: v for k, v in asr_options.items() if v is not None}
            
            # Run ASR
            result = self._model.transcribe(audio_path, **asr_options)
            
            # Align word timestamps (WhisperX key feature)
            if word_timestamps or should_diarize:
                # Load alignment model based on language
                detected_language = result.get("language", language)
                if detected_language is None:
                    logger.warning("Language not detected for alignment, defaulting to English")
                    detected_language = "en"
                
                # Get alignment model from manager
                alignment_model = self.model_manager.load_alignment_model(detected_language)
                
                # Align words
                result = whisperx.align(
                    result["segments"],
                    alignment_model,
                    audio_path,
                    self.model_manager.device,
                    return_char_alignments=False
                )
            
            # Perform diarization if requested
            if should_diarize:
                # Load diarization model if needed
                self._load_diarization_model()
                
                # Get number of speakers
                speaker_count = num_speakers or self.num_speakers
                diarize_options = {}
                
                if speaker_count is not None:
                    diarize_options["num_speakers"] = speaker_count
                else:
                    diarize_options["min_speakers"] = min_speakers
                    diarize_options["max_speakers"] = max_speakers
                
                # Run diarization
                diarize_segments = self._diarization_model(audio_path, **diarize_options)
                
                # Assign speakers to segments
                result = whisperx.assign_word_speakers(diarize_segments, result)
                
                # Process speakers
                speakers = {}
                for segment in result.get("segments", []):
                    for word in segment.get("words", []):
                        speaker = word.get("speaker", "UNKNOWN")
                        if speaker not in speakers:
                            speakers[speaker] = []
                        
                        speakers[speaker].append({
                            "text": word.get("word", ""),
                            "start": word.get("start", 0),
                            "end": word.get("end", 0)
                        })
                
                # Add speakers to result
                result["speakers"] = speakers
                result["num_speakers"] = len(speakers)
            
            # Format result to match our standard output structure
            formatted_result = self._format_result(result, word_timestamps, should_diarize)
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise STTError(
                message=f"Failed to transcribe with WhisperX: {str(e)}", 
                details={
                    "audio_path": audio_path, 
                    "language": language, 
                    "task": task,
                    "diarize": should_diarize
                }
            ) from e
    
    def _format_result(self, result: Dict[str, Any], word_timestamps: bool = False, 
                     include_speakers: bool = False) -> Dict[str, Any]:
        """
        Format the WhisperX output into a standardized format.
        
        Args:
            result: Raw WhisperX output
            word_timestamps: Whether to include word timestamps
            include_speakers: Whether to include speaker information
            
        Returns:
            Formatted result dictionary
        """
        formatted = {
            "text": " ".join([segment.get("text", "") for segment in result.get("segments", [])]),
            "language": result.get("language"),
        }
        
        # Format segments
        formatted_segments = []
        for i, segment in enumerate(result.get("segments", [])):
            formatted_segment = {
                "id": i,
                "text": segment.get("text", ""),
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
            }
            
            # Add speaker if available
            if include_speakers and "speaker" in segment:
                formatted_segment["speaker"] = segment["speaker"]
            
            # Add word timestamps if available and requested
            if word_timestamps and "words" in segment:
                formatted_segment["words"] = [
                    {
                        "text": word.get("word", ""),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0),
                        "speaker": word.get("speaker") if include_speakers and "speaker" in word else None
                    }
                    for word in segment["words"]
                ]
            
            formatted_segments.append(formatted_segment)
        
        formatted["segments"] = formatted_segments
        
        # Add speaker information if available
        if include_speakers and "speakers" in result:
            formatted["speakers"] = result["speakers"]
            formatted["num_speakers"] = result.get("num_speakers", len(result["speakers"]))
        
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
                "large-v1": 0.9,
                "large-v2": 0.9,
                "large-v3": 0.95
            }
            formatted["confidence"] = confidence_map.get(self.model_size, 0.75)
        
        return formatted
    
    @handle_audio_processing_errors
    def transcribe_stream(self, audio_stream: BinaryIO, chunk_size_ms: int = 5000, 
                        language: Optional[str] = None, diarize: Optional[bool] = None,
                        **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Perform streaming transcription from an audio stream.
        
        Args:
            audio_stream: Audio data stream
            chunk_size_ms: Size of each chunk in milliseconds
            language: Language code for transcription
            diarize: Whether to perform speaker diarization
            **kwargs: Additional parameters for transcription
            
        Yields:
            Dictionaries with partial transcription results
        """
        # Call the parent method to update statistics
        super().transcribe_stream(audio_stream, chunk_size_ms, language, **kwargs)
        
        self._load_model()
        
        # Determine whether to apply diarization
        should_diarize = self.diarize if diarize is None else diarize
        
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
            
            # For diarization, we might need to process the entire file first
            if should_diarize:
                # Load diarization model
                self._load_diarization_model()
                
                # Process the whole file for diarization
                diarize_segments = self._diarization_model(temp_path, **kwargs)
            
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
                        diarize=False,  # We handle diarization separately for streaming
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
                    
                    # Add speaker information if diarization was requested
                    if should_diarize and "diarize_segments" in locals():
                        # Match words with speaker segments based on timestamps
                        # This is a simplified approach, WhisperX would do this more accurately
                        try:
                            import whisperx
                            # Create a small result dict with the current segment
                            mini_result = {"segments": result.get("segments", [])}
                            # Assign speakers using WhisperX's function
                            with_speakers = whisperx.assign_word_speakers(diarize_segments, mini_result)
                            # Update the segments with speaker info
                            result["segments"] = with_speakers.get("segments", result.get("segments", []))
                        except Exception as speaker_err:
                            logger.warning(f"Could not assign speakers to streaming chunk: {speaker_err}")
                    
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
                details={"language": language, "diarize": should_diarize}
            ) from e
    
    @handle_stt_errors
    def list_models(self) -> Dict[str, str]:
        """
        List available WhisperX models.
        
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
        # WhisperX supports the same languages as Whisper
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
        self._alignment_model = None
        self._diarization_model = None
        
        # Clear the cache
        if self.model_manager:
            self.model_manager.clear_cache()
            
        logger.info("Cleaned up WhisperX resources") 