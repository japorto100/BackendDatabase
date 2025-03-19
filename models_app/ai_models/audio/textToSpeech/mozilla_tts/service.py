"""
Service class for Mozilla TTS that handles text-to-speech conversion.
"""

import os
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple, Any, Union, Generator, BinaryIO
import tempfile
import time
import uuid

from django.conf import settings

# Import base service
from models_app.ai_models.audio.textToSpeech.base_tts_service import (
    BaseTTSService,
    handle_tts_errors,
    handle_audio_processing_errors,
    synthesis_operation
)

# Import error handling
from models_app.ai_models.utils.common.errors import (
    AudioModelError,
    TTSError,
    AudioProcessingError
)

from .model_manager import MozillaTTSModelManager

logger = logging.getLogger(__name__)

class MozillaTTSService(BaseTTSService):
    """
    Service for Mozilla TTS text-to-speech conversion.
    
    This class:
    - Manages the TTS models through MozillaTTSModelManager
    - Provides text-to-speech conversion functionality
    - Handles audio output in different formats
    """
    
    def __init__(self, voice_id: str = "default", device: Optional[str] = None,
                cache_dir: Optional[str] = None, language: str = "en", 
                model_name: Optional[str] = None):
        """
        Initialize the Mozilla TTS service.
        
        Args:
            voice_id: ID of the voice to use
            device: Device to use for inference ('cpu', 'cuda', etc.)
            cache_dir: Directory to cache models
            language: Language code for synthesis
            model_name: Name of the model to use. If None, uses default model.
        """
        # Initialize the base class
        super().__init__(voice_id=voice_id, device=device, cache_dir=cache_dir, language=language)
        
        self.model_manager = MozillaTTSModelManager(cache_dir)
        self.model_name = model_name or MozillaTTSModelManager.DEFAULT_MODEL_NAME
        self.tts_model = None
        self.synthesizer = None
        
        # Additional parameters
        self.output_dir = os.path.join(settings.MEDIA_ROOT, 'tts_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
    @handle_tts_errors
    def _load_model(self) -> bool:
        """
        Load a Mozilla TTS model.
        
        Returns:
            True if model was loaded successfully, False otherwise
        """
        # If model is already loaded, return True
        if self.synthesizer is not None:
            return True
            
        # Make sure model is downloaded
        if not self.model_manager.ensure_model_available(self.model_name):
            logger.error(f"Failed to ensure model {self.model_name} is available")
            raise TTSError(
                message=f"Failed to download model {self.model_name}",
                details={"model_name": self.model_name}
            )
        
        try:
            # Import here to avoid dependency issues if TTS is not installed
            from TTS.utils.manage import ModelManager
            from TTS.utils.synthesizer import Synthesizer
            
            model_path = self.model_manager.get_model_path(self.model_name)
            
            # Different handling based on model type
            if self.model_name == "tts_en_ljspeech":
                # Load model directly
                config_path = os.path.join(model_path, "config.json")
                model_file = os.path.join(model_path, os.path.basename(self.model_manager.MODELS_INFO[self.model_name]["url"]))
                
                self.synthesizer = Synthesizer(
                    tts_checkpoint=model_file,
                    tts_config_path=config_path,
                    use_cuda=(self.device == "cuda")
                )
            else:
                # For newer models that follow the TTS download structure
                # Use TTS ModelManager to find the appropriate files
                tts_manager = ModelManager()
                
                if self.model_name == "tts_en_multispeaker":
                    model_path = tts_manager.download_model("tts_models/en/vctk/vits")
                    self.synthesizer = Synthesizer(
                        tts_checkpoint=model_path,
                        use_cuda=(self.device == "cuda")
                    )
                elif self.model_name == "tts_multilingual":
                    model_path = tts_manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
                    self.synthesizer = Synthesizer(
                        tts_checkpoint=model_path,
                        use_cuda=(self.device == "cuda")
                    )
            
            return True
        except Exception as e:
            logger.error(f"Error loading Mozilla TTS model: {str(e)}")
            raise TTSError(
                message=f"Error loading Mozilla TTS model: {str(e)}",
                details={"model_name": self.model_name, "error": str(e)}
            ) from e
    
    @synthesis_operation
    def synthesize(self, text: str, voice_id: Optional[str] = None,
                 output_path: Optional[str] = None, **kwargs) -> Union[str, bytes]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_id: ID of the voice to use (overrides the default)
            output_path: Path to save audio file. If None, returns audio data as bytes.
            **kwargs: Additional parameters for synthesis
            
        Returns:
            Path to the audio file if output_path is provided, otherwise audio data as bytes
        """
        # Call the parent method to update statistics
        super().synthesize(text, voice_id, output_path, **kwargs)
        
        if not text:
            raise TTSError(
                message="Empty text provided for speech synthesis",
                details={"text": text}
            )
        
        # If voice_id is provided, update the current voice
        if voice_id and voice_id != self.voice_id:
            self.voice_id = voice_id
            
        # Load model if not already loaded
        self._load_model()
        
        # Set output path if not provided
        if not output_path:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name
        
        try:
            # Prepare synthesis parameters
            params = {}
            
            # If using a multi-speaker model, set speaker ID from voice_id
            if self.model_name in ["tts_en_multispeaker", "tts_multilingual"]:
                params["speaker_id"] = self.voice_id
            
            # For multilingual models, set language
            if self.model_name == "tts_multilingual":
                params["language"] = kwargs.get("language", self.language)
            
            # Get speed parameter
            speed = kwargs.get("speed", 1.0)
            
            # Synthesize speech
            waves = self.synthesizer.tts(text=text, **params)
            
            # Adjust speed if needed
            if speed != 1.0:
                import librosa
                waves = librosa.effects.time_stretch(waves, rate=speed)
            
            # Save to file
            self.synthesizer.save_wav(waves, output_path)
            
            if not os.path.exists(output_path):
                raise TTSError(
                    message="Speech synthesis failed - output file not created",
                    details={"text": text, "output_path": output_path}
                )
            
            logger.info(f"Speech synthesis successful, saved to {output_path}")
            
            # If the original output_path was None, read the file and return bytes
            if kwargs.get("return_bytes", True) and not kwargs.get("original_output_path"):
                with open(output_path, "rb") as f:
                    audio_data = f.read()
                # Clean up temporary file
                os.unlink(output_path)
                return audio_data
                
            return output_path
                
        except Exception as e:
            logger.error(f"Error during speech synthesis: {str(e)}")
            raise TTSError(
                message=f"Failed to synthesize speech: {str(e)}",
                details={"text": text, "output_path": output_path}
            ) from e
    
    def synthesize_stream(self, text_stream: Union[str, List[str]], 
                        chunk_size: int = 100, **kwargs) -> Generator[bytes, None, None]:
        """
        Perform streaming synthesis from a text stream.
        
        Args:
            text_stream: Text stream or list of text chunks to synthesize
            chunk_size: Size of each chunk in characters
            **kwargs: Additional parameters for synthesis
            
        Yields:
            Audio data chunks as bytes
        """
        # Call the parent method to update statistics
        super().synthesize_stream(text_stream, chunk_size, **kwargs)
        
        # Load model if not already loaded
        self._load_model()
        
        # Process input text
        if isinstance(text_stream, str):
            # Split text into sentences or chunks
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text_stream)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                    
            if current_chunk:
                chunks.append(current_chunk)
        else:
            # Use provided chunks
            chunks = text_stream
        
        # Synthesize each chunk
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            # Synthesize to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name
            
            try:
                # Synthesize chunk
                self.synthesize(
                    text=chunk,
                    output_path=output_path,
                    original_output_path=True,
                    return_bytes=False,
                    **kwargs
                )
                
                # Read chunk and yield
                with open(output_path, "rb") as f:
                    audio_data = f.read()
                
                yield audio_data
                
            finally:
                # Clean up temporary file
                if os.path.exists(output_path):
                    os.unlink(output_path)
    
    @handle_tts_errors
    def list_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        List available voices.
        
        Returns:
            Dictionary mapping voice IDs to voice information
        """
        voices = {}
        
        if self.model_name == "tts_en_ljspeech":
            voices["default"] = {
                "name": "LJSpeech Female",
                "gender": "female",
                "language": "en"
            }
        elif self.model_name == "tts_en_multispeaker":
            # Try to load model to get speakers
            self._load_model()
            
            if self.synthesizer and hasattr(self.synthesizer.tts_model, "speaker_manager"):
                speaker_names = self.synthesizer.tts_model.speaker_manager.speaker_names
                for i, name in enumerate(speaker_names):
                    voices[str(i)] = {
                        "name": name,
                        "gender": "unknown",
                        "language": "en"
                    }
        elif self.model_name == "tts_multilingual":
            # XTTS standard voices
            voices = {
                "male_en": {"name": "English Male", "gender": "male", "language": "en"},
                "female_en": {"name": "English Female", "gender": "female", "language": "en"},
                "male_es": {"name": "Spanish Male", "gender": "male", "language": "es"},
                "female_es": {"name": "Spanish Female", "gender": "female", "language": "es"},
                "male_fr": {"name": "French Male", "gender": "male", "language": "fr"},
                "female_fr": {"name": "French Female", "gender": "female", "language": "fr"},
                "male_de": {"name": "German Male", "gender": "male", "language": "de"},
                "female_de": {"name": "German Female", "gender": "female", "language": "de"}
            }
            
        # If no voices found, add default
        if not voices:
            voices["default"] = {
                "name": "Default Voice",
                "gender": "neutral",
                "language": "en"
            }
            
        return voices
    
    @handle_tts_errors
    def list_languages(self) -> List[str]:
        """
        List supported languages.
        
        Returns:
            List of supported language codes
        """
        if self.model_name == "tts_multilingual":
            # Typical languages supported by XTTS
            return ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn"]
        elif self.model_name == "tts_en_ljspeech" or self.model_name == "tts_en_multispeaker":
            return ["en"]
        
        return ["en"]
    
    @handle_tts_errors
    def adjust_voice(self, rate: Optional[float] = None, pitch: Optional[float] = None,
                    volume: Optional[float] = None) -> None:
        """
        Adjust voice properties.
        
        Args:
            rate: Speech rate multiplier (1.0 is normal speed)
            pitch: Voice pitch multiplier (1.0 is normal pitch)
            volume: Volume level (1.0 is normal volume)
        """
        # Mozilla TTS has limited direct voice adjustment
        # Rate can be handled during synthesis, but pitch and volume
        # would need to be handled in post-processing
        logger.info("Voice parameter adjustment in Mozilla TTS is limited to speech rate during synthesis")
    
    @handle_tts_errors
    def cleanup(self):
        """Release resources and clear model cache."""
        # Release model resources
        self.synthesizer = None
        self.tts_model = None
        
        # Clear the cache if model manager supports it
        if hasattr(self.model_manager, 'clear_cache'):
            self.model_manager.clear_cache()
            
        logger.info("Cleaned up Mozilla TTS resources") 