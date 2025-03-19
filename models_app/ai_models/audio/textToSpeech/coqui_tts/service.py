"""
Service for text-to-speech conversion using Coqui-TTS.
"""

import os
import uuid
import logging
import tempfile
from typing import Dict, Optional, Any, List, Tuple, Union, Generator, BinaryIO
from pathlib import Path
import numpy as np

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

from .model_manager import CoquiTTSModelManager

logger = logging.getLogger(__name__)

class CoquiTTSService(BaseTTSService):
    """
    Service for text-to-speech conversion using Coqui-TTS.
    
    Handles:
    - Loading Coqui-TTS models
    - Converting text to speech with various models
    - Supporting multiple languages
    """
    
    def __init__(self, voice_id: str = "en_female", device: Optional[str] = None,
                cache_dir: Optional[str] = None, language: str = "en"):
        """
        Initialize the CoquiTTSService.
        
        Args:
            voice_id: ID of the voice to use
            device: Device to use for inference ('cpu', 'cuda', etc.)
            cache_dir: Directory to cache models
            language: Language code for synthesis
        """
        # Initialize the base class
        super().__init__(voice_id=voice_id, device=device, cache_dir=cache_dir, language=language)
        
        self.model_key = voice_id
        self.model_manager = CoquiTTSModelManager()
        self.model = None
        self.vocoder = None
        self.output_dir = os.path.join(settings.MEDIA_ROOT, 'tts_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
    @handle_tts_errors
    def _load_model(self) -> bool:
        """
        Load a Coqui-TTS model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        model_key = self.voice_id
        
        # If model is already loaded and it's the same model, return True
        if self.model is not None and hasattr(self, 'current_model_key') and self.current_model_key == model_key:
            return True
        
        # Ensure model is downloaded
        if not self.model_manager.ensure_model_available(model_key):
            logger.error(f"Failed to download model {model_key}")
            raise TTSError(
                message=f"Failed to download model {model_key}",
                details={"model_key": model_key}
            )
        
        # Get model information
        model_info = self.model_manager.get_model_info(model_key)
        if not model_info:
            logger.error(f"Model information not found for {model_key}")
            raise TTSError(
                message=f"Model information not found for {model_key}",
                details={"model_key": model_key}
            )
        
        try:
            # Import TTS
            try:
                from TTS.utils.manage import ModelManager
                from TTS.utils.synthesizer import Synthesizer
            except ImportError:
                # Need to install Coqui-TTS
                logger.warning("Installing Coqui-TTS package")
                import subprocess
                import sys
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "TTS"
                ])
                # Try importing again after installation
                from TTS.utils.manage import ModelManager
                from TTS.utils.synthesizer import Synthesizer
            
            # Initialize synthesizer
            synthesizer = Synthesizer(
                model_info["path"],
                model_info.get("vocoder_path"),
                use_cuda=(self.device == "cuda")
            )
            
            # Store model
            self.model = synthesizer
            self.current_model_key = model_key
            self.current_model_info = model_info
            
            logger.info(f"Successfully loaded Coqui-TTS model {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Coqui-TTS model {model_key}: {str(e)}")
            self.model = None
            raise TTSError(
                message=f"Error loading Coqui-TTS model: {str(e)}",
                details={"model_key": model_key, "error": str(e)}
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
            # Set speaker ID if needed
            speaker_id = self.current_model_info.get("speaker_id", None)
            
            # Generate speech
            if speaker_id:
                wav = self.model.tts(text, speaker_id=speaker_id)
            else:
                wav = self.model.tts(text)
            
            # Save to file
            import soundfile as sf
            sf.write(output_path, wav, 22050)
            
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
            logger.error(f"Error in speech synthesis: {str(e)}")
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
    def adjust_voice(self, rate: Optional[float] = None, pitch: Optional[float] = None,
                    volume: Optional[float] = None) -> None:
        """
        Adjust voice properties.
        
        Args:
            rate: Speech rate multiplier (1.0 is normal speed)
            pitch: Voice pitch multiplier (1.0 is normal pitch)
            volume: Volume level (1.0 is normal volume)
        """
        # Coqui-TTS doesn't support direct voice parameter adjustment
        # Instead, these adjustments would be applied during post-processing
        logger.info("Voice parameter adjustment in Coqui-TTS is handled during post-processing")
    
    @handle_tts_errors
    def list_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        List available voices.
        
        Returns:
            Dictionary mapping voice IDs to voice information
        """
        voices = {}
        
        # Convert available models to presets
        for model_key, model_info in self.model_manager.list_available_models().items():
            voices[model_key] = {
                "name": f"{model_info['language'].upper()} {model_info['voice_type'].capitalize()}",
                "language": model_info["language"],
                "gender": model_info["voice_type"],
                "downloaded": model_info["downloaded"]
            }
            
        return voices
    
    @handle_tts_errors
    def list_languages(self) -> List[str]:
        """
        List supported languages.
        
        Returns:
            List of supported language codes
        """
        languages = set()
        
        for model_info in self.model_manager.AVAILABLE_MODELS.values():
            languages.add(model_info["language"])
            
        return list(languages)
        
    @handle_tts_errors
    def cleanup(self):
        """Release resources and clear model cache."""
        # Release model resources
        self.model = None
        
        # Clear the cache if model manager supports it
        if hasattr(self.model_manager, 'clear_cache'):
            self.model_manager.clear_cache()
            
        logger.info("Cleaned up Coqui TTS resources") 