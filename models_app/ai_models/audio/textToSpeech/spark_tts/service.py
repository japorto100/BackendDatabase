"""
Service for text-to-speech conversion using Spark-TTS.
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

from .model_manager import SparkTTSModelManager

logger = logging.getLogger(__name__)

class SparkTTSService(BaseTTSService):
    """
    Service for text-to-speech conversion using Spark-TTS.
    
    Handles:
    - Loading the Spark-TTS model
    - Converting text to speech
    - Voice cloning from reference audio
    - Voice customization with parameters
    """
    
    def __init__(self, voice_id: str = "default", model_name: str = SparkTTSModelManager.DEFAULT_MODEL_NAME,
                device: Optional[str] = None, cache_dir: Optional[str] = None, language: str = "en"):
        """
        Initialize the SparkTTSService.
        
        Args:
            voice_id: ID of the voice to use
            model_name: Name of the Spark-TTS model to use
            device: Device to use for inference ('cpu', 'cuda', etc.)
            cache_dir: Directory to cache models
            language: Language code for synthesis
        """
        # Initialize the base class
        super().__init__(voice_id=voice_id, device=device, cache_dir=cache_dir, language=language)
        
        self.model_name = model_name
        self.model_manager = SparkTTSModelManager()
        self.model = None
        self.tokenizer = None
        self.output_dir = os.path.join(settings.MEDIA_ROOT, 'tts_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Voice parameters with defaults
        self.voice_params = {
            "gender": "neutral",
            "pitch": 0.0,
            "speaking_rate": 1.0,
            "energy": 1.0,
            "voice_age": "adult"
        }
        
    @handle_tts_errors
    def _load_model(self) -> bool:
        """
        Load the Spark-TTS model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.model is not None:
            # Model already loaded
            return True
        
        # Ensure model is downloaded
        if not self.model_manager.ensure_model_available(self.model_name):
            logger.error(f"Failed to download model {self.model_name}")
            raise TTSError(
                message=f"Failed to download model {self.model_name}",
                details={"model_name": self.model_name}
            )
        
        model_path = self.model_manager.get_model_path(self.model_name)
        if not model_path:
            logger.error(f"Model path not found for {self.model_name}")
            raise TTSError(
                message=f"Model path not found for {self.model_name}",
                details={"model_name": self.model_name}
            )
        
        try:
            # Import packages here to avoid loading them unless necessary
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer and model
            logger.info(f"Loading Spark-TTS model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=self.device if self.device != "cpu" else None,
                torch_dtype=torch.float16
            )
            
            # Import inference components
            try:
                from sparktts.cli.inference import load_audio, get_text_completion
            except ImportError:
                # Need to install Spark-TTS package
                logger.warning("Installing Spark-TTS package")
                import subprocess
                import sys
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "git+https://github.com/SparkAudio/Spark-TTS.git"
                ])
                # Try importing again after installation
                from sparktts.cli.inference import load_audio, get_text_completion
            
            self._load_audio = load_audio
            self._get_text_completion = get_text_completion
            
            logger.info(f"Successfully loaded model {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            self.model = None
            self.tokenizer = None
            raise TTSError(
                message=f"Error loading Spark-TTS model: {str(e)}",
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
        
        # Load model if not already loaded
        self._load_model()
        
        # Apply voice parameters if provided
        if voice_id and voice_id != self.voice_id:
            self.voice_id = voice_id
            
        # Set output path if not provided
        if not output_path:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name
        
        try:
            import torch
            from sparktts.cli.inference import synthesize
            
            # Run speech synthesis
            synthesize(
                model=self.model,
                tokenizer=self.tokenizer,
                text=text,
                device=self.device,
                save_path=output_path
            )
            
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
    def clone_voice(self, text: str, reference_audio_path: str, 
                   output_path: Optional[str] = None) -> str:
        """
        Clone a voice from reference audio and synthesize new speech.
        
        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio file for voice cloning
            output_path: Path to save audio file. If None, generates a unique path.
            
        Returns:
            Path to the generated audio file
        """
        # Load model if not already loaded
        self._load_model()
        
        if not os.path.exists(reference_audio_path):
            raise TTSError(
                message=f"Reference audio file not found: {reference_audio_path}",
                details={"reference_audio_path": reference_audio_path}
            )
        
        if not output_path:
            output_path = os.path.join(self.output_dir, f"{uuid.uuid4()}.wav")
        
        try:
            import torch
            from sparktts.cli.inference import inference
            
            device_int = 0 if self.device == "cuda" else -1
            
            # Extract reference prompt text and audio
            # In a real implementation, you would need to know what text corresponds to the audio
            # Here we'll use a placeholder that will need to be replaced with actual text
            prompt_text = "This is a reference audio sample."
            
            # Run inference with voice cloning
            inference(
                text=text,
                device=device_int,
                save_dir=os.path.dirname(output_path),
                save_name=os.path.basename(output_path),
                model_dir=self.model_manager.get_model_path(self.model_name),
                prompt_text=prompt_text,
                prompt_speech_path=reference_audio_path
            )
            
            if not os.path.exists(output_path):
                raise TTSError(
                    message="Voice cloning failed - output file not created",
                    details={"text": text, "reference_audio": reference_audio_path}
                )
                
            logger.info(f"Voice cloning successful, saved to {output_path}")
            return output_path
                
        except Exception as e:
            logger.error(f"Error in voice cloning: {str(e)}")
            raise TTSError(
                message=f"Failed to clone voice: {str(e)}",
                details={"text": text, "reference_audio": reference_audio_path}
            ) from e
    
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
        if rate is not None:
            self.voice_params["speaking_rate"] = max(0.5, min(2.0, rate))
        
        if pitch is not None:
            self.voice_params["pitch"] = max(-1.0, min(1.0, pitch))
            
        if volume is not None:
            self.voice_params["energy"] = max(0.5, min(1.5, volume))
    
    @handle_tts_errors
    def list_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        List available voice presets.
        
        Returns:
            Dictionary mapping voice IDs to voice information
        """
        return {
            "default": {
                "name": "Default Voice",
                "gender": "neutral",
                "language": "en"
            },
            "male_adult": {
                "name": "Male Adult",
                "gender": "male",
                "voice_age": "adult",
                "language": "en"
            },
            "female_adult": {
                "name": "Female Adult",
                "gender": "female",
                "voice_age": "adult",
                "language": "en"
            },
            "child": {
                "name": "Child Voice",
                "gender": "neutral",
                "voice_age": "child",
                "language": "en"
            },
            "senior": {
                "name": "Senior Voice",
                "gender": "neutral",
                "voice_age": "senior",
                "language": "en"
            }
        }
    
    @handle_tts_errors
    def cleanup(self):
        """Release resources and clear model cache."""
        # Release model resources
        self.model = None
        self.tokenizer = None
        
        # Clear the cache if model manager supports it
        if hasattr(self.model_manager, 'clear_cache'):
            self.model_manager.clear_cache()
            
        logger.info("Cleaned up Spark TTS resources") 