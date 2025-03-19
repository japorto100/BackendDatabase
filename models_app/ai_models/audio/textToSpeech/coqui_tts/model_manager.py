"""
Manages Coqui-TTS models, including downloading, updating, and version tracking.
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from django.conf import settings

logger = logging.getLogger(__name__)

class CoquiTTSModelManager:
    """
    Manages the downloading and versioning of Coqui-TTS models.
    
    This class handles:
    - Checking if models are downloaded
    - Downloading models from Hugging Face
    - Tracking model versions
    - Ensuring model files are in the correct location
    """
    
    DEFAULT_MODEL_NAME = "tts_models--en--ljspeech--tacotron2-DDC"
    DEFAULT_VOCODER_NAME = "vocoder_models--en--ljspeech--multiband-melgan"
    AVAILABLE_MODELS = {
        "en_female": {
            "model": "tts_models--en--ljspeech--tacotron2-DDC",
            "vocoder": "vocoder_models--en--ljspeech--multiband-melgan",
            "language": "en",
            "voice_type": "female"
        },
        "en_male": {
            "model": "tts_models--en--vctk--vits",
            "vocoder": None,  # VITS doesn't need a separate vocoder
            "language": "en",
            "voice_type": "male",
            "speaker_id": "p273"  # Male voice from VCTK
        },
        "de_female": {
            "model": "tts_models--de--thorsten--tacotron2-DDC",
            "vocoder": "vocoder_models--de--thorsten--fullband-melgan",
            "language": "de",
            "voice_type": "female"
        },
        "fr_female": {
            "model": "tts_models--fr--mai--tacotron2-DDC",
            "vocoder": "vocoder_models--universal--libri-tts--fullband-melgan",
            "language": "fr",
            "voice_type": "female"
        },
        "es_female": {
            "model": "tts_models--es--css10--vits",
            "vocoder": None,
            "language": "es",
            "voice_type": "female"
        }
    }
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the CoquiTTSModelManager.
        
        Args:
            model_dir: Directory to store models. If None, uses default location.
        """
        self.model_dir = model_dir or os.path.join(settings.BASE_DIR, "pretrained_models", "coqui_tts")
        self.ensure_dir_exists(self.model_dir)
        self.models = self._scan_available_models()
        
    def ensure_dir_exists(self, directory: str) -> None:
        """Create directory if it doesn't exist."""
        os.makedirs(directory, exist_ok=True)
        
    def _scan_available_models(self) -> Dict[str, Dict]:
        """
        Scan the model directory for available models.
        
        Returns:
            Dict mapping model keys to their information
        """
        downloaded_models = {}
        
        for model_key, model_info in self.AVAILABLE_MODELS.items():
            model_name = model_info["model"]
            vocoder_name = model_info["vocoder"]
            
            model_path = os.path.join(self.model_dir, model_name)
            
            # Check if model exists
            if os.path.exists(model_path) and os.path.isdir(model_path):
                # Add model to downloaded models
                downloaded_models[model_key] = model_info.copy()
                downloaded_models[model_key]["path"] = model_path
                
                # Check if vocoder exists if required
                if vocoder_name:
                    vocoder_path = os.path.join(self.model_dir, vocoder_name)
                    if os.path.exists(vocoder_path) and os.path.isdir(vocoder_path):
                        downloaded_models[model_key]["vocoder_path"] = vocoder_path
                    else:
                        # Vocoder is missing, but required
                        downloaded_models[model_key]["vocoder_path"] = None
                        logger.warning(f"Model {model_key} is missing vocoder {vocoder_name}")
                else:
                    # No vocoder required
                    downloaded_models[model_key]["vocoder_path"] = None
                    
        return downloaded_models
    
    def download_model(self, model_key: str) -> bool:
        """
        Download a Coqui-TTS model.
        
        Args:
            model_key: Key of the model to download from AVAILABLE_MODELS
            
        Returns:
            True if download successful, False otherwise
        """
        if model_key not in self.AVAILABLE_MODELS:
            logger.error(f"Model key {model_key} not found in available models")
            return False
            
        model_info = self.AVAILABLE_MODELS[model_key]
        model_name = model_info["model"]
        vocoder_name = model_info["vocoder"]
        
        # Ensure model directory exists
        self.ensure_dir_exists(self.model_dir)
        
        try:
            # Import TTS to download models
            try:
                from TTS.utils.manage import ModelManager
                manager = ModelManager(os.path.join(self.model_dir, ".models.json"))
            except ImportError:
                # Need to install Coqui-TTS
                logger.warning("Installing Coqui-TTS package")
                subprocess.check_call([
                    "pip", "install", "TTS"
                ])
                from TTS.utils.manage import ModelManager
                manager = ModelManager(os.path.join(self.model_dir, ".models.json"))
            
            # Download model
            logger.info(f"Downloading Coqui-TTS model {model_name}")
            model_path, model_config = manager.download_model(model_name)
            
            # Download vocoder if needed
            vocoder_path = None
            if vocoder_name:
                logger.info(f"Downloading Coqui-TTS vocoder {vocoder_name}")
                vocoder_path, vocoder_config = manager.download_model(vocoder_name)
            
            # Update available models
            self.models[model_key] = {
                **model_info,
                "path": model_path,
                "vocoder_path": vocoder_path
            }
            
            return True
        except Exception as e:
            logger.error(f"Error downloading Coqui-TTS model {model_key}: {str(e)}")
            return False
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """
        Get information about a downloaded model.
        
        Args:
            model_key: Key of the model
            
        Returns:
            Dict with model information or None if not found
        """
        return self.models.get(model_key)
    
    def list_available_models(self) -> Dict[str, Dict]:
        """
        List all available models, both downloaded and not downloaded.
        
        Returns:
            Dict mapping model keys to their information
        """
        available_models = {}
        
        for model_key, model_info in self.AVAILABLE_MODELS.items():
            downloaded = model_key in self.models
            available_models[model_key] = {
                **model_info,
                "downloaded": downloaded
            }
            
        return available_models
    
    def list_downloaded_models(self) -> List[str]:
        """
        List all downloaded models.
        
        Returns:
            List of model keys
        """
        return list(self.models.keys())
    
    def ensure_model_available(self, model_key: str = "en_female") -> bool:
        """
        Ensure a model is available, downloading it if necessary.
        
        Args:
            model_key: Key of the model
            
        Returns:
            True if model is available, False otherwise
        """
        if model_key in self.models:
            return True
        
        # Model not available, try to download it
        return self.download_model(model_key)
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """
        Check if all required dependencies for Coqui-TTS are installed.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Try importing required packages
            import TTS
            return True, "All dependencies are satisfied"
        except ImportError as e:
            missing_package = str(e).split("'")[1]
            return False, f"Missing dependency: {missing_package}" 