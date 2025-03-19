"""
Manages Mozilla TTS models, including downloading and version tracking.
"""

import os
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from django.conf import settings

logger = logging.getLogger(__name__)

class MozillaTTSModelManager:
    """
    Manages the Mozilla TTS models.
    
    This class handles:
    - Checking if models are downloaded
    - Downloading models
    - Tracking model versions
    """
    
    DEFAULT_MODEL_NAME = "tts_en_ljspeech"
    MODELS_INFO = {
        "tts_en_ljspeech": {
            "url": "https://github.com/mozilla/TTS/releases/download/v0.0.12/tts_model_ljspeech.pth.tar",
            "config_url": "https://raw.githubusercontent.com/mozilla/TTS/v0.0.12/TTS/server/conf/config_ljspeech.json",
            "language": "en",
            "type": "female",
            "size_mb": 50
        },
        "tts_en_multispeaker": {
            "url": "https://github.com/mozilla/TTS/releases/download/v0.9.0/tts_models--en--vctk--vits.zip",
            "language": "en",
            "type": "multispeaker",
            "size_mb": 120
        },
        "tts_multilingual": {
            "url": "https://github.com/mozilla/TTS/releases/download/v0.9.0/tts_models--multilingual--multi-dataset--xtts_v2.zip",
            "language": "multilingual",
            "type": "xtts",
            "size_mb": 300
        }
    }
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the MozillaTTSModelManager.
        
        Args:
            model_dir: Directory to store models. If None, uses default location.
        """
        self.model_dir = model_dir or os.path.join(settings.BASE_DIR, "pretrained_models", "mozilla_tts")
        self.ensure_dir_exists(self.model_dir)
        self.models_file = os.path.join(self.model_dir, "models.json")
        self.models = self._load_models_info()
        
    def ensure_dir_exists(self, directory: str) -> None:
        """Create directory if it doesn't exist."""
        os.makedirs(directory, exist_ok=True)
    
    def _load_models_info(self) -> Dict[str, Dict]:
        """
        Load information about downloaded models.
        
        Returns:
            Dict mapping model names to their information
        """
        if os.path.exists(self.models_file):
            try:
                with open(self.models_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading models info: {str(e)}")
                return {}
        return {}
    
    def _save_models_info(self) -> None:
        """Save information about downloaded models."""
        try:
            with open(self.models_file, 'w') as f:
                json.dump(self.models, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving models info: {str(e)}")
    
    def download_model(self, model_name: str = DEFAULT_MODEL_NAME) -> bool:
        """
        Download a Mozilla TTS model.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            True if download successful, False otherwise
        """
        if model_name not in self.MODELS_INFO:
            logger.error(f"Model {model_name} not found in available models")
            return False
        
        model_info = self.MODELS_INFO[model_name]
        model_path = os.path.join(self.model_dir, model_name)
        
        # Check if model already exists
        if model_name in self.models and os.path.exists(model_path):
            logger.info(f"Model {model_name} already exists")
            return True
        
        # Ensure model directory exists
        self.ensure_dir_exists(model_path)
        
        try:
            # Try to import required packages
            try:
                import wget
                import TTS
            except ImportError:
                # Need to install Mozilla TTS
                logger.warning("Installing Mozilla TTS and dependencies")
                subprocess.check_call([
                    "pip", "install", "TTS", "wget"
                ])
                import wget
                import TTS
            
            # Download model
            logger.info(f"Downloading Mozilla TTS model {model_name}")
            model_url = model_info["url"]
            model_file = os.path.join(model_path, os.path.basename(model_url))
            wget.download(model_url, model_file)
            
            # Download config if available
            if "config_url" in model_info:
                config_url = model_info["config_url"]
                config_file = os.path.join(model_path, "config.json")
                wget.download(config_url, config_file)
            
            # Extract if it's a zip file
            if model_file.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(model_file, 'r') as zip_ref:
                    zip_ref.extractall(model_path)
            
            # Update models info
            self.models[model_name] = {
                "path": model_path,
                "language": model_info["language"],
                "type": model_info["type"]
            }
            self._save_models_info()
            
            return True
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}")
            return False
    
    def get_model_path(self, model_name: str = DEFAULT_MODEL_NAME) -> Optional[str]:
        """
        Get the path to a downloaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the model directory or None if not found
        """
        if model_name in self.models:
            return self.models[model_name]["path"]
        return None
    
    def list_available_models(self) -> Dict[str, Dict]:
        """
        List all available models (both downloaded and not downloaded).
        
        Returns:
            Dict mapping model names to their information
        """
        available_models = {}
        
        for model_name, model_info in self.MODELS_INFO.items():
            downloaded = model_name in self.models
            available_models[model_name] = {
                **model_info,
                "downloaded": downloaded
            }
            
        return available_models
    
    def list_downloaded_models(self) -> List[str]:
        """
        List all downloaded models.
        
        Returns:
            List of model names
        """
        return list(self.models.keys())
    
    def ensure_model_available(self, model_name: str = DEFAULT_MODEL_NAME) -> bool:
        """
        Ensure a model is available, downloading it if necessary.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is available, False otherwise
        """
        if model_name in self.models:
            model_path = self.models[model_name]["path"]
            if os.path.exists(model_path):
                return True
            
        # Model not available or path doesn't exist, try to download it
        return self.download_model(model_name)
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """
        Check if all required dependencies for Mozilla TTS are installed.
        
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