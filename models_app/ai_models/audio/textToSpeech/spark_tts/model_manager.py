"""
Manages Spark-TTS models, including downloading, updating, and version tracking.
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from django.conf import settings

logger = logging.getLogger(__name__)

class SparkTTSModelManager:
    """
    Manages the downloading and versioning of Spark-TTS models.
    
    This class handles:
    - Checking if models are downloaded
    - Downloading models from Hugging Face
    - Tracking model versions
    - Ensuring model files are in the correct location
    """
    
    DEFAULT_MODEL_NAME = "Spark-TTS-0.5B"
    DEFAULT_MODEL_REPO = "SparkAudio/Spark-TTS-0.5B"
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the SparkTTSModelManager.
        
        Args:
            model_dir: Directory to store models. If None, uses default location.
        """
        self.model_dir = model_dir or os.path.join(settings.BASE_DIR, "pretrained_models")
        self.ensure_dir_exists(self.model_dir)
        self.models = self._scan_available_models()
        
    def ensure_dir_exists(self, directory: str) -> None:
        """Create directory if it doesn't exist."""
        os.makedirs(directory, exist_ok=True)
        
    def _scan_available_models(self) -> Dict[str, str]:
        """
        Scan the model directory for available models.
        
        Returns:
            Dict mapping model names to their paths
        """
        models = {}
        if os.path.exists(self.model_dir):
            for item in os.listdir(self.model_dir):
                item_path = os.path.join(self.model_dir, item)
                if os.path.isdir(item_path):
                    # Check if this is a valid Spark-TTS model directory
                    if self._is_valid_model_dir(item_path):
                        models[item] = item_path
        return models
    
    def _is_valid_model_dir(self, directory: str) -> bool:
        """
        Check if a directory contains a valid Spark-TTS model.
        
        Args:
            directory: Path to check
            
        Returns:
            True if directory contains a valid model
        """
        # Check for essential files that indicate this is a Spark-TTS model
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(directory, file)):
                return False
        return True
    
    def download_model(self, model_name: str = DEFAULT_MODEL_NAME, 
                      model_repo: str = DEFAULT_MODEL_REPO) -> bool:
        """
        Download a Spark-TTS model from Hugging Face.
        
        Args:
            model_name: Name to save the model as
            model_repo: Hugging Face repository path
            
        Returns:
            True if download successful, False otherwise
        """
        target_dir = os.path.join(self.model_dir, model_name)
        
        # Check if model already exists
        if os.path.exists(target_dir) and self._is_valid_model_dir(target_dir):
            logger.info(f"Model {model_name} already exists at {target_dir}")
            return True
            
        # Ensure model directory exists
        self.ensure_dir_exists(self.model_dir)
        
        try:
            # Use huggingface_hub to download the model
            logger.info(f"Downloading model {model_repo} to {target_dir}")
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id=model_repo,
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            
            # Verify the download
            if self._is_valid_model_dir(target_dir):
                logger.info(f"Successfully downloaded model {model_name}")
                # Update available models
                self.models[model_name] = target_dir
                return True
            else:
                logger.error(f"Downloaded model {model_name} is invalid")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading model {model_repo}: {str(e)}")
            return False
    
    def get_model_path(self, model_name: str = DEFAULT_MODEL_NAME) -> Optional[str]:
        """
        Get the path to a downloaded model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to the model directory or None if not found
        """
        return self.models.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """
        List all available downloaded models.
        
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
            return True
        
        # Model not available, try to download it
        return self.download_model(model_name)
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """
        Check if all required dependencies for Spark-TTS are installed.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Try importing required packages
            import torch
            import numpy
            import tqdm
            import huggingface_hub
            return True, "All dependencies are satisfied"
        except ImportError as e:
            missing_package = str(e).split("'")[1]
            return False, f"Missing dependency: {missing_package}" 