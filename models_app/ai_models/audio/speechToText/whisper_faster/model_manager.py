"""
Model manager for Faster Whisper.

This module handles the loading, caching, and management of models
for the Faster Whisper implementation.
"""

import os
import logging
from typing import Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class WhisperFasterModelManager:
    """
    Manager for Faster Whisper models.
    
    This class handles model loading, caching, verification,
    and resource management for Faster Whisper models.
    """
    
    # Model size to parameter count mapping (for reference)
    MODEL_SIZES = {
        "tiny": "39M",
        "base": "74M",
        "small": "244M",
        "medium": "769M",
        "large-v2": "1550M",
        "large": "1550M"  # For compatibility
    }
    
    def __init__(self, cache_dir: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            cache_dir: Directory to cache models. If None, uses default cache.
            device: Device to load models on ('cpu', 'cuda')
        """
        self.device = device or self._get_default_device()
        self.cache_dir = cache_dir
        
        # Map of loaded models keyed by model_id + device + compute_type
        self.models: Dict[str, Any] = {}
        
        # Create cache directory if specified and doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_default_device(self) -> str:
        """Get the default device based on available hardware."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"
    
    def _get_model_key(self, model_size: str, compute_type: str) -> str:
        """
        Generate a unique key for the model based on size, device, and compute type.
        
        Args:
            model_size: Size of the model ('tiny', 'base', 'small', etc.)
            compute_type: Compute type ('float16', 'float32', 'int8')
            
        Returns:
            String key for caching the model
        """
        return f"{model_size}_{self.device}_{compute_type}"
    
    def load_model(self, model_size: str, compute_type: str = "float16") -> Any:
        """
        Load a Faster Whisper model with caching.
        
        Args:
            model_size: Size of the model to load
            compute_type: Compute type for model inference
            
        Returns:
            Loaded model
            
        Raises:
            ImportError: If required packages are not installed
            RuntimeError: If model loading fails
        """
        model_key = self._get_model_key(model_size, compute_type)
        
        # Return cached model if available
        if model_key in self.models:
            logger.info(f"Using cached model: {model_key}")
            return self.models[model_key]
        
        try:
            from faster_whisper import WhisperModel
            
            # Try to verify the installation
            try:
                import ctranslate2
                logger.info(f"Using CTranslate2 version: {ctranslate2.__version__}")
            except (ImportError, AttributeError):
                logger.warning("Could not import ctranslate2 directly")
            
            # Determine model ID
            model_id = model_size
            
            # Load model
            logger.info(f"Loading model {model_id} on {self.device} with {compute_type}")
            model = WhisperModel(
                model_id,
                device=self.device,
                compute_type=compute_type,
                download_root=self.cache_dir
            )
            
            # Cache the loaded model
            self.models[model_key] = model
            
            logger.info(f"Successfully loaded and cached model: {model_key}")
            return model
            
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            raise ImportError(
                "Faster Whisper requires the faster-whisper package. "
                "Install with: pip install faster-whisper"
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_size}: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def verify_model_files(self, model_size: str) -> bool:
        """
        Verify that the model files exist and are complete.
        
        Args:
            model_size: Size of the model to verify
            
        Returns:
            True if model files are verified, False otherwise
        """
        try:
            model_path = os.path.join(self.cache_dir, model_size) if self.cache_dir else None
            
            # Try to check if model exists in cache
            if model_path and os.path.exists(model_path):
                # Check for key files that should be present
                required_files = ["model.bin", "tokenizer.json", "config.json"]
                for file in required_files:
                    if not os.path.exists(os.path.join(model_path, file)):
                        logger.warning(f"Model file missing: {file}")
                        return False
                return True
            
            # If model doesn't exist, try to load it which will download if needed
            from faster_whisper import download_model
            download_model(model_size, self.cache_dir)
            return True
            
        except Exception as e:
            logger.error(f"Error verifying model files: {e}")
            return False
    
    def clear_cache(self, model_size: Optional[str] = None):
        """
        Clear cached models from memory.
        
        Args:
            model_size: Specific model size to clear, or None for all models
        """
        if model_size:
            # Clear specific model size from cache
            keys_to_clear = [
                key for key in self.models.keys() 
                if key.startswith(f"{model_size}_")
            ]
            
            for key in keys_to_clear:
                if key in self.models:
                    logger.info(f"Clearing cached model: {key}")
                    # Release CUDA memory if applicable
                    model = self.models[key]
                    del model
                    del self.models[key]
        else:
            # Clear all models
            logger.info("Clearing all cached models")
            self.models.clear()
            
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def get_model_info(self, model_size: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_size: Size of the model
            
        Returns:
            Dictionary with model information
        """
        return {
            "size": model_size,
            "parameters": self.MODEL_SIZES.get(model_size, "Unknown"),
            "is_downloaded": self.verify_model_files(model_size),
            "device": self.device
        }
    
    def list_available_models(self) -> Dict[str, str]:
        """
        List all available Whisper models.
        
        Returns:
            Dictionary mapping model IDs to descriptive names
        """
        result = {}
        for size, params in self.MODEL_SIZES.items():
            result[size] = f"{size.capitalize()} ({params} parameters)"
        return result 