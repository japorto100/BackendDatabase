"""
Model manager for Insanely Fast Whisper.

This module handles the loading, caching, and management of Whisper models
for the Insanely Fast Whisper implementation.
"""

import os
import logging
import hashlib
import json
from typing import Dict, Optional, Any, Tuple
import torch

logger = logging.getLogger(__name__)

class WhisperInsanelyFastModelManager:
    """
    Manager for Insanely Fast Whisper models.
    
    This class handles model loading, caching, verification,
    and resource management for Whisper models.
    """
    
    # Model size to parameter count mapping (for reference)
    MODEL_SIZES = {
        "tiny": "39M",
        "base": "74M",
        "small": "244M",
        "medium": "769M",
        "large-v2": "1550M",
        "large-v3": "1550M"
    }
    
    def __init__(self, cache_dir: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            cache_dir: Directory to cache models. If None, uses default HF cache.
            device: Device to load models on ('cpu', 'cuda', 'mps')
        """
        self.device = device or self._get_default_device()
        self.cache_dir = cache_dir
        
        # Map of loaded models keyed by model_id + device + compute_type
        self.models: Dict[str, Tuple[Any, Any]] = {}
        
        # Create cache directory if specified and doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_default_device(self) -> str:
        """Get the default device based on available hardware."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
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
    
    def load_model(self, model_size: str, compute_type: str = "float16") -> Tuple[Any, Any]:
        """
        Load a Whisper model and processor with caching.
        
        Args:
            model_size: Size of the model to load
            compute_type: Compute type for model inference
            
        Returns:
            Tuple of (model, processor)
            
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
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            from optimum.bettertransformer import BetterTransformer
            
            # Try to import insanely_fast_whisper as verification
            try:
                import insanely_fast_whisper
                logger.info(f"Using insanely-fast-whisper version: {insanely_fast_whisper.__version__}")
            except (ImportError, AttributeError):
                logger.warning("Could not import insanely-fast-whisper directly, "
                              "but continuing with transformers and optimum")
            
            # Determine model ID
            model_id = f"openai/whisper-{model_size}"
            
            # Determine torch dtype based on compute_type
            if compute_type == "float16":
                torch_dtype = torch.float16
            elif compute_type == "float32":
                torch_dtype = torch.float32
            elif compute_type == "int8":
                torch_dtype = torch.int8
            else:
                torch_dtype = torch.float16
            
            # Load processor
            logger.info(f"Loading processor for {model_id}")
            processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=self.cache_dir
            )
            
            # Load model
            logger.info(f"Loading model {model_id} on {self.device} with {compute_type}")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                cache_dir=self.cache_dir
            )
            
            # Apply BetterTransformer optimization if not on CPU
            if self.device != "cpu" and compute_type != "int8":
                model = BetterTransformer.transform(model)
            
            # Move model to device
            model = model.to(self.device)
            
            # Cache the loaded model
            self.models[model_key] = (model, processor)
            
            logger.info(f"Successfully loaded and cached model: {model_key}")
            return model, processor
            
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            raise ImportError(
                "Insanely Fast Whisper requires transformers and optimum-bettertransformer. "
                "Install with: pip install transformers optimum[bettertransformer] torch"
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
            from huggingface_hub import snapshot_download
            from transformers import AutoConfig
            
            model_id = f"openai/whisper-{model_size}"
            
            # Check if model is already downloaded
            try:
                config = AutoConfig.from_pretrained(
                    model_id, 
                    cache_dir=self.cache_dir
                )
                return True
            except Exception as e:
                logger.warning(f"Could not load model config, model may not be downloaded: {e}")
                
                # Try to download model snapshot
                try:
                    snapshot_download(
                        repo_id=model_id,
                        cache_dir=self.cache_dir
                    )
                    return True
                except Exception as download_err:
                    logger.error(f"Failed to download model: {download_err}")
                    return False
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
                    del self.models[key]
        else:
            # Clear all models
            logger.info("Clearing all cached models")
            self.models.clear()
            
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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