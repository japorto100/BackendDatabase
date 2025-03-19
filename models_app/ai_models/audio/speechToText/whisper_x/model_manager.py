"""
Model manager for WhisperX.

This module handles the loading, caching, and management of models
for the WhisperX implementation, including ASR and diarization models.
"""

import os
import logging
from typing import Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class WhisperXModelManager:
    """
    Manager for WhisperX models.
    
    This class handles model loading, caching, verification,
    and resource management for WhisperX models, including
    ASR models, alignment models, and diarization models.
    """
    
    # Model size to parameter count mapping (for reference)
    MODEL_SIZES = {
        "tiny": "39M",
        "base": "74M",
        "small": "244M",
        "medium": "769M",
        "large-v2": "1550M"
    }
    
    def __init__(self, cache_dir: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            cache_dir: Directory to cache models. If None, uses default HF cache.
            device: Device to load models on ('cpu', 'cuda')
        """
        self.device = device or self._get_default_device()
        self.cache_dir = cache_dir
        
        # Map of loaded models keyed by model_id + device + compute_type
        self.asr_models: Dict[str, Any] = {}
        self.alignment_models: Dict[str, Any] = {}
        self.diarization_model = None
        
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
            compute_type: Compute type ('float16', 'float32')
            
        Returns:
            String key for caching the model
        """
        return f"{model_size}_{self.device}_{compute_type}"
    
    def load_asr_model(self, model_size: str, compute_type: str = "float16") -> Any:
        """
        Load a WhisperX ASR model with caching.
        
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
        if model_key in self.asr_models:
            logger.info(f"Using cached ASR model: {model_key}")
            return self.asr_models[model_key]
        
        try:
            import whisperx
            
            # Try to verify the installation
            try:
                import torch
                logger.info(f"Using PyTorch version: {torch.__version__}")
                logger.info(f"Using WhisperX")
            except (ImportError, AttributeError):
                logger.warning("Could not verify WhisperX installation details")
            
            # Load ASR model
            logger.info(f"Loading WhisperX ASR model {model_size} on {self.device} with {compute_type}")
            model = whisperx.load_model(
                model_size,
                self.device,
                compute_type=compute_type,
                download_root=self.cache_dir
            )
            
            # Cache the loaded model
            self.asr_models[model_key] = model
            
            logger.info(f"Successfully loaded and cached ASR model: {model_key}")
            return model
            
        except ImportError as e:
            logger.error(f"Failed to import required packages: {e}")
            raise ImportError(
                "WhisperX requires the whisperx package. "
                "Install with: pip install git+https://github.com/m-bain/whisperx.git"
            )
        except Exception as e:
            logger.error(f"Failed to load ASR model {model_size}: {e}")
            raise RuntimeError(f"Failed to load ASR model: {e}")
    
    def load_alignment_model(self, language_code: str) -> Any:
        """
        Load a WhisperX alignment model with caching.
        
        Args:
            language_code: Language code for the alignment model
            
        Returns:
            Loaded alignment model
            
        Raises:
            ImportError: If required packages are not installed
            RuntimeError: If model loading fails
        """
        model_key = f"align_{language_code}_{self.device}"
        
        # Return cached model if available
        if model_key in self.alignment_models:
            logger.info(f"Using cached alignment model: {model_key}")
            return self.alignment_models[model_key]
        
        try:
            import whisperx
            
            # Load alignment model
            logger.info(f"Loading alignment model for language '{language_code}' on {self.device}")
            model = whisperx.load_align_model(
                language_code=language_code,
                device=self.device
            )
            
            # Cache the loaded model
            self.alignment_models[model_key] = model
            
            logger.info(f"Successfully loaded and cached alignment model: {model_key}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load alignment model for language '{language_code}': {e}")
            raise RuntimeError(f"Failed to load alignment model: {e}")
    
    def load_diarization_model(self) -> Any:
        """
        Load a WhisperX diarization model with caching.
        
        Returns:
            Loaded diarization model
            
        Raises:
            ImportError: If required packages are not installed
            RuntimeError: If model loading fails
        """
        # Return cached model if available
        if self.diarization_model is not None:
            logger.info("Using cached diarization model")
            return self.diarization_model
        
        try:
            import whisperx
            
            # Load diarization model
            logger.info(f"Loading diarization model on {self.device}")
            model = whisperx.DiarizationPipeline(
                use_auth_token=None,
                device=self.device
            )
            
            # Cache the loaded model
            self.diarization_model = model
            
            logger.info("Successfully loaded and cached diarization model")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load diarization model: {e}")
            raise RuntimeError(f"Failed to load diarization model: {e}")
    
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
            
            model_id = f"openai/whisper-{model_size}"
            
            # Try to download model snapshot to verify it exists
            try:
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=self.cache_dir
                )
                return True
            except Exception as download_err:
                logger.error(f"Failed to download/verify model: {download_err}")
                return False
                
        except Exception as e:
            logger.error(f"Error verifying model files: {e}")
            return False
    
    def clear_cache(self, model_type: Optional[str] = None, language_code: Optional[str] = None):
        """
        Clear cached models from memory.
        
        Args:
            model_type: Type of model to clear ('asr', 'align', 'diarize'), or None for all
            language_code: For alignment models, language code to clear
        """
        if model_type == 'asr' or model_type is None:
            # Clear ASR models
            logger.info("Clearing ASR model cache")
            self.asr_models.clear()
            
        if model_type == 'align' or model_type is None:
            # Clear alignment models
            if language_code:
                # Clear only specified language
                keys_to_clear = [
                    key for key in self.alignment_models.keys() 
                    if key.startswith(f"align_{language_code}_")
                ]
                
                for key in keys_to_clear:
                    if key in self.alignment_models:
                        logger.info(f"Clearing cached alignment model: {key}")
                        del self.alignment_models[key]
            else:
                # Clear all alignment models
                logger.info("Clearing all alignment models")
                self.alignment_models.clear()
                
        if model_type == 'diarize' or model_type is None:
            # Clear diarization model
            if self.diarization_model is not None:
                logger.info("Clearing diarization model")
                self.diarization_model = None
            
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