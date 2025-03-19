"""
QwenVisionModelManager

Manages Qwen Vision models, focusing on Qwen2-VL, a powerful multimodal model.
"""

import os
import logging
import torch
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class QwenVisionModelManager:
    """
    Manages configuration and initialization of Qwen Vision models.
    
    This class:
    1. Tracks available Qwen vision models
    2. Handles model loading and configuration
    3. Manages device placement and optimization
    """
    
    # Available model configurations
    AVAILABLE_MODELS = {
        'Qwen/Qwen2-VL-7B': {
            'description': 'Qwen2 Vision-Language model (7B parameters)',
            'class': 'Qwen2VLForConditionalGeneration',
            'processor': 'AutoProcessor',
            'context_length': 4096,
            'requirements': {
                'min_ram_mb': 16000,
                'min_gpu_mb': 14000,
                'preferred_device': 'cuda'
            }
        },
        'Qwen/Qwen2-VL-7B-Instruct': {
            'description': 'Qwen2 Vision-Language model with instruction tuning (7B parameters)',
            'class': 'Qwen2VLForConditionalGeneration',
            'processor': 'AutoProcessor',
            'context_length': 4096,
            'requirements': {
                'min_ram_mb': 16000,
                'min_gpu_mb': 14000,
                'preferred_device': 'cuda'
            }
        },
        'Qwen/Qwen-VL': {
            'description': 'Original Qwen Vision-Language model',
            'class': 'QwenVLForConditionalGeneration',
            'processor': 'QwenVLProcessor',
            'context_length': 2048,
            'requirements': {
                'min_ram_mb': 8000,
                'min_gpu_mb': 8000,
                'preferred_device': 'cuda'
            }
        },
        'Qwen/Qwen-VL-Chat': {
            'description': 'Original Qwen Vision-Language model with chat capability',
            'class': 'QwenVLForConditionalGeneration',
            'processor': 'QwenVLProcessor',
            'context_length': 2048,
            'requirements': {
                'min_ram_mb': 8000,
                'min_gpu_mb': 8000,
                'preferred_device': 'cuda'
            }
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Qwen Vision model manager.
        
        Args:
            config: Configuration dictionary including model name, quantization settings
        """
        self.config = config or {}
        self.model_name = self.config.get('model', 'Qwen/Qwen2-VL-7B-Instruct')
        
        # If model specified doesn't include path, check if it's just the variant name
        if '/' not in self.model_name and self.model_name not in self.AVAILABLE_MODELS:
            # Try to match with a known model
            matching_models = [name for name in self.AVAILABLE_MODELS.keys() if self.model_name in name]
            if matching_models:
                self.model_name = matching_models[0]
                logger.info(f"Using model {self.model_name} based on specified model: {self.config.get('model')}")
        
        # Set quantization level
        self.quantization_level = self.config.get('quantization_level', 'none').lower()
        
        # Determine device
        self.device_name = self.config.get('device', 'auto')
        if self.device_name == 'auto':
            self.device = self._get_best_device()
        else:
            self.device = torch.device(self.device_name)
        
        # Track initialization state
        self.initialized = False
        self.model = None
        self.processor = None
    
    def _get_best_device(self) -> torch.device:
        """
        Determine the best available device for the Qwen model.
        
        Returns:
            torch.device: The most suitable device
        """
        # Check CUDA availability
        if torch.cuda.is_available():
            # Get model requirements
            if self.model_name in self.AVAILABLE_MODELS:
                requirements = self.AVAILABLE_MODELS[self.model_name]['requirements']
                preferred_device = requirements.get('preferred_device', 'cuda')
                
                if preferred_device == 'cuda':
                    # Check if enough GPU memory is available
                    required_mem = requirements.get('min_gpu_mb', 0)
                    try:
                        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                        if gpu_mem >= required_mem:
                            return torch.device('cuda')
                    except:
                        pass
            
            # Default to CUDA if above checks don't apply
            return torch.device('cuda')
            
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
            
        # Fall back to CPU
        return torch.device('cpu')
    
    def is_available(self) -> bool:
        """
        Check if the selected Qwen Vision model is available on this system.
        
        Returns:
            bool: True if the model can be used, False otherwise
        """
        # Check if the model is known
        if self.model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown Qwen Vision model: {self.model_name}")
            return False
        
        # Check hardware requirements
        requirements = self.AVAILABLE_MODELS[self.model_name]['requirements']
        
        # Check RAM requirements
        try:
            import psutil
            available_ram = psutil.virtual_memory().available / (1024 * 1024)
            if available_ram < requirements['min_ram_mb']:
                logger.warning(f"Insufficient RAM for {self.model_name}. Required: {requirements['min_ram_mb']}MB, Available: {available_ram:.0f}MB")
                return False
        except ImportError:
            # Can't check RAM, assume it's sufficient
            pass
        
        # If model requires GPU, check if it's available with sufficient memory
        if requirements['min_gpu_mb'] > 0 and requirements['preferred_device'] == 'cuda':
            if not torch.cuda.is_available():
                logger.warning(f"GPU required for {self.model_name} but not available")
                # Allow CPU fallback with strong warning
                logger.warning("Running on CPU may be extremely slow and memory intensive")
            else:
                try:
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    if gpu_mem < requirements['min_gpu_mb']:
                        logger.warning(f"Insufficient GPU memory for optimal {self.model_name} performance. Recommended: {requirements['min_gpu_mb']}MB, Available: {gpu_mem:.0f}MB")
                        # Don't return False as it might still work with reduced performance
                except:
                    pass
        
        # If we already have an initialized model, return True
        if self.initialized and self.model is not None:
            return True
        
        # Otherwise, try checking import
        try:
            # Just check import, don't actually load the model
            import transformers
            return True
        except ImportError as e:
            logger.warning(f"Required package 'transformers' not available: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured Qwen Vision model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        if self.model_name in self.AVAILABLE_MODELS:
            model_info = self.AVAILABLE_MODELS[self.model_name].copy()
            model_info.update({
                'name': self.model_name,
                'device': str(self.device),
                'quantization': self.quantization_level,
                'initialized': self.initialized
            })
            return model_info
        else:
            return {
                'name': self.model_name,
                'error': f"Unknown model: {self.model_name}",
                'initialized': False
            }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available Qwen Vision models.
        
        Returns:
            List[Dict[str, Any]]: Information about available models
        """
        result = []
        for name, info in self.AVAILABLE_MODELS.items():
            model_entry = {
                'name': name,
                'description': info['description'],
                'requirements': info['requirements']
            }
            result.append(model_entry)
        return result
    
    def initialize_model(self) -> Tuple[Any, Any]:
        """
        Initialize the selected Qwen Vision model.
        
        Returns:
            Tuple[Any, Any]: Tuple of (model, processor)
            
        Raises:
            ImportError: If required packages are not installed
            RuntimeError: If model initialization fails
        """
        if self.initialized and self.model is not None and self.processor is not None:
            return self.model, self.processor
        
        # Check if model is known
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown Qwen Vision model: {self.model_name}")
        
        # Get model configuration
        model_config = self.AVAILABLE_MODELS[self.model_name]
        model_class = model_config['class']
        processor_class = model_config['processor']
        
        try:
            # Import necessary classes
            from transformers import AutoProcessor, AutoTokenizer
            
            # Import the correct model class
            if model_class == 'Qwen2VLForConditionalGeneration':
                from transformers import Qwen2VLForConditionalGeneration as ModelClass
            elif model_class == 'QwenVLForConditionalGeneration':
                from transformers import QwenVLForConditionalGeneration as ModelClass
            else:
                raise ImportError(f"Unknown model class: {model_class}")
            
            # Import the correct processor class
            if processor_class == 'AutoProcessor':
                from transformers import AutoProcessor as ProcessorClass
            elif processor_class == 'QwenVLProcessor':
                from transformers import QwenVLProcessor as ProcessorClass
            else:
                raise ImportError(f"Unknown processor class: {processor_class}")
            
            logger.info(f"Loading Qwen Vision model: {self.model_name}")
            
            # Load the processor
            self.processor = ProcessorClass.from_pretrained(self.model_name)
            
            # Determine torch dtype based on device and quantization
            torch_dtype = torch.float16 if self.device.type != 'cpu' else torch.float32
            
            # Load the model with appropriate quantization
            if self.quantization_level == '8bit':
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    load_in_8bit=True,
                    device_map="auto"
                )
            elif self.quantization_level == '4bit':
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    load_in_4bit=True,
                    device_map="auto"
                )
            else:
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype
                ).to(self.device)
            
            self.initialized = True
            logger.info(f"Successfully initialized Qwen Vision model: {self.model_name} on {self.device}")
            
            return self.model, self.processor
            
        except ImportError as e:
            logger.error(f"Failed to import required library: {e}")
            raise
        except Exception as e:
            logger.error(f"Error initializing Qwen Vision model: {e}")
            raise RuntimeError(f"Failed to initialize Qwen Vision model: {str(e)}") 