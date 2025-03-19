"""
LightweightVisionModelManager

Manages lightweight vision models that can run efficiently on modest hardware.
"""

import os
import logging
import torch
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class LightweightVisionModelManager:
    """
    Manages configuration and availability of lightweight vision models.
    
    This class:
    1. Tracks available models like CLIP, BLIP, and other lightweight models
    2. Handles model loading and configuration
    3. Manages model weights and caching
    """
    
    # Available model types and their configurations
    AVAILABLE_MODELS = {
        'clip': {
            'default': 'openai/clip-vit-base-patch32',
            'variants': {
                'base': 'openai/clip-vit-base-patch32',
                'large': 'openai/clip-vit-large-patch14',
                'small': 'openai/clip-vit-base-patch16'
            },
            'description': 'CLIP models for zero-shot image classification',
            'requirements': {
                'min_ram_mb': 2000,
                'min_gpu_mb': 2000,
                'preferred_device': 'cuda'
            }
        },
        'blip': {
            'default': 'Salesforce/blip-image-captioning-base',
            'variants': {
                'base': 'Salesforce/blip-image-captioning-base',
                'large': 'Salesforce/blip-image-captioning-large',
                'vqa': 'Salesforce/blip-vqa-base'
            },
            'description': 'BLIP models for image captioning and visual QA',
            'requirements': {
                'min_ram_mb': 4000,
                'min_gpu_mb': 4000,
                'preferred_device': 'cuda'
            }
        },
        'paligemma': {
            'default': 'google/paligemma-small-7b',
            'variants': {
                'small': 'google/paligemma-small-7b',
            },
            'description': 'PaliGemma models for multimodal tasks',
            'requirements': {
                'min_ram_mb': 8000,
                'min_gpu_mb': 8000,
                'preferred_device': 'cuda'
            }
        },
        'clip_phi': {
            'default': 'clip-phi-text-similarity',
            'variants': {
                'base': 'clip-phi-text-similarity',
            },
            'description': 'Combination of CLIP embeddings with Phi models for lightweight image understanding',
            'requirements': {
                'min_ram_mb': 2000,
                'min_gpu_mb': 0,  # Can run on CPU
                'preferred_device': 'cpu'
            }
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the lightweight vision model manager.
        
        Args:
            config: Configuration dictionary including model type, variant, and quantization settings
        """
        self.config = config or {}
        self.model_type = self.config.get('model_type', 'clip').lower()
        self.variant = self.config.get('variant', 'base').lower()
        self.quantization_level = self.config.get('quantization_level', 'none').lower()
        self.model_path = self.config.get('model_path', None)
        
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
        
        # Set the model path based on type and variant
        if not self.model_path:
            self._set_model_path()
    
    def _get_best_device(self) -> torch.device:
        """
        Determine the best available device for the selected model.
        
        Returns:
            torch.device: The most suitable device
        """
        # Check CUDA availability
        if torch.cuda.is_available():
            if self.model_type in self.AVAILABLE_MODELS:
                requirements = self.AVAILABLE_MODELS[self.model_type]['requirements']
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
    
    def _set_model_path(self):
        """Set the model path based on the selected model type and variant."""
        if self.model_type in self.AVAILABLE_MODELS:
            variants = self.AVAILABLE_MODELS[self.model_type]['variants']
            if self.variant in variants:
                self.model_path = variants[self.variant]
            else:
                # Fall back to default variant
                self.model_path = self.AVAILABLE_MODELS[self.model_type]['default']
                logger.warning(f"Unknown variant '{self.variant}' for model type '{self.model_type}'. Using default: {self.model_path}")
        else:
            # Fall back to default CLIP model
            self.model_type = 'clip'
            self.model_path = self.AVAILABLE_MODELS['clip']['default']
            logger.warning(f"Unknown model type: {self.model_type}. Using default CLIP model: {self.model_path}")
    
    def is_available(self) -> bool:
        """
        Check if the selected lightweight vision model is available on this system.
        
        Returns:
            bool: True if the model can be used, False otherwise
        """
        # Check if the model type is known
        if self.model_type not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown model type: {self.model_type}")
            return False
        
        # Check hardware requirements
        requirements = self.AVAILABLE_MODELS[self.model_type]['requirements']
        
        # Check RAM requirements
        try:
            import psutil
            available_ram = psutil.virtual_memory().available / (1024 * 1024)
            if available_ram < requirements['min_ram_mb']:
                logger.warning(f"Insufficient RAM for {self.model_type}. Required: {requirements['min_ram_mb']}MB, Available: {available_ram:.0f}MB")
                return False
        except ImportError:
            # Can't check RAM, assume it's sufficient
            pass
        
        # If model requires GPU, check if it's available with sufficient memory
        if requirements['min_gpu_mb'] > 0 and requirements['preferred_device'] == 'cuda':
            if not torch.cuda.is_available():
                logger.warning(f"GPU required for optimal {self.model_type} performance but not available")
                # Don't return False as it might still work on CPU
            else:
                try:
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    if gpu_mem < requirements['min_gpu_mb']:
                        logger.warning(f"Insufficient GPU memory for optimal {self.model_type} performance. Recommended: {requirements['min_gpu_mb']}MB, Available: {gpu_mem:.0f}MB")
                        # Don't return False as it might still work with reduced performance
                except:
                    pass
        
        # If we already have an initialized model, return True
        if self.initialized and self.model is not None:
            return True
        
        # Otherwise, try initializing it
        try:
            # Just check import, don't actually load the model
            if self.model_type == 'clip':
                import transformers.models.clip
            elif self.model_type == 'blip':
                import transformers.models.blip
            elif self.model_type == 'paligemma':
                import peft  # Needed for some paligemma variants
            
            return True
        except ImportError as e:
            logger.warning(f"Required package not available for {self.model_type}: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured lightweight vision model.
        
        Returns:
            Dict[str, Any]: Model information including type, path, and requirements
        """
        if self.model_type in self.AVAILABLE_MODELS:
            model_info = self.AVAILABLE_MODELS[self.model_type].copy()
            model_info.update({
                'type': self.model_type,
                'variant': self.variant,
                'path': self.model_path,
                'device': str(self.device),
                'quantization': self.quantization_level,
                'initialized': self.initialized
            })
            return model_info
        else:
            return {
                'type': self.model_type,
                'error': f"Unknown model type: {self.model_type}",
                'initialized': False
            }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available lightweight vision models.
        
        Returns:
            List[Dict[str, Any]]: Information about available models
        """
        result = []
        for model_type, info in self.AVAILABLE_MODELS.items():
            model_entry = {
                'type': model_type,
                'description': info['description'],
                'default_path': info['default'],
                'variants': list(info['variants'].keys()),
                'requirements': info['requirements']
            }
            result.append(model_entry)
        return result
    
    def initialize_model(self) -> Tuple[Any, Any]:
        """
        Initialize the selected lightweight vision model.
        
        Returns:
            Tuple[Any, Any]: Tuple of (model, processor)
            
        Raises:
            ImportError: If required packages are not installed
            RuntimeError: If model initialization fails
        """
        if self.initialized and self.model is not None and self.processor is not None:
            return self.model, self.processor
        
        try:
            if self.model_type == 'clip':
                return self._initialize_clip()
            elif self.model_type == 'blip':
                return self._initialize_blip()
            elif self.model_type == 'paligemma':
                return self._initialize_paligemma()
            elif self.model_type == 'clip_phi':
                return self._initialize_clip_phi()
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Error initializing {self.model_type} model: {e}")
            raise RuntimeError(f"Failed to initialize vision model: {str(e)}")
    
    def _initialize_clip(self) -> Tuple[Any, Any]:
        """Initialize a CLIP model for image understanding."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            # Load the processor
            self.processor = CLIPProcessor.from_pretrained(self.model_path)
            
            # Load the model with quantization if specified
            if self.quantization_level == '8bit':
                self.model = CLIPModel.from_pretrained(self.model_path, load_in_8bit=True, device_map=self.device)
            elif self.quantization_level == '4bit':
                self.model = CLIPModel.from_pretrained(self.model_path, load_in_4bit=True, device_map=self.device)
            else:
                self.model = CLIPModel.from_pretrained(self.model_path).to(self.device)
            
            self.initialized = True
            logger.info(f"Successfully initialized CLIP model: {self.model_path} on {self.device}")
            
            return self.model, self.processor
            
        except ImportError:
            logger.error("Failed to import transformers library. Please install it with: pip install transformers")
            raise
    
    def _initialize_blip(self) -> Tuple[Any, Any]:
        """Initialize a BLIP model for image captioning and visual QA."""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            # Load the processor
            self.processor = BlipProcessor.from_pretrained(self.model_path)
            
            # Load the model with quantization if specified
            if self.quantization_level == '8bit':
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_path, load_in_8bit=True, device_map=self.device
                )
            elif self.quantization_level == '4bit':
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_path, load_in_4bit=True, device_map=self.device
                )
            else:
                self.model = BlipForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
            
            self.initialized = True
            logger.info(f"Successfully initialized BLIP model: {self.model_path} on {self.device}")
            
            return self.model, self.processor
            
        except ImportError:
            logger.error("Failed to import transformers library. Please install it with: pip install transformers")
            raise
    
    def _initialize_paligemma(self) -> Tuple[Any, Any]:
        """Initialize a PaliGemma model for multimodal understanding."""
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # Load the model with quantization if specified
            if self.quantization_level == '8bit':
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path, load_in_8bit=True, device_map=self.device
                )
            elif self.quantization_level == '4bit':
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path, load_in_4bit=True, device_map=self.device
                )
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_path).to(self.device)
            
            self.initialized = True
            logger.info(f"Successfully initialized PaliGemma model: {self.model_path} on {self.device}")
            
            return self.model, self.processor
            
        except ImportError:
            logger.error("Failed to import transformers/peft. Please install: pip install transformers peft")
            raise
    
    def _initialize_clip_phi(self) -> Tuple[Any, Any]:
        """
        Initialize a lightweight CLIP+Phi combination.
        
        This is a custom lightweight approach that uses CLIP embeddings
        with a small Phi model for text generation.
        """
        try:
            from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
            
            # Load CLIP for image embeddings
            clip_model_path = "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(clip_model_path)
            self.clip_model = CLIPModel.from_pretrained(clip_model_path).to(self.device)
            
            # Load a small Phi model for text generation from image embeddings
            phi_model_path = "microsoft/phi-2"
            self.phi_tokenizer = AutoTokenizer.from_pretrained(phi_model_path)
            
            # Load with quantization if specified
            if self.quantization_level == '8bit':
                self.phi_model = AutoModelForCausalLM.from_pretrained(
                    phi_model_path, load_in_8bit=True, device_map=self.device
                )
            elif self.quantization_level == '4bit':
                self.phi_model = AutoModelForCausalLM.from_pretrained(
                    phi_model_path, load_in_4bit=True, device_map=self.device
                )
            else:
                self.phi_model = AutoModelForCausalLM.from_pretrained(phi_model_path).to(self.device)
            
            # Custom hybrid model (CLIP+Phi)
            class ClipPhiModel:
                def __init__(self, clip_model, phi_model, clip_processor, phi_tokenizer, device):
                    self.clip_model = clip_model
                    self.phi_model = phi_model
                    self.clip_processor = clip_processor
                    self.phi_tokenizer = phi_tokenizer
                    self.device = device
                
                def to(self, device):
                    self.clip_model = self.clip_model.to(device)
                    self.phi_model = self.phi_model.to(device)
                    self.device = device
                    return self
            
            self.model = ClipPhiModel(self.clip_model, self.phi_model, self.processor, self.phi_tokenizer, self.device)
            
            self.initialized = True
            logger.info(f"Successfully initialized CLIP+Phi hybrid model on {self.device}")
            
            return self.model, self.processor
            
        except ImportError:
            logger.error("Failed to import required libraries. Please install: pip install transformers")
            raise 