"""
QwenLLMModelManager

Manages Qwen and QwQ large language models.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import torch

logger = logging.getLogger(__name__)

class QwenLLMModelManager:
    """
    Manages configuration and initialization of Qwen LLM models.
    
    This class:
    1. Tracks available Qwen models
    2. Handles model loading and configuration
    3. Manages device placement and optimization
    """
    
    # Available model configurations
    AVAILABLE_MODELS = {
        'Qwen/QwQ-32B': {
            'description': 'QwQ-32B - high-performance model for reasoning and complex tasks',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 32768,
            'requirements': {
                'gpu_memory': 24000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'reasoning', 'document_processing']
        },
        'Qwen/Qwen2-72B-Instruct': {
            'description': 'Qwen2-72B-Instruct - large instruction-tuned model with strong reasoning',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 32768,
            'requirements': {
                'gpu_memory': 40000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'reasoning', 'document_processing', 'task_decomposition']
        },
        'Qwen/Qwen1.5-14B-Chat': {
            'description': 'Qwen1.5-14B-Chat - balanced chat model for general use',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 8192,
            'requirements': {
                'gpu_memory': 16000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'summarization', 'simple_qa']
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Qwen LLM model manager.
        
        Args:
            config: Configuration for the model manager
        """
        self.config = config or {}
        self.model_name = self.config.get('model', 'Qwen/QwQ-32B')
        
        # Model configuration
        self.model_config = self.AVAILABLE_MODELS.get(self.model_name, self.AVAILABLE_MODELS['Qwen/QwQ-32B'])
        
        # Parse additional model-specific configurations
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.9)
        self.repetition_penalty = self.config.get('repetition_penalty', 1.1)
        
        # Quantization level for optimization
        self.quantization_level = self.config.get('quantization_level', '4bit')
        
        # Check CUDA availability
        self.device = self._detect_device()
        
        # Model instances
        self.model = None
        self.tokenizer = None
        
    def _detect_device(self) -> str:
        """
        Detect the best available device.
        
        Returns:
            str: Device to use (cuda, mps, or cpu)
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
            
    def is_available(self) -> bool:
        """
        Check if the Qwen model can be loaded on the current device.
        
        Returns:
            bool: True if available, False otherwise
        """
        # Check if CUDA is available for GPU models
        if self.model_config.get('requirements', {}).get('gpu_memory', 0) > 0 and self.device == 'cpu':
            logger.warning(f"Qwen model {self.model_name} requires GPU but none is available")
            return False
            
        try:
            # Check if required libraries are available
            import transformers
            
            return True
        except ImportError:
            logger.error("Transformers package not installed. Run: pip install transformers")
            return False
            
    def _initialize_model(self):
        """Initialize the Qwen model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import gc
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"Loading Qwen model {self.model_name} on {self.device}...")
            
            # Configure quantization if needed
            quantization_config = None
            if self.quantization_level == '4bit' and self.device == 'cuda':
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True
                )
            elif self.quantization_level == '8bit' and self.device == 'cuda':
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Load model with appropriate quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device if self.device == 'cuda' else None,
                trust_remote_code=True,
            )
            
            # Move model to device if not using device_map
            if self.device != 'cuda':
                self.model = self.model.to(self.device)
                
            logger.info(f"Qwen model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Qwen model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently selected model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'name': self.model_name,
            'provider': 'Qwen',
            'description': self.model_config.get('description', ''),
            'context_length': self.model_config.get('context_length', 0),
            'capabilities': self.model_config.get('capabilities', []),
            'device': self.device,
            'quantization_level': self.quantization_level
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available Qwen models with their configurations.
        
        Returns:
            List[Dict[str, Any]]: List of model configurations
        """
        models = []
        
        for name, config in self.AVAILABLE_MODELS.items():
            models.append({
                'name': name,
                'description': config.get('description', ''),
                'context_length': config.get('context_length', 0),
                'capabilities': config.get('capabilities', []),
                'gpu_memory_required': config.get('requirements', {}).get('gpu_memory', 0)
            })
            
        return models

    def prepare_generation_parameters(self, prompt: str, system_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare generation parameters for the Qwen model.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            
        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Initialize model if needed
        if self.model is None or self.tokenizer is None:
            self._initialize_model()
            
        # Prepare chat template if system message is available
        if system_message:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                # Build messages array
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                
                # Apply chat template
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback if apply_chat_template not available
                formatted_prompt = f"{system_message}\n\nUser: {prompt}\n\nAssistant:"
        else:
            # Simple prompt without system message
            formatted_prompt = prompt
            
        # Encode the prompt
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Prepare generation parameters
        generation_params = {
            "inputs": inputs,
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.temperature > 0.0,
        }
        
        return generation_params 