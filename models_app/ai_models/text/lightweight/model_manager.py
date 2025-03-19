"""
LightweightLLMModelManager

Manages lightweight large language models like Phi, Gemma, and Mistral.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import torch

logger = logging.getLogger(__name__)

class LightweightLLMModelManager:
    """
    Manages configuration and initialization of lightweight LLM models.
    
    This class:
    1. Tracks available lightweight models
    2. Handles model loading and configuration
    3. Manages device placement and optimization
    4. Supports ONNX runtime acceleration
    """
    
    # Available model configurations - all requiring at most 8GB GPU memory
    AVAILABLE_MODELS = {
        'microsoft/phi-3-mini-4k-instruct': {
            'description': 'Phi-3-mini - 3.8B efficient model for general use cases (4k context)',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 4096,
            'requirements': {
                'gpu_memory': 6000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'summarization', 'simple_qa'],
            'supports_onnx': True
        },
        'google/gemma-2b-it': {
            'description': 'Gemma 2B - Google\'s lightweight instruction-tuned model with excellent performance',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 8192,
            'requirements': {
                'gpu_memory': 4000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'summarization', 'simple_qa'],
            'supports_onnx': True
        },
        'microsoft/phi-2': {
            'description': 'Phi-2 - 2.7B compact model with synthetic data training, good for code and reasoning',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 2048,
            'requirements': {
                'gpu_memory': 5000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'coding', 'simple_reasoning'],
            'supports_onnx': True
        },
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0': {
            'description': 'TinyLlama 1.1B - Ultra-lightweight model for basic chat tasks',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 2048,
            'requirements': {
                'gpu_memory': 2500,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'simple_qa'],
            'supports_onnx': True
        },
        'stabilityai/stablelm-zephyr-3b': {
            'description': 'StableLM Zephyr 3B - Stability AI\'s efficient 3B instruction model',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 4096,
            'requirements': {
                'gpu_memory': 6000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'summarization', 'simple_qa'],
            'supports_onnx': True
        },
        'PygmalionAI/mythalion-13b-8k-GGUF': {
            'description': 'Mythalion 13B GGUF - Efficient role-playing model in GGUF format (4-bit)',
            'model_type': 'GGUF',
            'tokenizer_type': 'Auto',
            'context_length': 8192,
            'requirements': {
                'gpu_memory': 7000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'creative_writing', 'role_playing'],
            'supports_onnx': False,
            'file_extension': '.gguf'
        },
        'PaddlePaddle/ernie-3.0-nano-zh': {
            'description': 'ERNIE-Tiny - Baidu\'s lightweight model for Chinese tasks, ultra-low memory usage',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 2048,
            'requirements': {
                'gpu_memory': 2000,
                'quantization': '4bit'
            },
            'capabilities': ['chinese_chat', 'summarization'],
            'supports_onnx': True
        },
        'PaddlePaddle/ernie-3.0-micro-zh': {
            'description': 'ERNIE-Micro - Compact Chinese model from Baidu with good performance-to-size ratio',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 2048,
            'requirements': {
                'gpu_memory': 2000,
                'quantization': '4bit'
            },
            'capabilities': ['chinese_chat', 'summarization'],
            'supports_onnx': True
        },
        'Qwen/Qwen1.5-0.5B-Chat': {
            'description': 'Qwen 0.5B - Ultra small but capable Chinese/English bilingual assistant model',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 2048,
            'requirements': {
                'gpu_memory': 1500,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'simple_qa', 'bilingual'],
            'supports_onnx': True
        },
        'THUDM/chatglm3-6b-gguf': {
            'description': 'ChatGLM3 6B GGUF - Efficient Chinese/English bilingual model in GGUF format',
            'model_type': 'GGUF',
            'tokenizer_type': 'Auto',
            'context_length': 8192,
            'requirements': {
                'gpu_memory': 7000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'bilingual', 'math'],
            'supports_onnx': False,
            'file_extension': '.gguf'
        },
        'Neural-Research-Group/jais-30b-v3-gguf': {
            'description': 'Jais 30B GGUF - Arabic/English bilingual model, 4-bit quantized GGUF',
            'model_type': 'GGUF',
            'tokenizer_type': 'Auto',
            'context_length': 4096,
            'requirements': {
                'gpu_memory': 8000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'arabic_language', 'bilingual'],
            'supports_onnx': False,
            'file_extension': '.gguf'
        },
        'BlinkDL/rwkv-4-raven-1b5': {
            'description': 'RWKV-4-Raven 1.5B - RNN-based model requiring less memory than transformers',
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            'context_length': 4096,
            'requirements': {
                'gpu_memory': 3000,
                'quantization': '4bit'
            },
            'capabilities': ['chat', 'creative_writing'],
            'supports_onnx': False
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Lightweight LLM model manager.
        
        Args:
            config: Configuration for the model manager
        """
        self.config = config or {}
        self.model_name = self.config.get('model', 'microsoft/phi-3-mini-4k-instruct')
        
        # Model configuration
        self.model_config = self.AVAILABLE_MODELS.get(self.model_name, self.AVAILABLE_MODELS['microsoft/phi-3-mini-4k-instruct'])
        
        # Parse additional model-specific configurations
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.9)
        self.repetition_penalty = self.config.get('repetition_penalty', 1.1)
        
        # Quantization and optimization
        self.quantization_level = self.config.get('quantization_level', '4bit')
        self.use_onnx = self.config.get('use_onnx', False) and self.model_config.get('supports_onnx', False)
        
        # Check device availability
        self.device = self._detect_device()
        
        # Model instances
        self.model = None
        self.tokenizer = None
        self.onnx_session = None
        
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
        Check if the model can be loaded on the current device.
        
        Returns:
            bool: True if available, False otherwise
        """
        # For ONNX runtime, we're more flexible with hardware requirements
        if self.use_onnx:
            try:
                import onnxruntime
                return True
            except ImportError:
                logger.error("ONNX Runtime not installed. Run: pip install onnxruntime")
                return False
                
        # For regular PyTorch models, check CUDA for GPU-intensive models
        if self.model_config.get('requirements', {}).get('gpu_memory', 0) > 4000 and self.device == 'cpu':
            # We can still run with CPU, but it will be slow
            logger.warning(f"Model {self.model_name} performs better with GPU but will run on CPU")
            
        # For GGUF models, check if we have the required libraries
        if self.model_config.get('model_type') == 'GGUF':
            try:
                import llama_cpp
                return True
            except ImportError:
                try:
                    import ctransformers
                    return True
                except ImportError:
                    logger.error("Neither llama-cpp-python nor ctransformers is installed for GGUF models")
                    logger.error("Run: pip install llama-cpp-python or pip install ctransformers")
                    return False
            
        try:
            # Check if required libraries are available
            import transformers
            
            return True
        except ImportError:
            logger.error("Transformers package not installed. Run: pip install transformers")
            return False
            
    def _initialize_model(self):
        """Initialize the model and tokenizer."""
        if (self.model is not None and self.tokenizer is not None) or (self.use_onnx and self.onnx_session is not None):
            return
            
        try:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            logger.info(f"Loading lightweight model {self.model_name} on {self.device}...")
            
            # Check if it's a GGUF model
            if self.model_config.get('model_type') == 'GGUF':
                self._initialize_gguf_model()
                return
            
            # For HuggingFace models    
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Load tokenizer (needed for both PyTorch and ONNX)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Load model with PyTorch or ONNX based on configuration
            if self.use_onnx:
                self._initialize_onnx_model()
            else:
                self._initialize_pytorch_model()
                
        except Exception as e:
            logger.error(f"Error loading lightweight model: {str(e)}")
            raise
            
    def _initialize_pytorch_model(self):
        """Initialize the model using PyTorch."""
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
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
            
        logger.info(f"PyTorch model {self.model_name} loaded successfully")
    
    def _initialize_gguf_model(self):
        """Initialize GGUF model using llama-cpp or ctransformers."""
        try:
            # Try using llama-cpp-python first
            try:
                from llama_cpp import Llama
                
                # Configure parameters
                params = {
                    "model_path": self.model_name if os.path.exists(self.model_name) else self._download_gguf_model(),
                    "n_ctx": self.model_config.get('context_length', 4096),
                    "n_threads": os.cpu_count() or 4
                }
                
                # Add GPU offload if available
                if self.device == 'cuda':
                    params["n_gpu_layers"] = -1  # All layers on GPU
                
                # Create llama-cpp model instance
                self.model = Llama(**params)
                
                # For compatibility, create a simple wrapper for the tokenizer
                self.tokenizer = self._create_gguf_tokenizer_wrapper(self.model)
                
                logger.info(f"GGUF model loaded with llama-cpp-python")
                
            except ImportError:
                # Fallback to ctransformers
                logger.info("llama-cpp-python not found, trying ctransformers...")
                
                from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
                
                model_path = self.model_name if os.path.exists(self.model_name) else self._download_gguf_model()
                
                # Load model with ctransformers
                self.model = CTAutoModelForCausalLM.from_pretrained(
                    model_path,
                    model_type="llama",
                    lib="avx2" if self.device == "cpu" else "cuda"
                )
                
                # Create a simple wrapper for compatibility
                self.tokenizer = self._create_gguf_tokenizer_wrapper(self.model)
                
                logger.info(f"GGUF model loaded with ctransformers")
                
        except Exception as e:
            logger.error(f"Error loading GGUF model: {str(e)}")
            raise

    def _download_gguf_model(self):
        """Download GGUF model if needed and return the path."""
        try:
            from huggingface_hub import hf_hub_download
            
            # If model is a HuggingFace repo ID, parse it
            if '/' in self.model_name and not os.path.exists(self.model_name):
                # Check file extension
                file_extension = self.model_config.get('file_extension', '.gguf')
                
                # Split repo_id and filename if necessary
                if ':' in self.model_name:
                    repo_id, filename = self.model_name.split(':', 1)
                else:
                    repo_id = self.model_name
                    # Guess filename from repo_id, using the last part of the path
                    model_name = repo_id.split('/')[-1]
                    filename = f"{model_name}{file_extension}"
                
                # Create models directory if it doesn't exist
                models_dir = os.path.expanduser("~/models/gguf")
                os.makedirs(models_dir, exist_ok=True)
                
                # Download the file
                logger.info(f"Downloading GGUF model from HuggingFace: {repo_id}")
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=models_dir
                )
                
                logger.info(f"GGUF model downloaded to {model_path}")
                return model_path
                
        except Exception as e:
            logger.error(f"Error downloading GGUF model: {str(e)}")
            raise ValueError(f"Failed to download GGUF model: {str(e)}")
            
    def _create_gguf_tokenizer_wrapper(self, model):
        """Create a simple tokenizer wrapper for GGUF models."""
        class GGUFTokenizerWrapper:
            def __init__(self, model):
                self.model = model
                
            def __call__(self, text, return_tensors=None):
                if hasattr(self.model, "tokenize"):
                    tokens = self.model.tokenize(text)
                    return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.ones_like(torch.tensor([tokens]))}
                else:
                    # Use model's encode method
                    return {"input_ids": torch.tensor([self.model.encode(text)]), "attention_mask": torch.ones_like(torch.tensor([self.model.encode(text)]))}
                    
            def decode(self, tokens, skip_special_tokens=True):
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                    
                if hasattr(self.model, "detokenize"):
                    return self.model.detokenize(tokens)
                elif hasattr(self.model, "decode"):
                    return self.model.decode(tokens)
                else:
                    # Fallback
                    return "".join([chr(t) if t < 128 else " " for t in tokens])
                    
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                """Simple chat template for GGUF models."""
                result = ""
                for msg in messages:
                    role = msg.get("role", "").capitalize()
                    content = msg.get("content", "")
                    result += f"{role}: {content}\n"
                    
                if add_generation_prompt:
                    result += "Assistant: "
                    
                return result
        
        return GGUFTokenizerWrapper(model)
            
    def _initialize_onnx_model(self):
        """Initialize the model using ONNX runtime."""
        try:
            import onnxruntime as ort
            import gc
            
            # Set up ONNX session options
            options = ort.SessionOptions()
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.enable_mem_reuse = False
            
            # Determine providers based on available hardware
            if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
                
            # Check if ONNX model exists or needs conversion
            model_path = f"onnx_models/{self.model_name.replace('/', '_')}"
            if not os.path.exists(model_path):
                # Create directory for ONNX models if it doesn't exist
                os.makedirs("onnx_models", exist_ok=True)
                
                # Convert model to ONNX format
                logger.info(f"Converting {self.model_name} to ONNX format...")
                from transformers import AutoModelForCausalLM
                from transformers.onnx import export
                
                # Load PyTorch model temporarily
                temp_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                
                # Export to ONNX
                export(
                    preprocessor=self.tokenizer,
                    model=temp_model,
                    output=model_path,
                    opset=13
                )
                
                # Clean up temporary model
                del temp_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            # Load ONNX model
            self.onnx_session = ort.InferenceSession(model_path, options, providers=providers)
            logger.info(f"ONNX model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            # Fall back to PyTorch
            logger.info("Falling back to PyTorch implementation")
            self.use_onnx = False
            self._initialize_pytorch_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently selected model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'name': self.model_name,
            'provider': 'Lightweight',
            'description': self.model_config.get('description', ''),
            'context_length': self.model_config.get('context_length', 0),
            'capabilities': self.model_config.get('capabilities', []),
            'device': self.device,
            'quantization_level': self.quantization_level,
            'using_onnx': self.use_onnx
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available lightweight models with their configurations.
        
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
                'gpu_memory_required': config.get('requirements', {}).get('gpu_memory', 0),
                'supports_onnx': config.get('supports_onnx', False),
                'model_type': config.get('model_type', 'AutoModelForCausalLM'),
                'languages': self._extract_languages(config)
            })
            
        return models
        
    def _extract_languages(self, config: Dict[str, Any]) -> List[str]:
        """Extract supported languages from model capabilities."""
        languages = []
        capabilities = config.get('capabilities', [])
        
        language_mappings = {
            'chinese_chat': 'Chinese',
            'bilingual': 'English, Chinese',
            'arabic_language': 'Arabic, English'
        }
        
        for capability in capabilities:
            if capability in language_mappings:
                for lang in language_mappings[capability].split(', '):
                    if lang not in languages:
                        languages.append(lang)
                        
        # Default to English if no specific language found
        if not languages:
            languages.append('English')
            
        return languages

    def prepare_generation_parameters(self, prompt: str, system_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare generation parameters for the model.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            
        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Initialize model if needed
        if (self.model is None and self.onnx_session is None) or self.tokenizer is None:
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
            
        # For GGUF models, return the prompt directly
        if self.model_config.get('model_type') == 'GGUF':
            return {
                "prompt": formatted_prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repeat_penalty": self.repetition_penalty,
            }
            
        # Encode the prompt for HF/ONNX models
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if not self.use_onnx and self.device != 'cpu':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prepare generation parameters
        if self.use_onnx:
            # Parameters for ONNX runtime
            return {
                "inputs": inputs,
                "max_length": inputs['input_ids'].shape[1] + self.max_tokens,
                "do_sample": self.temperature > 0.0,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        else:
            # Parameters for PyTorch
            return {
                "inputs": inputs,
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "do_sample": self.temperature > 0.0,
            } 