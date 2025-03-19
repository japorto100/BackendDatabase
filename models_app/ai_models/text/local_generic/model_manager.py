"""
LocalGenericLLMModelManager

Manages generic local large language models from local storage or custom sources.
"""

import os
import logging
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
import torch
import glob
import shutil
import urllib.parse
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Dictionary of recommended lightweight models that can run on minimal hardware
RECOMMENDED_MODELS = {
    "Baidu ERNIE-Tiny": {
        "description": "Lightweight model from Baidu's ERNIE family, suitable for basic NLP tasks",
        "model_url": "https://huggingface.co/PaddlePaddle/ernie-3.0-nano-zh",
        "parameters": "312M",
        "context_length": 2048,
        "min_requirements": "4GB RAM, CPU only",
        "quantization": "4-bit supported",
        "languages": ["Chinese"]
    },
    "Baidu ERNIE-Micro": {
        "description": "Compact model from Baidu's ERNIE family with good performance-to-size ratio",
        "model_url": "https://huggingface.co/PaddlePaddle/ernie-3.0-micro-zh",
        "parameters": "384M",
        "context_length": 2048,
        "min_requirements": "4GB RAM, CPU only",
        "quantization": "4-bit supported",
        "languages": ["Chinese"]
    },
    "Baidu ERNIE-Mini": {
        "description": "Small but capable model from Baidu's ERNIE family for various NLP tasks",
        "model_url": "https://huggingface.co/PaddlePaddle/ernie-3.0-mini-zh",
        "parameters": "384M",
        "context_length": 2048,
        "min_requirements": "8GB RAM, CPU only",
        "quantization": "4-bit supported",
        "languages": ["Chinese"]
    },
    "Microsoft Phi-2": {
        "description": "Small but powerful 2.7B parameter model, trained on synthetic data",
        "model_url": "https://huggingface.co/microsoft/phi-2",
        "parameters": "2.7B",
        "context_length": 2048,
        "min_requirements": "8GB RAM, 4GB VRAM",
        "quantization": "4-bit supported",
        "languages": ["English"]
    },
    "TinyLlama-1.1B": {
        "description": "Compact LLM with surprisingly good capabilities for its size",
        "model_url": "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "parameters": "1.1B",
        "context_length": 2048,
        "min_requirements": "4GB RAM, CPU only",
        "quantization": "4-bit supported",
        "languages": ["English"]
    },
    "Gemma-2B": {
        "description": "Google's lightweight open model with good performance on limited hardware",
        "model_url": "https://huggingface.co/google/gemma-2b-it",
        "parameters": "2B",
        "context_length": 8192,
        "min_requirements": "8GB RAM, 4GB VRAM",
        "quantization": "4-bit supported",
        "languages": ["English"]
    },
    "Mistral-7B-Instruct-v0.2": {
        "description": "Excellent performance-to-size ratio, works well with 4-bit quantization",
        "model_url": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2",
        "parameters": "7B",
        "context_length": 8192,
        "min_requirements": "16GB RAM, 8GB VRAM (4-bit quantized)",
        "quantization": "4-bit supported, GGUF available",
        "languages": ["English", "French", "German", "Italian", "Spanish"]
    },
    "RWKV-4-Raven-1.5B": {
        "description": "RNN-based model that requires less memory than transformer models",
        "model_url": "https://huggingface.co/BlinkDL/rwkv-4-raven-1b5",
        "parameters": "1.5B",
        "context_length": 4096,
        "min_requirements": "6GB RAM, CPU only",
        "quantization": "GGUF available",
        "languages": ["English"]
    }
}

class LocalGenericLLMModelManager:
    """
    Manages configuration and initialization of generic local LLM models.
    
    This class:
    1. Detects available local models
    2. Handles model loading and configuration
    3. Manages device placement and optimization
    4. Supports custom model formats and configurations
    5. Allows downloading models from URLs and repositories
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Local Generic LLM model manager.
        
        Args:
            config: Configuration for the model manager
        """
        self.config = config or {}
        
        # Default models directory can be overridden in config
        self.models_dir = self.config.get('models_dir', os.path.join(os.path.expanduser('~'), 'models'))
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model can be specified as path, name, or URL
        self.model_path = self.config.get('model_path', None)
        self.model_name = self.config.get('model', None)
        self.model_url = self.config.get('model_url', None)
        
        # If URL is provided, download the model
        if self.model_url and not self.model_path:
            self.model_path = self.download_model_from_url(self.model_url)
        
        # If neither is provided, we'll scan for available models
        if not self.model_path and not self.model_name:
            available_models = self.scan_available_models()
            if available_models:
                self.model_path = available_models[0]['path']
                self.model_name = available_models[0]['name']
            else:
                logger.warning("No local models found. Please specify model_path, model_name, or model_url in config.")
        
        # Parse additional model-specific configurations
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.9)
        self.repetition_penalty = self.config.get('repetition_penalty', 1.1)
        
        # Model format and optimization
        self.model_format = self.config.get('model_format', 'auto')  # 'auto', 'huggingface', 'gguf', 'onnx', etc.
        self.quantization_level = self.config.get('quantization_level', '4bit')
        self.context_length = self.config.get('context_length', 4096)
        
        # Device configuration
        self.device = self._detect_device()
        
        # Model instances
        self.model = None
        self.tokenizer = None
        
    def download_model_from_url(self, url: str) -> str:
        """
        Download a model from a URL.
        
        Supports:
        - Hugging Face Hub models (huggingface.co/...)
        - GitHub repositories (github.com/...)
        - Direct file downloads (.gguf, .bin, .onnx, etc.)
        - ZIP archives
        
        Args:
            url: URL to download the model from
            
        Returns:
            str: Path to the downloaded model
        """
        logger.info(f"Downloading model from URL: {url}")
        
        # Create a safe directory name based on the URL
        parsed_url = urllib.parse.urlparse(url)
        
        # Handle different URL types
        if "huggingface.co" in url or "hf.co" in url:
            return self._download_from_huggingface(url)
        elif "github.com" in url:
            return self._download_from_github(url)
        elif url.endswith(".gguf") or url.endswith(".bin") or url.endswith(".onnx") or url.endswith(".pt"):
            return self._download_direct_file(url)
        elif url.endswith(".zip"):
            return self._download_and_extract_zip(url)
        else:
            # Try to treat as a Hugging Face model ID
            if "/" in url and not url.startswith("http"):
                logger.info(f"Treating as Hugging Face model ID: {url}")
                return self._download_from_huggingface(f"https://huggingface.co/{url}")
            else:
                logger.warning(f"Unrecognized URL format: {url}, attempting direct download")
                return self._download_direct_file(url)
            
    def _download_from_huggingface(self, url: str) -> str:
        """
        Download a model from Hugging Face Hub.
        
        Args:
            url: Hugging Face URL or model ID
            
        Returns:
            str: Path to the downloaded model
        """
        try:
            from huggingface_hub import snapshot_download
            
            # Extract model ID from URL
            match = re.search(r'(?:huggingface\.co|hf\.co)/([^/]+/[^/]+)', url)
            if match:
                model_id = match.group(1)
            else:
                # Remove any URL prefix if present
                model_id = url.replace("https://huggingface.co/", "").replace("https://hf.co/", "")
                
            # Create target directory
            target_dir = os.path.join(self.models_dir, model_id.replace("/", "_"))
            os.makedirs(target_dir, exist_ok=True)
            
            # Download the model
            logger.info(f"Downloading model {model_id} from Hugging Face Hub...")
            local_dir = snapshot_download(
                repo_id=model_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"Model downloaded to {local_dir}")
            return local_dir
            
        except ImportError:
            logger.error("huggingface_hub package not installed. Run: pip install huggingface_hub")
            # Fall back to using transformers directly
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                # Extract model ID from URL
                match = re.search(r'(?:huggingface\.co|hf\.co)/([^/]+/[^/]+)', url)
                if match:
                    model_id = match.group(1)
                else:
                    model_id = url.replace("https://huggingface.co/", "").replace("https://hf.co/", "")
                
                # Create target directory using the model name
                model_name = model_id.replace("/", "_")
                target_dir = os.path.join(self.models_dir, model_name)
                os.makedirs(target_dir, exist_ok=True)
                
                # Download tokenizer and model
                logger.info(f"Downloading model {model_id} using transformers...")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
                
                # Save to disk
                tokenizer.save_pretrained(target_dir)
                model.save_pretrained(target_dir)
                
                logger.info(f"Model downloaded to {target_dir}")
                return target_dir
                
            except Exception as e:
                logger.error(f"Error downloading from Hugging Face: {str(e)}")
                raise
                
    def _download_from_github(self, url: str) -> str:
        """
        Download a repository from GitHub.
        
        Args:
            url: GitHub repository URL
            
        Returns:
            str: Path to the downloaded repository
        """
        try:
            # Extract repo owner and name
            match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
            if not match:
                raise ValueError(f"Invalid GitHub URL: {url}")
                
            owner, repo = match.group(1), match.group(2)
            
            # Remove .git extension if present
            repo = repo.replace(".git", "")
            
            # Create target directory
            target_dir = os.path.join(self.models_dir, f"{owner}_{repo}")
            os.makedirs(target_dir, exist_ok=True)
            
            # Use git clone to download the repository
            import subprocess
            logger.info(f"Cloning repository {owner}/{repo} from GitHub...")
            subprocess.run(["git", "clone", url, target_dir], check=True)
            
            logger.info(f"Repository cloned to {target_dir}")
            return target_dir
            
        except Exception as e:
            logger.error(f"Error downloading from GitHub: {str(e)}")
            raise
            
    def _download_direct_file(self, url: str) -> str:
        """
        Download a file directly from a URL.
        
        Args:
            url: URL to the file
            
        Returns:
            str: Path to the downloaded file
        """
        try:
            # Extract filename from URL
            filename = os.path.basename(urllib.parse.urlparse(url).path)
            if not filename:
                filename = f"model_{hash(url)}"
                
            # Add extension based on URL if missing
            if "." not in filename:
                if url.endswith(".gguf"):
                    filename += ".gguf"
                elif url.endswith(".bin"):
                    filename += ".bin"
                elif url.endswith(".onnx"):
                    filename += ".onnx"
                elif url.endswith(".pt"):
                    filename += ".pt"
                    
            # Create target directory
            target_dir = os.path.join(self.models_dir, "downloads")
            os.makedirs(target_dir, exist_ok=True)
            
            # Full path to save the file
            target_path = os.path.join(target_dir, filename)
            
            # Download the file with progress bar
            logger.info(f"Downloading file from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            
            with open(target_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    progress_bar.update(size)
                    
            logger.info(f"File downloaded to {target_path}")
            return target_path
            
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise
            
    def _download_and_extract_zip(self, url: str) -> str:
        """
        Download and extract a ZIP archive.
        
        Args:
            url: URL to the ZIP archive
            
        Returns:
            str: Path to the extracted directory
        """
        try:
            import zipfile
            
            # Download the ZIP file
            zip_path = self._download_direct_file(url)
            
            # Create extraction directory (same name as zip without extension)
            extract_dir = os.path.splitext(zip_path)[0]
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract the ZIP file
            logger.info(f"Extracting ZIP archive to {extract_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            logger.info(f"ZIP archive extracted to {extract_dir}")
            return extract_dir
            
        except Exception as e:
            logger.error(f"Error extracting ZIP archive: {str(e)}")
            raise
    
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
            
    def scan_available_models(self) -> List[Dict[str, Any]]:
        """
        Scan for available local models in configured directories.
        
        Returns:
            List[Dict[str, Any]]: List of available models and their formats
        """
        models = []
        
        # Make sure models directory exists
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            return models
            
        # Check for Hugging Face-style models (containing config.json)
        hf_models = glob.glob(os.path.join(self.models_dir, "**", "config.json"), recursive=True)
        for model_config in hf_models:
            model_dir = os.path.dirname(model_config)
            model_name = os.path.basename(model_dir)
            models.append({
                'name': model_name,
                'path': model_dir,
                'format': 'huggingface',
                'type': 'local'
            })
            
        # Check for GGUF format models
        gguf_models = glob.glob(os.path.join(self.models_dir, "**", "*.gguf"), recursive=True)
        for model_path in gguf_models:
            model_name = os.path.basename(model_path)
            models.append({
                'name': model_name,
                'path': model_path,
                'format': 'gguf',
                'type': 'local'
            })
            
        # Check for ONNX format models
        onnx_models = glob.glob(os.path.join(self.models_dir, "**", "*.onnx"), recursive=True)
        for model_path in onnx_models:
            model_name = os.path.basename(model_path)
            models.append({
                'name': model_name,
                'path': model_path,
                'format': 'onnx',
                'type': 'local'
            })
            
        # Check for downloaded models in special directories
        downloads_dir = os.path.join(self.models_dir, "downloads")
        if os.path.exists(downloads_dir):
            for item in os.listdir(downloads_dir):
                item_path = os.path.join(downloads_dir, item)
                if os.path.isfile(item_path):
                    # Determine format based on extension
                    if item.endswith(".gguf"):
                        format_type = "gguf"
                    elif item.endswith(".onnx"):
                        format_type = "onnx"
                    elif item.endswith(".bin") or item.endswith(".pt"):
                        format_type = "huggingface"
                    else:
                        format_type = "unknown"
                        
                    models.append({
                        'name': item,
                        'path': item_path,
                        'format': format_type,
                        'type': 'downloaded'
                    })
                    
        return models
            
    def is_available(self) -> bool:
        """
        Check if the local model can be loaded on the current device.
        
        Returns:
            bool: True if available, False otherwise
        """
        # Check if we have a valid model path or name
        if not self.model_path and not self.model_name:
            logger.error("No model path or name specified")
            return False
            
        # Check if file exists when using a path
        if self.model_path and not os.path.exists(self.model_path):
            logger.error(f"Model path does not exist: {self.model_path}")
            return False
            
        # Check required libraries
        try:
            import transformers
            return True
        except ImportError:
            logger.error("Transformers package not installed. Run: pip install transformers")
            return False
            
    def _detect_model_format(self) -> str:
        """
        Detect the format of the model based on its path or inspection.
        
        Returns:
            str: Model format (huggingface, gguf, onnx, etc.)
        """
        if self.model_format != 'auto':
            return self.model_format
            
        if self.model_path:
            # Check file extension for common formats
            if self.model_path.endswith('.gguf'):
                return 'gguf'
            elif self.model_path.endswith('.onnx'):
                return 'onnx'
            elif os.path.isdir(self.model_path) and os.path.exists(os.path.join(self.model_path, 'config.json')):
                return 'huggingface'
                
        # Default to huggingface format
        return 'huggingface'
            
    def _initialize_model(self):
        """Initialize the local model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return
            
        # Detect model format
        format_type = self._detect_model_format()
        
        try:
            if format_type == 'huggingface':
                self._initialize_huggingface_model()
            elif format_type == 'gguf':
                self._initialize_gguf_model()
            elif format_type == 'onnx':
                self._initialize_onnx_model()
            else:
                logger.error(f"Unsupported model format: {format_type}")
                raise ValueError(f"Unsupported model format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            raise
            
    def _initialize_huggingface_model(self):
        """Initialize model using Hugging Face Transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import gc
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Determine model path/name
            model_identifier = self.model_path if self.model_path else self.model_name
            
            logger.info(f"Loading local model {model_identifier} on {self.device}...")
            
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
                model_identifier, 
                trust_remote_code=True
            )
            
            # Load model with appropriate quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                quantization_config=quantization_config,
                device_map=self.device if self.device == 'cuda' else None,
                trust_remote_code=True,
            )
            
            # Move model to device if not using device_map
            if self.device != 'cuda':
                self.model = self.model.to(self.device)
                
            logger.info(f"Local model {model_identifier} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading local Hugging Face model: {str(e)}")
            raise
            
    def _initialize_gguf_model(self):
        """Initialize model using llama-cpp or ctransformers for GGUF format."""
        try:
            # Try using llama-cpp-python first
            try:
                from llama_cpp import Llama
                
                # Determine model path
                model_path = self.model_path
                
                # Configure parameters
                params = {
                    "model_path": model_path,
                    "n_ctx": self.context_length,
                    "n_threads": os.cpu_count() or 4
                }
                
                # Add GPU offload if available
                if self.device == 'cuda':
                    params["n_gpu_layers"] = -1  # All layers on GPU
                
                # Create llama-cpp model instance
                self.model = Llama(**params)
                
                # For compatibility, we'll build a simple wrapper for the tokenizer
                self.tokenizer = self._create_gguf_tokenizer_wrapper(self.model)
                
                logger.info(f"GGUF model {model_path} loaded with llama-cpp-python")
                
            except ImportError:
                # Fallback to ctransformers
                logger.info("llama-cpp-python not found, trying ctransformers...")
                
                from ctransformers import AutoModelForCausalLM as CTAutoModelForCausalLM
                from transformers import AutoTokenizer
                
                model_path = self.model_path
                
                # Load model with ctransformers
                self.model = CTAutoModelForCausalLM.from_pretrained(
                    model_path,
                    model_type="llama",
                    lib="avx2" if self.device == "cpu" else "cuda"
                )
                
                # Try to load a matching tokenizer, or create a simple one
                try:
                    tokenizer_path = os.path.dirname(model_path)
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                except:
                    # Create a simple wrapper for compatibility
                    self.tokenizer = self._create_gguf_tokenizer_wrapper(self.model)
                
                logger.info(f"GGUF model {model_path} loaded with ctransformers")
                
        except Exception as e:
            logger.error(f"Error loading GGUF model: {str(e)}")
            raise
            
    def _initialize_onnx_model(self):
        """Initialize model using ONNX runtime."""
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
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
                
            # Load ONNX model
            model_path = self.model_path
            self.model = ort.InferenceSession(model_path, options, providers=providers)
            
            # Try to load a matching tokenizer, or use a default one
            try:
                # Try to find tokenizer in the same directory
                tokenizer_path = os.path.dirname(model_path)
                if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                else:
                    # Use a default tokenizer as fallback
                    self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            except Exception as e:
                logger.warning(f"Error loading tokenizer for ONNX model: {str(e)}")
                # Use a default tokenizer as fallback
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            
            logger.info(f"ONNX model {model_path} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ONNX model: {str(e)}")
            raise
            
    def _create_gguf_tokenizer_wrapper(self, model):
        """
        Create a simple tokenizer wrapper for GGUF models.
        
        Args:
            model: GGUF model instance
            
        Returns:
            object: A simple tokenizer wrapper compatible with our interface
        """
        # Simple wrapper class for compatibility
        class GGUFTokenizerWrapper:
            def __init__(self, model):
                self.model = model
                
            def __call__(self, text, return_tensors=None):
                # Simple encoding for compatibility
                if hasattr(self.model, "tokenize"):
                    tokens = self.model.tokenize(text)
                    return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.ones_like(torch.tensor([tokens]))}
                else:
                    # Fallback using model's encode method
                    return {"input_ids": torch.tensor([self.model.encode(text)]), "attention_mask": torch.ones_like(torch.tensor([self.model.encode(text)]))}
                    
            def decode(self, tokens, skip_special_tokens=True):
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                    
                # Try different decode methods based on the model
                if hasattr(self.model, "detokenize"):
                    return self.model.detokenize(tokens)
                elif hasattr(self.model, "decode"):
                    return self.model.decode(tokens)
                else:
                    # Simple character-by-character fallback
                    return "".join([chr(t) if t < 128 else " " for t in tokens])
        
        return GGUFTokenizerWrapper(model)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently selected model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        model_format = self._detect_model_format()
        model_identifier = self.model_path if self.model_path else self.model_name
        
        return {
            'name': os.path.basename(model_identifier) if model_identifier else "Unknown",
            'path': self.model_path,
            'provider': 'Local Generic',
            'format': model_format,
            'context_length': self.context_length,
            'device': self.device,
            'quantization_level': self.quantization_level
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available local models with their configurations.
        
        Returns:
            List[Dict[str, Any]]: List of model configurations
        """
        local_models = self.scan_available_models()
        
        # Add a recommended models section
        recommended_models = []
        for name, info in RECOMMENDED_MODELS.items():
            recommended_models.append({
                'name': name,
                'path': info['model_url'],
                'format': 'huggingface',
                'type': 'recommended',
                'description': info['description'],
                'parameters': info['parameters'],
                'context_length': info['context_length'],
                'min_requirements': info['min_requirements'],
                'languages': info['languages']
            })
            
        # Combine local and recommended models
        return local_models + recommended_models
        
    def add_model_from_url(self, url: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a new model from a URL.
        
        Args:
            url: URL to download the model from
            name: Optional custom name for the model
            
        Returns:
            Dict[str, Any]: Information about the downloaded model
        """
        try:
            model_path = self.download_model_from_url(url)
            
            # Determine format and name
            if not name:
                name = os.path.basename(model_path)
                
            format_type = 'unknown'
            if model_path.endswith('.gguf'):
                format_type = 'gguf'
            elif model_path.endswith('.onnx'):
                format_type = 'onnx'
            elif os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'config.json')):
                format_type = 'huggingface'
                
            model_info = {
                'name': name,
                'path': model_path,
                'format': format_type,
                'type': 'downloaded',
                'url': url
            }
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error adding model from URL: {str(e)}")
            raise
            
    def get_recommended_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of recommended lightweight models.
        
        Returns:
            List[Dict[str, Any]]: List of recommended model configurations
        """
        result = []
        for name, info in RECOMMENDED_MODELS.items():
            result.append({
                'name': name,
                'url': info['model_url'],
                'description': info['description'],
                'parameters': info['parameters'],
                'context_length': info['context_length'],
                'min_requirements': info['min_requirements'],
                'quantization': info.get('quantization', 'Not specified'),
                'languages': info['languages']
            })
            
        return result
        
    def estimate_hardware_requirements(self, model_size: str) -> Dict[str, Any]:
        """
        Estimate hardware requirements for a model of given size.
        
        Args:
            model_size: Model size in parameters (e.g., "7B", "1.5B")
            
        Returns:
            Dict[str, Any]: Estimated hardware requirements
        """
        # Convert size string to number in billions
        size_in_b = 0
        if "B" in model_size:
            size_str = model_size.replace("B", "")
            if "." in size_str:
                size_in_b = float(size_str)
            else:
                size_in_b = int(size_str) / 1000  # Convert M to B
                
        # Estimate RAM and VRAM requirements
        ram_gb = 0
        vram_gb = 0
        
        if size_in_b <= 1:
            ram_gb = 8
            vram_gb = 4
        elif size_in_b <= 3:
            ram_gb = 16
            vram_gb = 8
        elif size_in_b <= 7:
            ram_gb = 32
            vram_gb = 16
        elif size_in_b <= 13:
            ram_gb = 64
            vram_gb = 24
        else:
            ram_gb = 128
            vram_gb = 48
            
        # Adjust for quantization
        if self.quantization_level == '4bit':
            vram_gb = max(2, vram_gb // 2)
            
        return {
            'ram_gb': ram_gb,
            'vram_gb': vram_gb,
            'cpu_only_possible': size_in_b <= 3,
            'recommended_quantization': '4bit' if size_in_b > 1 else 'none'
        }

    def prepare_generation_parameters(self, prompt: str, system_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare generation parameters for the local model.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            
        Returns:
            Dict[str, Any]: Generation parameters
        """
        # Initialize model if needed
        if self.model is None or self.tokenizer is None:
            self._initialize_model()
            
        model_format = self._detect_model_format()
        
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
                # Fallback simple template
                formatted_prompt = f"{system_message}\n\nUser: {prompt}\n\nAssistant:"
        else:
            # Simple prompt without system message
            formatted_prompt = prompt
            
        # Format parameters based on model type
        if model_format == 'huggingface':
            # Encode the prompt for HF models
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
            
        elif model_format == 'gguf':
            # Parameters for llama-cpp or ctransformers
            generation_params = {
                "prompt": formatted_prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repeat_penalty": self.repetition_penalty,
            }
            
        elif model_format == 'onnx':
            # Parameters for ONNX
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            generation_params = {
                "inputs": inputs,
                "max_length": self.max_tokens + inputs['input_ids'].shape[1],
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        
        return generation_params 