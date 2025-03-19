"""
Text LLM Provider Factory

Creates and manages text LLM providers based on configuration parameters.
This factory provides centralized creation, management, and recommendation
of text providers based on hardware capabilities and use case requirements.
"""

import logging
import os
import re
from typing import Dict, Any, Optional, List

# Import the base provider
from .base_text_provider import BaseLLMProvider

# Import our provider services
from .openai.service import OpenAILLMService
from .anthropic.service import AnthropicLLMService
from .qwen.service import QwenLLMService
from .lightweight.service import LightweightLLMService
from .deepseek.service import DeepSeekLLMService
from .local_generic.service import LocalGenericLLMService

# Import error handling and registry from common utilities
from models_app.ai_models.utils.common.handlers import handle_llm_errors
from models_app.ai_models.utils.common.ai_base_service import (
    ModelRegistry, register_service_type, create_service, get_service, get_or_create_service
)

logger = logging.getLogger(__name__)

# Register all text providers with the model registry
def register_text_providers():
    """
    Register all text LLM providers with the ModelRegistry.
    
    This function ensures all text providers are properly registered
    in the central registry, allowing for lookup and management.
    """
    register_service_type('openai', OpenAILLMService)
    register_service_type('anthropic', AnthropicLLMService)
    register_service_type('qwen', QwenLLMService)
    register_service_type('lightweight', LightweightLLMService)
    register_service_type('deepseek', DeepSeekLLMService)
    register_service_type('local_generic', LocalGenericLLMService)
    logger.info("Registered all text LLM providers with the ModelRegistry")

# Register providers on import
try:
    register_text_providers()
except Exception as e:
    logger.error(f"Error registering text providers: {str(e)}")

class ProviderFactory:
    """
    Factory class for creating Text LLM providers based on configurations.
    
    This class provides methods for:
    1. Creating appropriate provider instances based on configuration
    2. Detecting device capabilities to inform provider selection
    3. Recommending providers based on task requirements and hardware
    """
    
    @staticmethod
    @handle_llm_errors
    def create_provider(config: Dict[str, Any]) -> BaseLLMProvider:
        """
        Creates a provider based on the configuration.
        
        Args:
            config: Configuration for the provider, containing parameters like:
                   - provider_type: Type of provider to create
                   - model_name: Name of the model to use
                   - temperature: Generation temperature
                   - max_tokens: Maximum tokens to generate
                   - and other provider-specific parameters
            
        Returns:
            BaseLLMProvider: A provider for the specified model
            
        Raises:
            ValueError: If the provider type is not supported
        """
        provider_type = config.get('provider_type', '').lower()
        model_name = config.get('model_name', config.get('model', ''))  # Support both for backward compatibility
        
        # Ensure model_name is a string
        if model_name is None:
            model_name = ''
        
        # Standardize config to use model_name
        if 'model' in config and 'model_name' not in config:
            config['model_name'] = config['model']
            
        # First check if a service already exists in the registry
        if provider_type:
            existing_service = get_service(provider_type, model_name)
            if existing_service:
                logger.info(f"Using existing {provider_type} provider for model {model_name}")
                return existing_service
        
        # Cloud providers
        if provider_type == 'openai' or any(name in model_name.lower() for name in ['gpt-3.5', 'gpt-4']):
            logger.info(f"Creating OpenAI provider for model {model_name}")
            service = OpenAILLMService(config)
            ModelRegistry.get_instance().register_service('openai', model_name, service)
            return service
        
        elif provider_type == 'anthropic' or any(name in model_name.lower() for name in ['claude-3', 'claude-2']):
            logger.info(f"Creating Anthropic provider for model {model_name}")
            service = AnthropicLLMService(config)
            ModelRegistry.get_instance().register_service('anthropic', model_name, service)
            return service
        
        # Local providers
        elif provider_type == 'deepseek' or 'deepseek' in model_name.lower():
            logger.info(f"Creating DeepSeek provider for model {model_name}")
            service = DeepSeekLLMService(config)
            ModelRegistry.get_instance().register_service('deepseek', model_name, service)
            return service
        
        elif provider_type == 'qwen' or any(name in model_name.lower() for name in ['qwq', 'qwen']):
            logger.info(f"Creating Qwen provider for model {model_name}")
            service = QwenLLMService(config)
            ModelRegistry.get_instance().register_service('qwen', model_name, service)
            return service
        
        elif provider_type == 'lightweight' or any(name in model_name.lower() for name in ['phi', 'gemma', 'mistral-7b', 'llama-3-8b']):
            logger.info(f"Creating Lightweight provider for model {model_name}")
            service = LightweightLLMService(config)
            ModelRegistry.get_instance().register_service('lightweight', model_name, service)
            return service
            
        elif provider_type == 'local' or provider_type == 'local_generic':
            logger.info(f"Creating generic local provider for model {model_name}")
            service = LocalGenericLLMService(config)
            ModelRegistry.get_instance().register_service('local_generic', model_name, service)
            return service
        
        # URL-based detection 
        elif 'url' in config:
            return ProviderFactory.create_provider_from_url(config['url'], config)
        
        # Default fallback
        else:
            logger.warning(f"Unknown provider type: {provider_type}, using Local Generic provider")
            service = LocalGenericLLMService(config)
            ModelRegistry.get_instance().register_service('local_generic', 'fallback', service)
            return service
    
    @staticmethod
    @handle_llm_errors
    def create_provider_from_url(model_url: str, config: Dict[str, Any]) -> BaseLLMProvider:
        """
        Creates a provider based on a model URL.
        
        This method detects the appropriate provider based on the URL pattern
        and creates the corresponding provider instance.
        
        Args:
            model_url: URL to the model, such as a Hugging Face model URL
            config: Configuration for the provider
            
        Returns:
            BaseLLMProvider: A provider for the specified model
        """
        service_type = 'local_generic'
        model_name = 'unknown'
        service = None
        
        # Hugging Face Hub URL
        if "huggingface.co" in model_url or "hf.co" in model_url:
            # Extract model name from URL
            match = re.search(r'(?:huggingface\.co|hf\.co)/([^/]+/[^/]+)', model_url)
            if match:
                model_name = match.group(1)
                config['model_name'] = model_name  # Standardized to model_name
                
                # Check for special models
                if 'deepseek' in model_name.lower():
                    service_type = 'deepseek'
                    service = DeepSeekLLMService(config)
                elif 'qwen' in model_name.lower():
                    service_type = 'qwen'
                    service = QwenLLMService(config)
                elif any(name in model_name.lower() for name in ['phi', 'gemma', 'mistral']):
                    service_type = 'lightweight'
                    service = LightweightLLMService(config)
                else:
                    service = LocalGenericLLMService(config)
        
        # GitHub URL
        elif "github.com" in model_url:
            # For GitHub repos (e.g., Mistral) use the local provider
            service = LocalGenericLLMService(config)
        
        # Local model file
        elif os.path.exists(model_url):
            config['model_path'] = model_url
            model_name = os.path.basename(model_url)
            config['model_name'] = model_name  # Standardized to model_name
            service = LocalGenericLLMService(config)
        
        # Fallback: Try to treat it as a Hugging Face model name
        else:
            config['model_name'] = model_url  # Standardized to model_name
            model_name = model_url
            service = LocalGenericLLMService(config)
            
        # Register the service
        if service:
            ModelRegistry.get_instance().register_service(service_type, model_name, service)
            
        return service
    
    @staticmethod
    def detect_device_capabilities() -> Dict[str, Any]:
        """
        Detects available device capabilities for optimal model selection.
        
        This method checks for available hardware resources including:
        - GPU availability and specifications
        - CPU capabilities
        - System information
        - Python and torch versions
        
        Returns:
            Dict[str, Any]: Device capabilities information
        """
        import torch
        import platform
        
        # Define default capabilities
        capabilities = {
            'has_gpu': False,
            'gpu_name': 'N/A',
            'gpu_memory_mb': 0,
            'cpu_count': os.cpu_count() or 1,
            'system': platform.system(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'has_mps': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
        
        # Check for CUDA GPU
        if torch.cuda.is_available():
            capabilities['has_gpu'] = True
            capabilities['gpu_count'] = torch.cuda.device_count()
            capabilities['gpu_name'] = torch.cuda.get_device_name(0)
            try:
                # Try to get GPU memory
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                capabilities['gpu_memory_mb'] = gpu_mem // (1024 * 1024)
            except:
                # Fallback if the memory query fails
                capabilities['gpu_memory_mb'] = 'Unknown'
        
        # Check for Apple Silicon MPS
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            capabilities['has_gpu'] = True
            capabilities['gpu_name'] = 'Apple Silicon'
            capabilities['gpu_count'] = 1
            # Apple doesn't expose GPU memory info easily
            capabilities['gpu_memory_mb'] = 'Unknown'
        
        return capabilities
    
    @staticmethod
    def get_recommended_provider(task: str, device_capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Determines the recommended provider based on the task and available hardware.
        
        This method selects the most appropriate provider and model for a given task
        by considering the available hardware resources and task requirements.
        
        Args:
            task: Task type ('chat', 'summarization', 'qa', 'code', etc.)
            device_capabilities: Device capabilities (if None, will be detected)
            
        Returns:
            Dict[str, Any]: Recommended provider configuration with parameters:
                - provider_type: Type of recommended provider
                - model_name: Recommended model
                - and various provider-specific parameters
        """
        # Detect capabilities if not provided
        if device_capabilities is None:
            device_capabilities = ProviderFactory.detect_device_capabilities()
        
        has_gpu = device_capabilities.get('has_gpu', False)
        gpu_memory = device_capabilities.get('gpu_memory_mb', 0)
        
        # Default configuration
        config = {
            'provider_type': 'local_generic',
            'model_name': 'microsoft/phi-2',  # Standardized to model_name
            'use_memory': True,
            'memory_config': {
                'max_history': 10,
                'summary_threshold': 20
            }
        }
        
        # Adjust based on hardware
        if not has_gpu:
            # CPU-only recommendations
            config['provider_type'] = 'openai'  # Fall back to cloud services if no GPU
            config['model_name'] = 'gpt-3.5-turbo'  # Standardized to model_name
        elif isinstance(gpu_memory, (int, float)) and gpu_memory > 16000:
            # High-end GPU
            if task in ['chat', 'creative_writing']:
                config['provider_type'] = 'local_generic'
                config['model_name'] = 'meta-llama/Llama-3-8B-Instruct'  # Standardized to model_name
            else:
                config['provider_type'] = 'local_generic'
                config['model_name'] = 'mistralai/Mistral-7B-Instruct-v0.2'  # Standardized to model_name
        elif isinstance(gpu_memory, (int, float)) and gpu_memory > 8000:
            # Mid-range GPU
            config['provider_type'] = 'lightweight'
            config['model_name'] = 'microsoft/phi-2'  # Standardized to model_name
        else:
            # Low-end GPU or unknown
            config['provider_type'] = 'lightweight'
            config['model_name'] = 'google/gemma-2b-instruct'  # Standardized to model_name
        
        # Task-specific adjustments
        if task == 'code':
            if has_gpu:
                config['model_name'] = 'deepseek-ai/deepseek-coder-6.7b-instruct'  # Standardized to model_name
                config['provider_type'] = 'deepseek'
            else:
                config['model_name'] = 'gpt-4'  # Standardized to model_name
                config['provider_type'] = 'openai'
        
        # Add recommended parameters
        if config['provider_type'] in ['local_generic', 'lightweight', 'deepseek', 'qwen']:
            config['quantization'] = '4bit' if has_gpu else '8bit'
            config['context_window'] = 4096
        
        return config
        
    @staticmethod
    def list_available_models(provider_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists all available models, optionally filtered by provider type.
        
        Args:
            provider_type: Optional provider type to filter results
            
        Returns:
            List[Dict[str, Any]]: List of available models with metadata
        """
        models = []
        
        # Get registry instance
        registry = ModelRegistry.get_instance()
        
        # If provider_type is specified, only get models for that provider
        if provider_type:
            provider_models = registry.list_services(provider_type)
            for model_name, service in provider_models.items():
                model_info = service.get_model_info() if hasattr(service, 'get_model_info') else {}
                models.append({
                    'provider': provider_type,
                    'name': model_name,
                    'info': model_info
                })
            return models
            
        # Otherwise, get models for all providers
        provider_types = [
            'openai', 'anthropic', 'qwen', 'lightweight', 
            'deepseek', 'local_generic'
        ]
        
        for provider in provider_types:
            provider_models = registry.list_services(provider)
            for model_name, service in provider_models.items():
                model_info = service.get_model_info() if hasattr(service, 'get_model_info') else {}
                models.append({
                    'provider': provider,
                    'name': model_name,
                    'info': model_info
                })
                
        return models 