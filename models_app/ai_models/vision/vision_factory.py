"""
Vision Provider Factory

Creates vision model providers based on model types and configurations.
"""

import logging
import os
import platform
from typing import Dict, Any, Optional, Union, List
import torch
from PIL import Image

from models_app.ai_models.vision.base_vision_provider import BaseVisionProvider
from models_app.ai_models.utils.common.errors import ModelError, ModelUnavailableError, VisionModelError, ProviderNotFoundError
from models_app.ai_models.utils.common.handlers import handle_vision_errors
from models_app.ai_models.utils.common.metrics import get_vision_metrics
from models_app.ai_models.utils.common.config import VisionConfig

from models_app.ai_models.utils.vision.image_processing import (
    extract_urls_from_text,
    support_multiple_images,
    download_and_process_image,
    handle_high_resolution_image
)

# Add ModelRegistry imports
from models_app.ai_models.utils.common.ai_base_service import (
    ModelRegistry, register_service_type, create_service, get_service, get_or_create_service
)

# Add functools for caching
import functools
import time

from Levenshtein import distance

# Register vision providers function
def register_vision_providers():
    """
    Register all vision providers with the ModelRegistry.
    """
    # Import vision providers
    from models_app.ai_models.vision.qwen.service import QwenVisionService
    from models_app.ai_models.vision.gemini.service import GeminiVisionService
    from models_app.ai_models.vision.gpt4v.service import GPT4VisionService
    from models_app.ai_models.vision.lightweight.service import LightweightVisionService
    
    # Register with the ModelRegistry
    register_service_type('qwen_vision', QwenVisionService)
    register_service_type('gemini_vision', GeminiVisionService)
    register_service_type('gpt4_vision', GPT4VisionService)
    register_service_type('lightweight_vision', LightweightVisionService)
    
    logger.info("Registered all vision providers with the ModelRegistry")

# Register providers on import
try:
    register_vision_providers()
except Exception as e:
    logger.error(f"Error registering vision providers: {str(e)}")

# Store capabilities cache time
_capabilities_cache = None
_capabilities_timestamp = 0
_CAPABILITIES_CACHE_TTL = 300  # 5 minutes

class VisionProviderFactory:
    """
    Factory class for creating vision model providers.
    
    This class:
    1. Creates appropriate vision providers based on model type and configuration
    2. Handles provider initialization and error handling
    3. Recommends suitable providers based on task requirements and device capabilities
    4. Uses the ModelRegistry for provider management
    5. Provides failover mechanisms when primary providers are unavailable
    """
    
    @staticmethod
    @handle_vision_errors
    def create_provider(config: Dict[str, Any]) -> BaseVisionProvider:
        """
        Create a vision provider based on the configuration.
        
        Args:
            config: Provider configuration
            
        Returns:
            BaseVisionProvider: A provider for vision models
        """
        # Get metrics collector for the factory
        metrics = get_vision_metrics("vision_factory")
        op_time = metrics.start_operation("create_provider")
        
        if not config:
            logger.warning("Empty configuration provided, using default lightweight provider")
            config = {'vision_type': 'lightweight'}
            
        vision_type = config.get('vision_type', '').lower()
        model_name = config.get('model', '').lower()
        
        try:
            # Map vision_type to service_type in ModelRegistry
            service_type_map = {
                'qwen': 'qwen_vision',
                'gemini': 'gemini_vision',
                'gpt4v': 'gpt4_vision',
                'lightweight': 'lightweight_vision'
            }
            
            service_type = service_type_map.get(vision_type, 'lightweight_vision')
            
            # Record provider creation attempt
            metrics.record_custom_metric("provider_creation", "vision_type", vision_type)
            metrics.record_custom_metric("provider_creation", "model_name", model_name)
            
            # Check if we already have this provider in the registry
            existing_provider = get_service(service_type, model_name)
            if existing_provider:
                logger.info(f"Using existing {service_type} provider for model {model_name}")
                metrics.record_custom_metric("provider_creation", "from_registry", True)
                metrics.stop_operation("create_provider", op_time, success=True)
                return existing_provider
            
            # Create a new service through the registry
            provider = create_service(service_type, model_name, model_name, **config)
            
            # If creation succeeded
            if provider:
                # Initialize the provider
                provider.initialize()
                metrics.record_custom_metric("provider_creation", "new_created", True)
                metrics.stop_operation("create_provider", op_time, success=True)
                return provider
                
            # If we get here, service creation failed but didn't throw an exception
            # Fall back to lightweight provider
            logger.warning(f"Failed to create {service_type} provider, falling back to lightweight")
            metrics.record_custom_metric("provider_creation", "fallback_used", True)
            fallback = create_service('lightweight_vision', 'fallback', 'lightweight', **config)
            fallback.initialize()
            metrics.stop_operation("create_provider", op_time, success=True)
            return fallback
                
        except ImportError as e:
            logger.error(f"Error importing vision provider: {e}")
            metrics.record_vision_error("provider_import_error", {
                "error": str(e),
                "vision_type": vision_type,
                "model_name": model_name
            })
            
            # Try to fall back to lightweight provider
            try:
                logger.warning("Falling back to lightweight vision service due to import error")
                
                # Record fallback attempt
                metrics.record_custom_metric("provider_creation", "fallback_used", True)
                
                fallback = create_service('lightweight_vision', 'fallback', 'lightweight', **config)
                fallback.initialize()
                metrics.stop_operation("create_provider", op_time, success=True)
                return fallback
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback provider: {fallback_error}")
                metrics.stop_operation("create_provider", op_time, success=False)
                raise ModelUnavailableError(
                    f"Failed to create vision provider: {e}. Fallback also failed: {fallback_error}",
                    model_name=model_name
                )
                
        except Exception as e:
            logger.error(f"Error creating vision provider: {e}")
            metrics.stop_operation("create_provider", op_time, success=False)
            metrics.record_vision_error("provider_creation_error", {
                "error": str(e),
                "vision_type": vision_type,
                "model_name": model_name
            })
            raise ModelUnavailableError(f"Failed to create vision provider: {e}", model_name=model_name)
    
    @staticmethod
    @handle_vision_errors
    def process_image_input(image_input: Union[str, Image.Image, bytes, List], provider_config: Dict[str, Any]) -> Union[Image.Image, List[Image.Image]]:
        """
        Process and standardize image input before sending to provider.
        
        Args:
            image_input: The image(s) to process, can be various formats
            provider_config: Configuration for the provider
            
        Returns:
            Union[Image.Image, List[Image.Image]]: Processed image(s) ready for the provider
        """
        # Get metrics collector
        metrics = get_vision_metrics("vision_factory")
        op_time = metrics.start_operation("process_image_input")
        
        try:
            # Extract URLs if the input is text containing URLs
            if isinstance(image_input, str) and not image_input.startswith(('http', 'data:')):
                urls = extract_urls_from_text(image_input)
                if urls:
                    # Process multiple images if the provider supports it
                    if provider_config.get('supports_multiple_images', False):
                        result = support_multiple_images(urls)
                        metrics.record_custom_metric("image_processing", "url_count", len(urls))
                        metrics.record_custom_metric("image_processing", "processed_all_urls", True)
                        metrics.stop_operation("process_image_input", op_time, success=True)
                        return result
                    else:
                        # Just use the first URL for providers that don't support multiple images
                        try:
                            result = download_and_process_image(urls[0])
                            metrics.record_custom_metric("image_processing", "url_count", len(urls))
                            metrics.record_custom_metric("image_processing", "processed_all_urls", False)
                            metrics.stop_operation("process_image_input", op_time, success=True)
                            return result
                        except Exception as e:
                            logger.error(f"Error downloading image from URL: {e}")
                            metrics.record_vision_error("image_download_error", {"error": str(e), "url": urls[0]})
                            metrics.stop_operation("process_image_input", op_time, success=False)
                            raise VisionModelError(f"Failed to download image from URL: {urls[0]}")
            
            # Handle multiple images case
            if isinstance(image_input, list):
                if provider_config.get('supports_multiple_images', False):
                    result = support_multiple_images(image_input)
                    metrics.record_custom_metric("image_processing", "image_count", len(image_input))
                    metrics.stop_operation("process_image_input", op_time, success=True)
                    return result
                else:
                    # If provider doesn't support multiple images, use just the first one
                    logger.warning("Provider doesn't support multiple images, using only the first image")
                    if not image_input:
                        metrics.record_vision_error("empty_image_list", {})
                        metrics.stop_operation("process_image_input", op_time, success=False)
                        raise VisionModelError("Empty image list provided")
                    
                    metrics.record_custom_metric("image_processing", "image_count", len(image_input))
                    metrics.record_custom_metric("image_processing", "using_first_only", True)
                    image_input = image_input[0]
            
            # Handle high-resolution image case
            if isinstance(image_input, Image.Image):
                max_image_size = provider_config.get('max_image_size', 1024)
                if max(image_input.size) > max_image_size:
                    if provider_config.get('supports_tiling', False):
                        tile_size = provider_config.get('tile_size', 512)
                        max_tiles = provider_config.get('max_tiles', 6)
                        result = handle_high_resolution_image(
                            image_input, 
                            method="tile", 
                            tile_size=tile_size, 
                            max_tiles=max_tiles
                        )
                        metrics.record_custom_metric("image_processing", "high_res_method", "tile")
                        metrics.record_custom_metric("image_processing", "original_size", max(image_input.size))
                        metrics.stop_operation("process_image_input", op_time, success=True)
                        return result
                    else:
                        result = handle_high_resolution_image(
                            image_input, 
                            method="resize",
                            target_size=(max_image_size, max_image_size)
                        )
                        metrics.record_custom_metric("image_processing", "high_res_method", "resize")
                        metrics.record_custom_metric("image_processing", "original_size", max(image_input.size))
                        metrics.stop_operation("process_image_input", op_time, success=True)
                        return result
            
            # Return as is for other cases, the provider's preprocess_image will handle it
            metrics.stop_operation("process_image_input", op_time, success=True)
            return image_input
            
        except Exception as e:
            metrics.stop_operation("process_image_input", op_time, success=False)
            metrics.record_vision_error("image_processing_error", {"error": str(e)})
            raise VisionModelError(f"Error processing image input: {str(e)}")
    
    @staticmethod
    def detect_device_capabilities() -> Dict[str, Any]:
        """
        Detect and return the capabilities of the current device.
        
        Uses caching to avoid repeated expensive detection operations.
        
        Returns:
            Dict[str, Any]: Device capabilities including GPU, memory, etc.
        """
        global _capabilities_cache, _capabilities_timestamp
        
        # Get metrics collector
        metrics = get_vision_metrics("vision_factory")
        op_time = metrics.start_operation("detect_device_capabilities")
        
        # Check if we have a recent cache
        current_time = time.time()
        if _capabilities_cache is not None and (current_time - _capabilities_timestamp) < _CAPABILITIES_CACHE_TTL:
            logger.debug("Using cached device capabilities")
            metrics.record_custom_metric("device_capabilities", "cache_hit", True)
            metrics.stop_operation("detect_device_capabilities", op_time, success=True)
            return _capabilities_cache
            
        metrics.record_custom_metric("device_capabilities", "cache_hit", False)
        
        capabilities = {
            'has_gpu': False,
            'gpu_memory_mb': 0,
            'cpu_memory_mb': 8192,  # Default assumption
            'internet_available': True,  # Assume internet is available by default
            'platform': platform.system(),
            'python_version': platform.python_version(),
        }
        
        try:
            # Detect GPU
            try:
                has_cuda = torch.cuda.is_available()
                capabilities['has_gpu'] = has_cuda
                
                if has_cuda:
                    # Get GPU memory
                    gpu_properties = torch.cuda.get_device_properties(0)
                    capabilities['gpu_memory_mb'] = gpu_properties.total_memory / (1024 * 1024)
                    capabilities['gpu_name'] = gpu_properties.name
                    capabilities['cuda_version'] = torch.version.cuda
                    
                    metrics.record_custom_metric("device_capabilities", "gpu_type", "cuda")
                    metrics.record_custom_metric("device_capabilities", "gpu_memory_mb", capabilities['gpu_memory_mb'])
                else:
                    # Check for MPS (Apple Silicon)
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        capabilities['has_gpu'] = True
                        capabilities['gpu_name'] = 'Apple Silicon (MPS)'
                        # Conservative estimate for M1/M2 shared memory
                        capabilities['gpu_memory_mb'] = 4000
                        
                        metrics.record_custom_metric("device_capabilities", "gpu_type", "mps")
                        metrics.record_custom_metric("device_capabilities", "gpu_memory_mb", capabilities['gpu_memory_mb'])
            except Exception as e:
                logger.warning(f"Error detecting GPU capabilities: {e}")
                metrics.record_vision_error("gpu_detection_error", {"error": str(e)})
            
            # Detect CPU memory
            try:
                import psutil
                capabilities['cpu_memory_mb'] = psutil.virtual_memory().total / (1024 * 1024)
                metrics.record_custom_metric("device_capabilities", "cpu_memory_mb", capabilities['cpu_memory_mb'])
            except ImportError:
                logger.warning("psutil not available, using default CPU memory estimate")
            
            # Check internet connectivity
            try:
                import socket
                socket.create_connection(("www.google.com", 80), timeout=1)
                metrics.record_custom_metric("device_capabilities", "internet_available", True)
            except OSError:
                capabilities['internet_available'] = False
                logger.warning("Internet connection not available")
                metrics.record_custom_metric("device_capabilities", "internet_available", False)
            
            # Record platform information
            metrics.record_custom_metric("device_capabilities", "platform", capabilities['platform'])
            
            # Cache the capabilities
            _capabilities_cache = capabilities
            _capabilities_timestamp = current_time
            
            # Successfully completed device detection
            metrics.stop_operation("detect_device_capabilities", op_time, success=True)
            return capabilities
            
        except Exception as e:
            logger.error(f"Error detecting device capabilities: {e}")
            metrics.stop_operation("detect_device_capabilities", op_time, success=False)
            metrics.record_vision_error("device_detection_error", {"error": str(e)})
            return capabilities  # Return default capabilities on error
    
    @staticmethod
    @handle_vision_errors
    def get_provider(provider_name: str, model_name: str, use_cache: bool = True, **kwargs) -> BaseVisionProvider:
        """
        Get a configured vision provider by name and model.
        
        Args:
            provider_name: Name of the provider to use (e.g., 'gemini', 'gpt4v', 'lightweight')
            model_name: Name of the model to use
            use_cache: Whether to use cached providers from the registry (default: True)
            **kwargs: Additional configuration options
            
        Returns:
            BaseVisionProvider: An initialized vision provider
            
        Raises:
            ProviderNotFoundError: If the requested provider is not found
            ModelUnavailableError: If the requested model is unavailable
        """
        # Get metrics collector
        metrics = get_vision_metrics("vision_factory")
        op_time = metrics.start_operation("get_provider")
        
        try:
            # Map provider_name to service_type
            service_type_map = {
                # Standard providers
                'gemini': 'gemini_vision',
                'google': 'gemini_vision',
                'google-gemini': 'gemini_vision',
                
                'gpt4v': 'gpt4_vision',
                'gpt-4-vision': 'gpt4_vision',
                'gpt4-vision': 'gpt4_vision',
                'openai': 'gpt4_vision',
                
                'qwen': 'qwen_vision',
                'qwen-vl': 'qwen_vision',
                'alibaba': 'qwen_vision',
                
                # Lightweight models and aliases
                'lightweight': 'lightweight_vision',
                'clip': 'lightweight_vision',
                'blip': 'lightweight_vision',
                'phi': 'lightweight_vision',
                'clip_phi': 'lightweight_vision',
                'minigpt4': 'lightweight_vision',
                'paligemma': 'lightweight_vision'
            }
            
            # Check if provider exists
            service_type = service_type_map.get(provider_name.lower())
            if not service_type:
                raise ProviderNotFoundError(f"Vision provider '{provider_name}' not found")
                
            # Record provider request in metrics
            metrics.record_custom_metric("provider_request", "provider_name", provider_name)
            metrics.record_custom_metric("provider_request", "model_name", model_name)
            
            # Check ModelRegistry for existing provider if using cache
            if use_cache:
                existing_provider = get_service(service_type, model_name)
                if existing_provider:
                    metrics.record_custom_metric("provider_request", "from_registry", True)
                    metrics.stop_operation("get_provider", op_time, success=True)
                    return existing_provider
            
            # Create configuration for the provider
            config = {
                'vision_type': provider_name,
                'provider_type': provider_name,
                'model': model_name,
                'model_name': model_name,  # For backward compatibility
                **kwargs  # Additional configuration
            }
            
            # Create and initialize the provider through factory
            provider = VisionProviderFactory.create_provider(config)
            
            metrics.stop_operation("get_provider", op_time, success=True)
            return provider
            
        except ProviderNotFoundError as e:
            # Record error and see if we can suggest alternatives
            metrics.stop_operation("get_provider", op_time, success=False)
            metrics.record_vision_error("provider_not_found", {
                "error": str(e),
                "provider_name": provider_name
            })
            
            # Suggest similar providers
            similar_providers = VisionProviderFactory._find_similar_providers(provider_name)
            if similar_providers:
                suggestion = f"Provider '{provider_name}' not found. Did you mean: {', '.join(similar_providers)}?"
                logger.warning(suggestion)
                raise ProviderNotFoundError(suggestion)
            raise e
            
        except ModelUnavailableError as e:
            # Record error and try fallback if available
            metrics.stop_operation("get_provider", op_time, success=False)
            metrics.record_vision_error("model_unavailable", {
                "error": str(e),
                "provider_name": provider_name,
                "model_name": model_name
            })
            
            # Try to find a fallback model
            fallback_model = VisionProviderFactory._find_fallback_model(provider_name, model_name)
            if fallback_model and kwargs.get('allow_fallback', True):
                logger.warning(f"Model {model_name} unavailable, falling back to {fallback_model}")
                # Call get_provider recursively with fallback model
                return VisionProviderFactory.get_provider(
                    provider_name, 
                    fallback_model,
                    use_cache=use_cache,
                    **{**kwargs, 'allow_fallback': False}  # Prevent infinite recursion
                )
            raise e
            
        except Exception as e:
            # Record unexpected error
            metrics.stop_operation("get_provider", op_time, success=False)
            metrics.record_vision_error("get_provider_error", {
                "error": str(e),
                "provider_name": provider_name,
                "model_name": model_name
            })
            logger.error(f"Error getting vision provider: {e}")
            raise ModelUnavailableError(
                f"Failed to get vision provider: {e}",
                model_name=model_name
            )
    
    @staticmethod
    def list_providers() -> List[Dict[str, Any]]:
        """
        List all available vision providers with their capabilities.
        
        Returns:
            List[Dict[str, Any]]: List of provider information dictionaries
        """
        # Get metrics collector
        metrics = get_vision_metrics("vision_factory")
        op_time = metrics.start_operation("list_providers")
        
        try:
            # Define known providers and their capabilities
            providers = [
                {
                    'name': 'gemini',
                    'display_name': 'Google Gemini Vision',
                    'description': 'Google\'s multimodal AI model capable of understanding images',
                    'requires_api_key': True,
                    'supports_multiple_images': True,
                    'is_cloud_based': True,
                    'max_image_size': 4096
                },
                {
                    'name': 'gpt4v',
                    'display_name': 'OpenAI GPT-4 Vision',
                    'description': 'OpenAI\'s vision-capable large language model',
                    'requires_api_key': True,
                    'supports_multiple_images': True,
                    'is_cloud_based': True,
                    'max_image_size': 2048
                },
                {
                    'name': 'qwen',
                    'display_name': 'Qwen-VL',
                    'description': 'Vision-language model from Alibaba that runs locally',
                    'requires_api_key': False,
                    'supports_multiple_images': False,
                    'is_cloud_based': False,
                    'max_image_size': 1024
                },
                {
                    'name': 'lightweight',
                    'display_name': 'Lightweight Vision Models',
                    'description': 'Collection of efficient vision models for various tasks',
                    'requires_api_key': False,
                    'supports_multiple_images': False,
                    'is_cloud_based': False,
                    'max_image_size': 1024
                }
            ]
            
            # Check which providers are actually available (based on imports)
            available_providers = []
            for provider in providers:
                try:
                    # Import check - don't actually create the provider
                    if provider['name'] == 'gemini':
                        from models_app.ai_models.vision.gemini.service import GeminiVisionService
                        provider['available'] = True
                    elif provider['name'] == 'gpt4v':
                        from models_app.ai_models.vision.gpt4v.service import GPT4VisionService
                        provider['available'] = True
                    elif provider['name'] == 'qwen':
                        from models_app.ai_models.vision.qwen.service import QwenVisionService
                        provider['available'] = True
                    elif provider['name'] == 'lightweight':
                        from models_app.ai_models.vision.lightweight.service import LightweightVisionService
                        provider['available'] = True
                    else:
                        provider['available'] = False
                        
                    # Check for API keys if needed
                    if provider['requires_api_key']:
                        if provider['name'] == 'gemini':
                            provider['has_api_key'] = bool(os.environ.get('GOOGLE_API_KEY', ''))
                        elif provider['name'] == 'gpt4v':
                            provider['has_api_key'] = bool(os.environ.get('OPENAI_API_KEY', ''))
                        else:
                            provider['has_api_key'] = False
                    else:
                        provider['has_api_key'] = True  # Not required
                        
                    available_providers.append(provider)
                except ImportError:
                    # Provider module not available
                    provider['available'] = False
                    provider['has_api_key'] = False
                    available_providers.append(provider)
            
            # Record metrics
            metrics.record_custom_metric("provider_listing", "provider_count", len(available_providers))
            metrics.record_custom_metric("provider_listing", "available_count", 
                                     sum(1 for p in available_providers if p.get('available', False)))
            
            metrics.stop_operation("list_providers", op_time, success=True)
            return available_providers
            
        except Exception as e:
            logger.error(f"Error listing providers: {e}")
            metrics.stop_operation("list_providers", op_time, success=False)
            metrics.record_vision_error("list_providers_error", {"error": str(e)})
            return []  # Return empty list on error
    
    @staticmethod
    def list_models(provider_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        List available models for all providers or a specific provider.
        
        Args:
            provider_name: Optional name of a specific provider
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Mapping from provider names to lists of model information
            
        Raises:
            ProviderNotFoundError: If a specific provider was requested but not found
        """
        # Get metrics collector
        metrics = get_vision_metrics("vision_factory")
        op_time = metrics.start_operation("list_models")
        
        try:
            # Define models for each provider
            all_models = {
                'gemini': [
                    {
                        'name': 'gemini-pro-vision',
                        'display_name': 'Gemini Pro Vision',
                        'description': 'Google\'s vision-capable large language model',
                        'supports_multiple_images': True,
                        'max_image_size': 4096,
                        'is_default': True
                    }
                ],
                'gpt4v': [
                    {
                        'name': 'gpt-4-vision-preview',
                        'display_name': 'GPT-4 Vision (Preview)',
                        'description': 'OpenAI\'s vision-capable GPT-4 model',
                        'supports_multiple_images': True,
                        'max_image_size': 2048,
                        'is_default': False
                    },
                    {
                        'name': 'gpt-4o',
                        'display_name': 'GPT-4o',
                        'description': 'OpenAI\'s latest omnimodal model with vision capabilities',
                        'supports_multiple_images': True,
                        'max_image_size': 2048,
                        'is_default': True
                    }
                ],
                'qwen': [
                    {
                        'name': 'Qwen/Qwen-VL-Chat',
                        'display_name': 'Qwen-VL-Chat',
                        'description': 'Qwen vision-language model optimized for chat',
                        'supports_multiple_images': False,
                        'max_image_size': 1024,
                        'is_default': False
                    },
                    {
                        'name': 'Qwen/Qwen2-VL-7B-Instruct',
                        'display_name': 'Qwen2-VL-7B',
                        'description': 'Qwen2 vision-language model (7B parameters)',
                        'supports_multiple_images': False,
                        'max_image_size': 1024,
                        'is_default': True
                    }
                ],
                'lightweight': [
                    {
                        'name': 'clip',
                        'display_name': 'CLIP',
                        'description': 'Contrastive Language-Image Pre-training model',
                        'supports_multiple_images': False,
                        'max_image_size': 224,
                        'is_default': False
                    },
                    {
                        'name': 'clip_phi',
                        'display_name': 'CLIP-Phi',
                        'description': 'CLIP with Phi small language model for efficient understanding',
                        'supports_multiple_images': False,
                        'max_image_size': 224,
                        'is_default': True
                    },
                    {
                        'name': 'blip',
                        'display_name': 'BLIP',
                        'description': 'Bootstrapping Language-Image Pre-training model',
                        'supports_multiple_images': False,
                        'max_image_size': 384,
                        'is_default': False
                    },
                    {
                        'name': 'paligemma',
                        'display_name': 'PaliGemma',
                        'description': 'Google\'s efficient vision-language model',
                        'supports_multiple_images': False,
                        'max_image_size': 448,
                        'is_default': False
                    }
                ]
            }
            
            # If a specific provider was requested, return only its models
            if provider_name:
                provider_name = provider_name.lower()
                if provider_name not in all_models:
                    metrics.stop_operation("list_models", op_time, success=False)
                    metrics.record_vision_error("provider_not_found", {"provider_name": provider_name})
                    raise ProviderNotFoundError(f"Vision provider '{provider_name}' not found")
                
                metrics.record_custom_metric("model_listing", "provider", provider_name)
                metrics.record_custom_metric("model_listing", "model_count", len(all_models[provider_name]))
                metrics.stop_operation("list_models", op_time, success=True)
                
                return {provider_name: all_models[provider_name]}
            
            # Otherwise return all models
            metrics.record_custom_metric("model_listing", "provider_count", len(all_models))
            metrics.record_custom_metric("model_listing", "total_model_count", 
                                     sum(len(models) for models in all_models.values()))
            metrics.stop_operation("list_models", op_time, success=True)
            
            return all_models
            
        except ProviderNotFoundError:
            # Let this propagate up
            raise
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            metrics.stop_operation("list_models", op_time, success=False)
            metrics.record_vision_error("list_models_error", {"error": str(e)})
            
            if provider_name:
                raise ProviderNotFoundError(f"Error listing models for provider '{provider_name}': {e}")
            return {}  # Return empty dict on error

    @staticmethod
    @handle_vision_errors
    def get_recommended_provider(task: str, device_capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recommend the best vision provider for a specific task based on
        available device capabilities.
        
        Args:
            task: The task to perform (image_understanding, document_processing, visual_qa, etc.)
            device_capabilities: Information about available hardware (optional)
            
        Returns:
            Dict: Configuration for the recommended provider
        """
        # Get metrics collector
        metrics = get_vision_metrics("vision_factory")
        op_time = metrics.start_operation("get_recommended_provider") 
        
        try:
            if device_capabilities is None:
                device_capabilities = VisionProviderFactory.detect_device_capabilities()
            
            # Extract device capabilities
            has_gpu = device_capabilities.get('has_gpu', False)
            gpu_memory = device_capabilities.get('gpu_memory_mb', 0)
            cpu_memory = device_capabilities.get('cpu_memory_mb', 8192)  # 8GB as default
            internet_available = device_capabilities.get('internet_available', True)
            
            # Check for API keys for cloud services
            has_openai_key = bool(os.environ.get('OPENAI_API_KEY', ''))
            has_google_key = bool(os.environ.get('GOOGLE_API_KEY', ''))
            
            # Record recommendation request
            metrics.record_custom_metric("provider_recommendation", "task", task)
            metrics.record_custom_metric("provider_recommendation", "has_gpu", has_gpu)
            metrics.record_custom_metric("provider_recommendation", "has_openai_key", has_openai_key)
            metrics.record_custom_metric("provider_recommendation", "has_google_key", has_google_key)
            
            # Base configuration
            config = {
                'provider_type': 'vision',
                'temperature': 0.7,
                'max_image_size': 1024
            }
            
            # Select provider based on task, device capabilities, and available credentials
            if task == 'image_understanding':
                if internet_available and has_openai_key:
                    # Prefer GPT-4 Vision if available
                    config.update({
                        'vision_type': 'gpt4v',
                        'model': 'gpt-4o',
                        'detail_level': 'auto'
                    })
                elif internet_available and has_google_key:
                    # Or use Gemini if available
                    config.update({
                        'vision_type': 'gemini',
                        'model': 'gemini-pro-vision'
                    })
                elif has_gpu and gpu_memory >= 16000:
                    # For high-end local GPU
                    config.update({
                        'vision_type': 'qwen',
                        'model': 'Qwen/Qwen2-VL-7B-Instruct',
                        'quantization_level': '4bit'
                    })
                else:
                    # For low resource environments
                    config.update({
                        'vision_type': 'lightweight',
                        'model_type': 'clip_phi',
                        'quantization_level': '8bit'
                    })
            
            elif task == 'document_processing':
                if internet_available and has_openai_key:
                    config.update({
                        'vision_type': 'gpt4v',
                        'model': 'gpt-4o',
                        'detail_level': 'high'
                    })
                elif has_gpu and gpu_memory >= 24000:
                    config.update({
                        'vision_type': 'qwen',
                        'model': 'Qwen/Qwen2-VL-7B-Instruct',
                        'quantization_level': '4bit'
                    })
                else:
                    # For lower resources - try lightweight model with high quality output
                    config.update({
                        'vision_type': 'lightweight',
                        'model_type': 'blip',
                        'quantization_level': '8bit'
                    })
            
            elif task == 'visual_qa':
                if has_gpu and gpu_memory >= 16000:
                    config.update({
                        'vision_type': 'qwen',
                        'model': 'Qwen/Qwen2-VL-7B-Instruct',
                        'quantization_level': '4bit',
                        'supports_multiple_images': True
                    })
                elif internet_available and has_google_key:
                    config.update({
                        'vision_type': 'gemini',
                        'model': 'gemini-pro-vision'
                    })
                else:
                    config.update({
                        'vision_type': 'lightweight',
                        'model_type': 'blip',
                        'use_vqa_mode': True, 
                        'quantization_level': '4bit'
                    })
            
            else:  # General default
                if has_gpu and gpu_memory >= 8000:
                    config.update({
                        'vision_type': 'lightweight',
                        'model_type': 'paligemma',
                        'quantization_level': '4bit'
                    })
                else:
                    config.update({
                        'vision_type': 'lightweight',
                        'model_type': 'clip_phi',
                        'quantization_level': '8bit'
                    })
            
            # Record the recommendation result
            metrics.record_custom_metric("provider_recommendation", "recommended_type", config.get('vision_type'))
            metrics.record_custom_metric("provider_recommendation", "recommended_model", config.get('model', config.get('model_type', 'unknown')))
            
            # Stop timing and return config
            metrics.stop_operation("get_recommended_provider", op_time, success=True)
            return config
            
        except Exception as e:
            # Log error and record in metrics
            logger.error(f"Error recommending vision provider: {e}")
            metrics.stop_operation("get_recommended_provider", op_time, success=False)
            metrics.record_vision_error("recommendation_error", {"error": str(e), "task": task})
            
            # Return lightweight fallback config on error
            fallback = {
                'provider_type': 'vision',
                'vision_type': 'lightweight',
                'model_type': 'clip_phi',
                'quantization_level': '8bit',
                'error': str(e)
            }
            return fallback 

    @staticmethod
    def _find_similar_providers(provider_name: str) -> List[str]:
        """Find similar provider names to suggest alternatives"""
        # List of available provider names (keys from the map in get_provider)
        available_providers = ['gemini', 'gpt4v', 'qwen', 'lightweight',
                             'clip', 'blip', 'phi', 'clip_phi', 'minigpt4']
        
        # Find providers with similar names
        provider_name = provider_name.lower()
        similar = []
        
        for name in available_providers:
            # Simple fuzzy matching based on substring and distance
            if name in provider_name or provider_name in name:
                similar.append(name)
            elif distance(name, provider_name) <= 2:  # Allow up to 2 edits
                similar.append(name)
                
        return similar[:3]  # Return up to 3 suggestions
    
    @staticmethod
    def _find_fallback_model(provider_name: str, model_name: str) -> Optional[str]:
        """Find a fallback model for the specified provider and model"""
        # Map of fallback models for each provider
        fallback_map = {
            'gemini': {
                'gemini-pro-vision': 'gemini-1.0-pro-vision',
                'gemini-1.5-flash': 'gemini-1.0-pro-vision'
            },
            'gpt4v': {
                'gpt-4-vision-preview': 'gpt-4-turbo-vision',
                'gpt-4-vision': 'gpt-4-turbo-vision',
                'gpt-4v': 'gpt-4-turbo-vision',
            },
            'qwen': {
                'qwen-vl-plus': 'qwen-vl-chat',
                'qwen-vl-max': 'qwen-vl-chat',
            },
            'lightweight': {
                'clip-vit-large': 'clip-vit-base',
                'blip2-flan-t5': 'blip2-opt',
            }
        }
        
        # Get fallbacks for this provider
        provider_fallbacks = fallback_map.get(provider_name.lower(), {})
        
        # Check if we have a specific fallback for this model
        if model_name in provider_fallbacks:
            return provider_fallbacks[model_name]
            
        # Return first fallback model as default if any exist
        if provider_fallbacks:
            return list(provider_fallbacks.values())[0]
            
        return None

    @staticmethod
    def clear_cache(provider_name: Optional[str] = None, model_name: Optional[str] = None):
        """
        Clear the provider and/or capabilities cache.
        
        Args:
            provider_name: Optional provider name to clear specific provider cache
            model_name: Optional model name (used with provider_name) to clear specific model cache
        """
        global _provider_cache, _capabilities_cache, _capabilities_timestamp
        
        metrics = get_vision_metrics("vision_factory")
        
        if provider_name and model_name:
            # Clear specific provider/model
            prefix = f"{provider_name.lower()}:{model_name.lower()}"
            keys_to_remove = [k for k in _provider_cache.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                _provider_cache.pop(k, None)
            logger.debug(f"Cleared cache for provider {provider_name} and model {model_name}")
            metrics.record_custom_metric("cache_operations", "specific_clear", len(keys_to_remove))
            
        elif provider_name:
            # Clear all models for this provider
            prefix = f"{provider_name.lower()}:"
            keys_to_remove = [k for k in _provider_cache.keys() if k.startswith(prefix)]
            for k in keys_to_remove:
                _provider_cache.pop(k, None)
            logger.debug(f"Cleared cache for provider {provider_name}")
            metrics.record_custom_metric("cache_operations", "provider_clear", len(keys_to_remove))
            
        else:
            # Clear all caches
            count = len(_provider_cache)
            _provider_cache.clear()
            _capabilities_cache = None
            _capabilities_timestamp = 0
            logger.debug("Cleared all provider and capabilities caches")
            metrics.record_custom_metric("cache_operations", "full_clear", count) 

    @staticmethod
    def cleanup_providers():
        """
        Clean up all providers managed by the Registry and factory.
        
        This method ensures proper resource cleanup for all vision providers.
        """
        # Get metrics collector
        metrics = get_vision_metrics("vision_factory")
        op_time = metrics.start_operation("cleanup_providers")
        
        try:
            # Get all vision service types
            vision_services = [
                'gemini_vision',
                'gpt4_vision', 
                'qwen_vision',
                'lightweight_vision'
            ]
            
            cleanup_count = 0
            
            # Clean up each service type
            for service_type in vision_services:
                # Get all registered instances of this service type
                services = ModelRegistry.get_all_services_of_type(service_type)
                
                for service_id, service in services.items():
                    if hasattr(service, 'cleanup') and callable(service.cleanup):
                        try:
                            service.cleanup()
                            cleanup_count += 1
                            logger.debug(f"Cleaned up vision provider: {service_id}")
                        except Exception as e:
                            logger.error(f"Error cleaning up vision provider {service_id}: {e}")
                            
                # Remove all services of this type from registry
                ModelRegistry.remove_all_services_of_type(service_type)
            
            metrics.record_custom_metric("cleanup_operations", "provider_cleanup", cleanup_count)
            logger.info(f"Cleaned up {cleanup_count} vision providers")
            
            metrics.stop_operation("cleanup_providers", op_time, success=True)
        except Exception as e:
            metrics.stop_operation("cleanup_providers", op_time, success=False)
            logger.error(f"Error during provider cleanup: {e}")

    @staticmethod
    def _initialize_provider(provider, model_name):
        """Initialize a vision provider and handle any errors"""
        success = provider.initialize()
        if not success:
            raise ModelUnavailableError(
                f"Failed to initialize vision model: {model_name}",
                model_name=model_name
            )
        return provider

# Register vision providers with the ModelRegistry on module import
register_vision_providers() 