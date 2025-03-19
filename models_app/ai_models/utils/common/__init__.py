"""
Common utilities for AI models.

This package provides common utilities for AI models, including:
- Error handling and definitions
- Configuration management
- Metrics collection
- Base service classes
"""

# Import error types for easier access
from models_app.ai_models.utils.common.errors import (
    # General model errors
    ModelError,
    ModelNotFoundError,
    
    # Text/LLM errors
    LLMError,
    ProviderConnectionError,
    RateLimitError,
    TokenLimitError,
    ProviderResponseError,
    ModelUnavailableError,
    
    # Audio errors
    AudioModelError,
    STTError,
    TTSError,
    AudioProcessingError,
    
    # Vision errors
    VisionModelError,
    ImageProcessingError,
    OCRError,
    DocumentVisionError,
    MultiImageProcessingError
)

# Import configuration utilities
from models_app.ai_models.utils.common.config import (
    BaseConfig,
    LLMConfig,
    BaseAudioConfig,
    STTConfig,
    TTSConfig,
    ConfigManager,
    get_stt_config,
    get_tts_config,
    get_llm_config,
    set_stt_config,
    set_tts_config,
    set_llm_config
)

# Import metrics utilities
from models_app.ai_models.utils.common.metrics import (
    MetricsCollector,
    STTMetricsCollector,
    TTSMetricsCollector,
    LLMMetricsCollector,
    VisionMetricsCollector,
    get_stt_metrics,
    get_tts_metrics,
    get_llm_metrics,
    get_vision_metrics,
    export_all_metrics,
)

# Import service utilities
from models_app.ai_models.utils.common.ai_base_service import (
    BaseModelService,
    ModelRegistry,
    register_service_type,
    get_service,
    create_service,
    get_or_create_service,
    list_services,
    cleanup_all,
)

# Import handler utilities
from models_app.ai_models.utils.common.handlers import (
    # General handlers
    default_error_handler,
    handle_model_errors,
    retry_on_error,
    time_execution,
    
    # LLM handlers
    handle_llm_errors,
    handle_provider_connection,
    handle_rate_limits,
    validate_token_limits,
    with_model_fallback,
    handle_streaming_errors,
    LLMErrorHandlingService,
    error_handler,
    
    # Audio handlers
    handle_audio_errors,
    handle_stt_errors,
    handle_tts_errors,
    
    # Vision handlers
    handle_vision_errors,
    handle_image_processing_errors,
    handle_document_vision_errors,
    handle_multi_image_processing_errors,
    handle_ocr_errors,
    vision_processing,
    image_processing,
    document_vision_processing,
    multi_image_processing,
    ocr_processing
)

__all__ = [
    # Error types
    'ModelError',
    'ModelNotFoundError',
    
    # LLM error types
    'LLMError',
    'ProviderConnectionError',
    'RateLimitError',
    'TokenLimitError',
    'ProviderResponseError',
    'ModelUnavailableError',
    
    # Audio error types
    'AudioModelError',
    'STTError',
    'TTSError',
    'AudioProcessingError',
    
    # Vision error types
    'VisionModelError',
    'ImageProcessingError',
    'OCRError',
    'DocumentVisionError',
    'MultiImageProcessingError',
    
    # Configuration
    'BaseConfig',
    'LLMConfig',
    'BaseAudioConfig',
    'STTConfig',
    'TTSConfig',
    'ConfigManager',
    'get_stt_config',
    'get_tts_config',
    'get_llm_config',
    'set_stt_config',
    'set_tts_config',
    'set_llm_config',
    
    # Metrics
    'MetricsCollector',
    'STTMetricsCollector',
    'TTSMetricsCollector',
    'LLMMetricsCollector',
    'VisionMetricsCollector',
    'get_stt_metrics',
    'get_tts_metrics',
    'get_llm_metrics',
    'get_vision_metrics',
    'export_all_metrics',
    
    # Services
    'BaseModelService',
    'ModelRegistry',
    'register_service_type',
    'get_service',
    'create_service',
    'get_or_create_service',
    'list_services',
    'cleanup_all',
    
    # General Handlers
    'default_error_handler',
    'handle_model_errors',
    'retry_on_error',
    'time_execution',
    
    # LLM Handlers
    'handle_llm_errors',
    'handle_provider_connection',
    'handle_rate_limits',
    'validate_token_limits',
    'with_model_fallback',
    'handle_streaming_errors',
    'LLMErrorHandlingService',
    'error_handler',
    
    # Audio Handlers
    'handle_audio_errors',
    'handle_stt_errors',
    'handle_tts_errors',
    
    # Vision Handlers
    'handle_vision_errors',
    'handle_image_processing_errors',
    'handle_document_vision_errors',
    'handle_multi_image_processing_errors',
    'handle_ocr_errors',
    'vision_processing',
    'image_processing',
    'document_vision_processing',
    'multi_image_processing',
    'ocr_processing',
] 