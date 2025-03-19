"""
Error handling decorators for AI model operations.

This module provides error handling decorators for all AI model operations,
including text (LLM), audio, and vision models. It consolidates handler
functionality across different model types for consistent error management.
"""

import logging
import functools
import asyncio
import time
from typing import Callable, Optional, Dict, Any, List, AsyncGenerator, Type, Union, Tuple

from error_handlers.common_handlers import handle_errors, measure_time, retry
from models_app.ai_models.utils.common.errors import (
    # General model errors
    ModelError,
    ModelNotFoundError,
    # Audio errors
    AudioModelError,
    TTSError,
    STTError,
    AudioProcessingError,
    # Vision errors
    VisionModelError,
    ImageProcessingError,
    OCRError,
    DocumentVisionError,
    MultiImageProcessingError
)

# Import LLM errors for text module compatibility
try:
    from models_app.llm_providers.utils.error_handling.errors import (
        LLMError,
        ProviderConnectionError,
        RateLimitError,
        TokenLimitError,
        ProviderResponseError,
        ModelUnavailableError
    )
    HAS_LLM_ERRORS = True
except ImportError:
    HAS_LLM_ERRORS = False
    # Create placeholder classes to avoid errors
    class LLMError(ModelError): pass
    class ProviderConnectionError(ModelError): pass
    class RateLimitError(ModelError): pass
    class TokenLimitError(ModelError): pass
    class ProviderResponseError(ModelError): pass
    class ModelUnavailableError(ModelError): pass

logger = logging.getLogger(__name__)

# Generic model error handlers
def default_error_handler(func: Callable) -> Callable:
    """Generic decorator to handle all model errors."""
    return handle_errors(error_types=[Exception], error_class=ModelError)(func)

def handle_model_errors(func: Callable) -> Callable:
    """Decorator to handle model-specific errors."""
    return handle_errors(error_types=[ModelError], error_class=ModelError)(func)

def retry_on_error(max_attempts: int = 3, delay: float = 1.0) -> Callable:
    """Decorator to retry operations on temporary errors."""
    return retry(max_attempts=max_attempts, delay=delay, exceptions=(ModelError,))

def time_execution(func: Callable) -> Callable:
    """Decorator to time the execution of model operations."""
    return measure_time(func)

# Audio-specific handlers
def handle_audio_errors(func: Callable) -> Callable:
    """Decorator to handle audio processing errors."""
    return handle_errors(error_types=[AudioModelError], error_class=AudioModelError)(func)

def handle_stt_errors(func: Callable) -> Callable:
    """Decorator to handle speech-to-text errors."""
    return handle_errors(error_types=[STTError], error_class=STTError)(func)

def handle_tts_errors(func: Callable) -> Callable:
    """Decorator to handle text-to-speech errors."""
    return handle_errors(error_types=[TTSError], error_class=TTSError)(func)

# Vision-specific handlers
def handle_vision_errors(func: Callable) -> Callable:
    """Decorator to handle vision processing errors."""
    return handle_errors(error_types=[VisionModelError], error_class=VisionModelError)(func)

def handle_image_processing_errors(func: Callable) -> Callable:
    """Decorator to handle image processing errors."""
    return handle_errors(error_types=[ImageProcessingError], error_class=ImageProcessingError)(func)

def handle_document_vision_errors(func: Callable) -> Callable:
    """Decorator to handle document vision processing errors."""
    return handle_errors(error_types=[DocumentVisionError], error_class=DocumentVisionError)(func)

def handle_multi_image_processing_errors(func: Callable) -> Callable:
    """Decorator to handle errors when processing multiple images."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error processing multiple images: {str(e)}")
            
            # Get image count if available
            image_count = None
            if 'images' in kwargs and isinstance(kwargs['images'], list):
                image_count = len(kwargs['images'])
            
            # If already a MultiImageProcessingError, re-raise
            if isinstance(e, MultiImageProcessingError):
                raise
            
            # Otherwise wrap in MultiImageProcessingError
            raise MultiImageProcessingError(
                f"Error processing multiple images: {str(e)}",
                image_count=image_count,
                cause=e
            )
    return wrapper

def handle_ocr_errors(func: Callable) -> Callable:
    """Decorator to handle OCR processing errors."""
    return handle_errors(error_types=[OCRError], error_class=OCRError)(func)

# Text/LLM-specific handlers
def handle_llm_errors(func: Callable) -> Callable:
    """Decorator to handle LLM provider errors."""
    return handle_errors(error_types=[Exception], error_class=LLMError)(func)

def handle_provider_connection(func: Callable) -> Callable:
    """Decorator to handle provider connection errors with retry."""
    @functools.wraps(func)
    @retry(max_attempts=3, exceptions=(ProviderConnectionError,))
    @handle_errors(error_types=[Exception], error_class=ProviderConnectionError)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def handle_rate_limits(func: Callable) -> Callable:
    """Decorator to handle rate limit errors with exponential backoff."""
    @functools.wraps(func)
    @retry(max_attempts=3, delay=2.0, backoff_factor=3.0, 
           exceptions=(RateLimitError,))
    @handle_errors(error_types=[Exception], error_class=RateLimitError)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def validate_token_limits(max_tokens: int) -> Callable:
    """
    Decorator to validate token limits before making API calls.
    
    Args:
        max_tokens: Maximum allowed tokens
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        @handle_errors(error_types=[Exception], error_class=TokenLimitError)
        def wrapper(*args, **kwargs):
            # Get token count from kwargs or calculate
            token_count = kwargs.get('token_count', 0)
            if token_count > max_tokens:
                raise TokenLimitError(
                    f"Token limit exceeded: {token_count} > {max_tokens}",
                    token_count=token_count,
                    max_tokens=max_tokens
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Model fallback handler
def with_model_fallback(fallback_models: List[str]) -> Callable:
    """
    Decorator to handle model fallbacks.
    
    Args:
        fallback_models: List of model names to try in order
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for model in fallback_models:
                try:
                    kwargs['model'] = model
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(f"Model {model} failed: {str(e)}, trying next model")
            
            # If all models failed, raise the last error
            if last_error:
                if HAS_LLM_ERRORS:
                    raise ModelUnavailableError(
                        f"All models failed: {str(last_error)}",
                        model_name=fallback_models[-1]
                    ) from last_error
                else:
                    raise ModelError(
                        f"All models failed: {str(last_error)}",
                    ) from last_error
        return wrapper
    return decorator

def handle_streaming_errors(func: Callable) -> Callable:
    """Decorator to handle errors in streaming responses."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> AsyncGenerator:
        try:
            async for chunk in func(*args, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            # Yield error message as final chunk
            yield {
                "error": True,
                "message": str(e),
                "error_type": type(e).__name__
            }
            raise
    return wrapper

def manage_context_window(
    max_context_length: int,
    truncation_strategy: str = "end"  # or "start" or "middle"
) -> Callable:
    """
    Decorator to manage context window size.
    
    Args:
        max_context_length: Maximum context length in tokens
        truncation_strategy: How to truncate if context is too long
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = kwargs.get('context', '')
            if len(context) > max_context_length:
                if truncation_strategy == "end":
                    context = context[:max_context_length]
                elif truncation_strategy == "start":
                    context = context[-max_context_length:]
                else:  # middle
                    half = max_context_length // 2
                    context = context[:half] + context[-half:]
                
                kwargs['context'] = context
                logger.warning(f"Context truncated using {truncation_strategy} strategy")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Composite handlers
llm_processing = handle_llm_errors
llm_processing_with_retry = handle_provider_connection
llm_processing_with_rate_limit = handle_rate_limits

# Audio processing specific handlers
stt_processing = handle_stt_errors
tts_processing = handle_tts_errors

# Vision processing composite handlers
vision_processing = handle_vision_errors
image_processing = handle_image_processing_errors
document_vision_processing = handle_document_vision_errors
multi_image_processing = handle_multi_image_processing_errors
ocr_processing = handle_ocr_errors

# Complete processing with all checks and fallback
llm_processing_complete = validate_token_limits(4096)(
    with_model_fallback(['gpt-4', 'gpt-3.5-turbo'])(
        handle_rate_limits(
            handle_provider_connection(
                measure_time(handle_llm_errors)
            )
        )
    )
)

# Streaming processing with all checks
streaming_processing_complete = handle_streaming_errors(
    validate_token_limits(4096)(
        handle_rate_limits(
            handle_provider_connection(
                measure_time(handle_llm_errors)
            )
        )
    )
) 