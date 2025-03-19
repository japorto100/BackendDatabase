"""
Error types for LLM providers.

This module defines error types specific to LLM provider operations,
inheriting from application-wide base errors.
"""

from error_handlers.base_errors import BaseApplicationError
from typing import Optional, Any

class ModelError(BaseApplicationError):
    """Base exception for all model errors."""
    
    def __init__(self, message: str, error_code: str = None, **kwargs):
        super().__init__(message, error_code=error_code or "model_error", **kwargs)
        self.details = self.details or {}
        self.details["module"] = "ai_models"

class ModelNotFoundError(ModelError):
    """Error when a model is not found."""
    
    def __init__(self, message: str, model_name: str = None, **kwargs):
        super().__init__(message, error_code="model_not_found", **kwargs)
        self.model_name = model_name
        self.details["model_name"] = model_name

# Text/LLM errors
class LLMError(ModelError):
    """Base exception for LLM processing errors."""
    
    def __init__(self, message: str, error_code: str = None, **kwargs):
        super().__init__(message, error_code=error_code or "llm_error", **kwargs)
        self.details = self.details or {}
        self.details["module"] = "llm_providers"

class ProviderConnectionError(LLMError):
    """Error connecting to LLM provider."""
    
    def __init__(self, message: str, provider: str = None, **kwargs):
        super().__init__(message, error_code="provider_connection_error", **kwargs)
        self.provider = provider
        self.details["provider"] = provider

class RateLimitError(LLMError):
    """Rate limit exceeded for LLM provider."""
    
    def __init__(self, message: str, provider: str = None, 
                 limit_type: str = None, **kwargs):
        super().__init__(message, error_code="rate_limit_error", **kwargs)
        self.provider = provider
        self.limit_type = limit_type
        self.details.update({
            "provider": provider,
            "limit_type": limit_type
        })

class TokenLimitError(LLMError):
    """Token limit exceeded for LLM request."""
    
    def __init__(self, message: str, token_count: int = None, 
                 max_tokens: int = None, **kwargs):
        super().__init__(message, error_code="token_limit_error", **kwargs)
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.details.update({
            "token_count": token_count,
            "max_tokens": max_tokens
        })

class ProviderResponseError(LLMError):
    """Error in provider response."""
    
    def __init__(self, message: str, provider: str = None, 
                 response_code: str = None, **kwargs):
        super().__init__(message, error_code="provider_response_error", **kwargs)
        self.provider = provider
        self.response_code = response_code
        self.details.update({
            "provider": provider,
            "response_code": response_code
        })

class ModelUnavailableError(LLMError):
    """Requested model is unavailable."""
    
    def __init__(self, message: str, model_name: str = None, 
                 provider: str = None, **kwargs):
        super().__init__(message, error_code="model_unavailable_error", **kwargs)
        self.model_name = model_name
        self.provider = provider
        self.details.update({
            "model_name": model_name,
            "provider": provider
        })

# Vision errors
class VisionModelError(ModelError):
    """Base exception for vision model errors."""
    def __init__(self, message="Error in vision model processing", **kwargs):
        super().__init__(message, error_code="vision_model_error", **kwargs)

class ImageProcessingError(VisionModelError):
    """Error during image processing or analysis."""
    def __init__(self, message="Error processing image", details=None, **kwargs):
        super().__init__(message, error_code="image_processing_error", **kwargs)
        if details:
            self.details.update(details)

class OCRError(VisionModelError):
    """Error during optical character recognition."""
    def __init__(self, message="Error in text extraction from image", details=None, **kwargs):
        super().__init__(message, error_code="ocr_error", **kwargs)
        if details:
            self.details.update(details)

class DocumentVisionError(VisionModelError):
    """Error during document processing with vision models."""
    def __init__(self, message="Error in document vision processing", details=None, **kwargs):
        super().__init__(message, error_code="document_vision_error", **kwargs)
        if details:
            self.details.update(details)

# Audio errors
class AudioModelError(ModelError):
    """Base exception for audio model errors."""
    def __init__(self, message="Error in audio model processing", **kwargs):
        super().__init__(message, error_code="audio_model_error", **kwargs)

class TTSError(AudioModelError):
    """Error during text-to-speech conversion."""
    def __init__(self, message="Error in text-to-speech conversion", details=None, **kwargs):
        super().__init__(message, error_code="tts_error", **kwargs)
        if details:
            self.details.update(details)

class STTError(AudioModelError):
    """Error during speech-to-text conversion."""
    def __init__(self, message="Error in speech-to-text conversion", details=None, **kwargs):
        super().__init__(message, error_code="stt_error", **kwargs)
        if details:
            self.details.update(details)

class AudioProcessingError(AudioModelError):
    """Error during audio processing operations."""
    def __init__(self, message="Error processing audio data", details=None, **kwargs):
        super().__init__(message, error_code="audio_processing_error", **kwargs)
        if details:
            self.details.update(details)

class MultiImageProcessingError(VisionModelError):
    """Error raised when processing multiple images fails."""
    
    def __init__(self, message: str, *, model_name: Optional[str] = None, 
                provider: Optional[str] = None, image_count: Optional[int] = None, 
                cause: Optional[Exception] = None):
        self.image_count = image_count
        super().__init__(message, model_name=model_name, provider=provider, cause=cause) 