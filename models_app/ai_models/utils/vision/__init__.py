"""
Vision Utilities Module

Provides utilities for processing and managing images for vision model providers.
This module centralizes common image processing functions, error handling, 
and data structures used across different vision model providers.
"""

# Import image processing utilities
from .image_processing import (
    decode_base64_image,
    encode_image_to_base64,
    resize_image_for_model,
    download_and_process_image,
    handle_high_resolution_image,
    support_multiple_images
)

# Import error handling from common
from models_app.ai_models.utils.common.errors import (
    VisionModelError,
    ImageProcessingError,
    OCRError,
    DocumentVisionError,
    MultiImageProcessingError
)

from models_app.ai_models.utils.common.handlers import (
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

# Import metrics collection
from models_app.ai_models.utils.common.metrics import (
    get_vision_metrics,
    VisionMetricsCollector
)

# Export utilities
__all__ = [
    # Image processing
    'decode_base64_image',
    'encode_image_to_base64',
    'resize_image_for_model',
    'download_and_process_image',
    'handle_high_resolution_image',
    'support_multiple_images',
    
    # Error types
    'VisionModelError',
    'ImageProcessingError',
    'OCRError',
    'DocumentVisionError',
    'MultiImageProcessingError',
    
    # Error handlers
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
    
    # Metrics
    'get_vision_metrics',
    'VisionMetricsCollector'
]

# This module will contain vision-specific utilities in the future.
# For now, it's a placeholder for organization purposes. 