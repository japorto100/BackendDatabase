"""
Enhanced error handling system with analytics integration.

This module provides a comprehensive error handling system that integrates with:
- Analytics monitoring
- Performance measurement
- Input validation
- Retry mechanisms
"""

import logging
import functools
import traceback
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Union
import time
from datetime import datetime

from models_app.vision.utils.error_handling.vision_errors import (
    VisionError,
    DocumentProcessingError,
    ValidationError,
    ResourceError
)
from analytics_app.utils import (
    monitor_document_performance,
    monitor_ocr_performance,
    monitor_colpali_performance,
    monitor_fusion_performance,
    monitor_selector_performance
)

logger = logging.getLogger(__name__)
T = TypeVar('T')

class ErrorHandlingService:
    """
    Centralized error handling service with analytics integration.
    
    This class provides:
    - Error handling decorators
    - Performance monitoring
    - Analytics integration
    - Retry mechanisms
    - Input validation
    """
    
    def __init__(self, logger=None):
        """Initialize the error handling service."""
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_document_processing_error(self, func: Callable) -> Callable:
        """Decorator for handling document processing errors."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except VisionError as e:
                # Already a vision error, just log and return
                self.logger.error(f"Vision error in {func.__name__}: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                return {
                    "error": True,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "error_code": e.error_code,
                    "details": e.details,
                    "function": func.__name__,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                # Wrap in DocumentProcessingError
                self.logger.error(f"Error in {func.__name__}: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                
                # Extract metadata if available
                metadata = kwargs.get('metadata', {})
                if not metadata and len(args) > 1 and isinstance(args[1], dict):
                    metadata = args[1]
                
                error = DocumentProcessingError(
                    message=str(e),
                    processing_stage=func.__name__,
                    details=metadata
                )
                
                return {
                    "error": True,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "error_code": error.error_code,
                    "details": error.details,
                    "function": func.__name__,
                    "timestamp": datetime.now().isoformat()
                }
        return wrapper

    # ... rest of the existing methods ...

# Create singleton instance
error_handler = ErrorHandlingService()

# Pre-defined combined decorators
document_processing = error_handler.with_error_handling_and_analytics(monitor_document_performance)
ocr_processing = error_handler.with_error_handling_and_analytics(monitor_ocr_performance)
colpali_processing = error_handler.with_error_handling_and_analytics(monitor_colpali_performance)
fusion_processing = error_handler.with_error_handling_and_analytics(monitor_fusion_performance)
selector_processing = error_handler.with_error_handling_and_analytics(monitor_selector_performance)

# Pre-defined combined decorators with timing
document_processing_with_timing = error_handler.with_timing(document_processing)
ocr_processing_with_timing = error_handler.with_timing(ocr_processing)
colpali_processing_with_timing = error_handler.with_timing(colpali_processing)
fusion_processing_with_timing = error_handler.with_timing(fusion_processing)
selector_processing_with_timing = error_handler.with_timing(selector_processing)

# Complete versions with validation, timing, error handling, and analytics
document_processing_complete = error_handler.with_validation(document_processing_with_timing)
ocr_processing_complete = error_handler.with_validation(ocr_processing_with_timing)
colpali_processing_complete = error_handler.with_validation(colpali_processing_with_timing)
fusion_processing_complete = error_handler.with_validation(fusion_processing_with_timing)
selector_processing_complete = error_handler.with_validation(selector_processing_with_timing) 