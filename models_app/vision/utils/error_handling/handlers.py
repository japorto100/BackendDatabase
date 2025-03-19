"""
Vision-level error handling decorators that inherit from common handlers.
"""

from functools import wraps
import time
import logging
from typing import Any, Callable, Dict, Optional
import psutil

from error_handlers.common_handlers import (
    handle_errors,
    measure_time,
    retry
)
from .errors import (
    VisionError,
    VisionProcessingError,
    VisionResourceError,
    VisionTimeoutError
)

logger = logging.getLogger(__name__)

def handle_vision_errors(func: Callable) -> Callable:
    """
    Vision-level error handler that inherits from common error handler.
    Parent handler for document and other vision subsystem errors.
    """
    @wraps(func)
    @handle_errors(error_type=VisionError)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except VisionError as e:
            logger.error(f"Vision error in {func.__name__}: {str(e)}")
            # Add error to metadata context if available
            if 'metadata_context' in kwargs:
                kwargs['metadata_context'].record_error(
                    component="vision_system",
                    message=str(e),
                    error_type=type(e).__name__,
                    is_fatal=True
                )
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise VisionProcessingError(
                message=f"Unexpected error: {str(e)}",
                processing_stage="unknown",
                component=func.__name__
            )
    return wrapper

def handle_processing_errors(func: Callable) -> Callable:
    """
    Vision-level processing error handler that inherits from common error handler.
    """
    @wraps(func)
    @handle_errors(error_type=VisionProcessingError)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except VisionProcessingError as e:
            logger.error(f"Processing error in {func.__name__}: {str(e)}")
            if 'metadata_context' in kwargs:
                kwargs['metadata_context'].record_error(
                    component=e.component or "vision_processing",
                    message=str(e),
                    error_type=type(e).__name__,
                    is_fatal=True
                )
            raise
        except Exception as e:
            logger.error(f"Unexpected processing error in {func.__name__}: {str(e)}")
            raise VisionProcessingError(
                message=f"Processing failed: {str(e)}",
                processing_stage="unknown",
                component=func.__name__
            )
    return wrapper

def processing_complete(func: Callable) -> Callable:
    """
    Vision-level processing completion handler that includes timing measurement.
    """
    @wraps(func)
    @measure_time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # Log success
            logger.info(
                f"Vision processing completed: {func.__name__}, "
                f"time: {processing_time:.2f}s"
            )
            
            # Add processing metadata
            if isinstance(result, dict):
                result["vision_processing_metadata"] = {
                    "success": True,
                    "processing_time": processing_time,
                    "function": func.__name__
                }
            
            # Update metadata context if available
            if 'metadata_context' in kwargs:
                kwargs['metadata_context'].record_processor_performance(
                    processor_name=func.__name__,
                    document_type="unknown",  # Subclasses should override this
                    success=True,
                    processing_time=processing_time
                )
                
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Vision processing failed: {func.__name__}, "
                f"time: {processing_time:.2f}s, error: {str(e)}"
            )
            
            # Update metadata context if available
            if 'metadata_context' in kwargs:
                kwargs['metadata_context'].record_processor_performance(
                    processor_name=func.__name__,
                    document_type="unknown",
                    success=False,
                    processing_time=processing_time,
                    metrics={"error": str(e)}
                )
            raise
    return wrapper

def handle_resource_limits(func: Callable) -> Callable:
    """
    Vision-level resource limit handler with retry capability.
    """
    @wraps(func)
    @retry(max_attempts=2, exceptions=(VisionResourceError,))
    def wrapper(*args, **kwargs):
        # Check initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        try:
            # Monitor CPU usage
            if process.cpu_percent(interval=0.1) > 90:
                raise VisionResourceError(
                    message="CPU usage too high",
                    resource_type="cpu"
                )
            
            # Monitor memory usage
            if psutil.virtual_memory().percent > 90:
                raise VisionResourceError(
                    message="System memory usage too high",
                    resource_type="memory"
                )
            
            result = func(*args, **kwargs)
            
            # Check memory growth
            final_memory = process.memory_info().rss
            memory_growth = final_memory - initial_memory
            if memory_growth > 1024 * 1024 * 1024:  # 1GB
                logger.warning(f"High memory growth in {func.__name__}: {memory_growth / (1024*1024):.2f}MB")
            
            return result
            
        except VisionResourceError:
            raise
        except Exception as e:
            logger.error(f"Resource error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def handle_timeout(timeout_seconds: int = 300) -> Callable:
    """
    Vision-level timeout handler with retry capability.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @retry(max_attempts=2, exceptions=(VisionTimeoutError,))
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Check if execution time exceeded timeout
                if time.time() - start_time > timeout_seconds:
                    raise VisionTimeoutError(
                        message=f"Vision operation timed out after {timeout_seconds} seconds",
                        timeout_seconds=timeout_seconds,
                        operation=func.__name__
                    )
                
                return result
                
            except VisionTimeoutError:
                raise
            except Exception as e:
                if time.time() - start_time > timeout_seconds:
                    raise VisionTimeoutError(
                        message=f"Vision operation timed out after {timeout_seconds} seconds",
                        timeout_seconds=timeout_seconds,
                        operation=func.__name__
                    )
                raise
        return wrapper
    return decorator 