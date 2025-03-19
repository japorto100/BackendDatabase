"""
Unified error handlers for document processing.
Provides decorators and functions for consistent error handling.
"""

import logging
import time
import functools
import traceback
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast
from datetime import datetime

from models_app.vision.document.utils.error_handling.errors import (
    VisionError,
    DocumentError,
    DocumentProcessingError,
    DocumentTimeoutError,
    DocumentResourceError,
    DocumentValidationError,
    AdapterError,
    ConfigurationError
)
from models_app.vision.document.utils.core.next_layer_interface import (
    NextLayerInterface,
    ProcessingEventType
)

logger = logging.getLogger(__name__)

# Type variable for function annotations
F = TypeVar('F', bound=Callable[..., Any])

def handle_vision_errors(func: F) -> F:
    """
    Decorator to handle all vision-related errors.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except VisionError as e:
            # Already a vision error, so just log and re-raise
            logger.error(f"Vision error in {func.__name__}: {str(e)}")
            _emit_error_event(e, func.__name__)
            raise
        except Exception as e:
            # Convert to VisionError
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            vision_error = VisionError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                details={"original_error": str(e), "traceback": traceback.format_exc()}
            )
            _emit_error_event(vision_error, func.__name__)
            raise vision_error from e
    
    return cast(F, wrapper)

def handle_document_errors(func: F) -> F:
    """
    Decorator to handle document-related errors.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DocumentError as e:
            # Already a document error, so just log and re-raise
            logger.error(f"Document error in {func.__name__}: {str(e)}")
            _emit_error_event(e, func.__name__)
            raise
        except VisionError as e:
            # Convert to DocumentError
            logger.error(f"Vision error in document operation {func.__name__}: {str(e)}")
            doc_error = DocumentError(
                str(e),
                details=e.details
            )
            _emit_error_event(doc_error, func.__name__)
            raise doc_error from e
        except Exception as e:
            # Convert to DocumentProcessingError
            logger.error(f"Unexpected error in document operation {func.__name__}: {str(e)}")
            processing_error = DocumentProcessingError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                component=func.__module__,
                details={"original_error": str(e), "traceback": traceback.format_exc()}
            )
            _emit_error_event(processing_error, func.__name__)
            raise processing_error from e
    
    return cast(F, wrapper)

def handle_adapter_errors(adapter_name: str) -> Callable[[F], F]:
    """
    Decorator to handle adapter-specific errors.
    
    Args:
        adapter_name: Name of the adapter
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AdapterError as e:
                # Already an adapter error, so just log and re-raise
                logger.error(f"Adapter error in {adapter_name}.{func.__name__}: {str(e)}")
                _emit_error_event(e, f"{adapter_name}.{func.__name__}")
                raise
            except DocumentError as e:
                # Convert to AdapterError
                logger.error(f"Document error in adapter {adapter_name}.{func.__name__}: {str(e)}")
                adapter_error = AdapterError(
                    str(e),
                    adapter_name=adapter_name,
                    operation=func.__name__,
                    details=e.details
                )
                _emit_error_event(adapter_error, f"{adapter_name}.{func.__name__}")
                raise adapter_error from e
            except Exception as e:
                # Convert to AdapterError
                logger.error(f"Unexpected error in adapter {adapter_name}.{func.__name__}: {str(e)}")
                adapter_error = AdapterError(
                    f"Error in adapter {adapter_name}: {str(e)}",
                    adapter_name=adapter_name,
                    operation=func.__name__,
                    details={"original_error": str(e), "traceback": traceback.format_exc()}
                )
                _emit_error_event(adapter_error, f"{adapter_name}.{func.__name__}")
                raise adapter_error from e
        
        return cast(F, wrapper)
    
    return decorator

def handle_timeout(timeout_seconds: float) -> Callable[[F], F]:
    """
    Decorator to handle function timeout.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract document_path from args or kwargs if available
            document_path = None
            if args and hasattr(args[0], '__dict__') and 'document_path' in args[0].__dict__:
                document_path = args[0].document_path
            elif 'document_path' in kwargs:
                document_path = kwargs['document_path']
            
            try:
                result = func(*args, **kwargs)
                
                # Check if execution time exceeded timeout
                execution_time = time.time() - start_time
                if execution_time > timeout_seconds:
                    logger.warning(
                        f"Function {func.__name__} completed but exceeded timeout "
                        f"({execution_time:.2f}s > {timeout_seconds:.2f}s)"
                    )
                
                return result
                
            except Exception as e:
                # Check if this is a timeout-related exception
                if isinstance(e, TimeoutError) or "timeout" in str(e).lower():
                    timeout_error = DocumentTimeoutError(
                        f"Operation timed out after {timeout_seconds} seconds",
                        timeout_seconds=timeout_seconds,
                        document_path=document_path,
                        operation=func.__name__
                    )
                    _emit_error_event(timeout_error, func.__name__)
                    raise timeout_error from e
                raise
                
        return cast(F, wrapper)
    
    return decorator

def handle_resource_limits(func: F) -> F:
    """
    Decorator to handle resource limit checks.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get document_path if available
        document_path = None
        if args and hasattr(args[0], '__dict__') and 'document_path' in args[0].__dict__:
            document_path = args[0].document_path
        elif 'document_path' in kwargs:
            document_path = kwargs['document_path']
        
        try:
            # Check resource usage before executing
            _check_resource_limits(document_path)
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Check resource usage after executing
            _check_resource_limits(document_path)
            
            return result
            
        except DocumentResourceError as e:
            # Already a resource error, so just log and re-raise
            logger.error(f"Resource error in {func.__name__}: {str(e)}")
            _emit_error_event(e, func.__name__)
            raise
        except Exception as e:
            # If not a resource error, re-raise original exception
            raise
    
    return cast(F, wrapper)

def measure_processing_time(func: F) -> F:
    """
    Decorator to measure and log function processing time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Get document_path if available
        document_path = None
        if args and hasattr(args[0], '__dict__') and 'document_path' in args[0].__dict__:
            document_path = args[0].document_path
        elif 'document_path' in kwargs:
            document_path = kwargs['document_path']
        
        try:
            # Emit start event
            _emit_phase_event(
                ProcessingEventType.PROCESSING_PHASE_START,
                document_path,
                {
                    "function": func.__name__,
                    "start_time": start_time
                }
            )
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Emit completion event
            _emit_phase_event(
                ProcessingEventType.PROCESSING_PHASE_END,
                document_path,
                {
                    "function": func.__name__,
                    "processing_time": processing_time,
                    "success": True
                }
            )
            
            logger.debug(f"Function {func.__name__} took {processing_time:.4f} seconds")
            return result
            
        except Exception as e:
            # Calculate processing time on error
            processing_time = time.time() - start_time
            
            # Emit error event
            _emit_phase_event(
                ProcessingEventType.PROCESSING_PHASE_END,
                document_path,
                {
                    "function": func.__name__,
                    "processing_time": processing_time,
                    "success": False,
                    "error": str(e)
                }
            )
            
            logger.debug(f"Function {func.__name__} failed after {processing_time:.4f} seconds")
            raise
    
    return cast(F, wrapper)

def validate_document_path(func: F) -> F:
    """
    Decorator to validate document path exists.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get document_path from args or kwargs
        document_path = None
        if 'document_path' in kwargs:
            document_path = kwargs['document_path']
        else:
            for arg in args:
                if isinstance(arg, str) and arg.endswith(('.pdf', '.docx', '.txt', '.jpg', '.png', '.tif')):
                    document_path = arg
                    break
        
        if document_path:
            import os
            if not os.path.exists(document_path):
                validation_error = DocumentValidationError(
                    f"Document path does not exist: {document_path}",
                    document_path=document_path,
                    validator=func.__name__
                )
                _emit_error_event(validation_error, func.__name__)
                raise validation_error
        
        return func(*args, **kwargs)
    
    return cast(F, wrapper)

def document_processing_complete(func: F) -> F:
    """
    Decorator to mark document processing completion.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Extract document_path and metadata_context if available
        document_path = None
        metadata_context = None
        
        if 'document_path' in kwargs:
            document_path = kwargs['document_path']
        elif args and len(args) > 0 and isinstance(args[0], str):
            document_path = args[0]
            
        if 'metadata_context' in kwargs:
            metadata_context = kwargs['metadata_context']
        elif args and len(args) > 1:
            metadata_context = args[1]
        
        # Emit processing start event
        if document_path:
            _emit_simple_event(
                ProcessingEventType.DOCUMENT_RECEIVED,
                document_path,
                {
                    "operation": func.__name__,
                    "start_time": start_time
                }
            )
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Emit processing complete event
            if document_path:
                _emit_simple_event(
                    ProcessingEventType.PROCESSING_COMPLETE,
                    document_path,
                    {
                        "operation": func.__name__,
                        "processing_time": processing_time,
                        "success": True,
                        "result_type": type(result).__name__
                    }
                )
            
            return result
            
        except Exception as e:
            # Calculate processing time on error
            processing_time = time.time() - start_time
            
            # Emit error event
            if document_path:
                _emit_simple_event(
                    ProcessingEventType.ERROR_OCCURRED,
                    document_path,
                    {
                        "operation": func.__name__,
                        "processing_time": processing_time,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
            
            raise
    
    return cast(F, wrapper)

def _check_resource_limits(document_path: Optional[str] = None) -> None:
    """
    Check if system resources are within acceptable limits.
    
    Args:
        document_path: Optional path to the document being processed
        
    Raises:
        DocumentResourceError: If resource limits are exceeded
    """
    import psutil
    
    # Get current resource usage
    process = psutil.Process()
    memory_percent = process.memory_percent()
    cpu_percent = process.cpu_percent(interval=0.1)
    
    # Check memory usage
    if memory_percent > 90:  # 90% memory usage threshold
        error = DocumentResourceError(
            f"Memory usage too high: {memory_percent:.1f}%",
            resource_type="memory",
            current_value=memory_percent,
            threshold=90,
            document_path=document_path
        )
        _emit_error_event(error, "_check_resource_limits")
        raise error
    
    # Check CPU usage
    if cpu_percent > 95:  # 95% CPU usage threshold
        error = DocumentResourceError(
            f"CPU usage too high: {cpu_percent:.1f}%",
            resource_type="cpu",
            current_value=cpu_percent,
            threshold=95,
            document_path=document_path
        )
        _emit_error_event(error, "_check_resource_limits")
        raise error

def _emit_error_event(error: VisionError, function_name: str) -> None:
    """
    Emit error event to NextLayerInterface.
    
    Args:
        error: Error that occurred
        function_name: Name of the function where the error occurred
    """
    try:
        # Get document_path if available
        document_id = "unknown"
        if isinstance(error, DocumentError) and error.document_path:
            document_id = error.document_path
        
        # Get the NextLayerInterface singleton
        next_layer = NextLayerInterface.get_instance()
        
        # Prepare event data
        event_data = {
            "error_message": str(error),
            "error_type": type(error).__name__,
            "error_code": getattr(error, "error_code", "unknown"),
            "function": function_name,
            "details": getattr(error, "details", {})
        }
        
        # Emit error event
        next_layer.emit_simple_event(
            ProcessingEventType.ERROR_OCCURRED,
            document_id,
            event_data
        )
        
    except Exception as e:
        # Don't let errors in error handling cause issues
        logger.error(f"Error emitting error event: {str(e)}")

def _emit_phase_event(
    event_type: ProcessingEventType,
    document_id: Optional[str],
    data: Dict[str, Any]
) -> None:
    """
    Emit phase event to NextLayerInterface.
    
    Args:
        event_type: Type of event to emit
        document_id: ID of the document being processed
        data: Event data
    """
    try:
        # Use a safe document ID
        safe_document_id = document_id if document_id else "unknown"
        
        # Get the NextLayerInterface singleton
        next_layer = NextLayerInterface.get_instance()
        
        # Emit event
        next_layer.emit_simple_event(event_type, safe_document_id, data)
        
    except Exception as e:
        # Don't let errors in event emission cause issues
        logger.error(f"Error emitting phase event: {str(e)}")

def _emit_simple_event(
    event_type: ProcessingEventType,
    document_id: str,
    data: Dict[str, Any]
) -> None:
    """
    Simplified method to emit events.
    
    Args:
        event_type: Type of event to emit
        document_id: ID of the document being processed
        data: Event data
    """
    try:
        # Get the NextLayerInterface singleton
        next_layer = NextLayerInterface.get_instance()
        
        # Emit event
        next_layer.emit_simple_event(event_type, document_id, data)
        
    except Exception as e:
        # Don't let errors in event emission cause issues
        logger.error(f"Error emitting simple event: {str(e)}") 