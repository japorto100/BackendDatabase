"""
Common error handling utilities and decorators.

This module provides common error handling functionality that can be
used across different parts of the application.
"""

import logging
import functools
import traceback
from typing import Dict, Any, Optional, Callable, Type, Union
from datetime import datetime
import time

from error_handlers.base_errors import BaseApplicationError

logger = logging.getLogger(__name__)

def handle_errors(error_type: Type[BaseApplicationError] = BaseApplicationError):
    """
    Generic error handling decorator.
    
    Args:
        error_type: The type of error to catch and wrap other errors in
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                # Already the correct error type, just log and return
                logger.error(f"{error_type.__name__} in {func.__name__}: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
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
                # Wrap in specified error type
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                error = error_type(
                    message=str(e),
                    details={
                        "original_error": type(e).__name__,
                        "function": func.__name__
                    }
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
    return decorator

def measure_time(func: Callable) -> Callable:
    """Decorator for measuring execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Add timing to result if it's a dict
            if isinstance(result, dict):
                result['execution_time'] = duration
            
            logger.info(f"{func.__name__} execution time: {duration:.2f} seconds")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.2f} seconds: {str(e)}")
            raise
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0, 
         backoff_factor: float = 2.0, 
         exceptions: tuple = (Exception,)) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"Final retry attempt failed for {func.__name__}: {str(e)}")
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {current_delay:.1f} seconds..."
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    attempt += 1
        return wrapper
    return decorator 