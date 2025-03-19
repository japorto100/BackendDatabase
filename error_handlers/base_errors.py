"""
Base error classes for the entire application.

This module provides the foundational error classes that all
application-specific errors should inherit from.
"""

from typing import Optional, Dict, Any
from datetime import datetime

class BaseApplicationError(Exception):
    """Base class for all application errors."""
    
    def __init__(self, 
                 message: str, 
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "base_error"
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        self.module = "base"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "module": self.module,
            "timestamp": self.timestamp,
            "details": self.details
        }

class ResourceError(BaseApplicationError):
    """Base class for resource-related errors."""
    def __init__(self, 
                 message: str,
                 resource_type: str,
                 resource_id: Optional[str] = None,
                 **kwargs):
        super().__init__(message, error_code="resource_error", **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.details.update({
            "resource_type": resource_type,
            "resource_id": resource_id
        })

class ValidationError(BaseApplicationError):
    """Base class for validation errors."""
    def __init__(self,
                 message: str,
                 validation_type: str,
                 invalid_value: Any = None,
                 **kwargs):
        super().__init__(message, error_code="validation_error", **kwargs)
        self.validation_type = validation_type
        self.invalid_value = invalid_value
        self.details.update({
            "validation_type": validation_type,
            "invalid_value": str(invalid_value)
        })

class ProcessingError(BaseApplicationError):
    """Base class for processing errors."""
    def __init__(self,
                 message: str,
                 processing_stage: str,
                 component: Optional[str] = None,
                 **kwargs):
        super().__init__(message, error_code="processing_error", **kwargs)
        self.processing_stage = processing_stage
        self.component = component
        self.details.update({
            "processing_stage": processing_stage,
            "component": component
        })

class TimeoutError(BaseApplicationError):
    """Base class for timeout errors."""
    def __init__(self,
                 message: str,
                 timeout_seconds: float,
                 operation: str,
                 **kwargs):
        super().__init__(message, error_code="timeout_error", **kwargs)
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self.details.update({
            "timeout_seconds": timeout_seconds,
            "operation": operation
        }) 