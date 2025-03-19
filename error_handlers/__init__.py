"""Base error handling functionality for localGPT-Vision"""

class LocalGPTError(Exception):
    """Base exception for all localGPT-Vision errors"""
    def __init__(self, message, error_code=None, details=None):
        self.message = message
        self.error_code = error_code or "unknown_error"
        self.details = details or {}
        super().__init__(self.message)
        
    def to_dict(self):
        """Convert error to dictionary format for API responses"""
        return {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details
        }

# Import specific error modules
from .models_app_errors import *
try:
    from .search_app_errors import *
except ImportError:
    pass  # Module might not exist yet in some environments
