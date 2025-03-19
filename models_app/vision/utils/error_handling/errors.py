"""
Vision-specific error types and base classes.

This module defines error types specific to vision processing operations,
inheriting from application-wide base errors but adding vision-specific
context and handling.
"""

from typing import Optional, Dict, Any
from error_handlers.base_errors import (
    BaseApplicationError,
    ProcessingError,
    ValidationError,
    ResourceError,
    TimeoutError
)

class VisionError(BaseApplicationError):
    """Base class for all vision-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code=error_code or "vision_error", details=details)
        self.module = "vision"

class VisionProcessingError(ProcessingError):
    """Base class for vision processing errors."""
    
    def __init__(self, message: str, processing_stage: str,
                 component: Optional[str] = None, **kwargs):
        super().__init__(message, processing_stage, component, **kwargs)
        self.module = "vision"
        self.error_code = "vision_processing_error"

class VisionResourceError(ResourceError):
    """Vision-specific resource errors."""
    
    def __init__(self, message: str, resource_type: str,
                 resource_id: Optional[str] = None, **kwargs):
        super().__init__(message, resource_type, resource_id, **kwargs)
        self.module = "vision"
        self.error_code = "vision_resource_error"

class VisionValidationError(ValidationError):
    """Vision-specific validation errors."""
    
    def __init__(self, message: str, validation_type: str,
                 invalid_value: Any = None, **kwargs):
        super().__init__(message, validation_type, invalid_value, **kwargs)
        self.module = "vision"
        self.error_code = "vision_validation_error"

class VisionTimeoutError(TimeoutError):
    """Vision-specific timeout errors."""
    
    def __init__(self, message: str, timeout_seconds: float,
                 operation: str, **kwargs):
        super().__init__(message, timeout_seconds, operation, **kwargs)
        self.module = "vision"
        self.error_code = "vision_timeout_error"

# Document Processing Specific Errors
class DocumentProcessingError(VisionProcessingError):
    """Error during document processing operations."""
    
    def __init__(self, message: str, processing_stage: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, processing_stage, "document_processor", **kwargs)
        self.document_path = document_path
        self.error_code = "document_processing_error"
        self.details["document_path"] = document_path

class OCRError(DocumentProcessingError):
    """Error during OCR operations."""
    
    def __init__(self, message: str, ocr_engine: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, "ocr", document_path, **kwargs)
        self.ocr_engine = ocr_engine
        self.error_code = "ocr_error"
        self.details["ocr_engine"] = ocr_engine

class DocumentAnalysisError(DocumentProcessingError):
    """Error during document analysis."""
    
    def __init__(self, message: str, analysis_type: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, "analysis", document_path, **kwargs)
        self.analysis_type = analysis_type
        self.error_code = "analysis_error"
        self.details["analysis_type"] = analysis_type

class ChunkingError(DocumentProcessingError):
    """Error during document chunking."""
    
    def __init__(self, message: str, chunk_size: Optional[int] = None,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, "chunking", document_path, **kwargs)
        self.chunk_size = chunk_size
        self.error_code = "chunking_error"
        self.details["chunk_size"] = chunk_size

class AdapterError(DocumentProcessingError):
    """Error in document adapter."""
    
    def __init__(self, message: str, adapter_name: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, "adapter", document_path, **kwargs)
        self.adapter_name = adapter_name
        self.error_code = "adapter_error"
        self.details["adapter_name"] = adapter_name

class KnowledgeGraphError(DocumentProcessingError):
    """Error in knowledge graph operations."""
    
    def __init__(self, message: str, operation: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, "knowledge_graph", document_path, **kwargs)
        self.operation = operation
        self.error_code = "kg_error"
        self.details["operation"] = operation 