"""
Unified error handling system for document processing.
Provides a consistent error hierarchy for all components.
"""

from typing import Optional, Dict, Any
import traceback

class VisionError(Exception):
    """Base class for all vision-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                details: Optional[Dict[str, Any]] = None):
        """
        Initialize the error.
        
        Args:
            message: Error message
            error_code: Error code for categorization
            details: Optional details about the error
        """
        super().__init__(message)
        self.error_code = error_code or "VISION_ERROR"
        self.details = details or {}
        self.traceback = traceback.format_exc()

class DocumentError(VisionError):
    """Base class for all document-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                details: Optional[Dict[str, Any]] = None, 
                document_path: Optional[str] = None):
        """
        Initialize the document error.
        
        Args:
            message: Error message
            error_code: Error code for categorization
            details: Optional details about the error
            document_path: Path to the document causing the error
        """
        super().__init__(message, error_code or "DOCUMENT_ERROR", details)
        self.document_path = document_path
        if document_path:
            self.details["document_path"] = document_path

class DocumentProcessingError(DocumentError):
    """Error occurred during document processing."""
    
    def __init__(self, message: str, document_path: Optional[str] = None, 
                component: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the processing error.
        
        Args:
            message: Error message
            document_path: Path to the document causing the error
            component: Name of the component where the error occurred
            details: Optional details about the error
        """
        error_details = details or {}
        if component:
            error_details["component"] = component
        super().__init__(message, "DOCUMENT_PROCESSING_ERROR", error_details, document_path)
        self.component = component

class DocumentValidationError(DocumentError):
    """Error in document validation before processing."""
    
    def __init__(self, message: str, document_path: Optional[str] = None, 
                validator: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the validation error.
        
        Args:
            message: Error message
            document_path: Path to the document causing the error
            validator: Name of the validator where the error occurred
            details: Optional details about the error
        """
        error_details = details or {}
        if validator:
            error_details["validator"] = validator
        super().__init__(message, "DOCUMENT_VALIDATION_ERROR", error_details, document_path)
        self.validator = validator

class DocumentResourceError(DocumentError):
    """Error related to document resource availability or limits."""
    
    def __init__(self, message: str, resource_type: str, 
                current_value: Optional[float] = None, 
                threshold: Optional[float] = None,
                document_path: Optional[str] = None):
        """
        Initialize the resource error.
        
        Args:
            message: Error message
            resource_type: Type of resource causing the error (memory, CPU, etc.)
            current_value: Current resource usage value
            threshold: Resource threshold that was exceeded
            document_path: Path to the document related to the error
        """
        error_details = {
            "resource_type": resource_type
        }
        if current_value is not None:
            error_details["current_value"] = current_value
        if threshold is not None:
            error_details["threshold"] = threshold
        
        super().__init__(message, "DOCUMENT_RESOURCE_ERROR", error_details, document_path)
        self.resource_type = resource_type
        self.current_value = current_value
        self.threshold = threshold

class DocumentTimeoutError(DocumentError):
    """Error when document processing exceeds time limits."""
    
    def __init__(self, message: str, timeout_seconds: float, 
                document_path: Optional[str] = None, 
                operation: Optional[str] = None):
        """
        Initialize the timeout error.
        
        Args:
            message: Error message
            timeout_seconds: Timeout in seconds that was exceeded
            document_path: Path to the document causing the error
            operation: Operation that timed out
        """
        error_details = {
            "timeout_seconds": timeout_seconds
        }
        if operation:
            error_details["operation"] = operation
        
        super().__init__(message, "DOCUMENT_TIMEOUT_ERROR", error_details, document_path)
        self.timeout_seconds = timeout_seconds
        self.operation = operation

class AdapterError(DocumentError):
    """Error in document adapter functionality."""
    
    def __init__(self, message: str, adapter_name: str, 
                document_path: Optional[str] = None, 
                operation: Optional[str] = None,
                details: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter error.
        
        Args:
            message: Error message
            adapter_name: Name of the adapter where the error occurred
            document_path: Path to the document causing the error
            operation: Operation that failed
            details: Optional details about the error
        """
        error_details = details or {}
        error_details["adapter_name"] = adapter_name
        if operation:
            error_details["operation"] = operation
        
        super().__init__(message, "ADAPTER_ERROR", error_details, document_path)
        self.adapter_name = adapter_name
        self.operation = operation

class AnalysisError(DocumentError):
    """Error in document analysis or classification."""
    
    def __init__(self, message: str, analyzer_name: str, 
                document_path: Optional[str] = None, 
                analysis_type: Optional[str] = None,
                details: Optional[Dict[str, Any]] = None):
        """
        Initialize the analysis error.
        
        Args:
            message: Error message
            analyzer_name: Name of the analyzer where the error occurred
            document_path: Path to the document causing the error
            analysis_type: Type of analysis that failed
            details: Optional details about the error
        """
        error_details = details or {}
        error_details["analyzer_name"] = analyzer_name
        if analysis_type:
            error_details["analysis_type"] = analysis_type
        
        super().__init__(message, "ANALYSIS_ERROR", error_details, document_path)
        self.analyzer_name = analyzer_name
        self.analysis_type = analysis_type

class KnowledgeGraphError(VisionError):
    """Error in knowledge graph integration."""
    
    def __init__(self, message: str, entity_type: Optional[str] = None, 
                document_id: Optional[str] = None,
                details: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge graph error.
        
        Args:
            message: Error message
            entity_type: Type of entity related to the error
            document_id: ID of the document related to the error
            details: Optional details about the error
        """
        error_details = details or {}
        if entity_type:
            error_details["entity_type"] = entity_type
        if document_id:
            error_details["document_id"] = document_id
        
        super().__init__(message, "KNOWLEDGE_GRAPH_ERROR", error_details)
        self.entity_type = entity_type
        self.document_id = document_id

class ProcessorError(VisionError):
    """Error in content processors."""
    
    def __init__(self, message: str, processor_name: str, 
                content_type: Optional[str] = None,
                details: Optional[Dict[str, Any]] = None):
        """
        Initialize the processor error.
        
        Args:
            message: Error message
            processor_name: Name of the processor where the error occurred
            content_type: Type of content being processed
            details: Optional details about the error
        """
        error_details = details or {}
        error_details["processor_name"] = processor_name
        if content_type:
            error_details["content_type"] = content_type
        
        super().__init__(message, "PROCESSOR_ERROR", error_details)
        self.processor_name = processor_name
        self.content_type = content_type

class ConfigurationError(VisionError):
    """Error in system configuration."""
    
    def __init__(self, message: str, component: str, 
                config_key: Optional[str] = None,
                details: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration error.
        
        Args:
            message: Error message
            component: Name of the component with configuration issues
            config_key: Specific configuration key that caused the error
            details: Optional details about the error
        """
        error_details = details or {}
        error_details["component"] = component
        if config_key:
            error_details["config_key"] = config_key
        
        super().__init__(message, "CONFIGURATION_ERROR", error_details)
        self.component = component
        self.config_key = config_key

# Specific Document Processing Errors
class OCRError(DocumentProcessingError):
    """Error during OCR operations."""
    
    def __init__(self, message: str, ocr_engine: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, "ocr", document_path, "ocr_processor", **kwargs)
        self.ocr_engine = ocr_engine
        self.error_code = "ocr_error"
        self.details["ocr_engine"] = ocr_engine

class DocumentAnalysisError(DocumentProcessingError):
    """Error during document analysis."""
    
    def __init__(self, message: str, analysis_type: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, "analysis", document_path, "document_analyzer", **kwargs)
        self.analysis_type = analysis_type
        self.error_code = "analysis_error"
        self.details["analysis_type"] = analysis_type

class ChunkingError(DocumentProcessingError):
    """Error during document chunking."""
    
    def __init__(self, message: str, chunk_size: Optional[int] = None,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, "chunking", document_path, "chunking_service", **kwargs)
        self.chunk_size = chunk_size
        self.error_code = "chunking_error"
        if chunk_size:
            self.details["chunk_size"] = chunk_size

class DocumentCapabilityError(DocumentError):
    """Error related to document processing capabilities."""
    
    def __init__(self, message: str, capability: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="capability_error", document_path=document_path, **kwargs)
        self.capability = capability
        self.details["capability"] = capability

class DocumentMetadataError(DocumentError):
    """Error related to document metadata operations."""
    
    def __init__(self, message: str, metadata_type: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="metadata_error", document_path=document_path, **kwargs)
        self.metadata_type = metadata_type
        self.details["metadata_type"] = metadata_type

class DocumentPipelineError(DocumentProcessingError):
    """Error in document processing pipeline."""
    
    def __init__(self, message: str, pipeline_stage: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, pipeline_stage, document_path, "pipeline_manager", **kwargs)
        self.pipeline_stage = pipeline_stage
        self.error_code = "pipeline_error"
        self.details["pipeline_stage"] = pipeline_stage

class DocumentCacheError(DocumentError):
    """Error related to document processing cache operations."""
    
    def __init__(self, message: str, cache_operation: str,
                 document_path: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="cache_error", document_path=document_path, **kwargs)
        self.cache_operation = cache_operation
        self.details["cache_operation"] = cache_operation 