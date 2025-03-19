"""
DocumentFormatAdapter: Adapters for different document formats.
Provides specialized adapters for handling specific document types.
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any, ClassVar
import logging
from abc import abstractmethod
from datetime import datetime
from functools import wraps

from models_app.vision.document.adapters.document_base_adapter import DocumentBaseAdapter
from models_app.vision.document.utils.core.next_layer_interface import (
    ProcessingEventType
)
from models_app.vision.document.utils.core.processing_metadata_context import (
    ProcessingMetadataContext,
    PipelinePhase
)
from models_app.vision.document.utils.error_handling.handlers import (
    handle_document_errors,
    handle_adapter_errors
)
from models_app.vision.document.utils.error_handling.errors import (
    DocumentProcessingError,
    DocumentValidationError
)
from models_app.vision.document.utils.optimization import ProcessingOptimizer

logger = logging.getLogger(__name__)

def with_transaction(func):
    """Decorator to handle metadata context transactions."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        metadata_context = kwargs.get('metadata_context', None)
        if metadata_context:
            metadata_context.start_transaction()
        try:
            result = func(self, *args, **kwargs)
            if metadata_context:
                metadata_context.commit_transaction()
            return result
        except Exception as e:
            if metadata_context:
                metadata_context.rollback_transaction()
            raise
    return wrapper

def with_phase_tracking(phase: PipelinePhase):
    """Decorator to track pipeline phases."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            metadata_context = kwargs.get('metadata_context', None)
            if metadata_context:
                metadata_context.start_phase(phase)
            try:
                result = func(self, *args, **kwargs)
                if metadata_context:
                    metadata_context.end_phase(phase)
                return result
            except Exception as e:
                if metadata_context:
                    metadata_context.record_phase_error(phase)
                raise
        return wrapper
    return decorator

class DocumentFormatAdapter(DocumentBaseAdapter):
    """
    Base class for document format adapters.
    Adapters are specific implementations for different document formats.
    """
    
    # Add supported formats as class attributes
    SUPPORTED_FORMATS: ClassVar[List[str]] = []  # List of supported file extensions
    SUPPORTED_MIME_TYPES: ClassVar[List[str]] = []  # List of supported MIME types
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the document format adapter.
        
        Args:
            config: Configuration dictionary with adapter-specific settings.
        """
        super().__init__(config)
        self.format_name = "Generic Format"
    
    def _initialize_adapter_components(self) -> None:
        """
        Initialize adapter components.
        Overrides the base method to add format-specific initialization.
        """
        # Call parent implementation first
        super()._initialize_adapter_components()
        
        # Format adapter specific initializations
        self._initialize_optimizer()
        
    def _initialize_optimizer(self) -> None:
        """Initialize processing optimizer with format-specific settings."""
        optimizer_config = self.config.get("optimizer", {})
        optimizer_config.update({
            "format": self.format_name,
            "capabilities": self.get_capabilities()
        })
        self.optimizer = ProcessingOptimizer(optimizer_config)
    
    def get_adapter_type(self) -> str:
        """Return the adapter type for metadata tracking."""
        return self.format_name
    
    def optimize_processing(self, document_path: str, content: Any,
                          metadata_context: Optional[ProcessingMetadataContext] = None) -> Any:
        """
        Apply format-specific optimizations to the processing pipeline.
        
        Args:
            document_path: Path to the document
            content: Content to optimize
            metadata_context: Optional metadata context
            
        Returns:
            Optimized content
        """
        if not hasattr(self, 'optimizer') or not self.optimizer.should_optimize(content):
            return content
            
        try:
            if metadata_context:
                metadata_context.start_timing("optimization")
            
            optimized = self.optimizer.optimize(content)
            
            if metadata_context:
                self.next_layer.emit_simple_event(
                    ProcessingEventType.OPTIMIZATION_APPLIED,
                    document_path,
                    {
                        "optimization_type": self.optimizer.last_optimization,
                        "improvement": self.optimizer.last_improvement
                    }
                )
            
            return optimized
            
        finally:
            if metadata_context:
                metadata_context.end_timing("optimization")
    
    @handle_document_errors
    @with_transaction
    @with_phase_tracking(PipelinePhase.ANALYSIS)
    @abstractmethod
    def extract_structure(self, document_path: str, metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Extract the document's structure (headings, paragraphs, tables, etc.).
        
        Args:
            document_path: Path to the document file.
            metadata_context: Optional metadata context for tracking decisions.
            
        Returns:
            Dict: Structured representation of the document.
        """
        pass
    
    @handle_document_errors
    @with_transaction
    @with_phase_tracking(PipelinePhase.PROCESSING)
    def extract_text(self, document_path: str, metadata_context: Optional[ProcessingMetadataContext] = None) -> str:
        """
        Extract plain text from the document.
        
        Args:
            document_path: Path to the document file.
            metadata_context: Optional metadata context for tracking decisions.
            
        Returns:
            str: Plain text content of the document.
        """
        if metadata_context:
            metadata_context.start_timing("text_extraction")
            
        try:
            # Default implementation could use the structure to extract text
            structure = self.extract_structure(document_path, metadata_context)
            text = self._structure_to_text(structure)
            
            if metadata_context:
                metadata_context.end_timing("text_extraction")
                metadata_context.record_analysis_result(
                    "text_extraction",
                    {"text_length": len(text)},
                    component=self.__class__.__name__
                )
                
                # Emit text extraction event
                self.next_layer.emit_simple_event(
                    ProcessingEventType.PROCESSING_COMPLETE,
                    document_path,
                    {
                        "operation": "text_extraction",
                        "text_length": len(text)
                    }
                )
            
            return text
            
        except Exception as e:
            if metadata_context:
                metadata_context.end_timing("text_extraction")
                metadata_context.record_error(
                    component=self.__class__.__name__,
                    message=f"Text extraction failed: {str(e)}",
                    error_type=type(e).__name__
                )
                
                # Emit error event
                self.next_layer.emit_simple_event(
                    ProcessingEventType.ERROR_OCCURRED,
                    document_path,
                    {
                        "operation": "text_extraction",
                        "error": str(e)
                    }
                )
            raise
    
    def _structure_to_text(self, structure: Dict[str, Any]) -> str:
        """
        Convert a structured document representation to plain text.
        
        Args:
            structure: Structured representation of the document.
            
        Returns:
            str: Plain text version of the document.
        """
        # This is a simple implementation; subclasses may override with format-specific logic
        if isinstance(structure, dict):
            if "text" in structure:
                return structure["text"]
            
            parts = []
            for key, value in structure.items():
                if key == "elements" and isinstance(value, list):
                    for element in value:
                        parts.append(self._structure_to_text(element))
                elif isinstance(value, (dict, list)):
                    parts.append(self._structure_to_text(value))
            
            return "\n".join([p for p in parts if p])
        
        elif isinstance(structure, list):
            return "\n".join([self._structure_to_text(item) for item in structure if item])
        
        elif isinstance(structure, str):
            return structure
        
        return ""
    
    @handle_document_errors
    @with_transaction
    @with_phase_tracking(PipelinePhase.PREPROCESSING)
    def get_document_preview(self, document_path: str, page: int = 0, metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Generate a preview for the document.
        
        Args:
            document_path: Path to the document file.
            page: Page number to preview (0-indexed).
            metadata_context: Optional metadata context for tracking decisions.
            
        Returns:
            Dict: Preview data, which may include text, images, and metadata.
        """
        if metadata_context:
            metadata_context.start_timing("preview_generation")
            child_context = metadata_context.create_child_context()
        
        try:
            metadata = self.extract_metadata(document_path, metadata_context)
            text_preview = self.extract_text(document_path, child_context if metadata_context else None)
            
            # Truncate the preview if it's too long
            max_preview_length = self.config.get("max_preview_length", 1000)
            if len(text_preview) > max_preview_length:
                text_preview = text_preview[:max_preview_length] + "..."
            
            preview_result = {
                "metadata": metadata,
                "text_preview": text_preview,
                "format": self.format_name,
                "page": page,
                "has_more_pages": False  # Subclasses will override if multi-page
            }
            
            # Emit preview generation event
            self.next_layer.emit_simple_event(
                ProcessingEventType.PROCESSING_COMPLETE,
                document_path,
                {
                    "operation": "preview_generation",
                    "preview_result": preview_result
                }
            )
            
            if metadata_context:
                metadata_context.end_timing("preview_generation")
                metadata_context.merge_child_context(child_context)
            
            return preview_result
            
        except Exception as e:
            if metadata_context:
                metadata_context.end_timing("preview_generation")
                if child_context:
                    metadata_context.merge_child_context(child_context)
                metadata_context.record_error(
                    component=self.__class__.__name__,
                    message=f"Preview generation failed: {str(e)}",
                    error_type=type(e).__name__
                )
                
                # Emit error event
                self.next_layer.emit_simple_event(
                    ProcessingEventType.ERROR_OCCURRED,
                    document_path,
                    {
                        "operation": "preview_generation",
                        "error": str(e)
                    }
                )
            
            logger.error(f"Error generating preview for {document_path}: {str(e)}")
            return {
                "error": str(e),
                "format": self.format_name
            }
    
    def validate_format(self, document_path: str) -> bool:
        """
        Validate if the document format is supported by this adapter.
        
        Args:
            document_path: Path to the document file.
            
        Returns:
            bool: True if format is supported, False otherwise.
            
        Raises:
            DocumentValidationError: If format validation fails.
        """
        if not os.path.exists(document_path):
            raise DocumentValidationError(
                f"Document not found: {document_path}",
                document_path=document_path,
                validator=self.__class__.__name__
            )
            
        file_ext = os.path.splitext(document_path)[1].lower()
        if not file_ext:
            raise DocumentValidationError(
                "Document has no file extension",
                document_path=document_path,
                validator=self.__class__.__name__
            )
            
        # Check file extension
        if self.SUPPORTED_FORMATS and file_ext not in self.SUPPORTED_FORMATS:
            raise DocumentValidationError(
                f"Unsupported file extension {file_ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}",
                document_path=document_path,
                validator=self.__class__.__name__
            )
            
        # Check MIME type if available
        try:
            import magic
            mime_type = magic.from_file(document_path, mime=True)
            if self.SUPPORTED_MIME_TYPES and mime_type not in self.SUPPORTED_MIME_TYPES:
                raise DocumentValidationError(
                    f"Unsupported MIME type {mime_type}. "
                    f"Supported types: {', '.join(self.SUPPORTED_MIME_TYPES)}",
                    document_path=document_path,
                    validator=self.__class__.__name__
                )
        except ImportError:
            logger.warning("python-magic not installed. MIME type validation skipped.")
            
        return True
        
    def _cleanup_resources(self) -> None:
        """Clean up resources used by the adapter."""
        try:
            # Clean up optimizer resources
            if hasattr(self, 'optimizer'):
                self.optimizer.cleanup()
                
            # Call parent cleanup
            super()._cleanup_resources()
            
        except Exception as e:
            logger.error(f"Error during cleanup in {self.__class__.__name__}: {str(e)}")
            
    def get_capabilities(self) -> Dict[str, float]:
        """
        Get format-specific capabilities with confidence scores.
        
        Returns:
            Dict[str, float]: Mapping of capability names to confidence scores
        """
        # Start with base capabilities from DocumentBaseAdapter
        capabilities = super().get_capabilities()
        
        # Add format-specific capabilities
        format_capabilities = {
            "text_extraction": 1.0,
            "structure_preservation": 1.0,
            "metadata_extraction": 1.0
        }
        
        # Combine capabilities, preferring format-specific scores
        capabilities.update(format_capabilities)
        
        return capabilities
    
    def _process_file(self, file_path: str, options: Dict[str, Any], 
                    metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process a file and extract information.
        
        Args:
            file_path: Path to the file
            options: Processing options
            metadata_context: Context for tracking metadata during processing
            
        Returns:
            Dictionary with extracted information
        """
        # First, validate the format
        self.validate_format(file_path)
        
        # Extract structure
        structure = self.extract_structure(file_path, metadata_context)
        
        # Extract text from structure
        text = self._structure_to_text(structure)
        
        # Combine results
        result = {
            "text": text,
            "structure": structure,
            "format": self.format_name,
            "confidence": 0.8  # Default confidence, should be overridden by subclasses
        }
        
        # Add metadata
        result["metadata"] = self.extract_metadata(file_path, metadata_context)
        
        return result 