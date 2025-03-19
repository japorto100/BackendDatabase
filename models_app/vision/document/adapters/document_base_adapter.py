"""
DocumentProcessor: Base class for document processing.
Provides a unified interface for handling various document types.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, Type, ClassVar
import logging
import numpy as np
from PIL import Image
import hashlib
import mimetypes
from datetime import datetime
import platform
import traceback
from pathlib import Path
import tempfile
import shutil
from functools import wraps

# Richtiger Import für IOService
from models_app.vision.document.utils.io.io_service import IOService
from models_app.vision.document.utils.core.processing_metadata_context import ProcessingMetadataContext
from models_app.vision.document.utils.core.next_layer_interface import (
    NextLayerInterface,
    ProcessingEvent,
    ProcessingEventType
)
from models_app.vision.document.utils.error_handling.errors import (
    AdapterError,
    DocumentProcessingError,
    DocumentValidationError,
    DocumentError,
    DocumentTimeoutError
)
from models_app.vision.document.utils.error_handling.handlers import (
    handle_document_errors,
    handle_adapter_errors,
    measure_processing_time,
    validate_document_path
)

logger = logging.getLogger(__name__)

class DocumentBaseAdapter(ABC):
    """
    Base class for all document processors.
    Provides methods for handling different document formats and extracting text and structure.
    """
    
    # Class-level constants
    VERSION: ClassVar[str] = "1.0.0"
    CAPABILITIES: ClassVar[Dict[str, float]] = {}
    SUPPORTED_FORMATS: ClassVar[List[str]] = []
    PRIORITY: ClassVar[int] = 50
    
    # Common registry for all adapters
    _registry = {}
    
    @classmethod
    def register_adapter(cls, adapter_type, adapter_class):
        """Register an adapter class for a specific document type."""
        cls._registry[adapter_type] = adapter_class
    
    @classmethod
    def get_adapter_class(cls, adapter_type):
        """Get the registered adapter class for a document type."""
        return cls._registry.get(adapter_type)
    
    @classmethod
    def get_registered_adapters(cls):
        """Get all registered adapters."""
        return cls._registry
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the document processor.
        
        Args:
            config: Configuration dictionary with processor-specific settings.
        """
        self.config = config or {}
        self.is_initialized = False
        self._supported_extensions = []
        self._processor_name = self.__class__.__name__
        self._temp_files = []
        
        # Use IOService instead of FormatConverter
        self.io_service = IOService(config)
        
        # Use Singleton instance of NextLayerInterface
        self.next_layer = NextLayerInterface.get_instance()
        
        # Template Method pattern for initialization
        self.initialize()
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the adapter, following the Template Method pattern.
        
        Returns:
            Dict[str, Any]: Initialization metadata
        """
        try:
            # Emit initialization event
            self.next_layer.emit_simple_event(
                ProcessingEventType.INITIALIZATION,
                f"adapter_{self._processor_name}",
                {"adapter_class": self.__class__.__name__}
            )
            
            # 1. Initialize supported extensions
            self._initialize_supported_extensions()
            
            # 2. Initialize adapter-specific components
            self._initialize_adapter_components()
            
            # 3. Configure adapter-specific settings
            self._configure_settings()
            
            # 4. Validate configuration
            self._validate_configuration()
            
            # 5. Register with Next Layer
            self._register_with_next_layer()
            
            # Mark as initialized after template method steps
            self.is_initialized = True
            
            # Return initialization metadata
            initialization_meta = {
                "success": True,
                "adapter": self._processor_name,
                "capabilities": self.get_capabilities(),
                "supported_formats": self._supported_extensions,
                "version": self.VERSION
            }
            
            # Emit completion event
            self.next_layer.emit_simple_event(
                ProcessingEventType.INITIALIZATION_COMPLETE,
                f"adapter_{self._processor_name}",
                initialization_meta
            )
            
            return initialization_meta
            
        except Exception as e:
            # Emit error event
            self.next_layer.emit_simple_event(
                ProcessingEventType.ERROR_OCCURRED,
                f"adapter_{self._processor_name}",
                {
                    "adapter_class": self.__class__.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            logger.error(f"Error during initialization of {self.__class__.__name__}: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "adapter": self._processor_name
            }
    
    def _initialize_supported_extensions(self) -> None:
        """Initialize supported file extensions from class attribute."""
        self._supported_extensions = self.SUPPORTED_FORMATS.copy()
    
    def _initialize_adapter_components(self) -> None:
        """Initialize adapter-specific components. Override in subclasses."""
        pass
    
    def _configure_settings(self) -> None:
        """
        Configure adapter-specific settings (Template Method).
        
        This method is part of the template method pattern and should be
        overridden by subclasses to configure specific settings.
        """
        # Configure standard timeouts
        self.timeout_config = self.config.get("timeouts", {})
        if not self.timeout_config:
            self.timeout_config = {
                "processing": 300,  # 5 minutes
                "extraction": 120,  # 2 minutes
                "analysis": 60      # 1 minute
            }
    
    def _validate_configuration(self) -> None:
        """
        Validate adapter configuration (Template Method).
        
        This method is part of the template method pattern and should be
        overridden by subclasses to validate their configuration.
        
        Raises:
            DocumentValidationError: If configuration is invalid
        """
        # Validate general configuration parameters
        required_params = self.config.get("required_params", [])
        for param in required_params:
            if param not in self.config:
                error_msg = f"Missing required configuration parameter: {param}"
                logger.error(f"{self._processor_name}: {error_msg}")
                raise DocumentValidationError(
                    error_msg,
                    validator=self.__class__.__name__
                )
    
    def _register_with_next_layer(self) -> None:
        """Register this adapter with the next layer interface."""
        # For now, do nothing, but subclasses could register listeners
        pass
    
    def extract_document_type(self, file_path: str) -> str:
        """
        Extract document type based on file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Document type (file extension without dot)
        """
        return self._get_document_type(file_path)
    
    def _get_document_type(self, document_path: str) -> str:
        """Get the document type based on extension."""
        _, ext = os.path.splitext(document_path)
        ext = ext.lower()
        
        if ext in ['.pdf']:
            return "pdf"
        elif ext in ['.doc', '.docx', '.odt']:
            return "document"
        elif ext in ['.xls', '.xlsx', '.ods']:
            return "spreadsheet"
        elif ext in ['.ppt', '.pptx', '.odp']:
            return "presentation"
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return "image"
        elif ext in ['.txt', '.md', '.rst']:
            return "text"
        elif ext in ['.html', '.htm', '.xml']:
            return "markup"
        elif ext in ['.csv', '.tsv']:
            return "data"
        else:
            return "unknown"
    
    def _create_temp_file(self, suffix: str = None) -> str:
        """
        Create a temporary file for processing.
        
        Args:
            suffix: Optional file suffix (extension)
            
        Returns:
            str: Path to the temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.close()
        self._temp_files.append(temp_file.name)
        return temp_file.name
    
    def can_process(self, file_path: str) -> bool:
        """
        Check if this processor can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the processor can handle this file, False otherwise
        """
        if not os.path.exists(file_path):
            return False
        
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in self._supported_extensions
    
    def get_supported_formats(self) -> List[str]:
        """
        Get the list of file formats supported by this processor.
        
        Returns:
            List[str]: List of supported file extensions.
        """
        return self._supported_extensions
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about this document processor.
        
        Returns:
            Dict: Processor information and capabilities.
        """
        return {
            "name": self._processor_name,
            "supported_formats": self.get_supported_formats(),
            "initialized": self.is_initialized,
            "capabilities": self.get_capabilities(),
            "version": self.VERSION,
            "priority": self.PRIORITY
        }
    
    def get_capabilities(self) -> Dict[str, float]:
        """
        Get the capabilities of this adapter with confidence levels.
        
        Capabilities are reported with confidence values (0.0-1.0) indicating
        how well the adapter can perform each capability. This information is
        used by the CapabilityBasedSelector to choose the most appropriate
        adapter for each document.
        
        Returns:
            Dict: Dictionary of capability names and confidence values (0.0-1.0).
        """
        # Start with class-level capabilities
        capabilities = dict(self.CAPABILITIES)
        
        # Add base capabilities if not present
        base_capabilities = {
            "text_extraction": 0.5,  # Basic capability present in all adapters
            "structure_preservation": 0.0,
            "metadata_extraction": 0.5,
            "image_extraction": 0.0,
            "table_extraction": 0.0
        }
        
        for name, value in base_capabilities.items():
            if name not in capabilities:
                capabilities[name] = value
        
        return capabilities
    
    @handle_document_errors
    @measure_processing_time
    def extract_metadata(self, document_path: str, 
                       metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Extract metadata from a document.
        
        Args:
            document_path: Path to the document file
            metadata_context: Context for tracking metadata during processing
            
        Returns:
            Dictionary with metadata
            
        Raises:
            DocumentProcessingError: If metadata extraction fails
        """
        if metadata_context:
            metadata_context.start_timing("extract_metadata")
            
        try:
            # Use IOService for metadata extraction if possible
            if hasattr(self, 'io_service'):
                metadata = self.io_service.get_file_metadata(document_path)
                metadata["processor"] = self._processor_name
                
                if metadata_context:
                    metadata_context.add_document_metadata("file_metadata", metadata)
                    metadata_context.end_timing("extract_metadata")
                    
                # Emit metadata extraction event
                self.next_layer.emit_simple_event(
                    ProcessingEventType.PROCESSING_COMPLETE,
                    document_path,
                    {
                        "operation": "metadata_extraction",
                        "metadata_keys": list(metadata.keys())
                    }
                )
                    
                return metadata
            
            # Fallback to own implementation
            file_size = os.path.getsize(document_path)
            file_name = os.path.basename(document_path)
            file_extension = os.path.splitext(file_name)[1].lower()
            mime_type, _ = mimetypes.guess_type(document_path)
            
            # File hashing for identification
            file_hash = self._calculate_file_hash(document_path)
            
            # Dates
            creation_date = self._extract_creation_date(document_path)
            modification_date = self._extract_modification_date(document_path)
            
            metadata = {
                "filename": file_name,
                "extension": file_extension,
                "mime_type": mime_type,
                "file_size": file_size,
                "file_hash": file_hash,
                "creation_date": creation_date,
                "modification_date": modification_date,
                "processor": self._processor_name
            }
            
            if metadata_context:
                metadata_context.add_document_metadata("file_metadata", metadata)
                metadata_context.end_timing("extract_metadata")
                
            # Emit metadata extraction event
            self.next_layer.emit_simple_event(
                ProcessingEventType.PROCESSING_COMPLETE,
                document_path,
                {
                    "operation": "metadata_extraction",
                    "metadata_keys": list(metadata.keys())
                }
            )
                
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            if metadata_context:
                metadata_context.record_error(
                    component=self._processor_name,
                    message=f"Error extracting metadata: {str(e)}",
                    error_type=type(e).__name__
                )
                metadata_context.end_timing("extract_metadata")
                
            # Emit error event
            self.next_layer.emit_simple_event(
                ProcessingEventType.ERROR_OCCURRED,
                document_path,
                {
                    "operation": "metadata_extraction",
                    "error": str(e)
                }
            )
                
            raise DocumentProcessingError(f"Error extracting metadata: {str(e)}")

    def _calculate_file_hash(self, file_path: str, block_size: int = 65536) -> str:
        """Calculate MD5 hash of a file."""
        try:
            md5 = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(block_size), b''):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating file hash: {str(e)}")
            return "hash_error"
    
    def _extract_creation_date(self, file_path: str) -> str:
        """Extract file creation date"""
        try:
            stat = os.stat(file_path)
            # Use ctime on Windows, creation time on other platforms
            if platform.system() == 'Windows':
                return datetime.fromtimestamp(stat.st_ctime).isoformat()
            else:
                # For Unix-like systems
                return datetime.fromtimestamp(stat.st_mtime).isoformat()
        except Exception:
            return datetime.now().isoformat()
        
    def _extract_modification_date(self, file_path: str) -> str:
        """Extract file modification date"""
        try:
            return datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        except Exception:
            return datetime.now().isoformat()

    @measure_processing_time
    @handle_document_errors
    @validate_document_path
    def process_document(self, document_path: str, options: Dict[str, Any] = None,
                        metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process a document and extract information.
        
        Args:
            document_path: Path to the document file
            options: Processing options
            metadata_context: Context for tracking metadata during processing
            
        Returns:
            Dictionary with extracted information
            
        Raises:
            DocumentProcessingError: If document processing fails
            DocumentValidationError: If document format is not supported
        """
        options = options or {}
        
        # Create a new metadata context if none is provided
        if metadata_context is None:
            metadata_context = ProcessingMetadataContext(document_path)
            
        metadata_context.start_timing(f"{self._processor_name}_processing")
        
        # Event for document reception
        self.next_layer.emit_simple_event(
            ProcessingEventType.DOCUMENT_RECEIVED,
            document_path,
            {
                "processor": self._processor_name,
                "options": options
            }
        )
        
        try:
            # Check if the file can be processed
            if not self.can_process(document_path):
                error_msg = f"Document format not supported by {self._processor_name}"
                metadata_context.record_error(
                    component=self._processor_name,
                    message=error_msg,
                    error_type="UnsupportedFormat",
                    is_fatal=True
                )
                metadata_context.end_timing(f"{self._processor_name}_processing")
                
                # Event for error
                self.next_layer.emit_simple_event(
                    ProcessingEventType.ERROR_OCCURRED,
                    document_path,
                    {
                        "processor": self._processor_name,
                        "error": error_msg,
                        "error_type": "UnsupportedFormat"
                    }
                )
                
                raise DocumentValidationError(error_msg, document_path=document_path)
            
            # Initialize if not already done
            if not self.is_initialized:
                metadata_context.start_timing(f"{self._processor_name}_initialization")
                init_result = self.initialize()
                metadata_context.end_timing(f"{self._processor_name}_initialization")
                
                if not init_result.get("success", False):
                    error_msg = f"Failed to initialize {self._processor_name}: {init_result.get('error', 'Unknown error')}"
                    metadata_context.record_error(
                        component=self._processor_name,
                        message=error_msg,
                        error_type="InitializationError",
                        is_fatal=True
                    )
                    metadata_context.end_timing(f"{self._processor_name}_processing")
                    
                    # Event for error
                    self.next_layer.emit_simple_event(
                        ProcessingEventType.ERROR_OCCURRED,
                        document_path,
                        {
                            "processor": self._processor_name,
                            "error": error_msg,
                            "error_type": "InitializationError"
                        }
                    )
                    
                    raise DocumentProcessingError(error_msg)
            
            # Process document
            metadata_context.start_timing(f"{self._processor_name}_file_processing")
            result = self._process_file(document_path, options, metadata_context)
            metadata_context.end_timing(f"{self._processor_name}_file_processing")
            
            # Add metadata
            if "metadata" not in result:
                metadata_context.start_timing("metadata_extraction")
                result["metadata"] = self.extract_metadata(document_path, metadata_context)
                metadata_context.end_timing("metadata_extraction")
            
            # Event for successful processing
            self.next_layer.emit_simple_event(
                ProcessingEventType.PROCESSING_COMPLETE,
                document_path,
                {
                    "processor": self._processor_name,
                    "success": True,
                    "result_keys": list(result.keys())
                }
            )
            
            metadata_context.end_timing(f"{self._processor_name}_processing")
            return result
            
        except DocumentValidationError:
            # Re-raise validation errors without wrapping
            raise
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            metadata_context.record_error(
                component=self._processor_name,
                message=error_msg,
                error_type=type(e).__name__,
                is_fatal=True
            )
            metadata_context.end_timing(f"{self._processor_name}_processing")
            
            # Event for error
            self.next_layer.emit_simple_event(
                ProcessingEventType.ERROR_OCCURRED,
                document_path,
                {
                    "processor": self._processor_name,
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            
            raise DocumentProcessingError(error_msg)
        finally:
            # Clean up temporary resources after processing
            self._cleanup_resources()

    @abstractmethod
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
        pass

    def cleanup(self) -> None:
        """
        Clean up all adapter resources.
        
        This public method handles cleanup by calling the template method
        and emitting appropriate events.
        """
        try:
            # Emit cleanup event
            self.next_layer.emit_simple_event(
                ProcessingEventType.CLEANUP,
                f"adapter_{self._processor_name}",
                {"adapter": self.__class__.__name__}
            )
            
            # Call template method for actual cleanup
            self._cleanup_resources()
            
            # Emit completion event
            self.next_layer.emit_simple_event(
                ProcessingEventType.CLEANUP_COMPLETE,
                f"adapter_{self._processor_name}",
                {"adapter": self.__class__.__name__}
            )
            
        except Exception as e:
            logger.error(f"Error during cleanup of {self.__class__.__name__}: {str(e)}")
            
            # Emit error event
            self.next_layer.emit_simple_event(
                ProcessingEventType.ERROR_OCCURRED,
                f"adapter_{self._processor_name}",
                {
                    "adapter_class": self.__class__.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
    
    def _cleanup_resources(self) -> None:
        """
        Clean up resources used by the adapter.
        
        This method cleans up temporary files and resources.
        """
        # Clean up temporary files
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary file {temp_file}: {str(e)}")
        
        self._temp_files = []
    
    @handle_document_errors
    def prepare_for_extraction(self, document_path: str, **kwargs) -> Dict[str, Any]:
        """
        Prepare document for entity extraction with standardized output format.
        This creates a consistent format for all knowledge graph processing.
        """
        # Get document processing result
        process_result = self.process_document(document_path)
        
        # Create standardized output
        extraction_data = {
            "document_id": hashlib.md5(document_path.encode()).hexdigest(),
            "document_path": document_path,
            "document_type": self._get_document_type(document_path),
            
            # Content flags
            "has_text": len(process_result.get("text", "")) > 0,
            "has_visual_elements": len(process_result.get("images", [])) > 0 or len(process_result.get("visual_elements", [])) > 0,
            
            # Structured content
            "content": {
                "text": process_result.get("text", ""),
                "sections": process_result.get("sections", []),
                "blocks": process_result.get("blocks", []),
                "images": process_result.get("images", []),
                "tables": process_result.get("tables", []),
                "charts": process_result.get("charts", []),
                "forms": process_result.get("forms", [])
            },
            
            # Enhanced metadata
            "metadata": {
                "source": document_path,
                "creation_date": self._extract_creation_date(document_path),
                "modification_date": self._extract_modification_date(document_path),
                "title": process_result.get("metadata", {}).get("title", os.path.basename(document_path)),
                "author": process_result.get("metadata", {}).get("author", "unknown"),
                "language": process_result.get("metadata", {}).get("language", "unknown"),
                "mime_type": mimetypes.guess_type(document_path)[0] or "application/octet-stream",
                "extraction_timestamp": datetime.now().isoformat()
            },
            
            # Analysis results
            "analysis": {
                "ocr_results": process_result.get("ocr_results", {}),
                "colpali_results": process_result.get("colpali_results", {}),
                "confidence": process_result.get("confidence", 0.0)
            }
        }
        
        return extraction_data
        
    async def check_health(self) -> Dict[str, Any]:
        """
        Check adapter health status.
        
        Returns:
            Dict containing health check results
        """
        try:
            # Prepare health info
            health_info = {
                "status": "healthy",
                "adapter": self.__class__.__name__,
                "version": self.VERSION,
                "capabilities": self.get_capabilities(),
                "timestamp": datetime.now().isoformat()
            }
            
            return health_info
            
        except Exception as e:
            return {
                "status": "error",
                "adapter": self.__class__.__name__,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }