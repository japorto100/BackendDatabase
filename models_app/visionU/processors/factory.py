from typing import Dict, Any, Optional, List, Type
import logging
import os
from pathlib import Path

from models_app.visionU.core.interfaces.processor_interface import ProcessorInterface
from models_app.visionU.processors.content_types.text.text_processor import TextProcessor
from models_app.visionU.processors.content_types.image.image_processor import ImageProcessor
from models_app.visionU.processors.content_types.pdf.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)

class ProcessorFactory:
    """
    Factory class for creating and managing document processors.
    Handles processor registration, selection, and instantiation.
    """
    
    def __init__(self):
        """Initialize the processor factory."""
        self._processors: Dict[str, Type[ProcessorInterface]] = {}
        self._register_default_processors()
    
    def _register_default_processors(self) -> None:
        """Register default document processors."""
        self.register_processor("text", TextProcessor)
        self.register_processor("image", ImageProcessor)
        self.register_processor("pdf", PDFProcessor)
    
    def register_processor(self, processor_type: str, processor_class: Type[ProcessorInterface]) -> None:
        """
        Register a new processor type.
        
        Args:
            processor_type: Type identifier for the processor
            processor_class: Processor class to register
        """
        if not issubclass(processor_class, ProcessorInterface):
            raise ValueError(f"Processor class must implement ProcessorInterface: {processor_class}")
        
        self._processors[processor_type] = processor_class
        logger.info(f"Registered processor {processor_class.__name__} for type {processor_type}")
    
    def get_processor(self, processor_type: str, config: Optional[Dict[str, Any]] = None) -> ProcessorInterface:
        """
        Get an instance of a processor by type.
        
        Args:
            processor_type: Type of processor to create
            config: Optional configuration for the processor
            
        Returns:
            Instance of the requested processor
            
        Raises:
            ValueError: If processor type is not registered
        """
        if processor_type not in self._processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        processor_class = self._processors[processor_type]
        return processor_class(config)
    
    def select_processor(self, document_path: str, content_type: Optional[str] = None) -> ProcessorInterface:
        """
        Select an appropriate processor for a document.
        
        Args:
            document_path: Path to the document
            content_type: Optional content type hint
            
        Returns:
            Instance of the selected processor
            
        Raises:
            ValueError: If no suitable processor is found
        """
        # Get file extension
        ext = Path(document_path).suffix.lower()
        
        # Try content type first if provided
        if content_type:
            for processor_type, processor_class in self._processors.items():
                if content_type in processor_class.SUPPORTED_CONTENT_TYPES:
                    logger.info(f"Selected {processor_class.__name__} based on content type {content_type}")
                    return processor_class()
        
        # Try file extension
        for processor_type, processor_class in self._processors.items():
            if ext in processor_class.SUPPORTED_FORMATS:
                logger.info(f"Selected {processor_class.__name__} based on extension {ext}")
                return processor_class()
        
        # Default to text processor for unknown document types
        if ext in [".txt", ".doc", ".docx", ".rtf", ".odt"]:
            logger.info("Defaulting to TextProcessor for document")
            return TextProcessor()
        
        raise ValueError(f"No suitable processor found for document: {document_path}")
    
    def get_available_processors(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered processors.
        
        Returns:
            Dict containing processor information
        """
        info = {}
        for processor_type, processor_class in self._processors.items():
            processor = processor_class()
            info[processor_type] = {
                "name": processor_class.__name__,
                "version": processor.version,
                "supported_formats": processor.supported_formats,
                "supported_content_types": processor.supported_content_types,
                "capabilities": processor.get_capabilities()
            }
        return info
    
    def validate_processor_compatibility(self, document_path: str, processor_type: str) -> bool:
        """
        Check if a specific processor type is compatible with a document.
        
        Args:
            document_path: Path to the document
            processor_type: Type of processor to check
            
        Returns:
            bool: True if processor is compatible
        """
        if processor_type not in self._processors:
            return False
            
        processor_class = self._processors[processor_type]
        processor = processor_class()
        
        try:
            return processor.validate_input(document_path)
        except Exception as e:
            logger.error(f"Error validating processor compatibility: {str(e)}")
            return False 