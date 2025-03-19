from typing import Dict, Any, Optional, List, ClassVar
import logging
from datetime import datetime

from models_app.visionU.core.base.processor_base import ProcessorBase

logger = logging.getLogger(__name__)

class TextProcessor(ProcessorBase):
    """
    Processor for handling text-based documents.
    Provides capabilities for text extraction, analysis, and structuring.
    """
    
    VERSION: ClassVar[str] = "1.0.0"
    SUPPORTED_FORMATS: ClassVar[List[str]] = [".txt", ".doc", ".docx", ".rtf", ".odt"]
    SUPPORTED_CONTENT_TYPES: ClassVar[List[str]] = ["text", "document"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the text processor with configuration."""
        super().__init__(config or {})
        self._setup_text_processing()
    
    def _setup_text_processing(self) -> None:
        """Set up text processing components."""
        self.text_config = {
            "min_confidence": self.config.get("min_confidence", 0.7),
            "max_text_length": self.config.get("max_text_length", 100000),
            "language_detection": self.config.get("language_detection", True),
            "text_cleaning": self.config.get("text_cleaning", True)
        }
    
    def validate_input(self, document_path: str) -> bool:
        """
        Validate if the input document is suitable for text processing.
        
        Args:
            document_path: Path to the document
            
        Returns:
            bool: True if document is valid for processing
        """
        if not super().validate_input(document_path):
            return False
            
        try:
            # Check file size
            file_size = self._get_file_size(document_path)
            if file_size > self.config.get("max_file_size", 10 * 1024 * 1024):  # 10MB default
                logger.warning(f"File size {file_size} exceeds maximum allowed size")
                return False
            
            # Validate file extension
            if not any(document_path.lower().endswith(ext) for ext in self.SUPPORTED_FORMATS):
                logger.warning(f"Unsupported file format for {document_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating document {document_path}: {str(e)}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the text processor.
        
        Returns:
            Dict containing processor capabilities
        """
        return {
            "text_extraction": True,
            "language_detection": self.text_config["language_detection"],
            "text_cleaning": self.text_config["text_cleaning"],
            "confidence_scoring": True,
            "metadata_extraction": True,
            "supported_formats": self.SUPPORTED_FORMATS,
            "supported_content_types": self.SUPPORTED_CONTENT_TYPES,
            "max_text_length": self.text_config["max_text_length"]
        }
    
    def _process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a text document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dict containing processed results
        """
        try:
            # Extract text content
            text_content = self._extract_text(document_path)
            
            # Clean text if enabled
            if self.text_config["text_cleaning"]:
                text_content = self._clean_text(text_content)
            
            # Detect language if enabled
            language = None
            if self.text_config["language_detection"]:
                language = self._detect_language(text_content)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(text_content)
            
            # Structure the result
            result = {
                "content": text_content,
                "confidence": confidence_score,
                "metadata": {
                    "processor": self.__class__.__name__,
                    "version": self.VERSION,
                    "timestamp": datetime.now().isoformat(),
                    "file_path": document_path,
                    "language": language,
                    "text_length": len(text_content),
                    "processing_config": self.text_config
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {str(e)}")
            raise
    
    def _extract_text(self, document_path: str) -> str:
        """Extract text from document."""
        # Implementation depends on file type
        # For simplicity, reading as text file here
        with open(document_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Basic text cleaning
        text = text.strip()
        # Remove multiple spaces
        text = ' '.join(text.split())
        return text
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the text."""
        try:
            # Simple language detection logic
            # In practice, use a proper language detection library
            return "en"  # Default to English
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return None
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for the processed text."""
        # Simple confidence calculation
        if not text:
            return 0.0
        
        # Basic metrics for confidence
        has_content = bool(text.strip())
        reasonable_length = 10 <= len(text) <= self.text_config["max_text_length"]
        well_formed = text.count('.') > 0  # Has at least one sentence
        
        # Calculate confidence score
        confidence = 0.0
        if has_content:
            confidence += 0.4
        if reasonable_length:
            confidence += 0.3
        if well_formed:
            confidence += 0.3
            
        return min(confidence, 1.0) 