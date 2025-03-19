import os
import logging
from typing import Dict, Any, Optional, List, ClassVar
from datetime import datetime

from ..interfaces.processor_interface import ProcessorInterface
from ...pipeline.quality.validation import QualityValidator
from ...integration.monitoring.metrics import ProcessingMetrics

logger = logging.getLogger(__name__)

class ProcessorBase(ProcessorInterface):
    """Base class for all document processors."""
    
    VERSION: ClassVar[str] = "1.0.0"
    SUPPORTED_FORMATS: ClassVar[List[str]] = []
    SUPPORTED_CONTENT_TYPES: ClassVar[List[str]] = []
    DEFAULT_CAPABILITIES: ClassVar[Dict[str, float]] = {
        "text_extraction": 0.0,
        "layout_analysis": 0.0,
        "table_extraction": 0.0,
        "image_processing": 0.0,
        "formula_recognition": 0.0
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the base processor."""
        self.config = config or {}
        self.is_initialized = False
        self.quality_validator = QualityValidator()
        self.metrics = ProcessingMetrics()
        self._initialize_monitoring()
    
    def _initialize_monitoring(self) -> None:
        """Initialize monitoring components."""
        self.processing_stats = {
            "total_processed": 0,
            "successful_processed": 0,
            "failed_processed": 0,
            "processing_times": [],
            "last_processed": None
        }
    
    def initialize(self) -> bool:
        """Initialize the processor."""
        if self.is_initialized:
            return True
            
        try:
            self._setup_processor()
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize processor: {str(e)}")
            return False
    
    def _setup_processor(self) -> None:
        """Setup processor-specific components."""
        pass
    
    def validate_input(self, document_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Validate input document."""
        if not os.path.exists(document_path):
            logger.error(f"Document not found: {document_path}")
            return False
            
        file_extension = os.path.splitext(document_path)[1].lower()
        if file_extension not in self.supported_formats:
            logger.error(f"Unsupported format: {file_extension}")
            return False
            
        return True
    
    def process(self, document_path: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process the document."""
        start_time = datetime.now()
        
        try:
            if not self.is_initialized:
                self.initialize()
            
            if not self.validate_input(document_path):
                raise ValueError(f"Invalid input: {document_path}")
            
            # Process document
            result = self._process_document(document_path, options or {})
            
            # Validate quality
            quality_result = self.quality_validator.validate_result(result)
            if not quality_result["is_valid"]:
                logger.warning(f"Quality validation failed: {quality_result['reason']}")
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(True, processing_time)
            
            return {
                "result": result,
                "quality": quality_result,
                "processing_time": processing_time,
                "processor_version": self.version
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(False, processing_time)
            raise
    
    def _process_document(self, document_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Implement actual document processing logic."""
        raise NotImplementedError("Subclasses must implement _process_document")
    
    def _update_metrics(self, success: bool, processing_time: float) -> None:
        """Update processing metrics."""
        self.processing_stats["total_processed"] += 1
        if success:
            self.processing_stats["successful_processed"] += 1
        else:
            self.processing_stats["failed_processed"] += 1
        
        self.processing_stats["processing_times"].append(processing_time)
        self.processing_stats["last_processed"] = datetime.now()
        
        # Keep only last 100 processing times
        if len(self.processing_stats["processing_times"]) > 100:
            self.processing_stats["processing_times"] = self.processing_stats["processing_times"][-100:]
        
        # Update metrics service
        self.metrics.record_processing_time(processing_time)
        self.metrics.record_processing_result(success)
    
    def get_capabilities(self) -> Dict[str, float]:
        """Get processor capabilities."""
        return self.DEFAULT_CAPABILITIES.copy()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.is_initialized = False
        self.metrics.flush()
    
    @property
    def version(self) -> str:
        """Get processor version."""
        return self.VERSION
    
    @property
    def supported_formats(self) -> List[str]:
        """Get supported formats."""
        return self.SUPPORTED_FORMATS
    
    @property
    def supported_content_types(self) -> List[str]:
        """Get supported content types."""
        return self.SUPPORTED_CONTENT_TYPES 