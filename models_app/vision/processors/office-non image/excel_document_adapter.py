"""
ExcelDocumentAdapter: Adapter for processing Microsoft Excel (.xlsx, .xls) files.
"""

import os
import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

from models_app.vision.document.adapters.document_format_adapter import DocumentFormatAdapter
from models_app.vision.document.utils.core.processing_metadata_context import ProcessingMetadataContext
from models_app.vision.document.utils.error_handling.errors import (
    DocumentProcessingError,
    DocumentValidationError
)
from models_app.vision.document.utils.error_handling.handlers import (
    handle_document_errors,
    measure_processing_time
)
from models_app.vision.document.utils.core.next_layer_interface import ProcessingEventType

logger = logging.getLogger(__name__)

class ExcelDocumentAdapter(DocumentFormatAdapter):
    """
    Adapter for processing Microsoft Excel (.xlsx, .xls) files.
    Uses pandas and openpyxl to extract data, structure, and metadata.
    
    Features:
    - Extracts tabular data from sheets
    - Preserves worksheet structure
    - Handles formulas and cell references
    - Extracts chart data when possible
    - Retrieves document metadata
    """
    
    # Class-level constants
    VERSION = "1.0.0"
    CAPABILITIES = {
        "tabular_data_extraction": 0.9,
        "worksheet_structure": 0.8,
        "metadata_extraction": 0.7,
        "formula_handling": 0.7,
        "chart_extraction": 0.5
    }
    SUPPORTED_FORMATS = [".xlsx", ".xls", ".xlsm", ".ods"]
    PRIORITY = 80  # Relatively high priority for Excel-specific files
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Excel document adapter.
        
        Args:
            config: Configuration dictionary with adapter-specific settings.
        """
        super().__init__(config)
        self.pandas_module = None
        self.openpyxl_module = None
        self.xlrd_module = None
    
    def _initialize_adapter_components(self) -> None:
        """Initialize adapter-specific components."""
        try:
            # Attempt to import pandas and openpyxl
            import pandas as pd
            import openpyxl
            self.pandas_module = pd
            self.openpyxl_module = openpyxl
            
            # For older Excel formats
            try:
                import xlrd
                self.xlrd_module = xlrd
            except ImportError:
                logger.info("xlrd not available - fallback mechanisms will be used for .xls files")
            
            logger.info("Successfully initialized Excel document processing components")
            
        except ImportError as e:
            logger.warning(f"Could not import Excel processing libraries: {str(e)}. Using fallback mode.")
            self._initialize_fallback_components()
    
    def _initialize_fallback_components(self) -> None:
        """Initialize fallback components when main libraries are unavailable."""
        self.can_process_excel = False
        logger.info("Initialized fallback mode - document extraction will be limited")
    
    @handle_document_errors
    @measure_processing_time
    def _process_file(self, file_path: str, options: Dict[str, Any], 
                     metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process an Excel file and extract information.
        
        Args:
            file_path: Path to the Excel file
            options: Processing options
            metadata_context: Context for tracking metadata during processing
            
        Returns:
            Dictionary with extracted information
        """
        if metadata_context:
            metadata_context.start_timing("excel_processing")
            metadata_context.record_adapter_processing(
                adapter=self.__class__.__name__,
                file_path=file_path,
                capabilities=self.CAPABILITIES
            )
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Record processing start for performance tracking
        start_time = time.time()
        
        # Emit start event
        self.next_layer.emit_simple_event(
            event_type=ProcessingEventType.PROCESSING_PHASE_START,
            document_id=file_path,
            data={
                "adapter": self.__class__.__name__,
                "file_extension": file_extension,
                "phase": "excel_processing"
            }
        )
        
        try:
            # Process based on file extension
            if file_extension in ['.xlsx', '.xlsm']:
                result = self._process_xlsx_file(file_path, options, metadata_context)
            elif file_extension == '.xls':
                result = self._process_xls_file(file_path, options, metadata_context)
            elif file_extension == '.ods':
                result = self._process_ods_file(file_path, options, metadata_context)
            else:
                raise DocumentValidationError(
                    f"Unsupported file format: {file_extension}",
                    document_path=file_path
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add processor info to result
            result = {
                **result,
                "processor": {
                    "name": self.__class__.__name__,
                    "version": self.VERSION,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Emit performance metric
            self.next_layer.emit_simple_event(
                event_type=ProcessingEventType.PERFORMANCE_METRIC,
                document_id=file_path,
                data={
                    "adapter": self.__class__.__name__,
                    "processing_time": processing_time,
                    "success": True,
                    "file_extension": file_extension
                }
            )
            
            # Emit completion event
            self.next_layer.emit_simple_event(
                event_type=ProcessingEventType.PROCESSING_PHASE_END,
                document_id=file_path,
                data={
                    "adapter": self.__class__.__name__,
                    "phase": "excel_processing",
                    "success": True,
                    "processing_time": processing_time
                }
            )
            
            if metadata_context:
                metadata_context.end_timing("excel_processing")
                
            return result
            
        except Exception as e:
            # Calculate error processing time
            error_time = time.time() - start_time
            
            # Emit error event
            self.next_layer.emit_simple_event(
                event_type=ProcessingEventType.ERROR_OCCURRED,
                document_id=file_path,
                data={
                    "adapter": self.__class__.__name__,
                    "phase": "excel_processing",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time": error_time
                }
            )
            
            if metadata_context:
                metadata_context.record_error(
                    component=self.__class__.__name__,
                    message=f"Error processing Excel document: {str(e)}",
                    error_type=type(e).__name__
                )
                metadata_context.end_timing("excel_processing")
            
            logger.error(f"Error processing Excel document: {str(e)}")
            logger.debug(traceback.format_exc())
            
            raise DocumentProcessingError(f"Error processing Excel document: {str(e)}")

# Register this adapter
DocumentBaseAdapter.register_adapter("excel", ExcelDocumentAdapter)
