import logging
from typing import Dict, Any, Optional, List, ClassVar
from datetime import datetime
import re

from ....core.base.processor_base import ProcessorBase
from ....pipeline.quality.validation import QualityValidator

logger = logging.getLogger(__name__)

class InvoiceProcessor(ProcessorBase):
    """Specialized processor for invoice documents."""
    
    VERSION: ClassVar[str] = "1.0.0"
    SUPPORTED_FORMATS: ClassVar[List[str]] = [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]
    SUPPORTED_CONTENT_TYPES: ClassVar[List[str]] = ["invoice", "receipt", "financial"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize invoice processor."""
        super().__init__(config)
        self.invoice_config = self.config.get("invoice", {})
        self._setup_field_extractors()
    
    def _setup_field_extractors(self) -> None:
        """Setup invoice field extraction patterns."""
        self.field_patterns = {
            "invoice_number": [
                r"Invoice\s*#?\s*[:]\s*([A-Z0-9-]+)",
                r"Invoice\s*Number\s*[:]\s*([A-Z0-9-]+)",
                r"Bill\s*Number\s*[:]\s*([A-Z0-9-]+)"
            ],
            "date": [
                r"Date\s*[:]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
                r"Invoice\s*Date\s*[:]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
            ],
            "total_amount": [
                r"Total\s*[:]\s*[$€£]?\s*([\d,]+\.?\d*)",
                r"Amount\s*Due\s*[:]\s*[$€£]?\s*([\d,]+\.?\d*)"
            ],
            "tax_amount": [
                r"Tax\s*[:]\s*[$€£]?\s*([\d,]+\.?\d*)",
                r"VAT\s*[:]\s*[$€£]?\s*([\d,]+\.?\d*)"
            ]
        }
    
    def get_capabilities(self) -> Dict[str, float]:
        """Get invoice processing capabilities."""
        capabilities = super().get_capabilities()
        capabilities.update({
            "text_extraction": 0.9,
            "layout_analysis": 0.8,
            "table_extraction": 0.9,
            "invoice_processing": 1.0,
            "field_extraction": 0.9
        })
        return capabilities
    
    def _process_document(self, document_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process invoice document."""
        try:
            # Get OCR text from document (assuming OCR is done in preprocessing)
            text_content = self._get_text_content(document_path)
            
            # Extract invoice fields
            extracted_fields = self._extract_invoice_fields(text_content)
            
            # Extract table data
            table_data = self._extract_table_data(text_content)
            
            # Calculate confidence scores
            field_confidence = self._calculate_field_confidence(extracted_fields)
            table_confidence = self._calculate_table_confidence(table_data)
            overall_confidence = (field_confidence + table_confidence) / 2
            
            # Structure result
            result = {
                "fields": extracted_fields,
                "table_data": table_data,
                "confidence": overall_confidence,
                "elements": []
            }
            
            # Add field elements with positions
            for field_name, field_data in extracted_fields.items():
                if "position" in field_data:
                    element = {
                        "text": field_data["value"],
                        "confidence": field_data["confidence"],
                        "bbox": field_data["position"],
                        "type": "field",
                        "field_name": field_name
                    }
                    result["elements"].append(element)
            
            # Add table elements
            for row in table_data:
                element = {
                    "text": str(row),
                    "confidence": row.get("confidence", 0.0),
                    "bbox": row.get("position", {}),
                    "type": "table_row"
                }
                result["elements"].append(element)
            
            # Add metadata
            result["metadata"] = {
                "processor": "invoice",
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "document_type": "invoice"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Invoice processing failed: {str(e)}")
            raise
    
    def _get_text_content(self, document_path: str) -> str:
        """Get text content from document."""
        # TODO: Implement text extraction or get from preprocessing
        return ""
    
    def _extract_invoice_fields(self, text_content: str) -> Dict[str, Any]:
        """Extract invoice fields using patterns."""
        extracted_fields = {}
        
        for field_name, patterns in self.field_patterns.items():
            field_data = {
                "value": None,
                "confidence": 0.0,
                "position": None
            }
            
            # Try each pattern
            for pattern in patterns:
                match = re.search(pattern, text_content)
                if match:
                    field_data["value"] = match.group(1)
                    field_data["confidence"] = 0.9  # High confidence for regex match
                    # TODO: Add position extraction
                    break
            
            extracted_fields[field_name] = field_data
        
        return extracted_fields
    
    def _extract_table_data(self, text_content: str) -> List[Dict[str, Any]]:
        """Extract table data from invoice."""
        # TODO: Implement table extraction
        return []
    
    def _calculate_field_confidence(self, fields: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted fields."""
        if not fields:
            return 0.0
            
        confidences = [
            field_data["confidence"]
            for field_data in fields.values()
            if field_data["value"] is not None
        ]
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _calculate_table_confidence(self, table_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for extracted table data."""
        if not table_data:
            return 0.0
            
        confidences = [
            row.get("confidence", 0.0)
            for row in table_data
        ]
        
        return sum(confidences) / len(confidences) if confidences else 0.0 