import os
import logging
from typing import Dict, Any, Optional, List, ClassVar
import pytesseract
from PIL import Image

from .....core.base.processor_base import ProcessorBase

logger = logging.getLogger(__name__)

class TesseractProcessor(ProcessorBase):
    """Tesseract OCR processor implementation."""
    
    VERSION: ClassVar[str] = "1.0.0"
    SUPPORTED_FORMATS: ClassVar[List[str]] = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    SUPPORTED_CONTENT_TYPES: ClassVar[List[str]] = ["text", "mixed"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Tesseract processor."""
        super().__init__(config)
        self.tesseract_config = self.config.get("tesseract", {})
        self.language = self.tesseract_config.get("language", "eng")
        self.psm = self.tesseract_config.get("psm", 3)
        self.oem = self.tesseract_config.get("oem", 3)
    
    def _setup_processor(self) -> None:
        """Setup Tesseract-specific components."""
        try:
            # Check if tesseract is installed
            pytesseract.get_tesseract_version()
            
            # Set custom path if provided
            if "tesseract_cmd" in self.tesseract_config:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_config["tesseract_cmd"]
            
            logger.info(f"Initialized Tesseract {self.version}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {str(e)}")
            raise
    
    def get_capabilities(self) -> Dict[str, float]:
        """Get Tesseract capabilities."""
        capabilities = super().get_capabilities()
        capabilities.update({
            "text_extraction": 0.8,
            "layout_analysis": 0.6,
            "table_extraction": 0.4,
            "formula_recognition": 0.3
        })
        return capabilities
    
    def _process_document(self, document_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process document using Tesseract OCR."""
        try:
            # Open and preprocess image
            image = Image.open(document_path)
            
            # Configure OCR options
            custom_config = f'--oem {self.oem} --psm {self.psm}'
            if "config" in options:
                custom_config += f' {options["config"]}'
            
            # Perform OCR
            ocr_result = pytesseract.image_to_data(
                image,
                lang=self.language,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract confidence scores
            confidences = [conf for conf in ocr_result["conf"] if conf != -1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Structure result
            result = {
                "text": " ".join(ocr_result["text"]),
                "confidence": avg_confidence / 100.0,  # Convert to 0-1 range
                "elements": []
            }
            
            # Add individual text elements with positions
            for i in range(len(ocr_result["text"])):
                if ocr_result["conf"][i] != -1:  # Skip empty elements
                    element = {
                        "text": ocr_result["text"][i],
                        "confidence": ocr_result["conf"][i] / 100.0,
                        "bbox": {
                            "x": ocr_result["left"][i],
                            "y": ocr_result["top"][i],
                            "width": ocr_result["width"][i],
                            "height": ocr_result["height"][i]
                        },
                        "type": "word",
                        "page": ocr_result["page_num"][i]
                    }
                    result["elements"].append(element)
            
            # Add metadata
            result["metadata"] = {
                "processor": "tesseract",
                "version": self.version,
                "language": self.language,
                "psm": self.psm,
                "oem": self.oem
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Tesseract processing failed: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Clean up Tesseract resources."""
        super().cleanup()
        # No specific cleanup needed for Tesseract
        pass 