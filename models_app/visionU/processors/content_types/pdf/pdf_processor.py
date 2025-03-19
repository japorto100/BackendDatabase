from typing import Dict, Any, Optional, List, ClassVar
import logging
from datetime import datetime
import os
import fitz  # PyMuPDF
import numpy as np

from models_app.visionU.core.base.processor_base import ProcessorBase

logger = logging.getLogger(__name__)

class PDFProcessor(ProcessorBase):
    """
    Processor for handling PDF documents.
    Provides capabilities for PDF text extraction, structure analysis, and metadata extraction.
    """
    
    VERSION: ClassVar[str] = "1.0.0"
    SUPPORTED_FORMATS: ClassVar[List[str]] = [".pdf"]
    SUPPORTED_CONTENT_TYPES: ClassVar[List[str]] = ["pdf", "document"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PDF processor with configuration."""
        super().__init__(config or {})
        self._setup_pdf_processing()
    
    def _setup_pdf_processing(self) -> None:
        """Set up PDF processing components."""
        self.pdf_config = {
            "min_confidence": self.config.get("min_confidence", 0.7),
            "extract_images": self.config.get("extract_images", True),
            "extract_tables": self.config.get("extract_tables", True),
            "max_pages": self.config.get("max_pages", 1000),
            "dpi": self.config.get("dpi", 300),
            "password": self.config.get("password", None)
        }
    
    def validate_input(self, document_path: str) -> bool:
        """
        Validate if the input document is suitable for PDF processing.
        
        Args:
            document_path: Path to the document
            
        Returns:
            bool: True if document is valid for processing
        """
        if not super().validate_input(document_path):
            return False
            
        try:
            # Check file extension
            if not document_path.lower().endswith(".pdf"):
                logger.warning(f"Not a PDF file: {document_path}")
                return False
            
            # Try opening the PDF
            with fitz.open(document_path) as doc:
                # Check if encrypted
                if doc.is_encrypted and not self.pdf_config["password"]:
                    logger.warning("PDF is encrypted and no password provided")
                    return False
                
                # Check number of pages
                if len(doc) > self.pdf_config["max_pages"]:
                    logger.warning(f"PDF has too many pages: {len(doc)}")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error validating PDF {document_path}: {str(e)}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the PDF processor.
        
        Returns:
            Dict containing processor capabilities
        """
        return {
            "text_extraction": True,
            "image_extraction": self.pdf_config["extract_images"],
            "table_extraction": self.pdf_config["extract_tables"],
            "metadata_extraction": True,
            "structure_analysis": True,
            "supported_formats": self.SUPPORTED_FORMATS,
            "supported_content_types": self.SUPPORTED_CONTENT_TYPES,
            "max_pages": self.pdf_config["max_pages"],
            "dpi": self.pdf_config["dpi"]
        }
    
    def _process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a PDF document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dict containing processed results
        """
        try:
            with fitz.open(document_path) as doc:
                # Extract metadata
                metadata = self._extract_metadata(doc)
                
                # Process pages
                pages = []
                images = []
                tables = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Extract text with blocks
                    page_text = page.get_text("dict")
                    pages.append(self._process_page(page_text, page_num))
                    
                    # Extract images if enabled
                    if self.pdf_config["extract_images"]:
                        page_images = self._extract_images(page, page_num)
                        images.extend(page_images)
                    
                    # Extract tables if enabled
                    if self.pdf_config["extract_tables"]:
                        page_tables = self._extract_tables(page, page_num)
                        tables.extend(page_tables)
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence(pages, images, tables)
                
                # Structure the result
                result = {
                    "metadata": {
                        **metadata,
                        "processor": self.__class__.__name__,
                        "version": self.VERSION,
                        "timestamp": datetime.now().isoformat(),
                        "file_path": document_path,
                        "total_pages": len(doc)
                    },
                    "content": {
                        "pages": pages,
                        "images": images if self.pdf_config["extract_images"] else [],
                        "tables": tables if self.pdf_config["extract_tables"] else []
                    },
                    "confidence": confidence_score
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error processing PDF {document_path}: {str(e)}")
            raise
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = doc.metadata
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
            "encryption": doc.is_encrypted,
            "pdf_version": doc.version
        }
    
    def _process_page(self, page_dict: Dict[str, Any], page_num: int) -> Dict[str, Any]:
        """Process a page's text content."""
        blocks = page_dict.get("blocks", [])
        processed_blocks = []
        
        for block in blocks:
            if block.get("type") == 0:  # Text block
                processed_blocks.append({
                    "type": "text",
                    "bbox": block.get("bbox", []),
                    "text": "".join(span.get("text", "") for line in block.get("lines", [])
                                  for span in line.get("spans", [])),
                    "font": block.get("lines", [{}])[0].get("spans", [{}])[0].get("font", "")
                    if block.get("lines") else ""
                })
            elif block.get("type") == 1:  # Image block
                processed_blocks.append({
                    "type": "image",
                    "bbox": block.get("bbox", [])
                })
        
        return {
            "page_number": page_num + 1,
            "blocks": processed_blocks,
            "rotation": page_dict.get("rotation", 0),
            "dimensions": {
                "width": page_dict.get("width", 0),
                "height": page_dict.get("height", 0)
            }
        }
    
    def _extract_images(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from a page."""
        images = []
        image_list = page.get_images()
        
        for img_idx, img in enumerate(image_list):
            xref = img[0]
            base_image = page.parent.extract_image(xref)
            
            if base_image:
                images.append({
                    "page_number": page_num + 1,
                    "image_index": img_idx + 1,
                    "width": base_image["width"],
                    "height": base_image["height"],
                    "colorspace": base_image["colorspace"],
                    "bpc": base_image["bpc"],
                    "type": base_image["ext"],
                    "size": len(base_image["image"])
                })
        
        return images
    
    def _extract_tables(self, page: fitz.Page, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables from a page."""
        # Simple table detection based on text layout
        # In practice, you'd want to use a dedicated table extraction library
        tables = []
        words = page.get_text("words")
        
        if words:
            # Group words by their vertical position (simple table row detection)
            rows = {}
            for word in words:
                y_pos = round(word[3])  # bottom y-coordinate
                if y_pos not in rows:
                    rows[y_pos] = []
                rows[y_pos].append(word[4])  # word text
            
            # If we have multiple rows with similar number of cells, consider it a table
            row_lengths = [len(row) for row in rows.values()]
            if len(row_lengths) > 2 and len(set(row_lengths)) < len(row_lengths) / 2:
                tables.append({
                    "page_number": page_num + 1,
                    "rows": list(rows.values()),
                    "row_count": len(rows),
                    "estimated_columns": max(row_lengths)
                })
        
        return tables
    
    def _calculate_confidence(self, pages: List[Dict[str, Any]], 
                            images: List[Dict[str, Any]], 
                            tables: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the processed PDF."""
        confidence = 0.0
        
        # Text quality check
        if pages:
            text_blocks = sum(len([b for b in page["blocks"] if b["type"] == "text"]) 
                            for page in pages)
            if text_blocks > 0:
                confidence += 0.4
        
        # Image quality check
        if self.pdf_config["extract_images"]:
            if images:
                image_quality = sum(1 for img in images 
                                  if img["width"] >= 100 and img["height"] >= 100) / len(images)
                confidence += 0.3 * image_quality
        else:
            confidence += 0.3
        
        # Table quality check
        if self.pdf_config["extract_tables"]:
            if tables:
                table_quality = sum(1 for table in tables 
                                  if table["row_count"] > 2) / len(tables)
                confidence += 0.3 * table_quality
        else:
            confidence += 0.3
        
        return min(confidence, 1.0) 