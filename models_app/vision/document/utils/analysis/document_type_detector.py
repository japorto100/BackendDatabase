"""
Enhanced DocumentTypeDetector with more sophisticated document type analysis.
"""

import os
import magic
import logging
from typing import Dict, List, Any, Optional, Tuple
import mimetypes
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class DocumentTypeDetector:
    """Enhanced detector for document types with advanced PDF analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the document type detector.
        
        Args:
            config: Configuration dictionary with detector-specific settings.
        """
        self.config = config or {}
        self._init_mime_types()
        
        # Configure thresholds
        self.scanned_text_density_threshold = self.config.get("scanned_text_density_threshold", 0.02)
        self.scanned_image_ratio_threshold = self.config.get("scanned_image_ratio_threshold", 0.8)
        
        # OCR-based formats
        self.image_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp']
        
        # Standard text formats
        self.text_formats = ['.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm']
        
        # Office formats
        self.office_formats = ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.odt', '.ods', '.odp']
    
    def _init_mime_types(self):
        """Initialize MIME type mapping."""
        # Ensure common MIME types are registered
        mimetypes.init()
        # Add any missing types
        mimetypes.add_type('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx')
        mimetypes.add_type('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', '.xlsx')
        mimetypes.add_type('application/vnd.openxmlformats-officedocument.presentationml.presentation', '.pptx')
    
    def detect_document_type(self, file_path: str) -> Dict[str, Any]:
        """
        Detect document type and analyze its properties.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            Dict: Document type information.
        """
        if not os.path.exists(file_path):
            return self._get_unknown_type(file_path)
        
        # Basic file information
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        extension = os.path.splitext(file_name)[1].lower()
        
        # Get MIME type
        try:
            mime_type = magic.from_file(file_path, mime=True)
        except:
            mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        
        # Get basic document type
        doc_type = self._get_document_type(mime_type, extension)
        
        # Enhanced analysis for PDFs to determine if scanned
        if extension == '.pdf':
            is_scanned, image_ratio, text_blocks = self._analyze_pdf(file_path)
            doc_type.update({
                "is_scanned": is_scanned,
                "image_ratio": image_ratio,
                "text_blocks": text_blocks
            })
        
        # Determine processing category
        category = self._determine_processing_category(doc_type, extension)
        doc_type["category"] = category
        
        # Add priority for processing decisions
        doc_type["priority"] = self.get_processing_priority(file_path)
        
        return doc_type
    
    def _get_document_type(self, mime_type: str, extension: str) -> Dict[str, Any]:
        """
        Get the document type based on MIME type and extension.
        
        Args:
            mime_type: MIME type of the document.
            extension: File extension.
            
        Returns:
            Dict: Document type information.
        """
        # Default document type
        doc_type = {
            "mime_type": mime_type,
            "extension": extension,
            "type": "unknown"
        }
        
        # Determine document type
        if mime_type.startswith("image/"):
            doc_type["type"] = "image"
        elif mime_type == "application/pdf":
            doc_type["type"] = "pdf"
        elif mime_type.startswith("text/"):
            doc_type["type"] = "text"
        elif "spreadsheet" in mime_type or extension in ['.xlsx', '.xls', '.ods', '.csv']:
            doc_type["type"] = "spreadsheet"
        elif "presentation" in mime_type or extension in ['.pptx', '.ppt', '.odp']:
            doc_type["type"] = "presentation"
        elif "word" in mime_type or "document" in mime_type or extension in ['.docx', '.doc', '.odt', '.rtf']:
            doc_type["type"] = "document"
        
        return doc_type
    
    def _determine_processing_category(self, doc_type: Dict[str, Any], extension: str) -> str:
        """
        Determine the processing category for routing decisions.
        
        Args:
            doc_type: Document type information.
            extension: File extension.
            
        Returns:
            str: Processing category ('text', 'image', 'pdf', 'office', 'hybrid')
        """
        doc_type_str = doc_type.get("type", "unknown")
        
        if doc_type_str == "image" or extension in self.image_formats:
            return "image"
        
        elif doc_type_str == "pdf":
            # For PDFs, check if it's scanned or has high image content
            if doc_type.get("is_scanned", False) or doc_type.get("image_ratio", 0) > 0.6:
                return "image"  # Route to image processing pipeline
            elif doc_type.get("image_ratio", 0) > 0.3:
                return "hybrid"  # Mixed content, may need both pipelines
            else:
                return "pdf"  # Primarily text-based PDF
        
        elif doc_type_str in ["document", "spreadsheet", "presentation"] or extension in self.office_formats:
            return "office"
        
        elif doc_type_str == "text" or extension in self.text_formats:
            return "text"
        
        # Default for unknown types
        return "hybrid"
    
    def _get_unknown_type(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Get unknown document type information.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            Dict: Unknown document type information.
        """
        extension = os.path.splitext(file_path)[1].lower() if file_path else ""
        
        return {
            "mime_type": "application/octet-stream",
            "extension": extension,
            "type": "unknown",
            "category": "unknown",
            "priority": 0
        }
    
    def _analyze_pdf(self, file_path: str) -> Tuple[bool, float, int]:
        """
        Analyze PDF to determine if it's scanned and its image content ratio.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Tuple[bool, float, int]: 
                - is_scanned: Whether the PDF appears to be a scanned document
                - image_ratio: Ratio of image content to total content
                - text_blocks: Number of text blocks found
        """
        try:
            # Open PDF
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                return False, 0.0, 0
            
            # Initialize counters
            total_images = 0
            total_text_blocks = 0
            total_text_area = 0
            total_page_area = 0
            
            # Sample pages (up to 5) to determine PDF characteristics
            pages_to_sample = min(total_pages, 5)
            sampled_pages = list(range(min(3, pages_to_sample))) + list(range(max(0, total_pages - 2), total_pages))
            sampled_pages = sorted(list(set(sampled_pages)))[:pages_to_sample]
            
            for i in sampled_pages:
                page = doc[i]
                width, height = page.rect.width, page.rect.height
                page_area = width * height
                total_page_area += page_area
                
                # Extract text blocks
                blocks = page.get_text("blocks")
                total_text_blocks += len(blocks)
                
                # Calculate text area
                text_area = sum([(b[2]-b[0]) * (b[3]-b[1]) for b in blocks])
                total_text_area += text_area
                
                # Extract images
                images = page.get_images()
                total_images += len(images)
                
                # Check if page appears to be scanned
                # For scanned documents, typically few text blocks with low density
                text_density = text_area / page_area if page_area > 0 else 0
                
                # If page has very low text density and an image, it's likely scanned
                if text_density < self.scanned_text_density_threshold and len(images) > 0:
                    doc.close()
                    return True, 1.0, total_text_blocks
            
            # Calculate image ratio
            avg_text_area = total_text_area / total_page_area if total_page_area > 0 else 0
            image_ratio = 1.0 - avg_text_area
            
            # If overall low text coverage, it's likely a scanned document
            is_scanned = image_ratio > self.scanned_image_ratio_threshold
            
            doc.close()
            return is_scanned, image_ratio, total_text_blocks
            
        except Exception as e:
            logger.error(f"Error analyzing PDF {file_path}: {str(e)}")
            return False, 0.0, 0
    
    def is_office_document(self, file_path: str) -> bool:
        """
        Check if the file is an Office document.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            bool: True if the document is an Office document.
        """
        extension = os.path.splitext(file_path)[1].lower()
        return extension in self.office_formats
    
    def is_image(self, file_path: str) -> bool:
        """
        Check if the file is an image.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            bool: True if the document is an image.
        """
        extension = os.path.splitext(file_path)[1].lower()
        return extension in self.image_formats
    
    def is_pdf(self, file_path: str) -> bool:
        """
        Check if the file is a PDF.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            bool: True if the document is a PDF.
        """
        extension = os.path.splitext(file_path)[1].lower()
        return extension == '.pdf'
    
    def get_processing_priority(self, file_path: str) -> int:
        """
        Get the processing priority for a file type.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            int: Processing priority (higher values = higher priority).
        """
        extension = os.path.splitext(file_path)[1].lower()
        
        # Prioritize common formats
        if extension == '.pdf':
            return 100
        elif extension in ['.docx', '.xlsx', '.pptx']:
            return 90
        elif extension in self.image_formats:
            return 80
        elif extension in ['.doc', '.xls', '.ppt']:
            return 70
        elif extension in self.text_formats:
            return 60
        
        return 50  # Default priority
    
    def get_supported_document_types(self) -> List[str]:
        """
        Get a list of supported document types.
        
        Returns:
            List[str]: List of supported document types.
        """
        return [
            "pdf", "image", "document", "spreadsheet", 
            "presentation", "text", "html", "xml"
        ]
    
    def get_document_type_info(self, doc_type: str) -> Dict[str, Any]:
        """
        Get information about a specific document type.
        
        Args:
            doc_type: Document type name.
            
        Returns:
            Dict: Document type information.
        """
        type_info = {
            "type": doc_type,
            "supported": doc_type in self.get_supported_document_types()
        }
        
        # Add type-specific information
        if doc_type == "pdf":
            type_info.update({
                "can_be_scanned": True,
                "can_contain_images": True,
                "can_contain_tables": True,
                "needs_ocr_analysis": True,
                "extensions": [".pdf"]
            })
        elif doc_type == "image":
            type_info.update({
                "needs_ocr": True,
                "needs_vision_analysis": True,
                "extensions": self.image_formats
            })
        elif doc_type == "document":
            type_info.update({
                "can_contain_images": True,
                "can_contain_tables": True,
                "extensions": [".docx", ".doc", ".odt", ".rtf"]
            })
        elif doc_type == "spreadsheet":
            type_info.update({
                "tabular_data": True,
                "extensions": [".xlsx", ".xls", ".ods", ".csv"]
            })
        elif doc_type == "presentation":
            type_info.update({
                "can_contain_images": True,
                "can_contain_tables": True,
                "slide_based": True,
                "extensions": [".pptx", ".ppt", ".odp"]
            })
        elif doc_type == "text":
            type_info.update({
                "plain_text": True,
                "extensions": [".txt", ".md"]
            })
        
        return type_info
    
    def analyze_pdf_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Perform detailed analysis of PDF structure.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Dict: Detailed PDF structure information.
        """
        if not self.is_pdf(file_path):
            return {"error": "Not a PDF file"}
            
        try:
            doc = fitz.open(file_path)
            result = {
                "page_count": len(doc),
                "has_toc": bool(doc.get_toc()),
                "metadata": doc.metadata,
                "pages": []
            }
            
            # Sample up to 10 pages
            pages_to_sample = min(len(doc), 10)
            
            for i in range(pages_to_sample):
                page = doc[i]
                page_info = {
                    "page_number": i + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation,
                    "text_blocks": len(page.get_text("blocks")),
                    "image_count": len(page.get_images()),
                }
                result["pages"].append(page_info)
            
            doc.close()
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing PDF structure {file_path}: {str(e)}")
            return {"error": f"Error analyzing PDF: {str(e)}"}
    
    def get_document_complexity(self, file_path: str) -> float:
        """
        Calculate document complexity score for routing decisions.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            float: Complexity score (0.0-1.0)
        """
        try:
            # Initialize complexity factors
            complexity = 0.0
            
            # Check file type
            extension = os.path.splitext(file_path)[1].lower()
            
            # PDF documents
            if extension == ".pdf":
                result = self.analyze_pdf_structure(file_path)
                
                # Consider factors from PDF analysis
                if "error" not in result:
                    # Page count factor
                    page_count = result.get("page_count", 0)
                    page_factor = min(page_count / 20.0, 1.0)  # Scale by pages (caps at 20)
                    
                    # Check for table of contents (indicates structure)
                    toc_factor = 0.1 if result.get("has_toc", False) else 0.0
                    
                    # Image content factor
                    image_counts = [p.get("image_count", 0) for p in result.get("pages", [])]
                    avg_images = sum(image_counts) / len(image_counts) if image_counts else 0
                    image_factor = min(avg_images / 5.0, 0.3)  # Scale by avg images (caps at 0.3)
                    
                    # Text complexity
                    text_blocks = [p.get("text_blocks", 0) for p in result.get("pages", [])]
                    avg_blocks = sum(text_blocks) / len(text_blocks) if text_blocks else 0
                    text_factor = min(avg_blocks / 20.0, 0.3)  # Scale by text blocks (caps at 0.3)
                    
                    # Calculate overall complexity
                    complexity = 0.2 + (page_factor * 0.2) + toc_factor + image_factor + text_factor
            
            # Office documents
            elif extension in self.office_formats:
                # Office documents can be complex
                complexity = 0.6
                
                # Additional complexity for certain formats
                if extension in [".docx", ".doc"]:
                    complexity += 0.1  # Word docs can have complex layout
                elif extension in [".pptx", ".ppt"]:
                    complexity += 0.2  # Presentations often have mixed content
            
            # Images
            elif extension in self.image_formats:
                try:
                    # Check image complexity
                    img = Image.open(file_path)
                    width, height = img.size
                    
                    # Size factor (larger images may be more complex)
                    size_factor = min((width * height) / (1920 * 1080), 0.2)
                    
                    # Color complexity
                    color_mode = img.mode
                    color_factor = 0.1 if color_mode in ["RGB", "RGBA", "CMYK"] else 0.0
                    
                    # Resolution factor
                    try:
                        dpi = img.info.get("dpi", (72, 72))
                        dpi_factor = min(max(dpi) / 300.0, 0.1)  # Scale by DPI
                    except:
                        dpi_factor = 0.0
                        
                    complexity = 0.3 + size_factor + color_factor + dpi_factor
                    
                    img.close()
                except Exception as e:
                    logger.error(f"Error analyzing image {file_path}: {str(e)}")
                    complexity = 0.3  # Default for images
            
            # Cap complexity between 0 and 1
            return max(0.0, min(complexity, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating document complexity for {file_path}: {str(e)}")
            return 0.5  # Default moderate complexity

    def classify_document_type(self, document_path: str) -> Dict[str, float]:
        """
        Enhanced document type classification with confidence scores.
        
        This advanced classification method uses multiple signals to determine 
        the document type with confidence scores, rather than a binary classification.
        
        Args:
            document_path: Path to the document file.
            
        Returns:
            Dict: Document types with confidence scores (0-1).
        """
        classification = {
            "invoice": 0.0,
            "receipt": 0.0,
            "contract": 0.0,
            "letter": 0.0,
            "resume": 0.0,
            "report": 0.0,
            "form": 0.0,
            "article": 0.0,
            "presentation": 0.0,
            "spreadsheet": 0.0,
            "academic_paper": 0.0,
            "legal_document": 0.0,
            "manual": 0.0,
            "certificate": 0.0,
            "id_document": 0.0,
            "medical_record": 0.0
        }
        
        # Basic document type information
        doc_type_info = self.detect_document_type(document_path)
        
        try:
            # For PDFs and images, perform deep analysis
            if doc_type_info["type"] in ["pdf", "image"]:
                # Analyze PDF structure to get detailed insights
                if doc_type_info["type"] == "pdf" and PYMUPDF_AVAILABLE:
                    structure = self.analyze_pdf_structure(document_path)
                    
                    # Use structure features for classification
                    if structure.get("has_tables", False):
                        classification["report"] += 0.4
                        classification["invoice"] += 0.3
                        classification["spreadsheet"] += 0.3
                    
                    if structure.get("has_forms", False):
                        classification["form"] += 0.7
                        
                    if structure.get("has_signatures", False):
                        classification["contract"] += 0.5
                        classification["letter"] += 0.3
                        classification["legal_document"] += 0.4
                    
                    # Text-based analysis
                    text_blocks = structure.get("text_blocks", 0)
                    if text_blocks > 100:
                        classification["report"] += 0.3
                        classification["article"] += 0.5
                        classification["academic_paper"] += 0.3
                    
                    # Layout features
                    if structure.get("has_header_footer", False):
                        classification["letter"] += 0.4
                        classification["report"] += 0.3
                    
                    # Check for academic paper characteristics
                    if structure.get("has_sections", False) and structure.get("has_references", False):
                        classification["academic_paper"] += 0.6
                    
                    # Check for financial document characteristics
                    if structure.get("has_currency_symbols", False):
                        classification["invoice"] += 0.5
                        classification["receipt"] += 0.6
                    
                    # Font analysis (if available in structure)
                    fonts = structure.get("fonts", {})
                    if fonts:
                        # Legal documents often use specific fonts
                        if any(font in ["Times New Roman", "Garamond", "Baskerville"] for font in fonts.keys()):
                            classification["legal_document"] += 0.3
                            classification["contract"] += 0.2
                        
                        # Technical documents/manuals often use sans-serif fonts
                        if any(font in ["Arial", "Helvetica", "Calibri"] for font in fonts.keys()):
                            classification["manual"] += 0.3
                            classification["report"] += 0.2
                
                # Image analysis for type detection
                elif doc_type_info["type"] == "image" and OPENCV_AVAILABLE:
                    try:
                        from models_app.vision.utils.image_processing.detection import detect_form_elements
                        
                        # Load image
                        image = cv2.imread(document_path)
                        
                        # Detect form elements
                        form_elements = detect_form_elements(image)
                        if form_elements.get("has_form_elements", False):
                            classification["form"] += 0.8
                            
                            # Specific form types
                            form_type = form_elements.get("form_type", "")
                            if form_type == "questionnaire":
                                classification["form"] += 0.1  # Boost confidence
                            elif form_type == "id_card":
                                classification["id_document"] += 0.7
                        
                        # Check for receipt/invoice characteristics in the image
                        # (simplified - in practice would use OCR + pattern matching)
                        aspect_ratio = image.shape[1] / image.shape[0]
                        if 0.3 < aspect_ratio < 0.5:  # Typical receipt aspect ratio
                            classification["receipt"] += 0.4
                        
                    except Exception as e:
                        logger.warning(f"Error in image analysis for document classification: {str(e)}")
            
            # Office document specific classification
            elif doc_type_info["type"] == "document":
                # Word documents - check filename for hints
                filename = os.path.basename(document_path).lower()
                
                if any(term in filename for term in ["contract", "agreement", "legal"]):
                    classification["contract"] += 0.6
                    classification["legal_document"] += 0.5
                    
                if any(term in filename for term in ["cv", "resume", "lebenslauf"]):
                    classification["resume"] += 0.7
                    
                if any(term in filename for term in ["letter", "brief", "cover"]):
                    classification["letter"] += 0.6
                    
                if any(term in filename for term in ["report", "bericht"]):
                    classification["report"] += 0.6
                
            elif doc_type_info["type"] == "spreadsheet":
                classification["spreadsheet"] += 0.8
                
                # Financial spreadsheets often have certain keywords
                try:
                    # This would be expanded with actual spreadsheet parsing in a real implementation
                    filename = os.path.basename(document_path).lower()
                    
                    if any(term in filename for term in ["invoice", "billing", "rechnung"]):
                        classification["invoice"] += 0.7
                        
                    if any(term in filename for term in ["financial", "finance", "budget"]):
                        classification["report"] += 0.5
                except:
                    pass
                
            elif doc_type_info["type"] == "presentation":
                classification["presentation"] += 0.9  # High confidence for presentations
            
            # Calculate confidence normalization factor to ensure most likely types stand out
            max_confidence = max(classification.values())
            if max_confidence > 0:
                # Apply a softmax-like normalization to accentuate the highest probabilities
                # while keeping the relative relationships between scores
                for doc_type in classification:
                    # Boost high confidences, diminish low ones
                    classification[doc_type] = min(1.0, (classification[doc_type] / max_confidence) ** 0.7)
        
        except Exception as e:
            logger.error(f"Error during document classification: {str(e)}")
        
        return classification

    def analyze_document_metadata(self, document_path: str) -> Dict[str, Any]:
        """
        Analyze document metadata for classification enhancement.
        
        Args:
            document_path: Path to the document file.
            
        Returns:
            Dict: Metadata extracted from the document.
        """
        metadata = {
            "title": None,
            "author": None,
            "creator": None,
            "producer": None,
            "creation_date": None,
            "modification_date": None,
            "page_count": 0,
            "keywords": [],
            "subject": None,
            "language": None,
            "document_properties": {}
        }
        
        try:
            # Try to extract metadata based on file type
            file_extension = os.path.splitext(document_path)[1].lower()
            
            # PDF metadata extraction
            if file_extension == '.pdf' and PYMUPDF_AVAILABLE:
                doc = fitz.open(document_path)
                
                # Extract standard metadata
                pdf_metadata = doc.metadata
                if pdf_metadata:
                    metadata["title"] = pdf_metadata.get("title")
                    metadata["author"] = pdf_metadata.get("author")
                    metadata["creator"] = pdf_metadata.get("creator")
                    metadata["producer"] = pdf_metadata.get("producer")
                    metadata["creation_date"] = pdf_metadata.get("creationDate")
                    metadata["modification_date"] = pdf_metadata.get("modDate")
                    metadata["subject"] = pdf_metadata.get("subject")
                    
                    # Keywords may be in comma-separated format
                    if pdf_metadata.get("keywords"):
                        metadata["keywords"] = [k.strip() for k in pdf_metadata.get("keywords", "").split(",") if k.strip()]
                
                # Add page count
                metadata["page_count"] = len(doc)
                
                # Attempt to detect language (simplified)
                if doc.page_count > 0:
                    first_page_text = doc[0].get_text()
                    metadata["language"] = self._detect_language(first_page_text)
                
                doc.close()
                
            # Add more metadata extractors for other file types here
            # (Word documents, images with EXIF data, etc.)
            
        except Exception as e:
            logger.warning(f"Error extracting document metadata: {str(e)}")
        
        return metadata

    def _detect_language(self, text: str) -> Optional[str]:
        """
        Simple language detection based on character frequency.
        In a real implementation, use a proper language detection library.
        
        Args:
            text: Text to analyze
            
        Returns:
            str: Detected language code or None
        """
        if not text or len(text) < 20:
            return None
        
        # Very simplified language detection based on character sets
        # In production, use a library like langdetect or fastText
        
        # Count character frequencies
        text = text.lower()[:1000]  # First 1000 chars
        
        # Check for non-latin alphabets
        if any('\u0400' <= c <= '\u04FF' for c in text):
            return "ru"  # Cyrillic (Russian as example)
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return "zh"  # Chinese
        if any('\u3040' <= c <= '\u309F' for c in text) or any('\u30A0' <= c <= '\u30FF' for c in text):
            return "ja"  # Japanese
        
        # For Latin alphabets, check common letter patterns
        # (this is extremely simplified)
        text = ''.join(c for c in text if c.isalpha())
        
        # Count common letter combinations
        de_patterns = ['der', 'die', 'und', 'ist']
        en_patterns = ['the', 'and', 'ing', 'ion']
        es_patterns = ['los', 'las', 'del', 'con']
        fr_patterns = ['les', 'des', 'que', 'est']
        
        counts = {
            "de": sum(text.count(p) for p in de_patterns),
            "en": sum(text.count(p) for p in en_patterns),
            "es": sum(text.count(p) for p in es_patterns),
            "fr": sum(text.count(p) for p in fr_patterns)
        }
        
        # Get language with highest pattern count
        if max(counts.values()) > 0:
            return max(counts, key=counts.get)
        
        return None