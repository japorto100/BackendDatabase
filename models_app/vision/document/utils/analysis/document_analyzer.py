"""
Document analyzer for detailed document structure analysis.
Works with DocumentTypeDetector for intelligent document routing.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import mimetypes
from datetime import datetime
from dataclasses import dataclass
import pytesseract
import cv2

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

@dataclass
class AnalysisResult:
    """Results from document analysis."""
    document_type: str
    confidence: float
    features: Dict[str, Any]
    quality_metrics: Dict[str, float]
    content_summary: Dict[str, Any]

@dataclass
class DocumentFeatures:
    """Extracted document features."""
    has_text: bool = False
    has_tables: bool = False
    has_images: bool = False
    has_forms: bool = False
    has_signatures: bool = False
    has_charts: bool = False
    layout_complexity: float = 0.0
    text_density: float = 0.0
    image_ratio: float = 0.0
    language_info: Dict[str, float] = None

class DocumentAnalyzer:
    """Enhanced document analyzer with comprehensive feature extraction."""
    
    VERSION = "2.0.0"
    
    def __init__(self):
        self.supported_languages = ["eng", "deu", "fra", "spa"]  # Add more as needed
        self.min_confidence = 0.6
    
    def analyze_document(self, document_path: str) -> AnalysisResult:
        """
        Perform comprehensive document analysis.
        
        Args:
            document_path: Path to the document
            
        Returns:
            AnalysisResult containing analysis details
        """
        # Extract basic features
        features = self._extract_features(document_path)
        
        # Analyze document quality
        quality_metrics = self._analyze_quality(document_path)
        
        # Analyze content and structure
        content_summary = self._analyze_content(document_path, features)
        
        # Determine document type and confidence
        doc_type, confidence = self._determine_document_type(features, content_summary)
        
        return AnalysisResult(
            document_type=doc_type,
            confidence=confidence,
            features=features.__dict__,
            quality_metrics=quality_metrics,
            content_summary=content_summary
        )
    
    def _extract_features(self, document_path: str) -> DocumentFeatures:
        """Extract document features using computer vision and OCR."""
        features = DocumentFeatures()
        
        try:
            # Load image
            image = cv2.imread(document_path)
            if image is None:
                raise ValueError(f"Could not load image from {document_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Text detection
            text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            features.has_text = len(text_data["text"]) > 0
            
            # Calculate text density
            if features.has_text:
                text_area = sum(w * h for w, h in zip(text_data["width"], text_data["height"]))
                total_area = image.shape[0] * image.shape[1]
                features.text_density = text_area / total_area
            
            # Table detection
            features.has_tables = self._detect_tables(gray)
            
            # Form detection
            features.has_forms = self._detect_forms(gray)
            
            # Image content ratio
            features.image_ratio = self._calculate_image_ratio(image)
            
            # Layout complexity
            features.layout_complexity = self._analyze_layout_complexity(gray)
            
            # Language detection
            features.language_info = self._detect_languages(text_data["text"])
            
            return features
            
        except Exception as e:
            raise DocumentAnalysisError(f"Feature extraction failed: {str(e)}")
    
    def _detect_tables(self, gray_image: np.ndarray) -> bool:
        """Detect presence of tables in the document."""
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Find lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1):
                    horizontal_lines += 1
                else:
                    vertical_lines += 1
            
            # If we have both horizontal and vertical lines, likely a table
            return horizontal_lines > 2 and vertical_lines > 2
        
        return False
    
    def _detect_forms(self, gray_image: np.ndarray) -> bool:
        """Detect if the document is a form."""
        # Look for checkboxes and input fields
        rectangles = self._find_rectangles(gray_image)
        if len(rectangles) > 5:  # Threshold for form detection
            return True
            
        # Look for aligned text that might be labels
        text_data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)
        aligned_text = self._check_text_alignment(text_data)
        
        return aligned_text > 0.7  # 70% aligned text suggests a form
    
    def _calculate_image_ratio(self, image: np.ndarray) -> float:
        """Calculate the ratio of image content to total document area."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate saturation mean
        saturation = hsv[:, :, 1]
        return np.mean(saturation) / 255.0
    
    def _analyze_layout_complexity(self, gray_image: np.ndarray) -> float:
        """Analyze the complexity of the document layout."""
        # Use edge detection to measure layout complexity
        edges = cv2.Canny(gray_image, 50, 150)
        complexity = np.sum(edges > 0) / (gray_image.shape[0] * gray_image.shape[1])
        
        return min(1.0, complexity * 3)  # Normalize to [0, 1]
    
    def _detect_languages(self, text_list: List[str]) -> Dict[str, float]:
        """Detect languages present in the document."""
        if not text_list:
            return {"unknown": 1.0}
            
        try:
            # Use pytesseract's language detection
            lang_scores = {}
            for lang in self.supported_languages:
                data = pytesseract.image_to_data(Image.fromarray(np.zeros((1, 1))), 
                                               lang=lang,
                                               output_type=pytesseract.Output.DICT)
                if data["conf"] and max(data["conf"]) > self.min_confidence:
                    lang_scores[lang] = float(max(data["conf"])) / 100
            
            # Normalize scores
            total = sum(lang_scores.values())
            if total > 0:
                return {k: v/total for k, v in lang_scores.items()}
            
            return {"unknown": 1.0}
            
        except Exception:
            return {"unknown": 1.0}
    
    def _analyze_quality(self, document_path: str) -> Dict[str, float]:
        """Analyze document quality metrics."""
        try:
            image = cv2.imread(document_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate various quality metrics
            metrics = {
                "resolution": self._calculate_resolution_score(image),
                "contrast": self._calculate_contrast_score(gray),
                "brightness": self._calculate_brightness_score(gray),
                "noise": self._calculate_noise_score(gray),
                "blur": self._calculate_blur_score(gray)
            }
            
            # Calculate overall quality score
            metrics["overall"] = sum(metrics.values()) / len(metrics)
            
            return metrics
            
        except Exception as e:
            raise DocumentAnalysisError(f"Quality analysis failed: {str(e)}")
    
    def _calculate_resolution_score(self, image: np.ndarray) -> float:
        """Calculate resolution quality score."""
        height, width = image.shape[:2]
        min_dimension = min(height, width)
        return min(1.0, min_dimension / 1000.0)  # Normalize to [0, 1]
    
    def _calculate_contrast_score(self, gray_image: np.ndarray) -> float:
        """Calculate contrast quality score."""
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        return 1.0 - np.std(hist)  # Higher contrast = lower histogram std
    
    def _calculate_brightness_score(self, gray_image: np.ndarray) -> float:
        """Calculate brightness quality score."""
        mean_brightness = np.mean(gray_image) / 255.0
        return 1.0 - abs(0.5 - mean_brightness) * 2  # Optimal at 0.5
    
    def _calculate_noise_score(self, gray_image: np.ndarray) -> float:
        """Calculate noise quality score."""
        denoised = cv2.fastNlMeansDenoising(gray_image)
        noise_diff = np.abs(gray_image - denoised)
        return 1.0 - np.mean(noise_diff) / 255.0
    
    def _calculate_blur_score(self, gray_image: np.ndarray) -> float:
        """Calculate blur quality score using Laplacian variance."""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return min(1.0, np.var(laplacian) / 500)  # Normalize to [0, 1]
    
    def _analyze_content(self, document_path: str, features: DocumentFeatures) -> Dict[str, Any]:
        """Analyze document content and structure."""
        content_summary = {
            "structure": {
                "sections": self._detect_sections(document_path),
                "hierarchy_level": self._analyze_hierarchy(),
                "formatting_consistency": self._analyze_formatting()
            },
            "content_types": {
                "text_blocks": features.has_text,
                "tables": features.has_tables,
                "images": features.has_images,
                "forms": features.has_forms,
                "signatures": features.has_signatures,
                "charts": features.has_charts
            },
            "statistics": {
                "text_density": features.text_density,
                "image_ratio": features.image_ratio,
                "layout_complexity": features.layout_complexity
            }
        }
        
        return content_summary
    
    def _determine_document_type(
        self, 
        features: DocumentFeatures,
        content_summary: Dict[str, Any]
    ) -> tuple[str, float]:
        """Determine document type and confidence level."""
        # Implement document type classification logic
        type_scores = {
            "text": 0.0,
            "form": 0.0,
            "image": 0.0,
            "mixed": 0.0
        }
        
        # Score based on features
        if features.has_text:
            type_scores["text"] += features.text_density
        if features.has_forms:
            type_scores["form"] += 0.8
        if features.has_images:
            type_scores["image"] += features.image_ratio
        if features.layout_complexity > 0.7:
            type_scores["mixed"] += 0.6
        
        # Get highest scoring type
        doc_type, confidence = max(type_scores.items(), key=lambda x: x[1])
        
        return doc_type, min(1.0, confidence)

    def analyze_layout_structure(self, document_path: str) -> Dict[str, Any]:
        """Performs advanced layout analysis for better structure understanding."""
        layout_info = {
            "column_structure": self._detect_columns(document_path),
            "section_hierarchy": self._analyze_section_hierarchy(document_path),
            "form_elements": self._detect_form_elements(document_path),
            "layout_complexity": 0.0  # Will be calculated
        }
        # Calculate layout complexity
        layout_info["layout_complexity"] = self._calculate_layout_complexity(layout_info)
        return layout_info

    def _detect_form_elements(self, document_path: str) -> Dict[str, Any]:
        """
        Detect form elements in a document such as checkboxes, radio buttons, text fields, etc.
        
        Args:
            document_path: Path to the document file.
            
        Returns:
            Dict: Information about detected form elements
        """
        # Initialize result structure
        form_elements = {
            "has_form_elements": False,
            "form_type": "unknown",
            "elements": [],
            "confidence": 0.0,
            "structured_data": {}
        }
        
        try:
            # Handle different document types
            if document_path.lower().endswith('.pdf'):
                if not PYMUPDF_AVAILABLE:
                    logger.warning("PyMuPDF not available, skipping form detection for PDF")
                    return form_elements
                
                # Open PDF document
                doc = fitz.open(document_path)
                
                # Check if document has form fields
                has_form_fields = False
                total_fields = 0
                form_data = {}
                
                # Check each page for form elements
                for page_idx, page in enumerate(doc):
                    # Check for interactive form fields
                    if page.widgets:
                        has_form_fields = True
                        
                        # Extract form field information
                        for widget in page.widgets:
                            field_type = widget.field_type
                            field_name = widget.field_name
                            field_value = widget.field_value
                            field_rect = widget.rect
                            
                            # Skip fields without names or out of page bounds
                            if not field_name or not field_rect.intersects(page.rect):
                                continue
                            
                            total_fields += 1
                            
                            # Create form element entry
                            element = {
                                "type": self._map_pdf_field_type(field_type),
                                "name": field_name,
                                "value": field_value if field_value is not None else "",
                                "bbox": [field_rect.x0, field_rect.y0, field_rect.x1, field_rect.y1],
                                "page": page_idx,
                                "confidence": 1.0  # Interactive fields have perfect confidence
                            }
                            
                            form_elements["elements"].append(element)
                            
                            # Add to structured data dictionary
                            form_data[field_name] = field_value
                    
                    # Also check for non-interactive form elements using visual detection
                    if OPENCV_AVAILABLE:
                        # Get page as image
                        pix = page.get_pixmap()
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                        
                        # Convert to grayscale if needed
                        if pix.n == 4:  # RGBA
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
                        elif pix.n == 3:  # RGB
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                            
                        # Detect visual form elements
                        visual_elements = self._detect_visual_form_elements(img_array, page_idx)
                        
                        # Add visual elements to results
                        for elem in visual_elements:
                            form_elements["elements"].append(elem)
                            total_fields += 1
                
                # Update form detection result
                if total_fields > 0:
                    form_elements["has_form_elements"] = True
                    form_elements["confidence"] = 1.0 if has_form_fields else 0.8
                    form_elements["form_type"] = self._determine_form_type(form_elements["elements"])
                    form_elements["structured_data"] = form_data
                
                doc.close()
                
            elif document_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp')):
                if not OPENCV_AVAILABLE:
                    logger.warning("OpenCV not available, skipping form detection for image")
                    return form_elements
                
                # Load image
                img = cv2.imread(document_path)
                if img is None:
                    logger.error(f"Failed to load image: {document_path}")
                    return form_elements
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                
                # Detect visual form elements
                visual_elements = self._detect_visual_form_elements(gray, 0)
                
                # Update form detection result
                if visual_elements:
                    form_elements["has_form_elements"] = True
                    form_elements["confidence"] = 0.8
                    form_elements["elements"] = visual_elements
                    form_elements["form_type"] = self._determine_form_type(visual_elements)
                    
                    # Build structured data from visual elements
                    structured_data = {}
                    for elem in visual_elements:
                        if "name" in elem and elem["name"]:
                            structured_data[elem["name"]] = elem.get("value", "")
                    form_elements["structured_data"] = structured_data
            
            return form_elements
            
        except Exception as e:
            logger.error(f"Error detecting form elements: {str(e)}")
            return form_elements
    
    def _detect_visual_form_elements(self, gray_image: np.ndarray, page_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Detect visual form elements in an image using computer vision techniques.
        
        Args:
            gray_image: Grayscale image as numpy array
            page_idx: Page index for multi-page documents
            
        Returns:
            List of detected form elements
        """
        elements = []
        
        try:
            # Detect checkboxes
            checkboxes = self._detect_checkboxes(gray_image)
            for checkbox in checkboxes:
                x, y, w, h = checkbox["bbox"]
                is_checked = checkbox["checked"]
                confidence = checkbox["confidence"]
                
                # Try to find associated label using relative position
                label = self._find_element_label(gray_image, x, y, w, h)
                
                elements.append({
                    "type": "checkbox",
                    "bbox": [x, y, x+w, y+h],
                    "checked": is_checked,
                    "name": label or f"checkbox_{len(elements)}",
                    "value": "checked" if is_checked else "unchecked",
                    "page": page_idx,
                    "confidence": confidence
                })
            
            # Detect radio buttons
            radio_buttons = self._detect_radio_buttons(gray_image)
            for radio in radio_buttons:
                x, y, w, h = radio["bbox"]
                is_selected = radio["selected"]
                confidence = radio["confidence"]
                
                # Try to find associated label
                label = self._find_element_label(gray_image, x, y, w, h)
                
                elements.append({
                    "type": "radio",
                    "bbox": [x, y, x+w, y+h],
                    "selected": is_selected,
                    "name": label or f"radio_{len(elements)}",
                    "value": "selected" if is_selected else "unselected",
                    "page": page_idx,
                    "confidence": confidence
                })
            
            # Detect text fields (using contours and layout analysis)
            text_fields = self._detect_text_fields(gray_image)
            for field in text_fields:
                x, y, w, h = field["bbox"]
                confidence = field["confidence"]
                
                # Try to find associated label
                label = self._find_element_label(gray_image, x, y, w, h)
                
                elements.append({
                    "type": "text_field",
                    "bbox": [x, y, x+w, y+h],
                    "name": label or f"field_{len(elements)}",
                    "value": "",  # Empty as we can't extract text here
                    "page": page_idx,
                    "confidence": confidence
                })
                
            return elements
            
        except Exception as e:
            logger.error(f"Error detecting visual form elements: {str(e)}")
            return []
    
    def _detect_checkboxes(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect checkboxes in a grayscale image.
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            List of detected checkboxes with positions and states
        """
        checkboxes = []
        
        # Apply adaptive threshold
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's square-ish and the right size
            aspect_ratio = float(w) / h
            area = w * h
            
            # Checkboxes are typically square and small
            if 0.8 < aspect_ratio < 1.2 and 100 < area < 2500:
                # Check if it's a checkbox (has 4 corners)
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    # Extract the checkbox region
                    checkbox_roi = binary[y:y+h, x:x+w]
                    
                    # Check if it's filled (checked)
                    total_pixels = checkbox_roi.shape[0] * checkbox_roi.shape[1]
                    filled_pixels = cv2.countNonZero(checkbox_roi)
                    fill_ratio = filled_pixels / total_pixels
                    
                    # Decision boundary for checked/unchecked
                    is_checked = fill_ratio > 0.2
                    
                    # Calculate confidence based on shape and fill ratio
                    shape_confidence = 1.0 - abs(aspect_ratio - 1.0)
                    fill_confidence = 1.0 if (fill_ratio < 0.1 or fill_ratio > 0.3) else 0.5
                    confidence = (shape_confidence + fill_confidence) / 2
                    
                    checkboxes.append({
                        "bbox": (x, y, w, h),
                        "checked": is_checked,
                        "confidence": confidence,
                        "fill_ratio": fill_ratio
                    })
        
        return checkboxes
    
    def _detect_radio_buttons(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect radio buttons in a grayscale image.
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            List of detected radio buttons with positions and states
        """
        radio_buttons = []
        
        # Apply threshold
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's circular and the right size
            aspect_ratio = float(w) / h
            area = w * h
            
            # Radio buttons are circular and small
            if 0.8 < aspect_ratio < 1.2 and 100 < area < 2500:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                if circularity > 0.7:  # Close to a perfect circle
                    # Extract the radio button region
                    radio_roi = binary[y:y+h, x:x+w]
                    
                    # Check if it's filled (selected)
                    total_pixels = radio_roi.shape[0] * radio_roi.shape[1]
                    filled_pixels = cv2.countNonZero(radio_roi)
                    fill_ratio = filled_pixels / total_pixels
                    
                    # Decision boundary for selected/unselected
                    is_selected = fill_ratio > 0.2
                    
                    # Calculate confidence based on shape and fill ratio
                    shape_confidence = circularity
                    fill_confidence = 1.0 if (fill_ratio < 0.1 or fill_ratio > 0.3) else 0.5
                    confidence = (shape_confidence + fill_confidence) / 2
                    
                    radio_buttons.append({
                        "bbox": (x, y, w, h),
                        "selected": is_selected,
                        "confidence": confidence,
                        "fill_ratio": fill_ratio
                    })
        
        return radio_buttons
    
    def _detect_text_fields(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect text fields in a grayscale image.
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            List of detected text fields
        """
        text_fields = []
        
        # Apply threshold
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to find horizontal lines
        kernel = np.ones((1, 50), np.uint8)
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio and size
            aspect_ratio = float(w) / h
            
            # Text fields are typically wide rectangles
            if aspect_ratio > 4 and w > 100:
                # Calculate confidence based on shape
                confidence = min(1.0, max(0.5, aspect_ratio / 10))
                
                text_fields.append({
                    "bbox": (x, y, w, h),
                    "confidence": confidence
                })
        
        return text_fields
    
    def _find_element_label(self, gray_image: np.ndarray, 
                           x: int, y: int, w: int, h: int, 
                           search_distance: int = 150) -> Optional[str]:
        """
        Find a label associated with a form element using OCR.
        This is a placeholder that would use an OCR engine in a real implementation.
        
        Args:
            gray_image: Grayscale image as numpy array
            x, y, w, h: Form element bounding box
            search_distance: Maximum distance to search for a label
            
        Returns:
            Label text if found, None otherwise
        """
        # This would require OCR integration which is beyond the scope of this implementation
        # In a real system, we would:
        # 1. Define a search region (left/above/right of the element)
        # 2. Use OCR to extract text in that region
        # 3. Return the closest text as the label
        
        # For now, return None to indicate no label was found
        return None
    
    def _map_pdf_field_type(self, field_type: int) -> str:
        """
        Map PyMuPDF field type codes to readable types.
        
        Args:
            field_type: PyMuPDF field type code
            
        Returns:
            String representation of the field type
        """
        # PyMuPDF field type mapping
        field_map = {
            1: "text_field",
            2: "checkbox",
            3: "radio",
            4: "listbox",
            5: "combobox",
            6: "signature"
        }
        return field_map.get(field_type, "unknown")
    
    def _determine_form_type(self, elements: List[Dict[str, Any]]) -> str:
        """
        Determine the overall form type based on detected elements.
        
        Args:
            elements: List of detected form elements
            
        Returns:
            Form type classification
        """
        # Count element types
        type_counts = {}
        for elem in elements:
            elem_type = elem.get("type", "unknown")
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
        
        # No elements
        if not type_counts:
            return "unknown"
        
        # Check for dominant types
        checkbox_count = type_counts.get("checkbox", 0)
        text_field_count = type_counts.get("text_field", 0)
        radio_count = type_counts.get("radio", 0)
        
        # Decision logic
        if checkbox_count > 5 and checkbox_count > text_field_count:
            return "checklist"
        elif text_field_count > 5 and text_field_count > checkbox_count:
            return "data_entry"
        elif radio_count > 3:
            return "questionnaire"
        elif text_field_count > 0 and checkbox_count > 0:
            return "mixed_form"
        else:
            return "simple_form"

    def _generate_kg_hints(self, quality_metrics: Dict[str, float], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate knowledge graph extraction hints based on quality metrics and analysis.
        
        Args:
            quality_metrics: Document quality metrics
            analysis_result: Document analysis results
            
        Returns:
            Dict: KG extraction hints
        """
        kg_hints = {
            "quality_concerns": quality_metrics["overall_quality"] < 0.4,
            "text_reliability": max(0.1, quality_metrics["blur_score"] * quality_metrics["contrast_score"]),
            "extraction_confidence_factor": min(1.0, quality_metrics["overall_quality"] + 0.2)
        }
        
        # Add document-specific hints
        if "has_tables" in analysis_result and analysis_result["has_tables"]:
            kg_hints["prioritize_table_extraction"] = True
            
        if "content_complexity" in analysis_result:
            complexity = analysis_result["content_complexity"]
            kg_hints["content_complexity"] = complexity
            
            if complexity > 0.7:
                kg_hints["use_advanced_extraction"] = True
                
        # If the document has very poor quality, suggest visual extraction priority
        if quality_metrics["overall_quality"] < 0.3:
            kg_hints["prioritize_visual_extraction"] = True
            
        # Add form-specific hints if document has forms
        if "form_elements" in analysis_result and analysis_result["form_elements"].get("has_form_elements", False):
            form_data = analysis_result["form_elements"]
            kg_hints["has_form_data"] = True
            kg_hints["form_type"] = form_data.get("form_type", "unknown")
            kg_hints["form_confidence"] = form_data.get("confidence", 0.0)
            
            # Add specific extraction strategies based on form type
            if form_data.get("form_type") == "checklist":
                kg_hints["extract_checkbox_relationships"] = True
            elif form_data.get("form_type") == "data_entry":
                kg_hints["extract_key_value_pairs"] = True
            elif form_data.get("form_type") == "questionnaire":
                kg_hints["extract_question_answers"] = True
        
        return kg_hints

class DocumentQualityAnalyzer:
    """Analyzes document image quality for optimal processing selection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the image quality analyzer.
        
        Args:
            config: Configuration dictionary with analyzer-specific settings.
        """
        self.config = config or {}
    
    def analyze_image_quality(self, image_path: str) -> Dict[str, Any]:
        """Analyzes image quality metrics for better processing decisions."""
        try:
            # Load image
            if isinstance(image_path, str):
                img = Image.open(image_path)
                img_array = np.array(img)
            else:
                # Handle case where image is already loaded
                img_array = np.array(image_path)
                
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
                
            # Calculate quality metrics
            quality_metrics = {
                "blur_score": self._detect_blur(gray),
                "contrast_score": self._analyze_contrast(gray),
                "noise_level": self._analyze_noise(gray),
                "resolution_adequacy": self._check_resolution(img_array),
                "overall_quality": 0.0  # Will be calculated
            }
            
            # Calculate overall quality score
            quality_metrics["overall_quality"] = self._calculate_overall_score(quality_metrics)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing image quality: {str(e)}")
            return {
                "blur_score": 0.5,
                "contrast_score": 0.5,
                "noise_level": 0.5,
                "resolution_adequacy": 0.5,
                "overall_quality": 0.5,
                "error": str(e)
            }
    
    def _detect_blur(self, gray_image: np.ndarray) -> float:
        """
        Detect image blur using Laplacian variance.
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            float: Blur score (0-1, higher is better/less blur)
        """
        try:
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            # Normalize to 0-1 range (higher is better, less blur)
            blur_score = min(1.0, laplacian_var / 500)
            return blur_score
        except Exception as e:
            logger.error(f"Error detecting blur: {str(e)}")
            return 0.5
    
    def _analyze_contrast(self, gray_image: np.ndarray) -> float:
        """
        Analyze image contrast.
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            float: Contrast score (0-1, higher is better contrast)
        """
        try:
            # Calculate histogram
            hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
            
            # Calculate standard deviation of histogram as measure of contrast
            std_dev = np.std(hist)
            
            # Normalize to 0-1 range (higher is better contrast)
            contrast_score = min(1.0, std_dev / 50)
            
            return contrast_score
        except Exception as e:
            logger.error(f"Error analyzing contrast: {str(e)}")
            return 0.5
    
    def _analyze_noise(self, gray_image: np.ndarray) -> float:
        """
        Analyze image noise level.
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            float: Noise level (0-1, lower is better/less noise)
        """
        try:
            # Apply median filter to remove noise
            denoised = cv2.medianBlur(gray_image, 5)
            
            # Calculate difference between original and denoised
            diff = cv2.absdiff(gray_image, denoised)
            
            # Noise level is the mean of the difference
            noise_level = np.mean(diff) / 255
            
            return noise_level
        except Exception as e:
            logger.error(f"Error analyzing noise: {str(e)}")
            return 0.5
    
    def _check_resolution(self, image: np.ndarray) -> float:
        """
        Check if image resolution is adequate for OCR.
        
        Args:
            image: Image as numpy array
            
        Returns:
            float: Resolution adequacy score (0-1, higher is better)
        """
        try:
            height, width = image.shape[:2]
            
            # Check if dimensions are reasonable for OCR
            min_dimension = min(height, width)
            
            # Normalize to 0-1 range (higher is better resolution)
            if min_dimension < 100:
                return 0.1
            elif min_dimension < 300:
                return 0.3
            elif min_dimension < 500:
                return 0.6
            elif min_dimension < 1000:
                return 0.8
            else:
                return 1.0
        except Exception as e:
            logger.error(f"Error checking resolution: {str(e)}")
            return 0.5
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall image quality score.
        
        Args:
            metrics: Dictionary of individual quality metrics
            
        Returns:
            float: Overall quality score (0-1, higher is better)
        """
        try:
            # Weighted average of individual metrics
            weights = {
                "blur_score": 0.35,
                "contrast_score": 0.30,
                "noise_level": 0.20,  # Inverted: lower is better
                "resolution_adequacy": 0.15
            }
            
            score = (
                weights["blur_score"] * metrics["blur_score"] +
                weights["contrast_score"] * metrics["contrast_score"] +
                weights["noise_level"] * (1.0 - metrics["noise_level"]) +  # Invert noise score
                weights["resolution_adequacy"] * metrics["resolution_adequacy"]
            )
            
            return max(0.0, min(score, 1.0))
        except Exception as e:
            logger.error(f"Error calculating overall score: {str(e)}")
            return 0.5

    def generate_preprocessing_recommendations(self, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate intelligent preprocessing recommendations based on quality metrics.
        
        This method analyzes document quality metrics and provides specific, actionable
        preprocessing recommendations to improve document processing results.
        Each recommendation includes a method, priority, parameters, and expected impact.
        
        Args:
            quality_metrics: Dictionary of quality metrics from analyze_image_quality
            
        Returns:
            Dict: Comprehensive preprocessing recommendations and strategy
        """
        recommendations = {
            "preprocessing_required": quality_metrics["overall_quality"] < 0.7,
            "overall_quality": quality_metrics["overall_quality"],
            "quality_issues": [],
            "recommended_methods": [],
            "preprocessing_strategy": "standard"
        }
        
        # Identify quality issues
        if quality_metrics["blur_score"] < 0.5:
            recommendations["quality_issues"].append({
                "type": "blur",
                "severity": "high" if quality_metrics["blur_score"] < 0.3 else "medium",
                "metric": quality_metrics["blur_score"]
            })
            
        if quality_metrics["contrast_score"] < 0.5:
            recommendations["quality_issues"].append({
                "type": "low_contrast",
                "severity": "high" if quality_metrics["contrast_score"] < 0.3 else "medium",
                "metric": quality_metrics["contrast_score"]
            })
            
        if quality_metrics["noise_level"] > 0.3:
            recommendations["quality_issues"].append({
                "type": "noise",
                "severity": "high" if quality_metrics["noise_level"] > 0.5 else "medium",
                "metric": quality_metrics["noise_level"]
            })
            
        if quality_metrics["resolution_adequacy"] < 0.5:
            recommendations["quality_issues"].append({
                "type": "low_resolution",
                "severity": "high" if quality_metrics["resolution_adequacy"] < 0.3 else "medium",
                "metric": quality_metrics["resolution_adequacy"]
            })
            
        # Determine overall preprocessing strategy
        if quality_metrics["overall_quality"] < 0.3:
            recommendations["preprocessing_strategy"] = "aggressive"
        elif quality_metrics["overall_quality"] < 0.5:
            recommendations["preprocessing_strategy"] = "balanced"
        elif quality_metrics["overall_quality"] < 0.7:
            recommendations["preprocessing_strategy"] = "conservative"
        else:
            recommendations["preprocessing_strategy"] = "minimal"
            
        # Generate method-specific recommendations
        if quality_metrics["blur_score"] < 0.5:
            blur_rec = {
                "method": "deblur",
                "priority": "high" if quality_metrics["blur_score"] < 0.3 else "medium",
                "params": {
                    "strength": min(1.0, 1.0 - quality_metrics["blur_score"]),
                    "method": "wiener" if quality_metrics["blur_score"] < 0.3 else "unsharp_mask"
                },
                "expected_improvement": {
                    "ocr_accuracy": "significant" if quality_metrics["blur_score"] < 0.3 else "moderate",
                    "entity_extraction": "moderate"
                }
            }
            recommendations["recommended_methods"].append(blur_rec)
            
        if quality_metrics["contrast_score"] < 0.5:
            contrast_methods = ["adaptive_histogram_equalization", "clahe"]
            if quality_metrics["contrast_score"] < 0.3:
                contrast_method = "clahe"
                contrast_strength = min(1.0, 1.0 - quality_metrics["contrast_score"])
            else:
                contrast_method = "adaptive_histogram_equalization"
                contrast_strength = min(0.8, 1.0 - quality_metrics["contrast_score"])
                
            contrast_rec = {
                "method": "enhance_contrast",
                "priority": "high" if quality_metrics["contrast_score"] < 0.3 else "medium",
                "params": {
                    "strength": contrast_strength,
                    "algorithm": contrast_method,
                    "local_adaptation": True if quality_metrics["contrast_score"] < 0.3 else False
                },
                "expected_improvement": {
                    "ocr_accuracy": "significant" if quality_metrics["contrast_score"] < 0.3 else "moderate",
                    "entity_extraction": "moderate"
                }
            }
            recommendations["recommended_methods"].append(contrast_rec)
            
        if quality_metrics["noise_level"] > 0.3:
            noise_methods = ["bilateral_filter", "non_local_means", "gaussian"]
            if quality_metrics["noise_level"] > 0.6:
                noise_method = "non_local_means"  # Best for high noise
                noise_strength = min(1.0, quality_metrics["noise_level"])
            elif quality_metrics["noise_level"] > 0.4:
                noise_method = "bilateral_filter"  # Good balance
                noise_strength = min(0.8, quality_metrics["noise_level"])
            else:
                noise_method = "gaussian"  # Gentle noise reduction
                noise_strength = min(0.6, quality_metrics["noise_level"])
                
            noise_rec = {
                "method": "denoise",
                "priority": "high" if quality_metrics["noise_level"] > 0.5 else "medium",
                "params": {
                    "strength": noise_strength,
                    "algorithm": noise_method,
                    "preserve_edges": True if "text" in noise_method else False
                },
                "expected_improvement": {
                    "ocr_accuracy": "significant" if quality_metrics["noise_level"] > 0.5 else "moderate",
                    "entity_extraction": "moderate"
                }
            }
            recommendations["recommended_methods"].append(noise_rec)
            
        if quality_metrics["resolution_adequacy"] < 0.5:
            # Choose upscaling method based on document type and quality
            if quality_metrics["resolution_adequacy"] < 0.3:
                # For very low resolution, use AI-based upscaling
                upscale_method = "sr_cnn"  # Super-resolution CNN
                upscale_factor = 3.0
            else:
                # For moderately low resolution
                upscale_method = "bicubic"
                upscale_factor = 2.0
                
            upscale_rec = {
                "method": "upscale",
                "priority": "high" if quality_metrics["resolution_adequacy"] < 0.3 else "medium",
                "params": {
                    "factor": upscale_factor,
                    "algorithm": upscale_method,
                    "preserve_text_edges": True
                },
                "expected_improvement": {
                    "ocr_accuracy": "significant" if quality_metrics["resolution_adequacy"] < 0.3 else "moderate",
                    "entity_extraction": "moderate"
                }
            }
            recommendations["recommended_methods"].append(upscale_rec)
        
        # Add document-specific preprocessing based on combined issues
        has_multiple_issues = len(recommendations["quality_issues"]) > 1
        is_severely_degraded = quality_metrics["overall_quality"] < 0.3
        
        if has_multiple_issues:
            # When multiple issues exist, add a binarization recommendation
            # which can often help with several problems at once
            binarize_rec = {
                "method": "binarize",
                "priority": "high" if is_severely_degraded else "medium",
                "params": {
                    "algorithm": "adaptive_otsu" if is_severely_degraded else "otsu",
                    "block_size": 15 if is_severely_degraded else 11
                },
                "expected_improvement": {
                    "ocr_accuracy": "significant" if is_severely_degraded else "moderate",
                    "entity_extraction": "significant" if is_severely_degraded else "moderate"
                }
            }
            recommendations["recommended_methods"].append(binarize_rec)
        
        # For severely degraded documents, recommend a multi-stage approach
        if is_severely_degraded:
            recommendations["preprocessing_pipeline"] = {
                "type": "multi_stage",
                "stages": [
                    {"stage": "noise_reduction", "methods": ["denoise"]},
                    {"stage": "enhancement", "methods": ["enhance_contrast", "deblur"]},
                    {"stage": "binarization", "methods": ["binarize"]},
                    {"stage": "resolution", "methods": ["upscale"]}
                ],
                "approach": "iterative",
                "quality_checkpoints": True
            }
        
        # Add OCR engine recommendations based on document quality
        recommendations["ocr_recommendations"] = self._recommend_ocr_settings(quality_metrics)
            
        # Sort methods by priority
        recommendations["recommended_methods"].sort(
            key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]]
        )
        
        return recommendations
        
    def _recommend_ocr_settings(self, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Recommend OCR settings based on document quality.
        
        Diese Methode gibt Empfehlungen für OCR-Engines und deren Einstellungen
        basierend auf den Qualitätsmetriken des Dokuments. Berücksichtigt werden alle
        verfügbaren OCR-Adapter und Spezialfälle wie Formelerkennung, Tabellenerkennung,
        und die Integration mit ColPaLi für Bilder ohne erkennbaren Text.
        
        Args:
            quality_metrics: Document quality metrics
            
        Returns:
            Dict: OCR engine recommendations
        """
        ocr_recommendations = {
            "preferred_engines": [],
            "engine_settings": {},
            "approach": "standard",
            "colpali_integration": False
        }
        
        # Determine overall document quality category
        if quality_metrics["overall_quality"] < 0.3:
            quality_category = "very_low"
        elif quality_metrics["overall_quality"] < 0.5:
            quality_category = "low"
        elif quality_metrics["overall_quality"] < 0.7:
            quality_category = "medium"
        else:
            quality_category = "high"
            
        # Empfehlungen für OCR-Engines basierend auf Qualität und Dokumenttyp
        # Wir berücksichtigen hier alle verfügbaren OCR-Adapter
        
        # 1. Allgemeine OCR-Engines basierend auf Qualität
        if quality_category == "very_low":
            # Für sehr schlechte Qualität sind robustere Engines besser
            ocr_recommendations["preferred_engines"] = ["tesseract", "doctr", "layoutlmv3"]
            ocr_recommendations["approach"] = "conservative"
            ocr_recommendations["colpali_integration"] = True  # ColPaLi als Backup bei sehr schlechter Textqualität
            
            # Engine-spezifische Einstellungen
            ocr_recommendations["engine_settings"] = {
                "tesseract": {
                    "oem": 1,  # LSTM only
                    "psm": 11,  # Sparse text with OSD
                    "config_params": "--dpi 300"
                },
                "doctr": {
                    "model": "db_resnet50",
                    "adjust_thresholds": True
                },
                "layoutlmv3": {
                    "detection_threshold": 0.4,  # Niedrigerer Schwellwert für schwierige Bilder
                    "preserve_layout": True
                }
            }
        elif quality_category == "low":
            ocr_recommendations["preferred_engines"] = ["tesseract", "paddleocr", "easyocr", "layoutlmv3"]
            ocr_recommendations["approach"] = "balanced"
            ocr_recommendations["engine_settings"] = {
                "tesseract": {
                    "oem": 3,  # Default, based on what's available
                    "psm": 6,  # Assume uniform block of text
                },
                "paddleocr": {
                    "use_angle_cls": True,
                    "det_db_thresh": 0.3
                }
            }
        elif quality_category == "medium":
            ocr_recommendations["preferred_engines"] = ["paddleocr", "easyocr", "microsoft_read", "donut"]
            ocr_recommendations["approach"] = "standard"
            ocr_recommendations["engine_settings"] = {
                "paddleocr": {
                    "use_angle_cls": True
                },
                "donut": {
                    "half_precision": True
                }
            }
        else:  # Hohe Qualität
            ocr_recommendations["preferred_engines"] = ["microsoft_read", "paddleocr", "donut", "doctr"]
            ocr_recommendations["approach"] = "aggressive"
            
        # 2. Spezifische Engines für besondere Dokumenttypen
        # Diese Empfehlungen werden im OCRModelSelector weiter verfeinert, basierend auf zusätzlicher Analyse
        
        # Für akademische Dokumente und Formeln
        has_formulas = quality_metrics.get("has_formulas", False)
        if has_formulas:
            # Füge spezielle Formelerkennungs-Engines hinzu
            formula_engines = ["formula_recognition", "nougat"]
            for engine in formula_engines:
                if engine not in ocr_recommendations["preferred_engines"]:
                    ocr_recommendations["preferred_engines"].insert(0, engine)  # An den Anfang stellen
                    
            # Spezifische Einstellungen für Formelerkennungs-Engines
            ocr_recommendations["engine_settings"]["formula_recognition"] = {
                "detection_threshold": 0.6
            }
            ocr_recommendations["engine_settings"]["nougat"] = {
                "math_mode": True,
                "batch_size": 1
            }
            
        # Für Dokumente mit Tabellen
        has_tables = quality_metrics.get("has_tables", False)
        if has_tables:
            # Füge tabellenorientierte Engines hinzu
            table_engines = ["table_extraction", "layoutlmv3"]
            for engine in table_engines:
                if engine not in ocr_recommendations["preferred_engines"]:
                    ocr_recommendations["preferred_engines"].insert(0, engine)
                    
            # Spezifische Einstellungen für Tabellenerkennungs-Engines
            ocr_recommendations["engine_settings"]["table_extraction"] = {
                "detect_borders": True,
                "reconstruct_layout": True
            }
            
        # Für Dokumente mit komplexem Layout
        has_complex_layout = quality_metrics.get("has_complex_layout", False)
        if has_complex_layout:
            # Füge layoutorientierte Engines hinzu
            layout_engines = ["layoutlmv3", "donut", "microsoft_read"]
            for engine in layout_engines:
                if engine not in ocr_recommendations["preferred_engines"]:
                    # Einfügen, aber nicht unbedingt an erste Stelle
                    if engine not in ocr_recommendations["preferred_engines"][:2]:
                        # Nach den ersten beiden Engines einfügen
                        ocr_recommendations["preferred_engines"].insert(2, engine)
                        
        # 3. Berücksichtigung von Bildqualitätsmetriken für spezifische Probleme
        
        # Bei verschwommenen Bildern
        blur_score = quality_metrics.get("blur_score", 0.5)
        if blur_score < 0.4:
            # Füge Engines hinzu, die besser mit Unschärfe umgehen können
            blur_robust_engines = ["microsoft_read", "paddleocr"]
            for engine in blur_robust_engines:
                if engine not in ocr_recommendations["preferred_engines"][:3]:
                    # In die Top 3 einfügen
                    ocr_recommendations["preferred_engines"].insert(2, engine)
                    
        # Bei niedrigem Kontrast
        contrast_score = quality_metrics.get("contrast_score", 0.5)
        if contrast_score < 0.4:
            # Füge Engines hinzu, die besser mit niedrigem Kontrast umgehen können
            contrast_robust_engines = ["easyocr", "layoutlmv3"]
            for engine in contrast_robust_engines:
                if engine not in ocr_recommendations["preferred_engines"][:3]:
                    # In die Top 3 einfügen
                    ocr_recommendations["preferred_engines"].insert(2, engine)
                    
        # Bei verrauschten Bildern
        noise_level = quality_metrics.get("noise_level", 0.3)
        if noise_level > 0.5:
            # Füge Engines hinzu, die besser mit Rauschen umgehen können
            noise_robust_engines = ["doctr", "microsoft_read"]
            for engine in noise_robust_engines:
                if engine not in ocr_recommendations["preferred_engines"][:3]:
                    # In die Top 3 einfügen
                    ocr_recommendations["preferred_engines"].insert(2, engine)
                    
        # Bei niedriger Auflösung
        resolution_adequacy = quality_metrics.get("resolution_adequacy", 0.5)
        if resolution_adequacy < 0.4:
            # Füge Engines hinzu, die besser mit niedriger Auflösung umgehen können
            resolution_robust_engines = ["tesseract", "microsoft_read"]
            for engine in resolution_robust_engines:
                if engine not in ocr_recommendations["preferred_engines"][:3]:
                    # In die Top 3 einfügen
                    ocr_recommendations["preferred_engines"].insert(2, engine)
        
        # 4. ColPaLi-Integration basierend auf Textwahrscheinlichkeit
        # Wenn das Dokument möglicherweise kein oder wenig Text enthält, empfehlen wir ColPaLi
        text_confidence = quality_metrics.get("text_confidence", 0.8)
        if text_confidence < 0.5:
            ocr_recommendations["colpali_integration"] = True
            ocr_recommendations["colpali_priority"] = "high"
            # Wir können trotzdem OCR ausführen, aber mit niedrigerer Priorität
            ocr_recommendations["ocr_priority"] = "low"
        elif text_confidence < 0.7:
            ocr_recommendations["colpali_integration"] = True
            ocr_recommendations["colpali_priority"] = "medium"
            ocr_recommendations["ocr_priority"] = "medium"
        
        # 5. Multi-Engine-Empfehlungen für bestimmte Fälle
        
        # Bei mittlerer Qualität oder gemischten Problemen
        if 0.4 < quality_metrics["overall_quality"] < 0.7 or len(quality_metrics.get("quality_issues", [])) > 1:
            ocr_recommendations["multi_engine"] = True
            
            # Strategie basierend auf Dokumenttyp und Qualitätsproblemen
            if has_complex_layout:
                ocr_recommendations["multi_engine_strategy"] = "region_based"
            elif quality_metrics["overall_quality"] < 0.5:
                ocr_recommendations["multi_engine_strategy"] = "voting_confidence"
            else:
                ocr_recommendations["multi_engine_strategy"] = "voting"
                
        # Bei extrem schlechter Qualität, hybrid-Ansatz empfehlen
        if quality_metrics["overall_quality"] < 0.3:
            ocr_recommendations["hybrid_approach"] = {
                "enabled": True,
                "methods": ["ocr", "colpali", "visual_analysis"],
                "fusion_strategy": "confidence_weighted"
            }
            
        # Entferne Duplikate in der Engine-Liste, während die Reihenfolge beibehalten wird
        unique_engines = []
        for engine in ocr_recommendations["preferred_engines"]:
            if engine not in unique_engines:
                unique_engines.append(engine)
                
        ocr_recommendations["preferred_engines"] = unique_engines[:5]  # Begrenze auf Top 5
        
        return ocr_recommendations 