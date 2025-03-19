"""
Feature detection utilities for locating elements in document images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Union, Optional, Dict, Any
import logging
from .core import convert_to_array, get_image_grayscale

logger = logging.getLogger(__name__)

def detect_text_regions(image, min_area: int = 100, max_area: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
    """
    Detect potential text regions in an image.
    
    Args:
        image: Image in any supported format
        min_area: Minimum area of text region
        max_area: Maximum area of text region (None = no limit)
        
    Returns:
        List of bounding boxes in format (x, y, width, height)
    """
    try:
        # Convert to grayscale if needed
        gray = get_image_grayscale(image)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find connected components (potential text regions)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        binary = cv2.dilate(binary, kernel, iterations=3)
        binary = cv2.erode(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        boxes = []
        img_h, img_w = gray.shape
        if max_area is None:
            max_area = img_h * img_w * 0.9  # Default: 90% of image area
            
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by area and aspect ratio
            if min_area <= area <= max_area and w > h:
                boxes.append((x, y, w, h))
                
        return boxes
        
    except Exception as e:
        logger.error(f"Error detecting text regions: {str(e)}")
        return []

def detect_tables(image, min_confidence: float = 0.5) -> List[dict]:
    """
    Detect tables in an image.
    
    Args:
        image: Image in any supported format
        min_confidence: Minimum confidence for detected tables
        
    Returns:
        List of detected tables with bounding boxes and confidence
    """
    try:
        # Convert to numpy array
        np_image = convert_to_array(image)
        gray = get_image_grayscale(np_image)
        
        # Method 1: Line detection approach
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Detect lines
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines
        table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Find contours of table regions
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and format results
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate confidence based on line density
            line_density = cv2.countNonZero(table_mask[y:y+h, x:x+w]) / (w * h)
            confidence = min(line_density * 2, 1.0)  # Scale to 0-1
            
            if confidence >= min_confidence:
                tables.append({
                    'bbox': (x, y, w, h),
                    'confidence': float(confidence),
                    'type': 'table'
                })
                
        return tables
        
    except Exception as e:
        logger.error(f"Error detecting tables: {str(e)}")
        return []

def detect_formulas(image, min_confidence: float = 0.5) -> List[dict]:
    """
    Detect mathematical formulas in an image.
    
    Args:
        image: Image in any supported format
        min_confidence: Minimum confidence for detected formulas
        
    Returns:
        List of detected formulas with bounding boxes and confidence
    """
    try:
        # Convert to grayscale
        gray = get_image_grayscale(image)
        
        # Simple approach: Look for regions with high density of mathematical symbols
        # This is a simplified heuristic - more advanced models would be better
        
        # Threshold to binary
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find symbol-like connected components
        connectivity = 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity, cv2.CV_32S)
        
        # Group nearby symbols
        symbol_points = []
        for i in range(1, num_labels):  # Skip background label 0
            if 10 < stats[i, cv2.CC_STAT_AREA] < 500:  # Size filter for symbols
                symbol_points.append((
                    int(centroids[i, 0]),
                    int(centroids[i, 1])
                ))
        
        if not symbol_points:
            return []
            
        # Cluster points to find formula regions (simplified)
        formulas = []
        
        # Convert symbol points to numpy array
        points = np.array(symbol_points, dtype=np.float32)
        
        # Simple clustering: Use distance-based criteria
        min_points = 5
        max_distance = 50
        
        # Process all points
        remaining_points = set(range(len(points)))
        
        while remaining_points:
            # Take first point as seed
            seed_idx = next(iter(remaining_points))
            cluster = {seed_idx}
            remaining_points.remove(seed_idx)
            
            # Find all points within distance
            for idx in list(remaining_points):
                seed_point = points[seed_idx]
                point = points[idx]
                distance = np.sqrt(np.sum((seed_point - point) ** 2))
                
                if distance < max_distance:
                    cluster.add(idx)
                    remaining_points.remove(idx)
            
            # If enough points, consider it a formula
            if len(cluster) >= min_points:
                # Calculate bounding box of cluster
                cluster_points = points[[i for i in cluster]]
                min_x = int(np.min(cluster_points[:, 0]))
                min_y = int(np.min(cluster_points[:, 1]))
                max_x = int(np.max(cluster_points[:, 0]))
                max_y = int(np.max(cluster_points[:, 1]))
                
                # Calculate confidence based on density
                width = max_x - min_x
                height = max_y - min_y
                area = width * height
                density = len(cluster) / area if area > 0 else 0
                confidence = min(density * 10000, 1.0)  # Scale to 0-1
                
                if confidence >= min_confidence:
                    formulas.append({
                        'bbox': (min_x, min_y, width, height),
                        'confidence': float(confidence),
                        'type': 'formula'
                    })
        
        return formulas
        
    except Exception as e:
        logger.error(f"Error detecting formulas: {str(e)}")
        return []

def detect_lines(image, min_length: int = 50) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Detect horizontal and vertical lines in an image.
    
    Args:
        image: Image in any supported format
        min_length: Minimum line length to detect
        
    Returns:
        Tuple of (horizontal_lines, vertical_lines) where each line is (x1, y1, x2, y2)
    """
    try:
        # Convert to grayscale
        gray = get_image_grayscale(image)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=min_length, maxLineGap=10)
        
        if lines is None:
            return [], []
            
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 == 0:  # Vertical line
                angle = 90
            else:
                angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                
            # Classify line as horizontal or vertical
            if angle < 20:  # Near horizontal
                horizontal_lines.append((x1, y1, x2, y2))
            elif angle > 70:  # Near vertical
                vertical_lines.append((x1, y1, x2, y2))
                
        return horizontal_lines, vertical_lines
        
    except Exception as e:
        logger.error(f"Error detecting lines: {str(e)}")
        return [], []

def detect_paragraphs(image, text_regions: Optional[List[Tuple]] = None) -> List[dict]:
    """
    Detect paragraphs in an image by grouping text regions.
    
    Args:
        image: Image in any supported format
        text_regions: Optional list of text regions as (x, y, w, h)
        
    Returns:
        List of paragraph regions as dictionaries with bbox and confidence
    """
    try:
        # Detect text regions if not provided
        if text_regions is None:
            text_regions = detect_text_regions(image)
            
        if not text_regions:
            return []
            
        # Sort text regions by y-coordinate (top to bottom)
        sorted_regions = sorted(text_regions, key=lambda x: x[1])
        
        # Group into paragraphs based on vertical proximity
        paragraphs = []
        current_paragraph = [sorted_regions[0]]
        
        for i in range(1, len(sorted_regions)):
            prev_region = current_paragraph[-1]
            curr_region = sorted_regions[i]
            
            # Get y-coordinates
            prev_bottom = prev_region[1] + prev_region[3]
            curr_top = curr_region[1]
            
            # Check if regions are close enough vertically
            if curr_top - prev_bottom < 20:  # Threshold for line spacing
                current_paragraph.append(curr_region)
            else:
                # End current paragraph and start a new one
                paragraphs.append(current_paragraph)
                current_paragraph = [curr_region]
                
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)
            
        # Convert grouped regions to paragraph bounding boxes
        paragraph_boxes = []
        for para in paragraphs:
            if not para:
                continue
                
            # Calculate bounding box that encompasses all regions in the paragraph
            min_x = min(r[0] for r in para)
            min_y = min(r[1] for r in para)
            max_x = max(r[0] + r[2] for r in para)
            max_y = max(r[1] + r[3] for r in para)
            
            width = max_x - min_x
            height = max_y - min_y
            
            paragraph_boxes.append({
                'bbox': (min_x, min_y, width, height),
                'confidence': 0.9,  # Default confidence
                'type': 'paragraph'
            })
            
        return paragraph_boxes
        
    except Exception as e:
        logger.error(f"Error detecting paragraphs: {str(e)}")
        return []

def detect_images(document_image) -> List[dict]:
    """
    Detect embedded images within a document image.
    
    Args:
        document_image: Image to analyze
        
    Returns:
        List of detected images with bounding boxes and confidence
    """
    try:
        # Convert to array
        np_image = convert_to_array(document_image)
        
        # Convert to grayscale
        gray = get_image_grayscale(np_image)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that might be images
        image_regions = []
        for contour in contours:
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Skip very small regions
            if area < 1000:
                continue
                
            # Check image characteristics - look for regions with:
            # 1. Reasonable aspect ratio (not too thin)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue
                
            # 2. Higher variance (images tend to have more variance than text)
            roi = gray[y:y+h, x:x+w]
            variance = np.var(roi)
            
            # 3. More complex texture than text
            # Simplified texture measure: standard deviation of Sobel derivatives
            sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            texture_complexity = np.std(np.abs(sobel_x) + np.abs(sobel_y))
            
            # Calculate confidence based on these factors
            var_factor = min(variance / 1000, 1.0)
            texture_factor = min(texture_complexity / 50, 1.0)
            confidence = (var_factor + texture_factor) / 2
            
            if confidence > 0.3:  # Minimum confidence threshold
                image_regions.append({
                    'bbox': (x, y, w, h),
                    'confidence': float(confidence),
                    'type': 'image'
                })
                
        return image_regions
        
    except Exception as e:
        logger.error(f"Error detecting images: {str(e)}")
        return []

def detect_form_elements(image, min_confidence: float = 0.5) -> Dict[str, Any]:
    """
    Detect form elements in an image (checkboxes, radio buttons, text fields).
    
    Args:
        image: Image in any supported format
        min_confidence: Minimum confidence for detected elements
        
    Returns:
        Dictionary with detected form elements and their properties
    """
    try:
        # Convert to numpy array
        np_image = convert_to_array(image)
        gray = get_image_grayscale(np_image)
        
        # Initialize result
        result = {
            "has_form_elements": False,
            "form_type": "unknown",
            "elements": [],
            "confidence": 0.0
        }
        
        # Detect checkboxes
        checkboxes = detect_checkboxes(gray, min_confidence)
        
        # Detect radio buttons
        radio_buttons = detect_radio_buttons(gray, min_confidence)
        
        # Detect text fields
        text_fields = detect_text_fields(gray, min_confidence)
        
        # Combine all elements
        elements = checkboxes + radio_buttons + text_fields
        
        # If we found elements, update the result
        if elements:
            result["has_form_elements"] = True
            result["elements"] = elements
            
            # Calculate average confidence
            avg_confidence = sum(e.get("confidence", 0) for e in elements) / len(elements)
            result["confidence"] = avg_confidence
            
            # Determine form type
            result["form_type"] = _determine_form_type(elements)
        
        return result
        
    except Exception as e:
        logger.error(f"Error detecting form elements: {str(e)}")
        return {
            "has_form_elements": False,
            "form_type": "unknown",
            "elements": [],
            "confidence": 0.0
        }

def detect_checkboxes(gray_image: np.ndarray, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """
    Detect checkboxes in a grayscale image.
    
    Args:
        gray_image: Grayscale image as numpy array
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of detected checkboxes with positions and states
    """
    checkboxes = []
    
    try:
        # Apply threshold
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
                    
                    if confidence >= min_confidence:
                        checkboxes.append({
                            "type": "checkbox",
                            "bbox": (x, y, w, h),
                            "checked": is_checked,
                            "confidence": confidence,
                            "attributes": {
                                "fill_ratio": fill_ratio
                            }
                        })
        
        return checkboxes
        
    except Exception as e:
        logger.error(f"Error detecting checkboxes: {str(e)}")
        return []

def detect_radio_buttons(gray_image: np.ndarray, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """
    Detect radio buttons in a grayscale image.
    
    Args:
        gray_image: Grayscale image as numpy array
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of detected radio buttons with positions and states
    """
    radio_buttons = []
    
    try:
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
                    
                    if confidence >= min_confidence:
                        radio_buttons.append({
                            "type": "radio",
                            "bbox": (x, y, w, h),
                            "selected": is_selected,
                            "confidence": confidence,
                            "attributes": {
                                "fill_ratio": fill_ratio,
                                "circularity": circularity
                            }
                        })
        
        return radio_buttons
        
    except Exception as e:
        logger.error(f"Error detecting radio buttons: {str(e)}")
        return []

def detect_text_fields(gray_image: np.ndarray, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """
    Detect text fields in a grayscale image.
    
    Args:
        gray_image: Grayscale image as numpy array
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of detected text fields
    """
    text_fields = []
    
    try:
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
                
                if confidence >= min_confidence:
                    text_fields.append({
                        "type": "text_field",
                        "bbox": (x, y, w, h),
                        "confidence": confidence,
                        "attributes": {
                            "aspect_ratio": aspect_ratio
                        }
                    })
        
        return text_fields
        
    except Exception as e:
        logger.error(f"Error detecting text fields: {str(e)}")
        return []

def _determine_form_type(elements: List[Dict[str, Any]]) -> str:
    """
    Determine the form type based on detected elements.
    
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