"""
Visualization utilities for debugging image processing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Any
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .core import convert_to_array, convert_to_pil, save_image

logger = logging.getLogger(__name__)

def draw_bounding_boxes(image, boxes: List[Union[Tuple, Dict]], 
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2,
                       labels: Optional[List[str]] = None,
                       confidences: Optional[List[float]] = None) -> np.ndarray:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: Image in any supported format
        boxes: List of bounding boxes as (x, y, w, h) tuples or dicts with 'bbox' key
        color: RGB color for boxes (default: green)
        thickness: Line thickness
        labels: Optional list of labels for each box
        confidences: Optional list of confidence scores
        
    Returns:
        Image with bounding boxes drawn on it
    """
    try:
        # Convert to numpy array
        np_image = convert_to_array(image).copy()
        
        # Convert color from RGB to BGR (for OpenCV)
        if len(color) == 3:
            color = (color[2], color[1], color[0])
            
        # Process each box
        for i, box in enumerate(boxes):
            # Extract coordinates from tuple or dict
            if isinstance(box, tuple) and len(box) == 4:
                x, y, w, h = box
            elif isinstance(box, dict) and 'bbox' in box:
                x, y, w, h = box['bbox']
            else:
                logger.warning(f"Invalid box format: {box}")
                continue
                
            # Draw the box
            cv2.rectangle(np_image, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)
            
            # Draw label if provided
            if labels is not None and i < len(labels):
                label = labels[i]
                if confidences is not None and i < len(confidences):
                    label = f"{label}: {confidences[i]:.2f}"
                
                # Calculate text position
                text_y = y - 5 if y - 5 > 15 else y + h + 15
                
                # Draw background rectangle for text
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(np_image, (x, text_y - text_size[1] - 5), 
                             (x + text_size[0], text_y), color, -1)
                
                # Draw text
                cv2.putText(np_image, label, (x, text_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        return np_image
        
    except Exception as e:
        logger.error(f"Error drawing bounding boxes: {str(e)}")
        return convert_to_array(image)

def draw_text_regions(image, regions: List[Dict], 
                     highlight_paragraphs: bool = True,
                     region_color: Tuple[int, int, int] = (0, 255, 0),
                     paragraph_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    Draw text regions with optional paragraph grouping.
    
    Args:
        image: Image in any supported format
        regions: List of text region data
        highlight_paragraphs: Whether to group regions into paragraphs
        region_color: Color for individual text regions
        paragraph_color: Color for paragraph boundaries
        
    Returns:
        Image with text regions visualized
    """
    try:
        np_image = convert_to_array(image).copy()
        
        # Draw individual text regions
        for region in regions:
            if isinstance(region, tuple) and len(region) == 4:
                x, y, w, h = region
            elif isinstance(region, dict) and 'bbox' in region:
                x, y, w, h = region['bbox']
            else:
                continue
                
            cv2.rectangle(np_image, (int(x), int(y)), (int(x+w), int(y+h)), 
                         (region_color[2], region_color[1], region_color[0]), 1)
        
        # Group into paragraphs if requested
        if highlight_paragraphs:
            from .detection import detect_paragraphs
            paragraphs = detect_paragraphs(image, regions)
            
            for para in paragraphs:
                x, y, w, h = para['bbox']
                cv2.rectangle(np_image, (int(x), int(y)), (int(x+w), int(y+h)), 
                             (paragraph_color[2], paragraph_color[1], paragraph_color[0]), 2)
                
        return np_image
        
    except Exception as e:
        logger.error(f"Error drawing text regions: {str(e)}")
        return convert_to_array(image)

def visualize_processing_steps(image, steps: List[Tuple[str, np.ndarray]]) -> np.ndarray:
    """
    Create a visualization of multiple processing steps.
    
    Args:
        image: Original image
        steps: List of (step_name, processed_image) tuples
        
    Returns:
        Combined visualization image
    """
    try:
        # Ensure all images are numpy arrays
        orig_image = convert_to_array(image)
        
        # Include original in steps
        all_steps = [("Original", orig_image)] + steps
        n_steps = len(all_steps)
        
        # Determine grid layout
        cols = min(3, n_steps)
        rows = (n_steps + cols - 1) // cols
        
        # Create figure
        plt.figure(figsize=(15, 5 * rows))
        
        # Add each image to the plot
        for i, (name, img) in enumerate(all_steps):
            plt.subplot(rows, cols, i + 1)
            
            # Convert to RGB for matplotlib if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                # OpenCV uses BGR, convert to RGB for matplotlib
                plt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                plt_img = img
                
            plt.imshow(plt_img, cmap='gray' if len(img.shape) == 2 else None)
            plt.title(name)
            plt.axis('off')
            
        plt.tight_layout()
        
        # Convert matplotlib figure to numpy array
        fig = plt.gcf()
        fig.canvas.draw()
        
        # Get the RGB buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 3)
        
        plt.close()
        return buf
        
    except Exception as e:
        logger.error(f"Error visualizing processing steps: {str(e)}")
        return convert_to_array(image)

def create_debug_image(image, detection_results: Dict[str, List], 
                      color_map: Dict[str, Tuple[int, int, int]] = None,
                      show_labels: bool = True,
                      show_confidence: bool = True) -> np.ndarray:
    """
    Create a debug visualization with multiple detection types.
    
    Args:
        image: Source image
        detection_results: Dict mapping detection types to lists of detections
        color_map: Dict mapping detection types to colors
        show_labels: Whether to show detection type labels
        show_confidence: Whether to show confidence scores
        
    Returns:
        Debug visualization image
    """
    try:
        np_image = convert_to_array(image).copy()
        
        # Default color map if not provided
        if color_map is None:
            color_map = {
                'text': (0, 255, 0),     # Green
                'paragraph': (0, 0, 255), # Blue
                'table': (255, 0, 0),     # Red
                'image': (255, 255, 0),   # Yellow
                'formula': (255, 0, 255)  # Magenta
            }
        
        # Process each detection type
        for det_type, detections in detection_results.items():
            color = color_map.get(det_type, (125, 125, 125))  # Default gray
            
            # Convert to BGR for OpenCV
            bgr_color = (color[2], color[1], color[0])
            
            for detection in detections:
                if isinstance(detection, dict) and 'bbox' in detection:
                    x, y, w, h = detection['bbox']
                    confidence = detection.get('confidence', 1.0)
                else:
                    continue
                    
                # Draw rectangle
                cv2.rectangle(np_image, (int(x), int(y)), (int(x+w), int(y+h)), bgr_color, 2)
                
                # Draw label if requested
                if show_labels:
                    label = det_type
                    if show_confidence and 'confidence' in detection:
                        label = f"{label}: {confidence:.2f}"
                        
                    # Calculate text position
                    text_y = y - 5 if y - 5 > 15 else y + h + 15
                    
                    # Draw background rectangle for text
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(np_image, (x, text_y - text_size[1] - 5), 
                                 (x + text_size[0], text_y), bgr_color, -1)
                    
                    # Draw text
                    cv2.putText(np_image, label, (x, text_y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        return np_image
        
    except Exception as e:
        logger.error(f"Error creating debug image: {str(e)}")
        return convert_to_array(image)

def overlay_segmentation_mask(image, mask, alpha: float = 0.5, 
                             colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlay a segmentation mask on an image.
    
    Args:
        image: Source image
        mask: Binary or multi-class segmentation mask
        alpha: Transparency of overlay (0-1)
        colormap: OpenCV colormap to apply to mask
        
    Returns:
        Image with overlaid segmentation mask
    """
    try:
        np_image = convert_to_array(image).copy()
        
        # Ensure mask has proper format
        if mask.max() > 1:
            # Normalize to 0-1 if needed
            mask_norm = mask.astype(np.float32) / 255.0
        else:
            mask_norm = mask.astype(np.float32)
            
        # Convert mask to color visualization
        mask_uint8 = (mask_norm * 255).astype(np.uint8)
        colored_mask = cv2.applyColorMap(mask_uint8, colormap)
        
        # Create overlay
        overlay = cv2.addWeighted(np_image, 1.0, colored_mask, alpha, 0)
        
        return overlay
        
    except Exception as e:
        logger.error(f"Error overlaying segmentation mask: {str(e)}")
        return convert_to_array(image)

def draw_keypoints(image, keypoints, color: Tuple[int, int, int] = (0, 255, 0), 
                  size: int = 5, connections: Optional[List[Tuple[int, int]]] = None,
                  connection_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    Draw keypoints and optional connections on an image.
    
    Args:
        image: Source image
        keypoints: List of (x, y) coordinates
        color: Color for keypoints
        size: Size of keypoint markers
        connections: Optional list of (index1, index2) pairs defining connections
        connection_color: Color for connections
        
    Returns:
        Image with keypoints and connections drawn on it
    """
    try:
        np_image = convert_to_array(image).copy()
        
        # Convert colors to BGR for OpenCV
        bgr_color = (color[2], color[1], color[0])
        bgr_connection = (connection_color[2], connection_color[1], connection_color[0])
        
        # Draw connections first (so they appear behind points)
        if connections is not None:
            for idx1, idx2 in connections:
                if idx1 < len(keypoints) and idx2 < len(keypoints):
                    pt1 = tuple(map(int, keypoints[idx1]))
                    pt2 = tuple(map(int, keypoints[idx2]))
                    cv2.line(np_image, pt1, pt2, bgr_connection, 2)
        
        # Draw keypoints
        for kp in keypoints:
            x, y = map(int, kp)
            cv2.circle(np_image, (x, y), size, bgr_color, -1)
            
        return np_image
        
    except Exception as e:
        logger.error(f"Error drawing keypoints: {str(e)}")
        return convert_to_array(image)