"""
Image transformation utilities for manipulating images.
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Tuple, List, Union, Optional

from .core import convert_to_array, convert_to_pil, get_image_grayscale

logger = logging.getLogger(__name__)

def resize_image(image, size: Tuple[int, int], keep_aspect_ratio: bool = True, 
                 interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
    """
    Resize an image to the specified dimensions.
    
    Args:
        image: Image in any supported format
        size: Target size as (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        interpolation: Interpolation method
        
    Returns:
        Resized image as numpy array
    """
    try:
        np_image = convert_to_array(image)
        target_width, target_height = size
        
        if keep_aspect_ratio:
            # Calculate new dimensions preserving aspect ratio
            h, w = np_image.shape[:2]
            aspect_ratio = w / h
            
            if w > h:  # Landscape
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
                if new_height > target_height:
                    new_height = target_height
                    new_width = int(new_height * aspect_ratio)
            else:  # Portrait
                new_height = target_height
                new_width = int(new_height * aspect_ratio)
                if new_width > target_width:
                    new_width = target_width
                    new_height = int(new_width / aspect_ratio)
        else:
            new_width, new_height = target_width, target_height
            
        # Resize the image
        resized = cv2.resize(np_image, (new_width, new_height), interpolation=interpolation)
        
        # If keep_aspect_ratio is True, pad to target size
        if keep_aspect_ratio and (new_width < target_width or new_height < target_height):
            # Create blank image with target size
            padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            if len(np_image.shape) == 2:  # Grayscale
                padded = np.zeros((target_height, target_width), dtype=np.uint8)
                
            # Calculate position for centered image
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # Place resized image in center
            if len(np_image.shape) == 2:  # Grayscale
                padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            else:  # Color
                padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized
                
            return padded
        
        return resized
        
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        np_image = convert_to_array(image)
        return np_image

def rotate_image(image, angle: float, center: Optional[Tuple[int, int]] = None, 
                scale: float = 1.0, border_value: Tuple = (0, 0, 0)) -> np.ndarray:
    """
    Rotate an image by the specified angle.
    
    Args:
        image: Image in any supported format
        angle: Rotation angle in degrees (positive = counterclockwise)
        center: Center of rotation (None = center of image)
        scale: Scaling factor
        border_value: Value used for pixels outside the rotated image
        
    Returns:
        Rotated image as numpy array
    """
    try:
        np_image = convert_to_array(image)
        height, width = np_image.shape[:2]
        
        if center is None:
            center = (width // 2, height // 2)
            
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Apply affine transformation
        rotated = cv2.warpAffine(np_image, rotation_matrix, (width, height), 
                                 borderValue=border_value)
        
        return rotated
        
    except Exception as e:
        logger.error(f"Error rotating image: {str(e)}")
        np_image = convert_to_array(image)
        return np_image

def crop_image(image, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop an image to the specified bounding box.
    
    Args:
        image: Image in any supported format
        bbox: Bounding box as (x, y, width, height) or (x1, y1, x2, y2)
        
    Returns:
        Cropped image as numpy array
    """
    try:
        np_image = convert_to_array(image)
        
        # Handle different bbox formats
        if len(bbox) == 4:
            if bbox[2] > 0 and bbox[3] > 0:  # (x, y, w, h)
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
            else:  # (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox
                
        # Ensure coordinates are within image bounds
        height, width = np_image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1, min(x2, width))
        y2 = max(y1, min(y2, height))
        
        # Crop the image
        cropped = np_image[y1:y2, x1:x2]
        
        return cropped
        
    except Exception as e:
        logger.error(f"Error cropping image: {str(e)}")
        np_image = convert_to_array(image)
        return np_image

def apply_perspective_transform(image, points: List[Tuple[int, int]]) -> np.ndarray:
    """
    Apply perspective transform to an image (e.g., for document deskewing).
    
    Args:
        image: Image in any supported format
        points: Four points defining the quadrilateral to transform
                in order: top-left, top-right, bottom-right, bottom-left
        
    Returns:
        Transformed image as numpy array
    """
    try:
        np_image = convert_to_array(image)
        height, width = np_image.shape[:2]
        
        if len(points) != 4:
            raise ValueError("Four points are required for perspective transform")
            
        # Convert points to numpy array
        pts1 = np.float32(points)
        
        # Compute the width and height of the target image
        # Width is max of top and bottom side lengths
        width_top = np.sqrt(((pts1[1][0] - pts1[0][0]) ** 2) + ((pts1[1][1] - pts1[0][1]) ** 2))
        width_bottom = np.sqrt(((pts1[2][0] - pts1[3][0]) ** 2) + ((pts1[2][1] - pts1[3][1]) ** 2))
        target_width = max(int(width_top), int(width_bottom))
        
        # Height is max of left and right side lengths
        height_left = np.sqrt(((pts1[3][0] - pts1[0][0]) ** 2) + ((pts1[3][1] - pts1[0][1]) ** 2))
        height_right = np.sqrt(((pts1[2][0] - pts1[1][0]) ** 2) + ((pts1[2][1] - pts1[1][1]) ** 2))
        target_height = max(int(height_left), int(height_right))
        
        # Define the target points (rectangle)
        pts2 = np.float32([[0, 0], [target_width, 0], 
                            [target_width, target_height], [0, target_height]])
        
        # Compute the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        # Apply the perspective transformation
        transformed = cv2.warpPerspective(np_image, matrix, (target_width, target_height))
        
        return transformed
        
    except Exception as e:
        logger.error(f"Error applying perspective transform: {str(e)}")
        np_image = convert_to_array(image)
        return np_image

def normalize_image(image, alpha: int = 0, beta: int = 255) -> np.ndarray:
    """
    Normalize image pixel values to the specified range.
    
    Args:
        image: Image in any supported format
        alpha: Lower bound of the output range
        beta: Upper bound of the output range
        
    Returns:
        Normalized image as numpy array
    """
    try:
        np_image = convert_to_array(image)
        
        # Convert to float for normalization
        np_float = np_image.astype(np.float32)
        
        # Get current min and max values
        min_val = np.min(np_float)
        max_val = np.max(np_float)
        
        # Avoid division by zero
        if max_val == min_val:
            return np.ones_like(np_image) * alpha
            
        # Normalize to [0, 1]
        normalized = (np_float - min_val) / (max_val - min_val)
        
        # Scale to target range [alpha, beta]
        normalized = normalized * (beta - alpha) + alpha
        
        # Convert back to original data type
        if np_image.dtype == np.uint8:
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        return normalized
        
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        np_image = convert_to_array(image)
        return np_image