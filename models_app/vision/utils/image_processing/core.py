"""
Core image loading and conversion utilities.
"""

import os
import numpy as np
import cv2
from PIL import Image
import logging
from typing import Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)

def load_image(image_path_or_array: Union[str, np.ndarray, Image.Image]) -> Tuple[Optional[Image.Image], Optional[np.ndarray], Optional[str]]:
    """
    Load an image from a file path, numpy array, or PIL Image.
    
    Args:
        image_path_or_array: Image source (file path, numpy array, or PIL Image)
        
    Returns:
        Tuple of (PIL Image, numpy array, original path or None)
    """
    pil_image = None
    np_image = None
    original_path = None
    
    try:
        if isinstance(image_path_or_array, str):
            # Load from file path
            if not os.path.exists(image_path_or_array):
                raise FileNotFoundError(f"Image file not found: {image_path_or_array}")
                
            original_path = image_path_or_array
            pil_image = Image.open(image_path_or_array).convert('RGB')
            np_image = np.array(pil_image)
            
        elif isinstance(image_path_or_array, np.ndarray):
            # Use numpy array directly
            np_image = image_path_or_array
            
            # Convert BGR to RGB if it's a color image from OpenCV
            if len(np_image.shape) == 3 and np_image.shape[2] == 3:
                np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
                
            pil_image = Image.fromarray(np_image)
            
        elif isinstance(image_path_or_array, Image.Image):
            # Use PIL image directly
            pil_image = image_path_or_array
            np_image = np.array(pil_image)
            
        else:
            raise TypeError(f"Unsupported image type: {type(image_path_or_array)}")
            
        return pil_image, np_image, original_path
        
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        return None, None, None

def save_image(image: Union[np.ndarray, Image.Image], output_path: str, format: Optional[str] = None) -> bool:
    """
    Save an image to a file.
    
    Args:
        image: Image as numpy array or PIL Image
        output_path: Path to save the image
        format: Image format (jpg, png, etc.) - auto-detected from extension if None
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if isinstance(image, np.ndarray):
            # Save numpy array with OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert RGB to BGR for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image)
        else:
            # Save PIL Image
            pil_image = convert_to_pil(image)
            pil_image.save(output_path, format=format)
            
        return True
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return False

def convert_to_array(image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Convert various image formats to numpy array.
    
    Args:
        image: Image as file path, PIL Image, or numpy array
        
    Returns:
        Image as numpy array (RGB)
    """
    _, np_image, _ = load_image(image)
    return np_image

def convert_to_pil(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Convert various image formats to PIL Image.
    
    Args:
        image: Image as file path, PIL Image, or numpy array
        
    Returns:
        Image as PIL Image
    """
    pil_image, _, _ = load_image(image)
    return pil_image

def get_image_grayscale(image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Get grayscale version of an image.
    
    Args:
        image: Image in any supported format
        
    Returns:
        Grayscale image as numpy array
    """
    np_image = convert_to_array(image)
    
    if len(np_image.shape) == 2:
        # Already grayscale
        return np_image
        
    return cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)