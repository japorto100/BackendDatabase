"""
Image enhancement utilities for improving image quality before processing.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Union, Tuple, Optional

from .core import convert_to_array, convert_to_pil, get_image_grayscale

logger = logging.getLogger(__name__)

def denoise_image(image: Union[str, Image.Image, np.ndarray], method: str = 'fastNL', strength: int = 10) -> np.ndarray:
    """
    Apply denoising to an image.
    
    Args:
        image: Image in any supported format
        method: Denoising method ('fastNL', 'gaussian', 'bilateral', 'median')
        strength: Denoising strength
        
    Returns:
        Denoised image as numpy array
    """
    gray = get_image_grayscale(image)
    
    try:
        if method == 'fastNL':
            # Non-local means denoising (best quality but slower)
            return cv2.fastNlMeansDenoising(gray, None, strength, 7, 21)
        elif method == 'gaussian':
            # Gaussian blur (fast but may blur details)
            return cv2.GaussianBlur(gray, (5, 5), 0)
        elif method == 'bilateral':
            # Bilateral filter (preserves edges)
            return cv2.bilateralFilter(gray, 9, 75, 75)
        elif method == 'median':
            # Median filter (good for salt-and-pepper noise)
            return cv2.medianBlur(gray, 5)
        else:
            logger.warning(f"Unknown denoising method: {method}, using fastNL")
            return cv2.fastNlMeansDenoising(gray, None, strength, 7, 21)
    except Exception as e:
        logger.warning(f"Error applying denoising: {str(e)}")
        return gray

def binarize_image(image: Union[str, Image.Image, np.ndarray], method: str = 'adaptive', 
                   block_size: int = 11, c: int = 2, threshold: int = 127) -> np.ndarray:
    """
    Convert image to binary (black and white).
    
    Args:
        image: Image in any supported format
        method: Binarization method ('adaptive', 'otsu', 'simple')
        block_size: Block size for adaptive thresholding (odd number)
        c: Constant for adaptive thresholding
        threshold: Threshold value for simple thresholding
        
    Returns:
        Binarized image as numpy array
    """
    gray = get_image_grayscale(image)
    
    try:
        if method == 'adaptive':
            # Adaptive thresholding for better results with uneven lighting
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, c
            )
        elif method == 'otsu':
            # Otsu's method for automatic thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        elif method == 'simple':
            # Simple global thresholding
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            return binary
        else:
            logger.warning(f"Unknown binarization method: {method}, using adaptive")
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, c
            )
    except Exception as e:
        logger.warning(f"Error applying binarization: {str(e)}")
        return gray

def deskew_image(image: Union[str, Image.Image, np.ndarray], max_angle: float = 20.0) -> np.ndarray:
    """
    Correct image skew.
    
    Args:
        image: Image in any supported format
        max_angle: Maximum angle to correct (degrees)
        
    Returns:
        Deskewed image as numpy array
    """
    gray = get_image_grayscale(image)
    
    try:
        # Find edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is None or len(lines) == 0:
            return gray
            
        # Calculate skew angle
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Only consider mostly horizontal or vertical lines
            if theta < np.pi/4 or theta > 3*np.pi/4:
                angles.append(theta)
                
        if not angles:
            return gray
            
        median_angle = np.median(angles)
        
        # Calculate correction angle
        if median_angle < np.pi/4:
            angle = median_angle * 180 / np.pi
        else:
            angle = (median_angle - np.pi/2) * 180 / np.pi
            
        # Limit correction to max_angle
        angle = max(min(angle, max_angle), -max_angle)
            
        # Rotate image
        (h, w) = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    except Exception as e:
        logger.warning(f"Error deskewing image: {str(e)}")
        return gray

def enhance_contrast(image: Union[str, Image.Image, np.ndarray], method: str = 'clahe', 
                     clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Enhance image contrast.
    
    Args:
        image: Image in any supported format
        method: Enhancement method ('clahe', 'histogram_eq', 'pil')
        clip_limit: Clipping limit for CLAHE
        tile_grid_size: Grid size for CLAHE
        
    Returns:
        Contrast-enhanced image
    """
    gray = get_image_grayscale(image)
    
    try:
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(gray)
        elif method == 'histogram_eq':
            # Simple histogram equalization
            return cv2.equalizeHist(gray)
        elif method == 'pil':
            # Use PIL for contrast enhancement
            pil_img = convert_to_pil(gray)
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = enhancer.enhance(2.0)  # Enhance contrast by factor of 2
            return np.array(enhanced)
        else:
            logger.warning(f"Unknown contrast enhancement method: {method}, using CLAHE")
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(gray)
    except Exception as e:
        logger.warning(f"Error enhancing contrast: {str(e)}")
        return gray

def normalize_image(image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Normalize image pixel values.
    
    Args:
        image: Image in any supported format
        
    Returns:
        Normalized image as numpy array
    """
    np_image = convert_to_array(image)
    
    try:
        # Convert to float and normalize to 0-1 range
        normalized = np_image.astype(np.float32) / 255.0
        
        # Standardize (zero mean, unit variance)
        mean = np.mean(normalized)
        std = np.std(normalized)
        if std > 0:
            normalized = (normalized - mean) / std
        
        # Scale back to 0-255 range and convert to uint8
        normalized = np.clip(normalized * 64 + 128, 0, 255).astype(np.uint8)
        
        return normalized
    except Exception as e:
        logger.warning(f"Error normalizing image: {str(e)}")
        return np_image

def sharpen_image(image: Union[str, Image.Image, np.ndarray], method: str = 'unsharp_mask', 
                 strength: float = 1.5) -> np.ndarray:
    """
    Sharpen image details.
    
    Args:
        image: Image in any supported format
        method: Sharpening method ('unsharp_mask', 'laplacian', 'pil')
        strength: Sharpening strength
        
    Returns:
        Sharpened image
    """
    np_image = convert_to_array(image)
    
    try:
        if method == 'unsharp_mask':
            # Unsharp mask sharpening
            blur = cv2.GaussianBlur(np_image, (0, 0), 3)
            return cv2.addWeighted(np_image, 1.5, blur, -0.5, 0)
        elif method == 'laplacian':
            # Laplacian sharpening
            laplacian = cv2.Laplacian(np_image, cv2.CV_64F)
            return np.uint8(np.clip(np_image + strength * laplacian, 0, 255))
        elif method == 'pil':
            # PIL sharpening
            pil_img = convert_to_pil(np_image)
            enhanced = pil_img.filter(ImageFilter.SHARPEN)
            return np.array(enhanced)
        else:
            logger.warning(f"Unknown sharpening method: {method}, using unsharp mask")
            blur = cv2.GaussianBlur(np_image, (0, 0), 3)
            return cv2.addWeighted(np_image, 1.5, blur, -0.5, 0)
    except Exception as e:
        logger.warning(f"Error sharpening image: {str(e)}")
        return np_image

def enhance_for_ocr(image: Union[str, Image.Image, np.ndarray], 
                   denoise: bool = True, 
                   binarize: bool = True, 
                   deskew: bool = True,
                   contrast: bool = True) -> np.ndarray:
    """
    Apply a complete set of enhancements for optimal OCR results.
    
    Args:
        image: Image in any supported format
        denoise: Whether to apply denoising
        binarize: Whether to binarize the image
        deskew: Whether to correct skew
        contrast: Whether to enhance contrast
        
    Returns:
        Enhanced image ready for OCR
    """
    # Load and convert to grayscale
    np_image = get_image_grayscale(image)
    
    # Apply enhancements in optimal order
    if deskew:
        np_image = deskew_image(np_image)
        
    if contrast:
        np_image = enhance_contrast(np_image)
        
    if denoise:
        np_image = denoise_image(np_image)
        
    if binarize:
        np_image = binarize_image(np_image)
        
    return np_image

def enhance_for_tesseract(image, dpi=300):
    """Apply optimal preprocessing for Tesseract OCR"""
    img = denoise_image(image, method='fastNL')
    img = deskew_image(img)
    img = binarize_image(img, method='adaptive')
    return img

def enhance_for_easyocr(image):
    """Preprocessing optimized for EasyOCR"""
    img = denoise_image(image, method='gaussian')
    img = enhance_contrast(img, method='clahe')
    return img

def enhance_for_paddleocr(image):
    """Preprocessing optimized for PaddleOCR"""
    img = denoise_image(image, method='bilateral')
    img = sharpen_image(img)
    img = enhance_contrast(img)
    return img

def enhance_for_donut(image):
    """Preprocessing optimized for Donut document understanding"""
    img = convert_to_array(image)
    img = denoise_image(img, method='fastNL')
    img = enhance_contrast(img, method='clahe')
    return img

def enhance_for_formula_recognition(image):
    """Preprocessing optimized for formula recognition"""
    img = convert_to_array(image)
    img = denoise_image(img, method='gaussian')
    img = binarize_image(img, method='adaptive', block_size=15, c=5)
    img = sharpen_image(img, strength=2.0)
    return img

def enhance_for_microsoft(image):
    """Preprocessing optimized for Microsoft Read API"""
    img = convert_to_array(image)
    img = enhance_contrast(img)
    img = sharpen_image(img, method='unsharp_mask')
    return img

def enhance_for_nougat(image):
    """Preprocessing optimized for Nougat scientific document understanding"""
    img = convert_to_array(image)
    img = denoise_image(img, method='gaussian')
    img = deskew_image(img)
    img = normalize_image(img)
    return img

def enhance_for_table_extraction(image):
    """Preprocessing optimized for table detection and extraction"""
    img = convert_to_array(image)
    img = denoise_image(img)
    img = sharpen_image(img)
    img = binarize_image(img, method='otsu')
    return img

def enhance_for_layoutlmv3(image):
    """Preprocessing optimized for LayoutLMv3"""
    img = convert_to_array(image)
    img = denoise_image(img, method='gaussian')
    img = enhance_contrast(img)
    return img

def enhance_for_doctr(image):
    """Preprocessing optimized for DocTR"""
    img = convert_to_array(image)
    img = denoise_image(img)
    img = deskew_image(img)
    img = enhance_contrast(img)
    return img