"""
Image analysis utilities for measuring document properties.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Union, Optional, Any
import logging
from scipy import ndimage
from .core import convert_to_array, get_image_grayscale
from .detection import detect_text_regions, detect_tables, detect_formulas, detect_images

logger = logging.getLogger(__name__)

def analyze_image_complexity(image) -> Dict[str, float]:
    """
    Analyze the complexity of an image based on various metrics.
    
    Args:
        image: Image in any supported format
        
    Returns:
        Dictionary of complexity metrics
    """
    try:
        # Convert to grayscale
        gray = get_image_grayscale(image)
        
        # Calculate edge density (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.mean(edge_magnitude) / 255.0
        
        # Calculate texture complexity (GLCM)
        # Simplified texture measure using local variance
        local_variance = ndimage.variance(gray, size=5)
        texture_complexity = np.mean(local_variance) / 255.0
        
        # Calculate histogram entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalize
        entropy = -np.sum(hist * np.log2(hist + 1e-7))  # Add small constant to avoid log(0)
        normalized_entropy = entropy / 8.0  # Max entropy for 8-bit is 8
        
        # Overall complexity score (weighted average)
        complexity_score = (
            0.4 * edge_density + 
            0.3 * texture_complexity + 
            0.3 * normalized_entropy
        )
        
        return {
            'edge_density': float(edge_density),
            'texture_complexity': float(texture_complexity),
            'entropy': float(normalized_entropy),
            'overall_complexity': float(complexity_score)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image complexity: {str(e)}")
        return {
            'edge_density': 0.0,
            'texture_complexity': 0.0,
            'entropy': 0.0,
            'overall_complexity': 0.0
        }

def analyze_image_quality(image) -> Dict[str, float]:
    """
    Analyze the quality of an image.
    
    Args:
        image: Image in any supported format
        
    Returns:
        Dictionary of quality metrics
    """
    try:
        # Convert to grayscale
        gray = get_image_grayscale(image)
        
        # Brightness
        brightness = np.mean(gray) / 255.0
        
        # Contrast
        contrast = np.std(gray) / 255.0
        
        # Blur detection using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian)
        # Normalize blur score (higher = less blurry)
        normalized_blur = min(blur_score / 1000, 1.0)
        
        # Noise estimation using median filter difference
        median_filtered = cv2.medianBlur(gray, 3)
        noise = np.mean(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))) / 255.0
        noise_level = min(noise * 10, 1.0)  # Scale up for better visibility
        
        # Overall quality score (weighted average)
        # Prefer: moderate brightness, higher contrast, less blur, less noise
        brightness_score = 1.0 - 2.0 * abs(brightness - 0.5)  # Penalize if too bright or too dark
        quality_score = (
            0.25 * brightness_score +
            0.25 * contrast +
            0.35 * normalized_blur +
            0.15 * (1.0 - noise_level)
        )
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'blur_level': float(1.0 - normalized_blur),  # Invert so higher = more blur
            'noise_level': float(noise_level),
            'overall_quality': float(quality_score)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image quality: {str(e)}")
        return {
            'brightness': 0.0,
            'contrast': 0.0,
            'blur_level': 1.0,
            'noise_level': 1.0,
            'overall_quality': 0.0
        }

def estimate_text_density(image) -> Dict[str, float]:
    """
    Estimate the density of text in an image.
    
    Args:
        image: Image in any supported format
        
    Returns:
        Dictionary with text density metrics
    """
    try:
        # Detect text regions
        text_regions = detect_text_regions(image)
        
        # Convert to array for dimensions
        np_image = convert_to_array(image)
        image_height, image_width = np_image.shape[:2]
        image_area = image_height * image_width
        
        # Calculate total text area
        text_area = 0
        for x, y, w, h in text_regions:
            text_area += w * h
            
        # Calculate density metrics
        text_coverage = text_area / image_area if image_area > 0 else 0
        region_count = len(text_regions)
        region_density = region_count / (image_area / 100000)  # Normalized by 100k pixels
        
        # Characterize density
        if text_coverage < 0.1:
            density_category = "sparse"
        elif text_coverage < 0.3:
            density_category = "medium"
        else:
            density_category = "dense"
            
        return {
            'text_coverage': float(text_coverage),
            'text_region_count': region_count,
            'region_density': float(region_density),
            'density_category': density_category
        }
        
    except Exception as e:
        logger.error(f"Error estimating text density: {str(e)}")
        return {
            'text_coverage': 0.0,
            'text_region_count': 0,
            'region_density': 0.0,
            'density_category': "unknown"
        }

def detect_image_content_type(image) -> Dict[str, Any]:
    """
    Analyze image to detect the type of content (text, table, formula, mixed).
    
    Args:
        image: Image in any supported format
        
    Returns:
        Dictionary with content type analysis
    """
    try:
        # Detect various content elements
        text_regions = detect_text_regions(image)
        tables = detect_tables(image)
        formulas = detect_formulas(image)
        images = detect_images(image)
        
        # Convert to array for dimensions
        np_image = convert_to_array(image)
        image_height, image_width = np_image.shape[:2]
        image_area = image_height * image_width
        
        # Calculate area coverage for each element type
        text_area = sum(w * h for x, y, w, h in text_regions)
        table_area = sum(bbox[2] * bbox[3] for item in tables for bbox in [item['bbox']])
        formula_area = sum(bbox[2] * bbox[3] for item in formulas for bbox in [item['bbox']])
        image_area_coverage = sum(bbox[2] * bbox[3] for item in images for bbox in [item['bbox']])
        
        # Calculate coverage percentages
        total_area = image_area if image_area > 0 else 1
        text_coverage = text_area / total_area
        table_coverage = table_area / total_area
        formula_coverage = formula_area / total_area
        image_coverage = image_area_coverage / total_area
        
        # Determine content classification
        content_types = []
        if text_coverage > 0.1:
            content_types.append("text")
        if table_coverage > 0.05:
            content_types.append("table")
        if formula_coverage > 0.05:
            content_types.append("formula")
        if image_coverage > 0.1:
            content_types.append("image")
            
        if not content_types:
            content_types = ["unknown"]
            
        # Determine primary type (highest coverage)
        coverage_map = {
            "text": text_coverage,
            "table": table_coverage,
            "formula": formula_coverage,
            "image": image_coverage
        }
        
        primary_type = max(coverage_map.items(), key=lambda x: x[1])[0] if coverage_map else "unknown"
        
        # Determine complexity
        is_complex = len(content_types) > 1 or primary_type in ["table", "formula"]
        
        return {
            'content_types': content_types,
            'primary_type': primary_type,
            'is_complex': is_complex,
            'coverage': {
                'text': float(text_coverage),
                'table': float(table_coverage),
                'formula': float(formula_coverage),
                'image': float(image_coverage)
            },
            'counts': {
                'text_regions': len(text_regions),
                'tables': len(tables),
                'formulas': len(formulas),
                'images': len(images)
            }
        }
        
    except Exception as e:
        logger.error(f"Error detecting image content type: {str(e)}")
        return {
            'content_types': ["unknown"],
            'primary_type': "unknown",
            'is_complex': False,
            'coverage': {
                'text': 0.0,
                'table': 0.0,
                'formula': 0.0,
                'image': 0.0
            },
            'counts': {
                'text_regions': 0,
                'tables': 0,
                'formulas': 0,
                'images': 0
            }
        }