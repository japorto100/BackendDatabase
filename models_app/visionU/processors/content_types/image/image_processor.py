from typing import Dict, Any, Optional, List, ClassVar, Tuple
import logging
from datetime import datetime
import os
from PIL import Image
import numpy as np

from models_app.visionU.core.base.processor_base import ProcessorBase

logger = logging.getLogger(__name__)

class ImageProcessor(ProcessorBase):
    """
    Processor for handling image-based documents.
    Provides capabilities for image analysis, OCR, and visual understanding.
    """
    
    VERSION: ClassVar[str] = "1.0.0"
    SUPPORTED_FORMATS: ClassVar[List[str]] = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    SUPPORTED_CONTENT_TYPES: ClassVar[List[str]] = ["image", "photo", "diagram", "chart"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the image processor with configuration."""
        super().__init__(config or {})
        self._setup_image_processing()
    
    def _setup_image_processing(self) -> None:
        """Set up image processing components."""
        self.image_config = {
            "min_confidence": self.config.get("min_confidence", 0.7),
            "min_resolution": self.config.get("min_resolution", (100, 100)),
            "max_resolution": self.config.get("max_resolution", (4096, 4096)),
            "max_file_size": self.config.get("max_file_size", 10 * 1024 * 1024),  # 10MB
            "color_analysis": self.config.get("color_analysis", True),
            "quality_analysis": self.config.get("quality_analysis", True)
        }
    
    def validate_input(self, document_path: str) -> bool:
        """
        Validate if the input document is suitable for image processing.
        
        Args:
            document_path: Path to the document
            
        Returns:
            bool: True if document is valid for processing
        """
        if not super().validate_input(document_path):
            return False
            
        try:
            # Check file size
            file_size = self._get_file_size(document_path)
            if file_size > self.image_config["max_file_size"]:
                logger.warning(f"File size {file_size} exceeds maximum allowed size")
                return False
            
            # Validate file extension
            if not any(document_path.lower().endswith(ext) for ext in self.SUPPORTED_FORMATS):
                logger.warning(f"Unsupported file format for {document_path}")
                return False
            
            # Check image dimensions
            with Image.open(document_path) as img:
                width, height = img.size
                if not self._validate_dimensions((width, height)):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating document {document_path}: {str(e)}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the image processor.
        
        Returns:
            Dict containing processor capabilities
        """
        return {
            "image_analysis": True,
            "color_analysis": self.image_config["color_analysis"],
            "quality_analysis": self.image_config["quality_analysis"],
            "resolution_range": {
                "min": self.image_config["min_resolution"],
                "max": self.image_config["max_resolution"]
            },
            "supported_formats": self.SUPPORTED_FORMATS,
            "supported_content_types": self.SUPPORTED_CONTENT_TYPES,
            "max_file_size": self.image_config["max_file_size"]
        }
    
    def _process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process an image document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dict containing processed results
        """
        try:
            # Load and analyze image
            with Image.open(document_path) as img:
                # Basic image analysis
                dimensions = img.size
                format_info = img.format
                mode = img.mode
                
                # Color analysis if enabled
                color_info = None
                if self.image_config["color_analysis"]:
                    color_info = self._analyze_colors(img)
                
                # Quality analysis if enabled
                quality_info = None
                if self.image_config["quality_analysis"]:
                    quality_info = self._analyze_quality(img)
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence(img, quality_info)
            
            # Structure the result
            result = {
                "metadata": {
                    "processor": self.__class__.__name__,
                    "version": self.VERSION,
                    "timestamp": datetime.now().isoformat(),
                    "file_path": document_path,
                    "dimensions": dimensions,
                    "format": format_info,
                    "color_mode": mode,
                    "file_size": self._get_file_size(document_path)
                },
                "confidence": confidence_score,
                "analysis": {
                    "color_info": color_info,
                    "quality_info": quality_info
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {str(e)}")
            raise
    
    def _validate_dimensions(self, dimensions: Tuple[int, int]) -> bool:
        """Validate image dimensions against configured limits."""
        width, height = dimensions
        min_width, min_height = self.image_config["min_resolution"]
        max_width, max_height = self.image_config["max_resolution"]
        
        if width < min_width or height < min_height:
            logger.warning(f"Image dimensions {dimensions} below minimum {self.image_config['min_resolution']}")
            return False
            
        if width > max_width or height > max_height:
            logger.warning(f"Image dimensions {dimensions} exceed maximum {self.image_config['max_resolution']}")
            return False
            
        return True
    
    def _analyze_colors(self, img: Image.Image) -> Dict[str, Any]:
        """Analyze color information in the image."""
        try:
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Get color histogram
            histogram = img.histogram()
            
            # Calculate basic color statistics
            pixels = img.width * img.height
            colors_used = sum(1 for count in histogram if count > 0)
            
            return {
                "mode": img.mode,
                "unique_colors": colors_used,
                "color_depth": len(histogram) // 3,  # For RGB
                "pixels": pixels
            }
            
        except Exception as e:
            logger.warning(f"Color analysis failed: {str(e)}")
            return None
    
    def _analyze_quality(self, img: Image.Image) -> Dict[str, Any]:
        """Analyze image quality metrics."""
        try:
            # Convert to grayscale for some calculations
            gray_img = img.convert("L")
            gray_data = np.array(gray_img)
            
            # Calculate basic quality metrics
            contrast = np.std(gray_data)
            brightness = np.mean(gray_data)
            
            return {
                "contrast": float(contrast),
                "brightness": float(brightness),
                "sharpness": self._estimate_sharpness(gray_data),
                "noise_level": self._estimate_noise(gray_data)
            }
            
        except Exception as e:
            logger.warning(f"Quality analysis failed: {str(e)}")
            return None
    
    def _estimate_sharpness(self, gray_data: np.ndarray) -> float:
        """Estimate image sharpness using Laplacian variance."""
        try:
            from scipy.ndimage import laplace
            return float(np.var(laplace(gray_data)))
        except ImportError:
            return 0.0
    
    def _estimate_noise(self, gray_data: np.ndarray) -> float:
        """Estimate image noise level."""
        try:
            # Simple noise estimation using local variance
            from scipy.ndimage import uniform_filter
            mean = uniform_filter(gray_data, size=3)
            noise = np.sqrt(uniform_filter((gray_data - mean)**2, size=3))
            return float(np.mean(noise))
        except ImportError:
            return 0.0
    
    def _calculate_confidence(self, img: Image.Image, quality_info: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score for the processed image."""
        confidence = 0.0
        
        # Basic image checks
        if img.width >= self.image_config["min_resolution"][0] and img.height >= self.image_config["min_resolution"][1]:
            confidence += 0.3
        
        # Format and mode checks
        if img.mode in ["RGB", "RGBA"]:
            confidence += 0.2
        
        # Quality metrics if available
        if quality_info:
            # Normalize quality metrics to 0-1 range
            contrast_score = min(quality_info["contrast"] / 100.0, 1.0) * 0.2
            brightness_score = (1.0 - abs(quality_info["brightness"] - 128) / 128) * 0.2
            sharpness_score = min(quality_info["sharpness"] / 1000.0, 1.0) * 0.1
            
            confidence += contrast_score + brightness_score + sharpness_score
        
        return min(confidence, 1.0) 