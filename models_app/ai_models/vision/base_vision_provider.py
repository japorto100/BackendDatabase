"""
BaseVisionProvider

Base class for all vision model providers, defining the common interface and functionality.
This class serves as the foundation for all vision processing providers and ensures
consistent behavior across different implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import os
from PIL import Image
from io import BytesIO
import base64
import requests
import time
import re
import uuid
import threading
import psutil
import gc
import torch

# Import base model service for inheritance
from models_app.ai_models.utils.common.ai_base_service import BaseModelService

# Import common utilities for error handling and metrics
from models_app.ai_models.utils.common.handlers import handle_vision_errors, handle_model_errors
from models_app.ai_models.utils.common.errors import ModelError, ModelUnavailableError, ImageProcessingError
from models_app.ai_models.utils.common.metrics import get_vision_metrics

# Import our vision utility functions
from models_app.ai_models.utils.vision.image_processing import (
    decode_base64_image, 
    encode_image_to_base64,
    resize_image_for_model, 
    download_and_process_image,
    handle_high_resolution_image,
    support_multiple_images
)

logger = logging.getLogger(__name__)

class BaseVisionProvider(BaseModelService):
    """
    Abstract base class for all vision model providers.
    
    This class inherits from BaseModelService to integrate with the common
    model service architecture while maintaining vision-specific functionality.
    
    All vision providers should inherit from this class and implement its
    abstract methods to ensure a consistent interface across providers.
    
    Key features:
    - Image processing with text prompts
    - Multiple image handling for providers that support it
    - Standardized error handling and classification
    - Consistent metrics collection
    - Resource cleanup
    """
    
    # Define common error patterns for classification
    TRANSIENT_ERROR_PATTERNS = [
        r'timeout', 
        r'connection.*(?:refused|reset|aborted)',
        r'temporarily unavailable',
        r'rate limit',
        r'too many requests',
        r'server (is )?overloaded',
        r'insufficient.*resources',
        r'try again later',
        r'service unavailable',
        r'internal server error',
        r'bad gateway',
        r'gateway timeout'
    ]
    
    PERMANENT_ERROR_PATTERNS = [
        r'authentication failed',
        r'unauthorized',
        r'forbidden',
        r'not found',
        r'invalid.*(?:request|format|parameter|token|key)',
        r'unsupported',
        r'bad request',
        r'access denied',
        r'permission denied',
        r'quota exceeded',
        r'account.*(?:suspended|deactivated|disabled)',
        r'method not allowed',
        r'unsupported media type',
        r'unprocessable entity'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vision provider with configuration.
        
        Args:
            config: Configuration dictionary with provider settings
        """
        super().__init__(config)
        
        # If config is a dict, convert it to VisionConfig
        if isinstance(config, dict):
            self.config = config
        
        # Get model name from config
        self.model_name = config.get('model_name', 'default')
        
        # Supporting multi-image by default is False
        # Each provider should set this to True if it supports multiple images
        self.supports_multiple_images = False
        
        # Flag to track initialization status
        self.initialized = False
        
        # Memory tracking for long-running operations
        self.memory_timeline = []
        self.memory_tracking_enabled = False
        self.memory_sampling_interval = 1.0  # seconds
        self.memory_tracker_thread = None
        self.memory_tracking_lock = threading.Lock()
        
        # Initialize base model service with appropriate parameters
        name = config.get('provider_type', 'vision')
        model_name = config.get('model_name', config.get('model', 'base'))  # Support both for backward compatibility
        super().__init__(name, model_name, config)
        
        # Store vision-specific configuration
        self.max_image_size = config.get('max_image_size', 1024)
        self.processor = None
        self.device = None
        
        # Add memory interface if specified
        self.memory = config.get('memory', None)
        if self.memory is None and config.get('use_memory', False):
            from mem0 import Memory
            self.memory = Memory(config.get('memory_config', {}))
        
        # Initialize metrics collector
        self.metrics = None
        if config.get('enable_metrics', True):
            self.metrics = get_vision_metrics(f"{name}_{model_name}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the vision model and processor.
        
        This method loads the model and processor, and prepares them for use.
        It should be called before using the provider for inference.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        
        Raises:
            ModelUnavailableError: If the model cannot be initialized
        """
        pass
    
    @abstractmethod
    @handle_vision_errors
    def process_image(self, 
                     image: Union[str, Image.Image, bytes], 
                     prompt: str = "Describe this image in detail.", 
                     max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Process an image with the vision model.
        
        Args:
            image: The image to process, can be a file path, URL, PIL image, or bytes
            prompt: The text prompt to guide the model's understanding of the image
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Tuple[str, float]: The generated text and a confidence score
            
        Raises:
            ImageProcessingError: If image processing fails
            ModelError: If model inference fails
        """
        pass
    
    @abstractmethod
    @handle_vision_errors
    def process_multiple_images(self, 
                               images: List[Union[str, Image.Image, bytes]], 
                               prompt: str = "Describe these images in detail.", 
                               max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Process multiple images with the vision model.
        
        Args:
            images: List of images to process
            prompt: The text prompt to guide the model's understanding of the images
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Tuple[str, float]: The generated text and a confidence score
            
        Raises:
            ImageProcessingError: If image processing fails
            ModelError: If model inference fails
        """
        pass
    
    @abstractmethod
    @handle_vision_errors
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate text based on a text-only prompt.
        
        Args:
            prompt: The text prompt
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Tuple[str, float]: The generated text and a confidence score
            
        Raises:
            ModelError: If text generation fails
        """
        pass
    
    @handle_vision_errors
    def generate_batch(self, prompts: List[str], max_tokens: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Generate text for multiple text-only prompts in batch.
        
        Args:
            prompts: List of text prompts
            max_tokens: Maximum number of tokens for each response
            
        Returns:
            List[Tuple[str, float]]: List of generated texts and confidence scores
            
        Raises:
            ModelError: If batch generation fails
        """
        # Start metrics collection if enabled
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("generate_batch")
            
        try:
            results = []
            start_time = time.time()
            for prompt in prompts:
                try:
                    result = self.generate_text(prompt, max_tokens)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    results.append(("Error processing prompt", 0.0))
                    
                    # Record error in metrics
                    if self.metrics:
                        self.metrics.record_vision_error("batch_processing_error", {
                            "error": str(e),
                            "prompt": prompt[:100]  # Include part of the prompt for context
                        })
            
            # Record batch metrics
            if self.metrics:
                batch_time_ms = (time.time() - start_time) * 1000
                self.metrics.record_custom_metric("batch_processing", "prompts_per_batch", len(prompts))
                self.metrics.record_custom_metric("batch_processing", "batch_time_ms", batch_time_ms)
                self.metrics.record_custom_metric("batch_processing", "avg_time_per_prompt_ms", 
                                               batch_time_ms / len(prompts) if prompts else 0)
                self.metrics.stop_operation("generate_batch", op_time, success=True)
                
            return results
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("generate_batch", op_time, success=False)
                self.metrics.record_vision_error("batch_generation_error", {"error": str(e)})
            logger.error(f"Error in batch generation: {str(e)}")
            raise ModelError(f"Failed to generate batch responses: {str(e)}")
    
    def preprocess_image(self, image: Union[str, Image.Image, bytes]) -> Image.Image:
        """
        Preprocess an image for the vision model.
        
        Args:
            image: The image to preprocess, can be a file path, URL, PIL image, or bytes
            
        Returns:
            Image.Image: The preprocessed PIL image
            
        Raises:
            ImageProcessingError: If image preprocessing fails
        """
        # Start metrics collection if enabled
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("preprocess_image")
        
        try:
            # Convert the input to a PIL Image
            if isinstance(image, str):
                if image.startswith(('http://', 'https://')):
                    # It's a URL - use our utility function
                    try:
                        result = download_and_process_image(image)
                        if self.metrics:
                            self.metrics.record_custom_metric("image_processing", "source_type", "url")
                        return result
                    except Exception as e:
                        logger.error(f"Error downloading image from URL: {e}")
                        raise ImageProcessingError(f"Failed to download image from URL: {image}", cause=e)
                elif image.startswith('data:image'):
                    # It's a base64 encoded image - use our utility function
                    try:
                        result = decode_base64_image(image)
                        if self.metrics:
                            self.metrics.record_custom_metric("image_processing", "source_type", "base64")
                        return result
                    except Exception as e:
                        logger.error(f"Error decoding base64 image: {e}")
                        raise ImageProcessingError("Failed to decode base64 image", cause=e)
                else:
                    # It's a file path
                    if not os.path.exists(image):
                        raise ImageProcessingError(f"Image file not found: {image}")
                    result = Image.open(image).convert('RGB')
                    if self.metrics:
                        self.metrics.record_custom_metric("image_processing", "source_type", "file_path")
                    return result
            elif isinstance(image, bytes):
                # It's binary data
                result = Image.open(BytesIO(image)).convert('RGB')
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "bytes")
                return result
            elif isinstance(image, Image.Image):
                # It's already a PIL Image
                result = image.convert('RGB')
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "pil_image")
                return result
            else:
                raise ImageProcessingError(f"Unsupported image type: {type(image)}")
                
            # Stop metrics collection if enabled
            if self.metrics and op_time:
                self.metrics.stop_operation("preprocess_image", op_time, success=True)
                
            return result
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("preprocess_image", op_time, success=False)
                self.metrics.record_vision_error("image_preprocessing_error", {"error": str(e)})
            
            # If it's already an ImageProcessingError, re-raise it
            if isinstance(e, ImageProcessingError):
                raise e
                
            # Otherwise, wrap it in an ImageProcessingError
            raise ImageProcessingError(f"Failed to preprocess image: {str(e)}", cause=e)
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize an image to fit within the maximum dimensions while preserving aspect ratio.
        
        Args:
            image: The PIL image to resize
            
        Returns:
            Image.Image: The resized PIL image
            
        Raises:
            ImageProcessingError: If image resizing fails
        """
        try:
            # Start metrics collection if enabled
            start_time = time.time()
            original_width, original_height = image.size
            original_format = getattr(image, 'format', 'UNKNOWN')
            
            # Use our utility function with the provider's max_image_size
            result = resize_image_for_model(image, (self.max_image_size, self.max_image_size))
            
            # Record metrics for the resized image
            if self.metrics:
                resized_width, resized_height = result.size
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Record detailed image processing metrics
                self.metrics.record_image_processed(
                    image_format=original_format,
                    original_dimensions=(original_width, original_height),
                    processed_dimensions=(resized_width, resized_height),
                    processing_time_ms=processing_time_ms,
                    was_resized=True
                )
                
            return result
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            if self.metrics:
                self.metrics.record_vision_error("image_resize_error", {"error": str(e)})
            raise ImageProcessingError(f"Failed to resize image: {str(e)}", cause=e)
    
    def prepare_high_res_image(self, image: Image.Image) -> Union[Image.Image, List[Image.Image]]:
        """
        Prepare high resolution images based on model capabilities.
        
        Args:
            image: The PIL image to process
            
        Returns:
            Union[Image.Image, List[Image.Image]]: Processed image or list of image tiles
            
        Raises:
            ImageProcessingError: If high-resolution image handling fails
        """
        try:
            # Start metrics collection if enabled
            if self.metrics:
                width, height = image.size
                self.metrics.record_custom_metric("image_processing", "high_res_width", width)
                self.metrics.record_custom_metric("image_processing", "high_res_height", height)
                
            if self.config.get('supports_tiling', False):
                tile_size = self.config.get('tile_size', 512)
                max_tiles = self.config.get('max_tiles', 6)
                result = handle_high_resolution_image(image, "tile", tile_size, max_tiles)
                
                # Record metrics for tiling
                if self.metrics:
                    if isinstance(result, list):
                        self.metrics.record_custom_metric("image_processing", "tiling_method", "tile")
                        self.metrics.record_custom_metric("image_processing", "tile_count", len(result))
                    
                return result
            else:
                result = handle_high_resolution_image(image, "resize", target_size=(self.max_image_size, self.max_image_size))
                
                # Record metrics for resizing
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "tiling_method", "resize")
                    
                return result
        except Exception as e:
            logger.error(f"Error preparing high-resolution image: {e}")
            raise ImageProcessingError(f"Failed to prepare high-resolution image: {str(e)}", cause=e)
    
    def prepare_multiple_images(self, images: List[Union[str, Image.Image, bytes]], max_images: int = 5) -> List[Image.Image]:
        """
        Process and prepare multiple images for models that support multi-image input.
        
        Args:
            images: List of images to process
            max_images: Maximum number of images to process
            
        Returns:
            List[Image.Image]: List of processed PIL images
            
        Raises:
            ImageProcessingError: If multi-image processing fails
        """
        try:
            # Start metrics collection if enabled
            start_time = time.time()
            
            result = support_multiple_images(images, max_images)
            
            # Record metrics for multi-image processing
            if self.metrics:
                processing_time_ms = (time.time() - start_time) * 1000
                self.metrics.record_multi_image_processed(
                    image_count=len(result),
                    processing_time_ms=processing_time_ms
                )
                
            return result
        except Exception as e:
            logger.error(f"Error preparing multiple images: {e}")
            if self.metrics:
                self.metrics.record_vision_error("multi_image_processing_error", {
                    "error": str(e),
                    "image_count": len(images) if isinstance(images, list) else 0
                })
            raise ImageProcessingError(f"Failed to prepare multiple images: {str(e)}", cause=e)
    
    def analyze_image(self, image: Union[str, Image.Image, bytes]) -> Dict[str, Any]:
        """
        Perform basic analysis on an image.
        
        This method extracts basic properties and metadata from an image.
        
        Args:
            image: The image to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing image analysis results
        """
        # Start metrics collection if enabled
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("analyze_image")
            
        try:
            # Preprocess the image
            pil_image = self.preprocess_image(image)
            
            # Extract basic properties
            width, height = pil_image.size
            mode = pil_image.mode
            format = getattr(pil_image, 'format', 'Unknown')
            
            # Calculate aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            
            # Detect if image is grayscale
            is_grayscale = mode == 'L' or (mode == 'RGB' and self._is_grayscale(pil_image))
            
            # Create result dictionary
            result = {
                'width': width,
                'height': height,
                'mode': mode,
                'format': format,
                'aspect_ratio': aspect_ratio,
                'is_grayscale': is_grayscale,
                'size_kb': self._estimate_size_kb(pil_image),
                'success': True
            }
            
            # Stop metrics collection if enabled
            if self.metrics and op_time:
                self.metrics.stop_operation("analyze_image", op_time, success=True)
                # Record image properties
                self.metrics.record_custom_metric("image_analysis", "width", width)
                self.metrics.record_custom_metric("image_analysis", "height", height)
                self.metrics.record_custom_metric("image_analysis", "aspect_ratio", aspect_ratio)
                self.metrics.record_custom_metric("image_analysis", "is_grayscale", int(is_grayscale))
            
            return result
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("analyze_image", op_time, success=False)
                self.metrics.record_vision_error("image_analysis_error", {"error": str(e)})
                
            logger.error(f"Error analyzing image: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def _is_grayscale(self, img: Image.Image) -> bool:
        """Check if an RGB image is grayscale."""
        if img.mode != 'RGB':
            return False
        
        # Sample up to 100 pixels for efficiency
        width, height = img.size
        pixels = []
        for x in range(0, width, max(1, width // 10)):
            for y in range(0, height, max(1, height // 10)):
                if len(pixels) >= 100:
                    break
                pixels.append(img.getpixel((x, y)))
        
        # Check if R, G, and B values are equal (or very close)
        for r, g, b in pixels:
            if abs(r - g) > 5 or abs(r - b) > 5 or abs(g - b) > 5:
                return False
        
        return True
    
    def _estimate_size_kb(self, img: Image.Image) -> float:
        """Estimate the size of the image in KB."""
        width, height = img.size
        bytes_per_pixel = len(img.getbands())
        estimated_bytes = width * height * bytes_per_pixel
        return estimated_bytes / 1024
    
    @handle_vision_errors
    def process_document_with_images(self, 
                                   document: Dict[str, Any], 
                                   query: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document that contains text and images.
        
        Args:
            document: Document containing text and image paths/data
            query: Optional query to guide the processing
            
        Returns:
            Dict[str, Any]: Processing results
            
        Raises:
            ImageProcessingError: If image processing fails
            ModelError: If model inference fails
        """
        # Start metrics collection if enabled
        op_time = None
        start_time = time.time()
        if self.metrics:
            op_time = self.metrics.start_operation("process_document_with_images")
            
        try:
            # Extract text and images from the document
            text = document.get("text", "")
            images = document.get("images", [])
            
            if not images:
                error_msg = "No images found in document"
                # Record error in metrics
                if self.metrics:
                    self.metrics.record_vision_error("document_processing_error", {"error": error_msg})
                
                return {
                    "error": error_msg,
                    "success": False
                }
            
            # Prepare the prompt
            if query:
                prompt = f"Based on the document and images, please answer: {query}\n\nDocument text: {text}"
            else:
                prompt = f"Analyze this document and its images.\n\nDocument text: {text}"
            
            # Process the first image (or more if the model supports it)
            response = ""
            confidence = 0.0
            
            if len(images) == 1 or not self.supports_multiple_images:
                response, confidence = self.process_image(images[0], prompt)
                
                # Record metrics
                if self.metrics:
                    self.metrics.record_custom_metric("document_processing", "images_processed", 1)
            else:
                # Use our utility to prepare multiple images
                processed_images = self.prepare_multiple_images(
                    images[:min(len(images), 5)], 
                    max_images=5
                )
                response, confidence = self.process_multiple_images(processed_images, prompt)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Record metrics for document processing
            if self.metrics:
                self.metrics.record_document_analysis(
                    document_length=len(text),
                    image_count=len(images),
                    processing_time_ms=processing_time_ms
                )
            
            result = {
                "response": response,
                "confidence": confidence,
                "model_used": self.get_model_info().get('name', self.model_name),
                "strategy": "multimodal",
                "images_processed": min(len(images), 5 if self.supports_multiple_images else 1),
                "processing_time_ms": processing_time_ms,
                "success": True
            }
            
            # Stop metrics collection if enabled
            if self.metrics and op_time:
                self.metrics.stop_operation("process_document_with_images", op_time, success=True)
                
            return result
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("process_document_with_images", op_time, success=False)
                self.metrics.record_vision_error("document_processing_error", {"error": str(e)})
                
            logger.error(f"Error processing document with images: {e}")
            return {
                "error": f"Failed to process document: {str(e)}",
                "success": False
            }
    
    def _load_model_impl(self) -> Any:
        """
        Implementation of model loading.
        
        This method bridges the BaseModelService's model loading with the
        vision provider's implementation. Subclasses should override this
        if they want to use BaseModelService's model loading capabilities.
        
        Returns:
            Any: Loaded model object
            
        Raises:
            NotImplementedError: Vision providers must implement their own model loading
        """
        # Default implementation that calls initialize() for backward compatibility
        success = self.initialize()
        if not success:
            raise ModelUnavailableError(f"Failed to initialize vision model: {self.model_name}")
        return self.model
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        This method returns details about the model being used,
        which is useful for logging and client applications.
        
        Returns:
            Dict[str, Any]: Model information including:
                - name: Model name
                - service: Service type
                - provider: Provider name
                - supports_multiple_images: Whether multiple images are supported
                - max_image_size: Maximum supported image dimensions
                - and provider-specific details
        """
        # Default implementation
        return {
            "name": self.model_name,
            "service": self.name,
            "provider": self.__class__.__name__,
            "loaded": self.model is not None,
            "supports_multiple_images": self.supports_multiple_images,
            "max_image_size": self.max_image_size
        }
    
    def classify_error(self, error: Exception) -> Dict[str, Any]:
        """
        Classify an error as either transient or permanent based on error patterns.
        
        Args:
            error: The exception to classify
            
        Returns:
            Dict[str, Any]: Classification result with error_id, is_transient flag, 
                            retry_recommended flag, and error_category
        """
        error_str = str(error).lower()
        error_id = str(uuid.uuid4())[:8]  # Generate a short unique ID for tracking
        
        # Check if error matches any transient patterns
        is_transient = any(re.search(pattern, error_str) for pattern in self.TRANSIENT_ERROR_PATTERNS)
        
        # Check if error matches any permanent patterns
        is_permanent = any(re.search(pattern, error_str) for pattern in self.PERMANENT_ERROR_PATTERNS)
        
        # Determine error category
        if isinstance(error, ModelUnavailableError):
            category = "model_unavailable"
            # Model unavailable could be transient if not explicitly permanent
            is_transient = not is_permanent
        elif isinstance(error, ImageProcessingError):
            category = "image_processing"
            # Image processing errors are usually permanent
            is_transient = False
        elif "memory" in error_str or "out of memory" in error_str or "cuda" in error_str:
            category = "resource_constraint"
            is_transient = True
        elif "timeout" in error_str:
            category = "timeout"
            is_transient = True
        elif "rate" in error_str and ("limit" in error_str or "exceeded" in error_str):
            category = "rate_limit"
            is_transient = True
        elif is_transient:
            category = "transient_service_issue"
        elif is_permanent:
            category = "permanent_service_issue"
        else:
            category = "unknown"
            # Default to assuming transient if we can't determine
            is_transient = True
        
        # Record the error classification in metrics
        if self.metrics:
            self.metrics.record_vision_error(
                category, 
                {
                    "error_id": error_id,
                    "error_message": str(error),
                    "is_transient": is_transient,
                    "provider": self.__class__.__name__
                }
            )
        
        return {
            "error_id": error_id,
            "is_transient": is_transient,
            "retry_recommended": is_transient,
            "error_category": category,
            "error_message": str(error)
        }
    
    def handle_error_with_context(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle an error with additional context, classifying and logging it appropriately.
        
        Args:
            error: The exception to handle
            context: Additional context about the error (e.g., operation being performed)
            
        Returns:
            Dict[str, Any]: Complete error info with classification and context
        """
        # Classify the error
        error_info = self.classify_error(error)
        
        # Add context
        if context:
            error_info.update({"context": context})
        
        # Log with appropriate level based on classification
        if error_info["is_transient"]:
            logger.warning(
                f"Transient error ({error_info['error_category']}): {error_info['error_message']} "
                f"[ID: {error_info['error_id']}]"
            )
        else:
            logger.error(
                f"Permanent error ({error_info['error_category']}): {error_info['error_message']} "
                f"[ID: {error_info['error_id']}]"
            )
        
        return error_info
    
    def cleanup(self, force: bool = False) -> bool:
        """
        Free resources used by the vision provider.
        
        This method should be called when the provider is no longer needed
        to ensure proper resource management, especially for GPU memory.
        
        Args:
            force: If True, force cleanup even if operations might be in progress
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        cleanup_start = time.time()
        
        try:
            # Record cleanup operation start
            if self.metrics:
                self.metrics.record_custom_metric(
                    "resource_management", 
                    "cleanup_triggered", 
                    1, 
                    {"force": force}
                )
            
            # Provider-specific cleanup should be implemented in subclasses
            # The base implementation just handles common resources
            
            # Remove any stored model references that might keep GPU memory
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'to'):
                    try:
                        # Move model to CPU if possible
                        self.model.to('cpu')
                    except Exception as e:
                        logger.warning(f"Failed to move model to CPU during cleanup: {e}")
                
                # Clear model reference
                self.model = None
                
            # Clear processor/tokenizer references
            if hasattr(self, 'processor') and self.processor is not None:
                self.processor = None
                
            # Reset initialization flag
            self.initialized = False
            
            # Try to trigger Python garbage collection
            gc.collect()
            
            # If using PyTorch, empty CUDA cache
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if self.metrics:
                        self.metrics.record_custom_metric(
                            "resource_management", 
                            "gpu_cache_cleared", 
                            1
                        )
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache during cleanup: {e}")
            
            # Record successful cleanup
            cleanup_time_ms = (time.time() - cleanup_start) * 1000
            if self.metrics:
                self.metrics.record_custom_metric(
                    "resource_management", 
                    "cleanup_time_ms", 
                    cleanup_time_ms
                )
                self.metrics.record_model_usage(
                    self.model_name, 
                    {"model_cleaned_up": True}
                )
            
            logger.info(f"Successfully cleaned up resources for {self.__class__.__name__}")
            return True
            
        except Exception as e:
            # Record cleanup failure
            if self.metrics:
                self.metrics.record_vision_error(
                    "cleanup_error", 
                    {"error": str(e), "force": force}
                )
            
            logger.error(f"Failed to clean up resources for {self.__class__.__name__}: {str(e)}")
            return False
    
    def __del__(self):
        """
        Destructor to ensure resources are cleaned up when the object is garbage collected.
        """
        try:
            self.cleanup()
        except Exception as e:
            # Just log, don't raise during garbage collection
            logger.warning(f"Error during auto-cleanup in __del__: {e}")

    # Context manager support for automatic cleanup
    def __enter__(self):
        """
        Context manager entry - allows using the provider in a 'with' statement.
        """
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - ensures cleanup when exiting a 'with' block.
        """
        self.cleanup()
        return False  # Don't suppress exceptions

    def start_memory_tracking(self) -> None:
        """
        Start tracking memory usage over time for long-running operations.
        This creates a background thread that samples memory usage at regular intervals.
        
        The results can be accessed via self.memory_timeline after the operation completes.
        """
        if self.memory_tracking_enabled:
            return  # Already tracking
            
        with self.memory_tracking_lock:
            self.memory_timeline = []
            self.memory_tracking_enabled = True
            
            # Define the tracking function
            def track_memory():
                process = psutil.Process()
                start_time = time.time()
                
                while self.memory_tracking_enabled:
                    # System memory
                    system_memory = psutil.virtual_memory()
                    
                    # Process memory
                    process_memory = process.memory_info()
                    
                    # GPU memory if available
                    gpu_memory = {}
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            gpu_memory[f"gpu_{i}"] = {
                                "allocated": torch.cuda.memory_allocated(i) / (1024 ** 2),  # MB
                                "reserved": torch.cuda.memory_reserved(i) / (1024 ** 2),    # MB
                                "max_allocated": torch.cuda.max_memory_allocated(i) / (1024 ** 2)  # MB
                            }
                    
                    # Collect memory metrics
                    memory_point = {
                        "timestamp": time.time() - start_time,
                        "system": {
                            "total": system_memory.total / (1024 ** 2),  # MB
                            "available": system_memory.available / (1024 ** 2),  # MB
                            "percent": system_memory.percent
                        },
                        "process": {
                            "rss": process_memory.rss / (1024 ** 2),  # MB
                            "vms": process_memory.vms / (1024 ** 2)   # MB
                        },
                        "gpu": gpu_memory,
                        "threads": threading.active_count()
                    }
                    
                    # Add to timeline
                    with self.memory_tracking_lock:
                        self.memory_timeline.append(memory_point)
                    
                    # Collect garbage to ensure accurate readings
                    gc.collect()
                    
                    # Sleep for the sampling interval
                    time.sleep(self.memory_sampling_interval)
            
            # Start tracking thread
            self.memory_tracker_thread = threading.Thread(
                target=track_memory, 
                daemon=True,  # Daemon thread will exit when main thread exits
                name="MemoryTracker"
            )
            self.memory_tracker_thread.start()
            
            if self.metrics:
                self.metrics.record_custom_metric("memory_tracking", "started", 1)
    
    def stop_memory_tracking(self) -> List[Dict[str, Any]]:
        """
        Stop tracking memory usage and return the collected timeline.
        
        Returns:
            List[Dict[str, Any]]: Timeline of memory usage samples
        """
        if not self.memory_tracking_enabled:
            return self.memory_timeline
            
        with self.memory_tracking_lock:
            self.memory_tracking_enabled = False
            
            # Wait for thread to finish if it exists
            if self.memory_tracker_thread and self.memory_tracker_thread.is_alive():
                self.memory_tracker_thread.join(timeout=2.0)
                
            # Clone the timeline to return
            timeline = list(self.memory_timeline)
            
            # Record metrics if available
            if self.metrics and timeline:
                # Record peak memory usage
                process_peak = max(point["process"]["rss"] for point in timeline)
                self.metrics.record_custom_metric("memory_tracking", "peak_process_mb", process_peak)
                
                # Record GPU peak if available
                if timeline[0].get("gpu") and any(timeline[0]["gpu"]):
                    for gpu_id in timeline[0]["gpu"].keys():
                        peak_gpu = max(point["gpu"][gpu_id]["allocated"] for point in timeline)
                        self.metrics.record_custom_metric(
                            "memory_tracking", 
                            f"peak_gpu_{gpu_id}_mb", 
                            peak_gpu
                        )
                
                # Record average usage
                avg_process = sum(point["process"]["rss"] for point in timeline) / len(timeline)
                self.metrics.record_custom_metric("memory_tracking", "avg_process_mb", avg_process)
                
                # Record memory growth (last - first)
                memory_growth = timeline[-1]["process"]["rss"] - timeline[0]["process"]["rss"]
                self.metrics.record_custom_metric("memory_tracking", "memory_growth_mb", memory_growth)
                
                # Record sample count
                self.metrics.record_custom_metric("memory_tracking", "sample_count", len(timeline))
            
            return timeline

    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """
        Get the current memory usage timeline without stopping tracking.
        
        Returns:
            List[Dict[str, Any]]: Current timeline of memory usage samples
        """
        with self.memory_tracking_lock:
            return list(self.memory_timeline)
            
    def detect_memory_bottlenecks(self) -> Dict[str, Any]:
        """
        Analyze the memory timeline to detect potential bottlenecks.
        
        Returns:
            Dict[str, Any]: Analysis results with identified bottlenecks
        """
        with self.memory_tracking_lock:
            if not self.memory_timeline:
                return {"error": "No memory timeline data available"}
                
            timeline = list(self.memory_timeline)
            
        # Calculate memory growth rate
        if len(timeline) < 2:
            return {"error": "Insufficient timeline data for analysis"}
            
        # Analyze process memory
        first_point = timeline[0]
        last_point = timeline[-1]
        duration = last_point["timestamp"] - first_point["timestamp"]
        
        if duration <= 0:
            return {"error": "Invalid timeline duration"}
            
        rss_growth = last_point["process"]["rss"] - first_point["process"]["rss"]
        growth_rate_mb_per_sec = rss_growth / duration
        
        # Detect sudden spikes
        rss_values = [point["process"]["rss"] for point in timeline]
        avg_rss = sum(rss_values) / len(rss_values)
        peak_rss = max(rss_values)
        peak_ratio = peak_rss / avg_rss if avg_rss > 0 else 1.0
        
        # Find the highest spike
        max_spike = 0
        for i in range(1, len(timeline)):
            spike = timeline[i]["process"]["rss"] - timeline[i-1]["process"]["rss"]
            max_spike = max(max_spike, spike)
        
        # GPU analysis if available
        gpu_analysis = {}
        if timeline[0].get("gpu") and any(timeline[0]["gpu"]):
            for gpu_id in timeline[0]["gpu"].keys():
                gpu_values = [point["gpu"][gpu_id]["allocated"] for point in timeline]
                avg_gpu = sum(gpu_values) / len(gpu_values)
                peak_gpu = max(gpu_values)
                gpu_growth = gpu_values[-1] - gpu_values[0]
                
                gpu_analysis[gpu_id] = {
                    "peak_mb": peak_gpu,
                    "average_mb": avg_gpu,
                    "growth_mb": gpu_growth,
                    "growth_rate_mb_per_sec": gpu_growth / duration,
                    "potential_leak": gpu_growth > 100 and gpu_growth / duration > 10
                }
        
        # Determine if we have potential memory issues
        has_memory_leak = growth_rate_mb_per_sec > 10 and duration > 5
        has_spike = peak_ratio > 2.0
        
        result = {
            "duration_seconds": duration,
            "process_memory": {
                "start_mb": first_point["process"]["rss"],
                "end_mb": last_point["process"]["rss"],
                "peak_mb": peak_rss,
                "growth_mb": rss_growth,
                "growth_rate_mb_per_sec": growth_rate_mb_per_sec,
                "max_spike_mb": max_spike,
                "potential_leak": has_memory_leak,
                "has_spike": has_spike
            },
            "gpu_memory": gpu_analysis,
            "system_memory": {
                "start_percent": first_point["system"]["percent"],
                "end_percent": last_point["system"]["percent"],
                "available_end_mb": last_point["system"]["available"]
            },
            "bottlenecks": []
        }
        
        # Identify bottlenecks
        if has_memory_leak:
            result["bottlenecks"].append({
                "type": "memory_leak",
                "severity": "high" if growth_rate_mb_per_sec > 50 else "medium",
                "details": f"Memory growing at {growth_rate_mb_per_sec:.2f} MB/sec"
            })
            
        if has_spike:
            result["bottlenecks"].append({
                "type": "memory_spike",
                "severity": "high" if peak_ratio > 5 else "medium",
                "details": f"Memory spike of {max_spike:.2f} MB detected"
            })
            
        if last_point["system"]["percent"] > 90:
            result["bottlenecks"].append({
                "type": "system_memory_pressure",
                "severity": "high",
                "details": f"System memory usage at {last_point['system']['percent']}%"
            })
            
        for gpu_id, analysis in gpu_analysis.items():
            if analysis.get("potential_leak"):
                result["bottlenecks"].append({
                    "type": "gpu_memory_leak",
                    "severity": "high",
                    "device": gpu_id,
                    "details": f"GPU memory growing at {analysis['growth_rate_mb_per_sec']:.2f} MB/sec"
                })
        
        return result 