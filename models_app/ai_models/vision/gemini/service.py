"""
GeminiVisionService

Provides image processing services using Google's Gemini Vision models.
"""

import os
import logging
import time
import base64
from PIL import Image
from typing import Dict, List, Any, Optional, Union, Tuple
from io import BytesIO
import requests

from models_app.ai_models.vision.base_vision_provider import BaseVisionProvider
from models_app.ai_models.vision.gemini.model_manager import GeminiVisionModelManager

# Import common utilities
from models_app.ai_models.utils.common.config import VisionConfig, get_vision_config
from models_app.ai_models.utils.common.handlers import (
    handle_vision_errors, 
    handle_image_processing_errors, 
    handle_multi_image_processing_errors
)
from models_app.ai_models.utils.common.errors import (
    VisionModelError, 
    ImageProcessingError,
    ModelUnavailableError
)

logger = logging.getLogger(__name__)

class GeminiVisionService(BaseVisionProvider):
    """
    Service for processing images with Google's Gemini Vision models.
    
    This service provides:
    1. Image understanding and captioning
    2. Visual question answering
    3. Processing of multiple images with context
    
    Thread Safety:
    This provider is NOT thread-safe for concurrent initialization or model operations.
    Each thread should use its own instance of GeminiVisionService to avoid race conditions
    with model state and API request handling. The _prepare_image method is thread-safe
    and can be used concurrently.
    
    Resource Usage:
    - API Usage: This provider uses Google's Gemini API and requires valid API credentials
    - Memory Usage: Moderate local memory usage for image preparation and response processing
    - Network Requirements: Requires stable internet connection for API communication
    
    Performance Characteristics:
    - Response Time: Typically 1-3 seconds for standard image processing
    - Throughput: Limited by API rate limits (see Google Cloud documentation)
    - Best for: Complex visual reasoning tasks and multi-image understanding
    
    Limitations:
    - Cannot process images larger than 20MB
    - Maximum context length limited by Gemini model specifications
    - Rate limited according to your Google Cloud quota
    - Response quality may vary based on image clarity and prompt specificity
    - Images must be in JPG, PNG, WEBP, HEIF, or GIF format
    
    Example Usage:
    ```python
    from models_app.ai_models.vision.gemini.service import GeminiVisionService
    
    # Initialize service
    service = GeminiVisionService({"model_name": "gemini-pro-vision"})
    service.initialize()
    
    # Process single image
    result, confidence = service.process_image(
        "path/to/image.jpg", 
        prompt="What objects are in this image?"
    )
    
    # Process multiple images
    images = ["image1.jpg", "image2.jpg"]
    result, confidence = service.process_multiple_images(
        images, 
        prompt="Compare these two images."
    )
    ```
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Gemini Vision service.
        
        Args:
            config: Configuration dictionary for model selection and parameters
        """
        super().__init__(config or {})
        
        # Convert raw config dict to VisionConfig if needed
        if not isinstance(self.config, VisionConfig):
            config_name = f"gemini_{self.model_name}"
            vision_config = get_vision_config(config_name)
            
            # Update config with values from the raw config
            for key, value in config.items():
                if hasattr(vision_config, key):
                    setattr(vision_config, key, value)
            
            self.config = vision_config
        
        # Create model manager
        self.model_manager = GeminiVisionModelManager(self.config)
        
        # Configure generation parameters
        self.max_tokens = self.config.max_tokens
        self.temperature = self.config.temperature
        self.top_p = getattr(self.config, 'top_p', 0.9)
        
        # Flag for initialization
        self.initialized = False
        self.model = None
        
        # Support for multiple images (Gemini supports multiple images)
        self.supports_multiple_images = True
    
    @handle_vision_errors
    def initialize(self) -> bool:
        """
        Initialize the Gemini Vision service.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
            
        Raises:
            ModelUnavailableError: If the model cannot be initialized
        """
        if self.initialized and self.model is not None:
            return True
        
        try:
            # Start metrics collection
            op_time = None
            if self.metrics:
                op_time = self.metrics.start_operation("initialize")
            
            # Check if model is available
            start_time = time.time()
            if not self.model_manager.is_available():
                if self.metrics and op_time:
                    self.metrics.stop_operation("initialize", op_time, success=False)
                    self.metrics.record_vision_error(
                        "model_unavailable", 
                        {"model": self.model_manager.model_name}
                    )
                logger.error(f"Model {self.model_manager.model_name} is not available")
                raise ModelUnavailableError(
                    f"Model {self.model_manager.model_name} is not available",
                    model_name=self.model_manager.model_name
                )
            
            # Initialize the model
            self.model = self.model_manager.initialize_model()
            self.initialized = True
            
            # Record metrics
            if self.metrics and op_time:
                init_time_ms = (time.time() - start_time) * 1000
                self.metrics.record_custom_metric(
                    "initialization", 
                    "initialization_time_ms", 
                    init_time_ms
                )
                self.metrics.record_model_usage(
                    self.model_name, 
                    {"model_initialized": True}
                )
                self.metrics.stop_operation("initialize", op_time, success=True)
            
            logger.info(f"Successfully initialized Gemini Vision service for model: {self.model_manager.model_name}")
            return True
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("initialize", op_time, success=False)
                self.metrics.record_vision_error("initialization_error", {"error": str(e)})
            
            logger.error(f"Failed to initialize Gemini Vision service: {str(e)}")
            raise ModelUnavailableError(
                f"Failed to initialize Gemini Vision model: {str(e)}",
                model_name=self.model_name
            )
    
    @handle_image_processing_errors
    def _prepare_image(self, image: Union[str, Image.Image, bytes]) -> Union[Image.Image, bytes]:
        """
        Prepare an image for the Gemini model.
        
        Args:
            image: The image to prepare (path, URL, PIL image, or bytes)
            
        Returns:
            Union[Image.Image, bytes]: Prepared image
            
        Raises:
            ImageProcessingError: If image preparation fails
        """
        # Start metrics collection if available
        start_time = time.time()
        
        try:
            # Handle different input types
            if isinstance(image, str):
                if image.startswith(('http://', 'https://')):
                    # It's a URL
                    try:
                        response = requests.get(image, stream=True, timeout=10)
                        response.raise_for_status()
                        result = Image.open(BytesIO(response.content))
                        if self.metrics:
                            self.metrics.record_custom_metric("image_processing", "source_type", "url")
                        return result
                    except Exception as e:
                        logger.error(f"Failed to load image from URL: {e}")
                        raise ImageProcessingError(f"Failed to load image from URL: {str(e)}", cause=e)
                
                elif image.startswith('data:image'):
                    # It's a base64 string
                    try:
                        # Extract the base64 part (after comma)
                        base64_data = image.split(',')[1]
                        image_data = base64.b64decode(base64_data)
                        result = Image.open(BytesIO(image_data))
                        if self.metrics:
                            self.metrics.record_custom_metric("image_processing", "source_type", "base64")
                        return result
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {e}")
                        raise ImageProcessingError(f"Failed to decode base64 image: {str(e)}", cause=e)
                
                elif os.path.exists(image):
                    # It's a file path
                    result = Image.open(image)
                    if self.metrics:
                        self.metrics.record_custom_metric("image_processing", "source_type", "file_path")
                    return result
                else:
                    raise ImageProcessingError(f"Image path does not exist: {image}")
            
            elif isinstance(image, bytes):
                # It's bytes data
                result = Image.open(BytesIO(image))
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "bytes")
                return result
            
            elif isinstance(image, Image.Image):
                # It's already a PIL Image
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "pil_image")
                return image
            
            else:
                raise ImageProcessingError(f"Unsupported image type: {type(image)}")
        
        except Exception as e:
            # If it's already an ImageProcessingError, re-raise it
            if isinstance(e, ImageProcessingError):
                raise e
            
            # Otherwise, wrap it
            logger.error(f"Error preparing image: {e}")
            raise ImageProcessingError(f"Failed to prepare image: {str(e)}", cause=e)
        
        finally:
            # Record timing for image preparation if metrics available
            if self.metrics:
                prep_time_ms = (time.time() - start_time) * 1000
                self.metrics.record_custom_metric("image_processing", "preparation_time_ms", prep_time_ms)
    
    @handle_vision_errors
    def process_image(self, 
                     image: Union[str, Image.Image, bytes], 
                     prompt: str = "Describe this image in detail.", 
                     max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Process an image with Gemini Vision.
        
        Args:
            image: The image to process (path, URL, PIL image, or bytes)
            prompt: The text prompt to guide processing of the image
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Tuple[str, float]: Generated text and a confidence score
            
        Raises:
            ImageProcessingError: If image processing fails
            VisionModelError: If model inference fails
        """
        # Start metrics collection
        start_time = time.time()
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("process_image")
        
        try:
            # Initialize if needed
            if not self.initialized:
                self.initialize()
            
            # Check if model is loaded
            if not self.model:
                raise VisionModelError("Model not initialized")
            
            # Use specified max_tokens if provided, otherwise use default
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Prepare the image
            processed_image = self._prepare_image(image)
            
            # Record image dimensions for metrics
            if self.metrics and isinstance(processed_image, Image.Image):
                width, height = processed_image.size
                image_format = getattr(processed_image, 'format', 'UNKNOWN')
                self.metrics.record_image_processed(
                    image_format=image_format,
                    original_dimensions=(width, height),
                    processed_dimensions=(width, height),
                    processing_time_ms=(time.time() - start_time) * 1000,
                    was_resized=False
                )
            
            # Process the image with Gemini
            generation_start = time.time()
            response = self.model.generate_content(
                [processed_image, prompt],
                generation_config={
                    "max_output_tokens": tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
            )
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Extract the response text
            response_text = response.text
            
            # Calculate a confidence score
            # Gemini doesn't provide confidence scores, so we estimate based on model internals
            if hasattr(response, 'candidates') and response.candidates:
                confidence = response.candidates[0].safety_ratings[0].probability if hasattr(response.candidates[0], 'safety_ratings') else 0.85
            else:
                confidence = 0.85
            
            # Record metrics
            if self.metrics:
                # Record total processing time
                total_time_ms = (time.time() - start_time) * 1000
                
                # Record inference metrics
                self.metrics.record_inference(
                    inference_time_ms=generation_time_ms,
                    confidence=confidence
                )
                
                # Record model usage
                self.metrics.record_model_usage(self.model_name, {
                    "prompt_length": len(prompt),
                    "temperature": self.temperature,
                    "max_tokens": tokens,
                    "response_length": len(response_text)
                })
                
                # Stop operation timing
                if op_time:
                    self.metrics.stop_operation("process_image", op_time, success=True)
            
            return response_text, confidence
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("process_image", op_time, success=False)
                self.metrics.record_vision_error("image_processing_error", {"error": str(e)})
            
            logger.error(f"Error processing image with Gemini Vision: {e}")
            raise VisionModelError(f"Error processing image with Gemini Vision: {str(e)}", cause=e)
    
    @handle_multi_image_processing_errors
    def process_multiple_images(self, 
                               images: List[Union[str, Image.Image, bytes]], 
                               prompt: str = "Describe these images in detail.", 
                               max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Process multiple images with Gemini Vision.
        
        Args:
            images: List of images to process
            prompt: The text prompt to guide processing of the images
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Tuple[str, float]: Generated text and a confidence score
            
        Raises:
            ImageProcessingError: If image processing fails
            VisionModelError: If model inference fails
        """
        # Start metrics collection
        start_time = time.time()
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("process_multiple_images")
        
        try:
            # Initialize if needed
            if not self.initialized:
                self.initialize()
            
            # Check if model is loaded
            if not self.model:
                raise VisionModelError("Model not initialized")
            
            # Use specified max_tokens if provided, otherwise use default
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Prepare all images
            preprocessing_start = time.time()
            processed_images = [self._prepare_image(img) for img in images]
            
            # Record multi-image processing
            preprocessing_time_ms = (time.time() - preprocessing_start) * 1000
            if self.metrics:
                self.metrics.record_multi_image_processed(
                    image_count=len(processed_images),
                    processing_time_ms=preprocessing_time_ms
                )
            
            # Create input content for the model
            # Gemini API takes a list of content parts
            content_parts = processed_images + [prompt]
            
            # Process the images with Gemini
            generation_start = time.time()
            response = self.model.generate_content(
                content_parts,
                generation_config={
                    "max_output_tokens": tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
            )
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Extract the response text
            response_text = response.text
            
            # Calculate a confidence score
            # For multiple images, we reduce the confidence slightly due to added complexity
            if hasattr(response, 'candidates') and response.candidates:
                confidence = response.candidates[0].safety_ratings[0].probability if hasattr(response.candidates[0], 'safety_ratings') else 0.8
            else:
                confidence = 0.8
            
            # Record metrics
            if self.metrics:
                # Record total processing time
                total_time_ms = (time.time() - start_time) * 1000
                
                # Record inference metrics
                self.metrics.record_inference(
                    inference_time_ms=generation_time_ms,
                    confidence=confidence
                )
                
                # Record model usage
                self.metrics.record_model_usage(self.model_name, {
                    "prompt_length": len(prompt),
                    "temperature": self.temperature,
                    "max_tokens": tokens,
                    "image_count": len(images),
                    "response_length": len(response_text)
                })
                
                # Stop operation timing
                if op_time:
                    self.metrics.stop_operation("process_multiple_images", op_time, success=True)
            
            return response_text, confidence
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("process_multiple_images", op_time, success=False)
                self.metrics.record_vision_error("multi_image_processing_error", {
                    "error": str(e),
                    "image_count": len(images) if isinstance(images, list) else 0
                })
            
            logger.error(f"Error processing multiple images with Gemini Vision: {e}")
            raise VisionModelError(f"Error processing multiple images: {str(e)}", cause=e)
    
    @handle_vision_errors
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate text based on a text-only prompt using Gemini.
        
        Args:
            prompt: The text prompt
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Tuple[str, float]: Generated text and a confidence score
            
        Raises:
            VisionModelError: If text generation fails
        """
        # Start metrics collection
        start_time = time.time()
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("generate_text")
        
        try:
            # Initialize if needed
            if not self.initialized:
                self.initialize()
            
            # Check if model is loaded
            if not self.model:
                raise VisionModelError("Model not initialized")
            
            # Use specified max_tokens if provided, otherwise use default
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Process with Gemini
            generation_start = time.time()
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p
                }
            )
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Extract the response text
            response_text = response.text
            
            # For text-only prompts, we generally assign higher confidence
            if hasattr(response, 'candidates') and response.candidates:
                confidence = response.candidates[0].safety_ratings[0].probability if hasattr(response.candidates[0], 'safety_ratings') else 0.9
            else:
                confidence = 0.9
            
            # Record metrics
            if self.metrics:
                # Record total processing time
                total_time_ms = (time.time() - start_time) * 1000
                
                # Record inference metrics
                self.metrics.record_inference(
                    inference_time_ms=generation_time_ms,
                    confidence=confidence
                )
                
                # Record model usage
                self.metrics.record_model_usage(self.model_name, {
                    "prompt_length": len(prompt),
                    "temperature": self.temperature,
                    "max_tokens": tokens,
                    "text_only": True,
                    "response_length": len(response_text)
                })
                
                # Stop operation timing
                if op_time:
                    self.metrics.stop_operation("generate_text", op_time, success=True)
            
            return response_text, confidence
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("generate_text", op_time, success=False)
                self.metrics.record_vision_error("text_generation_error", {"error": str(e)})
            
            logger.error(f"Error generating text with Gemini: {e}")
            raise VisionModelError(f"Error generating text: {str(e)}", cause=e)
    
    def list_speakers(self) -> List[str]:
        """Return a list of available speakers for the vision model."""
        return []  # Gemini Vision models don't have speakers
    
    def list_languages(self) -> List[str]:
        """Return a list of supported languages for the Gemini Vision model."""
        # Gemini models support multiple languages
        return ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "hi", "ar"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Gemini Vision model.
        
        Returns:
            Dict[str, Any]: Model information and capabilities
        """
        # Start metrics collection
        if self.metrics:
            op_time = self.metrics.start_operation("get_model_info")
        
        try:
            # Get information from model manager
            model_info = self.model_manager.get_model_info()
            
            # Add service-specific information
            model_info.update({
                "service_initialized": self.initialized,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "supports_multiple_images": self.supports_multiple_images,
            })
            
            # Stop metrics collection
            if self.metrics and op_time:
                self.metrics.stop_operation("get_model_info", op_time, success=True)
            
            return model_info
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("get_model_info", op_time, success=False)
                self.metrics.record_vision_error("info_retrieval_error", {"error": str(e)})
            
            logger.error(f"Error getting model info: {e}")
            return {
                "name": self.model_name,
                "error": str(e),
                "service": "gemini_vision",
                "provider": self.__class__.__name__
            } 