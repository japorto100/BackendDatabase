"""
GPT4VisionService

Provides image processing services using OpenAI's GPT-4 Vision models.
"""

import os
import logging
import base64
import time
import json
from PIL import Image
from typing import Dict, List, Any, Optional, Union, Tuple
from io import BytesIO
import requests

from models_app.ai_models.vision.base_vision_provider import BaseVisionProvider
from models_app.ai_models.vision.gpt4v.model_manager import GPT4VisionModelManager

# Import common utilities for vision services
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

class GPT4VisionService(BaseVisionProvider):
    """
    Service for processing images with OpenAI's GPT-4 Vision models.
    
    This service provides:
    1. Image understanding and captioning
    2. Visual question answering
    3. Processing of multiple images with context
    4. High quality image analysis with detail control
    
    Thread Safety:
    This provider is thread-safe for most operations as it uses the stateless OpenAI API client.
    However, the initialization method should not be called concurrently from multiple threads.
    Once initialized, the service can handle concurrent requests safely as the underlying
    client uses separate connections for each request.
    
    Resource Usage:
    - API Usage: Requires valid OpenAI API key with GPT-4 Vision access
    - Token Consumption: Uses OpenAI tokens for both input (images+text) and output
    - Network Requirements: Requires stable internet connection for API communication
    - Cost Considerations: API calls are billed based on input/output tokens and detail level
    
    Performance Characteristics:
    - Response Time: Typically 2-6 seconds for standard detail level
    - Response Quality: Generally highest quality among vision providers
    - Detail Levels: Supports 'low', 'high', and 'auto' detail settings
    - Throughput: Limited by OpenAI API rate limits (varies by account tier)
    - Best for: Complex visual reasoning and high-quality analysis
    
    Limitations:
    - Maximum 5 images per request (technically supports more but not recommended)
    - Token limits apply to combined input (images + text prompt)
    - High detail images consume significantly more tokens
    - Rate limited according to your OpenAI account tier
    - Image dimensions have recommended limits (see OpenAI documentation)
    - Poor internet connection can lead to timeout errors
    
    Example Usage:
    ```python
    from models_app.ai_models.vision.gpt4v.service import GPT4VisionService
    
    # Initialize service
    service = GPT4VisionService({"model_name": "gpt-4-vision-preview"})
    service.initialize()
    
    # Process single image with low detail
    config = {"detail_level": "low"}
    result, confidence = service.process_image(
        "path/to/image.jpg", 
        prompt="What objects are in this image?",
        **config
    )
    
    # Process multiple images
    images = ["image1.jpg", "image2.jpg"]
    result, confidence = service.process_multiple_images(
        images, 
        prompt="Compare these two images and tell me the differences."
    )
    ```
    
    Version Compatibility:
    - Compatible with OpenAI API v1.0.0 and newer
    - Supports all GPT-4 Vision models (including future updates)
    - Automatically adapts to API response format changes via error handling
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the GPT-4 Vision service.
        
        Args:
            config: Configuration dictionary for model selection and parameters
        """
        super().__init__(config or {})
        
        # Convert raw config dict to VisionConfig if needed
        if not isinstance(self.config, VisionConfig):
            config_name = f"gpt4v_{self.model_name}"
            vision_config = get_vision_config(config_name)
            
            # Update config with values from the raw config
            for key, value in config.items():
                if hasattr(vision_config, key):
                    setattr(vision_config, key, value)
            
            self.config = vision_config
        
        # Create model manager
        self.model_manager = GPT4VisionModelManager(self.config)
        
        # Configure generation parameters
        self.max_tokens = self.config.max_tokens
        self.temperature = self.config.temperature
        self.top_p = getattr(self.config, 'top_p', 0.9)
        self.detail_level = getattr(self.config, 'detail_level', 'auto')
        
        # Flag for initialization
        self.initialized = False
        self.client = None
        
        # Support for multiple images (GPT-4 Vision supports multiple images)
        self.supports_multiple_images = True
    
    @handle_vision_errors
    def initialize(self) -> bool:
        """
        Initialize the GPT-4 Vision service.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
            
        Raises:
            ModelUnavailableError: If the model cannot be initialized
        """
        if self.initialized and self.client is not None:
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
            
            # Initialize the OpenAI client
            self.client = self.model_manager.initialize_client()
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
            
            logger.info(f"Successfully initialized GPT-4 Vision service for model: {self.model_manager.model_name}")
            return True
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("initialize", op_time, success=False)
                self.metrics.record_vision_error("initialization_error", {"error": str(e)})
            
            logger.error(f"Failed to initialize GPT-4 Vision service: {str(e)}")
            raise ModelUnavailableError(
                f"Failed to initialize GPT-4 Vision model: {str(e)}",
                model_name=self.model_name
            )
    
    @handle_image_processing_errors
    def _prepare_image(self, image: Union[str, Image.Image, bytes]) -> Dict[str, Any]:
        """
        Prepare an image for the GPT-4 Vision API.
        
        Args:
            image: The image to prepare (path, URL, PIL image, or bytes)
            
        Returns:
            Dict[str, Any]: Image data formatted for GPT-4 Vision API
            
        Raises:
            ImageProcessingError: If image preparation fails
        """
        # Start metrics collection if available
        start_time = time.time()
        
        try:
            # URL format - simplest case, just return the URL
            if isinstance(image, str) and image.startswith(('http://', 'https://')):
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "url")
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": image,
                        "detail": self.detail_level
                    }
                }
            
            # For other types, we need to convert to base64
            # First get a PIL Image
            pil_image = None
            if isinstance(image, str):
                if image.startswith('data:image'):
                    # It's a base64 string
                    try:
                        # Extract the base64 part (after comma)
                        base64_data = image.split(',')[1]
                        image_data = base64.b64decode(base64_data)
                        pil_image = Image.open(BytesIO(image_data))
                        if self.metrics:
                            self.metrics.record_custom_metric("image_processing", "source_type", "base64")
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {e}")
                        raise ImageProcessingError(f"Failed to decode base64 image: {str(e)}", cause=e)
                elif os.path.exists(image):
                    # It's a file path
                    pil_image = Image.open(image)
                    if self.metrics:
                        self.metrics.record_custom_metric("image_processing", "source_type", "file_path")
                else:
                    raise ImageProcessingError(f"Image path does not exist: {image}")
            elif isinstance(image, bytes):
                # It's bytes data
                pil_image = Image.open(BytesIO(image))
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "bytes")
            elif isinstance(image, Image.Image):
                # It's already a PIL Image
                pil_image = image
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "pil_image")
            else:
                raise ImageProcessingError(f"Unsupported image type: {type(image)}")
            
            # Now convert PIL Image to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Return in OpenAI format
            result = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}",
                    "detail": self.detail_level
                }
            }
            
            return result
        
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
        Process an image with GPT-4 Vision.
        
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
            
            # Check if client is available
            if not self.client:
                raise VisionModelError("GPT-4 Vision client not initialized")
            
            # Use specified max_tokens if provided, otherwise use default
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Prepare the image
            processed_image = self._prepare_image(image)
            
            # Record image dimensions for metrics if it's a PIL Image
            if self.metrics and isinstance(image, Image.Image):
                width, height = image.size
                image_format = getattr(image, 'format', 'UNKNOWN')
                self.metrics.record_image_processed(
                    image_format=image_format,
                    original_dimensions=(width, height),
                    processed_dimensions=(width, height),
                    processing_time_ms=(time.time() - start_time) * 1000,
                    was_resized=False
                )
            
            # Prepare message for GPT-4 Vision
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        processed_image
                    ]
                }
            ]
            
            # Process with OpenAI
            generation_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Extract confidence from OpenAI response if available, otherwise estimate
            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'finish_reason'):
                # Adjust confidence based on finish reason
                if response.choices[0].finish_reason == 'stop':
                    confidence = 0.9
                elif response.choices[0].finish_reason == 'length':
                    confidence = 0.8  # Less confident if truncated
                else:
                    confidence = 0.7  # Less confident for other finish reasons
            else:
                confidence = 0.85  # Default if not available
            
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
                    "response_length": len(response_text),
                    "detail_level": self.detail_level
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
            
            logger.error(f"Error processing image with GPT-4 Vision: {e}")
            raise VisionModelError(f"Error processing image with GPT-4 Vision: {str(e)}", cause=e)
    
    @handle_multi_image_processing_errors
    def process_multiple_images(self, 
                               images: List[Union[str, Image.Image, bytes]], 
                               prompt: str = "Describe these images in detail.", 
                               max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Process multiple images with GPT-4 Vision.
        
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
            
            # Check if client is available
            if not self.client:
                raise VisionModelError("GPT-4 Vision client not initialized")
            
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
            
            # Create content for OpenAI format
            content = [{"type": "text", "text": prompt}]
            for processed_image in processed_images:
                content.append(processed_image)
            
            # Prepare message for GPT-4 Vision
            messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Process with OpenAI
            generation_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Calculate a confidence score
            # For multiple images, we reduce the confidence slightly due to added complexity
            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'finish_reason'):
                # Adjust confidence based on finish reason
                if response.choices[0].finish_reason == 'stop':
                    confidence = 0.85  # Slightly lower than single image due to complexity
                elif response.choices[0].finish_reason == 'length':
                    confidence = 0.75  # Less confident if truncated
                else:
                    confidence = 0.65  # Less confident for other finish reasons
            else:
                confidence = 0.8  # Default if not available
            
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
                    "response_length": len(response_text),
                    "detail_level": self.detail_level
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
            
            logger.error(f"Error processing multiple images with GPT-4 Vision: {e}")
            raise VisionModelError(f"Error processing multiple images: {str(e)}", cause=e)
    
    @handle_vision_errors
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate text based on a text-only prompt using GPT-4.
        
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
            
            # Check if client is available
            if not self.client:
                raise VisionModelError("GPT-4 client not initialized")
            
            # Use specified max_tokens if provided, otherwise use default
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Process with OpenAI
            generation_start = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # For text-only prompts, we generally assign higher confidence
            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'finish_reason'):
                # Adjust confidence based on finish reason
                if response.choices[0].finish_reason == 'stop':
                    confidence = 0.95  # Higher confidence for text-only
                elif response.choices[0].finish_reason == 'length':
                    confidence = 0.85  # Less confident if truncated
                else:
                    confidence = 0.8  # Less confident for other finish reasons
            else:
                confidence = 0.9  # Default if not available
            
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
            
            logger.error(f"Error generating text with GPT-4: {e}")
            raise VisionModelError(f"Error generating text: {str(e)}", cause=e)
    
    def list_speakers(self) -> List[str]:
        """Return a list of available speakers for the vision model."""
        return []  # GPT-4 Vision models don't have speakers
    
    def list_languages(self) -> List[str]:
        """Return a list of supported languages for the GPT-4 Vision model."""
        # GPT-4 Vision supports many languages
        return [
            "en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ru", 
            "ar", "hi", "nl", "tr", "pl", "fi", "sv", "da", "no"
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the GPT-4 Vision model.
        
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
                "detail_level": self.detail_level,
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
                "service": "gpt4v",
                "provider": self.__class__.__name__
            } 