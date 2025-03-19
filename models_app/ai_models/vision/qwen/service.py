"""
QwenVisionService

Provides image processing services using Qwen Vision models.
"""

import os
import logging
import torch
import numpy as np
import base64
import time
from PIL import Image
from typing import Dict, List, Any, Optional, Union, Tuple
from io import BytesIO
from pathlib import Path
import requests

from models_app.ai_models.vision.base_vision_provider import BaseVisionProvider
from models_app.ai_models.vision.qwen.model_manager import QwenVisionModelManager

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

class QwenVisionService(BaseVisionProvider):
    """
    Service for processing images with Qwen Vision models.
    
    This service provides:
    1. Image captioning
    2. Visual question answering
    3. Multimodal chat capabilities
    4. Processing of multiple images
    
    Thread Safety:
    This provider is not thread-safe for concurrent initialization or model operations.
    Each thread should use its own instance of QwenVisionService to avoid race conditions
    with the model and tokenizer state. The _prepare_image_for_model method is thread-safe
    and can be used concurrently.
    
    Resource Usage:
    - GPU Memory: The model requires significant GPU memory (4-8GB depending on the model).
    - CPU Usage: Preparation and post-processing are CPU-bound operations.
    - Consider freeing GPU resources when not in use for extended periods using explicit
      cleanup methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Qwen Vision service.
        
        Args:
            config: Configuration dictionary for model selection and parameters
        """
        super().__init__(config or {})
        
        # Convert raw config dict to VisionConfig if needed
        if not isinstance(self.config, VisionConfig):
            config_name = f"qwen_{self.model_name}"
            vision_config = get_vision_config(config_name)
            
            # Update config with values from the raw config
            for key, value in config.items():
                if hasattr(vision_config, key):
                    setattr(vision_config, key, value)
            
            self.config = vision_config
        
        # Create model manager
        self.model_manager = QwenVisionModelManager(self.config)
        
        # Configure generation parameters
        self.max_tokens = self.config.max_tokens
        self.temperature = self.config.temperature
        self.top_p = getattr(self.config, 'top_p', 0.9)
        self.repetition_penalty = getattr(self.config, 'repetition_penalty', 1.1)
        
        # Flag for initialization
        self.initialized = False
        self.model = None
        self.processor = None
        
        # Support for multiple images
        self.supports_multiple_images = self.config.supports_multiple_images
    
    @handle_vision_errors
    def initialize(self) -> bool:
        """
        Initialize the Qwen Vision service.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
            
        Raises:
            ModelUnavailableError: If the model cannot be initialized
        """
        if self.initialized and self.model is not None and self.processor is not None:
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
                logger.error(f"Model {self.model_manager.model_name} is not available on this system")
                raise ModelUnavailableError(
                    f"Model {self.model_manager.model_name} is not available on this system",
                    model_name=self.model_manager.model_name
                )
            
            # Initialize the model
            self.model, self.processor = self.model_manager.initialize_model()
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
            
            logger.info(f"Successfully initialized Qwen Vision service for model: {self.model_manager.model_name}")
            return True
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("initialize", op_time, success=False)
                self.metrics.record_vision_error("initialization_error", {"error": str(e)})
            
            logger.error(f"Failed to initialize Qwen Vision service: {str(e)}")
            raise ModelUnavailableError(
                f"Failed to initialize Qwen Vision model: {str(e)}",
                model_name=self.model_name
            )
    
    @handle_image_processing_errors
    def _prepare_image_for_model(self, image: Union[str, Image.Image, bytes]) -> Union[Image.Image, str]:
        """
        Prepare an image for the model.
        
        Args:
            image: The image to prepare (path, URL, PIL image, or bytes)
            
        Returns:
            Union[Image.Image, str]: Prepared image or image path
            
        Raises:
            ImageProcessingError: If image preparation fails
        """
        # Start metrics collection if available
        start_time = time.time()
        
        try:
            # If already a PIL Image, just return it
            if isinstance(image, Image.Image):
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "pil_image")
                return image
            
            # If bytes, convert to PIL Image
            if isinstance(image, bytes):
                result = Image.open(BytesIO(image))
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "bytes")
                return result
            
            # If string...
            if isinstance(image, str):
                # If URL, return as-is as some models can handle URLs directly
                if image.startswith(('http://', 'https://')):
                    # Try to fetch and load the image to verify it exists
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
                
                # If base64, decode to PIL Image
                elif image.startswith('data:image'):
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
                
                # If file path, check if it exists and return path or load image
                elif os.path.exists(image):
                    result = Image.open(image)
                    if self.metrics:
                        self.metrics.record_custom_metric("image_processing", "source_type", "file_path")
                    return result
                else:
                    raise ImageProcessingError(f"Image path does not exist: {image}")
            
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
        Process an image with Qwen Vision.
        
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
            if not self.model or not self.processor:
                raise VisionModelError("Model not initialized")
            
            # Use specified max_tokens if provided, otherwise use default
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Prepare the image
            processed_image = self._prepare_image_for_model(image)
            
            # Prepare inputs for the model
            generation_start = time.time()
            inputs = self.processor(
                text=prompt,
                images=processed_image,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty
            )
            
            # Decode the response
            response_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Calculate a confidence score based on model output
            # (Qwen doesn't provide confidence scores, so we estimate based on length and other factors)
            confidence = 0.85
            
            # Adjust confidence based on response characteristics
            if len(response_text) < 20:  # Very short responses might be less reliable
                confidence = 0.7
            elif any(phrase in response_text.lower() for phrase in ["i'm not sure", "cannot determine", "unclear"]):
                confidence = 0.6
            
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
            
            logger.error(f"Error processing image with Qwen Vision: {e}")
            raise VisionModelError(f"Error processing image with Qwen Vision: {str(e)}", cause=e)
    
    @handle_multi_image_processing_errors
    def process_multiple_images(self, 
                               images: List[Union[str, Image.Image, bytes]], 
                               prompt: str = "Describe these images in detail.", 
                               max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Process multiple images with Qwen Vision.
        
        Args:
            images: List of images to process
            prompt: The text prompt to guide the model's understanding of the images
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Tuple[str, float]: Generated text and confidence score
            
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
            if not self.model or not self.processor:
                raise VisionModelError("Model not initialized")
            
            # Check if multiple images are supported
            if not self.supports_multiple_images:
                logger.warning("This Qwen model doesn't support processing multiple images. Using only the first image.")
                if not images:
                    raise ImageProcessingError("No images provided")
                return self.process_image(images[0], prompt, max_tokens)
            
            # Use specified max_tokens if provided, otherwise use default
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Limit the number of images to process
            MAX_IMAGES = 5
            if len(images) > MAX_IMAGES:
                logger.warning(f"Too many images provided. Processing only the first {MAX_IMAGES}.")
                images = images[:MAX_IMAGES]
            
            # Preprocess all images
            preprocessing_start = time.time()
            processed_images = [self._prepare_image_for_model(img) for img in images]
            
            # Record multi-image processing
            preprocessing_time_ms = (time.time() - preprocessing_start) * 1000
            if self.metrics:
                self.metrics.record_multi_image_processed(
                    image_count=len(processed_images),
                    processing_time_ms=preprocessing_time_ms
                )
            
            # Build a prompt that references all images
            image_prompt = prompt
            
            # Prepare inputs for the model
            generation_start = time.time()
            
            # Process first image with text prompt
            inputs = self.processor(
                text=image_prompt,
                images=processed_images,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty
            )
            
            # Decode the response
            response_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Calculate a confidence score
            # For multiple images, we reduce the confidence slightly due to added complexity
            confidence = 0.8
            if len(response_text) < 50:  # Short responses might indicate struggles with multi-image context
                confidence = 0.6
            elif any(phrase in response_text.lower() for phrase in ["i'm not sure", "cannot determine", "unclear"]):
                confidence = 0.5
            
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
            
            logger.error(f"Error processing multiple images with Qwen Vision: {e}")
            raise VisionModelError(f"Error processing multiple images: {str(e)}", cause=e)
    
    @handle_vision_errors
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate text based on a text-only prompt using Qwen.
        
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
            if not self.model or not self.processor:
                raise VisionModelError("Model not initialized")
            
            # Use specified max_tokens if provided, otherwise use default
            tokens = max_tokens if max_tokens is not None else self.max_tokens
            
            # Prepare inputs for the model
            generation_start = time.time()
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty
            )
            
            # Decode the response
            response_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # For text-only prompts, we generally assign higher confidence
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
            
            logger.error(f"Error generating text with Qwen: {e}")
            raise VisionModelError(f"Error generating text: {str(e)}", cause=e)
    
    def list_speakers(self) -> List[str]:
        """Return a list of available speakers for the vision model."""
        return []  # Qwen Vision models don't have speakers
    
    def list_languages(self) -> List[str]:
        """Return a list of supported languages for the Qwen Vision model."""
        # Qwen models typically support mainly Chinese and English, but can work with other languages
        return ["zh", "en"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Qwen Vision model.
        
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
                "repetition_penalty": self.repetition_penalty,
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
                "service": "qwen_vision",
                "provider": self.__class__.__name__
            } 