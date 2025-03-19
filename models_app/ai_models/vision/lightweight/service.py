"""
LightweightVisionService

Provides image processing services using lightweight vision models (CLIP, BLIP, etc.).
"""

import os
import tempfile
import logging
import torch
import numpy as np
import base64
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from io import BytesIO
from pathlib import Path
from PIL import Image
import requests
import urllib.parse

from .model_manager import LightweightVisionModelManager

# Import base vision provider for inheritance
from models_app.ai_models.vision.base_vision_provider import BaseVisionProvider

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

class LightweightVisionService(BaseVisionProvider):
    """
    Service for processing images using lightweight vision models.
    
    This service provides:
    1. Image classification and embedding extraction
    2. Image captioning and visual question answering
    3. Zero-shot image classification
    4. Image text similarity
    
    Thread Safety:
    This provider is NOT thread-safe for most operations. The underlying models maintain
    internal state during operations that can cause race conditions when accessed from
    multiple threads. Each thread should use its own instance of LightweightVisionService.
    The only exception is the get_model_info() method which is thread-safe.
    
    Resource Usage:
    - GPU Memory: 
      * CLIP: 1-2GB GPU memory
      * BLIP: 2-3GB GPU memory
      * PaLI-Gemma: 4-6GB GPU memory
      * CLIP+Phi: 3-4GB GPU memory
    - CPU Usage: High during image preprocessing
    - Memory: 500MB-1GB system RAM required in addition to GPU memory
    - Disk Space: Models are cached locally (200MB-2GB depending on model)
    
    Performance Characteristics:
    - Inference Speed: 
      * CLIP: Very fast (50-150ms)
      * BLIP: Medium (200-400ms)
      * PaLI-Gemma: Slower (500-800ms)
      * CLIP+Phi: Medium-slow (300-600ms)
    - Batch Performance: CLIP models scale well with batching
    - Best for: 
      * CLIP: Classification, similarity tasks
      * BLIP: Image captioning, simple VQA
      * PaLI-Gemma: Detailed captioning, complex VQA
      * CLIP+Phi: Good all-around performance
    
    Limitations:
    - Maximum image resolution varies by model type
    - CLIP models cannot answer open-ended questions
    - BLIP provides limited detail compared to larger models
    - PaLI-Gemma requires significant resources
    - Multi-image processing only supported by some models
    - All models have limited context understanding compared to API-based alternatives
    - Model performance affected by image quality and lighting conditions
    
    Memory Management:
    GPU memory is not automatically freed after operations. Call the cleanup() method
    when the service is no longer needed to free GPU resources. In high-throughput
    scenarios, consider periodically freeing memory with cleanup() and re-initializing.
    
    Example Usage:
    ```python
    from models_app.ai_models.vision.lightweight.service import LightweightVisionService
    
    # Initialize with CLIP
    clip_service = LightweightVisionService({"model_type": "clip"})
    clip_service.initialize()
    
    # Classify an image against custom categories
    categories = ["dog", "cat", "bird", "car", "building"]
    result, confidence = clip_service.classify_image("image.jpg", categories)
    
    # Get image embeddings for similarity search
    embedding = clip_service.get_image_embedding("image.jpg")
    
    # Use BLIP for captioning
    blip_service = LightweightVisionService({"model_type": "blip"})
    blip_service.initialize()
    caption, confidence = blip_service.process_image("image.jpg")
    
    # Clean up when done to free GPU memory
    clip_service.cleanup()
    blip_service.cleanup()
    ```
    
    Version Compatibility:
    - Compatible with PyTorch 1.10+ and transformers 4.18+
    - CLIP models require torchvision 0.11+
    - BLIP models require timm 0.5.4+
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the lightweight vision service.
        
        Args:
            config: Configuration dictionary for model selection and parameters
        """
        super().__init__(config or {})
        
        # Convert raw config dict to VisionConfig if needed
        if not isinstance(self.config, VisionConfig):
            # Get appropriate configuration for the model type
            model_type = config.get('model_type', 'clip') if config else 'clip'
            config_name = f"lightweight_{model_type}"
            vision_config = get_vision_config(config_name)
            
            # Update config with values from the raw config
            if config:
                for key, value in config.items():
                    if hasattr(vision_config, key):
                        setattr(vision_config, key, value)
            
            self.config = vision_config
        
        # Create model manager
        self.model_manager = LightweightVisionModelManager(config)
        self.model = None
        self.processor = None
        self.initialized = False
        
        # Extract configuration parameters with appropriate defaults
        self.max_length = getattr(self.config, 'max_length', 50)
        self.temperature = getattr(self.config, 'temperature', 0.7)
        self.top_p = getattr(self.config, 'top_p', 0.9)
        self.top_k = getattr(self.config, 'top_k', 50)
        self.repetition_penalty = getattr(self.config, 'repetition_penalty', 1.0)
        
        # Add support for multiple images (some lightweight models don't support this)
        self.supports_multiple_images = getattr(self.config, 'supports_multiple_images', False)
        
        # Set model name for consistent access
        self.model_name = getattr(self.config, 'model_name', self.model_manager.model_type)
        
        # Track compatibility with tasks
        self._supported_tasks = self._get_supported_tasks()
    
    def _get_supported_tasks(self) -> Dict[str, bool]:
        """
        Determine which tasks are supported by the selected model.
        
        Returns:
            Dict[str, bool]: Map of task names to boolean indicating support
        """
        supported = {
            'image_classification': False,
            'image_captioning': False,
            'visual_question_answering': False,
            'image_embedding': False,
            'text_similarity': False
        }
        
        model_type = self.model_manager.model_type
        
        if model_type == 'clip':
            supported.update({
                'image_classification': True,
                'image_embedding': True,
                'text_similarity': True
            })
        elif model_type == 'blip':
            supported.update({
                'image_captioning': True,
                'visual_question_answering': True,
                'image_embedding': True
            })
        elif model_type == 'paligemma':
            supported.update({
                'image_captioning': True,
                'visual_question_answering': True,
                'image_embedding': True,
                'text_similarity': True
            })
        elif model_type == 'clip_phi':
            supported.update({
                'image_classification': True,
                'image_captioning': True,
                'visual_question_answering': True,
                'image_embedding': True
            })
        
        return supported
    
    def is_task_supported(self, task: str) -> bool:
        """
        Check if a specific task is supported by the current model.
        
        Args:
            task: Task name to check
            
        Returns:
            bool: True if the task is supported, False otherwise
        """
        return self._supported_tasks.get(task.lower(), False)
    
    @handle_vision_errors
    def initialize(self) -> bool:
        """
        Initialize the vision service by loading the model.
        
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
                        {"model": self.model_manager.model_type}
                    )
                logger.error(f"Model {self.model_manager.model_type} is not available on this system")
                raise ModelUnavailableError(
                    f"Model {self.model_manager.model_type} is not available",
                    model_name=self.model_manager.model_type
                )
            
            # Initialize the model
            self.model, self.processor = self.model_manager.initialize_model()
            self.initialized = True
            
            # Update supported tasks based on the loaded model
            self._supported_tasks = self._get_supported_tasks()
            
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
            
            logger.info(f"Successfully initialized {self.model_manager.model_type} service")
            return True
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("initialize", op_time, success=False)
                self.metrics.record_vision_error("initialization_error", {"error": str(e)})
            
            logger.error(f"Failed to initialize vision service: {str(e)}")
            raise ModelUnavailableError(
                f"Failed to initialize lightweight vision model: {str(e)}",
                model_name=self.model_manager.model_type
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        model_info = self.model_manager.get_model_info()
        model_info.update({
            'service_initialized': self.initialized,
            'supported_tasks': self._supported_tasks,
            'generation_params': {
                'max_length': self.max_length,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'top_k': self.top_k,
                'repetition_penalty': self.repetition_penalty
            },
            'supports_multiple_images': self.supports_multiple_images,
            'model_name': self.model_name
        })
        return model_info
    
    @handle_image_processing_errors
    def _prepare_image(self, image_input: Union[str, bytes, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load and preprocess image from various input formats.
        
        Args:
            image_input: Image input, can be:
                - Path to image file (str)
                - URL (str starting with http/https)
                - Base64 encoded string (str starting with data:image)
                - Bytes
                - PIL Image
                - Numpy array
        
        Returns:
            PIL.Image.Image: Preprocessed image
            
        Raises:
            ImageProcessingError: If image cannot be loaded or processed
        """
        # Start metrics collection
        start_time = time.time()
        
        try:
            # Load the image using the previous _load_image logic
            # If already a PIL Image, use it directly
            if isinstance(image_input, Image.Image):
                pil_image = image_input
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "pil_image")
            
            # If numpy array, convert to PIL Image
            elif isinstance(image_input, np.ndarray):
                pil_image = Image.fromarray(image_input.astype('uint8'))
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "numpy_array")
            
            # If bytes, convert to PIL Image
            elif isinstance(image_input, bytes):
                pil_image = Image.open(BytesIO(image_input))
                if self.metrics:
                    self.metrics.record_custom_metric("image_processing", "source_type", "bytes")
            
            # If string, handle different formats
            elif isinstance(image_input, str):
                # URL
                if image_input.startswith(('http://', 'https://')):
                    try:
                        response = requests.get(image_input, stream=True, timeout=10)
                        response.raise_for_status()
                        pil_image = Image.open(BytesIO(response.content))
                        if self.metrics:
                            self.metrics.record_custom_metric("image_processing", "source_type", "url")
                    except Exception as e:
                        logger.error(f"Failed to download image from URL: {e}")
                        raise ImageProcessingError(f"Failed to download image from URL: {str(e)}", cause=e)
                
                # Base64
                elif image_input.startswith('data:image'):
                    try:
                        # Extract the base64 part (after comma)
                        base64_data = image_input.split(',')[1]
                        image_data = base64.b64decode(base64_data)
                        pil_image = Image.open(BytesIO(image_data))
                        if self.metrics:
                            self.metrics.record_custom_metric("image_processing", "source_type", "base64")
                    except Exception as e:
                        logger.error(f"Failed to decode base64 image: {e}")
                        raise ImageProcessingError(f"Failed to decode base64 image: {str(e)}", cause=e)
                
                # File path
                else:
                    try:
                        pil_image = Image.open(image_input)
                        if self.metrics:
                            self.metrics.record_custom_metric("image_processing", "source_type", "file_path")
                    except Exception as e:
                        logger.error(f"Failed to open image file: {e}")
                        raise ImageProcessingError(f"Failed to open image file: {str(e)}", cause=e)
            else:
                # If we get here, the input format wasn't recognized
                raise ImageProcessingError(f"Unsupported image input type: {type(image_input)}")
            
            # Record original image properties if metrics available
            if self.metrics and hasattr(pil_image, 'size'):
                width, height = pil_image.size
                image_format = getattr(pil_image, 'format', 'UNKNOWN')
                self.metrics.record_custom_metric("image_processing", "original_width", width)
                self.metrics.record_custom_metric("image_processing", "original_height", height)
                self.metrics.record_custom_metric("image_processing", "image_format", str(image_format))
            
            # Perform preprocessing (converting to RGB and resizing)
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Resize based on model requirements
            model_type = self.model_manager.model_type
            if model_type == 'clip':
                # CLIP generally works with 224x224 or 336x336 images
                processed_image = pil_image.resize((224, 224))
            elif model_type == 'blip':
                # BLIP typically uses 384x384 images
                processed_image = pil_image.resize((384, 384))
            elif model_type == 'paligemma':
                # PaliGemma typically uses larger images
                processed_image = pil_image.resize((576, 576))
            else:
                # Default resize for unknown models
                processed_image = pil_image.resize((224, 224))
            
            # Record processed image properties and timing if metrics available
            if self.metrics and hasattr(processed_image, 'size'):
                processing_time_ms = (time.time() - start_time) * 1000
                new_width, new_height = processed_image.size
                was_resized = (width, height) != (new_width, new_height)
                
                self.metrics.record_image_processed(
                    image_format=image_format,
                    original_dimensions=(width, height),
                    processed_dimensions=(new_width, new_height),
                    processing_time_ms=processing_time_ms,
                    was_resized=was_resized
                )
            
            return processed_image
            
        except Exception as e:
            # If already an ImageProcessingError, re-raise
            if isinstance(e, ImageProcessingError):
                raise e
            
            # Otherwise wrap in ImageProcessingError
            logger.error(f"Error preparing image: {e}")
            raise ImageProcessingError(f"Failed to prepare image: {str(e)}", cause=e)
    
    def classify_image(
        self, 
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        candidate_labels: List[str] = None
    ) -> Dict[str, float]:
        """
        Classify image into one of several candidate categories.
        
        Args:
            image_input: Image to classify
            candidate_labels: List of text labels to classify the image as
            
        Returns:
            Dict[str, float]: Dictionary mapping labels to confidence scores
            
        Raises:
            ValueError: If image cannot be loaded or model doesn't support classification
        """
        if not self.is_task_supported('image_classification'):
            raise ValueError(f"Model {self.model_manager.model_type} does not support image classification")
        
        if not self.initialized:
            self.initialize()
        
        # Default candidate labels if none provided
        if candidate_labels is None or len(candidate_labels) == 0:
            candidate_labels = ["a photo of a person", "a photo of an animal", "a photo of a landscape", 
                               "a photo of food", "a photo of a building", "a photo of text"]
        
        try:
            # Load and preprocess image
            image = self._prepare_image(image_input)
            
            # Process with model
            model_type = self.model_manager.model_type
            
            if model_type == 'clip':
                # For CLIP, we use text-image similarity to classify
                inputs = self.processor(
                    text=candidate_labels,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
                
                # Create result dictionary
                result = {label: float(score) for label, score in zip(candidate_labels, probs)}
                
            elif model_type == 'clip_phi':
                # For CLIP+Phi hybrid, use CLIP part for classification
                clip_model = self.model.clip_model
                clip_processor = self.model.clip_processor
                
                inputs = clip_processor(
                    text=candidate_labels,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1).squeeze().cpu().numpy()
                
                # Create result dictionary
                result = {label: float(score) for label, score in zip(candidate_labels, probs)}
                
            else:
                # For other models, this shouldn't be reached due to the is_task_supported check
                raise ValueError(f"Model {model_type} does not support image classification")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in classify_image: {str(e)}")
            raise
    
    def generate_caption(
        self, 
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        prompt: str = None,
        max_length: int = None
    ) -> str:
        """
        Generate a caption for the image.
        
        Args:
            image_input: Image to caption
            prompt: Optional prompt to guide caption generation
            max_length: Maximum length of generated caption
            
        Returns:
            str: Generated caption
            
        Raises:
            ValueError: If image cannot be loaded or model doesn't support captioning
        """
        if not self.is_task_supported('image_captioning'):
            raise ValueError(f"Model {self.model_manager.model_type} does not support image captioning")
        
        if not self.initialized:
            self.initialize()
        
        max_length = max_length or self.max_length
        
        try:
            # Load and preprocess image
            image = self._prepare_image(image_input)
            
            # Process with model
            model_type = self.model_manager.model_type
            
            if model_type == 'blip':
                # For BLIP, we can use the conditional generation feature
                if prompt:
                    inputs = self.processor(image, prompt, return_tensors="pt").to(self.model_manager.device)
                else:
                    inputs = self.processor(image, return_tensors="pt").to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty
                    )
                
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                
            elif model_type == 'paligemma':
                # For PaliGemma, similar to BLIP
                if prompt:
                    inputs = self.processor(image, prompt, return_tensors="pt").to(self.model_manager.device)
                else:
                    inputs = self.processor(image, "Describe this image", return_tensors="pt").to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty
                    )
                
                caption = self.processor.decode(outputs[0], skip_special_tokens=True)
                
            elif model_type == 'clip_phi':
                # For CLIP+Phi hybrid, use CLIP for embeddings and Phi for generation
                clip_model = self.model.clip_model
                clip_processor = self.model.clip_processor
                phi_model = self.model.phi_model
                phi_tokenizer = self.model.phi_tokenizer
                
                # Get image embedding from CLIP
                inputs = clip_processor(images=image, return_tensors="pt").to(self.model_manager.device)
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Prepare prompt for Phi
                if prompt:
                    generator_prompt = f"{prompt}\nImage description:"
                else:
                    generator_prompt = "This image shows:"
                
                # Generate text with Phi model
                input_ids = phi_tokenizer(generator_prompt, return_tensors="pt").input_ids.to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = phi_model.generate(
                        input_ids,
                        max_length=len(input_ids[0]) + max_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty
                    )
                
                caption = phi_tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            
            else:
                # For other models, this shouldn't be reached due to the is_task_supported check
                raise ValueError(f"Model {model_type} does not support image captioning")
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Error in generate_caption: {str(e)}")
            raise
    
    def answer_question(
        self, 
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        question: str,
        max_length: int = None
    ) -> str:
        """
        Answer a question about the image.
        
        Args:
            image_input: Image to analyze
            question: Question about the image
            max_length: Maximum length of generated answer
            
        Returns:
            str: Answer to the question
            
        Raises:
            ValueError: If image cannot be loaded or model doesn't support VQA
        """
        if not self.is_task_supported('visual_question_answering'):
            raise ValueError(f"Model {self.model_manager.model_type} does not support visual question answering")
        
        if not self.initialized:
            self.initialize()
        
        max_length = max_length or self.max_length
        
        try:
            # Load and preprocess image
            image = self._prepare_image(image_input)
            
            # Process with model
            model_type = self.model_manager.model_type
            
            if model_type == 'blip':
                # For BLIP, use the VQA feature
                inputs = self.processor(image, question, return_tensors="pt").to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty
                    )
                
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                
            elif model_type == 'paligemma':
                # For PaliGemma
                inputs = self.processor(image, question, return_tensors="pt").to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty
                    )
                
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                
            elif model_type == 'clip_phi':
                # For CLIP+Phi hybrid, use CLIP for embeddings and Phi for generation
                clip_model = self.model.clip_model
                clip_processor = self.model.clip_processor
                phi_model = self.model.phi_model
                phi_tokenizer = self.model.phi_tokenizer
                
                # Get image embedding from CLIP
                inputs = clip_processor(images=image, return_tensors="pt").to(self.model_manager.device)
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Prepare prompt for Phi with the question
                generator_prompt = f"Question: {question}\nAnswer:"
                
                # Generate text with Phi model
                input_ids = phi_tokenizer(generator_prompt, return_tensors="pt").input_ids.to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = phi_model.generate(
                        input_ids,
                        max_length=len(input_ids[0]) + max_length,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty
                    )
                
                answer = phi_tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            
            else:
                # For other models, this shouldn't be reached due to the is_task_supported check
                raise ValueError(f"Model {model_type} does not support visual question answering")
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error in answer_question: {str(e)}")
            raise
    
    def get_image_embedding(
        self, 
        image_input: Union[str, bytes, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Extract embedding vector from image.
        
        Args:
            image_input: Image to embed
            
        Returns:
            np.ndarray: Embedding vector
            
        Raises:
            ValueError: If image cannot be loaded or model doesn't support embedding
        """
        if not self.is_task_supported('image_embedding'):
            raise ValueError(f"Model {self.model_manager.model_type} does not support image embedding")
        
        if not self.initialized:
            self.initialize()
        
        try:
            # Load and preprocess image
            image = self._prepare_image(image_input)
            
            # Process with model
            model_type = self.model_manager.model_type
            
            if model_type == 'clip':
                # For CLIP, extract image features
                inputs = self.processor(images=image, return_tensors="pt").to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    # Normalize embeddings
                    embedding = outputs / outputs.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embedding = embedding.cpu().numpy().squeeze()
                
            elif model_type == 'blip':
                # For BLIP, extract image features
                inputs = self.processor(images=image, return_tensors="pt").to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    # Normalize embeddings
                    embedding = outputs / outputs.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embedding = embedding.cpu().numpy().squeeze()
                
            elif model_type == 'paligemma':
                # For PaliGemma, extract image features
                inputs = self.processor(images=image, return_tensors="pt").to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    # Normalize embeddings if it has this method
                    if hasattr(outputs, 'norm'):
                        embedding = outputs / outputs.norm(dim=-1, keepdim=True)
                    else:
                        embedding = outputs
                
                # Convert to numpy
                embedding = embedding.cpu().numpy().squeeze()
                
            elif model_type == 'clip_phi':
                # For CLIP+Phi hybrid, use CLIP for embeddings
                clip_model = self.model.clip_model
                clip_processor = self.model.clip_processor
                
                # Get image embedding from CLIP
                inputs = clip_processor(images=image, return_tensors="pt").to(self.model_manager.device)
                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)
                    embedding = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embedding = embedding.cpu().numpy().squeeze()
            
            else:
                # For other models, this shouldn't be reached due to the is_task_supported check
                raise ValueError(f"Model {model_type} does not support image embedding")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error in get_image_embedding: {str(e)}")
            raise
    
    def compute_text_similarity(
        self, 
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        texts: List[str]
    ) -> Dict[str, float]:
        """
        Compute similarity between image and multiple text descriptions.
        
        Args:
            image_input: Image to compare
            texts: List of text descriptions to compare against
            
        Returns:
            Dict[str, float]: Dictionary mapping texts to similarity scores
            
        Raises:
            ValueError: If image cannot be loaded or model doesn't support text similarity
        """
        if not self.is_task_supported('text_similarity'):
            raise ValueError(f"Model {self.model_manager.model_type} does not support text similarity")
        
        if not self.initialized:
            self.initialize()
        
        try:
            # Load and preprocess image
            image = self._prepare_image(image_input)
            
            # Process with model
            model_type = self.model_manager.model_type
            
            if model_type == 'clip':
                # For CLIP, use text-image similarity
                inputs = self.processor(
                    text=texts,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.model_manager.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    similarity_scores = logits_per_image.squeeze().cpu().numpy()
                
                # Normalize scores to [0, 1] range
                if len(texts) > 1:
                    similarity_scores = similarity_scores / similarity_scores.sum()
                
                # Create result dictionary
                result = {text: float(score) for text, score in zip(texts, similarity_scores)}
                
            else:
                # For other models, this shouldn't be reached due to the is_task_supported check
                raise ValueError(f"Model {model_type} does not support text similarity")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in compute_text_similarity: {str(e)}")
            raise
    
    @handle_vision_errors
    def process_image(
        self, 
        image: Union[str, Image.Image, bytes], 
        prompt: str = "Describe this image in detail.", 
        max_tokens: Optional[int] = None
    ) -> Tuple[str, float]:
        """
        Process an image with the vision model.
        
        Args:
            image: The image to process (path, URL, PIL image, or bytes)
            prompt: The text prompt to guide the model's understanding of the image
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Tuple[str, float]: The generated text and a confidence score
            
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
            
            # Determine the most appropriate task based on the prompt
            task = 'caption'
            if '?' in prompt:
                task = 'vqa'
            
            # Check if the selected task is supported
            if task == 'vqa' and not self.is_task_supported('visual_question_answering'):
                task = 'caption'  # Fall back to caption if VQA not supported
                logger.warning(f"VQA not supported by {self.model_manager.model_type}, falling back to captioning")
            
            if task == 'caption' and not self.is_task_supported('image_captioning'):
                # If neither VQA nor captioning is supported, try classification
                if self.is_task_supported('image_classification'):
                    task = 'classify'
                else:
                    raise VisionModelError(f"Model {self.model_manager.model_type} does not support image captioning or VQA")
            
            # Use specified max_tokens if provided, otherwise use default
            max_length = max_tokens if max_tokens is not None else self.max_length
            
            # Process the image based on the selected task
            response_text = ""
            confidence = 0.0
            
            # Prepare the image
            processed_image = self._prepare_image(image)
            
            # Process with the appropriate method
            if task == 'vqa':
                response_text = self.answer_question(processed_image, prompt, max_length)
                confidence = 0.85  # Appropriate confidence for VQA
            elif task == 'caption':
                response_text = self.generate_caption(processed_image, prompt, max_length)
                confidence = 0.9  # Higher confidence for captioning
            elif task == 'classify':
                # Extract potential classes from the prompt
                potential_classes = [p.strip() for p in prompt.split(',') if p.strip()]
                if not potential_classes:
                    potential_classes = ["a photo of a person", "a photo of an animal", "a photo of a landscape", 
                                         "a photo of food", "a photo of a building", "a photo of text"]
                
                classifications = self.classify_image(processed_image, potential_classes)
                # Create text response from classifications
                lines = ["Image classification results:"]
                for label, score in sorted(classifications.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"- {label}: {score:.2f}")
                
                response_text = "\n".join(lines)
                # Use the highest confidence score
                confidence = max(classifications.values()) if classifications else 0.7
            
            # Record metrics
            if self.metrics:
                # Record total processing time
                total_time_ms = (time.time() - start_time) * 1000
                
                # Record inference metrics
                self.metrics.record_inference(
                    inference_time_ms=total_time_ms,
                    confidence=confidence
                )
                
                # Record model usage
                self.metrics.record_model_usage(self.model_name, {
                    "prompt_length": len(prompt),
                    "task": task,
                    "max_length": max_length,
                    "response_length": len(response_text),
                    "temperature": self.temperature
                })
                
                # Stop operation timing
                if op_time:
                    self.metrics.stop_operation("process_image", op_time, success=True)
            
            return response_text.strip(), confidence
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("process_image", op_time, success=False)
                self.metrics.record_vision_error("image_processing_error", {"error": str(e)})
            
            logger.error(f"Error processing image with {self.model_name}: {e}")
            raise VisionModelError(f"Error processing image with {self.model_name}: {str(e)}", cause=e)
    
    @handle_multi_image_processing_errors
    def process_multiple_images(
        self, 
        images: List[Union[str, Image.Image, bytes]], 
        prompt: str = "Describe these images in detail.", 
        max_tokens: Optional[int] = None
    ) -> Tuple[str, float]:
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
            
            # Check if multi-image support is available
            if not self.supports_multiple_images:
                if self.metrics and op_time:
                    self.metrics.stop_operation("process_multiple_images", op_time, success=False)
                    self.metrics.record_vision_error("multi_image_not_supported", {
                        "model": self.model_name,
                        "image_count": len(images)
                    })
                
                raise VisionModelError(f"Model {self.model_name} does not support multiple images")
            
            # Process images one by one and combine results
            responses = []
            total_confidence = 0.0
            
            for idx, img in enumerate(images):
                try:
                    # Adjust prompt for each image
                    img_prompt = f"Image {idx+1}: {prompt}"
                    
                    # Process each image individually
                    response, confidence = self.process_image(img, img_prompt, max_tokens)
                    responses.append(f"Image {idx+1}:\n{response}")
                    total_confidence += confidence
                except Exception as e:
                    logger.error(f"Error processing image {idx+1}: {e}")
                    responses.append(f"Image {idx+1}: Error - {str(e)}")
            
            # Combine responses with a summary
            combined_response = f"Analysis of {len(images)} images:\n\n" + "\n\n".join(responses)
            
            # Average confidence across all successful images
            avg_confidence = total_confidence / len(images) if images else 0.0
            
            # Record metrics
            if self.metrics:
                # Record processing time
                processing_time_ms = (time.time() - start_time) * 1000
                
                # Record multi-image metrics
                self.metrics.record_multi_image_processed(
                    image_count=len(images),
                    processing_time_ms=processing_time_ms
                )
                
                # Record model usage
                self.metrics.record_model_usage(self.model_name, {
                    "prompt_length": len(prompt),
                    "image_count": len(images),
                    "max_tokens": max_tokens,
                    "response_length": len(combined_response)
                })
                
                # Stop operation timing
                if op_time:
                    self.metrics.stop_operation("process_multiple_images", op_time, success=True)
            
            return combined_response, avg_confidence
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("process_multiple_images", op_time, success=False)
                self.metrics.record_vision_error("multi_image_processing_error", {
                    "error": str(e),
                    "image_count": len(images) if isinstance(images, list) else 0
                })
            
            logger.error(f"Error processing multiple images: {e}")
            raise VisionModelError(f"Error processing multiple images: {str(e)}", cause=e)
    
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
            
            # Check if text generation is supported
            model_type = self.model_manager.model_type
            
            if model_type != 'clip_phi':
                if self.metrics and op_time:
                    self.metrics.stop_operation("generate_text", op_time, success=False)
                    self.metrics.record_vision_error("text_generation_not_supported", {"model": self.model_name})
                
                raise VisionModelError(f"Model {self.model_name} does not support text-only generation")
            
            # Use specified max_tokens if provided, otherwise use default
            max_length = max_tokens if max_tokens is not None else self.max_length
            
            # Generate text with Phi model for clip_phi
            phi_model = self.model.phi_model
            phi_tokenizer = self.model.phi_tokenizer
            
            # Generate text with Phi model
            input_ids = phi_tokenizer(prompt, return_tensors="pt").input_ids.to(self.model_manager.device)
            
            with torch.no_grad():
                outputs = phi_model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + max_length,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty
                )
            
            # Decode the response
            response_text = phi_tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            
            # For text-only generation, we assign a reasonable confidence
            confidence = 0.85
            
            # Record metrics
            if self.metrics:
                # Record total processing time
                total_time_ms = (time.time() - start_time) * 1000
                
                # Record inference metrics
                self.metrics.record_inference(
                    inference_time_ms=total_time_ms,
                    confidence=confidence
                )
                
                # Record model usage
                self.metrics.record_model_usage(self.model_name, {
                    "prompt_length": len(prompt),
                    "text_only": True,
                    "max_length": max_length,
                    "response_length": len(response_text),
                    "temperature": self.temperature
                })
                
                # Stop operation timing
                if op_time:
                    self.metrics.stop_operation("generate_text", op_time, success=True)
            
            return response_text.strip(), confidence
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("generate_text", op_time, success=False)
                self.metrics.record_vision_error("text_generation_error", {"error": str(e)})
            
            logger.error(f"Error generating text with {self.model_name}: {e}")
            raise VisionModelError(f"Error generating text: {str(e)}", cause=e)
    
    def list_speakers(self) -> List[str]:
        """Return a list of available speakers for the vision model."""
        return []  # Lightweight vision models don't have speakers
    
    def list_languages(self) -> List[str]:
        """Return a list of supported languages for the vision model."""
        # Most lightweight models primarily support English
        return ["en"] 