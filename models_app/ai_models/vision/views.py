"""
Vision Model API Views

Views for processing images with various vision models.
"""

import logging
import os
import base64
import tempfile
import time
from io import BytesIO
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image

from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework.request import Request

from models_app.ai_models.vision.vision_factory import VisionProviderFactory
from models_app.ai_models.utils.common.errors import (
    VisionModelError,
    ImageProcessingError,
    MultiImageProcessingError,
    ProviderNotFoundError,
    ModelUnavailableError
)
from models_app.ai_models.utils.common.handlers import (
    handle_vision_errors,
    handle_image_processing_errors,
    vision_processing
)
from models_app.ai_models.utils.common.metrics import get_vision_metrics

logger = logging.getLogger(__name__)
metrics = get_vision_metrics("api_views")

@csrf_exempt
@require_http_methods(["POST"])
@handle_vision_errors
def process_image(request: HttpRequest) -> JsonResponse:
    """
    Process an image with a vision model and return the results.
    
    Expected request format:
    {
        "provider": "gemini",
        "model": "gemini-pro-vision",
        "image": "<base64-encoded image or URL>",
        "prompt": "Describe this image in detail.",
        "options": {
            "temperature": 0.7,
            "max_tokens": 512,
            ...
        }
    }
    
    Returns:
        JsonResponse: The processed results with confidence score
    """
    # Start metrics collection
    start_time = time.time()
    operation_time = metrics.start_operation("process_image_api")
    
    try:
        # Parse request body
        import json
        if isinstance(request, Request):
            data = request.data
        else:
            data = json.loads(request.body)
        
        provider_name = data.get('provider')
        model_name = data.get('model')
        image_data = data.get('image')
        prompt = data.get('prompt', 'Describe this image in detail.')
        options = data.get('options', {})
        
        # Validate required fields
        if not provider_name:
            return JsonResponse({'error': 'Provider name is required'}, status=400)
        if not model_name:
            return JsonResponse({'error': 'Model name is required'}, status=400)
        if not image_data:
            return JsonResponse({'error': 'Image data is required'}, status=400)
        
        # Get vision provider
        provider = VisionProviderFactory.get_provider(
            provider_name=provider_name,
            model_name=model_name,
            **options
        )
        
        # Process the image
        result, confidence = provider.process_image(image_data, prompt)
        
        # Record API metrics
        total_time_ms = (time.time() - start_time) * 1000
        metrics.record_inference(
            inference_time_ms=total_time_ms, 
            confidence=confidence,
            model_name=f"{provider_name}_{model_name}"
        )
        metrics.record_model_usage(model_name, {
            "provider": provider_name,
            "prompt_length": len(prompt),
            "result_length": len(result),
            "source": "api"
        })
        metrics.stop_operation("process_image_api", operation_time, success=True)
        
        # Return results
        return JsonResponse({
            'result': result,
            'confidence': confidence,
            'model': model_name,
            'provider': provider_name
        })
        
    except ProviderNotFoundError as e:
        metrics.stop_operation("process_image_api", operation_time, success=False)
        metrics.record_vision_error("provider_not_found", {"error": str(e)})
        logger.error(f"Provider not found: {e}")
        return JsonResponse({'error': str(e)}, status=404)
        
    except (VisionModelError, ImageProcessingError, ModelUnavailableError) as e:
        metrics.stop_operation("process_image_api", operation_time, success=False)
        metrics.record_vision_error("vision_processing_error", {"error": str(e)})
        logger.error(f"Vision processing error: {e}")
        return JsonResponse({'error': str(e)}, status=500)
        
    except Exception as e:
        metrics.stop_operation("process_image_api", operation_time, success=False)
        metrics.record_vision_error("unexpected_error", {"error": str(e)})
        logger.error(f"Unexpected error in process_image: {e}")
        return JsonResponse({'error': f"Unexpected error: {str(e)}"}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
@handle_vision_errors
def process_multiple_images(request: HttpRequest) -> JsonResponse:
    """
    Process multiple images with a vision model and return the results.
    
    Expected request format:
    {
        "provider": "gemini",
        "model": "gemini-pro-vision",
        "images": ["<base64-encoded image or URL>", ...],
        "prompt": "Compare these images.",
        "options": {
            "temperature": 0.7,
            "max_tokens": 512,
            ...
        }
    }
    
    Returns:
        JsonResponse: The processed results with confidence score
    """
    # Start metrics collection
    start_time = time.time()
    operation_time = metrics.start_operation("process_multiple_images_api")
    
    try:
        # Parse request body
        import json
        if isinstance(request, Request):
            data = request.data
        else:
            data = json.loads(request.body)
        
        provider_name = data.get('provider')
        model_name = data.get('model')
        images = data.get('images', [])
        prompt = data.get('prompt', 'Compare these images.')
        options = data.get('options', {})
        
        # Validate required fields
        if not provider_name:
            return JsonResponse({'error': 'Provider name is required'}, status=400)
        if not model_name:
            return JsonResponse({'error': 'Model name is required'}, status=400)
        if not images or not isinstance(images, list) or len(images) == 0:
            return JsonResponse({'error': 'At least one image is required'}, status=400)
        
        # Get vision provider
        provider = VisionProviderFactory.get_provider(
            provider_name=provider_name,
            model_name=model_name,
            **options
        )
        
        # Check if the provider supports multiple images
        if not provider.supports_multiple_images:
            return JsonResponse({
                'error': f"Provider '{provider_name}' does not support processing multiple images"
            }, status=400)
        
        # Process the images
        result, confidence = provider.process_multiple_images(images, prompt)
        
        # Record API metrics
        total_time_ms = (time.time() - start_time) * 1000
        metrics.record_inference(
            inference_time_ms=total_time_ms, 
            confidence=confidence,
            model_name=f"{provider_name}_{model_name}"
        )
        metrics.record_multi_image_processed(
            image_count=len(images),
            processing_time_ms=total_time_ms
        )
        metrics.record_model_usage(model_name, {
            "provider": provider_name,
            "prompt_length": len(prompt),
            "result_length": len(result),
            "source": "api",
            "image_count": len(images)
        })
        metrics.stop_operation("process_multiple_images_api", operation_time, success=True)
        
        # Return results
        return JsonResponse({
            'result': result,
            'confidence': confidence,
            'model': model_name,
            'provider': provider_name,
            'image_count': len(images)
        })
        
    except ProviderNotFoundError as e:
        metrics.stop_operation("process_multiple_images_api", operation_time, success=False)
        metrics.record_vision_error("provider_not_found", {"error": str(e)})
        logger.error(f"Provider not found: {e}")
        return JsonResponse({'error': str(e)}, status=404)
        
    except (VisionModelError, ImageProcessingError, MultiImageProcessingError, ModelUnavailableError) as e:
        metrics.stop_operation("process_multiple_images_api", operation_time, success=False)
        metrics.record_vision_error("vision_processing_error", {"error": str(e)})
        logger.error(f"Vision processing error: {e}")
        return JsonResponse({'error': str(e)}, status=500)
        
    except Exception as e:
        metrics.stop_operation("process_multiple_images_api", operation_time, success=False)
        metrics.record_vision_error("unexpected_error", {"error": str(e)})
        logger.error(f"Unexpected error in process_multiple_images: {e}")
        return JsonResponse({'error': f"Unexpected error: {str(e)}"}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def list_providers(request: HttpRequest) -> JsonResponse:
    """
    List all available vision providers.
    
    Returns:
        JsonResponse: A list of available vision providers
    """
    try:
        providers = VisionProviderFactory.list_providers()
        return JsonResponse({'providers': providers})
    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["GET"])
def list_models(request: HttpRequest) -> JsonResponse:
    """
    List available models for all providers or a specific provider.
    
    Query parameters:
    - provider (optional): Name of a specific provider to list models for
    
    Returns:
        JsonResponse: Dictionary mapping provider names to lists of model names
    """
    try:
        provider_name = request.GET.get('provider')
        models = VisionProviderFactory.list_models(provider_name)
        return JsonResponse({'models': models})
    except ProviderNotFoundError as e:
        logger.error(f"Provider not found: {e}")
        return JsonResponse({'error': str(e)}, status=404)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
@handle_image_processing_errors
def preprocess_image(request: HttpRequest) -> JsonResponse:
    """
    Preprocess an image without sending it to a vision model.
    
    Useful for:
    - Verifying image formatting before sending to a model
    - Getting image metadata
    - Converting image formats
    
    Expected request format:
    {
        "image": "<base64-encoded image or URL>",
        "format": "JPEG",  # Optional: output format
        "max_size": 1024,  # Optional: max dimension
        "return_data": true  # Optional: include processed image in response
    }
    
    Returns:
        JsonResponse: Information about the processed image
    """
    try:
        # Parse request body
        import json
        if isinstance(request, Request):
            data = request.data
        else:
            data = json.loads(request.body)
        
        image_data = data.get('image')
        output_format = data.get('format', 'JPEG')
        max_size = data.get('max_size', 1024)
        return_data = data.get('return_data', False)
        
        # Validate required fields
        if not image_data:
            return JsonResponse({'error': 'Image data is required'}, status=400)
        
        # Process the image using the base vision provider
        from models_app.ai_models.vision.base_vision_provider import BaseVisionProvider
        
        # Start metadata tracking
        start_time = time.time()
        
        # Create dummy provider to use preprocessing methods
        class DummyProvider(BaseVisionProvider):
            def process_image(self, *args, **kwargs):
                pass
        
        dummy_provider = DummyProvider({"max_image_size": max_size})
        
        # Preprocess the image
        pil_image = dummy_provider.preprocess_image(image_data)
        
        # Get image metadata
        width, height = pil_image.size
        mode = pil_image.mode
        format_name = pil_image.format or output_format
        
        # Prepare response
        response_data = {
            'width': width,
            'height': height,
            'format': format_name,
            'mode': mode,
            'processing_time_ms': (time.time() - start_time) * 1000
        }
        
        # Include processed image if requested
        if return_data:
            # Convert to the specified format
            buffer = BytesIO()
            pil_image.save(buffer, format=output_format)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            response_data['processed_image'] = f"data:image/{output_format.lower()};base64,{img_str}"
        
        return JsonResponse(response_data)
        
    except ImageProcessingError as e:
        logger.error(f"Image processing error: {e}")
        return JsonResponse({'error': str(e)}, status=400)
    except Exception as e:
        logger.error(f"Unexpected error in preprocess_image: {e}")
        return JsonResponse({'error': f"Unexpected error: {str(e)}"}, status=500) 