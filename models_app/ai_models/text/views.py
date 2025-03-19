"""
Text Models Views

This module provides views for directly working with text models via API endpoints.
"""

import logging
import json
import os
import time
from typing import Dict, List, Any, Optional
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser
from utils.config_handler import config

# Provider services
from .openai.service import OpenAILLMService
from .anthropic.service import AnthropicLLMService
from .qwen.service import QwenLLMService
from .lightweight.service import LightweightLLMService
from .deepseek.service import DeepSeekLLMService
from .local_generic.service import LocalGenericLLMService

# Provider model managers for dynamic model lists
from .openai.model_manager import OpenAILLMModelManager
from .anthropic.model_manager import AnthropicLLMModelManager
from .qwen.model_manager import QwenLLMModelManager
from .lightweight.model_manager import LightweightLLMModelManager
from .deepseek.model_manager import DeepSeekLLMModelManager
from .local_generic.model_manager import LocalGenericLLMModelManager

# Factory for provider creation
from .provider_factory import ProviderFactory

# Common utilities for error handling and metrics
from models_app.ai_models.utils.common.handlers import handle_llm_errors, handle_provider_connection
from models_app.ai_models.utils.common.errors import ProviderConnectionError, ModelUnavailableError
from models_app.ai_models.utils.common.metrics import get_llm_metrics

logger = logging.getLogger(__name__)

# Create API metrics collector for tracking endpoint performance
api_metrics = get_llm_metrics("text_api")

@csrf_exempt
@api_view(['POST'])
@parser_classes([JSONParser])
def openai_service(request):
    """API endpoint for OpenAI text service"""
    # Start metrics collection
    op_time = api_metrics.start_operation("openai_endpoint")
    
    try:
        data = request.data
        prompt = data.get('prompt', '')
        model_name = data.get('model', 'gpt-4-turbo')
        
        if not prompt:
            api_metrics.record_llm_error("missing_prompt", {"endpoint": "openai"})
            api_metrics.stop_operation("openai_endpoint", op_time, success=False)
            return JsonResponse({'error': 'Prompt is required'}, status=400)
        
        # Create OpenAI service with configuration
        config = {
            'model_name': model_name,  # Standardized to model_name
            'max_tokens': data.get('max_tokens', 1000),
            'temperature': data.get('temperature', 0.7)
        }
        
        # Measure service creation and text generation time
        start_time = time.time()
        service = OpenAILLMService(config)
        
        # Generate text
        response_text, confidence = service.generate_text(prompt)
        
        # Record API timing
        processing_time_ms = (time.time() - start_time) * 1000
        api_metrics.record_custom_metric("api_processing", "response_time_ms", processing_time_ms)
        api_metrics.record_custom_metric("api_usage", "model", model_name)
        api_metrics.stop_operation("openai_endpoint", op_time, success=True)
        
        return JsonResponse({
            'response': response_text,
            'confidence': confidence,
            'model': model_name,
            'processing_time_ms': processing_time_ms
        })
        
    except Exception as e:
        logger.error(f"Error in OpenAI text service: {str(e)}")
        api_metrics.record_llm_error("openai_endpoint_error", {"error": str(e)})
        api_metrics.stop_operation("openai_endpoint", op_time, success=False)
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([JSONParser])
def anthropic_service(request):
    """API endpoint for Anthropic text service"""
    # Start metrics collection
    op_time = api_metrics.start_operation("anthropic_endpoint")
    
    try:
        data = request.data
        prompt = data.get('prompt', '')
        model_name = data.get('model', 'claude-3-opus-20240229')
        
        if not prompt:
            api_metrics.record_llm_error("missing_prompt", {"endpoint": "anthropic"})
            api_metrics.stop_operation("anthropic_endpoint", op_time, success=False)
            return JsonResponse({'error': 'Prompt is required'}, status=400)
        
        # Create Anthropic service with configuration
        config = {
            'model_name': model_name,  # Standardized to model_name
            'max_tokens': data.get('max_tokens', 1000),
            'temperature': data.get('temperature', 0.7)
        }
        
        # Measure service creation and text generation time
        start_time = time.time()
        service = AnthropicLLMService(config)
        
        # Generate text
        response_text, confidence = service.generate_text(prompt)
        
        # Record API timing
        processing_time_ms = (time.time() - start_time) * 1000
        api_metrics.record_custom_metric("api_processing", "response_time_ms", processing_time_ms)
        api_metrics.record_custom_metric("api_usage", "model", model_name)
        api_metrics.stop_operation("anthropic_endpoint", op_time, success=True)
        
        return JsonResponse({
            'response': response_text,
            'confidence': confidence,
            'model': model_name,
            'processing_time_ms': processing_time_ms
        })
        
    except Exception as e:
        logger.error(f"Error in Anthropic text service: {str(e)}")
        api_metrics.record_llm_error("anthropic_endpoint_error", {"error": str(e)})
        api_metrics.stop_operation("anthropic_endpoint", op_time, success=False)
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([JSONParser])
def qwen_service(request):
    """API endpoint for Qwen text service"""
    try:
        data = request.data
        prompt = data.get('prompt', '')
        model_name = data.get('model', 'Qwen/QwQ-32B')
        
        if not prompt:
            return JsonResponse({'error': 'Prompt is required'}, status=400)
        
        # Create Qwen service with configuration
        config = {
            'model': model_name,
            'max_tokens': data.get('max_tokens', 1000),
            'temperature': data.get('temperature', 0.7),
            'quantization_level': data.get('quantization_level', '4bit')
        }
        
        service = QwenLLMService(config)
        
        # Generate text
        response_text, confidence = service.generate_text(prompt)
        
        return JsonResponse({
            'response': response_text,
            'confidence': confidence,
            'model': model_name
        })
        
    except Exception as e:
        logger.error(f"Error in Qwen text service: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([JSONParser])
def lightweight_service(request):
    """API endpoint for Lightweight text service"""
    try:
        data = request.data
        prompt = data.get('prompt', '')
        model_name = data.get('model', 'microsoft/phi-3-mini-4k-instruct')
        
        if not prompt:
            return JsonResponse({'error': 'Prompt is required'}, status=400)
        
        # Create Lightweight service with configuration
        config = {
            'model': model_name,
            'max_tokens': data.get('max_tokens', 1000),
            'temperature': data.get('temperature', 0.7),
            'quantization_level': data.get('quantization_level', '4bit')
        }
        
        service = LightweightLLMService(config)
        
        # Generate text
        response_text, confidence = service.generate_text(prompt)
        
        return JsonResponse({
            'response': response_text,
            'confidence': confidence,
            'model': model_name
        })
        
    except Exception as e:
        logger.error(f"Error in Lightweight text service: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([JSONParser])
def deepseek_service(request):
    """API endpoint for DeepSeek text service"""
    try:
        data = request.data
        prompt = data.get('prompt', '')
        model_name = data.get('model', 'deepseek-ai/deepseek-v3-7b')
        
        if not prompt:
            return JsonResponse({'error': 'Prompt is required'}, status=400)
        
        # Create DeepSeek service with configuration
        config = {
            'model': model_name,
            'max_tokens': data.get('max_tokens', 1000),
            'temperature': data.get('temperature', 0.7),
            'quantization_level': data.get('quantization_level', '4bit')
        }
        
        service = DeepSeekLLMService(config)
        
        # Generate text
        response_text, confidence = service.generate_text(prompt)
        
        return JsonResponse({
            'response': response_text,
            'confidence': confidence,
            'model': model_name
        })
        
    except Exception as e:
        logger.error(f"Error in DeepSeek text service: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([JSONParser])
def local_generic_service(request):
    """API endpoint for Local Generic text service"""
    try:
        data = request.data
        prompt = data.get('prompt', '')
        model_name = data.get('model', None)  # Let the manager choose default if not specified
        
        if not prompt:
            return JsonResponse({'error': 'Prompt is required'}, status=400)
        
        # Create Local Generic service with configuration
        config = {
            'max_tokens': data.get('max_tokens', 1000),
            'temperature': data.get('temperature', 0.7),
            'quantization_level': data.get('quantization_level', '4bit')
        }
        
        # Only add model if specified
        if model_name:
            config['model'] = model_name
        
        service = LocalGenericLLMService(config)
        
        # Generate text
        response_text, confidence = service.generate_text(prompt)
        
        # Get the model info
        model_info = service.get_model_info()
        
        return JsonResponse({
            'response': response_text,
            'confidence': confidence,
            'model': model_info.get('name', 'Unknown model')
        })
        
    except Exception as e:
        logger.error(f"Error in Local Generic text service: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([JSONParser])
@handle_llm_errors  # Add error handler decorator
def auto_select_service(request):
    """
    API endpoint that automatically selects the best text model for the task.
    
    This endpoint uses the ProviderFactory to determine the optimal model
    based on the requested task and available hardware resources.
    """
    # Start metrics collection
    op_time = api_metrics.start_operation("auto_select_endpoint")
    
    try:
        data = request.data
        prompt = data.get('prompt', '')
        task = data.get('task', 'chat')  # chat, document_processing, reasoning, summarization
        
        if not prompt:
            api_metrics.record_llm_error("missing_prompt", {"endpoint": "auto_select"})
            api_metrics.stop_operation("auto_select_endpoint", op_time, success=False)
            return JsonResponse({'error': 'Prompt is required'}, status=400)
        
        # Use factory to get recommended provider
        factory = ProviderFactory()
        
        # Get device capabilities for better model selection
        device_capabilities = factory.detect_device_capabilities()
        
        # Get recommended provider config for the task
        provider_config = factory.get_recommended_provider(task, device_capabilities)
        
        # Record the selected provider type
        api_metrics.record_custom_metric("api_usage", "selected_provider", provider_config.get('provider_type'))
        
        # Create provider using the config
        start_time = time.time()
        provider = factory.create_provider(provider_config)
        
        # Generate text
        response_text, confidence = provider.generate_text(prompt)
        
        # Get model info
        model_info = provider.get_model_info()
        
        # Record API timing
        processing_time_ms = (time.time() - start_time) * 1000
        api_metrics.record_custom_metric("api_processing", "response_time_ms", processing_time_ms)
        api_metrics.record_custom_metric("api_usage", "task", task)
        api_metrics.stop_operation("auto_select_endpoint", op_time, success=True)
        
        return JsonResponse({
            'response': response_text,
            'confidence': confidence,
            'selected_provider': provider_config.get('provider_type'),
            'selected_model': model_info.get('name', provider_config.get('model_name', 'auto')),
            'task': task,
            'processing_time_ms': processing_time_ms,
            'device_capabilities': {
                'has_gpu': device_capabilities.get('has_gpu', False),
                'gpu_name': device_capabilities.get('gpu_name', 'N/A'),
                'gpu_memory': device_capabilities.get('gpu_memory_mb', 0),
                'platform': device_capabilities.get('platform', 'unknown')
            }
        })
        
    except Exception as e:
        logger.error(f"Error in auto-select text service: {str(e)}")
        api_metrics.record_llm_error("auto_select_endpoint_error", {"error": str(e)})
        api_metrics.stop_operation("auto_select_endpoint", op_time, success=False)
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@handle_llm_errors  # Add error handler decorator
def document_service(request):
    """
    API endpoint for processing documents with text models.
    
    This endpoint accepts document uploads and processes them with the selected
    text model. It can extract text, analyze content, and answer queries about
    the document.
    """
    # Start metrics collection
    op_time = api_metrics.start_operation("document_endpoint")
    
    try:
        # Check if document file is provided
        if 'document' not in request.FILES:
            api_metrics.record_llm_error("missing_document", {"endpoint": "document_service"})
            api_metrics.stop_operation("document_endpoint", op_time, success=False)
            return JsonResponse({'error': 'Document file is required'}, status=400)
        
        document = request.FILES['document']
        query = request.data.get('query', None)
        model_name = request.data.get('model_name', None)  # Standardized to model_name
        provider_type = request.data.get('provider_type', None)
        
        # Record document metadata
        document_size = document.size
        document_type = os.path.splitext(document.name)[1]
        api_metrics.record_custom_metric("document_processing", "document_size_bytes", document_size)
        api_metrics.record_custom_metric("document_processing", "document_type", document_type)
        
        # Save document temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(document.name)[1]) as temp_file:
            for chunk in document.chunks():
                temp_file.write(chunk)
            document_path = temp_file.name
            
        # Extract text from document
        from models_app.utilities.document_utils import extract_text_from_document
        start_time = time.time()
        document_text = extract_text_from_document(document_path)
        extraction_time_ms = (time.time() - start_time) * 1000
        api_metrics.record_custom_metric("document_processing", "extraction_time_ms", extraction_time_ms)
        
        # Create document dict
        document_data = {
            'text': document_text,
            'path': document_path,
            'name': document.name,
            'type': os.path.splitext(document.name)[1]
        }
        
        # Configure provider
        config = {}
        if model_name:
            config['model_name'] = model_name  # Standardized to model_name
        if provider_type:
            config['provider_type'] = provider_type
            
        # Create provider through factory
        factory = ProviderFactory()
        provider = factory.create_provider(config)
        
        # Process document
        start_time = time.time()
        result = provider.process_document(document_data, query)
        processing_time_ms = (time.time() - start_time) * 1000
        api_metrics.record_custom_metric("document_processing", "processing_time_ms", processing_time_ms)
        
        # Clean up
        try:
            os.unlink(document_path)
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {document_path}: {str(e)}")
        
        # Add processing time to result
        result['processing_time_ms'] = processing_time_ms
        api_metrics.stop_operation("document_endpoint", op_time, success=True)
            
        # Return result
        return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"Error in document service: {str(e)}")
        api_metrics.record_llm_error("document_endpoint_error", {"error": str(e)})
        api_metrics.stop_operation("document_endpoint", op_time, success=False)
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['GET'])
def list_services(request):
    """
    List all available text services with capabilities.
    
    This endpoint provides information about all available text services,
    including the models they support, their capabilities, and hardware
    requirements.
    """
    # Start metrics collection
    op_time = api_metrics.start_operation("list_services_endpoint")
    
    try:
        # Get all available services by dynamically loading model information
        services = []
        
        # Add OpenAI service
        try:
            openai_manager = OpenAILLMModelManager()
            openai_models = [model['id'] for model in openai_manager.list_available_models()]
            services.append({
                'id': 'openai',
                'name': 'OpenAI',
                'models': openai_models,
                'provider': 'OpenAI',
                'capabilities': ['chat', 'reasoning', 'document_processing'],
                'requires_api_key': True
            })
        except Exception as e:
            logger.warning(f"Could not load OpenAI model information: {str(e)}")
            api_metrics.record_llm_error("model_listing_error", {"provider": "openai", "error": str(e)})
        
        # Add Anthropic service
        try:
            anthropic_manager = AnthropicLLMModelManager()
            anthropic_models = [model['id'] for model in anthropic_manager.list_available_models()]
            services.append({
                'id': 'anthropic',
                'name': 'Anthropic Claude',
                'models': anthropic_models,
                'provider': 'Anthropic',
                'capabilities': ['chat', 'reasoning', 'document_processing'],
                'requires_api_key': True
            })
        except Exception as e:
            logger.warning(f"Could not load Anthropic model information: {str(e)}")
            api_metrics.record_llm_error("model_listing_error", {"provider": "anthropic", "error": str(e)})
        
        # Add Qwen service
        try:
            qwen_manager = QwenLLMModelManager()
            qwen_models = [name for name in qwen_manager.AVAILABLE_MODELS.keys()]
            services.append({
                'id': 'qwen',
                'name': 'Qwen / QwQ',
                'models': qwen_models,
                'provider': 'Qwen',
                'capabilities': ['chat', 'reasoning', 'document_processing'],
                'requires_api_key': False,
                'requires_gpu': True
            })
        except Exception as e:
            logger.warning(f"Could not load Qwen model information: {str(e)}")
            api_metrics.record_llm_error("model_listing_error", {"provider": "qwen", "error": str(e)})
        
        # Add Lightweight service
        try:
            lightweight_manager = LightweightLLMModelManager()
            lightweight_models = [name for name in lightweight_manager.AVAILABLE_MODELS.keys()]
            services.append({
                'id': 'lightweight',
                'name': 'Lightweight Models',
                'models': lightweight_models,
                'provider': 'Local',
                'capabilities': ['chat', 'summarization'],
                'requires_api_key': False
            })
        except Exception as e:
            logger.warning(f"Could not load Lightweight model information: {str(e)}")
            api_metrics.record_llm_error("model_listing_error", {"provider": "lightweight", "error": str(e)})
        
        # Add DeepSeek service
        try:
            deepseek_manager = DeepSeekLLMModelManager()
            deepseek_models = [name for name in deepseek_manager.AVAILABLE_MODELS.keys()]
            services.append({
                'id': 'deepseek',
                'name': 'DeepSeek',
                'models': deepseek_models,
                'provider': 'DeepSeek',
                'capabilities': ['chat', 'code_generation', 'document_processing'],
                'requires_api_key': False,
                'requires_gpu': True
            })
        except Exception as e:
            logger.warning(f"Could not load DeepSeek model information: {str(e)}")
            api_metrics.record_llm_error("model_listing_error", {"provider": "deepseek", "error": str(e)})
        
        # Add Local Generic service
        try:
            local_generic_manager = LocalGenericLLMModelManager()
            local_generic_models = local_generic_manager.get_recommended_models()
            local_generic_model_names = [model.get('name') for model in local_generic_models]
            services.append({
                'id': 'local_generic',
                'name': 'Local Generic Models',
                'models': local_generic_model_names,
                'provider': 'Local',
                'capabilities': ['chat', 'summarization', 'document_processing'],
                'requires_api_key': False,
                'can_add_custom_models': True
            })
        except Exception as e:
            logger.warning(f"Could not load Local Generic model information: {str(e)}")
            api_metrics.record_llm_error("model_listing_error", {"provider": "local_generic", "error": str(e)})
        
        # Check for available hardware
        factory = ProviderFactory()
        device_capabilities = factory.detect_device_capabilities()
        
        # Mark services that are available with current hardware
        for service in services:
            if service.get('requires_gpu', False):
                service['available'] = device_capabilities.get('has_gpu', False)
            elif service.get('requires_api_key', False):
                # Check for API keys
                if service['id'] == 'openai':
                    service['available'] = bool(os.environ.get('OPENAI_API_KEY', ''))
                elif service['id'] == 'anthropic':
                    service['available'] = bool(os.environ.get('ANTHROPIC_API_KEY', ''))
            else:
                service['available'] = True
        
        # Get default text service from config
        default_service = config.get("MODELS", "DEFAULT_TEXT_SERVICE", "auto")
        
        # Record metrics
        api_metrics.record_custom_metric("api_usage", "service_listing_count", len(services))
        api_metrics.stop_operation("list_services_endpoint", op_time, success=True)
        
        return JsonResponse({
            'services': services,
            'default_service': default_service,
            'device_capabilities': {
                'has_gpu': device_capabilities.get('has_gpu', False),
                'gpu_name': device_capabilities.get('gpu_name', 'N/A'),
                'gpu_memory': device_capabilities.get('gpu_memory_mb', 0),
                'platform': device_capabilities.get('platform', 'unknown')
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing text services: {str(e)}")
        api_metrics.record_llm_error("service_listing_error", {"error": str(e)})
        api_metrics.stop_operation("list_services_endpoint", op_time, success=False)
        return JsonResponse({'error': str(e)}, status=500) 