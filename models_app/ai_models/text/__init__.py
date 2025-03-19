"""
Text AI Models module.

This module provides classes for working with text-based AI language models.
It includes provider services, model managers, and factory utilities.
"""

# Base provider and factory
from models_app.ai_models.text.base_text_provider import BaseLLMProvider
from models_app.ai_models.text.provider_factory import ProviderFactory, register_text_providers

# Provider services
from models_app.ai_models.text.openai.service import OpenAILLMService
from models_app.ai_models.text.anthropic.service import AnthropicLLMService
from models_app.ai_models.text.qwen.service import QwenLLMService
from models_app.ai_models.text.lightweight.service import LightweightLLMService
from models_app.ai_models.text.deepseek.service import DeepSeekLLMService
from models_app.ai_models.text.local_generic.service import LocalGenericLLMService

# Model managers
from models_app.ai_models.text.openai.model_manager import OpenAILLMModelManager
from models_app.ai_models.text.anthropic.model_manager import AnthropicLLMModelManager
from models_app.ai_models.text.qwen.model_manager import QwenLLMModelManager
from models_app.ai_models.text.lightweight.model_manager import LightweightLLMModelManager
from models_app.ai_models.text.deepseek.model_manager import DeepSeekLLMModelManager
from models_app.ai_models.text.local_generic.model_manager import LocalGenericLLMModelManager

# Common utilities - imported here for convenience so LLM services can access
# error handling decorators and metrics collection without additional imports
from models_app.ai_models.utils.common.handlers import handle_llm_errors, handle_provider_connection
from models_app.ai_models.utils.common.metrics import get_llm_metrics

# Define public API
__all__ = [
    # Base classes
    'BaseLLMProvider',
    'ProviderFactory',
    'register_text_providers',
    
    # Provider services
    'OpenAILLMService',
    'AnthropicLLMService',
    'QwenLLMService',
    'LightweightLLMService',
    'DeepSeekLLMService',
    'LocalGenericLLMService',
    
    # Model managers
    'OpenAILLMModelManager',
    'AnthropicLLMModelManager',
    'QwenLLMModelManager',
    'LightweightLLMModelManager',
    'DeepSeekLLMModelManager',
    'LocalGenericLLMModelManager',
    
    # Common utilities
    'handle_llm_errors',
    'handle_provider_connection',
    'get_llm_metrics'
] 