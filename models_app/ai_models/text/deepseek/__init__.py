"""
DeepSeek LLM Provider

This module provides access to DeepSeek large language models.
"""

from models_app.ai_models.text.deepseek.service import DeepSeekLLMService
from models_app.ai_models.text.deepseek.model_manager import DeepSeekLLMModelManager

# Define public API
__all__ = [
    'DeepSeekLLMService',
    'DeepSeekLLMModelManager'
] 