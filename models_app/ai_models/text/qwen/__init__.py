"""
Qwen LLM Provider

This module provides access to Qwen and QwQ large language models.
"""

from models_app.ai_models.text.qwen.service import QwenLLMService
from models_app.ai_models.text.qwen.model_manager import QwenLLMModelManager

# Define public API
__all__ = [
    'QwenLLMService',
    'QwenLLMModelManager'
] 