"""
Local Generic LLM Provider

This module provides access to generic local large language models.
"""

from models_app.ai_models.text.local_generic.service import LocalGenericLLMService
from models_app.ai_models.text.local_generic.model_manager import LocalGenericLLMModelManager

# Define public API
__all__ = [
    'LocalGenericLLMService',
    'LocalGenericLLMModelManager'
] 