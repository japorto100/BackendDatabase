"""
Common Utilities for AI Models

This package provides utilities for all AI model types:
- Text processing utilities
- Common error handling, metrics, and configuration
- Utility functions for providers
"""

# Text-specific utilities
from .token_management import determine_token_strategy, select_relevant_chunks
from .prompt_templates import create_document_prompt, create_summary_prompt
from .chunking import (
    fixed_size_chunking, 
    semantic_chunking, 
    adaptive_chunking, 
    chunk_text_for_model
)
from .results_combiner import (
    combine_chunk_results,
    _simple_combine,
    _weighted_combine,
    _hierarchical_combine
)

# Import common utilities
from .common.errors import (
    ModelError, ModelNotFoundError, LLMError, ProviderConnectionError, 
    RateLimitError, TokenLimitError, ProviderResponseError, ModelUnavailableError
)
from .common.config import (
    BaseConfig, LLMConfig, ConfigManager, get_llm_config, set_llm_config
)
from .common.metrics import (
    MetricsCollector, LLMMetricsCollector, get_llm_metrics, export_all_metrics
)
from .common.handlers import (
    default_error_handler, handle_model_errors, handle_llm_errors, 
    handle_provider_connection, handle_rate_limits
)
from .common.ai_base_service import (
    BaseModelService, ModelRegistry, register_service_type, 
    get_service, create_service
)

__all__ = [
    # Text utilities
    'determine_token_strategy',
    'select_relevant_chunks',
    'create_document_prompt',
    'create_summary_prompt',
    'fixed_size_chunking',
    'semantic_chunking',
    'adaptive_chunking',
    'chunk_text_for_model',
    'combine_chunk_results',
    
    # Common error types
    'ModelError',
    'ModelNotFoundError',
    'LLMError',
    'ProviderConnectionError',
    'RateLimitError',
    'TokenLimitError',
    'ProviderResponseError',
    'ModelUnavailableError',
    
    # Configuration
    'BaseConfig',
    'LLMConfig',
    'ConfigManager',
    'get_llm_config',
    'set_llm_config',
    
    # Metrics
    'MetricsCollector',
    'LLMMetricsCollector',
    'get_llm_metrics',
    'export_all_metrics',
    
    # Handlers
    'default_error_handler',
    'handle_model_errors',
    'handle_llm_errors',
    'handle_provider_connection',
    'handle_rate_limits',
    
    # Service utilities
    'BaseModelService',
    'ModelRegistry',
    'register_service_type',
    'get_service',
    'create_service'
] 