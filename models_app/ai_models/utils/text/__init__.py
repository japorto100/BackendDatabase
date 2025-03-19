"""
Text Utilities Module

Provides utilities for processing and managing text for LLM providers.
"""

# Import utilities from parent directory for backward compatibility
from .. import chunking
from .. import prompt_templates
from .. import results_combiner
from .. import token_management

# Import common utilities for LLM
from ..common.handlers import handle_llm_errors, handle_provider_connection
from ..common.errors import ProviderConnectionError, TokenLimitError, ModelUnavailableError
from ..common.metrics import get_llm_metrics
from ..common.config import get_llm_config

# Re-export for direct imports from this module
chunk_text_for_model = chunking.chunk_text_for_model
create_document_prompt = prompt_templates.create_document_prompt
create_summary_prompt = prompt_templates.create_summary_prompt
combine_chunk_results = results_combiner.combine_chunk_results
determine_token_strategy = token_management.determine_token_strategy
select_relevant_chunks = token_management.select_relevant_chunks

# Map old error handler to new one for backward compatibility
error_handler = handle_llm_errors

__all__ = [
    # Text processing
    'chunk_text_for_model',
    'create_document_prompt',
    'create_summary_prompt',
    'combine_chunk_results',
    'determine_token_strategy',
    'select_relevant_chunks',
    
    # Error handling (both new and legacy for compatibility)
    'error_handler',
    'handle_llm_errors',
    'handle_provider_connection',
    
    # Error types
    'ProviderConnectionError',
    'TokenLimitError',
    'ModelUnavailableError',
    
    # Configuration and metrics
    'get_llm_config',
    'get_llm_metrics'
]