"""
AnthropicLLMService

Provides text generation services using Anthropic's Claude models.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import json

from models_app.ai_models.text.base_text_provider import BaseLLMProvider
from .model_manager import AnthropicLLMModelManager

# Import from common utilities for consistent error handling
from models_app.ai_models.utils.common.handlers import handle_llm_errors, handle_provider_connection
from models_app.ai_models.utils.common.errors import ProviderConnectionError, TokenLimitError

logger = logging.getLogger(__name__)

class AnthropicLLMService(BaseLLMProvider):
    """
    Service for text generation with Anthropic's Claude models.
    
    This service provides:
    1. Text generation with Claude models
    2. Document processing
    3. Conversational interfaces
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Anthropic LLM service.
        
        Args:
            config: Configuration for the service
        """
        super().__init__(config or {})
        self.model_manager = AnthropicLLMModelManager(config)
        self.system_message = config.get('system_message', 'You are Claude, a helpful AI assistant.')
        self.model_name = config.get('model_name', 'claude-3-opus-20240229')
        
        # Initialize the model if requested
        if config.get('initialize_on_creation', False):
            self.initialize()
    
    @handle_provider_connection        
    def initialize(self) -> bool:
        """
        Initialize the Anthropic LLM service.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Check if service is available
            if not self.model_manager.is_available():
                logger.error("Anthropic service is not available")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error initializing Anthropic service: {str(e)}")
            raise ProviderConnectionError(f"Failed to initialize Anthropic service: {str(e)}", provider="anthropic")
            
    @handle_llm_errors
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate text using the Anthropic model.
        
        Args:
            prompt: Input prompt for text generation
            max_tokens: Maximum number of tokens to generate (optional)
            
        Returns:
            Tuple[str, float]: Generated text and confidence score
        """
        # Start metrics collection if enabled
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("generate_text")
            
        # Initialize if not already initialized
        if not hasattr(self.model_manager, 'client') or self.model_manager.client is None:
            self.initialize()
            
        # Override max_tokens if provided
        effective_max_tokens = max_tokens if max_tokens is not None else self.model_manager.max_tokens
            
        # Prepare request parameters
        params = self.model_manager.prepare_request_parameters(prompt, self.system_message)
        
        try:
            # Make API call
            start_time = time.time()
            response = self.model_manager.client.messages.create(**params)
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Get response text
            response_text = response.content[0].text
            
            # Calculate approximate confidence based on temperature
            temperature = params.get('temperature', 0.7)
            confidence = 1.0 - temperature  # Simplified confidence calculation
            
            # Log timing
            logger.info(f"Anthropic generated {len(response_text)} characters in {generation_time_ms/1000:.2f} seconds")
            
            # Record metrics if enabled
            if self.metrics:
                # Get token usage from the response if available
                # Anthropic may handle tokens differently, so adapt accordingly
                if hasattr(response, 'usage'):
                    prompt_tokens = response.usage.input_tokens
                    completion_tokens = response.usage.output_tokens
                else:
                    # Estimate token usage if not provided by API
                    prompt_tokens = len(prompt) // 4  # Rough estimate
                    completion_tokens = len(response_text) // 4  # Rough estimate
                
                # Record token usage
                self.metrics.record_token_usage(prompt_tokens, completion_tokens)
                
                # Record generation metrics
                self.metrics.record_generation(
                    prompt_length=prompt_tokens,
                    response_length=completion_tokens,
                    total_time_ms=generation_time_ms
                )
                
                # Record model usage
                self.metrics.record_model_usage(
                    self.model_name,
                    parameters={
                        "temperature": temperature,
                        "max_tokens": effective_max_tokens,
                        "top_p": params.get('top_p', 1.0)
                    }
                )
                
                # Record context window usage
                context_window_size = getattr(self.model_manager, 'context_window', 8192)  # Default for Claude
                self.metrics.record_context_window_usage(
                    used_tokens=prompt_tokens + completion_tokens,
                    max_tokens=context_window_size
                )
                
                # Stop operation timer
                self.metrics.stop_operation("generate_text", op_time, success=True)
            
            return response_text, confidence
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("generate_text", op_time, success=False)
                self.metrics.record_llm_error("generation_error", {
                    "error": str(e),
                    "model": self.model_name,
                    "provider": "anthropic"
                })
            logger.error(f"Error generating text with Anthropic: {str(e)}")
            raise
            
    @handle_llm_errors
    def generate_batch(self, prompts: List[str], max_tokens: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum number of tokens to generate (optional)
            
        Returns:
            List[Tuple[str, float]]: List of generated texts and confidence scores
        """
        # Start metrics collection if enabled
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("generate_batch")
        
        try:
            results = []
            
            start_time = time.time()
            for prompt in prompts:
                result = self.generate_text(prompt, max_tokens)
                results.append(result)
                
                # We don't need to record metrics for individual generations
                # as they are already recorded in generate_text
            
            # Record batch-specific metrics
            if self.metrics:
                batch_time_ms = (time.time() - start_time) * 1000
                self.metrics.record_custom_metric(
                    "batch_processing",
                    "prompts_per_batch",
                    len(prompts)
                )
                self.metrics.record_custom_metric(
                    "batch_processing",
                    "batch_time_ms",
                    batch_time_ms
                )
                self.metrics.record_custom_metric(
                    "batch_processing",
                    "avg_time_per_prompt_ms",
                    batch_time_ms / len(prompts) if prompts else 0
                )
                
                # Stop operation timer
                self.metrics.stop_operation("generate_batch", op_time, success=True)
                
            return results
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("generate_batch", op_time, success=False)
                self.metrics.record_llm_error("batch_generation_error", {
                    "error": str(e),
                    "model": self.model_name,
                    "provider": "anthropic",
                    "batch_size": len(prompts)
                })
            logger.error(f"Error generating batch with Anthropic: {str(e)}")
            raise
        
    @handle_llm_errors
    def process_document(self, document: Dict[str, Any], query: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document with the Anthropic model.
        
        Args:
            document: Document to process
            query: Optional query to guide processing
            
        Returns:
            Dict[str, Any]: Processing results
        """
        # Start metrics collection if enabled
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("process_document")
            
        try:
            # Get document text
            if 'text' in document:
                document_text = document['text']
            elif 'path' in document:
                with open(document['path'], 'r', encoding='utf-8') as f:
                    document_text = f.read()
            else:
                error_msg = 'Document must contain text or path'
                if self.metrics:
                    self.metrics.record_llm_error("document_error", {
                        "error": error_msg,
                        "provider": "anthropic"
                    })
                return {'error': error_msg, 'success': False}
                
            # Create prompt for document
            if query:
                prompt = f"Please analyze the following document with this query in mind: {query}\n\nDocument: {document_text}"
            else:
                prompt = f"Please analyze the following document and extract key information:\n\nDocument: {document_text}"
                
            # Generate response
            start_time = time.time()
            response, confidence = self.generate_text(prompt)
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Record document processing metrics
            if self.metrics:
                doc_length = len(document_text)
                response_length = len(response)
                
                self.metrics.record_custom_metric(
                    "document_processing",
                    "document_size_bytes",
                    doc_length
                )
                self.metrics.record_custom_metric(
                    "document_processing",
                    "processing_time_ms",
                    processing_time_ms
                )
                self.metrics.record_custom_metric(
                    "document_processing",
                    "response_size_bytes",
                    response_length
                )
                
                # Record processing rate (bytes per second)
                if processing_time_ms > 0:
                    processing_rate = (doc_length / (processing_time_ms / 1000))
                    self.metrics.record_custom_metric(
                        "document_processing",
                        "processing_rate_bytes_per_second",
                        processing_rate
                    )
                
                # Stop operation timer
                self.metrics.stop_operation("process_document", op_time, success=True)
            
            # Return results
            return {
                'text': response,
                'confidence': confidence,
                'success': True,
                'model': self.model_manager.model_name,
                'document_info': {
                    'name': document.get('name', 'Unknown'),
                    'type': document.get('type', 'Unknown'),
                    'length': len(document_text)
                },
                'processing_time_ms': processing_time_ms
            }
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("process_document", op_time, success=False)
                self.metrics.record_llm_error("document_processing_error", {
                    "error": str(e),
                    "model": self.model_name,
                    "provider": "anthropic"
                })
            logger.error(f"Error processing document with Anthropic: {str(e)}")
            return {'error': str(e), 'success': False}
            
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        model_info = self.model_manager.get_model_info()
        
        # Record model information in metrics
        if self.metrics:
            self.metrics.record_custom_metric(
                "model_info",
                "provider",
                "anthropic"
            )
            self.metrics.record_custom_metric(
                "model_info",
                "model_name",
                model_info.get("name", self.model_name)
            )
        
        return model_info 