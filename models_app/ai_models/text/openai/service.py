"""
OpenAILLMService

Provides text generation services using OpenAI's large language models.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import json

from models_app.ai_models.text.base_text_provider import BaseLLMProvider
from .model_manager import OpenAILLMModelManager

# Import from common utilities for consistent error handling
from models_app.ai_models.utils.common.handlers import handle_llm_errors, handle_provider_connection
from models_app.ai_models.utils.common.errors import ProviderConnectionError, TokenLimitError

logger = logging.getLogger(__name__)

class OpenAILLMService(BaseLLMProvider):
    """
    Service for text generation with OpenAI's large language models.
    
    This service provides:
    1. Text generation with GPT models
    2. Document processing
    3. Conversational interfaces
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the OpenAI LLM service.
        
        Args:
            config: Configuration for the service
        """
        super().__init__(config or {})
        self.model_manager = OpenAILLMModelManager(config)
        self.system_message = config.get('system_message', 'You are a helpful AI assistant.')
        self.model_name = config.get('model_name', 'gpt-3.5-turbo')
        
        # Track token quota for monitoring
        self.token_quota = config.get('token_quota', 1000000)  # Default: 1M tokens
        self.token_reset_period = config.get('token_reset_period', 'monthly')  # When quota resets
        
        # Initialize the model if requested
        if config.get('initialize_on_creation', False):
            self.initialize()
            
    @handle_provider_connection
    def initialize(self) -> bool:
        """
        Initialize the OpenAI LLM service.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Check if service is available
            if not self.model_manager.is_available():
                logger.error("OpenAI service is not available")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error initializing OpenAI service: {str(e)}")
            error_info = self.handle_error_with_context(e, {
                "operation": "initialize",
                "model_name": self.model_name,
                "provider": "openai"
            })
            raise ProviderConnectionError(f"Failed to initialize OpenAI service: {str(e)}", provider="openai")
            
    @handle_llm_errors
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate text using the OpenAI model.
        
        Args:
            prompt: Input prompt for text generation
            max_tokens: Maximum number of tokens to generate (optional)
            
        Returns:
            Tuple[str, float]: Generated text and confidence score
        """
        # Start memory tracking for long operations
        self.start_memory_tracking()
        
        # Start metrics collection if enabled
        op_time = None
        first_token_time = None
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
            # Record start time for first token latency
            first_token_start = time.time()
            
            # Make API call
            start_time = time.time()
            response = self.model_manager.client.chat.completions.create(**params)
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Calculate first token time (if stream=True this would be more accurate)
            first_token_time_ms = generation_time_ms  # For non-streaming, this is a close approximation
            
            # Get response text
            response_text = response.choices[0].message.content
            
            # Calculate approximate confidence based on temperature
            temperature = params.get('temperature', 0.7)
            confidence = 1.0 - temperature  # Simplified confidence calculation
            
            # Log timing
            logger.info(f"OpenAI generated {len(response_text)} characters in {generation_time_ms/1000:.2f} seconds")
            
            # Record metrics if enabled
            if self.metrics:
                # Get token usage from the response if available
                usage = response.usage if hasattr(response, 'usage') else None
                
                if usage:
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                else:
                    # Estimate token usage if not provided by API
                    prompt_tokens = len(prompt) // 4  # Rough estimate
                    completion_tokens = len(response_text) // 4  # Rough estimate
                
                # Record token usage
                self.metrics.record_token_usage(prompt_tokens, completion_tokens)
                
                # Record generation metrics with first token time
                self.metrics.record_generation(
                    prompt_length=prompt_tokens,
                    response_length=completion_tokens,
                    total_time_ms=generation_time_ms,
                    first_token_time_ms=first_token_time_ms
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
                
                # Record context window usage for alerts monitoring
                context_window_size = getattr(self.model_manager, 'context_window', 4096)
                window_usage = prompt_tokens + completion_tokens
                window_percentage = (window_usage / context_window_size) * 100 if context_window_size > 0 else 0
                
                self.metrics.record_context_window_usage(
                    used_tokens=window_usage,
                    max_tokens=context_window_size
                )
                
                # Add additional context metrics for more detailed monitoring
                self.metrics.record_custom_metric(
                    "context_window", 
                    "utilization_percentage", 
                    window_percentage,
                    {
                        "model": self.model_name,
                        "critical_threshold": 98,
                        "warning_threshold": 80
                    }
                )
                
                # Record token quota usage for cost monitoring
                total_tokens = prompt_tokens + completion_tokens
                quota_percentage = (total_tokens / self.token_quota) * 100 if self.token_quota > 0 else 0
                
                self.metrics.record_custom_metric(
                    "token_usage",
                    "quota_percentage",
                    quota_percentage,
                    {
                        "model": self.model_name,
                        "quota": self.token_quota,
                        "reset_period": self.token_reset_period,
                        "critical_threshold": 95
                    }
                )
                
                # Stop operation timer
                self.metrics.stop_operation("generate_text", op_time, success=True)
            
            # Stop memory tracking and analyze results
            memory_timeline = self.stop_memory_tracking()
            if memory_timeline and len(memory_timeline) > 2:
                # Analyze memory usage for potential issues
                bottlenecks = self.detect_memory_bottlenecks()
                if bottlenecks and bottlenecks.get("bottlenecks"):
                    for issue in bottlenecks["bottlenecks"]:
                        if issue["severity"] == "high":
                            logger.warning(f"Memory issue detected: {issue['type']} - {issue['details']}")
                            
                            # Record memory bottleneck in metrics for alerting
                            if self.metrics:
                                self.metrics.record_custom_metric(
                                    "memory_tracking",
                                    "bottleneck_detected",
                                    1,
                                    {
                                        "type": issue["type"],
                                        "severity": issue["severity"],
                                        "details": issue["details"]
                                    }
                                )
            
            return response_text, confidence
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("generate_text", op_time, success=False)
                self.metrics.record_llm_error("generation_error", {
                    "error": str(e),
                    "model": self.model_name,
                    "provider": "openai"
                })
            
            # Stop memory tracking
            self.stop_memory_tracking()
            
            # Classify and handle the error for better debugging and monitoring
            error_info = self.handle_error_with_context(e, {
                "operation": "generate_text",
                "model": self.model_name,
                "provider": "openai",
                "prompt_length": len(prompt),
                "max_tokens": effective_max_tokens
            })
            
            logger.error(f"Error generating text with OpenAI: {str(e)}")
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
        # Start memory tracking for long operations
        self.start_memory_tracking()
        
        # Start metrics collection if enabled
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("generate_batch")
        
        try:
            results = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            
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
            
            # Stop memory tracking and analyze results
            memory_timeline = self.stop_memory_tracking()
            
            # Check for memory issues after batch processing
            if memory_timeline and len(memory_timeline) > 5:  # Need enough samples for analysis
                bottlenecks = self.detect_memory_bottlenecks()
                if bottlenecks and bottlenecks.get("bottlenecks"):
                    for issue in bottlenecks["bottlenecks"]:
                        if issue["severity"] in ["high", "medium"]:
                            logger.warning(f"Batch processing memory issue: {issue['type']} - {issue['details']}")
                
            return results
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("generate_batch", op_time, success=False)
                self.metrics.record_llm_error("batch_generation_error", {
                    "error": str(e),
                    "model": self.model_name,
                    "provider": "openai",
                    "batch_size": len(prompts)
                })
            
            # Stop memory tracking
            self.stop_memory_tracking()
            
            # Classify and handle the error
            error_info = self.handle_error_with_context(e, {
                "operation": "generate_batch",
                "model": self.model_name,
                "provider": "openai",
                "batch_size": len(prompts)
            })
            
            logger.error(f"Error generating batch with OpenAI: {str(e)}")
            raise
        
    @handle_llm_errors
    def process_document(self, document: Dict[str, Any], query: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document with the OpenAI model.
        
        Args:
            document: Document to process
            query: Optional query to guide processing
            
        Returns:
            Dict[str, Any]: Processing results
        """
        # Use the parent class implementation which handles chunking
        # and advanced document processing strategies
        return super().process_document(document, query)
            
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
                "openai"
            )
            self.metrics.record_custom_metric(
                "model_info",
                "model_name",
                model_info.get("name", self.model_name)
            )
            
            # Record context window size for monitoring
            context_window = model_info.get("context_window", 4096)
            self.metrics.record_custom_metric(
                "model_info",
                "context_window_size",
                context_window,
                {
                    "model": self.model_name,
                    "provider": "openai"
                }
            )
        
        return model_info

    def cleanup(self, force: bool = False) -> bool:
        """
        Free resources used by the OpenAI service.
        
        Args:
            force: Whether to force cleanup even if there are pending operations
            
        Returns:
            bool: Whether cleanup was successful
        """
        try:
            # Record final metrics before cleanup
            if self.metrics:
                # Export metrics if enabled in config
                if self.config.get('export_metrics_on_cleanup', False):
                    self.metrics.export_metrics()
                
                # Record cleanup event
                self.metrics.record_custom_metric(
                    "lifecycle",
                    "cleanup_initiated",
                    1,
                    {
                        "forced": force,
                        "provider": "openai",
                        "model": self.model_name
                    }
                )
            
            # Clean up model manager resources if available
            if hasattr(self, 'model_manager') and self.model_manager is not None:
                if hasattr(self.model_manager, 'cleanup'):
                    self.model_manager.cleanup()
            
            # Use parent class cleanup for standard resources
            return super().cleanup(force)
        except Exception as e:
            logger.error(f"Error during OpenAI service cleanup: {e}")
            return False 