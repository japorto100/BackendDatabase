"""
LightweightLLMService

Provides text generation services using lightweight large language models.
"""

import os
import logging
import time
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
import gc
import numpy as np

from models_app.ai_models.text.base_text_provider import BaseLLMProvider
from .model_manager import LightweightLLMModelManager

# Import from common utilities for consistent error handling
from models_app.ai_models.utils.common.handlers import handle_llm_errors, handle_provider_connection
from models_app.ai_models.utils.common.errors import ProviderConnectionError, TokenLimitError, ModelUnavailableError

logger = logging.getLogger(__name__)

class LightweightLLMService(BaseLLMProvider):
    """
    Service for text generation with lightweight models like Phi, Gemma, and Mistral.
    
    This service provides:
    1. Text generation with efficient models
    2. Document processing
    3. ONNX runtime acceleration for supported models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Lightweight LLM service.
        
        Args:
            config: Configuration for the service
        """
        super().__init__(config or {})
        self.model_manager = LightweightLLMModelManager(config)
        self.system_message = config.get('system_message', 'You are a helpful AI assistant.')
        self.model_name = config.get('model_name', 'lightweight-llm')
        
        # Initialize the model if requested
        if config.get('initialize_on_creation', False):
            self.initialize()
            
    @handle_provider_connection        
    def initialize(self) -> bool:
        """
        Initialize the Lightweight LLM service by loading the model.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Check if service is available
            if not self.model_manager.is_available():
                logger.error("Lightweight service is not available on this device")
                raise ModelUnavailableError(
                    "Lightweight model is not available on this device", 
                    model=self.model_name,
                    provider="lightweight"
                )
                
            # Initialize the model
            self.model_manager._initialize_model()
            return True
        except Exception as e:
            logger.error(f"Error initializing Lightweight service: {str(e)}")
            raise ProviderConnectionError(
                f"Failed to initialize Lightweight service: {str(e)}", 
                provider="lightweight",
                model=self.model_name
            )
            
    @handle_llm_errors
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate text using the lightweight model.
        
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
            
        # Override max_tokens if provided
        effective_max_tokens = max_tokens if max_tokens is not None else self.model_manager.max_tokens
            
        # Prepare generation parameters
        params = self.model_manager.prepare_generation_parameters(prompt, self.system_message)
        
        try:
            # Generate text
            start_time = time.time()
            
            # Extract input_ids from inputs
            inputs = params.pop("inputs")
            
            if self.model_manager.use_onnx:
                # Generate with ONNX runtime
                output_text = self._generate_with_onnx(inputs, params)
            else:
                # Generate with PyTorch
                output_text = self._generate_with_pytorch(inputs, params)
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Calculate approximate confidence based on temperature
            temperature = params.get('temperature', 0.7)
            confidence = max(0.6, 1.0 - temperature)  # Better confidence calculation for lightweight models
            
            # Log timing
            logger.info(f"Lightweight model generated {len(output_text)} characters in {generation_time_ms/1000:.2f} seconds")
            
            # Record metrics if enabled
            if self.metrics:
                # Tokenize input and output to get token counts
                prompt_tokens = len(self.model_manager.tokenizer.encode(prompt))
                completion_tokens = len(self.model_manager.tokenizer.encode(output_text))
                
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
                        "top_p": params.get('top_p', 0.9),
                        "use_onnx": self.model_manager.use_onnx
                    }
                )
                
                # Record hardware metrics like GPU memory usage if available
                if torch.cuda.is_available():
                    try:
                        self.metrics.record_custom_metric(
                            "hardware",
                            "gpu_memory_allocated_mb",
                            torch.cuda.memory_allocated() / (1024 * 1024)
                        )
                        self.metrics.record_custom_metric(
                            "hardware",
                            "gpu_memory_reserved_mb",
                            torch.cuda.memory_reserved() / (1024 * 1024)
                        )
                    except Exception as hw_error:
                        logger.warning(f"Failed to record hardware metrics: {str(hw_error)}")
                
                # Record context window usage
                context_window_size = getattr(self.model_manager, 'context_window', 2048)  # Default for lightweight models
                self.metrics.record_context_window_usage(
                    used_tokens=prompt_tokens + completion_tokens,
                    max_tokens=context_window_size
                )
                
                # Stop operation timer
                self.metrics.stop_operation("generate_text", op_time, success=True)
            
            # Clear CUDA cache if available to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return output_text, confidence
            
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("generate_text", op_time, success=False)
                self.metrics.record_llm_error("generation_error", {
                    "error": str(e),
                    "model": self.model_name,
                    "provider": "lightweight",
                    "use_onnx": getattr(self.model_manager, 'use_onnx', False)
                })
            
            logger.error(f"Error generating text with lightweight model: {str(e)}")
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            raise
            
    def _generate_with_pytorch(self, inputs: Dict[str, torch.Tensor], params: Dict[str, Any]) -> str:
        """Generate text using PyTorch model."""
        # Generate text with the model
        with torch.no_grad():
            outputs = self.model_manager.model.generate(
                **inputs,
                **params
            )
        
        # Decode the output tokens, skipping the input tokens
        output_text = self.model_manager.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return output_text
        
    def _generate_with_onnx(self, inputs: Dict[str, torch.Tensor], params: Dict[str, Any]) -> str:
        """Generate text using ONNX model."""
        # Convert torch tensors to numpy for ONNX
        onnx_inputs = {
            'input_ids': inputs['input_ids'].cpu().numpy(),
            'attention_mask': inputs['attention_mask'].cpu().numpy()
        }
        
        # Set up generation parameters
        max_length = params.get('max_length', inputs['input_ids'].shape[1] + 100)
        temperature = params.get('temperature', 0.7)
        top_p = params.get('top_p', 0.9)
        
        # Run ONNX inference loop to generate output tokens
        output_sequence = self.model_manager.generate_with_onnx(
            onnx_inputs, 
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        # Decode the output tokens, skipping the input tokens
        output_text = self.model_manager.tokenizer.decode(
            output_sequence[inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return output_text
            
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
                    "provider": "lightweight",
                    "batch_size": len(prompts)
                })
            logger.error(f"Error generating batch with lightweight model: {str(e)}")
            raise
        
    @handle_llm_errors
    def process_document(self, document: Dict[str, Any], query: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document with the lightweight model.
        
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
                        "provider": "lightweight"
                    })
                return {'error': error_msg, 'success': False}
                
            # Lightweight models may have context length limitations
            # We may need to truncate the document
            context_window_size = getattr(self.model_manager, 'context_window', 2048)
            max_safe_doc_length = context_window_size - 200  # Leave room for prompt and generation
            
            # Check document length based on token count
            doc_tokens = len(self.model_manager.tokenizer.encode(document_text))
            if doc_tokens > max_safe_doc_length:
                logger.warning(f"Document too long ({doc_tokens} tokens), truncating to {max_safe_doc_length} tokens")
                # Truncate document based on tokens
                truncated_tokens = self.model_manager.tokenizer.encode(document_text)[:max_safe_doc_length]
                document_text = self.model_manager.tokenizer.decode(truncated_tokens)
                
                if self.metrics:
                    self.metrics.record_custom_metric(
                        "document_processing",
                        "truncation_required",
                        1
                    )
                    self.metrics.record_custom_metric(
                        "document_processing",
                        "original_tokens",
                        doc_tokens
                    )
                    self.metrics.record_custom_metric(
                        "document_processing",
                        "truncated_tokens",
                        max_safe_doc_length
                    )
                    
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
                    'length': len(document_text),
                    'truncated': doc_tokens > max_safe_doc_length
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
                    "provider": "lightweight"
                })
            logger.error(f"Error processing document with lightweight model: {str(e)}")
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
                "lightweight"
            )
            self.metrics.record_custom_metric(
                "model_info",
                "model_name",
                model_info.get("name", self.model_name)
            )
            
            # Record additional information about optimizations
            self.metrics.record_custom_metric(
                "model_info",
                "use_onnx",
                model_info.get("use_onnx", False)
            )
            self.metrics.record_custom_metric(
                "model_info",
                "quantization",
                model_info.get("quantization", "none") 
            )
        
        return model_info
        
    def cleanup(self):
        """
        Clean up resources used by the model.
        """
        try:
            # Start metrics collection if enabled
            op_time = None
            if self.metrics:
                op_time = self.metrics.start_operation("cleanup")
                
            logger.info(f"Cleaning up resources for {self.model_name}...")
            
            # Free GPU memory if model was loaded
            if hasattr(self.model_manager, 'model') and self.model_manager.model is not None:
                # Record memory usage before cleanup
                if torch.cuda.is_available() and self.metrics:
                    try:
                        self.metrics.record_custom_metric(
                            "cleanup",
                            "gpu_memory_before_mb",
                            torch.cuda.memory_allocated() / (1024 * 1024)
                        )
                    except Exception:
                        pass
                
                # Set model to None to release references
                self.model_manager.model = None
                self.model_manager.tokenizer = None
                
                # Force garbage collection and empty CUDA cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                    # Record memory usage after cleanup
                    if self.metrics:
                        try:
                            self.metrics.record_custom_metric(
                                "cleanup",
                                "gpu_memory_after_mb",
                                torch.cuda.memory_allocated() / (1024 * 1024)
                            )
                        except Exception:
                            pass
                            
            logger.info(f"Cleanup complete for {self.model_name}")
            
            # Stop operation timer
            if self.metrics and op_time:
                self.metrics.stop_operation("cleanup", op_time, success=True)
                
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("cleanup", op_time, success=False)
                self.metrics.record_llm_error("cleanup_error", {
                    "error": str(e),
                    "model": self.model_name,
                    "provider": "lightweight"
                })
            logger.error(f"Error during cleanup of lightweight model: {str(e)}")