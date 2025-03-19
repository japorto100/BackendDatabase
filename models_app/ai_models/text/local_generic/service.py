"""
LocalGenericLLMService

Provides text generation services using generic local large language models.
"""

import os
import logging
import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import gc
import json
import requests
from urllib.parse import urlparse

from models_app.ai_models.text.base_text_provider import BaseLLMProvider
from .model_manager import LocalGenericLLMModelManager, RECOMMENDED_MODELS

# Import from common utilities for consistent error handling
from models_app.ai_models.utils.common.handlers import handle_llm_errors, handle_provider_connection
from models_app.ai_models.utils.common.errors import ProviderConnectionError, TokenLimitError, ModelUnavailableError

logger = logging.getLogger(__name__)

class LocalGenericLLMService(BaseLLMProvider):
    """
    Service for text generation with generic local models.
    
    This service provides:
    1. Text generation with local models in various formats
    2. Document processing and summarization
    3. Support for custom local models
    4. Model discovery and management
    5. Download models from URLs and add them to the local collection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the LocalGeneric LLM service.
        
        Args:
            config: Configuration for the service
        """
        super().__init__(config or {})
        self.model_manager = LocalGenericLLMModelManager(config)
        self.system_message = config.get('system_message', 'You are a helpful AI assistant.')
        self.model_name = config.get('model_name', 'local-generic-llm')
        
        # Initialize the model if requested
        if config.get('initialize_on_creation', False):
            self.initialize()
            
    @handle_provider_connection        
    def initialize(self) -> bool:
        """
        Initialize the LocalGeneric LLM service.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Check if service is available
            if not self.model_manager.is_available():
                logger.error("LocalGeneric LLM service is not available on this device")
                raise ModelUnavailableError(
                    "LocalGeneric model is not available on this device", 
                    model=self.model_name,
                    provider="local_generic"
                )
                
            # Initialize the model
            self.model_manager.initialize_model()
            return True
        except Exception as e:
            logger.error(f"Error initializing LocalGeneric LLM service: {str(e)}")
            raise ProviderConnectionError(
                f"Failed to initialize LocalGeneric LLM service: {str(e)}", 
                provider="local_generic",
                model=self.model_name
            )
            
    @handle_llm_errors
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate text using the local model.
        
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
            # Generate text based on model format
            start_time = time.time()
            
            if self.model_manager.model_format == 'huggingface':
                output_text = self._generate_with_huggingface(params)
            elif self.model_manager.model_format == 'gguf':
                output_text = self._generate_with_gguf(params)
            elif self.model_manager.model_format == 'onnx':
                output_text = self._generate_with_onnx(params)
            else:
                raise ValueError(f"Unsupported model format: {self.model_manager.model_format}")
            
            generation_time_ms = (time.time() - start_time) * 1000
            
            # Calculate approximate confidence (simplified for local models)
            temperature = params.get('temperature', 0.7)
            confidence = max(0.6, 1.0 - temperature)  # Better confidence calculation for local models
            
            # Log timing
            logger.info(f"LocalGeneric model generated {len(output_text)} characters in {generation_time_ms/1000:.2f} seconds")
            
            # Record metrics if enabled
            if self.metrics:
                # Get token counts - method depends on model format
                try:
                    if hasattr(self.model_manager, 'tokenizer'):
                        prompt_tokens = len(self.model_manager.tokenizer.encode(prompt))
                        completion_tokens = len(self.model_manager.tokenizer.encode(output_text))
                    else:
                        # Rough estimate if tokenizer not available
                        prompt_tokens = len(prompt) // 4  
                        completion_tokens = len(output_text) // 4
                
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
                            "model_format": self.model_manager.model_format
                        }
                    )
                    
                    # Record hardware metrics if available
                    if torch.cuda.is_available():
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
                    
                    # Record context window usage
                    context_window_size = getattr(self.model_manager, 'context_window', 2048)
                    self.metrics.record_context_window_usage(
                        used_tokens=prompt_tokens + completion_tokens,
                        max_tokens=context_window_size
                    )
                except Exception as metrics_error:
                    logger.warning(f"Failed to record metrics: {str(metrics_error)}")
                
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
                    "provider": "local_generic",
                    "model_format": self.model_manager.model_format
                })
            
            logger.error(f"Error generating text with LocalGeneric model: {str(e)}")
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            raise
            
    def _generate_with_huggingface(self, params: Dict[str, Any]) -> str:
        """Generate text using Hugging Face Transformers model."""
        # Extract input_ids from inputs
        inputs = params.pop("inputs")
        
        # Generate text with the model
        with torch.no_grad():
            outputs = self.model_manager.model.generate(
                **inputs,
                **params
            )
        
        # Decode the output tokens
        output_text = self.model_manager.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        return output_text
        
    def _generate_with_gguf(self, params: Dict[str, Any]) -> str:
        """Generate text using GGUF model via llama-cpp or ctransformers."""
        # Check which GGUF implementation we're using
        if hasattr(self.model_manager.model, 'generate'):
            # llama-cpp style
            result = self.model_manager.model.generate(
                params["prompt"],
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                repeat_penalty=params["repeat_penalty"]
            )
            return result['choices'][0]['text']
        else:
            # ctransformers style
            result = self.model_manager.model(
                params["prompt"],
                max_new_tokens=params["max_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                repetition_penalty=params["repeat_penalty"]
            )
            # Return everything after the prompt
            prompt_len = len(params["prompt"])
            return result[prompt_len:]
        
    def _generate_with_onnx(self, params: Dict[str, Any]) -> str:
        """Generate text using ONNX model."""
        # Extract inputs
        inputs = params.pop("inputs")
        
        # Convert to numpy for ONNX
        input_ids = inputs["input_ids"].numpy()
        attention_mask = inputs["attention_mask"].numpy()
        
        # Simple greedy decoding for ONNX models
        max_length = params.get("max_length", len(input_ids[0]) + 100)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 0.9)
        
        # Get model inputs and outputs
        model_inputs = self.model_manager.model.get_inputs()
        input_names = [input.name for input in model_inputs]
        
        # Build input feed
        input_feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # If the model requires additional inputs, add them
        if "position_ids" in input_names:
            position_ids = np.arange(input_ids.shape[1], dtype=np.int64)
            position_ids = np.expand_dims(position_ids, 0)
            input_feed["position_ids"] = position_ids
            
        # For simplicity, we'll just do greedy generation here
        generated_ids = input_ids.copy()
        
        for _ in range(min(params.get("max_length", 100), 500)):  # Limit maximum steps
            # Update inputs for next token prediction
            current_length = generated_ids.shape[1]
            position_ids = np.arange(current_length, dtype=np.int64)
            position_ids = np.expand_dims(position_ids, 0)
            attention_mask = np.ones((1, current_length), dtype=np.int64)
            
            input_feed = {
                "input_ids": generated_ids,
                "attention_mask": attention_mask
            }
            if "position_ids" in input_names:
                input_feed["position_ids"] = position_ids
                
            # Run inference
            outputs = self.model_manager.model.run(None, input_feed)
            
            # Typically the first output is the logits
            logits = outputs[0][:, -1, :]
            
            # Apply temperature and top-p
            if temperature > 0:
                logits = logits / temperature
                
            # Convert to probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            
            # Apply top-p sampling
            if top_p < 1.0:
                sorted_probs = np.sort(probs[0])[::-1]
                cumulative_probs = np.cumsum(sorted_probs)
                cutoff = sorted_probs[np.argmax(cumulative_probs > top_p)]
                probs[0, probs[0] < cutoff] = 0
                
            # Resample
            if temperature == 0:
                # Greedy selection
                next_token = np.argmax(probs, axis=-1)[0]
            else:
                # Sample from the distribution
                next_token = np.random.choice(probs.shape[-1], p=probs[0])
                
            # Add the next token
            next_token_array = np.array([[next_token]])
            generated_ids = np.concatenate([generated_ids, next_token_array], axis=1)
            
            # Check for EOS
            if next_token == self.model_manager.tokenizer.eos_token_id:
                break
                
        # Decode the generated sequence, skipping the input
        generated_text = self.model_manager.tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text
            
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
                    "provider": "local_generic",
                    "batch_size": len(prompts)
                })
            logger.error(f"Error generating batch with LocalGeneric model: {str(e)}")
            raise
        
    @handle_llm_errors
    def process_document(self, document: Dict[str, Any], query: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document with the local model.
        
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
                        "provider": "local_generic"
                    })
                return {'error': error_msg, 'success': False}
            
            # Check document length based on model context window
            context_window_size = getattr(self.model_manager, 'context_window', 2048)
            max_safe_doc_length = context_window_size - 200  # Leave room for prompt and generation
            
            # Check if document needs chunking
            doc_tokens = 0
            is_truncated = False
            try:
                if hasattr(self.model_manager, 'tokenizer'):
                    doc_tokens = len(self.model_manager.tokenizer.encode(document_text))
                    if doc_tokens > max_safe_doc_length:
                        logger.warning(f"Document too long ({doc_tokens} tokens), truncating to {max_safe_doc_length} tokens")
                        truncated_tokens = self.model_manager.tokenizer.encode(document_text)[:max_safe_doc_length]
                        document_text = self.model_manager.tokenizer.decode(truncated_tokens)
                        is_truncated = True
                        
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
            except Exception as tokenize_error:
                logger.warning(f"Could not tokenize document: {str(tokenize_error)}")
                # Fallback to character count estimate
                if len(document_text) > max_safe_doc_length * 4:  # Rough estimate
                    document_text = document_text[:max_safe_doc_length * 4]
                    is_truncated = True
                    
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
                'model': self.model_name,
                'document_info': {
                    'name': document.get('name', 'Unknown'),
                    'type': document.get('type', 'Unknown'),
                    'length': len(document_text),
                    'tokens': doc_tokens if doc_tokens > 0 else None,
                    'truncated': is_truncated
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
                    "provider": "local_generic"
                })
            logger.error(f"Error processing document with LocalGeneric model: {str(e)}")
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
                "local_generic"
            )
            self.metrics.record_custom_metric(
                "model_info",
                "model_name",
                model_info.get("name", self.model_name)
            )
            self.metrics.record_custom_metric(
                "model_info",
                "model_format",
                model_info.get("format", "unknown")
            )
            
            # Record detailed information if available
            if "parameters" in model_info:
                self.metrics.record_custom_metric(
                    "model_info",
                    "parameter_count",
                    model_info["parameters"]
                )
            
            if "quantization" in model_info:
                self.metrics.record_custom_metric(
                    "model_info",
                    "quantization",
                    model_info["quantization"]
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
            
            # Record memory usage before cleanup if available
            if torch.cuda.is_available() and self.metrics:
                try:
                    self.metrics.record_custom_metric(
                        "cleanup",
                        "gpu_memory_before_mb",
                        torch.cuda.memory_allocated() / (1024 * 1024)
                    )
                except Exception:
                    pass
            
            # Perform format-specific cleanup
            if self.model_manager.model:
                self.model_manager.unload_model()
                
                # Force garbage collection
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
                    "provider": "local_generic"
                })
            logger.error(f"Error during cleanup of LocalGeneric model: {str(e)}")

    def list_all_models(self) -> List[Dict[str, Any]]:
        """
        List all available models, including local models and recommended lightweight models.
        
        Returns:
            List[Dict[str, Any]]: List of all models with their metadata
        """
        return self.model_manager.list_available_models()
        
    def get_recommended_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of recommended lightweight models that work well on minimal hardware.
        
        Returns:
            List[Dict[str, Any]]: List of recommended models with their metadata
        """
        return self.model_manager.get_recommended_models()
        
    def add_model_from_url(self, url: str, name: Optional[str] = None, model_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Download and add a model from a URL.
        
        Args:
            url: URL to download the model from
            name: Optional custom name for the model
            model_format: Optional format hint (huggingface, gguf, onnx)
            
        Returns:
            Dict[str, Any]: Information about the added model
        """
        # First, validate the URL
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL format: {url}")
            
        # Check if the URL is accessible
        if not self._is_url_accessible(url):
            raise ValueError(f"URL is not accessible: {url}")
            
        # Add the model
        model_info = self.model_manager.add_model_from_url(url, name)
        
        # Return information about the added model
        return {
            'success': True,
            'model_info': model_info,
            'message': f"Model successfully added from {url}"
        }
        
    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL has a valid format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
            
    def _is_url_accessible(self, url: str) -> bool:
        """Check if a URL is accessible."""
        try:
            # For Hugging Face URLs, we just check if the domain exists
            if "huggingface.co" in url or "hf.co" in url:
                parsed_url = urlparse(url)
                domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
                response = requests.head(domain, timeout=5)
                return response.status_code < 400
                
            # For other URLs, we try a HEAD request
            response = requests.head(url, timeout=5)
            return response.status_code < 400
        except:
            return False
        
    def estimate_model_requirements(self, model_size: str) -> Dict[str, Any]:
        """
        Estimate hardware requirements for a model of given size.
        
        Args:
            model_size: Model size in parameters (e.g., "7B", "1.5B")
            
        Returns:
            Dict[str, Any]: Estimated hardware requirements
        """
        return self.model_manager.estimate_hardware_requirements(model_size)
        
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get information about the current device capabilities.
        
        Returns:
            Dict[str, Any]: Device information including RAM, VRAM, etc.
        """
        device_info = {
            'device_type': self.model_manager.device,
            'cpu_count': os.cpu_count() or 0,
            'ram_available_gb': 0,
            'vram_available_gb': 0,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
        
        # Get RAM information
        try:
            import psutil
            device_info['ram_available_gb'] = round(psutil.virtual_memory().available / (1024**3), 2)
        except ImportError:
            logger.warning("psutil not installed, cannot determine available RAM")
            
        # Get VRAM information for CUDA devices
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    free_mem = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    device_info[f'cuda:{i}_free_vram_gb'] = round(free_mem / (1024**3), 2)
                    if i == 0:  # Use the first GPU as the main VRAM reference
                        device_info['vram_available_gb'] = round(free_mem / (1024**3), 2)
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory info: {str(e)}")
                
        return device_info
    
    def check_model_compatibility(self, model_url: str) -> Dict[str, Any]:
        """
        Check if a model is compatible with the current system.
        
        Args:
            model_url: URL or name of the model to check
            
        Returns:
            Dict[str, Any]: Compatibility information
        """
        # Extract model size from URL or name
        model_size = self._extract_model_size_from_url(model_url)
        
        # Get device info
        device_info = self.get_device_info()
        
        # Get estimated requirements
        if model_size:
            requirements = self.model_manager.estimate_hardware_requirements(model_size)
        else:
            # Default to medium size if we can't determine
            requirements = self.model_manager.estimate_hardware_requirements("7B")
            
        # Determine compatibility
        is_compatible = True
        compatibility_issues = []
        
        # Check RAM
        if requirements['ram_gb'] > device_info.get('ram_available_gb', 0):
            is_compatible = False
            compatibility_issues.append(f"Insufficient RAM: Required {requirements['ram_gb']}GB, Available {device_info.get('ram_available_gb', 0)}GB")
            
        # Check VRAM if model needs GPU
        if not requirements['cpu_only_possible'] and requirements['vram_gb'] > device_info.get('vram_available_gb', 0):
            is_compatible = False
            compatibility_issues.append(f"Insufficient VRAM: Required {requirements['vram_gb']}GB, Available {device_info.get('vram_available_gb', 0)}GB")
            
        # Return compatibility information
        return {
            'is_compatible': is_compatible,
            'model_url': model_url,
            'estimated_model_size': model_size or "Unknown",
            'estimated_requirements': requirements,
            'device_info': device_info,
            'issues': compatibility_issues,
            'recommended_quantization': requirements['recommended_quantization']
        }
        
    def _extract_model_size_from_url(self, url: str) -> Optional[str]:
        """Extract model size from URL or name."""
        # Common patterns in model names
        patterns = [
            r'(\d+[.]\d+)[bB]',  # matches 1.5B
            r'(\d+)[bB]',        # matches 7B
            r'-(\d+)[bB]-',      # matches -7B-
            r'-(\d+)[bB]$',      # matches -7B at the end
            r'(\d+)[mM]'         # matches 770M
        ]
        
        # Extract the last part of the URL or name
        name_part = url.split('/')[-1]
        
        # Check for each pattern
        import re
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                size = match.group(1)
                if '.' in size or 'm' in size.lower():
                    # Convert M to B if needed
                    if 'm' in size.lower():
                        return f"{float(size.lower().replace('m', '')) / 1000}B"
                    return f"{size}B"
                else:
                    return f"{size}B"
                
        # If no pattern matched, return None
        return None 