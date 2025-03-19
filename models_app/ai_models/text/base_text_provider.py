"""
BaseLLMProvider

Base class for all LLM providers, defining the common interface and functionality.
This class serves as the foundation for all text generation providers and ensures
consistent behavior across different implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import time
import uuid
import re
import threading
import psutil
import gc
import torch
import os

# Import base model service for inheritance
from models_app.ai_models.utils.common.ai_base_service import BaseModelService

# Import LLM metrics collector
from models_app.ai_models.utils.common.metrics import get_llm_metrics, LLMMetricsCollector

# Import LLM configuration 
from models_app.ai_models.utils.common.config import LLMConfig, get_llm_config

# Import common utilities for error handling
from models_app.ai_models.utils.common.handlers import handle_llm_errors, handle_model_errors
from models_app.ai_models.utils.common.errors import ModelError, ModelUnavailableError, TokenLimitError

# Import necessary utils from the text utils module
from models_app.ai_models.utils.text import (
    determine_token_strategy,
    select_relevant_chunks,
    chunk_text_for_model,
    combine_chunk_results,
    create_document_prompt,
    create_summary_prompt
)

logger = logging.getLogger(__name__)

class BaseLLMProvider(BaseModelService):
    """
    Abstract base class for all LLM providers.
    
    This class inherits from BaseModelService to integrate with the common
    model service architecture while maintaining the existing LLM interface.
    
    All LLM providers should inherit from this class and implement its
    abstract methods to ensure a consistent interface across providers.
    
    Key features:
    - Text generation with various parameters
    - Batch text generation for efficiency
    - Document processing capabilities
    - Integration with memory systems
    - Metrics collection for performance monitoring
    - Resource management with memory timeline tracking
    - Context manager support for proper resource cleanup
    - Error classification and handling
    """
    
    # Define common error patterns for classification
    TRANSIENT_ERROR_PATTERNS = [
        r'timeout', 
        r'connection.*(?:refused|reset|aborted)',
        r'temporarily unavailable',
        r'rate limit',
        r'too many requests',
        r'server (is )?overloaded',
        r'insufficient.*resources',
        r'try again later',
        r'service unavailable',
        r'internal server error',
        r'bad gateway',
        r'gateway timeout'
    ]
    
    PERMANENT_ERROR_PATTERNS = [
        r'authentication failed',
        r'unauthorized',
        r'forbidden',
        r'not found',
        r'invalid.*(?:request|format|parameter|token|key)',
        r'unsupported',
        r'bad request',
        r'access denied',
        r'permission denied',
        r'quota exceeded',
        r'account.*(?:suspended|deactivated|disabled)',
        r'method not allowed',
        r'token.*expired'
    ]
    
    @abstractmethod
    def __init__(self, config: Union[Dict[str, Any], LLMConfig]):
        """
        Initialize the LLM provider.
        
        Args:
            config: Configuration for the provider, including:
                - provider_type: The type of provider
                - model_name: The name of the model to use
                - max_context: Maximum context window size
                - use_memory: Whether to use memory for conversational context
                - enable_metrics: Whether to enable metrics collection
                - and other provider-specific parameters
        """
        # Convert config to LLMConfig if it's a dict
        if isinstance(config, dict):
            self.config = config  # Keep the original dict for backward compatibility
        else:
            self.config = config.to_dict() if hasattr(config, 'to_dict') else config
            
        # Initialize base model service with appropriate parameters
        name = self.config.get('provider_type', 'llm')
        model_name = self.config.get('model_name', self.config.get('model', 'base'))  # Support both for backward compatibility
        super().__init__(name, model_name, self.config)
        
        # Store configuration
        self.model_name = model_name  # Ensure model_name is consistently available
        self.max_context = self.config.get('max_context', 4096)
        
        # Add memory interface
        self.memory = self.config.get('memory', None)
        if self.memory is None and self.config.get('use_memory', False):
            from mem0 import Memory
            self.memory = Memory(self.config.get('memory_config', {}))
        
        # Initialize metrics collector
        self.metrics = None
        if self.config.get('enable_metrics', True):
            self.metrics = get_llm_metrics(f"{name}_{model_name}")
            
        # Memory timeline tracking for long-running operations
        self.memory_timeline = []
        self.memory_tracking_enabled = False
        self.memory_sampling_interval = 1.0  # seconds
        self.memory_tracker_thread = None
        self.memory_tracking_lock = threading.Lock()
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> Tuple[str, float]:
        """
        Generate text based on a prompt.
        
        This is the core method for text generation that all providers must implement.
        It should handle the specifics of calling the underlying model API or local model.
        
        Args:
            prompt: The prompt for text generation
            max_tokens: Maximum number of tokens to generate (optional)
            
        Returns:
            Tuple[str, float]: Generated text and a confidence score
            
        Raises:
            Exception: If text generation fails, appropriate exceptions should be raised
        """
        pass
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], max_tokens: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Generate text for multiple prompts in a batch.
        
        This method should efficiently process multiple prompts, potentially
        taking advantage of batching capabilities in the underlying model.
        
        Args:
            prompts: List of prompts to generate responses for
            max_tokens: Maximum number of tokens per generation (optional)
            
        Returns:
            List[Tuple[str, float]]: List of generated texts and confidence scores
            
        Raises:
            Exception: If batch generation fails, appropriate exceptions should be raised
        """
        pass
    
    @handle_llm_errors
    def process_document(self, document: Dict[str, Any], query: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document with the LLM.
        
        This method handles document processing, including:
        - Analyzing document text
        - Answering questions about the document
        - Summarizing document content
        - Extracting information from documents
        
        The method automatically chooses between direct processing or chunking
        based on the document length and model context window.
        
        Args:
            document: A document object containing text, chunks, and metadata
            query: An optional query for document processing
            
        Returns:
            Dict: Processing result with response and metadata
        """
        # Start metrics collection if enabled
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("process_document")
            
        text = document.get("text", "")
        
        # Decide processing strategy based on text length
        estimated_tokens = len(text) // 4  # Rough estimate: ~4 characters per token
        strategy = determine_token_strategy(estimated_tokens, self.max_context)
        
        try:
            if strategy == "direct":
                # Text fits directly in the context
                if query:
                    prompt = create_document_prompt(text, query)
                else:
                    prompt = create_summary_prompt(text)
                    
                response, confidence = self.generate_text(prompt)
                
                result = {
                    "response": response,
                    "confidence": confidence,
                    "model_used": self.get_model_info().get('name', self.model_name),
                    "strategy": strategy
                }
            else:
                # Text is too long, use chunking
                result = self.process_document_with_chunking(document, query)
                
            # Stop metrics collection if enabled
            if self.metrics and op_time:
                self.metrics.stop_operation("process_document", op_time, success=True)
                
            return result
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("process_document", op_time, success=False)
                # Record specific error type
                self.metrics.record_llm_error("document_processing_error", {"error": str(e)})
            
            # Classify and handle the error
            error_info = self.handle_error_with_context(e, {
                "operation": "process_document",
                "document_length": len(text),
                "strategy": strategy
            })
            
            raise e
    
    @handle_llm_errors
    def process_long_document(self, document: Dict[str, Any], query: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a long document with special strategies.
        
        This method provides an entry point for handling exceptionally long documents
        that require specialized processing beyond the basic chunking strategy.
        
        Args:
            document: A document object containing text, chunks, and metadata
            query: An optional query for document processing
            
        Returns:
            Dict: Processing result with response and metadata
        """
        # Default implementation uses chunking
        return self.process_document_with_chunking(document, query)
    
    @handle_llm_errors
    def process_document_with_chunking(self, document: Dict[str, Any], query: Optional[str] = None, 
                                      chunk_size: int = 4000, overlap: int = 200) -> Dict[str, Any]:
        """
        Process a document with chunking strategies.
        
        This method implements a chunking strategy for processing documents that are
        too large to fit in the model's context window. It includes:
        - Chunking the document into manageable pieces
        - Processing each chunk with the model
        - Selecting relevant chunks when a query is provided
        - Combining the results into a coherent response
        
        Args:
            document: A document object containing text, chunks, and metadata
            query: An optional query for document processing
            chunk_size: Size of the chunks in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            Dict: Processing result with response and metadata
        """
        # Start metrics collection if enabled
        op_time = None
        if self.metrics:
            op_time = self.metrics.start_operation("process_document_with_chunking")
            
        try:
            text = document.get("text", "")
            metadata = document.get("metadata", {})
            
            # Extract chunks from the document or create them
            if "chunks" in document and document["chunks"]:
                chunks = document["chunks"]
            else:
                # Use the chunking utility
                chunks = chunk_text_for_model(text, metadata, strategy="adaptive")
            
            # If a query is present, filter relevant chunks
            if query:
                relevant_chunks = select_relevant_chunks(chunks, query)
                if not relevant_chunks and chunks:
                    relevant_chunks = chunks[:min(3, len(chunks))]  # Fallback: Use the first chunks
            else:
                # For summarization: Use all chunks or a subset if there are many
                if len(chunks) > 10:
                    relevant_chunks = chunks[:10]  # Limit to 10 chunks for summarization
                else:
                    relevant_chunks = chunks
            
            # Process each relevant chunk
            results = []
            start_time = time.time()
            for chunk in relevant_chunks:
                chunk_text = chunk["text"]
                if query:
                    prompt = create_document_prompt(chunk_text, query)
                else:
                    prompt = create_summary_prompt(chunk_text)
                
                response, confidence = self.generate_text(prompt)
                results.append({
                    "response": response,
                    "confidence": confidence,
                    "chunk_metadata": chunk.get("metadata", {})
                })
            
            # Record token usage and generation metrics if enabled
            if self.metrics:
                processing_time_ms = (time.time() - start_time) * 1000
                # Estimate token usage for metrics
                prompt_tokens = sum(len(chunk["text"]) // 4 for chunk in relevant_chunks)
                completion_tokens = sum(len(result["response"]) // 4 for result in results)
                self.metrics.record_token_usage(prompt_tokens, completion_tokens)
                self.metrics.record_generation(
                    prompt_length=prompt_tokens,
                    response_length=completion_tokens,
                    total_time_ms=processing_time_ms
                )
                # Record context window usage
                self.metrics.record_context_window_usage(
                    used_tokens=prompt_tokens + completion_tokens,
                    max_tokens=self.max_context
                )
            
            # Combine the results with the utility function
            combined_result = combine_chunk_results(results, strategy="hierarchical")
            
            # Get model info for better reporting
            model_info = self.get_model_info()
            combined_result["model_used"] = model_info.get('name', self.model_name)
            combined_result["model_provider"] = model_info.get('provider', 'Unknown')
            combined_result["chunk_count"] = len(relevant_chunks)
            combined_result["query"] = query
            
            # Stop metrics collection if enabled
            if self.metrics and op_time:
                self.metrics.stop_operation("process_document_with_chunking", op_time, success=True)
                
            return combined_result
        except Exception as e:
            # Record error in metrics
            if self.metrics and op_time:
                self.metrics.stop_operation("process_document_with_chunking", op_time, success=False)
                self.metrics.record_llm_error("chunking_error", {"error": str(e)})
            
            # Classify and handle the error
            error_info = self.handle_error_with_context(e, {
                "operation": "process_document_with_chunking",
                "chunk_count": len(document.get("chunks", [])),
                "document_length": len(document.get("text", ""))
            })
            
            raise e

    def _load_model_impl(self) -> Any:
        """
        Implementation of model loading.
        
        This method bridges the BaseModelService's model loading with the
        LLM provider's implementation. Subclasses should override this
        method if they want to use BaseModelService's model loading
        capabilities.
        
        Returns:
            Any: Loaded model object
            
        Raises:
            NotImplementedError: LLM providers must implement their own model loading
        """
        # This is a bridge method to implement BaseModelService's abstract method
        # Subclasses should override this if they want to use BaseModelService's model loading
        raise NotImplementedError("LLM providers must implement their own model loading")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        This method returns details about the model being used,
        which is useful for logging and client applications.
        
        Returns:
            Dict[str, Any]: Model information including:
                - name: Model name
                - service: Service type
                - provider: Provider name
                - loaded: Whether the model is loaded
                - and provider-specific details
        """
        # Default implementation that should be overridden by subclasses
        return {
            "name": self.model_name,
            "service": self.name,
            "provider": self.__class__.__name__,
            "loaded": self.loaded_model is not None,
        }
        
    def _cleanup_impl(self) -> None:
        """
        Implementation of resource cleanup.
        
        This method handles the cleanup of resources used by the LLM provider,
        including unloading models and exporting metrics. It can be overridden
        by subclasses to provide provider-specific cleanup procedures.
        """
        # Default implementation
        self.loaded_model = None
        
        # Export metrics if enabled
        if self.metrics and hasattr(self.config, 'metrics_export_on_cleanup') and self.config.metrics_export_on_cleanup:
            self.metrics.export_metrics()
            if hasattr(self.config, 'metrics_reset_on_cleanup') and self.config.metrics_reset_on_cleanup:
                self.metrics.reset()
        
    # List models implementation as required by BaseModelService
    def _list_models_impl(self) -> List[Dict[str, Any]]:
        """
        Implementation of listing available models.
        
        This method provides information about models available through
        this provider. The default implementation simply returns the
        currently loaded model, but subclasses should override this to
        provide comprehensive model listings.
        
        Returns:
            List[Dict[str, Any]]: List of available models with metadata
        """
        # Default implementation
        return [
            {"name": self.model_name, "description": "Currently loaded model"}
        ]
        
    def classify_error(self, error: Exception) -> Dict[str, Any]:
        """
        Classify an error as either transient or permanent based on error patterns.
        
        Args:
            error: The exception to classify
            
        Returns:
            Dict[str, Any]: Classification result with error_id, is_transient flag, 
                            retry_recommended flag, and error_category
        """
        error_str = str(error).lower()
        error_id = str(uuid.uuid4())[:8]  # Generate a short unique ID for tracking
        
        # Check if error matches any transient patterns
        is_transient = any(re.search(pattern, error_str) for pattern in self.TRANSIENT_ERROR_PATTERNS)
        
        # Check if error matches any permanent patterns
        is_permanent = any(re.search(pattern, error_str) for pattern in self.PERMANENT_ERROR_PATTERNS)
        
        # Determine error category
        if isinstance(error, ModelUnavailableError):
            category = "model_unavailable"
            # Model unavailable could be transient if not explicitly permanent
            is_transient = not is_permanent
        elif isinstance(error, TokenLimitError):
            category = "token_limit"
            # Token limit errors are permanent
            is_transient = False
        elif "memory" in error_str or "out of memory" in error_str or "cuda" in error_str:
            category = "resource_constraint"
            is_transient = True
        elif "timeout" in error_str:
            category = "timeout"
            is_transient = True
        elif "rate" in error_str and ("limit" in error_str or "exceeded" in error_str):
            category = "rate_limit"
            is_transient = True
        elif is_transient:
            category = "transient_service_issue"
        elif is_permanent:
            category = "permanent_service_issue"
        else:
            category = "unknown"
            # Default to assuming transient if we can't determine
            is_transient = True
        
        # Record the error classification in metrics
        if self.metrics:
            self.metrics.record_llm_error(
                category, 
                {
                    "error_id": error_id,
                    "error_message": str(error),
                    "is_transient": is_transient,
                    "provider": self.__class__.__name__
                }
            )
        
        return {
            "error_id": error_id,
            "is_transient": is_transient,
            "retry_recommended": is_transient,
            "error_category": category,
            "error_message": str(error)
        }
    
    def handle_error_with_context(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle an error with additional context, classifying and logging it appropriately.
        
        Args:
            error: The exception to handle
            context: Additional context about the error (e.g., operation being performed)
            
        Returns:
            Dict[str, Any]: Complete error info with classification and context
        """
        # Classify the error
        error_info = self.classify_error(error)
        
        # Add context
        if context:
            error_info.update({"context": context})
        
        # Log with appropriate level based on classification
        if error_info["is_transient"]:
            logger.warning(
                f"Transient error ({error_info['error_category']}): {error_info['error_message']} "
                f"[ID: {error_info['error_id']}]"
            )
        else:
            logger.error(
                f"Permanent error ({error_info['error_category']}): {error_info['error_message']} "
                f"[ID: {error_info['error_id']}]"
            )
        
        return error_info
        
    def start_memory_tracking(self) -> None:
        """
        Start tracking memory usage over time for long-running operations.
        This creates a background thread that samples memory usage at regular intervals.
        
        The results can be accessed via self.memory_timeline after the operation completes.
        """
        if self.memory_tracking_enabled:
            return  # Already tracking
            
        with self.memory_tracking_lock:
            self.memory_timeline = []
            self.memory_tracking_enabled = True
            
            # Define the tracking function
            def track_memory():
                process = psutil.Process()
                start_time = time.time()
                
                while self.memory_tracking_enabled:
                    # System memory
                    system_memory = psutil.virtual_memory()
                    
                    # Process memory
                    process_memory = process.memory_info()
                    
                    # GPU memory if available
                    gpu_memory = {}
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            gpu_memory[f"gpu_{i}"] = {
                                "allocated": torch.cuda.memory_allocated(i) / (1024 ** 2),  # MB
                                "reserved": torch.cuda.memory_reserved(i) / (1024 ** 2),    # MB
                                "max_allocated": torch.cuda.max_memory_allocated(i) / (1024 ** 2)  # MB
                            }
                    
                    # Collect memory metrics
                    memory_point = {
                        "timestamp": time.time() - start_time,
                        "system": {
                            "total": system_memory.total / (1024 ** 2),  # MB
                            "available": system_memory.available / (1024 ** 2),  # MB
                            "percent": system_memory.percent
                        },
                        "process": {
                            "rss": process_memory.rss / (1024 ** 2),  # MB
                            "vms": process_memory.vms / (1024 ** 2)   # MB
                        },
                        "gpu": gpu_memory,
                        "threads": threading.active_count()
                    }
                    
                    # Add to timeline
                    with self.memory_tracking_lock:
                        self.memory_timeline.append(memory_point)
                    
                    # Collect garbage to ensure accurate readings
                    gc.collect()
                    
                    # Sleep for the sampling interval
                    time.sleep(self.memory_sampling_interval)
            
            # Start tracking thread
            self.memory_tracker_thread = threading.Thread(
                target=track_memory, 
                daemon=True,  # Daemon thread will exit when main thread exits
                name="MemoryTracker"
            )
            self.memory_tracker_thread.start()
            
            if self.metrics:
                self.metrics.record_custom_metric("memory_tracking", "started", 1)
    
    def stop_memory_tracking(self) -> List[Dict[str, Any]]:
        """
        Stop tracking memory usage and return the collected timeline.
        
        Returns:
            List[Dict[str, Any]]: Timeline of memory usage samples
        """
        if not self.memory_tracking_enabled:
            return self.memory_timeline
            
        with self.memory_tracking_lock:
            self.memory_tracking_enabled = False
            
            # Wait for thread to finish if it exists
            if self.memory_tracker_thread and self.memory_tracker_thread.is_alive():
                self.memory_tracker_thread.join(timeout=2.0)
                
            # Clone the timeline to return
            timeline = list(self.memory_timeline)
            
            # Record metrics if available
            if self.metrics and timeline:
                # Record peak memory usage
                process_peak = max(point["process"]["rss"] for point in timeline)
                self.metrics.record_custom_metric("memory_tracking", "peak_process_mb", process_peak)
                
                # Record GPU peak if available
                if timeline[0].get("gpu") and any(timeline[0]["gpu"]):
                    for gpu_id in timeline[0]["gpu"].keys():
                        peak_gpu = max(point["gpu"][gpu_id]["allocated"] for point in timeline)
                        self.metrics.record_custom_metric(
                            "memory_tracking", 
                            f"peak_gpu_{gpu_id}_mb", 
                            peak_gpu
                        )
                
                # Record average usage
                avg_process = sum(point["process"]["rss"] for point in timeline) / len(timeline)
                self.metrics.record_custom_metric("memory_tracking", "avg_process_mb", avg_process)
                
                # Record memory growth (last - first)
                memory_growth = timeline[-1]["process"]["rss"] - timeline[0]["process"]["rss"]
                self.metrics.record_custom_metric("memory_tracking", "memory_growth_mb", memory_growth)
                
                # Record sample count
                self.metrics.record_custom_metric("memory_tracking", "sample_count", len(timeline))
            
            return timeline

    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """
        Get the current memory usage timeline without stopping tracking.
        
        Returns:
            List[Dict[str, Any]]: Current timeline of memory usage samples
        """
        with self.memory_tracking_lock:
            return list(self.memory_timeline)
            
    def detect_memory_bottlenecks(self) -> Dict[str, Any]:
        """
        Analyze the memory timeline to detect potential bottlenecks.
        
        Returns:
            Dict[str, Any]: Analysis results with identified bottlenecks
        """
        with self.memory_tracking_lock:
            if not self.memory_timeline:
                return {"error": "No memory timeline data available"}
                
            timeline = list(self.memory_timeline)
            
        # Calculate memory growth rate
        if len(timeline) < 2:
            return {"error": "Insufficient timeline data for analysis"}
            
        # Analyze process memory
        first_point = timeline[0]
        last_point = timeline[-1]
        duration = last_point["timestamp"] - first_point["timestamp"]
        
        if duration <= 0:
            return {"error": "Invalid timeline duration"}
            
        rss_growth = last_point["process"]["rss"] - first_point["process"]["rss"]
        growth_rate_mb_per_sec = rss_growth / duration
        
        # Detect sudden spikes
        rss_values = [point["process"]["rss"] for point in timeline]
        avg_rss = sum(rss_values) / len(rss_values)
        peak_rss = max(rss_values)
        peak_ratio = peak_rss / avg_rss if avg_rss > 0 else 1.0
        
        # Find the highest spike
        max_spike = 0
        for i in range(1, len(timeline)):
            spike = timeline[i]["process"]["rss"] - timeline[i-1]["process"]["rss"]
            max_spike = max(max_spike, spike)
        
        # GPU analysis if available
        gpu_analysis = {}
        if timeline[0].get("gpu") and any(timeline[0]["gpu"]):
            for gpu_id in timeline[0]["gpu"].keys():
                gpu_values = [point["gpu"][gpu_id]["allocated"] for point in timeline]
                avg_gpu = sum(gpu_values) / len(gpu_values)
                peak_gpu = max(gpu_values)
                gpu_growth = gpu_values[-1] - gpu_values[0]
                
                gpu_analysis[gpu_id] = {
                    "peak_mb": peak_gpu,
                    "average_mb": avg_gpu,
                    "growth_mb": gpu_growth,
                    "growth_rate_mb_per_sec": gpu_growth / duration,
                    "potential_leak": gpu_growth > 100 and gpu_growth / duration > 10
                }
        
        # Determine if we have potential memory issues
        has_memory_leak = growth_rate_mb_per_sec > 10 and duration > 5
        has_spike = peak_ratio > 2.0
        
        result = {
            "duration_seconds": duration,
            "process_memory": {
                "start_mb": first_point["process"]["rss"],
                "end_mb": last_point["process"]["rss"],
                "peak_mb": peak_rss,
                "growth_mb": rss_growth,
                "growth_rate_mb_per_sec": growth_rate_mb_per_sec,
                "max_spike_mb": max_spike,
                "potential_leak": has_memory_leak,
                "has_spike": has_spike
            },
            "gpu_memory": gpu_analysis,
            "system_memory": {
                "start_percent": first_point["system"]["percent"],
                "end_percent": last_point["system"]["percent"],
                "available_end_mb": last_point["system"]["available"]
            },
            "bottlenecks": []
        }
        
        # Identify bottlenecks
        if has_memory_leak:
            result["bottlenecks"].append({
                "type": "memory_leak",
                "severity": "high" if growth_rate_mb_per_sec > 50 else "medium",
                "details": f"Memory growing at {growth_rate_mb_per_sec:.2f} MB/sec"
            })
            
        if has_spike:
            result["bottlenecks"].append({
                "type": "memory_spike",
                "severity": "high" if peak_ratio > 5 else "medium",
                "details": f"Memory spike of {max_spike:.2f} MB detected"
            })
            
        if last_point["system"]["percent"] > 90:
            result["bottlenecks"].append({
                "type": "system_memory_pressure",
                "severity": "high",
                "details": f"System memory usage at {last_point['system']['percent']}%"
            })
            
        for gpu_id, analysis in gpu_analysis.items():
            if analysis.get("potential_leak"):
                result["bottlenecks"].append({
                    "type": "gpu_memory_leak",
                    "severity": "high",
                    "device": gpu_id,
                    "details": f"GPU memory growing at {analysis['growth_rate_mb_per_sec']:.2f} MB/sec"
                })
        
        return result
        
    def cleanup(self, force: bool = False) -> bool:
        """
        Free resources used by the LLM provider.
        
        This method should be called when the provider is no longer needed
        to ensure proper resource management, especially for GPU memory.
        
        Args:
            force: If True, force cleanup even if operations might be in progress
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        cleanup_start = time.time()
        
        try:
            # Record cleanup operation start
            if self.metrics:
                self.metrics.record_custom_metric(
                    "resource_management", 
                    "cleanup_triggered", 
                    1, 
                    {"force": force}
                )
            
            # Stop memory tracking if active
            if self.memory_tracking_enabled:
                self.stop_memory_tracking()
            
            # Provider-specific cleanup should be implemented in subclasses
            # The base implementation just handles common resources
            
            # Remove any stored model references that might keep GPU memory
            if hasattr(self, 'model') and self.model is not None:
                if hasattr(self.model, 'to'):
                    try:
                        # Move model to CPU if possible
                        self.model.to('cpu')
                    except Exception as e:
                        logger.warning(f"Failed to move model to CPU during cleanup: {e}")
                
                # Clear model reference
                self.model = None
                
            # Reset initialization flag
            if hasattr(self, 'initialized'):
                self.initialized = False
                
            # Try to trigger Python garbage collection
            gc.collect()
            
            # If using PyTorch, empty CUDA cache
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if self.metrics:
                        self.metrics.record_custom_metric(
                            "resource_management", 
                            "gpu_cache_cleared", 
                            1
                        )
            except Exception as e:
                logger.warning(f"Failed to clear CUDA cache during cleanup: {e}")
            
            # Call parent class cleanup
            self._cleanup_impl()
            
            # Record successful cleanup
            cleanup_time_ms = (time.time() - cleanup_start) * 1000
            if self.metrics:
                self.metrics.record_custom_metric(
                    "resource_management", 
                    "cleanup_time_ms", 
                    cleanup_time_ms
                )
                self.metrics.record_model_usage(
                    self.model_name, 
                    {"model_cleaned_up": True}
                )
            
            logger.info(f"Successfully cleaned up resources for {self.__class__.__name__}")
            return True
            
        except Exception as e:
            # Record cleanup failure
            if self.metrics:
                self.metrics.record_llm_error(
                    "cleanup_error", 
                    {"error": str(e), "force": force}
                )
            
            logger.error(f"Failed to clean up resources for {self.__class__.__name__}: {str(e)}")
            return False
    
    def __del__(self):
        """
        Destructor to ensure resources are cleaned up when the object is garbage collected.
        """
        try:
            self.cleanup()
        except Exception as e:
            # Just log, don't raise during garbage collection
            logger.warning(f"Error during auto-cleanup in __del__: {e}")

    # Context manager support for automatic cleanup
    def __enter__(self):
        """
        Context manager entry - allows using the provider in a 'with' statement.
        """
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - ensures cleanup when exiting a 'with' block.
        """
        self.cleanup()
        return False  # Don't suppress exceptions 