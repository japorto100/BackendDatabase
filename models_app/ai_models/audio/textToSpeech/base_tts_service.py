"""
Base Text-to-Speech (TTS) service interface.

This module defines the base abstract class for all TTS services,
providing a common interface and shared functionality.

Design Rationale:
- Establishes a common contract for all TTS implementations
- Centralizes error handling and performance tracking
- Enables plug-and-play substitution of different TTS engines
- Provides a consistent approach to model management and resources
"""

import os
import logging
import abc
from typing import Dict, Any, List, Optional, Union, Callable, BinaryIO
import time
from functools import wraps

from django.conf import settings

from models_app.ai_models.utils.common.errors import (
    AudioModelError, TTSError, AudioProcessingError, ModelNotFoundError
)
from models_app.ai_models.utils.common.metrics import get_tts_metrics
from models_app.ai_models.utils.common.config import TTSConfig
from models_app.ai_models.utils.common.ai_base_service import BaseModelService
from error_handlers.common_handlers import handle_errors, measure_time, retry

logger = logging.getLogger(__name__)

class BaseTTSService(BaseModelService):
    """
    Base abstract class for all Text-to-Speech services.
    
    This abstract class defines the interface that all TTS service 
    implementations must adhere to, providing a common contract and
    shared functionality.
    
    Attributes:
        name (str): Name of the TTS service
        model_name (str): Name of the model to use
        cache_dir (str): Directory to cache models
        device (str): Device to use for inference (cpu, cuda, etc.)
        config (TTSConfig): Configuration for the service
        metrics (TTSMetricsCollector): Collector for service metrics
    """
    
    def __init__(
        self, 
        name: str,
        model_name: str = "base", 
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[TTSConfig] = None
    ):
        """
        Initialize the TTS service.
        
        Args:
            name: Name of the TTS service
            model_name: Name of the model to use
            cache_dir: Directory to cache models
            device: Device to use for inference (cpu, cuda, etc.)
            config: Configuration for the service
        """
        # Initialize the base class
        super().__init__(name, model_name, config)
        
        # Initialize configuration
        self.config = config or TTSConfig()
        
        # Override config with explicit parameters if provided
        if cache_dir:
            self.config.cache_dir = cache_dir
        if device:
            self.config.device = device
            
        # Ensure cache directory exists
        if not os.path.exists(self.config.cache_dir):
            os.makedirs(self.config.cache_dir, exist_ok=True)
            
        # Initialize metrics collector
        self.metrics = get_tts_metrics(self.name)
        
        logger.info(f"Initialized {self.name} TTS service with model {model_name}")
    
    @handle_errors(
        error_types=[Exception], 
        fallback_return=None,
        error_class=AudioModelError
    )
    def _load_model(self) -> Any:
        """
        Load the TTS model.
        
        This method should be implemented by subclasses to load their specific
        model implementations. It is decorated with error handling to ensure
        consistent error management.
        
        Returns:
            Any: Loaded model object
            
        Raises:
            ModelNotFoundError: If the model is not found
            AudioModelError: For other model loading errors
        """
        # Start timing the operation
        start_time = self.metrics.start_operation("load_model")
        
        try:
            # Load model implementation (to be provided by subclass)
            model = self._load_model_impl()
            
            # Record successful operation
            self.metrics.stop_operation("load_model", start_time, success=True)
            return model
        except Exception as e:
            # Record failed operation
            self.metrics.stop_operation("load_model", start_time, success=False)
            # Re-raise the exception for the decorator to handle
            raise e
    
    @abc.abstractmethod
    def _load_model_impl(self) -> Any:
        """
        Implementation of model loading.
        
        This abstract method must be implemented by subclasses to load
        their specific model implementations.
        
        Returns:
            Any: Loaded model object
        """
        pass
    
    @handle_errors(
        error_types=[Exception],
        fallback_return={"audio_path": "", "error": "Synthesis failed"},
        error_class=TTSError
    )
    @measure_time
    def synthesize(self, text: str, output_path: Optional[str] = None, voice_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Synthesize text to speech.
        
        This method handles the synthesis process, ensuring the model is loaded,
        and providing error handling and performance measurement.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the audio file, if None a temporary file is used
            voice_id: ID of the voice to use
            **kwargs: Additional arguments for synthesis
            
        Returns:
            Dict[str, Any]: Synthesis result with audio path and metadata
            
        Raises:
            TTSError: If synthesis fails
            AudioProcessingError: If audio processing fails
        """
        # Start timing the operation
        start_time = self.metrics.start_operation("synthesize")
        
        try:
            # Process text (timing separately)
            text_start = self.metrics.start_operation("text_processing")
            processed_text = self._process_text(text, **kwargs)
            text_process_time = self.metrics.stop_operation("text_processing", text_start, success=True)
            
            # Record text processed
            self.metrics.record_text_processed(text, text_process_time)
            
            # Ensure model is loaded
            if self.loaded_model is None:
                self.loaded_model = self._load_model()
                
            # Perform synthesis (timing separately)
            synthesis_start = self.metrics.start_operation("synthesis")
            result = self._synthesize_impl(processed_text, output_path, voice_id, **kwargs)
            synthesis_time = self.metrics.stop_operation("synthesis", synthesis_start, success=True)
            
            # Get audio duration if available
            audio_duration = result.get("duration_seconds", 0)
            
            # Record synthesis metrics
            self.metrics.record_synthesis(audio_duration, synthesis_time, voice_id or "default")
            
            # Stop overall operation timing
            self.metrics.stop_operation("synthesize", start_time, success=True)
            
            return result
        except Exception as e:
            # Record failed operation
            self.metrics.stop_operation("synthesize", start_time, success=False)
            
            # Record error details
            self.metrics.record_synthesis_error(
                error_type=type(e).__name__,
                details={"message": str(e), "text_length": len(text)}
            )
            
            # Re-raise for the decorator to handle
            raise e
    
    @abc.abstractmethod
    def _synthesize_impl(self, processed_text: str, output_path: Optional[str], voice_id: str, **kwargs) -> Dict[str, Any]:
        """
        Implementation of text synthesis.
        
        This abstract method must be implemented by subclasses to perform
        the actual synthesis using their specific model implementations.
        
        Args:
            processed_text: Processed text to synthesize
            output_path: Path to save the audio file
            voice_id: ID of the voice to use
            **kwargs: Additional arguments for synthesis
            
        Returns:
            Dict[str, Any]: Synthesis result with audio path and metadata
        """
        pass
    
    @handle_errors(
        error_types=[Exception],
        fallback_return="",
        error_class=AudioProcessingError
    )
    def _process_text(self, text: str, **kwargs) -> str:
        """
        Process text for synthesis.
        
        This method handles text preprocessing, ensuring proper format for synthesis.
        
        Args:
            text: Text to process
            **kwargs: Additional arguments for text processing
            
        Returns:
            str: Processed text
            
        Raises:
            AudioProcessingError: If text processing fails
        """
        # Start timing the operation
        start_time = self.metrics.start_operation("process_text")
        
        try:
            # Process text implementation (to be provided by subclass)
            result = self._process_text_impl(text, **kwargs)
            
            # Record successful operation
            self.metrics.stop_operation("process_text", start_time, success=True)
            return result
        except Exception as e:
            # Record failed operation
            self.metrics.stop_operation("process_text", start_time, success=False)
            # Re-raise for the decorator to handle
            raise e
    
    def _process_text_impl(self, text: str, **kwargs) -> str:
        """
        Implementation of text processing.
        
        This method can be overridden by subclasses to provide a custom
        implementation of text processing.
        
        Args:
            text: Text to process
            **kwargs: Additional arguments for text processing
            
        Returns:
            str: Processed text
        """
        # Default implementation - can be overridden by subclasses
        return text
    
    @handle_errors(
        error_types=[Exception],
        fallback_return=[],
        error_class=AudioModelError
    )
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models for this service.
        
        This method provides information about all available models
        for this TTS service.
        
        Returns:
            List[Dict[str, Any]]: List of available models with metadata
            
        Raises:
            AudioModelError: If listing models fails
        """
        # Start timing the operation
        start_time = self.metrics.start_operation("list_models")
        
        try:
            # List models implementation (to be provided by subclass)
            result = self._list_models_impl()
            
            # Record successful operation
            self.metrics.stop_operation("list_models", start_time, success=True)
            return result
        except Exception as e:
            # Record failed operation
            self.metrics.stop_operation("list_models", start_time, success=False)
            # Re-raise for the decorator to handle
            raise e
    
    def _list_models_impl(self) -> List[Dict[str, Any]]:
        """
        Implementation of listing available models.
        
        This method can be overridden by subclasses to provide a custom
        implementation of listing available models.
        
        Returns:
            List[Dict[str, Any]]: List of available models with metadata
        """
        # Default implementation that can be overridden
        return [
            {"name": self.model_name, "description": "Currently loaded model"}
        ]
    
    @handle_errors(
        error_types=[Exception],
        fallback_return=[],
        error_class=AudioModelError
    )
    def list_voices(self) -> List[Dict[str, Any]]:
        """
        List available voices for this service.
        
        This method provides information about all available voices
        for this TTS service.
        
        Returns:
            List[Dict[str, Any]]: List of available voices with metadata
            
        Raises:
            AudioModelError: If listing voices fails
        """
        # Start timing the operation
        start_time = self.metrics.start_operation("list_voices")
        
        try:
            # List voices implementation (to be provided by subclass)
            result = self._list_voices_impl()
            
            # Record successful operation
            self.metrics.stop_operation("list_voices", start_time, success=True)
            return result
        except Exception as e:
            # Record failed operation
            self.metrics.stop_operation("list_voices", start_time, success=False)
            # Re-raise for the decorator to handle
            raise e
    
    def _list_voices_impl(self) -> List[Dict[str, Any]]:
        """
        Implementation of listing available voices.
        
        This method can be overridden by subclasses to provide a custom
        implementation of listing available voices.
        
        Returns:
            List[Dict[str, Any]]: List of available voices with metadata
        """
        # Default implementation that can be overridden
        return [
            {"id": "default", "name": "Default Voice", "gender": "neutral", "language": "en"}
        ]
    
    @handle_errors(
        error_types=[Exception],
        fallback_return=None,
        error_class=AudioModelError
    )
    def cleanup(self) -> None:
        """
        Clean up resources used by the service.
        
        This method handles cleaning up any resources used by the service,
        such as loaded models, cached files, etc.
        
        Raises:
            AudioModelError: If cleanup fails
        """
        # Start timing the operation
        start_time = self.metrics.start_operation("cleanup")
        
        try:
            # Cleanup implementation (to be provided by subclass)
            self._cleanup_impl()
            
            # Reset loaded model
            self.loaded_model = None
            
            # Export metrics before resetting
            if self.config.metrics_export_on_cleanup:
                self.metrics.export_metrics()
                
            # Record successful operation and reset metrics if configured
            self.metrics.stop_operation("cleanup", start_time, success=True)
            if self.config.metrics_reset_on_cleanup:
                self.metrics.reset()
        except Exception as e:
            # Record failed operation
            self.metrics.stop_operation("cleanup", start_time, success=False)
            # Re-raise for the decorator to handle
            raise e
    
    def _cleanup_impl(self) -> None:
        """
        Implementation of resource cleanup.
        
        This method can be overridden by subclasses to provide a custom
        implementation of resource cleanup.
        """
        # Default implementation that can be overridden
        self.loaded_model = None
        
        # Export metrics before resetting
        if self.config.metrics_export_on_cleanup:
            self.metrics.export_metrics()
            
        # Reset metrics if configured
        if self.config.metrics_reset_on_cleanup:
            self.metrics.reset()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for this service.
        
        Returns:
            Dict[str, Any]: Metrics for this service
        """
        return self.metrics.get_metrics()
    
    def export_metrics(self, file_path: Optional[str] = None) -> Optional[str]:
        """
        Export metrics to a file.
        
        Args:
            file_path: Path to export metrics to. If None, a default path is used.
            
        Returns:
            Optional[str]: Path to the exported metrics file, or None if export failed
        """
        return self.metrics.export_metrics(file_path) 