"""
Base Speech-to-Text (STT) service interface.

This module defines the base abstract class for all STT services,
providing a common interface and shared functionality.

Design Rationale:
- Establishes a common contract for all STT implementations
- Centralizes error handling and performance tracking
- Enables plug-and-play substitution of different STT engines
- Provides a consistent approach to model management and resources
"""

import os
import logging
import abc
from typing import Dict, Any, List, Optional, Union, Callable
import time
from functools import wraps

from django.conf import settings

from models_app.ai_models.utils.common.errors import (
    AudioModelError, STTError, AudioProcessingError, ModelNotFoundError
)
from models_app.ai_models.utils.common.metrics import get_stt_metrics
from models_app.ai_models.utils.common.config import STTConfig
from models_app.ai_models.utils.common.ai_base_service import BaseModelService
from error_handlers.common_handlers import handle_errors, measure_time, retry

logger = logging.getLogger(__name__)

class BaseSTTService(BaseModelService):
    """
    Base abstract class for all Speech-to-Text services.
    
    This abstract class defines the interface that all STT service 
    implementations must adhere to, providing a common contract and
    shared functionality.
    
    Attributes:
        name (str): Name of the STT service
        model_name (str): Name of the model to use
        cache_dir (str): Directory to cache models
        device (str): Device to use for inference (cpu, cuda, etc.)
        config (STTConfig): Configuration for the service
        metrics (STTMetricsCollector): Collector for service metrics
    """
    
    def __init__(
        self, 
        name: str,
        model_name: str = "base", 
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[STTConfig] = None
    ):
        """
        Initialize the STT service.
        
        Args:
            name: Name of the STT service
            model_name: Name of the model to use
            cache_dir: Directory to cache models
            device: Device to use for inference (cpu, cuda, etc.)
            config: Configuration for the service
        """
        # Initialize the base class
        super().__init__(name, model_name, config)
        
        # Initialize configuration
        self.config = config or STTConfig()
        
        # Override config with explicit parameters if provided
        if cache_dir:
            self.config.cache_dir = cache_dir
        if device:
            self.config.device = device
            
        # Ensure cache directory exists
        if not os.path.exists(self.config.cache_dir):
            os.makedirs(self.config.cache_dir, exist_ok=True)
            
        # Initialize metrics collector
        self.metrics = get_stt_metrics(self.name)
        
        logger.info(f"Initialized {self.name} STT service with model {model_name}")
    
    @handle_errors(
        error_types=[Exception], 
        fallback_return=None,
        error_class=AudioModelError
    )
    def _load_model(self) -> Any:
        """
        Load the STT model.
        
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
        fallback_return={"text": "", "error": "Transcription failed"},
        error_class=STTError
    )
    @measure_time
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        This method handles the transcription process, ensuring the model is loaded,
        and providing error handling and performance measurement.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional arguments for the transcription
            
        Returns:
            Dict[str, Any]: Transcription result with text and metadata
            
        Raises:
            STTError: If transcription fails
            AudioProcessingError: If audio processing fails
        """
        # Start timing the operation
        start_time = self.metrics.start_operation("transcribe")
        
        try:
            # Ensure model is loaded
            if self.loaded_model is None:
                self.loaded_model = self._load_model()
                
            # Process audio (timing separately)
            audio_start = self.metrics.start_operation("audio_processing")
            audio_data, duration = self._process_audio(audio_path, **kwargs)
            audio_process_time = self.metrics.stop_operation("audio_processing", audio_start, success=True)
            
            # Record audio processed
            self.metrics.record_audio_processed(duration, audio_process_time)
            
            # Perform transcription (timing separately)
            transcribe_start = self.metrics.start_operation("transcription")
            result = self._transcribe_impl(audio_data, **kwargs)
            transcribe_time = self.metrics.stop_operation("transcription", transcribe_start, success=True)
            
            # Record transcription
            self.metrics.record_transcription(result, transcribe_time)
            
            # Stop overall operation timing
            self.metrics.stop_operation("transcribe", start_time, success=True)
            
            return result
        except Exception as e:
            # Record failed operation
            self.metrics.stop_operation("transcribe", start_time, success=False)
            
            # Record error details
            self.metrics.record_transcription_error(
                error_type=type(e).__name__,
                details={"message": str(e)}
            )
            
            # Re-raise for the decorator to handle
            raise e
    
    @abc.abstractmethod
    def _transcribe_impl(self, audio_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Implementation of audio transcription.
        
        This abstract method must be implemented by subclasses to perform
        the actual transcription using their specific model implementations.
        
        Args:
            audio_data: Processed audio data
            **kwargs: Additional arguments for the transcription
            
        Returns:
            Dict[str, Any]: Transcription result with text and metadata
        """
        pass
    
    @handle_errors(
        error_types=[Exception],
        fallback_return=(None, 0),
        error_class=AudioProcessingError
    )
    def _process_audio(self, audio_path: str, **kwargs) -> tuple:
        """
        Process audio file for transcription.
        
        This method handles audio preprocessing, ensuring proper format and
        loading for the model.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional arguments for audio processing
            
        Returns:
            tuple: Processed audio data and duration
            
        Raises:
            AudioProcessingError: If audio processing fails
        """
        # Start timing the operation
        start_time = self.metrics.start_operation("process_audio")
        
        try:
            # Process audio implementation (to be provided by subclass)
            result = self._process_audio_impl(audio_path, **kwargs)
            
            # Record successful operation
            self.metrics.stop_operation("process_audio", start_time, success=True)
            return result
        except Exception as e:
            # Record failed operation
            self.metrics.stop_operation("process_audio", start_time, success=False)
            # Re-raise for the decorator to handle
            raise e
    
    @abc.abstractmethod
    def _process_audio_impl(self, audio_path: str, **kwargs) -> tuple:
        """
        Implementation of audio processing.
        
        This abstract method must be implemented by subclasses to perform
        the actual audio processing for their specific models.
        
        Args:
            audio_path: Path to the audio file
            **kwargs: Additional arguments for audio processing
            
        Returns:
            tuple: Processed audio data and duration
        """
        pass
    
    @handle_errors(
        error_types=[Exception],
        fallback_return=[],
        error_class=AudioModelError
    )
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models for this service.
        
        This method provides information about all available models
        for this STT service.
        
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
    def list_languages(self) -> List[Dict[str, Any]]:
        """
        List supported languages for this service.
        
        This method provides information about all supported languages
        for this STT service.
        
        Returns:
            List[Dict[str, Any]]: List of supported languages with metadata
            
        Raises:
            AudioModelError: If listing languages fails
        """
        # Start timing the operation
        start_time = self.metrics.start_operation("list_languages")
        
        try:
            # List languages implementation (to be provided by subclass)
            result = self._list_languages_impl()
            
            # Record successful operation
            self.metrics.stop_operation("list_languages", start_time, success=True)
            return result
        except Exception as e:
            # Record failed operation
            self.metrics.stop_operation("list_languages", start_time, success=False)
            # Re-raise for the decorator to handle
            raise e
    
    def _list_languages_impl(self) -> List[Dict[str, Any]]:
        """
        Implementation of listing supported languages.
        
        This method can be overridden by subclasses to provide a custom
        implementation of listing supported languages.
        
        Returns:
            List[Dict[str, Any]]: List of supported languages with metadata
        """
        # Default implementation that can be overridden
        return [
            {"code": "en", "name": "English"}
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