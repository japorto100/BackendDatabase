"""
Base service module for AI model services.

This module provides base classes and utilities for all AI model services,
defining common patterns and abstractions for model management, error handling,
and resource control.

Design Rationale:
- Establishes consistent service interfaces across the application
- Centralizes service lifecycle management (initialization, usage, cleanup)
- Provides unified approach to error handling and metrics collection
- Ensures consistent resource management across different model types
"""

import os
import logging
import abc
from typing import Dict, Any, List, Optional, Union, Type
import time
import threading
from dataclasses import dataclass

from django.conf import settings

from models_app.ai_models.utils.common.errors import ModelError, ModelNotFoundError
from models_app.ai_models.utils.common.config import BaseAudioConfig
from models_app.ai_models.utils.common.metrics import MetricsCollector
from error_handlers.common_handlers import handle_errors

logger = logging.getLogger(__name__)

class BaseModelService(abc.ABC):
    """
    Abstract base class for all AI model services.
    
    This class defines the common interface and functionality that all
    AI model services should implement, providing consistency and
    shared behavior across different types of services.
    
    The BaseModelService:
    - Handles initialization with consistent parameters
    - Manages model loading and caching
    - Provides error handling through decorators
    - Collects metrics on model usage
    - Manages resources through proper lifecycle methods
    
    Attributes:
        name (str): Name of the service
        model_name (str): Name of the model to use
        metrics (MetricsCollector): Collector for service metrics
        loaded_model: The loaded model object
    """
    
    def __init__(
        self,
        name: str,
        model_name: str = "base",
        config: Optional[BaseAudioConfig] = None
    ):
        """
        Initialize the model service.
        
        Args:
            name: Name of the service
            model_name: Name of the model to use
            config: Configuration for the service
        """
        self.name = name
        self.model_name = model_name
        self.loaded_model = None
        self.config = config
        self._model_lock = threading.RLock()
        
        logger.info(f"Initialized {self.name} service with model {model_name}")
    
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
        fallback_return=None,
        error_class=ModelError
    )
    def _load_model(self) -> Any:
        """
        Load the model.
        
        This method handles model loading with proper locking to ensure
        thread safety. It is decorated with error handling to ensure
        consistent error management.
        
        Returns:
            Any: Loaded model object
            
        Raises:
            ModelNotFoundError: If the model is not found
            ModelError: For other model loading errors
        """
        with self._model_lock:
            if self.loaded_model is not None:
                return self.loaded_model
                
            logger.info(f"Loading model {self.model_name} for {self.name} service")
            self.loaded_model = self._load_model_impl()
            return self.loaded_model
    
    @abc.abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models for this service.
        
        This method provides information about all available models
        for this service.
        
        Returns:
            List[Dict[str, Any]]: List of available models with metadata
        """
        pass
    
    @handle_errors(
        error_types=[Exception],
        fallback_return=None,
        error_class=ModelError
    )
    def cleanup(self) -> None:
        """
        Clean up resources used by the service.
        
        This method handles cleaning up any resources used by the service,
        such as loaded models, cached files, etc.
        
        Raises:
            ModelError: If cleanup fails
        """
        with self._model_lock:
            # Cleanup implementation (to be provided by subclass)
            self._cleanup_impl()
            self.loaded_model = None
    
    def _cleanup_impl(self) -> None:
        """
        Implementation of resource cleanup.
        
        This method can be overridden by subclasses to provide a custom
        implementation of resource cleanup.
        """
        # Default implementation that can be overridden
        self.loaded_model = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dict[str, Any]: Information about the model
        """
        return {
            "name": self.model_name,
            "service": self.name,
            "loaded": self.loaded_model is not None,
        }
    
    @classmethod
    def supported_models(cls) -> List[str]:
        """
        Get list of supported models for this service class.
        
        Returns:
            List[str]: List of supported model names
        """
        return ["base"]


class ModelRegistry:
    """
    Registry for AI model services.
    
    This class provides a centralized registry for AI model services,
    allowing for dynamic registration and retrieval of services.
    
    The ModelRegistry:
    - Maintains a dictionary of service instances by type and name
    - Provides methods to register, retrieve, and list services
    - Ensures singleton service instances for each type and name
    """
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'ModelRegistry':
        """
        Get singleton instance of ModelRegistry.
        
        Returns:
            ModelRegistry: Singleton instance
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the model registry."""
        self._services = {}
        self._service_types = {}
    
    def register_service_type(self, service_type: str, service_class: Type[BaseModelService]) -> None:
        """
        Register a service type with its class.
        
        Args:
            service_type: Type of the service (e.g., 'stt', 'tts')
            service_class: Class of the service
        """
        self._service_types[service_type] = service_class
    
    def register_service(self, service_type: str, service_name: str, service: BaseModelService) -> None:
        """
        Register a service instance.
        
        Args:
            service_type: Type of the service (e.g., 'stt', 'tts')
            service_name: Name of the service (e.g., 'whisper_faster')
            service: Service instance
        """
        if service_type not in self._services:
            self._services[service_type] = {}
        self._services[service_type][service_name] = service
    
    def get_service(self, service_type: str, service_name: str) -> Optional[BaseModelService]:
        """
        Get a service instance.
        
        Args:
            service_type: Type of the service (e.g., 'stt', 'tts')
            service_name: Name of the service (e.g., 'whisper_faster')
            
        Returns:
            Optional[BaseModelService]: Service instance, or None if not found
        """
        if service_type not in self._services:
            return None
        return self._services[service_type].get(service_name)
    
    def create_service(self, service_type: str, service_name: str, model_name: str = "base", **kwargs) -> Optional[BaseModelService]:
        """
        Create and register a service instance.
        
        Args:
            service_type: Type of the service (e.g., 'stt', 'tts')
            service_name: Name of the service (e.g., 'whisper_faster')
            model_name: Name of the model to use
            **kwargs: Additional arguments for service creation
            
        Returns:
            Optional[BaseModelService]: Service instance, or None if service type not registered
        """
        if service_type not in self._service_types:
            return None
            
        service_class = self._service_types[service_type]
        service = service_class(service_name, model_name, **kwargs)
        self.register_service(service_type, service_name, service)
        return service
    
    def get_or_create_service(self, service_type: str, service_name: str, model_name: str = "base", **kwargs) -> Optional[BaseModelService]:
        """
        Get or create a service instance.
        
        Args:
            service_type: Type of the service (e.g., 'stt', 'tts')
            service_name: Name of the service (e.g., 'whisper_faster')
            model_name: Name of the model to use
            **kwargs: Additional arguments for service creation
            
        Returns:
            Optional[BaseModelService]: Service instance, or None if service type not registered
        """
        service = self.get_service(service_type, service_name)
        if service is None:
            service = self.create_service(service_type, service_name, model_name, **kwargs)
        return service
    
    def list_services(self, service_type: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List available services.
        
        Args:
            service_type: Type of the service (e.g., 'stt', 'tts'), or None for all
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping service types to lists of service names
        """
        if service_type is not None:
            if service_type not in self._services:
                return {service_type: []}
            return {service_type: list(self._services[service_type].keys())}
            
        result = {}
        for stype in self._services:
            result[stype] = list(self._services[stype].keys())
        return result
    
    def cleanup_all(self) -> None:
        """Clean up all registered services."""
        for service_type in self._services:
            for service_name, service in self._services[service_type].items():
                try:
                    service.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up service {service_name} of type {service_type}: {str(e)}")


# Helper functions for easier access
def register_service_type(service_type: str, service_class: Type[BaseModelService]) -> None:
    """
    Register a service type with its class.
    
    Args:
        service_type: Type of the service (e.g., 'stt', 'tts')
        service_class: Class of the service
    """
    ModelRegistry.get_instance().register_service_type(service_type, service_class)

def get_service(service_type: str, service_name: str) -> Optional[BaseModelService]:
    """
    Get a service instance.
    
    Args:
        service_type: Type of the service (e.g., 'stt', 'tts')
        service_name: Name of the service (e.g., 'whisper_faster')
        
    Returns:
        Optional[BaseModelService]: Service instance, or None if not found
    """
    return ModelRegistry.get_instance().get_service(service_type, service_name)

def create_service(service_type: str, service_name: str, model_name: str = "base", **kwargs) -> Optional[BaseModelService]:
    """
    Create and register a service instance.
    
    Args:
        service_type: Type of the service (e.g., 'stt', 'tts')
        service_name: Name of the service (e.g., 'whisper_faster')
        model_name: Name of the model to use
        **kwargs: Additional arguments for service creation
        
    Returns:
        Optional[BaseModelService]: Service instance, or None if service type not registered
    """
    return ModelRegistry.get_instance().create_service(service_type, service_name, model_name, **kwargs)

def get_or_create_service(service_type: str, service_name: str, model_name: str = "base", **kwargs) -> Optional[BaseModelService]:
    """
    Get or create a service instance.
    
    Args:
        service_type: Type of the service (e.g., 'stt', 'tts')
        service_name: Name of the service (e.g., 'whisper_faster')
        model_name: Name of the model to use
        **kwargs: Additional arguments for service creation
        
    Returns:
        Optional[BaseModelService]: Service instance, or None if service type not registered
    """
    return ModelRegistry.get_instance().get_or_create_service(service_type, service_name, model_name, **kwargs)

def list_services(service_type: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List available services.
    
    Args:
        service_type: Type of the service (e.g., 'stt', 'tts'), or None for all
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping service types to lists of service names
    """
    return ModelRegistry.get_instance().list_services(service_type)

def cleanup_all() -> None:
    """Clean up all registered services."""
    ModelRegistry.get_instance().cleanup_all()
