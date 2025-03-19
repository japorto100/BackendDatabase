"""
GPT4VisionModelManager

Manages OpenAI's GPT-4 Vision API configuration and authentication.
"""

import os
import logging
import requests
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class GPT4VisionModelManager:
    """
    Manages configuration and authentication for OpenAI's GPT-4 Vision API.
    
    This class:
    1. Validates and manages API keys
    2. Tracks available models
    3. Handles configuration for different GPT-4 Vision models
    """
    
    # Available GPT-4 Vision models
    AVAILABLE_MODELS = [
        "gpt-4-vision-preview",
        "gpt-4o",
        "gpt-4o-mini"
    ]
    
    # API endpoint
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the GPT-4 Vision model manager.
        
        Args:
            config: Configuration dictionary including API keys and model settings
        """
        self.config = config or {}
        self.api_key = self._get_api_key()
        self.model_name = self.config.get('model', 'gpt-4-vision-preview')
        self.initialized = False
        self.headers = None
    
    def _get_api_key(self) -> str:
        """
        Get the OpenAI API key from config or environment variables.
        
        Returns:
            str: The API key or empty string if not found
        """
        # Try to get API key from config
        api_key = self.config.get('api_key', '')
        
        # If not in config, try environment variables
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY', '')
        
        return api_key
    
    def is_available(self) -> bool:
        """
        Check if the GPT-4 Vision API is available with valid credentials.
        
        Returns:
            bool: True if the API is accessible, False otherwise
        """
        if not self.api_key:
            logger.warning("OpenAI API key not found")
            return False
        
        if self.model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown GPT-4 Vision model: {self.model_name}")
            return False
        
        # If already initialized successfully, return True
        if self.initialized and self.headers:
            return True
            
        # Try to initialize
        try:
            self.initialize_client()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GPT-4 Vision client: {e}")
            return False
    
    def initialize_client(self):
        """
        Initialize the GPT-4 Vision client for API access.
        
        This method sets up the API headers and validates the connection.
        
        Raises:
            ValueError: If the API key is missing
            RuntimeError: If initialization fails
        """
        if self.initialized and self.headers:
            return
            
        if not self.api_key:
            raise ValueError("API key is required for GPT-4 Vision")
        
        # Set up the headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # Validate connection with a minimal API call
            response = requests.post(
                self.API_URL,
                headers=self.headers,
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, testing API connectivity."}
                    ],
                    "max_tokens": 10,
                    "temperature": 0.0
                }
            )
            
            response.raise_for_status()
            self.initialized = True
            logger.info(f"Successfully initialized GPT-4 Vision client for model: {self.model_name}")
            
        except requests.exceptions.RequestException as e:
            error_message = f"Failed to connect to OpenAI API: {str(e)}"
            if hasattr(e, 'response') and e.response:
                try:
                    error_details = e.response.json()
                    error_message += f" - Details: {error_details.get('error', {}).get('message', 'Unknown error')}"
                except:
                    pass
            
            logger.error(error_message)
            raise RuntimeError(error_message)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured GPT-4 Vision model.
        
        Returns:
            Dict[str, Any]: Model information including name, capabilities and configuration
        """
        model_info = {
            "name": self.model_name,
            "provider": "OpenAI",
            "capabilities": {
                "image_understanding": True,
                "multiple_images": True,
                "text_generation": True
            },
            "api_key_set": bool(self.api_key),
            "initialized": self.initialized
        }
        
        return model_info
    
    def list_available_models(self) -> List[str]:
        """
        List all available GPT-4 Vision models.
        
        Returns:
            List[str]: Names of available models
        """
        return self.AVAILABLE_MODELS.copy()
    
    def set_model(self, model_name: str) -> bool:
        """
        Change the active GPT-4 Vision model.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown GPT-4 Vision model: {model_name}")
            return False
            
        # Reset initialization if model changed
        if self.model_name != model_name:
            self.model_name = model_name
            self.initialized = False
            
        return True 