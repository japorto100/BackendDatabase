"""
GeminiVisionModelManager

Manages Google Gemini Vision API configuration and authentication.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class GeminiVisionModelManager:
    """
    Manages configuration and authentication for Google's Gemini Vision API.
    
    This class:
    1. Validates and manages API keys
    2. Tracks available models
    3. Handles configuration for different Gemini Vision models
    """
    
    # Available Gemini Vision models
    AVAILABLE_MODELS = [
        "gemini-pro-vision",
        "gemini-1.5-pro-vision",
        "gemini-1.5-flash-vision"
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Gemini Vision model manager.
        
        Args:
            config: Configuration dictionary including API keys and model settings
        """
        self.config = config or {}
        self.api_key = self._get_api_key()
        self.model_name = self.config.get('model', 'gemini-pro-vision')
        self.initialized = False
        self.client = None
    
    def _get_api_key(self) -> str:
        """
        Get the Google API key from config or environment variables.
        
        Returns:
            str: The API key or empty string if not found
        """
        # Try to get API key from config
        api_key = self.config.get('api_key', '')
        
        # If not in config, try environment variables
        if not api_key:
            api_key = os.environ.get('GOOGLE_API_KEY', '')
        
        return api_key
    
    def is_available(self) -> bool:
        """
        Check if the Gemini Vision API is available with valid credentials.
        
        Returns:
            bool: True if the API is accessible, False otherwise
        """
        if not self.api_key:
            logger.warning("Gemini Vision API key not found")
            return False
        
        if self.model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown Gemini Vision model: {self.model_name}")
            return False
        
        # If already initialized successfully, return True
        if self.initialized and self.client:
            return True
            
        # Try to initialize
        try:
            self.initialize_client()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Vision client: {e}")
            return False
    
    def initialize_client(self):
        """
        Initialize the Gemini client for API access.
        
        This method imports and configures the Google Generative AI library.
        
        Raises:
            ImportError: If the required library is not installed
            RuntimeError: If initialization fails
        """
        if self.initialized and self.client:
            return
            
        try:
            # Import the library
            import google.generativeai as genai
            
            # Configure with API key
            if not self.api_key:
                raise ValueError("API key is required for Gemini Vision")
                
            genai.configure(api_key=self.api_key)
            
            # Create a model instance to validate the connection
            self.client = genai.GenerativeModel(self.model_name)
            
            # Validate with a simple call (comment out in production if needed)
            # response = self.client.generate_content("Test connection")
            
            self.initialized = True
            logger.info(f"Successfully initialized Gemini Vision client for model: {self.model_name}")
            
        except ImportError:
            logger.error("Google Generative AI library not installed. Install with: pip install google-generativeai")
            raise ImportError("Required package 'google-generativeai' is not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Vision: {e}")
            raise RuntimeError(f"Failed to initialize Gemini Vision: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured Gemini Vision model.
        
        Returns:
            Dict[str, Any]: Model information including name, capabilities and configuration
        """
        model_info = {
            "name": self.model_name,
            "provider": "Google",
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
        List all available Gemini Vision models.
        
        Returns:
            List[str]: Names of available models
        """
        return self.AVAILABLE_MODELS.copy()
    
    def set_model(self, model_name: str) -> bool:
        """
        Change the active Gemini Vision model.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        if model_name not in self.AVAILABLE_MODELS:
            logger.warning(f"Unknown Gemini Vision model: {model_name}")
            return False
            
        # Reset initialization if model changed
        if self.model_name != model_name:
            self.model_name = model_name
            self.initialized = False
            self.client = None
            
        return True 