"""
AnthropicLLMModelManager

Manages Anthropic Claude large language models.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class AnthropicLLMModelManager:
    """
    Manages configuration and initialization of Anthropic Claude LLM models.
    
    This class:
    1. Tracks available Claude models
    2. Validates API keys and model configurations
    3. Handles model selection based on capabilities
    """
    
    # Available model configurations
    AVAILABLE_MODELS = {
        'claude-3-opus-20240229': {
            'description': 'Claude 3 Opus - most powerful model with superior reasoning and instruction following',
            'api_name': 'claude-3-opus-20240229',
            'context_length': 200000,
            'requirements': {
                'api_key': True
            },
            'capabilities': ['chat', 'reasoning', 'document_processing', 'task_decomposition']
        },
        'claude-3-sonnet-20240229': {
            'description': 'Claude 3 Sonnet - balanced model for most use cases, balancing performance and cost',
            'api_name': 'claude-3-sonnet-20240229',
            'context_length': 200000,
            'requirements': {
                'api_key': True
            },
            'capabilities': ['chat', 'reasoning', 'document_processing']
        },
        'claude-3-haiku-20240307': {
            'description': 'Claude 3 Haiku - fastest and most efficient model for simple tasks',
            'api_name': 'claude-3-haiku-20240307',
            'context_length': 200000,
            'requirements': {
                'api_key': True
            },
            'capabilities': ['chat', 'summarization', 'simple_qa']
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Anthropic LLM model manager.
        
        Args:
            config: Configuration for the model manager
        """
        self.config = config or {}
        self.model_name = self.config.get('model', 'claude-3-sonnet-20240229')
        
        # Check for API key in config or environment
        self.api_key = self.config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        self.base_url = self.config.get('base_url') or os.environ.get('ANTHROPIC_API_BASE', 'https://api.anthropic.com')
        
        # Model configuration
        self.model_config = self.AVAILABLE_MODELS.get(self.model_name, self.AVAILABLE_MODELS['claude-3-sonnet-20240229'])
        
        # Parse additional model-specific configurations
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 1.0)
        self.top_k = self.config.get('top_k', None)
        
        # Client instance (will be initialized on-demand)
        self.client = None
        
    def is_available(self) -> bool:
        """
        Check if the Anthropic service is available with valid credentials.
        
        Returns:
            bool: True if the service is available, False otherwise
        """
        # Check API key
        if not self.api_key:
            logger.warning("Anthropic API key not found")
            return False
            
        try:
            # Initialize client (this doesn't make an API call yet)
            self._initialize_client()
            
            # Successful initialization
            return True
        except Exception as e:
            logger.error(f"Error checking Anthropic service availability: {str(e)}")
            return False
            
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        if self.client is not None:
            return
            
        try:
            from anthropic import Anthropic
            
            self.client = Anthropic(api_key=self.api_key)
            logger.info(f"Anthropic client initialized for model {self.model_name}")
            
        except ImportError:
            logger.error("Anthropic Python package not installed. Run: pip install anthropic")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently selected model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'name': self.model_name,
            'provider': 'Anthropic',
            'description': self.model_config.get('description', ''),
            'context_length': self.model_config.get('context_length', 0),
            'capabilities': self.model_config.get('capabilities', []),
            'api_key_configured': bool(self.api_key)
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available Anthropic models with their configurations.
        
        Returns:
            List[Dict[str, Any]]: List of model configurations
        """
        models = []
        
        for name, config in self.AVAILABLE_MODELS.items():
            models.append({
                'name': name,
                'description': config.get('description', ''),
                'context_length': config.get('context_length', 0),
                'capabilities': config.get('capabilities', [])
            })
            
        return models

    def prepare_request_parameters(self, prompt: str, system_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare request parameters for the Anthropic API call.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message to set conversation context
            
        Returns:
            Dict[str, Any]: Request parameters
        """
        # Initialize client if needed
        self._initialize_client()
        
        # Prepare parameters
        params = {
            "model": self.model_config.get('api_name', self.model_name),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Add system message if provided
        if system_message:
            params["system"] = system_message
            
        # Add optional parameters if specified
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.top_k is not None:
            params["top_k"] = self.top_k
        
        return params 