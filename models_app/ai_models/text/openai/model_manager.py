"""
OpenAILLMModelManager

Manages OpenAI cloud-based large language models.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class OpenAILLMModelManager:
    """
    Manages configuration and initialization of OpenAI LLM models.
    
    This class:
    1. Tracks available OpenAI models
    2. Validates API keys and model configurations
    3. Handles model selection based on capabilities
    """
    
    # Available model configurations
    AVAILABLE_MODELS = {
        'gpt-4-turbo': {
            'description': 'GPT-4 Turbo - latest flagship model with optimal performance across a wide range of tasks',
            'api_name': 'gpt-4-turbo',
            'context_length': 128000,
            'requirements': {
                'api_key': True
            },
            'capabilities': ['chat', 'reasoning', 'document_processing', 'task_decomposition']
        },
        'gpt-4o': {
            'description': 'GPT-4o - optimized model balancing performance and cost',
            'api_name': 'gpt-4o',
            'context_length': 128000,
            'requirements': {
                'api_key': True
            },
            'capabilities': ['chat', 'reasoning', 'document_processing']
        },
        'gpt-3.5-turbo': {
            'description': 'GPT-3.5 Turbo - efficient and cost-effective for simpler tasks',
            'api_name': 'gpt-3.5-turbo',
            'context_length': 16385,
            'requirements': {
                'api_key': True
            },
            'capabilities': ['chat', 'summarization', 'simple_qa']
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the OpenAI LLM model manager.
        
        Args:
            config: Configuration for the model manager
        """
        self.config = config or {}
        self.model_name = self.config.get('model', 'gpt-4-turbo')
        
        # Check for API key in config or environment
        self.api_key = self.config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        self.base_url = self.config.get('base_url') or os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
        
        # Check API organization (optional)
        self.organization = self.config.get('organization') or os.environ.get('OPENAI_ORGANIZATION')
        
        # Model configuration
        self.model_config = self.AVAILABLE_MODELS.get(self.model_name, self.AVAILABLE_MODELS['gpt-4-turbo'])
        
        # Parse additional model-specific configurations
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 1.0)
        self.presence_penalty = self.config.get('presence_penalty', 0.0)
        self.frequency_penalty = self.config.get('frequency_penalty', 0.0)
        
        # Client instance (will be initialized on-demand)
        self.client = None
        
    def is_available(self) -> bool:
        """
        Check if the OpenAI service is available with valid credentials.
        
        Returns:
            bool: True if the service is available, False otherwise
        """
        # Check API key
        if not self.api_key:
            logger.warning("OpenAI API key not found")
            return False
            
        try:
            # Initialize client (this doesn't make an API call yet)
            self._initialize_client()
            
            # Successful initialization
            return True
        except Exception as e:
            logger.error(f"Error checking OpenAI service availability: {str(e)}")
            return False
            
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        if self.client is not None:
            return
            
        try:
            from openai import OpenAI
            
            client_args = {
                "api_key": self.api_key,
                "base_url": self.base_url
            }
            
            if self.organization:
                client_args["organization"] = self.organization
                
            self.client = OpenAI(**client_args)
            logger.info(f"OpenAI client initialized for model {self.model_name}")
            
        except ImportError:
            logger.error("OpenAI Python package not installed. Run: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently selected model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'name': self.model_name,
            'provider': 'OpenAI',
            'description': self.model_config.get('description', ''),
            'context_length': self.model_config.get('context_length', 0),
            'capabilities': self.model_config.get('capabilities', []),
            'api_key_configured': bool(self.api_key)
        }
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available OpenAI models with their configurations.
        
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
        Prepare request parameters for the OpenAI API call.
        
        Args:
            prompt: The user prompt
            system_message: Optional system message to set conversation context
            
        Returns:
            Dict[str, Any]: Request parameters
        """
        # Initialize client if needed
        self._initialize_client()
        
        # Prepare messages
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
            
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Prepare parameters
        params = {
            "model": self.model_config.get('api_name', self.model_name),
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty
        }
        
        return params 