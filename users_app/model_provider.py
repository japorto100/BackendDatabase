"""
Model Provider Manager

Verwaltet die Auswahl und Konfiguration von LLM-Providern basierend auf Benutzereinstellungen.
"""

import logging
from django.conf import settings
from .models import UserProfile, UserSettings
from models_app.llm_providers.template_provider import TemplateProviderFactory
from models_app.llm_providers.openai_provider import OpenAILLMProvider
from models_app.llm_providers.anthropic_provider import AnthropicLLMProvider
from models_app.llm_providers.deepseek_provider import DeepSeekLLMProvider
from models_app.llm_providers.local_provider import LocalLLMProvider

logger = logging.getLogger(__name__)

class ModelProviderManager:
    """
    Verwaltet die Auswahl und Konfiguration von LLM-Providern.
    """
    
    def __init__(self, user=None):
        """
        Initialisiert den Manager mit optionalem Benutzer.
        
        Args:
            user: Der Benutzer, dessen Einstellungen verwendet werden sollen
        """
        self.user = user
        self.default_provider = getattr(settings, 'DEFAULT_LLM_PROVIDER', 'openai')
        self.default_model = getattr(settings, 'DEFAULT_LLM_MODEL', 'gpt-3.5-turbo')
    
    def get_provider_config(self, provider_name=None, model_name=None):
        """
        Gibt die Konfiguration für einen Provider zurück.
        
        Args:
            provider_name: Name des Providers (optional)
            model_name: Name des Modells (optional)
            
        Returns:
            dict: Provider-Konfiguration
        """
        # Wenn kein Benutzer angegeben ist, verwende Standardeinstellungen
        if not self.user:
            return self._get_default_config(provider_name, model_name)
        
        try:
            profile = self.user.profile
            settings_obj = self.user.settings
            
            # Wenn kein Provider angegeben ist, verwende den Standard des Benutzers
            provider = provider_name or profile.default_provider
            model = model_name or profile.default_model
            
            config = {
                'provider': provider,
                'model': model,
                'temperature': profile.temperature,
                'max_tokens': profile.max_tokens,
                'top_p': profile.top_p,
            }
            
            # Provider-spezifische Konfiguration
            if provider == 'openai':
                config['api_key'] = profile.openai_api_key or getattr(settings, 'OPENAI_API_KEY', '')
            elif provider == 'anthropic':
                config['api_key'] = profile.anthropic_api_key or getattr(settings, 'ANTHROPIC_API_KEY', '')
            elif provider == 'deepseek':
                config['api_key'] = profile.deepseek_api_key or getattr(settings, 'DEEPSEEK_API_KEY', '')
            elif provider == 'local':
                config['use_gpu'] = profile.use_gpu
                config['quantization_level'] = profile.quantization_level
                config['local_models_path'] = settings_obj.local_models_path
                config['batch_size'] = settings_obj.batch_size
                config['cache_models'] = settings_obj.cache_models
            
            return config
            
        except Exception as e:
            logger.error(f"Error getting provider config: {str(e)}")
            return self._get_default_config(provider_name, model_name)
    
    def _get_default_config(self, provider_name=None, model_name=None):
        """
        Gibt die Standard-Konfiguration zurück.
        
        Args:
            provider_name: Name des Providers (optional)
            model_name: Name des Modells (optional)
            
        Returns:
            dict: Standard-Konfiguration
        """
        provider = provider_name or self.default_provider
        model = model_name or self.default_model
        
        config = {
            'provider': provider,
            'model': model,
            'temperature': 0.7,
            'max_tokens': 1000,
            'top_p': 0.9,
        }
        
        # Provider-spezifische Konfiguration
        if provider == 'openai':
            config['api_key'] = getattr(settings, 'OPENAI_API_KEY', '')
        elif provider == 'anthropic':
            config['api_key'] = getattr(settings, 'ANTHROPIC_API_KEY', '')
        elif provider == 'deepseek':
            config['api_key'] = getattr(settings, 'DEEPSEEK_API_KEY', '')
        elif provider == 'local':
            config['use_gpu'] = True
            config['quantization_level'] = '4bit'
            config['local_models_path'] = 'models/'
            config['batch_size'] = 1
            config['cache_models'] = True
        
        return config
    
    def get_provider_instance(self, provider_name=None, model_name=None):
        """
        Gibt eine Instanz des LLM-Providers zurück.
        
        Args:
            provider_name: Name des Providers (optional)
            model_name: Name des Modells (optional)
            
        Returns:
            object: Provider-Instanz
        """
        config = self.get_provider_config(provider_name, model_name)
        provider = config['provider']
        
        if provider == 'openai':
            from models_app.llm_providers.openai_provider import OpenAILLMProvider
            return OpenAILLMProvider(config)
        elif provider == 'anthropic':
            from models_app.llm_providers.anthropic_provider import AnthropicLLMProvider
            return AnthropicLLMProvider(config)
        elif provider == 'deepseek':
            from models_app.llm_providers.deepseek_provider import DeepSeekLLMProvider
            return DeepSeekLLMProvider(config)
        elif provider == 'local':
            from models_app.llm_providers.local_provider import LocalLLMProvider
            return LocalLLMProvider(config)
        elif provider == 'template':
            # Verwende den TemplateProviderFactory, um einen passenden Provider zu erstellen
            model_url = config.get('model_url', '')
            return TemplateProviderFactory.create_provider(model_url, config)
        else:
            # Fallback auf einen einfachen Provider
            from models_app.hyde_processor import SimpleLLMProvider
            return SimpleLLMProvider() 