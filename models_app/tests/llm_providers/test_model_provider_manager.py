"""
Tests for the Model Provider Manager
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
from django.test import TestCase
from django.contrib.auth.models import User

from users_app.model_provider import ModelProviderManager
from models_app.llm_providers.openai_provider import OpenAILLMProvider
from models_app.llm_providers.anthropic_provider import AnthropicLLMProvider
from models_app.llm_providers.deepseek_provider import DeepSeekLLMProvider
from models_app.llm_providers.local_provider import LocalLLMProvider
from models_app.llm_providers.template_provider import TemplateProviderFactory


class TestModelProviderManager(TestCase):
    """Test cases for ModelProviderManager"""

    def setUp(self):
        """Set up test environment"""
        # Create a test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpassword'
        )
        
        # Create user settings
        from users_app.models import UserSettings
        self.user_settings = UserSettings.objects.create(
            user=self.user,
            generation_model='gpt-3.5-turbo',
            hyde_model='deepseek-coder-1.3b-instruct',
            use_hyde=True
        )
        
        # Create the manager
        self.manager = ModelProviderManager(user=self.user)

    def test_initialization(self):
        """Test manager initialization"""
        # Verify the manager was initialized with the user
        self.assertEqual(self.manager.user, self.user)
        
        # Verify default provider config
        default_config = self.manager.get_provider_config()
        self.assertIsNotNone(default_config)
        self.assertIn('provider', default_config)

    def test_get_provider_config(self):
        """Test getting provider config"""
        # Test with default parameters
        config = self.manager.get_provider_config()
        self.assertEqual(config['provider'], 'openai')  # Assuming default is OpenAI
        
        # Test with specific provider
        config = self.manager.get_provider_config(provider_name='anthropic')
        self.assertEqual(config['provider'], 'anthropic')
        
        # Test with specific model
        config = self.manager.get_provider_config(model_name='claude-3-sonnet')
        self.assertEqual(config['provider'], 'anthropic')
        self.assertEqual(config['model'], 'claude-3-sonnet')
        
        # Test with non-existent model
        config = self.manager.get_provider_config(model_name='non-existent-model')
        self.assertIsNotNone(config)  # Should return a default config

    @patch('models_app.llm_providers.openai_provider.OpenAILLMProvider')
    def test_get_provider_instance_openai(self, mock_openai_provider):
        """Test getting OpenAI provider instance"""
        # Set up mock provider
        mock_instance = MagicMock()
        mock_openai_provider.return_value = mock_instance
        
        # Get provider instance
        provider = self.manager.get_provider_instance(provider_name='openai')
        
        # Verify the correct provider type was created
        self.assertEqual(provider, mock_instance)
        mock_openai_provider.assert_called_once()

    @patch('models_app.llm_providers.anthropic_provider.AnthropicLLMProvider')
    def test_get_provider_instance_anthropic(self, mock_anthropic_provider):
        """Test getting Anthropic provider instance"""
        # Set up mock provider
        mock_instance = MagicMock()
        mock_anthropic_provider.return_value = mock_instance
        
        # Get provider instance
        provider = self.manager.get_provider_instance(provider_name='anthropic')
        
        # Verify the correct provider type was created
        self.assertEqual(provider, mock_instance)
        mock_anthropic_provider.assert_called_once()

    @patch('models_app.llm_providers.deepseek_provider.DeepSeekLLMProvider')
    def test_get_provider_instance_deepseek(self, mock_deepseek_provider):
        """Test getting DeepSeek provider instance"""
        # Set up mock provider
        mock_instance = MagicMock()
        mock_deepseek_provider.return_value = mock_instance
        
        # Get provider instance
        provider = self.manager.get_provider_instance(provider_name='deepseek')
        
        # Verify the correct provider type was created
        self.assertEqual(provider, mock_instance)
        mock_deepseek_provider.assert_called_once()

    @patch('models_app.llm_providers.local_provider.LocalLLMProvider')
    def test_get_provider_instance_local(self, mock_local_provider):
        """Test getting Local provider instance"""
        # Set up mock provider
        mock_instance = MagicMock()
        mock_local_provider.return_value = mock_instance
        
        # Get provider instance
        provider = self.manager.get_provider_instance(provider_name='local')
        
        # Verify the correct provider type was created
        self.assertEqual(provider, mock_instance)
        mock_local_provider.assert_called_once()

    @patch('models_app.llm_providers.template_provider.TemplateProviderFactory.create_provider')
    def test_get_provider_instance_template(self, mock_template_factory):
        """Test getting Template provider instance"""
        # Set up mock provider
        mock_instance = MagicMock()
        mock_template_factory.return_value = mock_instance
        
        # Update user settings to use custom model
        self.user_settings.use_custom_model = True
        self.user_settings.custom_model_url = "https://huggingface.co/mistralai/Mistral-7B-v0.1"
        self.user_settings.save()
        
        # Get provider instance
        provider = self.manager.get_provider_instance(provider_name='template')
        
        # Verify the correct provider type was created
        self.assertEqual(provider, mock_instance)
        mock_template_factory.assert_called_once()
        
        # Verify the URL was passed correctly
        call_args = mock_template_factory.call_args
        self.assertEqual(call_args[0][0], "https://huggingface.co/mistralai/Mistral-7B-v0.1")

    def test_get_provider_for_user(self):
        """Test getting provider for user"""
        # Mock all provider instances
        with patch('users_app.model_provider.ModelProviderManager.get_provider_instance') as mock_get_provider:
            mock_instance = MagicMock()
            mock_get_provider.return_value = mock_instance
            
            # Get provider for user with default settings
            provider = self.manager.get_provider_for_user()
            
            # Verify the correct provider was returned
            self.assertEqual(provider, mock_instance)
            mock_get_provider.assert_called_once()
            
            # Reset mock
            mock_get_provider.reset_mock()
            
            # Update user settings to use custom model
            self.user_settings.use_custom_model = True
            self.user_settings.custom_model_url = "https://example.com/model"
            self.user_settings.save()
            
            # Get provider for user with custom model
            provider = self.manager.get_provider_for_user()
            
            # Verify the template provider was used
            self.assertEqual(provider, mock_instance)
            mock_get_provider.assert_called_once_with(provider_name='template')

    def test_get_hyde_provider(self):
        """Test getting HyDE provider"""
        # Mock all provider instances
        with patch('users_app.model_provider.ModelProviderManager.get_provider_instance') as mock_get_provider:
            mock_instance = MagicMock()
            mock_get_provider.return_value = mock_instance
            
            # Get HyDE provider
            provider = self.manager.get_hyde_provider()
            
            # Verify the correct provider was returned
            self.assertEqual(provider, mock_instance)
            mock_get_provider.assert_called_once()
            
            # Verify the model name was passed correctly
            call_args = mock_get_provider.call_args
            self.assertEqual(call_args[1]['model_name'], 'deepseek-coder-1.3b-instruct')

    def test_get_provider_for_model(self):
        """Test getting provider for specific model"""
        # Mock all provider instances
        with patch('users_app.model_provider.ModelProviderManager.get_provider_instance') as mock_get_provider:
            mock_instance = MagicMock()
            mock_get_provider.return_value = mock_instance
            
            # Get provider for GPT-4
            provider = self.manager.get_provider_for_model('gpt-4')
            
            # Verify the correct provider was returned
            self.assertEqual(provider, mock_instance)
            mock_get_provider.assert_called_once()
            
            # Verify the model name was passed correctly
            call_args = mock_get_provider.call_args
            self.assertEqual(call_args[1]['model_name'], 'gpt-4')
            
            # Reset mock
            mock_get_provider.reset_mock()
            
            # Get provider for Claude
            provider = self.manager.get_provider_for_model('claude-3-opus')
            
            # Verify the correct provider was returned
            self.assertEqual(provider, mock_instance)
            mock_get_provider.assert_called_once()
            
            # Verify the model name was passed correctly
            call_args = mock_get_provider.call_args
            self.assertEqual(call_args[1]['model_name'], 'claude-3-opus') 