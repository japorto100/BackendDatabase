"""
Tests for the Template Provider Factory
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
from django.test import TestCase

from models_app.llm_providers.template_provider import TemplateProviderFactory


class TestTemplateProviderFactory(TestCase):
    """Test cases for TemplateProviderFactory"""

    def setUp(self):
        """Set up test environment"""
        self.base_config = {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 1.0
        }

    @patch('models_app.llm_providers.template_provider.HuggingFaceTemplateLLMProvider')
    def test_huggingface_url_detection(self, mock_hf_provider):
        """Test detection of Hugging Face URLs"""
        # Set up mock provider
        mock_instance = MagicMock()
        mock_hf_provider.return_value = mock_instance
        
        # Test with various Hugging Face URLs
        urls = [
            "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            "https://huggingface.co/meta-llama/Llama-2-7b-chat-hf",
            "huggingface.co/facebook/opt-350m"
        ]
        
        for url in urls:
            config = self.base_config.copy()
            config["model_url"] = url
            
            # Create provider
            provider = TemplateProviderFactory.create_provider(url, config)
            
            # Verify the correct provider type was created
            self.assertEqual(provider, mock_instance)
            mock_hf_provider.assert_called_with(config)

    @patch('models_app.llm_providers.template_provider.GitHubTemplateLLMProvider')
    def test_github_url_detection(self, mock_github_provider):
        """Test detection of GitHub URLs"""
        # Set up mock provider
        mock_instance = MagicMock()
        mock_github_provider.return_value = mock_instance
        
        # Test with various GitHub URLs
        urls = [
            "https://github.com/stanford-crfm/mistral",
            "github.com/facebookresearch/llama",
            "https://github.com/EleutherAI/gpt-neox"
        ]
        
        for url in urls:
            config = self.base_config.copy()
            config["model_url"] = url
            
            # Create provider
            provider = TemplateProviderFactory.create_provider(url, config)
            
            # Verify the correct provider type was created
            self.assertEqual(provider, mock_instance)
            mock_github_provider.assert_called_with(config)

    @patch('models_app.llm_providers.template_provider.LocalFileTemplateLLMProvider')
    def test_local_path_detection(self, mock_local_provider):
        """Test detection of local file paths"""
        # Set up mock provider
        mock_instance = MagicMock()
        mock_local_provider.return_value = mock_instance
        
        # Test with various local paths
        paths = [
            "/home/user/models/mistral-7b",
            "C:\\Users\\user\\models\\llama",
            "./models/gpt-neox"
        ]
        
        for path in paths:
            config = self.base_config.copy()
            config["model_url"] = path
            
            # Create provider
            provider = TemplateProviderFactory.create_provider(path, config)
            
            # Verify the correct provider type was created
            self.assertEqual(provider, mock_instance)
            mock_local_provider.assert_called_with(config)

    @patch('models_app.llm_providers.template_provider.GenericTemplateLLMProvider')
    def test_fallback_provider(self, mock_generic_provider):
        """Test fallback to generic provider for unknown URLs"""
        # Set up mock provider
        mock_instance = MagicMock()
        mock_generic_provider.return_value = mock_instance
        
        # Test with various unknown URLs
        urls = [
            "https://example.com/model",
            "ftp://models.org/gpt",
            "unknown://path/to/model"
        ]
        
        for url in urls:
            config = self.base_config.copy()
            config["model_url"] = url
            
            # Create provider
            provider = TemplateProviderFactory.create_provider(url, config)
            
            # Verify the fallback provider was created
            self.assertEqual(provider, mock_instance)
            mock_generic_provider.assert_called_with(config)

    def test_provider_creation_with_invalid_input(self):
        """Test provider creation with invalid input"""
        # Test with None URL
        provider = TemplateProviderFactory.create_provider(None, self.base_config)
        self.assertIsNotNone(provider)  # Should return a generic provider
        
        # Test with empty URL
        provider = TemplateProviderFactory.create_provider("", self.base_config)
        self.assertIsNotNone(provider)  # Should return a generic provider
        
        # Test with invalid config
        provider = TemplateProviderFactory.create_provider("https://huggingface.co/model", None)
        self.assertIsNotNone(provider)  # Should handle None config gracefully 