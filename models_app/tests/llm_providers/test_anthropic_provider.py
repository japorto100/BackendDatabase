"""
Tests for the Anthropic LLM Provider
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
from django.test import TestCase

from models_app.llm_providers.anthropic_provider import AnthropicLLMProvider


class TestAnthropicLLMProvider(TestCase):
    """Test cases for AnthropicLLMProvider"""

    def setUp(self):
        """Set up test environment"""
        self.config = {
            "api_key": "test_api_key",
            "model": "claude-3-sonnet",
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 1.0
        }
        self.provider = AnthropicLLMProvider(self.config)

    @patch('anthropic.Anthropic')
    def test_initialization(self, mock_anthropic):
        """Test provider initialization"""
        # Verify the Anthropic client was initialized with the correct API key
        self.assertEqual(self.provider.model, "claude-3-sonnet")
        self.assertEqual(self.provider.temperature, 0.7)
        self.assertEqual(self.provider.max_tokens, 500)
        mock_anthropic.assert_called_once_with(api_key="test_api_key")

    @patch('anthropic.Anthropic')
    def test_generate_text(self, mock_anthropic):
        """Test text generation"""
        # Mock the Anthropic client response
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "This is a test response"
        mock_response.stop_reason = "stop"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        
        mock_client.messages.create.return_value = mock_response
        
        # Call the generate_text method
        prompt = "Test prompt"
        response, confidence = self.provider.generate_text(prompt)
        
        # Verify the response
        self.assertEqual(response, "This is a test response")
        self.assertGreater(confidence, 0.5)  # Confidence should be reasonable
        
        # Verify the Anthropic client was called with the correct parameters
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]
        self.assertEqual(call_args["model"], "claude-3-sonnet")
        self.assertEqual(call_args["temperature"], 0.7)
        self.assertEqual(call_args["max_tokens"], 500)
        self.assertEqual(len(call_args["messages"]), 1)
        self.assertEqual(call_args["messages"][0]["role"], "user")
        self.assertEqual(call_args["messages"][0]["content"], "Test prompt")

    @patch('anthropic.Anthropic')
    def test_generate_batch(self, mock_anthropic):
        """Test batch text generation"""
        # Mock the Anthropic client response
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        # Set up multiple mock responses
        def side_effect(**kwargs):
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            prompt = kwargs["messages"][0]["content"]
            mock_response.content[0].text = f"Response to: {prompt}"
            mock_response.stop_reason = "stop"
            mock_response.usage.input_tokens = 10
            mock_response.usage.output_tokens = 20
            return mock_response
        
        mock_client.messages.create.side_effect = side_effect
        
        # Call the generate_batch method
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = self.provider.generate_batch(prompts)
        
        # Verify the responses
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0][0], "Response to: Prompt 1")
        self.assertEqual(responses[1][0], "Response to: Prompt 2")
        self.assertEqual(responses[2][0], "Response to: Prompt 3")
        
        # Verify the Anthropic client was called the correct number of times
        self.assertEqual(mock_client.messages.create.call_count, 3)

    @patch('anthropic.Anthropic')
    def test_error_handling(self, mock_anthropic):
        """Test error handling during text generation"""
        # Mock the Anthropic client to raise an exception
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")
        
        # Call the generate_text method
        prompt = "Test prompt"
        response, confidence = self.provider.generate_text(prompt)
        
        # Verify the error response
        self.assertTrue("Error" in response)
        self.assertEqual(confidence, 0.0)  # Confidence should be zero on error 