"""
Tests for the OpenAI LLM Provider
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
from django.test import TestCase

from models_app.llm_providers.openai_provider import OpenAILLMProvider


class TestOpenAILLMProvider(TestCase):
    """Test cases for OpenAILLMProvider"""

    def setUp(self):
        """Set up test environment"""
        self.config = {
            "api_key": "test_api_key",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 1.0
        }
        self.provider = OpenAILLMProvider(self.config)

    @patch('openai.OpenAI')
    def test_initialization(self, mock_openai):
        """Test provider initialization"""
        # Verify the OpenAI client was initialized with the correct API key
        self.assertEqual(self.provider.model, "gpt-3.5-turbo")
        self.assertEqual(self.provider.temperature, 0.7)
        self.assertEqual(self.provider.max_tokens, 500)
        mock_openai.assert_called_once_with(api_key="test_api_key")

    @patch('openai.OpenAI')
    def test_generate_text(self, mock_openai):
        """Test text generation"""
        # Mock the OpenAI client response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Call the generate_text method
        prompt = "Test prompt"
        response, confidence = self.provider.generate_text(prompt)
        
        # Verify the response
        self.assertEqual(response, "This is a test response")
        self.assertGreater(confidence, 0.5)  # Confidence should be reasonable
        
        # Verify the OpenAI client was called with the correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-3.5-turbo")
        self.assertEqual(call_args["temperature"], 0.7)
        self.assertEqual(call_args["max_tokens"], 500)
        self.assertEqual(len(call_args["messages"]), 1)
        self.assertEqual(call_args["messages"][0]["role"], "user")
        self.assertEqual(call_args["messages"][0]["content"], "Test prompt")

    @patch('openai.OpenAI')
    def test_generate_batch(self, mock_openai):
        """Test batch text generation"""
        # Mock the OpenAI client response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Set up multiple mock responses
        def side_effect(**kwargs):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            prompt = kwargs["messages"][0]["content"]
            mock_response.choices[0].message.content = f"Response to: {prompt}"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            return mock_response
        
        mock_client.chat.completions.create.side_effect = side_effect
        
        # Call the generate_batch method
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = self.provider.generate_batch(prompts)
        
        # Verify the responses
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0][0], "Response to: Prompt 1")
        self.assertEqual(responses[1][0], "Response to: Prompt 2")
        self.assertEqual(responses[2][0], "Response to: Prompt 3")
        
        # Verify the OpenAI client was called the correct number of times
        self.assertEqual(mock_client.chat.completions.create.call_count, 3)

    @patch('openai.OpenAI')
    def test_error_handling(self, mock_openai):
        """Test error handling during text generation"""
        # Mock the OpenAI client to raise an exception
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        # Call the generate_text method
        prompt = "Test prompt"
        response, confidence = self.provider.generate_text(prompt)
        
        # Verify the error response
        self.assertTrue("Error" in response)
        self.assertEqual(confidence, 0.0)  # Confidence should be zero on error 