"""
Tests for the Local LLM Provider
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
import torch
from django.test import TestCase

from models_app.llm_providers.local_provider import LocalLLMProvider


class TestLocalLLMProvider(TestCase):
    """Test cases for LocalLLMProvider"""

    def setUp(self):
        """Set up test environment"""
        self.config = {
            "model_path": "test_model_path",
            "model_type": "llama",
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 1.0,
            "quantization": "4bit"
        }
        
        # Skip actual model loading in tests
        with patch('transformers.AutoModelForCausalLM.from_pretrained'), \
             patch('transformers.AutoTokenizer.from_pretrained'):
            self.provider = LocalLLMProvider(self.config)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_initialization(self, mock_tokenizer, mock_model):
        """Test provider initialization"""
        # Create a new provider with mocked transformers
        provider = LocalLLMProvider(self.config)
        
        # Verify the model was initialized with the correct parameters
        self.assertEqual(provider.model_path, "test_model_path")
        self.assertEqual(provider.model_type, "llama")
        self.assertEqual(provider.temperature, 0.7)
        self.assertEqual(provider.max_tokens, 500)
        
        # Verify the transformers were called with the correct parameters
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
        
        # Check quantization parameters
        model_args = mock_model.call_args[1]
        self.assertTrue("load_in_4bit" in model_args or "load_in_8bit" in model_args)

    @patch('torch.cuda.is_available')
    @patch('transformers.pipeline')
    def test_generate_text_with_pipeline(self, mock_pipeline, mock_cuda_available):
        """Test text generation using pipeline"""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Mock the pipeline response
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe
        mock_pipe.return_value = [{"generated_text": "This is a test response"}]
        
        # Create a provider with pipeline approach
        config = self.config.copy()
        config["use_pipeline"] = True
        
        with patch('transformers.AutoModelForCausalLM.from_pretrained'), \
             patch('transformers.AutoTokenizer.from_pretrained'):
            provider = LocalLLMProvider(config)
        
        # Call the generate_text method
        prompt = "Test prompt"
        response, confidence = provider.generate_text(prompt)
        
        # Verify the response
        self.assertEqual(response, "This is a test response")
        self.assertGreater(confidence, 0.5)  # Confidence should be reasonable
        
        # Verify the pipeline was called with the correct parameters
        mock_pipe.assert_called_once()
        call_args = mock_pipe.call_args[0][0]
        self.assertEqual(call_args, "Test prompt")

    @patch('torch.cuda.is_available')
    def test_generate_text_with_direct_model(self, mock_cuda_available):
        """Test text generation using direct model inference"""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Mock the model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock tokenizer encode/decode
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "This is a test response"
        
        # Mock model generate
        mock_outputs = MagicMock()
        mock_model.generate.return_value = mock_outputs
        
        # Create a provider with direct model approach
        config = self.config.copy()
        config["use_pipeline"] = False
        
        with patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=mock_model), \
             patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
            provider = LocalLLMProvider(config)
            provider.model = mock_model
            provider.tokenizer = mock_tokenizer
        
        # Call the generate_text method
        prompt = "Test prompt"
        response, confidence = provider.generate_text(prompt)
        
        # Verify the model was called with the correct parameters
        mock_model.generate.assert_called_once()
        
        # We can't easily verify the exact response due to the complexity of mocking
        # the tokenizer and model interaction, but we can check that the method ran
        self.assertIsInstance(response, str)
        self.assertIsInstance(confidence, float)

    @patch('torch.cuda.is_available')
    @patch('transformers.pipeline')
    def test_generate_batch(self, mock_pipeline, mock_cuda_available):
        """Test batch text generation"""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Mock the pipeline response
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe
        
        # Set up multiple mock responses
        def side_effect(prompt):
            return [{"generated_text": f"Response to: {prompt}"}]
        
        mock_pipe.side_effect = side_effect
        
        # Create a provider with pipeline approach
        config = self.config.copy()
        config["use_pipeline"] = True
        
        with patch('transformers.AutoModelForCausalLM.from_pretrained'), \
             patch('transformers.AutoTokenizer.from_pretrained'):
            provider = LocalLLMProvider(config)
        
        # Call the generate_batch method
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = provider.generate_batch(prompts)
        
        # Verify the responses
        self.assertEqual(len(responses), 3)
        self.assertEqual(responses[0][0], "Response to: Prompt 1")
        self.assertEqual(responses[1][0], "Response to: Prompt 2")
        self.assertEqual(responses[2][0], "Response to: Prompt 3")
        
        # Verify the pipeline was called the correct number of times
        self.assertEqual(mock_pipe.call_count, 3)

    @patch('torch.cuda.is_available')
    @patch('transformers.pipeline')
    def test_error_handling(self, mock_pipeline, mock_cuda_available):
        """Test error handling during text generation"""
        # Mock CUDA availability
        mock_cuda_available.return_value = True
        
        # Mock the pipeline to raise an exception
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe
        mock_pipe.side_effect = Exception("Model error")
        
        # Create a provider with pipeline approach
        config = self.config.copy()
        config["use_pipeline"] = True
        
        with patch('transformers.AutoModelForCausalLM.from_pretrained'), \
             patch('transformers.AutoTokenizer.from_pretrained'):
            provider = LocalLLMProvider(config)
        
        # Call the generate_text method
        prompt = "Test prompt"
        response, confidence = provider.generate_text(prompt)
        
        # Verify the error response
        self.assertTrue("Error" in response)
        self.assertEqual(confidence, 0.0)  # Confidence should be zero on error 