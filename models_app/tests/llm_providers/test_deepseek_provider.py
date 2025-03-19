"""
Tests für den DeepSeekLLMProvider.

Diese Tests überprüfen die Funktionalität des DeepSeekLLMProviders, einschließlich:
- Modellinitialisierung
- Textgenerierung
- Fehlerbehandlung
- Ressourcennutzung
"""

import unittest
from unittest.mock import patch, MagicMock
import torch
from django.test import TestCase

from models_app.llm_providers.deepseek_provider import DeepSeekLLMProvider


class TestDeepSeekLLMProvider(TestCase):
    """Tests für den DeepSeekLLMProvider."""
    
    def setUp(self):
        """Testumgebung einrichten."""
        # Standardkonfiguration für Tests
        self.config = {
            'model': 'deepseek-ai/deepseek-coder-1.3b-instruct',
            'temperature': 0.7,
            'max_tokens': 100,
            'top_p': 0.9,
            'use_gpu': False,  # CPU für Tests verwenden
            'quantization_level': '4bit'
        }
    
    @patch('models_app.llm_providers.deepseek_provider.AutoModelForCausalLM')
    @patch('models_app.llm_providers.deepseek_provider.AutoTokenizer')
    def test_initialization(self, mock_tokenizer, mock_model):
        """Test der Initialisierung des Providers."""
        # Mock-Objekte einrichten
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        # Provider initialisieren
        provider = DeepSeekLLMProvider(self.config)
        
        # Überprüfen, ob die Methoden aufgerufen wurden
        mock_tokenizer.from_pretrained.assert_called_once_with(self.config['model'])
        mock_model.from_pretrained.assert_called_once()
        
        # Überprüfen, ob die Konfiguration korrekt übernommen wurde
        self.assertEqual(provider.model_name, self.config['model'])
        self.assertEqual(provider.temperature, self.config['temperature'])
        self.assertEqual(provider.max_tokens, self.config['max_tokens'])
        self.assertEqual(provider.top_p, self.config['top_p'])
        self.assertEqual(provider.use_gpu, self.config['use_gpu'])
        self.assertEqual(provider.quantization, self.config['quantization_level'])
    
    @patch('models_app.llm_providers.deepseek_provider.AutoModelForCausalLM')
    @patch('models_app.llm_providers.deepseek_provider.AutoTokenizer')
    def test_generate_text(self, mock_tokenizer, mock_model):
        """Test der Textgenerierung."""
        # Mock-Objekte einrichten
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Tokenizer-Verhalten simulieren
        mock_tokenizer_instance.decode.side_effect = lambda ids, **kwargs: "This is a test response" if isinstance(ids, torch.Tensor) else "This is a test prompt"
        mock_tokenizer_instance.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))
        
        # Modell-Verhalten simulieren
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.device = 'cpu'
        
        # Provider initialisieren und Text generieren
        provider = DeepSeekLLMProvider(self.config)
        response, confidence = provider.generate_text("This is a test prompt")
        
        # Überprüfen, ob die Methoden aufgerufen wurden
        mock_model_instance.generate.assert_called_once()
        
        # Überprüfen der Antwort
        self.assertEqual(response, "This is a test response")
        self.assertGreater(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    @patch('models_app.llm_providers.deepseek_provider.AutoModelForCausalLM')
    @patch('models_app.llm_providers.deepseek_provider.AutoTokenizer')
    def test_generate_text_with_error(self, mock_tokenizer, mock_model):
        """Test der Fehlerbehandlung bei der Textgenerierung."""
        # Mock-Objekte einrichten
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        
        # Fehler bei der Generierung simulieren
        mock_model.from_pretrained.return_value.generate.side_effect = Exception("Test error")
        
        # Provider initialisieren und Text generieren
        provider = DeepSeekLLMProvider(self.config)
        response, confidence = provider.generate_text("This is a test prompt")
        
        # Überprüfen der Fehlerbehandlung
        self.assertIn("Error generating text", response)
        self.assertEqual(confidence, 0.0)
    
    @patch('models_app.llm_providers.deepseek_provider.AutoModelForCausalLM')
    @patch('models_app.llm_providers.deepseek_provider.AutoTokenizer')
    def test_generate_batch(self, mock_tokenizer, mock_model):
        """Test der Batch-Textgenerierung."""
        # Mock-Objekte einrichten
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Tokenizer-Verhalten simulieren
        mock_tokenizer_instance.decode.side_effect = lambda ids, **kwargs: "This is a test response" if isinstance(ids, torch.Tensor) else "This is a test prompt"
        mock_tokenizer_instance.return_value = MagicMock(input_ids=torch.tensor([[1, 2, 3]]))
        
        # Modell-Verhalten simulieren
        mock_model_instance.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_instance.device = 'cpu'
        
        # Provider initialisieren und Batch-Text generieren
        provider = DeepSeekLLMProvider(self.config)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = provider.generate_batch(prompts)
        
        # Überprüfen, ob die Methoden aufgerufen wurden
        self.assertEqual(mock_model_instance.generate.call_count, 3)
        
        # Überprüfen der Antworten
        self.assertEqual(len(responses), 3)
        for response, confidence in responses:
            self.assertEqual(response, "This is a test response")
            self.assertGreater(confidence, 0)
            self.assertLessEqual(confidence, 1)
    
    @unittest.skipIf(not torch.cuda.is_available(), "GPU not available")
    def test_gpu_usage(self):
        """Test der GPU-Nutzung (nur wenn GPU verfügbar)."""
        # Konfiguration mit GPU
        config = self.config.copy()
        config['use_gpu'] = True
        
        # Provider initialisieren
        provider = DeepSeekLLMProvider(config)
        
        # Überprüfen, ob das Modell auf der GPU ist
        self.assertIn('cuda', provider.model.device.type)


if __name__ == '__main__':
    unittest.main() 