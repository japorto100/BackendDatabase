"""
Tests für den LayoutLMv3Adapter unter Verwendung der Basis-Testklasse.
"""

import numpy as np
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
import unittest

from models_app.ocr.layoutlmv3_adapter import LayoutLMv3Adapter
from models_app.tests.ocr.test_base_adapter import BaseOCRAdapterTest

class TestLayoutLMv3Adapter(BaseOCRAdapterTest):
    """Test-Klasse für den LayoutLMv3Adapter."""
    
    ADAPTER_CLASS = LayoutLMv3Adapter
    CONFIG_DICT = {
            'model_name': 'microsoft/layoutlmv3-base',
        'task': 'document_understanding',
            'max_length': 512,
        'gpu': False
    }
    MOCK_IMPORTS = [
        'models_app.ocr.layoutlmv3_adapter.LayoutLMv3Processor',
        'models_app.ocr.layoutlmv3_adapter.LayoutLMv3Model',
        'models_app.ocr.layoutlmv3_adapter.LayoutLMv3ForTokenClassification'
    ]
    
    @patch('models_app.ocr.layoutlmv3_adapter.LAYOUTLM_AVAILABLE', False)
    def test_initialize_layoutlm_not_available(self):
        """Test der Initialisierung, wenn LayoutLM nicht verfügbar ist."""
        result = self.adapter.initialize()
        
        self.assertFalse(result)
        self.assertFalse(self.adapter.is_initialized)
        self.assertIn("error", result)
        self.assertIn("not available", result["metadata"]["error"])
    
    @patch('models_app.ocr.layoutlmv3_adapter.LAYOUTLM_AVAILABLE', True)
    @patch('models_app.ocr.layoutlmv3_adapter.torch.device')
    @patch('models_app.ocr.layoutlmv3_adapter.torch.cuda.is_available')
    @patch('models_app.ocr.layoutlmv3_adapter.LayoutLMv3Processor.from_pretrained')
    @patch('models_app.ocr.layoutlmv3_adapter.LayoutLMv3Model.from_pretrained')
    def test_initialize_model(self, mock_model, mock_processor, mock_cuda, mock_device):
        """Test der Initialisierung des Modells."""
        # Mocks einrichten
        mock_cuda.return_value = False
        mock_device.return_value = torch.device('cpu')
        
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Adapter mit document_understanding Task initialisieren
        self.adapter.config['task'] = 'document_understanding'
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        mock_processor.assert_called_once()
        mock_model.assert_called_once()
        mock_model_instance.to.assert_called_once()
    
    @patch('models_app.ocr.layoutlmv3_adapter.LAYOUTLM_AVAILABLE', True)
    @patch.object(LayoutLMv3Adapter, 'initialize')
    @patch.object(LayoutLMv3Adapter, '_process_with_layoutlmv3')
    def test_process_image_with_file_path(self, mock_process, mock_init):
        """Test der Bildverarbeitung mit einem Dateipfad."""
        # Mocks einrichten
        mock_init.return_value = True
        
        # Mockup-Ergebnis für die Verarbeitung
        mock_result = {
            "text": "This is a test document with LayoutLMv3.",
            "blocks": [
                {
                    "text": "This is a test document",
                    "bbox": [10, 10, 200, 50],
                    "conf": 0.95
                },
                {
                    "text": "with LayoutLMv3.",
                    "bbox": [10, 60, 200, 100],
                    "conf": 0.9
                }
            ],
            "confidence": 0.925,
            "model": "LayoutLMv3Adapter",
            "language": "en",
            "metadata": {
                "task": "document_understanding"
            }
        }
        mock_process.return_value = mock_result
        
        # Adapter als initialisiert markieren
        self.adapter.is_initialized = True
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertEqual(result, mock_result)
        mock_process.assert_called_once_with(self.test_image_path, None)
    
    @patch('models_app.ocr.layoutlmv3_adapter.LAYOUTLM_AVAILABLE', True)
    @patch.object(LayoutLMv3Adapter, 'initialize')
    @patch.object(LayoutLMv3Adapter, '_process_with_layoutlmv3')
    def test_process_image_with_options(self, mock_process, mock_init):
        """Test der Bildverarbeitung mit zusätzlichen Optionen."""
        # Mocks einrichten
        mock_init.return_value = True
        
        # Mockup-Ergebnis für die Verarbeitung
        mock_result = {
            "text": "Test with options",
            "blocks": [
                {
                    "text": "Test with options",
                    "bbox": [10, 10, 200, 50],
                    "conf": 0.95
                }
            ],
            "confidence": 0.95,
            "model": "LayoutLMv3Adapter",
            "language": "de",
            "metadata": {
                "task": "token_classification"
            }
        }
        mock_process.return_value = mock_result
        
        # Adapter als initialisiert markieren
        self.adapter.is_initialized = True
        
        # Optionen für die Verarbeitung
        options = {
            'task': 'token_classification',
            'language': 'de',
            'apply_ocr': True
        }
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_image_path, options)
        
        # Überprüfen der Ergebnisse
        self.assertEqual(result, mock_result)
        mock_process.assert_called_once_with(self.test_image_path, options)

if __name__ == '__main__':
    unittest.main() 