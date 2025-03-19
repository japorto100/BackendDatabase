"""
Tests für den NougatAdapter unter Verwendung der Basis-Testklasse.
"""

import numpy as np
import torch
from unittest.mock import patch, MagicMock
from PIL import Image

from models_app.ocr.nougat_adapter import NougatAdapter
from models_app.tests.ocr.test_base_adapter import BaseOCRAdapterTest

class TestNougatAdapter(BaseOCRAdapterTest):
    """Test-Klasse für den NougatAdapter."""
    
    ADAPTER_CLASS = NougatAdapter
    CONFIG_DICT = {
        'model_name': 'facebook/nougat-base',
        'max_length': 4096,
        'device': 'cpu',
        'return_markdown': True,
        'offload_folder': None
    }
    MOCK_IMPORTS = [
        'models_app.ocr.nougat_adapter.NougatModel',
        'models_app.ocr.nougat_adapter.get_checkpoint',
        'models_app.ocr.nougat_adapter.get_device'
    ]
    
    @patch('models_app.ocr.nougat_adapter.NougatModel')
    @patch('models_app.ocr.nougat_adapter.get_checkpoint')
    @patch('models_app.ocr.nougat_adapter.get_device')
    def test_initialization(self, mock_get_device, mock_get_checkpoint, mock_nougat_model):
        """Test der Initialisierung des Adapters."""
        # Mocks einrichten
        mock_get_device.return_value = 'cpu'
        mock_get_checkpoint.return_value = '/path/to/checkpoint'
        
        mock_model_instance = MagicMock()
        mock_nougat_model.from_pretrained.return_value = mock_model_instance
        
        # Adapter initialisieren
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        mock_get_device.assert_called_once()
        mock_get_checkpoint.assert_called_once()
        mock_nougat_model.from_pretrained.assert_called_once()
        mock_model_instance.to.assert_called_once_with('cpu')
    
    @patch('models_app.ocr.nougat_adapter.NOUGAT_AVAILABLE', False)
    def test_initialize_nougat_not_available(self):
        """Test der Initialisierung, wenn Nougat nicht verfügbar ist."""
        result = self.adapter.initialize()
        
        self.assertFalse(result)
        self.assertFalse(self.adapter.is_initialized)
        self.assertIn("error", result)
        self.assertIn("not available", result["metadata"]["error"])
    
    def test_dummy_model(self):
        """Test der Initialisierung eines Dummy-Modells."""
        # Konfiguration für Dummy-Modus setzen
        self.adapter.config['dummy_mode'] = True
        
        # Adapter initialisieren
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        self.assertIsNotNone(self.adapter.processor)
        self.assertIsNotNone(self.adapter.model)
    
    @patch('models_app.ocr.nougat_adapter.NougatModel')
    @patch('models_app.ocr.nougat_adapter.get_checkpoint')
    @patch('models_app.ocr.nougat_adapter.get_device')
    def test_process_image_with_file_path(self, mock_get_device, mock_get_checkpoint, mock_nougat_model):
        """Test der Bildverarbeitung mit einem Dateipfad."""
        # Mocks einrichten
        self._setup_nougat_mocks(mock_get_device, mock_get_checkpoint, mock_nougat_model, "This is a test")
        
        # Adapter initialisieren
        self.adapter.initialize()
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "This is a test")
        self.assertIn("blocks", result)
        self.assertEqual(len(result["blocks"]), 1)
        self.assertIn("confidence", result)
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["model"], "NougatAdapter")
    
    @patch('models_app.ocr.nougat_adapter.NougatModel')
    @patch('models_app.ocr.nougat_adapter.get_checkpoint')
    @patch('models_app.ocr.nougat_adapter.get_device')
    def test_process_image_with_options(self, mock_get_device, mock_get_checkpoint, mock_nougat_model):
        """Test der Bildverarbeitung mit zusätzlichen Optionen."""
        # Mocks einrichten
        self._setup_nougat_mocks(mock_get_device, mock_get_checkpoint, mock_nougat_model, "Scientific paper")
        
        # Adapter initialisieren
        self.adapter.initialize()
        
        # Optionen für die Verarbeitung
        options = {
            'max_length': 2048,
            'return_markdown': False,
            'preprocess': True
        }
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_image_path, options)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "Scientific paper")
        self.assertIn("metadata", result)
        self.assertEqual(result["metadata"]["max_length"], 2048)
        self.assertEqual(result["metadata"]["return_markdown"], False)
    
    def _setup_nougat_mocks(self, mock_get_device, mock_get_checkpoint, mock_nougat_model, output_text):
        """Hilfsmethode zum Einrichten der Nougat-Mocks."""
        mock_get_device.return_value = 'cpu'
        mock_get_checkpoint.return_value = '/path/to/checkpoint'
        
        # Mock für das Modell
        mock_model_instance = MagicMock()
        mock_nougat_model.from_pretrained.return_value = mock_model_instance
        
        # Generation simulieren
        mock_generated_ids = torch.ones((1, 10), dtype=torch.long)
        mock_model_instance.generate.return_value = mock_generated_ids
        
        # Mock für den Prozessor
        self.adapter.processor = MagicMock()
        
        # Mock für die Bildverarbeitung
        self.adapter.processor.return_value = MagicMock(pixel_values=torch.zeros((1, 3, 224, 224)))
        
        # Mock für die Decoder-Ausgabe
        self.adapter.processor.batch_decode.return_value = [output_text]
        
        # Modell setzen
        self.adapter.model = mock_model_instance 