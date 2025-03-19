"""
Tests für den DonutAdapter unter Verwendung der Basis-Testklasse.
"""

import numpy as np
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
import json

from models_app.ocr.donut_adapter import DonutAdapter
from models_app.tests.ocr.test_base_adapter import BaseOCRAdapterTest

class TestDonutAdapter(BaseOCRAdapterTest):
    """Test-Klasse für den DonutAdapter."""
    
    ADAPTER_CLASS = DonutAdapter
    CONFIG_DICT = {
        'model_name': 'naver-clova-ix/donut-base-finetuned-cord-v2',
        'task_prompt': 'parse this document',
        'max_length': 1024,
        'output_format': 'json',
        'gpu': False
    }
    MOCK_IMPORTS = [
        'models_app.ocr.donut_adapter.torch.device',
        'models_app.ocr.donut_adapter.torch.cuda.is_available',
        'models_app.ocr.donut_adapter.DonutProcessor.from_pretrained',
        'models_app.ocr.donut_adapter.VisionEncoderDecoderModel.from_pretrained'
    ]
    
    @patch('models_app.ocr.donut_adapter.DONUT_AVAILABLE', False)
    def test_initialize_donut_not_available(self):
        """Test der Initialisierung, wenn Donut nicht verfügbar ist."""
        result = self.adapter.initialize()
        
        self.assertFalse(result)
        self.assertFalse(self.adapter.is_initialized)
        self.assertIn("error", result)
        self.assertIn("not available", result["metadata"]["error"])
    
    @patch('models_app.ocr.donut_adapter.DONUT_AVAILABLE', True)
    @patch('models_app.ocr.donut_adapter.torch.device')
    @patch('models_app.ocr.donut_adapter.torch.cuda.is_available')
    @patch('models_app.ocr.donut_adapter.DonutProcessor.from_pretrained')
    @patch('models_app.ocr.donut_adapter.VisionEncoderDecoderModel.from_pretrained')
    def test_initialize(self, mock_model, mock_processor, mock_cuda, mock_device):
        """Test der Initialisierung des Adapters."""
        # Mocks einrichten
        mock_cuda.return_value = False
        mock_device.return_value = torch.device('cpu')
        
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        mock_processor.assert_called_once()
        mock_model.assert_called_once()
        mock_model_instance.to.assert_called_once()
    
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
    
    @patch('models_app.ocr.donut_adapter.DONUT_AVAILABLE', True)
    @patch('models_app.ocr.donut_adapter.torch')
    def test_process_image_with_file_path(self, mock_torch):
        """Test der Bildverarbeitung mit einem Dateipfad."""
        # Mock für die Bildverarbeitung
        mock_tensor = MagicMock()
        mock_torch.zeros.return_value = mock_tensor
        
        # Mock für die Bildverarbeitung mit dem Processor
        self.adapter.processor = MagicMock()
        self.adapter.processor.feature_extractor.return_value = {"pixel_values": mock_tensor}
        
        # Mock für das Tokenizer
        self.adapter.processor.tokenizer.return_value = {"input_ids": mock_tensor}
        
        # Mock für das Modell
        self.adapter.model = MagicMock()
        mock_generated_ids = MagicMock()
        self.adapter.model.generate.return_value = mock_generated_ids
        
        # Mock für den Decoder
        json_output = '{"invoice_number": "INV-2023-001", "date": "2023-04-15", "total": "$123.45"}'
        self.adapter.processor.decode.return_value = f'<s_cord-v2>{json_output}</s_cord-v2>'
        
        # Adapter initialisieren
        self.adapter.is_initialized = True
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertIn("invoice_number", result["text"])
        self.assertIn("date", result["text"])
        self.assertIn("total", result["text"])
        
        self.assertIn("structured_data", result)
        self.assertEqual(result["structured_data"]["invoice_number"], "INV-2023-001")
        self.assertEqual(result["structured_data"]["date"], "2023-04-15")
        self.assertEqual(result["structured_data"]["total"], "$123.45")
        
        self.assertEqual(result["model"], "DonutAdapter")
    
    @patch('models_app.ocr.donut_adapter.DONUT_AVAILABLE', True)
    @patch('models_app.ocr.donut_adapter.torch')
    def test_process_image_with_json_output(self, mock_torch):
        """Test der Bildverarbeitung mit JSON-Ausgabe."""
        # Mock für die Bildverarbeitung
        mock_tensor = MagicMock()
        mock_torch.zeros.return_value = mock_tensor
        
        # Mock für die Bildverarbeitung mit dem Processor
        self.adapter.processor = MagicMock()
        self.adapter.processor.feature_extractor.return_value = {"pixel_values": mock_tensor}
        
        # Mock für das Tokenizer
        self.adapter.processor.tokenizer.return_value = {"input_ids": mock_tensor}
        
        # Mock für das Modell
        self.adapter.model = MagicMock()
        mock_generated_ids = MagicMock()
        self.adapter.model.generate.return_value = mock_generated_ids
        
        # Mock für den Decoder
        json_output = '{"invoice_number": "INV-2023-001", "date": "2023-04-15", "total": "$123.45"}'
        self.adapter.processor.decode.return_value = f'<s_cord-v2>{json_output}</s_cord-v2>'
        
        # Adapter initialisieren
        self.adapter.is_initialized = True
        
        # Bild mit JSON-Ausgabeformat verarbeiten
        options = {'output_format': 'json'}
        result = self.adapter.process_image(self.test_image_path, options)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], json_output)
        
        self.assertIn("structured_data", result)
        self.assertEqual(result["structured_data"]["invoice_number"], "INV-2023-001")
        
        self.assertEqual(result["model"], "DonutAdapter")
    
    @patch('models_app.ocr.donut_adapter.DONUT_AVAILABLE', True)
    @patch('models_app.ocr.donut_adapter.torch')
    def test_process_image_with_raw_output(self, mock_torch):
        """Test der Bildverarbeitung mit Rohausgabe."""
        # Mock für die Bildverarbeitung
        mock_tensor = MagicMock()
        mock_torch.zeros.return_value = mock_tensor
        
        # Mock für die Bildverarbeitung mit dem Processor
        self.adapter.processor = MagicMock()
        self.adapter.processor.feature_extractor.return_value = {"pixel_values": mock_tensor}
        
        # Mock für das Tokenizer
        self.adapter.processor.tokenizer.return_value = {"input_ids": mock_tensor}
        
        # Mock für das Modell
        self.adapter.model = MagicMock()
        mock_generated_ids = MagicMock()
        self.adapter.model.generate.return_value = mock_generated_ids
        
        # Mock für den Decoder
        raw_output = '<s_cord-v2>{"key": "value"}</s_cord-v2>'
        self.adapter.processor.decode.return_value = raw_output
        
        # Adapter initialisieren
        self.adapter.is_initialized = True
        
        # Bild mit Rohausgabe verarbeiten
        options = {'output_format': 'raw'}
        result = self.adapter.process_image(self.test_image_path, options)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], raw_output)
        
        self.assertEqual(result["model"], "DonutAdapter")
    
    def test_extract_json(self):
        """Test für die JSON-Extraktion aus Donut-Ausgabe."""
        # Verschiedene Eingabeformate
        input_texts = [
            '<s_cord-v2>{"key": "value"}</s_cord-v2>',
            '{"key": "value"}',
            'Keine JSON-Daten hier'
        ]
        
        # Erwartete Ergebnisse
        expected_results = [
            '{"key": "value"}',
            '{"key": "value"}',
            None
        ]
        
        for input_text, expected in zip(input_texts, expected_results):
            result = self.adapter._extract_json(input_text)
            self.assertEqual(result, expected)
    
    def test_json_to_text(self):
        """Test für die Konvertierung von JSON in Text."""
        # JSON-Daten
        json_str = '{"invoice": {"number": "INV001", "date": "2023-04-15"}, "items": [{"name": "Item 1", "price": 10}, {"name": "Item 2", "price": 20}]}'
        json_data = json.loads(json_str)
        
        result = self.adapter._json_to_text(json_data)
        
        self.assertIsInstance(result, str)
        self.assertIn("INV001", result)
        self.assertIn("2023-04-15", result)
        self.assertIn("Item 1", result)
        self.assertIn("Item 2", result)
    
    def test_format_json_as_text(self):
        """Test für die Formatierung von JSON als Text."""
        # JSON-Daten
        json_str = '{"invoice": {"number": "INV001", "date": "2023-04-15"}, "items": [{"name": "Item 1", "price": 10}, {"name": "Item 2", "price": 20}]}'
        
        # Deep format
        result_deep = self.adapter._format_json_as_text(json_str, format_type='deep')
        self.assertIn("invoice:", result_deep)
        self.assertIn("number: INV001", result_deep)
        
        # Flat format
        result_flat = self.adapter._format_json_as_text(json_str, format_type='flat')
        self.assertIn("invoice.number: INV001", result_flat)
        
        # Simple format
        result_simple = self.adapter._format_json_as_text(json_str, format_type='simple')
        self.assertIn("INV001", result_simple)
    
    def test_get_supported_languages(self):
        """Test für das Abrufen unterstützter Sprachen."""
        languages = self.adapter.get_supported_languages()
        
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)
    
    def test_get_model_info(self):
        """Test für das Abrufen von Modellinformationen."""
        # Adapter initialisieren
        self.adapter.is_initialized = True
        self.adapter.model_name = 'naver-clova-ix/donut-base-finetuned-cord-v2'
        
        info = self.adapter.get_model_info()
        
        self.assertEqual(info["name"], "DonutAdapter")
        self.assertEqual(info["type"], "Donut")
        self.assertIn("document_understanding", info["capabilities"])
        self.assertTrue(info["capabilities"]["document_understanding"]) 