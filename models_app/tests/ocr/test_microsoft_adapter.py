"""
Tests für den MicrosoftReadAdapter unter Verwendung der Basis-Testklasse.
"""

import numpy as np
from unittest.mock import patch, MagicMock
import io
import unittest

from models_app.ocr.microsoft_adapter import MicrosoftReadAdapter
from models_app.tests.ocr.test_base_adapter import BaseOCRAdapterTest

class TestMicrosoftReadAdapter(BaseOCRAdapterTest):
    """Test-Klasse für den MicrosoftReadAdapter."""
    
    ADAPTER_CLASS = MicrosoftReadAdapter
    CONFIG_DICT = {
        'api_key': 'test_api_key',
        'endpoint': 'https://test-endpoint.cognitiveservices.azure.com/',
        'language': 'en',
        'model_version': 'latest'
    }
    MOCK_IMPORTS = [
        'models_app.ocr.microsoft_adapter.ComputerVisionClient',
        'models_app.ocr.microsoft_adapter.CognitiveServicesCredentials'
    ]
    
    @patch('models_app.ocr.microsoft_adapter.ComputerVisionClient')
    @patch('models_app.ocr.microsoft_adapter.CognitiveServicesCredentials')
    def test_initialization(self, mock_credentials, mock_client):
        """Test der Initialisierung des Adapters."""
        # Mocks einrichten
        mock_credentials_instance = MagicMock()
        mock_credentials.return_value = mock_credentials_instance
        
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        mock_credentials.assert_called_once_with('test_api_key')
        mock_client.assert_called_once_with(
            'https://test-endpoint.cognitiveservices.azure.com/',
            mock_credentials_instance
        )
    
    @patch('models_app.ocr.microsoft_adapter.AZURE_AVAILABLE', False)
    def test_initialize_azure_not_available(self):
        """Test der Initialisierung, wenn Azure nicht verfügbar ist."""
        result = self.adapter.initialize()
        
        self.assertFalse(result)
        self.assertFalse(self.adapter.is_initialized)
        self.assertIn("error", result)
        self.assertIn("not available", result["metadata"]["error"])
    
    @patch('models_app.ocr.microsoft_adapter.ComputerVisionClient')
    @patch('models_app.ocr.microsoft_adapter.CognitiveServicesCredentials')
    def test_process_image_with_file_path(self, mock_credentials, mock_client):
        """Test der Bildverarbeitung mit einem Dateipfad."""
        # Mocks einrichten
        mock_credentials_instance = MagicMock()
        mock_credentials.return_value = mock_credentials_instance
        
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        # Mock für die Read-Operation
        mock_operation = MagicMock()
        mock_operation.operation_id = 'test_operation_id'
        mock_client_instance.read_in_stream.return_value = mock_operation
        
        # Mock für das Ergebnis der Read-Operation
        mock_result = MagicMock()
        mock_client_instance.get_read_result.return_value = mock_result
        mock_result.status = 'succeeded'
        
        # Komplexes Ergebnisobjekt erstellen
        mock_page1 = MagicMock()
        mock_page1.page_number = 1
        mock_page1.width = 400
        mock_page1.height = 200
        
        mock_line1 = MagicMock()
        mock_line1.content = 'Hello'
        mock_line1.bounding_box = [10, 10, 100, 10, 100, 50, 10, 50]
        
        mock_line2 = MagicMock()
        mock_line2.content = 'World'
        mock_line2.bounding_box = [10, 60, 100, 60, 100, 100, 10, 100]
        
        mock_page1.lines = [mock_line1, mock_line2]
        
        mock_result.analyze_result.read_results = [mock_page1]
        
        # Adapter initialisieren
        self.adapter.client = mock_client_instance
        self.adapter.is_initialized = True
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "Hello\nWorld")
        self.assertIn("blocks", result)
        self.assertEqual(len(result["blocks"]), 2)
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["confidence"], 0.9)  # Azure gibt keine Konfidenz pro Element, daher Standard
        self.assertEqual(result["model"], "MicrosoftReadAdapter")
    
    def test_get_supported_languages(self):
        """Test des Abrufs unterstützter Sprachen."""
        languages = self.adapter.get_supported_languages()
        
        self.assertIsInstance(languages, list)
        self.assertTrue(len(languages) > 20)  # Azure Read API unterstützt viele Sprachen
        self.assertIn('en', languages)
        self.assertIn('de', languages)
        self.assertIn('fr', languages)

if __name__ == '__main__':
    unittest.main() 