"""
Tests für den DocTRAdapter unter Verwendung der Basis-Testklasse.
"""

import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from models_app.ocr.doctr_adapter import DocTRAdapter
from models_app.tests.ocr.test_base_adapter import BaseOCRAdapterTest

class TestDocTRAdapter(BaseOCRAdapterTest):
    """Test-Klasse für den DocTRAdapter."""
    
    ADAPTER_CLASS = DocTRAdapter
    CONFIG_DICT = {
        'det_arch': 'db_resnet50',
        'reco_arch': 'crnn_vgg16_bn',
        'pretrained': True,
        'assume_straight_pages': True,
        'straighten_pages': True,
        'gpu': False
    }
    MOCK_IMPORTS = [
        'models_app.ocr.doctr_adapter.ocr_predictor',
        'models_app.ocr.doctr_adapter.DocumentFile'
    ]
    
    @patch('models_app.ocr.doctr_adapter.DOCTR_AVAILABLE', True)
    @patch('models_app.ocr.doctr_adapter.ocr_predictor')
    def test_initialization(self, mock_ocr_predictor):
        """Test der Initialisierung des Adapters."""
        mock_instance = MagicMock()
        mock_ocr_predictor.return_value = mock_instance
        
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        self.assertEqual(self.adapter.detector, self.CONFIG_DICT['det_arch'])
        self.assertEqual(self.adapter.recognizer, self.CONFIG_DICT['reco_arch'])
        mock_ocr_predictor.assert_called_once()
    
    @patch('models_app.ocr.doctr_adapter.DOCTR_AVAILABLE', False)
    def test_initialize_doctr_not_available(self):
        """Test der Initialisierung, wenn DocTR nicht verfügbar ist."""
        result = self.adapter.initialize()
        
        self.assertFalse(result)
        self.assertFalse(self.adapter.is_initialized)
        self.assertIn("error", result)
        self.assertIn("not available", result["metadata"]["error"])
    
    @patch('models_app.ocr.doctr_adapter.DOCTR_AVAILABLE', True)
    @patch('models_app.ocr.doctr_adapter.ocr_predictor')
    @patch('models_app.ocr.doctr_adapter.DocumentFile')
    def test_process_image_with_file_path(self, mock_document_file, mock_ocr_predictor):
        """Test der Bildverarbeitung mit einem Dateipfad."""
        # Mock für DocumentFile
        mock_doc_instance = MagicMock()
        mock_document_file.return_value = mock_doc_instance
        mock_doc_instance.__iter__.return_value = [np.zeros((100, 100, 3), dtype=np.uint8)]
        
        # Mock für den Predictor
        mock_predictor = MagicMock()
        mock_ocr_predictor.return_value = mock_predictor
        
        # Mock für die Vorhersageergebnisse
        mock_doc_result = MagicMock()
        mock_predictor.return_value = mock_doc_result
        
        # Komplexe Dokumentstruktur für die Rückgabe erstellen
        mock_pages = []
        page = MagicMock()
        block = MagicMock()
        line = MagicMock()
        
        # Wort mit Koordinaten, Text und Konfidenz
        word1 = MagicMock()
        word1.value = "Hello"
        word1.confidence = 0.95
        word1.geometry = ((0.1, 0.1), (0.3, 0.2))
        
        word2 = MagicMock()
        word2.value = "World"
        word2.confidence = 0.9
        word2.geometry = ((0.35, 0.1), (0.6, 0.2))
        
        # Struktur aufbauen
        line.words = [word1, word2]
        block.lines = [line]
        page.blocks = [block]
        mock_pages.append(page)
        
        mock_doc_result.pages = mock_pages
        mock_doc_result.export.return_value = {
            'pages': [
                {
                    'blocks': [
                        {
                            'lines': [
                                {
                                    'words': [
                                        {'value': 'Hello', 'confidence': 0.95, 'geometry': [[0.1, 0.1], [0.3, 0.2]]},
                                        {'value': 'World', 'confidence': 0.9, 'geometry': [[0.35, 0.1], [0.6, 0.2]]}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        # Adapter initialisieren und Bild verarbeiten
        self.adapter.model = mock_predictor
        self.adapter.is_initialized = True
        result = self.adapter.process_image(self.test_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "Hello World")
        self.assertIn("blocks", result)
        self.assertGreater(len(result["blocks"]), 0)
        self.assertIn("confidence", result)
        self.assertGreater(result["confidence"], 0.9)
        self.assertEqual(result["model"], "DocTRAdapter")
    
    @patch('models_app.ocr.doctr_adapter.DOCTR_AVAILABLE', True)
    @patch('models_app.ocr.doctr_adapter.ocr_predictor')
    @patch('models_app.ocr.doctr_adapter.DocumentFile')
    def test_process_image_with_numpy_array(self, mock_document_file, mock_ocr_predictor):
        """Test der Bildverarbeitung mit einem NumPy-Array."""
        # Mock für den Predictor
        mock_predictor = MagicMock()
        mock_ocr_predictor.return_value = mock_predictor
        
        # Mock für die Vorhersageergebnisse
        mock_doc_result = MagicMock()
        mock_predictor.return_value = mock_doc_result
        
        # Komplexe Dokumentstruktur für die Rückgabe erstellen
        mock_pages = []
        page = MagicMock()
        block = MagicMock()
        line = MagicMock()
        
        # Wort mit Koordinaten, Text und Konfidenz
        word = MagicMock()
        word.value = "Test"
        word.confidence = 0.95
        word.geometry = ((0.1, 0.1), (0.3, 0.2))
        
        # Struktur aufbauen
        line.words = [word]
        block.lines = [line]
        page.blocks = [block]
        mock_pages.append(page)
        
        mock_doc_result.pages = mock_pages
        mock_doc_result.export.return_value = {
            'pages': [
                {
                    'blocks': [
                        {
                            'lines': [
                                {
                                    'words': [
                                        {'value': 'Test', 'confidence': 0.95, 'geometry': [[0.1, 0.1], [0.3, 0.2]]}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        # Bild in NumPy-Array konvertieren
        image_array = np.array(self.test_image)
        
        # Adapter initialisieren und Bild verarbeiten
        self.adapter.model = mock_predictor
        self.adapter.is_initialized = True
        result = self.adapter.process_image(image_array)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "Test")
        self.assertIn("blocks", result)
        self.assertEqual(len(result["blocks"]), 1)
        self.assertIn("confidence", result)
        self.assertEqual(result["confidence"], 0.95)
    
    @patch('models_app.ocr.doctr_adapter.doctr')
    def test_get_model_info(self, mock_doctr):
        """Test für das Abrufen von Modellinformationen."""
        mock_doctr.__version__ = "0.6.0"
        
        # Adapter initialisieren
        self.adapter.is_initialized = True
        info = self.adapter.get_model_info()
        
        # Überprüfen der Informationen
        self.assertEqual(info["name"], "DocTRAdapter")
        self.assertEqual(info["type"], "DocTR")
        self.assertEqual(info["version"], "0.6.0")
        self.assertTrue(info["capabilities"]["multi_language"])
        self.assertTrue(info["capabilities"]["table_extraction"])
    
    @patch('models_app.ocr.doctr_adapter.cv2')
    def test_preprocess_image(self, mock_cv2):
        """Test der Bildvorverarbeitung."""
        # Mock für OpenCV-Funktionen
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.GaussianBlur.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.adaptiveThreshold.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        # Bild vorverarbeiten
        result = self.adapter.preprocess_image(self.test_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIsInstance(result, np.ndarray)
        mock_cv2.imread.assert_called_once()
        mock_cv2.cvtColor.assert_called_once()

if __name__ == '__main__':
    unittest.main() 