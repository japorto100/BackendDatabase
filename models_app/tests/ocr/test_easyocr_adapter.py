"""
Tests für den EasyOCRAdapter unter Verwendung der Basis-Testklasse.
"""

import numpy as np
from unittest.mock import patch, MagicMock

from models_app.ocr.easyocr_adapter import EasyOCRAdapter
from models_app.tests.ocr.test_base_adapter import BaseOCRAdapterTest

class TestEasyOCRAdapter(BaseOCRAdapterTest):
    """Test-Klasse für den EasyOCRAdapter."""
    
    ADAPTER_CLASS = EasyOCRAdapter
    CONFIG_DICT = {
        'lang': ["en"],
        'gpu': False,
        'verbose': False,
        'preprocess': True
    }
    MOCK_IMPORTS = ['models_app.ocr.easyocr_adapter.easyocr']
    
    @patch('models_app.ocr.easyocr_adapter.easyocr')
    def test_initialization(self, mock_easyocr):
        """Test der Initialisierung des Adapters."""
        # Mock für EasyOCR-Reader
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader
        
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        mock_easyocr.Reader.assert_called_once_with(
            self.CONFIG_DICT['lang'],
            gpu=self.CONFIG_DICT['gpu'],
            verbose=self.CONFIG_DICT['verbose']
        )
    
    @patch('models_app.ocr.easyocr_adapter.easyocr')
    def test_process_image_with_file_path(self, mock_easyocr):
        """Test der Bildverarbeitung mit einem Dateipfad."""
        # Mock für EasyOCR-Reader
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader
        
        # Mock für die OCR-Ergebnisse
        mock_reader.readtext.return_value = [
            ([[10, 10], [100, 10], [100, 50], [10, 50]], "Hello", 0.95),
            ([[10, 60], [100, 60], [100, 100], [10, 100]], "World", 0.9)
        ]
        
        # Adapter initialisieren
        self.adapter.reader = mock_reader
        self.adapter.is_initialized = True
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "Hello\nWorld")
        self.assertIn("blocks", result)
        self.assertEqual(len(result["blocks"]), 2)
        self.assertIn("confidence", result)
        self.assertGreater(result["confidence"], 0.9)
        self.assertEqual(result["model"], "EasyOCRAdapter")
    
    @patch('models_app.ocr.easyocr_adapter.easyocr')
    def test_process_image_with_numpy_array(self, mock_easyocr):
        """Test der Bildverarbeitung mit einem NumPy-Array."""
        # Mock für EasyOCR-Reader
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader
        
        # Mock für die OCR-Ergebnisse
        mock_reader.readtext.return_value = [
            ([[10, 10], [100, 10], [100, 50], [10, 50]], "Test", 0.95)
        ]
        
        # Adapter initialisieren
        self.adapter.reader = mock_reader
        self.adapter.is_initialized = True
        
        # Bild in NumPy-Array konvertieren
        image_array = np.array(self.test_image)
        
        # Bild verarbeiten
        result = self.adapter.process_image(image_array)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "Test")
        self.assertIn("blocks", result)
        self.assertEqual(len(result["blocks"]), 1)
        self.assertIn("confidence", result)
        self.assertEqual(result["confidence"], 0.95)
    
    @patch('models_app.ocr.easyocr_adapter.easyocr')
    def test_process_image_with_options(self, mock_easyocr):
        """Test der Bildverarbeitung mit zusätzlichen Optionen."""
        # Mock für EasyOCR-Reader
        mock_reader = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader
        
        # Mock für die OCR-Ergebnisse
        mock_reader.readtext.return_value = [
            ([[10, 10], [100, 10], [100, 50], [10, 50]], "Test", 0.95)
        ]
        
        # Adapter initialisieren
        self.adapter.reader = mock_reader
        self.adapter.is_initialized = True
        
        # Optionen für die Verarbeitung
        options = {
            'detail': 1,
            'paragraph': True,
            'batch_size': 5,
            'lang': ['de']
        }
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_image_path, options)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "Test")
        mock_reader.readtext.assert_called_once_with(
            self.test_image_path,
            detail=1,
            paragraph=True,
            batch_size=5
        )
    
    @patch('models_app.ocr.easyocr_adapter.cv2')
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