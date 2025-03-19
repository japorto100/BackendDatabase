"""
Tests für den TesseractAdapter unter Verwendung der Basis-Testklasse.
"""

import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from models_app.ocr.tesseract_adapter import TesseractAdapter
from models_app.tests.ocr.test_base_adapter import BaseOCRAdapterTest

class TestTesseractAdapter(BaseOCRAdapterTest):
    """Test-Klasse für den TesseractAdapter."""
    
    ADAPTER_CLASS = TesseractAdapter
    CONFIG_DICT = {
        'lang': 'eng',
        'config': '',
        'path': None
    }
    MOCK_IMPORTS = ['models_app.ocr.tesseract_adapter.pytesseract']
    
    @patch('models_app.ocr.tesseract_adapter.pytesseract')
    def test_initialization(self, mock_pytesseract):
        """Test der Initialisierung des Adapters."""
        mock_pytesseract.get_tesseract_version.return_value = "4.1.1"
        
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        mock_pytesseract.get_tesseract_version.assert_called_once()
    
    @patch('models_app.ocr.tesseract_adapter.pytesseract')
    def test_process_image_with_file_path(self, mock_pytesseract):
        """Test der Bildverarbeitung mit einem Dateipfad."""
        # Mock the OCR results
        mock_pytesseract.image_to_string.return_value = "This is a test\nfor Tesseract OCR"
        mock_pytesseract.image_to_data.return_value = {
            'text': ['This', 'is', 'a', 'test', 'for', 'Tesseract', 'OCR'],
            'conf': [90, 85, 95, 92, 88, 86, 91],
            'left': [50, 80, 100, 120, 50, 80, 180],
            'top': [50, 50, 50, 50, 100, 100, 100],
            'width': [30, 20, 10, 40, 30, 100, 40],
            'height': [20, 20, 20, 20, 20, 20, 20],
            'line_num': [1, 1, 1, 1, 2, 2, 2],
            'block_num': [1, 1, 1, 1, 1, 1, 1],
            'page_num': [1, 1, 1, 1, 1, 1, 1]
        }
        
        self.adapter.initialize()
        result = self.adapter.process_image(self.test_image_path)
        
        self.assertIn("text", result)
        self.assertIn("blocks", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["model"], "tesseract")
        self.assertEqual(result["language"], "eng")
        self.assertEqual(result["text"], "This is a test\nfor Tesseract OCR")
        self.assertGreater(result["confidence"], 0.0)
    
    @patch('models_app.ocr.tesseract_adapter.pytesseract')
    def test_process_image_with_numpy_array(self, mock_pytesseract):
        """Test der Bildverarbeitung mit einem NumPy-Array."""
        # Mock the OCR results
        mock_pytesseract.image_to_string.return_value = "This is a test"
        mock_pytesseract.image_to_data.return_value = {
            'text': ['This', 'is', 'a', 'test'],
            'conf': [90, 85, 95, 92],
            'left': [50, 80, 100, 120],
            'top': [50, 50, 50, 50],
            'width': [30, 20, 10, 40],
            'height': [20, 20, 20, 20],
            'line_num': [1, 1, 1, 1],
            'block_num': [1, 1, 1, 1],
            'page_num': [1, 1, 1, 1]
        }
        
        # Konvertiere das Bild in ein NumPy-Array
        image_array = np.array(self.test_image)
        
        self.adapter.initialize()
        result = self.adapter.process_image(image_array)
        
        self.assertIn("text", result)
        self.assertEqual(result["text"], "This is a test")
        
    @patch('models_app.ocr.tesseract_adapter.cv2')
    def test_preprocess_image(self, mock_cv2):
        """Test der Bildvorverarbeitung."""
        # Mock OpenCV functions
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.fastNlMeansDenoising.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.adaptiveThreshold.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        result = self.adapter.preprocess_image(self.test_image_path)
        
        self.assertIsInstance(result, Image.Image)
        mock_cv2.imread.assert_called_once_with(self.test_image_path)
        mock_cv2.cvtColor.assert_called_once()
        mock_cv2.fastNlMeansDenoising.assert_called_once()
        mock_cv2.adaptiveThreshold.assert_called_once()