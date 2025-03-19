"""
Tests für den PaddleOCRAdapter unter Verwendung der Basis-Testklasse.
"""

import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image, ImageDraw, ImageFont

from models_app.ocr.paddle_adapter import PaddleOCRAdapter
from models_app.tests.ocr.test_base_adapter import BaseOCRAdapterTest

class TestPaddleOCRAdapter(BaseOCRAdapterTest):
    """Test-Klasse für den PaddleOCRAdapter."""
    
    ADAPTER_CLASS = PaddleOCRAdapter
    CONFIG_DICT = {
        'lang': 'en',
        'use_gpu': False,
        'enable_mkldnn': True,
        'preprocess': True,
        'handwriting_mode': False
    }
    MOCK_IMPORTS = ['models_app.ocr.paddle_adapter.PaddleOCR']
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'use_gpu': False,
            'lang': 'en'
        }
        self.adapter = PaddleOCRAdapter(config=self.config)
        
        # Create test images
        self.test_image_path = os.path.join(os.path.dirname(__file__), 'test_printed.png')
        self.test_handwriting_path = os.path.join(os.path.dirname(__file__), 'test_handwriting.png')
        self._create_test_images()
    
    def _create_test_images(self):
        """Create test images for printed text and handwriting."""
        # Create a test image with printed text
        img_printed = Image.new('RGB', (400, 200), color=(255, 255, 255))
        d_printed = ImageDraw.Draw(img_printed)
        
        try:
            # Try to use a standard font
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            # Fallback to default
            font = ImageFont.load_default()
        
        # Draw text with regular spacing and alignment
        d_printed.text((50, 50), "This is a test", fill=(0, 0, 0), font=font)
        d_printed.text((50, 100), "for OCR processing", fill=(0, 0, 0), font=font)
        
        img_printed.save(self.test_image_path)
        
        # Create a test image with simulated handwriting
        img_handwriting = Image.new('RGB', (400, 200), color=(255, 255, 255))
        d_handwriting = ImageDraw.Draw(img_handwriting)
        
        # Draw text with slight variations to simulate handwriting
        for i, char in enumerate("This is handwritten"):
            # Vary the y-position slightly for each character
            y_offset = 50 + np.random.randint(-5, 5)
            # Vary the angle slightly
            d_handwriting.text((50 + i*15, y_offset), char, fill=(0, 0, 0), font=font)
        
        for i, char in enumerate("text for testing"):
            y_offset = 100 + np.random.randint(-5, 5)
            d_handwriting.text((50 + i*15, y_offset), char, fill=(0, 0, 0), font=font)
        
        # Add some noise to simulate real handwriting
        pixels = img_handwriting.load()
        for i in range(img_handwriting.size[0]):
            for j in range(img_handwriting.size[1]):
                if np.random.random() > 0.99:  # Add random noise
                    pixels[i, j] = (int(pixels[i, j][0] * 0.8), 
                                   int(pixels[i, j][1] * 0.8), 
                                   int(pixels[i, j][2] * 0.8))
        
        img_handwriting.save(self.test_handwriting_path)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        if os.path.exists(self.test_handwriting_path):
            os.remove(self.test_handwriting_path)
    
    @patch('models_app.ocr.paddle_adapter.PaddleOCR')
    def test_initialization(self, mock_paddle_ocr):
        """Test der Initialisierung des Adapters."""
        # Mock für PaddleOCR
        mock_instance = MagicMock()
        mock_paddle_ocr.return_value = mock_instance
        
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        mock_paddle_ocr.assert_called_once()
    
    @patch('models_app.ocr.paddle_adapter.PaddleOCR')
    def test_process_image_with_file_path(self, mock_paddle_ocr):
        """Test der Bildverarbeitung mit einem Dateipfad."""
        # Mock für PaddleOCR
        mock_instance = MagicMock()
        mock_paddle_ocr.return_value = mock_instance
        
        # Mock für die OCR-Ergebnisse
        mock_instance.return_value = [
            [[[10, 10], [100, 10], [100, 50], [10, 50]], ('Hello', 0.95)],
            [[[10, 60], [100, 60], [100, 100], [10, 100]], ('World', 0.9)]
        ]
        
        # Adapter initialisieren
        self.adapter.model = mock_instance
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
        self.assertEqual(result["model"], "PaddleOCRAdapter")
    
    @patch('models_app.ocr.paddle_adapter.PaddleOCR')
    def test_process_image_with_numpy_array(self, mock_paddle_ocr):
        """Test der Bildverarbeitung mit einem NumPy-Array."""
        # Mock für PaddleOCR
        mock_instance = MagicMock()
        mock_paddle_ocr.return_value = mock_instance
        
        # Mock für die OCR-Ergebnisse
        mock_instance.return_value = [
            [[[10, 10], [100, 10], [100, 50], [10, 50]], ('Test', 0.95)]
        ]
        
        # Adapter initialisieren
        self.adapter.model = mock_instance
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
    
    @patch('models_app.ocr.paddle_adapter.PaddleOCR')
    def test_handwriting_mode(self, mock_paddle_ocr):
        """Test der Handschrifterkennung."""
        # Mock für PaddleOCR
        mock_instance = MagicMock()
        mock_paddle_ocr.return_value = mock_instance
        
        # Mock für die OCR-Ergebnisse
        mock_instance.return_value = [
            [[[10, 10], [100, 10], [100, 50], [10, 50]], ('Handwritten', 0.85)]
        ]
        
        # Adapter mit Handschrift-Modus konfigurieren
        self.adapter.config['handwriting_mode'] = True
        
        # Adapter initialisieren
        result = self.adapter.initialize()
        self.assertTrue(result)
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "Handwritten")
        self.assertEqual(result["model"], "PaddleOCRAdapter")
        self.assertEqual(result["metadata"]["mode"], "handwriting")

    def test_preprocess_image(self):
        """Test der Bildvorverarbeitung."""
        # Bild vorverarbeiten
        result = self.adapter.preprocess_image(self.test_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIsNotNone(result)
        self.assertTrue(isinstance(result, (np.ndarray, Image.Image)))
    
    def test_preprocess_handwriting(self):
        """Test handwriting-specific preprocessing."""
        # Create adapter with handwriting mode
        handwriting_config = {
            'use_gpu': False,
            'lang': 'en',
            'handwriting_mode': True
        }
        handwriting_adapter = PaddleOCRAdapter(config=handwriting_config)
        
        # Create a simple test image
        image = Image.new('RGB', (100, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "Test", fill=(0, 0, 0))
        
        processed = handwriting_adapter._preprocess_handwriting(image)
        
        # Check that the processed image has the right shape and type
        self.assertEqual(processed.shape, (100, 100, 3))
        self.assertEqual(processed.dtype, np.uint8)
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.adapter.get_supported_languages()
        self.assertIn("en", languages)
        self.assertIn("ch", languages)
        self.assertIn("german", languages)
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = self.adapter.get_model_info()
        self.assertEqual(info["name"], "PaddleOCRAdapter")
        self.assertEqual(info["type"], "PaddleOCR")
        self.assertIn("capabilities", info)
        self.assertIn("multi_language", info["capabilities"])
        
        # Test with handwriting mode
        self.adapter.is_handwriting_mode = True
        info = self.adapter.get_model_info()
        self.assertIn("handwriting", info["capabilities"])
        self.assertTrue(info["handwriting_mode"])

if __name__ == '__main__':
    unittest.main()
