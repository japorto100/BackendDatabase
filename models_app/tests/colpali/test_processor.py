"""
Tests für ColPali Processor
"""

import unittest
import os
import tempfile
from PIL import Image
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from models_app.colpali.processor import ColPaliProcessor

class TestColPaliProcessor(unittest.TestCase):
    """Test-Klasse für ColPaliProcessor"""

    def setUp(self):
        """Setup für Tests"""
        # Mock für die Modell-Initialisierung, damit wir keine echten Modelle laden müssen
        self.patcher = patch('models_app.colpali.processor.AutoModel')
        self.mock_auto_model = self.patcher.start()
        self.mock_model = MagicMock()
        self.mock_auto_model.from_pretrained.return_value = self.mock_model
        
        # Mock für den Image Processor
        self.image_patcher = patch('models_app.colpali.processor.AutoImageProcessor')
        self.mock_image_processor_class = self.image_patcher.start()
        self.mock_image_processor = MagicMock()
        self.mock_image_processor_class.from_pretrained.return_value = self.mock_image_processor
        
        # Erstelle eine Test-Instanz
        self.processor = ColPaliProcessor()
        
        # Erstelle ein Test-Bild
        self.test_image = self._create_test_image()
    
    def tearDown(self):
        """Cleanup nach Tests"""
        self.patcher.stop()
        self.image_patcher.stop()
        
        # Lösche Testdateien
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def _create_test_image(self):
        """Erstellt ein Testbild für die Tests"""
        # Erstelle ein einfaches 100x100 RGB-Bild
        img = Image.new('RGB', (100, 100), color='white')
        
        # Speichere das Bild temporär für Tests mit Dateipfaden
        fd, self.temp_file = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        img.save(self.temp_file)
        
        return img
    
    @patch('torch.Tensor')
    def test_process_image(self, mock_tensor):
        """Test für process_image Methode"""
        # Konfiguriere den Mock für die Modellausgabe
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(1, 10, 768)
        self.mock_model.return_value = mock_output
        
        # Teste mit PIL-Bild
        result = self.processor.process_image(self.test_image)
        self.assertIn('features', result)
        self.assertIn('dimensions', result)
        
        # Teste mit Dateipfad
        result = self.processor.process_image(self.temp_file)
        self.assertIn('features', result)
        self.assertIn('dimensions', result)
    
    def test_extract_features(self):
        """Test für extract_features Methode"""
        # Konfiguriere Mocks
        self.mock_image_processor.return_value = {'pixel_values': torch.rand(1, 3, 224, 224)}
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(1, 10, 768)
        self.mock_model.return_value = mock_output
        
        # Teste die Methode
        features = self.processor._extract_features(self.test_image)
        self.assertIsNotNone(features)
        self.assertTrue(isinstance(features, dict))
        
    def test_compare_images(self):
        """Test für compare_images Methode"""
        # Erstelle Mock-Features
        mock_features1 = {'features': torch.rand(1, 10, 768)}
        mock_features2 = {'features': torch.rand(1, 10, 768)}
        
        # Mock für die Ähnlichkeitsberechnung
        with patch.object(self.processor, '_calculate_similarity') as mock_calc:
            mock_calc.return_value = 0.75
            
            # Teste die Methode
            similarity = self.processor.compare_images(mock_features1, mock_features2)
            self.assertIsInstance(similarity, float)
            self.assertTrue(0 <= similarity <= 1)
            mock_calc.assert_called_once()

if __name__ == '__main__':
    unittest.main()