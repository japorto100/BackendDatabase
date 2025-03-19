"""
Tests für die Bildverarbeitungsfunktionen aus dem image_processing Modul.
"""

import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from models_app.ocr.utils.image_processing import (
    load_image, denoise_image, binarize_image, deskew_image,
    enhance_image_for_ocr, detect_text_regions, crop_image
)

class TestImageProcessing(unittest.TestCase):
    """Test-Klasse für Bildverarbeitungsfunktionen"""
    
    def setUp(self):
        """Setup für Tests"""
        # Temporäres Testbild erstellen
        self.test_image = Image.new('RGB', (100, 100), color='white')
        self.test_image_path = os.path.join(os.path.dirname(__file__), 'test_image.png')
        self.test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Cleanup nach Tests"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def test_load_image(self):
        """Test für load_image Funktion"""
        # Test mit Dateipfad
        pil_img, np_img, path = load_image(self.test_image_path)
        self.assertIsInstance(pil_img, Image.Image)
        self.assertIsInstance(np_img, np.ndarray)
        self.assertEqual(path, self.test_image_path)
        
        # Test mit PIL Image
        pil_img2, np_img2, path2 = load_image(self.test_image)
        self.assertIs(pil_img2, self.test_image)
        self.assertIsNone(path2)
        
        # Test mit NumPy Array
        np_image = np.array(self.test_image)
        pil_img3, np_img3, path3 = load_image(np_image)
        self.assertIsInstance(pil_img3, Image.Image)
        self.assertIsNone(path3)
        
        # Test mit nicht existierender Datei
        with self.assertRaises(FileNotFoundError):
            load_image("non_existent_file.jpg")
    
    @patch('models_app.ocr.utils.image_processing.cv2')
    def test_denoise_image(self, mock_cv2):
        """Test für denoise_image Funktion"""
        # Mock-Setup für OpenCV
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.fastNlMeansDenoising.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        # Test mit einem Farbbild
        rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = denoise_image(rgb_image)
        
        mock_cv2.cvtColor.assert_called_once()
        mock_cv2.fastNlMeansDenoising.assert_called_once()
        self.assertIsInstance(result, np.ndarray)
        
        # Test mit einem Graustufenbild
        mock_cv2.reset_mock()
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        result = denoise_image(gray_image)
        
        mock_cv2.cvtColor.assert_not_called()
        mock_cv2.fastNlMeansDenoising.assert_called_once()
        self.assertIsInstance(result, np.ndarray)
    
    @patch('models_app.ocr.utils.image_processing.cv2')
    def test_binarize_image(self, mock_cv2):
        """Test für binarize_image Funktion"""
        # Mock-Setup für OpenCV
        mock_cv2.adaptiveThreshold.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.threshold.return_value = (None, np.zeros((100, 100), dtype=np.uint8))
        
        # Test mit adaptivem Thresholding
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        result = binarize_image(gray_image, adaptive=True)
        
        mock_cv2.adaptiveThreshold.assert_called_once()
        self.assertIsInstance(result, np.ndarray)
        
        # Test mit globalem Thresholding
        mock_cv2.reset_mock()
        result = binarize_image(gray_image, adaptive=False)
        
        mock_cv2.threshold.assert_called_once()
        self.assertIsInstance(result, np.ndarray)
    
    @patch('models_app.ocr.utils.image_processing.cv2')
    def test_deskew_image(self, mock_cv2):
        """Test für deskew_image Funktion"""
        # Mock-Setup für OpenCV
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.Canny.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.HoughLines.return_value = np.array([[[50, np.pi/3]]])
        mock_cv2.getRotationMatrix2D.return_value = np.zeros((2, 3), dtype=np.float32)
        mock_cv2.warpAffine.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        # Test mit einem Farbbild
        rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = deskew_image(rgb_image)
        
        mock_cv2.cvtColor.assert_called_once()
        mock_cv2.Canny.assert_called_once()
        mock_cv2.HoughLines.assert_called_once()
        mock_cv2.getRotationMatrix2D.assert_called_once()
        mock_cv2.warpAffine.assert_called_once()
        self.assertIsInstance(result, np.ndarray)
        
        # Test mit einem Graustufenbild
        mock_cv2.reset_mock()
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.HoughLines.return_value = None  # Keine Linien gefunden
        result = deskew_image(gray_image)
        
        mock_cv2.cvtColor.assert_not_called()
        mock_cv2.Canny.assert_called_once()
        mock_cv2.HoughLines.assert_called_once()
        # Wenn keine Linien gefunden werden, sollte das Originalbild zurückgegeben werden
        self.assertIs(result, gray_image)
    
    @patch('models_app.ocr.utils.image_processing.denoise_image')
    @patch('models_app.ocr.utils.image_processing.binarize_image')
    @patch('models_app.ocr.utils.image_processing.deskew_image')
    def test_enhance_image_for_ocr(self, mock_deskew, mock_binarize, mock_denoise):
        """Test für enhance_image_for_ocr Funktion"""
        # Mock-Rückgabewerte
        mock_denoise.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_binarize.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_deskew.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        # Test mit allen Optionen aktiviert
        pil_image, np_image = enhance_image_for_ocr(
            self.test_image_path, 
            denoise=True, 
            binarize=True, 
            deskew=True
        )
        
        # Überprüfen, ob alle Funktionen aufgerufen wurden
        mock_denoise.assert_called_once()
        mock_binarize.assert_called_once()
        mock_deskew.assert_called_once()
        
        # Überprüfen der Rückgabetypen
        self.assertIsInstance(pil_image, Image.Image)
        self.assertIsInstance(np_image, np.ndarray)
        
        # Test mit deaktivierten Optionen
        mock_denoise.reset_mock()
        mock_binarize.reset_mock()
        mock_deskew.reset_mock()
        
        pil_image, np_image = enhance_image_for_ocr(
            self.test_image_path, 
            denoise=False, 
            binarize=False, 
            deskew=False
        )
        
        # Überprüfen, dass keine der Funktionen aufgerufen wurde
        mock_denoise.assert_not_called()
        mock_binarize.assert_not_called()
        mock_deskew.assert_not_called()
    
    @patch('models_app.ocr.utils.image_processing.cv2')
    def test_detect_text_regions(self, mock_cv2):
        """Test für detect_text_regions Funktion"""
        # Mock-Setup für OpenCV
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.getStructuringElement.return_value = np.ones((5, 5), dtype=np.uint8)
        mock_cv2.dilate.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.erode.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.findContours.return_value = (
            [np.array([[[10, 10]], [[50, 10]], [[50, 30]], [[10, 30]]])], 
            None
        )
        mock_cv2.boundingRect.return_value = (10, 10, 40, 20)
        
        # Test der Funktion
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        boxes = detect_text_regions(gray_image)
        
        # Überprüfen, ob die richtigen Funktionen aufgerufen wurden
        mock_cv2.getStructuringElement.assert_called_once()
        mock_cv2.dilate.assert_called_once()
        mock_cv2.erode.assert_called_once()
        mock_cv2.findContours.assert_called_once()
        mock_cv2.boundingRect.assert_called_once()
        
        # Überprüfen des Ergebnisses
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], (10, 10, 40, 20))
    
    def test_crop_image(self):
        """Test für crop_image Funktion"""
        # Test mit NumPy-Array
        np_image = np.zeros((100, 100, 3), dtype=np.uint8)
        np_image[20:40, 30:60] = 255  # Weißes Rechteck
        
        # Test mit (x, y, w, h) Format
        cropped = crop_image(np_image, (30, 20, 30, 20))
        self.assertEqual(cropped.shape, (20, 30, 3))
        self.assertTrue(np.all(cropped == 255))
        
        # Test mit (x1, y1, x2, y2) Format
        cropped = crop_image(np_image, (30, 20, 60, 40))
        self.assertEqual(cropped.shape, (20, 30, 3))
        self.assertTrue(np.all(cropped == 255))
        
        # Test mit PIL-Bild
        pil_image = Image.fromarray(np_image)
        cropped = crop_image(pil_image, (30, 20, 30, 20))
        self.assertEqual(cropped.shape, (20, 30, 3))
        self.assertTrue(np.all(cropped == 255))
        
        # Test mit ungültigen Koordinaten (sollten auf Bildgrenzen beschränkt werden)
        cropped = crop_image(np_image, (-10, -10, 150, 150))
        self.assertEqual(cropped.shape, (100, 100, 3))

if __name__ == '__main__':
    unittest.main()