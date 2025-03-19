"""
Tests für die Dokumentenanalyse aus dem document_analyzer Modul.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from models_app.ocr.utils.document_analyzer import (
    analyze_document_structure, detect_lines, detect_tables,
    detect_images, has_mathematical_formulas, classify_document_type,
    check_if_has_tables, get_document_complexity
)

class TestDocumentAnalyzer(unittest.TestCase):
    """Test-Klasse für die Dokumentenanalyse"""
    
    def setUp(self):
        """Setup für Tests"""
        # Testbild erstellen
        self.test_image = Image.new('RGB', (300, 400), color='white')
        
        # NumPy-Array des Testbilds
        self.test_np_image = np.array(self.test_image)
        
        # Graustufenbild
        self.test_gray_image = np.zeros((400, 300), dtype=np.uint8)
        self.test_gray_image[100:300, 50:250] = 255  # Weißes Rechteck
    
    @patch('models_app.ocr.utils.document_analyzer.cv2')
    @patch('models_app.ocr.utils.document_analyzer.np')
    @patch('models_app.ocr.utils.document_analyzer.detect_lines')
    @patch('models_app.ocr.utils.document_analyzer.detect_tables')
    @patch('models_app.ocr.utils.document_analyzer.detect_images')
    def test_analyze_document_structure(self, mock_detect_images, mock_detect_tables, 
                                       mock_detect_lines, mock_np, mock_cv2):
        """Test für die analyze_document_structure Funktion"""
        # Mock-Rückgabewerte
        mock_cv2.cvtColor.return_value = self.test_gray_image
        mock_detect_lines.return_value = (
            [(10, 50, 290, 50), (10, 100, 290, 100)],  # Horizontale Linien
            [(50, 10, 50, 390), (250, 10, 250, 390)]   # Vertikale Linien
        )
        mock_detect_tables.return_value = [(50, 50, 250, 250)]
        mock_detect_images.return_value = [(150, 300, 250, 350)]
        
        # Funktion aufrufen
        structure = analyze_document_structure(self.test_np_image)
        
        # Überprüfen, ob alle notwendigen Funktionen aufgerufen wurden
        mock_cv2.cvtColor.assert_called_once()
        mock_detect_lines.assert_called_once()
        mock_detect_tables.assert_called_once()
        mock_detect_images.assert_called_once()
        
        # Überprüfen der Struktur
        self.assertIn("dimensions", structure)
        self.assertEqual(structure["dimensions"], {"width": 300, "height": 400})
        
        self.assertIn("lines", structure)
        self.assertEqual(len(structure["lines"]["horizontal"]), 2)
        self.assertEqual(len(structure["lines"]["vertical"]), 2)
        
        self.assertIn("tables", structure)
        self.assertEqual(len(structure["tables"]), 1)
        
        self.assertIn("images", structure)
        self.assertEqual(len(structure["images"]), 1)
        
        self.assertIn("layout_complexity", structure)
    
    @patch('models_app.ocr.utils.document_analyzer.cv2')
    def test_detect_lines(self, mock_cv2):
        """Test für die detect_lines Funktion"""
        # Mock-Rückgabewerte
        # HoughLinesP für horizontale Linien
        mock_cv2.HoughLinesP.side_effect = [
            np.array([[10, 50, 290, 50], [10, 100, 290, 100]]),  # Horizontale Linien
            np.array([[50, 10, 50, 390], [250, 10, 250, 390]])   # Vertikale Linien
        ]
        
        # Funktion aufrufen
        horizontal_lines, vertical_lines = detect_lines(self.test_gray_image)
        
        # Überprüfen, ob HoughLinesP zweimal aufgerufen wurde (für horizontale und vertikale Linien)
        self.assertEqual(mock_cv2.HoughLinesP.call_count, 2)
        
        # Überprüfen der erkannten Linien
        self.assertEqual(len(horizontal_lines), 2)
        self.assertEqual(len(vertical_lines), 2)
        
        # Überprüfen der ersten horizontalen Linie
        self.assertEqual(horizontal_lines[0], (10, 50, 290, 50))
    
    @patch('models_app.ocr.utils.document_analyzer.cv2')
    def test_detect_tables(self, mock_cv2):
        """Test für die detect_tables Funktion"""
        # Mock-Rückgabewerte für die Linien
        horizontal_lines = [(10, 50, 290, 50), (10, 100, 290, 100)]
        vertical_lines = [(50, 10, 50, 390), (250, 10, 250, 390)]
        
        # Mock für contourArea und boundingRect
        mock_cv2.contourArea.return_value = 40000  # 200x200
        mock_cv2.boundingRect.return_value = (50, 50, 200, 200)
        
        # Mock für findContours
        mock_contour = np.array([[[50, 50]], [[250, 50]], [[250, 250]], [[50, 250]]])
        mock_cv2.findContours.return_value = ([mock_contour], None)
        
        # Funktion aufrufen
        tables = detect_tables(self.test_gray_image, horizontal_lines, vertical_lines)
        
        # Überprüfen, ob findContours aufgerufen wurde
        mock_cv2.findContours.assert_called_once()
        
        # Überprüfen der erkannten Tabellen
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0], (50, 50, 200, 200))
    
    @patch('models_app.ocr.utils.document_analyzer.cv2')
    def test_detect_images(self, mock_cv2):
        """Test für die detect_images Funktion"""
        # Mock-Rückgabewerte für die Bild-Erkennung
        
        # Mock für Canny
        mock_cv2.Canny.return_value = np.zeros((400, 300), dtype=np.uint8)
        
        # Mock für findContours
        mock_contour1 = np.array([[[50, 50]], [[250, 50]], [[250, 150]], [[50, 150]]])
        mock_contour2 = np.array([[[100, 200]], [[200, 200]], [[200, 300]], [[100, 300]]])
        mock_cv2.findContours.return_value = ([mock_contour1, mock_contour2], None)
        
        # Mock für contourArea und boundingRect
        mock_cv2.contourArea.side_effect = [20000, 10000]  # Flächen der Konturen
        mock_cv2.boundingRect.side_effect = [(50, 50, 200, 100), (100, 200, 100, 100)]
        
        # Funktion aufrufen
        images = detect_images(self.test_np_image)
        
        # Überprüfen, ob die notwendigen Funktionen aufgerufen wurden
        mock_cv2.Canny.assert_called_once()
        mock_cv2.findContours.assert_called_once()
        
        # Überprüfen der erkannten Bilder
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0], (50, 50, 200, 100))
        self.assertEqual(images[1], (100, 200, 100, 100))
    
    @patch('models_app.ocr.utils.document_analyzer.cv2')
    @patch('models_app.ocr.utils.document_analyzer.np')
    def test_has_mathematical_formulas(self, mock_np, mock_cv2):
        """Test für die has_mathematical_formulas Funktion"""
        # Mock-Rückgabewerte für die Formel-Erkennung
        
        # Mock für Template-Matching
        mock_cv2.imread.return_value = np.zeros((50, 50, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((50, 50), dtype=np.uint8)
        mock_cv2.resize.return_value = np.zeros((50, 50), dtype=np.uint8)
        
        # Mock für matchTemplate
        mock_template_result = np.zeros((350, 250), dtype=np.float32)
        mock_template_result[100, 100] = 0.9  # Hohe Übereinstimmung an einer Stelle
        mock_cv2.matchTemplate.return_value = mock_template_result
        
        # Mock für minMaxLoc
        mock_cv2.minMaxLoc.return_value = (0.0, 0.9, (0, 0), (100, 100))
        
        # Funktion aufrufen
        has_formulas, confidence = has_mathematical_formulas(self.test_np_image)
        
        # Überprüfen, ob die notwendigen Funktionen aufgerufen wurden
        mock_cv2.imread.assert_called()  # Wird mehrmals für verschiedene Vorlagen aufgerufen
        mock_cv2.matchTemplate.assert_called()  # Wird mehrmals aufgerufen
        
        # Überprüfen des Ergebnisses (angenommen, dass mit den Mock-Werten Formeln erkannt werden)
        self.assertTrue(has_formulas)
        self.assertGreater(confidence, 0.5)
    
    def test_classify_document_type(self):
        """Test für die classify_document_type Funktion"""
        # Dokument mit Tabellen
        structure_with_tables = {
            "tables": [(50, 50, 200, 200)],
            "lines": {
                "horizontal": [(10, 50, 290, 50), (10, 100, 290, 100)],
                "vertical": [(50, 10, 50, 390), (250, 10, 250, 390)]
            },
            "images": []
        }
        
        doc_type, confidence = classify_document_type(structure_with_tables)
        self.assertEqual(doc_type, "table")
        self.assertGreater(confidence, 0.5)
        
        # Dokument mit vielen Bildern
        structure_with_images = {
            "tables": [],
            "lines": {
                "horizontal": [],
                "vertical": []
            },
            "images": [(50, 50, 150, 150), (200, 50, 300, 150), (50, 200, 300, 300)]
        }
        
        doc_type, confidence = classify_document_type(structure_with_images)
        self.assertEqual(doc_type, "image_heavy")
        self.assertGreater(confidence, 0.5)
        
        # Dokument mit wenigen Strukturelementen
        structure_simple = {
            "tables": [],
            "lines": {
                "horizontal": [(10, 50, 290, 50)],
                "vertical": []
            },
            "images": []
        }
        
        doc_type, confidence = classify_document_type(structure_simple)
        self.assertEqual(doc_type, "text")
        self.assertGreater(confidence, 0.5)
    
    @patch('models_app.ocr.utils.document_analyzer.cv2')
    @patch('models_app.ocr.utils.document_analyzer.detect_lines')
    @patch('models_app.ocr.utils.document_analyzer.detect_tables')
    def test_check_if_has_tables(self, mock_detect_tables, mock_detect_lines, mock_cv2):
        """Test für die check_if_has_tables Funktion"""
        # Mock-Rückgabewerte
        mock_cv2.cvtColor.return_value = self.test_gray_image
        mock_detect_lines.return_value = (
            [(10, 50, 290, 50), (10, 100, 290, 100)],  # Horizontale Linien
            [(50, 10, 50, 390), (250, 10, 250, 390)]   # Vertikale Linien
        )
        mock_detect_tables.return_value = [(50, 50, 250, 250)]
        
        # Funktion aufrufen
        has_tables, confidence = check_if_has_tables(self.test_np_image)
        
        # Überprüfen, ob die notwendigen Funktionen aufgerufen wurden
        mock_cv2.cvtColor.assert_called_once()
        mock_detect_lines.assert_called_once()
        mock_detect_tables.assert_called_once()
        
        # Überprüfen des Ergebnisses
        self.assertTrue(has_tables)
        self.assertGreater(confidence, 0.5)
        
        # Test für den Fall ohne Tabellen
        mock_detect_tables.return_value = []
        
        has_tables, confidence = check_if_has_tables(self.test_np_image)
        self.assertFalse(has_tables)
        self.assertLess(confidence, 0.5)
    
    @patch('models_app.ocr.utils.document_analyzer.analyze_document_structure')
    def test_get_document_complexity(self, mock_analyze_structure):
        """Test für die get_document_complexity Funktion"""
        # Mock-Rückgabewerte
        mock_analyze_structure.return_value = {
            "dimensions": {"width": 300, "height": 400},
            "lines": {
                "horizontal": [(10, 50, 290, 50), (10, 100, 290, 100)],
                "vertical": [(50, 10, 50, 390), (250, 10, 250, 390)]
            },
            "tables": [(50, 50, 250, 250)],
            "images": [(150, 300, 250, 350)],
            "layout_complexity": 0.65
        }
        
        # Funktion aufrufen
        complexity = get_document_complexity(self.test_np_image)
        
        # Überprüfen, ob die analyze_document_structure-Funktion aufgerufen wurde
        mock_analyze_structure.assert_called_once_with(self.test_np_image)
        
        # Überprüfen des Ergebnisses
        self.assertEqual(complexity, 0.65)

if __name__ == '__main__':
    unittest.main()