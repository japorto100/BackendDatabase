"""
Tests für den TableExtractionAdapter unter Verwendung der Basis-Testklasse.
"""

import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from PIL import Image

from models_app.ocr.table_extraction_adapter import TableExtractionAdapter
from models_app.tests.ocr.test_base_adapter import BaseOCRAdapterTest

class TestTableExtractionAdapter(BaseOCRAdapterTest):
    """Test-Klasse für den TableExtractionAdapter."""
    
    ADAPTER_CLASS = TableExtractionAdapter
    CONFIG_DICT = {
        'table_engine': 'paddle',
        'use_gpu': False,
        'lang': 'en',
        'output_format': 'csv'
    }
    MOCK_IMPORTS = ['models_app.ocr.table_extraction_adapter.PPStructure']
    
    def setUp(self):
        """Setup für Tests"""
        super().setUp()
        # Erstelle ein spezielles Test-Tabellenbild
        self.test_table_image = self._create_test_table_image()
        self.test_table_image_path = self._save_test_image(self.test_table_image, "test_table.png")
    
    @patch('models_app.ocr.table_extraction_adapter.PPStructure')
    def test_initialize_paddle(self, mock_pp_structure):
        """Test der Initialisierung mit PaddleOCR Structure."""
        # Mock für PPStructure
        mock_instance = MagicMock()
        mock_pp_structure.return_value = mock_instance
        
        # Adapter initialisieren
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        mock_pp_structure.assert_called_once()
    
    @patch('models_app.ocr.table_extraction_adapter.PADDLE_STRUCTURE_AVAILABLE', False)
    def test_initialize_opencv(self):
        """Test der Initialisierung mit OpenCV Fallback."""
        # Adapter initialisieren
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        self.assertEqual(self.adapter.engine, 'opencv')
    
    @patch('models_app.ocr.table_extraction_adapter.PPStructure')
    def test_process_image_with_paddle(self, mock_pp_structure):
        """Test der Bildverarbeitung mit PaddleOCR Structure."""
        # Mock für PPStructure
        mock_instance = MagicMock()
        mock_pp_structure.return_value = mock_instance
        
        # Mock für die Tabellenerkennung
        table_result = [{
            'type': 'table',
            'bbox': [10, 10, 590, 390],
            'res': {
                'html': '<table><tr><td>Header 1</td><td>Header 2</td></tr><tr><td>Data 1</td><td>Data 2</td></tr></table>',
                'cells': [
                    {'text': 'Header 1', 'bbox': [10, 10, 300, 60]},
                    {'text': 'Header 2', 'bbox': [310, 10, 590, 60]},
                    {'text': 'Data 1', 'bbox': [10, 70, 300, 120]},
                    {'text': 'Data 2', 'bbox': [310, 70, 590, 120]}
                ]
            }
        }]
        mock_instance.return_value = table_result
        
        # Adapter initialisieren
        self.adapter.model = mock_instance
        self.adapter.is_initialized = True
        self.adapter.engine = 'paddle'
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_table_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertIn("tables", result)
        self.assertEqual(len(result["tables"]), 1)
        self.assertIn("html", result["tables"][0])
        self.assertIn("csv", result["tables"][0])
        self.assertIn("cells", result["tables"][0])
        self.assertEqual(result["model"], "TableExtractionAdapter")
    
    @patch('models_app.ocr.table_extraction_adapter.cv2')
    @patch('models_app.ocr.table_extraction_adapter.pytesseract')
    def test_process_image_with_opencv(self, mock_pytesseract, mock_cv2):
        """Test der Bildverarbeitung mit OpenCV Fallback."""
        # Mocks für OpenCV-Funktionen
        mock_cv2.cvtColor.return_value = np.zeros((400, 600), dtype=np.uint8)
        mock_cv2.threshold.return_value = (None, np.zeros((400, 600), dtype=np.uint8))
        mock_cv2.findContours.return_value = ([
            np.array([[10, 10], [590, 10], [590, 390], [10, 390]])
        ], None)
        mock_cv2.boundingRect.return_value = (10, 10, 580, 380)
        
        # Mock für pytesseract
        mock_pytesseract.image_to_string.return_value = "Header 1 Header 2\nData 1 Data 2"
        mock_pytesseract.image_to_data.return_value = {
            'text': ['Header 1', 'Header 2', 'Data 1', 'Data 2'],
            'conf': [90, 85, 95, 92],
            'left': [10, 310, 10, 310],
            'top': [10, 10, 70, 70],
            'width': [290, 280, 290, 280],
            'height': [50, 50, 50, 50]
        }
        
        # Adapter initialisieren
        self.adapter.is_initialized = True
        self.adapter.engine = 'opencv'
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_table_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertIn("tables", result)
        self.assertEqual(len(result["tables"]), 1)
        self.assertIn("csv", result["tables"][0])
        self.assertEqual(result["model"], "TableExtractionAdapter")
    
    def test_format_table(self):
        """Test der Tabellenformatierung."""
        # Testdaten
        table_data = [
            ["Header 1", "Header 2"],
            ["Data 1", "Data 2"]
        ]
        
        # CSV-Format
        csv_result = self.adapter._format_table(table_data, 'csv')
        self.assertIsInstance(csv_result, str)
        self.assertIn("Header 1", csv_result)
        self.assertIn("Data 1", csv_result)
        
        # HTML-Format
        html_result = self.adapter._format_table(table_data, 'html')
        self.assertIsInstance(html_result, str)
        self.assertIn("<table", html_result)
        self.assertIn("<tr", html_result)
        self.assertIn("<td", html_result)
        self.assertIn("Header 1", html_result)
        
        # JSON-Format
        json_result = self.adapter._format_table(table_data, 'json')
        self.assertIsInstance(json_result, str)
        self.assertIn("Header 1", json_result)
        self.assertIn("Data 1", json_result)
        
        # DataFrame-Format
        df_result = self.adapter._format_table(table_data, 'dataframe')
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(df_result.shape, (1, 2))
        self.assertEqual(df_result.columns.tolist(), ["Header 1", "Header 2"])
    
    def test_get_supported_languages(self):
        """Test für das Abrufen unterstützter Sprachen."""
        languages = self.adapter.get_supported_languages()
        
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)
    
    def test_get_model_info(self):
        """Test für das Abrufen von Modellinformationen."""
        # Adapter initialisieren
        self.adapter.is_initialized = True
        self.adapter.engine = 'paddle'
        
        info = self.adapter.get_model_info()
        
        self.assertEqual(info["name"], "TableExtractionAdapter")
        self.assertEqual(info["type"], "Table Extraction")
        self.assertIn("version", info)
        self.assertTrue(info["capabilities"]["table_extraction"]) 