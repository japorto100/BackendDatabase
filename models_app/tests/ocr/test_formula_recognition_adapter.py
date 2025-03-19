"""
Tests für den FormulaRecognitionAdapter unter Verwendung der Basis-Testklasse.
"""

import numpy as np
import torch
from unittest.mock import patch, MagicMock
from PIL import Image

from models_app.ocr.formula_recognition_adapter import FormulaRecognitionAdapter
from models_app.tests.ocr.test_base_adapter import BaseOCRAdapterTest

class TestFormulaRecognitionAdapter(BaseOCRAdapterTest):
    """Test-Klasse für den FormulaRecognitionAdapter."""
    
    ADAPTER_CLASS = FormulaRecognitionAdapter
    CONFIG_DICT = {
        'engine': 'pix2tex',
        'model_name': 'pix2tex/pix2tex-base',
        'max_length': 512,
        'gpu': False,
        'confidence_threshold': 0.7,
        'preprocess': True
    }
    MOCK_IMPORTS = [
        'models_app.ocr.formula_recognition_adapter.LatexOCR',
        'models_app.ocr.formula_recognition_adapter.torch.cuda.is_available',
        'models_app.ocr.formula_recognition_adapter.TrOCRProcessor.from_pretrained',
        'models_app.ocr.formula_recognition_adapter.VisionEncoderDecoderModel.from_pretrained'
    ]
    
    def setUp(self):
        """Setup für Tests"""
        super().setUp()
        # Erstelle ein spezielles Formel-Testbild
        self.test_formula_image = self._create_test_formula_image()
        self.test_formula_image_path = self._save_test_image(self.test_formula_image, "test_formula.png")
    
    @patch('models_app.ocr.formula_recognition_adapter.PIX2TEX_AVAILABLE', False)
    @patch('models_app.ocr.formula_recognition_adapter.LATEX_OCR_AVAILABLE', False)
    def test_initialize_engines_not_available(self):
        """Test der Initialisierung, wenn keine Engines verfügbar sind."""
        result = self.adapter.initialize()
        
        self.assertFalse(result)
        self.assertFalse(self.adapter.is_initialized)
        self.assertIn("error", result)
        self.assertIn("not available", result["metadata"]["error"])
    
    @patch('models_app.ocr.formula_recognition_adapter.PIX2TEX_AVAILABLE', True)
    @patch('models_app.ocr.formula_recognition_adapter.LatexOCR')
    def test_initialize_pix2tex(self, mock_latex_ocr):
        """Test der Initialisierung mit pix2tex."""
        # Mock für LatexOCR
        mock_instance = MagicMock()
        mock_latex_ocr.return_value = mock_instance
        
        # Adapter initialisieren
        self.adapter.config['engine'] = 'pix2tex'
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        self.assertEqual(self.adapter.engine, 'pix2tex')
        mock_latex_ocr.assert_called_once()
    
    @patch('models_app.ocr.formula_recognition_adapter.LATEX_OCR_AVAILABLE', True)
    @patch('models_app.ocr.formula_recognition_adapter.torch.cuda.is_available')
    @patch('models_app.ocr.formula_recognition_adapter.TrOCRProcessor.from_pretrained')
    @patch('models_app.ocr.formula_recognition_adapter.VisionEncoderDecoderModel.from_pretrained')
    def test_initialize_latex_ocr(self, mock_model, mock_processor, mock_cuda):
        """Test der Initialisierung mit latex-ocr."""
        # Mocks einrichten
        mock_cuda.return_value = False
        
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        
        # Adapter initialisieren
        self.adapter.config['engine'] = 'latex-ocr'
        result = self.adapter.initialize()
        
        self.assertTrue(result)
        self.assertTrue(self.adapter.is_initialized)
        self.assertEqual(self.adapter.engine, 'latex-ocr')
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
        self.assertIsNotNone(self.adapter.pix2tex_model)
    
    @patch('models_app.ocr.formula_recognition_adapter.PIX2TEX_AVAILABLE', True)
    def test_process_image_with_file_path_pix2tex(self):
        """Test der Bildverarbeitung mit pix2tex."""
        # Mock-Modell einrichten
        mock_model = MagicMock()
        mock_model.return_value = "E = mc^2"
        
        # Adapter initialisieren
        self.adapter.pix2tex_model = mock_model
        self.adapter.is_initialized = True
        self.adapter.engine = 'pix2tex'
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_formula_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "E = mc^2")
        self.assertIn("latex", result)
        self.assertEqual(result["latex"], "E = mc^2")
        self.assertEqual(result["model"], "FormulaRecognitionAdapter")
        self.assertEqual(result["metadata"]["engine"], "pix2tex")
    
    @patch('models_app.ocr.formula_recognition_adapter.LATEX_OCR_AVAILABLE', True)
    @patch('models_app.ocr.formula_recognition_adapter.torch')
    def test_process_image_with_file_path_latex_ocr(self, mock_torch):
        """Test der Bildverarbeitung mit latex-ocr."""
        # Mock für die Bildverarbeitung
        mock_tensor = MagicMock()
        mock_torch.zeros.return_value = mock_tensor
        
        # Mock für den Processor
        self.adapter.processor = MagicMock()
        self.adapter.processor.return_value = {"pixel_values": mock_tensor}
        
        # Mock für das Modell
        self.adapter.trocr_model = MagicMock()
        mock_generated_ids = MagicMock()
        self.adapter.trocr_model.generate.return_value = mock_generated_ids
        
        # Mock für den Decoder
        self.adapter.processor.batch_decode.return_value = ["\\alpha = \\beta^2 + \\gamma"]
        
        # Adapter initialisieren
        self.adapter.is_initialized = True
        self.adapter.engine = 'latex-ocr'
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_formula_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertIn("latex", result)
        self.assertEqual(result["latex"], "\\alpha = \\beta^2 + \\gamma")
        self.assertEqual(result["model"], "FormulaRecognitionAdapter")
        self.assertEqual(result["metadata"]["engine"], "latex-ocr")
    
    @patch('models_app.ocr.formula_recognition_adapter.PIX2TEX_AVAILABLE', True)
    def test_process_image_with_numpy_array(self):
        """Test der Bildverarbeitung mit einem NumPy-Array."""
        # Mock-Modell einrichten
        mock_model = MagicMock()
        mock_model.return_value = "\\frac{dx}{dt} = v"
        
        # Adapter initialisieren
        self.adapter.pix2tex_model = mock_model
        self.adapter.is_initialized = True
        self.adapter.engine = 'pix2tex'
        
        # Bild in NumPy-Array konvertieren
        image_array = np.array(self.test_formula_image)
        
        # Bild verarbeiten
        result = self.adapter.process_image(image_array)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "\\frac{dx}{dt} = v")
        self.assertIn("latex", result)
        self.assertEqual(result["latex"], "\\frac{dx}{dt} = v")
        self.assertEqual(result["model"], "FormulaRecognitionAdapter")
    
    @patch('models_app.ocr.formula_recognition_adapter.PIX2TEX_AVAILABLE', True)
    def test_process_image_with_text_output(self):
        """Test der Bildverarbeitung mit Textausgabe."""
        # Mock-Modell einrichten
        mock_model = MagicMock()
        mock_model.return_value = "E = mc^2"
        
        # Adapter initialisieren
        self.adapter.pix2tex_model = mock_model
        self.adapter.is_initialized = True
        self.adapter.engine = 'pix2tex'
        
        # Optionen für die Verarbeitung
        options = {'output_format': 'text'}
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_formula_image_path, options)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "E = mc^2")
        self.assertNotIn("formatted_text", result)
        self.assertEqual(result["model"], "FormulaRecognitionAdapter")
    
    @patch('models_app.ocr.formula_recognition_adapter.PIX2TEX_AVAILABLE', True)
    def test_process_image_with_region(self):
        """Test der Bildverarbeitung mit Regionsangabe."""
        # Mock-Modell einrichten
        mock_model = MagicMock()
        mock_model.return_value = "\\int_0^\\infty e^{-x} dx = 1"
        
        # Adapter initialisieren
        self.adapter.pix2tex_model = mock_model
        self.adapter.is_initialized = True
        self.adapter.engine = 'pix2tex'
        
        # Optionen für die Verarbeitung mit Region
        options = {'region': [10, 20, 100, 80]}
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_formula_image_path, options)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertEqual(result["text"], "\\int_0^\\infty e^{-x} dx = 1")
        self.assertIn("latex", result)
        self.assertEqual(result["latex"], "\\int_0^\\infty e^{-x} dx = 1")
        self.assertEqual(result["model"], "FormulaRecognitionAdapter")
    
    @patch('models_app.ocr.formula_recognition_adapter.PIX2TEX_AVAILABLE', True)
    @patch('models_app.ocr.formula_recognition_adapter.cv2')
    def test_detect_formula_regions(self, mock_cv2):
        """Test der Erkennung von Formelregionen."""
        # Mocks für OpenCV-Funktionen
        mock_cv2.cvtColor.return_value = np.zeros((200, 400), dtype=np.uint8)
        mock_cv2.adaptiveThreshold.return_value = np.zeros((200, 400), dtype=np.uint8)
        mock_cv2.findContours.return_value = ([
            np.array([[10, 10], [100, 10], [100, 50], [10, 50]]),
            np.array([[10, 60], [100, 60], [100, 100], [10, 100]])
        ], None)
        mock_cv2.contourArea.side_effect = [4500, 2000]
        mock_cv2.boundingRect.side_effect = [(10, 10, 90, 40), (10, 60, 90, 40)]
        
        # Formelregionen erkennen
        regions = self.adapter._detect_formula_regions(self.test_formula_image_path)
        
        # Überprüfen der Ergebnisse
        self.assertIsInstance(regions, list)
        self.assertEqual(len(regions), 2)
        self.assertEqual(len(regions[0]), 4)  # [x, y, w, h]
    
    @patch('models_app.ocr.formula_recognition_adapter.PIX2TEX_AVAILABLE', True)
    def test_process_image_with_formula_detection(self):
        """Test der Bildverarbeitung mit Formelerkennung."""
        # Mock-Modell einrichten
        mock_model = MagicMock()
        mock_model.return_value = "E = mc^2"
        
        # Mock für die Formelerkennung
        self.adapter._detect_formula_regions = MagicMock()
        self.adapter._detect_formula_regions.return_value = [
            [10, 10, 90, 40],
            [10, 60, 90, 40]
        ]
        
        # Adapter initialisieren
        self.adapter.pix2tex_model = mock_model
        self.adapter.is_initialized = True
        self.adapter.engine = 'pix2tex'
        
        # Optionen für die Verarbeitung mit automatischer Formelerkennung
        options = {'detect_formulas': True}
        
        # Bild verarbeiten
        result = self.adapter.process_image(self.test_formula_image_path, options)
        
        # Überprüfen der Ergebnisse
        self.assertIn("text", result)
        self.assertIn("formulas", result)
        self.assertEqual(len(result["formulas"]), 2)
        self.assertEqual(result["formulas"][0]["latex"], "E = mc^2")
        self.assertEqual(result["model"], "FormulaRecognitionAdapter")
    
    def test_latex_to_text(self):
        """Test für die Konvertierung von LaTeX in lesbaren Text."""
        # Verschiedene LaTeX-Formeln
        latex_formulas = [
            "E = mc^2",
            "\\frac{dx}{dt} = v",
            "\\int_0^{\\infty} e^{-x} dx = 1",
            "a^2 + b^2 = c^2",
            "\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}"
        ]
        
        for latex in latex_formulas:
            text = self.adapter._latex_to_text(latex)
            self.assertIsInstance(text, str)
            self.assertNotIn("\\", text)  # Keine LaTeX-Befehle mehr
    
    def test_get_supported_languages(self):
        """Test für das Abrufen unterstützter Sprachen."""
        languages = self.adapter.get_supported_languages()
        
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)  # Formelsprache ist universell
    
    def test_get_model_info(self):
        """Test für das Abrufen von Modellinformationen."""
        # Adapter initialisieren
        self.adapter.is_initialized = True
        self.adapter.engine = 'pix2tex'
        
        info = self.adapter.get_model_info()
        
        self.assertEqual(info["name"], "FormulaRecognitionAdapter")
        self.assertEqual(info["type"], "Formula Recognition")
        self.assertIn("capabilities", info)
        self.assertTrue(info["capabilities"]["formula_recognition"]) 