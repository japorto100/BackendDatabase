"""
Tests für die Fehlerbehandlung aus dem error_handler Modul.
"""

import unittest
from unittest.mock import patch, MagicMock
import traceback
import sys

from models_app.ocr.utils.error_handler import (
    OCRError, ModelNotAvailableError, ModelInitializationError,
    ImageProcessingError, UnsupportedFormatError,
    create_error_result, handle_ocr_errors
)

class TestErrorHandler(unittest.TestCase):
    """Test-Klasse für die OCR-Fehlerbehandlung"""
    
    def test_error_classes(self):
        """Test für die verschiedenen Fehlerklassen"""
        # Basisklasse
        error = OCRError("Testfehler", error_code="TEST_ERROR", details={"test": "detail"})
        self.assertEqual(str(error), "Testfehler")
        self.assertEqual(error.error_code, "TEST_ERROR")
        self.assertEqual(error.details, {"test": "detail"})
        
        # Abgeleitete Klassen
        model_error = ModelNotAvailableError("Modell nicht verfügbar")
        self.assertIsInstance(model_error, OCRError)
        self.assertEqual(str(model_error), "Modell nicht verfügbar")
        
        init_error = ModelInitializationError("Initialisierungsfehler")
        self.assertIsInstance(init_error, OCRError)
        
        img_error = ImageProcessingError("Bildverarbeitungsfehler")
        self.assertIsInstance(img_error, OCRError)
        
        format_error = UnsupportedFormatError("Nicht unterstütztes Format")
        self.assertIsInstance(format_error, OCRError)
    
    @patch('models_app.ocr.utils.error_handler.create_standard_result')
    def test_create_error_result(self, mock_create_result):
        """Test für die create_error_result Funktion"""
        # Mock-Rückgabewert einrichten
        mock_create_result.return_value = {
            "text": "Error: Testfehler",
            "confidence": 0.0,
            "model": "error",
            "metadata": {"error": "Testfehler", "error_type": "test_error", "error_code": "TEST_ERROR"}
        }
        
        # Funktion aufrufen
        result = create_error_result(
            error_message="Testfehler",
            error_type="test_error",
            error_code="TEST_ERROR",
            details={"test": "detail"}
        )
        
        # Überprüfen, ob create_standard_result mit den richtigen Parametern aufgerufen wurde
        mock_create_result.assert_called_once()
        call_args = mock_create_result.call_args[1]
        self.assertEqual(call_args["text"], "Error: Testfehler")
        self.assertEqual(call_args["confidence"], 0.0)
        self.assertEqual(call_args["model"], "error")
        
        # Überprüfen der Metadaten
        metadata = call_args["metadata"]
        self.assertEqual(metadata["error"], "Testfehler")
        self.assertEqual(metadata["error_type"], "test_error")
        self.assertEqual(metadata["error_code"], "TEST_ERROR")
        self.assertEqual(metadata["error_details"], {"test": "detail"})
        
        # Überprüfen des Ergebnisses
        self.assertEqual(result["text"], "Error: Testfehler")
        self.assertEqual(result["confidence"], 0.0)
    
    def test_handle_ocr_errors_decorator_success(self):
        """Test für den handle_ocr_errors Decorator bei erfolgreicher Ausführung"""
        # Testfunktion definieren
        @handle_ocr_errors
        def successful_function():
            return "Erfolg!"
        
        # Funktion aufrufen und Ergebnis überprüfen
        result = successful_function()
        self.assertEqual(result, "Erfolg!")
    
    def test_handle_ocr_errors_decorator_model_not_available(self):
        """Test für den handle_ocr_errors Decorator bei ModelNotAvailableError"""
        # Testfunktion definieren
        @handle_ocr_errors
        def failing_function():
            raise ModelNotAvailableError("Modell nicht verfügbar", details={"model": "test_model"})
        
        # Funktion aufrufen und Ergebnis überprüfen
        result = failing_function()
        
        self.assertIn("error", result)
        self.assertEqual(result["metadata"]["error"], "Modell nicht verfügbar")
        self.assertEqual(result["metadata"]["error_type"], "model_not_available")
        self.assertEqual(result["metadata"]["error_details"]["model"], "test_model")
        self.assertEqual(result["confidence"], 0.0)
    
    def test_handle_ocr_errors_decorator_model_initialization(self):
        """Test für den handle_ocr_errors Decorator bei ModelInitializationError"""
        # Testfunktion definieren
        @handle_ocr_errors
        def failing_function():
            raise ModelInitializationError("Initialisierungsfehler")
        
        # Funktion aufrufen und Ergebnis überprüfen
        result = failing_function()
        
        self.assertIn("error", result["metadata"])
        self.assertEqual(result["metadata"]["error"], "Initialisierungsfehler")
        self.assertEqual(result["metadata"]["error_type"], "model_initialization_error")
    
    def test_handle_ocr_errors_decorator_image_processing(self):
        """Test für den handle_ocr_errors Decorator bei ImageProcessingError"""
        # Testfunktion definieren
        @handle_ocr_errors
        def failing_function():
            raise ImageProcessingError("Bildverarbeitungsfehler")
        
        # Funktion aufrufen und Ergebnis überprüfen
        result = failing_function()
        
        self.assertIn("error", result["metadata"])
        self.assertEqual(result["metadata"]["error"], "Bildverarbeitungsfehler")
        self.assertEqual(result["metadata"]["error_type"], "image_processing_error")
    
    def test_handle_ocr_errors_decorator_unsupported_format(self):
        """Test für den handle_ocr_errors Decorator bei UnsupportedFormatError"""
        # Testfunktion definieren
        @handle_ocr_errors
        def failing_function():
            raise UnsupportedFormatError("Nicht unterstütztes Format")
        
        # Funktion aufrufen und Ergebnis überprüfen
        result = failing_function()
        
        self.assertIn("error", result["metadata"])
        self.assertEqual(result["metadata"]["error"], "Nicht unterstütztes Format")
        self.assertEqual(result["metadata"]["error_type"], "unsupported_format")
    
    def test_handle_ocr_errors_decorator_general_ocr_error(self):
        """Test für den handle_ocr_errors Decorator bei allgemeinem OCRError"""
        # Testfunktion definieren
        @handle_ocr_errors
        def failing_function():
            raise OCRError("Allgemeiner OCR-Fehler", error_code="GENERAL_ERROR")
        
        # Funktion aufrufen und Ergebnis überprüfen
        result = failing_function()
        
        self.assertIn("error", result["metadata"])
        self.assertEqual(result["metadata"]["error"], "Allgemeiner OCR-Fehler")
        self.assertEqual(result["metadata"]["error_type"], "ocr_error")
        self.assertEqual(result["metadata"]["error_code"], "GENERAL_ERROR")
    
    def test_handle_ocr_errors_decorator_unexpected_error(self):
        """Test für den handle_ocr_errors Decorator bei unerwartetem Fehler"""
        # Testfunktion definieren
        @handle_ocr_errors
        def failing_function():
            # Einen unerwarteten Fehler auslösen
            raise ValueError("Unerwarteter Fehler")
        
        # Funktion aufrufen und Ergebnis überprüfen
        result = failing_function()
        
        self.assertIn("error", result["metadata"])
        self.assertEqual(result["metadata"]["error"], "Unerwarteter Fehler")
        self.assertEqual(result["metadata"]["error_type"], "unexpected_error")
        # Überprüfen, ob der Traceback enthalten ist
        self.assertIn("traceback", result["metadata"]["error_details"])
        self.assertIn("ValueError", result["metadata"]["error_details"]["traceback"])

if __name__ == '__main__':
    unittest.main() 