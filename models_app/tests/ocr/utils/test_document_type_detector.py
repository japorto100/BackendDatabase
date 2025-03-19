"""
Tests für den Dokumenttyp-Detektor aus dem document_type_detector Modul.
"""

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
import io

from models_app.ocr.utils.document_type_detector import DocumentTypeDetector

class TestDocumentTypeDetector(unittest.TestCase):
    """Test-Klasse für den Dokumenttyp-Detektor"""
    
    def setUp(self):
        """Setup für Tests"""
        self.detector = DocumentTypeDetector()
        
        # Temporäre Testdateien erstellen
        self.temp_dir = tempfile.mkdtemp()
        
        # PDF-Datei
        self.pdf_path = os.path.join(self.temp_dir, "test.pdf")
        with open(self.pdf_path, 'wb') as f:
            f.write(b'%PDF-1.4\n')  # Minimaler PDF-Header
        
        # Bilddatei
        self.image_path = os.path.join(self.temp_dir, "test.jpg")
        with open(self.image_path, 'wb') as f:
            f.write(b'\xff\xd8\xff')  # Minimaler JPEG-Header
        
        # Word-Datei
        self.word_path = os.path.join(self.temp_dir, "test.docx")
        with open(self.word_path, 'wb') as f:
            f.write(b'PK\x03\x04')  # Minimaler ZIP-Header (DOCX ist ein ZIP-Archiv)
        
        # Textdatei
        self.text_path = os.path.join(self.temp_dir, "test.txt")
        with open(self.text_path, 'w') as f:
            f.write("This is a test file.")
    
    def tearDown(self):
        """Cleanup nach Tests"""
        # Temporäre Dateien löschen
        for filename in [self.pdf_path, self.image_path, self.word_path, self.text_path]:
            if os.path.exists(filename):
                os.unlink(filename)
        
        os.rmdir(self.temp_dir)
    
    @patch('models_app.ocr.utils.document_type_detector.magic')
    def test_detect_document_type(self, mock_magic):
        """Test für die detect_document_type Methode"""
        # Mock-Rückgabewerte für verschiedene Dateitypen
        mock_magic.from_file.side_effect = [
            "PDF document, version 1.4",
            "JPEG image data",
            "Microsoft Word 2007+",
            "ASCII text"
        ]
        
        # Test für PDF
        result = self.detector.detect_document_type(self.pdf_path)
        self.assertEqual(result["type"], "pdf")
        self.assertEqual(result["mime_type"], "application/pdf")
        self.assertEqual(result["extension"], "pdf")
        self.assertTrue(result["supported"])
        
        # Test für Bild
        result = self.detector.detect_document_type(self.image_path)
        self.assertEqual(result["type"], "image")
        self.assertEqual(result["mime_type"], "image/jpeg")
        self.assertEqual(result["extension"], "jpg")
        self.assertTrue(result["supported"])
        
        # Test für Word
        result = self.detector.detect_document_type(self.word_path)
        self.assertEqual(result["type"], "office")
        self.assertEqual(result["mime_type"], "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        self.assertEqual(result["extension"], "docx")
        self.assertTrue(result["supported"])
        
        # Test für Text
        result = self.detector.detect_document_type(self.text_path)
        self.assertEqual(result["type"], "text")
        self.assertEqual(result["mime_type"], "text/plain")
        self.assertEqual(result["extension"], "txt")
        self.assertTrue(result["supported"])
        
        # Test für nicht existierende Datei
        with self.assertRaises(FileNotFoundError):
            self.detector.detect_document_type("non_existent_file.txt")
    
    def test_get_document_type(self):
        """Test für die _get_document_type Methode"""
        # Test für PDF
        doc_type = self.detector._get_document_type("application/pdf", "pdf")
        self.assertEqual(doc_type["type"], "pdf")
        self.assertEqual(doc_type["mime_type"], "application/pdf")
        self.assertEqual(doc_type["extension"], "pdf")
        self.assertTrue(doc_type["supported"])
        
        # Test für JPEG-Bild
        doc_type = self.detector._get_document_type("image/jpeg", "jpg")
        self.assertEqual(doc_type["type"], "image")
        self.assertEqual(doc_type["mime_type"], "image/jpeg")
        self.assertEqual(doc_type["extension"], "jpg")
        self.assertTrue(doc_type["supported"])
        
        # Test für PNG-Bild
        doc_type = self.detector._get_document_type("image/png", "png")
        self.assertEqual(doc_type["type"], "image")
        self.assertEqual(doc_type["mime_type"], "image/png")
        self.assertEqual(doc_type["extension"], "png")
        self.assertTrue(doc_type["supported"])
        
        # Test für Word-Dokument
        doc_type = self.detector._get_document_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx")
        self.assertEqual(doc_type["type"], "office")
        self.assertEqual(doc_type["mime_type"], "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        self.assertEqual(doc_type["extension"], "docx")
        self.assertTrue(doc_type["supported"])
        
        # Test für Excel-Dokument
        doc_type = self.detector._get_document_type("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx")
        self.assertEqual(doc_type["type"], "office")
        self.assertEqual(doc_type["mime_type"], "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        self.assertEqual(doc_type["extension"], "xlsx")
        self.assertTrue(doc_type["supported"])
        
        # Test für Textdatei
        doc_type = self.detector._get_document_type("text/plain", "txt")
        self.assertEqual(doc_type["type"], "text")
        self.assertEqual(doc_type["mime_type"], "text/plain")
        self.assertEqual(doc_type["extension"], "txt")
        self.assertTrue(doc_type["supported"])
        
        # Test für nicht unterstützten Dateityp
        doc_type = self.detector._get_document_type("application/octet-stream", "bin")
        self.assertEqual(doc_type["type"], "unknown")
        self.assertEqual(doc_type["mime_type"], "application/octet-stream")
        self.assertEqual(doc_type["extension"], "bin")
        self.assertFalse(doc_type["supported"])
    
    def test_get_unknown_type(self):
        """Test für die _get_unknown_type Methode"""
        # Test ohne Dateipfad
        unknown_type = self.detector._get_unknown_type()
        self.assertEqual(unknown_type["type"], "unknown")
        self.assertEqual(unknown_type["mime_type"], "application/octet-stream")
        self.assertEqual(unknown_type["extension"], "")
        self.assertFalse(unknown_type["supported"])
        
        # Test mit Dateipfad
        unknown_type = self.detector._get_unknown_type("test.xyz")
        self.assertEqual(unknown_type["type"], "unknown")
        self.assertEqual(unknown_type["mime_type"], "application/octet-stream")
        self.assertEqual(unknown_type["extension"], "xyz")
        self.assertFalse(unknown_type["supported"])
    
    def test_is_office_document(self):
        """Test für die is_office_document Methode"""
        # Test für Word-Dokument
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "Microsoft Word 2007+"
            self.assertTrue(self.detector.is_office_document(self.word_path))
        
        # Test für Excel-Dokument
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "Microsoft Excel 2007+"
            self.assertTrue(self.detector.is_office_document("test.xlsx"))
        
        # Test für PowerPoint-Dokument
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "Microsoft PowerPoint 2007+"
            self.assertTrue(self.detector.is_office_document("test.pptx"))
        
        # Test für kein Office-Dokument
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "PDF document"
            self.assertFalse(self.detector.is_office_document(self.pdf_path))
    
    def test_is_image(self):
        """Test für die is_image Methode"""
        # Test für JPEG-Bild
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "JPEG image data"
            self.assertTrue(self.detector.is_image(self.image_path))
        
        # Test für PNG-Bild
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "PNG image data"
            self.assertTrue(self.detector.is_image("test.png"))
        
        # Test für kein Bild
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "PDF document"
            self.assertFalse(self.detector.is_image(self.pdf_path))
    
    def test_is_pdf(self):
        """Test für die is_pdf Methode"""
        # Test für PDF-Datei
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "PDF document, version 1.4"
            self.assertTrue(self.detector.is_pdf(self.pdf_path))
        
        # Test für keine PDF-Datei
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "JPEG image data"
            self.assertFalse(self.detector.is_pdf(self.image_path))
    
    def test_get_processing_priority(self):
        """Test für die get_processing_priority Methode"""
        # Test für Bild (höchste Priorität)
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "JPEG image data"
            priority = self.detector.get_processing_priority(self.image_path)
            self.assertEqual(priority, 1)
        
        # Test für PDF (mittlere Priorität)
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "PDF document, version 1.4"
            priority = self.detector.get_processing_priority(self.pdf_path)
            self.assertEqual(priority, 2)
        
        # Test für Office-Dokument (niedrigere Priorität)
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "Microsoft Word 2007+"
            priority = self.detector.get_processing_priority(self.word_path)
            self.assertEqual(priority, 3)
        
        # Test für nicht unterstützten Dateityp (niedrigste Priorität)
        with patch('models_app.ocr.utils.document_type_detector.magic') as mock_magic:
            mock_magic.from_file.return_value = "data"
            priority = self.detector.get_processing_priority("test.bin")
            self.assertEqual(priority, 4)
    
    def test_get_supported_document_types(self):
        """Test für die get_supported_document_types Methode"""
        # Funktion aufrufen
        supported_types = self.detector.get_supported_document_types()
        
        # Überprüfen der unterstützten Dateitypen
        self.assertIsInstance(supported_types, list)
        self.assertIn("pdf", supported_types)
        self.assertIn("image", supported_types)
        self.assertIn("office", supported_types)
        self.assertIn("text", supported_types)
    
    def test_get_document_type_info(self):
        """Test für die get_document_type_info Methode"""
        # Test für PDF
        pdf_info = self.detector.get_document_type_info("pdf")
        self.assertEqual(pdf_info["type"], "pdf")
        self.assertEqual(pdf_info["description"], "PDF Document")
        self.assertIn("mime_types", pdf_info)
        self.assertIn("extensions", pdf_info)
        
        # Test für Bild
        image_info = self.detector.get_document_type_info("image")
        self.assertEqual(image_info["type"], "image")
        self.assertEqual(image_info["description"], "Image File")
        self.assertIn("mime_types", image_info)
        self.assertIn("extensions", image_info)
        
        # Test für Office-Dokument
        office_info = self.detector.get_document_type_info("office")
        self.assertEqual(office_info["type"], "office")
        self.assertEqual(office_info["description"], "Microsoft Office Document")
        self.assertIn("mime_types", office_info)
        self.assertIn("extensions", office_info)
        
        # Test für nicht unterstützten Dateityp
        with self.assertRaises(KeyError):
            self.detector.get_document_type_info("unsupported_type")

if __name__ == '__main__':
    unittest.main() 