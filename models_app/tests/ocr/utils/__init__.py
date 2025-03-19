"""
Tests für die OCR-Utility-Module.
"""

# Hier könnten gemeinsame Test-Fixtures oder Hilfsfunktionen definiert werden 

from .test_image_processing import load_image, enhance_image_for_ocr
from .test_error_handler import handle_ocr_errors, OCRError
from .test_result_formatter import create_standard_result
from .test_document_analyzer import analyze_document_structure
from .test_document_type_detector import detect_document_type
from .test_metadata_extractor import extract_document_metadata 