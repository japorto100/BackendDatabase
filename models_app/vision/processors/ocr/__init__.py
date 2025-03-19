"""
OCR-Modul für die Texterkennung in Bildern und Dokumenten.
"""

from models_app.ocr.base_adapter import BaseOCRAdapter, OCRAdapter
from models_app.ocr.utils import (
    config_manager, format_converter, document_type_detector,
    registry, register_adapter, discover_adapters,
    get_adapter_instance, get_adapter_by_capability,
    list_available_adapters, get_adapter_info
)

# Initialisiere das Plugin-System
from models_app.ocr.utils.plugin_system import initialize_plugin_system
available_adapters = initialize_plugin_system()

from .base_adapter import OCRAdapter, BaseOCRAdapter
from .ocr_model_selector import OCRModelSelector
from .paddle_adapter import PaddleOCRAdapter
from .tesseract_adapter import TesseractAdapter
from .easyocr_adapter import EasyOCRAdapter
from .doctr_adapter import DocTRAdapter
from .microsoft_adapter import MicrosoftReadAdapter
from .nougat_adapter import NougatAdapter
from .layoutlmv3_adapter import LayoutLMv3Adapter
from .table_extraction_adapter import TableExtractionAdapter
from .donut_adapter import DonutAdapter
from .formula_recognition_adapter import FormulaRecognitionAdapter 