"""
Utility-Module für OCR-Adapter.
"""

from models_app.vision.processors.ocr.utils.image_processing import (
    load_image, denoise_image, binarize_image, deskew_image,
    enhance_image_for_ocr, detect_text_regions, crop_image
)

from models_app.vision.processors.ocr.utils.result_formatter import (
    create_standard_result, merge_results, format_as_text,
    format_as_html, format_as_json, format_as_markdown,
    format_table_as_csv, format_table_as_html, format_table_as_json
)

from models_app.vision.processors.ocr.utils.error_handler import (
    OCRError, ModelNotAvailableError, ModelInitializationError,
    ImageProcessingError, UnsupportedFormatError,
    create_error_result, handle_ocr_errors
)

from models_app.ocr.utils.config_manager import ConfigManager
from models_app.ocr.utils.metadata_extractor import extract_document_metadata
from models_app.ocr.utils.plugin_system import (
    registry, register_adapter, discover_adapters,
    get_adapter_instance, get_adapter_by_capability,
    list_available_adapters, get_adapter_info,
    register_adapter_hook, initialize_plugin_system
)

from models_app.ocr.utils.document_analyzer import (
    analyze_document_structure, has_mathematical_formulas,
    check_if_has_tables, get_document_complexity
)

# Nicht exportieren:
# from models_app.ocr.utils.format_converter import FormatConverter
# from models_app.ocr.utils.document_type_detector import DocumentTypeDetector

# Globale Instanzen für einfachen Zugriff
config_manager = ConfigManager()
# format_converter = FormatConverter()  # Nicht exportieren
# document_type_detector = DocumentTypeDetector()  # Nicht exportieren 