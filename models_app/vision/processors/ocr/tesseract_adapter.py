"""
Adapter für Tesseract OCR.
"""

import os
import logging
import numpy as np
import cv2
from typing import Dict, List, Union, Any, Optional
from PIL import Image

from .base_adapter import BaseOCRAdapter
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, ModelNotAvailableError
from models_app.vision.ocr.utils.plugin_system import register_adapter
from models_app.vision.utils.image_processing.enhancement import enhance_for_tesseract
from models_app.vision.utils.image_processing.core import load_image, convert_to_array
from models_app.vision.utils.testing.dummy_models import DummyModelFactory
from models_app.vision.utils.image_processing.adapter_preprocess import preprocess_for_tesseract

# Versuche, pytesseract zu importieren
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

logger = logging.getLogger(__name__)

@register_adapter(name="tesseract", info={
    "description": "Tesseract OCR Engine",
    "version": "4.0.0",
    "capabilities": {
        "multi_language": True,
        "handwriting": False,
        "table_extraction": False,
        "formula_recognition": False,
        "document_understanding": False
    },
    "priority": 50
})
class TesseractAdapter(BaseOCRAdapter):
    """Adapter für Tesseract OCR."""
    
    ADAPTER_NAME = "tesseract"
    ADAPTER_INFO = {
        "description": "Tesseract OCR Engine",
        "version": "4.0.0",
        "capabilities": {
            "multi_language": True,
            "handwriting": False,
            "table_extraction": False,
            "formula_recognition": False,
            "document_understanding": False
        },
        "priority": 50
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den Tesseract-Adapter.
        
        Args:
            config: Konfiguration für Tesseract
        """
        super().__init__(config)
        
        # Standardkonfiguration
        self.lang = self.config.get('lang', 'eng')
        self.config_str = self.config.get('config', '')
        self.path = self.config.get('path', None)
        
        # Tesseract-Pfad setzen, falls angegeben
        if self.path:
            pytesseract.pytesseract.tesseract_cmd = self.path
    
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert Tesseract OCR.
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if not TESSERACT_AVAILABLE:
            raise ModelNotAvailableError("Tesseract (pytesseract) ist nicht installiert.")
            
        try:
            # Tesseract-Version prüfen
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR Version: {version}")
            
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von Tesseract: {str(e)}")
            self.is_initialized = False
            return False
    
    @handle_ocr_errors
    def process_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet ein Bild mit Tesseract OCR.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
            
        Returns:
            Dictionary mit OCR-Ergebnissen
        """
        options = options or {}
        
        # Sicherstellen, dass Tesseract initialisiert ist
        if not self.is_initialized:
            self.initialize()
            
        # Optionen extrahieren
        lang = options.get('lang', self.lang)
        config_str = options.get('config', self.config_str)
        
        # Prüfen, ob das Bild bereits vorverarbeitet wurde
        already_preprocessed = options.get('already_preprocessed', False)
        preprocess = options.get('preprocess', True) and not already_preprocessed
        
        # Bild laden
        from models_app.vision.utils.image_processing.core import load_image
        _, np_image, _ = load_image(image_path_or_array)
        
        # Bild vorverarbeiten
        if preprocess:
            processed_image = self.preprocess_image(np_image, options)
        else:
            processed_image = np_image
        
        # OCR durchführen
        text = pytesseract.image_to_string(processed_image, lang=lang, config=config_str)
        
        # Detaillierte Informationen extrahieren
        data = pytesseract.image_to_data(processed_image, lang=lang, config=config_str, output_type=pytesseract.Output.DICT)
        
        # Textblöcke erstellen
        blocks = []
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                block = {
                    'text': data['text'][i],
                    'conf': float(data['conf'][i]) / 100.0 if data['conf'][i] != -1 else 0.0,
                    'bbox': [
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    ],
                    'line_num': data['line_num'][i],
                    'block_num': data['block_num'][i],
                    'page_num': data['page_num'][i]
                }
                blocks.append(block)
        
        # Durchschnittliche Konfidenz berechnen
        confidences = [block['conf'] for block in blocks if block['conf'] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Ergebnis erstellen
        result = self._create_result(
            text=text,
            blocks=blocks,
            confidence=avg_confidence,
            language=lang,
            metadata={
                'config': config_str,
                'tesseract_version': pytesseract.get_tesseract_version()
            }
        )
        
        # Post-Processing
        result = self.postprocess_result(result, options)
        
        return result
    
    def get_supported_languages(self) -> List[str]:
        """
        Gibt die unterstützten Sprachen zurück.
        
        Returns:
            Liste der unterstützten Sprachen
        """
        if not TESSERACT_AVAILABLE:
            return []
            
        try:
            return pytesseract.get_languages()
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der unterstützten Sprachen: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen zum Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        info = super().get_model_info()
        
        if TESSERACT_AVAILABLE:
            try:
                info.update({
                    "type": "Tesseract OCR",
                    "version": str(pytesseract.get_tesseract_version()),
                    "supported_languages": self.get_supported_languages()
                })
            except Exception as e:
                logger.error(f"Fehler beim Abrufen der Modellinformationen: {str(e)}")
                
        return info
    def _initialize_dummy_model(self):
        """Initialisiert ein Dummy-Modell für Tesseract."""
        from localgpt_vision_django.models_app.vision.utils.testing.dummy_models import DummyModelFactory
        
        # Verwende das Tesseract-spezifische Dummy-Modell
        self.pytesseract = DummyModelFactory._create_tesseract_dummy()

    def preprocess_image(self, image_path_or_array, options: Dict[str, Any] = None) -> np.ndarray:
        """
        Vorverarbeitung des Bildes für Tesseract OCR.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Optionen für die Vorverarbeitung
            
        Returns:
            np.ndarray: Vorverarbeitetes Bild
        """
        options = options or {}
        # Adapter-spezifische Optionen hinzufügen
        options["dpi"] = self.config.get("dpi", 300)
        options["threshold_method"] = self.config.get("threshold_method", "adaptive")
        
        # Zentralisierte Vorverarbeitung verwenden
        return preprocess_for_tesseract(image_path_or_array, options)