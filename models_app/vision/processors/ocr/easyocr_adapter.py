"""
EasyOCR-Adapter

Adapter für die Verwendung von EasyOCR mit der standardisierten OCR-Schnittstelle.
EasyOCR ist eine benutzerfreundliche OCR-Bibliothek mit Unterstützung für über 80 Sprachen,
einschließlich Deutsch.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import numpy as np
from PIL import Image
import cv2

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

from .base_adapter import BaseOCRAdapter
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, ModelNotAvailableError
from models_app.vision.ocr.utils.plugin_system import register_adapter
from models_app.vision.utils.image_processing.enhancement import denoise_image, binarize_image
from models_app.vision.utils.image_processing.transformation import resize_image, convert_to_array
from analytics_app.utils import monitor_ocr_performance
from models_app.vision.utils.testing.dummy_models import DummyModelFactory
from models_app.vision.utils.image_processing.adapter_preprocess import preprocess_for_easyocr

logger = logging.getLogger(__name__)

@register_adapter(name="easyocr", info={
    "description": "EasyOCR Engine for multilingual OCR",
    "version": "1.7.0",
    "capabilities": {
        "multi_language": True,
        "handwriting": True,
        "table_extraction": False,
        "formula_recognition": False,
        "document_understanding": False
    },
    "priority": 60
})
class EasyOCRAdapter(BaseOCRAdapter):
    """
    Adapter für EasyOCR.
    """
    
    ADAPTER_NAME = "easyocr"
    ADAPTER_INFO = {
        "description": "EasyOCR Engine for multilingual OCR",
        "version": "1.7.0",
        "capabilities": {
            "multi_language": True,
            "handwriting": True,
            "table_extraction": False,
            "formula_recognition": False,
            "document_understanding": False
        },
        "priority": 60
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den EasyOCR-Adapter.
        
        Args:
            config: Konfiguration für EasyOCR
                - lang: Liste der Sprachen für OCR (default: ['en', 'de'])
                - gpu: GPU verwenden (default: False)
                - detector: Detektor-Modell (default: True)
                - recognizer: Erkenner-Modell (default: True)
                - verbose: Ausführliche Ausgabe (default: False)
                - model_storage_directory: Verzeichnis für Modelle (default: None)
                - download_enabled: Modelle bei Bedarf herunterladen (default: True)
        """
        super().__init__(config or {})
        self.config = {
            'lang': ['en', 'de'],
            'gpu': False,
            'detector': True,
            'recognizer': True,
            'verbose': False,
            'model_storage_directory': None,
            'download_enabled': True,
            **config
        } if config else {
            'lang': ['en', 'de'],
            'gpu': False,
            'detector': True,
            'recognizer': True,
            'verbose': False,
            'model_storage_directory': None,
            'download_enabled': True
        }
        
        # Stelle sicher, dass lang eine Liste ist
        if isinstance(self.config['lang'], str):
            self.config['lang'] = [self.config['lang']]
        
        self.languages = self.config['lang']
        self.gpu = self.config['gpu']
        self.detector = self.config['detector']
        self.recognizer = self.config['recognizer']
        self.verbose = self.config['verbose']
        self.model_storage_directory = self.config['model_storage_directory']
        self.download_enabled = self.config['download_enabled']
        
        self.model = None
        self.is_initialized = False
        
        # Liste aller unterstützten Sprachen in EasyOCR
        self.supported_languages = [
            'abq', 'ady', 'af', 'ang', 'ar', 'as', 'ava', 'az', 'be', 'bg', 'bh', 'bho', 
            'bn', 'bs', 'ch', 'che', 'cs', 'cy', 'da', 'dar', 'de', 'en', 'es', 'et', 
            'fa', 'fr', 'ga', 'gom', 'hi', 'hr', 'hu', 'id', 'inh', 'is', 'it', 'ja', 
            'ka', 'kbd', 'kn', 'ko', 'ku', 'la', 'lbe', 'lez', 'lt', 'lv', 'mah', 'mai', 
            'mi', 'mn', 'mr', 'ms', 'mt', 'ne', 'new', 'nl', 'no', 'oc', 'pi', 'pl', 
            'pt', 'ro', 'ru', 'rs_cyrillic', 'rs_latin', 'sck', 'sk', 'sl', 'sq', 'sv', 
            'sw', 'ta', 'tab', 'te', 'th', 'tjk', 'tl', 'tr', 'ug', 'uk', 'ur', 'uz', 'vi'
        ]
    
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert das EasyOCR-Modell.
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if not EASYOCR_AVAILABLE:
            raise ModelNotAvailableError("EasyOCR ist nicht installiert.")
        
        try:
            # Prüfe, ob alle angegebenen Sprachen unterstützt werden
            for lang in self.languages:
                if lang not in self.supported_languages:
                    logger.warning(f"Sprache '{lang}' wird von EasyOCR nicht unterstützt")
            
            # Initialisiere das Modell mit den ausgewählten Sprachen
            logger.info(f"Initialisiere EasyOCR für Sprachen: {self.languages}")
            self.model = easyocr.Reader(
                lang_list=self.languages,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=self.download_enabled,
                detector=self.detector,
                recognizer=self.recognizer,
                verbose=self.verbose
            )
            
            self.is_initialized = True
            logger.info(f"EasyOCR erfolgreich initialisiert für Sprachen: {self.languages}")
            return True
            
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von EasyOCR: {str(e)}")
            
            # Erstelle ein Dummy-Modell für Tests
            self._initialize_dummy_model()
            return False
    
    def _initialize_dummy_model(self):
        """Initialisiert ein Dummy-Modell für EasyOCR."""
        self.model = DummyModelFactory.create_ocr_dummy("easyocr")
    
    @monitor_ocr_performance
    @handle_ocr_errors
    def process_image(self, image_path_or_array: Union[str, np.ndarray], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet ein Bild mit EasyOCR.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
                - detail: Detaillierte Ergebnisse zurückgeben (default: 1)
                - paragraph: Text in Absätze gruppieren (default: False)
                - min_size: Minimale Textgröße
                - contrast_ths: Kontrast-Schwellenwert
                - adjust_contrast: Kontrast anpassen
                - text_threshold: Text-Schwellenwert
                - link_threshold: Link-Schwellenwert
                - low_text: Niedriger Text-Schwellenwert
                - canvas_size: Canvas-Größe
                - mag_ratio: Vergrößerungsverhältnis
        
        Returns:
            Dict mit OCR-Ergebnissen:
                - text: Erkannter Text
                - blocks: Liste von Textblöcken mit Position und Konfidenz
                - confidence: Gesamtkonfidenz
                - language: Verwendete Sprache
                - model: Name des verwendeten Modells
        """
        if not self.is_initialized:
            success = self.initialize()
            if not success:
                return {"error": "EasyOCR konnte nicht initialisiert werden"}
        
        try:
            # Optionen verarbeiten
            options = options or {}
            detail = options.get("detail", 1)
            paragraph = options.get("paragraph", False)
            
            # Weitere Optionen für readtext extrahieren
            readtext_options = {}
            for key in ['min_size', 'contrast_ths', 'adjust_contrast', 'text_threshold',
                       'link_threshold', 'low_text', 'canvas_size', 'mag_ratio',
                       'slope_ths', 'ycenter_ths', 'height_ths', 'width_ths',
                       'add_margin', 'threshold', 'bbox_min_score', 'bbox_min_size',
                       'max_candidates', 'unclip_ratio']:
                if key in options:
                    readtext_options[key] = options[key]
            
            # Prüfen, ob das Bild bereits vorverarbeitet wurde
            already_preprocessed = options.get('already_preprocessed', False)
            preprocess = options.get('preprocess', True) and not already_preprocessed
            
            # Bildquelle vorbereiten
            if isinstance(image_path_or_array, str):
                if not os.path.exists(image_path_or_array):
                    return {"error": f"Bilddatei nicht gefunden: {image_path_or_array}"}
                
                # Für Pfade: Bild laden und ggf. vorverarbeiten
                if preprocess:
                    from models_app.vision.utils.image_processing.core import load_image, save_image
                    import tempfile
                    
                    # Bild laden
                    _, np_image, _ = load_image(image_path_or_array)
                    
                    # Vorverarbeitung anwenden
                    processed_image = self.preprocess_image(np_image, options)
                    
                    # Temporäre Datei für das vorverarbeitete Bild erstellen
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        save_image(processed_image, temp_file.name)
                        image_source = temp_file.name
                else:
                    # Original-Bilddatei ohne Vorverarbeitung verwenden
                    image_source = image_path_or_array
            else:
                # Für NumPy-Arrays
                if preprocess:
                    # Vorverarbeitung anwenden
                    processed_image = self.preprocess_image(image_path_or_array, options)
                    image_source = processed_image
                else:
                    # Original-Array ohne Vorverarbeitung verwenden
                    image_source = image_path_or_array
            
            # OCR durchführen
            ocr_result = self.model.readtext(image_source, detail=detail, paragraph=paragraph, **readtext_options)
            
            # Temporäre Datei löschen, wenn sie erstellt wurde
            if preprocess and isinstance(image_path_or_array, str) and 'temp_file' in locals():
                try:
                    os.unlink(image_source)
                except Exception as e:
                    logger.warning(f"Konnte temporäre Datei nicht löschen: {str(e)}")
            
            # Ergebnisse standardisieren
            standardized_result = self._standardize_result(ocr_result)
            
            return standardized_result
            
        except Exception as e:
            logger.error(f"Fehler bei der Bildverarbeitung mit EasyOCR: {str(e)}")
            return {
                "error": f"Verarbeitungsfehler: {str(e)}",
                "text": "",
                "blocks": [],
                "confidence": 0.0,
                "language": ", ".join(self.languages),
                "model": "easyocr"
            }
    
    def _standardize_result(self, easyocr_result) -> Dict[str, Any]:
        """
        Standardisiert die EasyOCR-Ergebnisse.
        
        Args:
            easyocr_result: Ergebnisse von EasyOCR
            
        Returns:
            Dict: Standardisierte Ergebnisse
        """
        try:
            texts = []
            boxes = []
            confidences = []
            
            # Extrahiere die erkannten Texte und ihre Positionen
            for item in easyocr_result:
                box, text, conf = item
                
                # In EasyOCR sind die Bounding Boxes als 4 Punkte (links oben, rechts oben, 
                # rechts unten, links unten) im Format [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] angegeben
                # Konvertieren zu [x1, y1, x3, y3] (links oben, rechts unten)
                x_min = min(point[0] for point in box)
                y_min = min(point[1] for point in box)
                x_max = max(point[0] for point in box)
                y_max = max(point[1] for point in box)
                
                texts.append(text)
                boxes.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'polygon': box  # Original-Polygon beibehalten
                })
                confidences.append(float(conf))
            
            # Volltextextraktion
            full_text = " ".join(texts)
            
            # Standardisiertes Ergebnis erstellen
            result = {
                "text": full_text,
                "blocks": [
                    {
                        "text": text,
                        "bbox": box['bbox'],
                        "polygon": box['polygon'],
                        "confidence": conf
                    }
                    for text, box, conf in zip(texts, boxes, confidences)
                ],
                "model": "easyocr",
                "language": ", ".join(self.languages),
                "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
                "page_count": 1  # EasyOCR verarbeitet standardmäßig nur eine Seite
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei der Standardisierung der EasyOCR-Ergebnisse: {str(e)}")
            return {
                "error": f"Standardisierungsfehler: {str(e)}",
                "text": "",
                "blocks": [],
                "confidence": 0.0,
                "language": ", ".join(self.languages),
                "model": "easyocr"
            }
    
    def get_supported_languages(self) -> List[str]:
        """
        Gibt die unterstützten Sprachen für EasyOCR zurück.
        
        Returns:
            List[str]: Liste von Sprachcodes
        """
        return self.supported_languages
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über das Modell zurück.
        
        Returns:
            Dict mit Modellinformationen
        """
        info = super().get_model_info()
        
        if self.is_initialized:
            info.update({
                "type": "EasyOCR",
                "languages": self.languages,
                "supported_languages": self.get_supported_languages(),
                "gpu": self.gpu,
                "detector": self.detector,
                "recognizer": self.recognizer
            })
                
        return info
    
    def preprocess_image(self, image_path_or_array: Union[str, np.ndarray], options: Dict[str, Any] = None) -> np.ndarray:
        """
        Vorverarbeitung des Bildes für EasyOCR.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Optionen für die Vorverarbeitung
            
        Returns:
            np.ndarray: Vorverarbeitetes Bild
        """
        options = options or {}
        # Adapter-spezifische Optionen hinzufügen
        options["enhance_contrast"] = self.config.get("enhance_contrast", True)
        options["denoise"] = self.config.get("denoise", True)
        options["denoise_strength"] = self.config.get("denoise_strength", 5)
        
        # Zentralisierte Vorverarbeitung verwenden
        return preprocess_for_easyocr(image_path_or_array, options) 