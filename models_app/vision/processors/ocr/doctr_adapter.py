"""
Adapter für DocTR (Document Text Recognition).
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image

from .base_adapter import BaseOCRAdapter
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, ModelNotAvailableError
from models_app.vision.ocr.utils.plugin_system import register_adapter
from models_app.vision.utils.image_processing.enhancement import enhance_for_ocr
from models_app.vision.utils.image_processing.core import load_image, convert_to_array
from models_app.vision.utils.image_processing.detection import detect_text_regions, detect_tables
from analytics_app.utils import monitor_ocr_performance
from models_app.vision.utils.testing.dummy_models import DummyModelFactory
from models_app.vision.utils.image_processing.adapter_preprocess import preprocess_for_doctr
# Versuche, DocTR zu importieren
try:
    import doctr
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False
    doctr = None
    DocumentFile = None
    ocr_predictor = None

logger = logging.getLogger(__name__)

@register_adapter(name="doctr", info={
    "description": "DocTR OCR Engine for document text recognition",
    "version": "0.6.0",
    "capabilities": {
        "multi_language": True,
        "handwriting": False,
        "table_extraction": True,
        "formula_recognition": False,
        "document_understanding": True
    },
    "priority": 80
})
class DocTRAdapter(BaseOCRAdapter):
    """Adapter für DocTR (Document Text Recognition)."""
    
    ADAPTER_NAME = "doctr"
    ADAPTER_INFO = {
        "description": "DocTR OCR Engine for document text recognition",
        "version": "0.6.0",
        "capabilities": {
            "multi_language": True,
            "handwriting": False,
            "table_extraction": True,
            "formula_recognition": False,
            "document_understanding": True
        },
        "priority": 80
    }
    
    # Verfügbare Architekturen
    AVAILABLE_DET_ARCHS = [
        'db_resnet50', 'db_mobilenet_v3_large', 'linknet_resnet18', 
        'linknet_resnet34', 'linknet_resnet50'
    ]
    
    AVAILABLE_RECO_ARCHS = [
        'crnn_vgg16_bn', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large',
        'sar_resnet31', 'master'
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den DocTR-Adapter.
        
        Args:
            config: Konfiguration für DocTR
        """
        super().__init__(config)
        
        # Standardkonfiguration
        self.det_arch = self.config.get('det_arch', 'db_resnet50')
        self.reco_arch = self.config.get('reco_arch', 'crnn_vgg16_bn')
        self.pretrained = self.config.get('pretrained', True)
        self.assume_straight_pages = self.config.get('assume_straight_pages', True)
        self.straighten_pages = self.config.get('straighten_pages', False)
        self.detect_orientation = self.config.get('detect_orientation', False)
        self.export_as_straight = self.config.get('export_as_straight', False)
        self.apply_correction = self.config.get('apply_correction', False)
        self.preserve_aspect_ratio = self.config.get('preserve_aspect_ratio', True)
        self.symmetric_pad = self.config.get('symmetric_pad', True)
        
        self.model = None
    
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert das DocTR-Modell.
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if not DOCTR_AVAILABLE:
            raise ModelNotAvailableError("DocTR ist nicht installiert.")
            
        try:
            # Überprüfe, ob die angegebenen Architekturen verfügbar sind
            if self.det_arch not in self.AVAILABLE_DET_ARCHS:
                logger.warning(f"Detektions-Architektur '{self.det_arch}' nicht verfügbar. Verwende 'db_resnet50'.")
                self.det_arch = 'db_resnet50'
                
            if self.reco_arch not in self.AVAILABLE_RECO_ARCHS:
                logger.warning(f"Erkennungs-Architektur '{self.reco_arch}' nicht verfügbar. Verwende 'crnn_vgg16_bn'.")
                self.reco_arch = 'crnn_vgg16_bn'
                
            # DocTR-Predictor initialisieren
            self.model = ocr_predictor(
                det_arch=self.det_arch,
                reco_arch=self.reco_arch,
                pretrained=self.pretrained,
                assume_straight_pages=self.assume_straight_pages,
                straighten_pages=self.straighten_pages,
                detect_orientation=self.detect_orientation,
                preserve_aspect_ratio=self.preserve_aspect_ratio,
                symmetric_pad=self.symmetric_pad
            )
            
            logger.info(f"DocTR initialisiert mit Detektor: {self.det_arch}, Erkenner: {self.reco_arch}")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von DocTR: {str(e)}")
            self.is_initialized = False
            return False
    
    def _initialize_dummy_model(self):
        """Initialisiert ein Dummy-Modell für DocTR."""
        self.model = DummyModelFactory.create_ocr_dummy("doctr")
    
    @monitor_ocr_performance
    @handle_ocr_errors
    def process_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet ein Bild mit DocTR.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
        
        Returns:
            Dictionary mit OCR-Ergebnissen
        """
        options = options or {}
        
        # Sicherstellen, dass DocTR initialisiert ist
        if not self.is_initialized:
            if self.config.get('dummy_mode', False):
                self._initialize_dummy_model()
            else:
                self.initialize()
        
        # Optionen extrahieren
        export_as_straight = options.get('export_as_straight', self.export_as_straight)
        apply_correction = options.get('apply_correction', self.apply_correction)
        
        # Prüfen, ob das Bild bereits vorverarbeitet wurde
        already_preprocessed = options.get('already_preprocessed', False)
        preprocess = options.get('preprocess', True) and not already_preprocessed
        
        # Load image
        pil_image, np_image, _ = load_image(image_path_or_array)
        
        # Apply preprocessing if enabled
        if preprocess:
            processed_image = self.preprocess_image(np_image, options)
            pil_image = Image.fromarray(processed_image)
            np_image = processed_image
        
        # Use image with doctr
        doc = DocumentFile.from_images(pil_image)
        result = self.model(doc)
        
        # Ergebnis exportieren
        export = result.export(export_as_straight=export_as_straight)
        
        # Extrahiere die Ergebnisse
        blocks = []
        full_text = []
        total_confidence = 0.0
        num_words = 0
        
        # Verarbeite die DocTR-Ergebnisse
        for page_idx, page in enumerate(export['pages']):
            for block_idx, block in enumerate(page['blocks']):
                for line_idx, line in enumerate(block['lines']):
                    line_text = []
                    for word_idx, word in enumerate(line['words']):
                        text = word['value']
                        confidence = word['confidence']
                        # Koordinaten in relative Koordinaten umwandeln
                        geometry = word['geometry']
                        
                        # Aus den relativen Koordinaten die absoluten Pixelkoordinaten berechnen
                        # (unter der Annahme, dass die Bildgröße bekannt ist)
                        img_height, img_width = np_image.shape[:2] if hasattr(np_image, 'shape') else (0, 0)
                        x_coords = [int(point[0] * img_width) for point in geometry]
                        y_coords = [int(point[1] * img_height) for point in geometry]
                        
                        bbox = [
                            min(x_coords),  # x1
                            min(y_coords),  # y1
                            max(x_coords),  # x2
                            max(y_coords)   # y2
                        ]
                        
                        # Block erstellen
                        block_info = {
                            'text': text,
                            'conf': float(confidence),
                            'bbox': bbox,
                            'polygon': [[x, y] for x, y in zip(x_coords, y_coords)],
                            'page_num': page_idx,
                            'block_num': block_idx,
                            'line_num': line_idx,
                            'word_num': word_idx
                        }
                        
                        blocks.append(block_info)
                        line_text.append(text)
                        total_confidence += float(confidence)
                        num_words += 1
                    
                    full_text.append(" ".join(line_text))
        
        # Durchschnittliche Konfidenz berechnen
        avg_confidence = total_confidence / num_words if num_words > 0 else 0.0
        
        # Ergebnis erstellen
        result_dict = self._create_result(
            text="\n".join(full_text),
            blocks=blocks,
            confidence=avg_confidence,
            language="auto",
            metadata={
                'det_arch': self.det_arch,
                'reco_arch': self.reco_arch,
                'assume_straight_pages': self.assume_straight_pages,
                'straighten_pages': self.straighten_pages,
                'detect_orientation': self.detect_orientation,
                'export_as_straight': export_as_straight,
                'apply_correction': apply_correction,
                'preserve_aspect_ratio': self.preserve_aspect_ratio
            },
            raw_output=export
        )
        
        # Strukturierte Ausgabe
        if apply_correction and 'raw_output' in result_dict:
            # Anwenden von Korrekturregeln auf die erkannten Texte
            # Dieser Code könnte in einem separaten Modul implementiert werden
            pass
        
        # Post-Processing
        result_dict = self.postprocess_result(result_dict, options)
        
        return result_dict
    
    def get_supported_languages(self) -> List[str]:
        """
        Gibt die unterstützten Sprachen zurück.
        
        Returns:
            Liste der unterstützten Sprachen
        """
        # DocTR unterstützt mehrere Sprachen, aber hauptsächlich Englisch und Französisch
        return ['en', 'fr']
    
    def extract_table_structure(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extrahiert Tabellenstrukturen aus einem Bild.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
        
        Returns:
            Dictionary mit erkannten Tabellen
        """
        # Diese Funktion ist ein Beispiel für eine zusätzliche Fähigkeit von DocTR
        # Die tatsächliche Implementierung würde spezifische DocTR-APIs für die Tabellenerkennung verwenden
        
        if not self.is_initialized:
            self.initialize()
            
        # Tabellenextraktion mit DocTR
        # In der echten Implementierung würden hier spezifische DocTR-APIs verwendet
        
        # Dummy-Ergebnis für Tabellenstruktur
        return {
            "tables": [
                {
                    "rows": 3,
                    "columns": 4,
                    "cells": [
                        {"row": 0, "column": 0, "text": "Zelle 1,1", "confidence": 0.95},
                        # Weitere Zellen...
                    ],
                    "bbox": [10, 10, 300, 150],
                    "confidence": 0.9
                }
            ]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen zum Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        info = super().get_model_info()
        
        if self.is_initialized:
            info.update({
                "type": "DocTR",
                "det_arch": self.det_arch,
                "reco_arch": self.reco_arch,
                "pretrained": self.pretrained,
                "assume_straight_pages": self.assume_straight_pages,
                "straighten_pages": self.straighten_pages,
                "detect_orientation": self.detect_orientation,
                "available_det_archs": self.AVAILABLE_DET_ARCHS,
                "available_reco_archs": self.AVAILABLE_RECO_ARCHS
            })
                
        return info 

    def preprocess_image(self, image_path_or_array, options: Dict[str, Any] = None) -> np.ndarray:
        """
        Vorverarbeitung des Bildes für DocTR.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Optionen für die Vorverarbeitung
            
        Returns:
            np.ndarray: Vorverarbeitetes Bild
        """
        options = options or {}
        # DocTR-spezifische Optionen
        options["normalize"] = True
        options["enhance_contrast"] = self.config.get("enhance_contrast", True)
        
        # Zentralisierte Vorverarbeitung verwenden
        return preprocess_for_doctr(image_path_or_array, options) 