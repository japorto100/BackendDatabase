"""
Adapter für Microsoft Azure Computer Vision Read API.
"""

import os
import time
import logging
import tempfile
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image
import io
import cv2
import json

from .base_adapter import BaseOCRAdapter
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, ModelNotAvailableError, OCRError
from models_app.vision.ocr.utils.plugin_system import register_adapter
from models_app.vision.utils.image_processing.core import load_image, convert_to_array, save_image
from models_app.vision.utils.image_processing.enhancement import enhance_for_microsoft
from analytics_app.utils import monitor_ocr_performance
from models_app.vision.utils.testing.dummy_models import DummyModelFactory
from models_app.vision.utils.image_processing.adapter_preprocess import preprocess_for_microsoft

# Versuche, Azure-SDK zu importieren
try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
    from msrest.authentication import CognitiveServicesCredentials
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    ComputerVisionClient = None
    OperationStatusCodes = None
    CognitiveServicesCredentials = None

logger = logging.getLogger(__name__)

@register_adapter(name="microsoft_read", info={
    "description": "Microsoft Azure Computer Vision Read API",
    "version": "3.1",
    "capabilities": {
        "multi_language": True,
        "handwriting": True,
        "table_extraction": True,
        "formula_recognition": False,
        "document_understanding": True
    },
    "priority": 90
})
class MicrosoftReadAdapter(BaseOCRAdapter):
    """Adapter für Microsoft Azure Computer Vision Read API."""
    
    ADAPTER_NAME = "microsoft_read"
    ADAPTER_INFO = {
        "description": "Microsoft Azure Computer Vision Read API",
        "version": "3.1",
        "capabilities": {
            "multi_language": True,
            "handwriting": True,
            "table_extraction": True,
            "formula_recognition": False,
            "document_understanding": True
        },
        "priority": 90
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den Microsoft Read-Adapter.
        
        Args:
            config: Konfiguration für Azure Computer Vision
        """
        super().__init__(config)
        
        # Standardkonfiguration
        self.api_key = self.config.get('api_key', os.environ.get('AZURE_VISION_API_KEY', ''))
        self.endpoint = self.config.get('endpoint', os.environ.get('AZURE_VISION_ENDPOINT', ''))
        self.language = self.config.get('language', 'auto')
        self.timeout = self.config.get('timeout', 30)
        self.poll_interval = self.config.get('poll_interval', 1.0)
        self.read_version = self.config.get('read_version', '3.2')
        self.model_version = self.config.get('model_version', 'latest')
        
        self.model = None
    
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert die Microsoft Read API.
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if not AZURE_AVAILABLE:
            raise ModelNotAvailableError("Microsoft Azure SDK ist nicht installiert.")
            
        if not self.api_key or not self.endpoint:
            raise ModelNotAvailableError(
                "API-Schlüssel oder Endpunkt für Microsoft Azure fehlt. "
                "Bitte in der Konfiguration oder als Umgebungsvariablen angeben."
            )
            
        try:
            # Azure Computer Vision Client initialisieren
            self.model = ComputerVisionClient(
                endpoint=self.endpoint,
                credentials=CognitiveServicesCredentials(self.api_key)
            )
            
            logger.info("Microsoft Azure Computer Vision Read API initialisiert.")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von Microsoft Read API: {str(e)}")
            self.is_initialized = False
            return False
    def _initialize_dummy_model(self):
        """Initialisiert ein Dummy-Modell für Microsoft Read API."""
        self.model = DummyModelFactory.create_ocr_dummy("microsoft")
    
    @monitor_ocr_performance
    @handle_ocr_errors
    def process_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet ein Bild mit Microsoft Read API.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
            
        Returns:
            Dictionary mit OCR-Ergebnissen
        """
        options = options or {}
        
        # Prüfen, ob das Bild bereits vorverarbeitet wurde
        already_preprocessed = options.get('already_preprocessed', False)
        preprocess = options.get('preprocess', True) and not already_preprocessed
        
        # Handle input as file path or array
        if isinstance(image_path_or_array, str) and os.path.isfile(image_path_or_array):
            # File path case
            if preprocess:
                # Load, preprocess and save to temp file
                from models_app.vision.utils.image_processing.core import load_image, save_image
                
                _, np_image, _ = load_image(image_path_or_array)
                processed_image = self.preprocess_image(np_image, options)
                
                # Save to temp file for API upload
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                save_image(processed_image, temp_file.name)
                
                with open(temp_file.name, "rb") as image_stream:
                    # Process with Microsoft Read API
                    read_response = self.model.read_in_stream(image_stream)
                    # Rest of code...
                
                # Clean up temp file
                os.unlink(temp_file.name)
            else:
                # Use original file
                with open(image_path_or_array, "rb") as image_stream:
                    # Process with Microsoft Read API
                    read_response = self.model.read_in_stream(image_stream)
        else:
            # Array or PIL image case
            from models_app.vision.utils.image_processing.core import load_image
            _, np_image, _ = load_image(image_path_or_array)
            
            if preprocess:
                processed_image = self.preprocess_image(np_image, options)
                np_image = processed_image
            
            # Convert to bytes for API
            _, buffer = cv2.imencode('.jpg', np_image)
            image_bytes = io.BytesIO(buffer)
            
            # Process with Microsoft Read API
            read_response = self.model.read_in_stream(image_bytes)
        
        # Ergebnis extrahieren
        blocks = []
        full_text = []
        total_confidence = 0.0
        num_words = 0
        # Verarbeite die Microsoft Read-Ergebnisse
        for read_result in read_response.analyze_result.read_results:
            for line in read_result.lines:
                line_text = line.text
                full_text.append(line_text)
                
                # Bounding box extrahieren (Format: [x1,y1,x2,y1,x2,y2,x1,y2])
                bbox = self._convert_polygon_to_bbox(line.bounding_box)
                
                # Konfidenz für die Linie berechnen (Durchschnitt der Wortkonfidenzen)
                line_confidence = 0.0
                line_words = []
                
                for word in line.words:
                    word_text = word.text
                    word_confidence = word.confidence
                    word_bbox = self._convert_polygon_to_bbox(word.bounding_box)
                    
                    line_words.append({
                        'text': word_text,
                        'conf': word_confidence,
                        'bbox': word_bbox,
                        'polygon': word.bounding_box
                    })
                    
                    line_confidence += word_confidence
                    total_confidence += word_confidence
                    num_words += 1
                
                avg_line_confidence = line_confidence / len(line.words) if line.words else 0.0
                
                # Block für die Linie erstellen
                block = {
                    'text': line_text,
                    'conf': avg_line_confidence,
                    'bbox': bbox,
                    'polygon': line.bounding_box,
                    'words': line_words,
                    'page': read_result.page
                }
                
                blocks.append(block)
        
        # Durchschnittliche Konfidenz berechnen
        avg_confidence = total_confidence / num_words if num_words > 0 else 0.0
        
        # Ergebnis erstellen
        result_dict = self._create_result(
            text="\n".join(full_text),
            blocks=blocks,
            confidence=avg_confidence,
            language=self.language,
            metadata={
                'api_version': self.read_version,
                'model_version': self.model_version,
                'operation_id': read_response.headers["Operation-Location"].split("/")[-1]
            }
        )
        
        # Post-Processing
        result_dict = self.postprocess_result(result_dict, options)
        
        return result_dict

    def _convert_polygon_to_bbox(self, polygon):
        """
        Konvertiert ein Polygon (8 Werte) in eine Bounding Box (4 Werte).
        
        Args:
            polygon: Liste von 8 Werten [x1,y1,x2,y1,x2,y2,x1,y2]
            
        Returns:
            [x1, y1, x2, y2] Bounding Box
        """
        if len(polygon) >= 8:
            x_coords = [polygon[i] for i in range(0, len(polygon), 2)]
            y_coords = [polygon[i+1] for i in range(0, len(polygon), 2)]
            
            return [
                min(x_coords),  # x1
                min(y_coords),  # y1
                max(x_coords),  # x2
                max(y_coords)   # y2
            ]
        else:
            # Fallback, wenn das Polygon nicht das erwartete Format hat
            return [0, 0, 0, 0]
    
    def get_supported_languages(self) -> List[str]:
        """
        Gibt die unterstützten Sprachen zurück.
        
        Returns:
            Liste der unterstützten Sprachen
        """
        # Microsoft Read API unterstützt viele Sprachen
        return [
            'auto', 'ar', 'bg', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et',
            'fi', 'fr', 'ga', 'gl', 'hr', 'hu', 'id', 'is', 'it', 'ja', 'ko', 'lt', 'lv',
            'mt', 'nb', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sv', 'th', 'tr',
            'uk', 'vi', 'zh-Hans', 'zh-Hant'
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen zum Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        info = super().get_model_info()
        
        if self.is_initialized:
            info.update({
                "type": "Microsoft Azure Computer Vision Read API",
                "api_version": self.read_version,
                "model_version": self.model_version,
                "language": self.language,
                "supported_languages": self.get_supported_languages(),
                "is_cloud_service": True
            })
                
        return info

    def preprocess_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Any:
        """
        Vorverarbeitung des Bildes für Microsoft Read API.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Optionen für die Vorverarbeitung
            
        Returns:
            Any: Vorverarbeitetes Bild im Format, das von der API erwartet wird
        """
        options = options or {}
        # Microsoft-spezifische Optionen
        options["denoise"] = self.config.get("denoise", True)
        options["enhance_contrast"] = self.config.get("enhance_contrast", True)
        options["return_pil"] = True  # Microsoft API erwartet oft ein PIL-Bild
        
        # Zentralisierte Vorverarbeitung verwenden
        return preprocess_for_microsoft(image_path_or_array, options)