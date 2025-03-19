"""
PaddleOCR-Adapter

Adapter für die Verwendung von PaddleOCR mit der standardisierten OCR-Schnittstelle.
Unterstützt sowohl gedruckten Text als auch Handschrift.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import os
import numpy as np
from PIL import Image
import cv2

from .base_adapter import OCRAdapter
from models_app.ocr.base_adapter import BaseOCRAdapter
from models_app.ocr.utils.error_handler import handle_ocr_errors, ModelNotAvailableError
from models_app.ocr.utils.plugin_system import register_adapter
from models_app.vision.utils.testing.dummy_models import DummyModelFactory
from models_app.vision.utils.image_processing.adapter_preprocess import preprocess_for_paddleocr

# Versuche, PaddleOCR zu importieren
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    PaddleOCR = None

logger = logging.getLogger(__name__)

@register_adapter(name="paddleocr", info={
    "description": "PaddleOCR Engine for multilingual OCR",
    "version": "2.6.0",
    "capabilities": {
        "multi_language": True,
        "handwriting": True,
        "table_extraction": False,
        "formula_recognition": False,
        "document_understanding": True
    },
    "priority": 70
})
class PaddleOCRAdapter(BaseOCRAdapter):
    """
    Adapter für PaddleOCR mit Unterstützung für gedruckten Text und Handschrift.
    """
    
    ADAPTER_NAME = "paddleocr"
    ADAPTER_INFO = {
        "description": "PaddleOCR Engine for multilingual OCR",
        "version": "2.6.0",
        "capabilities": {
            "multi_language": True,
            "handwriting": True,
            "table_extraction": False,
            "formula_recognition": False,
            "document_understanding": True
        },
        "priority": 70
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den PaddleOCR-Adapter.
        
        Args:
            config: Konfiguration für PaddleOCR
        """
        super().__init__(config)
        
        # Standardkonfiguration
        self.use_angle_cls = self.config.get('use_angle_cls', True)
        self.lang = self.config.get('lang', 'de')
        self.use_gpu = self.config.get('use_gpu', False)
        self.show_log = self.config.get('show_log', False)
        self.rec_batch_num = self.config.get('rec_batch_num', 6)
        self.det_db_box_thresh = self.config.get('det_db_box_thresh', 0.5)
        self.det_db_thresh = self.config.get('det_db_thresh', 0.3)
        self.det_db_unclip_ratio = self.config.get('det_db_unclip_ratio', 1.6)
        self.max_batch_size = self.config.get('max_batch_size', 10)
        self.use_mp = self.config.get('use_mp', False)  # Multiprocessing
        self.total_process_num = self.config.get('total_process_num', 1)
        self.drop_score = self.config.get('drop_score', 0.5)
        
        self.model = None
        
        # Unterstützte Sprachen für PaddleOCR
        self.supported_languages = [
            "ch", "en", "fr", "german", "korean", "japan", "chinese_cht", 
            "ta", "te", "ka", "latin", "arabic", "cyrillic", "devanagari"
        ]
        
        # Handschrift-spezifische Konfiguration
        self.is_handwriting_mode = self.config.get("handwriting_mode", False)
        
        # Modell-spezifische Parameter
        self.det_model_dir = self.config.get("det_model_dir", None)
        self.rec_model_dir = self.config.get("rec_model_dir", None)
        self.cls_model_dir = self.config.get("cls_model_dir", None)
        
        # Erkennungsparameter
        self.use_dilation = self.config.get("use_dilation", False)
        
        # Wenn Handschriftmodus aktiviert ist, passe Parameter an
        if self.is_handwriting_mode:
            # Optimierte Parameter für Handschrift
            self.det_db_thresh = self.config.get("det_db_thresh", 0.3)  # Niedrigerer Schwellenwert für blasse Handschrift
            self.det_db_box_thresh = self.config.get("det_db_box_thresh", 0.5)
            self.det_db_unclip_ratio = self.config.get("det_db_unclip_ratio", 1.8)  # Höherer Wert für verbundene Handschrift
            self.use_dilation = self.config.get("use_dilation", True)  # Hilft bei dünnen Strichen
            self.drop_score = self.config.get("drop_score", 0.5)  # Niedrigerer Konfidenzwert für Handschrift
            self.rec_algorithm = self.config.get("rec_algorithm", "SVTR_LCNet")  # Besser für Handschrift
        else:
            # Standard-Parameter für gedruckten Text
            self.rec_algorithm = self.config.get("rec_algorithm", "CRNN")
    
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert das PaddleOCR-Modell.
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if not PADDLE_AVAILABLE:
            raise ModelNotAvailableError("PaddleOCR ist nicht installiert.")
            
        try:
            # PaddleOCR initialisieren
            self.model = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=self.show_log,
                rec_batch_num=self.rec_batch_num,
                det_db_box_thresh=self.det_db_box_thresh,
                det_db_thresh=self.det_db_thresh,
                det_db_unclip_ratio=self.det_db_unclip_ratio,
                max_batch_size=self.max_batch_size,
                use_mp=self.use_mp,
                total_process_num=self.total_process_num,
                drop_score=self.drop_score
            )
            
            logger.info(f"PaddleOCR initialisiert mit Sprache: {self.lang}")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von PaddleOCR: {str(e)}")
            self.is_initialized = False
            return False
    
    def _initialize_dummy_model(self):
        """Initialisiert ein Dummy-Modell für PaddleOCR."""
        self.model = DummyModelFactory.create_ocr_dummy("paddleocr")
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> Union[np.ndarray, Image.Image]:
        """
        Vorverarbeitung des Bildes für PaddleOCR.
        
        Args:
            image: Pfad zum Bild, PIL-Bild oder NumPy-Array
            
        Returns:
            Union[np.ndarray, Image.Image]: Vorverarbeitetes Bild
        """
        options = {
            "enhance_contrast": self.config.get("enhance_contrast", True),
            "sharpen": self.config.get("sharpen", True),
            "clip_limit": self.config.get("clip_limit", 2.0)
        }
        
        # Zentralisierte Vorverarbeitung verwenden
        return preprocess_for_paddleocr(image, options)
    
    def _preprocess_handwriting(self, image: Image.Image) -> np.ndarray:
        """
        Spezielle Vorverarbeitung für Handschrift.
        
        Args:
            image: PIL-Bild
            
        Returns:
            np.ndarray: Vorverarbeitetes Bild für Handschrifterkennung
        """
        # Konvertiere PIL zu NumPy
        img_np = np.array(image)
        
        # Konvertiere zu Graustufen, falls es ein Farbbild ist
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np.copy()
        
        # Adaptive Schwellenwertbildung zur Verbesserung der Handschrift
        # Dies hilft bei unterschiedlichem Stiftdruck und Lichtverhältnissen
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Rauschentfernung
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Zurück zu RGB für PaddleOCR
        processed = cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)
        
        return processed
    
    @handle_ocr_errors
    def process_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet ein Bild mit PaddleOCR.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
            
        Returns:
            Dictionary mit OCR-Ergebnissen
        """
        options = options or {}
        
        # Sicherstellen, dass PaddleOCR initialisiert ist
        if not self.is_initialized:
            if self.config.get('dummy_mode', False):
                self._initialize_dummy_model()
            else:
                self.initialize()
        
        # Optionen extrahieren
        use_angle_cls = options.get('use_angle_cls', self.use_angle_cls)
        cls = options.get('cls', use_angle_cls)  # Kompatibler Parameter
        det = options.get('det', True)  # Text-Detection aktivieren/deaktivieren
        rec = options.get('rec', True)  # Text-Recognition aktivieren/deaktivieren
        
        # Prüfen, ob das Bild bereits vorverarbeitet wurde
        already_preprocessed = options.get('already_preprocessed', False)
        preprocess = options.get('preprocess', True) and not already_preprocessed
        
        # Bild laden
        from models_app.vision.utils.image_processing.core import load_image
        _, image_np, _ = load_image(image_path_or_array)
        
        # Bild vorverarbeiten
        if preprocess:
            processed_image = self.preprocess_image(image_np)
        else:
            processed_image = image_np
        
        # PaddleOCR kann mit Pfaden oder Arrays arbeiten
        if isinstance(image_path_or_array, str) and not preprocess:
            # Wenn kein Preprocessing und ein Pfad gegeben ist, nutze direkt den Pfad
            # (effiziente Methode für PaddleOCR)
            image_to_process = image_path_or_array
        else:
            # Sonst nutze das numpy Array (entweder das Original oder das vorverarbeitete)
            image_to_process = processed_image
        
        # OCR durchführen
        result = self.model(image_to_process, cls=cls, det=det, rec=rec)
        
        # Extrahiere die Ergebnisse
        blocks = []
        full_text = []
        total_confidence = 0.0
        num_blocks = 0
        
        # Verarbeite die PaddleOCR-Ergebnisse
        if result and isinstance(result, list) and len(result) > 0:
            for line in result[0]:
                if len(line) >= 2:
                    coords, (text, confidence) = line
                    
                    # Bounding Box extrahieren
                    if isinstance(coords, list) and len(coords) >= 4:
                        x_coords = [point[0] for point in coords]
                        y_coords = [point[1] for point in coords]
                        
                        bbox = [
                            min(x_coords),  # x1
                            min(y_coords),  # y1
                            max(x_coords),  # x2
                            max(y_coords)   # y2
                        ]
                        
                        block = {
                            'text': text,
                            'conf': float(confidence),
                            'bbox': bbox,
                            'polygon': coords
                        }
                        
                        blocks.append(block)
                        full_text.append(text)
                        total_confidence += float(confidence)
                        num_blocks += 1
        
        # Durchschnittliche Konfidenz berechnen
        avg_confidence = total_confidence / num_blocks if num_blocks > 0 else 0.0
        
        # Ergebnis erstellen
        result = self._create_result(
            text="\n".join(full_text),
            blocks=blocks,
            confidence=avg_confidence,
            language=self.lang,
            metadata={
                'use_angle_cls': use_angle_cls,
                'det': det,
                'rec': rec,
                'rec_batch_num': self.rec_batch_num,
                'det_db_box_thresh': self.det_db_box_thresh,
                'det_db_thresh': self.det_db_thresh,
                'det_db_unclip_ratio': self.det_db_unclip_ratio
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
        # PaddleOCR unterstützt diese Sprachen
        return [
            'ch', 'en', 'fr', 'german', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 
            'latin', 'arabic', 'cyrillic', 'devanagari'
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
                "type": "PaddleOCR",
                "language": self.lang,
                "supported_languages": self.get_supported_languages(),
                "use_gpu": self.use_gpu,
                "use_angle_cls": self.use_angle_cls,
                "det_db_thresh": self.det_db_thresh,
                "det_db_box_thresh": self.det_db_box_thresh,
                "det_db_unclip_ratio": self.det_db_unclip_ratio
            })
                
        return info 