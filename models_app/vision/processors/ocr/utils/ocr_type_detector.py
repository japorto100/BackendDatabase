"""
Funktionen zur Erkennung und Klassifizierung von Dokumenttypen.
"""

import os
import logging
import re
import magic
import mimetypes
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional, Union

from models_app.vision.utils.image_processing.unified_detector import UnifiedDetector
from models_app.vision.utils.image_processing.core import convert_to_array

logger = logging.getLogger(__name__)

class DocumentTypeDetector:
    """Klasse für die Erkennung und Klassifizierung von Dokumenttypen."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den DocumentTypeDetector.
        
        Args:
            config: Konfiguration für den Detektor
        """
        self.config = config or {}
        
        # MIME-Typen initialisieren
        mimetypes.init()
        
        # UnifiedDetector für Bildanalyse initialisieren
        self.unified_detector = UnifiedDetector()
        
        # Bekannte Dokumenttypen und ihre Eigenschaften
        self.document_types = {
            "pdf": {
                "mime_types": ["application/pdf"],
                "extensions": [".pdf"],
                "description": "PDF-Dokument",
                "category": "document",
                "processing_priority": 1
            },
            "word": {
                "mime_types": [
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "application/vnd.oasis.opendocument.text"
                ],
                "extensions": [".doc", ".docx", ".odt"],
                "description": "Word-Dokument",
                "category": "document",
                "processing_priority": 2
            },
            "excel": {
                "mime_types": [
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/vnd.oasis.opendocument.spreadsheet"
                ],
                "extensions": [".xls", ".xlsx", ".ods"],
                "description": "Excel-Tabelle",
                "category": "spreadsheet",
                "processing_priority": 3
            },
            "powerpoint": {
                "mime_types": [
                    "application/vnd.ms-powerpoint",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    "application/vnd.oasis.opendocument.presentation"
                ],
                "extensions": [".ppt", ".pptx", ".odp"],
                "description": "PowerPoint-Präsentation",
                "category": "presentation",
                "processing_priority": 4
            },
            "image": {
                "mime_types": [
                    "image/jpeg", "image/png", "image/gif", "image/tiff",
                    "image/bmp", "image/webp"
                ],
                "extensions": [".jpg", ".jpeg", ".png", ".gif", ".tiff", ".tif", ".bmp", ".webp"],
                "description": "Bilddatei",
                "category": "image",
                "processing_priority": 5
            },
            "text": {
                "mime_types": [
                    "text/plain", "text/markdown", "text/csv"
                ],
                "extensions": [".txt", ".md", ".csv"],
                "description": "Textdatei",
                "category": "text",
                "processing_priority": 6
            },
            "html": {
                "mime_types": [
                    "text/html", "application/xhtml+xml"
                ],
                "extensions": [".html", ".htm", ".xhtml"],
                "description": "HTML-Dokument",
                "category": "web",
                "processing_priority": 7
            },
            "email": {
                "mime_types": [
                    "message/rfc822", "application/vnd.ms-outlook"
                ],
                "extensions": [".eml", ".msg"],
                "description": "E-Mail",
                "category": "email",
                "processing_priority": 8
            },
            "archive": {
                "mime_types": [
                    "application/zip", "application/x-rar-compressed",
                    "application/x-tar", "application/gzip"
                ],
                "extensions": [".zip", ".rar", ".tar", ".gz", ".7z"],
                "description": "Archiv",
                "category": "archive",
                "processing_priority": 9
            },
            "unknown": {
                "mime_types": [],
                "extensions": [],
                "description": "Unbekannter Dokumenttyp",
                "category": "unknown",
                "processing_priority": 100
            }
        }
    
    def detect_document_type(self, file_path: str) -> Dict[str, Any]:
        """
        Erkennt den Dokumenttyp einer Datei.
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            Dictionary mit Informationen zum Dokumenttyp
        """
        if not os.path.exists(file_path):
            logger.error(f"Datei nicht gefunden: {file_path}")
            return self._get_unknown_type()
            
        try:
            # MIME-Typ mit python-magic bestimmen
            mime_type = magic.from_file(file_path, mime=True)
            
            # Dateiendung
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            # Dokumenttyp anhand von MIME-Typ und Dateiendung bestimmen
            doc_type = self._get_document_type(mime_type, ext)
            
            # Zusätzliche Informationen
            doc_info = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "mime_type": mime_type,
                "extension": ext,
                "last_modified": os.path.getmtime(file_path)
            }
            
            # Bei Bilddateien: Erweiterte Analyse mit UnifiedDetector
            if doc_type["type"] == "image":
                try:
                    image_properties = self.analyze_image_properties(file_path)
                    doc_info.update(image_properties)
                except Exception as e:
                    logger.warning(f"Fehler bei der Bildanalyse: {str(e)}")
            
            # Dokumenttyp-Informationen mit zusätzlichen Informationen zusammenführen
            result = {**doc_type, **doc_info}
            
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei der Dokumenttyperkennung: {str(e)}")
            return self._get_unknown_type(file_path=file_path)
    
    def analyze_image_properties(self, image_path: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Analysiert die Eigenschaften eines Bildes für OCR-Zwecke.
        
        Args:
            image_path: Pfad zum Bild oder Bilddaten
            
        Returns:
            Dictionary mit Bildeigenschaften
        """
        # UnifiedDetector für Basisanalysen verwenden
        img_array = convert_to_array(image_path) if isinstance(image_path, (str, Image.Image)) else image_path
        
        # Dokumenteigenschaften vom UnifiedDetector abrufen
        doc_properties = self.unified_detector.analyze_document_properties(img_array)
        
        # Text- und Tabellenregionen erkennen
        text_regions = self.unified_detector.detect_text(img_array)
        tables = self.unified_detector.detect_tables(img_array)
        
        # OCR-spezifische Eigenschaften berechnen
        ocr_properties = {
            "text_density": len(text_regions) / (img_array.shape[0] * img_array.shape[1]) * 1000000 if text_regions else 0,
            "has_tables": len(tables) > 0,
            "table_count": len(tables),
            "layout_complexity": doc_properties.get("complexity", 0),
            "has_handwriting": doc_properties.get("handwriting", 0) > 0.5,
            "is_multilingual": doc_properties.get("multilingual", 0) > 0.5,
            "has_equations": doc_properties.get("equations", 0) > 0.5,
            "is_academic": doc_properties.get("academic", 0) > 0.5
        }
        
        # OCR-Schwierigkeitsgrad basierend auf Eigenschaften berechnen
        difficulty_score = (
            0.3 * doc_properties.get("complexity", 0) +
            0.2 * doc_properties.get("handwriting", 0) +
            0.2 * (1 if len(tables) > 0 else 0) +
            0.15 * doc_properties.get("equations", 0) +
            0.15 * doc_properties.get("multilingual", 0)
        )
        
        ocr_properties["ocr_difficulty"] = min(1.0, difficulty_score)
        
        return ocr_properties
    
    def _get_document_type(self, mime_type: str, extension: str) -> Dict[str, Any]:
        """
        Bestimmt den Dokumenttyp anhand von MIME-Typ und Dateiendung.
        
        Args:
            mime_type: MIME-Typ der Datei
            extension: Dateiendung
            
        Returns:
            Dictionary mit Informationen zum Dokumenttyp
        """
        # Zuerst nach MIME-Typ suchen
        for doc_type, properties in self.document_types.items():
            if mime_type in properties["mime_types"]:
                result = properties.copy()
                result["type"] = doc_type
                result["confidence"] = 0.9  # Hohe Konfidenz bei MIME-Typ-Übereinstimmung
                return result
                
        # Dann nach Dateiendung suchen
        for doc_type, properties in self.document_types.items():
            if extension in properties["extensions"]:
                result = properties.copy()
                result["type"] = doc_type
                result["confidence"] = 0.7  # Mittlere Konfidenz bei Dateiendungs-Übereinstimmung
                return result
                
        # Unbekannter Typ
        return self._get_unknown_type()
    
    def _get_unknown_type(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Gibt Informationen für einen unbekannten Dokumenttyp zurück.
        
        Args:
            file_path: Pfad zur Datei (optional)
            
        Returns:
            Dictionary mit Informationen zum unbekannten Dokumenttyp
        """
        result = self.document_types["unknown"].copy()
        result["type"] = "unknown"
        result["confidence"] = 0.0
        
        if file_path:
            result["file_path"] = file_path
            if os.path.exists(file_path):
                result["file_size"] = os.path.getsize(file_path)
                result["last_modified"] = os.path.getmtime(file_path)
                _, ext = os.path.splitext(file_path)
                result["extension"] = ext.lower()
                
                try:
                    result["mime_type"] = magic.from_file(file_path, mime=True)
                except:
                    result["mime_type"] = "application/octet-stream"
            
        return result
    
    def is_office_document(self, file_path: str) -> bool:
        """
        Prüft, ob eine Datei ein Office-Dokument ist.
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            True, wenn es sich um ein Office-Dokument handelt, sonst False
        """
        doc_type = self.detect_document_type(file_path)
        return doc_type["type"] in ["word", "excel", "powerpoint"]
    
    def is_image(self, file_path: str) -> bool:
        """
        Prüft, ob eine Datei ein Bild ist.
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            True, wenn es sich um ein Bild handelt, sonst False
        """
        doc_type = self.detect_document_type(file_path)
        return doc_type["type"] == "image"
    
    def is_pdf(self, file_path: str) -> bool:
        """
        Prüft, ob eine Datei ein PDF-Dokument ist.
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            True, wenn es sich um ein PDF-Dokument handelt, sonst False
        """
        doc_type = self.detect_document_type(file_path)
        return doc_type["type"] == "pdf"
    
    def get_processing_priority(self, file_path: str) -> int:
        """
        Gibt die Verarbeitungspriorität für eine Datei zurück.
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            Verarbeitungspriorität (niedrigere Werte = höhere Priorität)
        """
        doc_type = self.detect_document_type(file_path)
        return doc_type.get("processing_priority", 100)
    
    def get_supported_document_types(self) -> List[str]:
        """
        Gibt eine Liste der unterstützten Dokumenttypen zurück.
        
        Returns:
            Liste der unterstützten Dokumenttypen
        """
        return [doc_type for doc_type in self.document_types.keys() if doc_type != "unknown"]
    
    def get_document_type_info(self, doc_type: str) -> Dict[str, Any]:
        """
        Gibt Informationen zu einem bestimmten Dokumenttyp zurück.
        
        Args:
            doc_type: Dokumenttyp
            
        Returns:
            Dictionary mit Informationen zum Dokumenttyp
        """
        if doc_type in self.document_types:
            result = self.document_types[doc_type].copy()
            result["type"] = doc_type
            return result
        else:
            return self._get_unknown_type()