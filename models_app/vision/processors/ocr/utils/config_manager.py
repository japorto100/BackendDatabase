"""
Einheitliches Konfigurationsmanagement für OCR-Adapter.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manager für OCR-Adapter-Konfigurationen."""
    
    def __init__(self, config_dir=None):
        """
        Initialisiert den ConfigManager.
        
        Args:
            config_dir: Verzeichnis für Konfigurationsdateien
        """
        self.config_dir = config_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs"
        )
        
        # Verzeichnis erstellen, falls es nicht existiert
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Cache für geladene Konfigurationen
        self.config_cache = {}
        
    def get_default_config(self, adapter_name: str) -> Dict[str, Any]:
        """
        Gibt die Standardkonfiguration für einen Adapter zurück.
        
        Args:
            adapter_name: Name des Adapters
            
        Returns:
            Standardkonfiguration als Dictionary
        """
        # Standardkonfigurationen für verschiedene Adapter
        default_configs = {
            "tesseract": {
                "lang": "eng",
                "config": "",
                "path": None,
                "preprocess": True
            },
            "easyocr": {
                "lang": ["en"],
                "gpu": False,
                "verbose": False,
                "preprocess": True
            },
            "paddle": {
                "lang": "en",
                "use_gpu": False,
                "enable_mkldnn": True,
                "preprocess": True,
                "handwriting_mode": False
            },
            "doctr": {
                "det_arch": "db_resnet50",
                "reco_arch": "crnn_vgg16_bn",
                "pretrained": True,
                "assume_straight_pages": True,
                "straighten_pages": True,
                "gpu": False
            },
            "microsoft": {
                "api_key": os.environ.get("AZURE_VISION_KEY", ""),
                "endpoint": os.environ.get("AZURE_VISION_ENDPOINT", ""),
                "language": "en",
                "model_version": "latest"
            },
            "layoutlmv3": {
                "model_name": "microsoft/layoutlmv3-base",
                "task": "document_understanding",
                "max_length": 512,
                "gpu": False
            },
            "nougat": {
                "model_name": "facebook/nougat-base",
                "max_length": 4096,
                "gpu": False,
                "preprocess": True
            },
            "donut": {
                "model_name": "naver-clova-ix/donut-base-finetuned-cord-v2",
                "task_prompt": "parse this document",
                "max_length": 1024,
                "output_format": "json",
                "gpu": False
            },
            "formula_recognition": {
                "engine": "pix2tex",
                "model_name": "pix2tex/pix2tex-base",
                "max_length": 512,
                "gpu": False,
                "confidence_threshold": 0.7,
                "preprocess": True
            },
            "table_extraction": {
                "table_engine": "paddle",
                "use_gpu": False,
                "lang": "en",
                "output_format": "csv"
            }
        }
        
        # Fallback für unbekannte Adapter
        if adapter_name not in default_configs:
            logger.warning(f"Keine Standardkonfiguration für Adapter '{adapter_name}' gefunden. Verwende leere Konfiguration.")
            return {}
            
        return default_configs[adapter_name].copy()
    
    def load_config(self, adapter_name: str) -> Dict[str, Any]:
        """
        Lädt die Konfiguration für einen Adapter.
        
        Args:
            adapter_name: Name des Adapters
            
        Returns:
            Konfiguration als Dictionary
        """
        # Prüfen, ob die Konfiguration bereits im Cache ist
        if adapter_name in self.config_cache:
            return self.config_cache[adapter_name].copy()
            
        # Konfigurationsdatei-Pfad
        config_file = os.path.join(self.config_dir, f"{adapter_name}.json")
        
        # Standardkonfiguration
        config = self.get_default_config(adapter_name)
        
        # Konfigurationsdatei laden, falls vorhanden
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    
                # Konfigurationen zusammenführen
                config.update(file_config)
                logger.debug(f"Konfiguration für '{adapter_name}' aus Datei geladen.")
            except Exception as e:
                logger.error(f"Fehler beim Laden der Konfiguration für '{adapter_name}': {str(e)}")
        
        # Konfiguration im Cache speichern
        self.config_cache[adapter_name] = config.copy()
        
        return config
    
    def save_config(self, adapter_name: str, config: Dict[str, Any]) -> bool:
        """
        Speichert die Konfiguration für einen Adapter.
        
        Args:
            adapter_name: Name des Adapters
            config: Konfiguration als Dictionary
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        try:
            # Konfigurationsdatei-Pfad
            config_file = os.path.join(self.config_dir, f"{adapter_name}.json")
            
            # Konfiguration speichern
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
                
            # Cache aktualisieren
            self.config_cache[adapter_name] = config.copy()
            
            logger.debug(f"Konfiguration für '{adapter_name}' gespeichert.")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration für '{adapter_name}': {str(e)}")
            return False
    
    def merge_config(self, adapter_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Führt eine neue Konfiguration mit der bestehenden zusammen.
        
        Args:
            adapter_name: Name des Adapters
            config: Neue Konfiguration als Dictionary
            
        Returns:
            Zusammengeführte Konfiguration
        """
        # Bestehende Konfiguration laden
        current_config = self.load_config(adapter_name)
        
        # Neue Konfiguration hinzufügen
        current_config.update(config)
        
        # Aktualisierte Konfiguration speichern
        self.save_config(adapter_name, current_config)
        
        return current_config.copy()
    
    def get_config_value(self, adapter_name: str, key: str, default: Any = None) -> Any:
        """
        Gibt einen bestimmten Wert aus der Konfiguration zurück.
        
        Args:
            adapter_name: Name des Adapters
            key: Schlüssel des Konfigurationswerts
            default: Standardwert, falls der Schlüssel nicht existiert
            
        Returns:
            Konfigurationswert oder Standardwert
        """
        config = self.load_config(adapter_name)
        return config.get(key, default)
    
    def set_config_value(self, adapter_name: str, key: str, value: Any) -> bool:
        """
        Setzt einen bestimmten Wert in der Konfiguration.
        
        Args:
            adapter_name: Name des Adapters
            key: Schlüssel des Konfigurationswerts
            value: Neuer Wert
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        config = self.load_config(adapter_name)
        config[key] = value
        return self.save_config(adapter_name, config)
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Gibt alle verfügbaren Konfigurationen zurück.
        
        Returns:
            Dictionary mit allen Konfigurationen
        """
        configs = {}
        
        # Konfigurationsdateien durchsuchen
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                adapter_name = filename[:-5]  # Entferne '.json'
                configs[adapter_name] = self.load_config(adapter_name)
                
        return configs
    
    def reset_config(self, adapter_name: str) -> bool:
        """
        Setzt die Konfiguration eines Adapters auf die Standardwerte zurück.
        
        Args:
            adapter_name: Name des Adapters
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        default_config = self.get_default_config(adapter_name)
        return self.save_config(adapter_name, default_config) 