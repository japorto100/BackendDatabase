"""Zentrale Logging-Konfiguration für Vision-Module."""

import logging
import os
from typing import Dict, Any, Optional

def configure_logging(log_level: str = "INFO", 
                     log_file: Optional[str] = None,
                     module_config: Optional[Dict[str, str]] = None) -> None:
    """
    Konfiguriert das Logging für alle Vision-Module.
    
    Args:
        log_level: Logging-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional. Pfad zur Log-Datei
        module_config: Optional. Spezifische Logging-Level für einzelne Module
    """
    level = getattr(logging, log_level.upper())
    
    handlers = []
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    handlers.append(console_handler)
    
    # File Handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)
    
    # Root Logger für vision konfigurieren
    logger = logging.getLogger('models_app.vision')
    logger.setLevel(level)
    
    # Alte Handler entfernen und neue hinzufügen
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    for handler in handlers:
        logger.addHandler(handler)
    
    # Modul-spezifische Logging-Level (optional)
    if module_config:
        for module, module_level in module_config.items():
            module_logger = logging.getLogger(f'models_app.vision.{module}')
            module_logger.setLevel(getattr(logging, module_level.upper()))
    
    logger.info(f"Logging configured with level {log_level}") 