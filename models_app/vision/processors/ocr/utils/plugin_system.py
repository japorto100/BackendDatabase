"""
Plugin-System für OCR-Adapter.

Dieses Modul implementiert ein flexibles Plugin-System für OCR-Adapter,
das die dynamische Registrierung, Entdeckung und Verwendung verschiedener
OCR-Implementierungen ermöglicht.
"""

import os
import importlib
import inspect
import logging
import sys
from typing import Dict, Any, List, Type, Optional, Callable

logger = logging.getLogger(__name__)

class AdapterRegistry:
    """Registry für OCR-Adapter."""
    
    def __init__(self):
        """Initialisiert die Registry."""
        self.adapters = {}
        self.adapter_info = {}
    
    def register(self, adapter_class, name=None, info=None):
        """
        Registriert einen OCR-Adapter.
        
        Args:
            adapter_class: Klasse des Adapters
            name: Name des Adapters (optional, Standard ist der Klassenname)
            info: Zusätzliche Informationen zum Adapter (optional)
            
        Returns:
            Die registrierte Adapter-Klasse
        """
        if name is None:
            name = adapter_class.__name__
            
        if name in self.adapters:
            logger.warning(f"Adapter '{name}' wird überschrieben.")
            
        self.adapters[name] = adapter_class
        self.adapter_info[name] = info or {}
        
        logger.debug(f"Adapter '{name}' registriert.")
        return adapter_class
    
    def unregister(self, name):
        """
        Entfernt einen OCR-Adapter aus der Registry.
        
        Args:
            name: Name des Adapters
            
        Returns:
            True, wenn erfolgreich, sonst False
        """
        if name in self.adapters:
            del self.adapters[name]
            del self.adapter_info[name]
            logger.debug(f"Adapter '{name}' entfernt.")
            return True
        else:
            logger.warning(f"Adapter '{name}' nicht gefunden.")
            return False
    
    def get(self, name):
        """
        Gibt einen OCR-Adapter aus der Registry zurück.
        
        Args:
            name: Name des Adapters
            
        Returns:
            Adapter-Klasse oder None, wenn nicht gefunden
        """
        return self.adapters.get(name)
    
    def get_info(self, name):
        """
        Gibt Informationen zu einem OCR-Adapter zurück.
        
        Args:
            name: Name des Adapters
            
        Returns:
            Adapter-Informationen oder None, wenn nicht gefunden
        """
        return self.adapter_info.get(name)
    
    def list_adapters(self):
        """
        Gibt eine Liste aller registrierten Adapter zurück.
        
        Returns:
            Liste von Adapter-Namen
        """
        return list(self.adapters.keys())
    
    def get_all_info(self):
        """
        Gibt Informationen zu allen registrierten Adaptern zurück.
        
        Returns:
            Dictionary mit Adapter-Informationen
        """
        return self.adapter_info.copy()
    
    def create_instance(self, name, config=None):
        """
        Erstellt eine Instanz eines registrierten Adapters.
        
        Args:
            name: Name des Adapters
            config: Konfiguration für den Adapter (optional)
            
        Returns:
            Instanz des Adapters oder None, wenn nicht gefunden
        """
        adapter_class = self.get(name)
        if adapter_class is None:
            logger.error(f"Adapter '{name}' nicht gefunden.")
            return None
            
        try:
            return adapter_class(config=config or {})
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Adapters '{name}': {str(e)}")
            return None
    
    def get_adapter_by_capability(self, capability, min_priority=0):
        """
        Findet Adapter mit einer bestimmten Fähigkeit.
        
        Args:
            capability: Gesuchte Fähigkeit (z.B. 'handwriting', 'table_extraction')
            min_priority: Minimale Priorität (0-100)
            
        Returns:
            Liste von Adapter-Namen, die die Fähigkeit unterstützen
        """
        matching_adapters = []
        
        for name, info in self.adapter_info.items():
            capabilities = info.get('capabilities', {})
            priority = info.get('priority', 0)
            
            if capabilities.get(capability, False) and priority >= min_priority:
                matching_adapters.append(name)
                
        # Nach Priorität sortieren (höchste zuerst)
        matching_adapters.sort(
            key=lambda name: self.adapter_info[name].get('priority', 0),
            reverse=True
        )
                
        return matching_adapters

# Globale Registry-Instanz
registry = AdapterRegistry()

def register_adapter(name=None, info=None):
    """
    Dekorator zum Registrieren eines OCR-Adapters.
    
    Args:
        name: Name des Adapters (optional)
        info: Zusätzliche Informationen zum Adapter (optional)
        
    Returns:
        Dekorator-Funktion
    """
    def decorator(adapter_class):
        return registry.register(adapter_class, name, info)
    return decorator

def discover_adapters(package_path=None, base_class=None):
    """
    Entdeckt und registriert automatisch OCR-Adapter in einem Paket.
    
    Args:
        package_path: Pfad zum Paket (optional, Standard ist das aktuelle Paket)
        base_class: Basisklasse für Adapter (optional)
        
    Returns:
        Liste der entdeckten Adapter-Namen
    """
    if package_path is None:
        # Standardmäßig im OCR-Paket suchen
        package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if base_class is None:
        # Versuche, die BaseOCRAdapter-Klasse zu importieren
        try:
            from models_app.ocr.base_adapter import BaseOCRAdapter
            base_class = BaseOCRAdapter
        except ImportError:
            logger.error("BaseOCRAdapter konnte nicht importiert werden.")
            return []
    
    discovered = []
    
    # Alle Python-Dateien im Paket durchsuchen
    for root, dirs, files in os.walk(package_path):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                # Relativen Modulpfad berechnen
                rel_path = os.path.relpath(os.path.join(root, file), os.path.dirname(package_path))
                module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
                
                # Modul importieren
                try:
                    module_name = f"models_app.{module_path}"
                    module = importlib.import_module(module_name)
                    
                    # Alle Klassen im Modul durchsuchen
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Prüfen, ob die Klasse von der Basisklasse erbt und nicht die Basisklasse selbst ist
                        if (issubclass(obj, base_class) and 
                            obj != base_class and 
                            obj.__module__ == module.__name__):
                            
                            # Adapter-Informationen extrahieren
                            info = getattr(obj, 'ADAPTER_INFO', {})
                            
                            # Adapter registrieren
                            registry.register(obj, name=getattr(obj, 'ADAPTER_NAME', name), info=info)
                            discovered.append(name)
                            logger.info(f"Adapter '{name}' aus Modul '{module_name}' entdeckt und registriert.")
                            
                except (ImportError, AttributeError) as e:
                    logger.debug(f"Fehler beim Importieren von Modul '{module_path}': {str(e)}")
    
    return discovered

def get_adapter_instance(name, config=None):
    """
    Erstellt eine Instanz eines registrierten Adapters.
    
    Args:
        name: Name des Adapters
        config: Konfiguration für den Adapter (optional)
        
    Returns:
        Instanz des Adapters oder None, wenn nicht gefunden
    """
    return registry.create_instance(name, config)

def get_adapter_by_capability(capability, min_priority=0):
    """
    Findet Adapter mit einer bestimmten Fähigkeit.
    
    Args:
        capability: Gesuchte Fähigkeit (z.B. 'handwriting', 'table_extraction')
        min_priority: Minimale Priorität (0-100)
        
    Returns:
        Liste von Adapter-Namen, die die Fähigkeit unterstützen
    """
    return registry.get_adapter_by_capability(capability, min_priority)

def list_available_adapters():
    """
    Gibt eine Liste aller verfügbaren Adapter zurück.
    
    Returns:
        Liste von Adapter-Namen
    """
    return registry.list_adapters()

def get_adapter_info(name=None):
    """
    Gibt Informationen zu einem oder allen Adaptern zurück.
    
    Args:
        name: Name des Adapters (optional, wenn None werden alle Informationen zurückgegeben)
        
    Returns:
        Adapter-Informationen
    """
    if name is None:
        return registry.get_all_info()
    else:
        return registry.get_info(name)

def register_adapter_hook(adapter_name, hook_name, hook_function):
    """
    Registriert einen Hook für einen Adapter.
    
    Args:
        adapter_name: Name des Adapters
        hook_name: Name des Hooks (z.B. 'pre_process', 'post_process')
        hook_function: Hook-Funktion
        
    Returns:
        True, wenn erfolgreich, sonst False
    """
    adapter_class = registry.get(adapter_name)
    if adapter_class is None:
        logger.error(f"Adapter '{adapter_name}' nicht gefunden.")
        return False
        
    if not hasattr(adapter_class, 'HOOKS'):
        adapter_class.HOOKS = {}
        
    if hook_name not in adapter_class.HOOKS:
        adapter_class.HOOKS[hook_name] = []
        
    adapter_class.HOOKS[hook_name].append(hook_function)
    logger.debug(f"Hook '{hook_name}' für Adapter '{adapter_name}' registriert.")
    
    return True

def initialize_plugin_system():
    """
    Initialisiert das Plugin-System und entdeckt alle verfügbaren Adapter.
    
    Returns:
        Liste der entdeckten Adapter-Namen
    """
    logger.info("Initialisiere OCR-Adapter Plugin-System...")
    discovered = discover_adapters()
    logger.info(f"{len(discovered)} OCR-Adapter entdeckt: {', '.join(discovered)}")
    return discovered 