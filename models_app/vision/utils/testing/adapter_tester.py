"""
Einheitliches Framework zum Testen von Adaptern und Prozessoren.
"""
from typing import Dict, Any, List, Tuple, Union, Optional, Type
import numpy as np
import os
import json
import tempfile
from pathlib import Path

class AdapterTester:
    """
    Zentralisierte Klasse zum Testen von Vision-Adaptern und -Prozessoren.
    Bietet Standardtests für verschiedene Adapter-Typen.
    """
    
    def __init__(self, test_data_dir: Optional[str] = None):
        """
        Initialisiert den AdapterTester.
        
        Args:
            test_data_dir: Verzeichnis mit Testdaten
        """
        self.test_data_dir = test_data_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_data"
        )
        self._ensure_test_data()
    
    def _ensure_test_data(self):
        """Stellt sicher, dass Testdaten vorhanden sind."""
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Grundlegende Testdaten erstellen, falls nicht vorhanden
        self._create_test_images()
        self._create_test_documents()
    
    def _create_test_images(self):
        """Erstellt Basis-Testbilder, falls nicht vorhanden."""
        # Implementierung hier...
        pass
    
    def _create_test_documents(self):
        """Erstellt Basis-Testdokumente, falls nicht vorhanden."""
        # Implementierung hier...
        pass
    
    def test_ocr_adapter(self, adapter_instance, test_level: str = "basic") -> Dict[str, Any]:
        """
        Führt Standardtests für OCR-Adapter durch.
        
        Args:
            adapter_instance: Instanz des zu testenden OCR-Adapters
            test_level: Detailgrad der Tests ('basic', 'full', 'performance')
            
        Returns:
            Dict mit Testergebnissen
        """
        results = {
            "adapter_name": getattr(adapter_instance, "ADAPTER_NAME", "unknown"),
            "initialization": False,
            "basic_processing": False,
            "error_handling": False
        }
        
        # Initialisierungstest
        try:
            adapter_instance.initialize()
            results["initialization"] = True
        except Exception as e:
            results["initialization_error"] = str(e)
        
        # Basis-Verarbeitungstest
        if results["initialization"]:
            test_image_path = os.path.join(self.test_data_dir, "basic_text.png")
            try:
                process_result = adapter_instance.process_image(test_image_path)
                results["basic_processing"] = "text" in process_result
                results["process_result"] = process_result
            except Exception as e:
                results["processing_error"] = str(e)
        
        # Fehlerbehandlungstest
        try:
            adapter_instance.process_image("non_existent_file.png")
            results["error_handling"] = False
            results["error_handling_issue"] = "Failed to handle non-existent file"
        except Exception:
            # Erwarteter Fehler
            results["error_handling"] = True
        
        # Erweiterte Tests basierend auf test_level
        if test_level in ["full", "performance"]:
            self._run_advanced_ocr_tests(adapter_instance, results)
        
        if test_level == "performance":
            self._run_performance_tests(adapter_instance, results)
        
        return results
    
    def test_document_adapter(self, adapter_instance, test_level: str = "basic") -> Dict[str, Any]:
        """Führt Standardtests für Dokument-Adapter durch."""
        # Ähnlich wie test_ocr_adapter, aber für Dokumentadapter
        # Implementierung hier...
        pass
    
    # Weitere Testmethoden für verschiedene Adaptertypen...

# Singleton-Instanz für einfachen Zugriff
adapter_tester = AdapterTester() 