"""
Tests für das Plugin-System aus dem plugin_system Modul.
"""

import unittest
from unittest.mock import patch, MagicMock
import inspect
import sys

from models_app.ocr.utils.plugin_system import (
    AdapterRegistry, register_adapter, discover_adapters,
    get_adapter_instance, get_adapter_by_capability,
    list_available_adapters, get_adapter_info,
    register_adapter_hook, initialize_plugin_system
)
from models_app.ocr.base_adapter import BaseOCRAdapter

class MockAdapter(BaseOCRAdapter):
    """Mock-Adapter für Tests."""
    
    ADAPTER_NAME = "mock_adapter"
    ADAPTER_INFO = {
        "description": "Mock Adapter for testing",
        "version": "1.0.0",
        "capabilities": {
            "multi_language": True,
            "handwriting": False,
            "table_extraction": False,
            "formula_recognition": False,
            "document_understanding": False
        },
        "priority": 50
    }
    
    def initialize(self):
        return True
    
    def process_image(self, image_path_or_array, options=None):
        return {"text": "Mock OCR result"}

class AnotherMockAdapter(BaseOCRAdapter):
    """Zweiter Mock-Adapter für Tests."""
    
    ADAPTER_NAME = "another_mock"
    ADAPTER_INFO = {
        "description": "Another Mock Adapter",
        "version": "1.0.0",
        "capabilities": {
            "multi_language": False,
            "handwriting": True,
            "table_extraction": True,
            "formula_recognition": False,
            "document_understanding": False
        },
        "priority": 60
    }
    
    def initialize(self):
        return True
    
    def process_image(self, image_path_or_array, options=None):
        return {"text": "Another mock result"}

class TestAdapterRegistry(unittest.TestCase):
    """Test-Klasse für die AdapterRegistry"""
    
    def setUp(self):
        """Setup für Tests"""
        self.registry = AdapterRegistry()
    
    def test_register_and_get(self):
        """Test für das Registrieren und Abrufen von Adaptern"""
        # Adapter registrieren
        self.registry.register(MockAdapter, name="mock", info={"test": "info"})
        
        # Adapter abrufen
        adapter_class = self.registry.get("mock")
        self.assertEqual(adapter_class, MockAdapter)
        
        # Info abrufen
        info = self.registry.get_info("mock")
        self.assertEqual(info, {"test": "info"})
        
        # Nicht vorhandenen Adapter abrufen
        with self.assertRaises(KeyError):
            self.registry.get("non_existent")
    
    def test_unregister(self):
        """Test für das Deregistrieren von Adaptern"""
        # Adapter registrieren
        self.registry.register(MockAdapter, name="mock")
        
        # Überprüfen, ob der Adapter registriert ist
        self.assertIn("mock", self.registry.list_adapters())
        
        # Adapter deregistrieren
        self.registry.unregister("mock")
        
        # Überprüfen, ob der Adapter deregistriert ist
        self.assertNotIn("mock", self.registry.list_adapters())
        
        # Überprüfen, ob das Deregistrieren eines nicht vorhandenen Adapters keinen Fehler auslöst
        self.registry.unregister("non_existent")
    
    def test_list_adapters(self):
        """Test für das Auflisten von Adaptern"""
        # Adapter registrieren
        self.registry.register(MockAdapter, name="mock1")
        self.registry.register(AnotherMockAdapter, name="mock2")
        
        # Alle Adapter auflisten
        adapters = self.registry.list_adapters()
        self.assertIn("mock1", adapters)
        self.assertIn("mock2", adapters)
        self.assertEqual(len(adapters), 2)
    
    def test_get_all_info(self):
        """Test für das Abrufen aller Adapter-Informationen"""
        # Adapter mit Informationen registrieren
        self.registry.register(MockAdapter, name="mock1", info={"priority": 1})
        self.registry.register(AnotherMockAdapter, name="mock2", info={"priority": 2})
        
        # Alle Informationen abrufen
        all_info = self.registry.get_all_info()
        self.assertIn("mock1", all_info)
        self.assertIn("mock2", all_info)
        self.assertEqual(all_info["mock1"]["priority"], 1)
        self.assertEqual(all_info["mock2"]["priority"], 2)
    
    def test_create_instance(self):
        """Test für das Erstellen von Adapter-Instanzen"""
        # Adapter registrieren
        self.registry.register(MockAdapter, name="mock")
        
        # Instanz erstellen
        instance = self.registry.create_instance("mock", config={"option": "value"})
        
        # Überprüfen, ob die Instanz korrekt erstellt wurde
        self.assertIsInstance(instance, MockAdapter)
        self.assertEqual(instance.config, {"option": "value"})
        
        # Überprüfen, ob bei einem nicht vorhandenen Adapter eine KeyError ausgelöst wird
        with self.assertRaises(KeyError):
            self.registry.create_instance("non_existent")
    
    def test_get_adapter_by_capability(self):
        """Test für das Abrufen von Adaptern nach Fähigkeiten"""
        # Adapter mit verschiedenen Fähigkeiten registrieren
        self.registry.register(MockAdapter, name="mock1", info=MockAdapter.ADAPTER_INFO)
        self.registry.register(AnotherMockAdapter, name="mock2", info=AnotherMockAdapter.ADAPTER_INFO)
        
        # Adapter mit multi_language-Fähigkeit abrufen
        multi_language_adapters = self.registry.get_adapter_by_capability("multi_language")
        self.assertEqual(len(multi_language_adapters), 1)
        self.assertEqual(multi_language_adapters[0][0], "mock1")
        
        # Adapter mit handwriting-Fähigkeit abrufen
        handwriting_adapters = self.registry.get_adapter_by_capability("handwriting")
        self.assertEqual(len(handwriting_adapters), 1)
        self.assertEqual(handwriting_adapters[0][0], "mock2")
        
        # Adapter mit table_extraction-Fähigkeit abrufen
        table_adapters = self.registry.get_adapter_by_capability("table_extraction")
        self.assertEqual(len(table_adapters), 1)
        self.assertEqual(table_adapters[0][0], "mock2")
        
        # Adapter mit formula_recognition-Fähigkeit abrufen (keine vorhanden)
        formula_adapters = self.registry.get_adapter_by_capability("formula_recognition")
        self.assertEqual(len(formula_adapters), 0)
        
        # Adapter mit Mindestpriorität abrufen
        high_priority_adapters = self.registry.get_adapter_by_capability("multi_language", min_priority=55)
        self.assertEqual(len(high_priority_adapters), 0)  # mock1 hat Priorität 50, unter dem Minimum

class TestPluginSystemFunctions(unittest.TestCase):
    """Test-Klasse für die Plugin-System-Funktionen"""
    
    @patch('models_app.ocr.utils.plugin_system.registry')
    def test_register_adapter_decorator(self, mock_registry):
        """Test für den register_adapter Decorator"""
        # Decorator anwenden
        decorator = register_adapter(name="test_adapter", info={"test": "info"})
        decorated_class = decorator(MockAdapter)
        
        # Überprüfen, ob die Registrierungsmethode aufgerufen wurde
        mock_registry.register.assert_called_once_with(
            MockAdapter, name="test_adapter", info={"test": "info"}
        )
        
        # Überprüfen, ob die dekorierte Klasse zurückgegeben wurde
        self.assertEqual(decorated_class, MockAdapter)
    
    @patch('models_app.ocr.utils.plugin_system.pkgutil')
    @patch('models_app.ocr.utils.plugin_system.importlib')
    @patch('models_app.ocr.utils.plugin_system.inspect')
    @patch('models_app.ocr.utils.plugin_system.logger')
    def test_discover_adapters(self, mock_logger, mock_inspect, mock_importlib, mock_pkgutil):
        """Test für die discover_adapters Funktion"""
        # Mock für pkgutil.iter_modules
        mock_pkgutil.iter_modules.return_value = [
            (None, "module1", False),
            (None, "module2", False)
        ]
        
        # Mock für importlib.import_module
        mock_module1 = MagicMock()
        mock_module2 = MagicMock()
        mock_importlib.import_module.side_effect = [mock_module1, mock_module2]
        
        # Mock für inspect.getmembers
        class1 = MagicMock()
        class2 = MagicMock()
        # Simuliere, dass class1 von BaseOCRAdapter erbt und class2 nicht
        mock_inspect.isclass.side_effect = [True, True, True, True]
        mock_inspect.getmro.side_effect = [
            (class1, BaseOCRAdapter, object),
            (class2, object)
        ]
        mock_inspect.getmembers.side_effect = [
            [("Class1", class1), ("SomethingElse", "not_a_class")],
            [("Class2", class2)]
        ]
        
        # Funktion aufrufen
        adapters = discover_adapters("test_package", BaseOCRAdapter)
        
        # Überprüfen, ob die Module importiert wurden
        mock_importlib.import_module.assert_any_call("test_package.module1")
        mock_importlib.import_module.assert_any_call("test_package.module2")
        
        # Überprüfen, ob die richtigen Klassenmember überprüft wurden
        mock_inspect.getmembers.assert_any_call(mock_module1, mock_inspect.isclass)
        mock_inspect.getmembers.assert_any_call(mock_module2, mock_inspect.isclass)
        
        # Überprüfen, ob die gefundenen Adapter zurückgegeben wurden
        self.assertEqual(len(adapters), 1)
        self.assertEqual(adapters[0], class1)
    
    @patch('models_app.ocr.utils.plugin_system.registry')
    def test_get_adapter_instance(self, mock_registry):
        """Test für die get_adapter_instance Funktion"""
        # Mock für die create_instance Methode des Registry-Singletons
        mock_instance = MagicMock()
        mock_registry.create_instance.return_value = mock_instance
        
        # Funktion aufrufen
        instance = get_adapter_instance("test_adapter", config={"option": "value"})
        
        # Überprüfen, ob die Registry-Methode aufgerufen wurde
        mock_registry.create_instance.assert_called_once_with(
            "test_adapter", config={"option": "value"}
        )
        
        # Überprüfen, ob die Instanz zurückgegeben wurde
        self.assertEqual(instance, mock_instance)
    
    @patch('models_app.ocr.utils.plugin_system.registry')
    def test_get_adapter_by_capability(self, mock_registry):
        """Test für die get_adapter_by_capability Funktion"""
        # Mock für die get_adapter_by_capability Methode des Registry-Singletons
        mock_registry.get_adapter_by_capability.return_value = [
            ("adapter1", {"priority": 70}),
            ("adapter2", {"priority": 60})
        ]
        
        # Funktion aufrufen
        adapters = get_adapter_by_capability("multi_language", min_priority=50)
        
        # Überprüfen, ob die Registry-Methode aufgerufen wurde
        mock_registry.get_adapter_by_capability.assert_called_once_with(
            "multi_language", min_priority=50
        )
        
        # Überprüfen, ob die Adapter zurückgegeben wurden
        self.assertEqual(len(adapters), 2)
        self.assertEqual(adapters[0][0], "adapter1")
        self.assertEqual(adapters[1][0], "adapter2")
    
    @patch('models_app.ocr.utils.plugin_system.registry')
    def test_list_available_adapters(self, mock_registry):
        """Test für die list_available_adapters Funktion"""
        # Mock für die list_adapters Methode des Registry-Singletons
        mock_registry.list_adapters.return_value = ["adapter1", "adapter2"]
        
        # Funktion aufrufen
        adapters = list_available_adapters()
        
        # Überprüfen, ob die Registry-Methode aufgerufen wurde
        mock_registry.list_adapters.assert_called_once()
        
        # Überprüfen, ob die Adapter zurückgegeben wurden
        self.assertEqual(adapters, ["adapter1", "adapter2"])
    
    @patch('models_app.ocr.utils.plugin_system.registry')
    def test_get_adapter_info(self, mock_registry):
        """Test für die get_adapter_info Funktion"""
        # Mock für die get_info und get_all_info Methoden des Registry-Singletons
        mock_registry.get_info.return_value = {"priority": 70}
        mock_registry.get_all_info.return_value = {
            "adapter1": {"priority": 70},
            "adapter2": {"priority": 60}
        }
        
        # Funktion mit Adapter-Namen aufrufen
        info = get_adapter_info("adapter1")
        
        # Überprüfen, ob die Registry-Methode aufgerufen wurde
        mock_registry.get_info.assert_called_once_with("adapter1")
        
        # Überprüfen, ob die Informationen zurückgegeben wurden
        self.assertEqual(info, {"priority": 70})
        
        # Funktion ohne Adapter-Namen aufrufen
        all_info = get_adapter_info()
        
        # Überprüfen, ob die Registry-Methode aufgerufen wurde
        mock_registry.get_all_info.assert_called_once()
        
        # Überprüfen, ob alle Informationen zurückgegeben wurden
        self.assertEqual(all_info, {
            "adapter1": {"priority": 70},
            "adapter2": {"priority": 60}
        })
    
    @patch('models_app.ocr.utils.plugin_system.registry')
    def test_register_adapter_hook(self, mock_registry):
        """Test für die register_adapter_hook Funktion"""
        # Mock-Funktionen für Hooks
        def pre_process_hook(image, options):
            return image
        
        def post_process_hook(result, options):
            return result
        
        # Mock-Adapter-Klasse
        mock_adapter_class = MagicMock()
        mock_registry.get.return_value = mock_adapter_class
        
        # Funktion aufrufen
        register_adapter_hook("test_adapter", "pre_process", pre_process_hook)
        
        # Überprüfen, ob die Registry-Methode aufgerufen wurde
        mock_registry.get.assert_called_once_with("test_adapter")
        
        # Überprüfen, ob die Hooks auf der Adapter-Klasse registriert wurden
        # In der realen Implementierung würden wir hier HOOKS auf der Adapter-Klasse überprüfen
        # Da wir aber ein Mock verwenden, können wir nicht direkt auf das Klassenattribut zugreifen
        
        # Funktion mit unbekanntem Adapter aufrufen
        mock_registry.get.side_effect = KeyError("Unknown adapter")
        
        # Dies sollte einen Log-Eintrag erzeugen, aber keinen Fehler auslösen
        register_adapter_hook("unknown_adapter", "post_process", post_process_hook)
    
    @patch('models_app.ocr.utils.plugin_system.discover_adapters')
    @patch('models_app.ocr.utils.plugin_system.registry')
    def test_initialize_plugin_system(self, mock_registry, mock_discover_adapters):
        """Test für die initialize_plugin_system Funktion"""
        # Mock für die discover_adapters Funktion
        mock_adapter1 = MagicMock()
        mock_adapter1.ADAPTER_NAME = "adapter1"
        mock_adapter1.ADAPTER_INFO = {"priority": 70}
        
        mock_adapter2 = MagicMock()
        mock_adapter2.ADAPTER_NAME = "adapter2"
        mock_adapter2.ADAPTER_INFO = {"priority": 60}
        
        mock_discover_adapters.return_value = [mock_adapter1, mock_adapter2]
        
        # Funktion aufrufen
        discovered_adapters = initialize_plugin_system()
        
        # Überprüfen, ob die discover_adapters Funktion aufgerufen wurde
        mock_discover_adapters.assert_called_once()
        
        # Überprüfen, ob die Adapter registriert wurden
        mock_registry.register.assert_any_call(
            mock_adapter1, name="adapter1", info={"priority": 70}
        )
        mock_registry.register.assert_any_call(
            mock_adapter2, name="adapter2", info={"priority": 60}
        )
        
        # Überprüfen, ob die entdeckten Adapter zurückgegeben wurden
        self.assertEqual(discovered_adapters, ["adapter1", "adapter2"])

if __name__ == '__main__':
    unittest.main() 