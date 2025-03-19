"""
Tests für das Konfigurationsmanagement aus dem config_manager Modul.
"""

import unittest
import os
import json
import tempfile
from unittest.mock import patch, MagicMock

from models_app.ocr.utils.config_manager import ConfigManager

class TestConfigManager(unittest.TestCase):
    """Test-Klasse für das OCR-Konfigurationsmanagement"""
    
    def setUp(self):
        """Setup für Tests"""
        # Temporäres Verzeichnis für Konfigurationsdateien erstellen
        self.temp_dir = tempfile.mkdtemp()
        
        # ConfigManager mit dem temporären Verzeichnis initialisieren
        self.config_manager = ConfigManager(config_dir=self.temp_dir)
    
    def tearDown(self):
        """Cleanup nach Tests"""
        # Temporäres Verzeichnis und Dateien löschen
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        os.rmdir(self.temp_dir)
    
    def test_get_default_config(self):
        """Test für die get_default_config Methode"""
        # Standard-Konfiguration für Tesseract abrufen
        config = self.config_manager.get_default_config("tesseract")
        
        # Überprüfen, ob die Konfiguration die erwarteten Werte enthält
        self.assertIn("lang", config)
        self.assertEqual(config["lang"], "eng")
        self.assertIn("config", config)
        self.assertIn("path", config)
        self.assertIn("preprocess", config)
        self.assertTrue(config["preprocess"])
        
        # Standard-Konfiguration für einen unbekannten Adapter abrufen
        config = self.config_manager.get_default_config("unknown_adapter")
        self.assertEqual(config, {})
    
    def test_load_config(self):
        """Test für die load_config Methode"""
        # Konfiguration für einen Adapter abrufen, der noch keine Konfigurationsdatei hat
        config = self.config_manager.load_config("tesseract")
        
        # Überprüfen, ob die Standardkonfiguration zurückgegeben wurde
        self.assertIn("lang", config)
        self.assertEqual(config["lang"], "eng")
        
        # Konfigurationsdatei erstellen
        config_file_path = os.path.join(self.temp_dir, "tesseract.json")
        with open(config_file_path, 'w') as f:
            json.dump({"lang": "deu", "custom_option": "value"}, f)
        
        # Cache zurücksetzen
        self.config_manager.config_cache = {}
        
        # Konfiguration erneut abrufen
        config = self.config_manager.load_config("tesseract")
        
        # Überprüfen, ob die Konfiguration aus der Datei geladen wurde
        self.assertEqual(config["lang"], "deu")
        self.assertEqual(config["custom_option"], "value")
        
        # Überprüfen, ob andere Standardwerte beibehalten wurden
        self.assertIn("preprocess", config)
        self.assertTrue(config["preprocess"])
    
    def test_save_config(self):
        """Test für die save_config Methode"""
        # Konfiguration speichern
        config = {
            "lang": "fra",
            "config": "--psm 6",
            "path": "/usr/bin/tesseract",
            "preprocess": False
        }
        
        result = self.config_manager.save_config("tesseract", config)
        
        # Überprüfen, ob das Speichern erfolgreich war
        self.assertTrue(result)
        
        # Überprüfen, ob die Datei erstellt wurde
        config_file_path = os.path.join(self.temp_dir, "tesseract.json")
        self.assertTrue(os.path.exists(config_file_path))
        
        # Dateiinhalt überprüfen
        with open(config_file_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config["lang"], "fra")
        self.assertEqual(loaded_config["config"], "--psm 6")
        self.assertEqual(loaded_config["path"], "/usr/bin/tesseract")
        self.assertFalse(loaded_config["preprocess"])
        
        # Cache überprüfen
        self.assertIn("tesseract", self.config_manager.config_cache)
        self.assertEqual(self.config_manager.config_cache["tesseract"], config)
    
    def test_merge_config(self):
        """Test für die merge_config Methode"""
        # Grundkonfiguration speichern
        base_config = {
            "lang": "eng",
            "config": "",
            "path": None,
            "preprocess": True
        }
        
        self.config_manager.save_config("tesseract", base_config)
        
        # Neue Konfiguration zusammenführen
        new_config = {
            "lang": "deu",
            "custom_option": "value"
        }
        
        merged_config = self.config_manager.merge_config("tesseract", new_config)
        
        # Überprüfen, ob die Konfigurationen korrekt zusammengeführt wurden
        self.assertEqual(merged_config["lang"], "deu")  # Überschrieben
        self.assertEqual(merged_config["config"], "")  # Beibehalten
        self.assertIsNone(merged_config["path"])  # Beibehalten
        self.assertTrue(merged_config["preprocess"])  # Beibehalten
        self.assertEqual(merged_config["custom_option"], "value")  # Neu hinzugefügt
        
        # Überprüfen, ob die Datei aktualisiert wurde
        config_file_path = os.path.join(self.temp_dir, "tesseract.json")
        with open(config_file_path, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config["lang"], "deu")
        self.assertEqual(loaded_config["custom_option"], "value")
    
    def test_get_config_value(self):
        """Test für die get_config_value Methode"""
        # Konfiguration speichern
        config = {
            "lang": "eng",
            "config": "--psm 6",
            "nested": {
                "option": "value"
            }
        }
        
        self.config_manager.save_config("tesseract", config)
        
        # Einzelne Werte abrufen
        self.assertEqual(self.config_manager.get_config_value("tesseract", "lang"), "eng")
        self.assertEqual(self.config_manager.get_config_value("tesseract", "config"), "--psm 6")
        self.assertEqual(self.config_manager.get_config_value("tesseract", "nested"), {"option": "value"})
        
        # Nicht vorhandenen Wert abrufen
        self.assertIsNone(self.config_manager.get_config_value("tesseract", "non_existent"))
        self.assertEqual(self.config_manager.get_config_value("tesseract", "non_existent", "default"), "default")
    
    def test_set_config_value(self):
        """Test für die set_config_value Methode"""
        # Grundkonfiguration speichern
        config = {
            "lang": "eng",
            "config": "",
            "preprocess": True
        }
        
        self.config_manager.save_config("tesseract", config)
        
        # Wert ändern
        result = self.config_manager.set_config_value("tesseract", "lang", "deu")
        
        # Überprüfen, ob das Ändern erfolgreich war
        self.assertTrue(result)
        
        # Überprüfen, ob der Wert aktualisiert wurde
        updated_config = self.config_manager.load_config("tesseract")
        self.assertEqual(updated_config["lang"], "deu")
        
        # Neuen Wert hinzufügen
        result = self.config_manager.set_config_value("tesseract", "custom_option", "value")
        
        # Überprüfen, ob das Hinzufügen erfolgreich war
        self.assertTrue(result)
        
        # Überprüfen, ob der Wert hinzugefügt wurde
        updated_config = self.config_manager.load_config("tesseract")
        self.assertEqual(updated_config["custom_option"], "value")
    
    def test_get_all_configs(self):
        """Test für die get_all_configs Methode"""
        # Mehrere Konfigurationen speichern
        self.config_manager.save_config("tesseract", {"lang": "eng"})
        self.config_manager.save_config("easyocr", {"lang": ["en"]})
        self.config_manager.save_config("paddle", {"lang": "en"})
        
        # Alle Konfigurationen abrufen
        all_configs = self.config_manager.get_all_configs()
        
        # Überprüfen, ob alle Konfigurationen enthalten sind
        self.assertIn("tesseract", all_configs)
        self.assertIn("easyocr", all_configs)
        self.assertIn("paddle", all_configs)
        
        # Überprüfen der Werte
        self.assertEqual(all_configs["tesseract"]["lang"], "eng")
        self.assertEqual(all_configs["easyocr"]["lang"], ["en"])
        self.assertEqual(all_configs["paddle"]["lang"], "en")
    
    def test_reset_config(self):
        """Test für die reset_config Methode"""
        # Benutzerdefinierte Konfiguration speichern
        custom_config = {
            "lang": "deu",
            "config": "--psm 6",
            "custom_option": "value"
        }
        
        self.config_manager.save_config("tesseract", custom_config)
        
        # Konfiguration zurücksetzen
        result = self.config_manager.reset_config("tesseract")
        
        # Überprüfen, ob das Zurücksetzen erfolgreich war
        self.assertTrue(result)
        
        # Zurückgesetzte Konfiguration abrufen
        reset_config = self.config_manager.load_config("tesseract")
        
        # Überprüfen, ob die Standardkonfiguration wiederhergestellt wurde
        self.assertEqual(reset_config["lang"], "eng")
        self.assertEqual(reset_config["config"], "")
        self.assertNotIn("custom_option", reset_config)

if __name__ == '__main__':
    unittest.main() 