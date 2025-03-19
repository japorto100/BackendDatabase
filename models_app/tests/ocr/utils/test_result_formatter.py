"""
Tests für die Ergebnisformatierung aus dem result_formatter Modul.
"""

import unittest
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from models_app.ocr.utils.result_formatter import (
    create_standard_result, merge_results, format_as_text,
    format_as_html, format_as_json, format_as_markdown,
    format_table_as_csv, format_table_as_html, format_table_as_json
)

class TestResultFormatter(unittest.TestCase):
    """Test-Klasse für die OCR-Ergebnisformatierung"""
    
    def setUp(self):
        """Setup für Tests"""
        # Standardergebnis für Tests erstellen
        self.sample_result = {
            "text": "Dies ist ein Beispieltext.",
            "blocks": [
                {
                    "text": "Dies ist ein",
                    "conf": 0.95,
                    "bbox": [10, 10, 110, 30]
                },
                {
                    "text": "Beispieltext.",
                    "conf": 0.90,
                    "bbox": [10, 40, 110, 60]
                }
            ],
            "confidence": 0.925,
            "model": "tesseract",
            "language": "deu",
            "metadata": {
                "tesseract_version": "4.1.1"
            }
        }
        
        # Beispiel-Tabellendaten
        self.sample_table_data = [
            ["Name", "Alter", "Stadt"],
            ["Max", "30", "Berlin"],
            ["Anna", "25", "München"]
        ]
        
        # Als DataFrame
        self.sample_dataframe = pd.DataFrame(
            self.sample_table_data[1:], 
            columns=self.sample_table_data[0]
        )
    
    def test_create_standard_result(self):
        """Test für die create_standard_result Funktion"""
        # Test mit minimalen Parametern
        result = create_standard_result(text="Test")
        self.assertEqual(result["text"], "Test")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["model"], "unknown")
        self.assertEqual(result["language"], "unknown")
        self.assertEqual(result["blocks"], [])
        self.assertEqual(result["metadata"], {})
        
        # Test mit allen Parametern
        result = create_standard_result(
            text="Vollständiger Test",
            blocks=[{"text": "Block", "conf": 0.9, "bbox": [0, 0, 10, 10]}],
            confidence=0.85,
            model="testmodel",
            language="eng",
            metadata={"test": "metadata"},
            raw_output="raw data"
        )
        
        self.assertEqual(result["text"], "Vollständiger Test")
        self.assertEqual(result["confidence"], 0.85)
        self.assertEqual(result["model"], "testmodel")
        self.assertEqual(result["language"], "eng")
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["metadata"], {"test": "metadata"})
        self.assertEqual(result["raw_output"], "raw data")
    
    def test_merge_results(self):
        """Test für die merge_results Funktion"""
        # Test mit einem leeren Array
        result = merge_results([])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["confidence"], 0.0)
        
        # Test mit einem einzelnen Ergebnis
        result = merge_results([self.sample_result])
        self.assertEqual(result, self.sample_result)
        
        # Test mit mehreren Ergebnissen
        second_result = {
            "text": "Dies ist ein zweiter Text.",
            "blocks": [
                {
                    "text": "Dies ist ein zweiter",
                    "conf": 0.85,
                    "bbox": [10, 70, 150, 90]
                },
                {
                    "text": "Text.",
                    "conf": 0.95,
                    "bbox": [10, 100, 50, 120]
                }
            ],
            "confidence": 0.90,
            "model": "easyocr",
            "language": "en",
            "metadata": {
                "version": "1.4"
            }
        }
        
        result = merge_results([self.sample_result, second_result])
        
        # Überprüfen der zusammengeführten Texte
        self.assertEqual(result["text"], "Dies ist ein Beispieltext.\nDies ist ein zweiter Text.")
        
        # Überprüfen der Blöcke
        self.assertEqual(len(result["blocks"]), 4)
        
        # Überprüfen der Konfidenz (gewichteter Durchschnitt)
        self.assertAlmostEqual(result["confidence"], 0.9125, places=4)
        
        # Überprüfen des Modells und der Sprache
        self.assertEqual(result["model"], "tesseract+easyocr")
        self.assertEqual(result["language"], "deu+en")
        
        # Überprüfen der Metadaten
        self.assertIn("source_0", result["metadata"])
        self.assertIn("source_1", result["metadata"])
    
    def test_format_as_text(self):
        """Test für die format_as_text Funktion"""
        # Einfacher Text-Export
        text_result = format_as_text(self.sample_result)
        self.assertEqual(text_result, "Dies ist ein Beispieltext.")
        
        # Test mit leerem Ergebnis
        empty_result = {}
        text_result = format_as_text(empty_result)
        self.assertEqual(text_result, "")
    
    def test_format_as_html(self):
        """Test für die format_as_html Funktion"""
        # HTML-Export mit Blöcken
        html_result = format_as_html(self.sample_result)
        
        # Grundlegende HTML-Struktur prüfen
        self.assertTrue(html_result.startswith("<div class='ocr-result'>"))
        self.assertTrue(html_result.endswith("</div>"))
        
        # Überprüfen, ob Block-DIVs mit Positionsinformationen enthalten sind
        self.assertIn("div class='ocr-block'", html_result)
        self.assertIn("style='position: absolute;", html_result)
        self.assertIn("data-confidence='", html_result)
        
        # Überprüfen, ob die Texte enthalten sind
        self.assertIn("Dies ist ein", html_result)
        self.assertIn("Beispieltext.", html_result)
        
        # Test ohne Blöcke
        result_no_blocks = {
            "text": "Text ohne Blöcke.\nMit Absatz.\n\nZweiter Absatz."
        }
        html_result = format_as_html(result_no_blocks)
        
        # Grundlegende HTML-Struktur prüfen
        self.assertTrue(html_result.startswith("<div class='ocr-result'>"))
        self.assertTrue(html_result.endswith("</div>"))
        
        # Überprüfen der Absätze
        self.assertIn("<p>", html_result)
        self.assertIn("</p>", html_result)
        self.assertIn("<br>", html_result)
    
    def test_format_as_json(self):
        """Test für die format_as_json Funktion"""
        # JSON-Export
        json_result = format_as_json(self.sample_result)
        
        # Überprüfen, ob es sich um gültiges JSON handelt
        parsed_json = json.loads(json_result)
        
        # Überprüfen der Inhalte
        self.assertEqual(parsed_json["text"], "Dies ist ein Beispieltext.")
        self.assertEqual(parsed_json["confidence"], 0.925)
        self.assertEqual(parsed_json["model"], "tesseract")
        self.assertEqual(len(parsed_json["blocks"]), 2)
        
        # Test mit nicht JSON-serialisierbaren Daten
        result_with_numpy = {
            "text": "Test mit NumPy",
            "array": np.array([1, 2, 3])
        }
        
        # Sollte eine Fehlermeldung zurückgeben
        json_result = format_as_json(result_with_numpy)
        self.assertIn("Fehler bei der JSON-Formatierung", json_result)
    
    def test_format_as_markdown(self):
        """Test für die format_as_markdown Funktion"""
        # Markdown-Export
        md_result = format_as_markdown(self.sample_result)
        
        # Überprüfen der Überschrift
        self.assertTrue(md_result.startswith("# OCR-Ergebnis"))
        
        # Überprüfen der Metadaten
        self.assertIn("**Modell:** tesseract", md_result)
        self.assertIn("**Konfidenz:** 0.93", md_result)
        
        # Überprüfen des Textes
        self.assertIn("Dies ist ein Beispieltext.", md_result)
    
    def test_format_table_as_csv(self):
        """Test für die format_table_as_csv Funktion"""
        # CSV-Export mit Liste von Listen
        csv_result = format_table_as_csv(self.sample_table_data)
        
        # Überprüfen, ob die CSV-Daten korrekt sind
        self.assertIn("Name,Alter,Stadt", csv_result)
        self.assertIn("Max,30,Berlin", csv_result)
        self.assertIn("Anna,25,München", csv_result)
        
        # CSV-Export mit DataFrame
        csv_result = format_table_as_csv(self.sample_dataframe)
        
        # Überprüfen, ob die CSV-Daten korrekt sind
        self.assertIn("Name,Alter,Stadt", csv_result)
        self.assertIn("Max,30,Berlin", csv_result)
        self.assertIn("Anna,25,München", csv_result)
        
        # Test mit ungültigen Daten
        with self.assertRaises(Exception):
            format_table_as_csv("Keine Tabelle")
    
    def test_format_table_as_html(self):
        """Test für die format_table_as_html Funktion"""
        # HTML-Export mit Liste von Listen
        html_result = format_table_as_html(self.sample_table_data)
        
        # Überprüfen, ob die HTML-Tabelle korrekt ist
        self.assertIn("<table", html_result)
        self.assertIn("</table>", html_result)
        self.assertIn("<th>Name</th>", html_result)
        self.assertIn("<td>Max</td>", html_result)
        
        # HTML-Export mit DataFrame
        html_result = format_table_as_html(self.sample_dataframe)
        
        # Überprüfen, ob die HTML-Tabelle korrekt ist
        self.assertIn("<table", html_result)
        self.assertIn("</table>", html_result)
        self.assertIn("Name", html_result)
        self.assertIn("Max", html_result)
        
        # Test mit ungültigen Daten
        html_result = format_table_as_html("Keine Tabelle")
        self.assertIn("Fehler bei der Formatierung", html_result)
    
    def test_format_table_as_json(self):
        """Test für die format_table_as_json Funktion"""
        # JSON-Export mit Liste von Listen
        json_result = format_table_as_json(self.sample_table_data)
        
        # Überprüfen, ob es sich um gültiges JSON handelt
        parsed_json = json.loads(json_result)
        
        # Überprüfen der Struktur
        self.assertIsInstance(parsed_json, list)
        self.assertEqual(len(parsed_json), 2)  # Zwei Datenzeilen (ohne Überschrift)
        
        # JSON-Export mit DataFrame
        json_result = format_table_as_json(self.sample_dataframe)
        
        # Überprüfen, ob es sich um gültiges JSON handelt
        parsed_json = json.loads(json_result)
        
        # Überprüfen der Struktur
        self.assertIsInstance(parsed_json, list)
        self.assertEqual(len(parsed_json), 2)
        self.assertEqual(parsed_json[0]["Name"], "Max")
        self.assertEqual(parsed_json[1]["Stadt"], "München")
        
        # Test mit ungültigen Daten
        json_result = format_table_as_json("Keine Tabelle")
        self.assertEqual(json_result, "[]")

if __name__ == '__main__':
    unittest.main() 