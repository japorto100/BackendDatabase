"""
Basisklasse für OCR-Adapter-Tests.
Bietet gemeinsame Funktionalität für alle OCR-Adapter-Tests.
"""

import unittest
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image, ImageDraw, ImageFont

class BaseOCRAdapterTest(unittest.TestCase):
    """Basis-Testklasse für alle OCR-Adapter."""
    
    # Konstanten für die abgeleiteten Klassen
    ADAPTER_CLASS = None  # Muss in abgeleiteten Klassen überschrieben werden
    CONFIG_DICT = {}      # Standardkonfiguration für den Adapter
    MOCK_IMPORTS = []     # Liste der zu mockenden Importe
    
    def setUp(self):
        """Gemeinsame Setup-Logik für alle OCR-Adapter-Tests."""
        self.temp_files = []  # Liste für temporäre Dateien zum Aufräumen
        
        # Erstelle Test-Images (die konkrete Implementierung kann überschrieben werden)
        self.test_image = self._create_test_image()
        self.test_image_path = self._save_test_image(self.test_image, "test_image.png")
        
        # Erstelle den Adapter (Implementierung in konkreter Klasse)
        self._setup_mocks()
        self.adapter = self._create_adapter()
    
    def _setup_mocks(self):
        """Erstellt die erforderlichen Mocks basierend auf MOCK_IMPORTS."""
        self.mocks = {}
        for import_path in self.MOCK_IMPORTS:
            patcher = patch(import_path)
            mock = patcher.start()
            self.mocks[import_path] = mock
            self.addCleanup(patcher.stop)
    
    def _create_adapter(self):
        """Erstellt eine Instanz des zu testenden Adapters."""
        if not self.ADAPTER_CLASS:
            raise NotImplementedError("ADAPTER_CLASS muss in abgeleiteter Klasse definiert werden")
        return self.ADAPTER_CLASS(config=self.CONFIG_DICT)
    
    def _create_test_image(self, width=400, height=200, text="Test OCR"):
        """Erstellt ein einfaches Test-Bild mit Text für OCR-Tests."""
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        
        try:
            # Versuche eine Standardschriftart zu verwenden
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            # Fallback auf Standardschriftart
            font = ImageFont.load_default()
        
        d.text((50, 50), text, fill=(0, 0, 0), font=font)
        
        return img
    
    def _save_test_image(self, image, filename):
        """Speichert ein Bild in einer temporären Datei und gibt den Pfad zurück."""
        fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(filename)[1])
        os.close(fd)
        image.save(temp_path)
        self.temp_files.append(temp_path)
        return temp_path
    
    def _create_test_table_image(self, width=600, height=400):
        """Erstellt ein Test-Bild mit einer einfachen Tabelle."""
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        
        # Zeichne Tabellenrahmen
        d.rectangle([(50, 50), (550, 350)], outline=(0, 0, 0), width=2)
        
        # Zeichne Spaltenlinien
        for x in [150, 250, 350, 450]:
            d.line([(x, 50), (x, 350)], fill=(0, 0, 0), width=1)
        
        # Zeichne Zeilenlinien
        for y in [100, 150, 200, 250, 300]:
            d.line([(50, y), (550, y)], fill=(0, 0, 0), width=1)
        
        # Füge Text hinzu
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        # Tabellenüberschriften
        headers = ["Header 1", "Header 2", "Header 3", "Header 4", "Header 5"]
        for i, header in enumerate(headers):
            x = 60 + i * 100
            d.text((x, 70), header, fill=(0, 0, 0), font=font)
        
        # Tabellendaten
        data = [
            ["Data 1,1", "Data 1,2", "Data 1,3", "Data 1,4", "Data 1,5"],
            ["Data 2,1", "Data 2,2", "Data 2,3", "Data 2,4", "Data 2,5"],
            ["Data 3,1", "Data 3,2", "Data 3,3", "Data 3,4", "Data 3,5"],
            ["Data 4,1", "Data 4,2", "Data 4,3", "Data 4,4", "Data 4,5"]
        ]
        
        for row_idx, row in enumerate(data):
            for col_idx, cell in enumerate(row):
                x = 60 + col_idx * 100
                y = 120 + row_idx * 50
                d.text((x, y), cell, fill=(0, 0, 0), font=font)
        
        return img
    
    def _create_test_formula_image(self, width=400, height=200):
        """Erstellt ein Test-Bild mit einer mathematischen Formel."""
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        # Eine einfache Formel zeichnen
        d.text((50, 50), "E = mc²", fill=(0, 0, 0), font=font)
        d.text((50, 100), "∫ f(x) dx = F(x) + C", fill=(0, 0, 0), font=font)
        
        return img
    
    def tearDown(self):
        """Gemeinsame Aufräumlogik für alle OCR-Adapter-Tests."""
        # Lösche temporäre Dateien
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # Gemeinsame Testmethoden
    
    def test_initialization(self):
        """Testet die grundlegende Initialisierung des Adapters."""
        # Diese Methode sollte in abgeleiteten Klassen überschrieben werden
        pass
    
    def test_process_image_with_file_path(self):
        """Testet die Verarbeitung eines Bildes von einem Dateipfad."""
        # Diese Methode sollte in abgeleiteten Klassen überschrieben werden
        pass
    
    def test_process_image_with_numpy_array(self):
        """Testet die Verarbeitung eines Bildes als NumPy-Array."""
        # Diese Methode sollte in abgeleiteten Klassen überschrieben werden
        pass
    
    def test_get_supported_languages(self):
        """Testet das Abrufen unterstützter Sprachen."""
        languages = self.adapter.get_supported_languages()
        self.assertIsInstance(languages, list)
        self.assertTrue(len(languages) > 0)
    
    def test_get_model_info(self):
        """Testet das Abrufen von Modellinformationen."""
        info = self.adapter.get_model_info()
        self.assertIsInstance(info, dict)
        self.assertIn("name", info)
        self.assertIn("type", info)
        self.assertIn("capabilities", info)