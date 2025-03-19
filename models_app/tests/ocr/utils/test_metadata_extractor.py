"""
Tests für die Metadatenextraktionsfunktionen aus dem metadata_extractor Modul.
"""

import unittest
from unittest.mock import patch, MagicMock

from models_app.ocr.utils.metadata_extractor import (
    extract_dates, extract_amounts, extract_invoice_number,
    extract_vat_number, extract_addresses, extract_metadata,
    classify_document_type, extract_document_metadata
)

class TestMetadataExtractor(unittest.TestCase):
    """Test-Klasse für die Metadatenextraktion"""
    
    def test_extract_dates(self):
        """Test für die extract_dates Funktion"""
        # Test mit verschiedenen Datumsformaten
        text = """
        Datum: 01.02.2023
        Date: 2023-03-15
        Lieferdatum: 20.04.2023
        Rechnungsdatum: 05/10/2023
        Invoice Date: 2023/06/30
        """
        
        dates = extract_dates(text)
        
        # Überprüfen, ob alle Datumsformate erkannt wurden
        self.assertGreaterEqual(len(dates), 5)
        
        # Ein paar spezifische Daten überprüfen
        date_values = [d.get("value") for d in dates]
        self.assertIn("01.02.2023", date_values)
        self.assertIn("2023-03-15", date_values)
        
        # Überprüfen der Kontextinformationen
        for date in dates:
            if date.get("value") == "01.02.2023":
                self.assertEqual(date.get("context"), "Datum")
            elif date.get("value") == "20.04.2023":
                self.assertEqual(date.get("context"), "Lieferdatum")
    
    def test_extract_amounts(self):
        """Test für die extract_amounts Funktion"""
        # Test mit verschiedenen Geldbetragsformaten
        text = """
        Gesamtbetrag: 1.234,56 €
        Einzelpreis: 42,99€
        Total: $99.95
        Amount Due: EUR 500.00
        Summe: 1200 EUR
        """
        
        amounts = extract_amounts(text)
        
        # Überprüfen, ob alle Geldbeträge erkannt wurden
        self.assertGreaterEqual(len(amounts), 5)
        
        # Ein paar spezifische Beträge überprüfen
        amount_values = [a.get("value") for a in amounts]
        self.assertIn("1.234,56 €", amount_values)
        self.assertIn("$99.95", amount_values)
        
        # Überprüfen der Kontextinformationen
        for amount in amounts:
            if amount.get("value") == "1.234,56 €":
                self.assertEqual(amount.get("context"), "Gesamtbetrag")
            elif amount.get("value") == "42,99€":
                self.assertEqual(amount.get("context"), "Einzelpreis")
    
    def test_extract_invoice_number(self):
        """Test für die extract_invoice_number Funktion"""
        # Test mit verschiedenen Rechnungsnummernformaten
        text = """
        Rechnungsnummer: RE-2023-12345
        Invoice No.: INV/2023/001
        Belegnummer: BLG-987654
        """
        
        # Test für erste Rechnungsnummer
        result = extract_invoice_number(text)
        self.assertIsNotNone(result)
        self.assertIn("value", result)
        self.assertEqual(result.get("value"), "RE-2023-12345")
        self.assertEqual(result.get("context"), "Rechnungsnummer")
        
        # Test mit anderer Rechnungsnummer
        text2 = "Invoice Number: ABC-123456-XYZ"
        result = extract_invoice_number(text2)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("value"), "ABC-123456-XYZ")
        
        # Test mit ungültigem Text
        text3 = "No invoice number here."
        result = extract_invoice_number(text3)
        self.assertIsNone(result)
    
    def test_extract_vat_number(self):
        """Test für die extract_vat_number Funktion"""
        # Test mit verschiedenen USt-ID-Formaten
        text = """
        USt-IdNr.: DE123456789
        VAT Number: GB987654321
        Umsatzsteuer-ID: AT U12345678
        """
        
        # Test für erste USt-ID
        result = extract_vat_number(text)
        self.assertIsNotNone(result)
        self.assertIn("value", result)
        self.assertEqual(result.get("value"), "DE123456789")
        self.assertEqual(result.get("context"), "USt-IdNr.")
        
        # Test mit anderer USt-ID
        text2 = "Tax ID: ESA12345678"
        result = extract_vat_number(text2)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("value"), "ESA12345678")
        
        # Test mit ungültigem Text
        text3 = "No VAT number here."
        result = extract_vat_number(text3)
        self.assertIsNone(result)
    
    def test_extract_addresses(self):
        """Test für die extract_addresses Funktion"""
        # Test mit verschiedenen Adressformaten
        text = """
        Lieferanschrift:
        Max Mustermann
        Hauptstraße 123
        10115 Berlin
        
        Rechnungsadresse:
        Firma GmbH
        Musterweg 45
        80331 München
        Deutschland
        """
        
        addresses = extract_addresses(text)
        
        # Überprüfen, ob beide Adressen erkannt wurden
        self.assertGreaterEqual(len(addresses), 2)
        
        # Überprüfen der Kontextinformationen
        address_contexts = [a.get("context") for a in addresses]
        self.assertIn("Lieferanschrift", address_contexts)
        self.assertIn("Rechnungsadresse", address_contexts)
        
        # Überprüfen der Adressinhalte
        for address in addresses:
            if address.get("context") == "Lieferanschrift":
                self.assertIn("Berlin", address.get("value"))
            elif address.get("context") == "Rechnungsadresse":
                self.assertIn("München", address.get("value"))
    
    def test_extract_metadata(self):
        """Test für die extract_metadata Funktion"""
        # Test mit einem vollständigen Dokument
        text = """
        Rechnung Nr.: INV-2023-001
        Datum: 15.03.2023
        
        Lieferant:
        Versand GmbH
        Paketweg 1
        22767 Hamburg
        USt-ID: DE987654321
        
        Empfänger:
        Max Mustermann
        Hauptstraße 123
        10115 Berlin
        
        Artikel 1: 29,99 €
        Artikel 2: 49,95 €
        
        Gesamtbetrag: 79,94 €
        """
        
        metadata = extract_metadata(text)
        
        # Überprüfen, ob alle Metadatenkategorien extrahiert wurden
        self.assertIn("dates", metadata)
        self.assertIn("amounts", metadata)
        self.assertIn("invoice_number", metadata)
        self.assertIn("vat_number", metadata)
        self.assertIn("addresses", metadata)
        
        # Überprüfen einiger spezifischer Werte
        self.assertEqual(metadata["invoice_number"]["value"], "INV-2023-001")
        self.assertEqual(metadata["vat_number"]["value"], "DE987654321")
        
        # Überprüfen der Anzahl der gefundenen Elemente
        self.assertGreaterEqual(len(metadata["dates"]), 1)
        self.assertGreaterEqual(len(metadata["amounts"]), 3)
        self.assertGreaterEqual(len(metadata["addresses"]), 2)
    
    def test_classify_document_type(self):
        """Test für die classify_document_type Funktion"""
        # Test für eine Rechnung
        invoice_text = """
        RECHNUNG
        Rechnungsnummer: INV-2023-001
        Betrag: 99,95 €
        Zahlungsbedingungen: 14 Tage netto
        """
        
        doc_type, confidence = classify_document_type(invoice_text)
        self.assertEqual(doc_type, "invoice")
        self.assertGreater(confidence, 0.6)
        
        # Test für einen Lieferschein
        delivery_text = """
        LIEFERSCHEIN
        Lieferscheinnummer: DEL-2023-001
        Lieferdatum: 10.03.2023
        Versandart: DHL
        """
        
        doc_type, confidence = classify_document_type(delivery_text)
        self.assertEqual(doc_type, "delivery_note")
        self.assertGreater(confidence, 0.6)
        
        # Test für ein unbekanntes Dokument
        unknown_text = """
        Dies ist ein allgemeiner Text ohne spezifische Dokumentmerkmale.
        Er enthält keine typischen Schlüsselwörter oder Strukturen.
        """
        
        doc_type, confidence = classify_document_type(unknown_text)
        self.assertEqual(doc_type, "unknown")
        self.assertLess(confidence, 0.5)
    
    def test_extract_document_metadata(self):
        """Test für die extract_document_metadata Funktion"""
        # Test mit einem vollständigen Dokument
        text = """
        RECHNUNG
        
        Rechnungsnummer: INV-2023-001
        Datum: 15.03.2023
        
        Lieferant:
        Versand GmbH
        Paketweg 1
        22767 Hamburg
        USt-ID: DE987654321
        
        Empfänger:
        Max Mustermann
        Hauptstraße 123
        10115 Berlin
        
        Artikel 1: 29,99 €
        Artikel 2: 49,95 €
        
        Gesamtbetrag: 79,94 €
        Zahlungsbedingungen: 14 Tage netto
        """
        
        metadata = extract_document_metadata(text)
        
        # Überprüfen der Dokumenttyperkennung
        self.assertIn("document_type", metadata)
        self.assertEqual(metadata["document_type"]["type"], "invoice")
        self.assertGreater(metadata["document_type"]["confidence"], 0.6)
        
        # Überprüfen der extrahierten Metadaten
        self.assertIn("metadata", metadata)
        self.assertIn("invoice_number", metadata["metadata"])
        self.assertIn("dates", metadata["metadata"])
        self.assertIn("amounts", metadata["metadata"])
        
        # Überprüfen einiger spezifischer Werte
        self.assertEqual(metadata["metadata"]["invoice_number"]["value"], "INV-2023-001")
        
        # Überprüfen, ob mindestens ein Betrag gefunden wurde
        self.assertGreaterEqual(len(metadata["metadata"]["amounts"]), 1)
        
        # Überprüfen, ob der Gesamtbetrag gefunden wurde
        found_total = False
        for amount in metadata["metadata"]["amounts"]:
            if amount.get("context") == "Gesamtbetrag":
                self.assertEqual(amount.get("value"), "79,94 €")
                found_total = True
                break
        self.assertTrue(found_total)

if __name__ == '__main__':
    unittest.main()