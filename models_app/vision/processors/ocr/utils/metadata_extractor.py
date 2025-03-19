"""
Funktionen zur Extraktion und Formatierung von Metadaten aus OCR-Ergebnissen.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

def extract_dates(text: str) -> List[Dict[str, Any]]:
    """
    Extrahiert Datumsangaben aus einem Text.
    
    Args:
        text: Text, aus dem Datumsangaben extrahiert werden sollen
        
    Returns:
        Liste von gefundenen Datumsangaben mit Position und Wert
    """
    dates = []
    
    # Verschiedene Datumsformate
    patterns = [
        # DD.MM.YYYY oder DD-MM-YYYY
        r'(\d{1,2})[.-](\d{1,2})[.-](\d{4})',
        # YYYY-MM-DD
        r'(\d{4})-(\d{1,2})-(\d{1,2})',
        # DD. Month YYYY
        r'(\d{1,2})\.\s*(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*(\d{4})',
        r'(\d{1,2})\.\s*(Jan|Feb|Mär|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)[a-z]*\s*(\d{4})',
        # Englische Datumsformate
        r'(\d{1,2})\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s*(\d{4})',
        r'(\d{1,2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(\d{4})'
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            date_str = match.group(0)
            start_pos = match.start()
            end_pos = match.end()
            
            try:
                # Versuche, das Datum zu parsen
                if re.match(r'\d{4}-\d{1,2}-\d{1,2}', date_str):
                    # YYYY-MM-DD
                    year, month, day = map(int, date_str.split('-'))
                elif re.search(r'\d{1,2}[.-]\d{1,2}[.-]\d{4}', date_str):
                    # DD.MM.YYYY oder DD-MM-YYYY
                    parts = re.split(r'[.-]', date_str)
                    day, month, year = map(int, parts)
                else:
                    # Textformat mit Monatsnamen
                    continue  # Komplexere Formate überspringen
                
                # Datum validieren
                if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                    date_obj = datetime(year, month, day)
                    
                    dates.append({
                        'value': date_obj.strftime('%Y-%m-%d'),
                        'original': date_str,
                        'position': (start_pos, end_pos)
                    })
            except (ValueError, OverflowError):
                # Ungültiges Datum ignorieren
                pass
    
    return dates

def extract_amounts(text: str) -> List[Dict[str, Any]]:
    """
    Extrahiert Geldbeträge aus einem Text.
    
    Args:
        text: Text, aus dem Geldbeträge extrahiert werden sollen
        
    Returns:
        Liste von gefundenen Geldbeträgen mit Position und Wert
    """
    amounts = []
    
    # Verschiedene Geldbetragsformate
    patterns = [
        # EUR-Beträge
        r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*(?:€|EUR|Euro)',
        r'(\d{1,3}(?:,\d{3})*\.\d{2})\s*(?:€|EUR|Euro)',
        # USD-Beträge
        r'\$\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
        # Allgemeine Beträge mit Währungssymbol
        r'(\d{1,3}(?:\.\d{3})*,\d{2})\s*([A-Z]{3})',
        r'(\d{1,3}(?:,\d{3})*\.\d{2})\s*([A-Z]{3})'
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            amount_str = match.group(0)
            start_pos = match.start()
            end_pos = match.end()
            
            # Währung bestimmen
            currency = None
            if '€' in amount_str or 'EUR' in amount_str or 'Euro' in amount_str:
                currency = 'EUR'
            elif '$' in amount_str:
                currency = 'USD'
            else:
                # Versuche, die Währung aus dem Match zu extrahieren
                currency_match = re.search(r'[A-Z]{3}', amount_str)
                if currency_match:
                    currency = currency_match.group(0)
            
            # Betrag extrahieren und normalisieren
            amount_match = re.search(r'\d[\d.,]*\d', amount_str)
            if amount_match:
                amount_value = amount_match.group(0)
                
                # Normalisieren (auf Punkt als Dezimaltrennzeichen)
                if ',' in amount_value and '.' in amount_value:
                    if amount_value.rindex('.') > amount_value.rindex(','):
                        # Format: 1,234.56
                        amount_value = amount_value.replace(',', '')
                    else:
                        # Format: 1.234,56
                        amount_value = amount_value.replace('.', '').replace(',', '.')
                elif ',' in amount_value:
                    # Prüfen, ob Komma als Dezimaltrennzeichen oder Tausendertrennzeichen verwendet wird
                    if len(amount_value.split(',')[-1]) == 2:
                        # Format: 1234,56 (Komma als Dezimaltrennzeichen)
                        amount_value = amount_value.replace(',', '.')
                
                try:
                    # Betrag als Float parsen
                    amount_float = float(amount_value)
                    
                    amounts.append({
                        'value': amount_float,
                        'currency': currency,
                        'original': amount_str,
                        'position': (start_pos, end_pos)
                    })
                except ValueError:
                    # Ungültigen Betrag ignorieren
                    pass
    
    return amounts

def extract_invoice_number(text: str) -> Optional[Dict[str, Any]]:
    """
    Extrahiert eine Rechnungsnummer aus einem Text.
    
    Args:
        text: Text, aus dem die Rechnungsnummer extrahiert werden soll
        
    Returns:
        Gefundene Rechnungsnummer mit Position und Wert oder None
    """
    # Verschiedene Formate für Rechnungsnummern
    patterns = [
        r'Rechnungs(?:[-\s])?(?:nummer|nr)\.?:?\s*([A-Za-z0-9][-A-Za-z0-9/]{3,20})',
        r'Invoice\s*(?:number|no)\.?:?\s*([A-Za-z0-9][-A-Za-z0-9/]{3,20})',
        r'(?:Rechnung|Invoice|Bill)\s*#\s*([A-Za-z0-9][-A-Za-z0-9/]{3,20})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            invoice_number = match.group(1).strip()
            start_pos = match.start(1)
            end_pos = match.end(1)
            
            return {
                'value': invoice_number,
                'position': (start_pos, end_pos)
            }
    
    return None

def extract_vat_number(text: str) -> Optional[Dict[str, Any]]:
    """
    Extrahiert eine Umsatzsteuer-ID aus einem Text.
    
    Args:
        text: Text, aus dem die Umsatzsteuer-ID extrahiert werden soll
        
    Returns:
        Gefundene Umsatzsteuer-ID mit Position und Wert oder None
    """
    # Verschiedene Formate für Umsatzsteuer-IDs
    patterns = [
        r'USt-IdNr\.?:?\s*([A-Z]{2}\d{9,12})',
        r'Umsatzsteuer-ID:?\s*([A-Z]{2}\d{9,12})',
        r'VAT\s*(?:number|ID):?\s*([A-Z]{2}\d{9,12})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            vat_number = match.group(1).strip()
            start_pos = match.start(1)
            end_pos = match.end(1)
            
            return {
                'value': vat_number,
                'position': (start_pos, end_pos)
            }
    
    return None

def extract_addresses(text: str) -> List[Dict[str, Any]]:
    """
    Extrahiert Adressen aus einem Text.
    
    Args:
        text: Text, aus dem Adressen extrahiert werden sollen
        
    Returns:
        Liste von gefundenen Adressen mit Position und Wert
    """
    addresses = []
    
    # Einfache Adresserkennung (Straße mit Hausnummer, PLZ und Ort)
    # Deutsches Format
    pattern_de = r'([A-Za-zäöüÄÖÜß][-A-Za-zäöüÄÖÜß\s]+\s\d+[a-z]?)\s*,?\s*(\d{5})\s+([A-Za-zäöüÄÖÜß][-A-Za-zäöüÄÖÜß\s]+)'
    
    # Englisches Format
    pattern_en = r'(\d+\s[A-Za-z][-A-Za-z\s]+)\s*,?\s*([A-Za-z][-A-Za-z\s]+)\s*,?\s*([A-Z]{2}\s\d{5}|\d{5})'
    
    # Deutsche Adressen suchen
    for match in re.finditer(pattern_de, text):
        street = match.group(1).strip()
        postal_code = match.group(2).strip()
        city = match.group(3).strip()
        
        start_pos = match.start()
        end_pos = match.end()
        
        addresses.append({
            'street': street,
            'postal_code': postal_code,
            'city': city,
            'country': 'DE',  # Annahme: Deutschland
            'original': match.group(0),
            'position': (start_pos, end_pos)
        })
    
    # Englische Adressen suchen
    for match in re.finditer(pattern_en, text):
        street = match.group(1).strip()
        city = match.group(2).strip()
        postal_code = match.group(3).strip()
        
        start_pos = match.start()
        end_pos = match.end()
        
        addresses.append({
            'street': street,
            'city': city,
            'postal_code': postal_code,
            'country': 'US',  # Annahme: USA
            'original': match.group(0),
            'position': (start_pos, end_pos)
        })
    
    return addresses

def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extrahiert verschiedene Metadaten aus einem Text.
    
    Args:
        text: Text, aus dem Metadaten extrahiert werden sollen
        
    Returns:
        Dictionary mit extrahierten Metadaten
    """
    metadata = {}
    
    # Datumsangaben extrahieren
    dates = extract_dates(text)
    if dates:
        metadata['dates'] = dates
    
    # Geldbeträge extrahieren
    amounts = extract_amounts(text)
    if amounts:
        metadata['amounts'] = amounts
    
    # Rechnungsnummer extrahieren
    invoice_number = extract_invoice_number(text)
    if invoice_number:
        metadata['invoice_number'] = invoice_number
    
    # Umsatzsteuer-ID extrahieren
    vat_number = extract_vat_number(text)
    if vat_number:
        metadata['vat_number'] = vat_number
    
    # Adressen extrahieren
    addresses = extract_addresses(text)
    if addresses:
        metadata['addresses'] = addresses
    
    return metadata

def classify_document_type(text: str) -> Tuple[str, float]:
    """
    Klassifiziert den Dokumenttyp anhand des Textes.
    
    Args:
        text: Text, anhand dessen der Dokumenttyp klassifiziert werden soll
        
    Returns:
        Tuple aus Dokumenttyp und Konfidenz
    """
    # Einfache regelbasierte Klassifikation
    text_lower = text.lower()
    
    # Schlüsselwörter für verschiedene Dokumenttypen
    keywords = {
        'invoice': ['rechnung', 'invoice', 'bill', 'payment', 'zahlung', 'betrag', 'amount', 'umsatzsteuer', 'vat'],
        'receipt': ['quittung', 'receipt', 'kassenbon', 'beleg', 'kaufbeleg'],
        'contract': ['vertrag', 'contract', 'agreement', 'vereinbarung', 'konditionen', 'terms'],
        'letter': ['sehr geehrte', 'dear', 'mit freundlichen grüßen', 'sincerely', 'best regards'],
        'form': ['formular', 'form', 'antrag', 'application', 'bitte ausfüllen', 'please fill'],
        'id_document': ['personalausweis', 'id card', 'reisepass', 'passport', 'führerschein', 'driver\'s license']
    }
    
    # Zähle Treffer für jeden Dokumenttyp
    scores = {doc_type: 0 for doc_type in keywords}
    
    for doc_type, words in keywords.items():
        for word in words:
            if word in text_lower:
                scores[doc_type] += 1
    
    # Bestimme den Dokumenttyp mit der höchsten Punktzahl
    max_score = 0
    best_type = 'unknown'
    
    for doc_type, score in scores.items():
        if score > max_score:
            max_score = score
            best_type = doc_type
    
    # Berechne Konfidenz (normalisiert auf [0, 1])
    total_keywords = sum(len(words) for words in keywords.values())
    confidence = min(1.0, max_score / (total_keywords * 0.2))  # 20% der Keywords als Maximum
    
    return best_type, confidence

def extract_document_metadata(text: str) -> Dict[str, Any]:
    """
    Extrahiert umfassende Dokumentmetadaten.
    
    Args:
        text: Text, aus dem Metadaten extrahiert werden sollen
        
    Returns:
        Dictionary mit Dokumentmetadaten
    """
    # Grundlegende Metadaten extrahieren
    metadata = extract_metadata(text)
    
    # Dokumenttyp klassifizieren
    doc_type, confidence = classify_document_type(text)
    metadata['document_type'] = {
        'type': doc_type,
        'confidence': confidence
    }
    
    # Textstatistiken
    words = text.split()
    metadata['statistics'] = {
        'character_count': len(text),
        'word_count': len(words),
        'line_count': text.count('\n') + 1
    }
    
    return metadata 