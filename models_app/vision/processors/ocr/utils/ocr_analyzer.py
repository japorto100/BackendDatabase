"""
Funktionen zur Analyse und Klassifikation von Dokumenten.
"""

import os
import logging
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image

from models_app.ocr.utils.image_processing import load_image, detect_text_regions

logger = logging.getLogger(__name__)

def analyze_document_structure(image_path_or_array) -> Dict[str, Any]:
    """
    Analysiert die Struktur eines Dokuments.
    
    Args:
        image_path_or_array: Pfad zum Bild oder NumPy-Array
        
    Returns:
        Dictionary mit Strukturinformationen
    """
    try:
        # Bild laden
        pil_image, np_image, _ = load_image(image_path_or_array)
        
        # Bildgröße
        height, width = np_image.shape[:2]
        
        # Zu Graustufen konvertieren, falls das Bild farbig ist
        if len(np_image.shape) == 3:
            gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np_image.copy()
            
        # Textregionen erkennen
        text_regions = detect_text_regions(gray)
        
        # Horizontale und vertikale Linien erkennen
        horizontal_lines, vertical_lines = detect_lines(gray)
        
        # Tabellen erkennen
        tables = detect_tables(gray, horizontal_lines, vertical_lines)
        
        # Bilder erkennen
        images = detect_images(np_image)
        
        # Dokumentstruktur erstellen
        structure = {
            "size": {
                "width": width,
                "height": height
            },
            "text_regions": [
                {"x": x, "y": y, "width": w, "height": h}
                for x, y, w, h in text_regions
            ],
            "horizontal_lines": [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                for x1, y1, x2, y2 in horizontal_lines
            ],
            "vertical_lines": [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                for x1, y1, x2, y2 in vertical_lines
            ],
            "tables": [
                {"x": x, "y": y, "width": w, "height": h}
                for x, y, w, h in tables
            ],
            "images": [
                {"x": x, "y": y, "width": w, "height": h}
                for x, y, w, h in images
            ]
        }
        
        # Dokumenttyp bestimmen
        doc_type, confidence = classify_document_type(structure)
        structure["document_type"] = {
            "type": doc_type,
            "confidence": confidence
        }
        
        return structure
    except Exception as e:
        logger.error(f"Fehler bei der Dokumentstrukturanalyse: {str(e)}")
        return {
            "error": str(e),
            "document_type": {
                "type": "unknown",
                "confidence": 0.0
            }
        }

def detect_lines(gray_image) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """
    Erkennt horizontale und vertikale Linien in einem Bild.
    
    Args:
        gray_image: Graustufenbild als NumPy-Array
        
    Returns:
        Tuple aus Listen von horizontalen und vertikalen Linien
    """
    try:
        # Kanten erkennen
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Linien erkennen
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        horizontal_lines = []
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Horizontale Linien (y-Koordinaten ähnlich)
                if abs(y2 - y1) < 10:
                    horizontal_lines.append((x1, y1, x2, y2))
                
                # Vertikale Linien (x-Koordinaten ähnlich)
                elif abs(x2 - x1) < 10:
                    vertical_lines.append((x1, y1, x2, y2))
        
        return horizontal_lines, vertical_lines
    except Exception as e:
        logger.warning(f"Fehler bei der Linienerkennung: {str(e)}")
        return [], []

def detect_tables(gray_image, horizontal_lines, vertical_lines) -> List[Tuple[int, int, int, int]]:
    """
    Erkennt Tabellen in einem Bild anhand von horizontalen und vertikalen Linien.
    
    Args:
        gray_image: Graustufenbild als NumPy-Array
        horizontal_lines: Liste von horizontalen Linien
        vertical_lines: Liste von vertikalen Linien
        
    Returns:
        Liste von Tabellen als (x, y, w, h)
    """
    try:
        if not horizontal_lines or not vertical_lines:
            return []
            
        # Bild für Liniendarstellung erstellen
        height, width = gray_image.shape
        line_image = np.zeros((height, width), dtype=np.uint8)
        
        # Linien zeichnen
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
            
        # Morphologische Operationen, um Linien zu verbinden
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        line_image = cv2.dilate(line_image, kernel, iterations=2)
        line_image = cv2.erode(line_image, kernel, iterations=1)
        
        # Konturen finden
        contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        min_table_area = width * height * 0.01  # Mindestgröße für Tabellen (1% der Bildfläche)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Prüfen, ob die Kontur groß genug ist und ein Rechteck bildet
            if area >= min_table_area:
                # Prüfen, ob genügend Linien in der Kontur vorhanden sind
                h_lines_in_contour = sum(1 for x1, y1, x2, y2 in horizontal_lines 
                                        if y1 >= y and y1 <= y + h)
                v_lines_in_contour = sum(1 for x1, y1, x2, y2 in vertical_lines 
                                        if x1 >= x and x1 <= x + w)
                
                if h_lines_in_contour >= 2 and v_lines_in_contour >= 2:
                    tables.append((x, y, w, h))
        
        return tables
    except Exception as e:
        logger.warning(f"Fehler bei der Tabellenerkennung: {str(e)}")
        return []

def detect_images(image) -> List[Tuple[int, int, int, int]]:
    """
    Erkennt Bilder/Grafiken in einem Dokument.
    
    Args:
        image: Bild als NumPy-Array
        
    Returns:
        Liste von Bildern als (x, y, w, h)
    """
    try:
        # Zu Graustufen konvertieren, falls das Bild farbig ist
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Binarisieren
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphologische Operationen, um Text zu entfernen
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Konturen finden
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        images_found = []
        height, width = gray.shape
        min_image_area = width * height * 0.01  # Mindestgröße für Bilder (1% der Bildfläche)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Prüfen, ob die Kontur groß genug ist
            if area >= min_image_area:
                # Region aus dem Originalbild extrahieren
                roi = gray[y:y+h, x:x+w]
                
                # Standardabweichung berechnen (Bilder haben in der Regel eine höhere Varianz als Text)
                std_dev = np.std(roi)
                
                if std_dev > 20:  # Schwellenwert für Bildregionen
                    images_found.append((x, y, w, h))
        
        return images_found
    except Exception as e:
        logger.warning(f"Fehler bei der Bilderkennung: {str(e)}")
        return []

def has_mathematical_formulas(image_path_or_array) -> Tuple[bool, float]:
    """
    Prüft, ob ein Dokument mathematische Formeln enthält.
    
    Args:
        image_path_or_array: Pfad zum Bild oder NumPy-Array
        
    Returns:
        Tuple aus Boolean (hat Formeln) und Konfidenz
    """
    try:
        # Bild laden
        pil_image, np_image, _ = load_image(image_path_or_array)
        
        # Zu Graustufen konvertieren, falls das Bild farbig ist
        if len(np_image.shape) == 3:
            gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np_image.copy()
            
        # Binarisieren
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Mathematische Symbole suchen
        math_symbols = ['=', '+', '-', '×', '÷', '∑', '∫', '√', 'π', '∞', '≠', '≈', '≤', '≥']
        
        # OCR für Symbolerkennung (vereinfacht)
        # In einer realen Implementierung würde hier ein OCR-Modell verwendet werden
        
        # Stattdessen verwenden wir eine Heuristik basierend auf Bildmerkmalen
        
        # Horizontale und vertikale Linien erkennen (für Brüche, Wurzeln, etc.)
        horizontal_lines, vertical_lines = detect_lines(gray)
        
        # Textregionen erkennen
        text_regions = detect_text_regions(gray)
        
        # Heuristik: Wenn viele kurze horizontale Linien und vertikale Linien vorhanden sind,
        # handelt es sich wahrscheinlich um mathematische Formeln
        short_h_lines = sum(1 for x1, y1, x2, y2 in horizontal_lines if abs(x2 - x1) < 50)
        
        # Konfidenz berechnen
        confidence = min(1.0, (short_h_lines + len(vertical_lines)) / 10.0)
        
        return confidence > 0.3, confidence
    except Exception as e:
        logger.warning(f"Fehler bei der Formelprüfung: {str(e)}")
        return False, 0.0

def classify_document_type(structure: Dict[str, Any]) -> Tuple[str, float]:
    """
    Klassifiziert den Dokumenttyp anhand der Struktur.
    
    Args:
        structure: Dokumentstruktur
        
    Returns:
        Tuple aus Dokumenttyp und Konfidenz
    """
    # Merkmale extrahieren
    text_regions = structure.get("text_regions", [])
    tables = structure.get("tables", [])
    images = structure.get("images", [])
    horizontal_lines = structure.get("horizontal_lines", [])
    vertical_lines = structure.get("vertical_lines", [])
    
    # Anzahl der Elemente
    num_text_regions = len(text_regions)
    num_tables = len(tables)
    num_images = len(images)
    num_h_lines = len(horizontal_lines)
    num_v_lines = len(vertical_lines)
    
    # Einfache regelbasierte Klassifikation
    if num_tables > 0 and num_text_regions > 0:
        if num_tables / num_text_regions > 0.5:
            return "table_document", 0.8
        else:
            return "form", 0.7
    elif num_h_lines > 5 and num_v_lines > 5:
        return "structured_document", 0.7
    elif num_images > 3:
        return "image_rich_document", 0.8
    elif num_text_regions > 10:
        return "text_document", 0.9
    else:
        return "general_document", 0.5

def check_if_has_tables(image_path_or_array) -> Tuple[bool, float]:
    """
    Prüft, ob ein Dokument Tabellen enthält.
    
    Args:
        image_path_or_array: Pfad zum Bild oder NumPy-Array
        
    Returns:
        Tuple aus Boolean (hat Tabellen) und Konfidenz
    """
    try:
        # Bild laden
        pil_image, np_image, _ = load_image(image_path_or_array)
        
        # Zu Graustufen konvertieren, falls das Bild farbig ist
        if len(np_image.shape) == 3:
            gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np_image.copy()
            
        # Horizontale und vertikale Linien erkennen
        horizontal_lines, vertical_lines = detect_lines(gray)
        
        # Tabellen erkennen
        tables = detect_tables(gray, horizontal_lines, vertical_lines)
        
        # Konfidenz berechnen
        if tables:
            confidence = min(1.0, len(tables) * 0.3)
        else:
            # Alternative Heuristik: Wenn viele sich kreuzende Linien vorhanden sind,
            # handelt es sich wahrscheinlich um Tabellen
            h_lines = len(horizontal_lines)
            v_lines = len(vertical_lines)
            
            if h_lines > 3 and v_lines > 3:
                confidence = min(1.0, (h_lines + v_lines) / 20.0)
            else:
                confidence = 0.0
        
        return confidence > 0.3, confidence
    except Exception as e:
        logger.warning(f"Fehler bei der Tabellenprüfung: {str(e)}")
        return False, 0.0

def get_document_complexity(image_path_or_array) -> float:
    """
    Bewertet die Komplexität eines Dokuments.
    
    Args:
        image_path_or_array: Pfad zum Bild oder NumPy-Array
        
    Returns:
        Komplexitätswert zwischen 0 und 1
    """
    try:
        # Dokumentstruktur analysieren
        structure = analyze_document_structure(image_path_or_array)
        
        # Merkmale extrahieren
        text_regions = structure.get("text_regions", [])
        tables = structure.get("tables", [])
        images = structure.get("images", [])
        horizontal_lines = structure.get("horizontal_lines", [])
        vertical_lines = structure.get("vertical_lines", [])
        
        # Anzahl der Elemente
        num_text_regions = len(text_regions)
        num_tables = len(tables)
        num_images = len(images)
        num_h_lines = len(horizontal_lines)
        num_v_lines = len(vertical_lines)
        
        # Komplexität berechnen
        complexity = 0.0
        complexity += num_text_regions * 0.05  # Textregionen
        complexity += num_tables * 0.2         # Tabellen
        complexity += num_images * 0.1         # Bilder
        complexity += (num_h_lines + num_v_lines) * 0.01  # Linien
        
        # Normalisieren
        complexity = min(1.0, complexity)
        
        return complexity
    except Exception as e:
        logger.warning(f"Fehler bei der Komplexitätsbewertung: {str(e)}")
        return 0.5  # Mittlere Komplexität als Standardwert 