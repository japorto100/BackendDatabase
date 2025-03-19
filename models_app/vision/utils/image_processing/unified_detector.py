"""
Vereinheitlichter Manager für alle Objekterkennungsaufgaben in Dokumenten und Bildern.
"""
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
import cv2
from PIL import Image
import logging

from .core import convert_to_array
from .detection import detect_text_regions, detect_tables, detect_formulas, detect_images

logger = logging.getLogger(__name__)

class UnifiedDetector:
    """
    Einheitliche Klasse für Objekterkennung in Dokumenten und Bildern.
    Unterstützt Text, Tabellen, Bilder, Formeln und Layout-Elemente.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den UnifiedDetector.
        
        Args:
            config: Konfigurationsoptionen für die Detektoren
        """
        self.config = config or {}
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialisiert die spezifischen Detektoren basierend auf der Konfiguration."""
        # Hier könnten spezifische Modelle geladen werden, wenn nötig
        pass
    
    def detect_all(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Führt alle Erkennungsaufgaben auf dem Bild aus.
        
        Args:
            image: Das zu analysierende Bild
            
        Returns:
            Dict mit allen erkannten Elementen (Text, Tabellen, Bilder, Formeln, Layout)
        """
        img_array = convert_to_array(image)
        
        # Alle Erkennungsaufgaben ausführen
        results = {
            "text_regions": self.detect_text(img_array),
            "tables": self.detect_tables(img_array),
            "images": self.detect_images(img_array),
            "formulas": self.detect_formulas(img_array),
            "layout": self.analyze_layout(img_array),
            "form_elements": self.detect_form_elements(img_array)
        }
        
        return results
    
    def detect_text(self, image: Union[str, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """Erkennt Textregionen im Bild."""
        img_array = convert_to_array(image)
        return detect_text_regions(img_array)
    
    def detect_tables(self, image: Union[str, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """Erkennt Tabellen im Bild."""
        img_array = convert_to_array(image)
        return detect_tables(img_array)
    
    def detect_images(self, image: Union[str, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """Erkennt eingebettete Bilder im Dokument."""
        img_array = convert_to_array(image)
        return detect_images(img_array)
    
    def detect_formulas(self, image: Union[str, np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """Erkennt mathematische Formeln im Bild."""
        img_array = convert_to_array(image)
        return detect_formulas(img_array)
    
    def analyze_layout(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Analysiert das Layout des Dokuments."""
        img_array = convert_to_array(image)
        
        # Layout-Analyse-Logik
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
        h, w = gray.shape
        
        # Einfache Layout-Analyse basierend auf Bildregionen
        return {
            "pages": 1,
            "orientation": "portrait" if h > w else "landscape",
            "regions": [
                {"type": "header", "bbox": [0, 0, w, int(h*0.1)]},
                {"type": "content", "bbox": [0, int(h*0.1), w, int(h*0.9)]},
                {"type": "footer", "bbox": [0, int(h*0.9), w, h]}
            ]
        }
    
    def analyze_document_properties(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, float]:
        """
        Umfassende Dokumentenanalyse mit spezifischen OCR-relevanten Merkmalen.
        
        Args:
            image: Das zu analysierende Bild
            
        Returns:
            Dict mit Dokumenteigenschaften und deren Wahrscheinlichkeiten (0-1)
        """
        img_array = convert_to_array(image)
        
        # Grundlegende Dokumenteigenschaften
        properties = {
            "complexity": self._analyze_layout_complexity(img_array),
            "tables": len(self.detect_tables(img_array)) > 0,
            "equations": self._check_if_has_equations(img_array),
            "multilingual": self._check_if_multilingual(img_array),
            "academic": self._check_if_academic(img_array),
            "handwriting": self._check_if_has_handwriting(img_array)
        }
        
        return properties
    
    def _analyze_layout_complexity(self, image: np.ndarray) -> float:
        """
        Analysiert die Komplexität des Layouts.
        
        Args:
            image: Bild als NumPy-Array
            
        Returns:
            float: Komplexitätswert zwischen 0 und 1
        """
        try:
            # Bild in Graustufen konvertieren
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            h, w = gray.shape
            
            # Binarisieren und Kanten finden
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Normalisiere Konturenanzahl
            contour_complexity = min(1.0, len(contours) / 100)
            
            # Varianz in der Größe und Form der Konturen
            areas = []
            aspect_ratios = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = float(w) / h if h > 0 else 0
                
                areas.append(area)
                aspect_ratios.append(aspect_ratio)
                
            if areas:
                area_variance = np.var(areas) / (np.mean(areas) ** 2) if np.mean(areas) > 0 else 0
                aspect_variance = np.var(aspect_ratios) if aspect_ratios else 0
                
                # Normalisieren
                area_complexity = min(1.0, area_variance)
                shape_complexity = min(1.0, aspect_variance * 10)
            else:
                area_complexity = 0
                shape_complexity = 0
                
            # Überprüfen auf Mehrspaltigkeit
            vert_sum = np.sum(gray, axis=0)
            vert_sum = vert_sum / np.max(vert_sum) if np.max(vert_sum) > 0 else vert_sum
            
            # Zähle deutliche Übergänge als Hinweis auf Spalten
            transitions = 0
            threshold = 0.2
            for i in range(1, len(vert_sum)):
                if abs(vert_sum[i] - vert_sum[i-1]) > threshold:
                    transitions += 1
                    
            column_complexity = min(1.0, transitions / 30)
            
            # Kombination der Merkmale
            complexity = 0.4 * contour_complexity + 0.2 * area_complexity + 0.2 * shape_complexity + 0.2 * column_complexity
            
            return complexity
            
        except Exception as e:
            logger.warning(f"Fehler bei der Layout-Komplexitätsanalyse: {str(e)}")
            return 0.5  # Mittlerer Wert als Fallback
    
    def _check_if_has_equations(self, image: np.ndarray) -> float:
        """
        Prüft, ob das Bild mathematische Gleichungen enthält.
        
        Args:
            image: Bild als NumPy-Array
            
        Returns:
            float: Wahrscheinlichkeit für das Vorhandensein von Gleichungen (0-1)
        """
        try:
            # Bild in Graustufen konvertieren
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Binarisieren
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Konturen finden
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Zähler für relevante Merkmale
            small_contours = 0
            horizontal_lines = 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Kleine Konturen zählen (potenzielle mathematische Symbole)
                if 5 < w < 50 and 5 < h < 50:
                    small_contours += 1
                
                # Horizontale Linien erkennen (potenzielle Bruchstriche)
                if w > 3*h and w > 20:
                    horizontal_lines += 1
            
            # Berechne Wahrscheinlichkeit basierend auf Heuristiken
            small_contour_density = min(1.0, small_contours / 50)  # Normalisieren
            horizontal_line_factor = min(1.0, horizontal_lines / 5)  # Normalisieren
            
            # Kombinierte Wahrscheinlichkeit
            probability = 0.6 * small_contour_density + 0.4 * horizontal_line_factor
            
            return probability
        except Exception as e:
            logger.warning(f"Fehler bei der Erkennung von Gleichungen: {str(e)}")
            return 0.2  # Fallback-Wert
    
    def _check_if_has_handwriting(self, image: np.ndarray) -> float:
        """Prüft, ob das Dokument Handschrift enthält."""
        # Einfache Heuristik basierend auf Konturenvariabilität
        try:
            # Bild in Graustufen konvertieren
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Binarisieren
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Konturen finden
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) < 10:
                return 0.1
                
            # Analyse der Konturenform und -variabilität
            perimeters = []
            areas = []
            
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                if area > 0:
                    perimeters.append(perimeter)
                    areas.append(area)
            
            if not areas:
                return 0.1
                
            # Berechne Formfaktor (Verhältnis von Umfang zu Fläche)
            # Handschrift hat typischerweise unregelmäßigere Formen
            form_factors = [p**2 / (4 * np.pi * a) if a > 0 else 0 for p, a in zip(perimeters, areas)]
            
            # Variabilität der Formfaktoren
            if len(form_factors) > 1:
                form_variance = np.var(form_factors)
                # Hohe Varianz deutet auf Handschrift hin
                probability = min(1.0, form_variance * 2)
                return probability
            else:
                return 0.1
                
        except Exception as e:
            logger.warning(f"Fehler bei der Handschrifterkennung: {str(e)}")
            return 0.2
    
    def _check_if_multilingual(self, image: np.ndarray) -> float:
        """
        Prüft, ob das Dokument mehrsprachig ist durch Erkennung verschiedener Zeichensätze.
        
        Args:
            image: Bild als NumPy-Array
            
        Returns:
            float: Wahrscheinlichkeit für mehrsprachigen Inhalt (0-1)
        """
        # Vereinfachte Implementierung - in der Praxis würde hier eine
        # OCR-basierte Analyse mit Spracherkennung stattfinden
        return 0.2  # Standardwert
    
    def _check_if_academic(self, image: np.ndarray) -> float:
        """
        Prüft, ob es sich um ein akademisches Dokument handelt.
        
        Args:
            image: Bild als NumPy-Array
            
        Returns:
            float: Wahrscheinlichkeit für akademisches Dokument (0-1)
        """
        # Kombiniert verschiedene Merkmale akademischer Dokumente
        has_equations = self._check_if_has_equations(image)
        layout_complexity = self._analyze_layout_complexity(image)
        
        # Akademische Dokumente haben oft komplexe Layouts und Gleichungen
        probability = 0.4 * has_equations + 0.3 * layout_complexity + 0.3 * 0.5  # Basiswahrscheinlichkeit
        
        return probability

    def detect_form_elements(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Erkennt Formularelemente wie Checkboxen, Radiobuttons und Textfelder im Bild.
        
        Args:
            image: Das zu analysierende Bild
            
        Returns:
            Dict mit erkannten Formularelementen
        """
        img_array = convert_to_array(image)
        
        # In Graustufen konvertieren, falls nötig
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
        
        # Initialize result structure
        form_elements = {
            "has_form_elements": False,
            "form_type": "unknown",
            "elements": [],
            "confidence": 0.0
        }
        
        try:
            # Detect checkboxes
            checkboxes = self._detect_checkboxes(gray)
            if checkboxes:
                form_elements["elements"].extend(checkboxes)
            
            # Detect radio buttons
            radio_buttons = self._detect_radio_buttons(gray)
            if radio_buttons:
                form_elements["elements"].extend(radio_buttons)
            
            # Detect text fields
            text_fields = self._detect_text_fields(gray)
            if text_fields:
                form_elements["elements"].extend(text_fields)
            
            # Update form detection result
            if form_elements["elements"]:
                form_elements["has_form_elements"] = True
                form_elements["confidence"] = self._calculate_form_confidence(form_elements["elements"])
                form_elements["form_type"] = self._determine_form_type(form_elements["elements"])
            
            return form_elements
            
        except Exception as e:
            logger.error(f"Error detecting form elements: {str(e)}")
            return form_elements
    
    def _detect_checkboxes(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Erkennt Checkboxen in einem Graustufenbild.
        
        Args:
            gray_image: Graustufenbild als NumPy-Array
            
        Returns:
            Liste erkannter Checkboxen mit Positionen und Status
        """
        checkboxes = []
        
        # Schwellenwert anwenden
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Konturen finden
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Bounding rectangle ermitteln
            x, y, w, h = cv2.boundingRect(contour)
            
            # Prüfen, ob es quadratisch und in der richtigen Größe ist
            aspect_ratio = float(w) / h
            area = w * h
            
            # Checkboxen sind typischerweise quadratisch und klein
            if 0.8 < aspect_ratio < 1.2 and 100 < area < 2500:
                # Prüfen, ob es eine Checkbox ist (hat 4 Ecken)
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:
                    # Checkbox-Region extrahieren
                    checkbox_roi = binary[y:y+h, x:x+w]
                    
                    # Prüfen, ob sie gefüllt ist (angekreuzt)
                    total_pixels = checkbox_roi.shape[0] * checkbox_roi.shape[1]
                    filled_pixels = cv2.countNonZero(checkbox_roi)
                    fill_ratio = filled_pixels / total_pixels
                    
                    # Schwellenwert für angekreuzt/nicht angekreuzt
                    is_checked = fill_ratio > 0.2
                    
                    # Konfidenz basierend auf Form und Füllverhältnis berechnen
                    shape_confidence = 1.0 - abs(aspect_ratio - 1.0)
                    fill_confidence = 1.0 if (fill_ratio < 0.1 or fill_ratio > 0.3) else 0.5
                    confidence = (shape_confidence + fill_confidence) / 2
                    
                    checkboxes.append({
                        "type": "checkbox",
                        "bbox": (x, y, w, h),
                        "checked": is_checked,
                        "confidence": confidence,
                        "fill_ratio": fill_ratio
                    })
        
        return checkboxes
    
    def _detect_radio_buttons(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Erkennt Radiobuttons in einem Graustufenbild.
        
        Args:
            gray_image: Graustufenbild als NumPy-Array
            
        Returns:
            Liste erkannter Radiobuttons mit Positionen und Status
        """
        radio_buttons = []
        
        # Schwellenwert anwenden
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Konturen finden
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Bounding rectangle ermitteln
            x, y, w, h = cv2.boundingRect(contour)
            
            # Prüfen, ob es kreisförmig und in der richtigen Größe ist
            aspect_ratio = float(w) / h
            area = w * h
            
            # Radiobuttons sind kreisförmig und klein
            if 0.8 < aspect_ratio < 1.2 and 100 < area < 2500:
                # Kreisförmigkeit prüfen
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                if circularity > 0.7:  # Nahe an einem perfekten Kreis
                    # Radiobutton-Region extrahieren
                    radio_roi = binary[y:y+h, x:x+w]
                    
                    # Prüfen, ob er gefüllt ist (ausgewählt)
                    total_pixels = radio_roi.shape[0] * radio_roi.shape[1]
                    filled_pixels = cv2.countNonZero(radio_roi)
                    fill_ratio = filled_pixels / total_pixels
                    
                    # Schwellenwert für ausgewählt/nicht ausgewählt
                    is_selected = fill_ratio > 0.2
                    
                    # Konfidenz basierend auf Form und Füllverhältnis berechnen
                    shape_confidence = circularity
                    fill_confidence = 1.0 if (fill_ratio < 0.1 or fill_ratio > 0.3) else 0.5
                    confidence = (shape_confidence + fill_confidence) / 2
                    
                    radio_buttons.append({
                        "type": "radio",
                        "bbox": (x, y, w, h),
                        "selected": is_selected,
                        "confidence": confidence,
                        "fill_ratio": fill_ratio
                    })
        
        return radio_buttons
    
    def _detect_text_fields(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Erkennt Textfelder in einem Graustufenbild.
        
        Args:
            gray_image: Graustufenbild als NumPy-Array
            
        Returns:
            Liste erkannter Textfelder
        """
        text_fields = []
        
        # Schwellenwert anwenden
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphologische Operationen anwenden, um horizontale Linien zu finden
        kernel = np.ones((1, 50), np.uint8)
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Konturen finden
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Bounding rectangle ermitteln
            x, y, w, h = cv2.boundingRect(contour)
            
            # Seitenverhältnis und Größe prüfen
            aspect_ratio = float(w) / h
            
            # Textfelder sind typischerweise breite Rechtecke
            if aspect_ratio > 4 and w > 100:
                # Konfidenz basierend auf Form berechnen
                confidence = min(1.0, max(0.5, aspect_ratio / 10))
                
                text_fields.append({
                    "type": "text_field",
                    "bbox": (x, y, w, h),
                    "confidence": confidence
                })
        
        return text_fields
    
    def _calculate_form_confidence(self, elements: List[Dict[str, Any]]) -> float:
        """
        Berechnet die Gesamtkonfidenz für die Formularerkennung basierend auf den erkannten Elementen.
        
        Args:
            elements: Liste der erkannten Formularelemente
            
        Returns:
            Konfidenzwert zwischen 0 und 1
        """
        if not elements:
            return 0.0
        
        # Durchschnittliche Konfidenz der erkannten Elemente
        confidence_sum = sum(elem.get("confidence", 0.0) for elem in elements)
        avg_confidence = confidence_sum / len(elements)
        
        # Zusätzliche Konfidenz basierend auf der Anzahl der Elemente
        count_factor = min(1.0, len(elements) / 10.0)  # Max. Bonus bei 10+ Elementen
        
        # Gewichtete Kombination
        return 0.7 * avg_confidence + 0.3 * count_factor
    
    def _determine_form_type(self, elements: List[Dict[str, Any]]) -> str:
        """
        Bestimmt den Formulartyp basierend auf den erkannten Elementen.
        
        Args:
            elements: Liste der erkannten Formularelemente
            
        Returns:
            Formulartyp-Klassifikation
        """
        # Elementtypen zählen
        type_counts = {}
        for elem in elements:
            elem_type = elem.get("type", "unknown")
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
        
        # Keine Elemente
        if not type_counts:
            return "unknown"
        
        # Dominante Typen prüfen
        checkbox_count = type_counts.get("checkbox", 0)
        text_field_count = type_counts.get("text_field", 0)
        radio_count = type_counts.get("radio", 0)
        
        # Entscheidungslogik
        if checkbox_count > 5 and checkbox_count > text_field_count:
            return "checklist"
        elif text_field_count > 5 and text_field_count > checkbox_count:
            return "data_entry"
        elif radio_count > 3:
            return "questionnaire"
        elif text_field_count > 0 and checkbox_count > 0:
            return "mixed_form"
        else:
            return "simple_form"

# Singleton-Instanz für einfachen Zugriff
unified_detector = UnifiedDetector()