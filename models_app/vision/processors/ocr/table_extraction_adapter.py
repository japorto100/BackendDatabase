import os
import numpy as np
import logging
from PIL import Image
import cv2
import pandas as pd
from typing import Dict, List, Union, Any, Optional, Tuple
import io
import json
import re

try:
    import pytesseract
    from paddleocr import PPStructure, save_structure_res
    PADDLE_STRUCTURE_AVAILABLE = True
except ImportError:
    PADDLE_STRUCTURE_AVAILABLE = False

from .base_adapter import BaseOCRAdapter
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, ModelNotAvailableError
from models_app.vision.ocr.utils.plugin_system import register_adapter
from models_app.vision.utils.image_processing.core import load_image, convert_to_array
from models_app.vision.utils.image_processing.enhancement import enhance_for_table_extraction
from models_app.vision.utils.image_processing.detection import detect_tables, detect_lines
from models_app.vision.utils.image_processing.adapter_preprocess import preprocess_for_table_extraction

# Versuche, Table-Transformer zu importieren
try:
    import torch
    from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor
    TABLE_TRANSFORMER_AVAILABLE = True
except ImportError:
    TABLE_TRANSFORMER_AVAILABLE = False
    torch = None
    TableTransformerForObjectDetection = None
    DetrFeatureExtractor = None

# Versuche, Tabulizer zu importieren
try:
    import cv2
    import pandas as pd
    TABULIZER_AVAILABLE = True
except ImportError:
    TABULIZER_AVAILABLE = False
    cv2 = None
    pd = None

from models_app.vision.utils.testing.dummy_models import DummyModelFactory

logger = logging.getLogger(__name__)

@register_adapter(name="table_extraction", info={
    "description": "Table Extraction Adapter for detecting and parsing tables in documents",
    "version": "1.0.0",
    "capabilities": {
        "multi_language": False,
        "handwriting": False,
        "table_extraction": True,
        "formula_recognition": False,
        "document_understanding": False
    },
    "priority": 65
})
class TableExtractionAdapter(BaseOCRAdapter):
    """Adapter für Tabellenextraktion aus Dokumenten."""
    
    ADAPTER_NAME = "table_extraction"
    ADAPTER_INFO = {
        "description": "Table Extraction Adapter for detecting and parsing tables in documents",
        "version": "1.0.0",
        "capabilities": {
            "multi_language": False,
            "handwriting": False,
            "table_extraction": True,
            "formula_recognition": False,
            "document_understanding": False
        },
        "priority": 65
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den TableExtraction-Adapter.
        
        Args:
            config: Konfiguration für den Tabellenextraktionsadapter
        """
        super().__init__(config)
        
        # Standardkonfiguration
        self.model_name = self.config.get('model_name', 'microsoft/table-transformer-detection')
        self.structure_model_name = self.config.get('structure_model_name', 'microsoft/table-transformer-structure-recognition')
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu') if TABLE_TRANSFORMER_AVAILABLE and torch else 'cpu'
        self.threshold = self.config.get('threshold', 0.7)
        self.max_tables = self.config.get('max_tables', 5)
        
        # Erkennungsmethode ('transformer' oder 'opencv')
        self.detection_method = self.config.get('detection_method', 'transformer')
        
        # Parameter für OpenCV-Methode
        self.opencv_line_scale = self.config.get('opencv_line_scale', 15)
        self.opencv_line_threshold = self.config.get('opencv_line_threshold', 150)
        
        # Feature Extractor und Modelle
        self.feature_extractor = None
        self.detection_model = None
        self.structure_model = None
        self.ocr_adapter = None
    
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert die Tabellenextraktionsmodelle.
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if self.detection_method == 'transformer' and not TABLE_TRANSFORMER_AVAILABLE:
            raise ModelNotAvailableError("Table-Transformer ist nicht installiert.")
            
        if self.detection_method == 'opencv' and not TABULIZER_AVAILABLE:
            raise ModelNotAvailableError("OpenCV und/oder Pandas sind nicht installiert.")
            
        try:
            if self.detection_method == 'transformer':
                # Feature Extractor initialisieren
                self.feature_extractor = DetrFeatureExtractor.from_pretrained(self.model_name)
                
                # Tabellenerkennungsmodell initialisieren
                self.detection_model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
                self.detection_model.to(self.device)
                
                # Tabellenstrukturerkennungsmodell initialisieren (optional)
                if self.structure_model_name:
                    self.structure_model = TableTransformerForObjectDetection.from_pretrained(self.structure_model_name)
                    self.structure_model.to(self.device)
                
                logger.info(f"Table-Transformer initialisiert mit Modell: {self.model_name} auf Gerät: {self.device}")
            else:
                # OpenCV-basierte Methode benötigt keine spezielle Initialisierung
                logger.info("OpenCV-basierte Tabellenextraktion initialisiert")
            
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung der Tabellenextraktion: {str(e)}")
            self.is_initialized = False
            return False
    
    def _initialize_dummy_model(self):
        """Initialisiert ein Dummy-Modell für Tabellen-Extraktion."""
        dummy = DummyModelFactory.create_ocr_dummy("table_extraction")
        self.feature_extractor = dummy.get("feature_extractor")
        self.model = dummy.get("model")
    
    def _detect_tables_transformer(self, image: Union[np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """
        Erkennt Tabellen in einem Bild mit Table-Transformer.
        
        Args:
            image: Bild als NumPy-Array oder PIL-Bild
        
        Returns:
            Liste von erkannten Tabellen mit Bounding-Boxen
        """
        # Sicherstellen, dass wir ein PIL-Bild haben
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Feature Extraction
        inputs = self.feature_extractor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Tabellenerkennung durchführen
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        
        # Post-Processing
        width, height = image_pil.size
        results = self.feature_extractor.post_process_object_detection(
            outputs, 
            threshold=self.threshold, 
            target_sizes=[(height, width)]
        )[0]
        
        # Erkannte Tabellen extrahieren
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i) for i in box.tolist()]
            label_name = self.detection_model.config.id2label[label.item()]
            
            if label_name in ["table", "table rotated"]:
                tables.append({
                    "bbox": box,  # [x1, y1, x2, y2]
                    "score": score.item(),
                    "label": label_name,
                    "rotated": label_name == "table rotated"
                })
        
        # Sortieren nach Konfidenz und limitieren
        tables = sorted(tables, key=lambda x: x["score"], reverse=True)[:self.max_tables]
        
        return tables
    
    def _detect_tables_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect tables using OpenCV methods."""
        try:
            # Use the utility function first
            tables = detect_tables(image)
            if tables:
                return tables
        except Exception as e:
            self.logger.warning(f"Falling back to custom table detection: {str(e)}")
            
        # If utility fails or returns no tables, use custom implementation
        processed_image = enhance_for_table_extraction(image)
        
        # Graustufenbild
        if len(processed_image.shape) == 3:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed_image
        
        # Binarisierung
        thresh, binary = cv2.threshold(gray, self.opencv_line_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Dilatation für horizontale und vertikale Linien
        kernel_length = np.array(binary).shape[1] // self.opencv_line_scale
        
        # Horizontale Linien
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        horizontal_lines = cv2.erode(binary, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel)
        
        # Vertikale Linien
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        vertical_lines = cv2.erode(binary, vertical_kernel)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel)
        
        # Kombinieren der Linien
        table_grid = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Konturen finden
        contours, hierarchy = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Tabellenbereiche identifizieren
        tables = []
        for i, contour in enumerate(contours):
            # Bounding-Box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter für kleine Konturen
            if w < 50 or h < 50:
                continue
            
            tables.append({
                "bbox": [x, y, x + w, y + h],
                "score": 0.9,  # Dummy-Score
                "label": "table",
                "rotated": False
            })
        
        return tables[:self.max_tables]
    
    def _analyze_table_structure(self, image: np.ndarray, table_bbox: List[int]) -> Dict[str, Any]:
        """
        Analysiert die Struktur einer erkannten Tabelle.
        
        Args:
            image: Bild als NumPy-Array
            table_bbox: Bounding-Box der Tabelle [x1, y1, x2, y2]
            
        Returns:
            Dict mit Tabellenstruktur (Zeilen, Spalten, Zellen)
        """
        # Tabelle aus dem Bild ausschneiden
        x1, y1, x2, y2 = table_bbox
        table_image = image[y1:y2, x1:x2]
        
        if self.structure_model is not None:
            # Transformer-basierte Strukturerkennung
            # Konvertiere zu PIL-Bild
            table_pil = Image.fromarray(table_image)
            
            # Feature Extraction
            inputs = self.feature_extractor(images=table_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Strukturerkennung durchführen
            with torch.no_grad():
                outputs = self.structure_model(**inputs)
            
            # Post-Processing
            width, height = table_pil.size
            results = self.feature_extractor.post_process_object_detection(
                outputs, 
                threshold=self.threshold, 
                target_sizes=[(height, width)]
            )[0]
            
            # Erkannte Strukturelemente (Zellen, Überschriften, etc.)
            cells = []
            rows = []
            columns = []
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i) for i in box.tolist()]
                label_name = self.structure_model.config.id2label[label.item()]
                
                element = {
                    "bbox": box,  # [x1, y1, x2, y2]
                    "score": score.item(),
                    "label": label_name
                }
                
                if "column" in label_name:
                    columns.append(element)
                elif "row" in label_name:
                    rows.append(element)
                elif "cell" in label_name or "spanning" in label_name:
                    cells.append(element)
            
            # Sortieren und indizieren
            columns = sorted(columns, key=lambda x: x["bbox"][0])
            rows = sorted(rows, key=lambda x: x["bbox"][1])
            
            # Rückgabe der Struktur
            return {
                "cells": cells,
                "rows": rows,
                "columns": columns,
                "num_rows": len(rows),
                "num_columns": len(columns)
            }
        else:
            # OpenCV-basierte Strukturerkennung
            # Graustufenbild
            if len(table_image.shape) == 3:
                gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = table_image
            
            # Binarisierung
            thresh, binary = cv2.threshold(gray, self.opencv_line_threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Dilatation für horizontale und vertikale Linien
            kernel_length = np.array(binary).shape[1] // self.opencv_line_scale
            
            # Horizontale Linien
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
            horizontal_lines = cv2.erode(binary, horizontal_kernel)
            horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel)
            
            # Vertikale Linien
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
            vertical_lines = cv2.erode(binary, vertical_kernel)
            vertical_lines = cv2.dilate(vertical_lines, vertical_kernel)
            
            # Horizontale und vertikale Linien finden
            h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            # Zeilen- und Spaltenkoordinaten extrahieren
            row_positions = set()
            col_positions = set()
            
            if h_lines is not None:
                for line in h_lines:
                    x1, y1, x2, y2 = line[0]
                    row_positions.add(y1)
            
            if v_lines is not None:
                for line in v_lines:
                    x1, y1, x2, y2 = line[0]
                    col_positions.add(x1)
            
            # Sortieren
            row_positions = sorted(row_positions)
            col_positions = sorted(col_positions)
            
            # Rückgabe der Struktur
            return {
                "row_positions": row_positions,
                "col_positions": col_positions,
                "num_rows": len(row_positions) - 1 if len(row_positions) > 1 else 0,
                "num_columns": len(col_positions) - 1 if len(col_positions) > 1 else 0
            }
    
    def _extract_table_content(self, image: np.ndarray, table_structure: Dict[str, Any], ocr_adapter=None) -> pd.DataFrame:
        """
        Extrahiert den Inhalt einer Tabelle mit der erkannten Struktur.
        
        Args:
            image: Bild als NumPy-Array
            table_structure: Struktur der Tabelle
            ocr_adapter: OCR-Adapter für die Textextraktion (optional)
        
        Returns:
            DataFrame mit dem Tabelleninhalt
        """
        if not pd:
            return None
            
        # OCR auf Zellen durchführen, je nach verfügbarer Strukturinformation
        if "cells" in table_structure:
            # Mit identifizierten Zellen
            cells_data = []
            for cell in table_structure["cells"]:
                x1, y1, x2, y2 = cell["bbox"]
                cell_image = image[y1:y2, x1:x2]
                
                # Zelltext mit OCR extrahieren
                if ocr_adapter:
                    try:
                        ocr_result = ocr_adapter.process_image(cell_image)
                        cell_text = ocr_result.get('text', '').strip()
                    except Exception:
                        cell_text = ""
                else:
                    # Einfache Tesseract-Erkennung über OpenCV
                    try:
                        import pytesseract
                        cell_text = pytesseract.image_to_string(cell_image).strip()
                    except Exception:
                        cell_text = ""
                
                # Position bestimmen
                row_idx = -1
                col_idx = -1
                
                for i, row in enumerate(table_structure["rows"]):
                    if y1 >= row["bbox"][1] and y2 <= row["bbox"][3]:
                        row_idx = i
                        break
                
                for i, col in enumerate(table_structure["columns"]):
                    if x1 >= col["bbox"][0] and x2 <= col["bbox"][2]:
                        col_idx = i
                        break
                
                cells_data.append({
                    "row": row_idx,
                    "column": col_idx,
                    "text": cell_text,
                    "bbox": [x1, y1, x2, y2]
                })
            
            # Zu DataFrame konvertieren
            if cells_data:
                # Dataframe vorbereiten
                max_row = max(cell["row"] for cell in cells_data if cell["row"] >= 0) + 1
                max_col = max(cell["column"] for cell in cells_data if cell["column"] >= 0) + 1
                
                # Leeres DataFrame erstellen
                df = pd.DataFrame(index=range(max_row), columns=range(max_col))
                
                # Zellen füllen
                for cell in cells_data:
                    if cell["row"] >= 0 and cell["column"] >= 0:
                        df.iloc[cell["row"], cell["column"]] = cell["text"]
                
                return df
            
            return pd.DataFrame()
            
        elif "row_positions" in table_structure and "col_positions" in table_structure:
            # Mit erkannten Linien
            row_pos = table_structure["row_positions"]
            col_pos = table_structure["col_positions"]
            
            if len(row_pos) < 2 or len(col_pos) < 2:
                return pd.DataFrame()
            
            # Dataframe vorbereiten
            df = pd.DataFrame(index=range(len(row_pos)-1), columns=range(len(col_pos)-1))
            
            # Für jede Zelle
            for i in range(len(row_pos)-1):
                for j in range(len(col_pos)-1):
                    # Zellenkoordinaten
                    x1, y1 = col_pos[j], row_pos[i]
                    x2, y2 = col_pos[j+1], row_pos[i+1]
                    
                    # Zellbild extrahieren
                    cell_image = image[y1:y2, x1:x2]
                    
                    # Zelltext mit OCR extrahieren
                    if ocr_adapter:
                        try:
                            ocr_result = ocr_adapter.process_image(cell_image)
                            cell_text = ocr_result.get('text', '').strip()
                        except Exception:
                            cell_text = ""
                    else:
                        # Einfache Tesseract-Erkennung über OpenCV
                        try:
                            import pytesseract
                            cell_text = pytesseract.image_to_string(cell_image).strip()
                        except Exception:
                            cell_text = ""
                    
                    # In DataFrame eintragen
                    df.iloc[i, j] = cell_text
            
            return df
        
        return pd.DataFrame()
    
    @handle_ocr_errors
    def process_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet ein Bild und extrahiert Tabellen.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
        
        Returns:
            Dictionary mit extrahierten Tabellen
        """
        options = options or {}
        
        # Sicherstellen, dass Tabellenextraktion initialisiert ist
        if not self.is_initialized:
            if self.config.get('dummy_mode', False):
                self._initialize_dummy_model()
            else:
                self.initialize()
        
        # Optionen extrahieren
        detection_method = options.get('detection_method', self.detection_method)
        threshold = options.get('threshold', self.threshold)
        max_tables = options.get('max_tables', self.max_tables)
        analyze_structure = options.get('analyze_structure', True)
        extract_content = options.get('extract_content', True)
        ocr_adapter = options.get('ocr_adapter', self.ocr_adapter)
        
        # Prüfen, ob das Bild bereits vorverarbeitet wurde
        already_preprocessed = options.get('already_preprocessed', False)
        preprocess = options.get('preprocess', True) and not already_preprocessed
        
        # Bild laden
        from models_app.vision.utils.image_processing.core import load_image
        _, np_image, _ = load_image(image_path_or_array)
        
        # Bild vorverarbeiten
        if preprocess:
            image = self.preprocess_image(np_image, options)
        else:
            image = np_image
        
        # Tabellen erkennen
        if detection_method == 'transformer' and TABLE_TRANSFORMER_AVAILABLE:
            tables = self._detect_tables_transformer(image)
        else:
            tables = self._detect_tables_opencv(image)
        
        # Tabellenstruktur und Inhalt analysieren
        if analyze_structure:
            for table in tables:
                # Struktur analysieren
                table['structure'] = self._analyze_table_structure(image, table['bbox'])
                
                # Inhalt extrahieren, wenn gewünscht
                if extract_content and pd:
                    df = self._extract_table_content(image, table['structure'], ocr_adapter)
                    
                    if df is not None:
                        # DataFrame zu HTML konvertieren
                        table['html'] = df.to_html(index=False, header=False)
                        
                        # DataFrame zu CSV konvertieren
                        table['csv'] = df.to_csv(index=False, header=False)
                        
                        # DataFrame zu JSON konvertieren
                        table['data'] = json.loads(df.to_json(orient='values'))
        
        # Gesamtergebnis erstellen
        full_text = f"Gefundene Tabellen: {len(tables)}"
        
        # Ergebnis erstellen
        result = self._create_result(
            text=full_text,
            blocks=[{
                'text': f"Tabelle {i+1}",
                'conf': table['score'],
                'bbox': table['bbox'],
                'type': 'table',
                'table': table
            } for i, table in enumerate(tables)],
            confidence=sum(table['score'] for table in tables) / len(tables) if tables else 0.0,
            language="any",
            metadata={
                'detection_method': detection_method,
                'threshold': threshold,
                'max_tables': max_tables,
                'tables_found': len(tables)
            }
        )
        
        # Post-Processing
        result = self.postprocess_result(result, options)
        
        return result
    
    def get_supported_languages(self) -> List[str]:
        """
        Gibt die unterstützten Sprachen zurück.
        
        Returns:
            Liste der unterstützten Sprachen
        """
        # Tabellenextraktion ist sprachunabhängig
        return ["any"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen zum Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        info = super().get_model_info()
        
        if self.is_initialized:
            info.update({
                "type": "Table Extraction",
                "detection_method": self.detection_method,
                "model_name": self.model_name if self.detection_method == 'transformer' else "OpenCV-based",
                "structure_model_name": self.structure_model_name if self.detection_method == 'transformer' else None,
                "device": self.device if self.detection_method == 'transformer' else "CPU",
                "threshold": self.threshold,
                "max_tables": self.max_tables
            })
                
        return info
    
    def set_ocr_adapter(self, adapter: 'BaseOCRAdapter') -> None:
        """
        Setzt einen OCR-Adapter für die Textextraktion aus Tabellenzellen.
        
        Args:
            adapter: OCR-Adapter
        """
        self.ocr_adapter = adapter 

    def preprocess_image(self, image_path_or_array, options: Dict[str, Any] = None) -> np.ndarray:
        """
        Vorverarbeitung des Bildes für die Tabellenextraktion.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Optionen für die Vorverarbeitung
        
        Returns:
            np.ndarray: Vorverarbeitetes Bild
        """
        options = options or {}
        # Tabellen-spezifische Optionen
        options["enhance_contrast"] = True
        options["sharpen"] = True
        options["binarize_for_lines"] = self.config.get("binarize_for_lines", False)
        
        # Zentralisierte Vorverarbeitung verwenden
        return preprocess_for_table_extraction(image_path_or_array, options)