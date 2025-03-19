"""
Adapter für Erkennung mathematischer Formeln.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image
import re

from .base_adapter import BaseOCRAdapter
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, ModelNotAvailableError
from models_app.vision.ocr.utils.plugin_system import register_adapter
from models_app.vision.utils.image_processing.enhancement import enhance_for_formula_recognition
from models_app.vision.utils.image_processing.detection import detect_formulas
from models_app.vision.utils.image_processing.core import load_image, convert_to_array, convert_to_pil
from analytics_app.utils import monitor_ocr_performance
from models_app.vision.utils.testing.dummy_models import DummyModelFactory
from models_app.vision.utils.image_processing.adapter_preprocess import preprocess_for_formula_recognition

# Versuche, LaTeX-OCR oder andere Modelle zu importieren
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    FORMULA_OCR_AVAILABLE = True
except ImportError:
    FORMULA_OCR_AVAILABLE = False
    torch = None
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None

# Versuche, pix2tex zu importieren
try:
    import pix2tex
    PIX2TEX_AVAILABLE = True
except ImportError:
    PIX2TEX_AVAILABLE = False
    pix2tex = None

logger = logging.getLogger(__name__)

@register_adapter(name="formula_recognition", info={
    "description": "Formula Recognition Adapter for detecting mathematical formulas",
    "version": "1.0.0",
    "capabilities": {
        "multi_language": False,
        "handwriting": True,
        "table_extraction": False,
        "formula_recognition": True,
        "document_understanding": False
    },
    "priority": 60
})
class FormulaRecognitionAdapter(BaseOCRAdapter):
    """Adapter für Erkennung mathematischer Formeln."""
    
    ADAPTER_NAME = "formula_recognition"
    ADAPTER_INFO = {
        "description": "Formula Recognition Adapter for detecting mathematical formulas",
        "version": "1.0.0",
        "capabilities": {
            "multi_language": False,
            "handwriting": True,
            "table_extraction": False,
            "formula_recognition": True,
            "document_understanding": False
        },
        "priority": 60
    }
    
    # Verfügbare Methoden für Formelerkennung
    AVAILABLE_METHODS = ["pix2tex", "trocr", "nougat"]
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den FormulaRecognition-Adapter.
        
        Args:
            config: Konfiguration für die Formelerkennung
        """
        super().__init__(config)
        
        # Standardkonfiguration
        self.method = self.config.get('method', 'pix2tex' if PIX2TEX_AVAILABLE else 'trocr')
        self.model_name = self.config.get('model_name', None)
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu') if FORMULA_OCR_AVAILABLE and torch else 'cpu'
        self.use_segmentation = self.config.get('use_segmentation', True)
        self.max_width = self.config.get('max_width', 800)
        self.max_height = self.config.get('max_height', 800)
        
        # Modellspezifische Parameter
        if self.method == 'pix2tex':
            if not self.model_name:
                self.model_name = 'pix2tex/pix2tex'
        elif self.method == 'trocr':
            if not self.model_name:
                self.model_name = 'microsoft/trocr-base-handwritten'
        elif self.method == 'nougat':
            if not self.model_name:
                self.model_name = 'facebook/nougat-base'
        
        # Modell, Prozessor und Segmentierungsmodell
        self.model = None
        self.processor = None
        self.segmentation_model = None
    
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert das Formelerkennungsmodell.
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if self.method == 'pix2tex' and not PIX2TEX_AVAILABLE:
            raise ModelNotAvailableError("pix2tex ist nicht installiert.")
            
        if (self.method == 'trocr' or self.method == 'nougat') and not FORMULA_OCR_AVAILABLE:
            raise ModelNotAvailableError(f"{self.method} Abhängigkeiten sind nicht installiert.")
            
        try:
            if self.method == 'pix2tex':
                # Pix2Tex-Modell initialisieren
                from pix2tex.cli import LatexOCR
                self.model = LatexOCR()
                
                logger.info("pix2tex-Modell für Formelerkennung initialisiert")
                self.is_initialized = True
                
            elif self.method == 'trocr':
                # TrOCR-Prozessor initialisieren
                self.processor = TrOCRProcessor.from_pretrained(self.model_name)
                
                # TrOCR-Modell initialisieren
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                
                logger.info(f"TrOCR-Modell für Formelerkennung initialisiert: {self.model_name} auf Gerät: {self.device}")
                self.is_initialized = True
                
            elif self.method == 'nougat':
                # Nougat importieren
                from transformers import NougatProcessor, VisionEncoderDecoderModel
                
                # Nougat-Prozessor initialisieren
                self.processor = NougatProcessor.from_pretrained(self.model_name)
                
                # Nougat-Modell initialisieren
                self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                
                logger.info(f"Nougat-Modell für Formelerkennung initialisiert: {self.model_name} auf Gerät: {self.device}")
                self.is_initialized = True
                
            # Segmentierungsmodell initialisieren, wenn aktiviert
            if self.use_segmentation:
                try:
                    import detectron2
                    from detectron2.config import get_cfg
                    from detectron2.engine import DefaultPredictor
                    from detectron2 import model_zoo
                    
                    # Konfiguration für Segmentierung
                    cfg = get_cfg()
                    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
                    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
                    cfg.MODEL.DEVICE = self.device
                    
                    self.segmentation_model = DefaultPredictor(cfg)
                    logger.info("Segmentierungsmodell für Formelerkennung initialisiert")
                    
                except ImportError:
                    logger.warning("detectron2 ist nicht installiert. Formelsegmentierung wird deaktiviert.")
                    self.use_segmentation = False
                
            return True
                
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung der Formelerkennung: {str(e)}")
            self.is_initialized = False
            return False
    
    def _initialize_dummy_model(self):
        """Initialisiert ein Dummy-Modell für Formel-Erkennung."""
        dummy = DummyModelFactory.create_ocr_dummy("formula")
        self.pix2tex_model = dummy.get("pix2tex")
        self.trocr_model = dummy.get("trocr")
        self.trocr_processor = dummy.get("trocr_processor")
        self.nougat_model = dummy.get("nougat")
        self.nougat_processor = dummy.get("nougat_processor")
    
    def _segment_formulas(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segmentiert mathematische Formeln im Bild.
        
        Args:
            image: Bild als NumPy-Array
        
        Returns:
            Liste von erkannten Formelregionen mit Bounding-Boxen
        """
        if not self.use_segmentation or self.segmentation_model is None:
            # Wenn keine Segmentierung, das ganze Bild verwenden
            height, width = image.shape[:2]
            return [{
                'bbox': [0, 0, width, height],
                'score': 1.0,
                'image': image
            }]
        
        try:
            # Segmentierung durchführen
            outputs = self.segmentation_model(image)
            
            # Interessante Klassen: Whiteboard, Dokument, Text, etc.
            interesting_classes = [0, 73, 77]  # person, laptop, book
            
            instances = outputs["instances"]
            formulas = []
            
            if len(instances) > 0:
                boxes = instances.pred_boxes if hasattr(instances, "pred_boxes") else None
                scores = instances.scores if hasattr(instances, "scores") else None
                classes = instances.pred_classes if hasattr(instances, "pred_classes") else None
                
                if boxes is not None:
                    for i in range(len(boxes)):
                        if classes is None or classes[i] in interesting_classes:
                            box = boxes[i].tensor.cpu().numpy()[0]
                            score = float(scores[i].cpu().numpy()) if scores is not None else 1.0
                            
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Bild ausschneiden
                            formula_image = image[y1:y2, x1:x2]
                            
                            formulas.append({
                                'bbox': [x1, y1, x2, y2],
                                'score': score,
                                'image': formula_image
                            })
            
            # Wenn keine Formeln gefunden wurden, das ganze Bild verwenden
            if not formulas:
                height, width = image.shape[:2]
                formulas.append({
                    'bbox': [0, 0, width, height],
                    'score': 1.0,
                    'image': image
                })
                
            return formulas
            
        except Exception as e:
            logger.warning(f"Fehler bei der Formelsegmentierung: {str(e)}")
            # Fallback: das ganze Bild verwenden
            height, width = image.shape[:2]
            return [{
                'bbox': [0, 0, width, height],
                'score': 1.0,
                'image': image
            }]
    
    def _preprocess_formula_image(self, image: np.ndarray) -> Image.Image:
        """
        Vorverarbeitung des Bildes für die Formelerkennung.
        
        Args:
            image: NumPy-Array des Bildes
            
        Returns:
            Image.Image: Vorverarbeitetes PIL-Bild
        """
        options = {
            "enhance_contrast": True,
            "deskew": True,
            "sharpen": True,
            "sharpen_strength": 2.0
        }
        
        # Zentralisierte Vorverarbeitung verwenden
        processed_img = preprocess_for_formula_recognition(image, options)
        
        # Konvertiere zurück zu PIL für die Modelle
        return Image.fromarray(processed_img)
    
    def _recognize_formula_pix2tex(self, image_pil: Image.Image) -> str:
        """
        Erkennt eine Formel mit pix2tex.
        
        Args:
            image_pil: Bild als PIL-Image
            
        Returns:
            Erkannte LaTeX-Formel als String
        """
        try:
            # pix2tex direkt aufrufen
            latex = self.model(image_pil)
            return latex
        except Exception as e:
            logger.error(f"Fehler bei der Formelerkennung mit pix2tex: {str(e)}")
            return ""
    
    def _recognize_formula_trocr(self, image_pil: Image.Image) -> str:
        """
        Erkennt eine Formel mit TrOCR.
        
        Args:
            image_pil: Bild als PIL-Image
        
        Returns:
            Erkannte LaTeX-Formel als String
        """
        try:
            # Bild vorverarbeiten
            pixel_values = self.processor(image_pil, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generierung durchführen
            with torch.no_grad():
                output_ids = self.model.generate(pixel_values)
            
            # Ausgabe dekodieren
            latex = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            return latex
        except Exception as e:
            logger.error(f"Fehler bei der Formelerkennung mit TrOCR: {str(e)}")
            return ""
    
    def _recognize_formula_nougat(self, image_pil: Image.Image) -> str:
        """
        Erkennt eine Formel mit Nougat.
        
        Args:
            image_pil: Bild als PIL-Image
        
        Returns:
            Erkannte LaTeX-Formel als String
        """
        try:
            # Bild vorverarbeiten
            inputs = self.processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generierung durchführen
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs.pixel_values,
                    max_length=4096,
                    early_stopping=True
                )
            
            # Ausgabe dekodieren
            latex = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            
            # Extrahiere LaTeX-Formeln aus dem Text
            formulas = []
            
            # Equation-Umgebungen suchen
            equation_pattern = r'\\begin\{equation\}(.*?)\\end\{equation\}'
            equations = re.findall(equation_pattern, latex, re.DOTALL)
            formulas.extend(equations)
            
            # Inline-Math suchen
            inline_pattern = r'\$(.*?)\$'
            inlines = re.findall(inline_pattern, latex)
            formulas.extend(inlines)
            
            # Wenn Formeln gefunden wurden, verwende diese, sonst den gesamten Text
            if formulas:
                return " ".join(formulas)
            else:
                return latex
                
        except Exception as e:
            logger.error(f"Fehler bei der Formelerkennung mit Nougat: {str(e)}")
            return ""
    
    def preprocess_image(self, image_path_or_array, options: Dict[str, Any] = None) -> np.ndarray:
        """
        Vorverarbeitung des Bildes für die Formelerkennung.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Optionen für die Vorverarbeitung
            
        Returns:
            np.ndarray: Vorverarbeitetes Bild
        """
        options = options or {}
        # Formel-spezifische Optionen
        options["enhance_contrast"] = self.config.get("enhance_contrast", True)
        options["sharpen"] = self.config.get("sharpen", True)
        options["sharpen_strength"] = self.config.get("sharpen_strength", 2.0)
        
        # Zentralisierte Vorverarbeitung verwenden
        return preprocess_for_formula_recognition(image_path_or_array, options)
    
    @monitor_ocr_performance
    @handle_ocr_errors
    def process_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet ein Bild und erkennt mathematische Formeln.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
        
        Returns:
            Dictionary mit erkannten Formeln
        """
        options = options or {}
        
        # Sicherstellen, dass Formelerkennung initialisiert ist
        if not self.is_initialized:
            if self.config.get('dummy_mode', False):
                self._initialize_dummy_model()
            else:
                self.initialize()
        
        # Optionen extrahieren
        method = options.get('method', self.method)
        use_segmentation = options.get('use_segmentation', self.use_segmentation)
        max_width = options.get('max_width', self.max_width)
        max_height = options.get('max_height', self.max_height)
        
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
        
        # Formeln segmentieren
        formula_regions = detect_formulas(image)
        
        # Ergebnisse für jede Region
        formula_results = []
        
        for region in formula_regions:
            # Bild der Region extrahieren
            region_image = region['image']
            
            # Bild für Formelerkennung vorverarbeiten
            pil_image = self._preprocess_formula_image(region_image)
            
            # Formel erkennen
            if method == 'pix2tex':
                latex = self._recognize_formula_pix2tex(pil_image)
            elif method == 'trocr':
                latex = self._recognize_formula_trocr(pil_image)
            elif method == 'nougat':
                latex = self._recognize_formula_nougat(pil_image)
            else:
                latex = ""
            
            # Ergebnis hinzufügen
            formula_results.append({
                'latex': latex,
                'bbox': region['bbox'],
                'score': region['score']
            })
        
        # Text- und LaTeX-Repräsentation erstellen
        full_text = "\n".join([fr['latex'] for fr in formula_results])
        
        # Blocks für das Ergebnis erstellen
        blocks = []
        for i, fr in enumerate(formula_results):
            block = {
                'text': fr['latex'],
                'latex': fr['latex'],
                'conf': fr['score'],
                'bbox': fr['bbox'],
                'type': 'formula',
                'formula_index': i
            }
            blocks.append(block)
        
        # Ergebnis erstellen
        result = self._create_result(
            text=full_text,
            blocks=blocks,
            confidence=sum(fr['score'] for fr in formula_results) / len(formula_results) if formula_results else 0.0,
            language="latex",
            metadata={
                'method': method,
                'use_segmentation': use_segmentation,
                'model_name': self.model_name,
                'formulas_found': len(formula_results)
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
        # Formelerkennung verwendet LaTeX, was sprachunabhängig ist
        return ["latex"]
    
    def latex_to_image(self, latex: str, options: Dict[str, Any] = None) -> np.ndarray:
        """
        Konvertiert eine LaTeX-Formel in ein Bild.
        
        Args:
            latex: LaTeX-Formel
            options: Zusätzliche Optionen für die Konvertierung
        
        Returns:
            Bild als NumPy-Array
        """
        options = options or {}
        
        # Optionen extrahieren
        dpi = options.get('dpi', 300)
        fontsize = options.get('fontsize', 12)
        color = options.get('color', 'black')
        
        try:
            # matplotlib für Formelrendering verwenden
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib import rc
            
            # LaTeX-Setup
            rc('text', usetex=True)
            rc('font', family='serif')
            
            # Figur erstellen
            fig = plt.figure(figsize=(1, 1), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            
            # Formel rendern
            ax.text(0.5, 0.5, f"${latex}$", fontsize=fontsize, color=color,
                   horizontalalignment='center', verticalalignment='center')
            
            # In Bild konvertieren
            fig.canvas.draw()
            
            # Figur zu NumPy-Array
            data = np.array(fig.canvas.renderer.buffer_rgba())
            
            # Figur schließen
            plt.close(fig)
            
            return data
            
        except Exception as e:
            logger.error(f"Fehler bei der Konvertierung von LaTeX zu Bild: {str(e)}")
            # Leeres Bild zurückgeben
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen zum Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        info = super().get_model_info()
        
        if self.is_initialized:
            info.update({
                "type": "Formula Recognition",
                "method": self.method,
                "model_name": self.model_name,
                "device": self.device,
                "use_segmentation": self.use_segmentation,
                "available_methods": self.AVAILABLE_METHODS
            })
                
        return info 