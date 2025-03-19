"""
Adapter für LayoutLMv3 für Document Understanding.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image

from .base_adapter import BaseOCRAdapter
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, ModelNotAvailableError
from models_app.vision.ocr.utils.plugin_system import register_adapter
from models_app.vision.utils.image_processing.detection import detect_text_regions
from models_app.vision.utils.image_processing.visualization import draw_bounding_boxes
from analytics_app.utils import monitor_ocr_performance
from models_app.vision.utils.testing.dummy_models import DummyModelFactory
from models_app.vision.utils.image_processing.adapter_preprocess import preprocess_for_layoutlmv3


# Versuche, LayoutLMv3 zu importieren
try:
    import torch
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, LayoutLMv3ForSequenceClassification
    LAYOUTLMV3_AVAILABLE = True
except ImportError:
    LAYOUTLMV3_AVAILABLE = False
    torch = None
    LayoutLMv3Processor = None
    LayoutLMv3ForTokenClassification = None
    LayoutLMv3ForSequenceClassification = None

logger = logging.getLogger(__name__)

@register_adapter(name="layoutlmv3", info={
    "description": "LayoutLMv3 for Document Understanding and Analysis",
    "version": "1.0.0",
    "capabilities": {
        "multi_language": True,  # Mit Tesseract-Integration
        "handwriting": False,
        "table_extraction": True,
        "formula_recognition": False,
        "document_understanding": True
    },
    "priority": 75
})
class LayoutLMv3Adapter(BaseOCRAdapter):
    """Adapter für LayoutLMv3 für Document Understanding."""
    
    ADAPTER_NAME = "layoutlmv3"
    ADAPTER_INFO = {
        "description": "LayoutLMv3 for Document Understanding and Analysis",
        "version": "1.0.0",
        "capabilities": {
            "multi_language": True,  # Mit Tesseract-Integration
            "handwriting": False,
            "table_extraction": True,
            "formula_recognition": False,
            "document_understanding": True
        },
        "priority": 75
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den LayoutLMv3-Adapter.
        
        Args:
            config: Konfiguration für LayoutLMv3
        """
        super().__init__(config)
        
        # Standardkonfiguration
        self.model_name = self.config.get('model_name', 'microsoft/layoutlmv3-base')
        self.token_classification_model = self.config.get('token_classification_model', 'microsoft/layoutlmv3-base-finetuned-funsd')
        self.sequence_classification_model = self.config.get('sequence_classification_model', None)  # Optional
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu') if LAYOUTLMV3_AVAILABLE and torch else 'cpu'
        self.task = self.config.get('task', 'token_classification')  # 'token_classification' oder 'sequence_classification'
        self.label_map = self.config.get('label_map', None)  # Benutzerdefinierte Label-Map
        
        # LayoutLMv3Processor Konfiguration
        self.apply_ocr = self.config.get('apply_ocr', True)  # OCR verwenden, um Text und Bounding-Boxen zu erhalten
        self.ocr_lang = self.config.get('ocr_lang', None)  # Standardmäßig Englisch, wenn None
        self.tesseract_config = self.config.get('tesseract_config', '')  # Benutzerdefinierte Tesseract-Parameter
        self.image_size = self.config.get('image_size', {"height": 224, "width": 224})
        
        self.processor = None
        self.model = None
    
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert das LayoutLMv3-Modell.
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if not LAYOUTLMV3_AVAILABLE:
            raise ModelNotAvailableError("LayoutLMv3 (transformers) ist nicht installiert.")
            
        try:
            # LayoutLMv3-Prozessor initialisieren
            self.processor = LayoutLMv3Processor.from_pretrained(
                self.model_name,
                apply_ocr=self.apply_ocr,
                ocr_lang=self.ocr_lang,
                tesseract_config=self.tesseract_config,
                size=self.image_size
            )
            
            # LayoutLMv3-Modell basierend auf der Aufgabe initialisieren
            if self.task == 'token_classification':
                model_name = self.token_classification_model or self.model_name
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
                
                # Label-Map initialisieren, falls nicht angegeben
                if self.label_map is None:
                    # FUNSD-Standardlabels, wenn das FUNSD-finetuned Modell verwendet wird
                    if 'funsd' in model_name.lower():
                        self.label_map = {0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION', 
                                         4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER', 7: 'B-OTHER', 8: 'I-OTHER'}
                    else:
                        self.label_map = {}  # Leere Map, wenn unbekannt
                
            elif self.task == 'sequence_classification':
                model_name = self.sequence_classification_model or self.model_name
                self.model = LayoutLMv3ForSequenceClassification.from_pretrained(model_name)
                
                # Label-Map für Dokumentenklassifikation
                if self.label_map is None:
                    self.label_map = {0: "scientific", 1: "invoice", 2: "letter", 3: "form", 4: "news"}
            else:
                raise ValueError(f"Unbekannte Aufgabe: {self.task}")
            
            # Gerät einstellen
            self.model.to(self.device)
            
            logger.info(f"LayoutLMv3 initialisiert mit Modell: {self.model_name} für Aufgabe: {self.task} auf Gerät: {self.device}")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von LayoutLMv3: {str(e)}")
            self.is_initialized = False
            return False
    
    def _initialize_dummy_model(self):
        """Initialisiert ein Dummy-Modell für LayoutLMv3."""
        dummy = DummyModelFactory.create_ocr_dummy("layoutlmv3")
        self.processor = dummy.get("processor")
        self.model = dummy.get("model")
    
    @monitor_ocr_performance
    @handle_ocr_errors
    def process_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet ein Bild mit LayoutLMv3.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
        
        Returns:
            Dictionary mit OCR-Ergebnissen
        """
        options = options or {}
        
        # Sicherstellen, dass LayoutLMv3 initialisiert ist
        if not self.is_initialized:
            if self.config.get('dummy_mode', False):
                self._initialize_dummy_model()
            else:
                self.initialize()
        
        # Optionen extrahieren
        task = options.get('task', self.task)
        apply_ocr = options.get('apply_ocr', self.apply_ocr)
        ocr_lang = options.get('ocr_lang', self.ocr_lang)
        tesseract_config = options.get('tesseract_config', self.tesseract_config)
        
        # Prüfen, ob das Bild bereits vorverarbeitet wurde
        already_preprocessed = options.get('already_preprocessed', False)
        preprocess = options.get('preprocess', True) and not already_preprocessed
        
        # Bild laden
        from models_app.vision.utils.image_processing.core import load_image
        _, np_image, pil_image = load_image(image_path_or_array)
        
        # Bei Bedarf vorverarbeiten, was LayoutLMv3 selbst nicht optimiert
        if preprocess:
            processed_image = self.preprocess_image(np_image, options)
            # Konvertieren zu PIL für den Processor
            from PIL import Image
            pil_image = Image.fromarray(processed_image)
        
        # Layout-Analyse mit LayoutLMv3
        # Prozessor führt automatisch OCR durch, wenn apply_ocr=True
        encoding = self.processor(
            images=pil_image,
            return_tensors="pt",
            apply_ocr=apply_ocr,
            ocr_lang=ocr_lang,
            tesseract_config=tesseract_config
        )
        
        # Auf das richtige Gerät übertragen
        for key, value in encoding.items():
            if torch.is_tensor(value):
                encoding[key] = value.to(self.device)
        
        # Modell ausführen
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Ergebnisse extrahieren
        logits = outputs.logits
        
        # Verarbeitung basierend auf der Aufgabe
        if task == 'token_classification':
            # Token-Klassifikation (z.B. FUNSD) - ermittle die Labels für jedes Token
            predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]
            
            # Token-IDs und entsprechende Texte extrahieren
            tokens = self.processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].cpu().numpy())
            
            # Bounding-Boxen extrahieren
            bboxes = encoding["bbox"][0].cpu().numpy()
            
            # Text aus Tokens rekonstruieren und in bedeutungsvollen Blöcken gruppieren
            blocks = []
            current_entity = None
            full_text = []
            
            for i in range(1, len(tokens) - 1):  # Ignoriere [CLS] und [SEP]
                token = tokens[i]
                if token.startswith("##"):
                    token = token[2:]  # Entferne Subtoken-Marker
                    if current_entity:
                        current_entity["text"] += token
                else:
                    # Wenn wir ein neues Token beginnen, könnte das vorherige abgeschlossen sein
                    if current_entity:
                        blocks.append(current_entity)
                        full_text.append(current_entity["text"])
                    
                    # Neues Token/Entity beginnen
                    label_id = predictions[i]
                    label = self.label_map.get(label_id, f"LABEL_{label_id}")
                    
                    # Bounding Box
                    bbox = bboxes[i].tolist()
                    
                    current_entity = {
                        "text": token,
                        "conf": 0.95,  # Dummy-Konfidenz
                        "bbox": bbox,
                        "label": label
                    }
            
            # Das letzte Token hinzufügen, wenn es nicht getan wurde
            if current_entity:
                blocks.append(current_entity)
                full_text.append(current_entity["text"])
            
            # Ergebnis erstellen mit Token-Klassifikation
            result = self._create_result(
                text="\n".join(full_text),
                blocks=blocks,
                confidence=0.95,  # Dummy-Konfidenz
                language=ocr_lang or "de",
                metadata={
                    'model_name': self.model_name if task == 'token_classification' else self.sequence_classification_model,
                    'task': task,
                    'apply_ocr': apply_ocr,
                    'ocr_lang': ocr_lang,
                    'tesseract_config': tesseract_config
                }
            )
            
        else:  # 'sequence_classification'
            # Dokumentenklassifikation - ermittle die Dokumentenklasse
            predictions = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class_id = np.argmax(predictions)
            confidence = float(predictions[predicted_class_id])
            predicted_class = self.label_map.get(predicted_class_id, f"CLASS_{predicted_class_id}")
            
            # Wenn OCR aktiviert ist, extrahiere Text aus den Tokens
            if apply_ocr and hasattr(encoding, "input_ids"):
                token_ids = encoding["input_ids"][0].cpu().numpy()
                tokens = self.processor.tokenizer.convert_ids_to_tokens(token_ids)
                full_text = self.processor.tokenizer.convert_tokens_to_string(tokens)
            else:
                full_text = f"Dokument klassifiziert als: {predicted_class}"
            
            # Ergebnis erstellen mit Dokumentenklassifikation
            result = self._create_result(
                text=full_text,
                blocks=[{
                    "text": full_text,
                    "conf": confidence,
                    "class": predicted_class,
                    "bbox": [0, 0, pil_image.width, pil_image.height]
                }],
                confidence=confidence,
                language=ocr_lang or "de",
                metadata={
                    'model_name': self.model_name if task == 'token_classification' else self.sequence_classification_model,
                    'task': task,
                    'predicted_class': predicted_class,
                    'class_probabilities': {self.label_map.get(i, f"CLASS_{i}"): float(prob) for i, prob in enumerate(predictions)},
                    'apply_ocr': apply_ocr,
                    'ocr_lang': ocr_lang,
                    'tesseract_config': tesseract_config
                }
            )
        
        # Post-Processing
        result = self.postprocess_result(result, options)
        
        return result
    
    def extract_document_structure(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extrahiert die Dokumentenstruktur mit LayoutLMv3.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
            
        Returns:
            Dictionary mit erkannter Dokumentenstruktur
        """
        options = options or {}
        
        # Stelle sicher, dass wir das Token-Klassifikationsmodell verwenden
        current_task = self.task
        self.task = 'token_classification'
        
        # Verarbeite das Bild
        result = self.process_image(image_path_or_array, options)
        
        # Struktur extrahieren und gruppieren
        structure = {
            "headers": [],
            "questions": [],
            "answers": [],
            "other": []
        }
        
        for block in result.get("blocks", []):
            label = block.get("label", "O")
            if label.startswith("B-HEADER") or label.startswith("I-HEADER"):
                structure["headers"].append(block)
            elif label.startswith("B-QUESTION") or label.startswith("I-QUESTION"):
                structure["questions"].append(block)
            elif label.startswith("B-ANSWER") or label.startswith("I-ANSWER"):
                structure["answers"].append(block)
            elif label.startswith("B-OTHER") or label.startswith("I-OTHER"):
                structure["other"].append(block)
        
        # Aufgabe zurücksetzen
        self.task = current_task
        
        # Use centralized utilities for text region detection
        text_regions = detect_text_regions(image_path_or_array)
        # Create visualization
        debug_image = draw_bounding_boxes(image_path_or_array, text_regions)
        
        return {
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "structure": structure,
            "metadata": result.get("metadata", {}),
            "debug_image": debug_image
        }
    
    def get_supported_languages(self) -> List[str]:
        """
        Gibt die unterstützten Sprachen zurück.
        
        Returns:
            Liste der unterstützten Sprachen
        """
        # LayoutLMv3 unterstützt die Sprachen, die Tesseract unterstützt
        # wenn apply_ocr=True, sonst nur Englisch
        if self.apply_ocr:
            # Tesseract unterstützt viele Sprachen
            return [
                'eng', 'deu', 'fra', 'spa', 'ita', 'por', 'nld', 'jpn', 'kor', 'chi_sim', 
                'chi_tra', 'ara', 'rus', 'hin', 'ben', 'tur', 'tha', 'vie'
            ]
        else:
            return ['de']
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen zum Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        info = super().get_model_info()
        
        if self.is_initialized:
            info.update({
                "type": "LayoutLMv3",
                "model_name": self.model_name,
                "task": self.task,
                "device": self.device,
                "apply_ocr": self.apply_ocr,
                "ocr_lang": self.ocr_lang,
                "label_map": self.label_map,
                "supported_languages": self.get_supported_languages()
            })
                
        return info 

    def preprocess_image(self, image_path_or_array, options: Dict[str, Any] = None) -> np.ndarray:
        """
        Vorverarbeitung des Bildes für LayoutLMv3.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Optionen für die Vorverarbeitung
            
        Returns:
            np.ndarray: Vorverarbeitetes Bild
        """
        options = options or {}
        # Adapter-spezifische Optionen hinzufügen
        options["normalize"] = self.config.get("normalize", True)
        options["denoise"] = self.config.get("denoise", True)
        
        # Zentralisierte Vorverarbeitung verwenden
        return preprocess_for_layoutlmv3(image_path_or_array, options) 