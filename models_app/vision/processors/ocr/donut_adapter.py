"""
Adapter für Donut (Document Understanding Transformer).
"""

import os
import logging
import numpy as np
import json
import re
from typing import Dict, List, Any, Optional
from PIL import Image

from .base_adapter import BaseOCRAdapter
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, ModelNotAvailableError
from models_app.vision.ocr.utils.plugin_system import register_adapter
from models_app.vision.utils.image_processing.core import load_image, convert_to_pil
from models_app.vision.utils.image_processing.enhancement import enhance_for_donut
from analytics_app.utils import monitor_ocr_performance
from models_app.vision.utils.testing.dummy_models import DummyProcessor, DummyModel, DummyModelFactory
from models_app.vision.utils.image_processing.adapter_preprocess import preprocess_for_donut

# Versuche, Donut zu importieren
try:
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    DONUT_AVAILABLE = True
except ImportError:
    DONUT_AVAILABLE = False
    torch = None
    DonutProcessor = None
    VisionEncoderDecoderModel = None

logger = logging.getLogger(__name__)

@register_adapter(name="donut", info={
    "description": "Donut Document Understanding Transformer",
    "version": "1.0.0",
    "capabilities": {
        "multi_language": False,
        "handwriting": False,
        "table_extraction": True,
        "formula_recognition": False,
        "document_understanding": True
    },
    "priority": 85
})
class DonutAdapter(BaseOCRAdapter):
    """Adapter für Donut (Document Understanding Transformer)."""
    
    ADAPTER_NAME = "donut"
    ADAPTER_INFO = {
        "description": "Donut Document Understanding Transformer",
        "version": "1.0.0",
        "capabilities": {
            "multi_language": False,
            "handwriting": False,
            "table_extraction": True,
            "formula_recognition": False,
            "document_understanding": True
        },
        "priority": 85
    }
    
    # Verfügbare vortrainierte Modelle für verschiedene Aufgaben
    AVAILABLE_MODELS = {
        "document-parsing": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "receipt-parsing": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "invoice-parsing": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "rvl-cdip": "naver-clova-ix/donut-base-finetuned-rvlcdip",
        "document-classification": "naver-clova-ix/donut-base-finetuned-rvlcdip",
        "docvqa": "naver-clova-ix/donut-base-finetuned-docvqa"
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den Donut-Adapter.
        
        Args:
            config: Konfiguration für Donut
        """
        super().__init__(config)
        
        # Standardkonfiguration
        self.task = self.config.get('task', 'document-parsing')
        self.model_name = self.config.get('model_name', self.AVAILABLE_MODELS.get(self.task))
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu') if DONUT_AVAILABLE and torch else 'cpu'
        self.max_length = self.config.get('max_length', 512)
        self.prompt = self.config.get('prompt', "")
        
        self.processor = None
        self.model = None
    
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert das Donut-Modell.
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if not DONUT_AVAILABLE:
            raise ModelNotAvailableError("Donut (transformers) ist nicht installiert.")
            
        try:
            # Modellnamen überprüfen
            if not self.model_name:
                if self.task in self.AVAILABLE_MODELS:
                    self.model_name = self.AVAILABLE_MODELS[self.task]
                else:
                    raise ValueError(f"Unbekannte Aufgabe: {self.task}. "
                                    f"Verfügbare Aufgaben: {', '.join(self.AVAILABLE_MODELS.keys())}")
            
            # Donut-Prozessor initialisieren
            self.processor = DonutProcessor.from_pretrained(self.model_name)
            
            # Donut-Modell initialisieren
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Gerät einstellen
            self.model.to(self.device)
            
            logger.info(f"Donut initialisiert mit Modell: {self.model_name} für Aufgabe: {self.task} auf Gerät: {self.device}")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von Donut: {str(e)}")
            self.is_initialized = False
            return False
    
    def _initialize_dummy_model(self):
        """Initialisiert ein Dummy-Modell für Donut."""
        from localgpt_vision_django.models_app.vision.utils.testing.dummy_models import DummyModelFactory
        dummy = DummyModelFactory.create_ocr_dummy("donut")
        self.processor = dummy.get("processor")
        self.model = dummy.get("model")
        self.is_initialized = True
        logger.info("Dummy-Modell für Donut initialisiert")
    
    def _parse_donut_output(self, output_text: str) -> Dict[str, Any]:
        """
        Parst die Ausgabe des Donut-Modells in ein strukturiertes Dictionary.
        
        Args:
            output_text: Ausgabetext des Donut-Modells
            
        Returns:
            Dictionary mit strukturierten Informationen
        """
        result = {}
        
        try:
            # Je nach Aufgabe unterschiedliche Parsing-Strategien
            if self.task in ["document-parsing", "receipt-parsing", "invoice-parsing"]:
                # Für Dokumente mit Schlüssel-Wert-Paaren (z.B. Rechnungen, Quittungen)
                import re
                
                # Alle Tags und ihren Inhalt finden
                pattern = r'<fim_([^>]+)>(.*?)</fim_\1>'
                matches = re.findall(pattern, output_text)
                
                for key, value in matches:
                    result[key] = value.strip()
                
            elif self.task in ["rvl-cdip", "document-classification"]:
                # Für Dokumentenklassifikation
                import re
                
                # Klasse extrahieren
                class_match = re.search(r'<fim_class>(.*?)</fim_class>', output_text)
                if class_match:
                    result["class"] = class_match.group(1).strip()
                
            elif self.task == "docvqa":
                # Für Dokumenten-VQA (Visual Question Answering)
                import re
                
                # Antwort extrahieren
                answer_match = re.search(r'<fim_answer>(.*?)</fim_answer>', output_text)
                if answer_match:
                    result["answer"] = answer_match.group(1).strip()
                
            else:
                # Allgemeiner Ansatz für andere Aufgaben
                import re
                
                # Alle Tags und ihren Inhalt finden
                pattern = r'<fim_([^>]+)>(.*?)</fim_\1>'
                matches = re.findall(pattern, output_text)
                
                for key, value in matches:
                    result[key] = value.strip()
            
            # Versuch, JSON zu extrahieren, falls vorhanden
            import re
            json_match = re.search(r'\{.*\}', output_text)
            if json_match:
                try:
                    json_data = json.loads(json_match.group(0))
                    # JSON-Daten in das Ergebnis integrieren
                    for key, value in json_data.items():
                        if key not in result:
                            result[key] = value
                except json.JSONDecodeError:
                    pass
                
        except Exception as e:
            logger.warning(f"Fehler beim Parsen der Donut-Ausgabe: {str(e)}")
        
        return result
    
    @monitor_ocr_performance
    @handle_ocr_errors
    def process_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet ein Bild mit Donut.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
        
        Returns:
            Dictionary mit extrahierten Informationen
        """
        options = options or {}
        
        # Sicherstellen, dass Donut initialisiert ist
        if not self.is_initialized:
            if self.config.get('dummy_mode', False):
                self._initialize_dummy_model()
            else:
                self.initialize()
        
        # Optionen extrahieren
        task = options.get('task', self.task)
        prompt = options.get('prompt', self.prompt)
        max_length = options.get('max_length', self.max_length)
        question = options.get('question', None)  # Für DocVQA
        
        # Prüfen, ob das Bild bereits vorverarbeitet wurde
        already_preprocessed = options.get('already_preprocessed', False)
        preprocess = options.get('preprocess', True) and not already_preprocessed
        
        # Prompt anpassen je nach Aufgabe
        if task == "docvqa" and question:
            prompt = f"{question}"
            
        # Bild laden
        pil_image, np_image, _ = load_image(image_path_or_array)
        
        # Bild vorverarbeiten
        if preprocess:
            # Vorverarbeitung mit der spezialisierten Methode
            pil_image = self.preprocess_image(np_image, options)
        
        # Feature Extraction
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
        
        # Auf das richtige Gerät übertragen
        pixel_values = pixel_values.to(self.device)
            
        # Prompt tokenisieren
        decoder_input_ids = self.processor.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        # Generierung durchführen
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True
            )
            
        # Konvertiere Ausgabe zu Text
        sequence = self.processor.tokenizer.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            
        # Parse die Ausgabe zu einem strukturierten Dictionary
        parsed_result = self._parse_donut_output(sequence)
        
        # Umwandeln in OCR-Ergebnis
        # Da Donut keine Bounding-Boxen für Text liefert, erstellen wir einen einzelnen Block
        full_text = sequence
        block = {
            'text': sequence,
            'conf': 0.9,  # Dummy-Konfidenz, da Donut keine liefert
            'bbox': [0, 0, pil_image.width, pil_image.height],
            'parsed': parsed_result
        }
            
            # Ergebnis erstellen
        result = self._create_result(
            text=full_text,
            blocks=[block],
            confidence=0.9,  # Dummy-Konfidenz
            language="en",
            metadata={
                'model_name': self.model_name,
                'task': task,
                'prompt': prompt,
                'parsed_data': parsed_result
            },
            raw_output=sequence
        )
        
        # Post-Processing
        result = self.postprocess_result(result, options)
        
        return result
    
    def answer_question(self, image_path_or_array, question: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Beantwortet eine Frage zu einem Dokument (DocVQA).
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            question: Frage zum Dokument
            options: Zusätzliche Optionen für die Verarbeitung
            
        Returns:
            Dictionary mit der Antwort auf die Frage
        """
        options = options or {}
        options['task'] = 'docvqa'
        options['question'] = question
        
        # Verarbeite das Bild mit der DocVQA-Aufgabe
        result = self.process_image(image_path_or_array, options)
        
        return result
    
    def extract_structure(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extrahiert die Struktur eines Dokuments.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
            
        Returns:
            Dictionary mit der Struktur des Dokuments
        """
        options = options or {}
        options['task'] = 'document-parsing'
        
        # Verarbeite das Bild mit der Document-Parsing-Aufgabe
        result = self.process_image(image_path_or_array, options)
        
        return result
    
    def classify_document(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Klassifiziert ein Dokument.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
            
        Returns:
            Dictionary mit der Dokumentenklasse
        """
        options = options or {}
        options['task'] = 'document-classification'
        
        # Verarbeite das Bild mit der Document-Classification-Aufgabe
        result = self.process_image(image_path_or_array, options)
        
        return result
    
    def get_supported_languages(self) -> List[str]:
        """
        Gibt die unterstützten Sprachen zurück.
        
        Returns:
            Liste der unterstützten Sprachen
        """
        # Donut unterstützt hauptsächlich Englisch
        return ['en']
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen zum Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        info = super().get_model_info()
        
        if self.is_initialized:
            info.update({
                "type": "Donut",
                "model_name": self.model_name,
                "task": self.task,
                "device": self.device,
                "max_length": self.max_length,
                "supported_tasks": list(self.AVAILABLE_MODELS.keys())
            })
                
        return info

    def preprocess_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Image.Image:
        """
        Vorverarbeitung des Bildes für Donut.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Optionen für die Vorverarbeitung
            
        Returns:
            Image.Image: Vorverarbeitetes PIL-Bild
        """
        options = options or {}
        # Donut-spezifische Optionen
        options["normalize"] = True
        
        # Zentralisierte Vorverarbeitung verwenden
        processed_img = preprocess_for_donut(image_path_or_array, options)
        
        # Konvertiere zu PIL für den Donut-Prozessor
        return Image.fromarray(processed_img) 