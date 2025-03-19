"""
Adapter für Nougat OCR für wissenschaftliche Dokumente.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image

from .base_adapter import BaseOCRAdapter
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, ModelNotAvailableError
from models_app.vision.ocr.utils.plugin_system import register_adapter
from models_app.vision.utils.image_processing.core import load_image, convert_to_pil
from models_app.vision.utils.image_processing.enhancement import enhance_for_nougat
from analytics_app.utils import monitor_ocr_performance
from models_app.vision.utils.testing.dummy_models import DummyModelFactory
from models_app.vision.utils.image_processing.adapter_preprocess import preprocess_for_nougat

# Versuche, Nougat zu importieren
try:
    import torch
    from transformers import NougatProcessor, VisionEncoderDecoderModel
    NOUGAT_AVAILABLE = True
except ImportError:
    NOUGAT_AVAILABLE = False
    torch = None
    NougatProcessor = None
    VisionEncoderDecoderModel = None

logger = logging.getLogger(__name__)

@register_adapter(name="nougat", info={
    "description": "Nougat OCR Engine for scientific documents with LaTeX support",
    "version": "0.1.0",
    "capabilities": {
        "multi_language": False,
        "handwriting": False,
        "table_extraction": True,
        "formula_recognition": True,
        "document_understanding": True
    },
    "priority": 85
})
class NougatAdapter(BaseOCRAdapter):
    """Adapter für Nougat OCR für wissenschaftliche Dokumente."""
    
    ADAPTER_NAME = "nougat"
    ADAPTER_INFO = {
        "description": "Nougat OCR Engine for scientific documents with LaTeX support",
        "version": "0.1.0",
        "capabilities": {
            "multi_language": False,
            "handwriting": False,
            "table_extraction": True,
            "formula_recognition": True,
            "document_understanding": True
        },
        "priority": 85
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialisiert den Nougat-Adapter.
        
        Args:
            config: Konfiguration für Nougat
        """
        super().__init__(config)
        
        # Standardkonfiguration
        self.model_name = self.config.get('model_name', 'facebook/nougat-base')
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu') if NOUGAT_AVAILABLE and torch else 'cpu'
        self.max_length = self.config.get('max_length', 4096)
        self.return_markdown = self.config.get('return_markdown', True)
        self.offload_folder = self.config.get('offload_folder', None)
        
        self.processor = None
        self.model = None
    
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert das Nougat-Modell.
        
        Returns:
            True, wenn die Initialisierung erfolgreich war, sonst False
        """
        if not NOUGAT_AVAILABLE:
            raise ModelNotAvailableError("Nougat (transformers) ist nicht installiert.")
            
        try:
            # Nougat-Prozessor initialisieren
            self.processor = NougatProcessor.from_pretrained(self.model_name)
            
            # Nougat-Modell initialisieren
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Gerät einstellen
            self.model.to(self.device)
            
            logger.info(f"Nougat initialisiert mit Modell: {self.model_name} auf Gerät: {self.device}")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung von Nougat: {str(e)}")
            self.is_initialized = False
            return False
    
    def _initialize_dummy_model(self):
        """Initialisiert ein Dummy-Modell für Nougat."""
        from localgpt_vision_django.models_app.vision.utils.testing.dummy_models import DummyModelFactory
        dummy = DummyModelFactory.create_ocr_dummy("nougat")
        self.processor = dummy.get("processor")
        self.model = dummy.get("model")
    
    @monitor_ocr_performance
    @handle_ocr_errors
    def process_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verarbeitet ein Bild mit Nougat.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
        
        Returns:
            Dictionary mit OCR-Ergebnissen
        """
        options = options or {}
        
        # Sicherstellen, dass Nougat initialisiert ist
        if not self.is_initialized:
            if self.config.get('dummy_mode', False):
                self._initialize_dummy_model()
            else:
                self.initialize()
        
        # Optionen extrahieren
        max_length = options.get('max_length', self.max_length)
        return_markdown = options.get('return_markdown', self.return_markdown)
        
        # Prüfen, ob das Bild bereits vorverarbeitet wurde
        already_preprocessed = options.get('already_preprocessed', False)
        preprocess = options.get('preprocess', True) and not already_preprocessed
        
        # Bild laden
        from models_app.vision.utils.image_processing.core import load_image, convert_to_pil
        pil_image, np_image, _ = load_image(image_path_or_array)
        
        # Bild vorverarbeiten wenn nötig
        if preprocess:
            # Vorverarbeitung mit der spezialisierten Methode
            pil_image = self.preprocess_image(np_image, options)
        
        # Bild mit Nougat-Prozessor verarbeiten
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Generierung durchführen
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.pixel_values,
                max_length=max_length,
                num_beams=1,
                early_stopping=True
            )
        
        # Generierte IDs in Text umwandeln
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Markdowntags entfernen, falls gewünscht
        if not return_markdown:
            # Einfache Markdown-Ersetzung (für komplexere Fälle würde ein Markdown-Parser verwendet)
            import re
            generated_text = re.sub(r'##+\s+', '', generated_text)  # Überschriften
            generated_text = re.sub(r'\*\*(.*?)\*\*', r'\1', generated_text)  # Fett
            generated_text = re.sub(r'\*(.*?)\*', r'\1', generated_text)  # Kursiv
            generated_text = re.sub(r'```[a-z]*\n', '', generated_text)  # Codeblock-Start
            generated_text = re.sub(r'```', '', generated_text)  # Codeblock-Ende
            
        # Da Nougat keine Bounding-Boxen liefert, erstellen wir ein minimales Block-Ergebnis
        blocks = [{
            'text': generated_text,
            'conf': 0.95,  # Dummy-Konfidenz, da Nougat keine liefert
            'bbox': [0, 0, pil_image.width, pil_image.height],
            'polygon': [[0, 0], [pil_image.width, 0], [pil_image.width, pil_image.height], [0, pil_image.height]]
        }]
        
        # Ergebnis erstellen
        result = self._create_result(
            text=generated_text,
            blocks=blocks,
            confidence=0.95,  # Dummy-Konfidenz
            language="en",
            metadata={
                'model_name': self.model_name,
                'device': self.device,
                'max_length': max_length,
                'return_markdown': return_markdown
            }
        )
        # Post-Processing
        result = self.postprocess_result(result, options)
        
        return result

    def preprocess_image(self, image_path_or_array, options: Dict[str, Any] = None) -> Image.Image:
        """
        Vorverarbeitung des Bildes für Nougat.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Optionen für die Vorverarbeitung
        
        Returns:
            Image.Image: Vorverarbeitetes PIL-Bild
        """
        options = options or {}
        # Nougat-spezifische Optionen
        options["normalize"] = True
        options["denoise"] = self.config.get("denoise", True)
        
        # Zentralisierte Vorverarbeitung verwenden
        processed_img = preprocess_for_nougat(image_path_or_array, options)
        
        # Konvertiere zu PIL für den Nougat-Prozessor
        return Image.fromarray(processed_img)

    def get_supported_languages(self) -> List[str]:
        """
        Gibt die unterstützten Sprachen zurück.
            
        Returns:
            Liste der unterstützten Sprachen
        """
        # Nougat unterstützt hauptsächlich Englisch
        return ['en']
    
    def extract_formulas(self, text: str) -> List[str]:
        """
        Extrahiert LaTeX-Formeln aus dem erkannten Text.
        
        Args:
            text: Der erkannte Text
            
        Returns:
            Liste der gefundenen LaTeX-Formeln
        """
        import re
        
        # LaTeX-Blöcke finden (Inline und Display)
        inline_math = re.findall(r'\$(.*?)\$', text)
        display_math = re.findall(r'\$\$(.*?)\$\$', text)
        
        # Auch Umgebungen wie \begin{equation} finden
        env_math = re.findall(r'\\begin\{(equation|align|gather|multline|eqnarray)[*]?\}(.*?)\\end\{\1[*]?\}', text, re.DOTALL)
        env_formulas = [formula for _, formula in env_math]
        
        # Alle Formeln zusammenführen
        formulas = inline_math + display_math + env_formulas
        
        return formulas
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen zum Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        info = super().get_model_info()
        
        if self.is_initialized:
            info.update({
                "type": "Nougat",
                "model_name": self.model_name,
                "device": self.device,
                "max_length": self.max_length,
                "return_markdown": self.return_markdown,
                "features": ["LaTeX formula recognition", "Scientific document understanding", "Table extraction"]
            })
                
        return info 