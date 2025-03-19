"""
Basisklasse für alle OCR-Adapter.

Diese Klasse definiert die gemeinsame Schnittstelle, die von allen OCR-Adaptern
implementiert werden muss, und stellt Hilfsfunktionen bereit.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional
import uuid
from datetime import datetime
import hashlib

from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, OCRError
from models_app.vision.ocr.utils.result_formatter import create_standard_result
from models_app.vision.utils.image_processing import load_image, enhance_image_for_ocr
from models_app.vision.document.utils.processing_metadata_context import ProcessingMetadataContext
# Keep import for type compatibility but remove implementation
from models_app.knowledge_graph.interfaces import EntityExtractorInterface
from models_app.next_layer.interfaces import NextLayerInterface
from models_app.next_layer.events import ProcessingEventType

logger = logging.getLogger(__name__)

class BaseOCRAdapter(ABC):
    """
    Basisklasse für alle OCR-Adapter.
    
    Diese Adapter führen die Texterkennung durch und liefern strukturierte Ergebnisse,
    die dann von spezialisierten Entity Extractors (wie DocumentEntityExtractor)
    weiterverarbeitet werden können, um Wissengraphen zu erstellen.
    
    Hinweis: Die direkte Extraktion von Knowledge Graph Entitäten soll nicht in 
    OCR Adaptern implementiert werden. Stattdessen sollten die strukturierten OCR-Ergebnisse
    an die spezialisierten Entity Extractors in models_app/vision/knowledge_graph/ 
    übergeben werden.
    """
    
    # Klassenattribute für das Plugin-System
    ADAPTER_NAME = "base"
    ADAPTER_INFO = {
        "description": "Base OCR Adapter",
        "version": "1.0.0",
        "capabilities": {}
    }
    
    def __init__(self, config=None):
        """Initialize the OCR adapter."""
        self.config = config or {}
        self.is_initialized = False
        self.next_layer = NextLayerInterface.get_instance()
        self.model = None
        self.hooks = getattr(self.__class__, 'HOOKS', {})
        self.performance_history = []
        # OCR-spezifische Hooks für Vor- und Nachverarbeitung
        self.preprocessing_hooks = []
        self.postprocessing_hooks = []
    
    @abstractmethod
    @handle_ocr_errors
    def initialize(self) -> bool:
        """
        Initialisiert das OCR-Modell.
        
        Returns:
            True wenn erfolgreich, sonst False
        """
        raise NotImplementedError("Subclasses must implement initialize()")
    
    @handle_ocr_errors
    def process_image(self, image_path_or_array, options=None) -> Dict[str, Any]:
        """Process an image using OCR."""
        # Emit processing event
        self.next_layer.emit_simple_event(
            ProcessingEventType.PROCESSING_PHASE_START,
            str(image_path_or_array) if isinstance(image_path_or_array, str) else "image_array",
            {
                "processor": self.__class__.__name__,
                "adapter": self.ADAPTER_NAME,
                "options": options
            }
        )
        
        try:
            # Ensure initialization
            if not self.is_initialized:
                self.initialize()
                
            # Process image
            result = self._process_image_internal(image_path_or_array, options)
            
            # Emit completion event
            self.next_layer.emit_simple_event(
                ProcessingEventType.PROCESSING_PHASE_END,
                str(image_path_or_array) if isinstance(image_path_or_array, str) else "image_array",
                {
                    "processor": self.__class__.__name__,
                    "success": True,
                    "result_keys": list(result.keys()) if result else []
                }
            )
            
            return result
            
        except Exception as e:
            # Emit error event
            self.next_layer.emit_simple_event(
                ProcessingEventType.ERROR_OCCURRED,
                str(image_path_or_array) if isinstance(image_path_or_array, str) else "image_array",
                {
                    "processor": self.__class__.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
    
    def get_supported_languages(self) -> List[str]:
        """
        Gibt die unterstützten Sprachen zurück.
        
        Returns:
            Liste der unterstützten Sprachen
        """
        return ["en"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen zum Modell zurück.
        
        Returns:
            Dictionary mit Modellinformationen
        """
        return {
            "name": self.__class__.__name__,
            "type": "OCR",
            "version": "1.0.0",
            "capabilities": self.ADAPTER_INFO.get("capabilities", {}),
            "info": self.ADAPTER_INFO,
            "config": {k: v for k, v in self.config.items() if k != "api_key"}
        }
    
    @handle_ocr_errors
    def preprocess_image(self, image_path_or_array, options: Dict[str, Any] = None,
                       metadata_context: Optional[ProcessingMetadataContext] = None) -> Any:
        """
        Vorverarbeitung eines Bildes für die OCR.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Vorverarbeitung
            metadata_context: Kontext für das Tracking von Metadaten
            
        Returns:
            Vorverarbeitetes Bild
        """
        options = options or {}
        
        # Wenn der Metadatenkontext vorhanden ist, starte Timing
        if metadata_context:
            metadata_context.start_timing("ocr_image_preprocessing")
        
        # Prüfen, ob das Bild bereits vorverarbeitet wurde
        if options.get('already_preprocessed', False):
            if metadata_context:
                metadata_context.record_decision(
                    component=self.__class__.__name__,
                    decision="Skipped preprocessing",
                    reason="Image already preprocessed"
                )
                metadata_context.end_timing("ocr_image_preprocessing")
            return image_path_or_array
        
        # Standardoptionen
        denoise = options.get('denoise', True)
        binarize = options.get('binarize', True)
        deskew = options.get('deskew', False)
        
        if metadata_context:
            metadata_context.record_preprocessing_step(
                method="ocr_preprocess",
                parameters={
                    "denoise": denoise,
                    "binarize": binarize,
                    "deskew": deskew
                },
                before_image_path=image_path_or_array if isinstance(image_path_or_array, str) else None
            )
        
        # Bild verbessern
        try:
            enhanced_pil, enhanced_np = enhance_image_for_ocr(
                image_path_or_array,
                denoise=denoise,
                binarize=binarize,
                deskew=deskew
            )
            
            # Pre-Processing-Hooks ausführen
            if 'pre_process' in self.hooks:
                for hook in self.hooks['pre_process']:
                    try:
                        enhanced_np = hook(enhanced_np, options)
                    except Exception as e:
                        logger.warning(f"Fehler beim Ausführen des Pre-Processing-Hooks: {str(e)}")
                        if metadata_context:
                            metadata_context.record_warning(
                                component=f"{self.__class__.__name__}.preprocess_image",
                                message=f"Error in preprocessing hook: {str(e)}"
                            )
            
            if metadata_context:
                metadata_context.end_timing("ocr_image_preprocessing")
                
            return enhanced_np
            
        except Exception as e:
            logger.error(f"Fehler bei der Bildvorverarbeitung: {str(e)}")
            if metadata_context:
                metadata_context.record_error(
                    component=f"{self.__class__.__name__}.preprocess_image",
                    message=f"Error in image preprocessing: {str(e)}",
                    error_type=type(e).__name__
                )
                metadata_context.end_timing("ocr_image_preprocessing")
            return image_path_or_array
    
    @handle_ocr_errors
    def postprocess_result(self, result: Dict[str, Any], options: Dict[str, Any] = None,
                         metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Nachbearbeitung des OCR-Ergebnisses.
        
        Args:
            result: OCR-Ergebnis
            options: Zusätzliche Optionen für die Nachbearbeitung
            metadata_context: Kontext für das Tracking von Metadaten
            
        Returns:
            Nachbearbeitetes OCR-Ergebnis
        """
        options = options or {}
        
        if metadata_context:
            metadata_context.start_timing("ocr_result_postprocessing")
        
        try:
            # Post-Processing-Hooks ausführen
            if 'post_process' in self.hooks:
                for hook in self.hooks['post_process']:
                    try:
                        result = hook(result, options)
                    except Exception as e:
                        logger.warning(f"Fehler beim Ausführen des Post-Processing-Hooks: {str(e)}")
                        if metadata_context:
                            metadata_context.record_warning(
                                component=f"{self.__class__.__name__}.postprocess_result",
                                message=f"Error in postprocessing hook: {str(e)}"
                            )
            
            if metadata_context:
                metadata_context.end_timing("ocr_result_postprocessing")
                
            return result
            
        except Exception as e:
            logger.error(f"Fehler bei der Ergebnisnachbearbeitung: {str(e)}")
            if metadata_context:
                metadata_context.record_error(
                    component=f"{self.__class__.__name__}.postprocess_result",
                    message=f"Error in result postprocessing: {str(e)}",
                    error_type=type(e).__name__
                )
                metadata_context.end_timing("ocr_result_postprocessing")
            return result
    
    @handle_ocr_errors
    def _execute_hooks(self, hook_name: str, data: Any, options: Dict[str, Any] = None,
                     metadata_context: Optional[ProcessingMetadataContext] = None) -> Any:
        """
        Führt Hooks eines bestimmten Typs aus.
        
        Args:
            hook_name: Name des Hooks
            data: Daten, die an den Hook übergeben werden
            options: Zusätzliche Optionen
            metadata_context: Kontext für das Tracking von Metadaten
            
        Returns:
            Verarbeitete Daten
        """
        options = options or {}
        
        if metadata_context:
            metadata_context.start_timing(f"execute_hook_{hook_name}")
        
        if hook_name in self.hooks:
            for i, hook in enumerate(self.hooks[hook_name]):
                try:
                    data = hook(data, options)
                    if metadata_context:
                        metadata_context.record_decision(
                            component=f"{self.__class__.__name__}._execute_hooks",
                            decision=f"Executed {hook_name} hook {i+1}",
                            reason="Normal processing flow"
                        )
                except Exception as e:
                    logger.warning(f"Fehler beim Ausführen des Hooks '{hook_name}': {str(e)}")
                    if metadata_context:
                        metadata_context.record_warning(
                            component=f"{self.__class__.__name__}._execute_hooks",
                            message=f"Error executing {hook_name} hook {i+1}: {str(e)}"
                        )
        
        if metadata_context:
            metadata_context.end_timing(f"execute_hook_{hook_name}")
            
        return data
    
    @handle_ocr_errors
    def _create_result(self, text: str = "", blocks: List[Dict[str, Any]] = None, 
                      confidence: float = 0.0, language: str = "unknown", 
                      metadata: Dict[str, Any] = None, raw_output: Any = None,
                      metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Erstellt ein standardisiertes OCR-Ergebnis.
        
        Args:
            text: Erkannter Text
            blocks: Liste von Textblöcken mit Position und Konfidenz
            confidence: Gesamtkonfidenz des Ergebnisses
            language: Erkannte oder verwendete Sprache
            metadata: Zusätzliche Metadaten
            raw_output: Rohausgabe des OCR-Modells
            metadata_context: Kontext für das Tracking von Metadaten
            
        Returns:
            Standardisiertes OCR-Ergebnis
        """
        if metadata_context:
            metadata_context.start_timing("create_standardized_result")
            
            # Zusätzliche Metadaten über das Ergebnis aufzeichnen
            metadata_context.add_adapter_data(
                adapter_name=self.__class__.__name__,
                key="ocr_result_stats",
                value={
                    "text_length": len(text),
                    "blocks_count": len(blocks or []),
                    "confidence": confidence,
                    "language": language
                }
            )
        
        result = create_standard_result(
            text=text,
            blocks=blocks or [],
            confidence=confidence,
            model=self.__class__.__name__,
            language=language,
            metadata=metadata or {},
            raw_output=raw_output
        )
        
        if metadata_context:
            metadata_context.end_timing("create_standardized_result")
            
        return result

    def prepare_for_extraction(self, image_path_or_array, options: Dict[str, Any] = None,
                             metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Prepare OCR results for entity extraction with standardized output format.
        
        Args:
            image_path_or_array: Pfad zum Bild oder NumPy-Array
            options: Zusätzliche Optionen für die Verarbeitung
            metadata_context: Kontext für das Tracking von Metadaten
            
        Returns:
            Standardisiertes Extraktionsergebnis
        """
        if metadata_context:
            metadata_context.start_timing("prepare_for_extraction")
        
        # Process the image with OCR
        ocr_result = self.process_image(image_path_or_array, options, metadata_context)
        
        # Calculate image ID
        image_id = None
        if isinstance(image_path_or_array, str):
            image_id = hashlib.md5(str(image_path_or_array).encode()).hexdigest()
        
        # Create standardized extraction-ready output
        extraction_data = {
            "source_type": "ocr",
            "processor_id": self.ADAPTER_NAME,
            "image_id": image_id,
            
            # Text content
            "text": ocr_result.get("text", ""),
            
            # Structured content
            "content": {
                "blocks": ocr_result.get("blocks", []),
                "lines": ocr_result.get("lines", []),
                "words": ocr_result.get("words", []),
                "tables": ocr_result.get("tables", []),
                "regions": ocr_result.get("regions", [])
            },
            
            # Spatial information
            "layout": {
                "page_dimensions": ocr_result.get("page_dimensions", {}),
                "text_regions": ocr_result.get("text_regions", []),
                "non_text_regions": ocr_result.get("non_text_regions", []),
                "reading_order": ocr_result.get("reading_order", [])
            },
            
            # Metadata
            "metadata": {
                "language": ocr_result.get("language", "unknown"),
                "confidence": ocr_result.get("confidence", 0.0),
                "processing_time": ocr_result.get("processing_time", 0.0),
                "timestamp": datetime.now().isoformat()
            },
            
            # Original results
            "raw_result": ocr_result
        }
        
        if metadata_context:
            metadata_context.add_adapter_data(
                adapter_name=self.__class__.__name__,
                key="extraction_preparation",
                value={
                    "source_type": "ocr",
                    "text_length": len(ocr_result.get("text", "")),
                    "has_blocks": len(ocr_result.get("blocks", [])) > 0,
                    "has_tables": len(ocr_result.get("tables", [])) > 0
                }
            )
            metadata_context.end_timing("prepare_for_extraction")
        
        return extraction_data

# Für Abwärtskompatibilität
OCRAdapter = BaseOCRAdapter