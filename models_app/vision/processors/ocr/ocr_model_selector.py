"""
Intelligenter OCR-Modell-Selektor

Wählt automatisch das am besten geeignete OCR-Modell basierend auf:
- Dokumententyp (akademisch, geschäftlich, allgemein)
- Sprachinhalt (einsprachig, mehrsprachig)
- Layout-Komplexität
- Vorhandensein von Formeln, Tabellen oder Diagrammen
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from PIL import Image
import time
import psutil
import copy
from datetime import datetime

from models_app.vision.ocr.base_adapter import BaseOCRAdapter
from error_handlers.models_app_errors.vision_errors.vision_errors import handle_ocr_errors, OCRError, DataQualityError, ProcessingTimeoutError, ResourceExhaustedError
from models_app.vision.ocr.utils.plugin_system import list_available_adapters, get_adapter_instance
from models_app.vision.utils.image_processing.core import load_image
from models_app.vision.utils.image_processing.analysis import analyze_image_complexity, detect_image_content_type
from analytics_app.utils import monitor_second_selector_performance

logger = logging.getLogger(__name__)

class OCRModelSelector:
    """
    Intelligenter Selektor für OCR-Modelle basierend auf Dokumenteneigenschaften.
    """
    
    def __init__(self, models_config: Optional[Dict] = None):
        """
        Initialisiert den OCR-Modell-Selektor.
        
        Args:
            models_config: Konfiguration für die verfügbaren OCR-Modelle
        """
        self.models_config = models_config or self._get_default_config()
        self.available_models = self._initialize_available_models()
        
        # Performance-Tracking für Modellauswahl
        self.model_performance = {}
    
    def _get_default_config(self) -> Dict:
        """
        Erstellt eine Standardkonfiguration für OCR-Modelle.
        
        Returns:
            Dict: Standardkonfiguration
        """
        return {
            "paddle_ocr": {
                "name": "PaddleOCR",
                "type": "general",
                "languages": ["en", "de", "fr", "es", "zh", "ja", "ko", "ru"],
                "strengths": ["multilingual", "performance", "accuracy"],
                "weaknesses": ["complex_layouts", "equations"],
                "open_source": True,
                "license": "Apache-2.0"
            },
            "easy_ocr": {
                "name": "EasyOCR",
                "type": "general",
                "languages": ["en", "de", "fr", "es", "zh", "ja", "ko", "ru", "ar", "hi"],
                "strengths": ["multilingual", "ease_of_use"],
                "weaknesses": ["complex_layouts", "performance"],
                "open_source": True,
                "license": "Apache-2.0"
            },
            "tesseract": {
                "name": "Tesseract OCR",
                "type": "general",
                "languages": ["en", "de", "fr", "es", "zh", "ja", "ko", "ru", "ar", "hi"],
                "strengths": ["widespread_use", "language_support"],
                "weaknesses": ["complex_layouts", "handwriting"],
                "open_source": True,
                "license": "Apache-2.0"
            },
            "doctr": {
                "name": "DocTR",
                "type": "document",
                "languages": ["en", "fr"],
                "strengths": ["document_layout", "tables", "forms"],
                "weaknesses": ["equations", "handwriting"],
                "open_source": True,
                "license": "Apache-2.0"
            },
            "nougat": {
                "name": "Nougat",
                "type": "academic",
                "languages": ["en"],
                "strengths": ["academic_papers", "equations", "complex_layouts"],
                "weaknesses": ["speed", "handwriting", "general_documents"],
                "open_source": True,
                "license": "Apache-2.0"
            },
            "layout_lm": {
                "name": "LayoutLM",
                "type": "document",
                "languages": ["en"],
                "strengths": ["layout_understanding", "form_understanding"],
                "weaknesses": ["multilingual", "equations"],
                "open_source": True,
                "license": "MIT"
            },
            "donut": {
                "name": "Donut",
                "type": "document",
                "languages": ["en"],
                "strengths": ["end_to_end", "document_understanding"],
                "weaknesses": ["multilingual", "customization"],
                "open_source": True,
                "license": "Apache-2.0"
            },
            "table_extraction": {
                "name": "TableExtractionAdapter",
                "type": "specialized",
                "languages": ["en", "de", "fr", "es", "zh", "ja", "ko", "ru"],
                "strengths": ["table_extraction", "structured_data"],
                "weaknesses": ["text_only_documents", "complex_tables"],
                "open_source": True,
                "license": "Custom"
            },
            "formula_recognition": {
                "name": "FormulaRecognitionAdapter",
                "type": "specialized",
                "languages": ["en"],
                "strengths": ["mathematical_formulas", "scientific_notation"],
                "weaknesses": ["general_text", "handwritten_formulas"],
                "open_source": True,
                "license": "Custom"
            },
            # Non-Open Source Modelle
            "microsoft_read": {
                "name": "Microsoft Read API",
                "type": "cloud",
                "languages": ["en", "de", "fr", "es", "zh", "ja", "ko", "ru"],
                "strengths": ["high_accuracy", "multilingual", "layout"],
                "weaknesses": ["cost", "privacy", "offline_usage"],
                "open_source": False,
                "license": "Commercial"
            },
            "google_vision": {
                "name": "Google Vision API",
                "type": "cloud",
                "languages": ["en", "de", "fr", "es", "zh", "ja", "ko", "ru"],
                "strengths": ["high_accuracy", "multilingual", "document_ai"],
                "weaknesses": ["cost", "privacy", "offline_usage"],
                "open_source": False,
                "license": "Commercial"
            },
            "amazon_textract": {
                "name": "Amazon Textract",
                "type": "cloud",
                "languages": ["en"],
                "strengths": ["forms", "tables", "documents"],
                "weaknesses": ["multilingual", "cost", "privacy"],
                "open_source": False,
                "license": "Commercial"
            },
            "abbyy": {
                "name": "ABBYY FineReader",
                "type": "commercial",
                "languages": ["en", "de", "fr", "es", "zh", "ja", "ko", "ru"],
                "strengths": ["high_accuracy", "layout_preservation", "format_conversion"],
                "weaknesses": ["cost", "integration_complexity"],
                "open_source": False,
                "license": "Commercial"
            }
        }
    
    def _initialize_available_models(self) -> Dict:
        """
        Initialisiert die verfügbaren OCR-Modelle basierend auf der Konfiguration.
        
        Returns:
            Dict: Dictionary mit verfügbaren Modellen
        """
        available_models = {}
        
        # PaddleOCR - Allgemeines OCR mit guter Mehrsprachenunterstützung
        if 'paddle_ocr' in self.models_config:
            available_models['paddle_ocr'] = {
                'name': 'PaddleOCR',
                'adapter': 'PaddleOCRAdapter',
                'config': self.models_config.get('paddle_ocr', {}),
                'priority': 80,
                'capabilities': {
                    'general_ocr': 0.9,
                    'multi_language': 0.8,
                    'handwriting': 0.7
                }
            }
        
        # Tesseract - Open-Source OCR mit breiter Sprachunterstützung
        if 'tesseract' in self.models_config:
            available_models['tesseract'] = {
                'name': 'Tesseract OCR',
                'adapter': 'TesseractAdapter',
                'config': self.models_config.get('tesseract', {}),
                'priority': 60,
                'capabilities': {
                    'general_ocr': 0.8,
                    'multi_language': 0.9,
                    'layout_analysis': 0.7
                }
            }
        
        # EasyOCR - Benutzerfreundliches OCR mit guter Mehrsprachenunterstützung
        if 'easyocr' in self.models_config:
            available_models['easyocr'] = {
                'name': 'EasyOCR',
                'adapter': 'EasyOCRAdapter',
                'config': self.models_config.get('easyocr', {}),
                'priority': 70,
                'capabilities': {
                    'general_ocr': 0.85,
                    'multi_language': 0.9,
                    'handwriting': 0.6
                }
            }
        
        # DocTR - Dokumenten-OCR mit Layout-Analyse
        if 'doctr' in self.models_config:
            available_models['doctr'] = {
                'name': 'DocTR',
                'adapter': 'DocTRAdapter',
                'config': self.models_config.get('doctr', {}),
                'priority': 75,
                'capabilities': {
                    'document_ocr': 0.9,
                    'layout_analysis': 0.8,
                    'table_detection': 0.7
                }
            }
        
        # Microsoft Read API - Proprietäres OCR mit hoher Genauigkeit
        if 'microsoft_read' in self.models_config:
            available_models['microsoft_read'] = {
                'name': 'Microsoft Read API',
                'adapter': 'MicrosoftReadAdapter',
                'config': self.models_config.get('microsoft_read', {}),
                'priority': 90,
                'capabilities': {
                    'general_ocr': 0.95,
                    'multi_language': 0.9,
                    'handwriting': 0.8,
                    'layout_analysis': 0.85
                }
            }
        
        # Nougat - Spezialisiert für akademische Dokumente und Formeln
        if 'nougat' in self.models_config:
            available_models['nougat'] = {
                'name': 'Nougat',
                'adapter': 'NougatAdapter',
                'config': self.models_config.get('nougat', {}),
                'priority': 85,
                'capabilities': {
                    'academic_documents': 0.95,
                    'math_formulas': 0.9,
                    'markdown_output': 1.0
                }
            }
        
        # LayoutLMv3 - Dokumentenverständnis mit Layout-Informationen
        if 'layoutlmv3' in self.models_config:
            available_models['layoutlmv3'] = {
                'name': 'LayoutLMv3',
                'adapter': 'LayoutLMv3Adapter',
                'config': self.models_config.get('layoutlmv3', {}),
                'priority': 80,
                'capabilities': {
                    'document_understanding': 0.9,
                    'layout_analysis': 0.95,
                    'entity_extraction': 0.85
                }
            }
        
        # TableExtractionAdapter - Spezialisiert für Tabellenerkennung
        if 'table_extraction' in self.models_config:
            available_models['table_extraction'] = {
                'name': 'Table Extraction',
                'adapter': 'TableExtractionAdapter',
                'config': self.models_config.get('table_extraction', {}),
                'priority': 75,
                'capabilities': {
                    'table_detection': 0.95,
                    'table_structure': 0.9,
                    'data_extraction': 0.85
                }
            }
        
        # DonutAdapter - Dokumentenverständnis und strukturierte Ausgabe
        if 'donut' in self.models_config:
            available_models['donut'] = {
                'name': 'Donut Document Understanding',
                'adapter': 'DonutAdapter',
                'config': self.models_config.get('donut', {}),
                'priority': 70,
                'capabilities': {
                    'document_understanding': 1.0,
                    'structured_output': 0.9,
                    'ocr_free': 1.0
                }
            }
        
        # FormulaRecognitionAdapter - Spezialisiert für mathematische Formeln
        if 'formula_recognition' in self.models_config:
            available_models['formula_recognition'] = {
                'name': 'Mathematical Formula Recognition',
                'adapter': 'FormulaRecognitionAdapter',
                'config': self.models_config.get('formula_recognition', {}),
                'priority': 65,
                'capabilities': {
                    'formula_recognition': 1.0,
                    'latex_output': 0.9,
                    'scientific_documents': 0.8
                }
            }
        
        return available_models
    
    @monitor_second_selector_performance
    @handle_ocr_errors
    def analyze_document(self, image) -> Dict[str, float]:
        """
        Analyze an image to determine its properties for model selection.
        
        Args:
            image: Path to image or image array
            
        Returns:
            Dictionary of document properties and scores
            
        Raises:
            OCRError: If there's an error analyzing the document
            DataQualityError: If the image quality is too low for analysis
            ProcessingTimeoutError: If analysis takes too long
            ResourceExhaustedError: If memory usage exceeds limits
        """
        start_time = time.time()
        
        try:
            # Ressourcennutzung überwachen
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # Ein Timeout für die Operation setzen
            max_processing_time = 60  # 60 Sekunden
            
            # Stelle sicher, dass wir ein Bild haben
            if isinstance(image, str) and os.path.isfile(image):
                # Wenn ein Dateipfad übergeben wurde, lade das Bild
                image_path = image
                image = cv2.imread(image)
                
                if image is None or image.size == 0:
                    raise DataQualityError("Failed to load image or empty image", error_code="empty_image")
            elif not isinstance(image, np.ndarray):
                raise OCRError("Invalid image format provided", error_code="invalid_image_format")
            else:
                image_path = None
                
            # Minimale Bildgröße prüfen
            if image.shape[0] < 50 or image.shape[1] < 50:
                raise DataQualityError("Image is too small for analysis", error_code="image_too_small")
                
            # Ressourcennutzung prüfen
            current_memory = process.memory_info().rss
            memory_used = (current_memory - initial_memory) / (1024*1024)  # MB
            if memory_used > 500:  # über 500MB Speichernutzung
                raise ResourceExhaustedError("Memory usage exceeded during image loading", 
                                          error_code="memory_limit_exceeded")
            
            # Dokumentauflösung berechnen (DPI-Schätzung)
            h, w = image.shape[:2]
            resolution = w * h / 1000000  # Megapixel
            
            # Analyze the document structure and content
            # 1. Detections from existing analyzers
            complexity = analyze_image_complexity(image)
            content_type = detect_image_content_type(image)
            
            # Überprüfe die verstrichene Zeit
            elapsed_time = time.time() - start_time
            if elapsed_time > max_processing_time:
                raise ProcessingTimeoutError(f"Document analysis exceeded time limit of {max_processing_time}s", 
                                          error_code="analysis_timeout")
            
            # 2. Dedicated table detection for model selection
            has_tables = self._check_if_has_tables(image) > 0.6
            
            # 3. Dedicated math formula detection
            has_formulas = self._check_if_has_equations(image) > 0.6
            
            # Get document complexity score (0-1)
            doc_complexity = self._check_if_complex_layout(image)
            
            # 4. Bildqualitätsmetriken und Preprocessing-Empfehlungen hinzufügen
            # Verwende den DocumentQualityAnalyzer für detaillierte Qualitätsanalyse
            try:
                from models_app.vision.document.factory.document_analyzer import DocumentQualityAnalyzer
                quality_analyzer = DocumentQualityAnalyzer()
                
                if image_path:
                    quality_metrics = quality_analyzer.analyze_image_quality(image_path)
                    # Preprocessing-Empfehlungen generieren
                    preprocessing_recommendations = quality_analyzer.generate_preprocessing_recommendations(quality_metrics)
                else:
                    quality_metrics = quality_analyzer.analyze_image_quality(image)
                    # Preprocessing-Empfehlungen generieren
                    preprocessing_recommendations = quality_analyzer.generate_preprocessing_recommendations(quality_metrics)
                    
                # Qualitätsmetriken direkt in die Ergebnisse integrieren
                quality_factors = {
                    'blur_score': quality_metrics.get('blur_score', 0.5),
                    'contrast_score': quality_metrics.get('contrast_score', 0.5),
                    'noise_level': quality_metrics.get('noise_level', 0.5),
                    'resolution_adequacy': quality_metrics.get('resolution_adequacy', 0.5),
                    'overall_quality': quality_metrics.get('overall_quality', 0.5)
                }
                
                # Komplexitätsanpassung basierend auf Bildqualität
                if quality_metrics.get('overall_quality', 0.5) < 0.4:
                    # Schlechte Bildqualität erhöht die Komplexität
                    complexity *= 1.5
                    # Bei niedriger Qualität mehr generelle OCR-Engines bevorzugen
                    is_low_quality = True
                else:
                    is_low_quality = False
                
                # OCR-Parameter-Empfehlungen basierend auf Bildqualität generieren
                ocr_params = self._generate_ocr_parameters(quality_metrics, preprocessing_recommendations)
                
            except Exception as e:
                logger.warning(f"Failed to analyze image quality: {e}")
                # Default-Werte setzen, wenn die Qualitätsanalyse fehlschlägt
                quality_factors = {
                    'blur_score': 0.5,
                    'contrast_score': 0.5,
                    'noise_level': 0.5,
                    'resolution_adequacy': 0.5,
                    'overall_quality': 0.5
                }
                preprocessing_recommendations = {"preprocessing_required": False, "recommended_methods": []}
                ocr_params = {}
                is_low_quality = False
                
            # 5. Weitere Dokumenteigenschaften prüfen
            academic_score = self._check_if_academic(image)
            is_multilingual = self._check_if_multilingual(image) > 0.6
            has_handwriting = self._check_if_has_handwriting(image) > 0.5
                
            # Ergebnisse zusammenfassen
            results = {
                "complexity": complexity,
                "content_type": content_type,
                "has_tables": has_tables,
                "has_formulas": has_formulas,
                "resolution": resolution,
                "academic_score": academic_score,
                "is_multilingual": is_multilingual,
                "has_handwriting": has_handwriting,
                "is_complex_layout": doc_complexity > self.complex_layout_threshold,
                "layout_complexity": doc_complexity,
                
                # Qualitätsmetriken hinzufügen
                **quality_factors,
                "low_quality": is_low_quality,
                
                # Preprocessing-Empfehlungen und OCR-Parameter hinzufügen
                "preprocessing_recommendations": preprocessing_recommendations,
                "ocr_params": ocr_params
            }
            
            logger.debug(f"Document analysis results: {results}")
            return results
            
        except (OCRError, DataQualityError, ProcessingTimeoutError, ResourceExhaustedError) as e:
            # Diese spezifischen Fehler durchreichen
            raise
        except Exception as e:
            # Allgemeine Fehler in OCRError umwandeln
            raise OCRError(f"Failed to analyze document: {str(e)}", error_code="analysis_failed") from e
            
    def _generate_ocr_parameters(self, quality_metrics: Dict[str, float], 
                                preprocessing_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generiert optimale OCR-Parameter basierend auf Bildqualitätsmetriken
        und Preprocessing-Empfehlungen.
        
        Args:
            quality_metrics: Dictionary mit Qualitätsmetriken
            preprocessing_recommendations: Dictionary mit Preprocessing-Empfehlungen
            
        Returns:
            Dictionary mit OCR-Parametern für verschiedene Engines
        """
        ocr_params = {
            "tesseract": {},
            "paddleocr": {},
            "doctr": {},
            "easyocr": {},
            "azure": {},
            "nougat": {},
            "default": {}
        }
        
        # Standardparameter für alle Engines
        default_params = {
            "preprocessing": [],
            "confidence_threshold": 0.3,
            "timeout_multiplier": 1.0
        }
        
        # Parameter basierend auf Qualitätsmetriken anpassen
        if quality_metrics.get("overall_quality", 0.5) < 0.3:
            # Bei sehr schlechter Qualität
            default_params["confidence_threshold"] = 0.2  # Niedrigerer Schwellwert
            default_params["timeout_multiplier"] = 1.5    # Mehr Zeit für die Verarbeitung
            
            # Spezifische Parameter für Tesseract
            ocr_params["tesseract"] = {
                "psm": 6,  # Page segmentation mode: Annahme eines einheitlichen Textblocks
                "oem": 1,  # OCR engine mode: LSTM only
                "config": "--dpi 300"
            }
            
            # Spezifische Parameter für PaddleOCR
            ocr_params["paddleocr"] = {
                "use_angle_cls": True,      # Rotationserkennung aktivieren
                "cls_thresh": 0.8,          # Höherer Schwellwert für Rotationserkennung
                "rec_thresh": 0.2,          # Niedrigerer Schwellwert für Zeichenerkennung
                "drop_score": 0.1           # Niedrigerer Schwellwert für Wortfilterung
            }
            
        elif quality_metrics.get("overall_quality", 0.5) < 0.6:
            # Bei mittlerer Qualität
            default_params["confidence_threshold"] = 0.25
            default_params["timeout_multiplier"] = 1.2
            
            ocr_params["tesseract"] = {
                "psm": 3,  # Automatische Seitensegmentierung ohne OSD
                "oem": 3   # Standard: LSTM + Legacy
            }
            
            ocr_params["paddleocr"] = {
                "use_angle_cls": True,
                "cls_thresh": 0.9,
                "rec_thresh": 0.3
            }
            
        # Parameter basierend auf spezifischen Qualitätsproblemen anpassen
        if quality_metrics.get("blur_score", 0.5) < 0.4:
            # Bei unscharfen Bildern
            if "preprocessing" not in default_params:
                default_params["preprocessing"] = []
            default_params["preprocessing"].append("deblur")
            
            # PaddleOCR ist besser bei unscharfen Bildern
            ocr_params["paddleocr"]["priority_boost"] = 0.2
            
        if quality_metrics.get("contrast_score", 0.5) < 0.4:
            # Bei Kontrastproblemen
            if "preprocessing" not in default_params:
                default_params["preprocessing"] = []
            default_params["preprocessing"].append("enhance_contrast")
            
            # EasyOCR ist oft besser bei schlechtem Kontrast
            ocr_params["easyocr"]["priority_boost"] = 0.15
            
        if quality_metrics.get("noise_level", 0.5) > 0.5:
            # Bei verrauschten Bildern
            if "preprocessing" not in default_params:
                default_params["preprocessing"] = []
            default_params["preprocessing"].append("denoise")
            
            # DocTR ist robuster bei Rauschen
            ocr_params["doctr"]["priority_boost"] = 0.2
            
        # Preprocessing-Methoden aus den Empfehlungen übernehmen
        for method in preprocessing_recommendations.get("recommended_methods", []):
            if method["method"] not in default_params.get("preprocessing", []):
                if "preprocessing" not in default_params:
                    default_params["preprocessing"] = []
                default_params["preprocessing"].append(method["method"])
                
                # Parameter für spezifische Methoden hinzufügen
                if "params" in method:
                    if method["method"] not in default_params:
                        default_params[method["method"]] = {}
                    default_params[method["method"]] = method["params"]
        
        # Default-Parameter auf alle Engines anwenden
        for engine in ocr_params:
            ocr_params[engine] = {**default_params, **ocr_params.get(engine, {})}
            
        ocr_params["default"] = default_params
        
        return ocr_params
    
    @monitor_second_selector_performance
    @handle_ocr_errors
    def select_model(self, image_path: str, metadata: Dict[str, Any] = None) -> BaseOCRAdapter:
        """Select the best OCR model for an image with metadata preservation."""
        # Ensure we don't modify original metadata
        metadata = copy.deepcopy(metadata) if metadata else {}
        
        try:
            # Analyze document if we don't have quality metrics
            if 'quality_metrics' not in metadata:
                analysis_result = self.analyze_document(image_path)
                ocr_metadata = {
                    'quality_metrics': analysis_result,
                    'ocr_analysis': {
                        'timestamp': datetime.now().isoformat(),
                        'processor': self.__class__.__name__
                    }
                }
                # Merge without overwriting
                metadata = self._merge_metadata(metadata, ocr_metadata)
            
            # 1. Dokument analysieren
            document_properties = self.analyze_document(image_path)
            logger.info(f"Document properties: {document_properties}")
            
            # 2. Sprachpriorität prüfen
            language_priority = metadata.get("language", "de").lower()
            
            # 3. Prüfe, ob nur Open-Source-Modelle verwendet werden sollen
            use_only_open_source = metadata.get("use_only_open_source", True)
            
            # 3.1 Prüfe, ob bestimmte Modelle ausgeschlossen werden sollen
            excluded_models = metadata.get("excluded_models", [])
            
            # Prüfe, ob Multi-Engine-Modus aktiviert werden soll
            multi_engine_mode = metadata.get("multi_engine", False)
            
            # Wenn Multi-Engine-Modus aktiviert ist, delegiere zu select_multi_engine
            if multi_engine_mode and not metadata.get("skip_multi_engine", False):
                engines = self.select_multi_engine(image_path, {**metadata, "skip_multi_engine": True})
                if engines:
                    return engines[0]  # Gib den ersten (besten) Adapter zurück
            
            # 4. Bildqualitätsmetriken und Preprocessing-Empfehlungen extrahieren
            low_quality = document_properties.get("low_quality", False)
            overall_quality = document_properties.get("overall_quality", 0.5)
            blur_score = document_properties.get("blur_score", 0.5)
            contrast_score = document_properties.get("contrast_score", 0.5)
            noise_level = document_properties.get("noise_level", 0.5)
            
            # OCR-Parameter extrahieren
            ocr_params = document_properties.get("ocr_params", {})
            preprocessing_recommendations = document_properties.get("preprocessing_recommendations", {})
            
            # Logge Bildqualitätsmetriken für Debugging-Zwecke
            if "overall_quality" in document_properties:
                logger.info(f"Image quality metrics: overall={overall_quality}, blur={blur_score}, contrast={contrast_score}, noise={noise_level}")
                logger.debug(f"OCR parameters: {ocr_params}")
                logger.debug(f"Preprocessing recommendations: {preprocessing_recommendations}")
            
            # 5. Erstelle eine gewichtete Liste von geeigneten Modellen
            model_scores = {}
            
            # Durchlaufe alle verfügbaren Modelle
            for model_id, model_info in self.models_config.items():
                # Prüfe, ob das Modell ausgeschlossen werden soll
                if model_id in excluded_models:
                    continue
                    
                # Prüfe, ob das Modell die Sprache unterstützt
                language_supported = language_priority in model_info["languages"]
                
                # Wenn die Sprache nicht unterstützt wird, prüfe, ob Englisch unterstützt wird
                if not language_supported and "en" in model_info["languages"]:
                    language_supported = True
                    language_penalty = 0.2  # Abwertung, wenn die bevorzugte Sprache nicht unterstützt wird
                else:
                    language_penalty = 0.0
                    
                # Prüfe, ob Open-Source-Anforderung erfüllt ist
                if use_only_open_source and not model_info.get("open_source", False):
                    continue
                    
                # Basispunkte
                base_score = model_info.get("base_score", 0.5)
                
                # Anfangspunktzahl basierend auf Basis und Sprache
                score = base_score - language_penalty
                
                # NEU: Qualitätsabhängige Bewertungsanpassung
                # Bei niedriger Qualität bevorzugen wir robustere Modelle
                if low_quality:
                    if "low_quality" in model_info.get("strengths", []):
                        score += 0.3
                    elif "low_quality" in model_info.get("weaknesses", []):
                        score -= 0.3
                        
                    # Allgemeine Qualitätsanpassungen
                    if blur_score < 0.4 and "blur" in model_info.get("weaknesses", []):
                        score -= 0.2
                    elif blur_score < 0.4 and "blur" in model_info.get("strengths", []):
                        score += 0.2
                        
                    if contrast_score < 0.4 and "low_contrast" in model_info.get("weaknesses", []):
                        score -= 0.2
                    elif contrast_score < 0.4 and "low_contrast" in model_info.get("strengths", []):
                        score += 0.2
                        
                    if noise_level > 0.5 and "noise" in model_info.get("weaknesses", []):
                        score -= 0.2
                    elif noise_level > 0.5 and "noise" in model_info.get("strengths", []):
                        score += 0.2
                
                # Dokumenttyp-spezifische Gewichtung
                if document_properties["academic_score"] > 0.7:
                    # Akademische Dokumente
                    if "academic_papers" in model_info.get("strengths", []) or "equations" in model_info.get("strengths", []):
                        score += 0.3
                    if "equations" in model_info.get("weaknesses", []) and document_properties["has_formulas"] > 0.5:
                        score -= 0.3
                
                # Komplexität des Layouts
                if document_properties["layout_complexity"] > 0.7:
                    if "complex_layouts" in model_info.get("strengths", []) or "document_layout" in model_info.get("strengths", []):
                        score += 0.2
                    if "complex_layouts" in model_info.get("weaknesses", []):
                        score -= 0.2
                        
                # Mehrsprachigkeit
                if document_properties["is_multilingual"]:
                    if "multilingual" in model_info.get("strengths", []):
                        score += 0.25
                    if "multilingual" in model_info.get("weaknesses", []):
                        score -= 0.25
                        
                # Tabellen
                if document_properties["has_tables"]:
                    if "tables" in model_info.get("strengths", []) or "table_extraction" in model_info.get("strengths", []):
                        score += 0.35
                    if "tables" in model_info.get("weaknesses", []):
                        score -= 0.2
                        
                # Formeln/Gleichungen
                if document_properties["has_formulas"]:
                    if "equations" in model_info.get("strengths", []) or "formula_recognition" in model_info.get("strengths", []):
                        score += 0.4
                    if "equations" in model_info.get("weaknesses", []):
                        score -= 0.3
                        
                # Handschrift
                if document_properties.get("has_handwriting", False):
                    if "handwriting" in model_info.get("strengths", []):
                        score += 0.3
                    if "handwriting" in model_info.get("weaknesses", []):
                        score -= 0.4
                        
                # Prioritätsboost aus OCR-Parametern hinzufügen, falls vorhanden
                if model_id in ocr_params and "priority_boost" in ocr_params[model_id]:
                    score += ocr_params[model_id]["priority_boost"]
                    
                # Füge das Modell zur Liste der bewerteten Modelle hinzu
                model_scores[model_id] = score
            
            # 6. Sortiere die Modelle nach Punktzahl
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            if not sorted_models:
                # Keine passenden Modelle gefunden
                logger.warning("No suitable OCR models found. Using default model.")
                # Fallback zum Default-Modell
                default_model = "paddleocr"  # Oder ein anderes Default-Modell
                
                # Suche nach ersten vorhandenen Fallback-Modell
                for fallback in ["paddleocr", "tesseract", "doctr", "easyocr"]:
                    if fallback in self.models_config and (not use_only_open_source or 
                                                        self.models_config[fallback].get("open_source", False)):
                        default_model = fallback
                        break
                        
                # Erstelle und konfiguriere den Adapter
                adapter = self._create_adapter_for_model(default_model)
                
                # OCR-Parameter anwenden, falls verfügbar
                self._apply_ocr_parameters(adapter, default_model, ocr_params, preprocessing_recommendations)
                
                return adapter
            
            # Das Modell mit der höchsten Punktzahl auswählen
            best_model_id = sorted_models[0][0]
            best_model_score = sorted_models[0][1]
            
            # Zweites Modell für Multi-Engine-Ansatz (falls vorhanden)
            second_best_model_id = sorted_models[1][0] if len(sorted_models) > 1 else None
            
            logger.info(f"Selected model {best_model_id} with score {best_model_score}")
            if second_best_model_id:
                logger.info(f"Second best model: {second_best_model_id} with score {model_scores[second_best_model_id]}")
            
            # 7. Spezialisierte Adapter für bestimmte Dokumenttypen
            # 7.1 Akademische Dokumente mit Formeln
            if document_properties["academic_score"] > 0.8 and document_properties["has_formulas"]:
                if "nougat" in self.models_config and not use_only_open_source:
                    logger.info("Academic document with formulas. Using Nougat.")
                    adapter = self.get_model_instance("nougat")
                    
                    # OCR-Parameter anwenden
                    self._apply_ocr_parameters(adapter, "nougat", ocr_params, preprocessing_recommendations)
                    
                    return adapter
            
            # 7.2 Tabellenerkennung
            if document_properties["has_tables"] or metadata.get('contains_tables', False):
                if "table_extraction" not in excluded_models:
                    logger.info("Document contains tables. Using TableExtractionAdapter.")
                    adapter = self.get_model_instance("table_extraction")
                    
                    # OCR-Parameter anwenden
                    self._apply_ocr_parameters(adapter, "table_extraction", ocr_params, preprocessing_recommendations)
                    
                    return adapter
                
            # 7.3 Formelerkennung
            if document_properties["has_formulas"] or metadata.get('contains_equations', False):
                logger.info("Document contains many equations. Using FormulaRecognitionAdapter or Nougat.")
                
                formula_model = None
                if "formula_recognition" not in excluded_models:
                    formula_model = "formula_recognition"
                elif "nougat" not in excluded_models and not use_only_open_source:
                    formula_model = "nougat"
                    
                if formula_model:
                    adapter = self.get_model_instance(formula_model)
                    
                    # OCR-Parameter anwenden
                    self._apply_ocr_parameters(adapter, formula_model, ocr_params, preprocessing_recommendations)
                    
                    return adapter
            
            # 8. Mehrere-Engine-Ansatz für komplexe Dokumente mit gemischten Inhalten
            # Wenn das Dokument Tabellen UND Text enthält, kombinieren wir ggf. mehrere Engines
            if (document_properties["has_tables"] and 
                second_best_model_id and not multi_engine_mode):
                # Prüfe, ob Multi-Engine-Processing aktiviert werden soll
                enable_multi_engine = (
                    metadata.get("enable_multi_engine", False) or
                    document_properties["layout_complexity"] > 0.7 or
                    (document_properties["has_tables"] and document_properties["has_formulas"])
                )
                
                if enable_multi_engine:
                    # Rekursiver Aufruf mit aktiviertem Multi-Engine-Modus
                    return self.select_model(image_path, {**metadata, "multi_engine": True})
            
            # 9. Standard-Fall: Das Modell mit der höchsten Punktzahl verwenden
            adapter = self.get_model_instance(best_model_id)
            
            # OCR-Parameter anwenden
            self._apply_ocr_parameters(adapter, best_model_id, ocr_params, preprocessing_recommendations)
            
            # Record selection in metadata
            selection_metadata = {
                'selected_model': best_model_id,
                'selection_reason': "highest_score",
                'confidence': best_model_score,
                'alternatives': [model_id for model_id, score in sorted_models[:3]]
            }
            
            # Merge without overwriting
            metadata['ocr_selection'] = selection_metadata
            
            return adapter
            
        except Exception as e:
            # Error handling
            logger.error(f"Error selecting OCR model: {str(e)}")
            # Fallback zur ersten Engine
            return self.get_model_instance("tesseract")
    
    def _merge_metadata(self, existing_metadata, ocr_metadata):
        """Merge OCR-specific metadata with existing metadata without overwriting."""
        if not existing_metadata:
            return ocr_metadata
        
        # Create deep copy to avoid modifying original
        result = copy.deepcopy(existing_metadata)
        
        # Only add OCR-specific keys that don't already exist
        for key, value in ocr_metadata.items():
            if key not in result:
                result[key] = value
            elif key == 'quality_metrics' and 'quality_metrics' in result:
                # Special handling for quality metrics - merge them
                result['quality_metrics'].update(value)
            
        return result
    
    def get_model_instance(self, model_id: str) -> BaseOCRAdapter:
        """
        Gibt eine Instanz des angeforderten OCR-Modells zurück.
        
        Args:
            model_id: ID des angeforderten Modells
            
        Returns:
            OCRAdapter: Adapter für das Modell oder None bei Fehler
        """
        if model_id not in self.available_models:
            logger.error(f"OCR-Modell {model_id} ist nicht verfügbar")
            return None
        
        model_info = self.available_models[model_id]
        
        # Lade das Modell, falls noch nicht geschehen
        if not model_info["loaded"]:
            try:
                # Modell über den entsprechenden Adapter laden
                adapter = self._create_adapter_for_model(model_id)
                if adapter and adapter.initialize():
                    model_info["instance"] = adapter
                    model_info["loaded"] = True
                    logger.info(f"OCR-Modell {model_id} wurde geladen")
                else:
                    logger.error(f"Fehler beim Laden des OCR-Modells {model_id}")
                    return None
            except Exception as e:
                logger.error(f"Fehler beim Laden des OCR-Modells {model_id}: {str(e)}")
                return None
        
        return model_info["instance"]
    
    def _create_adapter_for_model(self, model_id: str) -> BaseOCRAdapter:
        """
        Erstellt eine Adapter-Instanz für das angegebene Modell.
        
        Args:
            model_id: ID des Modells
            
        Returns:
            BaseOCRAdapter: Adapter-Instanz
        """
        if model_id not in self.available_models:
            raise ValueError(f"Unbekanntes Modell: {model_id}")
        
        model_info = self.available_models[model_id]
        adapter_name = model_info['adapter']
        config = model_info['config']
        
        # Importiere den entsprechenden Adapter
        if adapter_name == 'PaddleOCRAdapter':
            from .paddle_adapter import PaddleOCRAdapter
            return PaddleOCRAdapter(config=config)
        elif adapter_name == 'TesseractAdapter':
            from .tesseract_adapter import TesseractAdapter
            return TesseractAdapter(config=config)
        elif adapter_name == 'EasyOCRAdapter':
            from .easyocr_adapter import EasyOCRAdapter
            return EasyOCRAdapter(config=config)
        elif adapter_name == 'DocTRAdapter':
            from .doctr_adapter import DocTRAdapter
            return DocTRAdapter(config=config)
        elif adapter_name == 'MicrosoftReadAdapter':
            from .microsoft_adapter import MicrosoftReadAdapter
            return MicrosoftReadAdapter(config=config)
        elif adapter_name == 'NougatAdapter':
            from .nougat_adapter import NougatAdapter
            return NougatAdapter(config=config)
        elif adapter_name == 'LayoutLMv3Adapter':
            from .layoutlmv3_adapter import LayoutLMv3Adapter
            return LayoutLMv3Adapter(config=config)
        elif adapter_name == 'TableExtractionAdapter':
            from .table_extraction_adapter import TableExtractionAdapter
            return TableExtractionAdapter(config=config)
        elif adapter_name == 'DonutAdapter':
            from .donut_adapter import DonutAdapter
            return DonutAdapter(config=config)
        elif adapter_name == 'FormulaRecognitionAdapter':
            from .formula_recognition_adapter import FormulaRecognitionAdapter
            return FormulaRecognitionAdapter(config=config)
        else:
            raise ValueError(f"Unbekannter Adapter-Typ: {adapter_name}")
    
    # Hilfsmethoden zur Dokumentenanalyse
    
    def _check_if_academic(self, image) -> float:
        """
        Prüft, ob es sich um ein akademisches Dokument handelt.
        
        Erkennt Merkmale, die typisch für wissenschaftliche Papers, Lehrbücher und akademische
        Dokumente sind:
        - Zweispalten-Layout
        - Vorkommen von Referenzen/Zitaten
        - Gleichungen und mathematische Notation
        - Vorkommen von Abbildungsbeschriftungen
        
        Args:
            image: Bild als NumPy-Array oder Pfad
            
        Returns:
            float: Wahrscheinlichkeit für akademisches Dokument (0-1)
        """
        try:
            # Bild laden, falls es ein Pfad ist
            if isinstance(image, str):
                import cv2
                img = cv2.imread(image)
            else:
                img = image.copy()
                
            # Bild in Graustufen konvertieren
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Überprüfen auf Mehrspaltigkeit (typisch für akademische Papers)
            h, w = gray.shape
            
            # Vertikale Summe für Spaltenerkennung 
            vert_sum = np.sum(gray, axis=0)
            vert_sum = vert_sum / np.max(vert_sum)  # Normalisieren
            
            # Glättung für bessere Erkennung
            from scipy.signal import savgol_filter
            try:
                smooth_sum = savgol_filter(vert_sum, 51, 3)
            except:
                smooth_sum = vert_sum
                
            # Suche nach Tälern (weiße Spalten zwischen Textblöcken)
            valleys = []
            for i in range(1, len(smooth_sum)-1):
                if smooth_sum[i] < smooth_sum[i-1] and smooth_sum[i] < smooth_sum[i+1] and smooth_sum[i] < 0.5:
                    valleys.append(i)
                    
            # Bei 2+ Spalten: typisch für akademische Dokumente
            has_columns = len(valleys) >= 1 and 0.2*w < valleys[0] < 0.8*w
            
            # 2. Suche nach Referenzmustern ([1], [2] usw.) oder "et al."
            import pytesseract
            try:
                text = pytesseract.image_to_string(gray)
                has_references = "[" in text and "]" in text and any(f"[{i}]" in text for i in range(1, 10))
                has_citations = "et al." in text.lower() or "references" in text.lower()
            except:
                has_references = False
                has_citations = False
                
            # 3. Überprüfen auf Gleichungen (ruft unsere bestehende Methode auf)
            has_equations = self._check_if_has_equations(img) > 0.6
                
            # Kombination der Signale
            signals = [
                has_columns * 0.3,  # Spaltenlayout
                has_references * 0.3,  # Referenzmuster
                has_citations * 0.2,  # Zitationsstil
                has_equations * 0.4   # Gleichungen
            ]
            
            # Gewichtete Summe, maximal 1.0
            probability = min(1.0, sum(signals))
            
            return probability
            
        except Exception as e:
            logger.warning(f"Fehler bei der akademischen Dokumenterkennung: {str(e)}")
            return 0.3  # Fallback-Wert

    def _check_if_multilingual(self, image) -> float:
        """
        Prüft, ob das Dokument mehrsprachig ist durch Erkennung verschiedener Zeichensätze.
        
        Args:
            image: Bild als NumPy-Array oder Pfad
            
        Returns:
            float: Wahrscheinlichkeit für mehrsprachigen Inhalt (0-1)
        """
        try:
            # Text mit Tesseract extrahieren
            if isinstance(image, str):
                import cv2
                img = cv2.imread(image)
            else:
                img = image.copy()
            
            # Verschiedene Zeichensatzgruppen definieren
            import re
            char_groups = {
                'latin': r'[a-zA-Z]',
                'cyrillic': r'[а-яА-Я]',
                'arabic': r'[\u0600-\u06FF]',
                'chinese': r'[\u4e00-\u9fff]',
                'japanese': r'[\u3040-\u30ff]',
                'korean': r'[\uac00-\ud7a3]',
                'greek': r'[\u0370-\u03FF]',
                'devanagari': r'[\u0900-\u097F]'
            }
            
            # Text aus dem Bild extrahieren
            import pytesseract
            try:
                text = pytesseract.image_to_string(img, lang='eng+deu+fra+spa+rus+ara+jpn+chi_sim+kor')
            except:
                text = pytesseract.image_to_string(img)
                
            # Zählen, wie viele verschiedene Zeichensätze erkannt wurden
            char_groups_found = 0
            for group, pattern in char_groups.items():
                if re.search(pattern, text):
                    char_groups_found += 1
                    
            # Je mehr Zeichensätze, desto wahrscheinlicher ist das Dokument mehrsprachig
            if char_groups_found >= 3:
                return 0.9
            elif char_groups_found == 2:
                return 0.7
            elif char_groups_found == 1:
                return 0.1
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Fehler bei der Mehrsprachigkeitserkennung: {str(e)}")
            return 0.2  # Fallback-Wert

    def _check_if_complex_layout(self, image) -> float:
        """
        Prüft, ob das Dokument ein komplexes Layout hat (Mehrspaltigkeit, Bilder, Tabellen etc.).
        
        Args:
            image: Bild als NumPy-Array oder Pfad
            
        Returns:
            float: Komplexitätswert zwischen 0 und 1
        """
        try:
            # Bild laden, falls es ein Pfad ist
            if isinstance(image, str):
                import cv2
                img = cv2.imread(image)
            else:
                img = image.copy()
                
            # Bild in Graustufen konvertieren
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # 1. Anzahl der Konturen als Maß für Layoutkomplexität
            # Binarisieren und Kanten finden
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Normalisiere Konturenanzahl
            contour_complexity = min(1.0, len(contours) / 100)
            
            # 2. Varianz in der Größe und Form der Konturen (unterschiedliche Elemente)
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
                
            # 3. Überprüfen auf Mehrspaltigkeit
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
            complexity = 0.3 * contour_complexity + 0.3 * shape_complexity + 0.2 * area_complexity + 0.2 * column_complexity
            
            return complexity
            
        except Exception as e:
            logger.warning(f"Fehler bei der Layoutkomplexitätsanalyse: {str(e)}")
            return 0.5  # Fallback-Wert

    def _check_if_has_tables(self, image) -> float:
        """
        Prüft, ob das Dokument Tabellen enthält.
        
        Args:
            image: Bild als NumPy-Array oder Pfad
            
        Returns:
            float: Wahrscheinlichkeit für das Vorhandensein von Tabellen (0-1)
        """
        try:
            # Bild laden, falls es ein Pfad ist
            if isinstance(image, str):
                import cv2
                img = cv2.imread(image)
            else:
                img = image.copy()
                
            # Bild in Graustufen konvertieren
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Linien erkennen mit Hough-Transformation (typisch für Tabellen)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is None or len(lines) == 0:
                return 0.1  # Keine Linien gefunden
                
            # Horizontal und vertikal verlaufende Linien analysieren
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Horizontale Linie
                if abs(y2 - y1) < 10:
                    horizontal_lines += 1
                # Vertikale Linie
                if abs(x2 - x1) < 10:
                    vertical_lines += 1
                    
            # Kreuzende Linien sind typisch für Tabellen
            # Je mehr Linien, desto wahrscheinlicher ist eine Tabelle
            if horizontal_lines > 3 and vertical_lines > 3:
                return 0.9
            elif horizontal_lines > 2 and vertical_lines > 2:
                return 0.7
            elif horizontal_lines > 1 and vertical_lines > 1:
                return 0.5
            else:
                return 0.2
                
        except Exception as e:
            logger.warning(f"Fehler bei der Tabellenerkennung: {str(e)}")
            return 0.4  # Fallback-Wert
    
    def _check_if_has_equations(self, image) -> float:
        """
        Prüft, ob das Bild mathematische Gleichungen enthält.
        
        Args:
            image: Bild als NumPy-Array oder Pfad
            
        Returns:
            float: Wahrscheinlichkeit für das Vorhandensein von Gleichungen (0-1)
        """
        try:
            # Bild laden, falls es ein Pfad ist
            if isinstance(image, str):
                import cv2
                img = cv2.imread(image)
            else:
                img = image.copy()
            
            # Zu Graustufen konvertieren
            import cv2
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Kanten erkennen (mathematische Symbole haben oft klare Kanten)
            edges = cv2.Canny(gray, 50, 150)
            
            # Konturen finden
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Heuristiken für mathematische Symbole:
            # 1. Viele kleine Konturen (Symbole, Indizes, etc.)
            # 2. Horizontale Linien (Brüche)
            # 3. Quadratwurzel-ähnliche Formen
            
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
    
    def _check_if_has_handwriting(self, image) -> float:
        """Prüft, ob das Dokument Handschrift enthält."""
        # In der Produktion würde hier eine Handschrifterkennung stattfinden
        return 0.2  # 20% Wahrscheinlichkeit für Handschrift
    
    def _detect_handwriting(self, image_path: str) -> bool:
        """
        Detect if an image contains handwriting.
        This is a simple heuristic based on stroke variation and irregularity.
        
        Args:
            image_path: Path to the image file.
        
        Returns:
            True if handwriting is detected, False otherwise.
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contour properties
            if len(contours) > 0:
                # Calculate average contour area and perimeter
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
                perimeters = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) > 10]
                
                if not areas or not perimeters:
                    return False
                
                # Calculate average area and perimeter
                avg_area = sum(areas) / len(areas)
                avg_perimeter = sum(perimeters) / len(perimeters)
                
                # Calculate standard deviation
                area_std = np.std(areas)
                perimeter_std = np.std(perimeters)
                
                # Handwriting typically has higher variation in contour properties
                area_variation = area_std / avg_area if avg_area > 0 else 0
                perimeter_variation = perimeter_std / avg_perimeter if avg_perimeter > 0 else 0
                
                # Heuristic threshold for handwriting detection
                if area_variation > 0.5 and perimeter_variation > 0.3:
                    return True
            
            return False
        except Exception as e:
            logger.warning(f"Error in handwriting detection: {str(e)}")
            return False
    
    def get_available_models(self, only_open_source: bool = False) -> List[Dict[str, Any]]:
        """
        Gibt eine Liste aller verfügbaren OCR-Modelle zurück.
        
        Args:
            only_open_source: Wenn True, werden nur Open-Source-Modelle zurückgegeben
            
        Returns:
            Liste der verfügbaren Modelle mit ihren Eigenschaften
        """
        models = []
        
        for model_id, model_info in self.available_models.items():
            config = model_info["config"]
            
            # Filtere nach Open-Source-Status, falls gewünscht
            if only_open_source and not config.get("open_source", False):
                continue
            
            models.append({
                "id": model_id,
                "name": config["name"],
                "type": config["type"],
                "languages": config["languages"],
                "open_source": config.get("open_source", False),
                "license": config.get("license", "Unknown"),
                "loaded": model_info["loaded"]
            })
        
        return models
    
    def select_multi_engine(self, image_path: str, metadata: Dict[str, Any] = None) -> List[BaseOCRAdapter]:
        """
        Wählt mehrere OCR-Engines aus, die für verschiedene Aspekte eines komplexen Dokuments geeignet sind.
        
        Diese Methode wird verwendet, wenn ein Dokument gemischte Inhalte enthält (z.B. Text, Tabellen, Formeln),
        für die unterschiedliche OCR-Engines optimal sind.
        
        Args:
            image_path: Pfad zum Bild
            metadata: Zusätzliche Metadaten über das Dokument
            
        Returns:
            Liste von OCR-Adapter-Instanzen, sortiert nach Relevanz
        """
        metadata = metadata or {}
        metadata["multi_engine"] = True  # Markiere als Multi-Engine-Auswahl
        
        # Dokument analysieren
        document_properties = self.analyze_document(image_path)
        logger.info(f"Multi-engine selection: Document properties: {document_properties}")
        
        # Sprachpriorität prüfen
        language_priority = metadata.get("language", "de").lower()
        
        # Prüfe, ob nur Open-Source-Modelle verwendet werden sollen
        use_only_open_source = metadata.get("use_only_open_source", True)
        
        # Geeignete Engines sammeln
        selected_engines = []
        selected_engine_ids = set()  # Verhindere Duplikate
        
        # 1. Spezialisierte Adapter für Tabellen, wenn vorhanden
        if document_properties["has_tables"]:
            table_adapter = self.get_model_instance("table_extraction")
            selected_engines.append({
                "adapter": table_adapter,
                "purpose": "tables",
                "score": document_properties["has_tables"],
                "id": "table_extraction"
            })
            selected_engine_ids.add("table_extraction")
        
        # 2. Spezialisierte Adapter für Formeln, wenn vorhanden
        if document_properties["has_formulas"]:
            if "nougat" in self.models_config and (not use_only_open_source or self.models_config["nougat"].get("open_source", False)):
                formula_adapter = self.get_model_instance("nougat")
                formula_adapter_id = "nougat"
            else:
                formula_adapter = self.get_model_instance("formula_recognition")
                formula_adapter_id = "formula_recognition"
                
            selected_engines.append({
                "adapter": formula_adapter,
                "purpose": "formulas",
                "score": document_properties["has_formulas"],
                "id": formula_adapter_id
            })
            selected_engine_ids.add(formula_adapter_id)
        
        # 3. Allgemeiner Texterkennungsadapter
        # Das beste allgemeine OCR-Modell für den Rest des Dokuments auswählen
        # (Überschreibt die Fallback-Parameter, um sicherzustellen, dass wir ein anderes Modell als die bereits ausgewählten bekommen)
        main_adapter = self.select_model(image_path, {
            **metadata,
            "excluded_models": list(selected_engine_ids)
        })
        
        # ID des Hauptadapters ermitteln
        main_adapter_id = None
        for model_id, info in self.available_models.items():
            if info.get('adapter') == main_adapter.__class__.__name__:
                main_adapter_id = model_id
                break
                
        if main_adapter_id and main_adapter_id not in selected_engine_ids:
            selected_engines.append({
                "adapter": main_adapter,
                "purpose": "general_text",
                "score": 1.0,  # Hauptadapter hat immer hohe Priorität
                "id": main_adapter_id
            })
        
        # 4. Sortiere Engines nach ihrer Relevanz (Score)
        selected_engines.sort(key=lambda x: x["score"], reverse=True)
        
        # Nur die Adapter zurückgeben
        return [engine["adapter"] for engine in selected_engines]
    
    def combine_ocr_results(self, image_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Kombiniert die Ergebnisse mehrerer OCR-Engines für ein Dokument mit gemischten Inhalten.
        
        Diese Methode führt folgende Schritte aus:
        1. Wählt mehrere geeignete OCR-Engines für verschiedene Dokumentbereiche aus
        2. Verarbeitet das Dokument mit jeder Engine und sammelt die Ergebnisse
        3. Kombiniert die Ergebnisse basierend auf dem Dokumenttyp und den Enginefähigkeiten
        
        Args:
            image_path: Pfad zum Bild
            metadata: Zusätzliche Metadaten über das Dokument
            
        Returns:
            Dictionary mit kombinierten OCR-Ergebnissen
        """
        from models_app.vision.utils.image_processing.segmentation import segment_document
        from models_app.vision.utils.image_processing.core import load_image
        
        metadata = metadata or {}
        metadata["multi_engine"] = True
        
        # 1. Ausgewählte Engines abrufen
        engines = self.select_multi_engine(image_path, metadata)
        
        if not engines:
            logger.warning("No suitable engines found. Falling back to single engine mode.")
            return self.get_model_instance("tesseract").process_image(image_path, metadata)
        
        logger.info(f"Selected {len(engines)} engines for multi-engine OCR.")
        
        # 2. Dokument segmentieren, um Bereiche für spezialisierte Engines zu identifizieren
        try:
            # Bild laden
            image, np_image, _ = load_image(image_path)
            
            # Dokument in semantische Regionen segmentieren
            segments = segment_document(np_image)
            
            # Leeres Ergebnis vorbereiten
            combined_result = {
                "text": "",
                "blocks": [],
                "confidence": 0.0,
                "metadata": {
                    "multi_engine": True,
                    "engines_used": [e.__class__.__name__ for e in engines],
                    "processing_mode": "hybrid"
                }
            }
            
            # 3. Jeder Engine die passenden Segmente zuweisen
            # Segmenttypen: 'text', 'table', 'formula', 'image', 'title', 'caption', etc.
            engine_assignments = {}
            
            if segments:
                # 3.1 Segmente den spezialisierten Engines zuweisen
                for engine in engines:
                    engine_name = engine.__class__.__name__
                    engine_capabilities = getattr(engine, "ADAPTER_INFO", {}).get("capabilities", {})
                    
                    if engine_capabilities.get("table_extraction"):
                        # Tabellen-spezialisierte Engine
                        engine_assignments[engine_name] = [
                            segment for segment in segments if segment["type"] == "table"
                        ]
                    elif engine_capabilities.get("formula_recognition"):
                        # Formel-spezialisierte Engine
                        engine_assignments[engine_name] = [
                            segment for segment in segments if segment["type"] == "formula"
                        ]
                    else:
                        # Allgemeine OCR-Engine für Text
                        engine_assignments[engine_name] = [
                            segment for segment in segments 
                            if segment["type"] in ["text", "title", "paragraph", "caption", "header", "footer"]
                        ]
                        
                # 3.2 Wenn keine Segmentierung verfügbar ist, verwende die erste Engine für das gesamte Dokument
                if not any(engine_assignments.values()):
                    engine_name = engines[0].__class__.__name__
                    engine_assignments[engine_name] = [{"type": "document", "bbox": [0, 0, np_image.shape[1], np_image.shape[0]]}]
            else:
                # Keine Segmentierung verfügbar, verwende einfachen Ansatz
                engine_confidences = {}
                all_results = {}
                
                # Jede Engine verarbeitet das gesamte Dokument
                for engine in engines:
                    engine_name = engine.__class__.__name__
                    
                    # Füge bereits_verarbeitet-Flag hinzu, wenn mehrere Engines dasselbe Bild verarbeiten
                    engine_metadata = metadata.copy()
                    if len(engines) > 1:
                        engine_metadata["already_preprocessed"] = True
                        
                    # Verarbeite das Bild mit der Engine
                    engine_result = engine.process_image(image_path, engine_metadata)
                    
                    # Speichere Ergebnis und Konfidenz
                    all_results[engine_name] = engine_result
                    engine_confidences[engine_name] = engine_result.get("confidence", 0.0)
                    
                # Wähle das Ergebnis mit der höchsten Konfidenz
                if engine_confidences:
                    best_engine = max(engine_confidences.items(), key=lambda x: x[1])[0]
                    return all_results[best_engine]
                else:
                    return combined_result
            
            # 4. OCR auf segmentierten Bereichen mit entsprechenden Engines durchführen
            segment_results = []
            
            for engine in engines:
                engine_name = engine.__class__.__name__
                assigned_segments = engine_assignments.get(engine_name, [])
                
                for segment in assigned_segments:
                    # Extrahiere Segment aus dem Originalbild
                    x1, y1, x2, y2 = segment["bbox"]
                    segment_image = np_image[y1:y2, x1:x2]
                    
                    # OCR auf dem Segment durchführen
                    try:
                        segment_metadata = metadata.copy()
                        segment_metadata["segment_type"] = segment["type"]
                        
                        segment_result = engine.process_image(segment_image, segment_metadata)
                        
                        # Koordinaten der erkannten Blöcke auf das Originalbild normalisieren
                        for block in segment_result.get("blocks", []):
                            # Bounding-Box anpassen
                            if "bbox" in block:
                                bx1, by1, bx2, by2 = block["bbox"]
                                block["bbox"] = [x1 + bx1, y1 + by1, x1 + bx2, y1 + by2]
                                
                            # Polygon anpassen, falls vorhanden
                            if "polygon" in block:
                                polygon = block["polygon"]
                                block["polygon"] = [[x1 + p[0], y1 + p[1]] for p in polygon]
                                
                            # Segmenttyp hinzufügen
                            block["segment_type"] = segment["type"]
                            block["engine"] = engine_name
                        
                        # Ergebnis zum kombinierten Ergebnis hinzufügen
                        segment_results.append({
                            "text": segment_result.get("text", ""),
                            "blocks": segment_result.get("blocks", []),
                            "confidence": segment_result.get("confidence", 0.0),
                            "segment_type": segment["type"],
                            "engine": engine_name
                        })
                    except Exception as e:
                        logger.error(f"Error processing segment with {engine_name}: {str(e)}")
                        continue
            
            # 5. Ergebnisse der Segmente kombinieren
            all_blocks = []
            text_parts = []
            total_confidence = 0.0
            
            for result in segment_results:
                text_parts.append(result.get("text", ""))
                all_blocks.extend(result.get("blocks", []))
                total_confidence += result.get("confidence", 0.0)
            
            # Durchschnittliche Konfidenz berechnen
            avg_confidence = total_confidence / len(segment_results) if segment_results else 0.0
            
            # Blöcke nach Y-Position sortieren (von oben nach unten)
            all_blocks.sort(key=lambda b: b.get("bbox", [0, 0, 0, 0])[1])
            
            # Kombinierte Ergebnisse erstellen
            combined_result = {
                "text": "\n".join(text_parts),
                "blocks": all_blocks,
                "confidence": avg_confidence,
                "language": metadata.get("language", "de"),
                "metadata": {
                    "multi_engine": True,
                    "engines_used": [e.__class__.__name__ for e in engines],
                    "segment_count": len(segment_results),
                    "processing_mode": "segmented" if segments else "hybrid"
                }
            }
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error in multi-engine OCR: {str(e)}")
            # Fallback zur ersten Engine
            return engines[0].process_image(image_path, metadata) 