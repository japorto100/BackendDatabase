from typing import Dict, List, Optional, Any, ClassVar
from datetime import datetime
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import os
import logging

from models_app.vision.document.adapters.document_format_adapter import DocumentFormatAdapter
from models_app.vision.document.utils.document_analyzer import DocumentAnalyzer
from models_app.vision.document.utils.core.processing_metadata_context import (
    ProcessingMetadataContext,
    with_transaction
)
from models_app.vision.document.utils.error_handling.handlers import (
    handle_document_errors,
    handle_adapter_errors
)
from models_app.vision.document.utils.error_handling.errors import DocumentProcessingError, DocumentValidationError
from models_app.vision.document.utils.core.next_layer_interface import ProcessingEventType

# Verwende with_transaction aus dem richtigen Modul
from models_app.vision.document.utils.core.processing_metadata_context import with_transaction



# Wir müssen noch überprüfen, ob documentanalyzer hier in ganze angewendet werden soll oder nur einzelne funktionen



import logging
logger = logging.getLogger(__name__)

class HybridDocumentAdapter(DocumentFormatAdapter):
    """
    Adapter for processing hybrid documents that contain multiple content types.
    Utilizes multiple processors and fusion strategies for optimal results.
    
    Hybrid Document Types:
    ---------------------
    1. Complex Business Documents:
       - Invoices: Text + Tables + Logos + Signatures
       - Contracts: Text + Tables + Signatures + Stamps
       - Reports: Text + Charts + Images + Tables
       - Presentations: Slides + Images + Charts + Animations
    
    2. Technical Documents:
       - Technical Manuals: Text + Diagrams + Tables + Flowcharts
       - Engineering Drawings: Vector Graphics + Text + Measurements
       - Scientific Papers: Text + Formulas + Graphs + Tables
    
    3. Medical Documents:
       - Patient Records: Text + Medical Images + Charts + Forms
       - Lab Reports: Text + Tables + Graphs + Analysis Images
       - Medical Imaging Reports: Images + Annotations + Text + Measurements
    
    4. Legal Documents:
       - Court Documents: Text + Signatures + Stamps + Attachments
       - Property Documents: Text + Maps + Photos + Signatures
       - Regulatory Filings: Forms + Tables + Signatures + Appendices
    
    5. Educational Materials:
       - Textbooks: Text + Images + Exercises + Solutions
       - Course Materials: Slides + Notes + Diagrams + Examples
       - Assessment Materials: Questions + Images + Answer Keys
    
    Processing Strategy:
    ------------------
    1. Content Analysis:
       - Identifies different content types within document
       - Determines complexity and processing requirements
       - Maps content to appropriate processors
    
    2. Parallel Processing:
       - Processes different sections concurrently
       - Uses specialized processors for each content type
       - Maintains content relationships and structure
    
    3. Result Fusion:
       - Merges results from different processors
       - Preserves document structure and relationships
       - Ensures consistent output format
    
    Processor Integration:
    ----------------------------
    1. OCRProcessorSelector:
       - Handles image-based content extraction
       - Supports multiple OCR engines (Tesseract, Azure OCR, Google Vision)
       - Selects best OCR engine based on content type and quality
       - Handles: scanned documents, images with text, handwritten content
    
    2. OfficeSelector:
       - Processes structured office documents
       - Supports: DOCX, XLSX, PPTX, PDF, RTF, ODT
       - Maintains document structure and formatting
       - Extracts: text, tables, embedded images, metadata
    
    3. ColpaliProcessor:
       - Specialized for complex layouts and mixed content
       - Table detection and extraction
       - Form field recognition
       - Layout analysis and preservation
    
    4. HybridFusion:
       - Merges results from multiple processors
       - Resolves conflicts between different extractions
       - Maintains document structure
       - Optimizes output based on content type
    """
    
    # Class-level constants for adapter registration
    VERSION: ClassVar[str] = "1.0.0"
    CAPABILITIES: ClassVar[Dict[str, float]] = {
        "mixed_content": 0.9,
        "complex_layouts": 0.85,
        "office_documents": 0.8,
        "pdfs": 0.85,
        "images": 0.8,
        "text_extraction": 0.85,
        "structure_preservation": 0.8,
        "email_processing": 0.75,
        "container_formats": 0.7
    }
    PRIORITY: ClassVar[int] = 90
    '''SUPPORTED_FORMATS: ClassVar[List[str]] = [
        '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
        '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp',
        '.html', '.htm', '.xml', '.rtf', '.odt' 
    ]'''
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the hybrid document adapter."""
        self.format_name = "Hybrid Document"
        
        # Will be initialized in _initialize_adapter_components
        self.document_analyzer = None
        self.ocr_selector = None
        self.office_selector = None
        self.colpali_processor = None
        self.fusion_engine = None
        
        # Übergebe Kontrolle an die Basisklasse, die initialize() aufruft
        super().__init__(config)
    
    def _initialize_adapter_components(self) -> None:
        """
        Initialize adapter-specific components using template method pattern.
        
        The hybrid adapter uses multiple processors in combination:
        1. DocumentAnalyzer: For analyzing document content types
        2. OCR: For text extraction from images
        3. Office: For structured document processing
        4. ColPali: For visual understanding and complex layouts
        5. Fusion: For combining results from different processors
        """
        # Rufe erst die Basisimplementierung auf
        super()._initialize_adapter_components()
        
        try:
            # Initialisiere DocumentAnalyzer
            self.document_analyzer = DocumentAnalyzer()
            
            # Import OCR Processor with fallbacks
            OCRProcessorSelector = self._import_class(
                "models_app.vision.processors.ocr", 
                "OCRProcessorSelector",
                ["models_app.vision.processors.ocr.ocr_model_selector"]
            )
            
            # Import Office Selector with fallbacks
            OfficeSelector = self._import_class(
                "models_app.vision.processors.office", 
                "OfficeSelector", 
                ["models_app.vision.processors.office_non_image.office_selector"]
            )
            
            # If OfficeSelector not found, try using WordDocumentAdapter as fallback
            if OfficeSelector is None:
                WordDocumentAdapter = self._import_class(
                    "models_app.vision.processors.office_non_image.word_document_adapter",
                    "WordDocumentAdapter"
                )
                if WordDocumentAdapter:
                    logger.info("Using WordDocumentAdapter as fallback for OfficeSelector")
                    # Create a wrapper class that mimics OfficeSelector interface
                    class OfficeAdapter:
                        def __init__(self, **kwargs):
                            self.word_adapter = WordDocumentAdapter()
                            self.config = kwargs
                        
                        def process(self, content, config=None):
                            return self.word_adapter.process_document(content)
                    
                    OfficeSelector = OfficeAdapter
            
            # Import ColPali Processor with fallbacks
            ColpaliProcessor = self._import_class(
                "models_app.vision.processors.colpali", 
                "ColpaliProcessor",
                ["models_app.vision.processors.colpali.processor"]
            )
            
            # Import HybridFusion
            HybridFusion = self._import_class(
                "models_app.vision.processors.fusion.hybrid_fusion",
                "HybridFusion"
            )
            
            # Initialize OCR selector with configuration
            if OCRProcessorSelector:
                ocr_config = self.config.get("ocr_config", {})
                if not ocr_config:
                    ocr_config = {
                        "enable_gpu": True,
                        "batch_size": 2,
                        "confidence_threshold": 0.7
                    }
                self.ocr_selector = OCRProcessorSelector(config=ocr_config)
                logger.info("Successfully initialized OCR selector")
            else:
                logger.warning("OCR selector not available")
            
            # Initialize ColPali processor
            if ColpaliProcessor:
                colpali_config = self.config.get("colpali_config", {})
                if not colpali_config:
                    colpali_config = {
                        "model_name": "vidore/colpali-v1.2",
                        "device": "auto",
                        "batch_size": 1
                    }
                try:
                    self.colpali_processor = ColpaliProcessor(**colpali_config)
                    logger.info("Successfully initialized ColPali processor")
                except TypeError as e:
                    # Try alternative initialization if signature is different
                    logger.warning(f"Error initializing ColPali with parameters: {str(e)}")
                    try:
                        self.colpali_processor = ColpaliProcessor()
                        logger.info("Successfully initialized ColPali processor without parameters")
                    except Exception as e2:
                        logger.error(f"Failed to initialize ColPali processor: {str(e2)}")
            else:
                logger.warning("ColPali processor not available")
            
            # Initialize office selector
            if OfficeSelector:
                office_config = self.config.get("office_config", {})
                if not office_config:
                    office_config = {
                        "supported_formats": ["docx", "pdf", "xlsx", "pptx"],
                        "extract_metadata": True
                    }
                try:
                    self.office_selector = OfficeSelector(**office_config)
                    logger.info("Successfully initialized Office selector")
                except TypeError as e:
                    # Try alternative initialization if signature is different
                    logger.warning(f"Error initializing Office selector with parameters: {str(e)}")
                    try:
                        self.office_selector = OfficeSelector()
                        logger.info("Successfully initialized Office selector without parameters")
                    except Exception as e2:
                        logger.error(f"Failed to initialize Office selector: {str(e2)}")
            else:
                logger.warning("Office selector not available")
            
            # Initialize fusion engine
            if HybridFusion:
                fusion_config = self.config.get("fusion_config", {})
                self.fusion_engine = HybridFusion(config=fusion_config)
                logger.info("Successfully initialized Fusion engine")
            else:
                logger.warning("Fusion engine not available")
            
            # Emit processor initialization event
            available_processors = [
                p for p in ["OCR", "Office", "ColPali"] 
                if getattr(self, f"{p.lower()}_selector" if p != "ColPali" else "colpali_processor") is not None
            ]
            
            if available_processors:
                self.next_layer.emit_simple_event(
                    ProcessingEventType.INITIALIZATION_COMPLETE,
                    f"hybrid_adapter_components",
                    {
                        "available_processors": available_processors,
                        "adapter": self.__class__.__name__
                    }
                )
                logger.info(f"Hybrid adapter initialized with processors: {', '.join(available_processors)}")
            else:
                logger.error("No processors available for Hybrid Document Adapter")
                self.next_layer.emit_simple_event(
                    ProcessingEventType.ERROR_OCCURRED,
                    f"hybrid_adapter_components",
                    {
                        "error": "No processors available",
                        "adapter": self.__class__.__name__
                    }
                )
            
        except ImportError as e:
            logger.error(f"Failed to import processor classes: {str(e)}")
            self.next_layer.emit_simple_event(
                ProcessingEventType.ERROR_OCCURRED,
                f"hybrid_adapter_components",
                {
                    "error": f"Failed to import processor classes: {str(e)}",
                    "adapter": self.__class__.__name__
                }
            )
            # Processors remain None, will be checked before use
        except Exception as e:
            logger.error(f"Error initializing processors: {str(e)}")
            self.next_layer.emit_simple_event(
                ProcessingEventType.ERROR_OCCURRED,
                f"hybrid_adapter_components",
                {
                    "error": f"Error initializing processors: {str(e)}",
                    "adapter": self.__class__.__name__
                }
            )
            # Processors that failed to initialize remain None, will be checked before use
    
    def _import_class(self, module_path: str, class_name: str, fallback_paths: List[str] = None) -> Optional[Any]:
        """
        Safely import a class from a module with fallbacks.
        
        Args:
            module_path: Path to the module
            class_name: Name of the class to import
            fallback_paths: List of alternative module paths to try
            
        Returns:
            Class object or None if import fails
        """
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not import {class_name} from {module_path}: {str(e)}")
            
            # Try fallback paths if provided
            if fallback_paths:
                for path in fallback_paths:
                    try:
                        module = importlib.import_module(path)
                        return getattr(module, class_name)
                    except (ImportError, AttributeError) as e2:
                        logger.warning(f"Fallback import failed for {class_name} from {path}: {str(e2)}")
            
            return None
    
    def _validate_configuration(self) -> None:
        """
        Validate adapter configuration.
        
        Raises:
            DocumentValidationError: If configuration is invalid
        """
        super()._validate_configuration()
        
        # Überprüfe, ob mindestens ein Prozessor verfügbar ist
        if not any([self.ocr_selector, self.colpali_processor, self.office_selector]):
            logger.warning("No processing components available for hybrid documents")
    
    def validate_format(self, document_path: str) -> bool:
        """
        Validate if the document can be processed by checking content types and available processors.
        Uses document type information from metadata context if available.
        
        Args:
            document_path: Path to the document
            
        Returns:
            bool: True if document can be processed, False otherwise
            
        Raises:
            DocumentValidationError: If validation fails
        """
        try:
            # Check if any processors are initialized
            if not any([self.ocr_selector, self.colpali_processor, self.office_selector]):
                logger.warning("No processors available for hybrid document processing.")
                # Check for processors in project
                available_modules = []
                for path in [
                    "models_app.vision.processors.ocr",
                    "models_app.vision.processors.colpali",
                    "models_app.vision.processors.office",
                    "models_app.vision.processors.office_non_image"
                ]:
                    try:
                        importlib.import_module(path)
                        available_modules.append(path)
                    except ImportError:
                        pass
                
                if not available_modules:
                    raise DocumentValidationError(
                        "No processor modules found in the project. Check installation."
                    )
                else:
                    raise DocumentValidationError(
                        f"No processors available for hybrid document processing. "
                        f"Found modules: {', '.join(available_modules)}"
                    )
            
            # Get document type info from metadata context if available
            if hasattr(self, 'metadata_context') and self.metadata_context:
                doc_type_info = self.metadata_context.get_analysis_result("document_type", {})
                doc_classification = self.metadata_context.get_analysis_result("document_classification", {})
                
                # Document needs hybrid processing if:
                # 1. Already classified as hybrid
                # 2. Has high image ratio
                # 3. Has multiple high-confidence classifications
                # 4. Is a complex document type
                needs_hybrid = (
                    doc_type_info.get("category") == "hybrid" or
                    doc_type_info.get("image_ratio", 0) > 0.3 or
                    len([v for v in doc_classification.values() if v > 0.7]) > 1 or
                    any(doc_classification.get(t, 0) > 0.7 for t in [
                        "invoice", "contract", "report", "medical_record",
                        "academic_paper", "technical_manual"
                    ])
                )
                
                if needs_hybrid:
                    self.metadata_context.record_decision(
                        component="HybridDocumentAdapter.validate_format",
                        decision="Document requires hybrid processing",
                        reason=f"Type: {doc_type_info.get('type')}, "
                              f"Category: {doc_type_info.get('category')}, "
                              f"Classifications: {[k for k, v in doc_classification.items() if v > 0.7]}"
                    )
                    return True
            
            # If no metadata or not clearly hybrid, analyze document structure
            sections = self.document_analyzer.analyze_document(document_path)
            content_types = {section.get('type', 'unknown') for section in sections}
            
            # Document is hybrid if:
            # 1. Multiple content types present
            # 2. Contains mixed content sections
            # 3. Has complex layout
            # 4. Has high complexity sections
            is_hybrid = (
                len(content_types) > 1 or
                'mixed' in content_types or
                any(section.get('complexity', 0) > 0.7 for section in sections) or
                any(section.get('layout_score', 0) > 0.8 for section in sections)
            )
            
            if not is_hybrid:
                raise DocumentValidationError(
                    "Document does not require hybrid processing"
                )
            
            # Check if we have processors for the detected content types
            available_processors = {
                'image': self.colpali_processor is not None or self.ocr_selector is not None,
                'document': self.office_selector is not None,
                'text': self.office_selector is not None or self.ocr_selector is not None,
                'mixed': any([self.colpali_processor, self.ocr_selector, self.office_selector]),
                'unknown': any([self.colpali_processor, self.ocr_selector, self.office_selector])
            }
            
            missing_processors = [ct for ct in content_types if not available_processors.get(ct, False)]
            if missing_processors:
                raise DocumentValidationError(
                    f"No processors available for content types: {', '.join(missing_processors)}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Format validation failed: {str(e)}")
            raise DocumentValidationError(f"Format validation failed: {str(e)}")
    
    def _get_available_processors(self, content_type: str, section: Dict[str, Any]) -> List[Any]:
        """
        Get available processors based on content type and section characteristics.
        Uses existing fusion infrastructure for result combination.
        
        Args:
            content_type: Type of content to process ('image', 'document', 'text', 'mixed')
            section: Section details including content characteristics
            
        Returns:
            List of appropriate processors for the content
        """
        processors = []
        
        # Extract section characteristics
        has_text = section.get('has_text', False)
        has_layout = section.get('layout_complexity', 0) > 0.5
        has_tables = section.get('has_tables', False)
        image_quality = section.get('image_quality', 0.5)
        
        try:
            if content_type == 'image':
                # ColPali for all images (especially without text or complex layout)
                if self.colpali_processor is not None:
                    processors.append(self.colpali_processor)
                
                # Add OCR if text is detected and quality is good
                if has_text and image_quality >= 0.6:
                    if self.ocr_selector is not None:
                        processors.append(self.ocr_selector)
                
            elif content_type in ['document', 'text']:
                # Office processor for basic document processing
                if self.office_selector is not None:
                    processors.append(self.office_selector)
                
                # Add ColPali for complex layouts or tables
                if (has_layout or has_tables) and self.colpali_processor is not None:
                    processors.append(self.colpali_processor)
                
                # Fallback to OCR if no office processor available and text content
                if self.office_selector is None and has_text and self.ocr_selector is not None:
                    processors.append(self.ocr_selector)
                
            elif content_type == 'mixed':
                # For mixed content, use all available processors
                # HybridFusion will handle the combination
                if self.colpali_processor is not None:
                    processors.append(self.colpali_processor)
                if self.office_selector is not None:
                    processors.append(self.office_selector)
                if has_text and self.ocr_selector is not None:
                    processors.append(self.ocr_selector)
            
            # If no processors selected but we have some available, use any available as fallback
            if not processors:
                logger.warning(f"No processors selected for {content_type}, using fallbacks")
                for processor_attr in ['colpali_processor', 'office_selector', 'ocr_selector']:
                    processor = getattr(self, processor_attr, None)
                    if processor is not None:
                        processors.append(processor)
                        logger.info(f"Using {processor_attr} as fallback for {content_type}")
            
            # Record processor selection
            if hasattr(self, 'metadata_context') and self.metadata_context:
                self.metadata_context.record_decision(
                    component="HybridDocumentAdapter",
                    decision=f"Selected processors for {content_type}",
                    reason={
                        'content_type': content_type,
                        'characteristics': {
                            'has_text': has_text,
                            'has_layout': has_layout,
                            'has_tables': has_tables,
                            'image_quality': image_quality
                        },
                        'selected_processors': [p.__class__.__name__ for p in processors]
                    }
                )
            
            return processors
            
        except Exception as e:
            logger.error(f"Error selecting processors: {str(e)}")
            raise DocumentProcessingError(f"Processor selection failed: {str(e)}")
    
    @handle_document_errors
    @handle_adapter_errors
    @with_transaction
    def process_document(self, document_path: str, options: Dict[str, Any] = None, 
                        metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """Process a hybrid document using appropriate processors."""
        
        if metadata_context is None:
            metadata_context = ProcessingMetadataContext(document_path)
        
        try:
            # Event for document reception
            self.next_layer.emit_simple_event(
                ProcessingEventType.DOCUMENT_RECEIVED,
                document_path,
                {
                    "processor": self.__class__.__name__,
                    "options": options
                }
            )
            
            # Store metadata context for validation
            self.metadata_context = metadata_context
            
            # Validate format using metadata context
            if not self.validate_format(document_path):
                raise DocumentProcessingError(f"Unsupported document format: {document_path}")
            
            # Verify fusion engine is initialized
            if self.fusion_engine is None:
                HybridFusion = self._import_class(
                    "models_app.vision.processors.fusion.hybrid_fusion",
                    "HybridFusion"
                )
                if HybridFusion:
                    self.fusion_engine = HybridFusion()
                    logger.info("Initialized fusion engine on demand")
                else:
                    raise DocumentProcessingError("Fusion engine not available and could not be initialized")
            
            # Get document type info for processing strategy
            doc_type_info = metadata_context.get_analysis_result("document_type", {})
            doc_classification = metadata_context.get_analysis_result("document_classification", {})
            
            # Analyze document structure with type info
            sections = self.document_analyzer.analyze_document(
                document_path, 
                document_type=doc_type_info.get("type"),
                expected_content=doc_classification
            )
            
            # Record analysis in metadata
            metadata_context.record_analysis_result(
                "hybrid_document_analysis",
                {
                    "sections": len(sections),
                    "content_types": list({s.get('type') for s in sections}),
                    "complexity_scores": [s.get('complexity', 0) for s in sections],
                    "layout_scores": [s.get('layout_score', 0) for s in sections]
                }
            )
            
            # Process sections in parallel if possible
            with ThreadPoolExecutor() as executor:
                section_futures = []
                for section in sections:
                    future = executor.submit(
                        self._process_section,
                        section,
                        metadata_context.create_child_context(f"section_{section['id']}")
                    )
                    section_futures.append(future)
                
                # Collect results
                section_results = []
                for future in as_completed(section_futures):
                    try:
                        result = future.result()
                        section_results.append(result)
                    except Exception as e:
                        metadata_context.record_error(
                            "section_processing",
                            str(e),
                            error_type=type(e).__name__
                        )
            
            # Merge results using HybridFusion
            fused_features, strategy, confidence = self.fusion_engine.fuse_with_best_strategy(
                visual_features=[r.get('visual_features') for r in section_results if r.get('visual_features')],
                text_features=[r.get('text_features') for r in section_results if r.get('text_features')],
                document_metadata=metadata_context.get_metadata()
            )
            
            # Apply inherited optimization from DocumentFormatAdapter
            optimized_result = self.optimize_processing(document_path, fused_features, metadata_context)
            
            result = {
                "content": optimized_result,
                "metadata": metadata_context.finalize_context(),
                "processing_info": {
                    "sections_processed": len(sections),
                    "successful_sections": len(section_results),
                    "fusion_strategy": strategy,
                    "fusion_confidence": confidence
                }
            }
            
            # Event for successful processing
            self.next_layer.emit_simple_event(
                ProcessingEventType.PROCESSING_COMPLETE,
                document_path,
                {
                    "processor": self.__class__.__name__,
                    "success": True,
                    "result_keys": list(result.keys()) if isinstance(result, dict) else []
                }
            )
            
            return result
            
        except Exception as e:
            metadata_context.record_error(
                "hybrid_processing",
                str(e),
                error_type=type(e).__name__,
                is_fatal=True
            )
            
            # Event for error
            self.next_layer.emit_simple_event(
                ProcessingEventType.ERROR_OCCURRED,
                document_path,
                {
                    "processor": self.__class__.__name__,
                    "error": f"Hybrid document processing failed: {str(e)}",
                    "error_type": type(e).__name__
                }
            )
            
            raise DocumentProcessingError(f"Hybrid document processing failed: {str(e)}")
    
    def _process_section(self, section: Dict[str, Any], metadata_context: ProcessingMetadataContext) -> Dict[str, Any]:
        """Process a section using selected processors and existing fusion infrastructure."""
        try:
            metadata_context.start_timing(f"section_processing_{section['id']}")
            
            # Get processors
            processors = self._get_available_processors(section.get('type', 'unknown'), section)
            
            if not processors:
                metadata_context.record_warning(
                    component="HybridDocumentAdapter",
                    message=f"No suitable processors found for section {section['id']} of type {section.get('type', 'unknown')}",
                    details=section
                )
                metadata_context.end_timing(f"section_processing_{section['id']}")
                return {
                    'section_id': section['id'],
                    'section_type': section.get('type', 'unknown'),
                    'processing_status': 'skipped',
                    'reason': 'no_suitable_processors'
                }
            
            # Process with each processor
            processor_results = []
            for processor in processors:
                try:
                    # Configure processor based on content type if method exists
                    processor_config = {}
                    if hasattr(processor, 'configure_for_content_type'):
                        processor_config = processor.configure_for_content_type(
                            section.get('type', 'unknown'),
                            section
                        )
                    
                    # Process content
                    if hasattr(processor, 'process'):
                        # Standard interface
                        result = processor.process(
                            section['content'],
                            config=processor_config if processor_config else None
                        )
                    elif hasattr(processor, 'process_document'):
                        # Alternative interface
                        result = processor.process_document(section['content'])
                    else:
                        raise AttributeError(f"Processor {processor.__class__.__name__} has no process or process_document method")
                    
                    processor_results.append(result)
                    
                    metadata_context.record_processing_result(
                        processor.__class__.__name__,
                        {
                            "success": True,
                            "section_id": section['id'],
                            "confidence": result.get('confidence', 0)
                        }
                    )
                except Exception as e:
                    metadata_context.record_error(
                        processor.__class__.__name__,
                        str(e),
                        error_type=type(e).__name__
                    )
            
            # Let HybridFusion handle the combination
            section_result = {
                'visual_features': [r.get('visual_features') for r in processor_results if r.get('visual_features')],
                'text_features': [r.get('text_features') for r in processor_results if r.get('text_features')],
                'section_id': section['id'],
                'section_type': section.get('type', 'unknown')
            }
            
            metadata_context.end_timing(f"section_processing_{section['id']}")
            return section_result
            
        except Exception as e:
            metadata_context.end_timing(f"section_processing_{section['id']}")
            metadata_context.record_error(
                "section_processing",
                str(e),
                error_type=type(e).__name__
            )
            raise
    
    def _cleanup_resources(self) -> None:
        """Clean up adapter resources."""
        try:
            # Clean up fusion engine
            if self.fusion_engine is not None and hasattr(self.fusion_engine, 'cleanup'):
                self.fusion_engine.cleanup()
            
            # Clean up processors
            for processor in [self.ocr_selector, self.office_selector, self.colpali_processor]:
                if processor is not None and hasattr(processor, 'cleanup'):
                    processor.cleanup()
            
            # Call parent cleanup
            super()._cleanup_resources()
            
        except Exception as e:
            logger.error(f"Error during HybridDocumentAdapter cleanup: {str(e)}")
