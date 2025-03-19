"""
ImageDocumentAdapter: Processor for image-based documents.
Leverages OCR and ColPali for comprehensive document understanding.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
from PIL import Image
from datetime import datetime
import tempfile
import time
import importlib

from models_app.vision.document.adapters.document_format_adapter import DocumentFormatAdapter
from models_app.vision.document.utils.core.processing_metadata_context import ProcessingMetadataContext
from models_app.vision.document.utils.error_handling.handlers import (
    handle_document_errors,
    handle_adapter_errors
)
from models_app.vision.document.utils.error_handling.errors import DocumentProcessingError, DocumentValidationError
from models_app.vision.document.utils.core.next_layer_interface import ProcessingEventType

# Import specific utility functions if needed
from models_app.vision.utils.image_processing.core import load_image
from models_app.vision.utils.image_processing.detection import detect_text_regions, detect_tables
from models_app.vision.utils.image_processing.enhancement import enhance_for_ocr

logger = logging.getLogger(__name__)

class ImageDocumentAdapter(DocumentFormatAdapter):
    """
    Processes image-based documents by coordinating OCR and ColPali components.
    Handles images, scanned PDFs, and documents with high image content.
    """
    
    # Class constants for registration
    VERSION = "1.0.0"
    SUPPORTED_FORMATS = [
        '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp',
        '.pdf',  # PDFs can be processed as images too
        '.heic', '.heif'  # High Efficiency Image Format
    ]
    PRIORITY = 70
    CAPABILITIES = {
        "images": 0.95,
        "scanned_documents": 0.9,
        "photos": 0.95,
        "text_extraction": 0.7,
        "complex_layouts": 0.6,
        "diagrams": 0.8,
        "forms": 0.6,
        "charts": 0.7,
        "tables": 0.6,
        "pdfs": 0.6,
        "mixed_content": 0.7
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the image document processor.
        
        Args:
            config: Configuration dictionary with processor-specific settings.
        """
        self.format_name = "Image Document"
        
        # Will be initialized in _initialize_adapter_components
        self.ocr_processor = None
        self.colpali_processor = None
        self.fusion_engine = None
        
        # Pass control to the base class which calls initialize()
        super().__init__(config)
    
    def _initialize_adapter_components(self) -> None:
        """
        Initialize adapter-specific components using template method pattern.
        
        The image adapter uses multiple processors in combination:
        1. OCR: For text extraction from images
        2. ColPali: For visual understanding and complex layouts
        3. Fusion: For combining results from different processors
        """
        # Call the base implementation first
        super()._initialize_adapter_components()
        
        try:
            # Import OCR Processor with fallbacks
            OCRProcessor = self._import_class(
                "models_app.vision.processors.ocr", 
                "OCRProcessorSelector",
                [
                    "models_app.vision.processors.ocr.ocr_model_selector",
                    "models_app.vision.ocr.ocr_model_selector" 
                ]
            )
            
            # Import ColPali Processor with fallbacks
            ColPaliProcessor = self._import_class(
                "models_app.vision.processors.colpali", 
                "ColPaliProcessor",
                [
                    "models_app.vision.processors.colpali.processor",
                    "models_app.vision.colpali.processor"
                ]
            )
            
            # Import HybridFusion
            HybridFusion = self._import_class(
                "models_app.vision.processors.fusion.hybrid_fusion",
                "HybridFusion",
                ["models_app.vision.fusion.hybrid_fusion"]
            )
            
            # Initialize OCR processor
            if OCRProcessor:
                ocr_config = self.config.get("ocr_config", {})
                if not ocr_config:
                    ocr_config = {
                        "enable_gpu": True,
                        "batch_size": 2,
                        "confidence_threshold": 0.7
                    }
                try:
                    self.ocr_processor = OCRProcessor(config=ocr_config)
                    logger.info("Successfully initialized OCR processor")
                except TypeError as e:
                    # Try alternative initialization
                    logger.warning(f"Error initializing OCR with parameters: {str(e)}")
                    try:
                        self.ocr_processor = OCRProcessor()
                        logger.info("Successfully initialized OCR processor without parameters")
                    except Exception as e2:
                        logger.error(f"Failed to initialize OCR processor: {str(e2)}")
            else:
                logger.warning("OCR processor not available")
            
            # Initialize ColPali processor
            if ColPaliProcessor:
                colpali_config = self.config.get("colpali_config", {})
                if not colpali_config:
                    colpali_config = {
                        "model_name": "vidore/colpali-v1.2",
                        "device": "auto",
                        "batch_size": 1
                    }
                try:
                    self.colpali_processor = ColPaliProcessor(**colpali_config)
                    logger.info("Successfully initialized ColPali processor")
                except TypeError as e:
                    # Try alternative initialization
                    logger.warning(f"Error initializing ColPali with parameters: {str(e)}")
                    try:
                        self.colpali_processor = ColPaliProcessor()
                        logger.info("Successfully initialized ColPali processor without parameters")
                    except Exception as e2:
                        logger.error(f"Failed to initialize ColPali processor: {str(e2)}")
            else:
                logger.warning("ColPali processor not available")
            
            # Initialize fusion engine
            if HybridFusion:
                fusion_config = self.config.get("fusion_config", {})
                try:
                    self.fusion_engine = HybridFusion(config=fusion_config) if fusion_config else HybridFusion()
                    logger.info("Successfully initialized Fusion engine")
                except Exception as e:
                    logger.error(f"Failed to initialize Fusion engine: {str(e)}")
            else:
                logger.warning("Fusion engine not available")
            
            # Emit processor initialization event
            available_processors = []
            if self.ocr_processor is not None:
                available_processors.append("OCR")
            if self.colpali_processor is not None:
                available_processors.append("ColPali")
            if self.fusion_engine is not None:
                available_processors.append("Fusion")
            
            if available_processors:
                self.next_layer.emit_simple_event(
                    ProcessingEventType.INITIALIZATION_COMPLETE,
                    f"image_adapter_components",
                    {
                        "available_processors": available_processors,
                        "adapter": self.__class__.__name__
                    }
                )
                logger.info(f"Image adapter initialized with processors: {', '.join(available_processors)}")
            else:
                logger.error("No processors available for Image Document Adapter")
                self.next_layer.emit_simple_event(
                    ProcessingEventType.ERROR_OCCURRED,
                    f"image_adapter_components",
                    {
                        "error": "No processors available",
                        "adapter": self.__class__.__name__
                    }
                )
                
        except Exception as e:
            logger.error(f"Error initializing processors: {str(e)}")
            self.next_layer.emit_simple_event(
                ProcessingEventType.ERROR_OCCURRED,
                f"image_adapter_components",
                {
                    "error": str(e),
                    "adapter": self.__class__.__name__
                }
            )
    
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
        
        # Check if at least one processor is available
        if not any([self.ocr_processor, self.colpali_processor]):
            logger.warning("No processing components available for image documents")
    
    @handle_document_errors
    @handle_adapter_errors
    def process_document(self, document_path: str, options: Dict[str, Any] = None, 
                       metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process an image-based document using OCR and/or ColPali.
        
        Args:
            document_path: Path to the document file
            options: Processing options
            metadata_context: Context for tracking metadata
            
        Returns:
            Dict containing processing results
            
        Raises:
            DocumentProcessingError: If processing fails
            DocumentValidationError: If format validation fails
        """
        options = options or {}
        
        # Create metadata context if not provided
        if metadata_context is None:
            metadata_context = ProcessingMetadataContext(document_path)
            
        metadata_context.start_timing("image_document_processing")
        
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
            
            # Ensure initialization
            if not self.is_initialized:
                init_result = self.initialize()
                if not init_result.get("success", False):
                    raise DocumentProcessingError(f"Image adapter initialization failed: {init_result.get('error', 'Unknown error')}")
            
            # Validate document format
            if not self.validate_format(document_path):
                raise DocumentValidationError(f"Unsupported document format: {document_path}")
            
            # Record basic document info
            metadata_context.add_document_metadata("document_type", "image")
            metadata_context.add_document_metadata("processing_path", document_path)
            
            # Analyze image characteristics
            metadata_context.start_timing("image_analysis")
            image_info = self._analyze_image(document_path)
            metadata_context.record_analysis_result("image_characteristics", image_info)
            metadata_context.end_timing("image_analysis")
            
            # Determine which processors to use based on image characteristics
            has_text = image_info.get("has_text", False)
            has_complex_layout = image_info.get("has_complex_layout", False)
            
            # Select processing strategy
            use_ocr = has_text and self.ocr_processor is not None
            use_colpali = (has_complex_layout or not has_text) and self.colpali_processor is not None
            
            # Record processing strategy decision
            metadata_context.record_decision(
                component="ImageDocumentAdapter",
                decision=f"Processing strategy: OCR={use_ocr}, ColPali={use_colpali}",
                reason=f"Based on image analysis: text={has_text}, complex_layout={has_complex_layout}"
            )
            
            # Process with selected processors
            results = {}
            
            # Process with OCR if needed
            if use_ocr:
                metadata_context.start_timing("ocr_processing")
                ocr_result = self._process_with_ocr(document_path, options, metadata_context)
                results["ocr"] = ocr_result
                metadata_context.end_timing("ocr_processing")
                
            # Process with ColPali if needed
            if use_colpali:
                metadata_context.start_timing("colpali_processing")
                colpali_result = self._process_with_colpali(document_path, options, metadata_context)
                results["colpali"] = colpali_result
                metadata_context.end_timing("colpali_processing")
                
            # Apply fusion if multiple processors were used
            if use_ocr and use_colpali and self.fusion_engine is not None:
                metadata_context.start_timing("fusion_processing")
                # Extract features for fusion
                ocr_features = self._extract_features(results["ocr"], "text")
                colpali_features = self._extract_features(results["colpali"], "visual")
                
                # Apply fusion
                fused_content, strategy, confidence = self.fusion_engine.fuse_with_best_strategy(
                    visual_features=[colpali_features],
                    text_features=[ocr_features],
                    document_metadata=metadata_context.get_metadata()
                )
                
                # Record fusion information
                metadata_context.record_decision(
                    component="ImageDocumentAdapter",
                    decision=f"Used fusion strategy: {strategy}",
                    reason=f"Fusion confidence: {confidence}"
                )
                
                # Apply inherited optimization from DocumentFormatAdapter
                optimized_result = self.optimize_processing(document_path, fused_content, metadata_context)
                
                # Create final result
                result = {
                    "content": optimized_result,
                    "metadata": metadata_context.get_metadata(),
                    "processing_info": {
                        "fusion_strategy": strategy,
                        "fusion_confidence": confidence,
                        "ocr_applied": use_ocr,
                        "colpali_applied": use_colpali
                    }
                }
                
                metadata_context.end_timing("fusion_processing")
                
            # If only one processor was used, create result accordingly
            elif use_ocr:
                result = {
                    "content": results["ocr"].get("text", ""),
                    "blocks": results["ocr"].get("blocks", []),
                    "metadata": metadata_context.get_metadata(),
                    "processing_info": {
                        "processor": "ocr",
                        "confidence": results["ocr"].get("confidence", 0.0)
                    }
                }
                
            elif use_colpali:
                result = {
                    "content": results["colpali"].get("understanding", {}),
                    "text": results["colpali"].get("text", ""),
                    "metadata": metadata_context.get_metadata(),
                    "processing_info": {
                        "processor": "colpali",
                        "confidence": results["colpali"].get("confidence", 0.0)
                    }
                }
                
            else:
                # No processors were used
                raise DocumentProcessingError("No suitable processors available for this document")
            
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
                
            metadata_context.end_timing("image_document_processing")
            return result
            
        except DocumentValidationError:
            # Re-raise validation errors without wrapping
            raise
        except Exception as e:
            metadata_context.record_error(
                component="ImageDocumentAdapter",
                message=f"Image processing failed: {str(e)}",
                error_type=type(e).__name__,
                is_fatal=True
            )
            metadata_context.end_timing("image_document_processing")
            
            # Event for error
            self.next_layer.emit_simple_event(
                ProcessingEventType.ERROR_OCCURRED,
                document_path,
                {
                    "processor": self.__class__.__name__,
                    "error": f"Image processing failed: {str(e)}",
                    "error_type": type(e).__name__
                }
            )
            
            raise DocumentProcessingError(f"Image processing failed: {str(e)}")
    
    def _analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image characteristics to determine processing strategy.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dict containing image characteristics
        """
        try:
            # Load image
            img = load_image(image_path)
            
            # Get basic image info
            width, height = img.size
            aspect_ratio = width / height
            
            # Detect text regions to determine if image has text
            text_regions = detect_text_regions(img)
            has_text = len(text_regions) > 0
            
            # Detect tables
            tables = detect_tables(img)
            has_tables = len(tables) > 0
            
            # Simple heuristic for layout complexity
            layout_complexity = min(1.0, (len(text_regions) + len(tables)) / 10.0)
            
            return {
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "has_text": has_text,
                "has_tables": has_tables,
                "has_complex_layout": layout_complexity > 0.5,
                "layout_complexity": layout_complexity,
                "text_regions": len(text_regions),
                "tables": len(tables)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            # Return basic characteristics on error
            return {
                "has_text": True,  # Assume there's text if analysis fails
                "has_tables": False,
                "has_complex_layout": False,
                "error": str(e)
            }
    
    def _process_with_ocr(self, image_path: str, options: Dict[str, Any], 
                        metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process image with OCR.
        
        Args:
            image_path: Path to the image
            options: Processing options
            metadata_context: Context for tracking metadata
            
        Returns:
            OCR results
            
        Raises:
            DocumentProcessingError: If OCR processing fails
        """
        try:
            if self.ocr_processor is None:
                raise DocumentProcessingError("OCR processor not available")
                
            # Check for different processing methods
            if hasattr(self.ocr_processor, "process_image"):
                return self.ocr_processor.process_image(image_path, options)
            elif hasattr(self.ocr_processor, "process"):
                return self.ocr_processor.process(image_path, options)
            elif hasattr(self.ocr_processor, "process_document"):
                return self.ocr_processor.process_document(image_path)
            else:
                raise DocumentProcessingError("OCR processor has no valid processing method")
        
        except DocumentProcessingError:
            # Re-raise DocumentProcessingError without wrapping
            raise
        except Exception as e:
            logger.error(f"Error in OCR processing: {str(e)}")
            if metadata_context:
                metadata_context.record_error(
                    component="ImageDocumentAdapter._process_with_ocr",
                    message=f"OCR processing failed: {str(e)}",
                    error_type=type(e).__name__
                )
            raise DocumentProcessingError(f"OCR processing failed: {str(e)}")
    
    def _process_with_colpali(self, image_path: str, options: Dict[str, Any],
                            metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Process image with ColPali.
        
        Args:
            image_path: Path to the image
            options: Processing options
            metadata_context: Context for tracking metadata
            
        Returns:
            ColPali results
            
        Raises:
            DocumentProcessingError: If ColPali processing fails
        """
        try:
            if self.colpali_processor is None:
                raise DocumentProcessingError("ColPali processor not available")
                
            # Check for different processing methods
            if hasattr(self.colpali_processor, "process_image"):
                return self.colpali_processor.process_image(image_path, options)
            elif hasattr(self.colpali_processor, "process"):
                return self.colpali_processor.process(image_path, options)
            else:
                raise DocumentProcessingError("ColPali processor has no valid processing method")
                
        except DocumentProcessingError:
            # Re-raise DocumentProcessingError without wrapping
            raise
        except Exception as e:
            logger.error(f"Error in ColPali processing: {str(e)}")
            if metadata_context:
                metadata_context.record_error(
                    component="ImageDocumentAdapter._process_with_colpali",
                    message=f"ColPali processing failed: {str(e)}",
                    error_type=type(e).__name__
                )
            raise DocumentProcessingError(f"ColPali processing failed: {str(e)}")
    
    def _extract_features(self, result: Dict[str, Any], feature_type: str) -> Any:
        """
        Extract features from processing result.
        
        Args:
            result: Processing result
            feature_type: Type of features to extract ('text' or 'visual')
            
        Returns:
            Extracted features
        """
        if not result:
            return {} if feature_type == "visual" else ""
            
        if feature_type == "text":
            # Try to get text features
            if "text_features" in result:
                return result["text_features"]
            elif "features" in result:
                return result["features"]
            elif "text" in result:
                return result["text"]
            else:
                return ""
        else:
            # Try to get visual features
            if "visual_features" in result:
                return result["visual_features"]
            elif "embeddings" in result:
                return result["embeddings"]
            elif "features" in result:
                return result["features"]
            elif "understanding" in result:
                return result["understanding"]
            else:
                return {}
    
    def _cleanup_resources(self) -> None:
        """Clean up adapter resources."""
        try:
            # Clean up processors
            for processor in [self.ocr_processor, self.colpali_processor, self.fusion_engine]:
                if processor is not None and hasattr(processor, 'cleanup'):
                    processor.cleanup()
            
            # Call parent cleanup
            super()._cleanup_resources()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def extract_structure(self, document_path: str, metadata_context: Optional[ProcessingMetadataContext] = None) -> Dict[str, Any]:
        """
        Extract the document's structure (headings, paragraphs, tables, etc.).
        Implementation of the abstract method from DocumentFormatAdapter.
        
        Args:
            document_path: Path to the document file.
            metadata_context: Optional metadata context for tracking decisions.
            
        Returns:
            Dict: Structured representation of the document.
            
        Raises:
            DocumentProcessingError: If structure extraction fails
        """
        if metadata_context:
            metadata_context.start_timing("structure_extraction")
            
        try:
            # Analyze image characteristics
            image_info = self._analyze_image(document_path)
            
            # Determine processing strategy
            has_text = image_info.get("has_text", False)
            has_tables = image_info.get("has_tables", False)
            
            # Initialize structure
            structure = {
                "type": "image_document",
                "layout": "unknown",
                "elements": []
            }
            
            # If we have ColPali, use it for structure analysis
            if self.colpali_processor is not None:
                colpali_result = self._process_with_colpali(document_path, {}, metadata_context)
                
                # Extract layout information
                if "layout" in colpali_result:
                    structure["layout"] = colpali_result["layout"]
                
                # Extract visual elements
                if "visual_elements" in colpali_result:
                    structure["elements"].extend(colpali_result["visual_elements"])
            
            # If we have OCR and text is detected, extract text blocks
            if has_text and self.ocr_processor is not None:
                ocr_result = self._process_with_ocr(document_path, {}, metadata_context)
                
                # Extract text blocks
                if "blocks" in ocr_result:
                    for block in ocr_result["blocks"]:
                        structure["elements"].append({
                            "type": "text_block",
                            "content": block.get("text", ""),
                            "confidence": block.get("confidence", 0),
                            "bbox": block.get("bbox", [])
                        })
            
            # Add image information
            structure["metadata"] = {
                "width": image_info.get("width", 0),
                "height": image_info.get("height", 0),
                "has_text": has_text,
                "has_tables": has_tables,
                "has_complex_layout": image_info.get("has_complex_layout", False)
            }
            
            if metadata_context:
                metadata_context.end_timing("structure_extraction")
                
            return structure
            
        except Exception as e:
            logger.error(f"Error extracting structure: {str(e)}")
            if metadata_context:
                metadata_context.record_error(
                    component="ImageDocumentAdapter.extract_structure",
                    message=f"Structure extraction failed: {str(e)}",
                    error_type=type(e).__name__
                )
                metadata_context.end_timing("structure_extraction")
                
            raise DocumentProcessingError(f"Structure extraction failed: {str(e)}")