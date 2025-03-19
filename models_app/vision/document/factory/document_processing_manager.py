from __future__ import annotations

# Standard library imports
from typing import Dict, Any, Optional, List, Union, Type, Callable, TypeVar, Protocol
import os
import logging
import time
import datetime
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil
import importlib
import inspect
import pkgutil
from functools import wraps

# Django imports
from django.conf import settings
from django.core.cache import cache

# Core utils
from models_app.vision.utils.core.processing_metadata_context import (
    ProcessingMetadataContext,
    ProcessingPhase,
    ProcessingDecision,
    PipelinePhase
)

# Analysis utils
from models_app.vision.utils.analysis.document_analyzer import (
    DocumentAnalyzer,
    AnalysisResult,
    DocumentFeatures
)
from models_app.vision.utils.analysis.document_type_detector import (
    DocumentTypeDetector,
    DocumentType,
    DocumentClassification
)
from models_app.vision.utils.analysis.capability_based_selector import (
    CapabilityBasedSelector,
    CapabilityScore,
    AdapterCapabilities
)
from models_app.vision.utils.analysis.quality_analyzer import (
    DocumentQualityAnalyzer,
    QualityMetrics,
    PreprocessingRecommendation
)

# Processing utils
from models_app.vision.utils.processing.chunking_service import (
    ChunkingService,
    ChunkingStrategy,
    ChunkMetadata
)
from models_app.vision.utils.processing.preprocessing_service import (
    PreprocessingService,
    PreprocessingResult,
    ProcessingStrategy
)
from models_app.vision.document.utils.io.io_service import IOService

# Error handling
from models_app.vision.document.utils.error_handling.errors import (
    DocumentError,
    DocumentProcessingError,
    DocumentResourceError,
    DocumentValidationError,
    DocumentTimeoutError,
    OCRError,
    DocumentAnalysisError,
    ChunkingError,
    AdapterError,
    KnowledgeGraphError,
    DocumentCapabilityError,
    DocumentMetadataError,
    DocumentPipelineError,
    DocumentCacheError
)
from models_app.vision.utils.error_handling.handlers import (
    handle_vision_errors,
    handle_processing_errors,
    processing_complete,
    handle_resource_limits as vision_resource_handler,
    handle_timeout as vision_timeout_handler
)
from models_app.vision.utils.error_handling.service import error_handler

# Knowledge Graph components
from models_app.vision.knowledge_graph.document_entity_extractor import (
    DocumentEntityExtractor,
    EntityExtractionResult
)
from models_app.vision.knowledge_graph.visual_entity_extractor import (
    VisualEntityExtractor,
    VisualEntityResult
)
from models_app.vision.knowledge_graph.hybrid_entity_extractor import (
    HybridEntityExtractor,
    HybridExtractionResult
)
from models_app.vision.knowledge_graph.knowledge_graph_manager import (
    KnowledgeGraphManager,
    GraphProcessingResult
    KGFeedbackManager
)

# Base adapter and implementations
from models_app.vision.document.adapters.document_base_adapter import (
    DocumentBaseAdapter,
    AdapterResult,
    ProcessingOptions
)
from models_app.vision.document.adapters.word_document_adapter import WordDocumentAdapter
from models_app.vision.document.adapters.universal_document_adapter import UniversalDocumentAdapter
from models_app.vision.document.adapters.image_document_adapter import ImageDocumentAdapter
from models_app.vision.document.adapters.hybrid_document_adapter import HybridDocumentAdapter

# Analytics
from analytics_app.utils import monitor_selector_performance

# Type definitions
AdapterType = TypeVar('AdapterType', bound='DocumentBaseAdapter')
ProcessingResult = Dict[str, Any]
DocumentPath = Union[str, Path]

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MAX_BATCH_SIZE = 50
DEFAULT_TIMEOUT = 300  # seconds
CACHE_TIMEOUT = 3600  # 1 hour

T = TypeVar('T', bound='DocumentBaseAdapter')

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    max_batch_size: int = MAX_BATCH_SIZE
    enable_parallel_processing: bool = True
    extract_for_kg: bool = False
    prepare_for_kg: bool = False
    chunking_config: Optional[Dict[str, Any]] = None
    timeout: int = DEFAULT_TIMEOUT
    cache_enabled: bool = True
    cache_timeout: int = CACHE_TIMEOUT
    resource_limits: Dict[str, float] = None
    monitoring_interval: int = 60  # seconds

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ProcessingConfig':
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})

class DocumentProcessingManager:
    """
    Central manager for document processing pipeline.
    Handles:
    - Document analysis and preprocessing
    - Adapter management and selection
    - Knowledge graph extraction
    - Document embedding (for RAG)
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the document processing manager with configuration."""
        self.config = config or {}
        self._initialize_core_components()
        self._initialize_resource_management()
        self._initialize_cache_system()
        self._initialize_monitoring()
        self._initialize_adapters()
        self._initialize_metadata_services()
        
    def _initialize_core_components(self) -> None:
        """Initialize core processing components."""
        # Analysis components
        self.type_detector = DocumentTypeDetector()
        self.analyzer = DocumentAnalyzer()
        self.quality_analyzer = DocumentQualityAnalyzer()
        self.capability_selector = CapabilityBasedSelector()
        
        # Processing components
        self.chunking_service = ChunkingService(self.config.chunking_config)
        self.preprocessing_service = PreprocessingService()
        
        # I/O service for file operations
        self.io_service = IOService(self.config.get('io_config', {}))
        
        # Thread pool for parallel processing
        if self.config.enable_parallel_processing:
            self.thread_pool = ThreadPoolExecutor(max_workers=cpu_count())
        
        # Component versions for tracking
        self.component_versions = self._get_component_versions()
        
        # Capability configuration
        self._initialize_capability_weights()

    def _initialize_kg_components(self) -> None:
        """Initialize Knowledge Graph components if enabled."""
        if self.config.extract_for_kg or self.config.prepare_for_kg:
            self.kg_manager = KnowledgeGraphManager()
            self.document_entity_extractor = DocumentEntityExtractor()
            self.visual_entity_extractor = VisualEntityExtractor()
            self.hybrid_entity_extractor = HybridEntityExtractor()
        else:
            self.kg_manager = None
            self.document_entity_extractor = None
            self.visual_entity_extractor = None
            self.hybrid_entity_extractor = None

    def _initialize_monitoring(self) -> None:
        """Initialize performance monitoring and statistics."""
        self.performance_metrics = {
            "adapter_selection_time": {},    # Average selection time by file type
            "adaptation_success_rate": {},   # Success rate by adapter
            "processing_time": {},           # Average processing time by adapter
            "quality_scores": {},            # Quality scores by adapter
            "selection_history": [],         # Recent selection decisions
            "feedback_history": {}           # User feedback history
        }
        
        self.processing_stats = {
            "processed_documents": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "total_processing_time": 0,
            "avg_processing_time": 0,
            "document_types_stats": {},
            "capability_usage": {},
            "resource_usage": {
                "cpu_usage": [],
                "memory_usage": [],
                "processing_times": [],
                "gpu_usage": [],
                "gpu_memory": []
            }
        }

    def _initialize_capability_weights(self) -> None:
        """Initialize capability weights for different document types."""
        self.capability_weights = {
            "images": {
                "images": 1.5,
                "photos": 1.2,
                "text_extraction": 0.7,
                "mixed_content": 0.5
            },
            "pdf": {
                "pdfs": 1.5,
                "text_extraction": 1.0,
                "complex_layouts": 1.0,
                "mixed_content": 0.8
            },
            "document": {
                "office_documents": 1.5,
                "text_extraction": 1.2,
                "tables": 0.8
            },
            "spreadsheet": {
                "office_documents": 1.5,
                "tables": 1.5,
                "text_extraction": 0.8
            },
            "presentation": {
                "office_documents": 1.5,
                "images": 1.0,
                "mixed_content": 1.0
            },
            "mixed": {
                "mixed_content": 1.5,
                "complex_layouts": 1.2,
                "images": 1.0,
                "text_extraction": 1.0
            }
        }

    def _get_component_versions(self) -> Dict[str, str]:
        """Get versions of all components for tracking."""
        return {
            "DocumentTypeDetector": getattr(self.type_detector, "VERSION", "1.0"),
            "DocumentAnalyzer": getattr(self.analyzer, "VERSION", "1.0"),
            "CapabilityBasedSelector": getattr(self.capability_selector, "VERSION", "1.0"),
            "ChunkingService": getattr(self.chunking_service, "VERSION", "1.0"),
            "PreprocessingService": getattr(self.preprocessing_service, "VERSION", "1.0"),
            "DocumentQualityAnalyzer": getattr(self.quality_analyzer, "VERSION", "1.0")
        }

    def _initialize_adapters(self) -> None:
        """Initialize available adapters and discover plugins."""
        try:
            # Initialize adapter storage
            self.adapters: Dict[str, Type[DocumentBaseAdapter]] = {}
            self.adapter_info: Dict[str, Dict[str, Any]] = {}
            self.extension_mapping: Dict[str, str] = {}
            
            # Register built-in adapters with validation
            built_in_adapters = {
                "universal": UniversalDocumentAdapter,
                "image": ImageDocumentAdapter,
                "hybrid": HybridDocumentAdapter,
                "word": WordDocumentAdapter
            }
            
            for name, adapter_cls in built_in_adapters.items():
                self._register_adapter(name, adapter_cls)
            
            # Discover and register plugin adapters
            self._discover_and_register_plugins()
            
            # Build extension mappings
            self._build_extension_mappings()
            
            # Validate adapter configurations
            self._validate_adapter_setup()
            
            # Cache adapter capabilities for performance
            self._cache_adapter_capabilities()
            
        except Exception as e:
            logger.error(f"Error initializing adapters: {str(e)}")
            raise DocumentProcessingError(f"Adapter initialization failed: {str(e)}")
    
    def _register_adapter(self, name: str, adapter_cls: Type[DocumentBaseAdapter]) -> None:
        """
        Register a single adapter with validation.
        
        Args:
            name: Adapter name
            adapter_cls: Adapter class to register
        """
        try:
            # Validate adapter class
            if not issubclass(adapter_cls, DocumentBaseAdapter):
                raise ValueError(f"Invalid adapter class: {adapter_cls.__name__}")
            
            # Check for required attributes
            required_attrs = ["CAPABILITIES", "SUPPORTED_FORMATS", "VERSION"]
            missing_attrs = [attr for attr in required_attrs if not hasattr(adapter_cls, attr)]
            if missing_attrs:
                raise ValueError(f"Adapter {adapter_cls.__name__} missing required attributes: {missing_attrs}")
            
            # Register adapter
            self.adapters[name] = adapter_cls
            
            # Store adapter info
            self.adapter_info[name] = {
                "name": name,
                "class": adapter_cls.__name__,
                "capabilities": getattr(adapter_cls, "CAPABILITIES", {}),
                "priority": getattr(adapter_cls, "PRIORITY", 50),
                "supported_formats": getattr(adapter_cls, "SUPPORTED_FORMATS", []),
                "version": getattr(adapter_cls, "VERSION", "1.0"),
                "is_plugin": False
            }
            
            logger.info(f"Registered adapter: {name} ({adapter_cls.__name__})")
            
        except Exception as e:
            logger.error(f"Error registering adapter {name}: {str(e)}")
            raise DocumentProcessingError(f"Failed to register adapter {name}: {str(e)}")
    
    def _discover_and_register_plugins(self) -> None:
        """Discover and register adapter plugins from configured directories."""
        # Get plugin directory from settings
        plugin_dir = getattr(settings, "DOCUMENT_ADAPTER_PLUGINS_DIR", None)
        if not plugin_dir or not os.path.exists(plugin_dir):
            logger.info("No plugin directory configured or directory not found")
            return
            
        try:
            # Discover plugins
            for finder, name, ispkg in pkgutil.iter_modules([plugin_dir]):
                try:
                    # Import module
                    module = importlib.import_module(f"{plugin_dir}.{name}")
                    
                    # Find adapter classes
                    for item_name, item in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(item, DocumentBaseAdapter) and 
                            item != DocumentBaseAdapter):
                            
                            # Register plugin adapter
                            plugin_name = name.lower()
                            self._register_plugin_adapter(plugin_name, item)
                            
                except Exception as e:
                    logger.error(f"Error loading plugin {name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error discovering plugins: {str(e)}")
            raise DocumentProcessingError(f"Plugin discovery failed: {str(e)}")
    
    def _register_plugin_adapter(self, name: str, adapter_cls: Type[DocumentBaseAdapter]) -> None:
        """
        Register a plugin adapter with additional validation.
        
        Args:
            name: Plugin name
            adapter_cls: Plugin adapter class
        """
        try:
            # Validate plugin version compatibility
            if hasattr(adapter_cls, "MIN_MANAGER_VERSION"):
                current_version = self.__class__.VERSION
                min_version = adapter_cls.MIN_MANAGER_VERSION
                if current_version < min_version:
                    raise ValueError(
                        f"Plugin {name} requires manager version {min_version}, "
                        f"but current version is {current_version}"
                    )
            
            # Register adapter
            self._register_adapter(name, adapter_cls)
            
            # Update plugin-specific info
            self.adapter_info[name].update({
                "is_plugin": True,
                "plugin_version": getattr(adapter_cls, "PLUGIN_VERSION", "1.0"),
                "plugin_author": getattr(adapter_cls, "PLUGIN_AUTHOR", "Unknown"),
                "plugin_description": getattr(adapter_cls, "PLUGIN_DESCRIPTION", "")
            })
            
            logger.info(f"Registered plugin adapter: {name} ({adapter_cls.__name__})")
            
        except Exception as e:
            logger.error(f"Error registering plugin adapter {name}: {str(e)}")
            raise DocumentProcessingError(f"Failed to register plugin adapter {name}: {str(e)}")
    
    def _build_extension_mappings(self) -> None:
        """Build mappings from file extensions to adapter names with conflict resolution."""
        try:
            new_mappings: Dict[str, str] = {}
            conflicts: Dict[str, List[str]] = {}
            
            # First pass: collect all mappings and detect conflicts
            for name, adapter_cls in self.adapters.items():
                supported_formats = self.adapter_info[name]["supported_formats"]
                priority = self.adapter_info[name]["priority"]
                
                for ext in supported_formats:
                    ext = ext.lower()
                    if ext not in new_mappings:
                        new_mappings[ext] = name
                        conflicts[ext] = [name]
                    else:
                        conflicts[ext].append(name)
            
            # Second pass: resolve conflicts based on priority
            for ext, adapters in conflicts.items():
                if len(adapters) > 1:
                    # Sort adapters by priority
                    adapters.sort(
                        key=lambda x: (
                            self.adapter_info[x]["priority"],
                            not self.adapter_info[x]["is_plugin"]  # Prefer non-plugins in ties
                        ),
                        reverse=True
                    )
                    
                    # Use highest priority adapter
                    new_mappings[ext] = adapters[0]
                    
                    # Log conflict resolution
                    logger.info(
                        f"Extension conflict for {ext} resolved to {adapters[0]} "
                        f"(conflicting adapters: {', '.join(adapters[1:])})"
                    )
            
            # Update extension mapping
            self.extension_mapping = new_mappings
            
        except Exception as e:
            logger.error(f"Error building extension mappings: {str(e)}")
            raise DocumentProcessingError(f"Failed to build extension mappings: {str(e)}")
    
    def _validate_adapter_setup(self) -> None:
        """Validate the complete adapter setup."""
        try:
            # Check for required adapters
            required_adapters = {"universal"}
            missing_adapters = required_adapters - set(self.adapters.keys())
            if missing_adapters:
                raise DocumentValidationError(
                    f"Missing required adapters: {missing_adapters}",
                    validation_type="adapter_validation",
                    details={"missing_adapters": list(missing_adapters)}
                )
            
            # Validate extension mappings
            unmapped_formats = set()
            for adapter_info in self.adapter_info.values():
                unmapped_formats.update(
                    fmt for fmt in adapter_info["supported_formats"]
                    if fmt.lower() not in self.extension_mapping
                )
            if unmapped_formats:
                logger.warning(f"Unmapped formats found: {unmapped_formats}")
            
            # Validate adapter capabilities
            for name, adapter_info in self.adapter_info.items():
                if not adapter_info["capabilities"]:
                    logger.warning(f"Adapter {name} has no declared capabilities")
            
        except Exception as e:
            logger.error(f"Error validating adapter setup: {str(e)}")
            raise DocumentValidationError(
                f"Adapter validation failed: {str(e)}",
                validation_type="adapter_validation",
                details={"error": str(e)}
            )
    
    def _cache_adapter_capabilities(self) -> None:
        """Cache adapter capabilities for performance optimization."""
        if not self.config.cache_enabled:
            return
            
        try:
            cache_key = "document_manager_adapter_capabilities"
            
            # Prepare capability cache
            capability_cache = {
                name: {
                    "capabilities": info["capabilities"],
                    "supported_formats": info["supported_formats"],
                    "priority": info["priority"]
                }
                for name, info in self.adapter_info.items()
            }
            
            # Store in Django cache
            cache.set(cache_key, capability_cache, self.config.cache_timeout)
            
        except Exception as e:
            logger.warning(f"Failed to cache adapter capabilities: {str(e)}")

    @document_processing_complete
    @processing_complete
    @handle_resource_limits
    @handle_vision_errors
    def process_document(self, document_path: DocumentPath, metadata_context: Optional[ProcessingMetadataContext] = None) -> ProcessingResult:
        """
        Process a document and extract its contents.
        
        Args:
            document_path: Path to the document file
            metadata_context: Optional metadata context for tracking decisions
            
        Returns:
            Dict containing extracted content, structure, and metadata
        """
        start_time = time.time()
        selected_adapter = None
        
        # Create metadata context if not provided
        if metadata_context is None:
            metadata_context = ProcessingMetadataContext(document_path)
        
        # Start main processing transaction
        metadata_context.start_transaction()
        
        try:
            # Record component versions and configuration
            metadata_context.record_component_versions(self.component_versions)
            metadata_context.record_configuration(self.config)
            
            # Validate document path
            if not os.path.exists(document_path):
                raise DocumentProcessingError(f"Document not found: {document_path}")
            
            # Check file size limits
            file_size = os.path.getsize(document_path)
            if self.config.resource_limits and file_size > self.config.resource_limits.get("max_file_size", float("inf")):
                raise ResourceError(f"File size {file_size} exceeds limit")
            
            # PHASE 1: Initial Document Analysis
            analysis_context = metadata_context.create_child_context(PipelinePhase.ANALYSIS)
            analysis_result = self._execute_analysis_phase(document_path, analysis_context)
            metadata_context.merge_child_context(analysis_context)
            
            # PHASE 2: Adapter Selection
            selection_context = metadata_context.create_child_context(PipelinePhase.SELECTION)
            selected_adapter = self._execute_selection_phase(
                document_path,
                analysis_result,
                selection_context
            )
            metadata_context.merge_child_context(selection_context)
            
            # PHASE 3: Document Preprocessing
            preprocessing_context = metadata_context.create_child_context(PipelinePhase.PREPROCESSING)
            processing_path = self._execute_preprocessing_phase(
                document_path,
                selected_adapter,
                analysis_result,
                preprocessing_context
            )
            metadata_context.merge_child_context(preprocessing_context)
            
            # PHASE 4: Document Processing
            processing_context = metadata_context.create_child_context(PipelinePhase.PROCESSING)
            processing_result = self._execute_processing_phase(
                processing_path,
                selected_adapter,
                analysis_result,
                processing_context
            )
            metadata_context.merge_child_context(processing_context)
            
            # PHASE 5: Knowledge Graph Integration (if enabled)
            if self.config.extract_for_kg:
                kg_context = metadata_context.create_child_context(PipelinePhase.KNOWLEDGE_GRAPH)
                kg_result = self._execute_kg_phase(
                    processing_result,
                    document_path,
                    kg_context
                )
                processing_result["knowledge_graph"] = kg_result
                metadata_context.merge_child_context(kg_context)
            
            # Add metadata to result
            processing_result["metadata"] = metadata_context.finalize_context()
            
            # Update processing stats
            self._update_processing_stats(
                selected_adapter.__class__.__name__,
                document_path,
                time.time() - start_time,
                True
            )
            
            # Track document lineage
            self._track_document_lineage(document_path, processing_result, metadata_context)
            
            # Commit the main transaction
            metadata_context.commit_transaction()
            
            return processing_result
            
        except Exception as e:
            # Rollback the main transaction
            metadata_context.rollback_transaction()
            
            # Handle different types of errors
            error_info = self._handle_processing_error(e, document_path, metadata_context)
            
            # Update processing stats with failure
            if selected_adapter:
                self._update_processing_stats(
                    selected_adapter.__class__.__name__,
                    document_path,
                    time.time() - start_time,
                    False
                )
            
            return error_info
    
    @handle_document_errors
    @handle_vision_errors
    @handle_timeout(timeout_seconds=120)
    def _execute_analysis_phase(
        self,
        document_path: DocumentPath,
        metadata_context: ProcessingMetadataContext
    ) -> Dict[str, Any]:
        """Execute the analysis phase of document processing."""
        metadata_context.start_timing("phase1_analysis")
        
        try:
            # Perform document type detection
            doc_type_info = self.type_detector.detect_document_type(document_path)
            doc_classification = self.type_detector.classify_document_type(document_path)
            
            # Record type detection in metadata
            metadata_context.record_analysis_result(
                "document_type",
                doc_type_info,
                version=self.component_versions["DocumentTypeDetector"],
                component="DocumentTypeDetector"
            )
            metadata_context.record_analysis_result(
                "document_classification",
                doc_classification,
                version=self.component_versions["DocumentTypeDetector"],
                component="DocumentTypeDetector"
            )
            
            # Perform comprehensive document analysis
            analysis_result = self.analyzer.analyze_document(document_path)
            
            # Add document type info to analysis result
            analysis_result["document_type"] = doc_type_info
            analysis_result["document_classification"] = doc_classification
            
            # Record analysis in metadata
            metadata_context.record_analysis_result(
                "document_analysis",
                analysis_result,
                version=self.component_versions["DocumentAnalyzer"],
                component="DocumentAnalyzer"
            )
            
            # Quality analysis
            quality_metrics = self.quality_analyzer.analyze_image_quality(document_path)
            preprocessing_recommendations = self.quality_analyzer.generate_preprocessing_recommendations(quality_metrics)
            
            # Record quality analysis in metadata
            metadata_context.record_analysis_result(
                "quality_metrics",
                quality_metrics,
                component="DocumentQualityAnalyzer"
            )
            metadata_context.record_analysis_result(
                "preprocessing_recommendations",
                preprocessing_recommendations,
                component="DocumentQualityAnalyzer"
            )
            
            # Generate knowledge graph extraction hints if needed
            if self.config.extract_for_kg or self.config.prepare_for_kg:
                kg_extraction_hints = self._generate_kg_extraction_hints(
                    analysis_result,
                    doc_classification
                )
                metadata_context.record_analysis_result(
                    "kg_extraction_hints",
                    kg_extraction_hints,
                    component="DocumentProcessingManager"
                )
            
            metadata_context.end_timing("phase1_analysis")
            return analysis_result
            
        except Exception as e:
            metadata_context.end_timing("phase1_analysis")
            metadata_context.record_error(
                component="DocumentProcessingManager",
                message=f"Analysis phase failed: {str(e)}",
                error_type=type(e).__name__,
                is_fatal=True
            )
            raise DocumentAnalysisError(f"Analysis phase failed: {str(e)}")
    
    @handle_document_errors
    @handle_vision_errors
    @handle_timeout(timeout_seconds=120)
    def _execute_selection_phase(
        self,
        document_path: DocumentPath,
        analysis_result: Dict[str, Any],
        metadata_context: ProcessingMetadataContext
    ) -> Type[DocumentBaseAdapter]:
        """Execute the adapter selection phase."""
        metadata_context.start_timing("phase2_adapter_selection")
        
        try:
            # Configure capability selector based on document type
            primary_doc_type = analysis_result["document_type"].get("type", "unknown")
            
            if primary_doc_type in self.capability_weights:
                # Apply document-type specific weights
                self.capability_selector.set_capability_weights(
                    self.capability_weights[primary_doc_type]
                )
                metadata_context.record_decision(
                    component="DocumentProcessingManager",
                    decision=f"Applied capability weights for {primary_doc_type} documents",
                    reason="Document type specific optimization"
                )
            elif (analysis_result["document_classification"].get("mixed", 0) > 0.5):
                # Apply mixed content weights if mixed confidence is high
                self.capability_selector.set_capability_weights(
                    self.capability_weights["mixed"]
                )
                metadata_context.record_decision(
                    component="DocumentProcessingManager",
                    decision="Applied capability weights for mixed content documents",
                    reason=f"Mixed content classification score: {analysis_result['document_classification']['mixed']:.2f}"
                )
            
            # Derive required capabilities from analysis
            required_capabilities = self.capability_selector.derive_required_capabilities(
                analysis_result
            )
            
            # Apply content-specific capability requirements
            self._enhance_capabilities_from_content(
                required_capabilities,
                analysis_result,
                metadata_context
            )
            
            # Use capability-based selector to find the best adapter
            selected_adapter = self.capability_selector.select_adapter(
                document_path=document_path,
                document_analysis=analysis_result,
                metadata_context=metadata_context,
                required_capabilities=required_capabilities
            )
            
            metadata_context.end_timing("phase2_adapter_selection")
            return selected_adapter
            
        except Exception as e:
            metadata_context.end_timing("phase2_adapter_selection")
            metadata_context.record_error(
                component="DocumentProcessingManager",
                message=f"Adapter selection failed: {str(e)}",
                error_type=type(e).__name__,
                is_fatal=True
            )
            raise DocumentProcessingError(f"Adapter selection failed: {str(e)}")
    
    def _execute_preprocessing_phase(
        self,
        document_path: DocumentPath,
        selected_adapter: Type[DocumentBaseAdapter],
        analysis_result: Dict[str, Any],
        metadata_context: ProcessingMetadataContext
    ) -> str:
        """Execute the preprocessing phase."""
        metadata_context.start_timing("phase3_preprocessing")
        
        try:
            # Check if adaptive chunking is needed
            if self.chunking_service.should_apply_chunking(document_path, analysis_result):
                chunking_strategy = self.chunking_service.determine_chunking_strategy(
                    analysis_result
                )
                metadata_context.record_preprocessing_step(
                    step_name="chunking",
                    details=chunking_strategy,
                    component="ChunkingService"
                )
            
            # Get quality metrics and preprocessing recommendations
            quality_metrics = metadata_context.get_analysis_result("quality_metrics", {})
            preprocessing_recommendations = metadata_context.get_analysis_result(
                "preprocessing_recommendations",
                {}
            )
            
            # Update options with analysis results
            enhanced_options = self._prepare_options_with_analysis(
                self.config.__dict__,
                analysis_result,
                quality_metrics,
                preprocessing_recommendations
            )
            
            # Call the adapter's prepare_for_extraction method
            preparation_result = selected_adapter.prepare_for_extraction(
                document_path,
                options=enhanced_options,
                metadata_context=metadata_context
            )
            
            metadata_context.end_timing("phase3_preprocessing")
            
            # Return processed file path or original if no processing was needed
            return preparation_result.get("processed_file_path", document_path)
            
        except Exception as e:
            metadata_context.end_timing("phase3_preprocessing")
            metadata_context.record_error(
                component="DocumentProcessingManager",
                message=f"Preprocessing failed: {str(e)}",
                error_type=type(e).__name__,
                is_fatal=True
            )
            raise DocumentProcessingError(f"Preprocessing failed: {str(e)}")
    
    def _execute_processing_phase(
        self,
        processing_path: str,
        selected_adapter: Type[DocumentBaseAdapter],
        analysis_result: Dict[str, Any],
        metadata_context: ProcessingMetadataContext
    ) -> ProcessingResult:
        """Execute the main processing phase."""
        metadata_context.start_timing("phase4_processing")
        
        try:
            # Get enhanced options for processing
            enhanced_options = self._prepare_options_with_analysis(
                self.config.__dict__,
                analysis_result,
                metadata_context.get_analysis_result("quality_metrics", {}),
                metadata_context.get_analysis_result("preprocessing_recommendations", {})
            )
            
            # Process the document using the selected adapter
            processing_result = selected_adapter.process_document(
                processing_path,
                options=enhanced_options,
                metadata_context=metadata_context
            )
            
            metadata_context.end_timing("phase4_processing")
            return processing_result
            
        except Exception as e:
            metadata_context.end_timing("phase4_processing")
            metadata_context.record_error(
                component="DocumentProcessingManager",
                message=f"Processing failed: {str(e)}",
                error_type=type(e).__name__,
                is_fatal=True
            )
            raise DocumentProcessingError(f"Processing failed: {str(e)}")
    
    def _execute_kg_phase(
        self,
        processing_result: ProcessingResult,
        document_path: DocumentPath,
        metadata_context: ProcessingMetadataContext
    ) -> Optional[Dict[str, Any]]:
        """
        Execute knowledge graph integration phase.
        
        Args:
            processing_result: Result of document processing
            document_path: Path to the document
            metadata_context: Context for tracking metadata
        
        Returns:
            Optional knowledge graph data
        """
        if not self.config.prepare_for_kg:
            return None
        
        metadata_context.start_timing("kg_phase")
        
        try:
            # Initialize KG components if needed
            if not hasattr(self, "_kg_feedback_manager"):
                self._kg_feedback_manager = KGFeedbackManager()
            
            # Emit KG processing start event
            self._publish_processing_event(
                ProcessingEventType.KG_INTEGRATION_COMPLETE,
                document_path,
                {"status": "started"}
            )
            
            # Enhance document metadata using KG insights
            kg_enhanced = self._kg_feedback_manager.enhance_document_metadata(metadata_context)
            
            # Get document entities if available
            entities = []
            if processing_result.get("entities"):
                entities = processing_result["entities"]
            elif metadata_context.has_analysis_result("entities"):
                entities = metadata_context.get_analysis_result("entities")
            
            # Record enhanced entities in result
            if kg_enhanced and entities:
                # Get document ID
                document_id = metadata_context.document_metadata.get("processing_id")
                
                # Get context from knowledge graph
                kg_context = self._kg_feedback_manager.get_document_context(document_id)
                
                # Update processing result with KG context
                processing_result["knowledge_graph"] = {
                    "entities": entities,
                    "relationships": kg_context.get("relationships", []),
                    "context": kg_context
                }
                
                # Record decision
                metadata_context.record_decision(
                    component="DocumentProcessingManager._execute_kg_phase",
                    decision="Enhanced document with knowledge graph insights",
                    reason=f"Found {len(kg_context.get('relationships', []))} relationships from knowledge graph",
                    confidence=0.85
                )
                
                # Emit KG processing complete event
                self._publish_processing_event(
                    ProcessingEventType.KG_INTEGRATION_COMPLETE,
                    document_path,
                    {
                        "status": "complete",
                        "entity_count": len(entities),
                        "relationship_count": len(kg_context.get("relationships", []))
                    }
                )
            
            metadata_context.end_timing("kg_phase")
            return processing_result.get("knowledge_graph")
        
        except Exception as e:
            metadata_context.record_error(
                component="DocumentProcessingManager._execute_kg_phase",
                message=f"Knowledge graph integration failed: {str(e)}",
                error_type=type(e).__name__
            )
            metadata_context.end_timing("kg_phase")
            
            # Emit KG error event
            self._publish_processing_event(
                ProcessingEventType.ERROR_OCCURRED,
                document_path,
                {
                    "phase": "knowledge_graph",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            logger.error(f"Knowledge graph integration failed: {str(e)}")
            return None
    
    def _handle_processing_error(
        self,
        error: Exception,
        document_path: DocumentPath,
        metadata_context: ProcessingMetadataContext
    ) -> Dict[str, Any]:
        """Handle processing errors and generate appropriate error response."""
        error_type = type(error).__name__
        error_message = str(error)
        
        logger.error(f"Error processing document {document_path}: {error_message}")
        
        metadata_context.record_error(
            component="DocumentProcessingManager",
            message=error_message,
            error_type=error_type,
            is_fatal=True
        )
        
        return {
            "error": error_message,
            "error_type": error_type,
            "metadata": metadata_context.finalize_context(),
            "document_path": document_path
        }

    @handle_document_errors
    @handle_vision_errors
    @handle_timeout(timeout_seconds=60)
    def analyze_document_capabilities(self, document_path: str) -> Dict[str, Any]:
        """
        Analyze document capabilities without full processing.
        
        This method performs a lightweight analysis of the document to determine
        its required capabilities and suggest appropriate adapters without
        performing full document processing.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Dict containing document analysis, required capabilities, and recommended adapters
        """
        metadata_context = ProcessingMetadataContext(document_path)
        
        try:
            # Phase 1: Initial document analysis
            metadata_context.start_timing("capability_analysis")
            analysis_result = self.analyzer.analyze_document(document_path)
            metadata_context.record_analysis_result(
                analysis_type="document_analysis",
                result=analysis_result,
                component="DocumentAnalyzer"
            )
            
            # Phase 2: Derive required capabilities
            required_capabilities = self.capability_selector.derive_required_capabilities(
                document_path=document_path,
                document_analysis=analysis_result
            )
            metadata_context.record_analysis_result(
                analysis_type="required_capabilities",
                result=required_capabilities,
                component="CapabilityBasedSelector"
            )
            
            # Phase 3: Find matching adapters with scores
            adapter_matches = []
            for adapter in self.registry.get_all_adapters():
                adapter_name = adapter.__class__.__name__
                adapter_capabilities = adapter.get_capabilities()
                
                # Calculate match score
                match_score = self.capability_selector._calculate_capability_score(
                    adapter_capabilities, required_capabilities)
                
                adapter_matches.append({
                    "adapter_name": adapter_name,
                    "match_score": match_score,
                    "capabilities": adapter_capabilities
                })
            
            # Sort by match score
            adapter_matches.sort(key=lambda x: x["match_score"], reverse=True)
            
            # Record capability matches in metadata
            metadata_context.record_capability_matches(adapter_matches)
            metadata_context.end_timing("capability_analysis")
            
            # Prepare result
            return {
                "document_analysis": analysis_result,
                "required_capabilities": required_capabilities,
                "recommended_adapters": adapter_matches,
                "metadata": metadata_context.finalize_context()
            }
            
        except Exception as e:
            metadata_context.end_timing("capability_analysis")
            metadata_context.record_error(
                component="DocumentProcessingManager",
                message=f"Error analyzing document capabilities: {str(e)}",
                error_type=type(e).__name__,
                is_fatal=True
            )
            
            raise DocumentProcessingError(
                f"Error analyzing document capabilities: {str(e)}",
                error_code="capability_analysis_failed"
            )

    @batch_processing_complete
    @processing_complete
    @handle_document_errors
    @handle_vision_errors
    def batch_process_documents(self, document_paths: List[str], optimize_order: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch, optimizing order based on capabilities.
        
        Args:
            document_paths: List of paths to documents
            optimize_order: Whether to optimize the processing order based on capabilities
            
        Returns:
            List of processing results, one for each document
        """
        if not document_paths:
            return []
            
        batch_metadata = ProcessingMetadataContext("batch_processing")
        batch_metadata.start_timing("batch_processing")
        
        results = []
        failed_documents = []
        
        try:
            # Step 1: Analyze all documents to determine capabilities (lightweight)
            document_analyses = {}
            capability_groups = {}
            
            if optimize_order:
                batch_metadata.start_timing("batch_analysis")
                
                # Perform lightweight analysis on all documents
                for doc_path in document_paths:
                    try:
                        # Create individual metadata context for this analysis
                        doc_metadata = ProcessingMetadataContext(doc_path)
                        
                        # Perform document analysis
                        analysis = self.analyze_document_capabilities(doc_path)
                        document_analyses[doc_path] = analysis
                        
                        # Group by primary capability requirements
                        primary_capability = max(
                            analysis["required_capabilities"].items(), 
                            key=lambda x: x[1]
                        )[0] if analysis["required_capabilities"] else "default"
                        
                        if primary_capability not in capability_groups:
                            capability_groups[primary_capability] = []
                            
                        capability_groups[primary_capability].append(doc_path)
                        
                    except Exception as e:
                        logger.warning(f"Error during pre-analysis of {doc_path}: {str(e)}")
                        # Add to failed documents
                        failed_documents.append({
                            "document_path": doc_path,
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
                
                batch_metadata.end_timing("batch_analysis")
                
                # Record the capability grouping strategy
                batch_metadata.record_analysis_result(
                    analysis_type="batch_optimization",
                    result={
                        "capability_groups": {k: len(v) for k, v in capability_groups.items()},
                        "total_documents": len(document_paths),
                        "pre_analyzed_documents": len(document_analyses)
                    },
                    component="DocumentProcessingManager"
                )
                
                # Step 2: Process documents in capability-based order
                # Process each capability group
                for capability, docs in capability_groups.items():
                    for doc_path in docs:
                        try:
                            # Create metadata context with batch information
                            doc_metadata = ProcessingMetadataContext(doc_path)
                            doc_metadata.record_preprocessing_step(
                                step_name="batch_assignment",
                                details={"capability_group": capability, "group_size": len(docs)},
                                component="DocumentProcessingManager"
                            )
                            
                            # Process with pre-analyzed capabilities
                            pre_analysis = document_analyses.get(doc_path, {})
                            result = self.process_document(doc_path, metadata_context=doc_metadata)
                            results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Error processing document {doc_path}: {str(e)}")
                            # Add to failed documents if not already added
                            if doc_path not in [d["document_path"] for d in failed_documents]:
                                failed_documents.append({
                                    "document_path": doc_path,
                                    "error": str(e),
                                    "error_type": type(e).__name__
                                })
            else:
                # Simple sequential processing without optimization
                for doc_path in document_paths:
                    try:
                        # Create metadata context
                        doc_metadata = ProcessingMetadataContext(doc_path)
                        
                        # Process document
                        result = self.process_document(doc_path, metadata_context=doc_metadata)
                        results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Error processing document {doc_path}: {str(e)}")
                        failed_documents.append({
                            "document_path": doc_path,
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
            
            # Add failure information to batch metadata
            batch_metadata.record_analysis_result(
                analysis_type="batch_failures",
                result={
                    "total_failed": len(failed_documents),
                    "failures": failed_documents
                },
                component="DocumentProcessingManager"
            )
            
            batch_metadata.end_timing("batch_processing")
            
            # Add batch metadata to each result
            batch_summary = batch_metadata.finalize_context()
            for result in results:
                if "metadata" in result:
                    result["metadata"]["batch_processing"] = batch_summary
            
            return results
            
        except Exception as e:
            batch_metadata.end_timing("batch_processing")
            batch_metadata.record_error(
                component="DocumentProcessingManager",
                message=f"Error in batch processing: {str(e)}",
                error_type=type(e).__name__,
                is_fatal=True
            )
            
            logger.error(f"Batch processing error: {str(e)}")
            raise DocumentProcessingError(
                f"Error in batch processing: {str(e)}",
                error_code="batch_processing_failed"
            )
    
    @handle_document_errors
    @handle_vision_errors
    @handle_timeout(timeout_seconds=300)
    def _process_document_internal(
        self,
        document_path: DocumentPath,
        metadata_context: ProcessingMetadataContext,
        is_chunk: bool = False
    ) -> ProcessingResult:
        """
        Internal method for document processing, used by both main processing and chunk processing.
        
        Args:
            document_path: Path to document or chunk
            metadata_context: Metadata context for tracking decisions
            is_chunk: Whether this is processing a chunk of a larger document
            
        Returns:
            Dict containing extracted content, structure, and metadata
        """
        try:
            # Skip chunking check if already processing a chunk
            if not is_chunk:
                # Check if chunking is needed
                chunking_result = self.chunking_service.check_if_chunking_needed(
                    document_path,
                    metadata_context
                )
                
                if chunking_result["chunking_needed"]:
                    return self._handle_document_chunking(
                        document_path,
                        metadata_context,
                        chunking_result
                    )
            
            # Proceed with normal processing phases
            analysis_result = self._execute_analysis_phase(document_path, metadata_context)
            selected_adapter = self._execute_selection_phase(
                document_path,
                analysis_result,
                metadata_context
            )
            
            processing_path = self._execute_preprocessing_phase(
                document_path,
                selected_adapter,
                analysis_result,
                metadata_context
            )
            
            processing_result = self._execute_processing_phase(
                processing_path,
                selected_adapter,
                analysis_result,
                metadata_context
            )
            
            # Add chunk-specific metadata if processing a chunk
            if is_chunk:
                processing_result["is_chunk"] = True
                processing_result["chunk_metadata"] = metadata_context.get_preprocessing_step("chunking")
            
            # At the beginning of processing:
            self._publish_processing_event(
                ProcessingEventType.DOCUMENT_RECEIVED,
                document_path,
                {"is_chunk": is_chunk}
            )
            
            # After successful processing:
            self._publish_processing_event(
                ProcessingEventType.PROCESSING_COMPLETE,
                document_path,
                {"processing_time": metadata_context.get_timing("total_processing_time")}
            )
            
            return processing_result
            
        except Exception as e:
            error_info = self._handle_processing_error(e, document_path, metadata_context)
            if is_chunk:
                error_info["is_chunk"] = True
            
            # In error handling:
            self._publish_processing_event(
                ProcessingEventType.ERROR_OCCURRED,
                document_path,
                {"error": str(e), "error_type": type(e).__name__}
            )
            
            # After adding to DLQ:
            try:
                dlq_manager = DLQManager()
                dlq_path = dlq_manager.add_to_dlq(document_path, e, metadata_context)
                self._publish_processing_event(
                    ProcessingEventType.DOCUMENT_QUEUED,
                    document_path,
                    {"queue": "dlq", "dlq_path": dlq_path}
                )
            except Exception as dlq_error:
                logger.error(f"Error adding document to DLQ: {str(dlq_error)}")
                
            return error_info

    def _handle_document_chunking(
        self,
        document_path: DocumentPath,
        metadata_context: ProcessingMetadataContext,
        chunking_result: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Handle document chunking based on size and complexity.
        
        Args:
            document_path: Path to the document file
            metadata_context: Context for tracking processing metadata
            chunking_result: Result of chunking analysis
        
        Returns:
            ProcessingResult: Combined result from all chunks
        """
        metadata_context.start_timing("document_chunking")
        
        # Get chunking service
        chunking_service = ChunkingService(config=self.config.chunking_config)
        
        # Classify document into size category
        doc_classification = chunking_service.classify_document(document_path)
        
        # Record classification decision
        metadata_context.record_decision(
            component="DocumentProcessingManager",
            decision=f"Document classified as '{doc_classification}'",
            reason="Based on file size and page count analysis",
            confidence=0.95
        )
        
        # Process based on classification
        processing_result = None
        
        try:
            if doc_classification == "large":
                # Process large document with specialized strategy
                processing_result = self._process_large_document(
                    document_path, 
                    metadata_context,
                    chunking_result
                )
            elif doc_classification == "single_page":
                # Process single page document directly
                processing_result = self._process_single_page_document(
                    document_path, 
                    metadata_context
                )
            else:  # "small"
                # Process small document with standard strategy
                processing_result = self._process_small_document(
                    document_path, 
                    metadata_context,
                    chunking_result
                )
                
            metadata_context.end_timing("document_chunking")
            return processing_result
            
        except Exception as e:
            metadata_context.record_error(
                component="DocumentProcessingManager._handle_document_chunking",
                message=f"Error during document chunking: {str(e)}",
                error_type=type(e).__name__,
                is_fatal=True
            )
            metadata_context.end_timing("document_chunking")
            
            # Add to DLQ if available
            try:
                dlq_manager = DLQManager()
                dlq_manager.add_to_dlq(document_path, e, metadata_context)
            except Exception as dlq_error:
                logger.error(f"Error adding document to DLQ: {str(dlq_error)}")
                
            raise DocumentProcessingError(f"Document chunking failed: {str(e)}")
    
    def _process_large_document(
        self,
        document_path: DocumentPath,
        metadata_context: ProcessingMetadataContext,
        chunking_result: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Process a large document by breaking it into manageable chunks.
        
        Args:
            document_path: Path to the document
            metadata_context: Metadata context
            chunking_result: Result of chunking analysis
        
        Returns:
            ProcessingResult: Combined result from all chunks
        """
        chunking_service = ChunkingService(config=self.config.chunking_config)
        
        # Create a chunking strategy
        doc_info = {
            "file_size": os.path.getsize(document_path),
            "content_types": chunking_result.get("content_types", ["text"])
        }
        
        strategy = chunking_service.determine_chunking_strategy(doc_info)
        
        # Create document chunks
        chunks = chunking_service.create_chunks(document_path, strategy, metadata_context)
        
        # Process each chunk
        chunk_results = []
        for chunk in chunks:
            try:
                # Create sub-context for chunk
                chunk_context = metadata_context.create_child_context()
                
                # Process chunk
                result = self._process_document_internal(
                    DocumentPath(os.path.join(document_path, chunk.chunk_id)),
                    chunk_context,
                    is_chunk=True
                )
                
                chunk_results.append(result)
                
                # Merge child context
                metadata_context.merge_child_context(chunk_context)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk.chunk_id}: {str(e)}")
                metadata_context.record_error(
                    component="DocumentProcessingManager._process_large_document",
                    message=f"Chunk processing error: {str(e)}",
                    error_type=type(e).__name__
                )
        
        # Merge chunk results
        section_fusion = SectionFusion()
        combined_result = section_fusion.fuse_sections(
            [r.get("text_sections", []) for r in chunk_results if r],
            [r.get("image_sections", []) for r in chunk_results if r],
            metadata_context.get_metadata()
        )
        
        return ProcessingResult(
            document_path=document_path,
            content=combined_result.get("text", ""),
            structure=combined_result.get("structure", {}),
            metadata=combined_result.get("metadata", {})
        )

    def _process_small_document(
        self,
        document_path: DocumentPath,
        metadata_context: ProcessingMetadataContext,
        chunking_result: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Process a small document with normal processing flow.
        
        Args:
            document_path: Path to the document
            metadata_context: Metadata context
            chunking_result: Result of chunking analysis
        
        Returns:
            ProcessingResult: Processing result
        """
        # Use standard processing flow but with optimized parameters
        return self._process_document_internal(
            document_path,
            metadata_context,
            is_chunk=False,
            optimization_settings={"resource_priority": "speed"}
        )

    def _process_single_page_document(
        self,
        document_path: DocumentPath,
        metadata_context: ProcessingMetadataContext
    ) -> ProcessingResult:
        """
        Process a single page document with optimized flow.
        
        Args:
            document_path: Path to the document
            metadata_context: Metadata context
        
        Returns:
            ProcessingResult: Processing result
        """
        # Use standard processing flow but with single-page optimizations
        return self._process_document_internal(
            document_path,
            metadata_context,
            is_chunk=False,
            optimization_settings={"resource_priority": "quality", "single_page_optimized": True}
        )

    def _enhance_capabilities_from_content(self, required_capabilities: Dict[str, float],
                                         analysis_result: Dict[str, Any],
                                         metadata_context: ProcessingMetadataContext) -> None:
        """
        Enhance required capabilities based on specific content features.
        
        Args:
            required_capabilities: The required capabilities to enhance
            analysis_result: Document analysis results
            metadata_context: Metadata context for recording decisions
        """
        # Check for specific content types that need specialized adapters
        
        # Check for form content
        if analysis_result.get("is_form", False):
            required_capabilities["forms"] = max(required_capabilities.get("forms", 0), 0.8)
            metadata_context.record_decision(
                component="DocumentProcessingManager",
                decision="Enhanced form capability requirement",
                reason="Document contains form elements"
            )
        
        # Check for tables
        table_count = analysis_result.get("table_count", 0)
        if table_count > 0:
            required_capabilities["tables"] = max(required_capabilities.get("tables", 0), 0.7)
            if table_count > 5:
                required_capabilities["tables"] = max(required_capabilities.get("tables", 0), 0.85)
                metadata_context.record_decision(
                    component="DocumentProcessingManager",
                    decision="Enhanced table capability requirement",
                    reason=f"Document contains {table_count} tables"
                )

        # Check for complex layout
        if analysis_result.get("has_complex_layout", False):
            required_capabilities["complex_layouts"] = max(required_capabilities.get("complex_layouts", 0), 0.75)
            metadata_context.record_decision(
                component="DocumentProcessingManager",
                decision="Enhanced complex layout capability requirement",
                reason="Document has complex layout"
            )
            
        # Check for language-specific requirements
        languages = analysis_result.get("languages", {})
        if languages:
            primary_language = max(languages.items(), key=lambda x: x[1])[0]
            if primary_language not in ["en", "english"]:
                metadata_context.record_decision(
                    component="DocumentProcessingManager",
                    decision=f"Noted non-English document language: {primary_language}",
                    reason="May require specialized language processing"
                )

        # Record enhanced capabilities
        metadata_context.record_capability_requirements(
            component="DocumentProcessingManager",
            capabilities=required_capabilities,
            reason="Enhanced from content analysis"
        )
    
    @handle_document_errors
    def _prepare_options_with_analysis(self, options: Dict[str, Any],
                                     analysis_result: Dict[str, Any],
                                     quality_metrics: Dict[str, Any],
                                     preprocessing_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepares enhanced options dictionary by incorporating analysis results.
        This ensures adapters and processors have access to analysis information.
        
        Args:
            options: Original options dictionary
            analysis_result: Results from document analysis
            quality_metrics: Document quality metrics
            preprocessing_recommendations: Preprocessing recommendations
            
        Returns:
            Dict: Enhanced options with analysis results
        """
        enhanced_options = options.copy()
        
        # Add analysis results to options for adapters and processors to use
        enhanced_options["document_analysis"] = analysis_result
        enhanced_options["document_type"] = analysis_result.get("document_type", {})
        enhanced_options["quality_metrics"] = quality_metrics
        enhanced_options["preprocessing_recommendations"] = preprocessing_recommendations
        
        # Extract specialized processing hints
        enhanced_options["has_tables"] = analysis_result.get("has_tables", False)
        enhanced_options["has_images"] = analysis_result.get("has_images", False)
        enhanced_options["has_forms"] = analysis_result.get("is_form", False)
        enhanced_options["has_formulas"] = analysis_result.get("has_formulas", False)
        enhanced_options["document_complexity"] = analysis_result.get("complexity", 0.5)
        enhanced_options["image_ratio"] = analysis_result.get("image_ratio", 0.0)
        
        # Add capability information for adapter internal decisions
        doc_type_info = analysis_result.get("document_type", {})
        primary_doc_type = doc_type_info.get("type", "unknown")
        
        # Include capability weights used for selection
        if primary_doc_type in self.capability_weights:
            enhanced_options["capability_weights"] = self.capability_weights[primary_doc_type]
        elif "mixed" in analysis_result.get("document_classification", {}) and analysis_result["document_classification"]["mixed"] > 0.5:
            enhanced_options["capability_weights"] = self.capability_weights["mixed"]
        else:
            # Default weights
            enhanced_options["capability_weights"] = self.capability_selector.capability_weights
        
        # Derive required capabilities if not already in options
        if "required_capabilities" not in enhanced_options:
            required_capabilities = self.capability_selector.derive_required_capabilities(analysis_result)
            enhanced_options["required_capabilities"] = required_capabilities
        
        # Processing strategy hints based on document type and capabilities
        enhanced_options["processing_strategy"] = self._determine_processing_strategy(
            enhanced_options["required_capabilities"],
            doc_type_info,
            analysis_result.get("document_classification", {})
        )
        
        # OCR-specific options
        if "ocr_params" in preprocessing_recommendations:
            enhanced_options["ocr_params"] = preprocessing_recommendations["ocr_params"]
        
        return enhanced_options
        
    @handle_document_errors
    def _determine_processing_strategy(self, required_capabilities: Dict[str, float],
                                     doc_type_info: Dict[str, Any],
                                     doc_classification: Dict[str, float]) -> Dict[str, Any]:
        """
        Determine the optimal processing strategy based on document characteristics and required capabilities.
        
        Args:
            required_capabilities: Capabilities required for processing this document
            doc_type_info: Document type information
            doc_classification: Document classification scores
            
        Returns:
            Dict: Processing strategy configuration
        """
        strategy = {
            "name": "standard",
            "chunking": False,
            "priority_capabilities": [],
            "preprocessing_level": "auto"
        }
        
        # Determine primary document type
        doc_type = doc_type_info.get("type", "unknown")
        is_scanned = doc_type_info.get("is_scanned", False)
        
        # Check for scanned documents
        if is_scanned:
            strategy["name"] = "ocr_priority"
            strategy["priority_capabilities"] = ["text_extraction", "scanned_documents"]
            strategy["preprocessing_level"] = "aggressive"
        
        # Check for image-heavy documents
        elif doc_type == "image" or required_capabilities.get("images", 0) > 0.8:
            strategy["name"] = "image_priority"
            strategy["priority_capabilities"] = ["images", "diagrams", "charts"]
            strategy["preprocessing_level"] = "balanced"
        
        # Check for PDFs with complex structure
        elif doc_type == "pdf" and required_capabilities.get("complex_layouts", 0) > 0.7:
            strategy["name"] = "structure_priority"
            strategy["priority_capabilities"] = ["complex_layouts", "pdfs", "mixed_content"]
            strategy["preprocessing_level"] = "minimal"
        
        # Check for office documents with tables
        elif doc_type in ["document", "spreadsheet"] and required_capabilities.get("tables", 0) > 0.7:
            strategy["name"] = "table_priority"
            strategy["priority_capabilities"] = ["tables", "office_documents"]
            strategy["preprocessing_level"] = "minimal"
        
        # Mixed content documents
        elif required_capabilities.get("mixed_content", 0) > 0.7:
            strategy["name"] = "hybrid"
            strategy["priority_capabilities"] = ["mixed_content", "complex_layouts"]
            strategy["preprocessing_level"] = "balanced"
        
        # Check for chunking needs
        if "pdf" in doc_classification and doc_classification["pdf"] > 0.7:
            if doc_type_info.get("page_count", 0) > 20:
                strategy["chunking"] = True
                strategy["chunk_size"] = "medium"  # small, medium, large
        
        return strategy
    
    @handle_document_errors
    def _generate_kg_extraction_hints(self, analysis_result: Dict[str, Any],
                                    doc_classification: Dict[str, float]) -> Dict[str, Any]:
        """
        Generates knowledge graph extraction hints based on document analysis.
        
        Args:
            analysis_result: Document analysis result
            doc_classification: Document classification result
            
        Returns:
            Dict: Knowledge graph extraction hints
        """
        # Start with basic hints
        kg_hints = {
            "entity_types": [],
            "prioritize_relationships": False,
            "extraction_strategy": "standard"
        }
        
        # Add entity types based on document classification
        document_type = max(doc_classification.items(), key=lambda x: x[1])[0] if doc_classification else None
        
        if document_type:
            if document_type == "invoice":
                kg_hints["entity_types"] = ["Organization", "Person", "Date", "Currency", "Product", "Service"]
                kg_hints["extraction_strategy"] = "financial"
            elif document_type == "contract":
                kg_hints["entity_types"] = ["Organization", "Person", "Date", "Location", "Clause", "Obligation"]
                kg_hints["prioritize_relationships"] = True
                kg_hints["extraction_strategy"] = "legal"
            elif document_type == "academic_paper":
                kg_hints["entity_types"] = ["Person", "Organization", "Concept", "Citation", "Formula"]
                kg_hints["extraction_strategy"] = "academic"
            elif document_type == "form":
                kg_hints["entity_types"] = ["Person", "Organization", "Date", "FormField", "FieldValue"]
                kg_hints["extraction_strategy"] = "form"
                kg_hints["prioritize_relationships"] = True
        
        # Add hints based on analysis results
        if analysis_result.get("has_tables", False):
            kg_hints["has_structured_data"] = True
            kg_hints["entity_types"].append("Table")
        
        if analysis_result.get("has_formulas", False):
            kg_hints["has_mathematical_content"] = True
            if "Formula" not in kg_hints["entity_types"]:
                kg_hints["entity_types"].append("Formula")
                
        if analysis_result.get("is_form", False):
            kg_hints["has_form_elements"] = True
            kg_hints["extraction_strategy"] = "form"
            
            # Add form-specific entity types if not already present
            form_entities = ["FormField", "FieldValue", "Checkbox", "RadioButton", "TextField"]
            for entity in form_entities:
                if entity not in kg_hints["entity_types"]:
                    kg_hints["entity_types"].append(entity)
        
        return kg_hints

    def _update_processing_stats(self, adapter_name: str, document_path: str, 
                              processing_time: float, success: bool) -> None:
        """
        Update processing statistics.
        
        Args:
            adapter_name: Name of the used adapter
            document_path: Path to the processed document
            processing_time: Time taken for processing
            success: Whether processing was successful
        """
        # Get document extension
        doc_ext = os.path.splitext(document_path)[1].lower()
        
        # Update general stats
        self.processing_stats["processed_documents"] += 1
        
        if success:
            self.processing_stats["successful_processing"] += 1
        else:
            self.processing_stats["failed_processing"] += 1
            
        self.processing_stats["total_processing_time"] += processing_time
        self.processing_stats["avg_processing_time"] = (
            self.processing_stats["total_processing_time"] / 
            self.processing_stats["processed_documents"]
        )
        
        # Update document type stats
        if doc_ext not in self.processing_stats["document_types_stats"]:
            self.processing_stats["document_types_stats"][doc_ext] = {
                "count": 0,
                "success_count": 0,
                "fail_count": 0,
                "total_time": 0,
                "avg_time": 0
            }
            
        doc_stats = self.processing_stats["document_types_stats"][doc_ext]
        doc_stats["count"] += 1
        
        if success:
            doc_stats["success_count"] += 1
        else:
            doc_stats["fail_count"] += 1
            
        doc_stats["total_time"] += processing_time
        doc_stats["avg_time"] = doc_stats["total_time"] / doc_stats["count"]
        
        # Update adapter usage
        if adapter_name not in self.processing_stats["capability_usage"]:
            self.processing_stats["capability_usage"][adapter_name] = {
                "document_count": 0,
                "success_count": 0,
                "total_time": 0
            }
            
        adapter_stats = self.processing_stats["capability_usage"][adapter_name]
        adapter_stats["document_count"] += 1
        
        if success:
            adapter_stats["success_count"] += 1
            
        adapter_stats["total_time"] += processing_time

    def _create_document_embedding(
        self,
        processing_result: Dict[str, Any],
        metadata_context: ProcessingMetadataContext
    ) -> Optional[Dict[str, Any]]:
        """
        Creates document embeddings for RAG.
        This is a placeholder that will be implemented when adding RAG support.
        
        Args:
            processing_result: Results from document processing
            metadata_context: Metadata context
            
        Returns:
            Optional[Dict]: Embedding results if successful
        """
        # TODO: Implement when adding RAG support
        metadata_context.record_decision(
            component="DocumentProcessingManager",
            decision="Embedding creation deferred",
            reason="RAG support pending implementation"
        )
        return None

    @handle_document_errors
    @handle_vision_errors
    @handle_resource_limits
    @vision_resource_handler
    def _monitor_resource_usage(self) -> Dict[str, float]:
        """Monitor current resource usage including GPU if available."""
        try:
            process = psutil.Process()
            
            # Get CPU and memory usage
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get disk I/O
            disk_io = process.io_counters()
            
            resource_stats = {
                "cpu_percent": cpu_percent,
                "memory_used_mb": memory_info.rss / (1024 * 1024),
                "memory_percent": memory_percent,
                "disk_read_mb": disk_io.read_bytes / (1024 * 1024),
                "disk_write_mb": disk_io.write_bytes / (1024 * 1024)
            }

            # Monitor GPU if available
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                
                gpu_stats = []
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    gpu_stats.append({
                        "device_id": i,
                        "memory_used_mb": info.used / (1024 * 1024),
                        "memory_total_mb": info.total / (1024 * 1024),
                        "memory_percent": (info.used / info.total) * 100,
                        "gpu_utilization": utilization.gpu,
                        "memory_utilization": utilization.memory
                    })
                
                resource_stats["gpu_stats"] = gpu_stats
                
                # Update GPU usage history
                if gpu_stats:
                    avg_gpu_util = sum(g["gpu_utilization"] for g in gpu_stats) / len(gpu_stats)
                    avg_gpu_mem = sum(g["memory_percent"] for g in gpu_stats) / len(gpu_stats)
                    
                    if "gpu_usage" not in self.processing_stats["resource_usage"]:
                        self.processing_stats["resource_usage"]["gpu_usage"] = []
                        self.processing_stats["resource_usage"]["gpu_memory"] = []
                    
                    self.processing_stats["resource_usage"]["gpu_usage"].append(avg_gpu_util)
                    self.processing_stats["resource_usage"]["gpu_memory"].append(avg_gpu_mem)
                    
                    # Keep only last 100 measurements for GPU as well
                    max_history = 100
                    if len(self.processing_stats["resource_usage"]["gpu_usage"]) > max_history:
                        self.processing_stats["resource_usage"]["gpu_usage"] = self.processing_stats["resource_usage"]["gpu_usage"][-max_history:]
                        self.processing_stats["resource_usage"]["gpu_memory"] = self.processing_stats["resource_usage"]["gpu_memory"][-max_history:]
                
            except ImportError:
                logger.debug("GPU monitoring not available - pynvml not installed")
            except Exception as e:
                logger.warning(f"Error monitoring GPU: {str(e)}")
            
            # Update CPU and memory history
            self.processing_stats["resource_usage"]["cpu_usage"].append(cpu_percent)
            self.processing_stats["resource_usage"]["memory_usage"].append(memory_percent)
            
            # Keep only last 100 measurements
            max_history = 100
            if len(self.processing_stats["resource_usage"]["cpu_usage"]) > max_history:
                self.processing_stats["resource_usage"]["cpu_usage"] = self.processing_stats["resource_usage"]["cpu_usage"][-max_history:]
                self.processing_stats["resource_usage"]["memory_usage"] = self.processing_stats["resource_usage"]["memory_usage"][-max_history:]
            
            return resource_stats
            
        except Exception as e:
            logger.warning(f"Error monitoring resources: {str(e)}")
            return {}

    @handle_document_errors
    @handle_vision_errors
    @handle_resource_limits
    @vision_resource_handler
    def _check_resource_limits(self, resource_stats: Dict[str, float]) -> bool:
        """
        Check if current resource usage exceeds limits.
        
        Args:
            resource_stats: Current resource statistics
            
        Returns:
            bool: True if limits are exceeded
        """
        if not self.config.resource_limits:
            return False
            
        limits = self.config.resource_limits
        
        # Check CPU limit
        if "max_cpu_percent" in limits and resource_stats.get("cpu_percent", 0) > limits["max_cpu_percent"]:
            raise DocumentResourceError(
                "CPU usage exceeds limit",
                resource_type="cpu",
                details={
                    "current": resource_stats.get("cpu_percent", 0),
                    "limit": limits["max_cpu_percent"]
                }
            )
        
        # Check memory limit
        if "max_memory_percent" in limits and resource_stats.get("memory_percent", 0) > limits["max_memory_percent"]:
            raise DocumentResourceError(
                "Memory usage exceeds limit",
                resource_type="memory",
                details={
                    "current": resource_stats.get("memory_percent", 0),
                    "limit": limits["max_memory_percent"]
                }
            )
        
        # Check disk I/O limits
        if "max_disk_read_mb" in limits and resource_stats.get("disk_read_mb", 0) > limits["max_disk_read_mb"]:
            raise DocumentResourceError(
                "Disk read usage exceeds limit",
                resource_type="disk_read",
                details={
                    "current": resource_stats.get("disk_read_mb", 0),
                    "limit": limits["max_disk_read_mb"]
                }
            )
        
        if "max_disk_write_mb" in limits and resource_stats.get("disk_write_mb", 0) > limits["max_disk_write_mb"]:
            raise DocumentResourceError(
                "Disk write usage exceeds limit",
                resource_type="disk_write",
                details={
                    "current": resource_stats.get("disk_write_mb", 0),
                    "limit": limits["max_disk_write_mb"]
                }
            )
        
        return False

    def cleanup(self) -> None:
        """
        Cleanup resources used by the document processing manager.
        Should be called when the manager is no longer needed.
        """
        try:
            # Close thread pool if it exists
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
            
            # Clean up temporary files
            self._cleanup_temp_files()
            
            # Close any open adapters
            for adapter in self.adapters.values():
                if hasattr(adapter, 'cleanup') and callable(adapter.cleanup):
                    try:
                        adapter.cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up adapter {adapter.__class__.__name__}: {str(e)}")
            
            # Clear caches
            if self.config.cache_enabled:
                cache_keys = [
                    "document_manager_adapter_capabilities",
                    "document_manager_processing_stats"
                ]
                for key in cache_keys:
                    try:
                        cache.delete(key)
                    except Exception as e:
                        logger.warning(f"Error clearing cache key {key}: {str(e)}")
            
            logger.info("Document processing manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise DocumentProcessingError(f"Cleanup failed: {str(e)}")

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary files created during processing."""
        try:
            temp_dir = getattr(settings, 'DOCUMENT_PROCESSING_TEMP_DIR', None)
            if not temp_dir or not os.path.exists(temp_dir):
                return
                
            # Get list of temporary files
            temp_files = []
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Check if file is older than 24 hours
                    if time.time() - os.path.getctime(file_path) > 86400:  # 24 hours
                        temp_files.append(file_path)
            
            # Remove old temporary files
            for file_path in temp_files:
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed temporary file: {file_path}")
                except Exception as e:
                    logger.warning(f"Error removing temporary file {file_path}: {str(e)}")
            
            logger.info(f"Cleaned up {len(temp_files)} temporary files")
            
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

    def __enter__(self) -> 'DocumentProcessingManager':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()

    def _optimize_processing_pipeline(
        self,
        document_path: DocumentPath,
        analysis_result: Dict[str, Any],
        metadata_context: ProcessingMetadataContext
    ) -> Dict[str, Any]:
        """
        Optimize the processing pipeline based on document characteristics and system state.
        
        Args:
            document_path: Path to the document
            analysis_result: Document analysis results
            metadata_context: Metadata context
            
        Returns:
            Dict containing optimization settings
        """
        optimization_settings = {
            "use_parallel": False,
            "chunk_size": None,
            "cache_strategy": "default",
            "preprocessing_level": "standard"
        }
        
        try:
            # Check system resources
            resource_stats = self._monitor_resource_usage()
            
            # Determine if parallel processing would be beneficial
            file_size = os.path.getsize(document_path)
            doc_type = analysis_result.get("document_type", {}).get("type", "unknown")
            page_count = analysis_result.get("page_count", 1)
            
            # Enable parallel processing for large documents with multiple pages
            if (self.config.enable_parallel_processing and
                page_count > 5 and
                file_size > 5 * 1024 * 1024 and  # 5MB
                resource_stats.get("cpu_percent", 0) < 70):
                
                optimization_settings["use_parallel"] = True
                optimization_settings["worker_count"] = min(
                    cpu_count(),
                    page_count,
                    max(2, int(16 - resource_stats.get("cpu_percent", 0) / 10))
                )
            
            # Determine chunking strategy
            if page_count > 20 or file_size > 20 * 1024 * 1024:  # 20MB
                optimization_settings["chunk_size"] = "medium"
                if page_count > 50:
                    optimization_settings["chunk_size"] = "large"
            
            # Set caching strategy
            if self.config.cache_enabled:
                if file_size < 1024 * 1024:  # 1MB
                    optimization_settings["cache_strategy"] = "aggressive"
                elif file_size < 10 * 1024 * 1024:  # 10MB
                    optimization_settings["cache_strategy"] = "moderate"
                else:
                    optimization_settings["cache_strategy"] = "minimal"
            
            # Adjust preprocessing level based on document quality
            quality_metrics = metadata_context.get_analysis_result("quality_metrics", {})
            if quality_metrics:
                quality_score = quality_metrics.get("overall_quality", 0.5)
                if quality_score < 0.3:
                    optimization_settings["preprocessing_level"] = "aggressive"
                elif quality_score < 0.7:
                    optimization_settings["preprocessing_level"] = "balanced"
                else:
                    optimization_settings["preprocessing_level"] = "minimal"
            
            # Record optimization decisions
            metadata_context.record_decision(
                component="DocumentProcessingManager",
                decision=f"Applied optimization settings: {optimization_settings}",
                reason="Based on document characteristics and system state"
            )
            
            return optimization_settings
            
        except Exception as e:
            logger.warning(f"Error optimizing pipeline: {str(e)}")
            return optimization_settings

    def _get_cached_result(
        self,
        document_path: DocumentPath,
        metadata_context: ProcessingMetadataContext
    ) -> Optional[ProcessingResult]:
        """
        Try to get cached processing result.
        
        Args:
            document_path: Path to the document
            metadata_context: Metadata context
            
        Returns:
            Optional[Dict]: Cached result if available
        """
        if not self.config.cache_enabled:
            return None
            
        try:
            # Generate cache key based on file content hash and metadata
            file_hash = self._get_file_hash(document_path)
            cache_key = f"doc_processing_{file_hash}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result:
                # Verify cache is still valid
                if self._validate_cache_entry(cached_result, document_path):
                    metadata_context.record_decision(
                        component="DocumentProcessingManager",
                        decision="Used cached processing result",
                        reason=f"Valid cache found for {document_path}"
                    )
                    return cached_result
                
            return None
            
        except Exception as e:
            logger.warning(f"Error accessing cache: {str(e)}")
            return None

    def _cache_processing_result(
        self,
        document_path: DocumentPath,
        result: ProcessingResult,
        optimization_settings: Dict[str, Any]
    ) -> None:
        """
        Cache processing result based on optimization settings.
        
        Args:
            document_path: Path to the document
            result: Processing result to cache
            optimization_settings: Optimization settings including cache strategy
        """
        if not self.config.cache_enabled:
            return
            
        try:
            # Generate cache key
            file_hash = self._get_file_hash(document_path)
            cache_key = f"doc_processing_{file_hash}"
            
            # Determine cache timeout based on strategy
            cache_strategy = optimization_settings.get("cache_strategy", "default")
            if cache_strategy == "aggressive":
                timeout = self.config.cache_timeout * 2
            elif cache_strategy == "moderate":
                timeout = self.config.cache_timeout
            else:  # minimal
                timeout = self.config.cache_timeout // 2
            
            # Add cache metadata
            result["cache_metadata"] = {
                "cached_at": datetime.datetime.now().isoformat(),
                "cache_strategy": cache_strategy,
                "file_hash": file_hash
            }
            
            # Store in cache
            cache.set(cache_key, result, timeout)
            
        except Exception as e:
            logger.warning(f"Error caching result: {str(e)}")

    def _get_file_hash(self, file_path: str, block_size: int = 65536) -> str:
        """
        Get SHA-256 hash of file contents.
        
        Args:
            file_path: Path to the file
            block_size: Size of blocks to read
            
        Returns:
            str: Hash of file contents
        """
        import hashlib
        
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(block_size), b""):
                sha256_hash.update(byte_block)
                
        return sha256_hash.hexdigest()

    def _validate_cache_entry(self, cached_result: Dict[str, Any], document_path: str) -> bool:
        """Validate if cached result is still valid."""
        try:
            # Check if cache metadata exists
            cache_metadata = cached_result.get("cache_metadata")
            if not cache_metadata:
                raise DocumentCacheError(
                    "Invalid cache entry - missing metadata",
                    cache_operation="validation",
                    document_path=document_path
                )
                
            # Check if file has been modified
            cached_hash = cache_metadata.get("file_hash")
            current_hash = self._get_file_hash(document_path)
            if cached_hash != current_hash:
                raise DocumentCacheError(
                    "Cache entry invalid - document modified",
                    cache_operation="validation",
                    document_path=document_path,
                    details={
                        "cached_hash": cached_hash,
                        "current_hash": current_hash
                    }
                )
                
            # Check cache age
            cached_at = datetime.datetime.fromisoformat(cache_metadata["cached_at"])
            age = (datetime.datetime.now() - cached_at).total_seconds()
            
            # Validate based on cache strategy
            cache_strategy = cache_metadata.get("cache_strategy", "default")
            if cache_strategy == "aggressive":
                max_age = self.config.cache_timeout * 2
            elif cache_strategy == "moderate":
                max_age = self.config.cache_timeout
            else:  # minimal
                max_age = self.config.cache_timeout // 2
                
            if age >= max_age:
                raise DocumentCacheError(
                    "Cache entry expired",
                    cache_operation="validation",
                    document_path=document_path,
                    details={
                        "age": age,
                        "max_age": max_age,
                        "strategy": cache_strategy
                    }
                )
                
            return True
                
        except DocumentCacheError:
            raise
        except Exception as e:
            logger.warning(f"Error validating cache entry: {str(e)}")
            return False

    def _initialize_metadata_services(self) -> None:
        """Initialize metadata services for document lineage and governance."""
        try:
            # Initialize metadata tracking
            self.metadata_services = {
                "lineage": {
                    "enabled": True,
                    "version_tracking": True,
                    "source_tracking": True
                },
                "governance": {
                    "enabled": True,
                    "compliance_tracking": True,
                    "audit_logging": True
                },
                "pipeline_status": {
                    "enabled": True,
                    "status_updates": True,
                    "notifications": True
                }
            }
            
            # Initialize SNS topic for status updates if configured
            self.status_topic = getattr(settings, 'DOCUMENT_PROCESSING_STATUS_TOPIC', None)
            
            # Initialize lineage tracking
            self.document_registry = {}  # Track document versions and relationships
            self.processing_lineage = {}  # Track processing steps and dependencies
            
        except Exception as e:
            logger.error(f"Error initializing metadata services: {str(e)}")
            raise DocumentProcessingError(f"Metadata services initialization failed: {str(e)}")

    def _track_document_lineage(
        self,
        document_path: DocumentPath,
        processing_result: ProcessingResult,
        metadata_context: ProcessingMetadataContext
    ) -> None:
        """
        Track document lineage including versions, transformations, and relationships.
        
        Args:
            document_path: Original document path
            processing_result: Processing result containing outputs
            metadata_context: Processing metadata context
        """
        try:
            # Generate document identifier
            doc_id = self._get_file_hash(document_path)
            
            # Record document registration
            self.document_registry[doc_id] = {
                "original_path": document_path,
                "registration_time": datetime.datetime.now().isoformat(),
                "document_type": metadata_context.get_analysis_result("document_type"),
                "versions": [],
                "transformations": [],
                "relationships": []
            }
            
            # Track versions if document was modified
            if "processed_file_path" in processing_result:
                version_id = self._get_file_hash(processing_result["processed_file_path"])
                self.document_registry[doc_id]["versions"].append({
                    "version_id": version_id,
                    "file_path": processing_result["processed_file_path"],
                    "created_at": datetime.datetime.now().isoformat(),
                    "processing_phase": "preprocessing"
                })
            
            # Track transformations
            for phase in ["preprocessing", "processing", "chunking"]:
                if f"{phase}_metadata" in processing_result:
                    self.document_registry[doc_id]["transformations"].append({
                        "phase": phase,
                        "metadata": processing_result[f"{phase}_metadata"],
                        "timestamp": datetime.datetime.now().isoformat()
                    })
            
            # Track relationships (e.g., chunks, extracted components)
            if "chunking_metadata" in processing_result:
                for chunk in processing_result["chunking_metadata"].get("chunks", []):
                    chunk_id = self._get_file_hash(chunk["path"])
                    self.document_registry[doc_id]["relationships"].append({
                        "type": "chunk",
                        "target_id": chunk_id,
                        "metadata": chunk
                    })
            
            # Publish status update if configured
            if self.status_topic:
                self._publish_status_update(doc_id, "LINEAGE_UPDATED", {
                    "document_id": doc_id,
                    "status": "success",
                    "lineage_metadata": self.document_registry[doc_id]
                })
                
        except Exception as e:
            logger.error(f"Error tracking document lineage: {str(e)}")
            metadata_context.record_error(
                component="DocumentProcessingManager",
                message=f"Lineage tracking failed: {str(e)}",
                error_type=type(e).__name__
            )

    def _publish_status_update(
        self,
        document_id: str,
        status: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Publish processing status updates to configured notification service.
        
        Args:
            document_id: Document identifier
            status: Status message
            metadata: Additional metadata to include
        """
        try:
            if not self.status_topic:
                return
                
            message = {
                "document_id": document_id,
                "status": status,
                "timestamp": datetime.datetime.now().isoformat(),
                "metadata": metadata
            }
            
            # Here you would integrate with your actual notification service
            # For example, AWS SNS, Django signals, or a message queue
            logger.info(f"Status update published: {message}")
            
        except Exception as e:
            logger.warning(f"Error publishing status update: {str(e)}")

    @handle_document_errors
    @handle_vision_errors
    def check_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of the document processing system.
        
        Returns:
            Dict containing health status and metrics
        """
        health_status = {
            "status": "healthy",
            "components": {},
            "resource_metrics": {},
            "processing_metrics": {},
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            # Check core components
            self._check_component_health(health_status)
            
            # Check resource usage
            self._check_resource_health(health_status)
            
            # Check processing metrics
            self._check_processing_health(health_status)
            
            # Check adapter health
            self._check_adapter_health(health_status)
            
            # Overall status determination
            if any(comp["status"] == "critical" for comp in health_status["components"].values()):
                health_status["status"] = "critical"
            elif any(comp["status"] == "warning" for comp in health_status["components"].values()):
                health_status["status"] = "warning"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    def _check_component_health(self, health_status: Dict[str, Any]) -> None:
        """Check health of core processing components."""
        components = {
            "type_detector": self.type_detector,
            "analyzer": self.analyzer,
            "quality_analyzer": self.quality_analyzer,
            "capability_selector": self.capability_selector,
            "chunking_service": self.chunking_service,
            "preprocessing_service": self.preprocessing_service
        }
        
        for name, component in components.items():
            try:
                # Check if component has health check method
                if hasattr(component, 'check_health'):
                    component_health = component.check_health()
                else:
                    # Basic check - verify component is initialized
                    component_health = {
                        "status": "healthy" if component is not None else "critical",
                        "message": "Component initialized" if component is not None else "Component not initialized"
                    }
                
                health_status["components"][name] = component_health
                
            except Exception as e:
                health_status["components"][name] = {
                    "status": "error",
                    "message": str(e)
                }

    def _check_resource_health(self, health_status: Dict[str, Any]) -> None:
        """Check system resource health."""
        try:
            # Get current resource stats
            resource_stats = self._monitor_resource_usage()
            
            # CPU health
            cpu_percent = resource_stats.get("cpu_percent", 0)
            cpu_health = {
                "status": "healthy",
                "current_usage": cpu_percent,
                "threshold": 80
            }
            if cpu_percent > 80:
                cpu_health["status"] = "critical"
            elif cpu_percent > 60:
                cpu_health["status"] = "warning"
            
            # Memory health
            memory_percent = resource_stats.get("memory_percent", 0)
            memory_health = {
                "status": "healthy",
                "current_usage": memory_percent,
                "threshold": 85
            }
            if memory_percent > 85:
                memory_health["status"] = "critical"
            elif memory_percent > 70:
                memory_health["status"] = "warning"
            
            # GPU health if available
            gpu_health = {}
            if "gpu_stats" in resource_stats:
                for gpu in resource_stats["gpu_stats"]:
                    gpu_id = gpu["device_id"]
                    gpu_util = gpu["gpu_utilization"]
                    gpu_mem = gpu["memory_percent"]
                    
                    gpu_health[f"gpu_{gpu_id}"] = {
                        "status": "healthy",
                        "utilization": gpu_util,
                        "memory_usage": gpu_mem,
                        "thresholds": {
                            "utilization": 85,
                            "memory": 85
                        }
                    }
                    
                    if gpu_util > 85 or gpu_mem > 85:
                        gpu_health[f"gpu_{gpu_id}"]["status"] = "critical"
                    elif gpu_util > 70 or gpu_mem > 70:
                        gpu_health[f"gpu_{gpu_id}"]["status"] = "warning"
            
            health_status["resource_metrics"] = {
                "cpu": cpu_health,
                "memory": memory_health,
                "gpu": gpu_health if gpu_health else {"status": "not_available"}
            }
            
        except Exception as e:
            health_status["resource_metrics"] = {
                "status": "error",
                "message": str(e)
            }

    def _check_processing_health(self, health_status: Dict[str, Any]) -> None:
        """Check document processing metrics and performance."""
        try:
            # Calculate processing success rate
            total_docs = self.processing_stats["processed_documents"]
            if total_docs > 0:
                success_rate = (self.processing_stats["successful_processing"] / total_docs) * 100
                avg_processing_time = self.processing_stats["avg_processing_time"]
                
                processing_health = {
                    "status": "healthy",
                    "success_rate": success_rate,
                    "avg_processing_time": avg_processing_time,
                    "total_documents": total_docs,
                    "thresholds": {
                        "min_success_rate": 95,
                        "max_avg_processing_time": 300  # seconds
                    }
                }
                
                # Set status based on thresholds
                if success_rate < 95 or avg_processing_time > 300:
                    processing_health["status"] = "critical"
                elif success_rate < 98 or avg_processing_time > 200:
                    processing_health["status"] = "warning"
                
            else:
                processing_health = {
                    "status": "unknown",
                    "message": "No documents processed yet"
                }
            
            health_status["processing_metrics"] = processing_health
            
        except Exception as e:
            health_status["processing_metrics"] = {
                "status": "error",
                "message": str(e)
            }

    def _check_adapter_health(self, health_status: Dict[str, Any]) -> None:
        """Check health of document processing adapters."""
        try:
            adapter_health = {}
            
            for name, adapter in self.adapters.items():
                # Check adapter capabilities and version
                adapter_info = self.adapter_info[name]
                
                # Calculate adapter success rate
                adapter_stats = self.processing_stats["capability_usage"].get(name, {})
                total_docs = adapter_stats.get("document_count", 0)
                
                if total_docs > 0:
                    success_rate = (adapter_stats.get("success_count", 0) / total_docs) * 100
                    avg_time = adapter_stats.get("total_time", 0) / total_docs
                    
                    status = "healthy"
                    if success_rate < 90:
                        status = "critical"
                    elif success_rate < 95:
                        status = "warning"
                    
                    adapter_health[name] = {
                        "status": status,
                        "success_rate": success_rate,
                        "avg_processing_time": avg_time,
                        "total_documents": total_docs,
                        "version": adapter_info["version"],
                        "capabilities": adapter_info["capabilities"]
                    }
                else:
                    adapter_health[name] = {
                        "status": "unknown",
                        "message": "No documents processed",
                        "version": adapter_info["version"],
                        "capabilities": adapter_info["capabilities"]
                    }
            
            health_status["components"]["adapters"] = adapter_health
            
        except Exception as e:
            health_status["components"]["adapters"] = {
                "status": "error",
                "message": str(e)
            }

    def _initialize_resource_management(self):
        """Initialize resource monitoring and management"""
        self.resource_monitor = ResourceMonitor()
        self.resource_limits = self.config.get("resource_limits", {
            "memory": 0.8,  # 80% max memory usage
            "cpu": 0.9,    # 90% max CPU usage
            "gpu": 0.7     # 70% max GPU usage if available
        })
        self.check_interval = self.config.get("resource_check_interval", 60)
        
    def _initialize_cache_system(self):
        """Initialize caching system with configuration"""
        self.cache_manager = CacheManager(
            enabled=self.config.cache_enabled,
            timeout=self.config.cache_timeout
        )

    def _initialize_monitoring(self):
        """Initialize performance and health monitoring"""
        self.health_monitor = HealthMonitor()
        self.performance_tracker = PerformanceTracker()

    @handle_vision_errors
    @handle_resource_limits
    @document_processing_complete
    def process_document(self, document_path: str, metadata_context: Optional[ProcessingMetadataContext] = None) -> ProcessingResult:
        """Main document processing entry point with resource and cache management"""
        
        if metadata_context is None:
            metadata_context = ProcessingMetadataContext()

        # 1. Check cache first
        if cached_result := self._get_cached_result(document_path, metadata_context):
            metadata_context.record_decision(
                component="DocumentProcessingManager",
                decision="Using cached result",
                reason="Valid cache entry found"
            )
            return cached_result

        # 2. Check resources before processing
        if not self._check_resource_availability():
            raise ResourceLimitExceeded("Insufficient resources for processing")

        try:
            # 3. Execute processing pipeline
            result = self._execute_processing_pipeline(document_path, metadata_context)

            # 4. Cache result if successful
            self._cache_processing_result(document_path, result)

            # 5. Prepare for knowledge graph if needed
            if self.config.prepare_for_kg:
                result = self._prepare_for_knowledge_graph(result, metadata_context)

            # 6. Track performance metrics
            self.performance_tracker.record_processing(
                document_path=document_path,
                processing_time=metadata_context.get_total_time(),
                success=True
            )

            return result

        except Exception as e:
            # Record failure metrics
            self.performance_tracker.record_processing(
                document_path=document_path,
                processing_time=metadata_context.get_total_time(),
                success=False,
                error=str(e)
            )
            raise

        finally:
            # Update resource usage statistics
            self._update_resource_stats()

    def _check_resource_availability(self) -> bool:
        """Check if sufficient resources are available for processing"""
        try:
            resource_stats = self._monitor_resource_usage()
            for resource, limit in self.resource_limits.items():
                if resource_stats.get(resource, 0) > limit:
                    logger.warning(f"Resource limit exceeded for {resource}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error checking resources: {str(e)}")
            return False

    def _get_cached_result(self, document_path: str, metadata_context: ProcessingMetadataContext) -> Optional[ProcessingResult]:
        """Retrieve cached result if available and valid"""
        if not self.config.cache_enabled:
            return None
            
        return self.cache_manager.get_result(document_path)

    def _cache_processing_result(self, document_path: str, result: ProcessingResult):
        """Cache processing result if caching is enabled"""
        if not self.config.cache_enabled:
            return
            
        self.cache_manager.store_result(document_path, result)

    def _prepare_for_knowledge_graph(self, result: ProcessingResult, metadata_context: ProcessingMetadataContext) -> ProcessingResult:
        """Prepare processing results for knowledge graph integration"""
        if not hasattr(self, 'kg_manager'):
            return result
            
        try:
            metadata_context.start_timing("kg_preparation")
            kg_data = self.kg_manager.prepare_document(result)
            result.kg_preparation = kg_data
            metadata_context.end_timing("kg_preparation")
            return result
        except Exception as e:
            logger.warning(f"KG preparation failed: {e}")
            metadata_context.record_error(
                component="DocumentProcessingManager",
                message=f"KG preparation failed: {str(e)}",
                error_type=type(e).__name__
            )
            return result

    def _update_resource_stats(self):
        """Update resource usage statistics after processing"""
        self.resource_monitor.update_stats()
        
    @handle_vision_errors
    def check_health(self) -> Dict[str, Any]:
        """Check health status of the processing manager and its components"""
        return {
            "status": "healthy",
            "cache": self.cache_manager.get_stats(),
            "resources": self.resource_monitor.get_stats(),
            "performance": self.performance_tracker.get_stats()
        }

    def _publish_processing_event(self, event_type: ProcessingEventType, document_path: str, data: Dict[str, Any]) -> None:
        """
        Publish an event through NextLayerInterface.
        
        Args:
            event_type: Type of event to publish
            document_path: Path to the document
            data: Event data
        """
        try:
            next_layer = NextLayerInterface.get_instance()
            next_layer.emit_simple_event(
                event_type=event_type,
                document_id=document_path,
                data={
                    **data,
                    "manager": self.__class__.__name__,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.warning(f"Failed to publish event: {str(e)}")
