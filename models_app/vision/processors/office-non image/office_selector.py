"""
OfficeDocumentSelector: Selects the appropriate adapter for office document types.
"""

import os
import logging
import time
import copy
from datetime import datetime
import mimetypes
from typing import Dict, List, Optional, Any, Type, Tuple

from models_app.vision.document.adapters.document_format_adapter import DocumentFormatAdapter
from models_app.vision.document.utils.error_handling.errors import AdapterError
from models_app.vision.document.utils.core.processing_metadata_context import ProcessingMetadataContext
from models_app.vision.document.utils.core.next_layer_interface import NextLayerInterface, ProcessingEventType
from analytics_app.utils import monitor_selector_performance

logger = logging.getLogger(__name__)

class OfficeDocumentSelector:
    """
    Selects the appropriate office document adapter based on file type and content analysis.
    This allows for a pluggable architecture where document processing
    can be handled by the most appropriate specialized adapter.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the office document selector.
        
        Args:
            config: Configuration for the selector and adapters
        """
        self.config = config or {}
        self.adapters = {}
        self.adapter_capabilities = {}
        self.performance_stats = {}
        # Initialize NextLayerInterface once
        self.next_layer = NextLayerInterface.get_instance()
        self._initialize_adapters()
    
    def _initialize_adapters(self) -> None:
        """Initialize and register all available office document adapters."""
        try:
            # Import adapters here to avoid circular imports
            # Use proper module import syntax for directory with hyphen
            import importlib.util
            import sys
            
            # Define the module paths
            module_paths = {
                "word": "models_app.vision.processors.office-non image.word_document_adapter",
                "excel": "models_app.vision.processors.office-non image.excel_document_adapter",
                "powerpoint": "models_app.vision.processors.office-non image.powerpoint_document_adapter",
                "web": "models_app.vision.processors.office-non image.web_document_adapter",
                "email": "models_app.vision.processors.office-non image.email_document_adapter"
            }
            
            # Load modules dynamically
            adapter_classes = {}
            for name, path in module_paths.items():
                try:
                    # Handle the hyphen in the directory name by using importlib
                    module_name = path.split(".")[-1]
                    spec = importlib.util.find_spec(path.replace("-", "_"))
                    if spec is None:
                        # Try alternative path format
                        alt_path = path.replace("-", "/")
                        spec = importlib.util.find_spec(alt_path)
                        if spec is None:
                            raise ImportError(f"Could not find module {path}")
                    
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Extract adapter class
                    if name == "word":
                        adapter_classes[name] = module.WordDocumentAdapter
                    elif name == "excel":
                        adapter_classes[name] = module.ExcelDocumentAdapter
                    elif name == "powerpoint":
                        adapter_classes[name] = module.PowerPointDocumentAdapter
                    elif name == "web":
                        adapter_classes[name] = module.WebDocumentAdapter
                    elif name == "email":
                        adapter_classes[name] = module.EmailDocumentAdapter
                except Exception as e:
                    logger.error(f"Failed to import {name} adapter: {str(e)}")
                    adapter_classes[name] = None
            
            # Create adapter instances for the ones that were successfully imported
            adapters = {}
            for name, adapter_class in adapter_classes.items():
                if adapter_class:
                    adapters[name] = adapter_class(self.config)
                    logger.info(f"Successfully created {name} adapter")
            
            # Store adapter capabilities for content-based selection
            for name, adapter in adapters.items():
                if adapter:
                    self.adapter_capabilities[name] = {
                        "adapter": adapter,
                        "capabilities": getattr(adapter, "CAPABILITIES", {}),
                        "priority": getattr(adapter, "PRIORITY", 50)
                    }
            
            # Register adapters by document extensions
            if "word" in adapters and adapters["word"]:
                self.register_adapter(['.doc', '.docx', '.odt', '.rtf'], adapters["word"])
            if "excel" in adapters and adapters["excel"]:
                self.register_adapter(['.xls', '.xlsx', '.ods', '.csv'], adapters["excel"])
            if "powerpoint" in adapters and adapters["powerpoint"]:
                self.register_adapter(['.ppt', '.pptx', '.odp'], adapters["powerpoint"])
            if "web" in adapters and adapters["web"]:
                self.register_adapter(['.html', '.htm', '.mht', '.mhtml'], adapters["web"])
            if "email" in adapters and adapters["email"]:
                self.register_adapter(['.eml', '.msg'], adapters["email"])
            
            # Initialize performance tracking
            for adapter_name in self.adapter_capabilities:
                self.performance_stats[adapter_name] = {
                    "total_processed": 0,
                    "successes": 0,
                    "failures": 0,
                    "avg_processing_time": 0.0,
                    "last_used": None
                }
            
            logger.info(f"Successfully initialized {len(set(self.adapters.values()))} office document adapters")
        except Exception as e:
            logger.error(f"Error initializing office document adapters: {str(e)}")
            raise AdapterError(f"Failed to initialize office document adapters: {str(e)}")
    
    def register_adapter(self, extensions: List[str], adapter: DocumentFormatAdapter) -> None:
        """
        Register an adapter for the specified file extensions.
        
        Args:
            extensions: List of file extensions this adapter supports
            adapter: The adapter instance to register
        """
        for ext in extensions:
            self.adapters[ext.lower()] = adapter
            logger.debug(f"Registered adapter {adapter.__class__.__name__} for extension {ext}")
    
    @monitor_selector_performance
    def get_adapter(self, file_path: str, metadata_context: Optional[ProcessingMetadataContext] = None) -> Optional[DocumentFormatAdapter]:
        """
        Get the appropriate adapter for the given file.
        Uses file extension as primary selector, but falls back to content
        analysis for ambiguous formats.
        
        Args:
            file_path: Path to the document file
            metadata_context: Optional metadata context for tracking decisions
            
        Returns:
            DocumentFormatAdapter: The appropriate adapter, or None if not supported
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        start_time = time.time()
        
        # Emit start event
        self.next_layer.emit_simple_event(
            event_type=ProcessingEventType.PROCESSING_PHASE_START,
            document_id=file_path,
            data={
                "component": self.__class__.__name__,
                "phase": "office_adapter_selection",
                "file_extension": ext
            }
        )
        
        # Record decision point in metadata context
        if metadata_context:
            metadata_context.start_timing("office_adapter_selection")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            
            # Emit error event
            self.next_layer.emit_simple_event(
                event_type=ProcessingEventType.ERROR_OCCURRED,
                document_id=file_path,
                data={
                    "component": self.__class__.__name__,
                    "phase": "office_adapter_selection",
                    "error": "File not found",
                    "error_type": "FileNotFoundError"
                }
            )
            
            if metadata_context:
                metadata_context.record_decision(
                    component="OfficeDocumentSelector.get_adapter",
                    decision="Failed to select adapter",
                    reason="File not found",
                    confidence=1.0
                )
                metadata_context.end_timing("office_adapter_selection")
            return None
        
        adapter = self.adapters.get(ext)
        
        # For ambiguous or unknown formats, perform content analysis
        if not adapter or ext in ['.xml', '.txt', '.json']:
            logger.info(f"Using content analysis to select adapter for {ext} file")
            adapter, confidence, reason = self._select_by_content(file_path, ext)
            
            if metadata_context and adapter:
                adapter_name = adapter.__class__.__name__
                metadata_context.record_decision(
                    component="OfficeDocumentSelector.get_adapter",
                    decision=f"Selected {adapter_name} by content analysis",
                    reason=reason,
                    confidence=confidence
                )
        
        if adapter:
            adapter_name = adapter.__class__.__name__
            logger.info(f"Selected adapter {adapter_name} for file {file_path}")
            
            # Emit selection event
            self.next_layer.emit_simple_event(
                event_type=ProcessingEventType.PROCESSING_DECISION,
                document_id=file_path,
                data={
                    "component": self.__class__.__name__,
                    "decision": f"Selected {adapter_name}",
                    "file_extension": ext,
                    "selection_time": time.time() - start_time
                }
            )
            
            # Record in metadata context
            if metadata_context:
                metadata_context.record_adapter_selection(
                    component="OfficeDocumentSelector",
                    selected_adapter=adapter_name,
                    file_path=file_path,
                    file_extension=ext
                )
            
            # Update performance stats (will be completed after processing)
            self._start_performance_tracking(adapter_name, file_path)
            
            # Emit completion event
            self.next_layer.emit_simple_event(
                event_type=ProcessingEventType.PROCESSING_PHASE_END,
                document_id=file_path,
                data={
                    "component": self.__class__.__name__,
                    "phase": "office_adapter_selection",
                    "success": True,
                    "selection_time": time.time() - start_time
                }
            )
            
            if metadata_context:
                metadata_context.end_timing("office_adapter_selection")
            
            return adapter
        
        logger.warning(f"No office adapter found for extension {ext} (file: {file_path})")
        
        # Emit failure event
        self.next_layer.emit_simple_event(
            event_type=ProcessingEventType.ERROR_OCCURRED,
            document_id=file_path,
            data={
                "component": self.__class__.__name__,
                "phase": "office_adapter_selection",
                "error": f"No adapter found for extension {ext}",
                "error_type": "UnsupportedFormatError"
            }
        )
        
        if metadata_context:
            metadata_context.record_decision(
                component="OfficeDocumentSelector.get_adapter",
                decision="Failed to select adapter",
                reason=f"Unsupported format: {ext}",
                confidence=1.0
            )
            metadata_context.end_timing("office_adapter_selection")
        
        return None
    
    def _select_by_content(self, file_path: str, ext: str) -> Tuple[Optional[DocumentFormatAdapter], float, str]:
        """
        Analyzes file content to determine the best adapter when extension is ambiguous.
        
        Args:
            file_path: Path to the document file
            ext: File extension
            
        Returns:
            Tuple of (selected adapter, confidence, reason)
        """
        # Get MIME type for initial classification
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Default adapter and confidence
        adapter = None
        confidence = 0.0
        reason = "Unknown format"
        
        try:
            # For XML files, check if it's actually an Office XML format
            if ext == '.xml':
                with open(file_path, 'rb') as f:
                    content = f.read(4096)  # Read first 4KB
                    content_str = content.decode('utf-8', errors='ignore')
                    
                    # Check for Office Open XML signatures
                    if 'word/document.xml' in content_str or '<w:document' in content_str:
                        adapter = self._get_adapter_by_name("word")
                        confidence = 0.9
                        reason = "Detected Word XML signature"
                    elif 'xl/workbook.xml' in content_str or '<workbook' in content_str:
                        adapter = self._get_adapter_by_name("excel")
                        confidence = 0.9
                        reason = "Detected Excel XML signature"
                    elif 'ppt/presentation.xml' in content_str or '<p:presentation' in content_str:
                        adapter = self._get_adapter_by_name("powerpoint")
                        confidence = 0.9
                        reason = "Detected PowerPoint XML signature"
                    elif '<html' in content_str.lower():
                        adapter = self._get_adapter_by_name("web")
                        confidence = 0.8
                        reason = "Detected HTML content in XML file"
            
            # For text files, try to determine structure
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(8192)  # Read first 8KB
                    
                    # Count commas and tabs to identify potential CSV
                    comma_count = content.count(',')
                    tab_count = content.count('\t')
                    newline_count = content.count('\n')
                    
                    # Check for CSV structure
                    if (comma_count > 5 and newline_count > 1 and 
                        comma_count / max(1, newline_count) > 3):
                        adapter = self._get_adapter_by_name("excel")
                        confidence = 0.7
                        reason = "Detected CSV-like structure"
                    
                    # Check for tab-delimited structure
                    elif (tab_count > 5 and newline_count > 1 and 
                          tab_count / max(1, newline_count) > 2):
                        adapter = self._get_adapter_by_name("excel")
                        confidence = 0.7
                        reason = "Detected tab-delimited structure"
                    
                    # Check for email-like structure
                    elif 'From:' in content and 'To:' in content and 'Subject:' in content:
                        adapter = self._get_adapter_by_name("email")
                        confidence = 0.8
                        reason = "Detected email headers"
                    
                    # Check for HTML content
                    elif '<html' in content.lower() or '<body' in content.lower():
                        adapter = self._get_adapter_by_name("web")
                        confidence = 0.7
                        reason = "Detected HTML tags"
                    
                    # Default to Word for plain text
                    else:
                        adapter = self._get_adapter_by_name("word")
                        confidence = 0.4
                        reason = "Assuming plain text document"
            
            # For JSON files, check the structure
            elif ext == '.json':
                import json
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        data = json.load(f)
                        
                        # Check for tabular data
                        if isinstance(data, list) and all(isinstance(item, dict) for item in data[:10]):
                            adapter = self._get_adapter_by_name("excel")
                            confidence = 0.7
                            reason = "Detected tabular JSON data"
                        # Check for document-like JSON
                        elif isinstance(data, dict) and any(key in data for key in ['content', 'text', 'body', 'paragraphs']):
                            adapter = self._get_adapter_by_name("word")
                            confidence = 0.6
                            reason = "Detected document-like JSON structure"
                except json.JSONDecodeError:
                    # Not valid JSON, use default
                    adapter = self._get_adapter_by_name("word")
                    confidence = 0.3
                    reason = "Invalid JSON, treating as text"
            
            # Use fallback based on MIME type if no adapter selected yet
            if not adapter and mime_type:
                if 'text/html' in mime_type:
                    adapter = self._get_adapter_by_name("web")
                    confidence = 0.6
                    reason = f"Selected based on MIME type: {mime_type}"
                elif 'text/plain' in mime_type:
                    adapter = self._get_adapter_by_name("word")
                    confidence = 0.5
                    reason = f"Selected based on MIME type: {mime_type}"
                elif 'application/json' in mime_type:
                    adapter = self._get_adapter_by_name("excel")
                    confidence = 0.5
                    reason = f"Selected based on MIME type: {mime_type}"
                elif 'message/rfc822' in mime_type:
                    adapter = self._get_adapter_by_name("email")
                    confidence = 0.8
                    reason = f"Selected based on MIME type: {mime_type}"
        
        except Exception as e:
            logger.warning(f"Error during content analysis: {str(e)}")
            # If content analysis fails, use adapter with highest priority for this extension
            adapter = self._get_default_adapter_for_extension(ext)
            confidence = 0.3
            reason = f"Content analysis failed, using default: {str(e)}"
        
        # If we still don't have an adapter, try the word adapter as fallback
        if not adapter:
            adapter = self._get_adapter_by_name("word")
            confidence = 0.2
            reason = "No suitable adapter found, using Word adapter as fallback"
        
        return adapter, confidence, reason
    
    def _get_adapter_by_name(self, adapter_name: str) -> Optional[DocumentFormatAdapter]:
        """Get adapter instance by name"""
        if adapter_name in self.adapter_capabilities:
            return self.adapter_capabilities[adapter_name]["adapter"]
        return None
    
    def _get_default_adapter_for_extension(self, ext: str) -> Optional[DocumentFormatAdapter]:
        """Get default adapter based on file extension"""
        # Extension to adapter name mapping
        ext_mapping = {
            '.txt': 'word',
            '.json': 'excel',
            '.xml': 'word',
            '.csv': 'excel',
            '.html': 'web',
            '.htm': 'web'
        }
        
        adapter_name = ext_mapping.get(ext, 'word')
        return self._get_adapter_by_name(adapter_name)
    
    def can_process(self, file_path: str) -> bool:
        """
        Check if any registered adapter can process this file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            bool: True if an adapter is available, False otherwise
        """
        return self.get_adapter(file_path) is not None
    
    def get_supported_formats(self) -> List[str]:
        """
        Get all supported file formats across all registered adapters.
        
        Returns:
            List[str]: List of supported file extensions
        """
        return list(self.adapters.keys())
    
    def get_adapters_info(self) -> Dict[str, Any]:
        """
        Get information about all registered adapters.
        
        Returns:
            Dict: Information about registered adapters
        """
        result = {
            "total_adapters": len(set(self.adapters.values())),
            "supported_formats": self.get_supported_formats(),
            "adapters": {},
            "performance_stats": self.performance_stats
        }
        
        # Group by adapter class
        for adapter in set(self.adapters.values()):
            adapter_name = adapter.__class__.__name__
            result["adapters"][adapter_name] = adapter.get_processor_info()
        
        return result
    
    def _start_performance_tracking(self, adapter_name: str, file_path: str) -> None:
        """Start tracking performance for an adapter processing a file"""
        # Extract short adapter name from class name
        short_name = None
        for name in self.adapter_capabilities:
            if adapter_name.lower() in self.adapter_capabilities[name]["adapter"].__class__.__name__.lower():
                short_name = name
                break
        
        if not short_name:
            return
            
        stats = self.performance_stats.get(short_name)
        if not stats:
            return
            
        stats["last_used"] = datetime.now().isoformat()
        stats["current_file"] = file_path
        stats["start_time"] = time.time()
    
    def track_processing_result(self, adapter_name: str, processing_time: float, success: bool) -> None:
        """
        Track processing result for performance optimization.
        
        Args:
            adapter_name: Name of the adapter
            processing_time: Time taken to process the document in seconds
            success: Whether processing was successful
        """
        # Extract short adapter name from class name
        short_name = None
        for name in self.adapter_capabilities:
            if adapter_name.lower() in name.lower() or adapter_name.lower() in self.adapter_capabilities[name]["adapter"].__class__.__name__.lower():
                short_name = name
                break
        
        if not short_name:
            return
            
        stats = self.performance_stats.get(short_name)
        if not stats:
            return
            
        stats["total_processed"] += 1
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        
        # Update average processing time using a weighted average
        if stats["total_processed"] > 1:
            stats["avg_processing_time"] = (stats["avg_processing_time"] * (stats["total_processed"] - 1) + 
                                          processing_time) / stats["total_processed"]
        else:
            stats["avg_processing_time"] = processing_time
        
        # Clear the current file
        if "current_file" in stats:
            del stats["current_file"]
        if "start_time" in stats:
            del stats["start_time"]
    
    def merge_metadata(self, existing_metadata: Dict[str, Any], adapter_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge adapter-specific metadata with existing metadata without overwriting.
        
        Args:
            existing_metadata: Existing metadata dictionary
            adapter_metadata: Adapter-specific metadata to merge
            
        Returns:
            Dict[str, Any]: Merged metadata
        """
        if not existing_metadata:
            return adapter_metadata
        
        # Create deep copy to avoid modifying original
        result = copy.deepcopy(existing_metadata)
        
        # Only add adapter-specific keys that don't already exist
        for key, value in adapter_metadata.items():
            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                # Recursively merge nested dictionaries
                result[key] = self.merge_metadata(result[key], value)
            # Skip if the key already exists (don't overwrite)
        
        return result

# Create singleton instance
_office_selector_instance = None

def get_office_selector_instance(config: Dict[str, Any] = None) -> OfficeDocumentSelector:
    """Get singleton instance of OfficeDocumentSelector"""
    global _office_selector_instance
    if _office_selector_instance is None:
        _office_selector_instance = OfficeDocumentSelector(config)
    return _office_selector_instance
