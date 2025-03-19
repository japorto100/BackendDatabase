"""
ProcessingMetadataContext: Management of metadata during document processing.

This context tracks decisions and metadata throughout the entire
document processing pipeline to ensure consistency and traceability.
"""

import time
import logging
import uuid
import json
import os
import platform
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from .pipeline_phases import PipelinePhaseManager, PipelinePhase

logger = logging.getLogger(__name__)

class ProcessingMetadataContext:
    """
    Manages metadata and decisions during the document processing process.
    
    This class provides a central location for tracking metadata,
    decisions, and performance data throughout the processing pipeline.
    It ensures that information is consistently passed between components
    and enables detailed tracking of processing steps.
    
    Version 2.0: Enhanced with improved versioning, capability tracking,
    and standardized decision recording.
    """
    
    # Context schema version
    SCHEMA_VERSION = "2.0.0"
    
    def __init__(self, document_path: str, context_id: Optional[str] = None,
                 parent_context: Optional['ProcessingMetadataContext'] = None):
        """
        Initialize the ProcessingMetadataContext.
        
        Args:
            document_path: Path to the document being processed
            context_id: Optional ID for the context (generated if not provided)
            parent_context: Optional parent context for hierarchical processing
        """
        self.context_id = context_id or str(uuid.uuid4())
        self.document_path = document_path
        self.creation_time = self._get_timestamp()
        
        # Pipeline phase management
        self.phase_manager = PipelinePhaseManager()
        
        # Context hierarchy
        self.parent_context = parent_context
        self.child_contexts: List['ProcessingMetadataContext'] = []
        
        # Transaction support
        self._transaction_stack: List[Dict[str, Any]] = []
        
        # Basic document metadata
        self.document_metadata = {
            "path": document_path,
            "filename": os.path.basename(document_path),
            "processing_id": self.context_id
        }
        
        # Add file metadata
        try:
            file_stats = os.stat(document_path)
            self.document_metadata.update({
                "size_bytes": file_stats.st_size,
                "last_modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            })
        except Exception as e:
            logger.warning(f"Could not get file stats for {document_path}: {str(e)}")
        
        # Pipeline decisions
        self.decisions = []
        
        # Performance tracking
        self.performance_metrics = {
            "system_info": self._get_system_info(),
            "processing_start": self._get_timestamp()
        }
        self.step_timings = {}
        self.start_time = time.time()
        
        # Router/selector decisions
        self.routing_decisions = {}
        
        # Analysis results with versioning
        self.analysis_results = {}
        
        # Document processing flow tracking
        self.processing_flow = []
        
        # Preprocessing information
        self.preprocessing_info = {}
        
        # Adapter-specific data
        self.adapter_data = {}
        
        # Warnings and errors
        self.warnings = []
        self.errors = []
        
        # Capability requirements and matches
        self.required_capabilities = {}
        self.capability_matches = {}
        
        # Component versions
        self.component_versions = {}
        
        # Start initialization phase
        self.phase_manager.start_phase(PipelinePhase.INITIALIZATION)
        
        # Record context creation in flow
        self._record_flow_step("context_created", "ProcessingMetadataContext", 
                             {"schema_version": self.SCHEMA_VERSION})
        
        logger.debug(f"New ProcessingMetadataContext created: {self.context_id} for document: {document_path}")
        
        # End initialization phase
        self.phase_manager.end_phase(PipelinePhase.INITIALIZATION)
    
    def create_child_context(self, phase: Optional[PipelinePhase] = None) -> 'ProcessingMetadataContext':
        """
        Create a child context for a processing phase.
        
        Args:
            phase: Optional pipeline phase for the child context
            
        Returns:
            New ProcessingMetadataContext instance
        """
        child = ProcessingMetadataContext(
            self.document_path,
            context_id=f"{self.context_id}_child_{len(self.child_contexts)}",
            parent_context=self
        )
        
        self.child_contexts.append(child)
        
        if phase:
            child.phase_manager.start_phase(phase)
            
        return child
    
    def merge_child_context(self, child: 'ProcessingMetadataContext') -> None:
        """
        Merge a child context back into the parent.
        
        Args:
            child: Child context to merge
        """
        if child not in self.child_contexts:
            logger.warning("Attempting to merge unknown child context")
            return
            
        # Merge decisions
        self.decisions.extend(child.decisions)
        
        # Merge analysis results
        self.analysis_results.update(child.analysis_results)
        
        # Merge processing flow
        self.processing_flow.extend(child.processing_flow)
        
        # Merge warnings and errors
        self.warnings.extend(child.warnings)
        self.errors.extend(child.errors)
        
        # Merge capabilities
        if child.required_capabilities:
            self.required_capabilities.update(child.required_capabilities)
        
        # Merge adapter data
        for adapter, data in child.adapter_data.items():
            if adapter not in self.adapter_data:
                self.adapter_data[adapter] = {}
            self.adapter_data[adapter].update(data)
            
        logger.debug(f"Merged child context {child.context_id} into {self.context_id}")
    
    def start_transaction(self) -> None:
        """Start a new transaction to track changes."""
        state = {
            "decisions": self.decisions.copy(),
            "analysis_results": self.analysis_results.copy(),
            "processing_flow": self.processing_flow.copy(),
            "warnings": self.warnings.copy(),
            "errors": self.errors.copy(),
            "required_capabilities": self.required_capabilities.copy(),
            "adapter_data": self.adapter_data.copy()
        }
        self._transaction_stack.append(state)
        self.phase_manager.start_transaction()
        logger.debug("Started metadata transaction")
    
    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        if self._transaction_stack:
            self._transaction_stack.pop()
            self.phase_manager.commit_transaction()
            logger.debug("Committed metadata transaction")
    
    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        if self._transaction_stack:
            state = self._transaction_stack.pop()
            self.decisions = state["decisions"]
            self.analysis_results = state["analysis_results"]
            self.processing_flow = state["processing_flow"]
            self.warnings = state["warnings"]
            self.errors = state["errors"]
            self.required_capabilities = state["required_capabilities"]
            self.adapter_data = state["adapter_data"]
            self.phase_manager.rollback_transaction()
            logger.debug("Rolled back metadata transaction")
    
    def get_current_phase(self) -> Optional[PipelinePhase]:
        """Get the currently active pipeline phase."""
        return self.phase_manager.get_current_phase()
    
    def get_phase_status(self, phase: PipelinePhase) -> Optional[str]:
        """Get the status of a pipeline phase."""
        return self.phase_manager.get_phase_status(phase)
    
    def get_phase_summary(self, phase: PipelinePhase) -> Optional[Dict[str, Any]]:
        """Get a summary of a phase's execution."""
        return self.phase_manager.get_phase_summary(phase)
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire pipeline execution."""
        return self.phase_manager.get_pipeline_summary()
    
    def _get_timestamp(self) -> str:
        """Get a standardized ISO format timestamp with timezone information."""
        return datetime.now().astimezone().isoformat()
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get basic system information for performance context."""
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "node": platform.node()
        }
    
    def _record_flow_step(self, step_type: str, component: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a step in the processing flow.
        
        Args:
            step_type: Type of step (e.g., 'analysis', 'decision', 'processing')
            component: Component that performed the step
            details: Additional details about the step
        """
        flow_step = {
            "type": step_type,
            "component": component,
            "timestamp": self._get_timestamp(),
            "processing_time": time.time() - self.start_time
        }
        
        if details:
            flow_step["details"] = details
            
        self.processing_flow.append(flow_step)
    
    def record_decision(self, component: str, decision: str, reason: str, 
                      alternatives: Optional[List[Dict[str, Any]]] = None,
                      confidence: Optional[float] = None,
                      decision_id: Optional[str] = None) -> str:
        """
        Record a decision made during processing.
        
        Args:
            component: The component that made the decision
            decision: The decision that was made
            reason: The reason for the decision
            alternatives: Alternative options that were available, with details
            confidence: Confidence value for the decision (0-1)
            decision_id: Optional ID to reference this decision
            
        Returns:
            str: The decision ID
        """
        decision_id = decision_id or str(uuid.uuid4())
        
        decision_record = {
            "id": decision_id,
            "component": component,
            "decision": decision,
            "reason": reason,
            "timestamp": self._get_timestamp(),
            "processing_time": time.time() - self.start_time
        }
        
        if alternatives:
            decision_record["alternatives"] = alternatives
            
        if confidence is not None:
            decision_record["confidence"] = confidence
            
        self.decisions.append(decision_record)
        
        # Record in flow
        self._record_flow_step("decision", component, {
            "decision": decision,
            "reason": reason,
            "decision_id": decision_id,
            "confidence": confidence
        })
        
        logger.debug(f"Decision recorded: {component} -> {decision}")
        return decision_id
    
    def record_adapter_selection(self, adapter_name: str, reason: str, 
                               analysis_factors: Optional[Dict[str, Any]] = None,
                               alternatives: Optional[List[Dict[str, Any]]] = None,
                               confidence: Optional[float] = None) -> None:
        """
        Record adapter selection.
        
        Args:
            adapter_name: Name of the selected adapter
            reason: Reason for selection
            analysis_factors: Factors used for analysis
            alternatives: Alternative adapters that were considered
            confidence: Confidence in this selection (0-1)
        """
        selection = {
            "selected_adapter": adapter_name,
            "reason": reason,
            "timestamp": self._get_timestamp(),
            "processing_time": time.time() - self.start_time
        }
        
        if analysis_factors:
            selection["analysis_factors"] = analysis_factors
            
        if alternatives:
            selection["alternatives"] = alternatives
            
        if confidence is not None:
            selection["confidence"] = confidence
            
        self.routing_decisions["adapter_selection"] = selection
        
        # Record in flow
        self._record_flow_step("adapter_selection", "AdapterSelector", {
            "selected_adapter": adapter_name,
            "reason": reason,
            "confidence": confidence
        })
        
        logger.debug(f"Adapter selected: {adapter_name} because: {reason}")
    
    def record_ocr_selection(self, engine_name: str, reason: str, 
                           confidence: Optional[float] = None,
                           multi_engine: bool = False,
                           all_engines: Optional[List[str]] = None,
                           alternatives: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Record OCR engine selection.
        
        Args:
            engine_name: Name of the selected OCR engine
            reason: Reason for selection
            confidence: Confidence value for selection (0-1)
            multi_engine: Whether multiple engines are used
            all_engines: List of all engines used (if multi_engine=True)
            alternatives: Alternative engines that were considered
        """
        selection = {
            "selected_engine": engine_name,
            "reason": reason,
            "multi_engine": multi_engine,
            "timestamp": self._get_timestamp(),
            "processing_time": time.time() - self.start_time
        }
        
        if confidence is not None:
            selection["confidence"] = confidence
            
        if multi_engine and all_engines:
            selection["all_engines"] = all_engines
            
        if alternatives:
            selection["alternatives"] = alternatives
            
        self.routing_decisions["ocr_selection"] = selection
        
        # Record in flow
        self._record_flow_step("ocr_selection", "OCRSelector", {
            "selected_engine": engine_name,
            "reason": reason,
            "confidence": confidence,
            "multi_engine": multi_engine
        })
        
        logger.debug(f"OCR engine selected: {engine_name} because: {reason}")
    
    def record_component_version(self, component_name: str, version: str) -> None:
        """
        Record the version of a component.
        
        Args:
            component_name: Name of the component
            version: Version string
        """
        self.component_versions[component_name] = version
        logger.debug(f"Component version recorded: {component_name} -> {version}")
    
    def record_analysis_result(self, analysis_type: str, result: Any, 
                             version: Optional[str] = None,
                             component: Optional[str] = None) -> None:
        """
        Record the result of a document analysis step.
        
        Args:
            analysis_type: Type of analysis performed
            result: Analysis result data
            version: Optional version of the analysis method
            component: Optional component that performed the analysis
        """
        analysis_record = {
            "result": result,
            "timestamp": self._get_timestamp(),
            "processing_time": time.time() - self.start_time
        }
        
        if version:
            analysis_record["version"] = version
            
        if component:
            analysis_record["component"] = component
            
        self.analysis_results[analysis_type] = analysis_record
        
        # Record in flow
        self._record_flow_step("analysis", component or "DocumentAnalyzer", {
            "analysis_type": analysis_type,
            "version": version
        })
        
        logger.debug(f"Analysis result recorded: {analysis_type}")
    
    def record_preprocessing_step(self, method: str, parameters: Optional[Dict[str, Any]] = None,
                                before_image_path: Optional[str] = None, 
                                after_image_path: Optional[str] = None,
                                component: Optional[str] = None) -> None:
        """
        Record a preprocessing step applied to the document.
        
        Args:
            method: Name of the preprocessing method
            parameters: Parameters used for preprocessing
            before_image_path: Optional path to image before preprocessing
            after_image_path: Optional path to image after preprocessing
            component: Optional component that performed preprocessing
        """
        step = {
            "method": method,
            "timestamp": self._get_timestamp(),
            "processing_time": time.time() - self.start_time
        }
        
        if parameters:
            step["parameters"] = parameters
            
        if before_image_path:
            step["before_image"] = before_image_path
            
        if after_image_path:
            step["after_image"] = after_image_path
            
        if component:
            step["component"] = component
            
        if method not in self.preprocessing_info:
            self.preprocessing_info[method] = []
            
        self.preprocessing_info[method].append(step)
        
        # Record in flow
        self._record_flow_step("preprocessing", component or "Preprocessor", {
            "method": method,
            "parameters": parameters
        })
        
        logger.debug(f"Preprocessing step recorded: {method}")
    
    def record_processor_performance(self, processor_name: str, document_type: str,
                                   success: bool, processing_time: float,
                                   metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Record performance metrics for a document processor.
        
        Args:
            processor_name: Name of the processor
            document_type: Type of document processed
            success: Whether processing was successful
            processing_time: Time taken for processing
            metrics: Additional performance metrics
        """
        performance = {
            "processor": processor_name,
            "document_type": document_type,
            "success": success,
            "processing_time": processing_time,
            "timestamp": self._get_timestamp()
        }
        
        if metrics:
            performance["metrics"] = metrics
            
        if "processors" not in self.performance_metrics:
            self.performance_metrics["processors"] = []
            
        self.performance_metrics["processors"].append(performance)
        
        # Record in flow
        self._record_flow_step("performance", processor_name, {
            "success": success,
            "processing_time": processing_time
        })
        
        logger.debug(f"Processor performance recorded: {processor_name}")
    
    def record_capability_requirements(self, component: str, capabilities: Dict[str, float]) -> None:
        """
        Record required capabilities for document processing.
        
        Args:
            component: Component that determined requirements
            capabilities: Required capabilities with confidence values
        """
        self.required_capabilities = {
            "component": component,
            "capabilities": capabilities,
            "timestamp": self._get_timestamp()
        }
        
        # Record in flow
        self._record_flow_step("requirements", component, {
            "capabilities": capabilities
        })
        
        logger.debug(f"Capability requirements recorded by {component}")
    
    def record_capability_match(self, adapter_name: str, capabilities: Dict[str, float],
                              match_score: float, required_capabilities: Dict[str, float]) -> None:
        """
        Record capability matching results for an adapter.
        
        Args:
            adapter_name: Name of the adapter
            capabilities: Adapter's capabilities
            match_score: Overall match score
            required_capabilities: Required capabilities
        """
        match = {
            "adapter": adapter_name,
            "capabilities": capabilities,
            "match_score": match_score,
            "required_capabilities": required_capabilities,
            "timestamp": self._get_timestamp()
        }
        
        if adapter_name not in self.capability_matches:
            self.capability_matches[adapter_name] = []
            
        self.capability_matches[adapter_name].append(match)
        
        # Record in flow
        self._record_flow_step("capability_match", "CapabilityMatcher", {
            "adapter": adapter_name,
            "match_score": match_score
        })
        
        logger.debug(f"Capability match recorded for {adapter_name}: {match_score:.2f}")
    
    def start_timing(self, step_name: str) -> None:
        """
        Start timing a processing step.
        
        Args:
            step_name: Name of the step to time
        """
        if step_name in self.step_timings:
            logger.warning(f"Timing already started for step: {step_name}")
            return
            
        self.step_timings[step_name] = {
            "start": time.time(),
            "complete": False
        }
        
        logger.debug(f"Started timing for step: {step_name}")
    
    def end_timing(self, step_name: str) -> float:
        """
        End timing a processing step and return duration.
        
        Args:
            step_name: Name of the step
            
        Returns:
            float: Duration in seconds, or -1 if timing wasn't started
        """
        if step_name not in self.step_timings:
            logger.warning(f"No timing found for step: {step_name}")
            return -1
            
        if self.step_timings[step_name]["complete"]:
            logger.warning(f"Timing already completed for step: {step_name}")
            return self.step_timings[step_name]["duration"]
            
        end_time = time.time()
        duration = end_time - self.step_timings[step_name]["start"]
        
        self.step_timings[step_name].update({
            "end": end_time,
            "duration": duration,
            "complete": True
        })
        
        # Record in flow
        self._record_flow_step("timing", step_name, {
            "duration": duration
        })
        
        logger.debug(f"Completed timing for step: {step_name} ({duration:.4f}s)")
        return duration
    
    def record_warning(self, component: str, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a warning during processing.
        
        Args:
            component: Component that generated the warning
            message: Warning message
            details: Additional warning details
        """
        warning = {
            "component": component,
            "message": message,
            "timestamp": self._get_timestamp(),
            "processing_time": time.time() - self.start_time
        }
        
        if details:
            warning["details"] = details
            
        self.warnings.append(warning)
        
        # Record in flow
        self._record_flow_step("warning", component, {
            "message": message
        })
        
        logger.warning(f"Warning from {component}: {message}")
    
    def record_error(self, component: str, message: str, 
                   error_type: Optional[str] = None,
                   details: Optional[Dict[str, Any]] = None,
                   is_fatal: bool = False) -> None:
        """
        Record an error during processing.
        
        Args:
            component: Component where error occurred
            message: Error message
            error_type: Type of error
            details: Additional error details
            is_fatal: Whether this is a fatal error
        """
        error = {
            "component": component,
            "message": message,
            "is_fatal": is_fatal,
            "timestamp": self._get_timestamp(),
            "processing_time": time.time() - self.start_time
        }
        
        if error_type:
            error["type"] = error_type
            
        if details:
            error["details"] = details
            
        self.errors.append(error)
        
        # Record in flow
        self._record_flow_step("error", component, {
            "message": message,
            "type": error_type,
            "is_fatal": is_fatal
        })
        
        if is_fatal:
            logger.error(f"Fatal error in {component}: {message}")
        else:
            logger.error(f"Error in {component}: {message}")
    
    def add_document_metadata(self, key: str, value: Any) -> None:
        """
        Add additional metadata about the document.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.document_metadata[key] = value
        logger.debug(f"Added document metadata: {key}")
    
    def add_adapter_data(self, adapter_name: str, key: str, value: Any) -> None:
        """
        Add adapter-specific data.
        
        Args:
            adapter_name: Name of the adapter
            key: Data key
            value: Data value
        """
        if adapter_name not in self.adapter_data:
            self.adapter_data[adapter_name] = {}
            
        self.adapter_data[adapter_name][key] = value
        logger.debug(f"Added adapter data for {adapter_name}: {key}")
    
    def has_analysis_result(self, analysis_type: str) -> bool:
        """
        Check if an analysis result exists.
        
        Args:
            analysis_type: Type of analysis to check
            
        Returns:
            bool: Whether the analysis result exists
        """
        return analysis_type in self.analysis_results
    
    def get_analysis_result(self, analysis_type: str, default: Any = None) -> Any:
        """
        Get an analysis result.
        
        Args:
            analysis_type: Type of analysis to get
            default: Default value if not found
            
        Returns:
            Analysis result or default value
        """
        if analysis_type not in self.analysis_results:
            return default
            
        return self.analysis_results[analysis_type]["result"]
    
    def get_document_type(self) -> Tuple[Optional[str], float]:
        """
        Get the determined document type and confidence.
        
        Returns:
            Tuple of (document_type, confidence)
        """
        if "document_type" not in self.analysis_results:
            return None, 0.0
            
        result = self.analysis_results["document_type"]["result"]
        return (
            result.get("type"),
            result.get("confidence", 0.0)
        )
    
    def finalize_context(self) -> Dict[str, Any]:
        """
        Finalize the context and return complete metadata.
        
        Returns:
            Dict containing all context data
        """
        # Add end time and duration
        end_time = time.time()
        self.performance_metrics.update({
            "processing_end": self._get_timestamp(),
            "total_duration": end_time - self.start_time
        })
        
        # Compile complete metadata
        metadata = {
            "context_id": self.context_id,
            "schema_version": self.SCHEMA_VERSION,
            "document": self.document_metadata,
            "processing": {
                "start_time": self.creation_time,
                "end_time": self._get_timestamp(),
                "duration": end_time - self.start_time,
                "flow": self.processing_flow
            },
            "decisions": self.decisions,
            "routing": self.routing_decisions,
            "analysis": self.analysis_results,
            "preprocessing": self.preprocessing_info,
            "performance": self.performance_metrics,
            "timings": self.step_timings,
            "capabilities": {
                "required": self.required_capabilities,
                "matches": self.capability_matches
            },
            "components": self.component_versions,
            "adapter_data": self.adapter_data,
            "warnings": self.warnings,
            "errors": self.errors
        }
        
        # Record final state
        self._record_flow_step("finalized", "ProcessingMetadataContext", {
            "total_duration": end_time - self.start_time,
            "total_steps": len(self.processing_flow)
        })
        
        return metadata
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get the current processing status.
        
        Returns:
            Dict with current status information
        """
        return {
            "context_id": self.context_id,
            "document": self.document_metadata["filename"],
            "processing_time": time.time() - self.start_time,
            "steps_completed": len(self.processing_flow),
            "current_step": self.processing_flow[-1] if self.processing_flow else None,
            "warnings": len(self.warnings),
            "errors": len(self.errors)
        }
    
    def to_json(self) -> str:
        """
        Convert context to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.finalize_context(), indent=2)
    
    def save_to_file(self, file_path: Optional[str] = None) -> str:
        """
        Save context to a JSON file.
        
        Args:
            file_path: Optional file path (generated if not provided)
            
        Returns:
            str: Path to saved file
        """
        if file_path is None:
            # Generate filename based on document and context ID
            base_name = os.path.splitext(os.path.basename(self.document_path))[0]
            file_path = f"{base_name}_{self.context_id}_metadata.json"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.finalize_context(), f, indent=2)
            
        logger.info(f"Saved metadata context to: {file_path}")
        return file_path
    
    @classmethod
    def from_existing_metadata(cls, metadata: Dict[str, Any], document_path: str) -> 'ProcessingMetadataContext':
        """
        Create context from existing metadata.
        
        Args:
            metadata: Existing metadata dictionary
            document_path: Path to the document
            
        Returns:
            New ProcessingMetadataContext instance
        """
        # Create new instance
        context = cls(document_path, context_id=metadata.get("context_id"))
        
        # Restore metadata
        if "document" in metadata:
            context.document_metadata.update(metadata["document"])
            
        if "decisions" in metadata:
            context.decisions = metadata["decisions"]
            
        if "routing" in metadata:
            context.routing_decisions = metadata["routing"]
            
        if "analysis" in metadata:
            context.analysis_results = metadata["analysis"]
            
        if "preprocessing" in metadata:
            context.preprocessing_info = metadata["preprocessing"]
            
        if "performance" in metadata:
            context.performance_metrics.update(metadata["performance"])
            
        if "timings" in metadata:
            context.step_timings = metadata["timings"]
            
        if "capabilities" in metadata:
            if "required" in metadata["capabilities"]:
                context.required_capabilities = metadata["capabilities"]["required"]
            if "matches" in metadata["capabilities"]:
                context.capability_matches = metadata["capabilities"]["matches"]
                
        if "components" in metadata:
            context.component_versions = metadata["components"]
            
        if "adapter_data" in metadata:
            context.adapter_data = metadata["adapter_data"]
            
        if "warnings" in metadata:
            context.warnings = metadata["warnings"]
            
        if "errors" in metadata:
            context.errors = metadata["errors"]
            
        if "processing" in metadata and "flow" in metadata["processing"]:
            context.processing_flow = metadata["processing"]["flow"]
            
        logger.info(f"Created context from existing metadata: {context.context_id}")
        return context
    
    @classmethod
    def load_from_file(cls, file_path: str, document_path: Optional[str] = None) -> 'ProcessingMetadataContext':
        """
        Load context from a JSON file.
        
        Args:
            file_path: Path to metadata JSON file
            document_path: Optional document path (extracted from metadata if not provided)
            
        Returns:
            New ProcessingMetadataContext instance
        """
        # Load metadata from file
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        # Get document path
        if document_path is None:
            document_path = metadata.get("document", {}).get("path")
            if not document_path:
                raise ValueError("Document path not found in metadata and not provided")
                
        # Create from metadata
        context = cls.from_existing_metadata(metadata, document_path)
        logger.info(f"Loaded metadata context from file: {file_path}")
        return context 