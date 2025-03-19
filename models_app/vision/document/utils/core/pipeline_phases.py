from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import datetime
import logging

logger = logging.getLogger(__name__)

class PipelinePhase(Enum):
    """Document processing pipeline phases."""
    INITIALIZATION = "initialization"
    ANALYSIS = "analysis"
    SELECTION = "selection"
    PREPROCESSING = "preprocessing"
    CHUNKING = "chunking"
    PROCESSING = "processing"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    COMPLETION = "completion"

@dataclass
class PhaseContext:
    """Context for a pipeline phase."""
    phase: PipelinePhase
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class PipelinePhaseManager:
    """Manages document processing pipeline phases."""
    
    def __init__(self):
        self.phases: Dict[PipelinePhase, PhaseContext] = {}
        self.current_phase: Optional[PipelinePhase] = None
        self.phase_history: List[Dict[str, Any]] = []
        self._transaction_stack: List[Dict[str, Any]] = []
    
    def start_phase(self, phase: PipelinePhase, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start a new pipeline phase.
        
        Args:
            phase: Phase to start
            metadata: Optional metadata for the phase
        """
        if phase in self.phases and self.phases[phase].status == "running":
            logger.warning(f"Phase {phase.value} already running")
            return
            
        self.current_phase = phase
        self.phases[phase] = PhaseContext(
            phase=phase,
            start_time=datetime.datetime.now(),
            metadata=metadata or {},
            status="running"
        )
        
        self.phase_history.append({
            "phase": phase.value,
            "action": "start",
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        logger.debug(f"Started phase: {phase.value}")
    
    def end_phase(self, phase: PipelinePhase, status: str = "completed") -> None:
        """
        End a pipeline phase.
        
        Args:
            phase: Phase to end
            status: Final status of the phase
        """
        if phase not in self.phases:
            logger.warning(f"Phase {phase.value} not started")
            return
            
        self.phases[phase].end_time = datetime.datetime.now()
        self.phases[phase].status = status
        
        if phase == self.current_phase:
            self.current_phase = None
            
        self.phase_history.append({
            "phase": phase.value,
            "action": "end",
            "status": status,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        logger.debug(f"Ended phase: {phase.value} with status: {status}")
    
    def record_decision(self, phase: PipelinePhase, decision: str, reason: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a decision made during a phase.
        
        Args:
            phase: Phase where decision was made
            decision: Decision description
            reason: Reason for the decision
            metadata: Additional decision metadata
        """
        if phase not in self.phases:
            logger.warning(f"Cannot record decision - phase {phase.value} not started")
            return
            
        decision_record = {
            "decision": decision,
            "reason": reason,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if metadata:
            decision_record["metadata"] = metadata
            
        self.phases[phase].decisions.append(decision_record)
        logger.debug(f"Recorded decision in {phase.value}: {decision}")
    
    def record_error(self, phase: PipelinePhase, error: str,
                    is_fatal: bool = False,
                    details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an error that occurred during a phase.
        
        Args:
            phase: Phase where error occurred
            error: Error description
            is_fatal: Whether error is fatal
            details: Additional error details
        """
        if phase not in self.phases:
            logger.warning(f"Cannot record error - phase {phase.value} not started")
            return
            
        error_record = {
            "error": error,
            "is_fatal": is_fatal,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if details:
            error_record["details"] = details
            
        self.phases[phase].errors.append(error_record)
        
        if is_fatal:
            self.end_phase(phase, status="failed")
            
        log_level = logging.ERROR if is_fatal else logging.WARNING
        logger.log(log_level, f"Error in {phase.value}: {error}")
    
    def record_metric(self, phase: PipelinePhase, metric_name: str,
                     value: Any, metric_type: str = "performance") -> None:
        """
        Record a performance metric for a phase.
        
        Args:
            phase: Phase to record metric for
            metric_name: Name of the metric
            value: Metric value
            metric_type: Type of metric
        """
        if phase not in self.phases:
            logger.warning(f"Cannot record metric - phase {phase.value} not started")
            return
            
        if metric_type not in self.phases[phase].performance_metrics:
            self.phases[phase].performance_metrics[metric_type] = {}
            
        self.phases[phase].performance_metrics[metric_type][metric_name] = {
            "value": value,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.debug(f"Recorded {metric_type} metric in {phase.value}: {metric_name}={value}")
    
    def start_transaction(self) -> None:
        """Start a new transaction to track changes."""
        self._transaction_stack.append({
            "phases": self.phases.copy(),
            "current_phase": self.current_phase,
            "history": self.phase_history.copy()
        })
        logger.debug("Started phase transaction")
    
    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        if self._transaction_stack:
            self._transaction_stack.pop()
            logger.debug("Committed phase transaction")
    
    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        if self._transaction_stack:
            state = self._transaction_stack.pop()
            self.phases = state["phases"]
            self.current_phase = state["current_phase"]
            self.phase_history = state["history"]
            logger.debug("Rolled back phase transaction")
    
    def get_phase_duration(self, phase: PipelinePhase) -> Optional[float]:
        """
        Get the duration of a phase in seconds.
        
        Args:
            phase: Phase to get duration for
            
        Returns:
            float: Duration in seconds or None if phase not complete
        """
        if phase not in self.phases:
            return None
            
        context = self.phases[phase]
        if context.end_time:
            return (context.end_time - context.start_time).total_seconds()
        return None
    
    def get_current_phase(self) -> Optional[PipelinePhase]:
        """Get the currently active phase."""
        return self.current_phase
    
    def get_phase_status(self, phase: PipelinePhase) -> Optional[str]:
        """Get the status of a phase."""
        if phase not in self.phases:
            return None
        return self.phases[phase].status
    
    def get_phase_summary(self, phase: PipelinePhase) -> Optional[Dict[str, Any]]:
        """Get a summary of a phase's execution."""
        if phase not in self.phases:
            return None
            
        context = self.phases[phase]
        return {
            "status": context.status,
            "duration": self.get_phase_duration(phase),
            "decisions": len(context.decisions),
            "errors": len(context.errors),
            "metrics": context.performance_metrics,
            "metadata": context.metadata
        }
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the entire pipeline execution."""
        return {
            "phases": {
                phase.value: self.get_phase_summary(phase)
                for phase in self.phases
            },
            "current_phase": self.current_phase.value if self.current_phase else None,
            "total_duration": sum(
                self.get_phase_duration(phase) or 0 
                for phase in self.phases
            ),
            "history": self.phase_history,
            "status": "failed" if any(
                context.status == "failed" 
                for context in self.phases.values()
            ) else "completed" if all(
                context.status == "completed" 
                for context in self.phases.values()
            ) else "running"
        } 