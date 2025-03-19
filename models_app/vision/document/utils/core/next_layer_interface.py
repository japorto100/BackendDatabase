"""
Interface definitions for next layer integration.

This module defines the interfaces and event system for communicating
with the next layer in the document processing pipeline.
"""

from typing import Dict, Any, Optional, Protocol, List
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

class ProcessingEventType(Enum):
    """Types of events that can be emitted during processing."""
    DOCUMENT_RECEIVED = "document_received"
    ANALYSIS_COMPLETE = "analysis_complete"
    ADAPTER_SELECTED = "adapter_selected"
    PREPROCESSING_COMPLETE = "preprocessing_complete"
    PROCESSING_COMPLETE = "processing_complete"
    KG_INTEGRATION_COMPLETE = "kg_integration_complete"
    ERROR_OCCURRED = "error_occurred"
    RESOURCE_WARNING = "resource_warning"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    # New performance and resource events
    PERFORMANCE_METRIC = "performance_metric"
    RESOURCE_USAGE = "resource_usage"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    OPTIMIZATION_APPLIED = "optimization_applied"
    PROCESSING_PHASE_START = "processing_phase_start"
    PROCESSING_PHASE_END = "processing_phase_end"
    # Lifecycle events
    INITIALIZATION = "initialization"
    INITIALIZATION_COMPLETE = "initialization_complete"
    CLEANUP = "cleanup"
    CLEANUP_COMPLETE = "cleanup_complete"
    COMPLETION = "completion"

@dataclass
class ProcessingEvent:
    """Event emitted during document processing."""
    event_type: ProcessingEventType
    document_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, event_type: ProcessingEventType, document_id: str, data: Dict[str, Any], 
              metadata: Optional[Dict[str, Any]] = None) -> 'ProcessingEvent':
        """Factory method to create events with consistent timestamp."""
        return cls(
            event_type=event_type,
            document_id=document_id,
            timestamp=datetime.now(),
            data=data,
            metadata=metadata
        )

class ProcessingEventListener(Protocol):
    """Protocol for processing event listeners."""
    
    async def on_event(self, event: ProcessingEvent) -> None:
        """Handle a processing event."""
        ...

class NextLayerInterface:
    """Interface for communicating with the next layer."""
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """
        Initialize the next layer interface.
        
        Note: This should not be called directly. Use get_instance() instead.
        """
        if NextLayerInterface._instance is not None:
            logger.warning("NextLayerInterface should not be initialized directly. Use get_instance().")
            return
            
        self.event_listeners: Dict[ProcessingEventType, List[ProcessingEventListener]] = {}
        self.event_loop = asyncio.get_event_loop()
        self._cache: Dict[str, Any] = {}
        self._initialize_monitoring()
    
    def _initialize_monitoring(self) -> None:
        """Initialize monitoring components."""
        self.performance_metrics = {
            "processing_times": [],
            "memory_usage": [],
            "cpu_usage": [],
            "gpu_usage": [],
            "cache_hits": 0,
            "cache_misses": 0
        }
        self.resource_thresholds = {
            "memory": 0.8,  # 80% max memory usage
            "cpu": 0.9,     # 90% max CPU usage
            "gpu": 0.7      # 70% max GPU usage if available
        }
    
    def monitor_performance(self, document_id: str, phase: str, 
                          metrics: Dict[str, Any]) -> None:
        """
        Record performance metrics for a processing phase.
        
        Args:
            document_id: ID of the document being processed
            phase: Processing phase name
            metrics: Performance metrics to record
        """
        event = ProcessingEvent.create(
            event_type=ProcessingEventType.PERFORMANCE_METRIC,
            document_id=document_id,
            data={
                "phase": phase,
                "metrics": metrics
            }
        )
        self.emit_event_sync(event)
        
        # Update internal metrics
        self.performance_metrics["processing_times"].append(
            metrics.get("processing_time", 0)
        )
        if len(self.performance_metrics["processing_times"]) > 100:
            self.performance_metrics["processing_times"] = self.performance_metrics["processing_times"][-100:]
    
    def monitor_resources(self, document_id: str) -> Dict[str, Any]:
        """
        Monitor current resource usage.
        
        Args:
            document_id: ID of the document being processed
            
        Returns:
            Dict containing resource usage metrics
        """
        process = psutil.Process()
        
        # Get CPU and memory usage
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_percent = process.memory_percent()
        
        resource_data = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check for resource warnings
        if memory_percent > self.resource_thresholds["memory"] * 100:
            self.emit_event_sync(ProcessingEvent.create(
                event_type=ProcessingEventType.RESOURCE_WARNING,
                document_id=document_id,
                data={
                    "resource": "memory",
                    "current": memory_percent,
                    "threshold": self.resource_thresholds["memory"] * 100
                }
            ))
        
        if cpu_percent > self.resource_thresholds["cpu"] * 100:
            self.emit_event_sync(ProcessingEvent.create(
                event_type=ProcessingEventType.RESOURCE_WARNING,
                document_id=document_id,
                data={
                    "resource": "cpu",
                    "current": cpu_percent,
                    "threshold": self.resource_thresholds["cpu"] * 100
                }
            ))
        
        # Update internal metrics
        self.performance_metrics["cpu_usage"].append(cpu_percent)
        self.performance_metrics["memory_usage"].append(memory_percent)
        
        # Keep only last 100 measurements
        for metric in ["cpu_usage", "memory_usage"]:
            if len(self.performance_metrics[metric]) > 100:
                self.performance_metrics[metric] = self.performance_metrics[metric][-100:]
        
        return resource_data

    def register_listener(self, event_type: ProcessingEventType,
                         listener: ProcessingEventListener) -> None:
        """
        Register a listener for a specific event type.
        
        Args:
            event_type: Type of event to listen for
            listener: Listener to register
        """
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(listener)
        logger.debug(f"Registered listener for {event_type.value}")
    
    async def emit_event(self, event: ProcessingEvent) -> None:
        """
        Emit an event to all registered listeners.
        
        Args:
            event: Event to emit
        """
        if event.event_type not in self.event_listeners:
            return
            
        tasks = []
        for listener in self.event_listeners[event.event_type]:
            tasks.append(listener.on_event(event))
            
        if tasks:
            await asyncio.gather(*tasks)
            logger.debug(f"Emitted {event.event_type.value} event to {len(tasks)} listeners")
    
    def emit_event_sync(self, event: ProcessingEvent) -> None:
        """
        Emit an event synchronously.
        
        Args:
            event: Event to emit
        """
        try:
            self.event_loop.run_until_complete(self.emit_event(event))
        except RuntimeError:
            # Handle case where we're already in an event loop
            asyncio.create_task(self.emit_event(event))
    
    def emit_simple_event(self, event_type: ProcessingEventType, document_id: str, 
                         data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Simplified method to emit events without creating ProcessingEvent manually.
        
        Args:
            event_type: Type of event to emit
            document_id: ID of the document being processed
            data: Event data
            metadata: Optional metadata
        """
        event = ProcessingEvent.create(
            event_type=event_type,
            document_id=document_id,
            data=data,
            metadata=metadata
        )
        self.emit_event_sync(event)
    
    def cache_result(self, key: str, result: Any,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Cache a processing result.
        
        Args:
            key: Cache key
            result: Result to cache
            metadata: Optional result metadata
        """
        self._cache[key] = {
            "result": result,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        logger.debug(f"Cached result for key: {key}")
        
        # Update cache metrics
        self.performance_metrics["cache_hits"] += 1
    
    def get_cached_result(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        if key in self._cache:
            logger.debug(f"Cache hit for key: {key}")
            self.performance_metrics["cache_hits"] += 1
            return self._cache[key]
        logger.debug(f"Cache miss for key: {key}")
        self.performance_metrics["cache_misses"] += 1
        return None
    
    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """
        Invalidate cache entries.
        
        Args:
            key: Optional specific key to invalidate (all if None)
        """
        if key is None:
            self._cache.clear()
            logger.debug("Cleared entire cache")
        elif key in self._cache:
            del self._cache[key]
            logger.debug(f"Invalidated cache for key: {key}")
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the next layer interface.
        
        Returns:
            Dict containing health check results
        """
        return {
            "status": "healthy",
            "cache_size": len(self._cache),
            "event_listeners": {
                event_type.value: len(listeners)
                for event_type, listeners in self.event_listeners.items()
            },
            "performance_metrics": {
                "avg_processing_time": sum(self.performance_metrics["processing_times"]) / len(self.performance_metrics["processing_times"]) if self.performance_metrics["processing_times"] else 0,
                "avg_memory_usage": sum(self.performance_metrics["memory_usage"]) / len(self.performance_metrics["memory_usage"]) if self.performance_metrics["memory_usage"] else 0,
                "avg_cpu_usage": sum(self.performance_metrics["cpu_usage"]) / len(self.performance_metrics["cpu_usage"]) if self.performance_metrics["cpu_usage"] else 0,
                "cache_hit_rate": self.performance_metrics["cache_hits"] / (self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]) if (self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]) > 0 else 0
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def shutdown(self) -> None:
        """Clean up resources during shutdown."""
        self._cache.clear()
        self.event_listeners.clear()
        logger.info("Next layer interface shut down")
        
        # Reset singleton instance
        NextLayerInterface._instance = None 