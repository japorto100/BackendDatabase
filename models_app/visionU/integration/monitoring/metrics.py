import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import statistics
from collections import deque

logger = logging.getLogger(__name__)

class ProcessingMetrics:
    """Tracks and analyzes processing metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize metrics tracking."""
        self.config = config or {}
        self.window_size = self.config.get("window_size", 1000)
        self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialize metrics storage."""
        self.processing_times = deque(maxlen=self.window_size)
        self.success_count = 0
        self.failure_count = 0
        self.start_time = datetime.now()
        self.last_flush_time = self.start_time
        
        # Performance thresholds
        self.thresholds = {
            "processing_time": self.config.get("max_processing_time", 30.0),  # seconds
            "error_rate": self.config.get("max_error_rate", 0.1),  # 10%
            "success_rate": self.config.get("min_success_rate", 0.9)  # 90%
        }
    
    def record_processing_time(self, processing_time: float) -> None:
        """Record a processing time measurement."""
        self.processing_times.append(processing_time)
        
        # Check for performance issues
        if processing_time > self.thresholds["processing_time"]:
            logger.warning(f"Processing time {processing_time}s exceeded threshold "
                         f"{self.thresholds['processing_time']}s")
    
    def record_processing_result(self, success: bool) -> None:
        """Record a processing result."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        # Check error rate
        total = self.success_count + self.failure_count
        if total > 0:
            error_rate = self.failure_count / total
            if error_rate > self.thresholds["error_rate"]:
                logger.warning(f"Error rate {error_rate:.2%} exceeded threshold "
                             f"{self.thresholds['error_rate']:.2%}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        total_processed = self.success_count + self.failure_count
        uptime = datetime.now() - self.start_time
        
        metrics = {
            "total_processed": total_processed,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / total_processed if total_processed > 0 else 0.0,
            "error_rate": self.failure_count / total_processed if total_processed > 0 else 0.0,
            "uptime_seconds": uptime.total_seconds(),
            "processing_rate": total_processed / uptime.total_seconds() if uptime.total_seconds() > 0 else 0.0
        }
        
        # Add processing time statistics if available
        if self.processing_times:
            metrics.update({
                "avg_processing_time": statistics.mean(self.processing_times),
                "min_processing_time": min(self.processing_times),
                "max_processing_time": max(self.processing_times),
                "median_processing_time": statistics.median(self.processing_times)
            })
            
            # Add 95th percentile if enough samples
            if len(self.processing_times) >= 20:
                sorted_times = sorted(self.processing_times)
                idx = int(len(sorted_times) * 0.95)
                metrics["processing_time_95th"] = sorted_times[idx]
        
        return metrics
    
    def get_performance_issues(self) -> List[Dict[str, Any]]:
        """Get list of current performance issues."""
        issues = []
        metrics = self.get_current_metrics()
        
        # Check processing time
        if metrics.get("avg_processing_time", 0) > self.thresholds["processing_time"]:
            issues.append({
                "type": "high_processing_time",
                "current": metrics["avg_processing_time"],
                "threshold": self.thresholds["processing_time"]
            })
        
        # Check error rate
        if metrics["error_rate"] > self.thresholds["error_rate"]:
            issues.append({
                "type": "high_error_rate",
                "current": metrics["error_rate"],
                "threshold": self.thresholds["error_rate"]
            })
        
        # Check success rate
        if metrics["success_rate"] < self.thresholds["success_rate"]:
            issues.append({
                "type": "low_success_rate",
                "current": metrics["success_rate"],
                "threshold": self.thresholds["success_rate"]
            })
        
        return issues
    
    def should_flush(self) -> bool:
        """Check if metrics should be flushed to storage."""
        time_since_flush = datetime.now() - self.last_flush_time
        return time_since_flush > timedelta(hours=1)
    
    def flush(self) -> None:
        """Flush metrics to persistent storage."""
        if not self.should_flush():
            return
            
        try:
            metrics = self.get_current_metrics()
            # TODO: Implement persistent storage
            logger.info(f"Flushing metrics: {metrics}")
            self.last_flush_time = datetime.now()
        except Exception as e:
            logger.error(f"Failed to flush metrics: {str(e)}")
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._initialize_metrics() 