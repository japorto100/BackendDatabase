"""
Metrics collection and analysis system for AI models.

This module provides a comprehensive metrics collection system for AI models,
particularly for speech-to-text and text-to-speech services. It tracks various 
performance metrics, resource usage, and operational statistics.
"""

import time
import logging
import json
import os
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import threading
import psutil
from pathlib import Path
from enum import Enum
from collections import defaultdict

from django.conf import settings

logger = logging.getLogger(__name__)

class MetricsFormat(Enum):
    """Enum for supported metrics export formats."""
    JSON = "json"
    CSV = "csv"
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    ELASTICSEARCH = "elasticsearch"

class MetricsCollector:
    """
    Base metrics collector for AI model services.
    
    This class collects and manages metrics for AI model services,
    including timing, success rates, and resource usage.
    
    Attributes:
        service_name: Name of the service
        start_time: Time when metrics collection started
        operations: Dictionary of operation metrics
        resources: Dictionary of resource usage metrics
        custom_metrics: Dictionary of custom metrics
    """
    
    def __init__(self, service_name: str, collect_system_metrics: bool = True):
        """
        Initialize the metrics collector.
        
        Args:
            service_name: Name of the service
            collect_system_metrics: Whether to collect system metrics
        """
        self.service_name = service_name
        self.start_time = time.time()
        
        # Operation metrics
        self.operations = {}  # Dict[str, Dict[str, Any]]
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Resource usage
        self.resources = {
            "cpu_percent": [],
            "memory_percent": [],
            "gpu_utilization": [],
            "gpu_memory": []
        }
        
        # Custom metrics
        self.custom_metrics = {}
        
        # Monitoring thread
        self._monitor_thread = None
        self._stop_monitoring = False
        
        # Start monitoring if requested
        if collect_system_metrics:
            self._start_monitoring()
        
        # Track metrics by category for better organization and visualization
        self.metrics_by_category = defaultdict(dict)
        # Track time series data for visualization
        self.time_series_data = defaultdict(list)
        # Track last update time for each metric for dashboards that need this
        self.last_updated = {}
        # Track metadata for visualization tools to properly render metrics
        self.metrics_metadata = {}
        
        self.lock = threading.RLock()  # Use RLock for nested lock acquisition
    
    def _start_monitoring(self):
        """Start monitoring resources."""
        if self._monitor_thread is not None:
            return
            
        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._monitor_thread.start()
    
    def _stop_monitoring(self):
        """Stop monitoring resources."""
        if self._monitor_thread is None:
            return
            
        self._stop_monitoring = True
        self._monitor_thread.join(timeout=1.0)
        self._monitor_thread = None
    
    def _monitor_resources(self):
        """Monitor system resources."""
        interval = 5.0  # seconds
        
        while not self._stop_monitoring:
            try:
                # Get CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                self.resources["cpu_percent"].append(cpu_percent)
                self.resources["memory_percent"].append(memory_percent)
                
                # Get GPU metrics if available
                gpu_metrics = self._collect_gpu_metrics()
                if gpu_metrics:
                    if "utilization" in gpu_metrics:
                        self.resources["gpu_utilization"].append(gpu_metrics["utilization"])
                    if "memory_percent" in gpu_metrics:
                        self.resources["gpu_memory"].append(gpu_metrics["memory_percent"])
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
            
            # Sleep for the interval
            time.sleep(interval)
    
    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """
        Collect GPU metrics if available.
        
        Returns:
            Dict[str, float]: GPU metrics
        """
        metrics = {}
        
        try:
            # Try to import NVIDIA tools
            import pynvml
            
            # Initialize NVML
            pynvml.nvmlInit()
            
            # Get the first GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["utilization"] = utilization.gpu
            
            # Get memory usage
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["memory_total"] = memory.total
            metrics["memory_used"] = memory.used
            metrics["memory_percent"] = (memory.used / memory.total) * 100.0
            
            # Shutdown NVML
            pynvml.nvmlShutdown()
        except (ImportError, Exception) as e:
            # Failed to get GPU metrics, try using torch
            try:
                import torch
                if torch.cuda.is_available():
                    # Get the first GPU
                    device = torch.cuda.current_device()
                    
                    # Get memory usage
                    memory_allocated = torch.cuda.memory_allocated(device)
                    memory_reserved = torch.cuda.memory_reserved(device)
                    memory_total = torch.cuda.get_device_properties(device).total_memory
                    
                    metrics["memory_total"] = memory_total
                    metrics["memory_used"] = memory_allocated
                    metrics["memory_reserved"] = memory_reserved
                    metrics["memory_percent"] = (memory_allocated / memory_total) * 100.0
            except (ImportError, Exception) as e:
                # Failed to get GPU metrics, silently ignore
                pass
        
        return metrics
    
    def start_operation(self, operation_name: str) -> int:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            int: Start time in nanoseconds
        """
        # Create operation entry if not exists
        if operation_name not in self.operations:
            self.operations[operation_name] = {
                "count": 0,
                "successful_count": 0,
                "failed_count": 0,
                "total_time_ns": 0,
                "min_time_ns": float('inf'),
                "max_time_ns": 0,
                "times_ns": [],
                "failures": {}
            }
        
        # Return start time
        return time.time_ns()
    
    def stop_operation(self, operation_name: str, start_time_ns: int, success: bool = True) -> float:
        """
        Stop timing an operation and record metrics.
        
        Args:
            operation_name: Name of the operation
            start_time_ns: Start time in nanoseconds
            success: Whether the operation was successful
            
        Returns:
            float: Duration in milliseconds
        """
        # Calculate elapsed time
        end_time_ns = time.time_ns()
        elapsed_ns = end_time_ns - start_time_ns
        elapsed_ms = elapsed_ns / 1_000_000.0  # Convert to milliseconds
        
        # Update operation metrics
        if operation_name in self.operations:
            self.operations[operation_name]["count"] += 1
            
            if success:
                self.operations[operation_name]["successful_count"] += 1
            else:
                self.operations[operation_name]["failed_count"] += 1
            
            self.operations[operation_name]["total_time_ns"] += elapsed_ns
            self.operations[operation_name]["min_time_ns"] = min(
                self.operations[operation_name]["min_time_ns"], elapsed_ns
            )
            self.operations[operation_name]["max_time_ns"] = max(
                self.operations[operation_name]["max_time_ns"], elapsed_ns
            )
            self.operations[operation_name]["times_ns"].append(elapsed_ns)
        
        return elapsed_ms
    
    def record_cache_access(self, hit: bool):
        """
        Record a cache access.
        
        Args:
            hit: Whether the cache access was a hit
        """
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_custom_metric(self, category: str, name: str, value: Any, metadata: Dict[str, Any] = None) -> None:
        """
        Record a custom metric.
        
        Args:
            category: The metric category
            name: The metric name
            value: The metric value
            metadata: Optional metadata about the metric for visualization
        """
        with self.lock:
            # Store in flat metrics dictionary
            metric_key = f"{category}.{name}"
            self.custom_metrics[metric_key] = value
            
            # Store in categorized dictionary for easier visualization
            self.metrics_by_category[category][name] = value
            
            # Record timestamp for time series data
            timestamp = datetime.now().isoformat()
            self.time_series_data[metric_key].append({
                "timestamp": timestamp,
                "value": value
            })
            
            # Limit time series data to prevent memory issues
            if len(self.time_series_data[metric_key]) > 1000:  # Keep last 1000 points
                self.time_series_data[metric_key] = self.time_series_data[metric_key][-1000:]
                
            # Update last updated timestamp
            self.last_updated[metric_key] = timestamp
            
            # Store metadata if provided
            if metadata:
                self.metrics_metadata[metric_key] = metadata
    
    def calculate_percentiles(self):
        """
        Calculate percentiles for all recorded operation times.
        
        Returns:
            Dict[str, Dict[str, float]]: Percentiles for each operation
        """
        percentiles = {}
        
        for operation_name, metrics in self.operations.items():
            if not metrics["times_ns"]:
                continue
                
            times = sorted(metrics["times_ns"])
            count = len(times)
            
            percentiles[operation_name] = {
                "p50_ms": times[int(count * 0.50)] / 1_000_000.0,
                "p90_ms": times[int(count * 0.90)] / 1_000_000.0,
                "p95_ms": times[int(count * 0.95)] / 1_000_000.0,
                "p99_ms": times[int(count * 0.99)] / 1_000_000.0 if count >= 100 else None
            }
        
        return percentiles
    
    def calculate_resource_averages(self):
        """
        Calculate averages for resource usage.
        
        Returns:
            Dict[str, float]: Average resource usage
        """
        averages = {}
        
        for resource, values in self.resources.items():
            if values:
                averages[resource] = sum(values) / len(values)
        
        return averages
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.
        
        Returns:
            Dict[str, Any]: All metrics
        """
        # Calculate derived metrics
        percentiles = self.calculate_percentiles()
        resource_averages = self.calculate_resource_averages()
        
        # Calculate cache hit rate
        total_cache_accesses = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_accesses if total_cache_accesses > 0 else 0
        
        # Return all metrics
        return {
            "service_name": self.service_name,
            "start_time": self.start_time,
            "current_time": time.time(),
            "operations": self.operations,
            "operation_percentiles": percentiles,
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": cache_hit_rate
            },
            "resources": resource_averages,
            "custom_metrics": self.custom_metrics
        }
    
    def export_metrics(self, file_path: Optional[str] = None) -> Optional[str]:
        """
        Export metrics to a file.
        
        Args:
            file_path: Path to export metrics to
            
        Returns:
            Optional[str]: Path to the exported metrics file, or None if export failed
        """
        # Get metrics
        metrics = self.get_metrics()
        
        # Generate default path if not provided
        if file_path is None:
            metrics_dir = os.path.join(settings.MEDIA_ROOT, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(
                metrics_dir, 
                f"{self.service_name}_{timestamp}.json"
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Export metrics to file
        try:
            with open(file_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Exported metrics to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error exporting metrics to {file_path}: {str(e)}")
            return None
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = time.time()
        self.operations = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.resources = {k: [] for k in self.resources}
        self.custom_metrics = {}
    
    def __del__(self):
        """Clean up resources."""
        try:
            self._stop_monitoring = True
            if self._monitor_thread is not None:
                self._monitor_thread.join(timeout=1.0)
        except:
            pass

    def format_for_visualization(self, format_type: MetricsFormat = MetricsFormat.JSON) -> Any:
        """
        Format metrics data for visualization dashboards.
        
        Args:
            format_type: The desired output format
            
        Returns:
            The formatted metrics data
        """
        with self.lock:
            if format_type == MetricsFormat.JSON:
                # JSON format with nested structure for easy dashboard consumption
                return {
                    "metadata": {
                        "collector_type": self.__class__.__name__,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "metrics": self.metrics_by_category,
                    "time_series": self.time_series_data
                }
                
            elif format_type == MetricsFormat.PROMETHEUS:
                # Prometheus format
                lines = []
                for category, metrics in self.metrics_by_category.items():
                    for name, value in metrics.items():
                        # Skip non-numeric values
                        if not isinstance(value, (int, float)):
                            continue
                        
                        # Sanitize name for Prometheus (lowercase, underscores)
                        prom_name = f"{category}_{name}".lower().replace(".", "_").replace("-", "_")
                        
                        # Add metric line with type hint comment
                        lines.append(f"# TYPE {prom_name} gauge")
                        lines.append(f"{prom_name} {value}")
                
                return "\n".join(lines)
                
            elif format_type == MetricsFormat.GRAFANA:
                # Grafana-compatible JSON format (simplified)
                targets = []
                for metric_key, points in self.time_series_data.items():
                    if not points:
                        continue
                        
                    # Format data as [timestamp, value] pairs
                    datapoints = []
                    for point in points:
                        try:
                            # Convert ISO timestamp to milliseconds epoch
                            dt = datetime.fromisoformat(point["timestamp"])
                            ts_ms = int(dt.timestamp() * 1000)
                            
                            # Only include if value is numeric
                            if isinstance(point["value"], (int, float)):
                                datapoints.append([point["value"], ts_ms])
                        except (ValueError, TypeError):
                            continue
                    
                    targets.append({
                        "target": metric_key,
                        "datapoints": datapoints
                    })
                
                return targets
                
            elif format_type == MetricsFormat.ELASTICSEARCH:
                # Elasticsearch-compatible documents
                documents = []
                
                # Current metrics as individual documents
                for category, metrics in self.metrics_by_category.items():
                    for name, value in metrics.items():
                        doc = {
                            "@timestamp": datetime.now().isoformat(),
                            "category": category,
                            "name": name,
                            "value": value,
                            "metadata": self.metrics_metadata.get(f"{category}.{name}", {})
                        }
                        documents.append(doc)
                
                return documents
                
            elif format_type == MetricsFormat.CSV:
                # Simple CSV format
                lines = ["category,name,value,timestamp"]
                for category, metrics in self.metrics_by_category.items():
                    for name, value in metrics.items():
                        metric_key = f"{category}.{name}"
                        timestamp = self.last_updated.get(metric_key, datetime.now().isoformat())
                        lines.append(f"{category},{name},{value},{timestamp}")
                
                return "\n".join(lines)
                
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
    
    def get_visualization_url(self, dashboard_type: str = "grafana") -> str:
        """
        Get a URL to visualize these metrics in a dashboard.
        
        Args:
            dashboard_type: The type of dashboard to generate URL for
            
        Returns:
            str: URL to visualize metrics
        """
        # Example implementation - would be customized based on actual dashboard setup
        if dashboard_type == "grafana":
            # This is a placeholder that would be replaced with actual dashboard configuration
            return "/analytics/dashboards/grafana?collector=" + self.__class__.__name__
        elif dashboard_type == "kibana":
            return "/analytics/dashboards/kibana?collector=" + self.__class__.__name__
        else:
            return "/analytics/dashboards/generic?collector=" + self.__class__.__name__


class STTMetricsCollector(MetricsCollector):
    """
    Metrics collector for speech-to-text services.
    
    This class extends the base metrics collector with STT-specific metrics.
    """
    
    def __init__(self, service_name: str, collect_system_metrics: bool = True):
        """
        Initialize the STT metrics collector.
        
        Args:
            service_name: Name of the service
            collect_system_metrics: Whether to collect system metrics
        """
        super().__init__(service_name, collect_system_metrics)
        
        # STT-specific metrics
        self.total_audio_duration_seconds = 0
        self.total_audio_processing_time_ms = 0
        
        self.transcriptions = {
            "count": 0,
            "total_words": 0,
            "total_processing_time_ms": 0,
            "words_per_second": []
        }
        
        self.errors = {
            "count": 0,
            "by_type": {}
        }
    
    def record_audio_processed(self, duration_seconds: float, processing_time_ms: float):
        """
        Record audio processing metrics.
        
        Args:
            duration_seconds: Duration of the audio in seconds
            processing_time_ms: Processing time in milliseconds
        """
        self.total_audio_duration_seconds += duration_seconds
        self.total_audio_processing_time_ms += processing_time_ms
        
        # Calculate real-time factor
        rtf = processing_time_ms / (duration_seconds * 1000.0) if duration_seconds > 0 else 0
        
        # Record custom metric
        self.record_custom_metric(
            "audio_processing",
            "real_time_factor",
            rtf
        )
    
    def record_transcription(self, transcription: Dict[str, Any], transcription_time_ms: float):
        """
        Record transcription metrics.
        
        Args:
            transcription: Transcription result
            transcription_time_ms: Transcription time in milliseconds
        """
        # Get word count
        text = transcription.get("text", "")
        word_count = len(text.split()) if text else 0
        
        # Update transcription metrics
        self.transcriptions["count"] += 1
        self.transcriptions["total_words"] += word_count
        self.transcriptions["total_processing_time_ms"] += transcription_time_ms
        
        # Calculate words per second
        words_per_second = (word_count / transcription_time_ms) * 1000.0 if transcription_time_ms > 0 else 0
        self.transcriptions["words_per_second"].append(words_per_second)
        
        # Record custom metrics
        self.record_custom_metric(
            "transcription",
            "word_count",
            word_count
        )
        
        self.record_custom_metric(
            "transcription",
            "words_per_second",
            words_per_second
        )
        
        # Record confidence if available
        if "confidence" in transcription:
            self.record_custom_metric(
                "transcription",
                "confidence",
                transcription["confidence"]
            )
    
    def record_transcription_error(self, error_type: str, details: Dict[str, Any] = None):
        """
        Record a transcription error.
        
        Args:
            error_type: Type of the error
            details: Error details
        """
        # Update error metrics
        self.errors["count"] += 1
        
        if error_type not in self.errors["by_type"]:
            self.errors["by_type"][error_type] = {
                "count": 0,
                "details": []
            }
        
        self.errors["by_type"][error_type]["count"] += 1
        
        if details:
            self.errors["by_type"][error_type]["details"].append(details)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.
        
        Returns:
            Dict[str, Any]: All metrics
        """
        # Get base metrics
        metrics = super().get_metrics()
        
        # Calculate STT-specific metrics
        rtf = self.total_audio_processing_time_ms / (self.total_audio_duration_seconds * 1000.0) if self.total_audio_duration_seconds > 0 else 0
        avg_words_per_second = sum(self.transcriptions["words_per_second"]) / len(self.transcriptions["words_per_second"]) if self.transcriptions["words_per_second"] else 0
        
        # Add STT-specific metrics
        metrics["stt"] = {
            "audio": {
                "total_duration_seconds": self.total_audio_duration_seconds,
                "total_processing_time_ms": self.total_audio_processing_time_ms,
                "real_time_factor": rtf
            },
            "transcriptions": {
                "count": self.transcriptions["count"],
                "total_words": self.transcriptions["total_words"],
                "total_processing_time_ms": self.transcriptions["total_processing_time_ms"],
                "avg_words_per_second": avg_words_per_second
            },
            "errors": self.errors
        }
        
        return metrics


class TTSMetricsCollector(MetricsCollector):
    """
    Metrics collector for text-to-speech services.
    
    This class extends the base metrics collector with TTS-specific metrics.
    """
    
    def __init__(self, service_name: str, collect_system_metrics: bool = True):
        """
        Initialize the TTS metrics collector.
        
        Args:
            service_name: Name of the service
            collect_system_metrics: Whether to collect system metrics
        """
        super().__init__(service_name, collect_system_metrics)
        
        # TTS-specific metrics
        self.total_text_length = 0
        self.total_text_processing_time_ms = 0
        
        self.synthesis = {
            "count": 0,
            "total_audio_duration_seconds": 0,
            "total_synthesis_time_ms": 0,
            "voices": {}
        }
        
        self.errors = {
            "count": 0,
            "by_type": {}
        }
    
    def record_text_processed(self, text: str, processing_time_ms: float):
        """
        Record text processing metrics.
        
        Args:
            text: Processed text
            processing_time_ms: Processing time in milliseconds
        """
        text_length = len(text)
        self.total_text_length += text_length
        self.total_text_processing_time_ms += processing_time_ms
        
        # Calculate characters per millisecond
        chars_per_ms = text_length / processing_time_ms if processing_time_ms > 0 else 0
        
        # Record custom metric
        self.record_custom_metric(
            "text_processing",
            "chars_per_ms",
            chars_per_ms
        )
    
    def record_synthesis(self, audio_duration_seconds: float, synthesis_time_ms: float, voice_id: str):
        """
        Record synthesis metrics.
        
        Args:
            audio_duration_seconds: Duration of the generated audio in seconds
            synthesis_time_ms: Synthesis time in milliseconds
            voice_id: ID of the voice used
        """
        # Update synthesis metrics
        self.synthesis["count"] += 1
        self.synthesis["total_audio_duration_seconds"] += audio_duration_seconds
        self.synthesis["total_synthesis_time_ms"] += synthesis_time_ms
        
        # Update voice metrics
        if voice_id not in self.synthesis["voices"]:
            self.synthesis["voices"][voice_id] = {
                "count": 0,
                "total_audio_duration_seconds": 0,
                "total_synthesis_time_ms": 0
            }
        
        self.synthesis["voices"][voice_id]["count"] += 1
        self.synthesis["voices"][voice_id]["total_audio_duration_seconds"] += audio_duration_seconds
        self.synthesis["voices"][voice_id]["total_synthesis_time_ms"] += synthesis_time_ms
        
        # Calculate real-time factor
        rtf = synthesis_time_ms / (audio_duration_seconds * 1000.0) if audio_duration_seconds > 0 else 0
        
        # Record custom metrics
        self.record_custom_metric(
            "synthesis",
            "real_time_factor",
            rtf
        )
    
    def record_synthesis_error(self, error_type: str, details: Dict[str, Any] = None):
        """
        Record a synthesis error.
        
        Args:
            error_type: Type of the error
            details: Error details
        """
        # Update error metrics
        self.errors["count"] += 1
        
        if error_type not in self.errors["by_type"]:
            self.errors["by_type"][error_type] = {
                "count": 0,
                "details": []
            }
        
        self.errors["by_type"][error_type]["count"] += 1
        
        if details:
            self.errors["by_type"][error_type]["details"].append(details)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.
        
        Returns:
            Dict[str, Any]: All metrics
        """
        # Get base metrics
        metrics = super().get_metrics()
        
        # Calculate TTS-specific metrics
        overall_rtf = self.synthesis["total_synthesis_time_ms"] / (self.synthesis["total_audio_duration_seconds"] * 1000.0) if self.synthesis["total_audio_duration_seconds"] > 0 else 0
        chars_per_second = (self.total_text_length / self.total_text_processing_time_ms) * 1000.0 if self.total_text_processing_time_ms > 0 else 0
        
        # Add TTS-specific metrics
        metrics["tts"] = {
            "text": {
                "total_length": self.total_text_length,
                "total_processing_time_ms": self.total_text_processing_time_ms,
                "chars_per_second": chars_per_second
            },
            "synthesis": {
                "count": self.synthesis["count"],
                "total_audio_duration_seconds": self.synthesis["total_audio_duration_seconds"],
                "total_synthesis_time_ms": self.synthesis["total_synthesis_time_ms"],
                "real_time_factor": overall_rtf,
                "voices": self.synthesis["voices"]
            },
            "errors": self.errors
        }
        
        return metrics


class LLMMetricsCollector(MetricsCollector):
    """
    Metrics collector for language model (LLM) services.
    
    This class extends the base metrics collector with LLM-specific metrics.
    """
    
    def __init__(self, service_name: str, collect_system_metrics: bool = True):
        """
        Initialize the LLM metrics collector.
        
        Args:
            service_name: Name of the service
            collect_system_metrics: Whether to collect system metrics
        """
        super().__init__(service_name, collect_system_metrics)
        
        # LLM-specific metrics
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        self.generations = {
            "count": 0,
            "total_time_ms": 0,
            "tokens_per_second": [],
            "avg_first_token_time_ms": 0,
            "first_token_times_ms": []
        }
        
        self.context_window = {
            "utilization": [],
            "average_utilized_pct": 0
        }
        
        self.errors = {
            "count": 0,
            "by_type": {}
        }
        
        self.models = {}
    
    def record_token_usage(self, prompt_tokens: int, completion_tokens: int):
        """
        Record token usage.
        
        Args:
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
        """
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += prompt_tokens + completion_tokens
        
        # Record custom metrics
        self.record_custom_metric(
            "tokens",
            "prompt_tokens",
            prompt_tokens
        )
        
        self.record_custom_metric(
            "tokens",
            "completion_tokens",
            completion_tokens
        )
    
    def record_generation(self, prompt_length: int, response_length: int, total_time_ms: float, first_token_time_ms: Optional[float] = None):
        """
        Record generation metrics.
        
        Args:
            prompt_length: Length of the prompt (chars or tokens)
            response_length: Length of the response (chars or tokens)
            total_time_ms: Total generation time in milliseconds
            first_token_time_ms: Time to first token in milliseconds
        """
        # Update generation metrics
        self.generations["count"] += 1
        self.generations["total_time_ms"] += total_time_ms
        
        # Calculate tokens per second (assuming response_length is in tokens)
        tokens_per_second = (response_length / total_time_ms) * 1000.0 if total_time_ms > 0 else 0
        self.generations["tokens_per_second"].append(tokens_per_second)
        
        # Record first token time if provided
        if first_token_time_ms is not None:
            self.generations["first_token_times_ms"].append(first_token_time_ms)
            
            # Update average
            first_token_count = len(self.generations["first_token_times_ms"])
            self.generations["avg_first_token_time_ms"] = (
                sum(self.generations["first_token_times_ms"]) / first_token_count
            )
        
        # Record custom metrics
        self.record_custom_metric(
            "generation",
            "tokens_per_second",
            tokens_per_second
        )
        
        if first_token_time_ms is not None:
            self.record_custom_metric(
                "generation",
                "first_token_time_ms",
                first_token_time_ms
            )
    
    def record_context_window_usage(self, used_tokens: int, max_tokens: int):
        """
        Record context window usage.
        
        Args:
            used_tokens: Number of tokens used in the context window
            max_tokens: Maximum number of tokens in the context window
        """
        # Calculate utilization percentage
        utilization_pct = (used_tokens / max_tokens) * 100.0 if max_tokens > 0 else 0
        self.context_window["utilization"].append(utilization_pct)
        
        # Update average utilization
        utilization_count = len(self.context_window["utilization"])
        self.context_window["average_utilized_pct"] = (
            sum(self.context_window["utilization"]) / utilization_count
        )
        
        # Record custom metric
        self.record_custom_metric(
            "context_window",
            "utilization_pct",
            utilization_pct
        )
    
    def record_model_usage(self, model_name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Record model usage metrics.
        
        Args:
            model_name: Name of the model
            parameters: Parameters used for generation
        """
        if model_name not in self.models:
            self.models[model_name] = {
                "count": 0,
                "parameters": {}
            }
        
        self.models[model_name]["count"] += 1
        
        # Record parameters if provided
        if parameters:
            for param_name, param_value in parameters.items():
                if param_name not in self.models[model_name]["parameters"]:
                    self.models[model_name]["parameters"][param_name] = []
                
                self.models[model_name]["parameters"][param_name].append(param_value)
    
    def record_llm_error(self, error_type: str, details: Dict[str, Any] = None):
        """
        Record an LLM error.
        
        Args:
            error_type: Type of the error
            details: Error details
        """
        # Update error metrics
        self.errors["count"] += 1
        
        if error_type not in self.errors["by_type"]:
            self.errors["by_type"][error_type] = {
                "count": 0,
                "details": []
            }
        
        self.errors["by_type"][error_type]["count"] += 1
        
        if details:
            self.errors["by_type"][error_type]["details"].append(details)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.
        
        Returns:
            Dict[str, Any]: All metrics
        """
        # Get base metrics
        metrics = super().get_metrics()
        
        # Calculate LLM-specific metrics
        avg_tokens_per_second = sum(self.generations["tokens_per_second"]) / len(self.generations["tokens_per_second"]) if self.generations["tokens_per_second"] else 0
        avg_tokens_per_request = self.token_usage["total_tokens"] / self.generations["count"] if self.generations["count"] > 0 else 0
        
        # Add LLM-specific metrics
        metrics["llm"] = {
            "token_usage": self.token_usage,
            "generations": {
                "count": self.generations["count"],
                "total_time_ms": self.generations["total_time_ms"],
                "avg_tokens_per_second": avg_tokens_per_second,
                "avg_tokens_per_request": avg_tokens_per_request,
                "avg_first_token_time_ms": self.generations["avg_first_token_time_ms"]
            },
            "context_window": {
                "avg_utilization_pct": self.context_window["average_utilized_pct"]
            },
            "models": self.models,
            "errors": self.errors
        }
        
        return metrics


class VisionMetricsCollector(MetricsCollector):
    """
    Metrics collector for vision model services.
    
    This class extends the base metrics collector with vision-specific metrics
    including image processing statistics, model performance, and error tracking.
    """
    
    def __init__(self, service_name: str, collect_system_metrics: bool = True):
        """
        Initialize the vision metrics collector.
        
        Args:
            service_name: Name of the service
            collect_system_metrics: Whether to collect system metrics
        """
        super().__init__(service_name, collect_system_metrics)
        
        # Vision-specific metrics
        self.image_processing = {
            "count": 0,
            "total_processing_time_ms": 0,
            "resized_count": 0,
            "formats": {},
            "resolution_stats": {
                "original_width": [],
                "original_height": [],
                "processed_width": [],
                "processed_height": [],
            }
        }
        
        self.multi_image_processing = {
            "count": 0,
            "total_images": 0,
            "avg_images_per_request": 0,
            "max_images_per_request": 0,
        }
        
        self.inference = {
            "count": 0,
            "total_time_ms": 0,
            "confidence_scores": [],
            "avg_confidence": 0,
        }
        
        self.document_analysis = {
            "count": 0,
            "document_lengths": [],
            "avg_document_length": 0,
            "total_images_processed": 0,
        }
        
        self.errors = {
            "count": 0,
            "by_type": {}
        }
        
        self.models = {}
        
        # New granular statistics
        self.confidence_distribution = defaultdict(int)  # Bucket confidence scores
        self.response_time_distribution = defaultdict(int)  # Bucket response times
        self.hourly_usage = defaultdict(int)  # Track usage by hour
        self.daily_usage = defaultdict(int)  # Track usage by day
        self.content_type_distribution = defaultdict(int)  # Track content types
        self.error_types = defaultdict(int)  # Track error types
    
    def record_image_processed(self, 
                               image_format: str, 
                               original_dimensions: Tuple[int, int],
                               processed_dimensions: Tuple[int, int],
                               processing_time_ms: float,
                               was_resized: bool = False):
        """
        Record image processing metrics.
        
        Args:
            image_format: Format of the image (e.g., 'JPEG', 'PNG')
            original_dimensions: Original image dimensions (width, height)
            processed_dimensions: Processed image dimensions (width, height)
            processing_time_ms: Processing time in milliseconds
            was_resized: Whether the image was resized
        """
        # Update image processing metrics
        self.image_processing["count"] += 1
        self.image_processing["total_processing_time_ms"] += processing_time_ms
        
        if was_resized:
            self.image_processing["resized_count"] += 1
        
        # Update format stats
        if image_format not in self.image_processing["formats"]:
            self.image_processing["formats"][image_format] = 0
        self.image_processing["formats"][image_format] += 1
        
        # Update resolution stats
        original_width, original_height = original_dimensions
        processed_width, processed_height = processed_dimensions
        
        self.image_processing["resolution_stats"]["original_width"].append(original_width)
        self.image_processing["resolution_stats"]["original_height"].append(original_height)
        self.image_processing["resolution_stats"]["processed_width"].append(processed_width)
        self.image_processing["resolution_stats"]["processed_height"].append(processed_height)
        
        # Record custom metrics
        self.record_custom_metric(
            "image_processing",
            "processing_time_ms",
            processing_time_ms
        )
        
        self.record_custom_metric(
            "image_processing",
            "original_resolution",
            original_width * original_height
        )
        
        self.record_custom_metric(
            "image_processing",
            "processed_resolution",
            processed_width * processed_height
        )
        
        self.record_custom_metric(
            "image_processing",
            "resolution_reduction_pct",
            100 - (processed_width * processed_height) / (original_width * original_height) * 100 if original_width * original_height > 0 else 0
        )
    
    def record_multi_image_processed(self, image_count: int, processing_time_ms: float):
        """
        Record multi-image processing metrics.
        
        Args:
            image_count: Number of images processed
            processing_time_ms: Processing time in milliseconds
        """
        # Update multi-image processing metrics
        self.multi_image_processing["count"] += 1
        self.multi_image_processing["total_images"] += image_count
        
        # Update max images per request
        self.multi_image_processing["max_images_per_request"] = max(
            self.multi_image_processing["max_images_per_request"],
            image_count
        )
        
        # Update average images per request
        self.multi_image_processing["avg_images_per_request"] = (
            self.multi_image_processing["total_images"] / 
            self.multi_image_processing["count"]
        )
        
        # Record custom metrics
        self.record_custom_metric(
            "multi_image_processing",
            "image_count",
            image_count
        )
        
        self.record_custom_metric(
            "multi_image_processing",
            "processing_time_ms",
            processing_time_ms
        )
        
        self.record_custom_metric(
            "multi_image_processing",
            "time_per_image_ms",
            processing_time_ms / image_count if image_count > 0 else 0
        )
    
    def record_inference(self, inference_time_ms: float = None, confidence: float = None) -> None:
        """
        Record a model inference.
        
        Args:
            inference_time_ms: Time taken for inference in milliseconds
            confidence: Confidence score (0-1)
        """
        with self.lock:
            self.inference["count"] += 1
            
            # Update flat metrics
            if inference_time_ms is not None:
                # Update running average
                self.inference["total_time_ms"] = (
                    (self.inference["total_time_ms"] * (self.inference["count"] - 1) + inference_time_ms) / 
                    self.inference["count"]
                )
                self.inference["avg_confidence"] = (
                    (self.inference["avg_confidence"] * (self.inference["count"] - 1) + confidence) / 
                    self.inference["count"]
                )
                
                # Record to categorized metrics
                self.metrics_by_category["inference"]["avg_confidence"] = self.inference["avg_confidence"]
                self.metrics_by_category["inference"]["last_confidence"] = confidence
                
                # Add to time series data
                timestamp = datetime.now().isoformat()
                self.time_series_data["inference.confidence"].append({
                    "timestamp": timestamp,
                    "value": confidence
                })
                
                # Update response time distribution
                # Bucket by 100ms increments
                bucket = int(inference_time_ms / 100) * 100
                self.response_time_distribution[bucket] += 1
                
                # Update metadata for visualization tools
                self.metrics_metadata["inference.confidence"] = {
                    "unit": "score",
                    "min": 0,
                    "max": 1,
                    "visualization": "gauge",
                    "description": "Model confidence score"
                }
            
            # Update usage by time
            now = datetime.now()
            hour_key = now.strftime("%Y-%m-%d %H:00")
            day_key = now.strftime("%Y-%m-%d")
            
            self.hourly_usage[hour_key] += 1
            self.daily_usage[day_key] += 1
            
            # Update time series data for usage
            self.metrics_by_category["usage"]["hourly"] = dict(self.hourly_usage)
            self.metrics_by_category["usage"]["daily"] = dict(self.daily_usage)
            
            # Update metadata
            self.metrics_metadata["usage.hourly"] = {
                "visualization": "bar_chart",
                "description": "Usage by hour"
            }
            self.metrics_metadata["usage.daily"] = {
                "visualization": "bar_chart",
                "description": "Usage by day"
            }
    
    def record_document_analysis(self, document_length: int, image_count: int, processing_time_ms: float):
        """
        Record document analysis metrics.
        
        Args:
            document_length: Length of the document text
            image_count: Number of images in the document
            processing_time_ms: Processing time in milliseconds
        """
        # Update document analysis metrics
        self.document_analysis["count"] += 1
        self.document_analysis["document_lengths"].append(document_length)
        self.document_analysis["total_images_processed"] += image_count
        
        # Update average document length
        document_count = len(self.document_analysis["document_lengths"])
        self.document_analysis["avg_document_length"] = (
            sum(self.document_analysis["document_lengths"]) / document_count
        )
        
        # Record custom metrics
        self.record_custom_metric(
            "document_analysis",
            "document_length",
            document_length
        )
        
        self.record_custom_metric(
            "document_analysis",
            "image_count",
            image_count
        )
        
        self.record_custom_metric(
            "document_analysis",
            "processing_time_ms",
            processing_time_ms
        )
    
    def record_model_usage(self, model_name: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Record model usage metrics.
        
        Args:
            model_name: Name of the model
            parameters: Parameters used for inference
        """
        if model_name not in self.models:
            self.models[model_name] = {
                "count": 0,
                "parameters": {}
            }
        
        self.models[model_name]["count"] += 1
        
        # Record parameters if provided
        if parameters:
            for param_name, param_value in parameters.items():
                if param_name not in self.models[model_name]["parameters"]:
                    self.models[model_name]["parameters"][param_name] = []
                
                self.models[model_name]["parameters"][param_name].append(param_value)
    
    def record_vision_error(self, error_type: str, details: Dict[str, Any] = None) -> None:
        """
        Record a vision error.
        
        Args:
            error_type: Type of error
            details: Additional error details
        """
        with self.lock:
            self.errors["count"] += 1
            self.metrics["errors.count"] = self.errors["count"]
            
            # Record error type
            self.error_types[error_type] += 1
            self.metrics_by_category["errors"]["types"] = dict(self.error_types)
            
            # Track error rate
            error_rate = self.errors["count"] / max(1, self.inference["count"])
            self.metrics["errors.rate"] = error_rate
            self.metrics_by_category["errors"]["rate"] = error_rate
            
            # Add to time series
            timestamp = datetime.now().isoformat()
            self.time_series_data["errors.count"].append({
                "timestamp": timestamp,
                "value": self.errors["count"]
            })
            self.time_series_data["errors.rate"].append({
                "timestamp": timestamp,
                "value": error_rate
            })
            
            # Add metadata for visualization
            self.metrics_metadata["errors.rate"] = {
                "visualization": "alert_gauge",
                "warning_threshold": 0.05,
                "critical_threshold": 0.1,
                "description": "Error rate (errors/total inferences)"
            }
            
            if details:
                # Store the latest error details
                self.metrics["errors.latest"] = details
                self.metrics_by_category["errors"]["latest"] = details
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics.
        
        Returns:
            Dict[str, Any]: All metrics
        """
        # Get base metrics
        metrics = super().get_metrics()
        
        # Calculate vision-specific metrics
        avg_inference_time_ms = (
            self.inference["total_time_ms"] / self.inference["count"]
            if self.inference["count"] > 0 else 0
        )
        
        avg_processing_time_ms = (
            self.image_processing["total_processing_time_ms"] / self.image_processing["count"]
            if self.image_processing["count"] > 0 else 0
        )
        
        resized_pct = (
            (self.image_processing["resized_count"] / self.image_processing["count"]) * 100
            if self.image_processing["count"] > 0 else 0
        )
        
        # Add vision-specific metrics
        metrics["vision"] = {
            "image_processing": {
                "count": self.image_processing["count"],
                "total_processing_time_ms": self.image_processing["total_processing_time_ms"],
                "avg_processing_time_ms": avg_processing_time_ms,
                "resized_count": self.image_processing["resized_count"],
                "resized_percentage": resized_pct,
                "formats": self.image_processing["formats"],
                "avg_original_resolution": self._calculate_avg_resolution(
                    self.image_processing["resolution_stats"]["original_width"],
                    self.image_processing["resolution_stats"]["original_height"]
                ),
                "avg_processed_resolution": self._calculate_avg_resolution(
                    self.image_processing["resolution_stats"]["processed_width"],
                    self.image_processing["resolution_stats"]["processed_height"]
                )
            },
            "multi_image_processing": self.multi_image_processing,
            "inference": {
                "count": self.inference["count"],
                "total_time_ms": self.inference["total_time_ms"],
                "avg_time_ms": avg_inference_time_ms,
                "avg_confidence": self.inference["avg_confidence"]
            },
            "document_analysis": self.document_analysis,
            "models": self.models,
            "errors": self.errors
        }
        
        return metrics
    
    def _calculate_avg_resolution(self, widths: List[int], heights: List[int]) -> Dict[str, float]:
        """
        Calculate average resolution statistics.
        
        Args:
            widths: List of widths
            heights: List of heights
            
        Returns:
            Dict[str, float]: Average resolution statistics
        """
        if not widths or not heights:
            return {
                "width": 0,
                "height": 0,
                "megapixels": 0
            }
        
        avg_width = sum(widths) / len(widths)
        avg_height = sum(heights) / len(heights)
        avg_megapixels = sum(w * h for w, h in zip(widths, heights)) / len(widths) / 1_000_000
        
        return {
            "width": avg_width,
            "height": avg_height,
            "megapixels": avg_megapixels
        }


# Singleton vision metrics collectors by service name
_vision_metrics_collectors = {}

def get_vision_metrics(service_name: str) -> VisionMetricsCollector:
    """
    Get vision metrics collector for a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        VisionMetricsCollector: Metrics collector for the service
    """
    global _vision_metrics_collectors
    
    if service_name not in _vision_metrics_collectors:
        _vision_metrics_collectors[service_name] = VisionMetricsCollector(service_name)
    
    return _vision_metrics_collectors[service_name]


# Singleton metrics collectors by service name
_stt_metrics_collectors = {}
_tts_metrics_collectors = {}
_llm_metrics_collectors = {}

def get_stt_metrics(service_name: str) -> STTMetricsCollector:
    """
    Get STT metrics collector for a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        STTMetricsCollector: Metrics collector for the service
    """
    global _stt_metrics_collectors
    
    if service_name not in _stt_metrics_collectors:
        _stt_metrics_collectors[service_name] = STTMetricsCollector(service_name)
    
    return _stt_metrics_collectors[service_name]

def get_tts_metrics(service_name: str) -> TTSMetricsCollector:
    """
    Get TTS metrics collector for a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        TTSMetricsCollector: Metrics collector for the service
    """
    global _tts_metrics_collectors
    
    if service_name not in _tts_metrics_collectors:
        _tts_metrics_collectors[service_name] = TTSMetricsCollector(service_name)
    
    return _tts_metrics_collectors[service_name]

def get_llm_metrics(service_name: str) -> LLMMetricsCollector:
    """
    Get LLM metrics collector for a service.
    
    Args:
        service_name: Name of the service
        
    Returns:
        LLMMetricsCollector: Metrics collector for the service
    """
    global _llm_metrics_collectors
    
    if service_name not in _llm_metrics_collectors:
        _llm_metrics_collectors[service_name] = LLMMetricsCollector(service_name)
    
    return _llm_metrics_collectors[service_name]

def export_all_metrics(directory: Optional[str] = None) -> Dict[str, str]:
    """
    Export all metrics to files.
    
    Args:
        directory: Directory to export metrics to
        
    Returns:
        Dict[str, str]: Dictionary mapping service names to exported file paths
    """
    # Generate default directory if not provided
    if directory is None:
        directory = os.path.join(settings.MEDIA_ROOT, 'metrics', 
                                datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Export STT metrics
    stt_paths = {}
    for service_name, collector in _stt_metrics_collectors.items():
        file_path = os.path.join(directory, f"stt_{service_name}.json")
        exported_path = collector.export_metrics(file_path)
        if exported_path:
            stt_paths[service_name] = exported_path
    
    # Export TTS metrics
    tts_paths = {}
    for service_name, collector in _tts_metrics_collectors.items():
        file_path = os.path.join(directory, f"tts_{service_name}.json")
        exported_path = collector.export_metrics(file_path)
        if exported_path:
            tts_paths[service_name] = exported_path
            
    # Export LLM metrics
    llm_paths = {}
    for service_name, collector in _llm_metrics_collectors.items():
        file_path = os.path.join(directory, f"llm_{service_name}.json")
        exported_path = collector.export_metrics(file_path)
        if exported_path:
            llm_paths[service_name] = exported_path
    
    # Export Vision metrics
    vision_paths = {}
    for service_name, collector in _vision_metrics_collectors.items():
        file_path = os.path.join(directory, f"vision_{service_name}.json")
        exported_path = collector.export_metrics(file_path)
        if exported_path:
            vision_paths[service_name] = exported_path
    
    # Return all paths
    return {
        "stt": stt_paths,
        "tts": tts_paths,
        "llm": llm_paths,
        "vision": vision_paths
    }

def export_metrics_for_visualization(format_type: str = "json") -> Any:
    """
    Export all metrics in a format suitable for visualization.
    
    Args:
        format_type: Format to export in (json, prometheus, grafana, etc.)
        
    Returns:
        The formatted metrics data
    """
    # Map string to enum
    try:
        format_enum = MetricsFormat(format_type.lower())
    except ValueError:
        format_enum = MetricsFormat.JSON
    
    # Collect metrics from all collectors
    collectors = {
        "llm": get_llm_metrics() if "get_llm_metrics" in globals() else None,
        "vision": get_vision_metrics() if "get_vision_metrics" in globals() else None,
        "stt": get_stt_metrics() if "get_stt_metrics" in globals() else None,
        "tts": get_tts_metrics() if "get_tts_metrics" in globals() else None
    }
    
    # Format each collector's metrics
    formatted_metrics = {}
    for name, collector in collectors.items():
        if collector:
            formatted_metrics[name] = collector.format_for_visualization(format_enum)
    
    return formatted_metrics 