"""
Metrics Alert System for Vision and LLM Providers

This module provides alerting capabilities for metrics collected by the vision and LLM providers,
enabling automatic detection and notification of issues based on configurable thresholds.
"""

import logging
import time
import json
import threading
import os
from enum import Enum
from typing import Dict, Any, List, Callable, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from django.conf import settings

from models_app.ai_models.utils.common.metrics import (
    get_vision_metrics, 
    get_llm_metrics,
    get_stt_metrics,
    get_tts_metrics,
    export_metrics_for_visualization
)

logger = logging.getLogger(__name__)

# Alert severity levels
class AlertSeverity(Enum):
    """Severity levels for alerts"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# Default alert thresholds
DEFAULT_ALERT_THRESHOLDS = {
    "error_rate": {
        "warning": 0.05,  # 5% error rate
        "error": 0.10,    # 10% error rate
        "critical": 0.20  # 20% error rate
    },
    "response_time_ms": {
        "warning": 5000,  # 5 seconds
        "error": 10000,   # 10 seconds
        "critical": 30000 # 30 seconds
    },
    "memory_usage": {
        "warning": 0.75,  # 75% of available memory
        "error": 0.85,    # 85% of available memory
        "critical": 0.95  # 95% of available memory
    },
    "gpu_utilization": {
        "warning": 0.85,  # 85% GPU utilization
        "error": 0.95,    # 95% GPU utilization
        "critical": 0.98  # 98% GPU utilization
    },
    "error_count": {
        "warning": 5,     # 5 errors
        "error": 10,      # 10 errors
        "critical": 20    # 20 errors
    },
    "confidence": {
        "warning": 0.6,   # Confidence below 60%
        "error": 0.4,     # Confidence below 40%
        "critical": 0.2   # Confidence below 20%
    },
    # LLM-specific thresholds
    "token_usage": {
        "warning": 0.70,  # 70% of quota
        "error": 0.85,    # 85% of quota
        "critical": 0.95  # 95% of quota
    },
    "context_window": {
        "warning": 0.80,  # 80% of context window used
        "error": 0.90,    # 90% of context window used
        "critical": 0.98  # 98% of context window used
    },
    "llm_latency": {
        "warning": 3000,  # 3 seconds
        "error": 8000,    # 8 seconds
        "critical": 15000 # 15 seconds
    }
}

# Try to load custom thresholds from settings
CUSTOM_ALERT_THRESHOLDS = getattr(settings, 'ALERT_THRESHOLDS', {})

# Merge custom thresholds with defaults
ALERT_THRESHOLDS = DEFAULT_ALERT_THRESHOLDS.copy()
for metric, levels in CUSTOM_ALERT_THRESHOLDS.items():
    if metric in ALERT_THRESHOLDS:
        ALERT_THRESHOLDS[metric].update(levels)
    else:
        ALERT_THRESHOLDS[metric] = levels

# Routing configuration
DEFAULT_ROUTING_CONFIG = {
    AlertSeverity.INFO: ["log"],
    AlertSeverity.WARNING: ["log", "file"],
    AlertSeverity.ERROR: ["log", "file", "email"],
    AlertSeverity.CRITICAL: ["log", "file", "email", "webhook"]
}

# Try to load custom routing from settings
CUSTOM_ROUTING_CONFIG = getattr(settings, 'ALERT_ROUTING', {})

# Merge custom routing with defaults
ALERT_ROUTING = {}
for severity, routes in DEFAULT_ROUTING_CONFIG.items():
    ALERT_ROUTING[severity] = CUSTOM_ROUTING_CONFIG.get(severity.value, routes)

# Alert callback registry
_alert_callbacks: Dict[str, List[Callable]] = {
    "log": [],
    "file": [],
    "email": [],
    "webhook": [],
    "custom": []
}

# Cooldown tracking to prevent alert spam
_alert_cooldowns: Dict[str, float] = {}
DEFAULT_COOLDOWN_SECONDS = 300  # 5 minutes

# Metrics monitoring state
_monitoring_active = False
_monitor_thread = None
_check_interval = 60  # seconds
_last_checked: Dict[str, float] = {}

# Alert destinations for different teams/channels
ALERT_DESTINATIONS = {
    "vision_team": {
        "email": "vision-team@example.com",
        "slack": "#vision-alerts",
        "webhook": "https://hooks.slack.com/services/XXXXXX/YYYYYY/ZZZZZZ"
    },
    "operations": {
        "email": "ops@example.com",
        "slack": "#ops-alerts",
        "webhook": "https://hooks.slack.com/services/AAAAAA/BBBBBB/CCCCCC"
    },
    "admins": {
        "email": "admin@example.com",
        "slack": "#admin-alerts",
        "webhook": "https://hooks.slack.com/services/DDDDDD/EEEEEE/FFFFFF"
    },
    # Add LLM team
    "llm_team": {
        "email": "llm-team@example.com",
        "slack": "#llm-alerts",
        "webhook": "https://hooks.slack.com/services/LLLLLL/MMMMMM/NNNNNN"
    }
}

# Pre-defined alert routes for specific metrics or service types
METRIC_ROUTING = {
    "gpu_utilization": "operations",
    "memory_usage": "operations",
    "error_rate.critical": "admins",
    "error_rate": "vision_team",
    "confidence": "vision_team",
    "response_time_ms": "vision_team",
    # Add LLM-specific routing
    "token_usage": "llm_team",
    "context_window": "llm_team", 
    "llm_latency": "llm_team",
    "llm_error_rate": "llm_team",
    "llm_error_rate.critical": "admins",
}

def register_alert_callback(callback_type: str, callback_fn: Callable) -> None:
    """
    Register a callback function to handle alerts.
    
    Args:
        callback_type: Type of callback (log, file, email, webhook, custom)
        callback_fn: Function to call when alert is triggered
    """
    if callback_type not in _alert_callbacks:
        _alert_callbacks[callback_type] = []
    
    _alert_callbacks[callback_type].append(callback_fn)
    logger.info(f"Registered {callback_type} alert callback: {callback_fn.__name__}")

def trigger_alert(alert_type: str, details: Dict[str, Any], severity: AlertSeverity = AlertSeverity.WARNING) -> None:
    """
    Trigger an alert with the given details.
    
    Args:
        alert_type: Type of alert (e.g., error_rate, response_time)
        details: Alert details
        severity: Alert severity level
    """
    # Create unique key for this alert type to track cooldowns
    alert_key = f"{alert_type}:{details.get('provider', 'unknown')}:{details.get('model', 'unknown')}"
    
    # Check cooldown to prevent alert spam
    now = time.time()
    if alert_key in _alert_cooldowns:
        cooldown_time = details.get('cooldown_seconds', DEFAULT_COOLDOWN_SECONDS)
        if now - _alert_cooldowns[alert_key] < cooldown_time:
            # Still in cooldown, don't trigger alert
            return
    
    # Update cooldown timestamp
    _alert_cooldowns[alert_key] = now
    
    # Add timestamp and severity to details
    details['timestamp'] = datetime.now().isoformat()
    details['severity'] = severity.value
    details['alert_type'] = alert_type
    
    # Log all alerts
    log_level = logging.WARNING
    if severity == AlertSeverity.ERROR:
        log_level = logging.ERROR
    elif severity == AlertSeverity.CRITICAL:
        log_level = logging.CRITICAL
    elif severity == AlertSeverity.INFO:
        log_level = logging.INFO
    
    logger.log(log_level, f"ALERT [{severity.value.upper()}] {alert_type}: {details}")
    
    # Determine routing based on alert type and severity
    routing_destinations = set()
    
    # Add destinations based on severity routing
    for route in ALERT_ROUTING.get(severity, ["log"]):
        if route in _alert_callbacks and _alert_callbacks[route]:
            routing_destinations.add(route)
    
    # Add team routing based on metric type
    team = None
    if alert_type in METRIC_ROUTING:
        team = METRIC_ROUTING[alert_type]
    
    # For critical alerts, also check specific routing
    specific_key = f"{alert_type}.{severity.value}"
    if specific_key in METRIC_ROUTING:
        team = METRIC_ROUTING[specific_key]
    
    # Get team destination information if specified
    destination_info = {}
    if team and team in ALERT_DESTINATIONS:
        destination_info = ALERT_DESTINATIONS[team]
    
    # Add alert metadata
    details['routing'] = {
        'team': team,
        'destinations': list(routing_destinations),
        'destination_info': destination_info
    }
    
    # Call appropriate callbacks based on routing
    for route in routing_destinations:
        for callback in _alert_callbacks[route]:
            try:
                callback(alert_type, details)
            except Exception as e:
                logger.error(f"Error in alert callback {callback.__name__}: {e}")

def _default_file_callback(alert_type: str, details: Dict[str, Any]) -> None:
    """
    Default callback to write alerts to a file.
    
    Args:
        alert_type: Type of alert
        details: Alert details
    """
    try:
        # Define alert log file path
        alert_log_dir = Path(settings.BASE_DIR) / "logs" / "alerts"
        os.makedirs(alert_log_dir, exist_ok=True)
        
        # Create dated log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        alert_log_file = alert_log_dir / f"alerts_{date_str}.jsonl"
        
        # Write alert as JSON line
        with open(alert_log_file, "a") as f:
            f.write(json.dumps(details) + "\n")
    except Exception as e:
        logger.error(f"Error writing alert to file: {e}")

def _default_email_callback(alert_type: str, details: Dict[str, Any]) -> None:
    """
    Default callback to send email alerts.
    
    Args:
        alert_type: Type of alert
        details: Alert details
    """
    # This is a placeholder for actual email sending logic
    logger.info(f"Would send email for alert: {alert_type}")
    # In a real implementation, you would use Django's send_mail or similar
    
def _default_webhook_callback(alert_type: str, details: Dict[str, Any]) -> None:
    """
    Default callback to send webhook alerts.
    
    Args:
        alert_type: Type of alert
        details: Alert details
    """
    # This is a placeholder for actual webhook sending logic
    logger.info(f"Would send webhook for alert: {alert_type}")
    # In a real implementation, you would use requests.post or similar

class MetricsAlertMonitor:
    """
    Monitor for alerting on metrics.
    
    This class monitors metrics from various collectors and triggers alerts
    when thresholds are exceeded.
    """
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize the metrics alert monitor.
        
        Args:
            check_interval: Interval in seconds between checks
        """
        self.check_interval = check_interval
        self.stop_event = threading.Event()
        self.thread = None
        self.active = False
        
        # Keep track of last alert times by alert key to implement cooldown
        self.last_alert_times: Dict[str, float] = {}
        
        # Keep track of baseline values for detecting changes over time
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        # Track alert counts for prioritization
        self.alert_counts: Dict[str, int] = {}
        
        # Frequency of collecting new baselines
        self.baseline_update_interval = 86400  # 24 hours
        self.last_baseline_update = 0
    
    def start(self) -> None:
        """
        Start monitoring metrics.
        """
        if self.active:
            logger.warning("Metrics alert monitor already running")
            return
        
        self.active = True
        self.stop_event.clear()
        
        # Initialize baselines if needed
        if not self.baselines or (time.time() - self.last_baseline_update) > self.baseline_update_interval:
            self._update_baselines()
        
        # Start monitoring thread
        self.thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MetricsAlertMonitor"
        )
        self.thread.start()
        logger.info("Started metrics alert monitor")
    
    def stop(self) -> None:
        """
        Stop monitoring metrics.
        """
        if not self.active:
            return
        
        self.active = False
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        
        logger.info("Stopped metrics alert monitor")
    
    def _monitor_loop(self) -> None:
        """
        Main monitoring loop.
        """
        while self.active and not self.stop_event.is_set():
            try:
                # Check all providers
                self._check_all_providers()
                
                # Update baselines periodically
                if (time.time() - self.last_baseline_update) > self.baseline_update_interval:
                    self._update_baselines()
                
            except Exception as e:
                logger.error(f"Error in metrics monitor: {e}")
            
            # Sleep until next check
            self.stop_event.wait(self.check_interval)
    
    def _update_baselines(self) -> None:
        """
        Update baseline values for all metrics.
        """
        try:
            # Get current metrics for vision
            vision_metrics = get_vision_metrics()
            if vision_metrics:
                # Update vision baselines
                provider_data = vision_metrics.get_metrics()
                
                # Error rates
                error_count = provider_data.get('errors', {}).get('count', 0)
                inference_count = provider_data.get('inference', {}).get('count', 0)
                error_rate = error_count / max(1, inference_count)
                
                # Response times
                response_time = provider_data.get('inference', {}).get('avg_response_time_ms', 0)
                
                # Confidence
                confidence = provider_data.get('inference', {}).get('avg_confidence', 0)
                
                # Update baselines
                self.baselines['vision'] = {
                    'error_rate': error_rate,
                    'response_time_ms': response_time,
                    'confidence': confidence
                }
            
            # Get current metrics for LLM
            llm_metrics = get_llm_metrics()
            if llm_metrics:
                # Update LLM baselines
                provider_data = llm_metrics.get_metrics()
                
                # Get LLM metrics
                llm_data = provider_data.get('llm', {})
                
                # Error rates
                error_count = provider_data.get('errors', {}).get('count', 0)
                generation_count = llm_data.get('generations', {}).get('count', 0)
                error_rate = error_count / max(1, generation_count)
                
                # Token rate
                avg_tokens_per_second = llm_data.get('generations', {}).get('avg_tokens_per_second', 0)
                
                # First token latency
                first_token_time = llm_data.get('generations', {}).get('avg_first_token_time_ms', 0)
                
                # Context window utilization
                context_window_utilization = llm_data.get('context_window', {}).get('avg_utilization_pct', 0)
                
                # Update baselines
                self.baselines['llm'] = {
                    'error_rate': error_rate,
                    'tokens_per_second': avg_tokens_per_second,
                    'first_token_time_ms': first_token_time,
                    'context_window_utilization': context_window_utilization
                }
            
            self.last_baseline_update = time.time()
            logger.info(f"Updated metric baselines: {self.baselines}")
            
        except Exception as e:
            logger.error(f"Error updating baselines: {e}")
    
    def _check_all_providers(self) -> None:
        """
        Check metrics for all providers.
        """
        # Check vision metrics
        vision_metrics = get_vision_metrics()
        if vision_metrics:
            self._check_vision_metrics(vision_metrics)
        
        # Check LLM metrics
        llm_metrics = get_llm_metrics()  
        if llm_metrics:
            self._check_llm_metrics(llm_metrics)
            
        # Similar checks could be added for other metric types (TTS, STT)
    
    def _check_vision_metrics(self, metrics) -> None:
        """
        Check vision metrics for threshold violations.
        
        Args:
            metrics: Vision metrics collector
        """
        # Get current metrics
        provider_data = metrics.get_metrics()
        
        # Check error rate
        self._check_error_rate(provider_data)
        
        # Check response time
        self._check_response_time(provider_data)
        
        # Check memory usage
        self._check_memory_usage(provider_data)
        
        # Check GPU utilization
        self._check_gpu_utilization(provider_data)
        
        # Check confidence scores
        self._check_confidence_scores(provider_data)
    
    def _check_error_rate(self, provider_data: Dict[str, Any]) -> None:
        """
        Check error rate metrics and trigger alerts if needed.
        
        Args:
            provider_data: Provider metrics data
        """
        # Get error and inference counts
        error_count = provider_data.get('errors', {}).get('count', 0)
        inference_count = provider_data.get('inference', {}).get('count', 0)
        
        # Skip if no inferences have been made
        if inference_count == 0:
            return
        
        # Calculate error rate
        error_rate = error_count / inference_count
        
        # Get thresholds
        thresholds = ALERT_THRESHOLDS.get('error_rate', {})
        
        # Check against thresholds
        severity = None
        if error_rate >= thresholds.get('critical', 0.2):
            severity = AlertSeverity.CRITICAL
        elif error_rate >= thresholds.get('error', 0.1):
            severity = AlertSeverity.ERROR
        elif error_rate >= thresholds.get('warning', 0.05):
            severity = AlertSeverity.WARNING
        
        if severity:
            # Get baseline for comparison
            baseline = self.baselines.get('vision', {}).get('error_rate', 0)
            pct_increase = ((error_rate - baseline) / max(0.001, baseline)) * 100
            
            alert_details = {
                'error_rate': error_rate,
                'error_count': error_count,
                'inference_count': inference_count,
                'threshold': thresholds.get(severity.value, 0),
                'baseline': baseline,
                'percent_increase': pct_increase if pct_increase > 0 else 0,
                'provider': 'vision',
                'cooldown_seconds': 600  # 10 minutes cooldown for error rate alerts
            }
            
            # Include error types if available
            error_types = provider_data.get('errors', {}).get('by_type', {})
            if error_types:
                alert_details['error_types'] = error_types
            
            trigger_alert('error_rate', alert_details, severity)
            
            # Increment alert count for this metric
            self.alert_counts['error_rate'] = self.alert_counts.get('error_rate', 0) + 1
    
    def _check_response_time(self, provider_data: Dict[str, Any]) -> None:
        """
        Check response time metrics and trigger alerts if needed.
        
        Args:
            provider_data: Provider metrics data
        """
        # Get average response time
        avg_response_time = provider_data.get('inference', {}).get('avg_response_time_ms', 0)
        
        # Skip if no response time data
        if avg_response_time == 0:
            return
        
        # Get thresholds
        thresholds = ALERT_THRESHOLDS.get('response_time_ms', {})
        
        # Check against thresholds
        severity = None
        if avg_response_time >= thresholds.get('critical', 30000):
            severity = AlertSeverity.CRITICAL
        elif avg_response_time >= thresholds.get('error', 10000):
            severity = AlertSeverity.ERROR
        elif avg_response_time >= thresholds.get('warning', 5000):
            severity = AlertSeverity.WARNING
        
        if severity:
            # Get baseline for comparison
            baseline = self.baselines.get('vision', {}).get('response_time_ms', 0)
            pct_increase = ((avg_response_time - baseline) / max(0.001, baseline)) * 100
            
            alert_details = {
                'response_time_ms': avg_response_time,
                'threshold': thresholds.get(severity.value, 0),
                'baseline': baseline,
                'percent_increase': pct_increase if pct_increase > 0 else 0,
                'provider': 'vision',
                'cooldown_seconds': 300  # 5 minutes cooldown for response time alerts
            }
            
            trigger_alert('response_time_ms', alert_details, severity)
            
            # Increment alert count for this metric
            self.alert_counts['response_time_ms'] = self.alert_counts.get('response_time_ms', 0) + 1
    
    def _check_memory_usage(self, provider_data: Dict[str, Any]) -> None:
        """
        Check memory usage metrics and trigger alerts if needed.
        
        Args:
            provider_data: Provider metrics data
        """
        # Memory usage might be collected in system stats
        system_stats = provider_data.get('system', {})
        memory_percent = system_stats.get('memory_percent', 0)
        
        # Skip if no memory data
        if memory_percent == 0:
            return
        
        # Get thresholds
        thresholds = ALERT_THRESHOLDS.get('memory_usage', {})
        
        # Check against thresholds
        severity = None
        if memory_percent >= thresholds.get('critical', 0.95):
            severity = AlertSeverity.CRITICAL
        elif memory_percent >= thresholds.get('error', 0.85):
            severity = AlertSeverity.ERROR
        elif memory_percent >= thresholds.get('warning', 0.75):
            severity = AlertSeverity.WARNING
        
        if severity:
            alert_details = {
                'memory_percent': memory_percent,
                'threshold': thresholds.get(severity.value, 0),
                'provider': 'vision',
                'cooldown_seconds': 300,  # 5 minutes cooldown for memory alerts
                'total_memory': system_stats.get('memory_total', 0),
                'used_memory': system_stats.get('memory_used', 0)
            }
            
            trigger_alert('memory_usage', alert_details, severity)
            
            # Increment alert count for this metric
            self.alert_counts['memory_usage'] = self.alert_counts.get('memory_usage', 0) + 1
    
    def _check_gpu_utilization(self, provider_data: Dict[str, Any]) -> None:
        """
        Check GPU utilization metrics and trigger alerts if needed.
        
        Args:
            provider_data: Provider metrics data
        """
        # GPU usage might be collected in system stats
        system_stats = provider_data.get('system', {})
        gpu_stats = system_stats.get('gpu', {})
        
        # Skip if no GPU data
        if not gpu_stats:
            return
        
        # Check each GPU
        for gpu_id, stats in gpu_stats.items():
            gpu_util = stats.get('utilization', 0)
            gpu_memory = stats.get('memory_percent', 0)
            
            # Get thresholds
            util_thresholds = ALERT_THRESHOLDS.get('gpu_utilization', {})
            
            # Check utilization against thresholds
            severity = None
            if gpu_util >= util_thresholds.get('critical', 0.98):
                severity = AlertSeverity.CRITICAL
            elif gpu_util >= util_thresholds.get('error', 0.95):
                severity = AlertSeverity.ERROR
            elif gpu_util >= util_thresholds.get('warning', 0.85):
                severity = AlertSeverity.WARNING
            
            if severity:
                alert_details = {
                    'gpu_id': gpu_id,
                    'gpu_utilization': gpu_util,
                    'gpu_memory_percent': gpu_memory,
                    'threshold': util_thresholds.get(severity.value, 0),
                    'provider': 'vision',
                    'cooldown_seconds': 300  # 5 minutes cooldown for GPU alerts
                }
                
                trigger_alert('gpu_utilization', alert_details, severity)
                
                # Increment alert count for this metric
                self.alert_counts['gpu_utilization'] = self.alert_counts.get('gpu_utilization', 0) + 1
    
    def _check_confidence_scores(self, provider_data: Dict[str, Any]) -> None:
        """
        Check confidence score metrics and trigger alerts if needed.
        
        Args:
            provider_data: Provider metrics data
        """
        # Get average confidence
        avg_confidence = provider_data.get('inference', {}).get('avg_confidence', 0)
        
        # Skip if no confidence data or new service with few samples
        inference_count = provider_data.get('inference', {}).get('count', 0)
        if avg_confidence == 0 or inference_count < 10:
            return
        
        # Get thresholds - for confidence, lower is worse
        thresholds = ALERT_THRESHOLDS.get('confidence', {})
        
        # Check against thresholds
        severity = None
        if avg_confidence <= thresholds.get('critical', 0.2):
            severity = AlertSeverity.CRITICAL
        elif avg_confidence <= thresholds.get('error', 0.4):
            severity = AlertSeverity.ERROR
        elif avg_confidence <= thresholds.get('warning', 0.6):
            severity = AlertSeverity.WARNING
        
        if severity:
            # Get baseline for comparison
            baseline = self.baselines.get('vision', {}).get('confidence', 0)
            pct_decrease = ((baseline - avg_confidence) / max(0.001, baseline)) * 100
            
            alert_details = {
                'confidence': avg_confidence,
                'threshold': thresholds.get(severity.value, 0),
                'baseline': baseline,
                'percent_decrease': pct_decrease if pct_decrease > 0 else 0,
                'provider': 'vision',
                'inference_count': inference_count,
                'cooldown_seconds': 1800  # 30 minutes cooldown for confidence alerts
            }
            
            trigger_alert('confidence', alert_details, severity)
            
            # Increment alert count for this metric
            self.alert_counts['confidence'] = self.alert_counts.get('confidence', 0) + 1
    
    def _check_llm_metrics(self, metrics) -> None:
        """
        Check LLM metrics for threshold violations.
        
        Args:
            metrics: LLM metrics collector
        """
        # Get current metrics
        provider_data = metrics.get_metrics()
        llm_data = provider_data.get('llm', {})
        
        # Check error rate
        self._check_llm_error_rate(provider_data, llm_data)
        
        # Check token usage
        self._check_token_usage(llm_data)
        
        # Check context window utilization
        self._check_context_window(llm_data)
        
        # Check latency/response time
        self._check_llm_latency(llm_data)
        
        # Check memory usage (same as vision)
        self._check_memory_usage(provider_data)
        
        # Check GPU utilization (same as vision)  
        self._check_gpu_utilization(provider_data)
    
    def _check_llm_error_rate(self, provider_data: Dict[str, Any], llm_data: Dict[str, Any]) -> None:
        """
        Check LLM error rate metrics and trigger alerts if needed.
        
        Args:
            provider_data: Provider metrics data
            llm_data: LLM-specific metrics data
        """
        # Get error and generation counts
        error_count = provider_data.get('errors', {}).get('count', 0)
        generation_count = llm_data.get('generations', {}).get('count', 0)
        
        # Skip if no generations have been made
        if generation_count == 0:
            return
        
        # Calculate error rate
        error_rate = error_count / generation_count
        
        # Get thresholds
        thresholds = ALERT_THRESHOLDS.get('error_rate', {})
        
        # Check against thresholds
        severity = None
        if error_rate >= thresholds.get('critical', 0.2):
            severity = AlertSeverity.CRITICAL
        elif error_rate >= thresholds.get('error', 0.1):
            severity = AlertSeverity.ERROR
        elif error_rate >= thresholds.get('warning', 0.05):
            severity = AlertSeverity.WARNING
        
        if severity:
            # Get baseline for comparison
            baseline = self.baselines.get('llm', {}).get('error_rate', 0)
            pct_increase = ((error_rate - baseline) / max(0.001, baseline)) * 100
            
            alert_details = {
                'error_rate': error_rate,
                'error_count': error_count,
                'generation_count': generation_count,
                'threshold': thresholds.get(severity.value, 0),
                'baseline': baseline,
                'percent_increase': pct_increase if pct_increase > 0 else 0,
                'provider': 'llm',
                'cooldown_seconds': 600  # 10 minutes cooldown for error rate alerts
            }
            
            # Include error types if available
            error_types = provider_data.get('errors', {}).get('by_type', {})
            if error_types:
                alert_details['error_types'] = error_types
            
            trigger_alert('llm_error_rate', alert_details, severity)
            
            # Increment alert count for this metric
            self.alert_counts['llm_error_rate'] = self.alert_counts.get('llm_error_rate', 0) + 1
    
    def _check_token_usage(self, llm_data: Dict[str, Any]) -> None:
        """
        Check token usage metrics and trigger alerts if needed.
        
        Args:
            llm_data: LLM-specific metrics data
        """
        # Get token usage
        token_usage = llm_data.get('token_usage', {})
        total_tokens = token_usage.get('total_tokens', 0)
        
        # Skip if no token usage data
        if total_tokens == 0:
            return
        
        # For this example, we'll assume there's some token quota
        # In a real implementation, you would get this from provider settings
        token_quota = getattr(settings, 'LLM_TOKEN_QUOTA', 1000000)  # Default: 1M tokens
        
        # Calculate usage percentage
        usage_percent = (total_tokens / token_quota) * 100 if token_quota > 0 else 0
        
        # Get thresholds
        thresholds = ALERT_THRESHOLDS.get('token_usage', {})
        
        # Check against thresholds
        severity = None
        if usage_percent >= thresholds.get('critical', 95):
            severity = AlertSeverity.CRITICAL
        elif usage_percent >= thresholds.get('error', 85):
            severity = AlertSeverity.ERROR
        elif usage_percent >= thresholds.get('warning', 70):
            severity = AlertSeverity.WARNING
        
        if severity:
            alert_details = {
                'token_usage_percent': usage_percent,
                'total_tokens': total_tokens,
                'token_quota': token_quota,
                'threshold': thresholds.get(severity.value, 0),
                'provider': 'llm',
                'prompt_tokens': token_usage.get('prompt_tokens', 0),
                'completion_tokens': token_usage.get('completion_tokens', 0),
                'cooldown_seconds': 3600  # 1 hour cooldown for token usage alerts
            }
            
            trigger_alert('token_usage', alert_details, severity)
            
            # Increment alert count for this metric
            self.alert_counts['token_usage'] = self.alert_counts.get('token_usage', 0) + 1
    
    def _check_context_window(self, llm_data: Dict[str, Any]) -> None:
        """
        Check context window utilization metrics and trigger alerts if needed.
        
        Args:
            llm_data: LLM-specific metrics data
        """
        # Get context window utilization
        context_window = llm_data.get('context_window', {})
        utilization_pct = context_window.get('avg_utilization_pct', 0)
        
        # Skip if no context window data
        if utilization_pct == 0:
            return
        
        # Get thresholds
        thresholds = ALERT_THRESHOLDS.get('context_window', {})
        
        # Check against thresholds
        severity = None
        if utilization_pct >= thresholds.get('critical', 98):
            severity = AlertSeverity.CRITICAL
        elif utilization_pct >= thresholds.get('error', 90):
            severity = AlertSeverity.ERROR
        elif utilization_pct >= thresholds.get('warning', 80):
            severity = AlertSeverity.WARNING
        
        if severity:
            # Get baseline for comparison
            baseline = self.baselines.get('llm', {}).get('context_window_utilization', 0)
            pct_increase = ((utilization_pct - baseline) / max(0.001, baseline)) * 100
            
            alert_details = {
                'context_window_utilization': utilization_pct,
                'threshold': thresholds.get(severity.value, 0),
                'baseline': baseline,
                'percent_increase': pct_increase if pct_increase > 0 else 0,
                'provider': 'llm',
                'cooldown_seconds': 300  # 5 minutes cooldown for context window alerts
            }
            
            trigger_alert('context_window', alert_details, severity)
            
            # Increment alert count for this metric
            self.alert_counts['context_window'] = self.alert_counts.get('context_window', 0) + 1
    
    def _check_llm_latency(self, llm_data: Dict[str, Any]) -> None:
        """
        Check LLM latency metrics and trigger alerts if needed.
        
        Args:
            llm_data: LLM-specific metrics data
        """
        # Get latency metrics
        generations = llm_data.get('generations', {})
        first_token_time = generations.get('avg_first_token_time_ms', 0)
        
        # Skip if no latency data
        if first_token_time == 0:
            return
        
        # Get thresholds
        thresholds = ALERT_THRESHOLDS.get('llm_latency', {})
        
        # Check against thresholds
        severity = None
        if first_token_time >= thresholds.get('critical', 15000):
            severity = AlertSeverity.CRITICAL
        elif first_token_time >= thresholds.get('error', 8000):
            severity = AlertSeverity.ERROR
        elif first_token_time >= thresholds.get('warning', 3000):
            severity = AlertSeverity.WARNING
        
        if severity:
            # Get baseline for comparison
            baseline = self.baselines.get('llm', {}).get('first_token_time_ms', 0)
            pct_increase = ((first_token_time - baseline) / max(0.001, baseline)) * 100
            
            alert_details = {
                'first_token_time_ms': first_token_time,
                'threshold': thresholds.get(severity.value, 0),
                'baseline': baseline,
                'percent_increase': pct_increase if pct_increase > 0 else 0,
                'provider': 'llm',
                'cooldown_seconds': 300  # 5 minutes cooldown for latency alerts
            }
            
            trigger_alert('llm_latency', alert_details, severity)
            
            # Increment alert count for this metric
            self.alert_counts['llm_latency'] = self.alert_counts.get('llm_latency', 0) + 1
    
    def get_alert_status(self) -> Dict[str, Any]:
        """
        Get the current alert status.
        
        Returns:
            Dict[str, Any]: Alert status information
        """
        return {
            'active': self.active,
            'last_check': datetime.fromtimestamp(time.time()).isoformat(),
            'baselines': self.baselines,
            'alert_counts': self.alert_counts,
            'thresholds': ALERT_THRESHOLDS
        }
    
    def get_prioritized_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get a prioritized list of recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List[Dict[str, Any]]: Prioritized list of recent alerts
        """
        try:
            # First check if alerts log directory exists
            alert_log_dir = Path(settings.BASE_DIR) / "logs" / "alerts"
            if not os.path.exists(alert_log_dir):
                return []
                
            # Find all alert log files, sorted by date (newest first)
            alert_files = sorted(
                [f for f in os.listdir(alert_log_dir) if f.startswith("alerts_") and f.endswith(".jsonl")],
                reverse=True
            )
            
            if not alert_files:
                return []
            
            # Load alerts from files
            all_alerts = []
            max_alerts_to_check = 1000  # Limit how many alerts we load for performance
            alerts_loaded = 0
            
            for file_name in alert_files:
                file_path = alert_log_dir / file_name
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            try:
                                alert = json.loads(line.strip())
                                # Add file source for reference
                                alert["_source_file"] = file_name
                                all_alerts.append(alert)
                                alerts_loaded += 1
                                if alerts_loaded >= max_alerts_to_check:
                                    break
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    logger.error(f"Error reading alert file {file_path}: {e}")
                
                if alerts_loaded >= max_alerts_to_check:
                    break
            
            # If no alerts found, return empty list
            if not all_alerts:
                return []
            
            # Sort alerts by severity, recency, and frequency
            scored_alerts = []
            alert_counts = {}  # Count how many times each alert type appears
            
            # First pass: count frequencies
            for alert in all_alerts:
                alert_type = alert.get("alert_type", "unknown")
                alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
            
            # Second pass: score alerts
            for alert in all_alerts:
                # Basic alert data
                alert_type = alert.get("alert_type", "unknown")
                severity_str = alert.get("severity", "warning")
                
                # Calculate recency score (newer is higher)
                timestamp = alert.get("timestamp", "")
                if timestamp:
                    try:
                        alert_time = datetime.fromisoformat(timestamp)
                        # Calculate recency score: 1.0 for now, decreasing over time
                        time_diff = datetime.now() - alert_time
                        hours_old = time_diff.total_seconds() / 3600
                        recency_score = max(0, 1.0 - (hours_old / 24))  # Decay over 24 hours
                    except ValueError:
                        recency_score = 0.0
                else:
                    recency_score = 0.0
                
                # Calculate severity score
                severity_scores = {
                    "critical": 1.0,
                    "error": 0.7,
                    "warning": 0.4,
                    "info": 0.1
                }
                severity_score = severity_scores.get(severity_str, 0.1)
                
                # Calculate frequency score
                frequency = alert_counts.get(alert_type, 1)
                frequency_score = min(1.0, frequency / 10)  # Cap at 10 occurrences
                
                # Calculate total priority score
                priority_score = (severity_score * 0.5) + (recency_score * 0.3) + (frequency_score * 0.2)
                
                # Add team priority boost
                team = alert.get("routing", {}).get("team", "")
                # Boost priority for admin alerts
                if team == "admins":
                    priority_score *= 1.2
                
                # Add alert with its score
                scored_alerts.append((alert, priority_score))
            
            # Sort by score (highest first) and take the top 'limit' alerts
            prioritized = [
                alert for alert, _ in sorted(scored_alerts, key=lambda x: x[1], reverse=True)[:limit]
            ]
            
            # Add priority information to returned alerts
            for i, alert in enumerate(prioritized):
                alert["priority_rank"] = i + 1
                
            return prioritized
            
        except Exception as e:
            logger.error(f"Error retrieving prioritized alerts: {e}")
            return []

# Create a monitor instance
_monitor = MetricsAlertMonitor()

def start_monitoring() -> None:
    """
    Start monitoring metrics.
    """
    global _monitoring_active
    if not _monitoring_active:
        _monitor.start()
        _monitoring_active = True

def stop_monitoring() -> None:
    """
    Stop monitoring metrics.
    """
    global _monitoring_active
    if _monitoring_active:
        _monitor.stop()
        _monitoring_active = False

def get_alert_status() -> Dict[str, Any]:
    """
    Get the current alert status.
    
    Returns:
        Dict[str, Any]: Alert status information
    """
    return _monitor.get_alert_status()

# Register default callbacks if settings enable them
if getattr(settings, 'ENABLE_ALERT_FILE_LOGGING', True):
    register_alert_callback('file', _default_file_callback)

if getattr(settings, 'ENABLE_ALERT_EMAILS', False):
    register_alert_callback('email', _default_email_callback)

if getattr(settings, 'ENABLE_ALERT_WEBHOOKS', False):
    register_alert_callback('webhook', _default_webhook_callback)

# Auto-start monitoring if enabled in settings
if getattr(settings, 'AUTO_START_METRICS_MONITORING', True):
    start_monitoring() 