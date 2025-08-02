"""
Performance Monitoring and Metrics Collection
Provides comprehensive performance monitoring for the fraud detection system.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import functools
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric measurement"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Collection of performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_percent: float = 0.0
    active_threads: int = 0
    request_count: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    model_prediction_time: float = 0.0
    feature_computation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics = defaultdict(lambda: deque(maxlen=10000))
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.system_metrics = deque(maxlen=1000)
        self._lock = threading.Lock()
        
        # Start background system metrics collection
        self._start_system_metrics_collection()
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        
        with self._lock:
            metric_point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {}
            )
            self.metrics[name].append(metric_point)
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        
        key = f"{name}:{str(sorted((labels or {}).items()))}"
        with self._lock:
            self.counters[key] += 1
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record a timing metric"""
        
        key = f"{name}:{str(sorted((labels or {}).items()))}"
        with self._lock:
            self.timers[key].append(duration)
            
            # Keep only last 1000 measurements
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]
    
    def get_metric_summary(self, name: str, window_minutes: int = 60) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self._lock:
            recent_values = [
                point.value for point in self.metrics[name]
                if point.timestamp >= cutoff_time
            ]
        
        if not recent_values:
            return {}
        
        return {
            'count': len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'mean': statistics.mean(recent_values),
            'median': statistics.median(recent_values),
            'std_dev': statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0,
            'p95': self._percentile(recent_values, 0.95),
            'p99': self._percentile(recent_values, 0.99)
        }
    
    def get_counter_value(self, name: str, labels: Dict[str, str] = None) -> int:
        """Get current counter value"""
        
        key = f"{name}:{str(sorted((labels or {}).items()))}"
        with self._lock:
            return self.counters[key]
    
    def get_timer_summary(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get summary statistics for a timer"""
        
        key = f"{name}:{str(sorted((labels or {}).items()))}"
        
        with self._lock:
            values = self.timers[key].copy()
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'p95': self._percentile(values, 0.95),
            'p99': self._percentile(values, 0.99)
        }
    
    def get_system_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics"""
        
        with self._lock:
            if self.system_metrics:
                return self.system_metrics[-1]
        
        return PerformanceMetrics()
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary"""
        
        metrics_summary = {}
        
        # Individual metrics
        with self._lock:
            for name in self.metrics.keys():
                metrics_summary[f"metric_{name}"] = self.get_metric_summary(name)
            
            # Counters
            for key, value in self.counters.items():
                metrics_summary[f"counter_{key}"] = value
            
            # Timers
            for key in self.timers.keys():
                metrics_summary[f"timer_{key}"] = self.get_timer_summary(key.split(':')[0], {})
        
        # System metrics
        metrics_summary['system'] = self.get_system_metrics().__dict__
        
        return metrics_summary
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _start_system_metrics_collection(self):
        """Start background thread for system metrics collection"""
        
        def collect_system_metrics():
            while True:
                try:
                    # Get system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    active_threads = threading.active_count()
                    
                    # Get application-specific metrics
                    request_count = self.get_counter_value('http_requests_total')
                    
                    # Calculate rates and averages
                    response_time_summary = self.get_timer_summary('http_request_duration')
                    avg_response_time = response_time_summary.get('mean', 0.0)
                    
                    error_count = self.get_counter_value('http_requests_errors')
                    error_rate = (error_count / max(request_count, 1)) * 100
                    
                    cache_hits = self.get_counter_value('cache_hits')
                    cache_total = self.get_counter_value('cache_requests')
                    cache_hit_rate = (cache_hits / max(cache_total, 1)) * 100
                    
                    model_time_summary = self.get_timer_summary('model_prediction_duration')
                    model_prediction_time = model_time_summary.get('mean', 0.0)
                    
                    feature_time_summary = self.get_timer_summary('feature_computation_duration')
                    feature_computation_time = feature_time_summary.get('mean', 0.0)
                    
                    # Create metrics object
                    metrics = PerformanceMetrics(
                        cpu_usage=cpu_percent,
                        memory_usage=memory.used / (1024 * 1024),  # MB
                        memory_percent=memory.percent,
                        active_threads=active_threads,
                        request_count=request_count,
                        average_response_time=avg_response_time,
                        error_rate=error_rate,
                        cache_hit_rate=cache_hit_rate,
                        model_prediction_time=model_prediction_time,
                        feature_computation_time=feature_computation_time,
                        timestamp=datetime.now()
                    )
                    
                    with self._lock:
                        self.system_metrics.append(metrics)
                    
                    time.sleep(60)  # Collect every minute
                    
                except Exception as e:
                    logger.error(f"Error collecting system metrics: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self._lock:
            for name, metric_points in self.metrics.items():
                # Remove old metric points
                while metric_points and metric_points[0].timestamp < cutoff_time:
                    metric_points.popleft()


# Global metrics collector instance
metrics_collector = MetricsCollector()


def timer(name: str, labels: Dict[str, str] = None):
    """Decorator to time function execution"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = (time.time() - start_time) * 1000  # Convert to milliseconds
                metrics_collector.record_timer(name, duration, labels)
        
        return wrapper
    return decorator


def counter(name: str, labels: Dict[str, str] = None):
    """Decorator to count function calls"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                metrics_collector.increment_counter(name, labels)
                return result
            except Exception as e:
                error_labels = (labels or {}).copy()
                error_labels['error'] = type(e).__name__
                metrics_collector.increment_counter(f"{name}_errors", error_labels)
                raise
        
        return wrapper
    return decorator


class PerformanceProfiler:
    """Context manager for detailed performance profiling"""
    
    def __init__(self, operation_name: str, labels: Dict[str, str] = None):
        self.operation_name = operation_name
        self.labels = labels or {}
        self.start_time = None
        self.checkpoints = []
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        total_duration = (time.time() - self.start_time) * 1000
        
        # Record total duration
        metrics_collector.record_timer(
            f"{self.operation_name}_total_duration", 
            total_duration, 
            self.labels
        )
        
        # Record checkpoint durations
        prev_time = self.start_time
        for checkpoint_name, checkpoint_time in self.checkpoints:
            checkpoint_duration = (checkpoint_time - prev_time) * 1000
            metrics_collector.record_timer(
                f"{self.operation_name}_{checkpoint_name}_duration",
                checkpoint_duration,
                self.labels
            )
            prev_time = checkpoint_time
        
        # Count operations
        if exc_type is None:
            metrics_collector.increment_counter(f"{self.operation_name}_success", self.labels)
        else:
            error_labels = self.labels.copy()
            error_labels['error_type'] = exc_type.__name__
            metrics_collector.increment_counter(f"{self.operation_name}_errors", error_labels)
    
    def checkpoint(self, name: str):
        """Add a performance checkpoint"""
        self.checkpoints.append((name, time.time()))


class ResourceMonitor:
    """Monitor resource usage and alert on thresholds"""
    
    def __init__(self, cpu_threshold: float = 80.0, memory_threshold: float = 80.0):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.alerts_sent = set()
        self.alert_cooldown = timedelta(minutes=15)
        self.last_alert_times = {}
    
    def check_resources(self) -> List[Dict[str, Any]]:
        """Check resource usage and return alerts if necessary"""
        
        alerts = []
        current_time = datetime.now()
        
        # Get current metrics
        metrics = metrics_collector.get_system_metrics()
        
        # Check CPU usage
        if metrics.cpu_usage > self.cpu_threshold:
            alert_key = f"cpu_high_{metrics.cpu_usage:.1f}"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    'type': 'high_cpu_usage',
                    'value': metrics.cpu_usage,
                    'threshold': self.cpu_threshold,
                    'severity': 'warning' if metrics.cpu_usage < 90 else 'critical',
                    'timestamp': current_time.isoformat()
                })
                self.last_alert_times[alert_key] = current_time
        
        # Check memory usage
        if metrics.memory_percent > self.memory_threshold:
            alert_key = f"memory_high_{metrics.memory_percent:.1f}"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    'type': 'high_memory_usage',
                    'value': metrics.memory_percent,
                    'threshold': self.memory_threshold,
                    'severity': 'warning' if metrics.memory_percent < 90 else 'critical',
                    'timestamp': current_time.isoformat()
                })
                self.last_alert_times[alert_key] = current_time
        
        # Check error rate
        if metrics.error_rate > 5.0:  # 5% error rate
            alert_key = f"error_rate_high_{metrics.error_rate:.1f}"
            if self._should_send_alert(alert_key, current_time):
                alerts.append({
                    'type': 'high_error_rate',
                    'value': metrics.error_rate,
                    'threshold': 5.0,
                    'severity': 'critical',
                    'timestamp': current_time.isoformat()
                })
                self.last_alert_times[alert_key] = current_time
        
        return alerts
    
    def _should_send_alert(self, alert_key: str, current_time: datetime) -> bool:
        """Check if alert should be sent based on cooldown period"""
        
        last_alert_time = self.last_alert_times.get(alert_key)
        
        if last_alert_time is None:
            return True
        
        return current_time - last_alert_time > self.alert_cooldown


# Global resource monitor
resource_monitor = ResourceMonitor()


def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary"""
    
    return {
        'system_metrics': metrics_collector.get_system_metrics().__dict__,
        'all_metrics': metrics_collector.get_all_metrics(),
        'resource_alerts': resource_monitor.check_resources(),
        'timestamp': datetime.now().isoformat()
    }


def record_fraud_detection_metrics(processing_time: float, hub_score: float, 
                                 spoke_score: float, final_score: float,
                                 risk_level: str, action: str):
    """Record fraud detection specific metrics"""
    
    # Record processing time
    metrics_collector.record_timer('fraud_detection_processing_time', processing_time)
    
    # Record scores
    metrics_collector.record_metric('hub_risk_score', hub_score)
    metrics_collector.record_metric('spoke_fraud_score', spoke_score)
    metrics_collector.record_metric('final_fraud_score', final_score)
    
    # Count by risk level and action
    metrics_collector.increment_counter('fraud_detections_total', {'risk_level': risk_level})
    metrics_collector.increment_counter('fraud_actions_total', {'action': action})


def record_model_metrics(model_name: str, prediction_time: float, 
                        feature_count: int, model_version: str):
    """Record model-specific metrics"""
    
    labels = {
        'model_name': model_name,
        'model_version': model_version
    }
    
    metrics_collector.record_timer('model_prediction_time', prediction_time, labels)
    metrics_collector.record_metric('model_feature_count', feature_count, labels)
    metrics_collector.increment_counter('model_predictions_total', labels)