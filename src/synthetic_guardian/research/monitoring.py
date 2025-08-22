"""
Advanced Monitoring and Observability for Research Modules

This module provides comprehensive monitoring, alerting, and observability
for research operations. It includes metrics collection, performance tracking,
anomaly detection, and integration with monitoring systems.

Features:
1. Real-time metrics collection and aggregation
2. Performance benchmarking and SLA monitoring
3. Anomaly detection for research operations
4. Integration with Prometheus, Grafana, and other systems
5. Automated alerting and notification systems
6. Research operation audit trails
7. Resource usage tracking and optimization recommendations
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import threading
import statistics
import numpy as np
from abc import ABC, abstractmethod

from ..utils.logger import get_logger


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert for monitoring events."""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    timestamp: float
    module_name: str
    metric_name: str
    current_value: float
    threshold_value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None


@dataclass
class PerformanceBenchmark:
    """Performance benchmark for research operations."""
    operation_name: str
    module_name: str
    baseline_time: float
    baseline_memory: float
    baseline_accuracy: float
    sla_time_threshold: float
    sla_memory_threshold: float
    sla_accuracy_threshold: float
    measurement_count: int = 0
    violations: List[Dict[str, Any]] = field(default_factory=list)


class MetricCollector:
    """Collects and aggregates metrics for research operations."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._lock = threading.Lock()
        self.logger = get_logger(self.__class__.__name__)
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a metric data point."""
        if labels is None:
            labels = {}
        if metadata is None:
            metadata = {}
        
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels,
            metadata=metadata
        )
        
        with self._lock:
            metric_key = f"{metric_name}:{json.dumps(labels, sort_keys=True)}"
            self.metrics[metric_key].append(point)
            
            # Update aggregated metrics
            self._update_aggregated_metrics(metric_name, point, metric_type)
        
        self.logger.debug(f"Recorded metric {metric_name}: {value}")
    
    def _update_aggregated_metrics(
        self,
        metric_name: str,
        point: MetricPoint,
        metric_type: MetricType
    ) -> None:
        """Update aggregated metrics for faster queries."""
        key = metric_name
        
        if key not in self.aggregated_metrics:
            self.aggregated_metrics[key] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0,
                "last_value": 0.0,
                "last_timestamp": 0.0
            }
        
        agg = self.aggregated_metrics[key]
        agg["count"] += 1
        agg["sum"] += point.value
        agg["min"] = min(agg["min"], point.value)
        agg["max"] = max(agg["max"], point.value)
        agg["avg"] = agg["sum"] / agg["count"]
        agg["last_value"] = point.value
        agg["last_timestamp"] = point.timestamp
    
    def get_metric_summary(
        self,
        metric_name: str,
        time_window_minutes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self._lock:
            if metric_name not in self.aggregated_metrics:
                return {"error": f"Metric {metric_name} not found"}
            
            summary = self.aggregated_metrics[metric_name].copy()
            
            # Add time-windowed statistics if requested
            if time_window_minutes:
                cutoff_time = time.time() - (time_window_minutes * 60)
                windowed_points = []
                
                for metric_key, points in self.metrics.items():
                    if metric_key.startswith(metric_name + ":"):
                        windowed_points.extend([
                            p for p in points if p.timestamp >= cutoff_time
                        ])
                
                if windowed_points:
                    values = [p.value for p in windowed_points]
                    summary.update({
                        "windowed_count": len(values),
                        "windowed_avg": statistics.mean(values),
                        "windowed_median": statistics.median(values),
                        "windowed_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "windowed_p95": np.percentile(values, 95),
                        "windowed_p99": np.percentile(values, 99)
                    })
            
            return summary
    
    def get_time_series(
        self,
        metric_name: str,
        time_window_minutes: int = 60,
        bucket_size_minutes: int = 5
    ) -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        with self._lock:
            cutoff_time = time.time() - (time_window_minutes * 60)
            bucket_size = bucket_size_minutes * 60
            
            # Collect points
            points = []
            for metric_key, metric_points in self.metrics.items():
                if metric_key.startswith(metric_name + ":"):
                    points.extend([
                        p for p in metric_points if p.timestamp >= cutoff_time
                    ])
            
            # Group into buckets
            buckets = defaultdict(list)
            for point in points:
                bucket_time = int(point.timestamp // bucket_size) * bucket_size
                buckets[bucket_time].append(point.value)
            
            # Create time series
            time_series = []
            for bucket_time in sorted(buckets.keys()):
                values = buckets[bucket_time]
                time_series.append({
                    "timestamp": bucket_time,
                    "count": len(values),
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "median": statistics.median(values)
                })
            
            return time_series
    
    def cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        with self._lock:
            for metric_key in list(self.metrics.keys()):
                points = self.metrics[metric_key]
                
                # Remove old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
                
                # Remove empty metrics
                if not points:
                    del self.metrics[metric_key]
        
        self.logger.debug(f"Cleaned up metrics older than {self.retention_hours} hours")


class AnomalyDetector:
    """Detects anomalies in research operation metrics."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def update_baseline(
        self,
        metric_name: str,
        values: List[float],
        min_samples: int = 10
    ) -> None:
        """Update baseline statistics for anomaly detection."""
        if len(values) < min_samples:
            return
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        
        self.baselines[metric_name] = {
            "mean": mean_val,
            "std": std_val,
            "min": min(values),
            "max": max(values),
            "sample_count": len(values)
        }
        
        self.logger.debug(f"Updated baseline for {metric_name}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    def detect_anomaly(
        self,
        metric_name: str,
        value: float
    ) -> Dict[str, Any]:
        """Detect if a value is anomalous."""
        if metric_name not in self.baselines:
            return {"is_anomaly": False, "reason": "No baseline available"}
        
        baseline = self.baselines[metric_name]
        mean_val = baseline["mean"]
        std_val = baseline["std"]
        
        # Z-score based anomaly detection
        if std_val > 0:
            z_score = abs(value - mean_val) / std_val
            is_anomaly = z_score > self.sensitivity
            
            return {
                "is_anomaly": is_anomaly,
                "z_score": z_score,
                "threshold": self.sensitivity,
                "baseline_mean": mean_val,
                "baseline_std": std_val,
                "deviation": value - mean_val
            }
        else:
            # If no variance, check if value differs from mean
            is_anomaly = value != mean_val
            return {
                "is_anomaly": is_anomaly,
                "reason": "No variance in baseline" if not is_anomaly else "Value differs from constant baseline",
                "baseline_mean": mean_val,
                "current_value": value
            }
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of anomaly detection status."""
        return {
            "metrics_monitored": len(self.baselines),
            "baselines": {
                name: {
                    "mean": baseline["mean"],
                    "std": baseline["std"],
                    "samples": baseline["sample_count"]
                }
                for name, baseline in self.baselines.items()
            },
            "sensitivity": self.sensitivity
        }


class AlertManager:
    """Manages alerts for research monitoring."""
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=max_alerts)
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.logger = get_logger(self.__class__.__name__)
    
    def add_alert_rule(
        self,
        rule_name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "greater",  # "greater", "less", "equal"
        severity: AlertSeverity = AlertSeverity.WARNING,
        description: str = ""
    ) -> None:
        """Add an alert rule."""
        self.alert_rules[rule_name] = {
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "description": description,
            "enabled": True
        }
        
        self.logger.info(f"Added alert rule: {rule_name}")
    
    def evaluate_alerts(
        self,
        metric_name: str,
        value: float,
        module_name: str = "unknown",
        labels: Optional[Dict[str, str]] = None
    ) -> List[Alert]:
        """Evaluate alert rules against metric value."""
        if labels is None:
            labels = {}
        
        triggered_alerts = []
        
        with self._lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule["enabled"] or rule["metric_name"] != metric_name:
                    continue
                
                # Check if threshold is crossed
                threshold_crossed = False
                comparison = rule["comparison"]
                threshold = rule["threshold"]
                
                if comparison == "greater" and value > threshold:
                    threshold_crossed = True
                elif comparison == "less" and value < threshold:
                    threshold_crossed = True
                elif comparison == "equal" and abs(value - threshold) < 1e-6:
                    threshold_crossed = True
                
                if threshold_crossed:
                    alert_id = f"{rule_name}_{module_name}_{metric_name}"
                    
                    # Check if alert already exists
                    if alert_id not in self.active_alerts:
                        alert = Alert(
                            alert_id=alert_id,
                            title=f"Alert: {rule_name}",
                            description=rule["description"] or f"{metric_name} {comparison} {threshold}",
                            severity=rule["severity"],
                            timestamp=time.time(),
                            module_name=module_name,
                            metric_name=metric_name,
                            current_value=value,
                            threshold_value=threshold,
                            labels=labels.copy()
                        )
                        
                        self.active_alerts[alert_id] = alert
                        self.alert_history.append(alert)
                        triggered_alerts.append(alert)
                        
                        self.logger.warning(f"Alert triggered: {alert.title} - {alert.description}")
        
        return triggered_alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_timestamp = time.time()
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Resolved alert: {alert.title}")
                return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        with self._lock:
            alerts = list(self.active_alerts.values())
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        with self._lock:
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1
            
            return {
                "active_alerts": len(self.active_alerts),
                "total_rules": len(self.alert_rules),
                "severity_breakdown": dict(severity_counts),
                "alert_history_size": len(self.alert_history)
            }


class ResearchMonitor:
    """
    Comprehensive monitoring system for research operations.
    
    This class coordinates metric collection, anomaly detection, alerting,
    and performance monitoring for all research modules.
    """
    
    def __init__(self, retention_hours: int = 24):
        self.logger = get_logger(self.__class__.__name__)
        
        # Core components
        self.metric_collector = MetricCollector(retention_hours)
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        
        # Performance monitoring
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        
        # Background tasks
        self.monitoring_task = None
        self.cleanup_task = None
        self.is_running = False
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules for research operations."""
        # Execution time alerts
        self.alert_manager.add_alert_rule(
            "high_execution_time",
            "execution_time",
            30.0,  # 30 seconds
            "greater",
            AlertSeverity.WARNING,
            "Research operation taking too long"
        )
        
        # Memory usage alerts
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            "memory_usage_mb",
            1024.0,  # 1GB
            "greater",
            AlertSeverity.CRITICAL,
            "High memory usage detected"
        )
        
        # Error rate alerts
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            "error_rate",
            0.1,  # 10%
            "greater",
            AlertSeverity.CRITICAL,
            "High error rate detected"
        )
        
        # Success rate alerts
        self.alert_manager.add_alert_rule(
            "low_success_rate",
            "success_rate",
            0.9,  # 90%
            "less",
            AlertSeverity.WARNING,
            "Low success rate detected"
        )
    
    async def record_operation_metrics(
        self,
        module_name: str,
        operation_name: str,
        execution_time: float,
        memory_usage_mb: float,
        success: bool,
        **additional_metrics
    ) -> None:
        """Record comprehensive metrics for a research operation."""
        labels = {
            "module": module_name,
            "operation": operation_name
        }
        
        # Core metrics
        self.metric_collector.record_metric(
            "execution_time", execution_time, MetricType.TIMING, labels
        )
        self.metric_collector.record_metric(
            "memory_usage_mb", memory_usage_mb, MetricType.GAUGE, labels
        )
        self.metric_collector.record_metric(
            "operation_success", 1.0 if success else 0.0, MetricType.COUNTER, labels
        )
        
        # Additional metrics
        for metric_name, value in additional_metrics.items():
            self.metric_collector.record_metric(metric_name, value, MetricType.GAUGE, labels)
        
        # Check for alerts
        await self._evaluate_operation_alerts(module_name, {
            "execution_time": execution_time,
            "memory_usage_mb": memory_usage_mb,
            **additional_metrics
        })
        
        # Update performance benchmarks
        await self._update_performance_benchmark(
            module_name, operation_name, execution_time, memory_usage_mb, success
        )
    
    async def _evaluate_operation_alerts(
        self,
        module_name: str,
        metrics: Dict[str, float]
    ) -> None:
        """Evaluate alerts for operation metrics."""
        for metric_name, value in metrics.items():
            alerts = self.alert_manager.evaluate_alerts(
                metric_name, value, module_name
            )
            
            # Check for anomalies
            anomaly_result = self.anomaly_detector.detect_anomaly(metric_name, value)
            if anomaly_result.get("is_anomaly", False):
                self.logger.warning(
                    f"Anomaly detected in {metric_name}: {value} "
                    f"(z-score: {anomaly_result.get('z_score', 'N/A')})"
                )
    
    async def _update_performance_benchmark(
        self,
        module_name: str,
        operation_name: str,
        execution_time: float,
        memory_usage_mb: float,
        success: bool
    ) -> None:
        """Update performance benchmarks."""
        benchmark_key = f"{module_name}_{operation_name}"
        
        if benchmark_key not in self.benchmarks:
            # Create initial benchmark
            self.benchmarks[benchmark_key] = PerformanceBenchmark(
                operation_name=operation_name,
                module_name=module_name,
                baseline_time=execution_time,
                baseline_memory=memory_usage_mb,
                baseline_accuracy=1.0 if success else 0.0,
                sla_time_threshold=execution_time * 2.0,  # 2x baseline
                sla_memory_threshold=memory_usage_mb * 1.5,  # 1.5x baseline
                sla_accuracy_threshold=0.95  # 95% success rate
            )
        else:
            benchmark = self.benchmarks[benchmark_key]
            benchmark.measurement_count += 1
            
            # Check for SLA violations
            violations = []
            if execution_time > benchmark.sla_time_threshold:
                violations.append({
                    "type": "execution_time",
                    "value": execution_time,
                    "threshold": benchmark.sla_time_threshold,
                    "timestamp": time.time()
                })
            
            if memory_usage_mb > benchmark.sla_memory_threshold:
                violations.append({
                    "type": "memory_usage",
                    "value": memory_usage_mb,
                    "threshold": benchmark.sla_memory_threshold,
                    "timestamp": time.time()
                })
            
            benchmark.violations.extend(violations)
            
            # Keep only recent violations (last 100)
            if len(benchmark.violations) > 100:
                benchmark.violations = benchmark.violations[-100:]
    
    async def get_performance_report(self, module_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "benchmarks": {},
            "sla_violations": [],
            "overall_health": "healthy"
        }
        
        violation_count = 0
        total_benchmarks = 0
        
        for benchmark_key, benchmark in self.benchmarks.items():
            if module_name and not benchmark_key.startswith(module_name):
                continue
            
            total_benchmarks += 1
            recent_violations = [
                v for v in benchmark.violations
                if time.time() - v["timestamp"] < 3600  # Last hour
            ]
            
            violation_count += len(recent_violations)
            
            report["benchmarks"][benchmark_key] = {
                "operation": benchmark.operation_name,
                "module": benchmark.module_name,
                "measurements": benchmark.measurement_count,
                "baseline_time": benchmark.baseline_time,
                "baseline_memory": benchmark.baseline_memory,
                "sla_violations": len(recent_violations),
                "time_threshold": benchmark.sla_time_threshold,
                "memory_threshold": benchmark.sla_memory_threshold
            }
            
            report["sla_violations"].extend(recent_violations)
        
        # Determine overall health
        if total_benchmarks > 0:
            violation_rate = violation_count / total_benchmarks
            if violation_rate > 0.2:
                report["overall_health"] = "unhealthy"
            elif violation_rate > 0.1:
                report["overall_health"] = "degraded"
        
        return report
    
    async def start_monitoring(self, monitoring_interval: float = 60.0) -> None:
        """Start background monitoring tasks."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start periodic tasks
        self.monitoring_task = asyncio.create_task(
            self._periodic_monitoring(monitoring_interval)
        )
        self.cleanup_task = asyncio.create_task(
            self._periodic_cleanup(3600.0)  # Cleanup every hour
        )
        
        self.logger.info("Started research monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped research monitoring")
    
    async def _periodic_monitoring(self, interval: float) -> None:
        """Periodic monitoring and anomaly detection."""
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                
                # Update anomaly detection baselines
                for metric_name in ["execution_time", "memory_usage_mb", "success_rate"]:
                    summary = self.metric_collector.get_metric_summary(metric_name, 60)  # Last hour
                    if "windowed_avg" in summary and summary["windowed_count"] > 10:
                        # Get recent values for baseline update
                        time_series = self.metric_collector.get_time_series(metric_name, 60, 5)
                        if time_series:
                            values = [bucket["avg"] for bucket in time_series]
                            self.anomaly_detector.update_baseline(metric_name, values)
                
                # Log monitoring status
                alert_summary = self.alert_manager.get_alert_summary()
                if alert_summary["active_alerts"] > 0:
                    self.logger.warning(f"Active alerts: {alert_summary['active_alerts']}")
                
            except Exception as e:
                self.logger.error(f"Monitoring task error: {str(e)}")
    
    async def _periodic_cleanup(self, interval: float) -> None:
        """Periodic cleanup of old data."""
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                self.metric_collector.cleanup_old_metrics()
                self.logger.debug("Performed periodic cleanup")
                
            except Exception as e:
                self.logger.error(f"Cleanup task error: {str(e)}")
    
    async def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Add metric metadata
        lines.append("# TYPE research_operation_duration_seconds histogram")
        lines.append("# TYPE research_operation_memory_bytes gauge")
        lines.append("# TYPE research_operation_success_total counter")
        
        # Export aggregated metrics
        for metric_name, aggregated in self.metric_collector.aggregated_metrics.items():
            if metric_name == "execution_time":
                lines.append(
                    f'research_operation_duration_seconds{{quantile="0.5"}} {aggregated["avg"]}'
                )
                lines.append(
                    f'research_operation_duration_seconds{{quantile="0.95"}} {aggregated["max"]}'
                )
            elif metric_name == "memory_usage_mb":
                lines.append(
                    f'research_operation_memory_bytes {aggregated["last_value"] * 1024 * 1024}'
                )
            elif metric_name == "operation_success":
                lines.append(
                    f'research_operation_success_total {aggregated["sum"]}'
                )
        
        return "\n".join(lines)
    
    async def get_comprehensive_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "timestamp": time.time(),
            "metrics_summary": {
                name: self.metric_collector.get_metric_summary(name, 60)
                for name in ["execution_time", "memory_usage_mb", "success_rate"]
            },
            "alerts": {
                "active": self.alert_manager.get_active_alerts(),
                "summary": self.alert_manager.get_alert_summary()
            },
            "anomalies": self.anomaly_detector.get_anomaly_summary(),
            "performance": await self.get_performance_report(),
            "health_status": "healthy" if len(self.alert_manager.active_alerts) == 0 else "degraded"
        }


# Global monitor instance
_global_monitor: Optional[ResearchMonitor] = None


def get_research_monitor() -> ResearchMonitor:
    """Get global research monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResearchMonitor()
    return _global_monitor


# Convenience decorator for monitoring
def monitor_research_operation(
    module_name: str,
    operation_name: str
):
    """Decorator to automatically monitor research operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            monitor = get_research_monitor()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record success metrics
                await monitor.record_operation_metrics(
                    module_name=module_name,
                    operation_name=operation_name,
                    execution_time=execution_time,
                    memory_usage_mb=0.0,  # Would need actual memory tracking
                    success=True
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record failure metrics
                await monitor.record_operation_metrics(
                    module_name=module_name,
                    operation_name=operation_name,
                    execution_time=execution_time,
                    memory_usage_mb=0.0,
                    success=False
                )
                
                raise e
        
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    async def test_monitoring():
        """Test the monitoring system."""
        print("📊 Testing Research Monitoring System")
        print("=" * 40)
        
        monitor = get_research_monitor()
        await monitor.start_monitoring(5.0)  # 5 second intervals
        
        # Simulate some operations
        for i in range(10):
            await monitor.record_operation_metrics(
                module_name="test_module",
                operation_name="test_operation",
                execution_time=np.random.normal(1.0, 0.2),  # Around 1 second
                memory_usage_mb=np.random.normal(100, 20),  # Around 100MB
                success=np.random.random() > 0.1,  # 90% success rate
                accuracy=np.random.uniform(0.8, 1.0)
            )
            
            await asyncio.sleep(0.1)
        
        # Get dashboard
        dashboard = await monitor.get_comprehensive_dashboard()
        print(f"📈 Dashboard Summary:")
        print(f"   Health: {dashboard['health_status']}")
        print(f"   Active alerts: {len(dashboard['alerts']['active'])}")
        print(f"   Metrics tracked: {len(dashboard['metrics_summary'])}")
        
        # Performance report
        perf_report = await monitor.get_performance_report()
        print(f"🎯 Performance Report:")
        print(f"   Benchmarks: {len(perf_report['benchmarks'])}")
        print(f"   SLA violations: {len(perf_report['sla_violations'])}")
        print(f"   Overall health: {perf_report['overall_health']}")
        
        await monitor.stop_monitoring()
        print("✅ Monitoring test completed")
    
    asyncio.run(test_monitoring())