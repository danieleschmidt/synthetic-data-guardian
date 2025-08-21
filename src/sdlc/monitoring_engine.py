"""
Monitoring Engine - Comprehensive monitoring, metrics, and observability
"""

import asyncio
import time
import json
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import gc

from ..utils.logger import get_logger


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Metric data point."""
    name: str
    type: MetricType
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class Alert:
    """Alert definition."""
    name: str
    condition: str
    severity: AlertSeverity
    threshold: float
    description: str = ""
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes
    last_triggered: float = 0.0


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable
    enabled: bool = True
    timeout_seconds: int = 30
    interval_seconds: int = 60
    last_run: float = 0.0
    last_result: Optional[Dict[str, Any]] = None


class MonitoringEngine:
    """
    Monitoring Engine - Comprehensive monitoring and observability.
    
    Provides enterprise-grade monitoring including:
    - Metrics collection and aggregation
    - Health checks and probes
    - Alerting and notifications
    - Performance monitoring
    - Resource usage tracking
    - Distributed tracing
    - Real-time dashboards
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        """Initialize monitoring engine."""
        self.config = config or {}
        self.logger = logger or get_logger(self.__class__.__name__)
        
        # Metrics storage
        self.metrics: List[Metric] = []
        self.metric_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status = {'overall': 'healthy', 'checks': {}}
        
        # Alerts
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.performance_metrics = {
            'request_count': 0,
            'request_duration_sum': 0.0,
            'error_count': 0,
            'last_request_time': 0.0
        }
        
        # Resource monitoring
        self.resource_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.collection_interval = self.config.get('collection_interval', 60)  # seconds
        self.retention_hours = self.config.get('retention_hours', 24)
        
        # Threading
        self._lock = threading.RLock()
        self._background_task = None
        
        self._initialize_default_health_checks()
        self._initialize_default_alerts()
        
        self.logger.info("Monitoring Engine initialized")
    
    def _initialize_default_health_checks(self) -> None:
        """Initialize default health checks."""
        self.health_checks.update({
            'memory_usage': HealthCheck(
                name="Memory Usage",
                check_function=self._check_memory_usage,
                interval_seconds=30
            ),
            'cpu_usage': HealthCheck(
                name="CPU Usage",
                check_function=self._check_cpu_usage,
                interval_seconds=30
            ),
            'disk_usage': HealthCheck(
                name="Disk Usage",
                check_function=self._check_disk_usage,
                interval_seconds=60
            ),
            'system_load': HealthCheck(
                name="System Load",
                check_function=self._check_system_load,
                interval_seconds=30
            ),
            'process_health': HealthCheck(
                name="Process Health",
                check_function=self._check_process_health,
                interval_seconds=30
            )
        })
    
    def _initialize_default_alerts(self) -> None:
        """Initialize default alerts."""
        self.alerts.update({
            'high_memory_usage': Alert(
                name="High Memory Usage",
                condition="memory_percent > threshold",
                severity=AlertSeverity.WARNING,
                threshold=85.0,
                description="Memory usage is above 85%"
            ),
            'critical_memory_usage': Alert(
                name="Critical Memory Usage",
                condition="memory_percent > threshold",
                severity=AlertSeverity.CRITICAL,
                threshold=95.0,
                description="Memory usage is critically high"
            ),
            'high_cpu_usage': Alert(
                name="High CPU Usage",
                condition="cpu_percent > threshold",
                severity=AlertSeverity.WARNING,
                threshold=80.0,
                description="CPU usage is above 80%"
            ),
            'high_error_rate': Alert(
                name="High Error Rate",
                condition="error_rate > threshold",
                severity=AlertSeverity.ERROR,
                threshold=5.0,
                description="Error rate is above 5%"
            ),
            'slow_response_time': Alert(
                name="Slow Response Time",
                condition="avg_response_time > threshold",
                severity=AlertSeverity.WARNING,
                threshold=2000.0,  # 2 seconds
                description="Average response time is above 2 seconds"
            )
        })
    
    async def start_monitoring(self) -> None:
        """Start the monitoring engine."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._background_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Monitoring engine started")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring engine."""
        self.monitoring_active = False
        
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Monitoring engine stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Run health checks
                await self._run_health_checks()
                
                # Evaluate alerts
                await self._evaluate_alerts()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Short delay before retrying
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        current_time = time.time()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            await self.record_gauge('system_cpu_percent', cpu_percent, {
                'description': 'CPU usage percentage'
            })
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.record_gauge('system_memory_percent', memory.percent, {
                'description': 'Memory usage percentage'
            })
            await self.record_gauge('system_memory_available_bytes', memory.available, {
                'description': 'Available memory in bytes'
            })
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self.record_gauge('system_disk_percent', disk_percent, {
                'description': 'Disk usage percentage'
            })
            
            # Network metrics
            network = psutil.net_io_counters()
            await self.record_counter('system_network_bytes_sent', network.bytes_sent, {
                'description': 'Total bytes sent over network'
            })
            await self.record_counter('system_network_bytes_recv', network.bytes_recv, {
                'description': 'Total bytes received over network'
            })
            
            # Process metrics
            process = psutil.Process()
            await self.record_gauge('process_memory_rss_bytes', process.memory_info().rss, {
                'description': 'Process resident memory in bytes'
            })
            await self.record_gauge('process_cpu_percent', process.cpu_percent(), {
                'description': 'Process CPU usage percentage'
            })
            
            # Python-specific metrics
            await self.record_gauge('python_gc_objects_collected', sum(gc.get_count()), {
                'description': 'Total objects in garbage collector'
            })
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def record_counter(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a counter metric."""
        with self._lock:
            self.counters[name] += value
            
            metric = Metric(
                name=name,
                type=MetricType.COUNTER,
                value=self.counters[name],
                timestamp=time.time(),
                labels=labels or {},
                description=labels.get('description', '') if labels else ''
            )
            
            self.metrics.append(metric)
            self.metric_series[name].append(metric)
    
    async def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a gauge metric."""
        with self._lock:
            self.gauges[name] = value
            
            metric = Metric(
                name=name,
                type=MetricType.GAUGE,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                description=labels.get('description', '') if labels else ''
            )
            
            self.metrics.append(metric)
            self.metric_series[name].append(metric)
    
    async def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a histogram metric."""
        # Simplified histogram implementation
        with self._lock:
            metric = Metric(
                name=name,
                type=MetricType.HISTOGRAM,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                description=labels.get('description', '') if labels else ''
            )
            
            self.metrics.append(metric)
            self.metric_series[name].append(metric)
    
    async def _run_health_checks(self) -> None:
        """Run all enabled health checks."""
        current_time = time.time()
        overall_healthy = True
        
        for name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
            
            # Check if it's time to run this health check
            if current_time - health_check.last_run < health_check.interval_seconds:
                continue
            
            try:
                # Run health check with timeout
                result = await asyncio.wait_for(
                    health_check.check_function(),
                    timeout=health_check.timeout_seconds
                )
                
                health_check.last_run = current_time
                health_check.last_result = result
                
                # Update health status
                self.health_status['checks'][name] = result
                
                if not result.get('healthy', False):
                    overall_healthy = False
                
            except asyncio.TimeoutError:
                self.logger.error(f"Health check '{name}' timed out")
                result = {'healthy': False, 'error': 'timeout'}
                health_check.last_result = result
                self.health_status['checks'][name] = result
                overall_healthy = False
                
            except Exception as e:
                self.logger.error(f"Health check '{name}' failed: {e}")
                result = {'healthy': False, 'error': str(e)}
                health_check.last_result = result
                self.health_status['checks'][name] = result
                overall_healthy = False
        
        self.health_status['overall'] = 'healthy' if overall_healthy else 'unhealthy'
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage health."""
        memory = psutil.virtual_memory()
        
        healthy = memory.percent < self.resource_thresholds['memory_percent']
        
        return {
            'healthy': healthy,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'threshold': self.resource_thresholds['memory_percent']
        }
    
    async def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage health."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        healthy = cpu_percent < self.resource_thresholds['cpu_percent']
        
        return {
            'healthy': healthy,
            'cpu_percent': cpu_percent,
            'threshold': self.resource_thresholds['cpu_percent']
        }
    
    async def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage health."""
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        healthy = disk_percent < self.resource_thresholds['disk_percent']
        
        return {
            'healthy': healthy,
            'disk_percent': disk_percent,
            'disk_free_gb': disk.free / (1024**3),
            'threshold': self.resource_thresholds['disk_percent']
        }
    
    async def _check_system_load(self) -> Dict[str, Any]:
        """Check system load health."""
        load_avg = psutil.getloadavg()
        cpu_count = psutil.cpu_count()
        
        # Consider healthy if 15-min load average is below CPU count
        healthy = load_avg[2] < cpu_count
        
        return {
            'healthy': healthy,
            'load_1min': load_avg[0],
            'load_5min': load_avg[1],
            'load_15min': load_avg[2],
            'cpu_count': cpu_count
        }
    
    async def _check_process_health(self) -> Dict[str, Any]:
        """Check process health."""
        try:
            process = psutil.Process()
            
            # Check if process is running and responsive
            status = process.status()
            healthy = status in ['running', 'sleeping']
            
            return {
                'healthy': healthy,
                'status': status,
                'pid': process.pid,
                'create_time': process.create_time(),
                'num_threads': process.num_threads()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def _evaluate_alerts(self) -> None:
        """Evaluate all enabled alerts."""
        current_time = time.time()
        
        for alert_name, alert in self.alerts.items():
            if not alert.enabled:
                continue
            
            # Check cooldown period
            if current_time - alert.last_triggered < alert.cooldown_seconds:
                continue
            
            try:
                # Evaluate alert condition
                should_trigger = await self._evaluate_alert_condition(alert)
                
                if should_trigger:
                    await self._trigger_alert(alert_name, alert)
                    alert.last_triggered = current_time
                    
            except Exception as e:
                self.logger.error(f"Error evaluating alert '{alert_name}': {e}")
    
    async def _evaluate_alert_condition(self, alert: Alert) -> bool:
        """Evaluate if an alert condition is met."""
        # Get current values for evaluation
        current_values = await self._get_current_metric_values()
        
        # Simple condition evaluation (could be enhanced with a proper expression parser)
        if alert.condition == "memory_percent > threshold":
            return current_values.get('system_memory_percent', 0) > alert.threshold
            
        elif alert.condition == "cpu_percent > threshold":
            return current_values.get('system_cpu_percent', 0) > alert.threshold
            
        elif alert.condition == "error_rate > threshold":
            total_requests = self.performance_metrics['request_count']
            error_count = self.performance_metrics['error_count']
            error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
            return error_rate > alert.threshold
            
        elif alert.condition == "avg_response_time > threshold":
            request_count = self.performance_metrics['request_count']
            total_duration = self.performance_metrics['request_duration_sum']
            avg_response_time = (total_duration / request_count * 1000) if request_count > 0 else 0
            return avg_response_time > alert.threshold
        
        return False
    
    async def _get_current_metric_values(self) -> Dict[str, float]:
        """Get current values for all metrics."""
        return {
            **self.gauges,
            **self.counters
        }
    
    async def _trigger_alert(self, alert_name: str, alert: Alert) -> None:
        """Trigger an alert."""
        alert_data = {
            'name': alert_name,
            'severity': alert.severity.value,
            'description': alert.description,
            'threshold': alert.threshold,
            'timestamp': time.time(),
            'current_values': await self._get_current_metric_values()
        }
        
        self.active_alerts.append(alert_data)
        
        # Log alert
        self.logger.warning(
            f"ALERT TRIGGERED: {alert_name} ({alert.severity.value}) - {alert.description}"
        )
        
        # Record alert as metric
        await self.record_counter(f'alerts_triggered_total', 1, {
            'alert_name': alert_name,
            'severity': alert.severity.value
        })
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics and data."""
        current_time = time.time()
        cutoff_time = current_time - (self.retention_hours * 3600)
        
        # Clean up old metrics
        with self._lock:
            old_count = len(self.metrics)
            self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
            cleaned_count = old_count - len(self.metrics)
            
            if cleaned_count > 0:
                self.logger.debug(f"Cleaned up {cleaned_count} old metrics")
            
            # Clean up old alerts
            self.active_alerts = [
                alert for alert in self.active_alerts
                if current_time - alert['timestamp'] < 3600  # Keep alerts for 1 hour
            ]
    
    async def record_request(self, duration: float, success: bool, path: str = None) -> None:
        """Record request metrics."""
        current_time = time.time()
        
        self.performance_metrics['request_count'] += 1
        self.performance_metrics['request_duration_sum'] += duration
        self.performance_metrics['last_request_time'] = current_time
        
        if not success:
            self.performance_metrics['error_count'] += 1
        
        # Record as metrics
        await self.record_counter('http_requests_total', 1, {
            'status': 'success' if success else 'error',
            'path': path or 'unknown'
        })
        
        await self.record_histogram('http_request_duration_seconds', duration, {
            'path': path or 'unknown'
        })
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            'overall_status': self.health_status['overall'],
            'timestamp': time.time(),
            'checks': self.health_status['checks'].copy()
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        current_time = time.time()
        
        # Calculate performance metrics
        total_requests = self.performance_metrics['request_count']
        total_duration = self.performance_metrics['request_duration_sum']
        error_count = self.performance_metrics['error_count']
        
        avg_response_time = (total_duration / total_requests) if total_requests > 0 else 0
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'timestamp': current_time,
            'total_metrics': len(self.metrics),
            'active_alerts': len(self.active_alerts),
            'performance': {
                'total_requests': total_requests,
                'average_response_time_seconds': avg_response_time,
                'error_rate_percent': error_rate,
                'requests_per_minute': self._calculate_requests_per_minute()
            },
            'system_resources': {
                'cpu_percent': self.gauges.get('system_cpu_percent', 0),
                'memory_percent': self.gauges.get('system_memory_percent', 0),
                'disk_percent': self.gauges.get('system_disk_percent', 0)
            },
            'health_status': self.health_status['overall']
        }
    
    def _calculate_requests_per_minute(self) -> float:
        """Calculate requests per minute over the last minute."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Count requests in the last minute from metrics
        recent_requests = [
            m for m in self.metrics
            if m.name == 'http_requests_total' and m.timestamp > minute_ago
        ]
        
        return len(recent_requests)
    
    def get_metric_series(self, metric_name: str, duration_hours: int = 1) -> List[Dict[str, Any]]:
        """Get time series data for a specific metric."""
        current_time = time.time()
        cutoff_time = current_time - (duration_hours * 3600)
        
        if metric_name not in self.metric_series:
            return []
        
        series_data = []
        for metric in self.metric_series[metric_name]:
            if metric.timestamp > cutoff_time:
                series_data.append({
                    'timestamp': metric.timestamp,
                    'value': metric.value,
                    'labels': metric.labels
                })
        
        return sorted(series_data, key=lambda x: x['timestamp'])
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        health_status = self.get_health_status()
        metrics_summary = self.get_metrics_summary()
        
        # Get recent time series for key metrics
        key_metrics = [
            'system_cpu_percent',
            'system_memory_percent', 
            'system_disk_percent',
            'http_requests_total'
        ]
        
        time_series = {}
        for metric in key_metrics:
            time_series[metric] = self.get_metric_series(metric, duration_hours=1)
        
        # Get recent alerts
        current_time = time.time()
        recent_alerts = [
            alert for alert in self.active_alerts
            if current_time - alert['timestamp'] < 3600  # Last hour
        ]
        
        return {
            'health_status': health_status,
            'metrics_summary': metrics_summary,
            'time_series': time_series,
            'recent_alerts': recent_alerts,
            'alert_counts_by_severity': self._count_alerts_by_severity(recent_alerts),
            'top_error_paths': self._get_top_error_paths()
        }
    
    def _count_alerts_by_severity(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count alerts by severity level."""
        counts = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in alerts:
            severity = alert.get('severity', 'info')
            counts[severity] = counts.get(severity, 0) + 1
        
        return counts
    
    def _get_top_error_paths(self) -> List[Dict[str, Any]]:
        """Get top error paths from metrics."""
        current_time = time.time()
        hour_ago = current_time - 3600
        
        error_counts = defaultdict(int)
        
        # Count errors by path
        for metric in self.metrics:
            if (metric.name == 'http_requests_total' and 
                metric.timestamp > hour_ago and
                metric.labels.get('status') == 'error'):
                
                path = metric.labels.get('path', 'unknown')
                error_counts[path] += 1
        
        # Sort by count and return top 10
        top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [{'path': path, 'error_count': count} for path, count in top_errors]
    
    async def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        health_status = self.get_health_status()
        metrics_summary = self.get_metrics_summary()
        dashboard_data = self.get_dashboard_data()
        
        # Calculate uptime
        current_time = time.time()
        if self.metrics:
            first_metric_time = min(m.timestamp for m in self.metrics)
            uptime_hours = (current_time - first_metric_time) / 3600
        else:
            uptime_hours = 0
        
        # Performance analysis
        performance_score = 100.0
        if metrics_summary['performance']['error_rate_percent'] > 5:
            performance_score -= 30
        if metrics_summary['performance']['average_response_time_seconds'] > 2:
            performance_score -= 20
        if metrics_summary['system_resources']['cpu_percent'] > 80:
            performance_score -= 20
        if metrics_summary['system_resources']['memory_percent'] > 85:
            performance_score -= 20
        
        performance_score = max(performance_score, 0)
        
        return {
            'report_timestamp': current_time,
            'uptime_hours': uptime_hours,
            'health_status': health_status,
            'metrics_summary': metrics_summary,
            'performance_score': performance_score,
            'alert_summary': {
                'total_active_alerts': len(self.active_alerts),
                'alerts_by_severity': dashboard_data['alert_counts_by_severity'],
                'recent_alerts': dashboard_data['recent_alerts']
            },
            'resource_utilization': {
                'cpu_status': 'healthy' if metrics_summary['system_resources']['cpu_percent'] < 80 else 'warning',
                'memory_status': 'healthy' if metrics_summary['system_resources']['memory_percent'] < 85 else 'warning',
                'disk_status': 'healthy' if metrics_summary['system_resources']['disk_percent'] < 90 else 'warning'
            },
            'recommendations': self._generate_monitoring_recommendations(metrics_summary, health_status)
        }
    
    def _generate_monitoring_recommendations(self, metrics: Dict[str, Any], health: Dict[str, Any]) -> List[str]:
        """Generate monitoring recommendations."""
        recommendations = []
        
        # Resource recommendations
        if metrics['system_resources']['cpu_percent'] > 80:
            recommendations.append("High CPU usage detected - consider scaling or optimization")
        
        if metrics['system_resources']['memory_percent'] > 85:
            recommendations.append("High memory usage detected - investigate memory leaks")
        
        if metrics['system_resources']['disk_percent'] > 90:
            recommendations.append("Low disk space - clean up old files or expand storage")
        
        # Performance recommendations
        if metrics['performance']['error_rate_percent'] > 5:
            recommendations.append("High error rate detected - investigate and fix errors")
        
        if metrics['performance']['average_response_time_seconds'] > 2:
            recommendations.append("Slow response times - optimize performance or scale resources")
        
        # Health recommendations
        if health['overall_status'] != 'healthy':
            recommendations.append("Health checks failing - investigate unhealthy components")
        
        # General recommendations
        recommendations.extend([
            "Set up automated alerting for critical metrics",
            "Implement log aggregation for better observability",
            "Create custom dashboards for key business metrics",
            "Regularly review and tune alert thresholds"
        ])
        
        return recommendations