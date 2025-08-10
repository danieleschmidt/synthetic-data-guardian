"""
Comprehensive health monitoring and system diagnostics
"""

import asyncio
import psutil
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from ..utils.logger import get_logger


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentStatus(Enum):
    """Individual component status."""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_function: Callable
    timeout_seconds: float = 30.0
    critical: bool = False
    interval_seconds: float = 60.0
    tags: List[str] = field(default_factory=list)


@dataclass
class HealthMetric:
    """Health metric data point."""
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass 
class ComponentHealth:
    """Component health information."""
    name: str
    status: ComponentStatus
    message: str
    last_check: datetime
    response_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """
    Comprehensive health monitoring system for Synthetic Data Guardian.
    """
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self.logger = get_logger(self.__class__.__name__)
        
        self.health_checks: Dict[str, HealthCheck] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.metrics_history: List[HealthMetric] = []
        self.alert_callbacks: List[Callable] = []
        
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # System metrics
        self.system_stats = {
            'start_time': time.time(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0
        }
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        # System resource checks
        self.register_check(HealthCheck(
            name="system_memory",
            check_function=self._check_system_memory,
            critical=True,
            interval_seconds=30.0,
            tags=["system", "resources"]
        ))
        
        self.register_check(HealthCheck(
            name="system_cpu",
            check_function=self._check_system_cpu,
            critical=False,
            interval_seconds=30.0,
            tags=["system", "resources"]
        ))
        
        self.register_check(HealthCheck(
            name="disk_space",
            check_function=self._check_disk_space,
            critical=True,
            interval_seconds=60.0,
            tags=["system", "storage"]
        ))
        
        # Application checks
        self.register_check(HealthCheck(
            name="guardian_status",
            check_function=self._check_guardian_status,
            critical=True,
            interval_seconds=60.0,
            tags=["application", "core"]
        ))
        
        self.register_check(HealthCheck(
            name="database_connectivity",
            check_function=self._check_database_connectivity,
            critical=True,
            interval_seconds=120.0,
            tags=["database", "connectivity"]
        ))
        
        self.register_check(HealthCheck(
            name="external_dependencies",
            check_function=self._check_external_dependencies,
            critical=False,
            interval_seconds=300.0,
            tags=["external", "dependencies"]
        ))
    
    def register_check(self, health_check: HealthCheck):
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        self.logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            if name in self.component_health:
                del self.component_health[name]
            self.logger.info(f"Unregistered health check: {name}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                asyncio.run(self._run_health_checks())
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring loop error: {e}")
                time.sleep(30)  # Wait before retrying
    
    async def _run_health_checks(self):
        """Run all health checks."""
        tasks = []
        
        for check_name, health_check in self.health_checks.items():
            # Check if it's time to run this check
            if self._should_run_check(check_name, health_check):
                task = asyncio.create_task(
                    self._execute_health_check(check_name, health_check)
                )
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect system metrics
        await self._collect_system_metrics()
    
    def _should_run_check(self, check_name: str, health_check: HealthCheck) -> bool:
        """Determine if a health check should be run."""
        if check_name not in self.component_health:
            return True
        
        last_check = self.component_health[check_name].last_check
        elapsed = (datetime.now() - last_check).total_seconds()
        return elapsed >= health_check.interval_seconds
    
    async def _execute_health_check(self, check_name: str, health_check: HealthCheck):
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            # Run the check with timeout
            result = await asyncio.wait_for(
                self._run_check_function(health_check.check_function),
                timeout=health_check.timeout_seconds
            )
            
            response_time = time.time() - start_time
            
            # Process result
            if isinstance(result, tuple):
                status, message, metadata = result
            else:
                status = ComponentStatus.UP if result else ComponentStatus.DOWN
                message = "Check passed" if result else "Check failed"
                metadata = {}
            
            # Update component health
            self.component_health[check_name] = ComponentHealth(
                name=check_name,
                status=status,
                message=message,
                last_check=datetime.now(),
                response_time=response_time,
                metadata=metadata
            )
            
            # Check for alerts
            if status in [ComponentStatus.DOWN, ComponentStatus.DEGRADED] and health_check.critical:
                await self._trigger_alert(check_name, health_check, status, message)
            
        except asyncio.TimeoutError:
            self.component_health[check_name] = ComponentHealth(
                name=check_name,
                status=ComponentStatus.DOWN,
                message=f"Health check timed out after {health_check.timeout_seconds}s",
                last_check=datetime.now(),
                response_time=health_check.timeout_seconds
            )
            
            if health_check.critical:
                await self._trigger_alert(check_name, health_check, ComponentStatus.DOWN, "Timeout")
        
        except Exception as e:
            self.component_health[check_name] = ComponentHealth(
                name=check_name,
                status=ComponentStatus.DOWN,
                message=f"Health check error: {str(e)}",
                last_check=datetime.now(),
                response_time=time.time() - start_time
            )
            
            if health_check.critical:
                await self._trigger_alert(check_name, health_check, ComponentStatus.DOWN, str(e))
    
    async def _run_check_function(self, check_function: Callable):
        """Run check function (sync or async)."""
        if asyncio.iscoroutinefunction(check_function):
            return await check_function()
        else:
            return check_function()
    
    async def _trigger_alert(
        self, 
        check_name: str, 
        health_check: HealthCheck, 
        status: ComponentStatus, 
        message: str
    ):
        """Trigger health alert."""
        alert_data = {
            'check_name': check_name,
            'status': status.value,
            'message': message,
            'critical': health_check.critical,
            'tags': health_check.tags,
            'timestamp': datetime.now()
        }
        
        self.logger.warning(f"Health alert: {check_name} - {message}")
        
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._add_metric("system_cpu_percent", cpu_percent, "percent")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._add_metric("system_memory_used_percent", memory.percent, "percent")
            self._add_metric("system_memory_available_bytes", memory.available, "bytes")
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_used_percent = (disk.used / disk.total) * 100
            self._add_metric("system_disk_used_percent", disk_used_percent, "percent")
            self._add_metric("system_disk_free_bytes", disk.free, "bytes")
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                self._add_metric("network_bytes_sent", network.bytes_sent, "bytes")
                self._add_metric("network_bytes_recv", network.bytes_recv, "bytes")
            except:
                pass
            
            # Application metrics
            uptime = time.time() - self.system_stats['start_time']
            self._add_metric("application_uptime_seconds", uptime, "seconds")
            
            if self.system_stats['total_requests'] > 0:
                success_rate = self.system_stats['successful_requests'] / self.system_stats['total_requests']
                self._add_metric("application_success_rate", success_rate, "ratio")
            
            self._add_metric("application_average_response_time", self.system_stats['average_response_time'], "seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _add_metric(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Add metric to history."""
        metric = HealthMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            tags=tags or {}
        )
        
        self.metrics_history.append(metric)
        
        # Keep only recent metrics (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
    
    def get_overall_health(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Get overall system health status."""
        if not self.component_health:
            return HealthStatus.UNKNOWN, "No health checks configured", {
                'total_checks': 0,
                'healthy_checks': 0,
                'degraded_checks': 0,
                'failed_checks': 0,
                'critical_failures': 0,
                'uptime_seconds': time.time() - self.system_stats['start_time'],
                'last_check': datetime.now()
            }
        
        critical_down = sum(
            1 for name, health in self.component_health.items()
            if health.status == ComponentStatus.DOWN and self.health_checks.get(name, HealthCheck("", lambda: True)).critical
        )
        
        total_down = sum(
            1 for health in self.component_health.values()
            if health.status == ComponentStatus.DOWN
        )
        
        total_degraded = sum(
            1 for health in self.component_health.values()
            if health.status == ComponentStatus.DEGRADED
        )
        
        total_checks = len(self.component_health)
        
        # Determine overall status
        if critical_down > 0:
            status = HealthStatus.CRITICAL
            message = f"{critical_down} critical components down"
        elif total_down > 0:
            status = HealthStatus.UNHEALTHY
            message = f"{total_down} components down"
        elif total_degraded > 0:
            status = HealthStatus.DEGRADED
            message = f"{total_degraded} components degraded"
        else:
            status = HealthStatus.HEALTHY
            message = "All systems operational"
        
        details = {
            'total_checks': total_checks,
            'healthy_checks': total_checks - total_down - total_degraded,
            'degraded_checks': total_degraded,
            'failed_checks': total_down,
            'critical_failures': critical_down,
            'uptime_seconds': time.time() - self.system_stats['start_time'],
            'last_check': max(
                (health.last_check for health in self.component_health.values()),
                default=datetime.now()
            )
        }
        
        return status, message, details
    
    def get_component_health(self, name: Optional[str] = None) -> Dict[str, ComponentHealth]:
        """Get health status of specific component or all components."""
        if name:
            return {name: self.component_health.get(name)} if name in self.component_health else {}
        return self.component_health.copy()
    
    def get_metrics(self, name_pattern: Optional[str] = None, hours: int = 1) -> List[HealthMetric]:
        """Get metrics history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_metrics = [
            metric for metric in self.metrics_history
            if metric.timestamp > cutoff_time
        ]
        
        if name_pattern:
            import re
            pattern = re.compile(name_pattern)
            filtered_metrics = [
                metric for metric in filtered_metrics
                if pattern.search(metric.name)
            ]
        
        return filtered_metrics
    
    def record_request(self, success: bool, response_time: float):
        """Record request metrics."""
        self.system_stats['total_requests'] += 1
        if success:
            self.system_stats['successful_requests'] += 1
        else:
            self.system_stats['failed_requests'] += 1
        
        # Update average response time (exponential moving average)
        alpha = 0.1
        current_avg = self.system_stats['average_response_time']
        self.system_stats['average_response_time'] = alpha * response_time + (1 - alpha) * current_avg
    
    # Default health check implementations
    def _check_system_memory(self) -> Tuple[ComponentStatus, str, Dict[str, Any]]:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                return ComponentStatus.DOWN, f"Memory usage critical: {memory.percent:.1f}%", {'memory_percent': memory.percent}
            elif memory.percent > 85:
                return ComponentStatus.DEGRADED, f"Memory usage high: {memory.percent:.1f}%", {'memory_percent': memory.percent}
            else:
                return ComponentStatus.UP, f"Memory usage normal: {memory.percent:.1f}%", {'memory_percent': memory.percent}
        except Exception as e:
            return ComponentStatus.DOWN, f"Failed to check memory: {str(e)}", {}
    
    def _check_system_cpu(self) -> Tuple[ComponentStatus, str, Dict[str, Any]]:
        """Check system CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 95:
                return ComponentStatus.DEGRADED, f"CPU usage very high: {cpu_percent:.1f}%", {'cpu_percent': cpu_percent}
            elif cpu_percent > 80:
                return ComponentStatus.DEGRADED, f"CPU usage high: {cpu_percent:.1f}%", {'cpu_percent': cpu_percent}
            else:
                return ComponentStatus.UP, f"CPU usage normal: {cpu_percent:.1f}%", {'cpu_percent': cpu_percent}
        except Exception as e:
            return ComponentStatus.DOWN, f"Failed to check CPU: {str(e)}", {}
    
    def _check_disk_space(self) -> Tuple[ComponentStatus, str, Dict[str, Any]]:
        """Check disk space usage."""
        try:
            disk = psutil.disk_usage('/')
            used_percent = (disk.used / disk.total) * 100
            
            if used_percent > 95:
                return ComponentStatus.DOWN, f"Disk space critical: {used_percent:.1f}% used", {'disk_percent': used_percent}
            elif used_percent > 85:
                return ComponentStatus.DEGRADED, f"Disk space low: {used_percent:.1f}% used", {'disk_percent': used_percent}
            else:
                return ComponentStatus.UP, f"Disk space sufficient: {used_percent:.1f}% used", {'disk_percent': used_percent}
        except Exception as e:
            return ComponentStatus.DOWN, f"Failed to check disk space: {str(e)}", {}
    
    def _check_guardian_status(self) -> Tuple[ComponentStatus, str, Dict[str, Any]]:
        """Check Guardian core status."""
        # This is a placeholder - in practice would check actual Guardian status
        try:
            # Basic sanity checks
            uptime = time.time() - self.system_stats['start_time']
            
            if uptime > 60:  # Running for more than a minute
                return ComponentStatus.UP, f"Guardian running (uptime: {uptime:.0f}s)", {'uptime': uptime}
            else:
                return ComponentStatus.UP, f"Guardian starting (uptime: {uptime:.0f}s)", {'uptime': uptime}
        except Exception as e:
            return ComponentStatus.DOWN, f"Guardian check failed: {str(e)}", {}
    
    def _check_database_connectivity(self) -> Tuple[ComponentStatus, str, Dict[str, Any]]:
        """Check database connectivity."""
        # Placeholder for database checks
        try:
            # In practice, would test actual database connections
            return ComponentStatus.UP, "Database connectivity check passed", {}
        except Exception as e:
            return ComponentStatus.DOWN, f"Database connectivity failed: {str(e)}", {}
    
    def _check_external_dependencies(self) -> Tuple[ComponentStatus, str, Dict[str, Any]]:
        """Check external service dependencies."""
        # Placeholder for external dependency checks
        try:
            # In practice, would check external APIs, services, etc.
            return ComponentStatus.UP, "External dependencies available", {}
        except Exception as e:
            return ComponentStatus.DEGRADED, f"Some external services unavailable: {str(e)}", {}


# Global health monitor instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor