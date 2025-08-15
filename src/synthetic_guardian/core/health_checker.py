"""
Health Checker - Comprehensive system health monitoring and diagnostics
"""

import asyncio
import time
import psutil
import socket
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path
import subprocess
import sys


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = None
    duration_ms: float = 0
    timestamp: float = 0
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.timestamp == 0:
            self.timestamp = time.time()


class HealthChecker:
    """
    Comprehensive health checker for the Synthetic Data Guardian system.
    Monitors system resources, dependencies, and service health.
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.checks: Dict[str, callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.last_check_time = 0
        self.check_interval = 60  # seconds
        self._lock = threading.RLock()
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("system_memory", self._check_memory)
        self.register_check("system_cpu", self._check_cpu)
        self.register_check("system_disk", self._check_disk)
        self.register_check("python_environment", self._check_python_env)
        self.register_check("network_connectivity", self._check_network)
        self.register_check("dependencies", self._check_dependencies)
    
    def register_check(self, name: str, check_function: callable):
        """Register a custom health check."""
        with self._lock:
            self.checks[name] = check_function
            self.logger.debug(f"Registered health check: {name}")
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        start_time = time.time()
        
        with self._lock:
            checks_to_run = self.checks.copy()
        
        results = {}
        
        # Run checks concurrently
        tasks = []
        for name, check_func in checks_to_run.items():
            task = asyncio.create_task(self._run_single_check(name, check_func))
            tasks.append(task)
        
        # Wait for all checks to complete
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for (name, _), result in zip(checks_to_run.items(), check_results):
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(result)}",
                    details={'error': str(result)}
                )
            else:
                results[name] = result
        
        with self._lock:
            self.results.update(results)
            self.last_check_time = time.time()
        
        total_duration = (time.time() - start_time) * 1000
        self.logger.info(f"Completed {len(results)} health checks in {total_duration:.1f}ms")
        
        return results
    
    async def _run_single_check(self, name: str, check_func: callable) -> HealthCheckResult:
        """Run a single health check with timing and error handling."""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheckResult):
                result.duration_ms = duration_ms
                return result
            else:
                # Convert simple return to HealthCheckResult
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="Check passed",
                    duration_ms=duration_ms,
                    details=result if isinstance(result, dict) else {}
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Health check {name} failed: {e}")
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
                details={'error': str(e), 'error_type': type(e).__name__}
            )
    
    def _check_memory(self) -> HealthCheckResult:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_percent = memory.percent
            swap_percent = swap.percent if swap.total > 0 else 0
            
            details = {
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_percent': memory_percent,
                'swap_total_gb': round(swap.total / (1024**3), 2),
                'swap_used_gb': round(swap.used / (1024**3), 2),
                'swap_percent': swap_percent
            }
            
            if memory_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory_percent:.1f}%"
            elif memory_percent > 80:
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return HealthCheckResult(
                name="system_memory",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_memory",
                status=HealthStatus.CRITICAL,
                message=f"Memory check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    def _check_cpu(self) -> HealthCheckResult:
        """Check CPU usage."""
        try:
            # Get CPU usage over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            details = {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'load_avg_1min': load_avg[0],
                'load_avg_5min': load_avg[1],
                'load_avg_15min': load_avg[2]
            }
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheckResult(
                name="system_cpu",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_cpu",
                status=HealthStatus.CRITICAL,
                message=f"CPU check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    def _check_disk(self) -> HealthCheckResult:
        """Check disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            details = {
                'disk_total_gb': round(disk_usage.total / (1024**3), 2),
                'disk_used_gb': round(disk_usage.used / (1024**3), 2),
                'disk_free_gb': round(disk_usage.free / (1024**3), 2),
                'disk_percent': round(disk_percent, 1)
            }
            
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical disk usage: {disk_percent:.1f}%"
            elif disk_percent > 85:
                status = HealthStatus.WARNING
                message = f"High disk usage: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            return HealthCheckResult(
                name="system_disk",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_disk",
                status=HealthStatus.CRITICAL,
                message=f"Disk check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    def _check_python_env(self) -> HealthCheckResult:
        """Check Python environment health."""
        try:
            details = {
                'python_version': sys.version,
                'python_executable': sys.executable,
                'python_path': sys.path[:3],  # First 3 entries
                'platform': sys.platform,
                'max_recursion_limit': sys.getrecursionlimit()
            }
            
            # Check Python version
            version_info = sys.version_info
            if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 9):
                status = HealthStatus.WARNING
                message = f"Python version may be outdated: {version_info.major}.{version_info.minor}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Python environment healthy: {version_info.major}.{version_info.minor}.{version_info.micro}"
            
            return HealthCheckResult(
                name="python_environment",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="python_environment",
                status=HealthStatus.CRITICAL,
                message=f"Python environment check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _check_network(self) -> HealthCheckResult:
        """Check network connectivity."""
        try:
            # Test DNS resolution
            loop = asyncio.get_event_loop()
            
            async def check_host(host, port=80):
                try:
                    future = loop.run_in_executor(None, socket.gethostbyname, host)
                    ip = await asyncio.wait_for(future, timeout=5)
                    return True, ip
                except Exception as e:
                    return False, str(e)
            
            # Check connectivity to common hosts
            hosts_to_check = ['google.com', 'github.com', 'pypi.org']
            results = {}
            
            for host in hosts_to_check:
                success, result = await check_host(host)
                results[host] = {'success': success, 'result': result}
            
            successful_checks = sum(1 for r in results.values() if r['success'])
            total_checks = len(results)
            
            details = {
                'connectivity_checks': results,
                'successful_checks': successful_checks,
                'total_checks': total_checks
            }
            
            if successful_checks == 0:
                status = HealthStatus.CRITICAL
                message = "No network connectivity detected"
            elif successful_checks < total_checks:
                status = HealthStatus.WARNING
                message = f"Partial network connectivity: {successful_checks}/{total_checks}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Network connectivity healthy: {successful_checks}/{total_checks}"
            
            return HealthCheckResult(
                name="network_connectivity",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="network_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Network check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    def _check_dependencies(self) -> HealthCheckResult:
        """Check critical dependencies."""
        try:
            critical_modules = [
                'asyncio', 'pathlib', 'logging', 'json', 'time',
                'psutil', 'threading', 'subprocess'
            ]
            
            optional_modules = [
                'pandas', 'numpy', 'torch', 'transformers',
                'fastapi', 'pydantic', 'redis', 'neo4j'
            ]
            
            critical_results = {}
            optional_results = {}
            
            # Check critical modules
            for module in critical_modules:
                try:
                    __import__(module)
                    critical_results[module] = True
                except ImportError:
                    critical_results[module] = False
            
            # Check optional modules  
            for module in optional_modules:
                try:
                    __import__(module)
                    optional_results[module] = True
                except ImportError:
                    optional_results[module] = False
            
            critical_available = sum(critical_results.values())
            critical_total = len(critical_results)
            optional_available = sum(optional_results.values())
            optional_total = len(optional_results)
            
            details = {
                'critical_modules': critical_results,
                'optional_modules': optional_results,
                'critical_available': f"{critical_available}/{critical_total}",
                'optional_available': f"{optional_available}/{optional_total}"
            }
            
            if critical_available < critical_total:
                status = HealthStatus.CRITICAL
                message = f"Missing critical dependencies: {critical_total - critical_available}"
            elif optional_available < optional_total * 0.5:
                status = HealthStatus.WARNING
                message = f"Many optional dependencies missing: {optional_total - optional_available}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Dependencies healthy: {critical_available}/{critical_total} critical, {optional_available}/{optional_total} optional"
            
            return HealthCheckResult(
                name="dependencies",
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Dependency check failed: {str(e)}",
                details={'error': str(e)}
            )
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        with self._lock:
            if not self.results:
                return HealthStatus.UNKNOWN
            
            statuses = [result.status for result in self.results.values()]
            
            if HealthStatus.CRITICAL in statuses:
                return HealthStatus.CRITICAL
            elif HealthStatus.WARNING in statuses:
                return HealthStatus.WARNING
            elif all(status == HealthStatus.HEALTHY for status in statuses):
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        with self._lock:
            overall_status = self.get_overall_status()
            
            summary = {
                'overall_status': overall_status.value,
                'last_check_time': self.last_check_time,
                'checks_count': len(self.results),
                'healthy_checks': len([r for r in self.results.values() if r.status == HealthStatus.HEALTHY]),
                'warning_checks': len([r for r in self.results.values() if r.status == HealthStatus.WARNING]),
                'critical_checks': len([r for r in self.results.values() if r.status == HealthStatus.CRITICAL]),
                'checks': {name: {
                    'status': result.status.value,
                    'message': result.message,
                    'duration_ms': result.duration_ms,
                    'timestamp': result.timestamp
                } for name, result in self.results.items()}
            }
            
            return summary
    
    async def continuous_monitoring(self, interval: int = None):
        """Start continuous health monitoring."""
        if interval:
            self.check_interval = interval
            
        self.logger.info(f"Starting continuous health monitoring (interval: {self.check_interval}s)")
        
        while True:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                self.logger.info("Health monitoring cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(self.check_interval)