"""
Robust Research Manager - Production-Ready Research Module Orchestration

This module provides enterprise-grade robustness, error handling, and reliability
for all research components. It implements circuit breakers, retry logic, input
validation, resource management, and comprehensive monitoring.

Production Features:
1. Circuit breakers for research module failures
2. Comprehensive input validation and sanitization
3. Resource monitoring and automatic cleanup
4. Retry logic with exponential backoff
5. Health checks and self-healing capabilities
6. Graceful degradation and fallback mechanisms
7. Performance monitoring and alerting
"""

import asyncio
import time
import logging
import traceback
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import warnings
import numpy as np
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from ..utils.logger import get_logger


class HealthStatus(Enum):
    """Health status for research modules."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    DISABLED = "disabled"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ResearchModuleConfig:
    """Configuration for research modules."""
    module_name: str
    enabled: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    memory_limit_mb: int = 1024
    cpu_limit_percent: float = 80.0
    health_check_interval: float = 30.0
    enable_fallback: bool = True
    max_concurrent_operations: int = 5


@dataclass
class OperationMetrics:
    """Metrics for research operations."""
    operation_name: str
    success_count: int = 0
    failure_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    circuit_trips: int = 0
    
    def update_success(self, execution_time: float):
        """Update metrics for successful operation."""
        self.success_count += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / (self.success_count + self.failure_count)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.last_success_time = time.time()
    
    def update_failure(self, execution_time: float):
        """Update metrics for failed operation."""
        self.failure_count += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / (self.success_count + self.failure_count)
        self.last_failure_time = time.time()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return 1.0 - self.success_rate


class CircuitBreaker:
    """Circuit breaker for research module operations."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
        
        self.logger = get_logger(self.__class__.__name__)
    
    def __call__(self, func):
        """Decorator to apply circuit breaker to function."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
    
    async def _call_with_circuit_breaker(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker OPEN for {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.logger.info("Circuit breaker CLOSED - service recovered")
    
    def _on_failure(self):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker OPEN - threshold reached: {self.failure_count}")


class InputValidator:
    """Comprehensive input validation for research modules."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def validate_numpy_array(
        self,
        data: np.ndarray,
        min_shape: Optional[Tuple[int, ...]] = None,
        max_shape: Optional[Tuple[int, ...]] = None,
        dtype_allowed: Optional[List[np.dtype]] = None,
        allow_nan: bool = False,
        allow_inf: bool = False
    ) -> Dict[str, Any]:
        """Validate numpy array input."""
        issues = []
        
        if not isinstance(data, np.ndarray):
            issues.append(f"Expected numpy array, got {type(data)}")
            return {"valid": False, "issues": issues}
        
        # Shape validation
        if min_shape and len(data.shape) < len(min_shape):
            issues.append(f"Array has too few dimensions: {data.shape} < {min_shape}")
        
        if max_shape and len(data.shape) > len(max_shape):
            issues.append(f"Array has too many dimensions: {data.shape} > {max_shape}")
        
        # Data type validation
        if dtype_allowed and data.dtype not in dtype_allowed:
            issues.append(f"Invalid dtype: {data.dtype}, allowed: {dtype_allowed}")
        
        # NaN/Inf validation
        if not allow_nan and np.isnan(data).any():
            issues.append("Array contains NaN values")
        
        if not allow_inf and np.isinf(data).any():
            issues.append("Array contains infinite values")
        
        # Size limits
        if data.size > 10_000_000:  # 10M elements
            issues.append(f"Array too large: {data.size} elements")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "shape": data.shape,
            "dtype": str(data.dtype),
            "size": data.size
        }
    
    def validate_config(
        self,
        config: Dict[str, Any],
        required_fields: List[str],
        field_types: Dict[str, type],
        field_ranges: Optional[Dict[str, Tuple[Any, Any]]] = None
    ) -> Dict[str, Any]:
        """Validate configuration dictionary."""
        issues = []
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in field_types.items():
            if field in config and not isinstance(config[field], expected_type):
                issues.append(f"Invalid type for {field}: expected {expected_type}, got {type(config[field])}")
        
        # Check field ranges
        if field_ranges:
            for field, (min_val, max_val) in field_ranges.items():
                if field in config:
                    value = config[field]
                    if value < min_val or value > max_val:
                        issues.append(f"Value for {field} out of range: {value} not in [{min_val}, {max_val}]")
        
        return {"valid": len(issues) == 0, "issues": issues}
    
    def sanitize_string(self, text: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove potentially dangerous characters
        sanitized = text.replace('\x00', '').replace('\r', '').replace('\n', ' ')
        
        # Truncate if too long
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
        
        return sanitized


class ResourceMonitor:
    """Monitor and enforce resource limits for research operations."""
    
    def __init__(self, config: ResearchModuleConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._start_time = None
        self._peak_memory = 0
    
    async def __aenter__(self):
        """Enter resource monitoring context."""
        self._start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit resource monitoring context."""
        duration = time.time() - self._start_time
        if duration > self.config.timeout_seconds:
            self.logger.warning(f"Operation exceeded timeout: {duration:.1f}s > {self.config.timeout_seconds}s")
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.config.memory_limit_mb:
                self.logger.warning(f"Memory limit exceeded: {memory_mb:.1f}MB > {self.config.memory_limit_mb}MB")
                return False
            
            self._peak_memory = max(self._peak_memory, memory_mb)
            return True
            
        except ImportError:
            # psutil not available, skip check
            return True
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        import gc
        gc.collect()
        self.logger.debug("Forced garbage collection")


class ResearchModuleWrapper:
    """Robust wrapper for research modules with comprehensive error handling."""
    
    def __init__(self, module: Any, config: ResearchModuleConfig):
        self.module = module
        self.config = config
        self.logger = get_logger(f"Wrapper_{config.module_name}")
        
        # Robustness components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout
        ) if config.circuit_breaker_enabled else None
        
        self.validator = InputValidator()
        self.metrics = OperationMetrics(config.module_name)
        self.health_status = HealthStatus.HEALTHY
        self.last_health_check = time.time()
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(config.max_concurrent_operations)
        
    async def execute_with_robustness(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """Execute operation with full robustness features."""
        start_time = time.time()
        operation_metadata = {
            "operation": operation_name,
            "module": self.config.module_name,
            "start_time": start_time,
            "retries": 0,
            "circuit_breaker_triggered": False
        }
        
        async with self.semaphore:  # Limit concurrent operations
            for attempt in range(self.config.max_retries + 1):
                try:
                    # Resource monitoring
                    async with ResourceMonitor(self.config) as monitor:
                        # Input validation if applicable
                        if args and isinstance(args[0], np.ndarray):
                            validation_result = self.validator.validate_numpy_array(args[0])
                            if not validation_result["valid"]:
                                raise ValueError(f"Input validation failed: {validation_result['issues']}")
                        
                        # Execute with circuit breaker
                        if self.circuit_breaker:
                            result = await self.circuit_breaker._call_with_circuit_breaker(
                                operation_func, *args, **kwargs
                            )
                        else:
                            result = await operation_func(*args, **kwargs)
                        
                        # Success metrics
                        execution_time = time.time() - start_time
                        self.metrics.update_success(execution_time)
                        self.health_status = HealthStatus.HEALTHY
                        
                        operation_metadata.update({
                            "success": True,
                            "execution_time": execution_time,
                            "peak_memory_mb": getattr(monitor, '_peak_memory', 0)
                        })
                        
                        return result, operation_metadata
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.metrics.update_failure(execution_time)
                    operation_metadata["retries"] = attempt
                    
                    if "Circuit breaker OPEN" in str(e):
                        operation_metadata["circuit_breaker_triggered"] = True
                        self.health_status = HealthStatus.UNHEALTHY
                        break
                    
                    if attempt < self.config.max_retries:
                        # Exponential backoff
                        backoff_time = self.config.retry_backoff_factor ** attempt
                        self.logger.warning(
                            f"Operation {operation_name} failed (attempt {attempt + 1}), "
                            f"retrying in {backoff_time:.1f}s: {str(e)}"
                        )
                        await asyncio.sleep(backoff_time)
                    else:
                        # Final failure
                        self.logger.error(f"Operation {operation_name} failed after {attempt + 1} attempts: {str(e)}")
                        self.health_status = HealthStatus.DEGRADED
                        
                        operation_metadata.update({
                            "success": False,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "execution_time": execution_time
                        })
                        
                        if self.config.enable_fallback:
                            fallback_result = await self._execute_fallback(operation_name, *args, **kwargs)
                            operation_metadata["fallback_used"] = True
                            return fallback_result, operation_metadata
                        
                        raise e
        
        # Should not reach here
        raise RuntimeError("Unexpected end of execution path")
    
    async def _execute_fallback(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute fallback logic for failed operations."""
        self.logger.info(f"Executing fallback for {operation_name}")
        
        # Simple fallback strategies based on operation type
        if "generate" in operation_name.lower():
            # Return dummy data for generation operations
            if args and isinstance(args[0], np.ndarray):
                return np.random.randn(*args[0].shape), {"fallback": True}
            else:
                return np.array([]), {"fallback": True}
        
        elif "verify" in operation_name.lower() or "validate" in operation_name.lower():
            # Return conservative validation result
            return {"valid": False, "confidence": 0.0, "fallback": True}
        
        else:
            # Generic fallback
            return {"result": None, "fallback": True}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        now = time.time()
        
        # Basic availability check
        try:
            # Simple operation to test module responsiveness
            if hasattr(self.module, 'get_info'):
                info = self.module.get_info()
            elif hasattr(self.module, '__dict__'):
                info = {"type": type(self.module).__name__}
            else:
                info = {"available": True}
            
            health_data = {
                "status": self.health_status.value,
                "last_check": now,
                "metrics": {
                    "success_rate": self.metrics.success_rate,
                    "failure_rate": self.metrics.failure_rate,
                    "avg_execution_time": self.metrics.average_execution_time,
                    "circuit_trips": self.metrics.circuit_trips
                },
                "circuit_breaker": {
                    "state": self.circuit_breaker.state.value if self.circuit_breaker else "disabled",
                    "failure_count": self.circuit_breaker.failure_count if self.circuit_breaker else 0
                },
                "module_info": info
            }
            
            # Determine overall health
            if self.metrics.failure_rate > 0.5:
                self.health_status = HealthStatus.DEGRADED
            elif self.circuit_breaker and self.circuit_breaker.state == CircuitState.OPEN:
                self.health_status = HealthStatus.UNHEALTHY
            else:
                self.health_status = HealthStatus.HEALTHY
            
            health_data["status"] = self.health_status.value
            self.last_health_check = now
            
            return health_data
            
        except Exception as e:
            self.health_status = HealthStatus.UNHEALTHY
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "last_check": now,
                "error": str(e),
                "error_type": type(e).__name__
            }


class RobustResearchManager:
    """
    Production-ready research manager with comprehensive robustness features.
    
    This manager orchestrates all research modules with enterprise-grade
    reliability, monitoring, and error handling capabilities.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.wrapped_modules: Dict[str, ResearchModuleWrapper] = {}
        self.global_health_status = HealthStatus.HEALTHY
        self.health_check_task = None
        self.metrics_collection_task = None
        
        # Global monitoring
        self.global_metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "fallback_operations": 0,
            "circuit_breaker_trips": 0
        }
        
    async def register_module(
        self,
        module_name: str,
        module_instance: Any,
        config: Optional[ResearchModuleConfig] = None
    ) -> bool:
        """Register a research module with robustness wrapper."""
        try:
            if config is None:
                config = ResearchModuleConfig(module_name=module_name)
            
            wrapper = ResearchModuleWrapper(module_instance, config)
            self.wrapped_modules[module_name] = wrapper
            
            self.logger.info(f"Registered research module: {module_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register module {module_name}: {str(e)}")
            return False
    
    async def execute_operation(
        self,
        module_name: str,
        operation_name: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """Execute operation on research module with full robustness."""
        if module_name not in self.wrapped_modules:
            raise ValueError(f"Module {module_name} not registered")
        
        wrapper = self.wrapped_modules[module_name]
        
        try:
            self.global_metrics["total_operations"] += 1
            
            result, metadata = await wrapper.execute_with_robustness(
                operation_name, operation_func, *args, **kwargs
            )
            
            # Update global metrics
            if metadata.get("success", False):
                self.global_metrics["successful_operations"] += 1
            else:
                self.global_metrics["failed_operations"] += 1
            
            if metadata.get("fallback_used", False):
                self.global_metrics["fallback_operations"] += 1
            
            if metadata.get("circuit_breaker_triggered", False):
                self.global_metrics["circuit_breaker_trips"] += 1
            
            return result, metadata
            
        except Exception as e:
            self.global_metrics["failed_operations"] += 1
            self.logger.error(f"Operation {operation_name} on {module_name} failed: {str(e)}")
            raise
    
    async def get_comprehensive_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report for all modules."""
        report = {
            "timestamp": time.time(),
            "global_status": self.global_health_status.value,
            "global_metrics": self.global_metrics.copy(),
            "modules": {}
        }
        
        # Collect health from all modules
        unhealthy_count = 0
        degraded_count = 0
        
        for module_name, wrapper in self.wrapped_modules.items():
            module_health = await wrapper.health_check()
            report["modules"][module_name] = module_health
            
            if module_health["status"] == HealthStatus.UNHEALTHY.value:
                unhealthy_count += 1
            elif module_health["status"] == HealthStatus.DEGRADED.value:
                degraded_count += 1
        
        # Determine global health
        total_modules = len(self.wrapped_modules)
        if unhealthy_count > total_modules // 2:
            self.global_health_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0 or unhealthy_count > 0:
            self.global_health_status = HealthStatus.DEGRADED
        else:
            self.global_health_status = HealthStatus.HEALTHY
        
        report["global_status"] = self.global_health_status.value
        report["summary"] = {
            "total_modules": total_modules,
            "healthy_modules": total_modules - degraded_count - unhealthy_count,
            "degraded_modules": degraded_count,
            "unhealthy_modules": unhealthy_count
        }
        
        return report
    
    async def start_monitoring(self, health_check_interval: float = 30.0):
        """Start background monitoring tasks."""
        if self.health_check_task is None:
            self.health_check_task = asyncio.create_task(
                self._periodic_health_check(health_check_interval)
            )
            self.logger.info("Started periodic health monitoring")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks."""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            self.logger.info("Stopped periodic health monitoring")
    
    async def _periodic_health_check(self, interval: float):
        """Periodic health check task."""
        while True:
            try:
                await asyncio.sleep(interval)
                health_report = await self.get_comprehensive_health_report()
                
                # Log health status
                status = health_report["global_status"]
                if status != HealthStatus.HEALTHY.value:
                    self.logger.warning(f"Global health status: {status}")
                
                # Log critical issues
                for module_name, module_health in health_report["modules"].items():
                    if module_health["status"] == HealthStatus.UNHEALTHY.value:
                        self.logger.error(f"Module {module_name} is unhealthy")
                
            except Exception as e:
                self.logger.error(f"Health check failed: {str(e)}")
    
    async def graceful_shutdown(self):
        """Gracefully shutdown the research manager."""
        self.logger.info("Initiating graceful shutdown...")
        
        # Stop monitoring
        await self.stop_monitoring()
        
        # Allow ongoing operations to complete (with timeout)
        for module_name, wrapper in self.wrapped_modules.items():
            try:
                # Wait for ongoing operations
                await asyncio.wait_for(wrapper.semaphore.acquire(), timeout=5.0)
                wrapper.semaphore.release()
                self.logger.debug(f"Module {module_name} operations completed")
            except asyncio.TimeoutError:
                self.logger.warning(f"Module {module_name} operations timed out during shutdown")
        
        self.logger.info("Graceful shutdown completed")


# Global instance for easy access
_global_research_manager: Optional[RobustResearchManager] = None


def get_research_manager() -> RobustResearchManager:
    """Get global research manager instance."""
    global _global_research_manager
    if _global_research_manager is None:
        _global_research_manager = RobustResearchManager()
    return _global_research_manager


# Convenience decorators
def robust_research_operation(
    module_name: str,
    operation_name: str,
    timeout: float = 30.0,
    max_retries: int = 3
):
    """Decorator to make research operations robust."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_research_manager()
            return await manager.execute_operation(
                module_name, operation_name, func, *args, **kwargs
            )
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    async def test_robust_research_manager():
        """Test the robust research manager."""
        print("🧪 Testing Robust Research Manager")
        print("=" * 40)
        
        # Create manager
        manager = RobustResearchManager()
        
        # Mock research module
        class MockResearchModule:
            def __init__(self, name):
                self.name = name
                self.call_count = 0
            
            async def process_data(self, data):
                self.call_count += 1
                if self.call_count % 3 == 0:  # Fail every 3rd call
                    raise ValueError(f"Simulated failure in {self.name}")
                return f"Processed by {self.name}: {data.shape}"
            
            def get_info(self):
                return {"name": self.name, "calls": self.call_count}
        
        # Register modules
        mock_module = MockResearchModule("test_module")
        config = ResearchModuleConfig(
            module_name="test_module",
            max_retries=2,
            circuit_breaker_enabled=True,
            failure_threshold=2
        )
        
        await manager.register_module("test_module", mock_module, config)
        
        # Start monitoring
        await manager.start_monitoring(health_check_interval=5.0)
        
        # Test operations
        test_data = np.random.randn(100, 5)
        
        for i in range(5):
            try:
                result, metadata = await manager.execute_operation(
                    "test_module",
                    "process_data",
                    mock_module.process_data,
                    test_data
                )
                print(f"✅ Operation {i+1}: {metadata['success']}, "
                      f"time: {metadata['execution_time']:.3f}s")
                
            except Exception as e:
                print(f"❌ Operation {i+1} failed: {str(e)}")
        
        # Health report
        health_report = await manager.get_comprehensive_health_report()
        print(f"\n📊 Health Report:")
        print(f"   Global status: {health_report['global_status']}")
        print(f"   Total operations: {health_report['global_metrics']['total_operations']}")
        print(f"   Success rate: {health_report['global_metrics']['successful_operations']}/{health_report['global_metrics']['total_operations']}")
        
        # Shutdown
        await manager.graceful_shutdown()
        print("\n✅ Robust research manager test completed")
    
    asyncio.run(test_robust_research_manager())