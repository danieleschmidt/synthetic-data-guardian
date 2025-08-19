"""
Circuit Breaker Pattern Implementation for Robust Error Handling
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable, Union, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from functools import wraps
import traceback

from ..utils.logger import get_logger


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Circuit tripped, failing fast
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures to trip circuit
    recovery_timeout: int = 60          # Seconds before trying half-open
    expected_exception: type = Exception # Expected exception type
    success_threshold: int = 2          # Successes to close circuit from half-open
    timeout: Optional[float] = None     # Operation timeout
    name: str = "circuit_breaker"       # Circuit breaker name
    
    # Advanced settings
    slow_call_duration_threshold: float = 10.0  # Seconds
    slow_call_rate_threshold: float = 0.5       # Percentage
    minimum_number_of_calls: int = 10           # Min calls before rate calculation
    sliding_window_size: int = 100              # Sliding window for metrics


@dataclass 
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    slow_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    # Sliding window for call history
    call_history: list = field(default_factory=list)
    
    def add_success(self, duration: float):
        """Record successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = time.time()
        
        # Track slow calls
        if duration > 10.0:  # Default slow threshold
            self.slow_calls += 1
        
        # Update sliding window
        self._update_sliding_window(True, duration)
    
    def add_failure(self, duration: float):
        """Record failed call."""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = time.time()
        
        # Update sliding window
        self._update_sliding_window(False, duration)
    
    def _update_sliding_window(self, success: bool, duration: float):
        """Update sliding window of call history."""
        call_record = {
            'timestamp': time.time(),
            'success': success,
            'duration': duration
        }
        self.call_history.append(call_record)
        
        # Keep only recent calls (sliding window)
        window_size = 100  # Default size
        if len(self.call_history) > window_size:
            self.call_history = self.call_history[-window_size:]
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    @property
    def slow_call_rate(self) -> float:
        """Calculate slow call rate."""
        if self.total_calls == 0:
            return 0.0
        return self.slow_calls / self.total_calls


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, name: str, metrics: CircuitBreakerMetrics):
        self.name = name
        self.metrics = metrics
        super().__init__(f"Circuit breaker '{name}' is OPEN")


class CircuitBreaker:
    """Circuit breaker implementation with state management and metrics."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.logger = get_logger(f"CircuitBreaker[{config.name}]")
        self._lock = threading.RLock()
        self._last_state_change = time.time()
        
        self.logger.info(f"Circuit breaker '{config.name}' initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._check_state()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with result recording."""
        duration = time.time() - getattr(self, '_call_start_time', time.time())
        
        if exc_type is None:
            await self._record_success(duration)
        elif isinstance(exc_val, self.config.expected_exception):
            await self._record_failure(duration, exc_val)
        else:
            # Unexpected exception - don't count against circuit breaker
            self.logger.warning(f"Unexpected exception type: {exc_type.__name__}")
    
    def __call__(self, func: Callable):
        """Decorator for synchronous functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
    
    def async_call(self, func: Callable[..., Awaitable]):
        """Decorator for async functions."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with self:
                return await func(*args, **kwargs)
        return async_wrapper
    
    async def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        async with self:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
    
    async def _check_state(self):
        """Check and update circuit breaker state."""
        with self._lock:
            current_time = time.time()
            self._call_start_time = current_time
            
            if self.state == CircuitBreakerState.CLOSED:
                # Normal operation
                if self.metrics.consecutive_failures >= self.config.failure_threshold:
                    await self._trip_circuit()
                    raise CircuitBreakerOpenException(self.config.name, self.metrics)
            
            elif self.state == CircuitBreakerState.OPEN:
                # Circuit is open - check if recovery timeout elapsed
                if (current_time - self._last_state_change) >= self.config.recovery_timeout:
                    await self._attempt_reset()
                else:
                    raise CircuitBreakerOpenException(self.config.name, self.metrics)
            
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Testing recovery - allow limited calls through
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    await self._close_circuit()
                elif self.metrics.consecutive_failures > 0:
                    await self._trip_circuit()
                    raise CircuitBreakerOpenException(self.config.name, self.metrics)
    
    async def _record_success(self, duration: float):
        """Record successful call and update state."""
        with self._lock:
            self.metrics.add_success(duration)
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.config.success_threshold:
                    await self._close_circuit()
            
            self.logger.debug(
                f"Success recorded - Duration: {duration:.3f}s, "
                f"Consecutive: {self.metrics.consecutive_successes}, "
                f"State: {self.state.value}"
            )
    
    async def _record_failure(self, duration: float, exception: Exception):
        """Record failed call and update state."""
        with self._lock:
            self.metrics.add_failure(duration)
            
            self.logger.warning(
                f"Failure recorded - Duration: {duration:.3f}s, "
                f"Exception: {exception.__class__.__name__}: {str(exception)}, "
                f"Consecutive: {self.metrics.consecutive_failures}, "
                f"State: {self.state.value}"
            )
            
            if self.state == CircuitBreakerState.CLOSED:
                if self.metrics.consecutive_failures >= self.config.failure_threshold:
                    await self._trip_circuit()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Immediate trip on any failure in half-open state
                await self._trip_circuit()
    
    async def _trip_circuit(self):
        """Trip the circuit breaker to OPEN state."""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self._last_state_change = time.time()
        
        self.logger.error(
            f"Circuit breaker TRIPPED: {old_state.value} -> OPEN "
            f"(Failures: {self.metrics.consecutive_failures}/{self.config.failure_threshold})"
        )
    
    async def _attempt_reset(self):
        """Attempt to reset circuit breaker to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self._last_state_change = time.time()
        self.metrics.consecutive_failures = 0
        self.metrics.consecutive_successes = 0
        
        self.logger.info(f"Circuit breaker state: {old_state.value} -> HALF_OPEN (attempting recovery)")
    
    async def _close_circuit(self):
        """Close the circuit breaker to normal operation."""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self._last_state_change = time.time()
        self.metrics.consecutive_failures = 0
        
        self.logger.info(
            f"Circuit breaker RECOVERED: {old_state.value} -> CLOSED "
            f"(Successes: {self.metrics.consecutive_successes}/{self.config.success_threshold})"
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state and metrics."""
        with self._lock:
            return {
                'name': self.config.name,
                'state': self.state.value,
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'last_state_change': self._last_state_change,
                'metrics': {
                    'total_calls': self.metrics.total_calls,
                    'successful_calls': self.metrics.successful_calls,
                    'failed_calls': self.metrics.failed_calls,
                    'slow_calls': self.metrics.slow_calls,
                    'consecutive_failures': self.metrics.consecutive_failures,
                    'consecutive_successes': self.metrics.consecutive_successes,
                    'failure_rate': self.metrics.failure_rate,
                    'success_rate': self.metrics.success_rate,
                    'slow_call_rate': self.metrics.slow_call_rate,
                    'last_failure_time': self.metrics.last_failure_time,
                    'last_success_time': self.metrics.last_success_time
                }
            }
    
    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            old_state = self.state
            self.state = CircuitBreakerState.CLOSED
            self._last_state_change = time.time()
            self.metrics.consecutive_failures = 0
            self.metrics.consecutive_successes = 0
            
            self.logger.info(f"Circuit breaker manually reset: {old_state.value} -> CLOSED")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self.logger = get_logger(self.__class__.__name__)
        self._lock = threading.RLock()
    
    def create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
        **kwargs
    ) -> CircuitBreaker:
        """Create and register a new circuit breaker."""
        with self._lock:
            if name in self._breakers:
                self.logger.warning(f"Circuit breaker '{name}' already exists, returning existing")
                return self._breakers[name]
            
            config = CircuitBreakerConfig(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
                **kwargs
            )
            
            breaker = CircuitBreaker(config)
            self._breakers[name] = breaker
            
            self.logger.info(f"Created circuit breaker '{name}'")
            return breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)
    
    def remove(self, name: str) -> bool:
        """Remove circuit breaker."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                self.logger.info(f"Removed circuit breaker '{name}'")
                return True
            return False
    
    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all circuit breakers and their states."""
        with self._lock:
            return {
                name: breaker.get_state()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            self.logger.info("Reset all circuit breakers")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all circuit breakers."""
        with self._lock:
            total = len(self._breakers)
            open_count = sum(1 for b in self._breakers.values() if b.state == CircuitBreakerState.OPEN)
            half_open_count = sum(1 for b in self._breakers.values() if b.state == CircuitBreakerState.HALF_OPEN)
            closed_count = total - open_count - half_open_count
            
            return {
                'total_breakers': total,
                'closed': closed_count,
                'half_open': half_open_count,
                'open': open_count,
                'breakers': list(self._breakers.keys())
            }


# Global registry
_global_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get global circuit breaker registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = CircuitBreakerRegistry()
    return _global_registry


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: type = Exception,
    **kwargs
):
    """Decorator for circuit breaker functionality."""
    registry = get_circuit_breaker_registry()
    breaker = registry.create(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        expected_exception=expected_exception,
        **kwargs
    )
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            return breaker.async_call(func)
        else:
            return breaker(func)
    
    return decorator