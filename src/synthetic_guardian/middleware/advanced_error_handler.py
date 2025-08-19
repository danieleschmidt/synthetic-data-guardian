"""
Advanced Error Handler with Recovery Strategies and Comprehensive Logging
"""

import asyncio
import time
import traceback
import inspect
import sys
from typing import Dict, Any, Optional, List, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
import functools
import logging
from pathlib import Path

from ..utils.logger import get_logger
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker_registry


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAIL_FAST = "fail_fast"
    IGNORE = "ignore"


@dataclass
class RetryConfig:
    """Configuration for retry strategy."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])


@dataclass
class FallbackConfig:
    """Configuration for fallback strategy."""
    fallback_func: Optional[Callable] = None
    fallback_value: Any = None
    cache_fallback: bool = True


@dataclass
class ErrorHandlingRule:
    """Rule for handling specific errors."""
    exception_type: Type[Exception]
    severity: ErrorSeverity
    strategy: RecoveryStrategy
    retry_config: Optional[RetryConfig] = None
    fallback_config: Optional[FallbackConfig] = None
    circuit_breaker_name: Optional[str] = None
    custom_handler: Optional[Callable] = None
    description: str = ""


class ErrorContext:
    """Context for error handling with detailed information."""
    
    def __init__(
        self,
        exception: Exception,
        function_name: str,
        args: tuple = None,
        kwargs: dict = None,
        attempt: int = 1,
        max_attempts: int = 1
    ):
        self.exception = exception
        self.function_name = function_name
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.attempt = attempt
        self.max_attempts = max_attempts
        self.timestamp = time.time()
        self.traceback = traceback.format_exc()
        self.call_stack = self._get_call_stack()
        
        # Additional context
        self.error_id = f"{function_name}_{int(self.timestamp)}_{id(exception)}"
        self.recovery_attempts = []
        self.metadata = {}
    
    def _get_call_stack(self) -> List[Dict[str, Any]]:
        """Get detailed call stack information."""
        stack = []
        for frame_info in inspect.stack()[2:]:  # Skip current frame and error handler frame
            try:
                stack.append({
                    'filename': frame_info.filename,
                    'function': frame_info.function,
                    'line_number': frame_info.lineno,
                    'code': frame_info.code_context[0].strip() if frame_info.code_context else ""
                })
            except (IndexError, AttributeError):
                continue
        return stack
    
    def add_recovery_attempt(self, strategy: str, success: bool, details: str = ""):
        """Record recovery attempt."""
        self.recovery_attempts.append({
            'strategy': strategy,
            'success': success,
            'timestamp': time.time(),
            'details': details
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for logging."""
        return {
            'error_id': self.error_id,
            'exception_type': self.exception.__class__.__name__,
            'exception_message': str(self.exception),
            'function_name': self.function_name,
            'attempt': self.attempt,
            'max_attempts': self.max_attempts,
            'timestamp': self.timestamp,
            'call_stack': self.call_stack,
            'recovery_attempts': self.recovery_attempts,
            'metadata': self.metadata
        }


class AdvancedErrorHandler:
    """
    Advanced error handler with comprehensive recovery strategies.
    
    Features:
    - Multiple recovery strategies (retry, fallback, circuit breaker)
    - Intelligent error classification
    - Detailed logging and monitoring
    - Performance impact tracking
    - Custom error handling rules
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.logger = get_logger(f"ErrorHandler[{name}]")
        self._rules: List[ErrorHandlingRule] = []
        self._error_statistics: Dict[str, Dict] = {}
        self._fallback_cache: Dict[str, Any] = {}
        
        # Initialize with default rules
        self._setup_default_rules()
        
        self.logger.info(f"Advanced error handler '{name}' initialized")
    
    def _setup_default_rules(self):
        """Setup default error handling rules."""
        # Network/connectivity errors - retry with backoff
        self.add_rule(
            exception_type=ConnectionError,
            severity=ErrorSeverity.MEDIUM,
            strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(max_attempts=3, base_delay=2.0, exponential_backoff=True),
            description="Network connectivity issues"
        )
        
        # Timeout errors - retry with circuit breaker
        self.add_rule(
            exception_type=TimeoutError,
            severity=ErrorSeverity.HIGH,
            strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            circuit_breaker_name="timeout_breaker",
            description="Operation timeout"
        )
        
        # Memory errors - fail fast
        self.add_rule(
            exception_type=MemoryError,
            severity=ErrorSeverity.CRITICAL,
            strategy=RecoveryStrategy.FAIL_FAST,
            description="Memory exhaustion"
        )
        
        # Value errors - fallback with validation
        self.add_rule(
            exception_type=ValueError,
            severity=ErrorSeverity.LOW,
            strategy=RecoveryStrategy.FALLBACK,
            fallback_config=FallbackConfig(fallback_value=None, cache_fallback=False),
            description="Invalid input values"
        )
    
    def add_rule(
        self,
        exception_type: Type[Exception],
        severity: ErrorSeverity,
        strategy: RecoveryStrategy,
        retry_config: Optional[RetryConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
        circuit_breaker_name: Optional[str] = None,
        custom_handler: Optional[Callable] = None,
        description: str = ""
    ):
        """Add custom error handling rule."""
        rule = ErrorHandlingRule(
            exception_type=exception_type,
            severity=severity,
            strategy=strategy,
            retry_config=retry_config,
            fallback_config=fallback_config,
            circuit_breaker_name=circuit_breaker_name,
            custom_handler=custom_handler,
            description=description
        )
        self._rules.append(rule)
        
        self.logger.info(f"Added error handling rule for {exception_type.__name__}: {strategy.value}")
    
    def _find_rule(self, exception: Exception) -> Optional[ErrorHandlingRule]:
        """Find matching rule for exception."""
        exception_type = type(exception)
        
        # Look for exact match first
        for rule in self._rules:
            if rule.exception_type == exception_type:
                return rule
        
        # Look for parent class matches
        for rule in self._rules:
            if issubclass(exception_type, rule.exception_type):
                return rule
        
        return None
    
    async def handle_error(
        self,
        exception: Exception,
        function_name: str,
        args: tuple = None,
        kwargs: dict = None,
        attempt: int = 1,
        max_attempts: int = 1
    ) -> Tuple[bool, Any]:
        """
        Handle error with appropriate recovery strategy.
        
        Returns:
            Tuple of (recovered, result)
        """
        # Create error context
        context = ErrorContext(exception, function_name, args, kwargs, attempt, max_attempts)
        
        # Update statistics
        self._update_error_statistics(exception, context)
        
        # Find matching rule
        rule = self._find_rule(exception)
        if rule is None:
            # No specific rule - use default handling
            rule = ErrorHandlingRule(
                exception_type=Exception,
                severity=ErrorSeverity.MEDIUM,
                strategy=RecoveryStrategy.FAIL_FAST
            )
        
        # Log error with context
        self._log_error(exception, context, rule)
        
        # Apply recovery strategy
        try:
            return await self._apply_recovery_strategy(exception, context, rule)
        except Exception as recovery_error:
            self.logger.error(
                f"Recovery strategy failed for {context.error_id}: {recovery_error}",
                extra={'error_context': context.to_dict()}
            )
            return False, None
    
    async def _apply_recovery_strategy(
        self,
        exception: Exception,
        context: ErrorContext,
        rule: ErrorHandlingRule
    ) -> Tuple[bool, Any]:
        """Apply the appropriate recovery strategy."""
        strategy = rule.strategy
        
        if strategy == RecoveryStrategy.RETRY:
            return await self._handle_retry(exception, context, rule)
        
        elif strategy == RecoveryStrategy.FALLBACK:
            return await self._handle_fallback(exception, context, rule)
        
        elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._handle_circuit_breaker(exception, context, rule)
        
        elif strategy == RecoveryStrategy.IGNORE:
            context.add_recovery_attempt("ignore", True, "Exception ignored per rule")
            self.logger.info(f"Ignoring exception per rule: {context.error_id}")
            return True, None
        
        else:  # FAIL_FAST
            context.add_recovery_attempt("fail_fast", False, "No recovery attempted")
            return False, None
    
    async def _handle_retry(
        self,
        exception: Exception,
        context: ErrorContext,
        rule: ErrorHandlingRule
    ) -> Tuple[bool, Any]:
        """Handle retry recovery strategy."""
        retry_config = rule.retry_config or RetryConfig()
        
        if context.attempt >= retry_config.max_attempts:
            context.add_recovery_attempt("retry", False, "Max retry attempts reached")
            return False, None
        
        # Calculate delay with exponential backoff and jitter
        delay = retry_config.base_delay
        if retry_config.exponential_backoff:
            delay *= (2 ** (context.attempt - 1))
        
        # Apply max delay limit
        delay = min(delay, retry_config.max_delay)
        
        # Add jitter to prevent thundering herd
        if retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        self.logger.info(
            f"Retrying {context.function_name} in {delay:.2f}s "
            f"(attempt {context.attempt + 1}/{retry_config.max_attempts})"
        )
        
        await asyncio.sleep(delay)
        
        context.add_recovery_attempt(
            "retry", 
            True, 
            f"Delayed retry after {delay:.2f}s"
        )
        
        # Return success to indicate retry should be attempted
        return True, "retry"
    
    async def _handle_fallback(
        self,
        exception: Exception,
        context: ErrorContext,
        rule: ErrorHandlingRule
    ) -> Tuple[bool, Any]:
        """Handle fallback recovery strategy."""
        fallback_config = rule.fallback_config or FallbackConfig()
        
        # Check cache first
        cache_key = f"{context.function_name}_{hash(str(context.args))}"
        if fallback_config.cache_fallback and cache_key in self._fallback_cache:
            cached_result = self._fallback_cache[cache_key]
            context.add_recovery_attempt("fallback", True, "Used cached fallback value")
            self.logger.info(f"Using cached fallback for {context.function_name}")
            return True, cached_result
        
        # Use fallback function if provided
        if fallback_config.fallback_func:
            try:
                if asyncio.iscoroutinefunction(fallback_config.fallback_func):
                    result = await fallback_config.fallback_func(*context.args, **context.kwargs)
                else:
                    result = fallback_config.fallback_func(*context.args, **context.kwargs)
                
                # Cache result if enabled
                if fallback_config.cache_fallback:
                    self._fallback_cache[cache_key] = result
                
                context.add_recovery_attempt("fallback", True, "Fallback function succeeded")
                self.logger.info(f"Fallback function succeeded for {context.function_name}")
                return True, result
                
            except Exception as fallback_error:
                context.add_recovery_attempt(
                    "fallback", 
                    False, 
                    f"Fallback function failed: {fallback_error}"
                )
                self.logger.warning(f"Fallback function failed: {fallback_error}")
        
        # Use fallback value
        if fallback_config.fallback_value is not None:
            context.add_recovery_attempt("fallback", True, "Used fallback value")
            return True, fallback_config.fallback_value
        
        context.add_recovery_attempt("fallback", False, "No fallback available")
        return False, None
    
    async def _handle_circuit_breaker(
        self,
        exception: Exception,
        context: ErrorContext,
        rule: ErrorHandlingRule
    ) -> Tuple[bool, Any]:
        """Handle circuit breaker recovery strategy."""
        breaker_name = rule.circuit_breaker_name or f"{self.name}_breaker"
        
        # Get or create circuit breaker
        registry = get_circuit_breaker_registry()
        breaker = registry.get(breaker_name)
        if breaker is None:
            breaker = registry.create(
                name=breaker_name,
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=type(exception)
            )
        
        # Circuit breaker will handle the error internally
        context.add_recovery_attempt("circuit_breaker", False, "Delegated to circuit breaker")
        return False, None
    
    def _update_error_statistics(self, exception: Exception, context: ErrorContext):
        """Update error statistics for monitoring."""
        exc_name = exception.__class__.__name__
        func_name = context.function_name
        
        if exc_name not in self._error_statistics:
            self._error_statistics[exc_name] = {
                'count': 0,
                'first_seen': time.time(),
                'last_seen': time.time(),
                'functions': {}
            }
        
        stats = self._error_statistics[exc_name]
        stats['count'] += 1
        stats['last_seen'] = time.time()
        
        if func_name not in stats['functions']:
            stats['functions'][func_name] = 0
        stats['functions'][func_name] += 1
    
    def _log_error(
        self,
        exception: Exception,
        context: ErrorContext,
        rule: ErrorHandlingRule
    ):
        """Log error with comprehensive context."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(rule.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"Error in {context.function_name}: {exception.__class__.__name__}: {str(exception)}",
            extra={
                'error_context': context.to_dict(),
                'severity': rule.severity.value,
                'strategy': rule.strategy.value,
                'rule_description': rule.description
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            'handler_name': self.name,
            'total_rules': len(self._rules),
            'error_statistics': self._error_statistics,
            'fallback_cache_size': len(self._fallback_cache)
        }
    
    def clear_cache(self):
        """Clear fallback cache."""
        self._fallback_cache.clear()
        self.logger.info("Fallback cache cleared")
    
    def reset_statistics(self):
        """Reset error statistics."""
        self._error_statistics.clear()
        self.logger.info("Error statistics reset")


def robust_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exponential_backoff: bool = True,
    circuit_breaker_name: Optional[str] = None,
    fallback_func: Optional[Callable] = None,
    error_handler_name: str = "default"
):
    """
    Decorator for robust async function execution with comprehensive error handling.
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        exponential_backoff: Use exponential backoff
        circuit_breaker_name: Circuit breaker name
        fallback_func: Fallback function
        error_handler_name: Error handler name
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            handler = AdvancedErrorHandler(error_handler_name)
            
            for attempt in range(1, max_attempts + 1):
                try:
                    # Use circuit breaker if specified
                    if circuit_breaker_name:
                        registry = get_circuit_breaker_registry()
                        breaker = registry.get(circuit_breaker_name)
                        if breaker is None:
                            breaker = registry.create(circuit_breaker_name)
                        
                        async with breaker:
                            return await func(*args, **kwargs)
                    else:
                        return await func(*args, **kwargs)
                
                except Exception as e:
                    # Handle error through advanced error handler
                    recovered, result = await handler.handle_error(
                        e, func.__name__, args, kwargs, attempt, max_attempts
                    )
                    
                    if recovered and result == "retry" and attempt < max_attempts:
                        continue  # Retry
                    elif recovered:
                        return result  # Use fallback result
                    elif attempt == max_attempts:
                        raise  # Last attempt failed
            
            # Should not reach here
            raise RuntimeError(f"Function {func.__name__} failed after {max_attempts} attempts")
        
        return wrapper
    return decorator