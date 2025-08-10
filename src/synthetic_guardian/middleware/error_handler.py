"""
Comprehensive error handling middleware for the Synthetic Data Guardian platform
"""

import asyncio
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Type, Union
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils.logger import get_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    GENERATION = "generation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    RESOURCE = "resource"
    NETWORK = "network"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    DATA = "data"
    SECURITY = "security"
    UNKNOWN = "unknown"


@dataclass
class ErrorDetails:
    """Detailed error information."""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_error: Exception
    context: Dict[str, Any]
    stack_trace: str
    user_message: str
    remediation_steps: List[str]
    metadata: Dict[str, Any]


class SyntheticDataError(Exception):
    """Base exception for Synthetic Data Guardian."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        remediation_steps: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.user_message = user_message or message
        self.remediation_steps = remediation_steps or []
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.now()


class ValidationError(SyntheticDataError):
    """Error in data validation."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class GenerationError(SyntheticDataError):
    """Error in data generation."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.GENERATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class SecurityError(SyntheticDataError):
    """Security-related error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )


class ResourceError(SyntheticDataError):
    """Resource-related error (memory, storage, etc.)."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class DependencyError(SyntheticDataError):
    """External dependency error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class RateLimitError(SyntheticDataError):
    """Rate limiting error."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )


class ErrorHandler:
    """
    Comprehensive error handler for the Synthetic Data Guardian platform.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.error_statistics = {
            'total_errors': 0,
            'by_category': {cat.value: 0 for cat in ErrorCategory},
            'by_severity': {sev.value: 0 for sev in ErrorSeverity},
            'critical_errors': []
        }
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self.notification_callbacks: List[Callable] = []
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default error handlers."""
        self.register_handler(ValidationError, self._handle_validation_error)
        self.register_handler(GenerationError, self._handle_generation_error)
        self.register_handler(SecurityError, self._handle_security_error)
        self.register_handler(ResourceError, self._handle_resource_error)
        self.register_handler(DependencyError, self._handle_dependency_error)
        self.register_handler(RateLimitError, self._handle_rate_limit_error)
        self.register_handler(Exception, self._handle_generic_error)
    
    def register_handler(self, exception_type: Type[Exception], handler: Callable):
        """Register a custom error handler."""
        self.error_handlers[exception_type] = handler
        self.logger.info(f"Registered handler for {exception_type.__name__}")
    
    def add_notification_callback(self, callback: Callable):
        """Add notification callback for critical errors."""
        self.notification_callbacks.append(callback)
    
    async def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorDetails:
        """
        Handle an error with comprehensive logging and processing.
        
        Args:
            error: The exception to handle
            context: Additional context information
            
        Returns:
            ErrorDetails with complete error information
        """
        # Create error details
        error_details = self._create_error_details(error, context or {})
        
        # Update statistics
        self._update_statistics(error_details)
        
        # Log the error
        self._log_error(error_details)
        
        # Find and execute appropriate handler
        handler = self._find_handler(error)
        if handler:
            try:
                await handler(error_details)
            except Exception as handler_error:
                self.logger.error(f"Error handler failed: {handler_error}")
        
        # Notify if critical
        if error_details.severity == ErrorSeverity.CRITICAL:
            await self._notify_critical_error(error_details)
        
        # Store critical errors for analysis
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.error_statistics['critical_errors'].append(error_details)
            # Keep only last 100 critical errors
            if len(self.error_statistics['critical_errors']) > 100:
                self.error_statistics['critical_errors'].pop(0)
        
        return error_details
    
    def _create_error_details(self, error: Exception, context: Dict[str, Any]) -> ErrorDetails:
        """Create detailed error information."""
        # Extract information from custom errors
        if isinstance(error, SyntheticDataError):
            category = error.category
            severity = error.severity
            error_id = error.error_id
            timestamp = error.timestamp
            user_message = error.user_message
            remediation_steps = error.remediation_steps
            error_context = {**context, **error.context}
        else:
            # Classify unknown errors
            category = self._classify_error(error)
            severity = self._determine_severity(error)
            error_id = str(uuid.uuid4())
            timestamp = datetime.now()
            user_message = "An unexpected error occurred. Please try again."
            remediation_steps = ["Contact support if the problem persists"]
            error_context = context
        
        return ErrorDetails(
            error_id=error_id,
            timestamp=timestamp,
            category=category,
            severity=severity,
            message=str(error),
            original_error=error,
            context=error_context,
            stack_trace=traceback.format_exc(),
            user_message=user_message,
            remediation_steps=remediation_steps,
            metadata={}
        )
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify unknown errors into categories."""
        error_message = str(error).lower()
        
        if 'validation' in error_message or 'invalid' in error_message:
            return ErrorCategory.VALIDATION
        elif 'memory' in error_message or 'resource' in error_message:
            return ErrorCategory.RESOURCE
        elif 'permission' in error_message or 'access' in error_message:
            return ErrorCategory.AUTHORIZATION
        elif 'connection' in error_message or 'network' in error_message:
            return ErrorCategory.NETWORK
        elif 'config' in error_message:
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type."""
        if isinstance(error, (MemoryError, SystemError, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (IOError, OSError, ConnectionError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _find_handler(self, error: Exception) -> Optional[Callable]:
        """Find appropriate error handler."""
        # Try exact type match first
        for error_type, handler in self.error_handlers.items():
            if type(error) == error_type:
                return handler
        
        # Try inheritance match
        for error_type, handler in self.error_handlers.items():
            if isinstance(error, error_type):
                return handler
        
        return None
    
    def _update_statistics(self, error_details: ErrorDetails):
        """Update error statistics."""
        self.error_statistics['total_errors'] += 1
        self.error_statistics['by_category'][error_details.category.value] += 1
        self.error_statistics['by_severity'][error_details.severity.value] += 1
    
    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level."""
        log_data = {
            'error_id': error_details.error_id,
            'category': error_details.category.value,
            'severity': error_details.severity.value,
            'context': error_details.context,
            'remediation_steps': error_details.remediation_steps
        }
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_details.message, extra=log_data)
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(error_details.message, extra=log_data)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error_details.message, extra=log_data)
        else:
            self.logger.info(error_details.message, extra=log_data)
    
    async def _notify_critical_error(self, error_details: ErrorDetails):
        """Notify about critical errors."""
        for callback in self.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_details)
                else:
                    callback(error_details)
            except Exception as e:
                self.logger.error(f"Notification callback failed: {e}")
    
    # Default error handlers
    async def _handle_validation_error(self, error_details: ErrorDetails):
        """Handle validation errors."""
        self.logger.info(f"Handling validation error: {error_details.error_id}")
        
        # Add remediation steps
        error_details.remediation_steps.extend([
            "Check input data format and schema",
            "Verify required fields are present",
            "Ensure data types match expected schema"
        ])
    
    async def _handle_generation_error(self, error_details: ErrorDetails):
        """Handle generation errors."""
        self.logger.warning(f"Handling generation error: {error_details.error_id}")
        
        # Add remediation steps
        error_details.remediation_steps.extend([
            "Try reducing the number of records to generate",
            "Check if the model is properly initialized",
            "Verify system resources are available"
        ])
    
    async def _handle_security_error(self, error_details: ErrorDetails):
        """Handle security errors."""
        self.logger.critical(f"SECURITY ALERT: {error_details.error_id}")
        
        # Security errors need immediate attention
        error_details.user_message = "Access denied. Contact administrator."
        error_details.remediation_steps = [
            "Verify authentication credentials",
            "Check authorization permissions", 
            "Contact security team if unauthorized access suspected"
        ]
    
    async def _handle_resource_error(self, error_details: ErrorDetails):
        """Handle resource errors."""
        self.logger.error(f"Resource error detected: {error_details.error_id}")
        
        error_details.remediation_steps.extend([
            "Free up system memory",
            "Check available disk space",
            "Reduce batch size or number of concurrent operations"
        ])
    
    async def _handle_dependency_error(self, error_details: ErrorDetails):
        """Handle dependency errors."""
        self.logger.warning(f"Dependency error: {error_details.error_id}")
        
        error_details.remediation_steps.extend([
            "Check if required packages are installed",
            "Verify external service connectivity",
            "Update dependencies to compatible versions"
        ])
    
    async def _handle_rate_limit_error(self, error_details: ErrorDetails):
        """Handle rate limit errors."""
        self.logger.info(f"Rate limit exceeded: {error_details.error_id}")
        
        error_details.user_message = "Rate limit exceeded. Please try again later."
        error_details.remediation_steps = [
            "Wait for rate limit to reset",
            "Reduce request frequency",
            "Consider upgrading service plan if needed"
        ]
    
    async def _handle_generic_error(self, error_details: ErrorDetails):
        """Handle generic/unknown errors."""
        self.logger.error(f"Unhandled error: {error_details.error_id}")
        
        error_details.remediation_steps = [
            "Check system logs for more details",
            "Try the operation again",
            "Contact support if problem persists"
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return self.error_statistics.copy()
    
    def get_recent_critical_errors(self, limit: int = 10) -> List[ErrorDetails]:
        """Get recent critical errors."""
        return self.error_statistics['critical_errors'][-limit:]
    
    def clear_statistics(self):
        """Clear error statistics."""
        self.error_statistics = {
            'total_errors': 0,
            'by_category': {cat.value: 0 for cat in ErrorCategory},
            'by_severity': {sev.value: 0 for sev in ErrorSeverity},
            'critical_errors': []
        }
        self.logger.info("Error statistics cleared")


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def error_handler_decorator(func):
    """Decorator to automatically handle errors in async functions."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            handler = get_error_handler()
            error_details = await handler.handle_error(e, {
                'function': func.__name__,
                'args': str(args)[:500],  # Truncate for logging
                'kwargs': str(kwargs)[:500]
            })
            # Re-raise with error ID for tracing
            raise SyntheticDataError(
                f"Error {error_details.error_id}: {error_details.user_message}",
                category=error_details.category,
                severity=error_details.severity
            ) from e
    
    return wrapper


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = get_logger(self.__class__.__name__)
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        import time
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed operation."""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class CircuitBreakerOpenError(SyntheticDataError):
    """Error when circuit breaker is open."""
    
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(
            message,
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.HIGH,
            user_message="Service temporarily unavailable. Please try again later."
        )