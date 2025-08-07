"""
Advanced logging utilities with structured logging and monitoring integration
"""

import logging
import logging.handlers
import json
import sys
import os
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import threading

# Thread-local storage for correlation IDs
_correlation_context = threading.local()


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
        }
        
        # Add correlation ID if available
        correlation_id = getattr(_correlation_context, 'correlation_id', None)
        if correlation_id:
            log_entry['correlation_id'] = correlation_id
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in log_entry and not key.startswith('_'):
                # Handle complex objects
                try:
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records."""
    
    def filter(self, record):
        """Add correlation ID to record if available."""
        correlation_id = getattr(_correlation_context, 'correlation_id', None)
        if correlation_id:
            record.correlation_id = correlation_id
        return True


class SecurityFilter(logging.Filter):
    """Filter out sensitive information from logs."""
    
    SENSITIVE_KEYS = {
        'password', 'token', 'key', 'secret', 'api_key', 'auth',
        'authorization', 'credential', 'private', 'ssn', 'credit_card'
    }
    
    def filter(self, record):
        """Remove sensitive information from log message."""
        message = record.getMessage()
        
        # Basic sensitive data redaction
        for sensitive_key in self.SENSITIVE_KEYS:
            if sensitive_key.lower() in message.lower():
                # This is a simple approach - in production, use more sophisticated redaction
                record.msg = record.msg.replace(message, '[REDACTED]')
                break
        
        return True


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, logger, operation: str, threshold_ms: float = 1000.0):
        self.logger = logger
        self.operation = operation
        self.threshold_ms = threshold_ms
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000
            
            log_level = logging.WARNING if duration_ms > self.threshold_ms else logging.DEBUG
            
            self.logger.log(
                log_level,
                f"Completed operation: {self.operation}",
                extra={
                    'operation': self.operation,
                    'duration_ms': duration_ms,
                    'slow_operation': duration_ms > self.threshold_ms,
                    'success': exc_type is None
                }
            )
            
            if exc_type:
                self.logger.error(
                    f"Operation failed: {self.operation}",
                    extra={
                        'operation': self.operation,
                        'duration_ms': duration_ms,
                        'error_type': exc_type.__name__ if exc_type else None,
                        'error_message': str(exc_val) if exc_val else None
                    }
                )


class Logger:
    """Enhanced logger with additional functionality."""
    
    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self._logger = logger
        
    def debug(self, msg, *args, **kwargs):
        """Log debug message."""
        self._logger.debug(msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        """Log info message."""
        self._logger.info(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        """Log warning message."""
        self._logger.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        """Log error message."""
        self._logger.error(msg, *args, **kwargs)
        
    def critical(self, msg, *args, **kwargs):
        """Log critical message."""
        self._logger.critical(msg, *args, **kwargs)
        
    def exception(self, msg, *args, **kwargs):
        """Log exception with traceback."""
        self._logger.exception(msg, *args, **kwargs)
        
    def performance(self, operation: str, threshold_ms: float = 1000.0):
        """Create performance logging context."""
        return PerformanceLogger(self._logger, operation, threshold_ms)
        
    def with_correlation(self, correlation_id: str):
        """Create logger context with correlation ID."""
        return CorrelationContext(correlation_id)
        
    def audit(self, event: str, **kwargs):
        """Log audit event."""
        self._logger.info(
            f"AUDIT: {event}",
            extra={
                'audit_event': True,
                'event_type': event,
                **kwargs
            }
        )
        
    def security(self, event: str, **kwargs):
        """Log security event."""
        self._logger.warning(
            f"SECURITY: {event}",
            extra={
                'security_event': True,
                'event_type': event,
                **kwargs
            }
        )


class CorrelationContext:
    """Context manager for correlation ID."""
    
    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.previous_id = None
        
    def __enter__(self):
        self.previous_id = getattr(_correlation_context, 'correlation_id', None)
        _correlation_context.correlation_id = self.correlation_id
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_id:
            _correlation_context.correlation_id = self.previous_id
        else:
            if hasattr(_correlation_context, 'correlation_id'):
                delattr(_correlation_context, 'correlation_id')


def configure_logging(
    level: str = 'INFO',
    format_type: str = 'json',
    log_file: Optional[Union[str, Path]] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    enable_security_filter: bool = True
) -> None:
    """Configure application-wide logging."""
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Create formatters
    if format_type == 'json':
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Add filters
        console_handler.addFilter(CorrelationFilter())
        if enable_security_filter:
            console_handler.addFilter(SecurityFilter())
            
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Add filters
        file_handler.addFilter(CorrelationFilter())
        if enable_security_filter:
            file_handler.addFilter(SecurityFilter())
            
        root_logger.addHandler(file_handler)
    
    # Error file handler
    if log_file:
        error_log_path = log_path.with_suffix('.error.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        error_handler.addFilter(CorrelationFilter())
        if enable_security_filter:
            error_handler.addFilter(SecurityFilter())
            
        root_logger.addHandler(error_handler)


def get_logger(name: str, level: Optional[str] = None) -> Logger:
    """Get enhanced logger instance."""
    logger = logging.getLogger(name)
    
    if level:
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(log_level)
    
    return Logger(name, logger)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current thread."""
    _correlation_context.correlation_id = correlation_id


def get_correlation_id() -> Optional[str]:
    """Get correlation ID for current thread."""
    return getattr(_correlation_context, 'correlation_id', None)


def clear_correlation_id() -> None:
    """Clear correlation ID for current thread."""
    if hasattr(_correlation_context, 'correlation_id'):
        delattr(_correlation_context, 'correlation_id')


# Configure default logging
if not logging.getLogger().handlers:
    configure_logging(
        level=os.getenv('LOG_LEVEL', 'INFO'),
        format_type=os.getenv('LOG_FORMAT', 'json'),
        log_file=os.getenv('LOG_FILE'),
        enable_console=os.getenv('LOG_CONSOLE', 'true').lower() == 'true'
    )
