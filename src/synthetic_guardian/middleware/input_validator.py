"""
Comprehensive input validation and sanitization middleware
"""

import re
import json
import html
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger
from .error_handler import ValidationError, SecurityError


class ValidationLevel(Enum):
    """Validation strictness levels."""
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    name: str
    validator: Callable
    error_message: str
    required: bool = True
    sanitizer: Optional[Callable] = None


class InputValidator:
    """
    Comprehensive input validation and sanitization system.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = get_logger(self.__class__.__name__)
        
        # Security patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b)",
            r"(UNION\s+SELECT)",
            r"(--|\*\/|\/\*)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+['\"][^'\"]*['\"])",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]
        
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e\\",
        ]
        
        # Common validation rules
        self.validation_rules = self._build_validation_rules()
    
    def _build_validation_rules(self) -> Dict[str, ValidationRule]:
        """Build standard validation rules."""
        rules = {}
        
        # String validations
        rules['safe_string'] = ValidationRule(
            name='safe_string',
            validator=self._validate_safe_string,
            error_message="String contains potentially dangerous content",
            sanitizer=self._sanitize_string
        )
        
        rules['email'] = ValidationRule(
            name='email',
            validator=self._validate_email,
            error_message="Invalid email format",
            sanitizer=self._sanitize_email
        )
        
        rules['numeric'] = ValidationRule(
            name='numeric',
            validator=self._validate_numeric,
            error_message="Value must be numeric"
        )
        
        rules['integer'] = ValidationRule(
            name='integer',
            validator=self._validate_integer,
            error_message="Value must be an integer"
        )
        
        rules['positive_integer'] = ValidationRule(
            name='positive_integer',
            validator=self._validate_positive_integer,
            error_message="Value must be a positive integer"
        )
        
        rules['uuid'] = ValidationRule(
            name='uuid',
            validator=self._validate_uuid,
            error_message="Invalid UUID format"
        )
        
        rules['filename'] = ValidationRule(
            name='filename',
            validator=self._validate_filename,
            error_message="Invalid filename",
            sanitizer=self._sanitize_filename
        )
        
        rules['json'] = ValidationRule(
            name='json',
            validator=self._validate_json,
            error_message="Invalid JSON format"
        )
        
        # Data generation specific validations
        rules['generator_type'] = ValidationRule(
            name='generator_type',
            validator=lambda x: x in ['tabular', 'timeseries', 'text', 'image', 'graph'],
            error_message="Invalid generator type"
        )
        
        rules['data_type'] = ValidationRule(
            name='data_type',
            validator=lambda x: x in ['tabular', 'timeseries', 'text', 'image', 'graph'],
            error_message="Invalid data type"
        )
        
        rules['record_count'] = ValidationRule(
            name='record_count',
            validator=lambda x: isinstance(x, int) and 1 <= x <= 10_000_000,
            error_message="Record count must be between 1 and 10,000,000"
        )
        
        return rules
    
    def validate_input(
        self, 
        data: Any, 
        rules: List[str], 
        field_name: str = "input"
    ) -> Any:
        """
        Validate input data against specified rules.
        
        Args:
            data: Input data to validate
            rules: List of validation rule names
            field_name: Name of the field being validated
            
        Returns:
            Sanitized data if validation passes
            
        Raises:
            ValidationError: If validation fails
            SecurityError: If security threat detected
        """
        self.logger.debug(f"Validating {field_name} with rules: {rules}")
        
        # Security check first
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            self._security_scan(data, field_name)
        
        sanitized_data = data
        
        for rule_name in rules:
            if rule_name not in self.validation_rules:
                raise ValidationError(f"Unknown validation rule: {rule_name}")
            
            rule = self.validation_rules[rule_name]
            
            try:
                # Run validator
                if not rule.validator(sanitized_data):
                    raise ValidationError(
                        f"Validation failed for {field_name}: {rule.error_message}",
                        context={
                            'field': field_name,
                            'rule': rule_name,
                            'value': str(sanitized_data)[:100]  # Truncate for security
                        }
                    )
                
                # Apply sanitizer if available
                if rule.sanitizer:
                    sanitized_data = rule.sanitizer(sanitized_data)
                    
            except Exception as e:
                if isinstance(e, (ValidationError, SecurityError)):
                    raise
                else:
                    raise ValidationError(
                        f"Validation error for {field_name}: {str(e)}",
                        context={'field': field_name, 'rule': rule_name}
                    )
        
        return sanitized_data
    
    def validate_dict(
        self, 
        data: Dict[str, Any], 
        field_rules: Dict[str, List[str]],
        allow_extra_fields: bool = True
    ) -> Dict[str, Any]:
        """
        Validate dictionary with field-specific rules.
        
        Args:
            data: Dictionary to validate
            field_rules: Rules for each field
            allow_extra_fields: Whether to allow fields not in field_rules
            
        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            raise ValidationError("Expected dictionary input")
        
        sanitized = {}
        
        # Validate known fields
        for field_name, rules in field_rules.items():
            if field_name in data:
                sanitized[field_name] = self.validate_input(
                    data[field_name], 
                    rules, 
                    field_name
                )
            else:
                # Check if field is required
                required = any(
                    self.validation_rules[rule].required 
                    for rule in rules 
                    if rule in self.validation_rules
                )
                if required:
                    raise ValidationError(f"Required field missing: {field_name}")
        
        # Handle extra fields
        if allow_extra_fields:
            for field_name, value in data.items():
                if field_name not in field_rules:
                    # Apply basic security sanitization to unknown fields
                    sanitized[field_name] = self.validate_input(
                        value, 
                        ['safe_string'] if isinstance(value, str) else [],
                        field_name
                    )
        else:
            extra_fields = set(data.keys()) - set(field_rules.keys())
            if extra_fields:
                raise ValidationError(f"Unexpected fields: {extra_fields}")
        
        return sanitized
    
    def _security_scan(self, data: Any, field_name: str):
        """Perform security scanning on input data."""
        if not isinstance(data, str):
            return
        
        data_lower = data.lower()
        
        # Check for SQL injection
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                self.logger.security(
                    f"SQL injection attempt detected in {field_name}",
                    pattern=pattern,
                    field=field_name
                )
                raise SecurityError(
                    "Potentially malicious SQL content detected",
                    context={'field': field_name, 'pattern': pattern}
                )
        
        # Check for XSS
        for pattern in self.xss_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                self.logger.security(
                    f"XSS attempt detected in {field_name}",
                    pattern=pattern,
                    field=field_name
                )
                raise SecurityError(
                    "Potentially malicious script content detected",
                    context={'field': field_name, 'pattern': pattern}
                )
        
        # Check for path traversal
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                self.logger.security(
                    f"Path traversal attempt detected in {field_name}",
                    pattern=pattern,
                    field=field_name
                )
                raise SecurityError(
                    "Path traversal attempt detected",
                    context={'field': field_name, 'pattern': pattern}
                )
    
    # Validator functions
    def _validate_safe_string(self, value: Any) -> bool:
        """Validate string is safe from common attacks."""
        if not isinstance(value, str):
            return False
        
        # Basic safety checks
        dangerous_chars = ['<', '>', '&', '"', "'", '\\', '\x00']
        return not any(char in value for char in dangerous_chars)
    
    def _validate_email(self, value: Any) -> bool:
        """Validate email format."""
        if not isinstance(value, str):
            return False
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, value) is not None
    
    def _validate_numeric(self, value: Any) -> bool:
        """Validate numeric value."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _validate_integer(self, value: Any) -> bool:
        """Validate integer value."""
        if isinstance(value, int):
            return True
        if isinstance(value, str):
            try:
                int(value)
                return True
            except ValueError:
                return False
        return False
    
    def _validate_positive_integer(self, value: Any) -> bool:
        """Validate positive integer."""
        if not self._validate_integer(value):
            return False
        
        num = int(value)
        return num > 0
    
    def _validate_uuid(self, value: Any) -> bool:
        """Validate UUID format."""
        if not isinstance(value, str):
            return False
        
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        return re.match(uuid_pattern, value.lower()) is not None
    
    def _validate_filename(self, value: Any) -> bool:
        """Validate filename is safe."""
        if not isinstance(value, str):
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'\.\./', r'\.\.\\', r'^\.', r'\x00', r'[<>:"|?*]'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value):
                return False
        
        # Check length
        if len(value) > 255:
            return False
        
        return True
    
    def _validate_json(self, value: Any) -> bool:
        """Validate JSON format."""
        if isinstance(value, (dict, list)):
            try:
                json.dumps(value)
                return True
            except (TypeError, ValueError):
                return False
        elif isinstance(value, str):
            try:
                json.loads(value)
                return True
            except (json.JSONDecodeError, ValueError):
                return False
        
        return False
    
    # Sanitizer functions
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string by escaping HTML and removing dangerous content."""
        if not isinstance(value, str):
            return str(value)
        
        # HTML escape
        sanitized = html.escape(value)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def _sanitize_email(self, value: str) -> str:
        """Sanitize email address."""
        if not isinstance(value, str):
            return str(value)
        
        # Convert to lowercase and strip whitespace
        return value.lower().strip()
    
    def _sanitize_filename(self, value: str) -> str:
        """Sanitize filename by removing dangerous characters."""
        if not isinstance(value, str):
            return str(value)
        
        # Replace dangerous characters
        sanitized = re.sub(r'[<>:"|?*\\]', '_', value)
        
        # Remove path separators
        sanitized = sanitized.replace('/', '_').replace('\\', '_')
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Ensure reasonable length
        if len(sanitized) > 200:
            name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
            sanitized = name[:190] + ('.' + ext if ext else '')
        
        return sanitized
    
    def add_custom_rule(self, rule: ValidationRule):
        """Add custom validation rule."""
        self.validation_rules[rule.name] = rule
        self.logger.info(f"Added custom validation rule: {rule.name}")
    
    def get_available_rules(self) -> List[str]:
        """Get list of available validation rules."""
        return list(self.validation_rules.keys())


# Convenience decorators for common validations
def validate_generation_params(func):
    """Decorator to validate generation parameters."""
    async def wrapper(self, *args, **kwargs):
        validator = InputValidator()
        
        # Validate common parameters
        if 'num_records' in kwargs:
            kwargs['num_records'] = validator.validate_input(
                kwargs['num_records'], 
                ['record_count'], 
                'num_records'
            )
        
        if 'seed' in kwargs and kwargs['seed'] is not None:
            kwargs['seed'] = validator.validate_input(
                kwargs['seed'], 
                ['integer'], 
                'seed'
            )
        
        return await func(self, *args, **kwargs)
    
    return wrapper


def validate_pipeline_config(func):
    """Decorator to validate pipeline configuration."""
    async def wrapper(*args, **kwargs):
        validator = InputValidator()
        
        # Find pipeline config in arguments
        config = None
        if args and isinstance(args[0], dict):
            config = args[0]
        elif 'config' in kwargs:
            config = kwargs['config']
        elif 'pipeline_config' in kwargs:
            config = kwargs['pipeline_config']
        
        if config and isinstance(config, dict):
            # Validate configuration
            field_rules = {
                'name': ['safe_string'],
                'generator_type': ['generator_type'],
                'data_type': ['data_type']
            }
            
            sanitized_config = validator.validate_dict(config, field_rules)
            
            # Update the config in arguments
            if args and isinstance(args[0], dict):
                args = (sanitized_config,) + args[1:]
            elif 'config' in kwargs:
                kwargs['config'] = sanitized_config
            elif 'pipeline_config' in kwargs:
                kwargs['pipeline_config'] = sanitized_config
        
        return await func(*args, **kwargs)
    
    return wrapper


# Global validator instance
_global_validator: Optional[InputValidator] = None


def get_input_validator() -> InputValidator:
    """Get global input validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = InputValidator()
    return _global_validator