"""
Advanced Security Middleware - Comprehensive security measures and threat detection
"""

import asyncio
import time
import hashlib
import hmac
import secrets
import re
import json
import logging
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path
import ipaddress
from collections import defaultdict, deque
import jwt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import os


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Types of security events."""
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_INPUT = "suspicious_input"
    INJECTION_ATTEMPT = "injection_attempt"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    AUTHENTICATION_FAILURE = "authentication_failure"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION_ATTEMPT = "data_exfiltration_attempt"


@dataclass
class SecurityAlert:
    """Security alert information."""
    event_type: SecurityEvent
    threat_level: ThreatLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class SecurityConfig:
    """Security configuration settings."""
    
    def __init__(self):
        # Rate limiting
        self.rate_limit_requests_per_minute = 60
        self.rate_limit_burst_size = 10
        
        # Input validation
        self.max_input_length = 10000
        self.max_field_count = 100
        self.max_nesting_depth = 10
        
        # Pattern detection
        self.enable_sql_injection_detection = True
        self.enable_xss_detection = True
        self.enable_path_traversal_detection = True
        self.enable_command_injection_detection = True
        
        # Authentication
        self.jwt_secret_key = secrets.token_urlsafe(32)
        self.jwt_expiration_hours = 24
        self.require_authentication = False
        
        # Encryption
        self.encryption_key = secrets.token_bytes(32)
        self.enable_field_encryption = True
        
        # Monitoring
        self.alert_threshold_per_minute = 5
        self.suspicious_behavior_window = 300  # 5 minutes
        self.max_failed_attempts = 5
        
        # IP filtering
        self.allowed_ip_ranges: List[str] = []
        self.blocked_ip_ranges: List[str] = []
        self.enable_geoip_filtering = False


class InputSanitizer:
    """Advanced input sanitization and validation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Dangerous patterns
        self.sql_injection_patterns = [
            r"(?i)\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b",
            r"(?i)(\-\-|\#|\/\*|\*\/)",
            r"(?i)(\'|\"|`|;|\||&)",
            r"(?i)\b(or|and)\s+(\d+\s*=\s*\d+|true|false)",
        ]
        
        self.xss_patterns = [
            r"(?i)<script[^>]*>.*?</script>",
            r"(?i)javascript:",
            r"(?i)on\w+\s*=",
            r"(?i)<iframe[^>]*>",
            r"(?i)(eval|alert|confirm|prompt)\s*\(",
        ]
        
        self.path_traversal_patterns = [
            r"\.\.[\\/]",
            r"[\\/]\.\.[\\/]",
            r"(?i)\.\.%2f",
            r"(?i)%2e%2e%2f",
        ]
        
        self.command_injection_patterns = [
            r"(?i)(;|\||\&|\$\(|\`)",
            r"(?i)\b(cat|ls|pwd|whoami|id|uname|netstat|ps|top|chmod|chown)\b",
            r"(?i)(>|<|>>|<<)",
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = {
            'sql_injection': [re.compile(p) for p in self.sql_injection_patterns],
            'xss': [re.compile(p) for p in self.xss_patterns],
            'path_traversal': [re.compile(p) for p in self.path_traversal_patterns],
            'command_injection': [re.compile(p) for p in self.command_injection_patterns],
        }
    
    def sanitize_input(self, data: Any, field_name: str = "unknown") -> tuple[Any, List[SecurityAlert]]:
        """Sanitize input data and detect threats."""
        alerts = []
        
        if isinstance(data, str):
            sanitized_data, string_alerts = self._sanitize_string(data, field_name)
            alerts.extend(string_alerts)
            return sanitized_data, alerts
        
        elif isinstance(data, dict):
            return self._sanitize_dict(data, field_name, alerts)
        
        elif isinstance(data, list):
            return self._sanitize_list(data, field_name, alerts)
        
        else:
            # For other types, just validate size/type
            if hasattr(data, '__len__') and len(str(data)) > self.config.max_input_length:
                alerts.append(SecurityAlert(
                    event_type=SecurityEvent.SUSPICIOUS_INPUT,
                    threat_level=ThreatLevel.MEDIUM,
                    message=f"Input too large in field {field_name}",
                    details={'field': field_name, 'size': len(str(data))}
                ))
            
            return data, alerts
    
    def _sanitize_string(self, text: str, field_name: str) -> tuple[str, List[SecurityAlert]]:
        """Sanitize string input and detect threats."""
        alerts = []
        
        # Length check
        if len(text) > self.config.max_input_length:
            alerts.append(SecurityAlert(
                event_type=SecurityEvent.SUSPICIOUS_INPUT,
                threat_level=ThreatLevel.MEDIUM,
                message=f"String too long in field {field_name}",
                details={'field': field_name, 'length': len(text)}
            ))
            text = text[:self.config.max_input_length]
        
        # Pattern detection
        for pattern_type, patterns in self.compiled_patterns.items():
            if not getattr(self.config, f'enable_{pattern_type}_detection', True):
                continue
                
            for pattern in patterns:
                if pattern.search(text):
                    threat_level = ThreatLevel.HIGH if pattern_type in ['sql_injection', 'command_injection'] else ThreatLevel.MEDIUM
                    alerts.append(SecurityAlert(
                        event_type=SecurityEvent.INJECTION_ATTEMPT,
                        threat_level=threat_level,
                        message=f"Potential {pattern_type.replace('_', ' ')} detected in field {field_name}",
                        details={
                            'field': field_name,
                            'pattern_type': pattern_type,
                            'matched_pattern': pattern.pattern[:100]
                        }
                    ))
                    break  # Only report first match per type
        
        # Basic sanitization
        sanitized = self._basic_string_sanitization(text)
        
        return sanitized, alerts
    
    def _basic_string_sanitization(self, text: str) -> str:
        """Apply basic string sanitization."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove potentially dangerous Unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text.strip()
    
    def _sanitize_dict(self, data: dict, field_name: str, alerts: List[SecurityAlert], depth: int = 0) -> tuple[dict, List[SecurityAlert]]:
        """Sanitize dictionary data."""
        if depth > self.config.max_nesting_depth:
            alerts.append(SecurityAlert(
                event_type=SecurityEvent.SUSPICIOUS_INPUT,
                threat_level=ThreatLevel.MEDIUM,
                message=f"Maximum nesting depth exceeded in field {field_name}",
                details={'field': field_name, 'depth': depth}
            ))
            return {}, alerts
        
        if len(data) > self.config.max_field_count:
            alerts.append(SecurityAlert(
                event_type=SecurityEvent.SUSPICIOUS_INPUT,
                threat_level=ThreatLevel.MEDIUM,
                message=f"Too many fields in {field_name}",
                details={'field': field_name, 'count': len(data)}
            ))
            # Truncate to max allowed
            data = dict(list(data.items())[:self.config.max_field_count])
        
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            clean_key, key_alerts = self._sanitize_string(str(key), f"{field_name}.{key}")
            alerts.extend(key_alerts)
            
            # Sanitize value
            clean_value, value_alerts = self.sanitize_input(value, f"{field_name}.{clean_key}")
            alerts.extend(value_alerts)
            
            sanitized[clean_key] = clean_value
        
        return sanitized, alerts
    
    def _sanitize_list(self, data: list, field_name: str, alerts: List[SecurityAlert], depth: int = 0) -> tuple[list, List[SecurityAlert]]:
        """Sanitize list data."""
        if depth > self.config.max_nesting_depth:
            alerts.append(SecurityAlert(
                event_type=SecurityEvent.SUSPICIOUS_INPUT,
                threat_level=ThreatLevel.MEDIUM,
                message=f"Maximum nesting depth exceeded in field {field_name}",
                details={'field': field_name, 'depth': depth}
            ))
            return [], alerts
        
        if len(data) > self.config.max_field_count:
            alerts.append(SecurityAlert(
                event_type=SecurityEvent.SUSPICIOUS_INPUT,
                threat_level=ThreatLevel.MEDIUM,
                message=f"Too many items in list {field_name}",
                details={'field': field_name, 'count': len(data)}
            ))
            # Truncate to max allowed
            data = data[:self.config.max_field_count]
        
        sanitized = []
        for i, item in enumerate(data):
            clean_item, item_alerts = self.sanitize_input(item, f"{field_name}[{i}]")
            alerts.extend(item_alerts)
            sanitized.append(clean_item)
        
        return sanitized, alerts


class RateLimiter:
    """Advanced rate limiting with burst protection."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track requests by IP/user
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips: Dict[str, float] = {}  # IP -> block_until_timestamp
        self.suspicious_ips: Set[str] = set()
        
        self._lock = threading.RLock()
    
    def is_allowed(self, identifier: str, current_time: float = None) -> tuple[bool, Optional[SecurityAlert]]:
        """Check if request is allowed based on rate limits."""
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            # Check if IP is blocked
            if identifier in self.blocked_ips:
                if current_time < self.blocked_ips[identifier]:
                    return False, SecurityAlert(
                        event_type=SecurityEvent.RATE_LIMIT_EXCEEDED,
                        threat_level=ThreatLevel.HIGH,
                        message=f"Request from blocked identifier: {identifier}",
                        details={'identifier': identifier, 'blocked_until': self.blocked_ips[identifier]}
                    )
                else:
                    # Block expired, remove it
                    del self.blocked_ips[identifier]
            
            # Get request history for this identifier
            history = self.request_history[identifier]
            
            # Remove old requests (older than 1 minute)
            cutoff_time = current_time - 60
            while history and history[0] < cutoff_time:
                history.popleft()
            
            # Check rate limit
            requests_in_window = len(history)
            
            if requests_in_window >= self.config.rate_limit_requests_per_minute:
                # Check for burst (too many requests in short time)
                recent_requests = sum(1 for req_time in history if current_time - req_time < 10)
                
                if recent_requests >= self.config.rate_limit_burst_size:
                    # Block the IP for 5 minutes
                    self.blocked_ips[identifier] = current_time + 300
                    self.suspicious_ips.add(identifier)
                    
                    return False, SecurityAlert(
                        event_type=SecurityEvent.RATE_LIMIT_EXCEEDED,
                        threat_level=ThreatLevel.CRITICAL,
                        message=f"Burst rate limit exceeded, blocking identifier: {identifier}",
                        details={
                            'identifier': identifier,
                            'requests_in_window': requests_in_window,
                            'recent_requests': recent_requests,
                            'blocked_until': self.blocked_ips[identifier]
                        }
                    )
                
                return False, SecurityAlert(
                    event_type=SecurityEvent.RATE_LIMIT_EXCEEDED,
                    threat_level=ThreatLevel.MEDIUM,
                    message=f"Rate limit exceeded for identifier: {identifier}",
                    details={'identifier': identifier, 'requests_in_window': requests_in_window}
                )
            
            # Add current request to history
            history.append(current_time)
            
            return True, None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        with self._lock:
            current_time = time.time()
            active_blocks = {ip: until for ip, until in self.blocked_ips.items() if until > current_time}
            
            return {
                'total_tracked_identifiers': len(self.request_history),
                'currently_blocked': len(active_blocks),
                'suspicious_identifiers': len(self.suspicious_ips),
                'blocked_identifiers': list(active_blocks.keys()),
                'request_counts': {
                    identifier: len(history)
                    for identifier, history in self.request_history.items()
                    if len(history) > 0
                }
            }


class EncryptionManager:
    """Field-level encryption for sensitive data."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.key = config.encryption_key
        
        # Sensitive field patterns
        self.sensitive_patterns = [
            r'(?i).*password.*',
            r'(?i).*secret.*',
            r'(?i).*token.*',
            r'(?i).*key.*',
            r'(?i).*ssn.*',
            r'(?i).*social.*',
            r'(?i).*credit.*card.*',
            r'(?i).*email.*',
            r'(?i).*phone.*',
            r'(?i).*address.*',
        ]
        
        self.compiled_sensitive_patterns = [re.compile(p) for p in self.sensitive_patterns]
    
    def should_encrypt_field(self, field_name: str) -> bool:
        """Determine if a field should be encrypted."""
        if not self.config.enable_field_encryption:
            return False
        
        return any(pattern.match(field_name) for pattern in self.compiled_sensitive_patterns)
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a sensitive value."""
        if not isinstance(value, str):
            value = str(value)
        
        try:
            # Generate random IV
            iv = os.urandom(16)
            
            # Create cipher
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv))
            encryptor = cipher.encryptor()
            
            # Pad value to multiple of 16 bytes
            padded_value = self._pad_string(value.encode('utf-8'))
            
            # Encrypt
            encrypted = encryptor.update(padded_value) + encryptor.finalize()
            
            # Combine IV and encrypted data, then base64 encode
            combined = iv + encrypted
            return base64.b64encode(combined).decode('utf-8')
        
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return "[ENCRYPTED]"  # Fallback
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt an encrypted value."""
        try:
            # Base64 decode
            combined = base64.b64decode(encrypted_value.encode('utf-8'))
            
            # Extract IV and encrypted data
            iv = combined[:16]
            encrypted = combined[16:]
            
            # Create cipher
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv))
            decryptor = cipher.decryptor()
            
            # Decrypt
            padded_value = decryptor.update(encrypted) + decryptor.finalize()
            
            # Remove padding
            value = self._unpad_string(padded_value)
            
            return value.decode('utf-8')
        
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return "[DECRYPTION_FAILED]"
    
    def _pad_string(self, data: bytes) -> bytes:
        """PKCS7 padding."""
        padding_length = 16 - (len(data) % 16)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_string(self, data: bytes) -> bytes:
        """Remove PKCS7 padding."""
        padding_length = data[-1]
        return data[:-padding_length]


class SecurityMonitor:
    """Comprehensive security monitoring and alerting."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.alerts: List[SecurityAlert] = []
        self.alert_handlers: List[Callable[[SecurityAlert], None]] = []
        
        # Anomaly detection
        self.behavior_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_metrics: Dict[str, Dict] = {}
        
        self._lock = threading.RLock()
    
    def add_alert_handler(self, handler: Callable[[SecurityAlert], None]):
        """Add a custom alert handler."""
        self.alert_handlers.append(handler)
    
    def report_alert(self, alert: SecurityAlert):
        """Report a security alert."""
        with self._lock:
            self.alerts.append(alert)
            
            # Trigger alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")
            
            # Log alert
            level = logging.CRITICAL if alert.threat_level == ThreatLevel.CRITICAL else logging.WARNING
            self.logger.log(level, f"Security Alert: {alert.message} (Level: {alert.threat_level.value})")
    
    def analyze_behavior(self, identifier: str, action: str, metadata: Dict[str, Any] = None):
        """Analyze user behavior for anomalies."""
        current_time = time.time()
        
        behavior_event = {
            'timestamp': current_time,
            'action': action,
            'metadata': metadata or {}
        }
        
        with self._lock:
            # Add to behavior history
            self.behavior_patterns[identifier].append(behavior_event)
            
            # Analyze for anomalies
            anomalies = self._detect_anomalies(identifier, behavior_event)
            
            for anomaly in anomalies:
                self.report_alert(SecurityAlert(
                    event_type=SecurityEvent.ANOMALOUS_BEHAVIOR,
                    threat_level=ThreatLevel.MEDIUM,
                    message=f"Anomalous behavior detected for {identifier}: {anomaly}",
                    details={'identifier': identifier, 'anomaly': anomaly, 'action': action}
                ))
    
    def _detect_anomalies(self, identifier: str, current_event: Dict) -> List[str]:
        """Detect behavioral anomalies."""
        anomalies = []
        pattern_history = self.behavior_patterns[identifier]
        
        if len(pattern_history) < 10:  # Need enough history
            return anomalies
        
        current_time = current_event['timestamp']
        
        # Check for unusual timing patterns
        recent_events = [e for e in pattern_history if current_time - e['timestamp'] < 3600]  # Last hour
        if len(recent_events) > 100:  # More than 100 actions per hour
            anomalies.append("excessive_activity")
        
        # Check for action pattern changes
        recent_actions = [e['action'] for e in recent_events]
        unique_actions = set(recent_actions)
        if len(unique_actions) > 20:  # Too many different actions
            anomalies.append("diverse_action_pattern")
        
        # Check for off-hours activity
        from datetime import datetime
        current_hour = datetime.fromtimestamp(current_time).hour
        if current_hour < 6 or current_hour > 22:  # Outside business hours
            recent_off_hours = sum(1 for e in recent_events if 
                                 datetime.fromtimestamp(e['timestamp']).hour < 6 or 
                                 datetime.fromtimestamp(e['timestamp']).hour > 22)
            if recent_off_hours > 10:
                anomalies.append("unusual_timing")
        
        return anomalies
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary."""
        with self._lock:
            current_time = time.time()
            
            # Alert statistics
            recent_alerts = [a for a in self.alerts if current_time - a.timestamp < 3600]
            alert_counts = defaultdict(int)
            for alert in recent_alerts:
                alert_counts[alert.threat_level.value] += 1
            
            # Behavior statistics
            active_users = len([patterns for patterns in self.behavior_patterns.values() 
                              if patterns and current_time - patterns[-1]['timestamp'] < 3600])
            
            return {
                'timestamp': current_time,
                'total_alerts': len(self.alerts),
                'recent_alerts_1h': len(recent_alerts),
                'alert_counts_by_level': dict(alert_counts),
                'active_users_1h': active_users,
                'monitored_entities': len(self.behavior_patterns),
                'alert_handlers': len(self.alert_handlers)
            }


class AdvancedSecurityMiddleware:
    """Comprehensive security middleware orchestrator."""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize security components
        self.input_sanitizer = InputSanitizer(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.encryption_manager = EncryptionManager(self.config)
        self.security_monitor = SecurityMonitor(self.config)
        
        # Setup default alert handlers
        self._setup_default_alert_handlers()
        
        self.logger.info("Advanced Security Middleware initialized")
    
    def _setup_default_alert_handlers(self):
        """Setup default security alert handlers."""
        def log_alert(alert: SecurityAlert):
            level = logging.CRITICAL if alert.threat_level == ThreatLevel.CRITICAL else logging.WARNING
            self.logger.log(level, f"[{alert.threat_level.value.upper()}] {alert.message}")
        
        self.security_monitor.add_alert_handler(log_alert)
    
    async def process_request(self, request_data: Dict[str, Any], 
                            client_ip: str = None, 
                            user_id: str = None) -> tuple[Dict[str, Any], List[SecurityAlert]]:
        """Process incoming request with full security pipeline."""
        identifier = client_ip or user_id or "anonymous"
        all_alerts = []
        
        # 1. Rate limiting
        allowed, rate_alert = self.rate_limiter.is_allowed(identifier)
        if not allowed:
            all_alerts.append(rate_alert)
            self.security_monitor.report_alert(rate_alert)
            return {}, all_alerts
        
        # 2. Input sanitization
        sanitized_data, sanitization_alerts = self.input_sanitizer.sanitize_input(
            request_data, "request_data"
        )
        all_alerts.extend(sanitization_alerts)
        
        for alert in sanitization_alerts:
            self.security_monitor.report_alert(alert)
        
        # 3. Behavior analysis
        if user_id:
            self.security_monitor.analyze_behavior(
                user_id, 
                "request", 
                {"client_ip": client_ip, "data_size": len(str(request_data))}
            )
        
        # 4. Field encryption (for sensitive fields)
        processed_data = self._process_field_encryption(sanitized_data)
        
        return processed_data, all_alerts
    
    def _process_field_encryption(self, data: Any) -> Any:
        """Apply field-level encryption to sensitive data."""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if self.encryption_manager.should_encrypt_field(key) and isinstance(value, str):
                    result[key] = self.encryption_manager.encrypt_value(value)
                elif isinstance(value, (dict, list)):
                    result[key] = self._process_field_encryption(value)
                else:
                    result[key] = value
            return result
        elif isinstance(data, list):
            return [self._process_field_encryption(item) for item in data]
        else:
            return data
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'config': {
                'rate_limiting_enabled': True,
                'input_validation_enabled': True,
                'encryption_enabled': self.config.enable_field_encryption,
                'threat_detection_enabled': True
            },
            'rate_limiter': self.rate_limiter.get_stats(),
            'security_monitor': self.security_monitor.get_security_summary(),
            'recent_alerts': [
                {
                    'event_type': alert.event_type.value,
                    'threat_level': alert.threat_level.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                }
                for alert in self.security_monitor.alerts[-10:]  # Last 10 alerts
            ]
        }