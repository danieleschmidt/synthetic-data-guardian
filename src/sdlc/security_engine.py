"""
Security Engine - Comprehensive security validation and enforcement
"""

import asyncio
import hashlib
import hmac
import time
import jwt
import secrets
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import ipaddress
from pathlib import Path

from ..utils.logger import get_logger


class SecurityThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    MALICIOUS_PAYLOAD = "malicious_payload"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: SecurityEventType
    threat_level: SecurityThreatLevel
    timestamp: float
    source_ip: str
    user_id: Optional[str] = None
    request_path: Optional[str] = None
    payload_hash: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    action_taken: Optional[str] = None


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    name: str
    enabled: bool = True
    threat_level: SecurityThreatLevel = SecurityThreatLevel.MEDIUM
    action: str = "block"  # block, warn, log
    parameters: Dict[str, Any] = field(default_factory=dict)


class SecurityEngine:
    """
    Security Engine - Comprehensive security validation and protection.
    
    Provides enterprise-grade security including:
    - Input validation and sanitization
    - Authentication and authorization
    - Rate limiting and DDoS protection
    - Injection attack prevention
    - Anomaly detection
    - Security event logging and alerting
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        """Initialize security engine."""
        self.config = config or {}
        self.logger = logger or get_logger(self.__class__.__name__)
        
        # Security state
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, List[float]] = {}  # IP -> timestamps
        self.rate_limits: Dict[str, List[float]] = {}  # Key -> timestamps
        self.failed_attempts: Dict[str, int] = {}  # IP/User -> count
        
        # Security policies
        self.policies = self._initialize_security_policies()
        
        # Security patterns
        self.malicious_patterns = self._initialize_malicious_patterns()
        
        # JWT settings
        self.jwt_secret = self.config.get('jwt_secret', secrets.token_urlsafe(32))
        self.jwt_algorithm = self.config.get('jwt_algorithm', 'HS256')
        self.jwt_expiry = self.config.get('jwt_expiry_hours', 24)
        
        self.logger.info("Security Engine initialized with comprehensive protection")
    
    def _initialize_security_policies(self) -> Dict[str, SecurityPolicy]:
        """Initialize default security policies."""
        policies = {}
        
        # Rate limiting policies
        policies['api_rate_limit'] = SecurityPolicy(
            name="API Rate Limiting",
            threat_level=SecurityThreatLevel.MEDIUM,
            action="block",
            parameters={
                'requests_per_minute': 60,
                'burst_limit': 10,
                'window_minutes': 1
            }
        )
        
        policies['generation_rate_limit'] = SecurityPolicy(
            name="Generation Rate Limiting", 
            threat_level=SecurityThreatLevel.HIGH,
            action="block",
            parameters={
                'generations_per_hour': 10,
                'burst_limit': 3,
                'window_hours': 1
            }
        )
        
        # Input validation policies
        policies['injection_prevention'] = SecurityPolicy(
            name="SQL/NoSQL Injection Prevention",
            threat_level=SecurityThreatLevel.CRITICAL,
            action="block",
            parameters={
                'check_sql_patterns': True,
                'check_nosql_patterns': True,
                'check_command_injection': True
            }
        )
        
        policies['xss_prevention'] = SecurityPolicy(
            name="XSS Prevention",
            threat_level=SecurityThreatLevel.HIGH,
            action="block",
            parameters={
                'check_script_tags': True,
                'check_event_handlers': True,
                'check_javascript_urls': True
            }
        )
        
        # Authentication policies
        policies['brute_force_protection'] = SecurityPolicy(
            name="Brute Force Protection",
            threat_level=SecurityThreatLevel.HIGH,
            action="block",
            parameters={
                'max_failed_attempts': 5,
                'lockout_duration_minutes': 30,
                'progressive_delay': True
            }
        )
        
        # Data protection policies
        policies['sensitive_data_protection'] = SecurityPolicy(
            name="Sensitive Data Protection",
            threat_level=SecurityThreatLevel.CRITICAL,
            action="block",
            parameters={
                'check_pii_patterns': True,
                'check_credentials': True,
                'check_api_keys': True
            }
        )
        
        return policies
    
    def _initialize_malicious_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting malicious input."""
        return {
            'sql_injection': [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b)",
                r"(\b(UNION|OR|AND)\s+\d+\s*=\s*\d+)",
                r"(--|\/\*|\*\/)",
                r"(\b(SCRIPT|IFRAME|OBJECT|EMBED)\b)",
                r"('|\"|`|;|\||&)"
            ],
            'nosql_injection': [
                r"(\$where|\$ne|\$gt|\$lt|\$regex)",
                r"(\.find\(|\.aggregate\(|\.mapReduce\()",
                r"(new\s+Function|eval\(|setTimeout\()"
            ],
            'command_injection': [
                r"(;|\||&|`|\$\(|\${)",
                r"(\b(cat|ls|pwd|whoami|id|uname|ps|netstat)\b)",
                r"(\.\.\/|\.\.\\\\)",
                r"(\/etc\/passwd|\/etc\/shadow|\/proc\/)"
            ],
            'xss_patterns': [
                r"(<script[^>]*>|<\/script>)",
                r"(javascript:|vbscript:|data:)",
                r"(on\w+\s*=)",
                r"(<iframe|<object|<embed|<form)"
            ],
            'path_traversal': [
                r"(\.\.\/|\.\.\\\\)",
                r"(\/etc\/|\/proc\/|\/sys\/)",
                r"(\.\.%2F|\.\.%5C)",
                r"(%2e%2e%2f|%2e%2e%5c)"
            ],
            'sensitive_data': [
                r"(\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b)",  # Credit card
                r"(\b\d{3}-\d{2}-\d{4}\b)",  # SSN
                r"(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)",  # Email
                r"(\b(password|pwd|pass|secret|token|key)\s*[:=]\s*\S+)",  # Credentials
                r"(\b[A-Za-z0-9]{32,}\b)"  # API keys/tokens
            ]
        }
    
    async def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incoming request for security threats."""
        source_ip = request_data.get('source_ip', 'unknown')
        user_id = request_data.get('user_id')
        path = request_data.get('path', '/')
        payload = request_data.get('payload', {})
        
        validation_result = {
            'allowed': True,
            'security_score': 100.0,
            'threats_detected': [],
            'actions_taken': [],
            'security_headers': self._get_security_headers()
        }
        
        try:
            # Check if IP is blocked
            if await self._is_ip_blocked(source_ip):
                validation_result['allowed'] = False
                validation_result['threats_detected'].append('blocked_ip')
                return validation_result
            
            # Rate limiting check
            rate_limit_result = await self._check_rate_limits(source_ip, path)
            if not rate_limit_result['allowed']:
                validation_result.update(rate_limit_result)
                await self._record_security_event(
                    SecurityEventType.RATE_LIMIT_EXCEEDED,
                    SecurityThreatLevel.MEDIUM,
                    source_ip,
                    user_id,
                    path,
                    details={'rate_limit_exceeded': rate_limit_result['details']}
                )
                return validation_result
            
            # Input validation
            input_validation = await self._validate_input(payload, source_ip, user_id, path)
            if input_validation['threats_detected']:
                validation_result['security_score'] -= len(input_validation['threats_detected']) * 20
                validation_result['threats_detected'].extend(input_validation['threats_detected'])
                validation_result['actions_taken'].extend(input_validation['actions_taken'])
                
                # Block if critical threats detected
                if any(threat in ['sql_injection', 'command_injection', 'malicious_payload'] 
                      for threat in input_validation['threats_detected']):
                    validation_result['allowed'] = False
            
            # Behavioral analysis
            behavioral_result = await self._analyze_behavior(source_ip, user_id, path)
            if behavioral_result['suspicious']:
                validation_result['security_score'] -= 30
                validation_result['threats_detected'].append('suspicious_behavior')
                
                await self._record_security_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    SecurityThreatLevel.MEDIUM,
                    source_ip,
                    user_id,
                    path,
                    details=behavioral_result['details']
                )
            
            # Authentication validation (if user_id provided)
            if user_id:
                auth_result = await self._validate_authentication(user_id, request_data)
                if not auth_result['valid']:
                    validation_result['security_score'] -= 50
                    validation_result['threats_detected'].append('invalid_authentication')
                    
                    await self._record_security_event(
                        SecurityEventType.AUTHENTICATION_FAILURE,
                        SecurityThreatLevel.HIGH,
                        source_ip,
                        user_id,
                        path
                    )
            
            # Final security decision
            if validation_result['security_score'] < 50:
                validation_result['allowed'] = False
                validation_result['actions_taken'].append('request_blocked')
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Security validation error: {e}")
            # Fail secure - block request on error
            return {
                'allowed': False,
                'security_score': 0.0,
                'threats_detected': ['validation_error'],
                'actions_taken': ['request_blocked'],
                'error': str(e)
            }
    
    async def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP address is blocked."""
        return ip in self.blocked_ips
    
    async def _check_rate_limits(self, source_ip: str, path: str) -> Dict[str, Any]:
        """Check rate limiting policies."""
        current_time = time.time()
        result = {'allowed': True, 'details': {}}
        
        # API rate limiting
        api_policy = self.policies['api_rate_limit']
        if api_policy.enabled:
            key = f"api:{source_ip}"
            window = api_policy.parameters['window_minutes'] * 60
            limit = api_policy.parameters['requests_per_minute']
            
            if key not in self.rate_limits:
                self.rate_limits[key] = []
            
            # Clean old timestamps
            self.rate_limits[key] = [
                ts for ts in self.rate_limits[key] 
                if current_time - ts < window
            ]
            
            if len(self.rate_limits[key]) >= limit:
                result['allowed'] = False
                result['details']['api_rate_limit_exceeded'] = True
                return result
            
            self.rate_limits[key].append(current_time)
        
        # Generation-specific rate limiting
        if '/generate' in path:
            gen_policy = self.policies['generation_rate_limit']
            if gen_policy.enabled:
                key = f"generation:{source_ip}"
                window = gen_policy.parameters['window_hours'] * 3600
                limit = gen_policy.parameters['generations_per_hour']
                
                if key not in self.rate_limits:
                    self.rate_limits[key] = []
                
                # Clean old timestamps
                self.rate_limits[key] = [
                    ts for ts in self.rate_limits[key]
                    if current_time - ts < window
                ]
                
                if len(self.rate_limits[key]) >= limit:
                    result['allowed'] = False
                    result['details']['generation_rate_limit_exceeded'] = True
                    return result
                
                self.rate_limits[key].append(current_time)
        
        return result
    
    async def _validate_input(self, payload: Dict[str, Any], source_ip: str, 
                             user_id: Optional[str], path: str) -> Dict[str, Any]:
        """Validate input for malicious patterns."""
        threats_detected = []
        actions_taken = []
        
        # Convert payload to string for pattern matching
        payload_str = str(payload).lower()
        
        # Check for SQL injection
        if self.policies['injection_prevention'].enabled:
            for pattern in self.malicious_patterns['sql_injection']:
                if re.search(pattern, payload_str, re.IGNORECASE):
                    threats_detected.append('sql_injection')
                    actions_taken.append('blocked_sql_injection')
                    await self._record_security_event(
                        SecurityEventType.INJECTION_ATTEMPT,
                        SecurityThreatLevel.CRITICAL,
                        source_ip, user_id, path,
                        details={'injection_type': 'sql', 'pattern': pattern}
                    )
                    break
        
        # Check for NoSQL injection
        for pattern in self.malicious_patterns['nosql_injection']:
            if re.search(pattern, payload_str, re.IGNORECASE):
                threats_detected.append('nosql_injection')
                actions_taken.append('blocked_nosql_injection')
                await self._record_security_event(
                    SecurityEventType.INJECTION_ATTEMPT,
                    SecurityThreatLevel.CRITICAL,
                    source_ip, user_id, path,
                    details={'injection_type': 'nosql', 'pattern': pattern}
                )
                break
        
        # Check for command injection
        for pattern in self.malicious_patterns['command_injection']:
            if re.search(pattern, payload_str, re.IGNORECASE):
                threats_detected.append('command_injection')
                actions_taken.append('blocked_command_injection')
                await self._record_security_event(
                    SecurityEventType.INJECTION_ATTEMPT,
                    SecurityThreatLevel.CRITICAL,
                    source_ip, user_id, path,
                    details={'injection_type': 'command', 'pattern': pattern}
                )
                break
        
        # Check for XSS
        if self.policies['xss_prevention'].enabled:
            for pattern in self.malicious_patterns['xss_patterns']:
                if re.search(pattern, payload_str, re.IGNORECASE):
                    threats_detected.append('xss_attempt')
                    actions_taken.append('blocked_xss')
                    await self._record_security_event(
                        SecurityEventType.MALICIOUS_PAYLOAD,
                        SecurityThreatLevel.HIGH,
                        source_ip, user_id, path,
                        details={'attack_type': 'xss', 'pattern': pattern}
                    )
                    break
        
        # Check for path traversal
        for pattern in self.malicious_patterns['path_traversal']:
            if re.search(pattern, payload_str, re.IGNORECASE):
                threats_detected.append('path_traversal')
                actions_taken.append('blocked_path_traversal')
                await self._record_security_event(
                    SecurityEventType.MALICIOUS_PAYLOAD,
                    SecurityThreatLevel.HIGH,
                    source_ip, user_id, path,
                    details={'attack_type': 'path_traversal', 'pattern': pattern}
                )
                break
        
        # Check for sensitive data exposure
        if self.policies['sensitive_data_protection'].enabled:
            for pattern in self.malicious_patterns['sensitive_data']:
                if re.search(pattern, payload_str, re.IGNORECASE):
                    threats_detected.append('sensitive_data_exposure')
                    actions_taken.append('sanitized_sensitive_data')
                    await self._record_security_event(
                        SecurityEventType.DATA_BREACH_ATTEMPT,
                        SecurityThreatLevel.CRITICAL,
                        source_ip, user_id, path,
                        details={'data_type': 'sensitive', 'pattern': pattern}
                    )
                    break
        
        return {
            'threats_detected': threats_detected,
            'actions_taken': actions_taken
        }
    
    async def _analyze_behavior(self, source_ip: str, user_id: Optional[str], path: str) -> Dict[str, Any]:
        """Analyze user behavior for anomalies."""
        current_time = time.time()
        suspicious = False
        details = {}
        
        # Track IP activity
        if source_ip not in self.suspicious_ips:
            self.suspicious_ips[source_ip] = []
        
        # Clean old timestamps (last hour)
        hour_ago = current_time - 3600
        self.suspicious_ips[source_ip] = [
            ts for ts in self.suspicious_ips[source_ip] if ts > hour_ago
        ]
        
        self.suspicious_ips[source_ip].append(current_time)
        
        # Check for suspicious patterns
        recent_requests = len(self.suspicious_ips[source_ip])
        
        # Too many requests from single IP
        if recent_requests > 300:  # More than 300 requests per hour
            suspicious = True
            details['high_frequency_requests'] = recent_requests
        
        # Check for rapid-fire requests
        if len(self.suspicious_ips[source_ip]) >= 3:
            last_three = self.suspicious_ips[source_ip][-3:]
            if last_three[-1] - last_three[0] < 5:  # 3 requests in 5 seconds
                suspicious = True
                details['rapid_fire_requests'] = True
        
        # Check for unusual time patterns (3 AM - 6 AM might be suspicious)
        hour = time.localtime(current_time).tm_hour
        if 3 <= hour <= 6:
            details['unusual_time'] = hour
            # Don't mark as suspicious by default, just note it
        
        return {
            'suspicious': suspicious,
            'details': details
        }
    
    async def _validate_authentication(self, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate authentication credentials."""
        # Check for JWT token
        auth_header = request_data.get('headers', {}).get('authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return {'valid': False, 'reason': 'missing_bearer_token'}
        
        token = auth_header.replace('Bearer ', '')
        
        try:
            # Verify JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check token expiration
            if payload.get('exp', 0) < time.time():
                return {'valid': False, 'reason': 'token_expired'}
            
            # Check user_id matches
            if payload.get('user_id') != user_id:
                return {'valid': False, 'reason': 'user_id_mismatch'}
            
            return {'valid': True, 'payload': payload}
            
        except jwt.InvalidTokenError as e:
            return {'valid': False, 'reason': f'invalid_token: {str(e)}'}
    
    def generate_jwt_token(self, user_id: str, additional_claims: Dict[str, Any] = None) -> str:
        """Generate JWT token for user."""
        now = time.time()
        payload = {
            'user_id': user_id,
            'iat': now,
            'exp': now + (self.jwt_expiry * 3600),
            'iss': 'synthetic-data-guardian'
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    async def _record_security_event(self, event_type: SecurityEventType, 
                                   threat_level: SecurityThreatLevel,
                                   source_ip: str, user_id: Optional[str] = None,
                                   request_path: Optional[str] = None,
                                   details: Dict[str, Any] = None) -> None:
        """Record security event for analysis and alerting."""
        event = SecurityEvent(
            event_type=event_type,
            threat_level=threat_level,
            timestamp=time.time(),
            source_ip=source_ip,
            user_id=user_id,
            request_path=request_path,
            details=details or {}
        )
        
        self.security_events.append(event)
        
        # Log security event
        self.logger.warning(
            f"Security Event: {event_type.value} from {source_ip} "
            f"(threat: {threat_level.value})"
        )
        
        # Take automated action based on threat level
        if threat_level == SecurityThreatLevel.CRITICAL:
            await self._handle_critical_threat(event)
        elif threat_level == SecurityThreatLevel.HIGH:
            await self._handle_high_threat(event)
    
    async def _handle_critical_threat(self, event: SecurityEvent) -> None:
        """Handle critical security threats."""
        # Immediately block IP for critical threats
        self.blocked_ips.add(event.source_ip)
        
        # Record action taken
        event.action_taken = "ip_blocked"
        
        self.logger.critical(f"CRITICAL THREAT: Blocked IP {event.source_ip} due to {event.event_type.value}")
        
        # Additional actions could include:
        # - Alerting security team
        # - Triggering incident response
        # - Updating firewall rules
    
    async def _handle_high_threat(self, event: SecurityEvent) -> None:
        """Handle high-level security threats."""
        # Increase monitoring for this IP
        if event.source_ip not in self.suspicious_ips:
            self.suspicious_ips[event.source_ip] = []
        
        # Track failed attempts
        key = f"{event.source_ip}:{event.user_id or 'anonymous'}"
        self.failed_attempts[key] = self.failed_attempts.get(key, 0) + 1
        
        # Block IP after multiple high-threat events
        if self.failed_attempts[key] >= 3:
            self.blocked_ips.add(event.source_ip)
            event.action_taken = "ip_blocked_after_multiple_attempts"
            
            self.logger.error(f"HIGH THREAT: Blocked IP {event.source_ip} after multiple attempts")
    
    def _get_security_headers(self) -> Dict[str, str]:
        """Get security headers to include in responses."""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
    
    def sanitize_input(self, input_data: Any) -> Any:
        """Sanitize input data to prevent injection attacks."""
        if isinstance(input_data, str):
            # Remove dangerous characters
            sanitized = input_data
            
            # Replace SQL injection characters
            dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'xp_', 'sp_']
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            
            # Escape HTML characters
            html_escapes = {
                '<': '&lt;',
                '>': '&gt;',
                '&': '&amp;',
                '"': '&quot;',
                "'": '&#x27;'
            }
            
            for char, escape in html_escapes.items():
                sanitized = sanitized.replace(char, escape)
            
            return sanitized
            
        elif isinstance(input_data, dict):
            return {key: self.sanitize_input(value) for key, value in input_data.items()}
        elif isinstance(input_data, list):
            return [self.sanitize_input(item) for item in input_data]
        else:
            return input_data
    
    def hash_sensitive_data(self, data: str, salt: str = None) -> Dict[str, str]:
        """Hash sensitive data for secure storage."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for password hashing
        hashed = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        
        return {
            'hash': hashed.hex(),
            'salt': salt,
            'algorithm': 'pbkdf2_sha256'
        }
    
    def verify_hash(self, data: str, stored_hash: str, salt: str) -> bool:
        """Verify hashed data."""
        computed_hash = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return hmac.compare_digest(computed_hash.hex(), stored_hash)
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics."""
        current_time = time.time()
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        # Filter recent events
        recent_events = [e for e in self.security_events if e.timestamp > hour_ago]
        daily_events = [e for e in self.security_events if e.timestamp > day_ago]
        
        # Count by threat level
        threat_counts = {
            'critical': len([e for e in recent_events if e.threat_level == SecurityThreatLevel.CRITICAL]),
            'high': len([e for e in recent_events if e.threat_level == SecurityThreatLevel.HIGH]),
            'medium': len([e for e in recent_events if e.threat_level == SecurityThreatLevel.MEDIUM]),
            'low': len([e for e in recent_events if e.threat_level == SecurityThreatLevel.LOW])
        }
        
        # Count by event type
        event_type_counts = {}
        for event_type in SecurityEventType:
            event_type_counts[event_type.value] = len([
                e for e in recent_events if e.event_type == event_type
            ])
        
        return {
            'total_events_last_hour': len(recent_events),
            'total_events_last_day': len(daily_events),
            'blocked_ips_count': len(self.blocked_ips),
            'suspicious_ips_count': len(self.suspicious_ips),
            'threat_level_distribution': threat_counts,
            'event_type_distribution': event_type_counts,
            'active_rate_limits': len(self.rate_limits),
            'security_score': self._calculate_security_score()
        }
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score."""
        base_score = 100.0
        
        # Deduct points for recent critical events
        current_time = time.time()
        hour_ago = current_time - 3600
        
        critical_events = len([
            e for e in self.security_events 
            if e.timestamp > hour_ago and e.threat_level == SecurityThreatLevel.CRITICAL
        ])
        
        high_events = len([
            e for e in self.security_events
            if e.timestamp > hour_ago and e.threat_level == SecurityThreatLevel.HIGH
        ])
        
        # Deduct points for security events
        score = base_score - (critical_events * 20) - (high_events * 10)
        
        # Deduct points for blocked IPs (indicates attacks)
        score -= len(self.blocked_ips) * 2
        
        return max(score, 0.0)
    
    async def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        metrics = self.get_security_metrics()
        
        # Top threat sources
        ip_threat_counts = {}
        for event in self.security_events:
            if event.source_ip not in ip_threat_counts:
                ip_threat_counts[event.source_ip] = 0
            ip_threat_counts[event.source_ip] += 1
        
        top_threat_ips = sorted(ip_threat_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Recent critical events
        current_time = time.time()
        day_ago = current_time - 86400
        
        critical_events = [
            {
                'timestamp': event.timestamp,
                'event_type': event.event_type.value,
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'details': event.details
            }
            for event in self.security_events
            if event.timestamp > day_ago and event.threat_level == SecurityThreatLevel.CRITICAL
        ]
        
        return {
            'report_timestamp': current_time,
            'metrics': metrics,
            'top_threat_sources': top_threat_ips,
            'recent_critical_events': critical_events,
            'blocked_ips': list(self.blocked_ips),
            'security_policies_enabled': [
                name for name, policy in self.policies.items() if policy.enabled
            ],
            'recommendations': self._generate_security_recommendations(metrics)
        }
    
    def _generate_security_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on metrics."""
        recommendations = []
        
        if metrics['threat_level_distribution']['critical'] > 0:
            recommendations.append("Investigate critical security events immediately")
        
        if metrics['blocked_ips_count'] > 10:
            recommendations.append("Review blocked IPs and consider network-level blocking")
        
        if metrics['security_score'] < 80:
            recommendations.append("Security score is below optimal - review and address threats")
        
        if metrics['total_events_last_hour'] > 100:
            recommendations.append("High volume of security events - consider additional monitoring")
        
        recommendations.extend([
            "Regularly update security policies and patterns",
            "Implement automated threat intelligence feeds",
            "Consider implementing additional authentication factors",
            "Review and audit security logs regularly"
        ])
        
        return recommendations
    
    async def cleanup_old_events(self, retention_days: int = 30) -> None:
        """Clean up old security events."""
        cutoff_time = time.time() - (retention_days * 86400)
        
        old_count = len(self.security_events)
        self.security_events = [
            event for event in self.security_events
            if event.timestamp > cutoff_time
        ]
        
        cleaned_count = old_count - len(self.security_events)
        self.logger.info(f"Cleaned up {cleaned_count} old security events")
        
        # Also clean up old rate limit data
        current_time = time.time()
        for key in list(self.rate_limits.keys()):
            self.rate_limits[key] = [
                ts for ts in self.rate_limits[key]
                if current_time - ts < 3600  # Keep last hour
            ]
            
            if not self.rate_limits[key]:
                del self.rate_limits[key]