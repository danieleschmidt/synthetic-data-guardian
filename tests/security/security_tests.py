"""Security testing utilities and test cases."""

import pytest
import hashlib
import secrets
import re
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import json


class SecurityTester:
    """Security testing utilities."""
    
    @staticmethod
    def generate_sql_injection_payloads() -> List[str]:
        """Generate common SQL injection payloads."""
        return [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' OR 1=1 --",
            "'; SELECT * FROM users WHERE 't' = 't",
            "' UNION SELECT username, password FROM users --",
            "'; EXEC xp_cmdshell('dir'); --",
            "' OR 1=1 UNION SELECT null, table_name FROM information_schema.tables --",
            "admin' --",
            "admin' /*",
            "' OR SLEEP(5) --",
        ]
    
    @staticmethod
    def generate_xss_payloads() -> List[str]:
        """Generate common XSS payloads."""
        return [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>",
            "'-alert('XSS')-'",
        ]
    
    @staticmethod
    def generate_path_traversal_payloads() -> List[str]:
        """Generate path traversal payloads."""
        return [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "..%252F..%252F..%252Fetc%252Fpasswd",
            "....\\\\....\\\\....\\\\windows\\\\system32\\\\drivers\\\\etc\\\\hosts",
            "/etc/passwd%00.jpg",
            "....//....//....//....//etc/passwd",
        ]
    
    @staticmethod
    def generate_nosql_injection_payloads() -> List[Dict[str, Any]]:
        """Generate NoSQL injection payloads."""
        return [
            {"$ne": ""},
            {"$gt": ""},
            {"$regex": ".*"},
            {"$where": "function() { return true; }"},
            {"$or": [{"username": {"$ne": ""}}, {"password": {"$ne": ""}}]},
            {"username": {"$in": ["admin", "administrator", "root"]}},
            {"$expr": {"$eq": [1, 1]}},
        ]
    
    @staticmethod
    def is_sensitive_data_exposed(response_text: str) -> Dict[str, List[str]]:
        """Check if response contains sensitive data patterns."""
        patterns = {
            'credit_cards': [
                r'\b4[0-9]{12}(?:[0-9]{3})?\b',  # Visa
                r'\b5[1-5][0-9]{14}\b',  # MasterCard
                r'\b3[47][0-9]{13}\b',  # American Express
            ],
            'ssn': [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{9}\b',
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            'phone': [
                r'\b\d{3}-\d{3}-\d{4}\b',
                r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',
            ],
            'api_keys': [
                r'[Aa][Pp][Ii]_?[Kk][Ee][Yy].*[\'\"]\s*:\s*[\'\"]\w+[\'\"']',
                r'[Ss][Ee][Cc][Rr][Ee][Tt].*[\'\"]\s*:\s*[\'\"]\w+[\'\"']',
            ],
            'tokens': [
                r'[Bb][Ee][Aa][Rr][Ee][Rr]\s+\w+',
                r'[Tt][Oo][Kk][Ee][Nn].*[\'\"]\s*:\s*[\'\"]\w+[\'\"']',
            ],
        }
        
        findings = {}
        for category, pattern_list in patterns.items():
            matches = []
            for pattern in pattern_list:
                matches.extend(re.findall(pattern, response_text))
            if matches:
                findings[category] = matches
        
        return findings
    
    @staticmethod
    def check_password_strength(password: str) -> Dict[str, Any]:
        """Check password strength."""
        checks = {
            'length': len(password) >= 8,
            'uppercase': bool(re.search(r'[A-Z]', password)),
            'lowercase': bool(re.search(r'[a-z]', password)),
            'digits': bool(re.search(r'\d', password)),
            'special_chars': bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', password)),
            'no_common_patterns': not any(pattern in password.lower() for pattern in [
                'password', '123456', 'qwerty', 'admin', 'letmein'
            ]),
        }
        
        score = sum(checks.values())
        strength = 'weak' if score < 3 else 'medium' if score < 5 else 'strong'
        
        return {
            'score': score,
            'max_score': len(checks),
            'strength': strength,
            'checks': checks,
        }


@pytest.fixture
def security_tester():
    """Security tester fixture."""
    return SecurityTester()


@pytest.mark.security
class TestInputValidation:
    """Test input validation against various attacks."""
    
    def test_sql_injection_protection(self, security_tester):
        """Test SQL injection protection."""
        payloads = security_tester.generate_sql_injection_payloads()
        
        for payload in payloads:
            # Mock database query function
            with patch('your_app.database.execute_query') as mock_query:
                mock_query.return_value = {'rows': [], 'count': 0}
                
                # Test that payloads are properly sanitized
                # Replace with actual function that processes user input
                result = self.process_user_input(payload)
                
                # Verify that the payload didn't execute malicious SQL
                assert result is not None
                assert 'error' not in result.lower()
                # Add more specific assertions based on your implementation
    
    def test_xss_protection(self, security_tester):
        """Test XSS protection."""
        payloads = security_tester.generate_xss_payloads()
        
        for payload in payloads:
            # Test HTML sanitization
            sanitized = self.sanitize_html(payload)
            
            # Verify that script tags and event handlers are removed
            assert '<script>' not in sanitized.lower()
            assert 'javascript:' not in sanitized.lower()
            assert 'onerror=' not in sanitized.lower()
            assert 'onload=' not in sanitized.lower()
            assert 'onfocus=' not in sanitized.lower()
    
    def test_path_traversal_protection(self, security_tester):
        """Test path traversal protection."""
        payloads = security_tester.generate_path_traversal_payloads()
        
        for payload in payloads:
            # Test file path validation
            is_safe = self.validate_file_path(payload)
            
            # Verify that path traversal attempts are blocked
            assert not is_safe, f"Path traversal not blocked: {payload}"
    
    def test_nosql_injection_protection(self, security_tester):
        """Test NoSQL injection protection."""
        payloads = security_tester.generate_nosql_injection_payloads()
        
        for payload in payloads:
            # Test NoSQL query sanitization
            with patch('your_app.database.mongodb.find') as mock_query:
                mock_query.return_value = []
                
                # Test that payloads are properly sanitized
                result = self.process_nosql_query(payload)
                
                # Verify that malicious queries don't execute
                assert result is not None
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Mock function to process user input - replace with actual implementation."""
        # This should be replaced with your actual input processing function
        # that includes SQL injection protection
        return {'processed': True, 'input': user_input}
    
    def sanitize_html(self, html_input: str) -> str:
        """Mock HTML sanitization - replace with actual implementation."""
        # This should be replaced with your actual HTML sanitization function
        import html
        return html.escape(html_input)
    
    def validate_file_path(self, file_path: str) -> bool:
        """Mock file path validation - replace with actual implementation."""
        # This should be replaced with your actual path validation function
        normalized = file_path.replace('\\', '/').replace('//', '/')
        return not ('..' in normalized or normalized.startswith('/'))
    
    def process_nosql_query(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock NoSQL query processing - replace with actual implementation."""
        # This should be replaced with your actual NoSQL query processing
        return []


@pytest.mark.security
class TestAuthentication:
    """Test authentication and authorization."""
    
    def test_password_strength_requirements(self, security_tester):
        """Test password strength requirements."""
        weak_passwords = [
            'password',
            '123456',
            'qwerty',
            'admin',
            'pass',
            '12345678',
        ]
        
        strong_passwords = [
            'MyStr0ng!Pass',
            'C0mpl3x@P4ssw0rd',
            'Un1qu3$Str1ng!',
            'S3cur3&L0ng@Pass',
        ]
        
        for password in weak_passwords:
            result = security_tester.check_password_strength(password)
            assert result['strength'] != 'strong', f"Weak password rated as strong: {password}"
        
        for password in strong_passwords:
            result = security_tester.check_password_strength(password)
            assert result['strength'] == 'strong', f"Strong password not recognized: {password}"
    
    def test_rate_limiting(self):
        """Test rate limiting on authentication endpoints."""
        # Mock multiple failed login attempts
        with patch('your_app.auth.check_rate_limit') as mock_rate_limit:
            mock_rate_limit.side_effect = [True] * 5 + [False]  # Allow 5, then block
            
            # Simulate 6 login attempts
            for i in range(6):
                if i < 5:
                    result = self.attempt_login('user@example.com', 'wrong_password')
                    assert result['allowed'], f"Request {i+1} should be allowed"
                else:
                    result = self.attempt_login('user@example.com', 'wrong_password')
                    assert not result['allowed'], "Request should be rate limited"
    
    def test_jwt_token_validation(self):
        """Test JWT token validation."""
        # Test valid token
        valid_token = self.generate_test_jwt('user123')
        result = self.validate_jwt(valid_token)
        assert result['valid'], "Valid JWT should be accepted"
        assert result['user_id'] == 'user123'
        
        # Test expired token
        expired_token = self.generate_expired_jwt('user123')
        result = self.validate_jwt(expired_token)
        assert not result['valid'], "Expired JWT should be rejected"
        
        # Test tampered token
        tampered_token = valid_token[:-10] + 'tampered123'
        result = self.validate_jwt(tampered_token)
        assert not result['valid'], "Tampered JWT should be rejected"
    
    def attempt_login(self, email: str, password: str) -> Dict[str, Any]:
        """Mock login attempt - replace with actual implementation."""
        # This should be replaced with your actual login function
        return {'allowed': True, 'success': False}
    
    def generate_test_jwt(self, user_id: str) -> str:
        """Generate test JWT token."""
        # This should use your actual JWT generation logic
        return f"test.jwt.token.for.{user_id}"
    
    def generate_expired_jwt(self, user_id: str) -> str:
        """Generate expired JWT token."""
        # This should generate an actually expired token
        return f"expired.jwt.token.for.{user_id}"
    
    def validate_jwt(self, token: str) -> Dict[str, Any]:
        """Mock JWT validation - replace with actual implementation."""
        # This should be replaced with your actual JWT validation
        if 'tampered' in token:
            return {'valid': False, 'error': 'Invalid signature'}
        if 'expired' in token:
            return {'valid': False, 'error': 'Token expired'}
        
        user_id = token.split('.')[-1]
        return {'valid': True, 'user_id': user_id}


@pytest.mark.security
class TestDataPrivacy:
    """Test data privacy and encryption."""
    
    def test_sensitive_data_encryption(self):
        """Test that sensitive data is properly encrypted."""
        sensitive_data = {
            'ssn': '123-45-6789',
            'credit_card': '4111-1111-1111-1111',
            'email': 'user@example.com',
            'phone': '555-123-4567',
        }
        
        for field, value in sensitive_data.items():
            encrypted = self.encrypt_sensitive_data(value)
            
            # Verify data is encrypted (not plaintext)
            assert encrypted != value, f"Data not encrypted: {field}"
            assert len(encrypted) > len(value), f"Encrypted data too short: {field}"
            
            # Verify it can be decrypted
            decrypted = self.decrypt_sensitive_data(encrypted)
            assert decrypted == value, f"Decryption failed: {field}"
    
    def test_pii_data_masking(self, security_tester):
        """Test PII data masking in logs and responses."""
        response_with_pii = """
        {
            "user": {
                "email": "john.doe@example.com",
                "ssn": "123-45-6789",
                "credit_card": "4111-1111-1111-1111",
                "phone": "555-123-4567"
            }
        }
        """
        
        # Check if PII is exposed
        findings = security_tester.is_sensitive_data_exposed(response_with_pii)
        
        if findings:
            # If PII is found, it should be properly masked in production
            masked_response = self.mask_pii_data(response_with_pii)
            masked_findings = security_tester.is_sensitive_data_exposed(masked_response)
            
            # Verify PII is now masked
            assert not masked_findings, f"PII still exposed after masking: {masked_findings}"
    
    def test_data_anonymization(self):
        """Test data anonymization functions."""
        original_data = {
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'age': 30,
            'city': 'New York',
            'salary': 75000,
        }
        
        anonymized = self.anonymize_data(original_data)
        
        # Verify personally identifiable information is removed/changed
        assert anonymized['name'] != original_data['name']
        assert anonymized['email'] != original_data['email']
        
        # Verify non-PII data is preserved or generalized appropriately
        assert isinstance(anonymized['age'], int)
        assert anonymized['city'] in ['New York', 'Large City', '[CITY]']  # Allow for different anonymization strategies
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Mock encryption - replace with actual implementation."""
        # This should use actual encryption (AES, etc.)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Mock decryption - replace with actual implementation."""
        # In a real implementation, this would decrypt the data
        # For testing purposes, we'll use a simple mapping
        test_mappings = {
            hashlib.sha256('123-45-6789'.encode()).hexdigest(): '123-45-6789',
            hashlib.sha256('4111-1111-1111-1111'.encode()).hexdigest(): '4111-1111-1111-1111',
            hashlib.sha256('user@example.com'.encode()).hexdigest(): 'user@example.com',
            hashlib.sha256('555-123-4567'.encode()).hexdigest(): '555-123-4567',
        }
        return test_mappings.get(encrypted_data, 'DECRYPTION_FAILED')
    
    def mask_pii_data(self, data: str) -> str:
        """Mock PII masking - replace with actual implementation."""
        # This should implement actual PII masking
        masked = data
        masked = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX', masked)  # SSN
        masked = re.sub(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b', 'XXXX-XXXX-XXXX-XXXX', masked)  # Credit card
        masked = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', masked)  # Email
        masked = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', 'XXX-XXX-XXXX', masked)  # Phone
        return masked
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock data anonymization - replace with actual implementation."""
        # This should implement actual anonymization techniques
        anonymized = data.copy()
        
        # Simple anonymization for testing
        if 'name' in anonymized:
            anonymized['name'] = f"User_{secrets.token_hex(4)}"
        if 'email' in anonymized:
            anonymized['email'] = f"user_{secrets.token_hex(4)}@example.com"
        
        return anonymized


@pytest.mark.security
class TestApiSecurity:
    """Test API security measures."""
    
    def test_cors_headers(self):
        """Test CORS headers are properly configured."""
        response = self.make_api_request('GET', '/api/v1/health')
        headers = response.get('headers', {})
        
        # Check for proper CORS headers
        assert 'Access-Control-Allow-Origin' in headers
        assert headers['Access-Control-Allow-Origin'] != '*'  # Should not allow all origins in production
        
        if 'Access-Control-Allow-Methods' in headers:
            allowed_methods = headers['Access-Control-Allow-Methods'].split(', ')
            assert 'GET' in allowed_methods
            assert 'POST' in allowed_methods
    
    def test_security_headers(self):
        """Test security headers are present."""
        response = self.make_api_request('GET', '/api/v1/health')
        headers = response.get('headers', {})
        
        # Check for security headers
        security_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy',
        ]
        
        for header in security_headers:
            assert header in headers, f"Missing security header: {header}"
    
    def test_api_rate_limiting(self):
        """Test API rate limiting."""
        # Make multiple requests rapidly
        responses = []
        for i in range(10):
            response = self.make_api_request('GET', '/api/v1/test-endpoint')
            responses.append(response)
        
        # Check if rate limiting kicks in
        status_codes = [r.get('status_code', 200) for r in responses]
        rate_limited_responses = [code for code in status_codes if code == 429]
        
        # Expect some requests to be rate limited if making too many
        if len(responses) > 5:  # Assuming rate limit is 5 requests
            assert len(rate_limited_responses) > 0, "Rate limiting not working"
    
    def make_api_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Mock API request - replace with actual implementation."""
        # This should make actual HTTP requests to your API
        return {
            'status_code': 200,
            'headers': {
                'Access-Control-Allow-Origin': 'https://trusted-domain.com',
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
                'Content-Security-Policy': "default-src 'self'",
            },
            'body': {'message': 'OK'},
        }