"""
Comprehensive Security Testing Suite for Synthetic Data Guardian
Tests for various security vulnerabilities and compliance requirements
"""

import pytest
import requests
import time
import json
import hashlib
import hmac
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess
import os
import re
from urllib.parse import urljoin, urlparse
import ssl
import socket

class SecurityTestConfig:
    """Configuration for security tests"""
    
    def __init__(self):
        self.base_url = os.getenv('TEST_API_BASE_URL', 'http://localhost:8080')
        self.api_key = os.getenv('TEST_API_KEY', 'test-api-key')
        self.admin_key = os.getenv('TEST_ADMIN_KEY', 'admin-test-key')
        self.test_timeout = 30
        self.max_retries = 3
        
        # Security testing parameters
        self.rate_limit_threshold = 100  # requests per minute
        self.injection_payloads = self._load_injection_payloads()
        self.xss_payloads = self._load_xss_payloads()
        self.auth_bypass_attempts = self._load_auth_bypass_payloads()
        
    def _load_injection_payloads(self) -> List[str]:
        """Load SQL/NoSQL injection test payloads"""
        return [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "{\"$ne\": null}",
            "{\"$gt\": \"\"}",
            "'; EXEC xp_cmdshell('dir'); --",
            "admin'/*",
            "' OR 1=1#",
            "') OR ('1'='1",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --"
        ]
    
    def _load_xss_payloads(self) -> List[str]:
        """Load XSS test payloads"""
        return [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//",
            "<iframe src=\"javascript:alert('XSS')\"></iframe>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>"
        ]
    
    def _load_auth_bypass_payloads(self) -> List[str]:
        """Load authentication bypass test payloads"""
        return [
            "admin",
            "administrator",
            "root",
            "guest",
            "test",
            "demo",
            "user",
            "admin123",
            "password",
            "123456"
        ]

class SecurityTester:
    """Main security testing class"""
    
    def __init__(self, config: SecurityTestConfig):
        self.config = config
        self.session = requests.Session()
        self.session.timeout = config.test_timeout
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests and return results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'target': self.config.base_url,
            'tests': {}
        }
        
        test_methods = [
            ('ssl_tls_security', self.test_ssl_tls_security),
            ('authentication_security', self.test_authentication_security),
            ('authorization_security', self.test_authorization_security),
            ('input_validation', self.test_input_validation),
            ('injection_attacks', self.test_injection_attacks),
            ('xss_protection', self.test_xss_protection),
            ('rate_limiting', self.test_rate_limiting),
            ('security_headers', self.test_security_headers),
            ('data_exposure', self.test_data_exposure),
            ('api_security', self.test_api_security),
            ('privacy_compliance', self.test_privacy_compliance),
            ('encryption_security', self.test_encryption_security)
        ]
        
        for test_name, test_method in test_methods:
            try:
                print(f"Running {test_name}...")
                results['tests'][test_name] = test_method()
            except Exception as e:
                results['tests'][test_name] = {
                    'status': 'error',
                    'error': str(e),
                    'passed': False
                }
        
        return results
    
    def test_ssl_tls_security(self) -> Dict[str, Any]:
        """Test SSL/TLS configuration and security"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        if self.config.base_url.startswith('https'):
            try:
                # Test SSL certificate
                hostname = urlparse(self.config.base_url).hostname
                context = ssl.create_default_context()
                
                with socket.create_connection((hostname, 443), timeout=10) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        cert = ssock.getpeercert()
                        
                        # Check certificate validity
                        not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                        days_to_expiry = (not_after - datetime.now()).days
                        
                        results['details']['certificate_expiry_days'] = days_to_expiry
                        results['details']['tls_version'] = ssock.version()
                        
                        if days_to_expiry < 30:
                            results['issues'].append(f"Certificate expires in {days_to_expiry} days")
                            results['passed'] = False
                            
                        # Check for weak TLS versions
                        if ssock.version() in ['TLSv1', 'TLSv1.1']:
                            results['issues'].append(f"Weak TLS version: {ssock.version()}")
                            results['passed'] = False
                            
            except Exception as e:
                results['issues'].append(f"SSL/TLS test failed: {str(e)}")
                results['passed'] = False
        else:
            results['issues'].append("HTTP used instead of HTTPS")
            results['passed'] = False
            
        return results
    
    def test_authentication_security(self) -> Dict[str, Any]:
        """Test authentication mechanisms"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        # Test access without authentication
        try:
            response = self.session.get(f"{self.config.base_url}/api/v1/generate")
            if response.status_code != 401:
                results['issues'].append("Endpoint accessible without authentication")
                results['passed'] = False
        except Exception as e:
            results['issues'].append(f"Authentication test failed: {str(e)}")
            results['passed'] = False
        
        # Test weak authentication bypass attempts
        for payload in self.config.auth_bypass_attempts:
            try:
                headers = {'Authorization': f'Bearer {payload}'}
                response = self.session.get(f"{self.config.base_url}/api/v1/generate", headers=headers)
                if response.status_code == 200:
                    results['issues'].append(f"Authentication bypassed with: {payload}")
                    results['passed'] = False
            except:
                continue
        
        # Test JWT token security if applicable
        if self.config.api_key:
            try:
                # Attempt to decode JWT without verification
                decoded = jwt.decode(self.config.api_key, options={"verify_signature": False})
                results['details']['jwt_decoded'] = True
                
                # Check for weak secrets or algorithms
                if 'alg' in decoded and decoded.get('alg') == 'none':
                    results['issues'].append("JWT uses 'none' algorithm")
                    results['passed'] = False
                    
            except jwt.InvalidTokenError:
                results['details']['jwt_decoded'] = False
        
        return results
    
    def test_authorization_security(self) -> Dict[str, Any]:
        """Test authorization and access control"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        # Test privilege escalation
        try:
            # Try accessing admin endpoints with regular user token
            headers = {'Authorization': f'Bearer {self.config.api_key}'}
            admin_endpoints = [
                '/api/v1/admin/users',
                '/api/v1/admin/config',
                '/api/v1/admin/logs',
                '/api/v1/admin/metrics'
            ]
            
            for endpoint in admin_endpoints:
                response = self.session.get(f"{self.config.base_url}{endpoint}", headers=headers)
                if response.status_code == 200:
                    results['issues'].append(f"Regular user can access admin endpoint: {endpoint}")
                    results['passed'] = False
                    
        except Exception as e:
            results['issues'].append(f"Authorization test failed: {str(e)}")
            results['passed'] = False
        
        return results
    
    def test_input_validation(self) -> Dict[str, Any]:
        """Test input validation and sanitization"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        # Test various malformed inputs
        malformed_inputs = [
            {"numRecords": -1},
            {"numRecords": "invalid"},
            {"schema": "not_an_object"},
            {"generator": "<script>alert('xss')</script>"},
            {"dataType": "../../../etc/passwd"},
            {"seed": "A" * 10000}  # Very long string
        ]
        
        headers = {'Authorization': f'Bearer {self.config.api_key}', 'Content-Type': 'application/json'}
        
        for malformed_input in malformed_inputs:
            try:
                response = self.session.post(
                    f"{self.config.base_url}/api/v1/generate",
                    json=malformed_input,
                    headers=headers
                )
                
                # Should return 400 for malformed input
                if response.status_code not in [400, 422]:
                    results['issues'].append(f"Malformed input accepted: {malformed_input}")
                    results['passed'] = False
                    
            except Exception as e:
                results['issues'].append(f"Input validation test failed: {str(e)}")
                results['passed'] = False
        
        return results
    
    def test_injection_attacks(self) -> Dict[str, Any]:
        """Test for SQL/NoSQL injection vulnerabilities"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        headers = {'Authorization': f'Bearer {self.config.api_key}', 'Content-Type': 'application/json'}
        
        for payload in self.config.injection_payloads:
            try:
                # Test in various fields
                test_data = {
                    "generator": payload,
                    "dataType": payload,
                    "schema": {"field": payload},
                    "numRecords": 10
                }
                
                response = self.session.post(
                    f"{self.config.base_url}/api/v1/generate",
                    json=test_data,
                    headers=headers
                )
                
                # Look for signs of successful injection
                if response.status_code == 500:
                    response_text = response.text.lower()
                    if any(error in response_text for error in ['sql', 'syntax error', 'mysql', 'postgres', 'mongodb']):
                        results['issues'].append(f"Possible injection vulnerability with payload: {payload}")
                        results['passed'] = False
                        
            except Exception as e:
                continue
        
        return results
    
    def test_xss_protection(self) -> Dict[str, Any]:
        """Test for XSS vulnerabilities"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        headers = {'Authorization': f'Bearer {self.config.api_key}', 'Content-Type': 'application/json'}
        
        for payload in self.config.xss_payloads:
            try:
                test_data = {
                    "generator": "sdv",
                    "dataType": "tabular",
                    "description": payload,
                    "numRecords": 1
                }
                
                response = self.session.post(
                    f"{self.config.base_url}/api/v1/generate",
                    json=test_data,
                    headers=headers
                )
                
                # Check if payload is reflected in response
                if payload in response.text:
                    results['issues'].append(f"Possible XSS vulnerability with payload: {payload}")
                    results['passed'] = False
                    
            except Exception as e:
                continue
        
        return results
    
    def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting implementation"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        headers = {'Authorization': f'Bearer {self.config.api_key}'}
        
        # Send rapid requests
        request_count = 0
        rate_limited = False
        
        start_time = time.time()
        
        for i in range(150):  # Send more than typical rate limit
            try:
                response = self.session.get(f"{self.config.base_url}/health", headers=headers)
                request_count += 1
                
                if response.status_code == 429:  # Too Many Requests
                    rate_limited = True
                    break
                    
                if time.time() - start_time > 60:  # Stop after 1 minute
                    break
                    
            except Exception as e:
                break
        
        results['details']['requests_sent'] = request_count
        results['details']['rate_limited'] = rate_limited
        
        if not rate_limited and request_count > self.config.rate_limit_threshold:
            results['issues'].append(f"No rate limiting detected after {request_count} requests")
            results['passed'] = False
        
        return results
    
    def test_security_headers(self) -> Dict[str, Any]:
        """Test for security headers"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        required_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': None,  # Just check presence for HTTPS
            'Content-Security-Policy': None,
            'Referrer-Policy': None
        }
        
        try:
            response = self.session.get(f"{self.config.base_url}/health")
            headers = response.headers
            
            for header, expected_values in required_headers.items():
                if header not in headers:
                    results['issues'].append(f"Missing security header: {header}")
                    results['passed'] = False
                elif expected_values and isinstance(expected_values, list):
                    if headers[header] not in expected_values:
                        results['issues'].append(f"Incorrect {header}: {headers[header]}")
                        results['passed'] = False
                elif expected_values and headers[header] != expected_values:
                    results['issues'].append(f"Incorrect {header}: {headers[header]}")
                    results['passed'] = False
            
            results['details']['headers'] = dict(headers)
            
        except Exception as e:
            results['issues'].append(f"Security headers test failed: {str(e)}")
            results['passed'] = False
        
        return results
    
    def test_data_exposure(self) -> Dict[str, Any]:
        """Test for sensitive data exposure"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        # Test common sensitive file exposure
        sensitive_files = [
            '/.env',
            '/config.json',
            '/package.json',
            '/admin',
            '/debug',
            '/metrics',
            '/logs',
            '/.git/config',
            '/backup',
            '/test'
        ]
        
        for file_path in sensitive_files:
            try:
                response = self.session.get(f"{self.config.base_url}{file_path}")
                if response.status_code == 200:
                    results['issues'].append(f"Sensitive file exposed: {file_path}")
                    results['passed'] = False
            except:
                continue
        
        return results
    
    def test_api_security(self) -> Dict[str, Any]:
        """Test API-specific security issues"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        # Test for verbose error messages
        try:
            headers = {'Authorization': 'Bearer invalid_token'}
            response = self.session.get(f"{self.config.base_url}/api/v1/generate", headers=headers)
            
            if any(keyword in response.text.lower() for keyword in ['stack trace', 'exception', 'debug', 'traceback']):
                results['issues'].append("Verbose error messages exposed")
                results['passed'] = False
                
        except Exception as e:
            results['issues'].append(f"API security test failed: {str(e)}")
            results['passed'] = False
        
        return results
    
    def test_privacy_compliance(self) -> Dict[str, Any]:
        """Test privacy and compliance features"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        headers = {'Authorization': f'Bearer {self.config.api_key}', 'Content-Type': 'application/json'}
        
        # Test data generation with privacy requirements
        try:
            test_data = {
                "generator": "sdv",
                "dataType": "tabular",
                "schema": {"name": "string", "ssn": "string", "email": "email"},
                "numRecords": 10,
                "privacyMode": "differential",
                "epsilon": 1.0
            }
            
            response = self.session.post(
                f"{self.config.base_url}/api/v1/generate",
                json=test_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check privacy score
                if 'privacyScore' not in result or result['privacyScore'] < 0.95:
                    results['issues'].append("Generated data does not meet privacy requirements")
                    results['passed'] = False
                
                # Check for PII in output
                if 'data' in result:
                    data_str = str(result['data'])
                    pii_patterns = [
                        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email pattern
                    ]
                    
                    for pattern in pii_patterns:
                        if re.search(pattern, data_str):
                            results['issues'].append("Potential PII found in synthetic data")
                            results['passed'] = False
                            break
                            
        except Exception as e:
            results['issues'].append(f"Privacy compliance test failed: {str(e)}")
            results['passed'] = False
        
        return results
    
    def test_encryption_security(self) -> Dict[str, Any]:
        """Test encryption and cryptographic security"""
        results = {
            'passed': True,
            'details': {},
            'issues': []
        }
        
        # Test watermark verification
        headers = {'Authorization': f'Bearer {self.config.api_key}', 'Content-Type': 'application/json'}
        
        try:
            test_data = {
                "generator": "sdv",
                "dataType": "tabular",
                "schema": {"value": "integer"},
                "numRecords": 10,
                "watermark": True
            }
            
            response = self.session.post(
                f"{self.config.base_url}/api/v1/generate",
                json=test_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'watermarked' not in result or not result['watermarked']:
                    results['issues'].append("Watermarking not properly implemented")
                    results['passed'] = False
                    
        except Exception as e:
            results['issues'].append(f"Encryption security test failed: {str(e)}")
            results['passed'] = False
        
        return results

@pytest.fixture
def security_config():
    """Pytest fixture for security test configuration"""
    return SecurityTestConfig()

@pytest.fixture
def security_tester(security_config):
    """Pytest fixture for security tester"""
    return SecurityTester(security_config)

# Pytest test functions
def test_ssl_tls_security(security_tester):
    """Test SSL/TLS security configuration"""
    results = security_tester.test_ssl_tls_security()
    assert results['passed'], f"SSL/TLS security issues: {results['issues']}"

def test_authentication_security(security_tester):
    """Test authentication mechanisms"""
    results = security_tester.test_authentication_security()
    assert results['passed'], f"Authentication security issues: {results['issues']}"

def test_authorization_security(security_tester):
    """Test authorization and access control"""
    results = security_tester.test_authorization_security()
    assert results['passed'], f"Authorization security issues: {results['issues']}"

def test_input_validation(security_tester):
    """Test input validation"""
    results = security_tester.test_input_validation()
    assert results['passed'], f"Input validation issues: {results['issues']}"

def test_injection_attacks(security_tester):
    """Test for injection vulnerabilities"""
    results = security_tester.test_injection_attacks()
    assert results['passed'], f"Injection vulnerability issues: {results['issues']}"

def test_xss_protection(security_tester):
    """Test XSS protection"""
    results = security_tester.test_xss_protection()
    assert results['passed'], f"XSS protection issues: {results['issues']}"

def test_rate_limiting(security_tester):
    """Test rate limiting"""
    results = security_tester.test_rate_limiting()
    assert results['passed'], f"Rate limiting issues: {results['issues']}"

def test_security_headers(security_tester):
    """Test security headers"""
    results = security_tester.test_security_headers()
    assert results['passed'], f"Security headers issues: {results['issues']}"

def test_data_exposure(security_tester):
    """Test for data exposure"""
    results = security_tester.test_data_exposure()
    assert results['passed'], f"Data exposure issues: {results['issues']}"

def test_privacy_compliance(security_tester):
    """Test privacy compliance"""
    results = security_tester.test_privacy_compliance()
    assert results['passed'], f"Privacy compliance issues: {results['issues']}"

if __name__ == "__main__":
    # Run security tests as standalone script
    config = SecurityTestConfig()
    tester = SecurityTester(config)
    
    print("Starting comprehensive security testing...")
    results = tester.run_all_tests()
    
    # Generate report
    with open('test-results/security-report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    total_tests = len(results['tests'])
    passed_tests = sum(1 for test in results['tests'].values() if test.get('passed', False))
    
    print(f"\nSecurity Test Summary:")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests < total_tests:
        print("\n⚠️ Security issues detected! Review the detailed report.")
        exit(1)
    else:
        print("\n✅ All security tests passed!")
        exit(0)