"""
Test configuration and constants for Synthetic Data Guardian tests

This module centralizes test configuration to ensure consistency
across all test modules.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class TestConfig:
    """Central configuration for all tests"""
    
    # Test data settings
    DEFAULT_NUM_ROWS: int = 100
    LARGE_DATASET_ROWS: int = 10000
    PERFORMANCE_TEST_ROWS: int = 100000
    
    # Timeout settings (seconds)
    UNIT_TEST_TIMEOUT: int = 30
    INTEGRATION_TEST_TIMEOUT: int = 300
    PERFORMANCE_TEST_TIMEOUT: int = 1800
    
    # Performance thresholds
    MAX_GENERATION_TIME_1K: float = 5.0  # seconds
    MAX_GENERATION_TIME_10K: float = 30.0  # seconds
    MAX_MEMORY_USAGE_MB: int = 2048
    MIN_THROUGHPUT_RECORDS_PER_SEC: int = 100
    
    # Privacy thresholds
    MAX_REIDENTIFICATION_RISK: float = 0.01  # 1%
    MIN_K_ANONYMITY: int = 5
    DEFAULT_EPSILON: float = 1.0
    DEFAULT_DELTA: float = 1e-5
    
    # Quality thresholds
    MIN_STATISTICAL_FIDELITY: float = 0.90
    MAX_STATISTICAL_DIVERGENCE: float = 0.10
    MIN_CORRELATION_PRESERVATION: float = 0.85
    MAX_DISTRIBUTION_DISTANCE: float = 0.15
    
    # Security settings
    TEST_ENCRYPTION_KEY: str = "test_key_32_chars_long_12345678"
    TEST_WATERMARK_KEY: str = "test_watermark_key_123456789012"
    SECURITY_SCAN_TIMEOUT: int = 600
    
    # File paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    TEST_DATA_DIR: Path = PROJECT_ROOT / "tests" / "fixtures"
    TEST_OUTPUT_DIR: Path = PROJECT_ROOT / "test-results"
    TEMP_DIR: Path = PROJECT_ROOT / "tmp"
    
    # Database settings for testing
    TEST_DB_URL: str = "sqlite:///:memory:"
    TEST_REDIS_URL: str = "redis://localhost:6379/15"  # Use DB 15 for tests
    TEST_NEO4J_URL: str = "bolt://localhost:7687"
    
    # API testing
    API_BASE_URL: str = "http://localhost:8080"
    API_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    
    # External service mocking
    MOCK_EXTERNAL_APIS: bool = True
    MOCK_CLOUD_STORAGE: bool = True
    MOCK_DATABASES: bool = False  # Use real test databases
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"
    TEST_LOG_FILE: str = "test.log"
    ENABLE_DEBUG_LOGGING: bool = False


class TestDataConfig:
    """Configuration for test data generation"""
    
    # Tabular data schemas
    BASIC_SCHEMA = {
        "id": "integer",
        "name": "string",
        "age": "integer[18:80]",
        "income": "float[20000:200000]",
        "city": "categorical",
        "active": "boolean"
    }
    
    PII_SCHEMA = {
        "id": "integer",
        "name": "string",
        "email": "email",
        "phone": "phone",
        "ssn": "ssn",
        "address": "address",
        "date_of_birth": "date"
    }
    
    FINANCIAL_SCHEMA = {
        "transaction_id": "uuid",
        "amount": "float[0.01:10000]",
        "timestamp": "datetime",
        "merchant_category": "categorical[retail,food,transport,utilities]",
        "user_id": "integer",
        "location": "geo_point"
    }
    
    HEALTHCARE_SCHEMA = {
        "patient_id": "uuid",
        "age": "integer[0:120]",
        "gender": "categorical[M,F,Other]",
        "condition": "categorical[diabetes,hypertension,none]",
        "treatment_date": "date",
        "provider_id": "integer"
    }
    
    # Text data templates
    TEXT_TEMPLATES = {
        "customer_review": "This product is {sentiment}. I would {recommendation} it to others.",
        "medical_note": "Patient presents with {symptoms}. Diagnosis: {condition}. Treatment: {treatment}.",
        "support_ticket": "Issue: {issue_type}. Description: {description}. Priority: {priority}."
    }
    
    # Image generation parameters
    IMAGE_CONFIG = {
        "formats": ["JPEG", "PNG", "TIFF"],
        "sizes": [(256, 256), (512, 512), (1024, 1024)],
        "color_modes": ["RGB", "RGBA", "L"],
        "compression_quality": [70, 85, 95]
    }


class SecurityTestConfig:
    """Configuration for security testing"""
    
    # Common attack vectors to test against
    ATTACK_VECTORS = {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/**/OR/**/'1'='1",
            "1; EXEC xp_cmdshell('dir')"
        ],
        "xss": [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "&#60;script&#62;alert('XSS')&#60;/script&#62;"
        ],
        "path_traversal": [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ],
        "command_injection": [
            "; ls -la",
            "| whoami",
            "&& cat /etc/passwd",
            "`id`"
        ]
    }
    
    # Vulnerability categories to scan for
    VULNERABILITY_CATEGORIES = [
        "injection",
        "broken_authentication",
        "sensitive_data_exposure",
        "xml_external_entities",
        "broken_access_control",
        "security_misconfiguration",
        "cross_site_scripting",
        "insecure_deserialization",
        "known_vulnerabilities",
        "insufficient_logging"
    ]
    
    # Encryption algorithms to test
    ENCRYPTION_ALGORITHMS = [
        "AES-256-GCM",
        "ChaCha20-Poly1305",
        "RSA-4096",
        "ECDSA-P256"
    ]


class ComplianceTestConfig:
    """Configuration for compliance testing"""
    
    # GDPR test scenarios
    GDPR_SCENARIOS = {
        "data_minimization": {
            "description": "Ensure only necessary data is processed",
            "test_cases": ["minimal_fields", "purpose_limitation", "storage_limitation"]
        },
        "consent_management": {
            "description": "Verify consent mechanisms",
            "test_cases": ["explicit_consent", "consent_withdrawal", "consent_granularity"]
        },
        "data_subject_rights": {
            "description": "Test data subject rights implementation",
            "test_cases": ["access_right", "rectification", "erasure", "portability"]
        }
    }
    
    # HIPAA test scenarios
    HIPAA_SCENARIOS = {
        "safe_harbor": {
            "description": "Verify Safe Harbor de-identification",
            "identifiers": [
                "names", "geographic_subdivisions", "dates", "phone_numbers",
                "fax_numbers", "email_addresses", "ssn", "medical_record_numbers"
            ]
        },
        "minimum_necessary": {
            "description": "Test minimum necessary standard",
            "test_cases": ["access_controls", "role_based_access", "audit_logs"]
        }
    }
    
    # SOC 2 controls
    SOC2_CONTROLS = {
        "security": ["logical_access", "system_monitoring", "change_management"],
        "availability": ["system_monitoring", "incident_response", "backup_recovery"],
        "processing_integrity": ["data_validation", "error_handling", "monitoring"],
        "confidentiality": ["encryption", "access_controls", "data_classification"],
        "privacy": ["notice", "choice", "collection", "use_retention", "access"]
    }


class PerformanceTestConfig:
    """Configuration for performance testing"""
    
    # Load test scenarios
    LOAD_SCENARIOS = {
        "baseline": {
            "users": 10,
            "duration": "1m",
            "ramp_up": "10s"
        },
        "normal_load": {
            "users": 50,
            "duration": "5m",
            "ramp_up": "30s"
        },
        "stress_test": {
            "users": 200,
            "duration": "10m",
            "ramp_up": "2m"
        },
        "spike_test": {
            "users": 500,
            "duration": "30s",
            "ramp_up": "5s"
        }
    }
    
    # Performance benchmarks
    BENCHMARKS = {
        "generation_throughput": {
            "tabular_1k": {"min_rps": 10, "max_latency": 100},
            "tabular_10k": {"min_rps": 1, "max_latency": 1000},
            "tabular_100k": {"min_rps": 0.1, "max_latency": 10000}
        },
        "validation_performance": {
            "statistical_fidelity": {"max_latency": 50},
            "privacy_assessment": {"max_latency": 200},
            "bias_detection": {"max_latency": 100}
        },
        "api_performance": {
            "health_check": {"max_latency": 10},
            "generate_endpoint": {"max_latency": 5000},
            "validate_endpoint": {"max_latency": 1000}
        }
    }
    
    # Resource usage limits
    RESOURCE_LIMITS = {
        "memory_mb": 4096,
        "cpu_percent": 80,
        "disk_io_mbps": 100,
        "network_mbps": 50
    }


# Environment-specific configurations
class EnvironmentConfig:
    """Environment-specific test configurations"""
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get configuration based on current environment"""
        env = os.getenv("TEST_ENV", "local")
        
        configs = {
            "local": {
                "parallel_workers": 2,
                "timeout_multiplier": 1.0,
                "enable_gpu_tests": False,
                "external_service_tests": False
            },
            "ci": {
                "parallel_workers": 4,
                "timeout_multiplier": 2.0,
                "enable_gpu_tests": False,
                "external_service_tests": True
            },
            "staging": {
                "parallel_workers": 8,
                "timeout_multiplier": 1.5,
                "enable_gpu_tests": True,
                "external_service_tests": True
            },
            "production": {
                "parallel_workers": 1,
                "timeout_multiplier": 0.5,
                "enable_gpu_tests": False,
                "external_service_tests": False
            }
        }
        
        return configs.get(env, configs["local"])


# Export main configuration instance
CONFIG = TestConfig()
DATA_CONFIG = TestDataConfig()
SECURITY_CONFIG = SecurityTestConfig()
COMPLIANCE_CONFIG = ComplianceTestConfig()
PERFORMANCE_CONFIG = PerformanceTestConfig()