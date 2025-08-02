"""
Comprehensive Python Test Utilities
Provides common testing helpers, fixtures, and utilities for Python tests
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from faker import Faker

fake = Faker()


@dataclass
class TestConfig:
    """Test configuration settings"""
    database_url: str = "postgresql://test:test@localhost:5433/synthetic_guardian_test"
    redis_url: str = "redis://localhost:6380/0"
    api_base_url: str = "http://localhost:8080"
    test_data_path: str = "./tests/fixtures/data"
    performance_threshold_ms: int = 500
    load_test_duration: int = 30
    sample_size: int = 1000


test_config = TestConfig()


class TestDataFactory:
    """Factory for generating test data"""
    
    @staticmethod
    def create_sample_dataset(size: int = 1000) -> pd.DataFrame:
        """Create a sample tabular dataset for testing"""
        return pd.DataFrame({
            'user_id': [fake.uuid4() for _ in range(size)],
            'age': [fake.random_int(min=18, max=80) for _ in range(size)],
            'income': [fake.random_int(min=20000, max=200000) for _ in range(size)],
            'email': [fake.email() for _ in range(size)],
            'city': [fake.city() for _ in range(size)],
            'country': [fake.country() for _ in range(size)],
            'registration_date': [fake.date_time_this_year() for _ in range(size)],
            'is_premium': [fake.boolean() for _ in range(size)],
            'last_login': [fake.date_time_this_month() for _ in range(size)],
            'purchase_count': [fake.random_int(min=0, max=50) for _ in range(size)]
        })
    
    @staticmethod
    def create_medical_dataset(size: int = 500) -> pd.DataFrame:
        """Create a sample medical dataset for HIPAA testing"""
        diagnoses = ['Hypertension', 'Diabetes', 'Asthma', 'Depression', 'Arthritis', 'Migraine']
        treatments = ['Medication', 'Surgery', 'Therapy', 'Monitoring', 'Lifestyle Change']
        
        return pd.DataFrame({
            'patient_id': [fake.uuid4() for _ in range(size)],
            'age': [fake.random_int(min=0, max=100) for _ in range(size)],
            'gender': [fake.random_element(['M', 'F', 'O']) for _ in range(size)],
            'diagnosis': [fake.random_element(diagnoses) for _ in range(size)],
            'treatment': [fake.random_element(treatments) for _ in range(size)],
            'visit_date': [fake.date_this_year() for _ in range(size)],
            'cost': [fake.random_int(min=100, max=10000) for _ in range(size)],
            'insurance_id': [fake.random_int(min=100000, max=999999) for _ in range(size)],
            'provider_id': [fake.random_int(min=1000, max=9999) for _ in range(size)],
            'symptoms': [fake.text(max_nb_chars=100) for _ in range(size)]
        })
    
    @staticmethod
    def create_financial_dataset(size: int = 1000) -> pd.DataFrame:
        """Create a sample financial dataset for testing"""
        categories = ['retail', 'food', 'transport', 'utilities', 'entertainment', 'healthcare']
        statuses = ['pending', 'completed', 'failed', 'cancelled']
        
        return pd.DataFrame({
            'transaction_id': [fake.uuid4() for _ in range(size)],
            'user_id': [fake.uuid4() for _ in range(size)],
            'amount': [fake.random_int(min=1, max=10000) / 100 for _ in range(size)],
            'currency': [fake.currency_code() for _ in range(size)],
            'merchant': [fake.company() for _ in range(size)],
            'category': [fake.random_element(categories) for _ in range(size)],
            'status': [fake.random_element(statuses) for _ in range(size)],
            'timestamp': [fake.date_time_this_year() for _ in range(size)],
            'card_type': [fake.random_element(['credit', 'debit']) for _ in range(size)],
            'location': [fake.city() for _ in range(size)]
        })
    
    @staticmethod
    def create_time_series_data(size: int = 100, start_date: Optional[str] = None) -> pd.DataFrame:
        """Create time series data for testing"""
        if start_date:
            start = pd.to_datetime(start_date)
        else:
            start = pd.Timestamp.now() - pd.Timedelta(days=size)
        
        dates = pd.date_range(start=start, periods=size, freq='D')
        
        return pd.DataFrame({
            'timestamp': dates,
            'value': [fake.random_int(min=0, max=1000) for _ in range(size)],
            'metric': [fake.random_element(['cpu', 'memory', 'disk', 'network']) for _ in range(size)],
            'host': [fake.random_element(['server1', 'server2', 'server3']) for _ in range(size)],
            'environment': [fake.random_element(['dev', 'staging', 'prod']) for _ in range(size)]
        })
    
    @staticmethod
    def create_generation_request() -> Dict[str, Any]:
        """Create a sample generation request"""
        return {
            'pipeline': fake.word(),
            'num_records': fake.random_int(min=100, max=10000),
            'format': fake.random_element(['csv', 'json', 'parquet']),
            'parameters': {
                'model': fake.random_element(['gaussian_copula', 'ctgan', 'tvae']),
                'epochs': fake.random_int(min=50, max=500),
                'batch_size': fake.random_element([32, 64, 128, 256])
            },
            'validation': {
                'statistical': True,
                'privacy': True,
                'bias': True
            },
            'metadata': {
                'description': fake.text(max_nb_chars=100),
                'tags': [fake.word() for _ in range(3)],
                'created_by': fake.email()
            }
        }


class MockHelper:
    """Helper for creating mocks and patches"""
    
    @staticmethod
    def mock_database():
        """Create a mock database connection"""
        mock_db = MagicMock()
        mock_db.execute.return_value = MagicMock()
        mock_db.fetchall.return_value = []
        mock_db.fetchone.return_value = None
        mock_db.commit.return_value = None
        mock_db.rollback.return_value = None
        return mock_db
    
    @staticmethod
    def mock_redis():
        """Create a mock Redis client"""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = 0
        return mock_redis
    
    @staticmethod
    def mock_neo4j():
        """Create a mock Neo4j driver"""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_session.run.return_value = []
        return mock_driver
    
    @staticmethod
    def mock_logger():
        """Create a mock logger"""
        mock_logger = MagicMock()
        return mock_logger
    
    @staticmethod
    @contextmanager
    def patch_environment_variables(**env_vars):
        """Context manager to patch environment variables"""
        with patch.dict('os.environ', env_vars):
            yield


class PerformanceTester:
    """Utilities for performance testing"""
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    
    @staticmethod
    async def measure_async_execution_time(func, *args, **kwargs):
        """Measure execution time of an async function"""
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time
    
    @staticmethod
    def load_test(func, num_requests: int = 100, concurrency: int = 10):
        """Perform a simple load test"""
        import concurrent.futures
        import threading
        
        results = []
        times = []
        errors = []
        
        def worker():
            try:
                result, execution_time = PerformanceTester.measure_execution_time(func)
                results.append(result)
                times.append(execution_time)
            except Exception as e:
                errors.append(str(e))
        
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker) for _ in range(num_requests)]
            concurrent.futures.wait(futures)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        return {
            'total_requests': num_requests,
            'successful_requests': len(results),
            'failed_requests': len(errors),
            'success_rate': (len(results) / num_requests) * 100,
            'total_time': total_time,
            'average_response_time': sum(times) / len(times) if times else 0,
            'min_response_time': min(times) if times else 0,
            'max_response_time': max(times) if times else 0,
            'requests_per_second': num_requests / total_time,
            'errors': errors
        }


class SecurityTester:
    """Utilities for security testing"""
    
    @staticmethod
    def generate_sql_injection_payloads() -> List[str]:
        """Generate SQL injection test payloads"""
        return [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; EXEC xp_cmdshell('dir'); --",
            "' OR 1=1#",
            "admin'--",
            "admin' /*",
            "' OR 'x'='x",
            "1' AND '1'='1",
            "1' AND '1'='2"
        ]
    
    @staticmethod
    def generate_xss_payloads() -> List[str]:
        """Generate XSS test payloads"""
        return [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//",
            "<svg onload=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<body onload=alert('XSS')>",
            "<input type=image src=x onerror=alert('XSS')>",
            "<object data=javascript:alert('XSS')>",
            "<embed src=javascript:alert('XSS')>"
        ]
    
    @staticmethod
    def test_input_validation(func, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test input validation with malicious payloads"""
        results = {
            'sql_injection': [],
            'xss': [],
            'path_traversal': [],
            'command_injection': []
        }
        
        # Test SQL injection
        for payload in SecurityTester.generate_sql_injection_payloads():
            for key in test_data:
                if isinstance(test_data[key], str):
                    test_input = test_data.copy()
                    test_input[key] = payload
                    try:
                        result = func(test_input)
                        results['sql_injection'].append({
                            'payload': payload,
                            'field': key,
                            'result': 'vulnerable' if result else 'safe',
                            'error': None
                        })
                    except Exception as e:
                        results['sql_injection'].append({
                            'payload': payload,
                            'field': key,
                            'result': 'safe',
                            'error': str(e)
                        })
        
        # Test XSS
        for payload in SecurityTester.generate_xss_payloads():
            for key in test_data:
                if isinstance(test_data[key], str):
                    test_input = test_data.copy()
                    test_input[key] = payload
                    try:
                        result = func(test_input)
                        results['xss'].append({
                            'payload': payload,
                            'field': key,
                            'result': 'vulnerable' if result else 'safe',
                            'error': None
                        })
                    except Exception as e:
                        results['xss'].append({
                            'payload': payload,
                            'field': key,
                            'result': 'safe',
                            'error': str(e)
                        })
        
        return results


class ValidationHelper:
    """Helper for common validation tasks"""
    
    @staticmethod
    def assert_valid_uuid(value: str):
        """Assert that a string is a valid UUID"""
        try:
            uuid.UUID(value)
        except ValueError:
            pytest.fail(f"'{value}' is not a valid UUID")
    
    @staticmethod
    def assert_valid_email(value: str):
        """Assert that a string is a valid email"""
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            pytest.fail(f"'{value}' is not a valid email address")
    
    @staticmethod
    def assert_dataframe_shape(df: pd.DataFrame, expected_rows: int, expected_cols: int):
        """Assert DataFrame has expected shape"""
        actual_rows, actual_cols = df.shape
        assert actual_rows == expected_rows, f"Expected {expected_rows} rows, got {actual_rows}"
        assert actual_cols == expected_cols, f"Expected {expected_cols} columns, got {actual_cols}"
    
    @staticmethod
    def assert_no_null_values(df: pd.DataFrame, columns: Optional[List[str]] = None):
        """Assert DataFrame has no null values in specified columns"""
        if columns:
            check_df = df[columns]
        else:
            check_df = df
        
        null_counts = check_df.isnull().sum()
        null_columns = null_counts[null_counts > 0]
        
        if not null_columns.empty:
            pytest.fail(f"Found null values in columns: {null_columns.to_dict()}")
    
    @staticmethod
    def assert_response_time(execution_time_ms: float, threshold_ms: float = None):
        """Assert that execution time is within threshold"""
        if threshold_ms is None:
            threshold_ms = test_config.performance_threshold_ms
        
        assert execution_time_ms <= threshold_ms, \
            f"Execution time {execution_time_ms}ms exceeds threshold {threshold_ms}ms"
    
    @staticmethod
    def assert_privacy_score(score: float, minimum: float = 0.9):
        """Assert privacy score meets minimum threshold"""
        assert score >= minimum, f"Privacy score {score} below minimum {minimum}"
    
    @staticmethod
    def assert_quality_score(score: float, minimum: float = 0.8):
        """Assert quality score meets minimum threshold"""
        assert score >= minimum, f"Quality score {score} below minimum {minimum}"


@pytest.fixture
def sample_dataset():
    """Fixture providing a sample dataset"""
    return TestDataFactory.create_sample_dataset(100)


@pytest.fixture
def medical_dataset():
    """Fixture providing a medical dataset"""
    return TestDataFactory.create_medical_dataset(50)


@pytest.fixture
def financial_dataset():
    """Fixture providing a financial dataset"""
    return TestDataFactory.create_financial_dataset(100)


@pytest.fixture
def mock_database():
    """Fixture providing a mock database"""
    return MockHelper.mock_database()


@pytest.fixture
def mock_redis():
    """Fixture providing a mock Redis client"""
    return MockHelper.mock_redis()


@pytest.fixture
def mock_logger():
    """Fixture providing a mock logger"""
    return MockHelper.mock_logger()


@pytest.fixture
def performance_tester():
    """Fixture providing performance testing utilities"""
    return PerformanceTester()


@pytest.fixture
def security_tester():
    """Fixture providing security testing utilities"""
    return SecurityTester()


@pytest.fixture
def validation_helper():
    """Fixture providing validation utilities"""
    return ValidationHelper()


# Async context managers for testing
@asynccontextmanager
async def async_test_environment():
    """Async context manager for setting up test environment"""
    # Setup
    print("Setting up async test environment...")
    try:
        yield
    finally:
        # Cleanup
        print("Cleaning up async test environment...")


@contextmanager
def temp_environment_variables(**env_vars):
    """Context manager for temporary environment variables"""
    import os
    original_values = {}
    
    # Store original values and set new ones
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = str(value)
    
    try:
        yield
    finally:
        # Restore original values
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value