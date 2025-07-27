"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Any
import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_tabular_data() -> pd.DataFrame:
    """Generate sample tabular data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'credit_score': np.random.randint(300, 850, 1000),
        'employment_years': np.random.randint(0, 40, 1000),
        'has_loan': np.random.choice([True, False], 1000),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 1000)
    })


@pytest.fixture
def sample_time_series_data() -> pd.DataFrame:
    """Generate sample time series data for testing."""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)
    return pd.DataFrame({
        'timestamp': dates,
        'value': np.cumsum(np.random.randn(100)),
        'volume': np.random.randint(100, 1000, 100)
    })


@pytest.fixture
def mock_database_session():
    """Mock database session for testing."""
    return MagicMock()


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    return MagicMock()


@pytest.fixture
def mock_neo4j_driver():
    """Mock Neo4j driver for testing."""
    return MagicMock()


@pytest.fixture
def test_config() -> dict[str, Any]:
    """Test configuration settings."""
    return {
        'database_url': 'sqlite:///:memory:',
        'redis_url': 'redis://localhost:6379/15',
        'neo4j_uri': 'bolt://localhost:7687',
        'environment': 'testing',
        'debug': True,
        'testing': True,
        'secret_key': 'test-secret-key',
        'encryption_key': 'test-encryption-key-32-bytes-long',
        'watermark_secret_key': 'test-watermark-key-32-bytes-long'
    }


@pytest.fixture
def sample_pipeline_config() -> dict[str, Any]:
    """Sample pipeline configuration for testing."""
    return {
        'name': 'test_pipeline',
        'description': 'Test pipeline for unit tests',
        'generation': {
            'backend': 'sdv',
            'params': {
                'model': 'gaussian_copula',
                'epochs': 10
            },
            'schema': {
                'age': 'integer[18:80]',
                'income': 'float[20000:200000]',
                'email': 'email'
            }
        },
        'validation': {
            'statistical_fidelity': {
                'enabled': True,
                'threshold': 0.8
            },
            'privacy_preservation': {
                'enabled': True,
                'epsilon': 1.0
            }
        },
        'watermarking': {
            'enabled': True,
            'method': 'statistical'
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    test_env_vars = {
        'ENVIRONMENT': 'testing',
        'DATABASE_URL': 'sqlite:///:memory:',
        'REDIS_URL': 'redis://localhost:6379/15',
        'NEO4J_URI': 'bolt://localhost:7687',
        'SECRET_KEY': 'test-secret-key',
        'ENCRYPTION_KEY': 'test-encryption-key-32-bytes-long',
        'WATERMARK_SECRET_KEY': 'test-watermark-key-32-bytes-long',
        'TESTING': 'true'
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def sample_synthetic_data() -> pd.DataFrame:
    """Generate sample synthetic data that looks realistic."""
    np.random.seed(123)
    return pd.DataFrame({
        'user_id': range(1000, 2000),
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.lognormal(10.5, 0.5, 1000),
        'credit_score': np.random.normal(700, 100, 1000).astype(int),
        'transaction_amount': np.random.exponential(100, 1000),
        'risk_category': np.random.choice(['low', 'medium', 'high'], 1000)
    })


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Generated synthetic text"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Generated synthetic text"
    mock_client.messages.create.return_value = mock_response
    return mock_client


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "privacy: mark test as privacy-related"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security-related"
    )
    config.addinivalue_line(
        "markers", "generator: mark test as generator-related"
    )
    config.addinivalue_line(
        "markers", "validator: mark test as validator-related"
    )


@pytest.fixture
def mock_environment_variables(monkeypatch):
    """Set up mock environment variables for testing."""
    env_vars = {
        'OPENAI_API_KEY': 'test-openai-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'HUGGINGFACE_API_KEY': 'test-hf-key',
        'STABILITY_API_KEY': 'test-stability-key',
        'AWS_ACCESS_KEY_ID': 'test-aws-key',
        'AWS_SECRET_ACCESS_KEY': 'test-aws-secret',
        'S3_BUCKET_NAME': 'test-bucket'
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed_time
        
        @property
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()


# Database fixtures for integration tests
@pytest.fixture
def test_database_url():
    """Provide test database URL."""
    return "sqlite:///:memory:"


@pytest.fixture 
def cleanup_test_files():
    """Cleanup test files after tests."""
    test_files = []
    
    def add_file(filepath):
        test_files.append(filepath)
    
    yield add_file
    
    # Cleanup
    for filepath in test_files:
        if os.path.exists(filepath):
            os.remove(filepath)