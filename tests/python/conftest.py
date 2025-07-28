"""Pytest configuration and fixtures for Python tests."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock

# Set test environment
os.environ['TESTING'] = 'true'
os.environ['NODE_ENV'] = 'test'
os.environ['LOG_LEVEL'] = 'ERROR'


@pytest.fixture(scope='session')
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope='session')
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_csv_data():
    """Generate sample CSV data for testing."""
    return pd.DataFrame({
        'id': range(1, 101),
        'name': [f'User {i}' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 100),
        'is_active': np.random.choice([True, False], 100),
        'signup_date': pd.date_range('2020-01-01', periods=100, freq='D'),
    })


@pytest.fixture
def sample_schema():
    """Sample schema definition for testing."""
    return {
        'id': {'type': 'integer', 'constraints': {'min': 1}},
        'name': {'type': 'string', 'constraints': {'max_length': 100}},
        'age': {'type': 'integer', 'constraints': {'min': 18, 'max': 120}},
        'income': {'type': 'float', 'constraints': {'min': 0}},
        'city': {'type': 'categorical', 'constraints': {'categories': ['NYC', 'LA', 'Chicago', 'Houston']}},
        'is_active': {'type': 'boolean'},
        'signup_date': {'type': 'datetime'},
    }


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    db = Mock()
    db.connect = Mock(return_value=None)
    db.disconnect = Mock(return_value=None)
    db.execute = Mock(return_value={'rows': [], 'count': 0})
    db.transaction = Mock()
    return db


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_client = Mock()
    redis_client.get = Mock(return_value=None)
    redis_client.set = Mock(return_value=True)
    redis_client.delete = Mock(return_value=1)
    redis_client.exists = Mock(return_value=False)
    redis_client.expire = Mock(return_value=True)
    return redis_client


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j driver for testing."""
    driver = Mock()
    session = Mock()
    driver.session = Mock(return_value=session)
    session.run = Mock(return_value=Mock(records=[]))
    session.close = Mock(return_value=None)
    return driver


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    s3 = Mock()
    s3.upload_file = Mock(return_value=None)
    s3.download_file = Mock(return_value=None)
    s3.delete_object = Mock(return_value=None)
    s3.list_objects_v2 = Mock(return_value={'Contents': []})
    return s3


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    client = Mock()
    client.chat.completions.create = Mock(return_value=Mock(
        choices=[Mock(message=Mock(content='Mock generated text'))]
    ))
    return client


@pytest.fixture
def mock_external_apis():
    """Mock all external API clients."""
    return {
        'openai': mock_openai_client(),
        'anthropic': Mock(),
        'stability': Mock(),
        'huggingface': Mock(),
    }


@pytest.fixture
def generation_config():
    """Sample generation configuration for testing."""
    return {
        'generator': 'sdv',
        'model': 'gaussian_copula',
        'epochs': 100,
        'batch_size': 500,
        'privacy': {
            'enabled': True,
            'epsilon': 1.0,
            'delta': 1e-5,
        },
        'validation': {
            'statistical_threshold': 0.9,
            'privacy_threshold': 0.8,
            'bias_threshold': 0.1,
        },
        'watermarking': {
            'enabled': True,
            'method': 'statistical',
            'strength': 0.8,
        },
    }


@pytest.fixture
def validation_config():
    """Sample validation configuration for testing."""
    return {
        'validators': [
            {
                'type': 'statistical_fidelity',
                'config': {
                    'threshold': 0.9,
                    'metrics': ['ks_test', 'wasserstein', 'correlation'],
                },
            },
            {
                'type': 'privacy_preservation',
                'config': {
                    'epsilon': 1.0,
                    'attack_models': ['linkage', 'inference'],
                },
            },
            {
                'type': 'bias_detection',
                'config': {
                    'protected_attributes': ['age', 'city'],
                    'fairness_metrics': ['demographic_parity'],
                },
            },
        ],
    }


@pytest.fixture
def mock_user():
    """Mock user object for testing."""
    return {
        'id': 'test-user-123',
        'email': 'test@example.com',
        'role': 'data_scientist',
        'permissions': ['generate', 'validate', 'view_lineage'],
        'api_key': 'test-api-key-123',
    }


@pytest.fixture
def mock_job():
    """Mock generation job for testing."""
    return {
        'id': 'job-test-123',
        'user_id': 'test-user-123',
        'status': 'pending',
        'pipeline_config': {
            'generator': 'mock',
            'num_records': 100,
        },
        'created_at': '2024-01-15T10:00:00Z',
        'updated_at': '2024-01-15T10:00:00Z',
    }


@pytest.fixture(autouse=True)
def cleanup_temp_files(temp_dir):
    """Automatically clean up temporary files after each test."""
    yield
    # Cleanup is handled by temp_dir fixture


@pytest.fixture(scope='function')
def isolated_test():
    """Ensure each test runs in isolation."""
    # Setup isolation
    original_env = os.environ.copy()
    
    yield
    
    # Cleanup isolation
    os.environ.clear()
    os.environ.update(original_env)


# Pytest hooks
def pytest_configure(config):
    """Configure pytest settings."""
    # Register custom markers
    config.addinivalue_line('markers', 'unit: Unit tests')
    config.addinivalue_line('markers', 'integration: Integration tests')
    config.addinivalue_line('markers', 'slow: Slow running tests')
    config.addinivalue_line('markers', 'fast: Fast running tests')
    config.addinivalue_line('markers', 'security: Security-related tests')
    config.addinivalue_line('markers', 'privacy: Privacy-related tests')
    config.addinivalue_line('markers', 'performance: Performance tests')
    config.addinivalue_line('markers', 'compliance: Compliance tests')


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add markers based on test file path
        test_path = str(item.fspath)
        
        if '/unit/' in test_path:
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.fast)
        elif '/integration/' in test_path:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        elif '/performance/' in test_path:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Add markers based on test name
        if 'security' in item.name.lower():
            item.add_marker(pytest.mark.security)
        if 'privacy' in item.name.lower():
            item.add_marker(pytest.mark.privacy)
        if 'compliance' in item.name.lower():
            item.add_marker(pytest.mark.compliance)


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Skip tests that require external services in CI
    if item.get_closest_marker('requires_network') and os.getenv('CI'):
        pytest.skip('Skipping network-dependent test in CI')
    
    if item.get_closest_marker('requires_gpu') and not torch_gpu_available():
        pytest.skip('Skipping GPU test - GPU not available')
    
    if item.get_closest_marker('requires_docker') and not docker_available():
        pytest.skip('Skipping Docker test - Docker not available')


def torch_gpu_available() -> bool:
    """Check if PyTorch GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def docker_available() -> bool:
    """Check if Docker is available."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


# Performance test helpers
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        'min_rounds': 5,
        'max_time': 10.0,
        'warmup': True,
        'warmup_iterations': 2,
    }


# Database test helpers
@pytest.fixture
def db_test_data():
    """Test data for database tests."""
    return {
        'users': [
            {'id': 'user1', 'email': 'user1@test.com', 'role': 'data_scientist'},
            {'id': 'user2', 'email': 'user2@test.com', 'role': 'privacy_officer'},
        ],
        'datasets': [
            {'id': 'dataset1', 'name': 'Test Dataset 1', 'schema': {}},
            {'id': 'dataset2', 'name': 'Test Dataset 2', 'schema': {}},
        ],
        'jobs': [
            {'id': 'job1', 'user_id': 'user1', 'status': 'completed'},
            {'id': 'job2', 'user_id': 'user2', 'status': 'pending'},
        ],
    }


# Mock data generators
@pytest.fixture
def mock_data_generator():
    """Factory for generating mock data."""
    def _generate(data_type: str, size: int = 100, **kwargs) -> Dict[str, Any]:
        if data_type == 'tabular':
            return {
                'data': pd.DataFrame({
                    'id': range(size),
                    'value': np.random.randn(size),
                    'category': np.random.choice(['A', 'B', 'C'], size),
                }),
                'schema': {'id': 'integer', 'value': 'float', 'category': 'categorical'},
            }
        elif data_type == 'timeseries':
            dates = pd.date_range('2024-01-01', periods=size, freq='D')
            return {
                'data': pd.DataFrame({
                    'date': dates,
                    'value': np.random.randn(size).cumsum(),
                }),
                'schema': {'date': 'datetime', 'value': 'float'},
            }
        else:
            raise ValueError(f'Unknown data type: {data_type}')
    
    return _generate