"""
Common testing utilities for Synthetic Data Guardian

This module provides reusable utilities for testing data generation,
validation, privacy, and security functionality.
"""

import os
import json
import tempfile
import hashlib
import secrets
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
from faker import Faker


class TestDataGenerator:
    """Generate test data for various testing scenarios"""
    
    def __init__(self, seed: int = 42):
        self.fake = Faker()
        Faker.seed(seed)
        np.random.seed(seed)
    
    def generate_tabular_data(
        self, 
        num_rows: int = 100,
        include_pii: bool = False,
        include_sensitive: bool = False
    ) -> pd.DataFrame:
        """Generate realistic tabular test data"""
        data = {
            'id': range(1, num_rows + 1),
            'age': np.random.randint(18, 80, num_rows),
            'income': np.random.normal(50000, 15000, num_rows),
            'city': [self.fake.city() for _ in range(num_rows)],
            'category': np.random.choice(['A', 'B', 'C'], num_rows),
            'value': np.random.exponential(2, num_rows),
            'timestamp': pd.date_range('2023-01-01', periods=num_rows, freq='D')
        }
        
        if include_pii:
            data.update({
                'name': [self.fake.name() for _ in range(num_rows)],
                'email': [self.fake.email() for _ in range(num_rows)],
                'phone': [self.fake.phone_number() for _ in range(num_rows)],
                'ssn': [self.fake.ssn() for _ in range(num_rows)]
            })
        
        if include_sensitive:
            data.update({
                'medical_condition': np.random.choice(['diabetes', 'hypertension', 'none'], num_rows),
                'credit_score': np.random.randint(300, 850, num_rows),
                'salary_grade': np.random.choice(['junior', 'senior', 'executive'], num_rows)
            })
        
        return pd.DataFrame(data)
    
    def generate_time_series_data(self, num_points: int = 1000) -> pd.DataFrame:
        """Generate time series test data"""
        dates = pd.date_range('2023-01-01', periods=num_points, freq='H')
        values = np.cumsum(np.random.randn(num_points)) + 100
        noise = np.random.normal(0, 5, num_points)
        
        return pd.DataFrame({
            'timestamp': dates,
            'value': values + noise,
            'category': np.random.choice(['A', 'B', 'C'], num_points)
        })
    
    def generate_text_data(self, num_documents: int = 50) -> List[str]:
        """Generate text documents for testing"""
        documents = []
        for _ in range(num_documents):
            doc = self.fake.text(max_nb_chars=500)
            documents.append(doc)
        return documents
    
    def generate_image_metadata(self, num_images: int = 20) -> List[Dict]:
        """Generate image metadata for testing"""
        metadata = []
        for i in range(num_images):
            meta = {
                'filename': f'test_image_{i:03d}.jpg',
                'width': np.random.randint(256, 2048),
                'height': np.random.randint(256, 2048),
                'format': 'JPEG',
                'size_bytes': np.random.randint(50000, 5000000),
                'creation_date': self.fake.date_time().isoformat()
            }
            metadata.append(meta)
        return metadata


class SecurityTestHelper:
    """Helper utilities for security testing"""
    
    @staticmethod
    def generate_test_keys() -> Tuple[str, str]:
        """Generate test encryption keys"""
        private_key = secrets.token_hex(32)
        public_key = hashlib.sha256(private_key.encode()).hexdigest()
        return private_key, public_key
    
    @staticmethod
    def create_watermark_signature(data: str, key: str) -> str:
        """Create a test watermark signature"""
        combined = f"{data}:{key}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    @staticmethod
    def simulate_attack_vectors() -> Dict[str, List[str]]:
        """Simulate common attack vectors for testing"""
        return {
            'sql_injection': [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'/**/OR/**/'1'='1"
            ],
            'xss': [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>"
            ],
            'path_traversal': [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "....//....//....//etc/passwd"
            ]
        }


class PrivacyTestHelper:
    """Helper utilities for privacy testing"""
    
    @staticmethod
    def calculate_k_anonymity(df: pd.DataFrame, quasi_identifiers: List[str]) -> int:
        """Calculate k-anonymity for a dataset"""
        groups = df.groupby(quasi_identifiers).size()
        return groups.min()
    
    @staticmethod
    def detect_pii_patterns(text: str) -> Dict[str, List[str]]:
        """Detect common PII patterns in text"""
        import re
        
        patterns = {
            'ssn': re.findall(r'\b\d{3}-\d{2}-\d{4}\b', text),
            'email': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
            'phone': re.findall(r'\b\d{3}-\d{3}-\d{4}\b', text),
            'credit_card': re.findall(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', text)
        }
        
        return {k: v for k, v in patterns.items() if v}
    
    @staticmethod
    def assess_reidentification_risk(
        original_df: pd.DataFrame, 
        synthetic_df: pd.DataFrame,
        quasi_identifiers: List[str]
    ) -> float:
        """Assess re-identification risk between datasets"""
        # Simplified risk assessment for testing
        original_combinations = set(
            tuple(row) for row in original_df[quasi_identifiers].values
        )
        synthetic_combinations = set(
            tuple(row) for row in synthetic_df[quasi_identifiers].values
        )
        
        overlap = len(original_combinations.intersection(synthetic_combinations))
        total_synthetic = len(synthetic_combinations)
        
        return overlap / total_synthetic if total_synthetic > 0 else 0.0


class PerformanceTestHelper:
    """Helper utilities for performance testing"""
    
    @staticmethod
    @contextmanager
    def measure_time():
        """Context manager to measure execution time"""
        import time
        start_time = time.time()
        yield
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f} seconds")
    
    @staticmethod
    @contextmanager
    def measure_memory():
        """Context manager to measure memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        yield
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = end_memory - start_memory
        print(f"Memory usage: {memory_diff:.2f} MB")
    
    @staticmethod
    def create_load_test_data(size_mb: int) -> pd.DataFrame:
        """Create large dataset for load testing"""
        # Estimate rows needed for target size
        estimated_rows = size_mb * 1024 * 1024 // 100  # Rough estimate
        
        data_gen = TestDataGenerator()
        return data_gen.generate_tabular_data(
            num_rows=estimated_rows,
            include_pii=True,
            include_sensitive=True
        )


class MockingHelper:
    """Helper utilities for mocking external dependencies"""
    
    @staticmethod
    def mock_openai_api():
        """Mock OpenAI API responses"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(text="Generated synthetic text content")
        ]
        return patch('openai.Completion.create', return_value=mock_response)
    
    @staticmethod
    def mock_database_connection():
        """Mock database connections"""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        return mock_conn, mock_cursor
    
    @staticmethod
    def mock_s3_client():
        """Mock AWS S3 client"""
        mock_s3 = MagicMock()
        mock_s3.upload_file.return_value = None
        mock_s3.download_file.return_value = None
        return mock_s3
    
    @staticmethod
    def mock_neo4j_session():
        """Mock Neo4j database session"""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_session.run.return_value = mock_result
        return mock_session


class FileTestHelper:
    """Helper utilities for file-based testing"""
    
    @staticmethod
    @contextmanager
    def temp_directory():
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @staticmethod
    @contextmanager
    def temp_file(suffix: str = '.tmp', content: str = None):
        """Create temporary file for testing"""
        with tempfile.NamedTemporaryFile(
            mode='w+', 
            suffix=suffix, 
            delete=False
        ) as temp_file:
            if content:
                temp_file.write(content)
                temp_file.flush()
            yield temp_file.name
        
        # Cleanup
        try:
            os.unlink(temp_file.name)
        except OSError:
            pass
    
    @staticmethod
    def create_test_config(config_data: Dict) -> str:
        """Create temporary configuration file"""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.yaml', 
            delete=False
        ) as config_file:
            import yaml
            yaml.dump(config_data, config_file, default_flow_style=False)
            return config_file.name


class AssertionHelper:
    """Enhanced assertion utilities for testing"""
    
    @staticmethod
    def assert_dataframe_quality(df: pd.DataFrame, min_rows: int = 1):
        """Assert basic data quality requirements"""
        assert not df.empty, "DataFrame should not be empty"
        assert len(df) >= min_rows, f"DataFrame should have at least {min_rows} rows"
        assert not df.isnull().all().any(), "DataFrame should not have all-null columns"
    
    @staticmethod
    def assert_privacy_preserved(
        original_df: pd.DataFrame, 
        synthetic_df: pd.DataFrame,
        max_overlap: float = 0.1
    ):
        """Assert that privacy is preserved in synthetic data"""
        # Check for exact row matches
        original_rows = set(tuple(row) for row in original_df.values)
        synthetic_rows = set(tuple(row) for row in synthetic_df.values)
        
        overlap = len(original_rows.intersection(synthetic_rows))
        overlap_ratio = overlap / len(synthetic_rows) if len(synthetic_rows) > 0 else 0
        
        assert overlap_ratio <= max_overlap, (
            f"Too much overlap between original and synthetic data: {overlap_ratio:.2%}"
        )
    
    @staticmethod
    def assert_statistical_similarity(
        original_df: pd.DataFrame, 
        synthetic_df: pd.DataFrame,
        numeric_columns: List[str],
        tolerance: float = 0.2
    ):
        """Assert statistical similarity between datasets"""
        for col in numeric_columns:
            if col in original_df.columns and col in synthetic_df.columns:
                orig_mean = original_df[col].mean()
                synth_mean = synthetic_df[col].mean()
                
                if orig_mean != 0:
                    relative_diff = abs(orig_mean - synth_mean) / abs(orig_mean)
                    assert relative_diff <= tolerance, (
                        f"Mean difference too large for {col}: {relative_diff:.2%}"
                    )
    
    @staticmethod
    def assert_security_compliance(test_result: Dict[str, Any]):
        """Assert that security tests pass compliance requirements"""
        assert test_result.get('encryption_enabled', False), "Encryption must be enabled"
        assert test_result.get('access_control', False), "Access control must be implemented"
        assert not test_result.get('vulnerabilities', []), "No security vulnerabilities should exist"
        assert test_result.get('audit_logging', False), "Audit logging must be enabled"


# Pytest fixtures for common test scenarios
def pytest_fixtures():
    """Common pytest fixtures for testing"""
    
    import pytest
    
    @pytest.fixture
    def test_data_generator():
        """Provide test data generator"""
        return TestDataGenerator()
    
    @pytest.fixture
    def sample_tabular_data(test_data_generator):
        """Provide sample tabular data"""
        return test_data_generator.generate_tabular_data(num_rows=100)
    
    @pytest.fixture
    def sample_pii_data(test_data_generator):
        """Provide sample data with PII"""
        return test_data_generator.generate_tabular_data(
            num_rows=50, 
            include_pii=True
        )
    
    @pytest.fixture
    def temp_workspace():
        """Provide temporary workspace directory"""
        with FileTestHelper.temp_directory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_external_apis():
        """Mock all external API dependencies"""
        with MockingHelper.mock_openai_api():
            yield
    
    return [
        test_data_generator,
        sample_tabular_data,
        sample_pii_data,
        temp_workspace,
        mock_external_apis
    ]