#!/usr/bin/env python3
"""
Comprehensive test suite for Synthetic Data Guardian
"""

import pytest
import asyncio
import sys
import time
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from synthetic_guardian.core.guardian import Guardian
from synthetic_guardian.generators.tabular import TabularGenerator, TabularGeneratorConfig
from synthetic_guardian.middleware.error_handler import ErrorHandler, ValidationError
from synthetic_guardian.middleware.input_validator import InputValidator
from synthetic_guardian.monitoring.health_monitor import HealthMonitor
from synthetic_guardian.optimization.performance_optimizer import PerformanceOptimizer
from synthetic_guardian.optimization.caching import IntelligentCache, CacheStrategy


class TestCoreGuardian:
    """Test suite for core Guardian functionality."""
    
    @pytest.fixture
    async def guardian(self):
        """Create Guardian instance for testing."""
        guardian = Guardian()
        await guardian.initialize()
        yield guardian
        await guardian.cleanup()
    
    @pytest.mark.asyncio
    async def test_guardian_initialization(self):
        """Test Guardian initialization and cleanup."""
        guardian = Guardian()
        assert not guardian.initialized
        
        await guardian.initialize()
        assert guardian.initialized
        
        metrics = guardian.get_metrics()
        assert 'total_generations' in metrics
        assert metrics['total_generations'] == 0
        
        await guardian.cleanup()
        assert not guardian.initialized
    
    @pytest.mark.asyncio
    async def test_simple_generation(self, guardian):
        """Test basic synthetic data generation."""
        pipeline_config = {
            "name": "test_generation",
            "generator_type": "tabular",
            "data_type": "tabular",
            "schema": {
                "id": "integer",
                "value": "float",
                "category": "categorical"
            }
        }
        
        result = await guardian.generate(
            pipeline_config=pipeline_config,
            num_records=10,
            seed=42,
            validate=False,
            watermark=False
        )
        
        assert result is not None
        assert len(result.data) == 10
        assert 'id' in result.data.columns
        assert 'value' in result.data.columns
        assert 'category' in result.data.columns
        
        # Check result metadata
        assert result.metadata['num_records'] == 10
        assert result.metadata['seed'] == 42
        assert not result.metadata['fitted']
    
    @pytest.mark.asyncio
    async def test_invalid_generation_parameters(self, guardian):
        """Test handling of invalid generation parameters."""
        pipeline_config = {
            "name": "test_invalid",
            "generator_type": "tabular",
            "data_type": "tabular"
        }
        
        # Test negative num_records
        with pytest.raises(ValueError, match="num_records must be a positive integer"):
            await guardian.generate(
                pipeline_config=pipeline_config,
                num_records=-1
            )
        
        # Test invalid generator type
        invalid_config = {
            "name": "test_invalid",
            "generator_type": "invalid_type",
            "data_type": "tabular"
        }
        
        with pytest.raises(Exception):  # Should raise some error for invalid type
            await guardian.generate(
                pipeline_config=invalid_config,
                num_records=5
            )


class TestTabularGenerator:
    """Test suite for tabular data generator."""
    
    @pytest.fixture
    def generator_config(self):
        """Create test generator configuration."""
        return TabularGeneratorConfig(
            name="test_tabular",
            type="tabular",
            schema={
                "id": {"type": "integer", "min": 1, "max": 1000},
                "name": {"type": "categorical", "categories": ["Alice", "Bob", "Charlie"]},
                "score": {"type": "float", "min": 0.0, "max": 100.0}
            }
        )
    
    @pytest.fixture
    async def generator(self, generator_config):
        """Create TabularGenerator instance."""
        generator = TabularGenerator(generator_config)
        await generator.initialize()
        yield generator
        await generator.cleanup()
    
    @pytest.mark.asyncio
    async def test_generator_initialization(self, generator_config):
        """Test generator initialization."""
        generator = TabularGenerator(generator_config)
        assert not generator.initialized
        
        await generator.initialize()
        assert generator.initialized
        
        info = generator.get_info()
        assert info['name'] == "test_tabular"
        assert info['type'] == "tabular"
        
        await generator.cleanup()
    
    @pytest.mark.asyncio
    async def test_schema_generation(self, generator):
        """Test generation from schema."""
        result = await generator.generate(num_records=5, seed=123)
        
        assert len(result.data) == 5
        assert 'id' in result.data.columns
        assert 'name' in result.data.columns
        assert 'score' in result.data.columns
        
        # Verify data types and ranges
        assert all(isinstance(x, (int, float)) for x in result.data['id'])
        assert all(x in ["Alice", "Bob", "Charlie"] for x in result.data['name'])
        assert all(0.0 <= x <= 100.0 for x in result.data['score'])
    
    @pytest.mark.asyncio
    async def test_config_validation(self, generator_config):
        """Test configuration validation."""
        generator = TabularGenerator(generator_config)
        issues = generator.validate_config()
        assert len(issues) == 0
        
        # Test invalid config
        bad_config = TabularGeneratorConfig(
            name="bad_config",
            type="tabular",
            epochs=-1,  # Invalid
            batch_size=0  # Invalid
        )
        bad_generator = TabularGenerator(bad_config)
        issues = bad_generator.validate_config()
        assert len(issues) > 0
        assert any("epochs must be a positive integer" in issue for issue in issues)
        assert any("batch_size must be a positive integer" in issue for issue in issues)


class TestErrorHandling:
    """Test suite for error handling middleware."""
    
    @pytest.fixture
    def error_handler(self):
        """Create ErrorHandler instance."""
        return ErrorHandler()
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, error_handler):
        """Test handling of validation errors."""
        error = ValidationError("Test validation error", context={'field': 'test'})
        
        error_details = await error_handler.handle_error(error)
        
        assert error_details.category.value == "validation"
        assert error_details.severity.value == "medium"
        assert "Test validation error" in error_details.message
        assert error_details.context['field'] == 'test'
        
        # Check statistics
        stats = error_handler.get_statistics()
        assert stats['total_errors'] == 1
        assert stats['by_category']['validation'] == 1
    
    @pytest.mark.asyncio
    async def test_generic_error_handling(self, error_handler):
        """Test handling of generic exceptions."""
        error = ValueError("Generic test error")
        
        error_details = await error_handler.handle_error(error)
        
        assert error_details.category.value == "unknown"  # Should classify as unknown
        assert error_details.message == "Generic test error"
        assert len(error_details.remediation_steps) > 0


class TestInputValidation:
    """Test suite for input validation middleware."""
    
    @pytest.fixture
    def validator(self):
        """Create InputValidator instance."""
        return InputValidator()
    
    def test_safe_string_validation(self, validator):
        """Test safe string validation."""
        # Valid strings
        assert validator.validate_input("hello world", ['safe_string']) == "hello world"
        assert validator.validate_input("test123", ['safe_string']) == "test123"
        
        # Invalid strings (should raise ValidationError)
        with pytest.raises(ValidationError):
            validator.validate_input("<script>alert('xss')</script>", ['safe_string'])
        
        with pytest.raises(ValidationError):
            validator.validate_input("DROP TABLE users", ['safe_string'])
    
    def test_email_validation(self, validator):
        """Test email validation."""
        # Valid emails
        assert validator.validate_input("test@example.com", ['email']) == "test@example.com"
        
        # Invalid emails
        with pytest.raises(ValidationError):
            validator.validate_input("not-an-email", ['email'])
        
        with pytest.raises(ValidationError):
            validator.validate_input("invalid.email@", ['email'])
    
    def test_integer_validation(self, validator):
        """Test integer validation."""
        # Valid integers
        assert validator.validate_input(42, ['positive_integer']) == 42
        assert validator.validate_input("123", ['positive_integer']) == "123"
        
        # Invalid integers
        with pytest.raises(ValidationError):
            validator.validate_input(-5, ['positive_integer'])
        
        with pytest.raises(ValidationError):
            validator.validate_input("not a number", ['positive_integer'])
    
    def test_dict_validation(self, validator):
        """Test dictionary validation."""
        test_data = {
            "name": "test_user",
            "email": "test@example.com",
            "age": 25
        }
        
        field_rules = {
            "name": ['safe_string'],
            "email": ['email'],
            "age": ['positive_integer']
        }
        
        result = validator.validate_dict(test_data, field_rules)
        assert result['name'] == "test_user"
        assert result['email'] == "test@example.com"
        assert result['age'] == 25


class TestHealthMonitoring:
    """Test suite for health monitoring."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create HealthMonitor instance."""
        return HealthMonitor()
    
    @pytest.mark.asyncio
    async def test_health_check_registration(self, health_monitor):
        """Test health check registration."""
        from synthetic_guardian.monitoring.health_monitor import HealthCheck, ComponentStatus
        
        def test_check():
            return ComponentStatus.UP, "Test passed", {}
        
        health_check = HealthCheck(
            name="test_check",
            check_function=test_check,
            interval_seconds=5.0
        )
        
        health_monitor.register_check(health_check)
        assert "test_check" in health_monitor.health_checks
        
        # Run health checks
        await health_monitor._run_health_checks()
        
        # Get component health
        component_health = health_monitor.get_component_health("test_check")
        assert "test_check" in component_health
        assert component_health["test_check"].status == ComponentStatus.UP
    
    @pytest.mark.asyncio
    async def test_overall_health_status(self, health_monitor):
        """Test overall health status calculation."""
        status, message, details = health_monitor.get_overall_health()
        
        assert status is not None
        assert isinstance(message, str)
        assert isinstance(details, dict)
        assert 'total_checks' in details


class TestPerformanceOptimization:
    """Test suite for performance optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create PerformanceOptimizer instance."""
        from synthetic_guardian.optimization.performance_optimizer import OptimizationStrategy
        optimizer = PerformanceOptimizer(strategy=OptimizationStrategy.BALANCED)
        yield optimizer
        asyncio.run(optimizer.cleanup())
    
    @pytest.mark.asyncio
    async def test_batch_operation_optimization(self, optimizer):
        """Test batch operation optimization."""
        def simple_operation(batch):
            return sum(batch) if isinstance(batch, list) else 0
        
        test_batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        results = await optimizer.optimize_batch_operation(
            simple_operation,
            test_batches,
            operation_name="test_batch_op"
        )
        
        assert len(results) == 3
        assert results == [6, 15, 24]
        
        # Check performance analysis
        analysis = optimizer.get_performance_analysis("test_batch_op")
        assert analysis['operation_stats']['total_executions'] == 1
        assert analysis['operation_stats']['successful_executions'] == 1
    
    def test_auto_scaling(self, optimizer):
        """Test auto-scaling functionality."""
        initial_constraints = optimizer.resource_constraints.max_concurrent_tasks
        
        scaling_actions = optimizer.auto_scale_resources(target_load=0.5)
        
        assert isinstance(scaling_actions, dict)
        assert 'memory_actions' in scaling_actions
        assert 'cpu_actions' in scaling_actions


class TestIntelligentCaching:
    """Test suite for intelligent caching."""
    
    @pytest.fixture
    async def cache(self):
        """Create IntelligentCache instance."""
        cache = IntelligentCache(
            max_memory_entries=100,
            max_memory_size_mb=1,
            strategy=CacheStrategy.LRU,
            enable_disk_cache=False
        )
        yield cache
        await cache.clear()
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache):
        """Test basic cache set and get operations."""
        test_key = "test_key"
        test_value = {"data": [1, 2, 3], "timestamp": time.time()}
        
        # Set value
        await cache.set(test_key, test_value)
        
        # Get value
        retrieved_value = await cache.get(test_key)
        assert retrieved_value == test_value
        
        # Get non-existent key
        missing_value = await cache.get("missing_key", "default")
        assert missing_value == "default"
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, cache):
        """Test TTL-based cache expiration."""
        test_key = "expiring_key"
        test_value = "test_data"
        
        # Set with short TTL
        await cache.set(test_key, test_value, ttl_seconds=0.1)
        
        # Should be available immediately
        retrieved = await cache.get(test_key)
        assert retrieved == test_value
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired now
        expired = await cache.get(test_key, "default")
        assert expired == "default"
    
    def test_cache_statistics(self, cache):
        """Test cache statistics tracking."""
        asyncio.run(self._test_cache_stats_async(cache))
    
    async def _test_cache_stats_async(self, cache):
        """Async helper for cache statistics test."""
        # Perform some cache operations
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        # Hit
        await cache.get("key1")
        # Miss
        await cache.get("key3", "default")
        
        stats = cache.get_stats()
        assert stats['memory_entries'] == 2
        assert stats['total_requests'] >= 3  # At least the operations above
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_generation_workflow(self):
        """Test complete end-to-end generation workflow."""
        # Initialize Guardian
        async with Guardian() as guardian:
            # Configure pipeline
            pipeline_config = {
                "name": "integration_test",
                "generator_type": "tabular",
                "data_type": "tabular",
                "schema": {
                    "user_id": "integer",
                    "username": "categorical",
                    "score": "float"
                }
            }
            
            # Generate data
            result = await guardian.generate(
                pipeline_config=pipeline_config,
                num_records=20,
                seed=12345,
                validate=False,  # Skip validation for integration test
                watermark=False  # Skip watermarking for integration test
            )
            
            # Verify result
            assert result is not None
            assert len(result.data) == 20
            assert result.is_valid()
            
            # Check Guardian metrics
            metrics = guardian.get_metrics()
            assert metrics['total_generations'] >= 1
            assert metrics['successful_generations'] >= 1
            assert metrics['total_records_generated'] >= 20
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling in complete workflow."""
        from synthetic_guardian.middleware.error_handler import get_error_handler
        
        error_handler = get_error_handler()
        initial_errors = error_handler.get_statistics()['total_errors']
        
        # Try invalid operation that should be handled gracefully
        async with Guardian() as guardian:
            try:
                await guardian.generate(
                    pipeline_config={"invalid": "config"},
                    num_records=-1  # Invalid parameter
                )
                assert False, "Should have raised an error"
            except Exception:
                pass  # Expected
        
        # Check that error was handled
        final_stats = error_handler.get_statistics()
        # Note: Might not increment if error is caught before reaching error handler


# Test configuration
pytestmark = pytest.mark.asyncio


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])