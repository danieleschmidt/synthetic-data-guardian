"""
Generation 1: Basic Functionality Tests
Test core synthetic data generation capabilities
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_guardian import Guardian, GenerationPipeline, PipelineBuilder
import pytest
import pandas as pd
import numpy as np


async def test_basic_guardian_initialization():
    """Test basic Guardian initialization and setup."""
    print("🧪 Testing Guardian initialization...")
    
    # Test Guardian creation
    guardian = Guardian()
    assert guardian is not None
    assert guardian.config.name == "synthetic-data-guardian"
    assert guardian.initialized == False
    
    # Test initialization
    await guardian.initialize()
    assert guardian.initialized == True
    
    # Test available generators
    generators = list(guardian.generators.keys())
    print(f"Available generators: {generators}")
    
    # Test available validators  
    validators = list(guardian.validators.keys())
    print(f"Available validators: {validators}")
    
    # Test available watermarkers
    watermarkers = list(guardian.watermarkers.keys())
    print(f"Available watermarkers: {watermarkers}")
    
    await guardian.cleanup()
    print("✅ Guardian initialization test passed")


async def test_tabular_data_generation():
    """Test basic tabular synthetic data generation."""
    print("🧪 Testing tabular data generation...")
    
    async with Guardian() as guardian:
        # Create a simple pipeline config
        pipeline_config = {
            'id': 'test_tabular_pipeline',
            'name': 'Test Tabular Pipeline',
            'generator_type': 'tabular',
            'data_type': 'tabular',
            'schema': {
                'age': 'integer[18:80]',
                'income': 'float[20000:200000]',
                'name': 'string',
                'is_active': 'boolean'
            },
            'validation_config': {
                'validators': []  # No validators for basic test
            },
            'watermark_config': None  # No watermarking for basic test
        }
        
        # Generate synthetic data
        result = await guardian.generate(
            pipeline_config=pipeline_config,
            num_records=100,
            seed=42,
            validate=False,
            watermark=False
        )
        
        # Verify result structure
        assert result is not None
        assert result.data is not None
        assert len(result.data) == 100
        assert result.task_id is not None
        assert result.metadata is not None
        
        print(f"✅ Generated {len(result.data)} synthetic records")
        print(f"Generation time: {result.metadata.get('generation_time', 0):.2f}s")
        
    print("✅ Tabular data generation test passed")


async def test_pipeline_builder():
    """Test PipelineBuilder functionality."""
    print("🧪 Testing PipelineBuilder...")
    
    # Build pipeline programmatically
    pipeline = (PipelineBuilder()
        .with_generator('tabular')
        .with_schema({
            'user_id': 'integer',
            'age': 'integer[18:95]',
            'score': 'float[0:100]'
        })
        .build()
    )
    
    assert pipeline is not None
    assert pipeline.generator_type == 'tabular'
    assert 'user_id' in pipeline.schema
    assert 'age' in pipeline.schema
    assert 'score' in pipeline.schema
    
    print("✅ PipelineBuilder test passed")


async def test_validation_framework():
    """Test basic validation framework."""
    print("🧪 Testing validation framework...")
    
    async with Guardian() as guardian:
        # Create some dummy data
        test_data = pd.DataFrame({
            'values': np.random.normal(0, 1, 100),
            'categories': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Test validation (should handle missing reference data gracefully)
        validation_result = await guardian.validate(
            data=test_data,
            validators=['statistical', 'quality']
        )
        
        assert validation_result is not None
        print(f"✅ Validation completed with {len(validation_result.results)} validator results")
        
    print("✅ Validation framework test passed")


async def test_watermarking_framework():
    """Test basic watermarking framework."""
    print("🧪 Testing watermarking framework...")
    
    async with Guardian() as guardian:
        # Create some dummy data
        test_data = pd.DataFrame({
            'values': np.random.normal(0, 1, 50),
            'ids': range(50)
        })
        
        # Test watermarking
        watermark_result = await guardian.watermark(
            data=test_data,
            method='statistical',
            message='test_watermark_gen1'
        )
        
        assert watermark_result is not None
        print("✅ Watermarking applied successfully")
        
        # Test watermark verification
        verify_result = await guardian.verify_watermark(
            data=watermark_result.get('watermarked_data', test_data),
            method='statistical'
        )
        
        assert verify_result is not None
        print("✅ Watermark verification completed")
        
    print("✅ Watermarking framework test passed")


async def test_metrics_and_monitoring():
    """Test metrics collection and monitoring."""
    print("🧪 Testing metrics and monitoring...")
    
    async with Guardian() as guardian:
        # Get initial metrics
        initial_metrics = guardian.get_metrics()
        assert initial_metrics['total_generations'] == 0
        
        # Perform a generation
        pipeline_config = {
            'id': 'metrics_test_pipeline',
            'generator_type': 'tabular',
            'data_type': 'tabular',
            'schema': {'value': 'float[0:1]'},
            'validation_config': {'validators': []},
            'watermark_config': None
        }
        
        await guardian.generate(
            pipeline_config=pipeline_config,
            num_records=10,
            validate=False,
            watermark=False
        )
        
        # Check metrics updated
        updated_metrics = guardian.get_metrics()
        assert updated_metrics['total_generations'] == 1
        assert updated_metrics['successful_generations'] == 1
        assert updated_metrics['total_records_generated'] == 10
        
        print(f"✅ Metrics updated correctly: {updated_metrics}")
        
    print("✅ Metrics and monitoring test passed")


async def test_error_handling():
    """Test basic error handling."""
    print("🧪 Testing error handling...")
    
    async with Guardian() as guardian:
        # Test invalid pipeline config
        try:
            await guardian.generate(
                pipeline_config="nonexistent_pipeline",
                num_records=10
            )
            assert False, "Should have raised an error"
        except ValueError as e:
            print(f"✅ Correctly caught pipeline error: {e}")
        
        # Test invalid validator
        try:
            await guardian.validate(
                data=pd.DataFrame({'x': [1, 2, 3]}),
                validators=['nonexistent_validator']
            )
            # This should not raise an error, just skip the validator
            print("✅ Invalid validator handled gracefully")
        except Exception as e:
            print(f"⚠️  Unexpected error with invalid validator: {e}")
        
    print("✅ Error handling test passed")


async def test_concurrent_operations():
    """Test basic concurrent operations."""
    print("🧪 Testing concurrent operations...")
    
    async with Guardian() as guardian:
        # Create multiple pipeline configs
        pipelines = []
        for i in range(3):
            pipelines.append({
                'id': f'concurrent_pipeline_{i}',
                'generator_type': 'tabular',
                'data_type': 'tabular',
                'schema': {'value': f'integer[{i}:{i+10}]'},
                'validation_config': {'validators': []},
                'watermark_config': None
            })
        
        # Run concurrent generations
        tasks = [
            guardian.generate(
                pipeline_config=pipeline,
                num_records=20,
                validate=False,
                watermark=False
            )
            for pipeline in pipelines
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result is not None
            assert len(result.data) == 20
            print(f"✅ Concurrent generation {i+1} completed")
            
        # Check active tasks
        active_tasks = guardian.get_active_tasks()
        print(f"Active tasks after completion: {len(active_tasks)}")
        
    print("✅ Concurrent operations test passed")


async def run_all_tests():
    """Run all Generation 1 tests."""
    print("🚀 Starting Generation 1: Basic Functionality Tests")
    print("=" * 60)
    
    test_functions = [
        test_basic_guardian_initialization,
        test_tabular_data_generation,
        test_pipeline_builder,
        test_validation_framework,
        test_watermarking_framework,  
        test_metrics_and_monitoring,
        test_error_handling,
        test_concurrent_operations
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            await test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"🏁 Generation 1 Tests Summary:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 All Generation 1 tests passed! Ready for Generation 2.")
        return True
    else:
        print("⚠️  Some tests failed. Review issues before proceeding.")
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n🏆 Generation 1: MAKE IT WORK - COMPLETED SUCCESSFULLY!")
        print("Ready to proceed to Generation 2: MAKE IT ROBUST")
    else:
        print("\n🔧 Generation 1 needs fixes before proceeding to Generation 2")
        
    sys.exit(0 if success else 1)