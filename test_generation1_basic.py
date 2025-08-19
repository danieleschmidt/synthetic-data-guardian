#!/usr/bin/env python3
"""
Generation 1 Basic Functionality Test - MAKE IT WORK
Autonomous test to validate core synthetic data generation functionality.
"""

import asyncio
import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synthetic_guardian.core.guardian import Guardian, GuardianConfig
from synthetic_guardian.core.pipeline import GenerationPipeline, PipelineConfig
from synthetic_guardian.utils.logger import get_logger

logger = get_logger("Generation1Test")


async def test_basic_guardian_initialization():
    """Test basic Guardian initialization."""
    logger.info("Testing Guardian initialization...")
    
    try:
        # Create basic configuration
        config = GuardianConfig(
            name="test_guardian",
            log_level="DEBUG",
            max_records_per_generation=1000,
            enable_resource_monitoring=False  # Disable for simpler testing
        )
        
        # Initialize Guardian
        guardian = Guardian(config=config)
        await guardian.initialize()
        
        # Verify initialization
        assert guardian.initialized, "Guardian should be initialized"
        assert len(guardian.generators) > 0, "Should have at least one generator"
        
        logger.info("✅ Guardian initialization successful")
        
        # Cleanup
        await guardian.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ Guardian initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


async def test_basic_tabular_generation():
    """Test basic tabular data generation."""
    logger.info("Testing tabular data generation...")
    
    try:
        # Create Guardian with simple config
        config = GuardianConfig(
            name="tabular_test",
            log_level="DEBUG",
            max_records_per_generation=100,
            enable_validation=False,  # Disable for simpler test
            enable_watermarking=False,
            enable_lineage=False,
            enable_resource_monitoring=False
        )
        
        guardian = Guardian(config=config)
        await guardian.initialize()
        
        # Create simple pipeline configuration
        pipeline_config = {
            'id': 'test_tabular_pipeline',
            'name': 'Test Tabular Pipeline',
            'description': 'Simple tabular data generation test',
            'generator_type': 'tabular',
            'data_type': 'tabular',
            'generator_params': {
                'backend': 'simple'  # Use simple backend for reliability
            },
            'schema': {
                'user_id': {'type': 'integer', 'min': 1, 'max': 1000},
                'age': {'type': 'integer', 'min': 18, 'max': 80},
                'score': {'type': 'float', 'min': 0.0, 'max': 100.0},
                'category': {'type': 'categorical', 'values': ['A', 'B', 'C', 'D']},
                'active': {'type': 'boolean'}
            }
        }
        
        # Generate synthetic data
        result = await guardian.generate(
            pipeline_config=pipeline_config,
            num_records=50,
            seed=42,  # For reproducibility
            validate=False,
            watermark=False
        )
        
        # Validate result
        assert result is not None, "Generation result should not be None"
        assert result.data is not None, "Generated data should not be None"
        assert len(result.data) == 50, f"Expected 50 records, got {len(result.data)}"
        assert result.metadata is not None, "Metadata should be present"
        assert result.metadata['num_records'] == 50, "Metadata should reflect correct record count"
        
        # Validate data structure
        data = result.data
        expected_columns = {'user_id', 'age', 'score', 'category', 'active'}
        actual_columns = set(data.columns)
        assert expected_columns <= actual_columns, f"Missing columns: {expected_columns - actual_columns}"
        
        # Basic data validation
        assert data['user_id'].min() >= 1, "user_id should be >= 1"
        assert data['user_id'].max() <= 1000, "user_id should be <= 1000"
        assert data['age'].min() >= 18, "age should be >= 18"
        assert data['age'].max() <= 80, "age should be <= 80"
        assert data['score'].min() >= 0.0, "score should be >= 0.0"
        assert data['score'].max() <= 100.0, "score should be <= 100.0"
        assert set(data['category'].unique()).issubset({'A', 'B', 'C', 'D'}), "category should be in allowed values"
        assert set(data['active'].unique()).issubset({True, False}), "active should be boolean"
        
        logger.info("✅ Tabular data generation successful")
        logger.info(f"Generated data shape: {data.shape}")
        logger.info(f"Data types: {data.dtypes.to_dict()}")
        logger.info(f"Sample data:\n{data.head()}")
        
        # Cleanup
        await guardian.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ Tabular generation failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


async def test_pipeline_direct_usage():
    """Test direct pipeline usage."""
    logger.info("Testing direct pipeline usage...")
    
    try:
        # Create pipeline configuration
        config = PipelineConfig(
            name="direct_test_pipeline",
            description="Direct pipeline test",
            generator_type="tabular",
            data_type="tabular",
            generator_params={'backend': 'simple'},
            schema={
                'id': 'integer',
                'value': 'float',
                'name': 'text'
            }
        )
        
        # Create and initialize pipeline
        pipeline = GenerationPipeline(config=config)
        await pipeline.initialize()
        
        # Generate data
        result = await pipeline.generate(num_records=25, seed=123)
        
        # Validate
        assert result is not None, "Pipeline result should not be None"
        assert 'data' in result, "Result should contain data"
        assert 'metadata' in result, "Result should contain metadata"
        
        data = result['data']
        assert len(data) == 25, f"Expected 25 records, got {len(data)}"
        
        logger.info("✅ Direct pipeline usage successful")
        logger.info(f"Generated data shape: {data.shape}")
        
        # Cleanup
        await pipeline.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct pipeline usage failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


async def test_multiple_generations():
    """Test multiple sequential generations."""
    logger.info("Testing multiple sequential generations...")
    
    try:
        config = GuardianConfig(
            name="multi_test",
            enable_validation=False,
            enable_watermarking=False,
            enable_lineage=False,
            enable_resource_monitoring=False
        )
        
        guardian = Guardian(config=config)
        await guardian.initialize()
        
        pipeline_config = {
            'id': 'multi_test_pipeline',
            'name': 'Multi Test Pipeline',
            'generator_type': 'tabular',
            'data_type': 'tabular',
            'generator_params': {'backend': 'simple'},
            'schema': {
                'x': 'float',
                'y': 'float',
                'label': {'type': 'categorical', 'values': ['pos', 'neg']}
            }
        }
        
        # Run multiple generations
        for i in range(3):
            result = await guardian.generate(
                pipeline_config=pipeline_config,
                num_records=10,
                seed=42 + i,  # Different seeds
                validate=False,
                watermark=False
            )
            
            assert result is not None, f"Generation {i+1} failed"
            assert len(result.data) == 10, f"Generation {i+1} wrong record count"
            logger.info(f"✅ Generation {i+1} successful - {len(result.data)} records")
        
        # Check metrics
        metrics = guardian.get_metrics()
        assert metrics['total_generations'] >= 3, "Should have recorded multiple generations"
        assert metrics['successful_generations'] >= 3, "Should have successful generations"
        
        logger.info("✅ Multiple generations successful")
        logger.info(f"Final metrics: {metrics}")
        
        await guardian.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ Multiple generations failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


async def run_generation1_tests():
    """Run all Generation 1 basic functionality tests."""
    logger.info("🚀 Starting Generation 1: MAKE IT WORK Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Guardian Initialization", test_basic_guardian_initialization),
        ("Basic Tabular Generation", test_basic_tabular_generation),
        ("Direct Pipeline Usage", test_pipeline_direct_usage),
        ("Multiple Sequential Generations", test_multiple_generations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running test: {test_name}")
        logger.info("-" * 40)
        
        try:
            success = await test_func()
            if success:
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
            logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🏆 GENERATION 1 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 Generation 1: MAKE IT WORK - ALL TESTS PASSED!")
        logger.info("✨ Core synthetic data generation functionality is working!")
        return True
    else:
        logger.error(f"💥 Generation 1: MAKE IT WORK - {total - passed} tests failed")
        return False


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    
    # Run tests
    try:
        success = asyncio.run(run_generation1_tests())
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total execution time: {elapsed:.2f} seconds")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n💥 Test suite failed with critical error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)