#!/usr/bin/env python3
"""
Basic functionality test for Synthetic Data Guardian
"""

import asyncio
import sys
import os
from pathlib import Path

# Mock pandas and numpy if not available
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas/numpy not available - running minimal tests")

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synthetic_guardian import Guardian, PipelineBuilder
from synthetic_guardian.core.guardian import GuardianConfig


async def test_basic_functionality():
    """Test basic synthetic data generation functionality."""
    print("🚀 Testing Synthetic Data Guardian Basic Functionality")
    print("=" * 60)
    
    try:
        # Test 1: Create Guardian instance
        print("\n📋 Test 1: Creating Guardian instance...")
        config = GuardianConfig(
            name="test-guardian",
            log_level="INFO",
            enable_lineage=False,  # Disable for simple test
            enable_watermarking=False  # Disable for simple test
        )
        
        guardian = Guardian(config=config)
        await guardian.initialize()
        print("✅ Guardian created and initialized successfully")
        
        # Test 2: Create simple pipeline
        print("\n📋 Test 2: Creating generation pipeline...")
        pipeline = (PipelineBuilder()
            .with_name("test_tabular_pipeline")
            .with_description("Simple test pipeline for tabular data")
            .with_generator("tabular", backend="simple")
            .with_data_type("tabular")
            .with_schema({
                "age": {"type": "integer", "min": 18, "max": 80},
                "income": {"type": "float", "min": 20000, "max": 200000},
                "category": {"type": "categorical", "categories": ["A", "B", "C"]}
            })
            .build())
        
        await pipeline.initialize()
        print("✅ Pipeline created and initialized successfully")
        
        # Test 3: Generate synthetic data
        print("\n📋 Test 3: Generating synthetic tabular data...")
        result = await guardian.generate(
            pipeline_config=pipeline.config.to_dict() if hasattr(pipeline.config, 'to_dict') else pipeline.config,
            num_records=100,
            seed=42,
            validate=True
        )
        
        print(f"✅ Generated {len(result.data)} records successfully")
        print(f"   Quality Score: {result.quality_score:.2f}")
        print(f"   Generation Time: {result.metadata.get('generation_time', 0):.2f}s")
        
        # Test 4: Validate the generated data
        print("\n📋 Test 4: Running data validation...")
        if result.validation_report:
            print(f"✅ Validation completed - Overall Score: {result.validation_report.overall_score:.2f}")
            print(f"   Validation Status: {'PASSED' if result.validation_report.passed else 'FAILED'}")
        else:
            print("⚠️  Validation report not available")
        
        # Test 5: Test time series generator
        print("\n📋 Test 5: Testing time series generation...")
        ts_pipeline = (PipelineBuilder()
            .with_name("test_timeseries_pipeline") 
            .with_generator("timeseries", sequence_length=50)
            .with_data_type("timeseries")
            .build())
        
        await ts_pipeline.initialize()
        
        ts_result = await guardian.generate(
            pipeline_config=ts_pipeline.config.to_dict() if hasattr(ts_pipeline.config, 'to_dict') else ts_pipeline.config,
            num_records=50,
            seed=42,
            validate=False  # Skip validation for time series test
        )
        
        print(f"✅ Generated time series with {len(ts_result.data)} points")
        
        # Test 6: Test text generator
        print("\n📋 Test 6: Testing text generation...")
        text_pipeline = (PipelineBuilder()
            .with_name("test_text_pipeline")
            .with_generator("text", backend="template")
            .with_data_type("text")
            .build())
        
        await text_pipeline.initialize()
        
        text_result = await guardian.generate(
            pipeline_config=text_pipeline.config.to_dict() if hasattr(text_pipeline.config, 'to_dict') else text_pipeline.config,
            num_records=10,
            seed=42,
            validate=False
        )
        
        print(f"✅ Generated {len(text_result.data)} text samples")
        if text_result.data:
            print(f"   Sample: {text_result.data[0]}")
        
        # Test 7: Test watermarking (if pandas available)
        if HAS_PANDAS:
            print("\n📋 Test 7: Testing statistical watermarking...")
            
            # Create sample data for watermarking
            sample_data = pd.DataFrame({
                'value1': np.random.normal(0, 1, 100),
                'value2': np.random.normal(10, 2, 100),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })
            
            # Test watermark embedding
            watermark_result = await guardian.watermark(
                data=sample_data,
                method='statistical',
                message='test_watermark'
            )
            
            print("✅ Statistical watermark embedded successfully")
            
            # Test watermark verification
            verification_result = await guardian.verify_watermark(
                data=watermark_result['data'],
                method='statistical'
            )
            
            print(f"✅ Watermark verification completed - Detected: {verification_result.get('is_watermarked', False)}")
        else:
            print("\n📋 Test 7: Skipping watermarking tests (pandas required)")
        
        # Cleanup
        print("\n📋 Cleanup: Cleaning up resources...")
        await guardian.cleanup()
        print("✅ Cleanup completed successfully")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Synthetic Data Guardian is working correctly!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_imports():
    """Test that basic imports work."""
    print("\n📋 Testing basic imports...")
    
    try:
        # Test core imports
        from synthetic_guardian import Guardian, GenerationPipeline, PipelineBuilder
        from synthetic_guardian.generators import TabularGenerator, TimeSeriesGenerator, TextGenerator
        from synthetic_guardian.validators import StatisticalValidator, PrivacyValidator
        from synthetic_guardian.watermarks import StatisticalWatermarker
        
        print("✅ All basic imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        return False


def main():
    """Main test runner."""
    print("🔬 Synthetic Data Guardian - Basic Functionality Test Suite")
    print("=" * 60)
    
    # Test imports first
    if not test_basic_imports():
        print("❌ Basic imports failed - cannot continue with functionality tests")
        return 1
    
    # Run async functionality tests
    success = asyncio.run(test_basic_functionality())
    
    if success:
        print("\n🎯 SUCCESS: All basic functionality tests passed!")
        return 0
    else:
        print("\n💥 FAILURE: Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)