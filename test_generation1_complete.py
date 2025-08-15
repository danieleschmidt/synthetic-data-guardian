#!/usr/bin/env python3
"""
Generation 1 Complete Test - Basic Synthetic Data Generation Functionality
"""

import asyncio
import json
from pathlib import Path
from src.synthetic_guardian import Guardian, GenerationPipeline


async def test_basic_text_generation():
    """Test basic text generation functionality."""
    print("🧪 Testing Basic Text Generation...")
    
    guardian = Guardian()
    await guardian.initialize()
    
    # Create a simple text generation pipeline
    pipeline_config = {
        'id': 'test_text_pipeline',
        'name': 'Basic Text Generator',
        'description': 'Generate synthetic text data',
        'generator_type': 'text',
        'data_type': 'text',
        'generator_params': {
            'model_type': 'simple',
            'max_length': 100
        },
        'schema': {
            'content': 'text',
            'category': 'categorical[email,review,comment]'
        }
    }
    
    try:
        # Generate synthetic text data
        result = await guardian.generate(
            pipeline_config=pipeline_config,
            num_records=10,
            seed=42
        )
        
        print(f"✅ Generated {result.metadata['num_records']} text records")
        print(f"📊 Generation time: {result.metadata['generation_time']:.2f}s")
        print(f"🎯 Task ID: {result.task_id}")
        print(f"📝 Sample data: {result.data[:3] if result.data else 'No data'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False
    finally:
        await guardian.cleanup()


async def test_pipeline_management():
    """Test pipeline creation and management."""
    print("\n🔧 Testing Pipeline Management...")
    
    guardian = Guardian()
    await guardian.initialize()
    
    try:
        # Test pipeline creation
        pipeline = GenerationPipeline({
            'id': 'test_pipeline_mgmt',
            'name': 'Management Test Pipeline',
            'generator_type': 'text',
            'data_type': 'text'
        })
        
        # Add to guardian
        guardian.pipelines[pipeline.id] = pipeline
        
        # Test pipeline listing
        pipelines = guardian.get_pipelines()
        print(f"✅ Pipeline management working. Found {len(pipelines)} pipelines")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline management failed: {e}")
        return False
    finally:
        await guardian.cleanup()


async def test_validation_framework():
    """Test basic validation framework."""
    print("\n🔍 Testing Validation Framework...")
    
    guardian = Guardian()
    await guardian.initialize()
    
    try:
        # Test validation with sample data
        sample_data = [
            {"text": "This is sample text", "category": "email"},
            {"text": "Another sample", "category": "review"}
        ]
        
        # Run validation (even if no validators are loaded)
        validation_report = await guardian.validate(
            data=sample_data,
            validators=[]  # Empty validators list for basic test
        )
        
        print(f"✅ Validation framework working. Report: {validation_report.task_id}")
        return True
        
    except Exception as e:
        print(f"❌ Validation framework failed: {e}")
        return False
    finally:
        await guardian.cleanup()


async def test_configuration_system():
    """Test configuration and setup."""
    print("\n⚙️ Testing Configuration System...")
    
    try:
        # Test various Guardian configurations
        configs = [
            {'name': 'test-guardian-1', 'log_level': 'DEBUG'},
            {'name': 'test-guardian-2', 'max_concurrent_generations': 5},
            {'name': 'test-guardian-3', 'enable_lineage': False}
        ]
        
        for config in configs:
            guardian = Guardian(config=config)
            assert guardian.config.name == config['name']
            await guardian.cleanup()
        
        print("✅ Configuration system working")
        return True
        
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        return False


async def test_error_handling():
    """Test basic error handling."""
    print("\n🛡️ Testing Error Handling...")
    
    guardian = Guardian()
    await guardian.initialize()
    
    try:
        # Test with invalid configuration
        try:
            await guardian.generate(
                pipeline_config="invalid_pipeline_id",
                num_records=10
            )
            print("❌ Should have failed with invalid pipeline")
            return False
        except ValueError as e:
            print(f"✅ Proper error handling: {str(e)[:50]}...")
        
        # Test with invalid parameters
        try:
            await guardian.generate(
                pipeline_config={'generator_type': 'text'},
                num_records=-5  # Invalid number
            )
            print("❌ Should have failed with invalid num_records")
            return False
        except ValueError as e:
            print(f"✅ Proper parameter validation: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False
    finally:
        await guardian.cleanup()


async def main():
    """Run all Generation 1 tests."""
    print("🚀 GENERATION 1 FUNCTIONALITY TESTS")
    print("=" * 50)
    
    tests = [
        test_configuration_system,
        test_pipeline_management,
        test_validation_framework,
        test_error_handling,
        test_basic_text_generation,
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 GENERATION 1 TEST RESULTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 GENERATION 1 COMPLETE - BASIC FUNCTIONALITY WORKING!")
        return True
    else:
        print("⚠️ Some tests failed - needs additional work")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)