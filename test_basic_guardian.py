#!/usr/bin/env python3
"""
Test basic Guardian functionality
"""

import asyncio
import sys
sys.path.insert(0, '/root/repo/src')

async def test_guardian_creation():
    """Test Guardian creation and initialization"""
    print("🧪 Testing Guardian creation...")
    
    try:
        from synthetic_guardian.core.guardian import Guardian
        
        # Create guardian
        guardian = Guardian()
        
        # Initialize
        await guardian.initialize()
        
        # Test info
        info = guardian.get_metrics()
        print(f"✅ Guardian created with metrics: {info}")
        
        # Cleanup
        await guardian.cleanup()
        
        return True
    except Exception as e:
        print(f"❌ Guardian test failed: {e}")
        return False

async def test_pipeline_creation():
    """Test Pipeline creation"""
    print("🧪 Testing Pipeline creation...")
    
    try:
        from synthetic_guardian.core.pipeline import GenerationPipeline, PipelineBuilder
        
        # Create pipeline using builder
        pipeline = (PipelineBuilder()
                   .with_name("test_pipeline")
                   .with_generator("tabular")
                   .with_schema({
                       "id": "integer",
                       "name": "string",
                       "value": "float"
                   })
                   .build())
        
        # Initialize pipeline
        await pipeline.initialize()
        
        # Test info
        info = pipeline.get_info()
        print(f"✅ Pipeline created: {info['name']}")
        
        # Cleanup
        await pipeline.cleanup()
        
        return True
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

async def test_simple_generation():
    """Test simple data generation"""
    print("🧪 Testing simple data generation...")
    
    try:
        from synthetic_guardian.core.guardian import Guardian
        
        # Create pipeline config
        pipeline_config = {
            "name": "simple_test",
            "generator_type": "tabular",
            "data_type": "tabular",
            "schema": {
                "id": "integer",
                "value": "float",
                "category": "categorical"
            }
        }
        
        # Create and use Guardian
        async with Guardian() as guardian:
            result = await guardian.generate(
                pipeline_config=pipeline_config,
                num_records=10,
                seed=42,
                validate=False,  # Skip validation for now
                watermark=False  # Skip watermark for now
            )
            
            print(f"✅ Generated {len(result.data)} records")
            print(f"   Quality score: {result.quality_score}")
            print(f"   Generation time: {result.metadata.get('generation_time', 0):.3f}s")
            print(f"   Sample data: {result.data[:2] if hasattr(result.data, '__getitem__') else 'N/A'}")
        
        return True
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    print("🚀 Testing Basic Guardian Functionality")
    print("=" * 60)
    
    tests = [
        test_guardian_creation,
        test_pipeline_creation,
        test_simple_generation
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 SUCCESS: All {total} tests passed!")
        return 0
    else:
        print(f"💥 PARTIAL: {passed}/{total} tests passed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))