#!/usr/bin/env python3
"""
Test performance optimization and caching features
"""

import asyncio
import sys
import time
import random
sys.path.insert(0, '/root/repo/src')

async def test_performance_optimizer():
    """Test performance optimization system"""
    print("🧪 Testing performance optimizer...")
    
    try:
        from synthetic_guardian.optimization.performance_optimizer import (
            PerformanceOptimizer, OptimizationStrategy
        )
        
        # Test different optimization strategies
        strategies = [
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.MEMORY_OPTIMIZED,
            OptimizationStrategy.CPU_OPTIMIZED
        ]
        
        for strategy in strategies:
            optimizer = PerformanceOptimizer(strategy=strategy)
            
            # Test batch operation optimization
            def sample_batch_operation(batch, **kwargs):
                """Sample CPU-intensive operation"""
                # Simulate some work
                time.sleep(0.01)
                return sum(range(len(batch))) if hasattr(batch, '__len__') else 42
            
            # Create test batches
            test_batches = [
                list(range(i * 10, (i + 1) * 10))
                for i in range(5)
            ]
            
            # Run optimized batch operation
            results = await optimizer.optimize_batch_operation(
                sample_batch_operation,
                test_batches,
                operation_name=f"test_operation_{strategy.value}"
            )
            
            print(f"   ✅ {strategy.value}: Processed {len(results)} batches")
            
            # Get performance analysis
            analysis = optimizer.get_performance_analysis(f"test_operation_{strategy.value}")
            if analysis['operation_stats']:
                stats = analysis['operation_stats']
                print(f"      📊 Avg duration: {stats['avg_duration']:.3f}s, Success rate: {stats['successful_executions']}/{stats['total_executions']}")
            
            # Cleanup
            await optimizer.cleanup()
        
        return True
    except Exception as e:
        print(f"   ❌ Performance optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_intelligent_cache():
    """Test intelligent caching system"""
    print("🧪 Testing intelligent cache...")
    
    try:
        from synthetic_guardian.optimization.caching import (
            IntelligentCache, CacheStrategy, cached
        )
        
        # Test different cache strategies
        strategies = [
            CacheStrategy.LRU,
            CacheStrategy.LFU,
            CacheStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            cache = IntelligentCache(
                max_memory_entries=100,
                max_memory_size_mb=10,
                strategy=strategy,
                enable_disk_cache=False  # Disable for testing
            )
            
            # Test basic cache operations
            test_key = f"test_key_{strategy.value}"
            test_value = {"data": list(range(100)), "timestamp": time.time()}
            
            # Set value
            await cache.set(test_key, test_value, ttl_seconds=60)
            
            # Get value (should hit cache)
            retrieved_value = await cache.get(test_key)
            assert retrieved_value == test_value
            print(f"   ✅ {strategy.value}: Basic cache operations working")
            
            # Test cache with TTL expiration
            await cache.set("expiring_key", "test_data", ttl_seconds=0.1)
            await asyncio.sleep(0.2)
            expired_value = await cache.get("expiring_key", "DEFAULT")
            assert expired_value == "DEFAULT"
            print(f"   ✅ {strategy.value}: TTL expiration working")
            
            # Test cache statistics
            stats = cache.get_stats()
            print(f"      📊 Hit rate: {stats['hit_rate_percent']:.1f}%, Entries: {stats['memory_entries']}")
            
            # Clear cache
            await cache.clear()
        
        # Test caching decorator
        @cached(ttl_seconds=1.0)
        async def expensive_operation(x: int) -> int:
            """Simulate expensive operation"""
            await asyncio.sleep(0.01)
            return x * x
        
        # First call should be slow
        start_time = time.time()
        result1 = await expensive_operation(5)
        first_call_time = time.time() - start_time
        
        # Second call should be fast (cached)
        start_time = time.time()
        result2 = await expensive_operation(5)
        second_call_time = time.time() - start_time
        
        assert result1 == result2 == 25
        assert second_call_time < first_call_time * 0.5  # Should be much faster
        print(f"   ✅ Caching decorator: {first_call_time:.3f}s → {second_call_time:.3f}s (cached)")
        
        return True
    except Exception as e:
        print(f"   ❌ Intelligent cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_optimized_guardian():
    """Test Guardian with performance optimization"""
    print("🧪 Testing optimized Guardian...")
    
    try:
        from synthetic_guardian.core.guardian import Guardian
        from synthetic_guardian.optimization.performance_optimizer import get_performance_optimizer
        from synthetic_guardian.optimization.caching import get_cache
        
        # Initialize optimization
        optimizer = get_performance_optimizer()
        cache = get_cache()
        
        # Test generation with performance tracking
        async with Guardian() as guardian:
            pipeline_config = {
                "name": "optimized_test",
                "generator_type": "tabular",
                "data_type": "tabular",
                "schema": {
                    "id": "integer",
                    "value": "float",
                    "category": "categorical"
                }
            }
            
            # Test multiple generations to exercise optimization
            generation_times = []
            
            for i in range(3):
                start_time = time.time()
                
                result = await guardian.generate(
                    pipeline_config=pipeline_config,
                    num_records=10 + i * 5,
                    seed=42 + i,
                    validate=False,
                    watermark=False
                )
                
                generation_time = time.time() - start_time
                generation_times.append(generation_time)
                
                print(f"   ✅ Generation {i+1}: {len(result.data)} records in {generation_time:.3f}s")
            
            # Check if performance is improving (with caching/optimization)
            avg_time = sum(generation_times) / len(generation_times)
            print(f"   📊 Average generation time: {avg_time:.3f}s")
            
            # Get optimization statistics
            perf_analysis = optimizer.get_performance_analysis()
            print(f"   📊 Total operations: {perf_analysis['total_operations']}")
            print(f"   📊 Success rate: {perf_analysis['successful_operations']}/{perf_analysis['total_operations']}")
            
            # Get cache statistics
            cache_stats = cache.get_stats()
            print(f"   📊 Cache hit rate: {cache_stats['hit_rate_percent']:.1f}%")
            print(f"   📊 Cache entries: {cache_stats['memory_entries']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Optimized Guardian test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_concurrent_performance():
    """Test concurrent performance capabilities"""
    print("🧪 Testing concurrent performance...")
    
    try:
        from synthetic_guardian.optimization.performance_optimizer import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        
        async def concurrent_task(task_id: int):
            """Simulate concurrent work"""
            await asyncio.sleep(random.uniform(0.01, 0.03))
            return f"Task {task_id} completed"
        
        # Test concurrent execution
        num_tasks = 20
        tasks = [concurrent_task(i) for i in range(num_tasks)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        print(f"   ✅ Concurrent execution: {len(results)} tasks in {concurrent_time:.3f}s")
        
        # Test auto-scaling
        scaling_actions = optimizer.auto_scale_resources(target_load=0.7)
        print(f"   ✅ Auto-scaling completed: {len(sum(scaling_actions.values(), []))} actions")
        
        # Test memory optimization
        optimizer.optimize_memory_usage()
        print(f"   ✅ Memory optimization completed")
        
        await optimizer.cleanup()
        return True
    except Exception as e:
        print(f"   ❌ Concurrent performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    print("⚡ Testing Performance Optimization & Caching")
    print("=" * 60)
    
    tests = [
        test_intelligent_cache,
        test_performance_optimizer,
        test_concurrent_performance,
        test_optimized_guardian
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
        print(f"🎉 SUCCESS: All {total} optimization tests passed!")
        return 0
    else:
        print(f"💥 PARTIAL: {passed}/{total} tests passed")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))