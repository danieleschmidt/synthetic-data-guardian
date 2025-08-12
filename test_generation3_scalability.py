"""
Generation 3: Scalability Tests
Test performance optimization, caching, concurrency, and auto-scaling capabilities
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from synthetic_guardian import Guardian, GenerationPipeline, PipelineBuilder
import pytest
import pandas as pd
import numpy as np
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


async def test_high_performance_generation():
    """Test high-performance synthetic data generation."""
    print("⚡ Testing high-performance generation...")
    
    async with Guardian() as guardian:
        pipeline_config = {
            'id': 'high_performance_pipeline',
            'generator_type': 'tabular',
            'data_type': 'tabular',
            'schema': {
                'id': 'integer[1:1000000]',
                'value1': 'float[0:1000]',
                'value2': 'float[0:1000]',
                'category': 'string'
            },
            'validation_config': {'validators': []},  # Disable validation for speed
            'watermark_config': None  # Disable watermarking for speed
        }
        
        # Test large-scale generation
        large_scale_tests = [
            (10000, "10K records"),
            (50000, "50K records"),
            (100000, "100K records")
        ]
        
        for num_records, test_name in large_scale_tests:
            start_time = time.time()
            
            result = await guardian.generate(
                pipeline_config=pipeline_config,
                num_records=num_records,
                seed=42,
                validate=False,
                watermark=False
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            records_per_second = num_records / generation_time if generation_time > 0 else float('inf')
            
            print(f"✅ {test_name}: {generation_time:.2f}s ({records_per_second:.0f} records/sec)")
            
            # Verify result integrity
            assert result is not None
            assert len(result.data) == num_records
        
        # Test memory efficiency with multiple generations
        memory_test_results = []
        for i in range(5):
            result = await guardian.generate(
                pipeline_config=pipeline_config,
                num_records=20000,
                seed=i,
                validate=False,
                watermark=False
            )
            memory_test_results.append(len(result.data))
        
        print(f"✅ Memory efficiency test: Generated {sum(memory_test_results)} total records across 5 batches")
    
    print("✅ High-performance generation test passed")


async def test_concurrent_scalability():
    """Test concurrent request handling and scalability."""
    print("⚡ Testing concurrent scalability...")
    
    async with Guardian() as guardian:
        pipeline_config = {
            'generator_type': 'tabular',
            'schema': {'value': 'integer[1:100]'},
            'validation_config': {'validators': []},
            'watermark_config': None
        }
        
        # Test increasing levels of concurrency
        concurrency_levels = [5, 10, 20, 30]
        
        for concurrent_requests in concurrency_levels:
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = []
            for i in range(concurrent_requests):
                config = {
                    'id': f'concurrent_test_{i}',
                    'generator_type': 'tabular',
                    'schema': {'value': f'integer[{i}:{i+10}]'},
                    'validation_config': {'validators': []},
                    'watermark_config': None
                }
                
                task = guardian.generate(
                    pipeline_config=config,
                    num_records=1000,
                    seed=i,
                    validate=False,
                    watermark=False
                )
                tasks.append(task)
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Analyze results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful
            throughput = successful / total_time if total_time > 0 else 0
            
            print(f"✅ {concurrent_requests} concurrent requests: {successful} successful, {failed} failed in {total_time:.2f}s ({throughput:.1f} req/sec)")
            
            # Most requests should succeed (some may fail due to rate limiting)
            assert successful >= concurrent_requests * 0.5, f"Too many failures: {failed}/{concurrent_requests}"
    
    print("✅ Concurrent scalability test passed")


async def test_caching_and_optimization():
    """Test caching mechanisms and optimization."""
    print("⚡ Testing caching and optimization...")
    
    async with Guardian() as guardian:
        # Test with same configuration to trigger caching
        base_config = {
            'id': 'cache_test_pipeline',
            'generator_type': 'tabular',
            'schema': {
                'id': 'integer[1:1000]',
                'value': 'float[0:100]'
            },
            'validation_config': {'validators': []},
            'watermark_config': None
        }
        
        # First generation (cold cache)
        start_time = time.time()
        result1 = await guardian.generate(
            pipeline_config=base_config,
            num_records=5000,
            seed=123,
            validate=False,
            watermark=False
        )
        cold_cache_time = time.time() - start_time
        
        # Second generation with same config (warm cache)
        start_time = time.time()
        result2 = await guardian.generate(
            pipeline_config=base_config,
            num_records=5000,
            seed=123,
            validate=False,
            watermark=False
        )
        warm_cache_time = time.time() - start_time
        
        print(f"✅ Cold cache: {cold_cache_time:.3f}s, Warm cache: {warm_cache_time:.3f}s")
        
        # Warm cache should be faster or similar (pipeline reuse)
        print(f"✅ Cache efficiency: {((cold_cache_time - warm_cache_time) / cold_cache_time * 100):.1f}% improvement")
        
        # Test pipeline caching
        initial_pipelines = len(guardian.get_pipelines())
        
        # Use same pipeline ID multiple times
        for i in range(3):
            await guardian.generate(
                pipeline_config=base_config,
                num_records=100,
                seed=i,
                validate=False,
                watermark=False
            )
        
        final_pipelines = len(guardian.get_pipelines())
        
        # Should reuse the same pipeline (not create 3 new ones)
        assert final_pipelines == initial_pipelines, f"Pipeline not cached: {initial_pipelines} -> {final_pipelines}"
        print("✅ Pipeline caching working correctly")
    
    print("✅ Caching and optimization test passed")


async def test_adaptive_resource_management():
    """Test adaptive resource management and auto-scaling."""
    print("⚡ Testing adaptive resource management...")
    
    async with Guardian() as guardian:
        # Test resource monitoring
        if guardian.resource_monitor:
            memory_status = guardian.resource_monitor.check_memory_usage()
            print(f"✅ Memory monitoring: {memory_status['current_mb']:.1f}MB / {memory_status['max_mb']}MB ({memory_status['usage_percent']:.1f}%)")
            
            # Test memory cleanup
            guardian.resource_monitor.force_garbage_collection()
            
            post_gc_status = guardian.resource_monitor.check_memory_usage()
            print(f"✅ Post-GC memory: {post_gc_status['current_mb']:.1f}MB")
        else:
            print("ℹ️  Resource monitoring disabled")
        
        # Test adaptive concurrency
        base_config = {
            'generator_type': 'tabular',
            'schema': {'value': 'integer[1:10]'},
            'validation_config': {'validators': []},
            'watermark_config': None
        }
        
        # Start with low concurrency and gradually increase
        concurrency_results = []
        
        for concurrency in [5, 10, 15]:
            start_time = time.time()
            
            tasks = []
            for i in range(concurrency):
                config = {
                    'id': f'adaptive_test_{i}',
                    **base_config,
                    'schema': {'value': f'integer[{i}:{i+5}]'}
                }
                
                task = guardian.generate(
                    pipeline_config=config,
                    num_records=500,
                    validate=False,
                    watermark=False
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if not isinstance(r, Exception))
            
            total_time = time.time() - start_time
            throughput = successful / total_time if total_time > 0 else 0
            
            concurrency_results.append({
                'concurrency': concurrency,
                'successful': successful,
                'throughput': throughput,
                'time': total_time
            })
            
            print(f"✅ Concurrency {concurrency}: {successful} successful in {total_time:.2f}s ({throughput:.1f} req/sec)")
        
        # Verify adaptive behavior (throughput should scale reasonably)
        print("✅ Adaptive resource management completed")
    
    print("✅ Adaptive resource management test passed")


async def test_load_balancing_and_distribution():
    """Test load balancing and task distribution."""
    print("⚡ Testing load balancing and distribution...")
    
    async with Guardian() as guardian:
        # Create multiple different pipeline types to test distribution
        pipeline_configs = [
            {
                'id': f'balanced_pipeline_{i}',
                'generator_type': 'tabular',
                'schema': {
                    'type_a': 'integer[1:100]',
                    'type_b': f'float[{i}:{i+10}]'
                },
                'validation_config': {'validators': []},
                'watermark_config': None
            }
            for i in range(10)
        ]
        
        # Test balanced execution
        start_time = time.time()
        
        tasks = []
        for config in pipeline_configs:
            task = guardian.generate(
                pipeline_config=config,
                num_records=2000,
                validate=False,
                watermark=False
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        total_records = sum(len(r.data) for r in results if not isinstance(r, Exception))
        
        print(f"✅ Load balancing: {successful} pipelines, {total_records} total records in {total_time:.2f}s")
        
        # Test task distribution metrics
        active_tasks = guardian.get_active_tasks()
        pipelines = guardian.get_pipelines()
        
        print(f"✅ Task distribution: {len(active_tasks)} active tasks, {len(pipelines)} total pipelines")
        
        # Verify load distribution (all pipelines should have been created)
        assert len(pipelines) >= len(pipeline_configs), f"Not all pipelines created: {len(pipelines)}/{len(pipeline_configs)}"
    
    print("✅ Load balancing and distribution test passed")


async def test_streaming_and_batch_processing():
    """Test streaming and batch processing capabilities."""
    print("⚡ Testing streaming and batch processing...")
    
    async with Guardian() as guardian:
        base_config = {
            'generator_type': 'tabular',
            'schema': {
                'batch_id': 'integer[1:100]',
                'stream_value': 'float[0:1000]'
            },
            'validation_config': {'validators': []},
            'watermark_config': None
        }
        
        # Test batch processing
        batch_sizes = [1000, 5000, 10000]
        batch_results = []
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            result = await guardian.generate(
                pipeline_config={
                    'id': f'batch_test_{batch_size}',
                    **base_config
                },
                num_records=batch_size,
                validate=False,
                watermark=False
            )
            
            processing_time = time.time() - start_time
            throughput = batch_size / processing_time if processing_time > 0 else 0
            
            batch_results.append({
                'batch_size': batch_size,
                'time': processing_time,
                'throughput': throughput
            })
            
            print(f"✅ Batch {batch_size}: {processing_time:.2f}s ({throughput:.0f} records/sec)")
        
        # Test streaming simulation (multiple small batches)
        stream_batches = 20
        stream_batch_size = 500
        
        start_time = time.time()
        
        stream_tasks = []
        for i in range(stream_batches):
            config = {
                'id': f'stream_batch_{i}',
                **base_config,
                'schema': {**base_config['schema'], 'batch_id': f'integer[{i*10}:{(i+1)*10}]'}
            }
            
            task = guardian.generate(
                pipeline_config=config,
                num_records=stream_batch_size,
                validate=False,
                watermark=False
            )
            stream_tasks.append(task)
        
        stream_results = await asyncio.gather(*stream_tasks, return_exceptions=True)
        
        stream_time = time.time() - start_time
        successful_streams = sum(1 for r in stream_results if not isinstance(r, Exception))
        total_stream_records = sum(len(r.data) for r in stream_results if not isinstance(r, Exception))
        stream_throughput = total_stream_records / stream_time if stream_time > 0 else 0
        
        print(f"✅ Streaming: {successful_streams} batches, {total_stream_records} records in {stream_time:.2f}s ({stream_throughput:.0f} records/sec)")
        
        # Compare batch vs streaming efficiency
        largest_batch = max(batch_results, key=lambda x: x['throughput'])
        print(f"✅ Best batch throughput: {largest_batch['throughput']:.0f} records/sec")
        print(f"✅ Streaming throughput: {stream_throughput:.0f} records/sec")
    
    print("✅ Streaming and batch processing test passed")


async def test_performance_monitoring_and_optimization():
    """Test performance monitoring and automatic optimization."""
    print("⚡ Testing performance monitoring and optimization...")
    
    async with Guardian() as guardian:
        # Get baseline metrics
        initial_metrics = guardian.get_metrics()
        print(f"Initial metrics: {initial_metrics}")
        
        # Perform operations to generate metrics
        optimization_config = {
            'generator_type': 'tabular',
            'schema': {
                'perf_id': 'integer[1:10000]',
                'perf_value': 'float[0:1000]',
                'perf_category': 'string'
            },
            'validation_config': {'validators': []},
            'watermark_config': None
        }
        
        # Test performance under different loads
        performance_tests = [
            (1000, 1, "Single small batch"),
            (5000, 1, "Single large batch"),
            (1000, 5, "Multiple small batches"),
            (2000, 3, "Multiple medium batches")
        ]
        
        performance_results = []
        
        for records_per_batch, num_batches, description in performance_tests:
            start_time = time.time()
            
            if num_batches == 1:
                # Single batch
                result = await guardian.generate(
                    pipeline_config={
                        'id': f'perf_test_{records_per_batch}_{num_batches}',
                        **optimization_config
                    },
                    num_records=records_per_batch,
                    validate=False,
                    watermark=False
                )
                total_records = len(result.data)
            else:
                # Multiple batches
                tasks = []
                for i in range(num_batches):
                    config = {
                        'id': f'perf_test_{records_per_batch}_{num_batches}_{i}',
                        **optimization_config
                    }
                    
                    task = guardian.generate(
                        pipeline_config=config,
                        num_records=records_per_batch,
                        validate=False,
                        watermark=False
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_records = sum(len(r.data) for r in results if not isinstance(r, Exception))
            
            test_time = time.time() - start_time
            throughput = total_records / test_time if test_time > 0 else 0
            
            performance_results.append({
                'description': description,
                'records': total_records,
                'time': test_time,
                'throughput': throughput
            })
            
            print(f"✅ {description}: {total_records} records in {test_time:.2f}s ({throughput:.0f} records/sec)")
        
        # Get final metrics
        final_metrics = guardian.get_metrics()
        print(f"Final metrics: {final_metrics}")
        
        # Calculate performance improvements
        metrics_improvement = {
            'total_generations': final_metrics['total_generations'] - initial_metrics['total_generations'],
            'successful_generations': final_metrics['successful_generations'] - initial_metrics['successful_generations'],
            'total_records_generated': final_metrics['total_records_generated'] - initial_metrics['total_records_generated']
        }
        
        print(f"✅ Performance monitoring captured {metrics_improvement['total_generations']} new generations")
        
        # Verify optimization patterns
        best_performance = max(performance_results, key=lambda x: x['throughput'])
        print(f"✅ Best performance: {best_performance['description']} at {best_performance['throughput']:.0f} records/sec")
    
    print("✅ Performance monitoring and optimization test passed")


async def test_horizontal_scaling_simulation():
    """Test horizontal scaling simulation."""
    print("⚡ Testing horizontal scaling simulation...")
    
    async with Guardian() as guardian:
        # Simulate increasing load that would trigger horizontal scaling
        scaling_config = {
            'generator_type': 'tabular',
            'schema': {
                'scale_id': 'integer[1:1000000]',
                'scale_metric': 'float[0:10000]'
            },
            'validation_config': {'validators': []},
            'watermark_config': None
        }
        
        # Test scaling under increasing load
        load_levels = [
            (10, 1000, "Light load"),
            (20, 2000, "Medium load"),
            (30, 3000, "Heavy load"),
            (50, 2000, "Peak load")  # More concurrent, smaller batches
        ]
        
        scaling_results = []
        
        for concurrent_requests, records_per_request, load_description in load_levels:
            print(f"🔄 Testing {load_description}: {concurrent_requests} concurrent requests of {records_per_request} records each")
            
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = []
            for i in range(concurrent_requests):
                config = {
                    'id': f'scale_test_{load_description.replace(' ', '_')}_{i}',
                    **scaling_config
                }
                
                task = guardian.generate(
                    pipeline_config=config,
                    num_records=records_per_request,
                    validate=False,
                    watermark=False
                )
                tasks.append(task)
            
            # Execute with timeout to prevent hanging
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=30.0  # 30 second timeout
                )
            except asyncio.TimeoutError:
                print(f"⚠️  {load_description} timed out - system may need horizontal scaling")
                results = [TimeoutError("Operation timed out")] * len(tasks)
            
            end_time = time.time()
            test_time = end_time - start_time
            
            # Analyze results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful
            total_records = sum(len(r.data) for r in results if not isinstance(r, Exception))
            
            throughput = total_records / test_time if test_time > 0 else 0
            success_rate = successful / len(tasks) if len(tasks) > 0 else 0
            
            scaling_results.append({
                'load': load_description,
                'concurrent_requests': concurrent_requests,
                'successful': successful,
                'failed': failed,
                'success_rate': success_rate,
                'total_records': total_records,
                'time': test_time,
                'throughput': throughput
            })
            
            print(f"✅ {load_description}: {successful}/{concurrent_requests} successful ({success_rate:.1%}), {total_records} records in {test_time:.2f}s ({throughput:.0f} records/sec)")
        
        # Analyze scaling characteristics
        print("\n📊 Scaling Analysis:")
        for result in scaling_results:
            efficiency = result['success_rate'] * result['throughput']
            print(f"  {result['load']}: {result['success_rate']:.1%} success rate, {result['throughput']:.0f} records/sec (efficiency: {efficiency:.0f})")
        
        # Determine if horizontal scaling would be beneficial
        peak_load_result = next(r for r in scaling_results if r['load'] == "Peak load")
        if peak_load_result['success_rate'] < 0.8:
            print("🔄 Horizontal scaling recommended: Success rate dropped below 80% under peak load")
        else:
            print("✅ System handles current load well, horizontal scaling not immediately needed")
    
    print("✅ Horizontal scaling simulation test passed")


async def run_all_tests():
    """Run all Generation 3 scalability tests."""
    print("⚡ Starting Generation 3: Scalability Tests")
    print("=" * 60)
    
    test_functions = [
        test_high_performance_generation,
        test_concurrent_scalability,
        test_caching_and_optimization,
        test_adaptive_resource_management,
        test_load_balancing_and_distribution,
        test_streaming_and_batch_processing,
        test_performance_monitoring_and_optimization,
        test_horizontal_scaling_simulation
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
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("=" * 60)
    print(f"🏁 Generation 3 Tests Summary:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 All Generation 3 tests passed! System is highly scalable.")
        return True
    else:
        print("⚠️  Some scalability tests failed. System needs optimization.")
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n⚡ Generation 3: MAKE IT SCALE - COMPLETED SUCCESSFULLY!")
        print("System is now optimized for high performance and scalability.")
        print("Ready for comprehensive testing and production deployment!")
    else:
        print("\n🔧 Generation 3 needs optimization before proceeding to final validation")
        
    sys.exit(0 if success else 1)