"""Performance testing utilities and benchmarks for Python components."""

import time
import asyncio
import statistics
from typing import Dict, List, Callable, Any
from functools import wraps
from dataclasses import dataclass
import pytest
import pandas as pd
import numpy as np


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""
    name: str
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    iterations: int
    throughput: float  # operations per second


class PerformanceTester:
    """Performance testing utilities."""
    
    def __init__(self, warmup_iterations: int = 3, min_iterations: int = 10):
        self.warmup_iterations = warmup_iterations
        self.min_iterations = min_iterations
    
    def benchmark(self, name: str, iterations: int = None):
        """Decorator for benchmarking functions."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.run_benchmark(name, func, iterations or self.min_iterations, *args, **kwargs)
            return wrapper
        return decorator
    
    def run_benchmark(self, name: str, func: Callable, iterations: int, *args, **kwargs) -> BenchmarkResult:
        """Run benchmark for a function."""
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        # Actual benchmark
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        throughput = 1.0 / mean_time if mean_time > 0 else 0
        
        return BenchmarkResult(
            name=name,
            mean_time=mean_time,
            median_time=median_time,
            std_dev=std_dev,
            min_time=min_time,
            max_time=max_time,
            iterations=iterations,
            throughput=throughput
        )
    
    async def run_async_benchmark(self, name: str, func: Callable, iterations: int, *args, **kwargs) -> BenchmarkResult:
        """Run benchmark for an async function."""
        # Warmup
        for _ in range(self.warmup_iterations):
            await func(*args, **kwargs)
        
        # Actual benchmark
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            await func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        throughput = 1.0 / mean_time if mean_time > 0 else 0
        
        return BenchmarkResult(
            name=name,
            mean_time=mean_time,
            median_time=median_time,
            std_dev=std_dev,
            min_time=min_time,
            max_time=max_time,
            iterations=iterations,
            throughput=throughput
        )


class LoadTester:
    """Load testing utilities."""
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
    
    async def run_concurrent_load(self, func: Callable, total_requests: int, 
                                 concurrent_requests: int = 10) -> Dict[str, Any]:
        """Run concurrent load test."""
        concurrent_requests = min(concurrent_requests, self.max_concurrent)
        results = []
        errors = []
        
        async def run_single_request():
            try:
                start_time = time.perf_counter()
                result = await func()
                end_time = time.perf_counter()
                results.append({
                    'success': True,
                    'duration': end_time - start_time,
                    'result': result
                })
            except Exception as e:
                errors.append({
                    'error': str(e),
                    'type': type(e).__name__
                })
        
        # Run requests in batches
        batches = [total_requests // concurrent_requests] * concurrent_requests
        remainder = total_requests % concurrent_requests
        for i in range(remainder):
            batches[i] += 1
        
        start_time = time.perf_counter()
        
        for batch_size in batches:
            if batch_size > 0:
                tasks = [run_single_request() for _ in range(batch_size)]
                await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        return {
            'total_requests': total_requests,
            'successful_requests': len(results),
            'failed_requests': len(errors),
            'success_rate': len(results) / total_requests if total_requests > 0 else 0,
            'total_duration': total_duration,
            'requests_per_second': total_requests / total_duration if total_duration > 0 else 0,
            'average_response_time': statistics.mean([r['duration'] for r in results]) if results else 0,
            'errors': errors
        }


class MemoryProfiler:
    """Memory profiling utilities."""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0
    
    def profile_memory(self, func: Callable):
        """Decorator to profile memory usage of a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            initial_memory = self.get_memory_usage()
            result = func(*args, **kwargs)
            final_memory = self.get_memory_usage()
            memory_delta = final_memory - initial_memory
            
            return {
                'result': result,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_delta_mb': memory_delta
            }
        return wrapper


# Fixtures for performance testing
@pytest.fixture
def performance_tester():
    """Performance tester fixture."""
    return PerformanceTester()


@pytest.fixture
def load_tester():
    """Load tester fixture."""
    return LoadTester()


@pytest.fixture
def memory_profiler():
    """Memory profiler fixture."""
    return MemoryProfiler()


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    size = 10000
    return pd.DataFrame({
        'id': range(size),
        'name': [f'User {i}' for i in range(size)],
        'age': np.random.randint(18, 80, size),
        'income': np.random.normal(50000, 15000, size),
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], size),
        'is_active': np.random.choice([True, False], size),
        'score': np.random.uniform(0, 100, size),
        'signup_date': pd.date_range('2020-01-01', periods=size, freq='H'),
    })


# Example performance tests
@pytest.mark.performance
@pytest.mark.slow
class TestDataGenerationPerformance:
    """Performance tests for data generation."""
    
    def test_tabular_generation_speed(self, performance_tester, large_dataset):
        """Test tabular data generation speed."""
        def generate_data():
            # Mock data generation - replace with actual generator
            return large_dataset.sample(n=1000, replace=True)
        
        result = performance_tester.run_benchmark("tabular_generation", generate_data, iterations=10)
        
        # Assert performance requirements
        assert result.mean_time < 1.0, f"Generation too slow: {result.mean_time:.3f}s"
        assert result.throughput > 1.0, f"Throughput too low: {result.throughput:.3f} ops/s"
        
        print(f"Tabular Generation Performance:")
        print(f"  Mean time: {result.mean_time:.3f}s")
        print(f"  Throughput: {result.throughput:.3f} ops/s")
    
    def test_validation_performance(self, performance_tester, large_dataset):
        """Test validation performance."""
        def validate_data():
            # Mock validation - replace with actual validator
            return {
                'statistical_score': np.random.uniform(0.8, 1.0),
                'privacy_score': np.random.uniform(0.7, 1.0),
                'quality_score': np.random.uniform(0.8, 1.0)
            }
        
        result = performance_tester.run_benchmark("data_validation", validate_data, iterations=20)
        
        # Assert performance requirements
        assert result.mean_time < 0.5, f"Validation too slow: {result.mean_time:.3f}s"
        
        print(f"Validation Performance:")
        print(f"  Mean time: {result.mean_time:.3f}s")
        print(f"  Throughput: {result.throughput:.3f} ops/s")


@pytest.mark.performance
@pytest.mark.slow
class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self, load_tester):
        """Test API performance under concurrent load."""
        async def mock_api_request():
            # Simulate API request processing time
            await asyncio.sleep(0.1)
            return {'status': 'success', 'data': {'id': 123}}
        
        result = await load_tester.run_concurrent_load(
            mock_api_request,
            total_requests=100,
            concurrent_requests=10
        )
        
        # Assert performance requirements
        assert result['success_rate'] >= 0.95, f"Success rate too low: {result['success_rate']:.2%}"
        assert result['requests_per_second'] >= 80, f"RPS too low: {result['requests_per_second']:.1f}"
        assert result['average_response_time'] <= 0.2, f"Response time too high: {result['average_response_time']:.3f}s"
        
        print(f"API Load Test Results:")
        print(f"  Success rate: {result['success_rate']:.2%}")
        print(f"  Requests per second: {result['requests_per_second']:.1f}")
        print(f"  Average response time: {result['average_response_time']:.3f}s")


@pytest.mark.performance
class TestMemoryUsage:
    """Memory usage tests."""
    
    def test_data_generation_memory_usage(self, memory_profiler):
        """Test memory usage during data generation."""
        
        @memory_profiler.profile_memory
        def generate_large_dataset():
            # Simulate memory-intensive data generation
            size = 50000
            data = pd.DataFrame({
                'id': range(size),
                'data': [f'Large string data {i}' * 10 for i in range(size)],
                'values': np.random.randn(size),
            })
            return data
        
        result = generate_large_dataset()
        
        # Assert memory usage is reasonable
        memory_delta = result['memory_delta_mb']
        assert memory_delta < 500, f"Memory usage too high: {memory_delta:.1f}MB"
        
        print(f"Memory Usage:")
        print(f"  Initial: {result['initial_memory_mb']:.1f}MB")
        print(f"  Final: {result['final_memory_mb']:.1f}MB")
        print(f"  Delta: {memory_delta:.1f}MB")


# Benchmark comparison utilities
def compare_benchmarks(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compare multiple benchmark results."""
    if not results:
        return {}
    
    sorted_results = sorted(results, key=lambda x: x.mean_time)
    fastest = sorted_results[0]
    
    comparison = {
        'fastest': fastest.name,
        'comparisons': []
    }
    
    for result in sorted_results[1:]:
        speedup = result.mean_time / fastest.mean_time
        comparison['comparisons'].append({
            'name': result.name,
            'slowdown_factor': speedup,
            'time_difference': result.mean_time - fastest.mean_time
        })
    
    return comparison


def print_benchmark_report(results: List[BenchmarkResult]):
    """Print a formatted benchmark report."""
    print("\nBenchmark Results:")
    print("=" * 80)
    
    for result in sorted(results, key=lambda x: x.mean_time):
        print(f"{result.name}:")
        print(f"  Mean time: {result.mean_time:.6f}s")
        print(f"  Median time: {result.median_time:.6f}s")
        print(f"  Std dev: {result.std_dev:.6f}s")
        print(f"  Min time: {result.min_time:.6f}s")
        print(f"  Max time: {result.max_time:.6f}s")
        print(f"  Throughput: {result.throughput:.2f} ops/s")
        print(f"  Iterations: {result.iterations}")
        print("-" * 40)
    
    if len(results) > 1:
        comparison = compare_benchmarks(results)
        print(f"\nFastest: {comparison['fastest']}")
        for comp in comparison['comparisons']:
            print(f"{comp['name']}: {comp['slowdown_factor']:.2f}x slower (+{comp['time_difference']:.6f}s)")