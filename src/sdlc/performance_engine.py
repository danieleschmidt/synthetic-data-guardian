"""
Performance Engine - Advanced performance optimization and scaling
"""

import asyncio
import time
import threading
import multiprocessing
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref

from ..utils.logger import get_logger


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    BALANCED = "balanced"


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    throughput: float = 0.0
    concurrent_operations: int = 0
    optimization_score: float = 0.0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 1
    last_accessed: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    size_bytes: int = 0


class AdvancedCache:
    """Advanced multi-strategy cache implementation."""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        """Initialize advanced cache."""
        self.max_size = max_size
        self.strategy = strategy
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # For LRU
        self.frequency_counter = defaultdict(int)  # For LFU
        self._lock = threading.RLock()
        
        # Adaptive strategy parameters
        self.hit_rate_threshold = 0.8
        self.recent_operations = deque(maxlen=1000)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.entries:
                self.misses += 1
                self._record_operation('miss')
                return None
            
            entry = self.entries[key]
            
            # Check TTL
            if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                del self.entries[key]
                self.misses += 1
                self._record_operation('miss')
                return None
            
            # Update access metadata
            entry.last_accessed = time.time()
            entry.access_count += 1
            self.frequency_counter[key] += 1
            
            # Update LRU order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.hits += 1
            self._record_operation('hit')
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self._lock:
            current_time = time.time()
            
            # Calculate size (simplified)
            size_bytes = self._estimate_size(value)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=current_time,
                last_accessed=current_time,
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            # If key already exists, update it
            if key in self.entries:
                self.entries[key] = entry
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return
            
            # Check if we need to evict
            if len(self.entries) >= self.max_size:
                self._evict()
            
            # Add new entry
            self.entries[key] = entry
            self.access_order.append(key)
            self.frequency_counter[key] += 1
    
    def _evict(self) -> None:
        """Evict entries based on strategy."""
        if not self.entries:
            return
        
        evict_key = None
        
        if self.strategy == CacheStrategy.LRU:
            evict_key = self.access_order.popleft()
        elif self.strategy == CacheStrategy.LFU:
            evict_key = min(self.frequency_counter.items(), key=lambda x: x[1])[0]
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first
            current_time = time.time()
            for key, entry in self.entries.items():
                if entry.ttl and current_time - entry.timestamp > entry.ttl:
                    evict_key = key
                    break
            # If no expired entries, fallback to LRU
            if not evict_key:
                evict_key = self.access_order.popleft()
        elif self.strategy == CacheStrategy.ADAPTIVE:
            evict_key = self._adaptive_evict()
        
        if evict_key and evict_key in self.entries:
            del self.entries[evict_key]
            if evict_key in self.access_order:
                self.access_order.remove(evict_key)
            if evict_key in self.frequency_counter:
                del self.frequency_counter[evict_key]
            self.evictions += 1
    
    def _adaptive_evict(self) -> Optional[str]:
        """Adaptive eviction based on usage patterns."""
        current_hit_rate = self.get_hit_rate()
        
        if current_hit_rate < self.hit_rate_threshold:
            # Low hit rate - use LFU to remove least accessed items
            if self.frequency_counter:
                return min(self.frequency_counter.items(), key=lambda x: x[1])[0]
        else:
            # Good hit rate - use LRU to maintain recency
            if self.access_order:
                return self.access_order.popleft()
        
        return None
    
    def _record_operation(self, operation: str) -> None:
        """Record cache operation for adaptive strategy."""
        self.recent_operations.append({
            'operation': operation,
            'timestamp': time.time()
        })
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        import sys
        return sys.getsizeof(value)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.entries),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.get_hit_rate(),
            'strategy': self.strategy.value
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.entries.clear()
            self.access_order.clear()
            self.frequency_counter.clear()


class WorkerPool:
    """Advanced worker pool for concurrent processing."""
    
    def __init__(self, pool_type: str = "thread", max_workers: Optional[int] = None):
        """Initialize worker pool."""
        self.pool_type = pool_type
        self.max_workers = max_workers or self._get_optimal_worker_count()
        
        if pool_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        elif pool_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")
        
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self._lock = threading.Lock()
    
    def _get_optimal_worker_count(self) -> int:
        """Get optimal worker count based on system resources."""
        cpu_count = multiprocessing.cpu_count()
        
        if self.pool_type == "thread":
            # For I/O bound tasks, can use more threads than CPUs
            return min(cpu_count * 2, 32)
        else:
            # For CPU bound tasks, use CPU count
            return cpu_count
    
    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task to worker pool."""
        with self._lock:
            self.active_tasks += 1
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, func, *args, **kwargs)
            
            with self._lock:
                self.completed_tasks += 1
                self.active_tasks -= 1
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failed_tasks += 1
                self.active_tasks -= 1
            raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        return {
            'pool_type': self.pool_type,
            'max_workers': self.max_workers,
            'active_tasks': self.active_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.completed_tasks / (self.completed_tasks + self.failed_tasks) if (self.completed_tasks + self.failed_tasks) > 0 else 0.0
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown worker pool."""
        self.executor.shutdown(wait=wait)


class PerformanceEngine:
    """
    Performance Engine - Advanced performance optimization and scaling.
    
    Provides comprehensive performance optimization including:
    - Multi-tier caching with intelligent eviction
    - Adaptive worker pools for concurrent processing
    - Resource monitoring and auto-scaling
    - Memory optimization and garbage collection
    - Performance profiling and bottleneck detection
    - Load balancing and request distribution
    """
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        """Initialize performance engine."""
        self.config = config or {}
        self.logger = logger or get_logger(self.__class__.__name__)
        
        # Caching
        self.caches = {
            'default': AdvancedCache(max_size=1000, strategy=CacheStrategy.ADAPTIVE),
            'generation': AdvancedCache(max_size=100, strategy=CacheStrategy.LRU),
            'validation': AdvancedCache(max_size=500, strategy=CacheStrategy.TTL)
        }
        
        # Worker pools
        self.worker_pools = {
            'cpu_intensive': WorkerPool(pool_type="process", max_workers=multiprocessing.cpu_count()),
            'io_intensive': WorkerPool(pool_type="thread", max_workers=multiprocessing.cpu_count() * 2),
            'general': WorkerPool(pool_type="thread", max_workers=multiprocessing.cpu_count())
        }
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.performance_history = deque(maxlen=1000)
        self.bottlenecks = []
        
        # Resource monitoring
        self.resource_limits = {
            'max_memory_percent': 85.0,
            'max_cpu_percent': 80.0,
            'gc_threshold': 70.0
        }
        
        # Auto-scaling parameters
        self.scaling_config = {
            'enabled': True,
            'scale_up_threshold': 0.8,
            'scale_down_threshold': 0.3,
            'max_scale_factor': 4.0,
            'min_scale_factor': 0.5
        }
        
        self._optimization_strategies = self._initialize_optimization_strategies()
        self.logger.info("Performance Engine initialized")
    
    def _initialize_optimization_strategies(self) -> Dict[OptimizationStrategy, Dict[str, Any]]:
        """Initialize optimization strategies."""
        return {
            OptimizationStrategy.CPU_INTENSIVE: {
                'worker_pool': 'cpu_intensive',
                'cache_strategy': CacheStrategy.LFU,
                'batch_size': 100,
                'memory_optimization': True
            },
            OptimizationStrategy.IO_INTENSIVE: {
                'worker_pool': 'io_intensive',
                'cache_strategy': CacheStrategy.LRU,
                'batch_size': 50,
                'memory_optimization': False
            },
            OptimizationStrategy.MEMORY_INTENSIVE: {
                'worker_pool': 'general',
                'cache_strategy': CacheStrategy.TTL,
                'batch_size': 25,
                'memory_optimization': True
            },
            OptimizationStrategy.BALANCED: {
                'worker_pool': 'general',
                'cache_strategy': CacheStrategy.ADAPTIVE,
                'batch_size': 50,
                'memory_optimization': True
            }
        }
    
    async def optimize_operation(self, operation: Callable, *args, 
                                strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                                cache_key: Optional[str] = None,
                                **kwargs) -> Any:
        """Optimize operation execution with advanced strategies."""
        start_time = time.time()
        
        try:
            # Check cache first
            if cache_key:
                cached_result = self.get_cached_result(cache_key, 'default')
                if cached_result is not None:
                    self.logger.debug(f"Cache hit for key: {cache_key}")
                    return cached_result
            
            # Get optimization configuration
            opt_config = self._optimization_strategies[strategy]
            
            # Pre-optimization
            await self._pre_optimize(strategy)
            
            # Execute operation with appropriate worker pool
            pool_name = opt_config['worker_pool']
            result = await self.worker_pools[pool_name].submit(operation, *args, **kwargs)
            
            # Cache result if cache key provided
            if cache_key:
                self.cache_result(cache_key, result, 'default', ttl=3600)  # 1 hour TTL
            
            # Post-optimization
            await self._post_optimize(strategy, time.time() - start_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Operation optimization failed: {e}")
            raise
    
    async def _pre_optimize(self, strategy: OptimizationStrategy) -> None:
        """Pre-optimization tasks."""
        if strategy in [OptimizationStrategy.MEMORY_INTENSIVE, OptimizationStrategy.BALANCED]:
            # Check memory usage and trigger GC if needed
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.resource_limits['gc_threshold']:
                gc.collect()
                self.logger.debug("Triggered garbage collection for memory optimization")
        
        # Update metrics
        self.metrics.concurrent_operations += 1
    
    async def _post_optimize(self, strategy: OptimizationStrategy, execution_time: float) -> None:
        """Post-optimization tasks."""
        # Update metrics
        self.metrics.concurrent_operations -= 1
        self.metrics.execution_time = execution_time
        
        # Record performance data
        self.performance_history.append({
            'strategy': strategy.value,
            'execution_time': execution_time,
            'timestamp': time.time(),
            'memory_usage': psutil.virtual_memory().percent,
            'cpu_usage': psutil.cpu_percent()
        })
        
        # Check for auto-scaling needs
        if self.scaling_config['enabled']:
            await self._check_scaling_needs()
    
    def cache_result(self, key: str, value: Any, cache_name: str = 'default', ttl: Optional[float] = None) -> None:
        """Cache result with specified cache."""
        if cache_name in self.caches:
            self.caches[cache_name].put(key, value, ttl)
        else:
            self.logger.warning(f"Unknown cache: {cache_name}")
    
    def get_cached_result(self, key: str, cache_name: str = 'default') -> Optional[Any]:
        """Get cached result from specified cache."""
        if cache_name in self.caches:
            return self.caches[cache_name].get(key)
        else:
            self.logger.warning(f"Unknown cache: {cache_name}")
            return None
    
    async def batch_process(self, items: List[Any], operation: Callable,
                           strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                           **kwargs) -> List[Any]:
        """Process items in optimized batches."""
        if not items:
            return []
        
        opt_config = self._optimization_strategies[strategy]
        batch_size = opt_config['batch_size']
        
        # Split items into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches concurrently
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch, operation, strategy, **kwargs))
            tasks.append(task)
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error(f"Batch processing error: {batch_result}")
                continue
            results.extend(batch_result)
        
        return results
    
    async def _process_batch(self, batch: List[Any], operation: Callable,
                            strategy: OptimizationStrategy, **kwargs) -> List[Any]:
        """Process a single batch of items."""
        results = []
        
        for item in batch:
            try:
                result = await self.optimize_operation(operation, item, strategy=strategy, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Item processing error: {e}")
                results.append(None)  # or handle error differently
        
        return results
    
    async def _check_scaling_needs(self) -> None:
        """Check if scaling is needed based on performance metrics."""
        if len(self.performance_history) < 10:
            return  # Need more data points
        
        # Calculate recent performance metrics
        recent_data = list(self.performance_history)[-10:]
        avg_cpu = sum(data['cpu_usage'] for data in recent_data) / len(recent_data)
        avg_memory = sum(data['memory_usage'] for data in recent_data) / len(recent_data)
        avg_execution_time = sum(data['execution_time'] for data in recent_data) / len(recent_data)
        
        # Determine if scaling is needed
        resource_utilization = max(avg_cpu / 100, avg_memory / 100)
        
        if resource_utilization > self.scaling_config['scale_up_threshold']:
            await self._scale_up()
        elif resource_utilization < self.scaling_config['scale_down_threshold']:
            await self._scale_down()
    
    async def _scale_up(self) -> None:
        """Scale up resources."""
        current_workers = {}
        for name, pool in self.worker_pools.items():
            current_workers[name] = pool.max_workers
        
        # Increase worker pool sizes
        scale_factor = min(1.5, self.scaling_config['max_scale_factor'])
        
        for name, pool in self.worker_pools.items():
            new_size = int(pool.max_workers * scale_factor)
            # For now, just log the scaling action
            # In a real implementation, you'd recreate the pool with new size
            self.logger.info(f"Scaling up {name} pool from {pool.max_workers} to {new_size} workers")
    
    async def _scale_down(self) -> None:
        """Scale down resources."""
        # Decrease worker pool sizes
        scale_factor = max(0.8, self.scaling_config['min_scale_factor'])
        
        for name, pool in self.worker_pools.items():
            new_size = max(1, int(pool.max_workers * scale_factor))
            # For now, just log the scaling action
            self.logger.info(f"Scaling down {name} pool from {pool.max_workers} to {new_size} workers")
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        initial_memory = psutil.virtual_memory().percent
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear cache if memory usage is high
        if initial_memory > self.resource_limits['max_memory_percent']:
            for cache in self.caches.values():
                cache.clear()
            self.logger.info("Cleared caches due to high memory usage")
        
        # Clean up performance history if it's getting large
        if len(self.performance_history) > 500:
            # Keep only recent 250 entries
            while len(self.performance_history) > 250:
                self.performance_history.popleft()
        
        final_memory = psutil.virtual_memory().percent
        memory_freed = initial_memory - final_memory
        
        return {
            'initial_memory_percent': initial_memory,
            'final_memory_percent': final_memory,
            'memory_freed_percent': memory_freed,
            'objects_collected': collected
        }
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        if not self.performance_history:
            return bottlenecks
        
        recent_data = list(self.performance_history)[-50:]  # Last 50 operations
        
        # Check for slow operations
        avg_execution_time = sum(data['execution_time'] for data in recent_data) / len(recent_data)
        slow_threshold = avg_execution_time * 2  # 2x average is considered slow
        
        slow_operations = [data for data in recent_data if data['execution_time'] > slow_threshold]
        if slow_operations:
            bottlenecks.append({
                'type': 'slow_operations',
                'severity': 'medium',
                'count': len(slow_operations),
                'avg_time': sum(op['execution_time'] for op in slow_operations) / len(slow_operations),
                'recommendation': 'Optimize slow operations or use different strategy'
            })
        
        # Check for high resource usage
        avg_cpu = sum(data['cpu_usage'] for data in recent_data) / len(recent_data)
        avg_memory = sum(data['memory_usage'] for data in recent_data) / len(recent_data)
        
        if avg_cpu > self.resource_limits['max_cpu_percent']:
            bottlenecks.append({
                'type': 'high_cpu_usage',
                'severity': 'high',
                'value': avg_cpu,
                'threshold': self.resource_limits['max_cpu_percent'],
                'recommendation': 'Scale up CPU resources or optimize CPU-intensive operations'
            })
        
        if avg_memory > self.resource_limits['max_memory_percent']:
            bottlenecks.append({
                'type': 'high_memory_usage',
                'severity': 'high',
                'value': avg_memory,
                'threshold': self.resource_limits['max_memory_percent'],
                'recommendation': 'Optimize memory usage or scale up memory resources'
            })
        
        # Check cache performance
        for cache_name, cache in self.caches.items():
            hit_rate = cache.get_hit_rate()
            if hit_rate < 0.5:  # Less than 50% hit rate
                bottlenecks.append({
                    'type': 'low_cache_hit_rate',
                    'severity': 'medium',
                    'cache_name': cache_name,
                    'hit_rate': hit_rate,
                    'recommendation': 'Review cache strategy or increase cache size'
                })
        
        self.bottlenecks = bottlenecks
        return bottlenecks
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        # Calculate cache metrics
        cache_metrics = {}
        for name, cache in self.caches.items():
            cache_metrics[name] = cache.get_stats()
        
        # Calculate worker pool metrics
        pool_metrics = {}
        for name, pool in self.worker_pools.items():
            pool_metrics[name] = pool.get_stats()
        
        # Calculate recent performance
        if self.performance_history:
            recent_data = list(self.performance_history)[-10:]
            avg_execution_time = sum(data['execution_time'] for data in recent_data) / len(recent_data)
            avg_cpu = sum(data['cpu_usage'] for data in recent_data) / len(recent_data)
            avg_memory = sum(data['memory_usage'] for data in recent_data) / len(recent_data)
        else:
            avg_execution_time = avg_cpu = avg_memory = 0.0
        
        # Calculate overall performance score
        performance_score = self._calculate_performance_score()
        
        return {
            'timestamp': time.time(),
            'performance_score': performance_score,
            'cache_metrics': cache_metrics,
            'worker_pool_metrics': pool_metrics,
            'recent_performance': {
                'avg_execution_time': avg_execution_time,
                'avg_cpu_usage': avg_cpu,
                'avg_memory_usage': avg_memory
            },
            'current_resources': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100
            },
            'bottlenecks': self.bottlenecks,
            'concurrent_operations': self.metrics.concurrent_operations
        }
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        score = 100.0
        
        # Deduct for cache performance
        total_hit_rate = 0.0
        cache_count = 0
        for cache in self.caches.values():
            hit_rate = cache.get_hit_rate()
            if hit_rate > 0:  # Only count caches that have been used
                total_hit_rate += hit_rate
                cache_count += 1
        
        if cache_count > 0:
            avg_hit_rate = total_hit_rate / cache_count
            if avg_hit_rate < 0.8:
                score -= (0.8 - avg_hit_rate) * 50  # Up to 50 points for cache performance
        
        # Deduct for resource usage
        current_cpu = psutil.cpu_percent()
        current_memory = psutil.virtual_memory().percent
        
        if current_cpu > self.resource_limits['max_cpu_percent']:
            score -= (current_cpu - self.resource_limits['max_cpu_percent']) * 2
        
        if current_memory > self.resource_limits['max_memory_percent']:
            score -= (current_memory - self.resource_limits['max_memory_percent']) * 2
        
        # Deduct for bottlenecks
        score -= len(self.bottlenecks) * 10
        
        return max(score, 0.0)
    
    async def warm_cache(self, cache_name: str, preload_data: Dict[str, Any]) -> None:
        """Warm up cache with preloaded data."""
        if cache_name not in self.caches:
            self.logger.warning(f"Unknown cache: {cache_name}")
            return
        
        cache = self.caches[cache_name]
        
        for key, value in preload_data.items():
            cache.put(key, value)
        
        self.logger.info(f"Warmed cache '{cache_name}' with {len(preload_data)} entries")
    
    def create_custom_cache(self, name: str, max_size: int = 1000, 
                           strategy: CacheStrategy = CacheStrategy.LRU) -> None:
        """Create a custom cache with specified parameters."""
        self.caches[name] = AdvancedCache(max_size=max_size, strategy=strategy)
        self.logger.info(f"Created custom cache '{name}' with strategy {strategy.value}")
    
    async def benchmark_operation(self, operation: Callable, *args, 
                                 iterations: int = 10, **kwargs) -> Dict[str, Any]:
        """Benchmark operation performance."""
        execution_times = []
        memory_usage = []
        
        for i in range(iterations):
            # Measure memory before
            memory_before = psutil.virtual_memory().percent
            
            # Measure execution time
            start_time = time.time()
            try:
                await operation(*args, **kwargs)
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
            except Exception as e:
                self.logger.error(f"Benchmark iteration {i} failed: {e}")
                continue
            
            # Measure memory after
            memory_after = psutil.virtual_memory().percent
            memory_usage.append(memory_after - memory_before)
        
        if not execution_times:
            return {'error': 'All benchmark iterations failed'}
        
        # Calculate statistics
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        avg_memory = sum(memory_usage) / len(memory_usage)
        
        return {
            'iterations': len(execution_times),
            'avg_execution_time': avg_time,
            'min_execution_time': min_time,
            'max_execution_time': max_time,
            'avg_memory_usage': avg_memory,
            'throughput_per_second': 1.0 / avg_time if avg_time > 0 else 0,
            'execution_times': execution_times,
            'memory_usage': memory_usage
        }
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        metrics = self.get_performance_metrics()
        bottlenecks = self.detect_bottlenecks()
        
        # Performance analysis
        recommendations = []
        
        if metrics['performance_score'] < 70:
            recommendations.append("Overall performance is below optimal - review bottlenecks")
        
        for bottleneck in bottlenecks:
            recommendations.append(bottleneck.get('recommendation', ''))
        
        # Cache analysis
        cache_analysis = {}
        for cache_name, cache_stats in metrics['cache_metrics'].items():
            if cache_stats['hit_rate'] < 0.8:
                cache_analysis[cache_name] = f"Low hit rate ({cache_stats['hit_rate']:.2%})"
        
        return {
            'report_timestamp': time.time(),
            'performance_score': metrics['performance_score'],
            'metrics_summary': metrics,
            'bottlenecks': bottlenecks,
            'cache_analysis': cache_analysis,
            'recommendations': recommendations,
            'optimization_opportunities': self._identify_optimization_opportunities(metrics),
            'scaling_status': self._get_scaling_status()
        }
    
    def _identify_optimization_opportunities(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Check for underutilized caches
        for cache_name, cache_stats in metrics['cache_metrics'].items():
            utilization = cache_stats['size'] / cache_stats['max_size']
            if utilization < 0.3 and cache_stats['hits'] > 0:
                opportunities.append({
                    'type': 'cache_resize',
                    'target': cache_name,
                    'current_size': cache_stats['max_size'],
                    'recommended_size': int(cache_stats['max_size'] * 0.7),
                    'potential_benefit': 'Reduced memory usage'
                })
        
        # Check for worker pool optimization
        for pool_name, pool_stats in metrics['worker_pool_metrics'].items():
            if pool_stats['success_rate'] < 0.9:
                opportunities.append({
                    'type': 'worker_pool_optimization',
                    'target': pool_name,
                    'current_success_rate': pool_stats['success_rate'],
                    'potential_benefit': 'Improved reliability and performance'
                })
        
        return opportunities
    
    def _get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            'auto_scaling_enabled': self.scaling_config['enabled'],
            'scale_up_threshold': self.scaling_config['scale_up_threshold'],
            'scale_down_threshold': self.scaling_config['scale_down_threshold'],
            'current_utilization': max(
                psutil.cpu_percent() / 100,
                psutil.virtual_memory().percent / 100
            ),
            'worker_pools': {
                name: pool.max_workers for name, pool in self.worker_pools.items()
            }
        }
    
    async def cleanup(self) -> None:
        """Clean up performance engine resources."""
        # Clear all caches
        for cache in self.caches.values():
            cache.clear()
        
        # Shutdown worker pools
        for pool in self.worker_pools.values():
            pool.shutdown(wait=False)
        
        self.logger.info("Performance Engine cleanup completed")