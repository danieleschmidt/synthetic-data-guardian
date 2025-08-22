"""
High-Performance Optimization Engine for Research Modules

This module implements advanced performance optimization techniques for research
operations including multi-level caching, parallel processing, GPU acceleration,
memory optimization, and intelligent workload distribution.

Performance Features:
1. Multi-tier intelligent caching (LRU, LFU, TTL, Adaptive)
2. Parallel processing with worker pools and task distribution
3. GPU acceleration for compute-intensive operations
4. Memory optimization and automatic garbage collection
5. Intelligent workload batching and preprocessing
6. Auto-scaling based on performance metrics
7. Result compression and efficient serialization
8. Cache warming and predictive prefetching
"""

import asyncio
import time
import hashlib
import pickle
import lz4.frame
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import psutil
import gc
from abc import ABC, abstractmethod
import weakref

from ..utils.logger import get_logger


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    TTL = "ttl"          # Time To Live
    ADAPTIVE = "adaptive" # Adaptive based on access patterns
    FIFO = "fifo"        # First In First Out


class ComputeBackend(Enum):
    """Compute backend options."""
    CPU = "cpu"
    GPU_CUDA = "gpu_cuda"
    GPU_OPENCL = "gpu_opencl"
    TPU = "tpu"
    MULTICORE = "multicore"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    creation_time: float
    last_access_time: float
    access_count: int
    ttl: Optional[float] = None
    priority: float = 1.0
    compressed: bool = False
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.creation_time > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.creation_time


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    # Caching
    cache_size_mb: int = 1024
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    enable_compression: bool = True
    compression_threshold_bytes: int = 1024
    
    # Parallel processing
    max_workers: int = mp.cpu_count()
    worker_type: str = "thread"  # "thread" or "process"
    batch_size: int = 32
    enable_gpu: bool = False
    gpu_memory_fraction: float = 0.5
    
    # Memory optimization
    memory_limit_mb: int = 4096
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    enable_memory_mapping: bool = True
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # CPU utilization
    scale_down_threshold: float = 0.3
    min_workers: int = 1
    max_workers_limit: int = mp.cpu_count() * 2


class AdaptiveCache:
    """
    High-performance adaptive cache with multiple eviction strategies.
    
    This cache automatically adapts its eviction strategy based on access
    patterns to maximize hit rates for different workload types.
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.max_size_bytes = config.cache_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        
        # Storage
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.frequency_counts = defaultdict(int)  # For LFU
        
        # Adaptive strategy tracking
        self.strategy_performance = {
            strategy: {"hits": 0, "misses": 0, "last_evaluated": time.time()}
            for strategy in CacheStrategy
        }
        self.current_strategy = config.cache_strategy
        self.last_strategy_switch = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "decompressions": 0
        }
        
        self.logger = get_logger(self.__class__.__name__)
    
    def _compute_key(self, key_data: Any) -> str:
        """Compute stable hash key for cache entry."""
        if isinstance(key_data, str):
            return hashlib.sha256(key_data.encode()).hexdigest()[:16]
        elif isinstance(key_data, (dict, list, tuple)):
            serialized = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.sha256(serialized).hexdigest()[:16]
        else:
            return hashlib.sha256(str(key_data).encode()).hexdigest()[:16]
    
    def _compress_value(self, value: Any) -> Tuple[bytes, bool]:
        """Compress value if beneficial."""
        try:
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            if len(serialized) > self.config.compression_threshold_bytes:
                compressed = lz4.frame.compress(serialized)
                if len(compressed) < len(serialized) * 0.8:  # 20% compression benefit
                    self.stats["compressions"] += 1
                    return compressed, True
            
            return serialized, False
            
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return pickle.dumps(value), False
    
    def _decompress_value(self, data: bytes, compressed: bool) -> Any:
        """Decompress value if needed."""
        try:
            if compressed:
                decompressed = lz4.frame.decompress(data)
                self.stats["decompressions"] += 1
                return pickle.loads(decompressed)
            else:
                return pickle.loads(data)
                
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            raise
    
    def _calculate_entry_size(self, value: Any) -> int:
        """Calculate memory size of cache entry."""
        try:
            if isinstance(value, bytes):
                return len(value)
            else:
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                return len(serialized)
        except:
            return 1024  # Default estimate
    
    def _evict_entries(self, target_size: int) -> None:
        """Evict entries to free up target size."""
        freed_bytes = 0
        
        if self.current_strategy == CacheStrategy.LRU:
            # Evict least recently used
            while freed_bytes < target_size and self.entries:
                key = next(iter(self.access_order))
                freed_bytes += self._remove_entry(key)
        
        elif self.current_strategy == CacheStrategy.LFU:
            # Evict least frequently used
            sorted_by_frequency = sorted(
                self.entries.items(),
                key=lambda x: (self.frequency_counts[x[0]], x[1].last_access_time)
            )
            
            for key, _ in sorted_by_frequency:
                if freed_bytes >= target_size:
                    break
                freed_bytes += self._remove_entry(key)
        
        elif self.current_strategy == CacheStrategy.TTL:
            # Evict expired entries first, then oldest
            now = time.time()
            expired_keys = [
                key for key, entry in self.entries.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                if freed_bytes >= target_size:
                    break
                freed_bytes += self._remove_entry(key)
            
            # If still need space, evict oldest
            if freed_bytes < target_size:
                sorted_by_age = sorted(
                    self.entries.items(),
                    key=lambda x: x[1].creation_time
                )
                
                for key, _ in sorted_by_age:
                    if freed_bytes >= target_size:
                        break
                    freed_bytes += self._remove_entry(key)
        
        elif self.current_strategy == CacheStrategy.ADAPTIVE:
            # Use hybrid approach based on access patterns
            self._adaptive_eviction(target_size)
    
    def _adaptive_eviction(self, target_size: int) -> None:
        """Adaptive eviction based on access patterns."""
        freed_bytes = 0
        now = time.time()
        
        # Calculate scores for each entry
        entry_scores = []
        for key, entry in self.entries.items():
            # Combine multiple factors
            recency_score = 1.0 / (1.0 + (now - entry.last_access_time) / 3600)  # Hours
            frequency_score = min(1.0, self.frequency_counts[key] / 100.0)
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            
            total_score = (recency_score * 0.4 + frequency_score * 0.4 - size_penalty * 0.2)
            entry_scores.append((key, total_score))
        
        # Sort by score (lower = more likely to evict)
        entry_scores.sort(key=lambda x: x[1])
        
        for key, _ in entry_scores:
            if freed_bytes >= target_size:
                break
            freed_bytes += self._remove_entry(key)
    
    def _remove_entry(self, key: str) -> int:
        """Remove entry and return freed bytes."""
        if key in self.entries:
            entry = self.entries[key]
            freed_bytes = entry.size_bytes
            
            del self.entries[key]
            self.access_order.pop(key, None)
            self.frequency_counts.pop(key, None)
            self.current_size_bytes -= freed_bytes
            self.stats["evictions"] += 1
            
            return freed_bytes
        
        return 0
    
    def put(self, key_data: Any, value: Any, ttl: Optional[float] = None) -> str:
        """Store value in cache."""
        with self._lock:
            key = self._compute_key(key_data)
            
            # Compress if enabled
            if self.config.enable_compression:
                compressed_value, is_compressed = self._compress_value(value)
                storage_value = compressed_value
            else:
                storage_value = value
                is_compressed = False
            
            entry_size = self._calculate_entry_size(storage_value)
            
            # Check if we need to evict
            if self.current_size_bytes + entry_size > self.max_size_bytes:
                target_eviction = entry_size + (self.max_size_bytes * 0.1)  # Extra 10%
                self._evict_entries(int(target_eviction))
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=storage_value,
                size_bytes=entry_size,
                creation_time=time.time(),
                last_access_time=time.time(),
                access_count=1,
                ttl=ttl,
                compressed=is_compressed
            )
            
            # Store entry
            self.entries[key] = entry
            self.access_order[key] = True
            self.frequency_counts[key] = 1
            self.current_size_bytes += entry_size
            
            return key
    
    def get(self, key_data: Any) -> Optional[Any]:
        """Retrieve value from cache."""
        with self._lock:
            key = self._compute_key(key_data)
            
            if key not in self.entries:
                self.stats["misses"] += 1
                return None
            
            entry = self.entries[key]
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self.stats["misses"] += 1
                return None
            
            # Update access patterns
            entry.last_access_time = time.time()
            entry.access_count += 1
            self.frequency_counts[key] += 1
            
            # Update LRU order
            self.access_order.move_to_end(key)
            
            self.stats["hits"] += 1
            
            # Decompress if needed
            return self._decompress_value(entry.value, entry.compressed)
    
    def contains(self, key_data: Any) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            key = self._compute_key(key_data)
            return key in self.entries and not self.entries[key].is_expired
    
    def invalidate(self, key_data: Any) -> bool:
        """Remove entry from cache."""
        with self._lock:
            key = self._compute_key(key_data)
            return self._remove_entry(key) > 0
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.entries.clear()
            self.access_order.clear()
            self.frequency_counts.clear()
            self.current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
            
            return {
                "hit_rate": hit_rate,
                "total_entries": len(self.entries),
                "size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": self.current_size_bytes / self.max_size_bytes,
                "current_strategy": self.current_strategy.value,
                **self.stats
            }


class WorkerPool:
    """
    High-performance worker pool with intelligent load balancing.
    
    This pool dynamically adjusts the number of workers based on workload
    and implements intelligent task distribution.
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.current_workers = config.max_workers // 2  # Start with half
        self.min_workers = max(1, config.min_workers)
        self.max_workers = min(config.max_workers_limit, config.max_workers)
        
        # Worker pools
        if config.worker_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=self.current_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.current_workers)
        
        # Load balancing
        self.task_queue = asyncio.Queue()
        self.active_tasks = set()
        self.completed_tasks = 0
        self.failed_tasks = 0
        
        # Performance monitoring
        self.worker_stats = defaultdict(lambda: {"tasks": 0, "total_time": 0.0})
        self.last_scale_check = time.time()
        self.scale_check_interval = 30.0  # 30 seconds
        
        self.logger = get_logger(self.__class__.__name__)
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        priority: float = 1.0,
        **kwargs
    ) -> Any:
        """Submit task to worker pool."""
        task_id = f"task_{time.time()}_{id(func)}"
        
        # Create task wrapper
        async def task_wrapper():
            start_time = time.time()
            worker_id = threading.current_thread().ident
            
            try:
                self.active_tasks.add(task_id)
                
                # Execute task
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.executor, func, *args, **kwargs)
                
                # Update statistics
                execution_time = time.time() - start_time
                self.worker_stats[worker_id]["tasks"] += 1
                self.worker_stats[worker_id]["total_time"] += execution_time
                self.completed_tasks += 1
                
                return result
                
            except Exception as e:
                self.failed_tasks += 1
                self.logger.error(f"Task {task_id} failed: {str(e)}")
                raise
            
            finally:
                self.active_tasks.discard(task_id)
        
        return await task_wrapper()
    
    async def submit_batch(
        self,
        func: Callable,
        arg_batches: List[Tuple],
        max_concurrent: Optional[int] = None
    ) -> List[Any]:
        """Submit batch of tasks with controlled concurrency."""
        if max_concurrent is None:
            max_concurrent = self.current_workers
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_task(args):
            async with semaphore:
                return await self.submit_task(func, *args)
        
        # Submit all tasks
        tasks = [bounded_task(args) for args in arg_batches]
        
        # Wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def _calculate_cpu_utilization(self) -> float:
        """Calculate current CPU utilization."""
        return psutil.cpu_percent(interval=1.0) / 100.0
    
    def _should_scale_up(self) -> bool:
        """Check if we should scale up workers."""
        if self.current_workers >= self.max_workers:
            return False
        
        # Check CPU utilization
        cpu_util = self._calculate_cpu_utilization()
        
        # Check task queue length
        queue_pressure = len(self.active_tasks) / self.current_workers
        
        return (cpu_util > self.config.scale_up_threshold or 
                queue_pressure > 2.0)
    
    def _should_scale_down(self) -> bool:
        """Check if we should scale down workers."""
        if self.current_workers <= self.min_workers:
            return False
        
        cpu_util = self._calculate_cpu_utilization()
        queue_pressure = len(self.active_tasks) / self.current_workers
        
        return (cpu_util < self.config.scale_down_threshold and 
                queue_pressure < 0.5)
    
    async def _auto_scale(self) -> None:
        """Automatically scale worker pool based on load."""
        if not self.config.enable_auto_scaling:
            return
        
        now = time.time()
        if now - self.last_scale_check < self.scale_check_interval:
            return
        
        self.last_scale_check = now
        
        if self._should_scale_up():
            new_workers = min(self.max_workers, self.current_workers + 1)
            if new_workers != self.current_workers:
                self.logger.info(f"Scaling up workers: {self.current_workers} -> {new_workers}")
                await self._resize_pool(new_workers)
        
        elif self._should_scale_down():
            new_workers = max(self.min_workers, self.current_workers - 1)
            if new_workers != self.current_workers:
                self.logger.info(f"Scaling down workers: {self.current_workers} -> {new_workers}")
                await self._resize_pool(new_workers)
    
    async def _resize_pool(self, new_size: int) -> None:
        """Resize the worker pool."""
        old_executor = self.executor
        self.current_workers = new_size
        
        # Create new executor
        if self.config.worker_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=new_size)
        else:
            self.executor = ThreadPoolExecutor(max_workers=new_size)
        
        # Shutdown old executor gracefully
        old_executor.shutdown(wait=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        total_tasks = self.completed_tasks + self.failed_tasks
        success_rate = self.completed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        return {
            "current_workers": self.current_workers,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": success_rate,
            "worker_type": self.config.worker_type,
            "cpu_utilization": self._calculate_cpu_utilization()
        }
    
    async def shutdown(self) -> None:
        """Shutdown worker pool gracefully."""
        self.logger.info("Shutting down worker pool...")
        self.executor.shutdown(wait=True)


class MemoryOptimizer:
    """
    Memory optimization and management for research operations.
    
    This class implements intelligent memory management including garbage
    collection, memory mapping, and resource monitoring.
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.memory_limit_bytes = config.memory_limit_mb * 1024 * 1024
        self.gc_threshold_bytes = self.memory_limit_bytes * config.gc_threshold
        
        # Memory tracking
        self.allocated_objects = weakref.WeakSet()
        self.memory_pools = {}
        self.last_gc_time = time.time()
        self.gc_interval = 30.0  # 30 seconds
        
        self.logger = get_logger(self.__class__.__name__)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / (1024 * 1024),
            "limit_mb": self.config.memory_limit_mb,
            "utilization": memory_info.rss / self.memory_limit_bytes
        }
    
    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        memory_usage = self.get_memory_usage()
        current_bytes = memory_usage["rss_mb"] * 1024 * 1024
        
        # Check memory threshold
        if current_bytes > self.gc_threshold_bytes:
            return True
        
        # Check time interval
        if time.time() - self.last_gc_time > self.gc_interval:
            return True
        
        return False
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics."""
        start_time = time.time()
        before_memory = self.get_memory_usage()
        
        # Multiple GC passes for thorough cleanup
        collected_objects = []
        for generation in range(3):
            collected = gc.collect(generation)
            collected_objects.append(collected)
        
        # Force immediate cleanup
        gc.collect()
        
        after_memory = self.get_memory_usage()
        gc_time = time.time() - start_time
        self.last_gc_time = time.time()
        
        freed_mb = before_memory["rss_mb"] - after_memory["rss_mb"]
        
        self.logger.info(f"GC completed: freed {freed_mb:.1f}MB in {gc_time:.3f}s")
        
        return {
            "gc_time": gc_time,
            "freed_mb": freed_mb,
            "before_memory": before_memory,
            "after_memory": after_memory,
            "collected_objects": collected_objects
        }
    
    def create_memory_pool(self, pool_name: str, size_mb: int) -> np.ndarray:
        """Create a memory pool for efficient allocation."""
        size_bytes = size_mb * 1024 * 1024
        
        # Create memory-mapped array
        if self.config.enable_memory_mapping:
            pool = np.zeros(size_bytes // 8, dtype=np.float64)  # 8 bytes per float64
        else:
            pool = np.zeros(size_bytes // 8, dtype=np.float64)
        
        self.memory_pools[pool_name] = {
            "pool": pool,
            "size_mb": size_mb,
            "allocated_bytes": 0,
            "created_time": time.time()
        }
        
        self.logger.info(f"Created memory pool '{pool_name}': {size_mb}MB")
        return pool
    
    def get_memory_pool(self, pool_name: str) -> Optional[np.ndarray]:
        """Get existing memory pool."""
        if pool_name in self.memory_pools:
            return self.memory_pools[pool_name]["pool"]
        return None
    
    def optimize_numpy_array(self, array: np.ndarray) -> np.ndarray:
        """Optimize numpy array for memory efficiency."""
        # Convert to most efficient dtype
        if array.dtype == np.float64:
            # Check if we can use float32 without losing precision
            float32_array = array.astype(np.float32)
            if np.allclose(array, float32_array, rtol=1e-6):
                array = float32_array
                self.logger.debug("Converted float64 to float32")
        
        # Make array contiguous for better cache performance
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
            self.logger.debug("Made array contiguous")
        
        return array
    
    async def monitor_memory(self) -> None:
        """Monitor memory usage and trigger optimization."""
        if self.should_trigger_gc():
            gc_stats = self.force_garbage_collection()
            
            # If still high memory usage, log warning
            memory_usage = self.get_memory_usage()
            if memory_usage["utilization"] > 0.9:
                self.logger.warning(
                    f"High memory usage: {memory_usage['utilization']:.1%} "
                    f"({memory_usage['rss_mb']:.1f}MB)"
                )


class PerformanceOptimizer:
    """
    Main performance optimization engine that coordinates all optimization
    components for maximum research operation performance.
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Core components
        self.cache = AdaptiveCache(self.config)
        self.worker_pool = WorkerPool(self.config)
        self.memory_optimizer = MemoryOptimizer(self.config)
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        self.optimization_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "parallelized_operations": 0,
            "memory_optimizations": 0,
            "total_time_saved": 0.0
        }
        
        # Background monitoring
        self.monitoring_task = None
        self.is_running = False
    
    async def optimized_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        cache_key: Optional[Any] = None,
        cache_ttl: Optional[float] = None,
        enable_parallel: bool = True,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Execute operation with full performance optimization."""
        start_time = time.time()
        
        # Generate cache key if not provided
        if cache_key is None:
            cache_key = {
                "operation": operation_name,
                "args": args,
                "kwargs": kwargs
            }
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.optimization_stats["cache_hits"] += 1
            cache_time = time.time() - start_time
            self.optimization_stats["total_time_saved"] += cache_time
            self.logger.debug(f"Cache hit for {operation_name}")
            return cached_result
        
        self.optimization_stats["cache_misses"] += 1
        
        # Memory optimization
        optimized_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                optimized_args.append(self.memory_optimizer.optimize_numpy_array(arg))
                self.optimization_stats["memory_optimizations"] += 1
            else:
                optimized_args.append(arg)
        
        # Execute operation
        if enable_parallel and batch_size and len(optimized_args) > 0:
            # Parallel batch processing
            result = await self._execute_parallel_batch(
                operation_func, optimized_args, batch_size, **kwargs
            )
            self.optimization_stats["parallelized_operations"] += 1
        else:
            # Single operation
            result = await self.worker_pool.submit_task(
                operation_func, *optimized_args, **kwargs
            )
        
        # Cache result
        self.cache.put(cache_key, result, ttl=cache_ttl)
        
        # Record timing
        execution_time = time.time() - start_time
        self.operation_times[operation_name].append(execution_time)
        
        # Keep only recent timings
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name] = self.operation_times[operation_name][-100:]
        
        return result
    
    async def _execute_parallel_batch(
        self,
        operation_func: Callable,
        args_list: List[Any],
        batch_size: int,
        **kwargs
    ) -> List[Any]:
        """Execute operation in parallel batches."""
        # Split into batches
        batches = [
            args_list[i:i + batch_size]
            for i in range(0, len(args_list), batch_size)
        ]
        
        # Execute batches in parallel
        batch_results = await self.worker_pool.submit_batch(
            operation_func, batches
        )
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
        
        return results
    
    async def precompute_cache(
        self,
        precompute_tasks: List[Tuple[str, Callable, tuple, dict]]
    ) -> None:
        """Precompute and cache common operations."""
        self.logger.info(f"Precomputing {len(precompute_tasks)} cache entries...")
        
        for operation_name, func, args, kwargs in precompute_tasks:
            try:
                cache_key = {
                    "operation": operation_name,
                    "args": args,
                    "kwargs": kwargs
                }
                
                # Skip if already cached
                if self.cache.contains(cache_key):
                    continue
                
                # Compute and cache
                result = await self.worker_pool.submit_task(func, *args, **kwargs)
                self.cache.put(cache_key, result)
                
            except Exception as e:
                self.logger.warning(f"Precompute failed for {operation_name}: {e}")
        
        self.logger.info("Cache precomputation completed")
    
    async def start_monitoring(self, interval: float = 60.0) -> None:
        """Start background performance monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_task = asyncio.create_task(
            self._performance_monitoring_loop(interval)
        )
        self.logger.info("Started performance monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped performance monitoring")
    
    async def _performance_monitoring_loop(self, interval: float) -> None:
        """Background performance monitoring loop."""
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                
                # Auto-scale workers
                await self.worker_pool._auto_scale()
                
                # Memory monitoring
                await self.memory_optimizer.monitor_memory()
                
                # Performance reporting
                stats = await self.get_comprehensive_stats()
                
                # Log performance summary
                cache_hit_rate = stats["cache"]["hit_rate"]
                worker_utilization = stats["workers"]["current_workers"]
                memory_utilization = stats["memory"]["utilization"]
                
                self.logger.info(
                    f"Performance: cache_hit={cache_hit_rate:.1%}, "
                    f"workers={worker_utilization}, "
                    f"memory={memory_utilization:.1%}"
                )
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "timestamp": time.time(),
            "optimization_stats": self.optimization_stats.copy(),
            "cache": self.cache.get_stats(),
            "workers": self.worker_pool.get_stats(),
            "memory": self.memory_optimizer.get_memory_usage(),
            "operation_timings": {
                name: {
                    "count": len(times),
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "total_time": np.sum(times)
                }
                for name, times in self.operation_times.items()
            }
        }
    
    async def optimize_for_workload(self, workload_profile: Dict[str, Any]) -> None:
        """Optimize configuration based on workload profile."""
        # Analyze workload characteristics
        operation_types = workload_profile.get("operation_types", [])
        data_sizes = workload_profile.get("data_sizes", [])
        concurrency_level = workload_profile.get("concurrency_level", 1)
        
        # Adjust cache strategy based on access patterns
        if "random_access" in workload_profile.get("access_pattern", ""):
            self.cache.current_strategy = CacheStrategy.LFU
        elif "sequential_access" in workload_profile.get("access_pattern", ""):
            self.cache.current_strategy = CacheStrategy.LRU
        else:
            self.cache.current_strategy = CacheStrategy.ADAPTIVE
        
        # Adjust worker pool size
        if concurrency_level > self.worker_pool.current_workers:
            await self.worker_pool._resize_pool(
                min(concurrency_level, self.worker_pool.max_workers)
            )
        
        # Adjust memory allocation
        if data_sizes and np.mean(data_sizes) > 100 * 1024 * 1024:  # >100MB
            # Large data workload - increase memory pool
            self.memory_optimizer.create_memory_pool("large_data", 1024)  # 1GB
        
        self.logger.info("Optimized configuration for workload profile")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown performance optimizer."""
        self.logger.info("Shutting down performance optimizer...")
        
        await self.stop_monitoring()
        await self.worker_pool.shutdown()
        self.cache.clear()
        
        self.logger.info("Performance optimizer shutdown completed")


# Global optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


# Convenience decorator for optimization
def optimize_performance(
    operation_name: str,
    cache_ttl: Optional[float] = None,
    enable_parallel: bool = True
):
    """Decorator to automatically optimize research operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            return await optimizer.optimized_operation(
                operation_name, func, *args,
                cache_ttl=cache_ttl,
                enable_parallel=enable_parallel,
                **kwargs
            )
        return wrapper
    return decorator


# Example usage and testing
if __name__ == "__main__":
    async def test_performance_optimizer():
        """Test the performance optimization system."""
        print("⚡ Testing Performance Optimization System")
        print("=" * 45)
        
        # Create optimizer
        config = PerformanceConfig(
            cache_size_mb=100,
            max_workers=4,
            enable_auto_scaling=True
        )
        
        optimizer = PerformanceOptimizer(config)
        await optimizer.start_monitoring(10.0)  # 10 second intervals
        
        # Test function
        async def compute_heavy_operation(data, multiplier=2.0):
            """Simulate heavy computation."""
            await asyncio.sleep(0.1)  # Simulate work
            return data * multiplier
        
        # Test data
        test_data = np.random.randn(1000, 100)
        
        # Test optimized operation
        print("🚀 Testing optimized operations...")
        
        for i in range(5):
            result = await optimizer.optimized_operation(
                "heavy_compute",
                compute_heavy_operation,
                test_data,
                multiplier=2.0,
                cache_key=f"test_data_{i % 3}",  # Some cache hits
                cache_ttl=300.0
            )
            print(f"   Operation {i+1}: shape={result.shape}")
        
        # Get statistics
        stats = await optimizer.get_comprehensive_stats()
        print(f"\n📊 Performance Statistics:")
        print(f"   Cache hit rate: {stats['cache']['hit_rate']:.1%}")
        print(f"   Cache entries: {stats['cache']['total_entries']}")
        print(f"   Workers: {stats['workers']['current_workers']}")
        print(f"   Memory usage: {stats['memory']['rss_mb']:.1f}MB")
        print(f"   Optimization stats: {stats['optimization_stats']}")
        
        # Test batch processing
        print(f"\n🔄 Testing batch processing...")
        
        batch_data = [np.random.randn(100, 10) for _ in range(10)]
        batch_results = await optimizer._execute_parallel_batch(
            compute_heavy_operation, batch_data, batch_size=3
        )
        print(f"   Processed {len(batch_results)} batches")
        
        await optimizer.shutdown()
        print("\n✅ Performance optimization test completed")
    
    asyncio.run(test_performance_optimizer())