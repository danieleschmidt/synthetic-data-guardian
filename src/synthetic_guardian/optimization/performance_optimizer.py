"""
Performance optimization and scaling infrastructure for Synthetic Data Guardian
"""

import asyncio
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import queue
import weakref
import gc

from ..utils.logger import get_logger
from ..monitoring.health_monitor import get_health_monitor


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    MEMORY_OPTIMIZED = "memory_optimized"
    CPU_OPTIMIZED = "cpu_optimized"
    IO_OPTIMIZED = "io_optimized"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class ResourceType(Enum):
    """System resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_used: float
    cpu_percent: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceConstraints:
    """Resource usage constraints."""
    max_memory_mb: int = 8192
    max_cpu_cores: int = 4
    max_concurrent_tasks: int = 10
    max_batch_size: int = 10000
    timeout_seconds: int = 3600


class PerformanceOptimizer:
    """
    Advanced performance optimization system for scaling synthetic data generation.
    """
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self.logger = get_logger(self.__class__.__name__)
        self.health_monitor = get_health_monitor()
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, Dict[str, Any]] = {}
        
        # Resource management
        self.resource_constraints = ResourceConstraints()
        self.resource_usage = {
            ResourceType.CPU: 0.0,
            ResourceType.MEMORY: 0.0,
            ResourceType.DISK: 0.0,
            ResourceType.NETWORK: 0.0
        }
        
        # Execution pools
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Optimization settings
        self._configure_for_strategy()
        
        # Performance monitoring
        self._start_resource_monitoring()
    
    def _configure_for_strategy(self):
        """Configure optimizer based on selected strategy."""
        if self.strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
            self.resource_constraints.max_memory_mb = 4096
            self.resource_constraints.max_concurrent_tasks = 5
            self.resource_constraints.max_batch_size = 5000
            self._enable_memory_optimization()
            
        elif self.strategy == OptimizationStrategy.CPU_OPTIMIZED:
            self.resource_constraints.max_cpu_cores = multiprocessing.cpu_count()
            self.resource_constraints.max_concurrent_tasks = 20
            self.resource_constraints.max_batch_size = 20000
            self._enable_cpu_optimization()
            
        elif self.strategy == OptimizationStrategy.IO_OPTIMIZED:
            self.resource_constraints.max_concurrent_tasks = 50
            self.resource_constraints.timeout_seconds = 7200
            self._enable_io_optimization()
            
        elif self.strategy == OptimizationStrategy.AGGRESSIVE:
            self.resource_constraints.max_memory_mb = 16384
            self.resource_constraints.max_cpu_cores = multiprocessing.cpu_count() * 2
            self.resource_constraints.max_concurrent_tasks = 100
            self.resource_constraints.max_batch_size = 50000
            self._enable_aggressive_optimization()
            
        else:  # BALANCED
            self.resource_constraints.max_memory_mb = 8192
            self.resource_constraints.max_cpu_cores = min(8, multiprocessing.cpu_count())
            self.resource_constraints.max_concurrent_tasks = 10
            self.resource_constraints.max_batch_size = 10000
    
    def _enable_memory_optimization(self):
        """Enable memory-specific optimizations."""
        self.logger.info("Enabling memory optimization features")
        
        # Enable garbage collection optimization
        gc.set_threshold(100, 10, 10)
        
        # Use smaller thread pool to reduce memory overhead
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
    
    def _enable_cpu_optimization(self):
        """Enable CPU-specific optimizations."""
        self.logger.info("Enabling CPU optimization features")
        
        # Use process pool for CPU-intensive tasks
        cpu_count = multiprocessing.cpu_count()
        self.process_pool = ProcessPoolExecutor(max_workers=cpu_count)
        self.thread_pool = ThreadPoolExecutor(max_workers=cpu_count * 2)
    
    def _enable_io_optimization(self):
        """Enable I/O-specific optimizations."""
        self.logger.info("Enabling I/O optimization features")
        
        # Large thread pool for I/O-bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=50)
    
    def _enable_aggressive_optimization(self):
        """Enable aggressive optimization for maximum performance."""
        self.logger.info("Enabling aggressive optimization features")
        
        # Maximum resource utilization
        cpu_count = multiprocessing.cpu_count()
        self.process_pool = ProcessPoolExecutor(max_workers=cpu_count)
        self.thread_pool = ThreadPoolExecutor(max_workers=cpu_count * 4)
        
        # Optimize garbage collection for performance
        gc.set_threshold(1000, 100, 100)
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring."""
        def monitor_resources():
            while True:
                try:
                    import psutil
                    
                    # Update resource usage
                    self.resource_usage[ResourceType.CPU] = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    self.resource_usage[ResourceType.MEMORY] = memory.percent
                    disk = psutil.disk_usage('/')
                    self.resource_usage[ResourceType.DISK] = (disk.used / disk.total) * 100
                    
                    # Check for resource pressure
                    self._check_resource_pressure()
                    
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    time.sleep(30)  # Wait longer on error
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def _check_resource_pressure(self):
        """Check for resource pressure and adjust accordingly."""
        # Memory pressure
        if self.resource_usage[ResourceType.MEMORY] > 85:
            self.logger.warning(f"Memory pressure detected: {self.resource_usage[ResourceType.MEMORY]:.1f}%")
            self._handle_memory_pressure()
        
        # CPU pressure
        if self.resource_usage[ResourceType.CPU] > 90:
            self.logger.warning(f"CPU pressure detected: {self.resource_usage[ResourceType.CPU]:.1f}%")
            self._handle_cpu_pressure()
        
        # Disk pressure
        if self.resource_usage[ResourceType.DISK] > 90:
            self.logger.warning(f"Disk pressure detected: {self.resource_usage[ResourceType.DISK]:.1f}%")
            self._handle_disk_pressure()
    
    def _handle_memory_pressure(self):
        """Handle high memory usage."""
        # Force garbage collection
        collected = gc.collect()
        self.logger.info(f"Forced garbage collection, freed {collected} objects")
        
        # Reduce batch sizes temporarily
        if hasattr(self, '_original_max_batch_size'):
            self.resource_constraints.max_batch_size = min(
                self.resource_constraints.max_batch_size,
                self._original_max_batch_size // 2
            )
        else:
            self._original_max_batch_size = self.resource_constraints.max_batch_size
            self.resource_constraints.max_batch_size //= 2
    
    def _handle_cpu_pressure(self):
        """Handle high CPU usage."""
        # Reduce concurrent tasks temporarily
        if hasattr(self, '_original_max_concurrent'):
            self.resource_constraints.max_concurrent_tasks = max(
                1,
                self.resource_constraints.max_concurrent_tasks - 1
            )
        else:
            self._original_max_concurrent = self.resource_constraints.max_concurrent_tasks
            self.resource_constraints.max_concurrent_tasks = max(1, self.resource_constraints.max_concurrent_tasks // 2)
    
    def _handle_disk_pressure(self):
        """Handle high disk usage."""
        self.logger.warning("High disk usage detected - consider cleanup")
        
        # Clean up old metrics
        cutoff_time = time.time() - 3600  # Keep only last hour
        self.metrics_history = [
            m for m in self.metrics_history
            if m.start_time > cutoff_time
        ]
    
    async def optimize_batch_operation(
        self,
        operation_func: Callable,
        data_batches: List[Any],
        operation_name: str = "batch_operation",
        **kwargs
    ) -> List[Any]:
        """
        Optimize batch processing with intelligent scaling.
        
        Args:
            operation_func: Function to execute on each batch
            data_batches: List of data batches to process
            operation_name: Name for tracking/logging
            **kwargs: Additional arguments for operation_func
            
        Returns:
            List of results from batch operations
        """
        start_time = time.time()
        
        # Determine optimal batch processing strategy
        optimal_concurrency = self._calculate_optimal_concurrency(len(data_batches))
        
        self.logger.info(
            f"Starting optimized batch operation '{operation_name}' "
            f"with {len(data_batches)} batches, concurrency={optimal_concurrency}"
        )
        
        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(optimal_concurrency)
        
        async def process_batch_with_semaphore(batch, batch_index):
            async with semaphore:
                return await self._process_single_batch(
                    operation_func, batch, batch_index, operation_name, **kwargs
                )
        
        # Execute all batches concurrently with controlled concurrency
        tasks = [
            process_batch_with_semaphore(batch, i)
            for i, batch in enumerate(data_batches)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle any exceptions
        successful_results = []
        failed_batches = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch {i} failed: {result}")
                failed_batches.append((i, result))
            else:
                successful_results.append(result)
        
        # Record performance metrics
        total_time = time.time() - start_time
        self._record_performance_metrics(
            operation_name, start_time, total_time, 
            len(successful_results), len(failed_batches)
        )
        
        self.logger.info(
            f"Batch operation '{operation_name}' completed in {total_time:.2f}s "
            f"({len(successful_results)} successful, {len(failed_batches)} failed)"
        )
        
        return successful_results
    
    async def _process_single_batch(
        self, operation_func: Callable, batch: Any, batch_index: int, 
        operation_name: str, **kwargs
    ) -> Any:
        """Process a single batch with performance tracking."""
        batch_start = time.time()
        
        try:
            # Choose execution method based on operation type
            if self._should_use_process_pool(operation_func, batch):
                result = await self._execute_in_process_pool(operation_func, batch, **kwargs)
            elif self._should_use_thread_pool(operation_func, batch):
                result = await self._execute_in_thread_pool(operation_func, batch, **kwargs)
            else:
                # Direct async execution
                if asyncio.iscoroutinefunction(operation_func):
                    result = await operation_func(batch, **kwargs)
                else:
                    result = operation_func(batch, **kwargs)
            
            duration = time.time() - batch_start
            self.logger.debug(f"Batch {batch_index} completed in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - batch_start
            self.logger.error(f"Batch {batch_index} failed after {duration:.3f}s: {e}")
            raise
    
    def _calculate_optimal_concurrency(self, num_batches: int) -> int:
        """Calculate optimal concurrency level based on system resources."""
        # Base concurrency on system resources and constraints
        base_concurrency = min(
            self.resource_constraints.max_concurrent_tasks,
            multiprocessing.cpu_count() * 2,
            num_batches
        )
        
        # Adjust based on current resource usage
        memory_factor = max(0.1, (100 - self.resource_usage[ResourceType.MEMORY]) / 100)
        cpu_factor = max(0.1, (100 - self.resource_usage[ResourceType.CPU]) / 100)
        
        optimal_concurrency = int(base_concurrency * min(memory_factor, cpu_factor))
        
        return max(1, optimal_concurrency)
    
    def _should_use_process_pool(self, operation_func: Callable, batch: Any) -> bool:
        """Determine if operation should use process pool."""
        # Use process pool for CPU-intensive operations
        if self.strategy == OptimizationStrategy.CPU_OPTIMIZED and self.process_pool:
            return True
        
        # Use process pool for large batches that can benefit from parallelism
        if hasattr(batch, '__len__') and len(batch) > 1000 and self.process_pool:
            return True
        
        return False
    
    def _should_use_thread_pool(self, operation_func: Callable, batch: Any) -> bool:
        """Determine if operation should use thread pool."""
        # Use thread pool for I/O-bound operations
        if not asyncio.iscoroutinefunction(operation_func) and self.thread_pool:
            return True
        
        return False
    
    async def _execute_in_process_pool(self, operation_func: Callable, batch: Any, **kwargs) -> Any:
        """Execute operation in process pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, operation_func, batch, **kwargs)
    
    async def _execute_in_thread_pool(self, operation_func: Callable, batch: Any, **kwargs) -> Any:
        """Execute operation in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, operation_func, batch, **kwargs)
    
    def _record_performance_metrics(
        self, operation_name: str, start_time: float, duration: float,
        successful_batches: int, failed_batches: int
    ):
        """Record performance metrics for analysis."""
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
        except ImportError:
            memory_info = None
            cpu_percent = 0.0
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=start_time + duration,
            duration=duration,
            memory_used=memory_info.used / (1024 * 1024) if memory_info else 0.0,
            cpu_percent=cpu_percent,
            success=failed_batches == 0,
            metadata={
                'successful_batches': successful_batches,
                'failed_batches': failed_batches,
                'strategy': self.strategy.value,
                'concurrency_used': self.resource_constraints.max_concurrent_tasks
            }
        )
        
        self.metrics_history.append(metrics)
        
        # Update operation statistics
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
                'best_duration': float('inf'),
                'worst_duration': 0.0
            }
        
        stats = self.operation_stats[operation_name]
        stats['total_executions'] += 1
        stats['total_duration'] += duration
        
        if metrics.success:
            stats['successful_executions'] += 1
        else:
            stats['failed_executions'] += 1
        
        stats['avg_duration'] = stats['total_duration'] / stats['total_executions']
        stats['best_duration'] = min(stats['best_duration'], duration)
        stats['worst_duration'] = max(stats['worst_duration'], duration)
        
        # Keep metrics history manageable
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]  # Keep recent 500
    
    def get_performance_analysis(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive performance analysis."""
        if operation_name:
            return {
                'operation_stats': self.operation_stats.get(operation_name, {}),
                'recent_metrics': [
                    m for m in self.metrics_history[-50:]
                    if m.operation_name == operation_name
                ]
            }
        
        return {
            'all_operation_stats': self.operation_stats.copy(),
            'resource_usage': self.resource_usage.copy(),
            'resource_constraints': {
                'max_memory_mb': self.resource_constraints.max_memory_mb,
                'max_cpu_cores': self.resource_constraints.max_cpu_cores,
                'max_concurrent_tasks': self.resource_constraints.max_concurrent_tasks,
                'max_batch_size': self.resource_constraints.max_batch_size
            },
            'optimization_strategy': self.strategy.value,
            'total_operations': len(self.metrics_history),
            'successful_operations': sum(1 for m in self.metrics_history if m.success),
            'average_operation_time': sum(m.duration for m in self.metrics_history) / len(self.metrics_history) if self.metrics_history else 0
        }
    
    def optimize_memory_usage(self):
        """Trigger memory optimization procedures."""
        self.logger.info("Optimizing memory usage...")
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear old metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-50:]
        
        # Compact operation stats
        for operation_name, stats in self.operation_stats.items():
            if stats['total_executions'] == 0:
                del self.operation_stats[operation_name]
        
        self.logger.info(f"Memory optimization completed, collected {collected} objects")
    
    def auto_scale_resources(self, target_load: float = 0.8) -> Dict[str, Any]:
        """Auto-scale resources based on current load and target."""
        scaling_actions = {
            'memory_actions': [],
            'cpu_actions': [],
            'concurrency_actions': []
        }
        
        # Memory scaling
        current_memory_load = self.resource_usage[ResourceType.MEMORY] / 100
        if current_memory_load > target_load:
            if self.resource_constraints.max_memory_mb > 2048:
                self.resource_constraints.max_memory_mb = int(self.resource_constraints.max_memory_mb * 0.9)
                scaling_actions['memory_actions'].append(f"Reduced memory limit to {self.resource_constraints.max_memory_mb}MB")
        elif current_memory_load < target_load * 0.5:
            if self.resource_constraints.max_memory_mb < 16384:
                self.resource_constraints.max_memory_mb = int(self.resource_constraints.max_memory_mb * 1.1)
                scaling_actions['memory_actions'].append(f"Increased memory limit to {self.resource_constraints.max_memory_mb}MB")
        
        # CPU scaling
        current_cpu_load = self.resource_usage[ResourceType.CPU] / 100
        if current_cpu_load > target_load:
            if self.resource_constraints.max_concurrent_tasks > 1:
                self.resource_constraints.max_concurrent_tasks -= 1
                scaling_actions['cpu_actions'].append(f"Reduced concurrency to {self.resource_constraints.max_concurrent_tasks}")
        elif current_cpu_load < target_load * 0.5:
            max_possible = multiprocessing.cpu_count() * 2
            if self.resource_constraints.max_concurrent_tasks < max_possible:
                self.resource_constraints.max_concurrent_tasks += 1
                scaling_actions['cpu_actions'].append(f"Increased concurrency to {self.resource_constraints.max_concurrent_tasks}")
        
        self.logger.info(f"Auto-scaling completed: {scaling_actions}")
        return scaling_actions
    
    async def cleanup(self):
        """Clean up optimizer resources."""
        self.logger.info("Cleaning up performance optimizer...")
        
        # Shutdown executor pools
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        # Clear metrics
        self.metrics_history.clear()
        self.operation_stats.clear()
        
        self.logger.info("Performance optimizer cleanup completed")


# Global optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def optimize_performance(strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
    """Decorator to optimize function performance."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            
            # If function needs batch processing, use optimizer
            if 'batches' in kwargs or (args and hasattr(args[0], '__iter__') and len(args[0]) > 1):
                return await optimizer.optimize_batch_operation(func, *args, **kwargs)
            else:
                # Regular function execution with performance tracking
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Record metrics
                    optimizer._record_performance_metrics(
                        func.__name__, start_time, duration, 1, 0
                    )
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    optimizer._record_performance_metrics(
                        func.__name__, start_time, duration, 0, 1
                    )
                    raise
        
        return wrapper
    return decorator