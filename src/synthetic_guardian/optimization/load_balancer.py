"""
Advanced Load Balancer - Intelligent request distribution and auto-scaling
"""

import asyncio
import time
import random
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import heapq
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"


class WorkerStatus(Enum):
    """Worker status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class WorkerMetrics:
    """Metrics for a worker instance."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_connections: int = 0
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    queue_length: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def get_load_score(self) -> float:
        """Calculate composite load score (lower is better)."""
        # Weighted combination of metrics
        score = (
            self.cpu_percent * 0.3 +
            self.memory_percent * 0.3 +
            (self.active_connections / 100) * 0.2 +
            (self.average_response_time * 1000) * 0.1 +
            self.error_rate * 100 * 0.1
        )
        return score


@dataclass
class WorkerNode:
    """Represents a worker node in the load balancer."""
    worker_id: str
    weight: float = 1.0
    max_connections: int = 100
    status: WorkerStatus = WorkerStatus.HEALTHY
    metrics: WorkerMetrics = field(default_factory=WorkerMetrics)
    created_at: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    
    # Request tracking
    current_connections: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    request_count: int = 0
    error_count: int = 0
    
    def is_available(self) -> bool:
        """Check if worker is available for requests."""
        return (
            self.status in [WorkerStatus.HEALTHY, WorkerStatus.DEGRADED] and
            self.current_connections < self.max_connections
        )
    
    def update_metrics(self, response_time: float, success: bool) -> None:
        """Update worker metrics after request completion."""
        self.response_times.append(response_time)
        self.request_count += 1
        
        if not success:
            self.error_count += 1
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        # Update computed metrics
        if self.response_times:
            self.metrics.average_response_time = statistics.mean(self.response_times)
        
        if self.request_count > 0:
            self.metrics.error_rate = self.error_count / self.request_count
        
        # Update status based on metrics
        self._update_status()
    
    def _update_status(self) -> None:
        """Update worker status based on metrics."""
        if self.consecutive_failures >= 5:
            self.status = WorkerStatus.FAILED
        elif self.metrics.error_rate > 0.1:  # 10% error rate
            self.status = WorkerStatus.DEGRADED
        elif (self.metrics.cpu_percent > 90 or 
              self.metrics.memory_percent > 90 or
              self.current_connections >= self.max_connections):
            self.status = WorkerStatus.OVERLOADED
        else:
            self.status = WorkerStatus.HEALTHY


class RequestQueue:
    """Priority queue for managing requests."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: List[Tuple[float, int, Any]] = []  # (priority, sequence, request)
        self.sequence = 0
        self._lock = threading.RLock()
        
    def enqueue(self, request: Any, priority: float = 1.0) -> bool:
        """Add request to queue."""
        with self._lock:
            if len(self.queue) >= self.max_size:
                return False
            
            heapq.heappush(self.queue, (priority, self.sequence, request))
            self.sequence += 1
            return True
    
    def dequeue(self) -> Optional[Any]:
        """Remove and return highest priority request."""
        with self._lock:
            if not self.queue:
                return None
            
            _, _, request = heapq.heappop(self.queue)
            return request
    
    def size(self) -> int:
        """Get current queue size."""
        with self._lock:
            return len(self.queue)
    
    def is_full(self) -> bool:
        """Check if queue is full."""
        with self._lock:
            return len(self.queue) >= self.max_size


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time < self.timeout:
                    raise Exception("Circuit breaker is open")
                else:
                    self.state = "half_open"
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class LoadBalancer:
    """Advanced load balancer with auto-scaling and intelligent routing."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE,
                 auto_scaling_enabled: bool = True):
        self.strategy = strategy
        self.auto_scaling_enabled = auto_scaling_enabled
        
        self.workers: Dict[str, WorkerNode] = {}
        self.request_queue = RequestQueue()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Load balancing state
        self.round_robin_index = 0
        self.request_count = 0
        
        # Auto-scaling parameters
        self.min_workers = 1
        self.max_workers = 10
        self.scale_up_threshold = 0.8  # 80% average load
        self.scale_down_threshold = 0.3  # 30% average load
        self.scale_check_interval = 30  # seconds
        
        # Thread pool for worker management
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self) -> None:
        """Start background monitoring and scaling tasks."""
        if self.auto_scaling_enabled:
            self.executor.submit(self._auto_scaling_loop)
        self.executor.submit(self._health_check_loop)
        self.executor.submit(self._metrics_collection_loop)
    
    def add_worker(self, worker_id: str, weight: float = 1.0, 
                   max_connections: int = 100) -> None:
        """Add a new worker to the load balancer."""
        with self._lock:
            worker = WorkerNode(
                worker_id=worker_id,
                weight=weight,
                max_connections=max_connections
            )
            
            self.workers[worker_id] = worker
            self.circuit_breakers[worker_id] = CircuitBreaker()
            
            self.logger.info(f"Added worker {worker_id} with weight {weight}")
    
    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from the load balancer."""
        with self._lock:
            if worker_id in self.workers:
                # Wait for current connections to complete
                worker = self.workers[worker_id]
                worker.status = WorkerStatus.MAINTENANCE
                
                # TODO: Implement graceful shutdown with connection draining
                
                del self.workers[worker_id]
                del self.circuit_breakers[worker_id]
                
                self.logger.info(f"Removed worker {worker_id}")
    
    def get_next_worker(self) -> Optional[WorkerNode]:
        """Get next worker based on load balancing strategy."""
        with self._lock:
            available_workers = [
                worker for worker in self.workers.values()
                if worker.is_available()
            ]
            
            if not available_workers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_workers)
            
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(available_workers)
            
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(available_workers)
            
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_selection(available_workers)
            
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return self._resource_based_selection(available_workers)
            
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
                return self._adaptive_selection(available_workers)
            
            else:
                return random.choice(available_workers)
    
    def _round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin worker selection."""
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return worker
    
    def _least_connections_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least active connections."""
        return min(workers, key=lambda w: w.current_connections)
    
    def _weighted_round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin selection."""
        # Create weighted list
        weighted_workers = []
        for worker in workers:
            weighted_workers.extend([worker] * int(worker.weight * 10))
        
        if not weighted_workers:
            return random.choice(workers)
        
        worker = weighted_workers[self.round_robin_index % len(weighted_workers)]
        self.round_robin_index += 1
        return worker
    
    def _least_response_time_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with lowest average response time."""
        return min(workers, key=lambda w: w.metrics.average_response_time)
    
    def _resource_based_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker based on resource utilization."""
        return min(workers, key=lambda w: w.metrics.get_load_score())
    
    def _adaptive_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Adaptive selection based on multiple factors."""
        current_time = time.time()
        
        # Score workers based on multiple factors
        scored_workers = []
        for worker in workers:
            # Base score from resource utilization
            score = worker.metrics.get_load_score()
            
            # Adjust for connection count
            connection_factor = worker.current_connections / worker.max_connections
            score += connection_factor * 50
            
            # Adjust for recent errors
            if worker.consecutive_failures > 0:
                score += worker.consecutive_failures * 20
            
            # Adjust for staleness of metrics
            metrics_age = current_time - worker.metrics.last_updated
            if metrics_age > 60:  # Metrics older than 1 minute
                score += metrics_age * 0.1
            
            scored_workers.append((score, worker))
        
        # Select worker with lowest score
        scored_workers.sort(key=lambda x: x[0])
        return scored_workers[0][1]
    
    async def route_request(self, request: Any, priority: float = 1.0) -> Any:
        """Route request to appropriate worker."""
        worker = self.get_next_worker()
        
        if not worker:
            # No workers available, queue the request
            if self.request_queue.enqueue(request, priority):
                self.logger.warning("No workers available, request queued")
                # Wait for worker to become available
                await self._wait_for_worker()
                worker = self.get_next_worker()
            else:
                raise Exception("Request queue full, cannot process request")
        
        if not worker:
            raise Exception("No workers available after waiting")
        
        # Execute request with circuit breaker protection
        start_time = time.time()
        success = False
        
        try:
            worker.current_connections += 1
            circuit_breaker = self.circuit_breakers[worker.worker_id]
            
            # Simulate request processing (replace with actual worker call)
            result = await self._execute_request_on_worker(worker, request)
            success = True
            return result
            
        except Exception as e:
            self.logger.error(f"Request failed on worker {worker.worker_id}: {e}")
            raise
        
        finally:
            worker.current_connections -= 1
            response_time = time.time() - start_time
            worker.update_metrics(response_time, success)
    
    async def _execute_request_on_worker(self, worker: WorkerNode, request: Any) -> Any:
        """Execute request on specific worker (placeholder)."""
        # This is where you'd actually send the request to the worker
        # For now, simulate processing
        await asyncio.sleep(random.uniform(0.01, 0.1))
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Simulated worker error")
        
        return f"Processed by worker {worker.worker_id}"
    
    async def _wait_for_worker(self, timeout: float = 10.0) -> None:
        """Wait for a worker to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if any(worker.is_available() for worker in self.workers.values()):
                return
            await asyncio.sleep(0.1)
        
        raise Exception("Timeout waiting for available worker")
    
    def _auto_scaling_loop(self) -> None:
        """Background loop for auto-scaling decisions."""
        while True:
            try:
                time.sleep(self.scale_check_interval)
                self._check_scaling_needs()
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
    
    def _check_scaling_needs(self) -> None:
        """Check if scaling up or down is needed."""
        if not self.auto_scaling_enabled:
            return
        
        with self._lock:
            if not self.workers:
                return
            
            # Calculate average load
            healthy_workers = [
                w for w in self.workers.values()
                if w.status == WorkerStatus.HEALTHY
            ]
            
            if not healthy_workers:
                return
            
            avg_load = statistics.mean(
                w.metrics.get_load_score() for w in healthy_workers
            )
            
            total_workers = len(self.workers)
            queue_size = self.request_queue.size()
            
            # Scale up conditions
            if (avg_load > self.scale_up_threshold * 100 or 
                queue_size > 10) and total_workers < self.max_workers:
                self._scale_up()
            
            # Scale down conditions
            elif (avg_load < self.scale_down_threshold * 100 and 
                  queue_size == 0 and total_workers > self.min_workers):
                self._scale_down()
    
    def _scale_up(self) -> None:
        """Add a new worker instance."""
        new_worker_id = f"auto_worker_{int(time.time())}"
        self.add_worker(new_worker_id)
        self.logger.info(f"Scaled up: added worker {new_worker_id}")
    
    def _scale_down(self) -> None:
        """Remove a worker instance."""
        # Find least utilized worker
        candidates = [
            w for w in self.workers.values()
            if w.worker_id.startswith("auto_worker_")
        ]
        
        if candidates:
            least_utilized = min(candidates, key=lambda w: w.current_connections)
            self.remove_worker(least_utilized.worker_id)
            self.logger.info(f"Scaled down: removed worker {least_utilized.worker_id}")
    
    def _health_check_loop(self) -> None:
        """Background loop for worker health checks."""
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                self._perform_health_checks()
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all workers."""
        current_time = time.time()
        
        with self._lock:
            for worker in self.workers.values():
                # Simple health check based on recent activity
                if current_time - worker.last_health_check > 30:
                    # Worker hasn't been checked recently
                    if worker.consecutive_failures >= 5:
                        worker.status = WorkerStatus.FAILED
                    
                    worker.last_health_check = current_time
    
    def _metrics_collection_loop(self) -> None:
        """Background loop for collecting worker metrics."""
        while True:
            try:
                time.sleep(5)  # Collect every 5 seconds
                self._collect_worker_metrics()
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
    
    def _collect_worker_metrics(self) -> None:
        """Collect system metrics for workers."""
        # Simulate metrics collection
        # In a real implementation, this would query actual worker instances
        current_time = time.time()
        
        with self._lock:
            for worker in self.workers.values():
                # Update system metrics (simulated)
                worker.metrics.cpu_percent = random.uniform(10, 80)
                worker.metrics.memory_percent = random.uniform(20, 70)
                worker.metrics.queue_length = self.request_queue.size()
                worker.metrics.last_updated = current_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        with self._lock:
            worker_stats = {}
            for worker_id, worker in self.workers.items():
                worker_stats[worker_id] = {
                    'status': worker.status.value,
                    'current_connections': worker.current_connections,
                    'max_connections': worker.max_connections,
                    'request_count': worker.request_count,
                    'error_count': worker.error_count,
                    'error_rate': worker.metrics.error_rate,
                    'avg_response_time': worker.metrics.average_response_time,
                    'load_score': worker.metrics.get_load_score(),
                    'weight': worker.weight
                }
            
            return {
                'strategy': self.strategy.value,
                'total_workers': len(self.workers),
                'healthy_workers': len([
                    w for w in self.workers.values()
                    if w.status == WorkerStatus.HEALTHY
                ]),
                'queue_size': self.request_queue.size(),
                'auto_scaling_enabled': self.auto_scaling_enabled,
                'worker_stats': worker_stats
            }