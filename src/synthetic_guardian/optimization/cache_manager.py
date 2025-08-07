"""
Advanced caching system with multiple strategies and intelligent eviction
"""

import asyncio
import time
import pickle
import gzip
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from ..utils.logger import get_logger
from ..utils.config import Config

logger = get_logger(__name__)


class CacheStrategy(Enum):
    """Cache strategies."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    hit_rate: float = 0.0
    memory_usage_bytes: int = 0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0


@dataclass
class CacheItem:
    """Cache item with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """Check if item is expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class BaseCacheBackend(ABC):
    """Base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value by key."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached values."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class MemoryCacheBackend(BaseCacheBackend):
    """In-memory cache backend with configurable eviction strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        strategy: CacheStrategy = CacheStrategy.LRU,
        default_ttl: Optional[float] = None,
        enable_compression: bool = True,
        compression_threshold: int = 1024
    ):
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        
        self._cache: Dict[str, CacheItem] = {}
        self._stats = CacheStats(max_size=max_size)
        self._lock = threading.RLock()
        self._access_order: OrderedDict[str, None] = OrderedDict()
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start periodic cleanup task."""
        async def cleanup_task():
            while True:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_expired()
        
        self._cleanup_task = asyncio.create_task(cleanup_task())
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return None
            
            item = self._cache[key]
            
            # Check expiration
            if item.is_expired():
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.size -= 1
                self._stats.memory_usage_bytes -= item.size_bytes
                self._stats.update_hit_rate()
                return None
            
            # Update access metadata
            item.touch()
            
            # Update access order for LRU
            if self.strategy == CacheStrategy.LRU:
                self._access_order.move_to_end(key)
            
            self._stats.hits += 1
            self._stats.update_hit_rate()
            
            # Decompress if needed
            value = item.value
            if item.compressed:
                value = pickle.loads(gzip.decompress(value))
            
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value by key."""
        with self._lock:
            # Calculate size
            serialized_value = pickle.dumps(value)
            size_bytes = len(serialized_value)
            
            # Compress if beneficial
            compressed = False
            if (
                self.enable_compression and 
                size_bytes > self.compression_threshold
            ):
                compressed_value = gzip.compress(serialized_value)
                if len(compressed_value) < size_bytes * 0.9:  # 10% improvement
                    serialized_value = compressed_value
                    compressed = True
            
            # Use provided TTL or default
            item_ttl = ttl if ttl is not None else self.default_ttl
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=serialized_value,
                ttl=item_ttl,
                size_bytes=len(serialized_value),
                compressed=compressed
            )
            
            # Remove existing item if present
            if key in self._cache:
                old_item = self._cache[key]
                self._stats.memory_usage_bytes -= old_item.size_bytes
            else:
                self._stats.size += 1
            
            # Add new item
            self._cache[key] = item
            self._stats.memory_usage_bytes += item.size_bytes
            
            # Update access order
            if key in self._access_order:
                self._access_order.move_to_end(key)
            else:
                self._access_order[key] = None
            
            # Evict if necessary
            while self._stats.size > self.max_size:
                await self._evict_one()
    
    async def delete(self, key: str) -> bool:
        """Delete value by key."""
        with self._lock:
            if key not in self._cache:
                return False
            
            item = self._cache[key]
            del self._cache[key]
            
            if key in self._access_order:
                del self._access_order[key]
            
            self._stats.size -= 1
            self._stats.memory_usage_bytes -= item.size_bytes
            return True
    
    async def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats.size = 0
            self._stats.memory_usage_bytes = 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        with self._lock:
            if key not in self._cache:
                return False
            
            item = self._cache[key]
            if item.is_expired():
                del self._cache[key]
                if key in self._access_order:
                    del self._access_order[key]
                self._stats.size -= 1
                self._stats.evictions += 1
                self._stats.memory_usage_bytes -= item.size_bytes
                return False
            
            return True
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                max_size=self._stats.max_size,
                hit_rate=self._stats.hit_rate,
                memory_usage_bytes=self._stats.memory_usage_bytes
            )
    
    async def _evict_one(self) -> None:
        """Evict one item based on strategy."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            key = next(iter(self._access_order))
            
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            
        elif self.strategy == CacheStrategy.FIFO:
            # Remove oldest
            key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            
        else:  # Default to LRU
            key = next(iter(self._access_order))
        
        await self.delete(key)
        self._stats.evictions += 1
    
    async def _cleanup_expired(self) -> None:
        """Remove expired items."""
        expired_keys = []
        
        with self._lock:
            for key, item in self._cache.items():
                if item.is_expired():
                    expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete(key)
            self._stats.evictions += 1
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self.clear()


class RedisCacheBackend(BaseCacheBackend):
    """Redis cache backend (placeholder implementation)."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, **kwargs):
        self.host = host
        self.port = port
        self.db = db
        self.redis_client = None
        self._stats = CacheStats()
        
        logger.warning("Redis backend not fully implemented")
    
    async def get(self, key: str) -> Optional[Any]:
        # Placeholder implementation
        self._stats.misses += 1
        self._stats.update_hit_rate()
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        # Placeholder implementation
        pass
    
    async def delete(self, key: str) -> bool:
        # Placeholder implementation
        return False
    
    async def clear(self) -> None:
        # Placeholder implementation
        pass
    
    async def exists(self, key: str) -> bool:
        # Placeholder implementation
        return False
    
    def get_stats(self) -> CacheStats:
        return self._stats


class CacheManager:
    """Advanced cache manager with multiple tiers and intelligent routing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backends: Dict[str, BaseCacheBackend] = {}
        self.default_backend = "memory"
        self.logger = get_logger(self.__class__.__name__)
        self._key_routing: Dict[str, str] = {}  # Key pattern -> backend mapping
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize default backends
        asyncio.create_task(self._initialize_backends())
    
    async def _initialize_backends(self) -> None:
        """Initialize cache backends."""
        # Memory backend
        memory_config = self.config.get("memory", {})
        memory_backend = MemoryCacheBackend(
            max_size=memory_config.get("max_size", 1000),
            strategy=CacheStrategy(memory_config.get("strategy", "lru")),
            default_ttl=memory_config.get("default_ttl"),
            enable_compression=memory_config.get("compression", True),
            compression_threshold=memory_config.get("compression_threshold", 1024)
        )
        self.backends["memory"] = memory_backend
        
        # Redis backend (if configured)
        redis_config = self.config.get("redis", {})
        if redis_config.get("enabled", False):
            redis_backend = RedisCacheBackend(**redis_config)
            self.backends["redis"] = redis_backend
        
        self.logger.info(f"Initialized {len(self.backends)} cache backends")
    
    def _get_backend(self, key: str) -> BaseCacheBackend:
        """Get appropriate backend for key."""
        # Check key routing rules
        for pattern, backend_name in self._key_routing.items():
            if pattern in key:
                return self.backends.get(backend_name, self.backends[self.default_backend])
        
        return self.backends[self.default_backend]
    
    def _generate_cache_key(self, namespace: str, key: str, **kwargs) -> str:
        """Generate cache key with namespace and parameters."""
        if kwargs:
            # Include parameters in key
            param_str = json.dumps(kwargs, sort_keys=True, default=str)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
            return f"{namespace}:{key}:{param_hash}"
        return f"{namespace}:{key}"
    
    async def get(
        self,
        key: str,
        namespace: str = "default",
        **kwargs
    ) -> Optional[Any]:
        """Get cached value."""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        backend = self._get_backend(cache_key)
        
        try:
            return await backend.get(cache_key)
        except Exception as e:
            self.logger.error(f"Cache get error: {str(e)}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[float] = None,
        **kwargs
    ) -> None:
        """Set cached value."""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        backend = self._get_backend(cache_key)
        
        try:
            await backend.set(cache_key, value, ttl)
        except Exception as e:
            self.logger.error(f"Cache set error: {str(e)}")
    
    async def delete(
        self,
        key: str,
        namespace: str = "default",
        **kwargs
    ) -> bool:
        """Delete cached value."""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        backend = self._get_backend(cache_key)
        
        try:
            return await backend.delete(cache_key)
        except Exception as e:
            self.logger.error(f"Cache delete error: {str(e)}")
            return False
    
    async def clear(self, backend_name: Optional[str] = None) -> None:
        """Clear cache(s)."""
        if backend_name:
            if backend_name in self.backends:
                await self.backends[backend_name].clear()
        else:
            for backend in self.backends.values():
                await backend.clear()
    
    async def exists(
        self,
        key: str,
        namespace: str = "default",
        **kwargs
    ) -> bool:
        """Check if key exists in cache."""
        cache_key = self._generate_cache_key(namespace, key, **kwargs)
        backend = self._get_backend(cache_key)
        
        try:
            return await backend.exists(cache_key)
        except Exception as e:
            self.logger.error(f"Cache exists error: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all backends."""
        return {
            name: backend.get_stats()
            for name, backend in self.backends.items()
        }
    
    def configure_routing(self, pattern: str, backend: str) -> None:
        """Configure key routing to specific backend."""
        self._key_routing[pattern] = backend
        self.logger.info(f"Configured routing: {pattern} -> {backend}")
    
    async def memoize(
        self,
        func: Callable,
        key: Optional[str] = None,
        namespace: str = "memoize",
        ttl: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Memoize function result."""
        # Generate key from function and arguments
        if key is None:
            func_key = f"{func.__module__}.{func.__name__}"
            args_hash = hashlib.md5(str(kwargs).encode()).hexdigest()[:8]
            key = f"{func_key}:{args_hash}"
        
        # Try cache first
        cached_result = await self.get(key, namespace)
        if cached_result is not None:
            return cached_result
        
        # Execute function
        if asyncio.iscoroutinefunction(func):
            result = await func(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self._executor, func, **kwargs)
        
        # Cache result
        await self.set(key, result, namespace, ttl)
        return result
    
    def cache_decorator(
        self,
        namespace: str = "decorator",
        ttl: Optional[float] = None,
        key_func: Optional[Callable] = None
    ) -> Callable:
        """Cache decorator for functions."""
        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    args_str = json.dumps([str(arg) for arg in args], default=str)
                    kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
                    combined = f"{args_str}:{kwargs_str}"
                    cache_key = hashlib.md5(combined.encode()).hexdigest()
                
                return await self.memoize(
                    func,
                    key=cache_key,
                    namespace=namespace,
                    ttl=ttl,
                    *args,
                    **kwargs
                )
            return wrapper
        return decorator
    
    async def cleanup(self) -> None:
        """Clean up cache manager resources."""
        for backend in self.backends.values():
            if hasattr(backend, 'cleanup'):
                await backend.cleanup()
        
        self._executor.shutdown(wait=True)
        self.backends.clear()
        self.logger.info("Cache manager cleaned up")


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def cache(namespace: str = "default", ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Convenient cache decorator."""
    cache_manager = get_cache_manager()
    return cache_manager.cache_decorator(namespace, ttl, key_func)
