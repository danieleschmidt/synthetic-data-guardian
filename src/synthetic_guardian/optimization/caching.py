"""
Advanced caching system with intelligent cache management and distributed support
"""

import asyncio
import time
import hashlib
import pickle
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import weakref

from ..utils.logger import get_logger


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


class CacheLevel(Enum):
    """Cache levels for hierarchical caching."""
    L1_MEMORY = "l1_memory"  # Fast in-memory cache
    L2_DISK = "l2_disk"      # Slower disk-based cache
    L3_DISTRIBUTED = "l3_distributed"  # Network distributed cache


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.accessed_at = time.time()
        self.access_count += 1


class IntelligentCache:
    """
    Advanced multi-level caching system with adaptive strategies.
    """
    
    def __init__(
        self,
        max_memory_entries: int = 10000,
        max_memory_size_mb: int = 1024,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        enable_disk_cache: bool = True,
        disk_cache_dir: Optional[Path] = None
    ):
        self.max_memory_entries = max_memory_entries
        self.max_memory_size_bytes = max_memory_size_mb * 1024 * 1024
        self.strategy = strategy
        self.enable_disk_cache = enable_disk_cache
        
        self.logger = get_logger(self.__class__.__name__)
        
        # L1 Memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_size_bytes = 0
        
        # L2 Disk cache
        self.disk_cache_dir = disk_cache_dir or Path("/tmp/synthetic_guardian_cache")
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_hits': 0,
            'disk_misses': 0,
            'total_requests': 0
        }
        
        # Access patterns for adaptive strategy
        self.access_patterns: Dict[str, List[float]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background maintenance
        self._start_maintenance_thread()
    
    def _start_maintenance_thread(self):
        """Start background maintenance thread."""
        def maintenance_worker():
            while True:
                try:
                    self._maintenance_cycle()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    self.logger.error(f"Cache maintenance error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        maintenance_thread.start()
    
    def _maintenance_cycle(self):
        """Perform cache maintenance operations."""
        with self.lock:
            # Remove expired entries
            self._remove_expired_entries()
            
            # Optimize cache based on strategy
            if self.strategy == CacheStrategy.ADAPTIVE:
                self._adaptive_optimization()
            
            # Clean up access patterns
            self._cleanup_access_patterns()
            
            # Disk cache maintenance
            if self.enable_disk_cache:
                self._disk_cache_maintenance()
    
    def _remove_expired_entries(self):
        """Remove expired cache entries."""
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self._evict_from_memory(key)
            
        if expired_keys:
            self.logger.debug(f"Removed {len(expired_keys)} expired cache entries")
    
    def _adaptive_optimization(self):
        """Adapt cache strategy based on access patterns."""
        current_time = time.time()
        
        # Analyze access patterns for each key
        for key, access_times in self.access_patterns.items():
            # Remove old access times (older than 1 hour)
            recent_accesses = [t for t in access_times if current_time - t < 3600]
            self.access_patterns[key] = recent_accesses
            
            # If key is frequently accessed, ensure it stays in L1 cache
            if len(recent_accesses) > 10 and key in self.memory_cache:
                entry = self.memory_cache[key]
                # Boost priority by updating access time
                entry.accessed_at = current_time
                entry.access_count += len(recent_accesses) // 10
    
    def _cleanup_access_patterns(self):
        """Clean up old access pattern data."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, access_times in self.access_patterns.items():
            # Remove patterns for keys not accessed in last 24 hours
            if not access_times or current_time - max(access_times) > 86400:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.access_patterns[key]
    
    def _disk_cache_maintenance(self):
        """Maintain disk cache."""
        try:
            # Remove old disk cache files (older than 7 days)
            current_time = time.time()
            for cache_file in self.disk_cache_dir.glob("*.cache"):
                if current_time - cache_file.stat().st_mtime > 604800:  # 7 days
                    cache_file.unlink()
        except Exception as e:
            self.logger.error(f"Disk cache maintenance error: {e}")
    
    def _calculate_cache_key(self, key_data: Any) -> str:
        """Calculate consistent cache key from data."""
        if isinstance(key_data, str):
            return hashlib.sha256(key_data.encode()).hexdigest()[:16]
        else:
            # Serialize and hash
            serialized = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.sha256(serialized).hexdigest()[:16]
    
    def _calculate_entry_size(self, value: Any) -> int:
        """Calculate size of cache entry in bytes."""
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_entry_size(item) for item in value[:100])  # Sample first 100
            elif isinstance(value, dict):
                return sum(
                    self._calculate_entry_size(k) + self._calculate_entry_size(v)
                    for k, v in list(value.items())[:100]  # Sample first 100
                )
            else:
                return 1024  # Default estimate
    
    async def get(self, key: Any, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        cache_key = self._calculate_cache_key(key)
        
        with self.lock:
            self.stats['total_requests'] += 1
            
            # Check L1 memory cache
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                
                if not entry.is_expired():
                    entry.update_access()
                    self._update_access_pattern(cache_key)
                    self.stats['hits'] += 1
                    self.logger.debug(f"L1 cache hit for key: {cache_key}")
                    return entry.value
                else:
                    # Remove expired entry
                    self._evict_from_memory(cache_key)
            
            # Check L2 disk cache
            if self.enable_disk_cache:
                disk_value = await self._get_from_disk(cache_key)
                if disk_value is not None:
                    # Promote to L1 cache
                    await self._set_in_memory(cache_key, disk_value)
                    self._update_access_pattern(cache_key)
                    self.stats['disk_hits'] += 1
                    self.logger.debug(f"L2 cache hit for key: {cache_key}")
                    return disk_value
                else:
                    self.stats['disk_misses'] += 1
            
            self.stats['misses'] += 1
            return default
    
    async def set(
        self, 
        key: Any, 
        value: Any, 
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            metadata: Additional metadata
        """
        cache_key = self._calculate_cache_key(key)
        
        with self.lock:
            await self._set_in_memory(cache_key, value, ttl_seconds, metadata)
            
            # Also store in disk cache for persistence
            if self.enable_disk_cache:
                await self._set_in_disk(cache_key, value, ttl_seconds, metadata)
    
    async def _set_in_memory(
        self, 
        cache_key: str, 
        value: Any, 
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Set value in L1 memory cache."""
        size_bytes = self._calculate_entry_size(value)
        
        # Check if we need to make space
        while (
            len(self.memory_cache) >= self.max_memory_entries or
            self.memory_size_bytes + size_bytes > self.max_memory_size_bytes
        ):
            if not self._evict_one_entry():
                break  # Can't evict any more entries
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=1,
            size_bytes=size_bytes,
            ttl_seconds=ttl_seconds,
            metadata=metadata or {}
        )
        
        # Remove old entry if exists
        if cache_key in self.memory_cache:
            old_entry = self.memory_cache[cache_key]
            self.memory_size_bytes -= old_entry.size_bytes
        
        # Add new entry
        self.memory_cache[cache_key] = entry
        self.memory_size_bytes += size_bytes
        
        self.logger.debug(f"Cached entry: {cache_key} ({size_bytes} bytes)")
    
    async def _get_from_disk(self, cache_key: str) -> Optional[Any]:
        """Get value from L2 disk cache."""
        cache_file = self.disk_cache_dir / f"{cache_key}.cache"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Check TTL
                if 'ttl_seconds' in cached_data and cached_data['ttl_seconds'] is not None:
                    if time.time() - cached_data['created_at'] > cached_data['ttl_seconds']:
                        cache_file.unlink()  # Remove expired file
                        return None
                
                return cached_data['value']
        except Exception as e:
            self.logger.error(f"Error reading from disk cache: {e}")
            try:
                cache_file.unlink()  # Remove corrupted file
            except:
                pass
        
        return None
    
    async def _set_in_disk(
        self, 
        cache_key: str, 
        value: Any, 
        ttl_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Set value in L2 disk cache."""
        cache_file = self.disk_cache_dir / f"{cache_key}.cache"
        
        try:
            cached_data = {
                'value': value,
                'created_at': time.time(),
                'ttl_seconds': ttl_seconds,
                'metadata': metadata or {}
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        except Exception as e:
            self.logger.error(f"Error writing to disk cache: {e}")
    
    def _evict_one_entry(self) -> bool:
        """Evict one entry from memory cache based on strategy."""
        if not self.memory_cache:
            return False
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            victim_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].accessed_at
            )
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            victim_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].access_count
            )
        elif self.strategy == CacheStrategy.TTL:
            # Evict entries with shortest remaining TTL
            current_time = time.time()
            victim_key = min(
                self.memory_cache.keys(),
                key=lambda k: (
                    self.memory_cache[k].ttl_seconds - (current_time - self.memory_cache[k].created_at)
                    if self.memory_cache[k].ttl_seconds else float('inf')
                )
            )
        else:  # ADAPTIVE
            # Use combined score of recency and frequency
            current_time = time.time()
            victim_key = min(
                self.memory_cache.keys(),
                key=lambda k: (
                    self.memory_cache[k].access_count / 
                    max(1, current_time - self.memory_cache[k].accessed_at)
                )
            )
        
        self._evict_from_memory(victim_key)
        return True
    
    def _evict_from_memory(self, cache_key: str):
        """Evict specific entry from memory cache."""
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            self.memory_size_bytes -= entry.size_bytes
            del self.memory_cache[cache_key]
            self.stats['evictions'] += 1
            self.logger.debug(f"Evicted cache entry: {cache_key}")
    
    def _update_access_pattern(self, cache_key: str):
        """Update access pattern for adaptive caching."""
        current_time = time.time()
        
        if cache_key not in self.access_patterns:
            self.access_patterns[cache_key] = []
        
        self.access_patterns[cache_key].append(current_time)
        
        # Keep only recent access times (last hour)
        self.access_patterns[cache_key] = [
            t for t in self.access_patterns[cache_key]
            if current_time - t < 3600
        ]
    
    async def delete(self, key: Any):
        """Delete entry from cache."""
        cache_key = self._calculate_cache_key(key)
        
        with self.lock:
            # Remove from memory cache
            if cache_key in self.memory_cache:
                self._evict_from_memory(cache_key)
            
            # Remove from disk cache
            if self.enable_disk_cache:
                cache_file = self.disk_cache_dir / f"{cache_key}.cache"
                try:
                    if cache_file.exists():
                        cache_file.unlink()
                except Exception as e:
                    self.logger.error(f"Error deleting from disk cache: {e}")
    
    async def clear(self):
        """Clear all cache entries."""
        with self.lock:
            # Clear memory cache
            self.memory_cache.clear()
            self.memory_size_bytes = 0
            self.access_patterns.clear()
            
            # Clear disk cache
            if self.enable_disk_cache:
                try:
                    for cache_file in self.disk_cache_dir.glob("*.cache"):
                        cache_file.unlink()
                except Exception as e:
                    self.logger.error(f"Error clearing disk cache: {e}")
            
            # Reset stats
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'disk_hits': 0,
                'disk_misses': 0,
                'total_requests': 0
            }
            
            self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = (
                self.stats['hits'] / max(1, self.stats['total_requests'])
            ) * 100
            
            disk_hit_rate = (
                self.stats['disk_hits'] / max(1, self.stats['disk_misses'] + self.stats['disk_hits'])
            ) * 100 if self.enable_disk_cache else 0
            
            return {
                **self.stats,
                'memory_entries': len(self.memory_cache),
                'memory_size_mb': self.memory_size_bytes / (1024 * 1024),
                'memory_utilization_percent': (
                    self.memory_size_bytes / self.max_memory_size_bytes
                ) * 100,
                'hit_rate_percent': hit_rate,
                'disk_hit_rate_percent': disk_hit_rate,
                'strategy': self.strategy.value,
                'access_patterns_tracked': len(self.access_patterns)
            }
    
    def optimize_cache(self):
        """Manually trigger cache optimization."""
        with self.lock:
            self._maintenance_cycle()
            self.logger.info("Cache optimization completed")


# Global cache instance
_global_cache: Optional[IntelligentCache] = None


def get_cache() -> IntelligentCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache


def cached(
    ttl_seconds: Optional[float] = None,
    key_generator: Optional[Callable] = None,
    cache_instance: Optional[IntelligentCache] = None
):
    """Decorator to cache function results."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            cache = cache_instance or get_cache()
            
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl_seconds)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to run the async cache operations
            import asyncio
            
            cache = cache_instance or get_cache()
            
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(cache.get(cache_key))
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            loop.run_until_complete(cache.set(cache_key, result, ttl_seconds))
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator