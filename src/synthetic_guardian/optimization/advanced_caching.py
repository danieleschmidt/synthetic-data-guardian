"""
Advanced Caching System - Multi-tier intelligent caching with eviction policies
"""

import asyncio
import time
import hashlib
import pickle
import json
import logging
import threading
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import weakref
from collections import OrderedDict, defaultdict
from pathlib import Path
import redis
import zlib


class CacheLevel(Enum):
    """Cache tier levels."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    compressed: bool = False
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> None:
        """Record access to this entry."""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache:
    """High-performance in-memory cache with configurable eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, 
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_counts: defaultdict[str, int] = defaultdict(int)
        self.current_memory = 0
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (dict, list)):
                return len(json.dumps(value, default=str).encode())
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size estimate
    
    def _should_compress(self, size_bytes: int) -> bool:
        """Determine if value should be compressed."""
        return size_bytes > 1024  # Compress values larger than 1KB
    
    def _compress_value(self, value: Any) -> Tuple[bytes, bool]:
        """Compress value if beneficial."""
        try:
            serialized = pickle.dumps(value)
            if self._should_compress(len(serialized)):
                compressed = zlib.compress(serialized)
                if len(compressed) < len(serialized) * 0.8:  # Only if 20%+ reduction
                    return compressed, True
            return serialized, False
        except Exception:
            return pickle.dumps(value), False
    
    def _decompress_value(self, data: bytes, compressed: bool) -> Any:
        """Decompress and deserialize value."""
        try:
            if compressed:
                data = zlib.decompress(data)
            return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Failed to decompress cache value: {e}")
            return None
    
    def _evict_entries(self) -> None:
        """Evict entries based on policy."""
        if len(self.cache) <= self.max_size and self.current_memory <= self.max_memory_bytes:
            return
        
        entries_to_remove = []
        target_size = max(int(self.max_size * 0.8), 1)
        target_memory = int(self.max_memory_bytes * 0.8)
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed
            )
            
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )
            
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired entries first, then oldest
            current_time = time.time()
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (not x[1].is_expired(), x[1].created_at)
            )
            
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Remove oldest entries
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].created_at
            )
            
        else:  # ADAPTIVE
            # Adaptive policy based on access patterns
            current_time = time.time()
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (
                    x[1].access_count / max(current_time - x[1].created_at, 1),  # Access rate
                    -x[1].last_accessed  # Recent access (negative for reverse sort)
                )
            )
        
        # Remove entries until we're under limits
        for key, entry in sorted_entries:
            if (len(self.cache) <= target_size and 
                self.current_memory <= target_memory):
                break
                
            entries_to_remove.append(key)
        
        # Remove the entries
        for key in entries_to_remove:
            self._remove_entry(key)
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory -= entry.size_bytes
            del self.cache[key]
            self.evictions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access tracking
            entry.access()
            self.hits += 1
            
            # Move to end for LRU (if using OrderedDict)
            if self.eviction_policy == EvictionPolicy.LRU:
                self.cache.move_to_end(key)
            
            # Decompress if needed
            if isinstance(entry.value, bytes):
                return self._decompress_value(entry.value, entry.compressed)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"Value too large for cache: {size_bytes} bytes")
                return False
            
            # Compress if beneficial
            compressed_value, is_compressed = self._compress_value(value)
            actual_size = len(compressed_value) if isinstance(compressed_value, bytes) else size_bytes
            
            # Remove existing entry if exists
            if key in self.cache:
                self._remove_entry(key)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                ttl=ttl,
                size_bytes=actual_size,
                compressed=is_compressed
            )
            
            # Add to cache
            self.cache[key] = entry
            self.current_memory += actual_size
            
            # Evict if necessary
            self._evict_entries()
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.current_memory = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_mb': self.current_memory / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'eviction_policy': self.eviction_policy.value
            }


class RedisCache:
    """Redis-based distributed cache layer."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 namespace: str = "synthetic_guardian"):
        self.namespace = namespace
        self.logger = logging.getLogger(self.__class__.__name__)
        
        try:
            self.redis = redis.from_url(redis_url, decode_responses=False)
            self.redis.ping()  # Test connection
            self.available = True
            self.logger.info("Redis cache initialized successfully")
        except Exception as e:
            self.logger.warning(f"Redis not available: {e}")
            self.redis = None
            self.available = False
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self.namespace}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.available:
            return None
        
        try:
            redis_key = self._make_key(key)
            data = self.redis.get(redis_key)
            
            if data is None:
                return None
            
            # Deserialize
            return pickle.loads(data)
            
        except Exception as e:
            self.logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self.available:
            return False
        
        try:
            redis_key = self._make_key(key)
            serialized_value = pickle.dumps(value)
            
            if ttl:
                self.redis.setex(redis_key, ttl, serialized_value)
            else:
                self.redis.set(redis_key, serialized_value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self.available:
            return False
        
        try:
            redis_key = self._make_key(key)
            return bool(self.redis.delete(redis_key))
        except Exception as e:
            self.logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries with namespace."""
        if not self.available:
            return
        
        try:
            pattern = f"{self.namespace}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
        except Exception as e:
            self.logger.error(f"Redis clear error: {e}")


class DiskCache:
    """Disk-based cache for large or persistent data."""
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _get_meta_path(self, key: str) -> Path:
        """Get metadata file path for cache key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.meta"
    
    def _write_metadata(self, key: str, ttl: Optional[float]) -> None:
        """Write cache metadata."""
        meta_path = self._get_meta_path(key)
        metadata = {
            'created_at': time.time(),
            'ttl': ttl,
            'last_accessed': time.time(),
            'access_count': 1
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
    
    def _read_metadata(self, key: str) -> Optional[Dict]:
        """Read cache metadata."""
        meta_path = self._get_meta_path(key)
        if not meta_path.exists():
            return None
        
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _update_access(self, key: str) -> None:
        """Update access metadata."""
        metadata = self._read_metadata(key)
        if metadata:
            metadata['last_accessed'] = time.time()
            metadata['access_count'] = metadata.get('access_count', 0) + 1
            
            meta_path = self._get_meta_path(key)
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        metadata = self._read_metadata(key)
        if not metadata:
            return True
        
        ttl = metadata.get('ttl')
        if ttl is None:
            return False
        
        return time.time() - metadata['created_at'] > ttl
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        for meta_file in self.cache_dir.glob("*.meta"):
            key_hash = meta_file.stem
            cache_file = self.cache_dir / f"{key_hash}.cache"
            
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                ttl = metadata.get('ttl')
                if ttl and time.time() - metadata['created_at'] > ttl:
                    # Remove expired files
                    meta_file.unlink(missing_ok=True)
                    cache_file.unlink(missing_ok=True)
                    
            except Exception:
                # Remove corrupted metadata
                meta_file.unlink(missing_ok=True)
                cache_file.unlink(missing_ok=True)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                return None
            
            if self._is_expired(key):
                # Remove expired entry
                file_path.unlink(missing_ok=True)
                self._get_meta_path(key).unlink(missing_ok=True)
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Update access metadata
                self._update_access(key)
                
                return pickle.loads(data)
                
            except Exception as e:
                self.logger.error(f"Disk cache read error: {e}")
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in disk cache."""
        with self._lock:
            try:
                # Serialize value
                serialized_value = pickle.dumps(value)
                
                # Check size limits
                if len(serialized_value) > self.max_size_bytes:
                    return False
                
                # Write cache file
                file_path = self._get_file_path(key)
                with open(file_path, 'wb') as f:
                    f.write(serialized_value)
                
                # Write metadata
                self._write_metadata(key, ttl)
                
                # Cleanup expired entries periodically
                if hash(key) % 100 == 0:  # 1% chance
                    self._cleanup_expired()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Disk cache write error: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from disk cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            meta_path = self._get_meta_path(key)
            
            deleted = False
            if file_path.exists():
                file_path.unlink()
                deleted = True
            
            if meta_path.exists():
                meta_path.unlink()
                deleted = True
            
            return deleted
    
    async def clear(self) -> None:
        """Clear all disk cache entries."""
        with self._lock:
            for file_path in self.cache_dir.glob("*"):
                file_path.unlink()


class MultiTierCache:
    """Multi-tier cache system with intelligent promotion/demotion."""
    
    def __init__(self, memory_cache: MemoryCache, redis_cache: RedisCache, 
                 disk_cache: DiskCache, enable_promotion: bool = True):
        self.memory_cache = memory_cache
        self.redis_cache = redis_cache
        self.disk_cache = disk_cache
        self.enable_promotion = enable_promotion
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Performance statistics
        self.tier_stats = {
            CacheLevel.MEMORY: {'hits': 0, 'misses': 0},
            CacheLevel.REDIS: {'hits': 0, 'misses': 0},
            CacheLevel.DISK: {'hits': 0, 'misses': 0}
        }
    
    def _should_promote(self, key: str, tier: CacheLevel) -> bool:
        """Determine if key should be promoted to higher tier."""
        if not self.enable_promotion:
            return False
        
        # Simple promotion strategy based on access frequency
        # In a production system, this would be more sophisticated
        return hash(key) % 3 == 0  # Promote ~33% of accessed items
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-tier cache."""
        # Try memory cache first (L1)
        value = self.memory_cache.get(key)
        if value is not None:
            self.tier_stats[CacheLevel.MEMORY]['hits'] += 1
            return value
        
        self.tier_stats[CacheLevel.MEMORY]['misses'] += 1
        
        # Try Redis cache (L2)
        value = await self.redis_cache.get(key)
        if value is not None:
            self.tier_stats[CacheLevel.REDIS]['hits'] += 1
            
            # Promote to memory cache
            if self._should_promote(key, CacheLevel.REDIS):
                self.memory_cache.set(key, value)
            
            return value
        
        self.tier_stats[CacheLevel.REDIS]['misses'] += 1
        
        # Try disk cache (L3)
        value = await self.disk_cache.get(key)
        if value is not None:
            self.tier_stats[CacheLevel.DISK]['hits'] += 1
            
            # Promote to higher tiers
            if self._should_promote(key, CacheLevel.DISK):
                await self.redis_cache.set(key, value, ttl=3600)  # 1 hour
                self.memory_cache.set(key, value, ttl=300)  # 5 minutes
            
            return value
        
        self.tier_stats[CacheLevel.DISK]['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in multi-tier cache."""
        # Set in all tiers for maximum availability
        results = []
        
        # Memory cache (short TTL)
        memory_ttl = min(ttl or 300, 300)  # Max 5 minutes in memory
        results.append(self.memory_cache.set(key, value, memory_ttl))
        
        # Redis cache (medium TTL)
        redis_ttl = int(min(ttl or 3600, 3600))  # Max 1 hour in Redis
        results.append(await self.redis_cache.set(key, value, redis_ttl))
        
        # Disk cache (long TTL)
        disk_ttl = ttl or 86400  # Max 1 day on disk
        results.append(await self.disk_cache.set(key, value, disk_ttl))
        
        return any(results)  # Success if any tier succeeds
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache tiers."""
        results = []
        
        results.append(self.memory_cache.delete(key))
        results.append(await self.redis_cache.delete(key))
        results.append(await self.disk_cache.delete(key))
        
        return any(results)
    
    async def clear(self) -> None:
        """Clear all cache tiers."""
        self.memory_cache.clear()
        await self.redis_cache.clear()
        await self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        
        # Calculate overall hit rates
        total_hits = sum(stats['hits'] for stats in self.tier_stats.values())
        total_requests = sum(
            stats['hits'] + stats['misses'] for stats in self.tier_stats.values()
        )
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            'overall_hit_rate': overall_hit_rate,
            'memory_cache': memory_stats,
            'tier_stats': self.tier_stats,
            'redis_available': self.redis_cache.available,
            'promotion_enabled': self.enable_promotion
        }