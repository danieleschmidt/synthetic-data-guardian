/**
 * Advanced Caching Manager with Multiple Cache Strategies
 */

import { EventEmitter } from 'events';

export class AdvancedCacheManager extends EventEmitter {
  constructor(logger, options = {}) {
    super();
    this.logger = logger;
    this.options = {
      // Cache strategies
      strategies: {
        lru: { enabled: true, maxSize: 1000, ttl: 300000 }, // 5 minutes
        lfu: { enabled: true, maxSize: 500, ttl: 600000 }, // 10 minutes
        fifo: { enabled: false, maxSize: 200, ttl: 180000 }, // 3 minutes
        memory: { enabled: true, maxSize: 2000, ttl: 900000 }, // 15 minutes
        redis: { enabled: false, host: 'localhost', port: 6379, ttl: 1800000 }, // 30 minutes
      },

      // Cache tiers
      tiers: {
        hot: { strategy: 'memory', maxSize: 100, ttl: 60000 }, // 1 minute - frequently accessed
        warm: { strategy: 'lru', maxSize: 500, ttl: 300000 }, // 5 minutes - recently accessed
        cold: { strategy: 'lfu', maxSize: 1000, ttl: 900000 }, // 15 minutes - less frequently accessed
      },

      // Performance settings
      cleanupInterval: options.cleanupInterval || 60000, // 1 minute
      compressionThreshold: options.compressionThreshold || 1024, // 1KB
      enableMetrics: options.enableMetrics !== false,
      enableDistributed: options.enableDistributed === true,

      ...options,
    };

    this.caches = new Map();
    this.metrics = {
      hits: new Map(),
      misses: new Map(),
      evictions: new Map(),
      compressionSaved: 0,
      totalRequests: 0,
      averageResponseTime: 0,
    };

    this.initialize();
  }

  initialize() {
    this.logger.info('Initializing Advanced Cache Manager...');

    // Initialize cache strategies
    for (const [name, config] of Object.entries(this.options.strategies)) {
      if (config.enabled) {
        this.caches.set(name, this.createCache(name, config));
      }
    }

    // Initialize tier caches
    for (const [tier, config] of Object.entries(this.options.tiers)) {
      const cacheKey = `tier_${tier}`;
      this.caches.set(cacheKey, this.createCache(cacheKey, config));
    }

    // Start cleanup and metrics collection
    this.startCleanupInterval();

    if (this.options.enableMetrics) {
      this.startMetricsCollection();
    }

    this.logger.info('Advanced Cache Manager initialized', {
      strategies: Object.keys(this.options.strategies).filter(s => this.options.strategies[s].enabled),
      tiers: Object.keys(this.options.tiers),
    });
  }

  createCache(name, config) {
    switch (config.strategy) {
      case 'lru':
        return new LRUCache(config.maxSize, config.ttl, this.logger);
      case 'lfu':
        return new LFUCache(config.maxSize, config.ttl, this.logger);
      case 'fifo':
        return new FIFOCache(config.maxSize, config.ttl, this.logger);
      case 'memory':
        return new MemoryCache(config.maxSize, config.ttl, this.logger);
      case 'redis':
        return new RedisCache(config, this.logger);
      default:
        return new MemoryCache(config.maxSize, config.ttl, this.logger);
    }
  }

  async get(key, options = {}) {
    const startTime = Date.now();
    this.metrics.totalRequests++;

    try {
      // Determine cache tier based on access pattern
      const tier = this.determineCacheTier(key, options);
      const cache = this.caches.get(`tier_${tier}`) || this.caches.get('memory');

      const result = await cache.get(key);

      if (result !== null) {
        this.recordCacheHit(tier, Date.now() - startTime);
        this.updateAccessPattern(key, 'hit');

        // Promote to higher tier if frequently accessed
        if (tier !== 'hot' && this.shouldPromote(key)) {
          await this.promote(key, result, tier);
        }

        return this.deserializeValue(result);
      } else {
        this.recordCacheMiss(tier, Date.now() - startTime);
        this.updateAccessPattern(key, 'miss');
        return null;
      }
    } catch (error) {
      this.logger.error('Cache get error', { key, error: error.message });
      return null;
    }
  }

  async set(key, value, options = {}) {
    const startTime = Date.now();

    try {
      const tier = options.tier || this.determineCacheTier(key, options);
      const cache = this.caches.get(`tier_${tier}`) || this.caches.get('memory');
      const ttl = options.ttl || this.options.tiers[tier]?.ttl;

      const serializedValue = await this.serializeValue(value, options);
      await cache.set(key, serializedValue, ttl);

      this.updateAccessPattern(key, 'set');
      this.logger.debug('Cache set', {
        key,
        tier,
        size: JSON.stringify(serializedValue).length,
        ttl,
      });
    } catch (error) {
      this.logger.error('Cache set error', { key, error: error.message });
    }
  }

  async getMultiple(keys, options = {}) {
    const results = new Map();
    const promises = keys.map(async key => {
      const value = await this.get(key, options);
      return [key, value];
    });

    const resolved = await Promise.all(promises);
    for (const [key, value] of resolved) {
      if (value !== null) {
        results.set(key, value);
      }
    }

    return results;
  }

  async setMultiple(entries, options = {}) {
    const promises = Array.from(entries.entries()).map(([key, value]) => this.set(key, value, options));

    await Promise.all(promises);
  }

  async delete(key, options = {}) {
    try {
      // Delete from all caches to ensure consistency
      const deletePromises = [];

      for (const [name, cache] of this.caches) {
        deletePromises.push(cache.delete(key));
      }

      await Promise.all(deletePromises);
      this.clearAccessPattern(key);

      this.logger.debug('Cache delete', { key });
    } catch (error) {
      this.logger.error('Cache delete error', { key, error: error.message });
    }
  }

  async clear(cacheNames = []) {
    try {
      const cachesToClear = cacheNames.length > 0 ? cacheNames : Array.from(this.caches.keys());

      for (const cacheName of cachesToClear) {
        const cache = this.caches.get(cacheName);
        if (cache) {
          await cache.clear();
        }
      }

      this.accessPatterns?.clear();
      this.logger.info('Cache cleared', { caches: cachesToClear });
    } catch (error) {
      this.logger.error('Cache clear error', { error: error.message });
    }
  }

  determineCacheTier(key, options = {}) {
    if (options.tier) return options.tier;

    // Initialize access patterns if not exists
    if (!this.accessPatterns) {
      this.accessPatterns = new Map();
    }

    const pattern = this.accessPatterns.get(key);

    if (!pattern) {
      return 'warm'; // Default tier for new keys
    }

    const { accessCount, lastAccess, frequency } = pattern;
    const timeSinceLastAccess = Date.now() - lastAccess;

    // Hot tier: frequently accessed and recently accessed
    if (accessCount > 10 && timeSinceLastAccess < 300000 && frequency > 0.1) {
      return 'hot';
    }

    // Cold tier: infrequently accessed or old
    if (accessCount < 3 || timeSinceLastAccess > 1800000 || frequency < 0.01) {
      return 'cold';
    }

    // Warm tier: default
    return 'warm';
  }

  shouldPromote(key) {
    const pattern = this.accessPatterns?.get(key);
    if (!pattern) return false;

    return pattern.accessCount > 5 && pattern.frequency > 0.05;
  }

  async promote(key, value, fromTier) {
    const targetTier = fromTier === 'cold' ? 'warm' : 'hot';
    await this.set(key, value, { tier: targetTier });

    this.logger.debug('Cache promotion', { key, from: fromTier, to: targetTier });
  }

  updateAccessPattern(key, operation) {
    if (!this.accessPatterns) {
      this.accessPatterns = new Map();
    }

    const now = Date.now();
    const existing = this.accessPatterns.get(key) || {
      accessCount: 0,
      firstAccess: now,
      lastAccess: now,
      frequency: 0,
      operations: { hit: 0, miss: 0, set: 0 },
    };

    existing.accessCount++;
    existing.lastAccess = now;
    existing.operations[operation]++;

    // Calculate frequency (accesses per hour)
    const timeSpan = now - existing.firstAccess;
    existing.frequency = timeSpan > 0 ? existing.accessCount / (timeSpan / 3600000) : 0;

    this.accessPatterns.set(key, existing);
  }

  clearAccessPattern(key) {
    this.accessPatterns?.delete(key);
  }

  recordCacheHit(tier, responseTime) {
    if (!this.metrics.hits.has(tier)) {
      this.metrics.hits.set(tier, 0);
    }
    this.metrics.hits.set(tier, this.metrics.hits.get(tier) + 1);
    this.updateAverageResponseTime(responseTime);
  }

  recordCacheMiss(tier, responseTime) {
    if (!this.metrics.misses.has(tier)) {
      this.metrics.misses.set(tier, 0);
    }
    this.metrics.misses.set(tier, this.metrics.misses.get(tier) + 1);
    this.updateAverageResponseTime(responseTime);
  }

  updateAverageResponseTime(responseTime) {
    const { totalRequests, averageResponseTime } = this.metrics;
    this.metrics.averageResponseTime = (averageResponseTime * (totalRequests - 1) + responseTime) / totalRequests;
  }

  async serializeValue(value, options = {}) {
    let serialized = value;

    // Compress large values
    if (options.compress || JSON.stringify(value).length > this.options.compressionThreshold) {
      try {
        serialized = await this.compress(value);
        this.metrics.compressionSaved += JSON.stringify(value).length - JSON.stringify(serialized).length;
      } catch (error) {
        this.logger.warn('Compression failed, storing uncompressed', { error: error.message });
      }
    }

    return {
      data: serialized,
      compressed: serialized !== value,
      timestamp: Date.now(),
      metadata: options.metadata || {},
    };
  }

  deserializeValue(serializedValue) {
    if (!serializedValue || typeof serializedValue !== 'object') {
      return serializedValue;
    }

    const { data, compressed } = serializedValue;

    if (compressed) {
      try {
        return this.decompress(data);
      } catch (error) {
        this.logger.warn('Decompression failed', { error: error.message });
        return data;
      }
    }

    return data;
  }

  async compress(data) {
    const zlib = await import('zlib');
    const jsonString = JSON.stringify(data);

    return new Promise((resolve, reject) => {
      zlib.gzip(Buffer.from(jsonString), (error, result) => {
        if (error) {
          reject(error);
        } else {
          resolve({
            compressed: result.toString('base64'),
            originalSize: jsonString.length,
            compressedSize: result.length,
          });
        }
      });
    });
  }

  async decompress(compressedData) {
    const zlib = await import('zlib');

    return new Promise((resolve, reject) => {
      const buffer = Buffer.from(compressedData.compressed, 'base64');
      zlib.gunzip(buffer, (error, result) => {
        if (error) {
          reject(error);
        } else {
          try {
            resolve(JSON.parse(result.toString()));
          } catch (parseError) {
            reject(parseError);
          }
        }
      });
    });
  }

  startCleanupInterval() {
    this.cleanupInterval = setInterval(async () => {
      await this.performCleanup();
    }, this.options.cleanupInterval);
  }

  async performCleanup() {
    let totalCleaned = 0;

    for (const [name, cache] of this.caches) {
      if (cache.cleanup) {
        const cleaned = await cache.cleanup();
        totalCleaned += cleaned;
      }
    }

    // Clean up access patterns
    if (this.accessPatterns) {
      const cutoff = Date.now() - 24 * 60 * 60 * 1000; // 24 hours
      let patternsRemoved = 0;

      for (const [key, pattern] of this.accessPatterns) {
        if (pattern.lastAccess < cutoff) {
          this.accessPatterns.delete(key);
          patternsRemoved++;
        }
      }

      if (patternsRemoved > 0) {
        this.logger.debug('Access patterns cleaned up', { removed: patternsRemoved });
      }
    }

    if (totalCleaned > 0) {
      this.logger.debug('Cache cleanup completed', { entriesRemoved: totalCleaned });
    }
  }

  startMetricsCollection() {
    this.metricsInterval = setInterval(() => {
      this.logMetrics();
    }, 300000); // Every 5 minutes
  }

  logMetrics() {
    const totalHits = Array.from(this.metrics.hits.values()).reduce((sum, val) => sum + val, 0);
    const totalMisses = Array.from(this.metrics.misses.values()).reduce((sum, val) => sum + val, 0);
    const hitRate = totalHits + totalMisses > 0 ? totalHits / (totalHits + totalMisses) : 0;

    this.logger.info('Cache metrics', {
      hitRate: `${(hitRate * 100).toFixed(2)}%`,
      totalRequests: this.metrics.totalRequests,
      averageResponseTime: `${this.metrics.averageResponseTime.toFixed(2)}ms`,
      compressionSaved: `${Math.round(this.metrics.compressionSaved / 1024)}KB`,
      activeCaches: this.caches.size,
      accessPatterns: this.accessPatterns?.size || 0,
    });
  }

  getMetrics() {
    const cacheStats = new Map();

    for (const [name, cache] of this.caches) {
      cacheStats.set(name, cache.getStats ? cache.getStats() : { size: 0 });
    }

    return {
      hits: Object.fromEntries(this.metrics.hits),
      misses: Object.fromEntries(this.metrics.misses),
      totalRequests: this.metrics.totalRequests,
      averageResponseTime: this.metrics.averageResponseTime,
      compressionSaved: this.metrics.compressionSaved,
      cacheStats: Object.fromEntries(cacheStats),
      accessPatterns: this.accessPatterns?.size || 0,
    };
  }

  // Express middleware
  middleware(options = {}) {
    return async (req, res, next) => {
      const cacheKey = options.keyGenerator
        ? options.keyGenerator(req)
        : `${req.method}:${req.url}:${JSON.stringify(req.query)}`;

      const cached = await this.get(cacheKey);

      if (cached && !options.bypassCache) {
        res.set('X-Cache', 'HIT');
        return res.json(cached);
      }

      res.set('X-Cache', 'MISS');

      // Override res.json to cache the response
      const originalJson = res.json;
      res.json = data => {
        this.set(cacheKey, data, options).catch(error => {
          this.logger.warn('Cache middleware set error', { error: error.message });
        });
        return originalJson.call(res, data);
      };

      next();
    };
  }

  async cleanup() {
    this.logger.info('Cleaning up Advanced Cache Manager...');

    if (this.cleanupInterval) clearInterval(this.cleanupInterval);
    if (this.metricsInterval) clearInterval(this.metricsInterval);

    // Cleanup all caches
    for (const [name, cache] of this.caches) {
      if (cache.cleanup) {
        await cache.cleanup();
      }
    }

    this.caches.clear();
    this.accessPatterns?.clear();

    this.logger.info('Advanced Cache Manager cleanup completed');
  }
}

// Cache implementation classes
class MemoryCache {
  constructor(maxSize = 1000, ttl = 300000, logger) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.ttl = ttl;
    this.logger = logger;
  }

  async get(key) {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() > entry.expiry) {
      this.cache.delete(key);
      return null;
    }

    return entry.value;
  }

  async set(key, value, ttl = this.ttl) {
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, {
      value,
      expiry: Date.now() + ttl,
      created: Date.now(),
    });
  }

  async delete(key) {
    this.cache.delete(key);
  }

  async clear() {
    this.cache.clear();
  }

  async cleanup() {
    const now = Date.now();
    let removed = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiry) {
        this.cache.delete(key);
        removed++;
      }
    }

    return removed;
  }

  getStats() {
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      utilization: ((this.cache.size / this.maxSize) * 100).toFixed(2) + '%',
    };
  }
}

class LRUCache extends MemoryCache {
  async get(key) {
    const value = await super.get(key);
    if (value !== null) {
      // Move to end (most recently used)
      const entry = this.cache.get(key);
      this.cache.delete(key);
      this.cache.set(key, entry);
    }
    return value;
  }
}

class LFUCache {
  constructor(maxSize = 1000, ttl = 300000, logger) {
    this.cache = new Map();
    this.frequencies = new Map();
    this.maxSize = maxSize;
    this.ttl = ttl;
    this.logger = logger;
  }

  async get(key) {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() > entry.expiry) {
      this.cache.delete(key);
      this.frequencies.delete(key);
      return null;
    }

    // Increment frequency
    this.frequencies.set(key, (this.frequencies.get(key) || 0) + 1);

    return entry.value;
  }

  async set(key, value, ttl = this.ttl) {
    if (this.cache.size >= this.maxSize) {
      // Remove least frequently used
      let minFreq = Infinity;
      let leastUsedKey = null;

      for (const [k, freq] of this.frequencies) {
        if (freq < minFreq) {
          minFreq = freq;
          leastUsedKey = k;
        }
      }

      if (leastUsedKey) {
        this.cache.delete(leastUsedKey);
        this.frequencies.delete(leastUsedKey);
      }
    }

    this.cache.set(key, {
      value,
      expiry: Date.now() + ttl,
      created: Date.now(),
    });

    this.frequencies.set(key, 1);
  }

  async delete(key) {
    this.cache.delete(key);
    this.frequencies.delete(key);
  }

  async clear() {
    this.cache.clear();
    this.frequencies.clear();
  }

  async cleanup() {
    const now = Date.now();
    let removed = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiry) {
        this.cache.delete(key);
        this.frequencies.delete(key);
        removed++;
      }
    }

    return removed;
  }

  getStats() {
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      utilization: ((this.cache.size / this.maxSize) * 100).toFixed(2) + '%',
      averageFrequency:
        this.frequencies.size > 0
          ? Array.from(this.frequencies.values()).reduce((sum, freq) => sum + freq, 0) / this.frequencies.size
          : 0,
    };
  }
}

class FIFOCache extends MemoryCache {
  constructor(maxSize, ttl, logger) {
    super(maxSize, ttl, logger);
    this.insertionOrder = [];
  }

  async set(key, value, ttl = this.ttl) {
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      const oldestKey = this.insertionOrder.shift();
      this.cache.delete(oldestKey);
    }

    if (!this.cache.has(key)) {
      this.insertionOrder.push(key);
    }

    await super.set(key, value, ttl);
  }

  async delete(key) {
    await super.delete(key);
    const index = this.insertionOrder.indexOf(key);
    if (index > -1) {
      this.insertionOrder.splice(index, 1);
    }
  }

  async clear() {
    await super.clear();
    this.insertionOrder = [];
  }
}

// Placeholder for Redis cache - would require actual Redis connection
class RedisCache {
  constructor(config, logger) {
    this.config = config;
    this.logger = logger;
    // In real implementation, initialize Redis client here
  }

  async get(key) {
    // Redis implementation
    return null;
  }

  async set(key, value, ttl) {
    // Redis implementation
  }

  async delete(key) {
    // Redis implementation
  }

  async clear() {
    // Redis implementation
  }
}

export function createAdvancedCacheManager(logger, options = {}) {
  return new AdvancedCacheManager(logger, options);
}
