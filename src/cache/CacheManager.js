/**
 * CacheManager - Centralized caching layer with Redis backend
 */

export class CacheManager {
  constructor(db, logger) {
    this.redis = db;
    this.logger = logger;
    this.defaultTTL = 3600; // 1 hour in seconds
    this.keyPrefix = 'sdg:'; // Synthetic Data Guardian prefix
  }

  // Generate cache key with prefix
  generateKey(namespace, key) {
    return `${this.keyPrefix}${namespace}:${key}`;
  }

  // Basic cache operations
  async get(namespace, key) {
    try {
      const cacheKey = this.generateKey(namespace, key);
      const value = await this.redis.get(cacheKey);
      
      if (value) {
        this.logger.debug('Cache hit', { namespace, key });
        return JSON.parse(value);
      }
      
      this.logger.debug('Cache miss', { namespace, key });
      return null;
    } catch (error) {
      this.logger.error('Cache get failed', { namespace, key, error: error.message });
      return null;
    }
  }

  async set(namespace, key, value, ttl = null) {
    try {
      const cacheKey = this.generateKey(namespace, key);
      const serializedValue = JSON.stringify(value);
      const expiry = ttl || this.defaultTTL;
      
      await this.redis.set(cacheKey, serializedValue, expiry);
      this.logger.debug('Cache set', { namespace, key, ttl: expiry });
      return true;
    } catch (error) {
      this.logger.error('Cache set failed', { namespace, key, error: error.message });
      return false;
    }
  }

  async del(namespace, key) {
    try {
      const cacheKey = this.generateKey(namespace, key);
      const result = await this.redis.del(cacheKey);
      this.logger.debug('Cache delete', { namespace, key, deleted: result > 0 });
      return result > 0;
    } catch (error) {
      this.logger.error('Cache delete failed', { namespace, key, error: error.message });
      return false;
    }
  }

  async exists(namespace, key) {
    try {
      const cacheKey = this.generateKey(namespace, key);
      const result = await this.redis.exists(cacheKey);
      return result > 0;
    } catch (error) {
      this.logger.error('Cache exists check failed', { namespace, key, error: error.message });
      return false;
    }
  }

  // Hash operations for complex data structures
  async hset(namespace, hashKey, field, value) {
    try {
      const cacheKey = this.generateKey(namespace, hashKey);
      const serializedValue = JSON.stringify(value);
      await this.redis.hset(cacheKey, field, serializedValue);
      this.logger.debug('Cache hset', { namespace, hashKey, field });
      return true;
    } catch (error) {
      this.logger.error('Cache hset failed', { namespace, hashKey, field, error: error.message });
      return false;
    }
  }

  async hget(namespace, hashKey, field) {
    try {
      const cacheKey = this.generateKey(namespace, hashKey);
      const value = await this.redis.hget(cacheKey, field);
      
      if (value) {
        this.logger.debug('Cache hget hit', { namespace, hashKey, field });
        return JSON.parse(value);
      }
      
      this.logger.debug('Cache hget miss', { namespace, hashKey, field });
      return null;
    } catch (error) {
      this.logger.error('Cache hget failed', { namespace, hashKey, field, error: error.message });
      return null;
    }
  }

  async hgetall(namespace, hashKey) {
    try {
      const cacheKey = this.generateKey(namespace, hashKey);
      const hash = await this.redis.hgetall(cacheKey);
      
      if (Object.keys(hash).length > 0) {
        const parsed = {};
        for (const [field, value] of Object.entries(hash)) {
          parsed[field] = JSON.parse(value);
        }
        this.logger.debug('Cache hgetall hit', { namespace, hashKey, fields: Object.keys(parsed).length });
        return parsed;
      }
      
      this.logger.debug('Cache hgetall miss', { namespace, hashKey });
      return {};
    } catch (error) {
      this.logger.error('Cache hgetall failed', { namespace, hashKey, error: error.message });
      return {};
    }
  }

  // High-level caching methods for specific use cases

  // Pipeline caching
  async cachePipeline(pipelineId, pipelineData, ttl = 3600) {
    return await this.set('pipelines', pipelineId, pipelineData, ttl);
  }

  async getCachedPipeline(pipelineId) {
    return await this.get('pipelines', pipelineId);
  }

  async invalidatePipeline(pipelineId) {
    return await this.del('pipelines', pipelineId);
  }

  // Generation result caching
  async cacheGenerationResult(taskId, result, ttl = 7200) { // 2 hours
    return await this.set('results', taskId, result, ttl);
  }

  async getCachedResult(taskId) {
    return await this.get('results', taskId);
  }

  async invalidateResult(taskId) {
    return await this.del('results', taskId);
  }

  // Validation report caching
  async cacheValidationReport(resultId, report, ttl = 3600) {
    return await this.set('validation', resultId, report, ttl);
  }

  async getCachedValidationReport(resultId) {
    return await this.get('validation', resultId);
  }

  // Quality metrics caching
  async cacheQualityMetrics(dataHash, metrics, ttl = 86400) { // 24 hours
    return await this.set('quality', dataHash, metrics, ttl);
  }

  async getCachedQualityMetrics(dataHash) {
    return await this.get('quality', dataHash);
  }

  // Privacy analysis caching
  async cachePrivacyAnalysis(dataHash, analysis, ttl = 86400) {
    return await this.set('privacy', dataHash, analysis, ttl);
  }

  async getCachedPrivacyAnalysis(dataHash) {
    return await this.get('privacy', dataHash);
  }

  // Session and rate limiting
  async setRateLimit(clientId, count, windowSeconds = 60) {
    const key = `ratelimit:${clientId}`;
    try {
      await this.redis.set(this.generateKey('rate', key), count, windowSeconds);
      return true;
    } catch (error) {
      this.logger.error('Rate limit set failed', { clientId, error: error.message });
      return false;
    }
  }

  async getRateLimit(clientId) {
    const key = `ratelimit:${clientId}`;
    try {
      const count = await this.redis.get(this.generateKey('rate', key));
      return count ? parseInt(count) : 0;
    } catch (error) {
      this.logger.error('Rate limit get failed', { clientId, error: error.message });
      return 0;
    }
  }

  async incrementRateLimit(clientId, windowSeconds = 60) {
    const key = `ratelimit:${clientId}`;
    const cacheKey = this.generateKey('rate', key);
    
    try {
      const current = await this.redis.get(cacheKey);
      if (current) {
        return await this.redis.incr(cacheKey);
      } else {
        await this.redis.set(cacheKey, 1, windowSeconds);
        return 1;
      }
    } catch (error) {
      this.logger.error('Rate limit increment failed', { clientId, error: error.message });
      return 1;
    }
  }

  // Bulk operations
  async mget(namespace, keys) {
    try {
      const cacheKeys = keys.map(key => this.generateKey(namespace, key));
      const values = await this.redis.mget(...cacheKeys);
      
      const result = {};
      keys.forEach((key, index) => {
        if (values[index]) {
          result[key] = JSON.parse(values[index]);
        }
      });
      
      this.logger.debug('Cache mget', { namespace, requested: keys.length, found: Object.keys(result).length });
      return result;
    } catch (error) {
      this.logger.error('Cache mget failed', { namespace, keys, error: error.message });
      return {};
    }
  }

  async mset(namespace, keyValuePairs, ttl = null) {
    try {
      const operations = [];
      const expiry = ttl || this.defaultTTL;
      
      for (const [key, value] of Object.entries(keyValuePairs)) {
        const cacheKey = this.generateKey(namespace, key);
        const serializedValue = JSON.stringify(value);
        operations.push(['set', cacheKey, serializedValue, 'EX', expiry]);
      }
      
      // Use pipeline for better performance
      const pipeline = this.redis.pipeline();
      operations.forEach(op => pipeline.set(op[1], op[2], 'EX', op[4]));
      await pipeline.exec();
      
      this.logger.debug('Cache mset', { namespace, keys: Object.keys(keyValuePairs).length, ttl: expiry });
      return true;
    } catch (error) {
      this.logger.error('Cache mset failed', { namespace, error: error.message });
      return false;
    }
  }

  // Pattern-based operations
  async deletePattern(pattern) {
    try {
      const fullPattern = this.generateKey('*', pattern);
      const keys = await this.redis.keys(fullPattern);
      
      if (keys.length > 0) {
        await this.redis.del(...keys);
        this.logger.debug('Cache pattern delete', { pattern, deleted: keys.length });
        return keys.length;
      }
      
      return 0;
    } catch (error) {
      this.logger.error('Cache pattern delete failed', { pattern, error: error.message });
      return 0;
    }
  }

  async findKeys(pattern) {
    try {
      const fullPattern = this.generateKey('*', pattern);
      const keys = await this.redis.keys(fullPattern);
      return keys.map(key => key.replace(this.keyPrefix, ''));
    } catch (error) {
      this.logger.error('Cache find keys failed', { pattern, error: error.message });
      return [];
    }
  }

  // Cache statistics and monitoring
  async getStats() {
    try {
      const info = await this.redis.info('memory');
      const stats = {};
      
      info.split('\r\n').forEach(line => {
        if (line.includes(':')) {
          const [key, value] = line.split(':');
          stats[key] = value;
        }
      });
      
      return {
        memory: stats,
        keyCount: await this.getKeyCount(),
        uptime: stats.uptime_in_seconds
      };
    } catch (error) {
      this.logger.error('Cache stats failed', { error: error.message });
      return {};
    }
  }

  async getKeyCount() {
    try {
      const keys = await this.redis.keys(`${this.keyPrefix}*`);
      return keys.length;
    } catch (error) {
      this.logger.error('Cache key count failed', { error: error.message });
      return 0;
    }
  }

  // Cache warming and preloading
  async warmCache(namespace, dataLoader, keys, ttl = null) {
    this.logger.info('Warming cache', { namespace, keys: keys.length });
    
    const results = [];
    for (const key of keys) {
      try {
        const exists = await this.exists(namespace, key);
        if (!exists) {
          const data = await dataLoader(key);
          if (data) {
            await this.set(namespace, key, data, ttl);
            results.push({ key, status: 'loaded' });
          } else {
            results.push({ key, status: 'no_data' });
          }
        } else {
          results.push({ key, status: 'exists' });
        }
      } catch (error) {
        this.logger.error('Cache warm failed for key', { namespace, key, error: error.message });
        results.push({ key, status: 'error', error: error.message });
      }
    }
    
    this.logger.info('Cache warming completed', { 
      namespace, 
      total: keys.length,
      loaded: results.filter(r => r.status === 'loaded').length,
      exists: results.filter(r => r.status === 'exists').length,
      errors: results.filter(r => r.status === 'error').length
    });
    
    return results;
  }

  // Health check
  async healthCheck() {
    try {
      const start = Date.now();
      await this.redis.ping();
      const latency = Date.now() - start;
      
      const stats = await this.getStats();
      
      return {
        status: 'healthy',
        latency: latency,
        memory: stats.memory?.used_memory_human || 'unknown',
        keyCount: stats.keyCount || 0,
        uptime: stats.uptime || 0
      };
    } catch (error) {
      this.logger.error('Cache health check failed', { error: error.message });
      return {
        status: 'unhealthy',
        error: error.message
      };
    }
  }

  // Cleanup and maintenance
  async cleanup() {
    this.logger.info('Cache cleanup started');
    
    try {
      // Clean up expired keys (Redis handles this automatically, but we can force it)
      const expiredCount = await this.deletePattern('expired:*');
      
      // Clean up old rate limit entries
      const rateLimitCount = await this.deletePattern('rate:*');
      
      this.logger.info('Cache cleanup completed', { 
        expiredCount, 
        rateLimitCount 
      });
      
      return { expiredCount, rateLimitCount };
    } catch (error) {
      this.logger.error('Cache cleanup failed', { error: error.message });
      return { error: error.message };
    }
  }

  // Advanced caching patterns
  
  // Cache-aside pattern with automatic loading
  async getOrLoad(namespace, key, loader, ttl = null) {
    try {
      // Try to get from cache first
      let value = await this.get(namespace, key);
      
      if (value === null) {
        // Cache miss - load from source
        this.logger.debug('Cache miss, loading from source', { namespace, key });
        value = await loader(key);
        
        if (value !== null && value !== undefined) {
          // Store in cache for future requests
          await this.set(namespace, key, value, ttl);
        }
      }
      
      return value;
    } catch (error) {
      this.logger.error('Cache get-or-load failed', { namespace, key, error: error.message });
      throw error;
    }
  }

  // Write-through pattern
  async setAndPersist(namespace, key, value, persister, ttl = null) {
    try {
      // Persist to primary storage first
      await persister(key, value);
      
      // Then update cache
      await this.set(namespace, key, value, ttl);
      
      this.logger.debug('Write-through completed', { namespace, key });
      return true;
    } catch (error) {
      this.logger.error('Write-through failed', { namespace, key, error: error.message });
      throw error;
    }
  }
}