/**
 * Comprehensive Rate Limiting and Request Throttling
 */

import { HealthMetrics } from './metrics.js';

export class RateLimiter {
  constructor(logger, options = {}) {
    this.logger = logger;
    this.options = {
      windowMs: options.windowMs || 15 * 60 * 1000, // 15 minutes
      maxRequests: options.maxRequests || 100,
      skipSuccessfulRequests: options.skipSuccessfulRequests || false,
      skipFailedRequests: options.skipFailedRequests || false,
      keyGenerator: options.keyGenerator || this.defaultKeyGenerator,
      skip: options.skip || (() => false),
      handler: options.handler || this.defaultHandler.bind(this),
      onLimitReached: options.onLimitReached || this.defaultOnLimitReached.bind(this),
      store: options.store || new MemoryStore(),
    };

    this.stats = {
      totalRequests: 0,
      limitedRequests: 0,
      resetTime: Date.now() + this.options.windowMs,
    };
  }

  defaultKeyGenerator(req) {
    // Combine IP, User ID (if available), and User-Agent for more granular limiting
    const ip = req.ip || req.connection.remoteAddress || 'unknown';
    const userId = req.user?.id || 'anonymous';
    const userAgent = req.get('User-Agent') || 'unknown';

    return `${ip}:${userId}:${this.hashUserAgent(userAgent)}`;
  }

  hashUserAgent(userAgent) {
    // Simple hash to avoid extremely long keys while maintaining uniqueness
    let hash = 0;
    for (let i = 0; i < userAgent.length; i++) {
      const char = userAgent.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  async defaultHandler(req, res) {
    const retryAfter = Math.ceil(this.options.windowMs / 1000);

    res.status(429).json({
      success: false,
      error: {
        type: 'RATE_LIMIT_EXCEEDED',
        message: 'Too many requests, please try again later.',
        retryAfter: retryAfter,
        limit: this.options.maxRequests,
        window: this.options.windowMs,
        timestamp: new Date().toISOString(),
      },
    });
  }

  defaultOnLimitReached(req, key, totalHits) {
    this.logger.warn('Rate limit exceeded', {
      key: this.sanitizeKey(key),
      totalHits,
      limit: this.options.maxRequests,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      url: req.url,
      method: req.method,
    });

    HealthMetrics.recordRateLimit(key, totalHits);
  }

  sanitizeKey(key) {
    // Remove sensitive information from logs
    return key.replace(/:\w+:/g, ':***:');
  }

  middleware() {
    return async (req, res, next) => {
      try {
        if (this.options.skip(req)) {
          return next();
        }

        const key = this.options.keyGenerator(req);
        const now = Date.now();

        // Get current usage
        const current = await this.options.store.get(key);
        const resetTime = current ? current.resetTime : now + this.options.windowMs;

        // Check if window has expired
        if (now > resetTime) {
          await this.options.store.reset(key);
          const newData = {
            count: 0,
            resetTime: now + this.options.windowMs,
          };
          await this.options.store.set(key, newData);
        }

        // Get updated usage
        const usage = (await this.options.store.get(key)) || { count: 0, resetTime: now + this.options.windowMs };

        // Check if limit exceeded
        if (usage.count >= this.options.maxRequests) {
          this.stats.limitedRequests++;
          this.options.onLimitReached(req, key, usage.count);
          return this.options.handler(req, res);
        }

        // Increment usage
        usage.count++;
        await this.options.store.set(key, usage);
        this.stats.totalRequests++;

        // Add rate limit headers
        res.set({
          'X-RateLimit-Limit': this.options.maxRequests,
          'X-RateLimit-Remaining': Math.max(0, this.options.maxRequests - usage.count),
          'X-RateLimit-Reset': new Date(usage.resetTime).toISOString(),
        });

        next();
      } catch (error) {
        this.logger.error('Rate limiter error', { error: error.message });
        // Don't block requests on rate limiter errors
        next();
      }
    };
  }

  getStats() {
    return {
      ...this.stats,
      hitRate: this.stats.totalRequests > 0 ? this.stats.limitedRequests / this.stats.totalRequests : 0,
      storeStats: this.options.store.getStats ? this.options.store.getStats() : {},
    };
  }

  async reset() {
    await this.options.store.clear();
    this.stats.totalRequests = 0;
    this.stats.limitedRequests = 0;
    this.stats.resetTime = Date.now() + this.options.windowMs;
  }
}

export class MemoryStore {
  constructor() {
    this.data = new Map();
    this.cleanupInterval = setInterval(() => this.cleanup(), 5 * 60 * 1000); // Cleanup every 5 minutes
  }

  async get(key) {
    return this.data.get(key);
  }

  async set(key, value) {
    this.data.set(key, value);
  }

  async reset(key) {
    this.data.delete(key);
  }

  async clear() {
    this.data.clear();
  }

  cleanup() {
    const now = Date.now();
    for (const [key, value] of this.data) {
      if (value.resetTime < now) {
        this.data.delete(key);
      }
    }
  }

  getStats() {
    return {
      keys: this.data.size,
      memoryUsage: JSON.stringify([...this.data]).length,
    };
  }

  destroy() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    this.data.clear();
  }
}

// Adaptive rate limiter that adjusts based on system load
export class AdaptiveRateLimiter extends RateLimiter {
  constructor(logger, options = {}) {
    super(logger, options);
    this.systemLoadThreshold = options.systemLoadThreshold || 0.8;
    this.adaptiveFactor = options.adaptiveFactor || 0.5;
    this.checkInterval = options.checkInterval || 60000; // 1 minute

    this.originalMaxRequests = this.options.maxRequests;
    this.startSystemMonitoring();
  }

  startSystemMonitoring() {
    this.monitoringInterval = setInterval(() => {
      this.adjustRateLimit();
    }, this.checkInterval);
  }

  adjustRateLimit() {
    const memoryUsage = process.memoryUsage();
    const totalMemory = memoryUsage.heapTotal;
    const usedMemory = memoryUsage.heapUsed;
    const memoryUtilization = usedMemory / totalMemory;

    // Get event loop lag (approximate)
    const start = process.hrtime.bigint();
    setImmediate(() => {
      const lag = Number(process.hrtime.bigint() - start) / 1000000; // Convert to ms

      const systemLoad = Math.max(memoryUtilization, Math.min(lag / 100, 1));

      if (systemLoad > this.systemLoadThreshold) {
        // Reduce rate limit under high load
        const reductionFactor = 1 - (systemLoad - this.systemLoadThreshold) * this.adaptiveFactor;
        this.options.maxRequests = Math.max(
          Math.floor(this.originalMaxRequests * reductionFactor),
          Math.floor(this.originalMaxRequests * 0.1), // Never go below 10% of original
        );

        this.logger.warn('Adaptive rate limit reduced due to system load', {
          systemLoad,
          memoryUtilization,
          eventLoopLag: lag,
          newLimit: this.options.maxRequests,
          originalLimit: this.originalMaxRequests,
        });
      } else {
        // Restore original rate limit under normal conditions
        if (this.options.maxRequests !== this.originalMaxRequests) {
          this.options.maxRequests = this.originalMaxRequests;
          this.logger.info('Adaptive rate limit restored to normal', {
            systemLoad,
            limit: this.options.maxRequests,
          });
        }
      }
    });
  }

  destroy() {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
    }
    if (this.options.store.destroy) {
      this.options.store.destroy();
    }
  }
}

// Specialized rate limiters for different endpoints
export class TieredRateLimiter {
  constructor(logger) {
    this.logger = logger;
    this.limiters = new Map();
    this.tiers = {
      // Generation endpoints - more restrictive
      generation: { windowMs: 5 * 60 * 1000, maxRequests: 10 }, // 10 per 5 minutes
      // Validation endpoints - moderate
      validation: { windowMs: 60 * 1000, maxRequests: 30 }, // 30 per minute
      // Health/status endpoints - lenient
      health: { windowMs: 60 * 1000, maxRequests: 100 }, // 100 per minute
      // Default tier
      default: { windowMs: 15 * 60 * 1000, maxRequests: 100 }, // 100 per 15 minutes
    };
  }

  getLimiter(tier = 'default') {
    if (!this.limiters.has(tier)) {
      const config = this.tiers[tier] || this.tiers.default;
      this.limiters.set(tier, new AdaptiveRateLimiter(this.logger, config));
    }
    return this.limiters.get(tier);
  }

  middleware(tier = 'default') {
    const limiter = this.getLimiter(tier);
    return limiter.middleware();
  }

  getStats() {
    const stats = {};
    for (const [tier, limiter] of this.limiters) {
      stats[tier] = limiter.getStats();
    }
    return stats;
  }

  destroy() {
    for (const limiter of this.limiters.values()) {
      if (limiter.destroy) {
        limiter.destroy();
      }
    }
    this.limiters.clear();
  }
}

export function createRateLimiters(logger) {
  return new TieredRateLimiter(logger);
}
