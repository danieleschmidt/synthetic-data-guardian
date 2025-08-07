/**
 * Performance Optimization Engine
 * Provides caching, connection pooling, and performance monitoring
 */

import { EventEmitter } from 'events';
import { Worker } from 'worker_threads';
import cluster from 'cluster';
import os from 'os';

export class PerformanceOptimizer extends EventEmitter {
  constructor(logger, options = {}) {
    super();
    this.logger = logger;
    this.options = {
      enableCaching: options.enableCaching !== false,
      cacheSize: options.cacheSize || 1000,
      cacheTTL: options.cacheTTL || 300000, // 5 minutes
      enableConnectionPooling: options.enableConnectionPooling !== false,
      maxConnections: options.maxConnections || 100,
      enableWorkerPool: options.enableWorkerPool !== false,
      maxWorkers: options.maxWorkers || Math.min(os.cpus().length, 4),
      enableCluster: options.enableCluster && cluster.isMaster,
      clusterSize: options.clusterSize || os.cpus().length,
      enableCompression: options.enableCompression !== false,
      compressionLevel: options.compressionLevel || 6,
      enableMemoryMonitoring: options.enableMemoryMonitoring !== false,
      memoryThreshold: options.memoryThreshold || 0.8,
      ...options,
    };

    this.cache = new Map();
    this.connectionPools = new Map();
    this.workerPool = [];
    this.availableWorkers = [];
    this.performanceMetrics = {
      cacheHits: 0,
      cacheMisses: 0,
      totalRequests: 0,
      averageResponseTime: 0,
      connectionPoolStats: {},
      workerStats: { active: 0, idle: 0, total: 0 },
    };

    this.initialize();
  }

  async initialize() {
    try {
      this.logger.info('Initializing Performance Optimizer...');

      if (this.options.enableCaching) {
        this.initializeCache();
      }

      if (this.options.enableConnectionPooling) {
        this.initializeConnectionPools();
      }

      if (this.options.enableWorkerPool) {
        await this.initializeWorkerPool();
      }

      if (this.options.enableMemoryMonitoring) {
        this.startMemoryMonitoring();
      }

      // Start performance monitoring
      this.startPerformanceMonitoring();

      this.logger.info('Performance Optimizer initialized successfully', {
        caching: this.options.enableCaching,
        connectionPooling: this.options.enableConnectionPooling,
        workerPool: this.options.enableWorkerPool,
        maxWorkers: this.options.maxWorkers,
      });
    } catch (error) {
      this.logger.error('Performance Optimizer initialization failed', {
        error: error.message,
      });
      throw error;
    }
  }

  initializeCache() {
    this.logger.debug('Initializing performance cache', {
      size: this.options.cacheSize,
      ttl: this.options.cacheTTL,
    });

    // Setup cache cleanup interval
    this.cacheCleanupInterval = setInterval(() => {
      this.cleanupExpiredCacheEntries();
    }, 60000); // Every minute
  }

  initializeConnectionPools() {
    this.logger.debug('Initializing connection pools');

    // Generic connection pool factory
    this.connectionPools.set('database', {
      connections: [],
      available: [],
      pending: [],
      maxConnections: this.options.maxConnections,
      currentConnections: 0,
    });
  }

  async initializeWorkerPool() {
    this.logger.info('Initializing worker pool', { maxWorkers: this.options.maxWorkers });

    for (let i = 0; i < this.options.maxWorkers; i++) {
      try {
        const worker = new Worker(new URL('./worker.js', import.meta.url), {
          workerData: { workerId: i },
        });

        worker.on('error', error => {
          this.logger.error('Worker error', { workerId: i, error: error.message });
          this.replaceWorker(i);
        });

        worker.on('exit', code => {
          if (code !== 0) {
            this.logger.warn('Worker exited with error', { workerId: i, code });
            this.replaceWorker(i);
          }
        });

        this.workerPool[i] = worker;
        this.availableWorkers.push(worker);
      } catch (error) {
        this.logger.error('Failed to create worker', { workerId: i, error: error.message });
      }
    }

    this.performanceMetrics.workerStats.total = this.workerPool.length;
    this.performanceMetrics.workerStats.idle = this.availableWorkers.length;
  }

  async replaceWorker(workerId) {
    try {
      const oldWorker = this.workerPool[workerId];
      if (oldWorker) {
        await oldWorker.terminate();
      }

      const newWorker = new Worker(new URL('./worker.js', import.meta.url), {
        workerData: { workerId },
      });

      this.workerPool[workerId] = newWorker;
      this.availableWorkers.push(newWorker);

      this.logger.info('Worker replaced successfully', { workerId });
    } catch (error) {
      this.logger.error('Failed to replace worker', { workerId, error: error.message });
    }
  }

  startMemoryMonitoring() {
    this.memoryMonitorInterval = setInterval(() => {
      const memoryUsage = process.memoryUsage();
      const totalMemory = memoryUsage.heapTotal;
      const usedMemory = memoryUsage.heapUsed;
      const memoryUtilization = usedMemory / totalMemory;

      if (memoryUtilization > this.options.memoryThreshold) {
        this.logger.warn('High memory usage detected', {
          utilization: `${(memoryUtilization * 100).toFixed(2)}%`,
          heapUsed: `${Math.round(usedMemory / 1024 / 1024)}MB`,
          heapTotal: `${Math.round(totalMemory / 1024 / 1024)}MB`,
        });

        this.emit('memoryPressure', { memoryUtilization, memoryUsage });
        this.optimizeMemoryUsage();
      }
    }, 30000); // Every 30 seconds
  }

  startPerformanceMonitoring() {
    this.performanceInterval = setInterval(() => {
      this.emitPerformanceMetrics();
    }, 60000); // Every minute
  }

  optimizeMemoryUsage() {
    // Force garbage collection if available
    if (global.gc) {
      this.logger.info('Forcing garbage collection');
      global.gc();
    }

    // Clear expired cache entries
    this.cleanupExpiredCacheEntries();

    // Emit memory optimization event
    this.emit('memoryOptimized');
  }

  // Caching methods
  async get(key, fallback = null) {
    const startTime = Date.now();
    const cacheKey = this.generateCacheKey(key);

    if (this.cache.has(cacheKey)) {
      const entry = this.cache.get(cacheKey);
      if (Date.now() < entry.expiry) {
        this.performanceMetrics.cacheHits++;
        this.logger.debug('Cache hit', { key: cacheKey, ttl: entry.expiry - Date.now() });
        return entry.data;
      } else {
        this.cache.delete(cacheKey);
      }
    }

    this.performanceMetrics.cacheMisses++;

    if (fallback && typeof fallback === 'function') {
      const data = await fallback();
      await this.set(key, data);
      return data;
    }

    return null;
  }

  async set(key, data, ttl = this.options.cacheTTL) {
    const cacheKey = this.generateCacheKey(key);

    // Implement LRU eviction if cache is full
    if (this.cache.size >= this.options.cacheSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(cacheKey, {
      data,
      expiry: Date.now() + ttl,
      created: Date.now(),
    });

    this.logger.debug('Cache set', { key: cacheKey, ttl });
  }

  generateCacheKey(key) {
    if (typeof key === 'object') {
      return JSON.stringify(key);
    }
    return String(key);
  }

  cleanupExpiredCacheEntries() {
    const now = Date.now();
    let expiredCount = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now >= entry.expiry) {
        this.cache.delete(key);
        expiredCount++;
      }
    }

    if (expiredCount > 0) {
      this.logger.debug('Cache cleanup completed', {
        expiredEntries: expiredCount,
        remainingEntries: this.cache.size,
      });
    }
  }

  // Worker pool methods
  async executeInWorker(taskType, data, options = {}) {
    if (!this.options.enableWorkerPool || this.availableWorkers.length === 0) {
      throw new Error('No workers available');
    }

    const worker = this.availableWorkers.pop();
    this.performanceMetrics.workerStats.active++;
    this.performanceMetrics.workerStats.idle--;

    return new Promise((resolve, reject) => {
      const timeout = options.timeout || 30000;
      const timeoutId = setTimeout(() => {
        reject(new Error('Worker task timeout'));
      }, timeout);

      worker.once('message', result => {
        clearTimeout(timeoutId);

        // Return worker to pool
        this.availableWorkers.push(worker);
        this.performanceMetrics.workerStats.active--;
        this.performanceMetrics.workerStats.idle++;

        if (result.error) {
          reject(new Error(result.error));
        } else {
          resolve(result.data);
        }
      });

      worker.once('error', error => {
        clearTimeout(timeoutId);
        reject(error);
      });

      worker.postMessage({ taskType, data, options });
    });
  }

  // Connection pooling methods
  async getConnection(poolName = 'database') {
    const pool = this.connectionPools.get(poolName);
    if (!pool) {
      throw new Error(`Connection pool '${poolName}' not found`);
    }

    if (pool.available.length > 0) {
      return pool.available.pop();
    }

    if (pool.currentConnections < pool.maxConnections) {
      // Create new connection (mock implementation)
      const connection = this.createConnection(poolName);
      pool.currentConnections++;
      return connection;
    }

    // Wait for available connection
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Connection timeout'));
      }, 10000);

      pool.pending.push({ resolve, reject, timeout });
    });
  }

  releaseConnection(connection, poolName = 'database') {
    const pool = this.connectionPools.get(poolName);
    if (!pool) {
      return;
    }

    if (pool.pending.length > 0) {
      const { resolve, timeout } = pool.pending.shift();
      clearTimeout(timeout);
      resolve(connection);
    } else {
      pool.available.push(connection);
    }
  }

  createConnection(poolName) {
    // Mock connection object
    return {
      id: Date.now() + Math.random(),
      poolName,
      created: Date.now(),
      query: async (sql, params) => {
        // Mock query implementation
        await new Promise(resolve => setTimeout(resolve, 10));
        return { rows: [], affectedRows: 0 };
      },
    };
  }

  // Performance tracking
  trackRequest(req, res, next) {
    const startTime = Date.now();
    this.performanceMetrics.totalRequests++;

    res.on('finish', () => {
      const responseTime = Date.now() - startTime;
      this.updateAverageResponseTime(responseTime);

      this.logger.debug('Request performance', {
        method: req.method,
        url: req.url,
        responseTime,
        statusCode: res.statusCode,
      });
    });

    next();
  }

  updateAverageResponseTime(responseTime) {
    const { totalRequests, averageResponseTime } = this.performanceMetrics;
    this.performanceMetrics.averageResponseTime =
      (averageResponseTime * (totalRequests - 1) + responseTime) / totalRequests;
  }

  // Compression utilities
  async compress(data, algorithm = 'gzip') {
    if (!this.options.enableCompression) {
      return data;
    }

    try {
      const zlib = await import('zlib');
      const compressed = await new Promise((resolve, reject) => {
        zlib[algorithm](
          Buffer.from(JSON.stringify(data)),
          {
            level: this.options.compressionLevel,
          },
          (error, result) => {
            if (error) reject(error);
            else resolve(result);
          },
        );
      });

      this.logger.debug('Data compressed', {
        originalSize: JSON.stringify(data).length,
        compressedSize: compressed.length,
        ratio: (compressed.length / JSON.stringify(data).length).toFixed(2),
      });

      return compressed;
    } catch (error) {
      this.logger.warn('Compression failed', { error: error.message });
      return data;
    }
  }

  async decompress(compressedData, algorithm = 'gzip') {
    try {
      const zlib = await import('zlib');
      const decompressed = await new Promise((resolve, reject) => {
        zlib[algorithm === 'gzip' ? 'gunzip' : 'inflate'](compressedData, (error, result) => {
          if (error) reject(error);
          else resolve(JSON.parse(result.toString()));
        });
      });

      return decompressed;
    } catch (error) {
      this.logger.warn('Decompression failed', { error: error.message });
      return compressedData;
    }
  }

  // Performance metrics
  getPerformanceMetrics() {
    return {
      ...this.performanceMetrics,
      cacheStats: {
        size: this.cache.size,
        hitRate:
          this.performanceMetrics.totalRequests > 0
            ? this.performanceMetrics.cacheHits / this.performanceMetrics.totalRequests
            : 0,
        maxSize: this.options.cacheSize,
      },
      memoryStats: process.memoryUsage(),
      uptime: process.uptime(),
    };
  }

  emitPerformanceMetrics() {
    const metrics = this.getPerformanceMetrics();
    this.emit('performanceMetrics', metrics);

    this.logger.info('Performance metrics', {
      totalRequests: metrics.totalRequests,
      averageResponseTime: Math.round(metrics.averageResponseTime),
      cacheHitRate: (metrics.cacheStats.hitRate * 100).toFixed(1) + '%',
      activeWorkers: metrics.workerStats.active,
      memoryUsage: Math.round(metrics.memoryStats.heapUsed / 1024 / 1024) + 'MB',
    });
  }

  // Cluster management
  static setupCluster(workers = os.cpus().length) {
    if (cluster.isMaster) {
      console.log(`Setting up cluster with ${workers} workers`);

      for (let i = 0; i < workers; i++) {
        cluster.fork();
      }

      cluster.on('exit', (worker, code, signal) => {
        console.log(`Worker ${worker.process.pid} died. Restarting...`);
        cluster.fork();
      });

      return true;
    }
    return false;
  }

  // Express middleware factory
  middleware() {
    return {
      performance: this.trackRequest.bind(this),
      cache: ttl => async (req, res, next) => {
        const cacheKey = `${req.method}:${req.url}:${JSON.stringify(req.query)}`;
        const cached = await this.get(cacheKey);

        if (cached) {
          return res.json(cached);
        }

        const originalJson = res.json;
        res.json = function (data) {
          this.cache.set(cacheKey, data, ttl).catch(() => {});
          return originalJson.call(this, data);
        }.bind(this);

        next();
      },
    };
  }

  async cleanup() {
    this.logger.info('Cleaning up Performance Optimizer...');

    // Clear intervals
    if (this.cacheCleanupInterval) clearInterval(this.cacheCleanupInterval);
    if (this.memoryMonitorInterval) clearInterval(this.memoryMonitorInterval);
    if (this.performanceInterval) clearInterval(this.performanceInterval);

    // Terminate workers
    for (const worker of this.workerPool) {
      if (worker) {
        await worker.terminate();
      }
    }

    // Clear cache
    this.cache.clear();

    // Close connection pools
    for (const [name, pool] of this.connectionPools) {
      pool.available = [];
      pool.pending.forEach(({ reject, timeout }) => {
        clearTimeout(timeout);
        reject(new Error('Pool closing'));
      });
    }

    this.removeAllListeners();
    this.logger.info('Performance Optimizer cleanup completed');
  }
}

export function createPerformanceOptimizer(logger, options = {}) {
  return new PerformanceOptimizer(logger, options);
}
