/**
 * Advanced Load Balancer and Concurrent Processing Manager
 */

import { EventEmitter } from 'events';
import cluster from 'cluster';
import os from 'os';

export class LoadBalancer extends EventEmitter {
  constructor(logger, options = {}) {
    super();
    this.logger = logger;
    this.options = {
      // Load balancing algorithms
      algorithm: options.algorithm || 'round_robin', // round_robin, weighted, least_connections, ip_hash

      // Health checking
      healthCheck: {
        enabled: options.healthCheck?.enabled !== false,
        interval: options.healthCheck?.interval || 30000,
        timeout: options.healthCheck?.timeout || 5000,
        retries: options.healthCheck?.retries || 3,
        path: options.healthCheck?.path || '/health',
      },

      // Circuit breaker
      circuitBreaker: {
        enabled: options.circuitBreaker?.enabled !== false,
        threshold: options.circuitBreaker?.threshold || 5,
        timeout: options.circuitBreaker?.timeout || 60000,
        resetTimeout: options.circuitBreaker?.resetTimeout || 30000,
      },

      // Performance
      maxConcurrentRequests: options.maxConcurrentRequests || 1000,
      queueTimeout: options.queueTimeout || 30000,

      // Auto-scaling
      autoScaling: {
        enabled: options.autoScaling?.enabled === true,
        minInstances: options.autoScaling?.minInstances || 1,
        maxInstances: options.autoScaling?.maxInstances || os.cpus().length,
        scaleUpThreshold: options.autoScaling?.scaleUpThreshold || 0.8,
        scaleDownThreshold: options.autoScaling?.scaleDownThreshold || 0.3,
        cooldownPeriod: options.autoScaling?.cooldownPeriod || 300000, // 5 minutes
      },

      ...options,
    };

    this.backends = new Map();
    this.currentIndex = 0;
    this.requestQueue = [];
    this.activeRequests = 0;
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      queuedRequests: 0,
      rejectedRequests: 0,
    };

    this.circuitBreakers = new Map();
    this.lastScalingAction = 0;

    this.initialize();
  }

  initialize() {
    this.logger.info('Initializing Load Balancer...', {
      algorithm: this.options.algorithm,
      healthChecking: this.options.healthCheck.enabled,
      circuitBreaker: this.options.circuitBreaker.enabled,
      autoScaling: this.options.autoScaling.enabled,
    });

    if (this.options.healthCheck.enabled) {
      this.startHealthChecking();
    }

    if (this.options.autoScaling.enabled) {
      this.startAutoScaling();
    }

    // Start metrics collection
    this.startMetricsCollection();

    this.logger.info('Load Balancer initialized successfully');
  }

  addBackend(id, config) {
    const backend = {
      id,
      ...config,
      status: 'healthy',
      connections: 0,
      weight: config.weight || 1,
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageResponseTime: 0,
      lastHealthCheck: null,
      consecutiveFailures: 0,
    };

    this.backends.set(id, backend);

    if (this.options.circuitBreaker.enabled) {
      this.circuitBreakers.set(id, {
        state: 'closed',
        failures: 0,
        lastFailure: null,
        nextAttempt: Date.now(),
      });
    }

    this.logger.info('Backend added', {
      id,
      host: config.host,
      port: config.port,
      weight: backend.weight,
    });

    this.emit('backendAdded', backend);
  }

  removeBackend(id) {
    if (this.backends.delete(id)) {
      this.circuitBreakers.delete(id);
      this.logger.info('Backend removed', { id });
      this.emit('backendRemoved', id);
      return true;
    }
    return false;
  }

  getHealthyBackends() {
    return Array.from(this.backends.values()).filter(
      backend => backend.status === 'healthy' && this.isCircuitClosed(backend.id),
    );
  }

  selectBackend() {
    const healthyBackends = this.getHealthyBackends();

    if (healthyBackends.length === 0) {
      throw new Error('No healthy backends available');
    }

    switch (this.options.algorithm) {
      case 'round_robin':
        return this.selectRoundRobin(healthyBackends);
      case 'weighted':
        return this.selectWeighted(healthyBackends);
      case 'least_connections':
        return this.selectLeastConnections(healthyBackends);
      case 'ip_hash':
        return this.selectIPHash(healthyBackends);
      default:
        return this.selectRoundRobin(healthyBackends);
    }
  }

  selectRoundRobin(backends) {
    const backend = backends[this.currentIndex % backends.length];
    this.currentIndex++;
    return backend;
  }

  selectWeighted(backends) {
    const totalWeight = backends.reduce((sum, backend) => sum + backend.weight, 0);
    let random = Math.random() * totalWeight;

    for (const backend of backends) {
      random -= backend.weight;
      if (random <= 0) {
        return backend;
      }
    }

    return backends[backends.length - 1];
  }

  selectLeastConnections(backends) {
    return backends.reduce((least, current) => (current.connections < least.connections ? current : least));
  }

  selectIPHash(backends, clientIP = '') {
    // Simple hash-based selection
    let hash = 0;
    for (let i = 0; i < clientIP.length; i++) {
      const char = clientIP.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }

    const index = Math.abs(hash) % backends.length;
    return backends[index];
  }

  async processRequest(requestHandler, req, res, options = {}) {
    const startTime = Date.now();
    this.metrics.totalRequests++;

    // Check if we're at capacity
    if (this.activeRequests >= this.options.maxConcurrentRequests) {
      if (this.requestQueue.length >= this.options.maxConcurrentRequests) {
        this.metrics.rejectedRequests++;
        throw new Error('Request rejected - system at capacity');
      }

      // Queue the request
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          this.removeFromQueue(queueItem);
          reject(new Error('Request timeout while queued'));
        }, this.options.queueTimeout);

        const queueItem = { requestHandler, req, res, options, resolve, reject, timeout, startTime };
        this.requestQueue.push(queueItem);
        this.metrics.queuedRequests++;
      });
    }

    return this.executeRequest(requestHandler, req, res, options, startTime);
  }

  async executeRequest(requestHandler, req, res, options = {}, startTime) {
    this.activeRequests++;

    try {
      // Select backend
      const backend = options.backend || this.selectBackend();

      if (!backend) {
        throw new Error('No backend available');
      }

      backend.connections++;
      backend.totalRequests++;

      this.logger.debug('Request routed to backend', {
        backendId: backend.id,
        method: req.method,
        url: req.url,
        connections: backend.connections,
      });

      // Execute the request handler
      const result = await this.executeWithTimeout(() => requestHandler(req, res, backend), options.timeout || 30000);

      // Success handling
      const responseTime = Date.now() - startTime;
      this.updateBackendMetrics(backend, true, responseTime);
      this.updateGlobalMetrics(true, responseTime);

      this.logger.debug('Request completed successfully', {
        backendId: backend.id,
        responseTime,
        status: res.statusCode,
      });

      return result;
    } catch (error) {
      // Error handling
      const responseTime = Date.now() - startTime;

      if (options.backend) {
        this.updateBackendMetrics(options.backend, false, responseTime);
        this.handleBackendError(options.backend, error);
      }

      this.updateGlobalMetrics(false, responseTime);

      this.logger.warn('Request failed', {
        error: error.message,
        responseTime,
        activeRequests: this.activeRequests,
      });

      throw error;
    } finally {
      this.activeRequests--;

      if (options.backend) {
        options.backend.connections--;
      }

      // Process next queued request
      this.processQueue();
    }
  }

  processQueue() {
    if (this.requestQueue.length > 0 && this.activeRequests < this.options.maxConcurrentRequests) {
      const queueItem = this.requestQueue.shift();
      this.metrics.queuedRequests--;

      clearTimeout(queueItem.timeout);

      this.executeRequest(
        queueItem.requestHandler,
        queueItem.req,
        queueItem.res,
        queueItem.options,
        queueItem.startTime,
      )
        .then(queueItem.resolve)
        .catch(queueItem.reject);
    }
  }

  removeFromQueue(targetItem) {
    const index = this.requestQueue.indexOf(targetItem);
    if (index > -1) {
      this.requestQueue.splice(index, 1);
      this.metrics.queuedRequests--;
    }
  }

  async executeWithTimeout(fn, timeout) {
    return Promise.race([
      fn(),
      new Promise((_, reject) => setTimeout(() => reject(new Error('Operation timeout')), timeout)),
    ]);
  }

  updateBackendMetrics(backend, success, responseTime) {
    if (success) {
      backend.successfulRequests++;
      backend.consecutiveFailures = 0;
    } else {
      backend.failedRequests++;
      backend.consecutiveFailures++;
    }

    // Update average response time
    const totalRequests = backend.totalRequests;
    backend.averageResponseTime = (backend.averageResponseTime * (totalRequests - 1) + responseTime) / totalRequests;
  }

  updateGlobalMetrics(success, responseTime) {
    if (success) {
      this.metrics.successfulRequests++;
    } else {
      this.metrics.failedRequests++;
    }

    // Update average response time
    const totalRequests = this.metrics.totalRequests;
    this.metrics.averageResponseTime =
      (this.metrics.averageResponseTime * (totalRequests - 1) + responseTime) / totalRequests;
  }

  handleBackendError(backend, error) {
    if (this.options.circuitBreaker.enabled) {
      const breaker = this.circuitBreakers.get(backend.id);

      if (breaker) {
        breaker.failures++;
        breaker.lastFailure = Date.now();

        if (breaker.failures >= this.options.circuitBreaker.threshold) {
          breaker.state = 'open';
          breaker.nextAttempt = Date.now() + this.options.circuitBreaker.resetTimeout;

          this.logger.warn('Circuit breaker opened for backend', {
            backendId: backend.id,
            failures: breaker.failures,
          });

          this.emit('circuitBreakerOpened', backend.id);
        }
      }
    }

    // Mark backend as unhealthy if too many consecutive failures
    if (backend.consecutiveFailures >= 5) {
      backend.status = 'unhealthy';
      this.logger.warn('Backend marked as unhealthy', {
        backendId: backend.id,
        consecutiveFailures: backend.consecutiveFailures,
      });

      this.emit('backendUnhealthy', backend.id);
    }
  }

  isCircuitClosed(backendId) {
    if (!this.options.circuitBreaker.enabled) return true;

    const breaker = this.circuitBreakers.get(backendId);
    if (!breaker) return true;

    if (breaker.state === 'open') {
      if (Date.now() >= breaker.nextAttempt) {
        breaker.state = 'half-open';
        this.logger.info('Circuit breaker half-opened', { backendId });
        this.emit('circuitBreakerHalfOpened', backendId);
      } else {
        return false;
      }
    }

    return true;
  }

  startHealthChecking() {
    this.healthCheckInterval = setInterval(async () => {
      await this.performHealthChecks();
    }, this.options.healthCheck.interval);

    this.logger.info('Health checking started', {
      interval: this.options.healthCheck.interval,
    });
  }

  async performHealthChecks() {
    const healthCheckPromises = Array.from(this.backends.values()).map(backend => this.checkBackendHealth(backend));

    await Promise.allSettled(healthCheckPromises);
  }

  async checkBackendHealth(backend) {
    try {
      const startTime = Date.now();

      // Simulate health check (in real implementation, make HTTP request)
      await new Promise(resolve => setTimeout(resolve, 10));

      const responseTime = Date.now() - startTime;

      if (responseTime < this.options.healthCheck.timeout) {
        if (backend.status === 'unhealthy') {
          backend.status = 'healthy';
          backend.consecutiveFailures = 0;

          // Reset circuit breaker
          const breaker = this.circuitBreakers.get(backend.id);
          if (breaker && breaker.state === 'half-open') {
            breaker.state = 'closed';
            breaker.failures = 0;
            this.logger.info('Circuit breaker closed', { backendId: backend.id });
            this.emit('circuitBreakerClosed', backend.id);
          }

          this.logger.info('Backend recovered', { backendId: backend.id });
          this.emit('backendHealthy', backend.id);
        }

        backend.lastHealthCheck = Date.now();
      }
    } catch (error) {
      backend.consecutiveFailures++;

      if (backend.consecutiveFailures >= this.options.healthCheck.retries) {
        backend.status = 'unhealthy';
        this.logger.warn('Health check failed', {
          backendId: backend.id,
          error: error.message,
          consecutiveFailures: backend.consecutiveFailures,
        });

        this.emit('backendUnhealthy', backend.id);
      }
    }
  }

  startAutoScaling() {
    this.autoScalingInterval = setInterval(() => {
      this.evaluateAutoScaling();
    }, 60000); // Every minute

    this.logger.info('Auto-scaling monitoring started');
  }

  evaluateAutoScaling() {
    const now = Date.now();
    const { autoScaling } = this.options;

    // Cooldown check
    if (now - this.lastScalingAction < autoScaling.cooldownPeriod) {
      return;
    }

    const currentLoad = this.activeRequests / this.options.maxConcurrentRequests;
    const healthyBackends = this.getHealthyBackends().length;

    // Scale up if needed
    if (currentLoad > autoScaling.scaleUpThreshold && healthyBackends < autoScaling.maxInstances) {
      this.scaleUp();
      this.lastScalingAction = now;
    }

    // Scale down if needed
    else if (currentLoad < autoScaling.scaleDownThreshold && healthyBackends > autoScaling.minInstances) {
      this.scaleDown();
      this.lastScalingAction = now;
    }
  }

  scaleUp() {
    this.logger.info('Auto-scaling: scaling up', {
      currentLoad: (this.activeRequests / this.options.maxConcurrentRequests).toFixed(2),
      healthyBackends: this.getHealthyBackends().length,
    });

    this.emit('scaleUp', {
      currentBackends: this.backends.size,
      activeRequests: this.activeRequests,
    });
  }

  scaleDown() {
    const healthyBackends = this.getHealthyBackends();
    const leastUsedBackend = healthyBackends.reduce((least, current) =>
      current.connections < least.connections ? current : least,
    );

    if (leastUsedBackend && leastUsedBackend.connections === 0) {
      this.logger.info('Auto-scaling: scaling down', {
        removingBackend: leastUsedBackend.id,
        currentLoad: (this.activeRequests / this.options.maxConcurrentRequests).toFixed(2),
      });

      this.emit('scaleDown', {
        backendToRemove: leastUsedBackend.id,
        currentBackends: this.backends.size,
      });
    }
  }

  startMetricsCollection() {
    this.metricsInterval = setInterval(() => {
      this.logMetrics();
    }, 60000); // Every minute
  }

  logMetrics() {
    const healthyBackends = this.getHealthyBackends().length;
    const successRate =
      this.metrics.totalRequests > 0
        ? ((this.metrics.successfulRequests / this.metrics.totalRequests) * 100).toFixed(2)
        : 0;

    this.logger.info('Load balancer metrics', {
      algorithm: this.options.algorithm,
      totalRequests: this.metrics.totalRequests,
      successRate: `${successRate}%`,
      activeRequests: this.activeRequests,
      queuedRequests: this.metrics.queuedRequests,
      healthyBackends: healthyBackends,
      totalBackends: this.backends.size,
      averageResponseTime: `${Math.round(this.metrics.averageResponseTime)}ms`,
    });

    this.emit('metricsUpdate', this.getMetrics());
  }

  getMetrics() {
    const backendMetrics = {};

    for (const [id, backend] of this.backends) {
      backendMetrics[id] = {
        status: backend.status,
        connections: backend.connections,
        totalRequests: backend.totalRequests,
        successfulRequests: backend.successfulRequests,
        failedRequests: backend.failedRequests,
        averageResponseTime: Math.round(backend.averageResponseTime),
        weight: backend.weight,
      };
    }

    return {
      global: {
        ...this.metrics,
        activeRequests: this.activeRequests,
        averageResponseTime: Math.round(this.metrics.averageResponseTime),
      },
      backends: backendMetrics,
      algorithm: this.options.algorithm,
      healthyBackends: this.getHealthyBackends().length,
      totalBackends: this.backends.size,
    };
  }

  // Express middleware
  middleware() {
    return async (req, res, next) => {
      try {
        await this.processRequest(
          async (request, response, backend) => {
            // Add backend info to request
            request.backend = backend;
            request.loadBalancer = this;

            // Call next middleware
            return new Promise((resolve, reject) => {
              const originalEnd = response.end;
              response.end = function (...args) {
                resolve();
                return originalEnd.apply(this, args);
              };

              response.on('error', reject);
              next();
            });
          },
          req,
          res,
        );
      } catch (error) {
        next(error);
      }
    };
  }

  async cleanup() {
    this.logger.info('Cleaning up Load Balancer...');

    if (this.healthCheckInterval) clearInterval(this.healthCheckInterval);
    if (this.autoScalingInterval) clearInterval(this.autoScalingInterval);
    if (this.metricsInterval) clearInterval(this.metricsInterval);

    // Reject all queued requests
    for (const queueItem of this.requestQueue) {
      clearTimeout(queueItem.timeout);
      queueItem.reject(new Error('Load balancer shutting down'));
    }

    this.requestQueue = [];
    this.backends.clear();
    this.circuitBreakers.clear();
    this.removeAllListeners();

    this.logger.info('Load Balancer cleanup completed');
  }
}

// Cluster management for automatic scaling
export class ClusterManager {
  constructor(logger, options = {}) {
    this.logger = logger;
    this.options = {
      minWorkers: options.minWorkers || 1,
      maxWorkers: options.maxWorkers || os.cpus().length,
      autoRestart: options.autoRestart !== false,
      gracefulShutdownTimeout: options.gracefulShutdownTimeout || 30000,
      ...options,
    };

    this.workers = new Map();
    this.setupCluster();
  }

  setupCluster() {
    if (cluster.isMaster) {
      this.logger.info('Starting cluster manager', {
        minWorkers: this.options.minWorkers,
        maxWorkers: this.options.maxWorkers,
      });

      // Fork initial workers
      for (let i = 0; i < this.options.minWorkers; i++) {
        this.forkWorker();
      }

      // Handle worker events
      cluster.on('exit', (worker, code, signal) => {
        this.handleWorkerExit(worker, code, signal);
      });

      cluster.on('listening', (worker, address) => {
        this.logger.info('Worker listening', {
          workerId: worker.id,
          pid: worker.process.pid,
          port: address.port,
        });
      });

      // Graceful shutdown
      process.on('SIGTERM', () => this.shutdown());
      process.on('SIGINT', () => this.shutdown());

      return true;
    }

    return false;
  }

  forkWorker() {
    const worker = cluster.fork();
    this.workers.set(worker.id, {
      worker,
      startTime: Date.now(),
      restarts: 0,
    });

    this.logger.info('Worker forked', {
      workerId: worker.id,
      pid: worker.process.pid,
    });

    return worker;
  }

  handleWorkerExit(worker, code, signal) {
    const workerInfo = this.workers.get(worker.id);
    this.workers.delete(worker.id);

    this.logger.warn('Worker exited', {
      workerId: worker.id,
      pid: worker.process.pid,
      code,
      signal,
      restarts: workerInfo?.restarts || 0,
    });

    // Auto-restart if enabled and not graceful shutdown
    if (this.options.autoRestart && code !== 0 && signal !== 'SIGTERM') {
      if (!workerInfo || workerInfo.restarts < 5) {
        setTimeout(() => {
          const newWorker = this.forkWorker();
          if (workerInfo) {
            this.workers.get(newWorker.id).restarts = workerInfo.restarts + 1;
          }
        }, 1000);
      } else {
        this.logger.error('Worker exceeded restart limit', {
          workerId: worker.id,
          restarts: workerInfo.restarts,
        });
      }
    }
  }

  scaleWorkers(targetCount) {
    const currentCount = this.workers.size;

    if (targetCount > currentCount) {
      // Scale up
      for (let i = currentCount; i < targetCount && i < this.options.maxWorkers; i++) {
        this.forkWorker();
      }
    } else if (targetCount < currentCount) {
      // Scale down
      const workersToKill = currentCount - Math.max(targetCount, this.options.minWorkers);
      const workerIds = Array.from(this.workers.keys()).slice(0, workersToKill);

      for (const workerId of workerIds) {
        const workerInfo = this.workers.get(workerId);
        if (workerInfo) {
          this.logger.info('Scaling down worker', { workerId });
          workerInfo.worker.disconnect();
          workerInfo.worker.kill('SIGTERM');
        }
      }
    }
  }

  async shutdown() {
    this.logger.info('Initiating cluster shutdown...');

    // Disconnect all workers
    for (const workerInfo of this.workers.values()) {
      workerInfo.worker.disconnect();
    }

    // Wait for graceful shutdown or force kill
    const shutdownPromise = new Promise(resolve => {
      const checkInterval = setInterval(() => {
        if (this.workers.size === 0) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 1000);
    });

    const timeoutPromise = new Promise(resolve => {
      setTimeout(() => {
        this.logger.warn('Graceful shutdown timeout, force killing workers');
        for (const workerInfo of this.workers.values()) {
          workerInfo.worker.kill('SIGKILL');
        }
        resolve();
      }, this.options.gracefulShutdownTimeout);
    });

    await Promise.race([shutdownPromise, timeoutPromise]);
    this.logger.info('Cluster shutdown completed');
    process.exit(0);
  }
}

export function createLoadBalancer(logger, options = {}) {
  return new LoadBalancer(logger, options);
}

export function createClusterManager(logger, options = {}) {
  return new ClusterManager(logger, options);
}
