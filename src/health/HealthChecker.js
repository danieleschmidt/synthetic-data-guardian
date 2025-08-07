/**
 * Comprehensive Health Checking System
 */

import { HealthMetrics } from '../middleware/metrics.js';

export class HealthChecker {
  constructor(logger) {
    this.logger = logger;
    this.checks = new Map();
    this.lastResults = new Map();
    this.isRunning = false;
    this.checkInterval = 30000; // 30 seconds
    this.intervalId = null;
  }

  // Add a health check
  addCheck(name, checkFn, options = {}) {
    this.checks.set(name, {
      name,
      fn: checkFn,
      timeout: options.timeout || 5000,
      critical: options.critical || false,
      interval: options.interval || this.checkInterval,
      description: options.description || `Health check for ${name}`,
      enabled: options.enabled !== false,
    });

    this.logger.info('Health check added', {
      name,
      critical: options.critical,
      timeout: options.timeout || 5000,
    });
  }

  // Remove a health check
  removeCheck(name) {
    if (this.checks.delete(name)) {
      this.lastResults.delete(name);
      this.logger.info('Health check removed', { name });
      return true;
    }
    return false;
  }

  // Run a single health check
  async runSingleCheck(name) {
    const check = this.checks.get(name);
    if (!check || !check.enabled) {
      return {
        name,
        status: 'skipped',
        message: check ? 'Check is disabled' : 'Check not found',
      };
    }

    const startTime = Date.now();

    try {
      const result = await Promise.race([
        check.fn(),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error(`Health check timeout after ${check.timeout}ms`)), check.timeout),
        ),
      ]);

      const duration = Date.now() - startTime;
      const healthResult = {
        name,
        status: 'healthy',
        duration,
        timestamp: new Date().toISOString(),
        critical: check.critical,
        description: check.description,
        ...result,
      };

      this.lastResults.set(name, healthResult);
      HealthMetrics.recordHealthCheck(name, 'healthy');

      return healthResult;
    } catch (error) {
      const duration = Date.now() - startTime;
      const healthResult = {
        name,
        status: 'unhealthy',
        error: error.message,
        duration,
        timestamp: new Date().toISOString(),
        critical: check.critical,
        description: check.description,
      };

      this.lastResults.set(name, healthResult);
      HealthMetrics.recordHealthCheck(name, 'unhealthy');

      this.logger.warn('Health check failed', {
        name,
        error: error.message,
        duration,
        critical: check.critical,
      });

      return healthResult;
    }
  }

  // Run all health checks
  async runAllChecks() {
    const results = new Map();
    const checkPromises = [];

    this.logger.debug('Running all health checks', { checkCount: this.checks.size });

    // Run all checks in parallel
    for (const [name] of this.checks) {
      checkPromises.push(
        this.runSingleCheck(name).then(result => {
          results.set(name, result);
          return result;
        }),
      );
    }

    await Promise.allSettled(checkPromises);

    // Determine overall health
    let overallStatus = 'healthy';
    let healthyCount = 0;
    let unhealthyCount = 0;
    let criticalFailures = [];

    for (const [name, result] of results) {
      if (result.status === 'healthy') {
        healthyCount++;
      } else if (result.status === 'unhealthy') {
        unhealthyCount++;
        if (result.critical) {
          criticalFailures.push(name);
          overallStatus = 'unhealthy';
        }
      }
    }

    // If no critical failures but some unhealthy checks, mark as degraded
    if (overallStatus === 'healthy' && unhealthyCount > 0) {
      overallStatus = 'degraded';
    }

    const healthSummary = {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      summary: {
        total: this.checks.size,
        healthy: healthyCount,
        unhealthy: unhealthyCount,
        criticalFailures: criticalFailures.length,
        uptime: process.uptime(),
      },
      checks: Object.fromEntries(results),
      criticalFailures,
    };

    this.logger.info('Health check completed', {
      status: overallStatus,
      healthy: healthyCount,
      unhealthy: unhealthyCount,
      critical: criticalFailures.length,
    });

    return healthSummary;
  }

  // Get last known health status
  getLastKnownHealth() {
    if (this.lastResults.size === 0) {
      return {
        status: 'unknown',
        message: 'No health checks have been run yet',
        timestamp: new Date().toISOString(),
      };
    }

    let overallStatus = 'healthy';
    let criticalFailures = [];
    const checks = {};

    for (const [name, result] of this.lastResults) {
      checks[name] = result;

      if (result.status === 'unhealthy' && result.critical) {
        criticalFailures.push(name);
        overallStatus = 'unhealthy';
      } else if (result.status === 'unhealthy') {
        if (overallStatus === 'healthy') {
          overallStatus = 'degraded';
        }
      }
    }

    return {
      status: overallStatus,
      timestamp: new Date().toISOString(),
      summary: {
        total: this.lastResults.size,
        criticalFailures: criticalFailures.length,
      },
      checks,
      criticalFailures,
    };
  }

  // Start periodic health checks
  startPeriodicChecks(interval = this.checkInterval) {
    if (this.isRunning) {
      this.logger.warn('Periodic health checks already running');
      return;
    }

    this.checkInterval = interval;
    this.isRunning = true;

    this.logger.info('Starting periodic health checks', { interval });

    // Run initial check
    this.runAllChecks().catch(error => {
      this.logger.error('Initial health check failed', { error: error.message });
    });

    // Schedule periodic checks
    this.intervalId = setInterval(async () => {
      try {
        await this.runAllChecks();
      } catch (error) {
        this.logger.error('Periodic health check failed', { error: error.message });
      }
    }, interval);
  }

  // Stop periodic health checks
  stopPeriodicChecks() {
    if (!this.isRunning) {
      return;
    }

    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    this.isRunning = false;
    this.logger.info('Stopped periodic health checks');
  }

  // Get health checker statistics
  getStats() {
    return {
      isRunning: this.isRunning,
      checkInterval: this.checkInterval,
      totalChecks: this.checks.size,
      enabledChecks: Array.from(this.checks.values()).filter(c => c.enabled).length,
      lastCheckTime:
        this.lastResults.size > 0
          ? Math.max(...Array.from(this.lastResults.values()).map(r => new Date(r.timestamp).getTime()))
          : null,
    };
  }
}

// Default health checks
export function createDefaultHealthChecks(guardian, logger) {
  const healthChecker = new HealthChecker(logger);

  // Memory usage check
  healthChecker.addCheck(
    'memory',
    async () => {
      const memoryUsage = process.memoryUsage();
      const totalMemory = memoryUsage.heapTotal;
      const usedMemory = memoryUsage.heapUsed;
      const memoryUtilization = (usedMemory / totalMemory) * 100;

      const isHealthy = memoryUtilization < 90; // Alert if memory usage > 90%

      return {
        status: isHealthy ? 'healthy' : 'unhealthy',
        memoryUtilization: `${memoryUtilization.toFixed(2)}%`,
        heapUsed: `${Math.round(usedMemory / 1024 / 1024)}MB`,
        heapTotal: `${Math.round(totalMemory / 1024 / 1024)}MB`,
        rss: `${Math.round(memoryUsage.rss / 1024 / 1024)}MB`,
        external: `${Math.round(memoryUsage.external / 1024 / 1024)}MB`,
      };
    },
    {
      critical: false,
      description: 'Monitor memory usage and detect memory leaks',
    },
  );

  // Event loop lag check
  healthChecker.addCheck(
    'event_loop',
    async () => {
      const start = process.hrtime.bigint();

      return new Promise(resolve => {
        setImmediate(() => {
          const lag = Number(process.hrtime.bigint() - start) / 1000000; // Convert to milliseconds
          const isHealthy = lag < 100; // Alert if event loop lag > 100ms

          resolve({
            status: isHealthy ? 'healthy' : 'unhealthy',
            eventLoopLag: `${lag.toFixed(2)}ms`,
            threshold: '100ms',
          });
        });
      });
    },
    {
      critical: false,
      description: 'Monitor event loop performance',
    },
  );

  // Guardian component health
  healthChecker.addCheck(
    'guardian',
    async () => {
      const isInitialized = guardian && guardian.initialized;

      if (!isInitialized) {
        throw new Error('Guardian is not initialized');
      }

      const activeTasks = guardian.activeTasks?.size || 0;
      const pipelines = guardian.pipelines?.size || 0;

      return {
        initialized: isInitialized,
        activeTasks,
        registeredPipelines: pipelines,
        status: 'healthy',
      };
    },
    {
      critical: true,
      description: 'Check Guardian core component health',
    },
  );

  // File system check
  healthChecker.addCheck(
    'filesystem',
    async () => {
      const fs = await import('fs/promises');
      const os = await import('os');

      try {
        const tempDir = os.tmpdir();
        const testFile = `${tempDir}/health-check-${Date.now()}.tmp`;

        // Test write
        await fs.writeFile(testFile, 'health-check');

        // Test read
        const content = await fs.readFile(testFile, 'utf8');

        // Cleanup
        await fs.unlink(testFile);

        if (content !== 'health-check') {
          throw new Error('File content mismatch');
        }

        return {
          status: 'healthy',
          tempDir,
          canWrite: true,
          canRead: true,
        };
      } catch (error) {
        throw new Error(`Filesystem check failed: ${error.message}`);
      }
    },
    {
      critical: false,
      description: 'Test filesystem read/write capabilities',
    },
  );

  // Process info check
  healthChecker.addCheck(
    'process',
    async () => {
      const uptime = process.uptime();
      const version = process.version;
      const platform = process.platform;
      const arch = process.arch;
      const pid = process.pid;

      return {
        status: 'healthy',
        uptime: `${Math.floor(uptime)}s`,
        nodeVersion: version,
        platform,
        architecture: arch,
        processId: pid,
        memoryUsage: process.memoryUsage(),
        cpuUsage: process.cpuUsage(),
      };
    },
    {
      critical: false,
      description: 'Monitor process information and uptime',
    },
  );

  return healthChecker;
}

// Create readiness and liveness probes
export function createKubernetesProbes(healthChecker) {
  return {
    // Liveness probe - checks if the application is running
    liveness: async () => {
      const basicChecks = await healthChecker.runSingleCheck('process');

      if (basicChecks.status !== 'healthy') {
        throw new Error('Liveness check failed: process not healthy');
      }

      return {
        status: 'alive',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
      };
    },

    // Readiness probe - checks if the application can serve requests
    readiness: async () => {
      const health = await healthChecker.runAllChecks();

      // Only check critical components for readiness
      const criticalFailures = health.criticalFailures || [];

      if (criticalFailures.length > 0) {
        throw new Error(`Readiness check failed: critical components unhealthy - ${criticalFailures.join(', ')}`);
      }

      return {
        status: 'ready',
        timestamp: new Date().toISOString(),
        healthStatus: health.status,
      };
    },
  };
}
