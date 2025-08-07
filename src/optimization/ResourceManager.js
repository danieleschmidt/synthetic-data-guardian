/**
 * Advanced Resource Manager with Auto-scaling and Resource Optimization
 */

import { EventEmitter } from 'events';
import os from 'os';

export class ResourceManager extends EventEmitter {
  constructor(logger, options = {}) {
    super();
    this.logger = logger;
    this.options = {
      // Resource monitoring
      monitoring: {
        enabled: options.monitoring?.enabled !== false,
        interval: options.monitoring?.interval || 30000, // 30 seconds
        historySize: options.monitoring?.historySize || 120, // 2 hours at 1min intervals
      },

      // Resource thresholds
      thresholds: {
        cpu: {
          high: options.thresholds?.cpu?.high || 80,
          critical: options.thresholds?.cpu?.critical || 95,
          low: options.thresholds?.cpu?.low || 20,
        },
        memory: {
          high: options.thresholds?.memory?.high || 80,
          critical: options.thresholds?.memory?.critical || 95,
          low: options.thresholds?.memory?.low || 30,
        },
        eventLoop: {
          high: options.thresholds?.eventLoop?.high || 100, // ms
          critical: options.thresholds?.eventLoop?.critical || 500,
          low: options.thresholds?.eventLoop?.low || 10,
        },
        connections: {
          high: options.thresholds?.connections?.high || 1000,
          critical: options.thresholds?.connections?.critical || 2000,
          low: options.thresholds?.connections?.low || 100,
        },
      },

      // Auto-scaling configuration
      autoScaling: {
        enabled: options.autoScaling?.enabled === true,
        cooldownPeriod: options.autoScaling?.cooldownPeriod || 300000, // 5 minutes
        scaleUpFactor: options.autoScaling?.scaleUpFactor || 1.5,
        scaleDownFactor: options.autoScaling?.scaleDownFactor || 0.7,
        minInstances: options.autoScaling?.minInstances || 1,
        maxInstances: options.autoScaling?.maxInstances || os.cpus().length * 2,
        evaluationPeriods: options.autoScaling?.evaluationPeriods || 3, // Number of periods to evaluate
      },

      // Resource optimization
      optimization: {
        enabled: options.optimization?.enabled !== false,
        gcThreshold: options.optimization?.gcThreshold || 85, // Trigger GC at 85% memory
        cacheEvictionThreshold: options.optimization?.cacheEvictionThreshold || 90,
        processLimitEnabled: options.optimization?.processLimitEnabled !== false,
        maxProcesses: options.optimization?.maxProcesses || os.cpus().length * 4,
      },

      // Alerts and notifications
      alerts: {
        enabled: options.alerts?.enabled !== false,
        channels: options.alerts?.channels || ['log'], // log, webhook, email
        webhook: options.alerts?.webhook,
        email: options.alerts?.email,
      },

      ...options,
    };

    this.resourceHistory = {
      cpu: [],
      memory: [],
      eventLoop: [],
      connections: [],
      requests: [],
    };

    this.currentResources = {
      cpu: 0,
      memory: 0,
      eventLoop: 0,
      connections: 0,
      processes: 0,
      uptime: 0,
    };

    this.scalingHistory = [];
    this.lastScalingAction = 0;
    this.alertsSent = new Set();

    this.initialize();
  }

  initialize() {
    this.logger.info('Initializing Resource Manager...', {
      monitoring: this.options.monitoring.enabled,
      autoScaling: this.options.autoScaling.enabled,
      optimization: this.options.optimization.enabled,
      alerts: this.options.alerts.enabled,
    });

    if (this.options.monitoring.enabled) {
      this.startResourceMonitoring();
    }

    if (this.options.autoScaling.enabled) {
      this.startAutoScalingEvaluation();
    }

    // Set up process event handlers
    this.setupProcessHandlers();

    this.logger.info('Resource Manager initialized successfully');
  }

  startResourceMonitoring() {
    this.monitoringInterval = setInterval(async () => {
      await this.collectResourceMetrics();
    }, this.options.monitoring.interval);

    this.logger.info('Resource monitoring started', {
      interval: this.options.monitoring.interval,
      historySize: this.options.monitoring.historySize,
    });
  }

  async collectResourceMetrics() {
    try {
      const metrics = await this.gatherSystemMetrics();

      // Update current resources
      Object.assign(this.currentResources, metrics);

      // Add to history
      this.addToHistory(metrics);

      // Evaluate resource status
      const status = this.evaluateResourceStatus(metrics);

      // Emit resource update event
      this.emit('resourceUpdate', {
        current: metrics,
        status,
        timestamp: Date.now(),
      });

      // Check for alerts
      if (this.options.alerts.enabled) {
        this.checkResourceAlerts(metrics, status);
      }

      // Trigger optimization if needed
      if (this.options.optimization.enabled) {
        this.optimizeResources(metrics, status);
      }

      this.logger.debug('Resource metrics collected', {
        cpu: `${metrics.cpu.toFixed(1)}%`,
        memory: `${metrics.memory.toFixed(1)}%`,
        eventLoop: `${metrics.eventLoop.toFixed(1)}ms`,
        connections: metrics.connections,
      });
    } catch (error) {
      this.logger.error('Failed to collect resource metrics', {
        error: error.message,
      });
    }
  }

  async gatherSystemMetrics() {
    const cpuUsage = await this.getCPUUsage();
    const memoryUsage = this.getMemoryUsage();
    const eventLoopLag = await this.getEventLoopLag();
    const connectionCount = this.getConnectionCount();
    const processCount = this.getProcessCount();

    return {
      cpu: cpuUsage,
      memory: memoryUsage,
      eventLoop: eventLoopLag,
      connections: connectionCount,
      processes: processCount,
      uptime: process.uptime(),
      timestamp: Date.now(),
    };
  }

  async getCPUUsage() {
    // Get CPU usage by measuring over a short interval
    const startUsage = process.cpuUsage();
    const startTime = process.hrtime();

    await new Promise(resolve => setTimeout(resolve, 100));

    const endUsage = process.cpuUsage(startUsage);
    const endTime = process.hrtime(startTime);

    const totalTime = endTime[0] * 1e6 + endTime[1] / 1e3; // microseconds
    const totalUsage = endUsage.user + endUsage.system; // microseconds

    return (totalUsage / totalTime) * 100;
  }

  getMemoryUsage() {
    const usage = process.memoryUsage();
    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();
    const usedMemory = totalMemory - freeMemory;

    // Calculate percentage based on system memory
    return (usedMemory / totalMemory) * 100;
  }

  async getEventLoopLag() {
    const start = process.hrtime.bigint();

    return new Promise(resolve => {
      setImmediate(() => {
        const lag = Number(process.hrtime.bigint() - start) / 1000000; // Convert to milliseconds
        resolve(lag);
      });
    });
  }

  getConnectionCount() {
    // In a real implementation, this would track actual connections
    // For now, return a simulated count based on active handles
    return process._getActiveHandles ? process._getActiveHandles().length : 0;
  }

  getProcessCount() {
    // In a real implementation, this would count spawned child processes
    return 1; // Current process
  }

  addToHistory(metrics) {
    const maxSize = this.options.monitoring.historySize;

    for (const key of Object.keys(this.resourceHistory)) {
      if (metrics[key] !== undefined) {
        this.resourceHistory[key].push({
          value: metrics[key],
          timestamp: Date.now(),
        });

        // Trim history to max size
        if (this.resourceHistory[key].length > maxSize) {
          this.resourceHistory[key] = this.resourceHistory[key].slice(-maxSize);
        }
      }
    }
  }

  evaluateResourceStatus(metrics) {
    const status = {};
    const { thresholds } = this.options;

    for (const resource of ['cpu', 'memory', 'eventLoop']) {
      const value = metrics[resource];
      const threshold = thresholds[resource];

      if (value >= threshold.critical) {
        status[resource] = 'critical';
      } else if (value >= threshold.high) {
        status[resource] = 'high';
      } else if (value <= threshold.low) {
        status[resource] = 'low';
      } else {
        status[resource] = 'normal';
      }
    }

    // Special handling for connections
    const connections = metrics.connections;
    const connThreshold = thresholds.connections;

    if (connections >= connThreshold.critical) {
      status.connections = 'critical';
    } else if (connections >= connThreshold.high) {
      status.connections = 'high';
    } else if (connections <= connThreshold.low) {
      status.connections = 'low';
    } else {
      status.connections = 'normal';
    }

    // Overall status
    const criticalCount = Object.values(status).filter(s => s === 'critical').length;
    const highCount = Object.values(status).filter(s => s === 'high').length;

    if (criticalCount > 0) {
      status.overall = 'critical';
    } else if (highCount > 1) {
      status.overall = 'high';
    } else if (highCount === 1) {
      status.overall = 'warning';
    } else {
      status.overall = 'normal';
    }

    return status;
  }

  checkResourceAlerts(metrics, status) {
    const alertKey = `${status.overall}_${Date.now()}`;

    // Avoid duplicate alerts within a short time window
    const recentAlerts = Array.from(this.alertsSent).filter(
      alert => Date.now() - parseInt(alert.split('_')[1]) < 300000, // 5 minutes
    );

    if (recentAlerts.some(alert => alert.startsWith(status.overall))) {
      return;
    }

    if (status.overall === 'critical' || status.overall === 'high') {
      this.sendAlert({
        level: status.overall,
        message: this.buildAlertMessage(metrics, status),
        metrics: metrics,
        status: status,
        timestamp: new Date().toISOString(),
      });

      this.alertsSent.add(alertKey);

      // Clean up old alerts
      this.alertsSent = new Set(recentAlerts);
    }
  }

  buildAlertMessage(metrics, status) {
    const issues = [];

    if (status.cpu === 'critical' || status.cpu === 'high') {
      issues.push(`CPU usage: ${metrics.cpu.toFixed(1)}%`);
    }

    if (status.memory === 'critical' || status.memory === 'high') {
      issues.push(`Memory usage: ${metrics.memory.toFixed(1)}%`);
    }

    if (status.eventLoop === 'critical' || status.eventLoop === 'high') {
      issues.push(`Event loop lag: ${metrics.eventLoop.toFixed(1)}ms`);
    }

    if (status.connections === 'critical' || status.connections === 'high') {
      issues.push(`Connection count: ${metrics.connections}`);
    }

    return `Resource ${status.overall} alert: ${issues.join(', ')}`;
  }

  sendAlert(alert) {
    this.logger.warn('Resource alert triggered', alert);

    this.emit('resourceAlert', alert);

    // Send to configured channels
    for (const channel of this.options.alerts.channels) {
      switch (channel) {
        case 'webhook':
          this.sendWebhookAlert(alert);
          break;
        case 'email':
          this.sendEmailAlert(alert);
          break;
        case 'log':
        default:
          // Already logged above
          break;
      }
    }
  }

  async sendWebhookAlert(alert) {
    if (!this.options.alerts.webhook) return;

    try {
      const response = await fetch(this.options.alerts.webhook, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(alert),
      });

      if (!response.ok) {
        throw new Error(`Webhook failed: ${response.status}`);
      }

      this.logger.debug('Webhook alert sent successfully');
    } catch (error) {
      this.logger.error('Failed to send webhook alert', {
        error: error.message,
      });
    }
  }

  async sendEmailAlert(alert) {
    // Email implementation would go here
    this.logger.debug('Email alert would be sent', { alert });
  }

  optimizeResources(metrics, status) {
    const optimizations = [];

    // Memory optimization
    if (metrics.memory > this.options.optimization.gcThreshold) {
      this.performGarbageCollection();
      optimizations.push('garbage_collection');
    }

    // Cache eviction
    if (metrics.memory > this.options.optimization.cacheEvictionThreshold) {
      this.emit('cacheEvictionRequested', { reason: 'high_memory' });
      optimizations.push('cache_eviction');
    }

    // Process limit enforcement
    if (this.options.optimization.processLimitEnabled && metrics.processes > this.options.optimization.maxProcesses) {
      this.emit('processLimitExceeded', {
        current: metrics.processes,
        max: this.options.optimization.maxProcesses,
      });
      optimizations.push('process_limit_warning');
    }

    if (optimizations.length > 0) {
      this.logger.info('Resource optimizations performed', {
        optimizations,
        metrics: {
          cpu: `${metrics.cpu.toFixed(1)}%`,
          memory: `${metrics.memory.toFixed(1)}%`,
        },
      });
    }
  }

  performGarbageCollection() {
    if (global.gc) {
      global.gc();
      this.logger.debug('Garbage collection performed');
    } else {
      this.logger.warn('Garbage collection not available (use --expose-gc flag)');
    }
  }

  startAutoScalingEvaluation() {
    this.autoScalingInterval = setInterval(() => {
      this.evaluateAutoScaling();
    }, this.options.autoScaling.cooldownPeriod / 5); // Evaluate 5 times per cooldown period

    this.logger.info('Auto-scaling evaluation started');
  }

  evaluateAutoScaling() {
    const now = Date.now();

    // Check cooldown period
    if (now - this.lastScalingAction < this.options.autoScaling.cooldownPeriod) {
      return;
    }

    const recentMetrics = this.getRecentMetrics();
    if (recentMetrics.length < this.options.autoScaling.evaluationPeriods) {
      return; // Not enough data
    }

    const shouldScaleUp = this.shouldScaleUp(recentMetrics);
    const shouldScaleDown = this.shouldScaleDown(recentMetrics);

    if (shouldScaleUp) {
      this.triggerScaleUp(recentMetrics);
    } else if (shouldScaleDown) {
      this.triggerScaleDown(recentMetrics);
    }
  }

  getRecentMetrics() {
    const evaluationPeriods = this.options.autoScaling.evaluationPeriods;
    const cpuHistory = this.resourceHistory.cpu.slice(-evaluationPeriods);
    const memoryHistory = this.resourceHistory.memory.slice(-evaluationPeriods);

    return cpuHistory.map((cpu, index) => ({
      cpu: cpu.value,
      memory: memoryHistory[index]?.value || 0,
      timestamp: cpu.timestamp,
    }));
  }

  shouldScaleUp(recentMetrics) {
    const { thresholds } = this.options;
    const highThreshold = Math.max(thresholds.cpu.high, thresholds.memory.high);

    return recentMetrics.every(metric => metric.cpu >= highThreshold || metric.memory >= highThreshold);
  }

  shouldScaleDown(recentMetrics) {
    const { thresholds } = this.options;
    const lowThreshold = Math.max(thresholds.cpu.low, thresholds.memory.low);

    return recentMetrics.every(metric => metric.cpu <= lowThreshold && metric.memory <= lowThreshold);
  }

  triggerScaleUp(recentMetrics) {
    const avgCpu = recentMetrics.reduce((sum, m) => sum + m.cpu, 0) / recentMetrics.length;
    const avgMemory = recentMetrics.reduce((sum, m) => sum + m.memory, 0) / recentMetrics.length;

    this.logger.info('Auto-scaling: Scale up triggered', {
      avgCpu: `${avgCpu.toFixed(1)}%`,
      avgMemory: `${avgMemory.toFixed(1)}%`,
      evaluationPeriods: recentMetrics.length,
    });

    const scalingEvent = {
      action: 'scale_up',
      timestamp: Date.now(),
      reason: 'high_resource_usage',
      metrics: { cpu: avgCpu, memory: avgMemory },
      factor: this.options.autoScaling.scaleUpFactor,
    };

    this.scalingHistory.push(scalingEvent);
    this.lastScalingAction = Date.now();

    this.emit('autoScaleUp', scalingEvent);
  }

  triggerScaleDown(recentMetrics) {
    const avgCpu = recentMetrics.reduce((sum, m) => sum + m.cpu, 0) / recentMetrics.length;
    const avgMemory = recentMetrics.reduce((sum, m) => sum + m.memory, 0) / recentMetrics.length;

    this.logger.info('Auto-scaling: Scale down triggered', {
      avgCpu: `${avgCpu.toFixed(1)}%`,
      avgMemory: `${avgMemory.toFixed(1)}%`,
      evaluationPeriods: recentMetrics.length,
    });

    const scalingEvent = {
      action: 'scale_down',
      timestamp: Date.now(),
      reason: 'low_resource_usage',
      metrics: { cpu: avgCpu, memory: avgMemory },
      factor: this.options.autoScaling.scaleDownFactor,
    };

    this.scalingHistory.push(scalingEvent);
    this.lastScalingAction = Date.now();

    this.emit('autoScaleDown', scalingEvent);
  }

  setupProcessHandlers() {
    // Monitor process warnings
    process.on('warning', warning => {
      this.logger.warn('Process warning', {
        name: warning.name,
        message: warning.message,
        stack: warning.stack,
      });
    });

    // Monitor unhandled promises
    process.on('unhandledRejection', (reason, promise) => {
      this.logger.error('Unhandled promise rejection', {
        reason: reason?.message || reason,
        stack: reason?.stack,
      });
    });
  }

  // Resource prediction
  predictResourceNeeds(timeHorizon = 3600000) {
    // 1 hour
    const history = this.resourceHistory;
    const now = Date.now();

    const predictions = {};

    for (const resource of ['cpu', 'memory', 'connections']) {
      const resourceHistory = history[resource];
      if (resourceHistory.length < 5) {
        predictions[resource] = { value: 0, confidence: 0 };
        continue;
      }

      // Simple linear regression for trend analysis
      const recentData = resourceHistory.slice(-20); // Last 20 measurements
      const trend = this.calculateTrend(recentData);
      const currentValue = recentData[recentData.length - 1].value;

      const predictedValue = currentValue + trend * (timeHorizon / (10 * 60 * 1000)); // 10 minutes per measurement
      const confidence = Math.min(recentData.length / 20, 1); // Confidence based on data points

      predictions[resource] = {
        current: currentValue,
        predicted: Math.max(0, predictedValue),
        trend,
        confidence,
        timeHorizon,
      };
    }

    return predictions;
  }

  calculateTrend(data) {
    if (data.length < 2) return 0;

    const n = data.length;
    const sumX = data.reduce((sum, _, index) => sum + index, 0);
    const sumY = data.reduce((sum, point) => sum + point.value, 0);
    const sumXY = data.reduce((sum, point, index) => sum + index * point.value, 0);
    const sumX2 = data.reduce((sum, _, index) => sum + index * index, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    return isNaN(slope) ? 0 : slope;
  }

  // Get comprehensive resource report
  getResourceReport() {
    const now = Date.now();
    const currentStatus = this.evaluateResourceStatus(this.currentResources);
    const predictions = this.predictResourceNeeds();

    return {
      timestamp: now,
      current: {
        resources: this.currentResources,
        status: currentStatus,
      },
      history: {
        cpu: this.resourceHistory.cpu.slice(-10),
        memory: this.resourceHistory.memory.slice(-10),
        eventLoop: this.resourceHistory.eventLoop.slice(-10),
      },
      predictions,
      scaling: {
        history: this.scalingHistory.slice(-5),
        lastAction: this.lastScalingAction,
        cooldownRemaining: Math.max(0, this.options.autoScaling.cooldownPeriod - (now - this.lastScalingAction)),
      },
      configuration: {
        thresholds: this.options.thresholds,
        autoScaling: this.options.autoScaling,
        optimization: this.options.optimization,
      },
    };
  }

  // Express middleware for resource monitoring
  middleware() {
    return (req, res, next) => {
      const startTime = Date.now();

      // Track request start
      this.emit('requestStart', {
        method: req.method,
        url: req.url,
        timestamp: startTime,
      });

      // Track request completion
      res.on('finish', () => {
        const responseTime = Date.now() - startTime;
        this.emit('requestComplete', {
          method: req.method,
          url: req.url,
          statusCode: res.statusCode,
          responseTime,
          timestamp: Date.now(),
        });
      });

      next();
    };
  }

  async cleanup() {
    this.logger.info('Cleaning up Resource Manager...');

    if (this.monitoringInterval) clearInterval(this.monitoringInterval);
    if (this.autoScalingInterval) clearInterval(this.autoScalingInterval);

    // Clear history
    for (const key of Object.keys(this.resourceHistory)) {
      this.resourceHistory[key] = [];
    }

    this.scalingHistory = [];
    this.alertsSent.clear();
    this.removeAllListeners();

    this.logger.info('Resource Manager cleanup completed');
  }
}

export function createResourceManager(logger, options = {}) {
  return new ResourceManager(logger, options);
}
