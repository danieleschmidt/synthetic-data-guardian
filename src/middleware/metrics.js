/**
 * Comprehensive Metrics and Monitoring System
 */

import { EventEmitter } from 'events';

export class HealthMetrics extends EventEmitter {
  static instance = null;

  static getInstance(logger = null) {
    if (!HealthMetrics.instance) {
      HealthMetrics.instance = new HealthMetrics(logger);
    }
    return HealthMetrics.instance;
  }

  constructor(logger) {
    super();
    this.logger = logger;
    this.metrics = new Map();
    this.counters = new Map();
    this.histograms = new Map();
    this.gauges = new Map();
    this.timers = new Map();

    this.startTime = Date.now();
    this.resetTime = this.startTime;

    this.setupDefaultMetrics();
    this.startPeriodicReports();
  }

  setupDefaultMetrics() {
    // HTTP request metrics
    this.createCounter('http_requests_total', 'Total HTTP requests', ['method', 'status', 'path']);
    this.createHistogram('http_request_duration', 'HTTP request duration in ms', ['method', 'path']);
    this.createGauge('http_active_requests', 'Active HTTP requests');

    // Error metrics
    this.createCounter('errors_total', 'Total errors', ['type', 'severity']);
    this.createCounter('security_events_total', 'Total security events', ['type', 'ip']);
    this.createCounter('security_blocks_total', 'Total security blocks', ['reason', 'ip']);

    // Rate limiting metrics
    this.createCounter('rate_limit_hits_total', 'Total rate limit hits', ['key']);
    this.createGauge('rate_limit_active_keys', 'Active rate limit keys');

    // Health check metrics
    this.createCounter('health_checks_total', 'Total health checks', ['name', 'status']);
    this.createGauge('health_check_duration', 'Health check duration in ms', ['name']);

    // Generation metrics
    this.createCounter('generation_requests_total', 'Total generation requests', ['status']);
    this.createHistogram('generation_duration', 'Generation duration in ms', ['generator']);
    this.createGauge('generation_active_tasks', 'Active generation tasks');
    this.createHistogram('generation_records_count', 'Number of records generated', ['generator']);

    // Validation metrics
    this.createCounter('validation_checks_total', 'Total validation checks', ['type', 'status']);
    this.createHistogram('validation_score', 'Validation scores', ['type']);

    // System metrics
    this.createGauge('memory_usage_bytes', 'Memory usage in bytes', ['type']);
    this.createGauge('cpu_usage_percent', 'CPU usage percentage');
    this.createGauge('event_loop_lag_ms', 'Event loop lag in milliseconds');
    this.createGauge('active_handles', 'Active handles');
    this.createGauge('active_requests', 'Active requests');

    // Business metrics
    this.createCounter('pipeline_executions_total', 'Total pipeline executions', ['pipeline', 'status']);
    this.createHistogram('quality_scores', 'Quality assessment scores', ['metric']);
    this.createHistogram('privacy_scores', 'Privacy analysis scores', ['metric']);
  }

  createCounter(name, description, labels = []) {
    this.counters.set(name, {
      name,
      description,
      labels,
      values: new Map(),
      createdAt: Date.now(),
    });
  }

  createHistogram(name, description, labels = []) {
    this.histograms.set(name, {
      name,
      description,
      labels,
      buckets: [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000],
      values: new Map(),
      createdAt: Date.now(),
    });
  }

  createGauge(name, description, labels = []) {
    this.gauges.set(name, {
      name,
      description,
      labels,
      value: 0,
      values: labels.length > 0 ? new Map() : null,
      lastUpdated: Date.now(),
      createdAt: Date.now(),
    });
  }

  createTimer(name, description, labels = []) {
    this.timers.set(name, {
      name,
      description,
      labels,
      startTimes: new Map(),
      durations: new Map(),
      createdAt: Date.now(),
    });
  }

  // Counter operations
  incrementCounter(name, labels = {}, value = 1) {
    const counter = this.counters.get(name);
    if (!counter) return;

    const labelKey = this.generateLabelKey(labels);
    const currentValue = counter.values.get(labelKey) || 0;
    counter.values.set(labelKey, currentValue + value);

    this.emit('metric', { type: 'counter', name, labels, value: currentValue + value });
  }

  // Histogram operations
  recordHistogram(name, value, labels = {}) {
    const histogram = this.histograms.get(name);
    if (!histogram) return;

    const labelKey = this.generateLabelKey(labels);
    if (!histogram.values.has(labelKey)) {
      histogram.values.set(labelKey, {
        count: 0,
        sum: 0,
        buckets: new Map(histogram.buckets.map(bucket => [bucket, 0])),
        values: [], // Store raw values for percentiles
      });
    }

    const record = histogram.values.get(labelKey);
    record.count++;
    record.sum += value;
    record.values.push(value);

    // Keep only recent values for percentile calculation
    if (record.values.length > 1000) {
      record.values = record.values.slice(-1000);
    }

    // Update buckets
    for (const bucket of histogram.buckets) {
      if (value <= bucket) {
        record.buckets.set(bucket, record.buckets.get(bucket) + 1);
      }
    }

    this.emit('metric', { type: 'histogram', name, labels, value });
  }

  // Gauge operations
  setGauge(name, value, labels = {}) {
    const gauge = this.gauges.get(name);
    if (!gauge) return;

    gauge.lastUpdated = Date.now();

    if (gauge.values) {
      const labelKey = this.generateLabelKey(labels);
      gauge.values.set(labelKey, value);
    } else {
      gauge.value = value;
    }

    this.emit('metric', { type: 'gauge', name, labels, value });
  }

  incrementGauge(name, value = 1, labels = {}) {
    const gauge = this.gauges.get(name);
    if (!gauge) return;

    if (gauge.values) {
      const labelKey = this.generateLabelKey(labels);
      const currentValue = gauge.values.get(labelKey) || 0;
      gauge.values.set(labelKey, currentValue + value);
      gauge.lastUpdated = Date.now();
      this.emit('metric', { type: 'gauge', name, labels, value: currentValue + value });
    } else {
      gauge.value += value;
      gauge.lastUpdated = Date.now();
      this.emit('metric', { type: 'gauge', name, labels, value: gauge.value });
    }
  }

  decrementGauge(name, value = 1, labels = {}) {
    this.incrementGauge(name, -value, labels);
  }

  // Timer operations
  startTimer(name, id, labels = {}) {
    const timer = this.timers.get(name);
    if (!timer) return;

    const labelKey = this.generateLabelKey(labels);
    const key = `${labelKey}:${id}`;
    timer.startTimes.set(key, { start: Date.now(), labels });
  }

  stopTimer(name, id, labels = {}) {
    const timer = this.timers.get(name);
    if (!timer) return;

    const labelKey = this.generateLabelKey(labels);
    const key = `${labelKey}:${id}`;
    const startRecord = timer.startTimes.get(key);

    if (!startRecord) return;

    const duration = Date.now() - startRecord.start;
    timer.startTimes.delete(key);

    if (!timer.durations.has(labelKey)) {
      timer.durations.set(labelKey, []);
    }

    const durations = timer.durations.get(labelKey);
    durations.push(duration);

    // Keep only recent durations
    if (durations.length > 1000) {
      timer.durations.set(labelKey, durations.slice(-1000));
    }

    this.emit('metric', { type: 'timer', name, labels, value: duration });
    return duration;
  }

  generateLabelKey(labels) {
    if (Object.keys(labels).length === 0) return 'default';

    return Object.entries(labels)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, value]) => `${key}=${value}`)
      .join(',');
  }

  // Convenience methods for common metrics
  static recordHTTPRequest(method, path, status, duration) {
    const instance = HealthMetrics.getInstance();
    instance.incrementCounter('http_requests_total', { method, status, path });
    instance.recordHistogram('http_request_duration', duration, { method, path });
  }

  static recordError(type, severity) {
    const instance = HealthMetrics.getInstance();
    instance.incrementCounter('errors_total', { type, severity });
  }

  static recordSecurityEvent(type, ip) {
    const instance = HealthMetrics.getInstance();
    instance.incrementCounter('security_events_total', { type, ip: ip.substring(0, 10) }); // Truncate IP for privacy
  }

  static recordSecurityBlock(reason, ip) {
    const instance = HealthMetrics.getInstance();
    instance.incrementCounter('security_blocks_total', { reason, ip: ip.substring(0, 10) });
  }

  static recordRateLimit(key, hits) {
    const instance = HealthMetrics.getInstance();
    instance.incrementCounter('rate_limit_hits_total', { key: key.substring(0, 10) }); // Truncate for privacy
  }

  static recordHealthCheck(name, status) {
    const instance = HealthMetrics.getInstance();
    instance.incrementCounter('health_checks_total', { name, status });
  }

  static recordGeneration(status, duration, generator, recordCount) {
    const instance = HealthMetrics.getInstance();
    instance.incrementCounter('generation_requests_total', { status });
    if (duration) instance.recordHistogram('generation_duration', duration, { generator });
    if (recordCount) instance.recordHistogram('generation_records_count', recordCount, { generator });
  }

  static recordValidation(type, status, score) {
    const instance = HealthMetrics.getInstance();
    instance.incrementCounter('validation_checks_total', { type, status });
    if (score !== undefined) instance.recordHistogram('validation_score', score, { type });
  }

  static updateSystemMetrics() {
    const instance = HealthMetrics.getInstance();
    const memoryUsage = process.memoryUsage();

    instance.setGauge('memory_usage_bytes', memoryUsage.rss, { type: 'rss' });
    instance.setGauge('memory_usage_bytes', memoryUsage.heapTotal, { type: 'heap_total' });
    instance.setGauge('memory_usage_bytes', memoryUsage.heapUsed, { type: 'heap_used' });
    instance.setGauge('memory_usage_bytes', memoryUsage.external, { type: 'external' });

    // Event loop lag measurement
    const start = process.hrtime.bigint();
    setImmediate(() => {
      const lag = Number(process.hrtime.bigint() - start) / 1000000;
      instance.setGauge('event_loop_lag_ms', lag);
    });

    // Process statistics
    if (process._getActiveHandles) {
      instance.setGauge('active_handles', process._getActiveHandles().length);
    }
    if (process._getActiveRequests) {
      instance.setGauge('active_requests', process._getActiveRequests().length);
    }
  }

  startPeriodicReports() {
    // Update system metrics every 30 seconds
    this.systemMetricsInterval = setInterval(() => {
      HealthMetrics.updateSystemMetrics();
    }, 30000);

    // Log metrics summary every 5 minutes
    this.summaryInterval = setInterval(
      () => {
        this.logMetricsSummary();
      },
      5 * 60 * 1000,
    );
  }

  logMetricsSummary() {
    const summary = this.getMetricsSummary();

    this.logger.info('Metrics Summary', {
      uptime: Date.now() - this.startTime,
      counters: Object.keys(summary.counters).length,
      gauges: Object.keys(summary.gauges).length,
      histograms: Object.keys(summary.histograms).length,
      topCounters: this.getTopMetrics(summary.counters, 5),
      systemHealth: summary.gauges.memory_usage_bytes
        ? {
            memoryMB: Math.round(summary.gauges.memory_usage_bytes.rss / 1024 / 1024),
            eventLoopLag: summary.gauges.event_loop_lag_ms,
          }
        : null,
    });
  }

  getTopMetrics(metrics, limit = 10) {
    return Object.entries(metrics)
      .sort(([, a], [, b]) => (b.total || b.value || 0) - (a.total || a.value || 0))
      .slice(0, limit)
      .reduce((acc, [key, value]) => {
        acc[key] = value.total || value.value || 0;
        return acc;
      }, {});
  }

  // Export metrics in Prometheus format
  exportPrometheusMetrics() {
    let output = '';

    // Export counters
    for (const [name, counter] of this.counters) {
      output += `# HELP ${name} ${counter.description}\n`;
      output += `# TYPE ${name} counter\n`;

      for (const [labelKey, value] of counter.values) {
        const labels = labelKey === 'default' ? '' : `{${labelKey}}`;
        output += `${name}${labels} ${value}\n`;
      }
      output += '\n';
    }

    // Export gauges
    for (const [name, gauge] of this.gauges) {
      output += `# HELP ${name} ${gauge.description}\n`;
      output += `# TYPE ${name} gauge\n`;

      if (gauge.values) {
        for (const [labelKey, value] of gauge.values) {
          const labels = labelKey === 'default' ? '' : `{${labelKey}}`;
          output += `${name}${labels} ${value}\n`;
        }
      } else {
        output += `${name} ${gauge.value}\n`;
      }
      output += '\n';
    }

    // Export histograms
    for (const [name, histogram] of this.histograms) {
      output += `# HELP ${name} ${histogram.description}\n`;
      output += `# TYPE ${name} histogram\n`;

      for (const [labelKey, record] of histogram.values) {
        const baseLabels = labelKey === 'default' ? '' : labelKey;

        // Export buckets
        for (const [bucket, count] of record.buckets) {
          const labels = baseLabels ? `{${baseLabels},le="${bucket}"}` : `{le="${bucket}"}`;
          output += `${name}_bucket${labels} ${count}\n`;
        }

        // Export sum and count
        const labels = baseLabels ? `{${baseLabels}}` : '';
        output += `${name}_sum${labels} ${record.sum}\n`;
        output += `${name}_count${labels} ${record.count}\n`;
      }
      output += '\n';
    }

    return output;
  }

  // Get comprehensive metrics summary
  getMetricsSummary() {
    const summary = {
      timestamp: Date.now(),
      uptime: Date.now() - this.startTime,
      counters: {},
      gauges: {},
      histograms: {},
      timers: {},
    };

    // Summarize counters
    for (const [name, counter] of this.counters) {
      summary.counters[name] = {
        total: Array.from(counter.values.values()).reduce((sum, val) => sum + val, 0),
        labels: counter.values.size,
        description: counter.description,
      };
    }

    // Summarize gauges
    for (const [name, gauge] of this.gauges) {
      if (gauge.values) {
        const values = Array.from(gauge.values.values());
        summary.gauges[name] = {
          current: values.length > 0 ? Math.max(...values) : 0,
          average: values.length > 0 ? values.reduce((sum, val) => sum + val, 0) / values.length : 0,
          labels: gauge.values.size,
          description: gauge.description,
        };
      } else {
        summary.gauges[name] = {
          value: gauge.value,
          description: gauge.description,
          lastUpdated: gauge.lastUpdated,
        };
      }
    }

    // Summarize histograms
    for (const [name, histogram] of this.histograms) {
      const allRecords = Array.from(histogram.values.values());
      const totalCount = allRecords.reduce((sum, record) => sum + record.count, 0);
      const totalSum = allRecords.reduce((sum, record) => sum + record.sum, 0);

      summary.histograms[name] = {
        count: totalCount,
        sum: totalSum,
        average: totalCount > 0 ? totalSum / totalCount : 0,
        labels: histogram.values.size,
        description: histogram.description,
      };
    }

    return summary;
  }

  // Reset metrics
  reset() {
    for (const counter of this.counters.values()) {
      counter.values.clear();
    }

    for (const histogram of this.histograms.values()) {
      histogram.values.clear();
    }

    for (const gauge of this.gauges.values()) {
      if (gauge.values) {
        gauge.values.clear();
      } else {
        gauge.value = 0;
      }
    }

    for (const timer of this.timers.values()) {
      timer.startTimes.clear();
      timer.durations.clear();
    }

    this.resetTime = Date.now();
    this.logger.info('Metrics reset');
  }

  // Cleanup
  cleanup() {
    if (this.systemMetricsInterval) clearInterval(this.systemMetricsInterval);
    if (this.summaryInterval) clearInterval(this.summaryInterval);

    this.counters.clear();
    this.histograms.clear();
    this.gauges.clear();
    this.timers.clear();
    this.removeAllListeners();
  }
}

// HTTP request tracking middleware
export function createMetricsMiddleware(logger) {
  const metrics = HealthMetrics.getInstance(logger);

  return (req, res, next) => {
    const startTime = Date.now();
    const startTimer = metrics.startTimer || (() => {});
    const stopTimer = metrics.stopTimer || (() => {});

    metrics.incrementGauge('http_active_requests');

    res.on('finish', () => {
      const duration = Date.now() - startTime;
      const path = req.route?.path || req.path || req.url;

      HealthMetrics.recordHTTPRequest(req.method, path, res.statusCode, duration);
      metrics.decrementGauge('http_active_requests');
    });

    next();
  };
}
