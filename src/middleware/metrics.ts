import { Request, Response, NextFunction } from 'express';
import { register, collectDefaultMetrics, Counter, Histogram, Gauge } from 'prom-client';

// Enable default metrics collection
collectDefaultMetrics({ prefix: 'synthetic_guardian_' });

// Custom metrics
export const httpRequestsTotal = new Counter({
  name: 'synthetic_guardian_http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code'],
});

export const httpRequestDuration = new Histogram({
  name: 'synthetic_guardian_http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.1, 0.5, 1, 2, 5, 10],
});

export const activeConnections = new Gauge({
  name: 'synthetic_guardian_active_connections',
  help: 'Number of active connections',
});

// Business metrics
export const syntheticDataGenerations = new Counter({
  name: 'synthetic_guardian_data_generations_total',
  help: 'Total number of synthetic data generations',
  labelNames: ['generator_type', 'data_type', 'status'],
});

export const syntheticDataSize = new Histogram({
  name: 'synthetic_guardian_data_size_bytes',
  help: 'Size of generated synthetic data in bytes',
  labelNames: ['generator_type', 'data_type'],
  buckets: [1024, 10240, 102400, 1048576, 10485760, 104857600], // 1KB to 100MB
});

export const validationTime = new Histogram({
  name: 'synthetic_guardian_validation_duration_seconds',
  help: 'Time taken for data validation in seconds',
  labelNames: ['validator_type'],
  buckets: [1, 5, 10, 30, 60, 300],
});

export const qualityScores = new Histogram({
  name: 'synthetic_guardian_quality_score',
  help: 'Data quality scores',
  labelNames: ['metric_type'],
  buckets: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
});

// Security metrics
export const securityScans = new Counter({
  name: 'synthetic_guardian_security_scans_total',
  help: 'Total number of security scans performed',
  labelNames: ['scan_type', 'status'],
});

export const vulnerabilities = new Counter({
  name: 'synthetic_guardian_vulnerabilities_total',
  help: 'Total number of vulnerabilities found',
  labelNames: ['severity', 'type'],
});

// Performance metrics
export const databaseConnections = new Gauge({
  name: 'synthetic_guardian_database_connections',
  help: 'Number of active database connections',
  labelNames: ['database_type'],
});

export const cacheHitRate = new Gauge({
  name: 'synthetic_guardian_cache_hit_rate',
  help: 'Cache hit rate percentage',
  labelNames: ['cache_type'],
});

export const queueSize = new Gauge({
  name: 'synthetic_guardian_queue_size',
  help: 'Number of jobs in queue',
  labelNames: ['queue_name'],
});

// Error tracking
export const errors = new Counter({
  name: 'synthetic_guardian_errors_total',
  help: 'Total number of errors',
  labelNames: ['error_type', 'component'],
});

// Middleware to track HTTP metrics
export const metricsMiddleware = (req: Request, res: Response, next: NextFunction): void => {
  const start = Date.now();
  
  // Track active connections
  activeConnections.inc();
  
  // Continue with the request
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    const route = req.route?.path || req.path;
    const method = req.method;
    const statusCode = res.statusCode.toString();
    
    // Record metrics
    httpRequestsTotal.inc({
      method,
      route,
      status_code: statusCode,
    });
    
    httpRequestDuration.observe(
      {
        method,
        route,
        status_code: statusCode,
      },
      duration
    );
    
    // Decrease active connections
    activeConnections.dec();
  });
  
  next();
};

// Health check metrics
export const healthChecks = new Counter({
  name: 'synthetic_guardian_health_checks_total',
  help: 'Total number of health checks',
  labelNames: ['component', 'status'],
});

// Custom metrics for business logic
export class BusinessMetrics {
  static recordGeneration(
    generatorType: string,
    dataType: string,
    status: 'success' | 'failure',
    sizeBytes?: number
  ): void {
    syntheticDataGenerations.inc({
      generator_type: generatorType,
      data_type: dataType,
      status,
    });
    
    if (sizeBytes && status === 'success') {
      syntheticDataSize.observe(
        {
          generator_type: generatorType,
          data_type: dataType,
        },
        sizeBytes
      );
    }
  }
  
  static recordValidation(validatorType: string, durationSeconds: number): void {
    validationTime.observe({ validator_type: validatorType }, durationSeconds);
  }
  
  static recordQualityScore(metricType: string, score: number): void {
    qualityScores.observe({ metric_type: metricType }, score);
  }
  
  static recordError(errorType: string, component: string): void {
    errors.inc({ error_type: errorType, component });
  }
  
  static updateDatabaseConnections(databaseType: string, count: number): void {
    databaseConnections.set({ database_type: databaseType }, count);
  }
  
  static updateCacheHitRate(cacheType: string, rate: number): void {
    cacheHitRate.set({ cache_type: cacheType }, rate);
  }
  
  static updateQueueSize(queueName: string, size: number): void {
    queueSize.set({ queue_name: queueName }, size);
  }
}

// Security metrics class
export class SecurityMetrics {
  static recordScan(scanType: string, status: 'success' | 'failure'): void {
    securityScans.inc({ scan_type: scanType, status });
  }
  
  static recordVulnerability(severity: string, type: string): void {
    vulnerabilities.inc({ severity, type });
  }
}

// Health check helper
export class HealthMetrics {
  static recordHealthCheck(component: string, status: 'healthy' | 'unhealthy'): void {
    healthChecks.inc({ component, status });
  }
}

// Export the registry for the metrics endpoint
export { register };

// Function to get all metrics
export const getMetrics = async (): Promise<string> => {
  return register.metrics();
};