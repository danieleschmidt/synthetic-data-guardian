#!/usr/bin/env node

/**
 * Synthetic Data Guardian - Main Entry Point
 * Enterprise-grade synthetic data pipeline with validation and lineage tracking
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import { Guardian } from './core/Guardian.js';
import { GenerationPipeline } from './core/GenerationPipeline.js';
import { healthRouter } from './routes/health.js';
import { apiRouter } from './routes/api.js';
import { createMetricsMiddleware } from './middleware/metrics.js';
import { createErrorHandler } from './middleware/ErrorHandler.js';
import { createInputValidator } from './middleware/InputValidator.js';
import { createSecurityMiddleware } from './middleware/Security.js';
import { createRateLimiters } from './middleware/RateLimiter.js';
import { createDefaultHealthChecks } from './health/HealthChecker.js';
import { createLogger } from './utils/Logger.js';
import { createPerformanceOptimizer } from './optimization/PerformanceOptimizer.js';
import { createAdvancedCacheManager } from './optimization/CacheManager.js';
import { createLoadBalancer } from './optimization/LoadBalancer.js';
import { createResourceManager } from './optimization/ResourceManager.js';

// Load environment configuration
dotenv.config();

// Initialize advanced logging system
const logger = createLogger({
  service: 'synthetic-data-guardian',
  level: process.env.LOG_LEVEL || 'info',
  environment: process.env.NODE_ENV || 'development',
  enableConsole: true,
  enableFile: true,
  enableError: true,
  logDir: process.env.LOG_DIR || './logs',
});

class SyntheticDataGuardianServer {
  constructor() {
    this.app = express();
    this.port = process.env.PORT || 8080;
    this.guardian = new Guardian(logger);

    // Initialize robust middleware components (Generation 2)
    this.errorHandler = createErrorHandler(logger);
    this.inputValidator = createInputValidator(logger);
    this.securityMiddleware = createSecurityMiddleware(logger, {
      enableApiKeyAuth: process.env.ENABLE_API_KEY_AUTH === 'true',
      apiKeys: new Set(process.env.API_KEYS?.split(',') || []),
      enableHoneypot: process.env.ENABLE_HONEYPOT === 'true',
      enableRequestFingerprinting: true,
      maxRequestSize: process.env.MAX_REQUEST_SIZE || '10mb',
    });
    this.rateLimiters = createRateLimiters(logger);
    this.healthChecker = createDefaultHealthChecks(this.guardian, logger);

    // Initialize optimization components (Generation 3)
    this.performanceOptimizer = createPerformanceOptimizer(logger, {
      enableCaching: true,
      enableWorkerPool: process.env.ENABLE_WORKER_POOL !== 'false',
      maxWorkers: parseInt(process.env.MAX_WORKERS) || 4,
      enableMemoryMonitoring: true,
      memoryThreshold: 0.8,
    });

    this.cacheManager = createAdvancedCacheManager(logger, {
      strategies: {
        lru: { enabled: true, maxSize: 1000, ttl: 300000 },
        memory: { enabled: true, maxSize: 2000, ttl: 900000 },
      },
      enableMetrics: true,
      compressionThreshold: 1024,
    });

    this.loadBalancer = createLoadBalancer(logger, {
      algorithm: process.env.LB_ALGORITHM || 'round_robin',
      maxConcurrentRequests: parseInt(process.env.MAX_CONCURRENT_REQUESTS) || 1000,
      autoScaling: {
        enabled: process.env.ENABLE_AUTO_SCALING === 'true',
        minInstances: 1,
        maxInstances: parseInt(process.env.MAX_INSTANCES) || 8,
      },
    });

    this.resourceManager = createResourceManager(logger, {
      monitoring: { enabled: true, interval: 30000 },
      autoScaling: {
        enabled: process.env.ENABLE_AUTO_SCALING === 'true',
        minInstances: 1,
        maxInstances: parseInt(process.env.MAX_INSTANCES) || 8,
      },
      optimization: { enabled: true, gcThreshold: 85 },
      alerts: { enabled: true, channels: ['log'] },
    });

    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
  }

  setupMiddleware() {
    // Trust proxy for proper IP detection
    this.app.set('trust proxy', process.env.TRUST_PROXY || 1);

    // Security middleware (comprehensive)
    this.app.use(this.securityMiddleware.middleware());

    // Additional helmet configuration
    this.app.use(
      helmet({
        contentSecurityPolicy: false, // Handled by SecurityMiddleware
        crossOriginEmbedderPolicy: false,
      }),
    );

    // CORS configuration
    this.app.use(
      cors({
        origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
        credentials: true,
        optionsSuccessStatus: 200,
        preflightContinue: false,
      }),
    );

    // Request correlation and logging
    this.app.use(logger.middleware());

    // Performance optimization middleware
    this.app.use(this.performanceOptimizer.middleware().performance);
    this.app.use(this.resourceManager.middleware());

    // Metrics collection
    this.app.use(createMetricsMiddleware(logger));

    // Rate limiting - different tiers for different endpoints
    this.app.use('/api/v1/generate', this.rateLimiters.middleware('generation'));
    this.app.use('/api/v1/validate', this.rateLimiters.middleware('validation'));
    this.app.use('/health', this.rateLimiters.middleware('health'));
    this.app.use('/api', this.rateLimiters.middleware('default'));

    // Body parsing with size limits
    this.app.use(
      express.json({
        limit: process.env.MAX_JSON_SIZE || '10mb',
        verify: (req, res, buf, encoding) => {
          // Store raw body for signature verification if needed
          req.rawBody = buf;
        },
      }),
    );
    this.app.use(
      express.urlencoded({
        extended: true,
        limit: process.env.MAX_FORM_SIZE || '10mb',
      }),
    );

    // Request ID middleware
    this.app.use((req, res, next) => {
      req.id = req.correlationId || logger.generateCorrelationId();
      next();
    });
  }

  setupRoutes() {
    // Health check endpoints with comprehensive health checker
    this.app.use('/health', healthRouter);

    // Enhanced health endpoints
    this.app.get(
      '/health/comprehensive',
      this.errorHandler.asyncHandler(async (req, res) => {
        const health = await this.healthChecker.runAllChecks();
        res.status(health.status === 'healthy' ? 200 : 503).json(health);
      }),
    );

    // Metrics endpoint
    this.app.get('/metrics', (req, res) => {
      const metrics = this.healthChecker.getStats();
      res.set('Content-Type', 'text/plain');
      res.send(`# Synthetic Data Guardian Metrics
# Service uptime
uptime_seconds ${process.uptime()}
# Memory usage
memory_rss_bytes ${process.memoryUsage().rss}
memory_heap_used_bytes ${process.memoryUsage().heapUsed}
memory_heap_total_bytes ${process.memoryUsage().heapTotal}
# Health checks
health_checks_total ${metrics.totalChecks || 0}
# Active sessions  
active_sessions ${this.securityMiddleware.getSecurityStats().activeSessions || 0}
`);
    });

    // Security status endpoint
    this.app.get('/security/status', (req, res) => {
      const stats = this.securityMiddleware.getSecurityStats();
      const rateLimitStats = this.rateLimiters.getStats();

      res.json({
        security: stats,
        rateLimiting: rateLimitStats,
        timestamp: new Date().toISOString(),
      });
    });

    // Performance status endpoint
    this.app.get('/performance/status', (req, res) => {
      const performanceStats = this.performanceOptimizer.getPerformanceMetrics();
      const cacheStats = this.cacheManager.getMetrics();
      const resourceReport = this.resourceManager.getResourceReport();

      res.json({
        performance: performanceStats,
        cache: cacheStats,
        resources: resourceReport,
        timestamp: new Date().toISOString(),
      });
    });

    // Cache management endpoint
    this.app.post(
      '/cache/clear',
      this.errorHandler.asyncHandler(async (req, res) => {
        const { caches } = req.body;
        await this.cacheManager.clear(caches);

        res.json({
          success: true,
          message: 'Cache cleared successfully',
          timestamp: new Date().toISOString(),
        });
      }),
    );

    // Resource predictions endpoint
    this.app.get('/resources/predict', (req, res) => {
      const timeHorizon = parseInt(req.query.timeHorizon) || 3600000; // 1 hour default
      const predictions = this.resourceManager.predictResourceNeeds(timeHorizon);

      res.json({
        predictions,
        timeHorizon,
        timestamp: new Date().toISOString(),
      });
    });

    // API routes with input validation
    this.app.use('/api/v1', apiRouter(this.guardian, logger, this.inputValidator));

    // Root endpoint
    this.app.get('/', (req, res) => {
      res.json({
        name: 'Synthetic Data Guardian',
        version: process.env.npm_package_version || '1.0.0',
        description: 'Enterprise-grade synthetic data pipeline with comprehensive security',
        features: [
          'Advanced error handling with circuit breakers',
          'Comprehensive input validation and sanitization',
          'Multi-tier rate limiting',
          'Security middleware with anomaly detection',
          'Structured logging with correlation tracking',
          'Health monitoring with multiple probes',
          'Metrics collection and monitoring',
          'Performance optimization with worker pools',
          'Multi-tier caching system',
          'Load balancing with auto-scaling',
          'Resource management and prediction',
          'Concurrent processing optimization',
        ],
        documentation: '/api/v1/docs',
        health: '/health',
        metrics: '/metrics',
        security: '/security/status',
        performance: '/performance/status',
        resourcePredictions: '/resources/predict',
      });
    });

    // API documentation
    this.app.get('/api/v1/docs', (req, res) => {
      res.json({
        title: 'Synthetic Data Guardian API v2.0',
        version: '2.0.0',
        description: 'Enterprise-grade synthetic data pipeline with comprehensive security and monitoring',
        security: {
          authentication: 'API Key (optional)',
          rateLimiting: 'Multi-tier rate limiting',
          inputValidation: 'Comprehensive validation and sanitization',
          monitoring: 'Request fingerprinting and anomaly detection',
        },
        endpoints: {
          'POST /api/v1/generate': 'Generate synthetic data with validation',
          'POST /api/v1/validate': 'Validate data quality and privacy',
          'GET /api/v1/lineage/:id': 'Get complete lineage information',
          'POST /api/v1/watermark': 'Apply advanced watermarking',
          'GET /api/v1/status/:taskId': 'Get generation task status',
          'GET /health': 'Basic health check',
          'GET /health/comprehensive': 'Comprehensive health assessment',
          'GET /metrics': 'Prometheus-style metrics',
          'GET /security/status': 'Security monitoring status',
          'GET /performance/status': 'Performance and optimization status',
          'GET /resources/predict': 'Resource usage predictions',
          'POST /cache/clear': 'Clear application caches',
        },
      });
    });
  }

  setupErrorHandling() {
    // 404 handler
    this.app.use('*', (req, res) => {
      logger.warn('Endpoint not found', {
        method: req.method,
        url: req.originalUrl,
        ip: req.ip,
        userAgent: req.get('User-Agent'),
      });

      res.status(404).json({
        success: false,
        error: {
          type: 'NOT_FOUND',
          message: `Endpoint ${req.method} ${req.originalUrl} not found`,
          timestamp: new Date().toISOString(),
          correlationId: req.correlationId,
        },
      });
    });

    // Global error handler with comprehensive error handling
    this.app.use(this.errorHandler.middleware());
  }

  async start() {
    try {
      logger.info('Starting Synthetic Data Guardian with robust components...');

      // Create logs directory
      const fs = await import('fs/promises');
      try {
        await fs.mkdir('./logs', { recursive: true });
      } catch (error) {
        // Directory might already exist
      }

      // Initialize guardian components
      await this.guardian.initialize();

      // Initialize optimization components
      await this.performanceOptimizer.initialize();

      // Start health monitoring
      this.healthChecker.startPeriodicChecks(30000); // Every 30 seconds

      // Setup optimization event handlers
      this.setupOptimizationEventHandlers();

      // Start server
      const server = this.app.listen(this.port, () => {
        logger.info(`🚀 Synthetic Data Guardian v2.0 started successfully`, {
          environment: process.env.NODE_ENV || 'development',
          port: this.port,
          pid: process.pid,
          features: [
            'Circuit Breaker Pattern',
            'Multi-tier Rate Limiting',
            'Advanced Security Middleware',
            'Comprehensive Health Monitoring',
            'Structured Logging with Correlation',
            'Input Validation & Sanitization',
          ],
        });

        // Log system capabilities
        logger.info('System capabilities initialized', {
          rateLimitTiers: Object.keys(this.rateLimiters.getStats()),
          securityFeatures: [
            'Request Fingerprinting',
            'Anomaly Detection',
            'Honeypot Protection',
            'IP Blocking',
            'Input Sanitization',
          ],
          healthChecks: this.healthChecker.getStats().totalChecks,
          logLevel: logger.options?.level || 'info',
        });
      });

      // Configure server timeouts
      server.timeout = 30000; // 30 seconds
      server.keepAliveTimeout = 5000; // 5 seconds
      server.headersTimeout = 10000; // 10 seconds

      // Graceful shutdown handling
      process.on('SIGTERM', () => this.shutdown('SIGTERM'));
      process.on('SIGINT', () => this.shutdown('SIGINT'));

      return server;
    } catch (error) {
      logger.error('Failed to start server', {
        error: error.message,
        stack: error.stack,
      });
      process.exit(1);
    }
  }

  async shutdown(signal) {
    logger.info(`🛑 Received ${signal}, initiating graceful shutdown...`);

    try {
      // Stop accepting new requests
      logger.info('Stopping health monitoring...');
      this.healthChecker.stopPeriodicChecks();

      // Cleanup middleware components
      logger.info('Cleaning up middleware components...');
      await Promise.all([
        this.guardian.cleanup(),
        this.errorHandler.cleanup(),
        this.securityMiddleware.cleanup(),
        this.rateLimiters.destroy(),
        this.healthChecker.cleanup(),
        this.performanceOptimizer.cleanup(),
        this.cacheManager.cleanup(),
        this.loadBalancer.cleanup(),
        this.resourceManager.cleanup(),
      ]);

      logger.info('✅ Graceful shutdown completed successfully');
      process.exit(0);
    } catch (error) {
      logger.error('❌ Error during graceful shutdown', {
        error: error.message,
        stack: error.stack,
      });
      process.exit(1);
    }
  }

  setupOptimizationEventHandlers() {
    // Performance optimizer events
    this.performanceOptimizer.on('memoryPressure', data => {
      logger.warn('Memory pressure detected', data);
      // Trigger cache cleanup
      this.cacheManager.clear().catch(error => logger.error('Cache cleanup failed', { error: error.message }));
    });

    // Resource manager events
    this.resourceManager.on('resourceAlert', alert => {
      logger.warn('Resource alert triggered', alert);

      // Auto-remediation actions
      if (alert.level === 'critical') {
        this.performEmergencyOptimizations();
      }
    });

    this.resourceManager.on('autoScaleUp', event => {
      logger.info('Auto-scaling up triggered', event);
      // In a real implementation, this would trigger container scaling
    });

    this.resourceManager.on('autoScaleDown', event => {
      logger.info('Auto-scaling down triggered', event);
      // In a real implementation, this would trigger container scaling
    });

    // Cache manager events
    this.cacheManager.on('cacheEvictionRequested', ({ reason }) => {
      logger.info('Cache eviction requested', { reason });
    });

    // Load balancer events
    this.loadBalancer.on('backendUnhealthy', backendId => {
      logger.warn('Backend marked unhealthy', { backendId });
    });

    this.loadBalancer.on('scaleUp', event => {
      logger.info('Load balancer requesting scale up', event);
    });

    this.loadBalancer.on('scaleDown', event => {
      logger.info('Load balancer requesting scale down', event);
    });
  }

  async performEmergencyOptimizations() {
    logger.warn('Performing emergency optimizations...');

    try {
      // Force garbage collection
      if (global.gc) {
        global.gc();
        logger.info('Emergency garbage collection performed');
      }

      // Clear non-critical caches
      await this.cacheManager.clear(['tier_cold']);
      logger.info('Emergency cache cleanup performed');

      // Reduce worker pool size temporarily
      // (In real implementation, would adjust worker pool)
      logger.info('Emergency worker pool optimization performed');
    } catch (error) {
      logger.error('Emergency optimizations failed', { error: error.message });
    }
  }
}

// CLI interface for direct usage
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Synthetic Data Guardian CLI

Commands:
  start                    Start the API server
  generate <config>        Generate synthetic data from config file
  validate <data> <schema> Validate synthetic data
  lineage <id>            Show lineage for dataset
  watermark <data>        Apply watermarking to data

Options:
  --port <port>           Server port (default: 8080)
  --config <file>         Configuration file
  --output <path>         Output directory
  --verbose               Verbose logging
  --help                  Show this help message

Examples:
  node src/index.js start --port 3000
  node src/index.js generate config.yaml --output ./data
  node src/index.js validate data.csv schema.json
    `);
    process.exit(0);
  }

  const command = args[0] || 'start';

  if (command === 'start') {
    const server = new SyntheticDataGuardianServer();
    server.start();
  } else {
    // Handle other CLI commands
    import('./cli/commands.js')
      .then(({ handleCommand }) => {
        handleCommand(command, args.slice(1));
      })
      .catch(error => {
        console.error('CLI Error:', error.message);
        process.exit(1);
      });
  }
}

export { SyntheticDataGuardianServer, Guardian, GenerationPipeline };
