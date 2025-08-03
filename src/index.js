#!/usr/bin/env node

/**
 * Synthetic Data Guardian - Main Entry Point
 * Enterprise-grade synthetic data pipeline with validation and lineage tracking
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import winston from 'winston';
import { Guardian } from './core/Guardian.js';
import { GenerationPipeline } from './core/GenerationPipeline.js';
import { healthRouter } from './routes/health.js';
import { apiRouter } from './routes/api.js';
import { metricsMiddleware } from './middleware/metrics.js';
import { errorHandler } from './middleware/errorHandler.js';
import { validationMiddleware } from './middleware/validation.js';

// Load environment configuration
dotenv.config();

// Configure logging
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'synthetic-data-guardian' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

class SyntheticDataGuardianServer {
  constructor() {
    this.app = express();
    this.port = process.env.PORT || 8080;
    this.guardian = new Guardian(logger);
    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
  }

  setupMiddleware() {
    // Security middleware
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: ["'self'", "'unsafe-inline'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          imgSrc: ["'self'", "data:", "https:"]
        }
      }
    }));

    // CORS configuration
    this.app.use(cors({
      origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
      credentials: true
    }));

    // Body parsing
    this.app.use(express.json({ limit: '50mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '50mb' }));

    // Custom middleware
    this.app.use(metricsMiddleware);
    this.app.use(validationMiddleware);

    // Request logging
    this.app.use((req, res, next) => {
      logger.info(`${req.method} ${req.path}`, {
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        requestId: req.id
      });
      next();
    });
  }

  setupRoutes() {
    // Health check endpoints
    this.app.use('/health', healthRouter);
    
    // API routes
    this.app.use('/api/v1', apiRouter(this.guardian, logger));

    // Root endpoint
    this.app.get('/', (req, res) => {
      res.json({
        name: 'Synthetic Data Guardian',
        version: process.env.npm_package_version || '1.0.0',
        description: 'Enterprise-grade synthetic data pipeline',
        documentation: '/api/v1/docs',
        health: '/health'
      });
    });

    // API documentation
    this.app.get('/api/v1/docs', (req, res) => {
      res.json({
        title: 'Synthetic Data Guardian API',
        version: '1.0.0',
        endpoints: {
          'POST /api/v1/generate': 'Generate synthetic data',
          'POST /api/v1/validate': 'Validate data quality',
          'GET /api/v1/lineage/:id': 'Get lineage information',
          'POST /api/v1/watermark': 'Apply watermarking',
          'GET /api/v1/status': 'Get generation status'
        }
      });
    });
  }

  setupErrorHandling() {
    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        error: 'Not Found',
        message: `Endpoint ${req.method} ${req.originalUrl} not found`,
        timestamp: new Date().toISOString()
      });
    });

    // Global error handler
    this.app.use(errorHandler(logger));
  }

  async start() {
    try {
      // Initialize guardian components
      await this.guardian.initialize();
      
      // Start server
      this.app.listen(this.port, () => {
        logger.info(`Synthetic Data Guardian started on port ${this.port}`, {
          environment: process.env.NODE_ENV || 'development',
          port: this.port,
          pid: process.pid
        });
      });

      // Graceful shutdown handling
      process.on('SIGTERM', () => this.shutdown('SIGTERM'));
      process.on('SIGINT', () => this.shutdown('SIGINT'));

    } catch (error) {
      logger.error('Failed to start server', { error: error.message, stack: error.stack });
      process.exit(1);
    }
  }

  async shutdown(signal) {
    logger.info(`Received ${signal}, shutting down gracefully`);
    
    try {
      await this.guardian.cleanup();
      logger.info('Guardian cleanup completed');
      process.exit(0);
    } catch (error) {
      logger.error('Error during shutdown', { error: error.message });
      process.exit(1);
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
    import('./cli/commands.js').then(({ handleCommand }) => {
      handleCommand(command, args.slice(1));
    }).catch(error => {
      console.error('CLI Error:', error.message);
      process.exit(1);
    });
  }
}

export { SyntheticDataGuardianServer, Guardian, GenerationPipeline };