/**
 * Advanced Logging System with Structured Logging and Multiple Transports
 */

import winston from 'winston';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class AdvancedLogger {
  constructor(options = {}) {
    this.options = {
      level: options.level || process.env.LOG_LEVEL || 'info',
      service: options.service || 'synthetic-data-guardian',
      version: options.version || process.env.npm_package_version || '1.0.0',
      environment: options.environment || process.env.NODE_ENV || 'development',
      enableConsole: options.enableConsole !== false,
      enableFile: options.enableFile !== false,
      enableError: options.enableError !== false,
      logDir: options.logDir || path.join(process.cwd(), 'logs'),
      maxFileSize: options.maxFileSize || '20m',
      maxFiles: options.maxFiles || 5,
      enableStructured: options.enableStructured !== false,
      enableCorrelation: options.enableCorrelation !== false,
      sensitiveFields: options.sensitiveFields || ['password', 'token', 'apiKey', 'secret', 'key'],
      ...options,
    };

    this.logger = null;
    this.correlationStorage = new Map();
    this.requestCounter = 0;

    this.initializeLogger();
  }

  initializeLogger() {
    const transports = [];
    const format = this.createLogFormat();

    // Console transport
    if (this.options.enableConsole) {
      transports.push(
        new winston.transports.Console({
          level: this.options.level,
          format:
            this.options.environment === 'development'
              ? winston.format.combine(winston.format.colorize(), winston.format.simple())
              : format,
          handleExceptions: true,
          handleRejections: true,
        }),
      );
    }

    // File transports
    if (this.options.enableFile) {
      // General log file
      transports.push(
        new winston.transports.File({
          filename: path.join(this.options.logDir, 'app.log'),
          level: this.options.level,
          format,
          maxsize: this.parseSize(this.options.maxFileSize),
          maxFiles: this.options.maxFiles,
          handleExceptions: false,
        }),
      );

      // Combined log file (all levels)
      transports.push(
        new winston.transports.File({
          filename: path.join(this.options.logDir, 'combined.log'),
          level: 'silly',
          format,
          maxsize: this.parseSize(this.options.maxFileSize),
          maxFiles: this.options.maxFiles,
          handleExceptions: false,
        }),
      );
    }

    // Error log file
    if (this.options.enableError) {
      transports.push(
        new winston.transports.File({
          filename: path.join(this.options.logDir, 'error.log'),
          level: 'error',
          format,
          maxsize: this.parseSize(this.options.maxFileSize),
          maxFiles: this.options.maxFiles,
          handleExceptions: true,
          handleRejections: true,
        }),
      );
    }

    this.logger = winston.createLogger({
      level: this.options.level,
      format,
      transports,
      defaultMeta: this.getDefaultMeta(),
      exitOnError: false,
    });

    // Handle uncaught exceptions and rejections
    this.setupGlobalErrorHandlers();
  }

  createLogFormat() {
    const formats = [winston.format.timestamp()];

    if (this.options.enableStructured) {
      formats.push(winston.format.errors({ stack: true }));
      formats.push(winston.format.json());

      // Custom formatter for structured logging
      formats.push(
        winston.format.printf(info => {
          const { timestamp, level, message, service, version, environment, correlationId, ...meta } = info;

          const logEntry = {
            '@timestamp': timestamp,
            '@version': '1',
            level: level.toUpperCase(),
            message,
            service,
            version,
            environment,
            ...(correlationId && { correlationId }),
            ...(Object.keys(meta).length > 0 && { meta: this.sanitizeSensitiveData(meta) }),
          };

          return JSON.stringify(logEntry);
        }),
      );
    } else {
      formats.push(winston.format.simple());
    }

    return winston.format.combine(...formats);
  }

  getDefaultMeta() {
    return {
      service: this.options.service,
      version: this.options.version,
      environment: this.options.environment,
      hostname: process.env.HOSTNAME || os.hostname(),
      pid: process.pid,
    };
  }

  parseSize(size) {
    if (typeof size === 'number') return size;
    const units = { b: 1, k: 1024, m: 1024 ** 2, g: 1024 ** 3 };
    const match = size.toLowerCase().match(/^(\d+)([bkmg])?$/);
    return match ? parseInt(match[1]) * (units[match[2]] || 1) : 5 * 1024 * 1024; // Default 5MB
  }

  sanitizeSensitiveData(obj) {
    if (typeof obj !== 'object' || obj === null) {
      return obj;
    }

    const sanitized = Array.isArray(obj) ? [] : {};

    for (const [key, value] of Object.entries(obj)) {
      const keyLower = key.toLowerCase();
      const isSensitive = this.options.sensitiveFields.some(field => keyLower.includes(field.toLowerCase()));

      if (isSensitive) {
        sanitized[key] = '[REDACTED]';
      } else if (typeof value === 'object' && value !== null) {
        sanitized[key] = this.sanitizeSensitiveData(value);
      } else {
        sanitized[key] = value;
      }
    }

    return sanitized;
  }

  setupGlobalErrorHandlers() {
    process.on('uncaughtException', error => {
      this.logger.error('Uncaught Exception', {
        error: {
          name: error.name,
          message: error.message,
          stack: error.stack,
        },
        fatal: true,
      });

      // Allow some time for the log to be written
      setTimeout(() => {
        process.exit(1);
      }, 1000);
    });

    process.on('unhandledRejection', (reason, promise) => {
      this.logger.error('Unhandled Rejection', {
        reason: reason?.message || reason,
        stack: reason?.stack,
        promise: promise.toString(),
      });
    });

    process.on('warning', warning => {
      this.logger.warn('Process Warning', {
        name: warning.name,
        message: warning.message,
        stack: warning.stack,
      });
    });

    process.on('SIGTERM', () => {
      this.logger.info('SIGTERM received, shutting down gracefully');
    });

    process.on('SIGINT', () => {
      this.logger.info('SIGINT received, shutting down gracefully');
    });
  }

  // Correlation ID management for request tracking
  generateCorrelationId() {
    return `${Date.now()}-${++this.requestCounter}-${Math.random().toString(36).substr(2, 9)}`;
  }

  setCorrelationId(correlationId) {
    if (this.options.enableCorrelation) {
      this.correlationStorage.set('current', correlationId);
    }
  }

  getCorrelationId() {
    return this.correlationStorage.get('current');
  }

  withCorrelation(correlationId, fn) {
    const previousId = this.getCorrelationId();
    this.setCorrelationId(correlationId);

    try {
      return fn();
    } finally {
      if (previousId) {
        this.setCorrelationId(previousId);
      } else {
        this.correlationStorage.delete('current');
      }
    }
  }

  // Enhanced logging methods with context
  log(level, message, meta = {}) {
    const correlationId = this.getCorrelationId();
    const enrichedMeta = {
      ...meta,
      ...(correlationId && { correlationId }),
    };

    this.logger.log(level, message, enrichedMeta);
  }

  error(message, meta = {}) {
    this.log('error', message, meta);
  }

  warn(message, meta = {}) {
    this.log('warn', message, meta);
  }

  info(message, meta = {}) {
    this.log('info', message, meta);
  }

  debug(message, meta = {}) {
    this.log('debug', message, meta);
  }

  verbose(message, meta = {}) {
    this.log('verbose', message, meta);
  }

  silly(message, meta = {}) {
    this.log('silly', message, meta);
  }

  // Performance timing
  time(label) {
    const startTime = Date.now();
    const correlationId = this.getCorrelationId();

    return {
      end: (message = `Timer ${label} completed`, meta = {}) => {
        const duration = Date.now() - startTime;
        this.info(message, {
          ...meta,
          timing: { label, duration },
          ...(correlationId && { correlationId }),
        });
        return duration;
      },
    };
  }

  // Request/Response logging
  logRequest(req, meta = {}) {
    this.info('HTTP Request', {
      request: {
        method: req.method,
        url: req.url,
        userAgent: req.get('User-Agent'),
        ip: req.ip || req.connection?.remoteAddress,
        contentLength: req.get('Content-Length'),
        contentType: req.get('Content-Type'),
        referer: req.get('Referer'),
      },
      ...meta,
    });
  }

  logResponse(req, res, duration, meta = {}) {
    this.info('HTTP Response', {
      request: {
        method: req.method,
        url: req.url,
        ip: req.ip || req.connection?.remoteAddress,
      },
      response: {
        statusCode: res.statusCode,
        statusMessage: res.statusMessage,
        contentLength: res.get('Content-Length'),
        contentType: res.get('Content-Type'),
      },
      duration,
      ...meta,
    });
  }

  // Business event logging
  logBusinessEvent(event, data = {}) {
    this.info('Business Event', {
      eventType: event,
      eventData: this.sanitizeSensitiveData(data),
      timestamp: new Date().toISOString(),
    });
  }

  // Security event logging
  logSecurityEvent(event, severity = 'medium', data = {}) {
    this.warn('Security Event', {
      eventType: event,
      severity,
      eventData: this.sanitizeSensitiveData(data),
      timestamp: new Date().toISOString(),
    });
  }

  // Performance monitoring
  logPerformance(operation, duration, meta = {}) {
    const level = duration > 5000 ? 'warn' : duration > 1000 ? 'info' : 'debug';

    this.log(level, 'Performance Metric', {
      performance: {
        operation,
        duration,
        threshold: duration > 5000 ? 'slow' : duration > 1000 ? 'moderate' : 'fast',
      },
      ...meta,
    });
  }

  // Database query logging
  logQuery(query, duration, meta = {}) {
    this.debug('Database Query', {
      query: {
        sql: query.length > 500 ? query.substring(0, 500) + '...' : query,
        duration,
        bindings: meta.bindings ? '[REDACTED]' : undefined,
      },
      ...meta,
    });
  }

  // API call logging
  logApiCall(method, url, statusCode, duration, meta = {}) {
    const level = statusCode >= 400 ? 'warn' : statusCode >= 300 ? 'info' : 'debug';

    this.log(level, 'External API Call', {
      api: {
        method,
        url,
        statusCode,
        duration,
        success: statusCode < 400,
      },
      ...meta,
    });
  }

  // Middleware for Express
  middleware() {
    return (req, res, next) => {
      const correlationId = req.get('X-Correlation-ID') || this.generateCorrelationId();
      const startTime = Date.now();

      // Set correlation ID for this request
      this.setCorrelationId(correlationId);
      req.correlationId = correlationId;
      res.set('X-Correlation-ID', correlationId);

      // Log incoming request
      this.logRequest(req);

      // Override res.end to log response
      const originalEnd = res.end;
      res.end = (...args) => {
        const duration = Date.now() - startTime;
        this.logResponse(req, res, duration);
        originalEnd.apply(res, args);
      };

      next();
    };
  }

  // Create child logger with additional context
  child(meta) {
    return new Proxy(this, {
      get(target, prop) {
        if (
          typeof target[prop] === 'function' &&
          ['log', 'error', 'warn', 'info', 'debug', 'verbose', 'silly'].includes(prop)
        ) {
          return function (message, additionalMeta = {}) {
            return target[prop](message, { ...meta, ...additionalMeta });
          };
        }
        return target[prop];
      },
    });
  }

  // Stream for HTTP requests (for morgan middleware)
  createStream() {
    return {
      write: message => {
        this.info(message.trim());
      },
    };
  }

  // Health check for logger
  isHealthy() {
    try {
      this.debug('Logger health check');
      return {
        healthy: true,
        transports: this.logger.transports.length,
        level: this.options.level,
      };
    } catch (error) {
      return {
        healthy: false,
        error: error.message,
      };
    }
  }

  // Get logger statistics
  getStats() {
    return {
      service: this.options.service,
      level: this.options.level,
      transports: this.logger.transports.map(transport => ({
        name: transport.name,
        level: transport.level,
        filename: transport.filename,
      })),
      correlationEnabled: this.options.enableCorrelation,
      activeCorrelations: this.correlationStorage.size,
    };
  }

  // Cleanup
  cleanup() {
    if (this.logger) {
      this.logger.close();
    }
    this.correlationStorage.clear();
  }
}

// Factory function to create logger instances
export function createLogger(options = {}) {
  return new AdvancedLogger(options);
}

// Default logger instance
export const logger = createLogger();

export default logger;
