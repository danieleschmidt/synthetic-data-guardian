/**
 * Comprehensive Error Handling Middleware with Circuit Breaker Pattern
 */

import { HealthMetrics } from './metrics.js';

export class CircuitBreaker {
  constructor(options = {}) {
    this.threshold = options.threshold || 5;
    this.timeout = options.timeout || 60000; // 1 minute
    this.resetTimeout = options.resetTimeout || 30000; // 30 seconds

    this.failureCount = 0;
    this.lastFailureTime = null;
    this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
    this.nextAttempt = Date.now();

    this.successCount = 0;
    this.totalRequests = 0;
  }

  async execute(fn) {
    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN - service unavailable');
      }
      this.state = 'HALF_OPEN';
    }

    try {
      const result = await Promise.race([
        fn(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Circuit breaker timeout')), this.timeout)),
      ]);

      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  onSuccess() {
    this.failureCount = 0;
    this.successCount++;
    this.totalRequests++;

    if (this.state === 'HALF_OPEN') {
      this.state = 'CLOSED';
    }
  }

  onFailure() {
    this.failureCount++;
    this.totalRequests++;
    this.lastFailureTime = Date.now();

    if (this.failureCount >= this.threshold) {
      this.state = 'OPEN';
      this.nextAttempt = Date.now() + this.resetTimeout;
    }
  }

  getStats() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      totalRequests: this.totalRequests,
      successRate: this.totalRequests > 0 ? this.successCount / this.totalRequests : 0,
      nextAttempt: this.nextAttempt,
      lastFailureTime: this.lastFailureTime,
    };
  }

  reset() {
    this.failureCount = 0;
    this.state = 'CLOSED';
    this.nextAttempt = Date.now();
  }
}

export class ErrorHandler {
  constructor(logger) {
    this.logger = logger;
    this.circuitBreakers = new Map();
    this.errorCounts = new Map();
    this.recoveryStrategies = new Map();

    // Default recovery strategies
    this.setupDefaultRecoveryStrategies();
  }

  setupDefaultRecoveryStrategies() {
    // Database connection issues
    this.recoveryStrategies.set('DATABASE_ERROR', async (error, context) => {
      this.logger.warn('Database error detected, implementing recovery', { error: error.message });
      // Could implement connection pooling, retry logic, fallback to cache, etc.
      return { strategy: 'retry', delay: 1000, maxRetries: 3 };
    });

    // Memory pressure
    this.recoveryStrategies.set('MEMORY_ERROR', async (error, context) => {
      this.logger.warn('Memory pressure detected, implementing cleanup', { error: error.message });
      global.gc && global.gc(); // Force garbage collection if available
      return { strategy: 'throttle', delay: 5000 };
    });

    // External service failures
    this.recoveryStrategies.set('SERVICE_ERROR', async (error, context) => {
      this.logger.warn('External service error, using circuit breaker', { error: error.message });
      return { strategy: 'circuit_breaker', timeout: 30000 };
    });

    // Rate limiting
    this.recoveryStrategies.set('RATE_LIMIT_ERROR', async (error, context) => {
      this.logger.warn('Rate limit exceeded, implementing backoff', { error: error.message });
      return { strategy: 'backoff', delay: 2000, multiplier: 2 };
    });
  }

  getCircuitBreaker(service, options = {}) {
    if (!this.circuitBreakers.has(service)) {
      this.circuitBreakers.set(service, new CircuitBreaker(options));
    }
    return this.circuitBreakers.get(service);
  }

  async handleError(error, req = null, context = {}) {
    const errorInfo = this.analyzeError(error);
    const errorId = this.generateErrorId();

    // Log error with full context
    this.logger.error('Error occurred', {
      errorId,
      error: error.message,
      stack: error.stack,
      type: errorInfo.type,
      severity: errorInfo.severity,
      url: req?.url,
      method: req?.method,
      ip: req?.ip,
      userAgent: req?.get('User-Agent'),
      context,
    });

    // Record metrics
    HealthMetrics.recordError(errorInfo.type, errorInfo.severity);

    // Update error counts
    const errorKey = `${errorInfo.type}:${error.message}`;
    this.errorCounts.set(errorKey, (this.errorCounts.get(errorKey) || 0) + 1);

    // Implement recovery strategy if available
    const recovery = await this.implementRecoveryStrategy(error, errorInfo, context);

    return {
      errorId,
      type: errorInfo.type,
      severity: errorInfo.severity,
      message: this.getSafeErrorMessage(error, errorInfo.severity),
      recovery,
      timestamp: new Date().toISOString(),
    };
  }

  analyzeError(error) {
    const message = error.message.toLowerCase();
    const stack = error.stack || '';

    // Database errors
    if (message.includes('database') || message.includes('connection') || message.includes('sql')) {
      return { type: 'DATABASE_ERROR', severity: 'HIGH' };
    }

    // Memory errors
    if (message.includes('memory') || message.includes('heap') || error.name === 'RangeError') {
      return { type: 'MEMORY_ERROR', severity: 'CRITICAL' };
    }

    // Network errors
    if (message.includes('network') || message.includes('timeout') || message.includes('econnrefused')) {
      return { type: 'NETWORK_ERROR', severity: 'HIGH' };
    }

    // Validation errors
    if (message.includes('validation') || message.includes('invalid') || error.name === 'ValidationError') {
      return { type: 'VALIDATION_ERROR', severity: 'MEDIUM' };
    }

    // Authentication errors
    if (message.includes('unauthorized') || message.includes('forbidden') || message.includes('auth')) {
      return { type: 'AUTH_ERROR', severity: 'MEDIUM' };
    }

    // Rate limiting errors
    if (message.includes('rate limit') || message.includes('too many requests')) {
      return { type: 'RATE_LIMIT_ERROR', severity: 'LOW' };
    }

    // Service errors
    if (message.includes('service') || message.includes('external') || message.includes('api')) {
      return { type: 'SERVICE_ERROR', severity: 'HIGH' };
    }

    // Business logic errors
    if (error.name === 'BusinessError' || message.includes('business')) {
      return { type: 'BUSINESS_ERROR', severity: 'MEDIUM' };
    }

    // System errors
    if (message.includes('system') || message.includes('internal')) {
      return { type: 'SYSTEM_ERROR', severity: 'HIGH' };
    }

    // Default classification
    return { type: 'UNKNOWN_ERROR', severity: 'MEDIUM' };
  }

  async implementRecoveryStrategy(error, errorInfo, context) {
    const strategy = this.recoveryStrategies.get(errorInfo.type);

    if (!strategy) {
      return null;
    }

    try {
      const recovery = await strategy(error, context);

      this.logger.info('Recovery strategy implemented', {
        errorType: errorInfo.type,
        strategy: recovery.strategy,
      });

      return recovery;
    } catch (recoveryError) {
      this.logger.error('Recovery strategy failed', {
        errorType: errorInfo.type,
        recoveryError: recoveryError.message,
      });
      return null;
    }
  }

  getSafeErrorMessage(error, severity) {
    // Don't expose sensitive information in production
    const isProduction = process.env.NODE_ENV === 'production';

    if (isProduction && severity !== 'LOW') {
      return 'An internal error occurred. Please try again later.';
    }

    return error.message;
  }

  generateErrorId() {
    return `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Express middleware
  middleware() {
    return async (error, req, res, next) => {
      const errorResponse = await this.handleError(error, req, {
        requestId: req.id,
        userId: req.user?.id,
      });

      // Set appropriate HTTP status code
      const statusCode = this.getHttpStatusCode(errorResponse.type, errorResponse.severity);

      res.status(statusCode).json({
        success: false,
        error: {
          id: errorResponse.errorId,
          type: errorResponse.type,
          message: errorResponse.message,
          timestamp: errorResponse.timestamp,
          ...(errorResponse.recovery && { recovery: errorResponse.recovery }),
        },
      });
    };
  }

  getHttpStatusCode(type, severity) {
    switch (type) {
      case 'VALIDATION_ERROR':
        return 400;
      case 'AUTH_ERROR':
        return 401;
      case 'RATE_LIMIT_ERROR':
        return 429;
      case 'BUSINESS_ERROR':
        return 422;
      case 'NETWORK_ERROR':
      case 'SERVICE_ERROR':
        return 503;
      case 'MEMORY_ERROR':
      case 'SYSTEM_ERROR':
        return 500;
      default:
        return severity === 'CRITICAL' ? 500 : 400;
    }
  }

  // Async error wrapper for route handlers
  asyncHandler(fn) {
    return (req, res, next) => {
      Promise.resolve(fn(req, res, next)).catch(next);
    };
  }

  // Process-level error handlers
  setupProcessErrorHandlers() {
    process.on('uncaughtException', error => {
      this.logger.error('Uncaught Exception', {
        error: error.message,
        stack: error.stack,
      });

      HealthMetrics.recordError('UNCAUGHT_EXCEPTION', 'CRITICAL');

      // Graceful shutdown
      setTimeout(() => {
        process.exit(1);
      }, 5000);
    });

    process.on('unhandledRejection', (reason, promise) => {
      this.logger.error('Unhandled Rejection', {
        reason: reason?.message || reason,
        stack: reason?.stack,
        promise: promise.toString(),
      });

      HealthMetrics.recordError('UNHANDLED_REJECTION', 'HIGH');
    });

    process.on('warning', warning => {
      this.logger.warn('Process Warning', {
        name: warning.name,
        message: warning.message,
        stack: warning.stack,
      });
    });
  }

  // Get error statistics
  getErrorStats() {
    const stats = {
      circuitBreakers: {},
      errorCounts: Object.fromEntries(this.errorCounts),
      totalErrors: Array.from(this.errorCounts.values()).reduce((sum, count) => sum + count, 0),
    };

    for (const [service, breaker] of this.circuitBreakers) {
      stats.circuitBreakers[service] = breaker.getStats();
    }

    return stats;
  }

  // Reset error statistics
  resetStats() {
    this.errorCounts.clear();
    this.circuitBreakers.forEach(breaker => breaker.reset());
  }

  // Cleanup
  async cleanup() {
    this.circuitBreakers.clear();
    this.errorCounts.clear();
    this.recoveryStrategies.clear();
  }
}

// Custom error classes
export class BusinessError extends Error {
  constructor(message, code = 'BUSINESS_ERROR') {
    super(message);
    this.name = 'BusinessError';
    this.code = code;
  }
}

export class ValidationError extends Error {
  constructor(message, field = null) {
    super(message);
    this.name = 'ValidationError';
    this.field = field;
  }
}

export class ServiceUnavailableError extends Error {
  constructor(service, reason = 'Service unavailable') {
    super(`${service}: ${reason}`);
    this.name = 'ServiceUnavailableError';
    this.service = service;
  }
}

// Helper function to create error handler instance
export function createErrorHandler(logger) {
  const errorHandler = new ErrorHandler(logger);
  errorHandler.setupProcessErrorHandlers();
  return errorHandler;
}
