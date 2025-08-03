/**
 * Validation Middleware - Request validation and sanitization
 */

import crypto from 'crypto';

export function validationMiddleware(req, res, next) {
  // Add request ID for tracing
  req.id = crypto.randomUUID();
  
  // Add request timestamp
  req.timestamp = new Date().toISOString();

  // Validate content type for POST/PUT requests
  if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
    const contentType = req.get('Content-Type');
    
    if (!contentType || !contentType.includes('application/json')) {
      return res.status(400).json({
        error: 'Invalid Content Type',
        message: 'Content-Type must be application/json',
        timestamp: new Date().toISOString()
      });
    }
  }

  // Validate request size
  const contentLength = req.get('Content-Length');
  if (contentLength && parseInt(contentLength) > 50 * 1024 * 1024) { // 50MB limit
    return res.status(413).json({
      error: 'Payload Too Large',
      message: 'Request body exceeds maximum size limit',
      timestamp: new Date().toISOString()
    });
  }

  // Security headers validation
  const origin = req.get('Origin');
  const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'];
  
  if (origin && !allowedOrigins.includes(origin) && process.env.NODE_ENV === 'production') {
    return res.status(403).json({
      error: 'Forbidden',
      message: 'Origin not allowed',
      timestamp: new Date().toISOString()
    });
  }

  // Rate limiting check (basic implementation)
  const clientIp = req.ip || req.connection.remoteAddress;
  const rateLimitKey = `rate_limit:${clientIp}`;
  
  // In a real implementation, this would use Redis
  // For now, we'll use a simple in-memory store
  if (!global.rateLimitStore) {
    global.rateLimitStore = new Map();
  }

  const now = Date.now();
  const windowMs = 60 * 1000; // 1 minute window
  const maxRequests = 100; // Max 100 requests per minute

  const clientRequests = global.rateLimitStore.get(rateLimitKey) || [];
  const recentRequests = clientRequests.filter(timestamp => now - timestamp < windowMs);

  if (recentRequests.length >= maxRequests) {
    return res.status(429).json({
      error: 'Too Many Requests',
      message: 'Rate limit exceeded. Please try again later.',
      timestamp: new Date().toISOString(),
      retryAfter: Math.ceil(windowMs / 1000)
    });
  }

  // Update rate limit store
  recentRequests.push(now);
  global.rateLimitStore.set(rateLimitKey, recentRequests);

  // Clean up old entries periodically
  if (Math.random() < 0.01) { // 1% chance to cleanup
    for (const [key, timestamps] of global.rateLimitStore.entries()) {
      const recentTimestamps = timestamps.filter(timestamp => now - timestamp < windowMs);
      if (recentTimestamps.length === 0) {
        global.rateLimitStore.delete(key);
      } else {
        global.rateLimitStore.set(key, recentTimestamps);
      }
    }
  }

  next();
}