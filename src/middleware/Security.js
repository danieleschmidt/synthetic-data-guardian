/**
 * Comprehensive Security Middleware
 */

import crypto from 'crypto';
import { HealthMetrics } from './metrics.js';

export class SecurityMiddleware {
  constructor(logger, options = {}) {
    this.logger = logger;
    this.options = {
      // Security headers configuration
      contentSecurityPolicy: options.contentSecurityPolicy || {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          imgSrc: ["'self'", 'data:', 'https:'],
          connectSrc: ["'self'"],
          fontSrc: ["'self'"],
          objectSrc: ["'none'"],
          mediaSrc: ["'self'"],
          frameSrc: ["'none'"],
        },
      },

      // API security
      apiKeyHeader: options.apiKeyHeader || 'x-api-key',
      enableApiKeyAuth: options.enableApiKeyAuth || false,
      apiKeys: options.apiKeys || new Set(),

      // Request filtering
      maxRequestSize: options.maxRequestSize || '10mb',
      allowedMethods: options.allowedMethods || ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      blockedUserAgents: options.blockedUserAgents || [],
      allowedOrigins: options.allowedOrigins || [],

      // Advanced security features
      enableHoneypot: options.enableHoneypot || false,
      enableRequestFingerprinting: options.enableRequestFingerprinting || true,
      suspiciousActivityThreshold: options.suspiciousActivityThreshold || 10,

      // File upload security
      allowedFileTypes: options.allowedFileTypes || ['json', 'csv', 'txt'],
      maxFileSize: options.maxFileSize || 5 * 1024 * 1024, // 5MB

      // IP filtering
      blockedIPs: options.blockedIPs || new Set(),
      trustedProxies: options.trustedProxies || new Set(),

      // Encryption
      encryptionKey: options.encryptionKey || crypto.randomBytes(32),

      // Session security
      sessionTimeout: options.sessionTimeout || 30 * 60 * 1000, // 30 minutes
    };

    this.suspiciousActivities = new Map();
    this.requestFingerprints = new Map();
    this.activeSessions = new Map();

    // Start cleanup intervals
    this.startCleanupJobs();
  }

  startCleanupJobs() {
    // Clean up old suspicious activity records
    this.suspiciousCleanup = setInterval(
      () => {
        const cutoff = Date.now() - 24 * 60 * 60 * 1000; // 24 hours
        for (const [key, activities] of this.suspiciousActivities) {
          const filtered = activities.filter(activity => activity.timestamp > cutoff);
          if (filtered.length === 0) {
            this.suspiciousActivities.delete(key);
          } else {
            this.suspiciousActivities.set(key, filtered);
          }
        }
      },
      60 * 60 * 1000,
    ); // Every hour

    // Clean up old request fingerprints
    this.fingerprintCleanup = setInterval(
      () => {
        const cutoff = Date.now() - 60 * 60 * 1000; // 1 hour
        for (const [key, data] of this.requestFingerprints) {
          if (data.lastSeen < cutoff) {
            this.requestFingerprints.delete(key);
          }
        }
      },
      15 * 60 * 1000,
    ); // Every 15 minutes

    // Clean up expired sessions
    this.sessionCleanup = setInterval(
      () => {
        const now = Date.now();
        for (const [sessionId, session] of this.activeSessions) {
          if (now - session.lastActivity > this.options.sessionTimeout) {
            this.activeSessions.delete(sessionId);
          }
        }
      },
      5 * 60 * 1000,
    ); // Every 5 minutes
  }

  // Main security middleware
  middleware() {
    return async (req, res, next) => {
      try {
        // Add security headers
        this.addSecurityHeaders(res);

        // Block malicious IPs
        if (this.isBlockedIP(req)) {
          return this.blockRequest(req, res, 'BLOCKED_IP');
        }

        // Validate HTTP method
        if (!this.options.allowedMethods.includes(req.method)) {
          return this.blockRequest(req, res, 'INVALID_METHOD');
        }

        // Check user agent
        if (this.isBlockedUserAgent(req)) {
          return this.blockRequest(req, res, 'BLOCKED_USER_AGENT');
        }

        // Request size validation
        if (
          req.get('content-length') &&
          parseInt(req.get('content-length')) > this.parseSize(this.options.maxRequestSize)
        ) {
          return this.blockRequest(req, res, 'REQUEST_TOO_LARGE');
        }

        // API key authentication
        if (this.options.enableApiKeyAuth && !this.validateApiKey(req)) {
          return this.blockRequest(req, res, 'INVALID_API_KEY', 401);
        }

        // Request fingerprinting and anomaly detection
        if (this.options.enableRequestFingerprinting) {
          const fingerprint = this.generateRequestFingerprint(req);
          const isAnomalous = this.detectAnomalousRequest(req, fingerprint);

          if (isAnomalous) {
            this.recordSuspiciousActivity(req, 'ANOMALOUS_REQUEST');
            if (this.isSuspiciousClient(req)) {
              return this.blockRequest(req, res, 'SUSPICIOUS_ACTIVITY', 429);
            }
          }
        }

        // Honeypot detection
        if (this.options.enableHoneypot && this.isHoneypotRequest(req)) {
          this.recordSuspiciousActivity(req, 'HONEYPOT_TRIGGERED');
          return this.blockRequest(req, res, 'HONEYPOT', 404);
        }

        // Add request security metadata
        req.security = {
          ip: this.getRealIP(req),
          userAgent: req.get('User-Agent'),
          fingerprint: this.options.enableRequestFingerprinting ? this.generateRequestFingerprint(req) : null,
          timestamp: Date.now(),
          sessionId: this.getOrCreateSession(req),
        };

        // Continue to next middleware
        next();
      } catch (error) {
        this.logger.error('Security middleware error', {
          error: error.message,
          url: req.url,
          ip: this.getRealIP(req),
        });
        next();
      }
    };
  }

  addSecurityHeaders(res) {
    // Prevent MIME type sniffing
    res.setHeader('X-Content-Type-Options', 'nosniff');

    // Prevent clickjacking
    res.setHeader('X-Frame-Options', 'DENY');

    // XSS Protection
    res.setHeader('X-XSS-Protection', '1; mode=block');

    // Strict Transport Security
    res.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');

    // Referrer Policy
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');

    // Permissions Policy
    res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');

    // Content Security Policy
    const csp = this.buildCSPHeader();
    res.setHeader('Content-Security-Policy', csp);

    // Remove server information
    res.removeHeader('X-Powered-By');

    // Custom security headers
    res.setHeader('X-API-Version', '1.0');
    res.setHeader('X-Request-ID', crypto.randomUUID());
  }

  buildCSPHeader() {
    const directives = [];
    for (const [directive, sources] of Object.entries(this.options.contentSecurityPolicy.directives)) {
      const kebabCase = directive.replace(/([A-Z])/g, '-$1').toLowerCase();
      directives.push(`${kebabCase} ${sources.join(' ')}`);
    }
    return directives.join('; ');
  }

  isBlockedIP(req) {
    const ip = this.getRealIP(req);
    return this.options.blockedIPs.has(ip);
  }

  isBlockedUserAgent(req) {
    const userAgent = req.get('User-Agent') || '';
    return this.options.blockedUserAgents.some(blocked => userAgent.toLowerCase().includes(blocked.toLowerCase()));
  }

  validateApiKey(req) {
    const apiKey = req.get(this.options.apiKeyHeader);
    return apiKey && this.options.apiKeys.has(apiKey);
  }

  generateRequestFingerprint(req) {
    const components = [
      this.getRealIP(req),
      req.get('User-Agent') || '',
      req.get('Accept') || '',
      req.get('Accept-Language') || '',
      req.get('Accept-Encoding') || '',
      req.method,
      req.url,
    ];

    return crypto.createHash('sha256').update(components.join('|')).digest('hex').substring(0, 16);
  }

  detectAnomalousRequest(req, fingerprint) {
    const now = Date.now();
    const existing = this.requestFingerprints.get(fingerprint);

    if (!existing) {
      this.requestFingerprints.set(fingerprint, {
        firstSeen: now,
        lastSeen: now,
        count: 1,
        urls: [req.url],
        methods: [req.method],
      });
      return false;
    }

    existing.lastSeen = now;
    existing.count++;

    if (!existing.urls.includes(req.url)) {
      existing.urls.push(req.url);
    }

    if (!existing.methods.includes(req.method)) {
      existing.methods.push(req.method);
    }

    // Detect anomalies
    const timeDiff = now - existing.firstSeen;
    const requestRate = existing.count / (timeDiff / 1000); // requests per second

    // High request rate from same fingerprint
    if (requestRate > 10) {
      return true;
    }

    // Too many different URLs from same fingerprint
    if (existing.urls.length > 20) {
      return true;
    }

    // Unusual method diversity
    if (existing.methods.length > 4) {
      return true;
    }

    return false;
  }

  isHoneypotRequest(req) {
    // Common honeypot paths that attackers typically probe
    const honeypotPaths = [
      '/admin',
      '/wp-admin',
      '/phpmyadmin',
      '/config',
      '/.env',
      '/.git',
      '/backup',
      '/test',
      '/robots.txt',
      '/sitemap.xml',
      '/.well-known',
    ];

    return honeypotPaths.some(path => req.url.toLowerCase().startsWith(path));
  }

  recordSuspiciousActivity(req, type) {
    const ip = this.getRealIP(req);
    const key = `${ip}:${type}`;
    const activity = {
      type,
      timestamp: Date.now(),
      url: req.url,
      method: req.method,
      userAgent: req.get('User-Agent'),
    };

    if (!this.suspiciousActivities.has(key)) {
      this.suspiciousActivities.set(key, []);
    }

    this.suspiciousActivities.get(key).push(activity);

    this.logger.warn('Suspicious activity detected', {
      ip,
      type,
      url: req.url,
      userAgent: req.get('User-Agent'),
    });

    HealthMetrics.recordSecurityEvent(type, ip);
  }

  isSuspiciousClient(req) {
    const ip = this.getRealIP(req);
    let totalSuspiciousActivities = 0;

    // Count all suspicious activities from this IP
    for (const [key, activities] of this.suspiciousActivities) {
      if (key.startsWith(`${ip}:`)) {
        totalSuspiciousActivities += activities.length;
      }
    }

    return totalSuspiciousActivities >= this.options.suspiciousActivityThreshold;
  }

  getRealIP(req) {
    // Handle proxies and load balancers
    const forwarded = req.get('X-Forwarded-For');
    const realIP = req.get('X-Real-IP');
    const remoteIP = req.connection.remoteAddress || req.socket.remoteAddress;

    if (forwarded) {
      const ips = forwarded.split(',').map(ip => ip.trim());
      // Return the first non-trusted proxy IP
      for (const ip of ips) {
        if (!this.options.trustedProxies.has(ip)) {
          return ip;
        }
      }
    }

    return realIP || remoteIP || 'unknown';
  }

  getOrCreateSession(req) {
    // Simple session management based on IP + User Agent hash
    const sessionKey = crypto
      .createHash('sha256')
      .update(`${this.getRealIP(req)}:${req.get('User-Agent') || ''}`)
      .digest('hex')
      .substring(0, 16);

    const now = Date.now();

    if (!this.activeSessions.has(sessionKey)) {
      this.activeSessions.set(sessionKey, {
        created: now,
        lastActivity: now,
        requestCount: 0,
      });
    }

    const session = this.activeSessions.get(sessionKey);
    session.lastActivity = now;
    session.requestCount++;

    return sessionKey;
  }

  blockRequest(req, res, reason, statusCode = 403) {
    const ip = this.getRealIP(req);

    this.logger.warn('Request blocked', {
      ip,
      reason,
      url: req.url,
      method: req.method,
      userAgent: req.get('User-Agent'),
    });

    HealthMetrics.recordSecurityBlock(reason, ip);

    res.status(statusCode).json({
      success: false,
      error: {
        type: 'SECURITY_VIOLATION',
        message: 'Request blocked for security reasons',
        code: reason,
        timestamp: new Date().toISOString(),
      },
    });
  }

  parseSize(size) {
    if (typeof size === 'number') return size;
    const units = { b: 1, kb: 1024, mb: 1024 ** 2, gb: 1024 ** 3 };
    const match = size.toLowerCase().match(/^(\d+)(b|kb|mb|gb)?$/);
    return match ? parseInt(match[1]) * (units[match[2]] || 1) : 0;
  }

  // File upload security
  validateFileUpload(file) {
    if (!file) return { valid: false, reason: 'No file provided' };

    // Check file size
    if (file.size > this.options.maxFileSize) {
      return { valid: false, reason: 'File too large' };
    }

    // Check file type
    const extension = file.originalname.split('.').pop().toLowerCase();
    if (!this.options.allowedFileTypes.includes(extension)) {
      return { valid: false, reason: 'File type not allowed' };
    }

    // Check for malicious content in filename
    if (file.originalname.includes('..') || file.originalname.includes('/') || file.originalname.includes('\\')) {
      return { valid: false, reason: 'Invalid filename' };
    }

    return { valid: true };
  }

  // Encrypt sensitive data
  encrypt(data) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher('aes-256-cbc', this.options.encryptionKey);
    cipher.setAutoPadding(true);

    let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
    encrypted += cipher.final('hex');

    return {
      encrypted,
      iv: iv.toString('hex'),
    };
  }

  // Decrypt sensitive data
  decrypt(encryptedData, iv) {
    const decipher = crypto.createDecipher('aes-256-cbc', this.options.encryptionKey);
    decipher.setAutoPadding(true);

    let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
    decrypted += decipher.final('utf8');

    return JSON.parse(decrypted);
  }

  // Get security statistics
  getSecurityStats() {
    return {
      suspiciousActivities: this.suspiciousActivities.size,
      activeFingerprints: this.requestFingerprints.size,
      activeSessions: this.activeSessions.size,
      blockedIPs: this.options.blockedIPs.size,
      trustedProxies: this.options.trustedProxies.size,
    };
  }

  // Add IP to blocklist
  blockIP(ip) {
    this.options.blockedIPs.add(ip);
    this.logger.info('IP added to blocklist', { ip });
  }

  // Remove IP from blocklist
  unblockIP(ip) {
    this.options.blockedIPs.delete(ip);
    this.logger.info('IP removed from blocklist', { ip });
  }

  // Cleanup
  cleanup() {
    if (this.suspiciousCleanup) clearInterval(this.suspiciousCleanup);
    if (this.fingerprintCleanup) clearInterval(this.fingerprintCleanup);
    if (this.sessionCleanup) clearInterval(this.sessionCleanup);

    this.suspiciousActivities.clear();
    this.requestFingerprints.clear();
    this.activeSessions.clear();
  }
}

export function createSecurityMiddleware(logger, options = {}) {
  return new SecurityMiddleware(logger, options);
}
