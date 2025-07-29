# Advanced Security Configuration

## Overview
This document provides comprehensive security configurations for the Synthetic Data Guardian project, implementing defense-in-depth strategies appropriate for enterprise synthetic data handling.

## Container Security

### 1. Multi-stage Dockerfile Security Enhancement

```dockerfile
# Production Dockerfile with security hardening
FROM node:18-alpine AS base
RUN apk add --no-cache dumb-init
RUN addgroup -g 1001 -S nodejs
RUN adduser -S guardian -u 1001

FROM base AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

FROM base AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM base AS runtime
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=build /app/package.json ./

USER 1001
EXPOSE 8080

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/index.js"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/api/v1/health || exit 1

LABEL org.opencontainers.image.title="Synthetic Data Guardian"
LABEL org.opencontainers.image.description="Enterprise synthetic data pipeline"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/your-org/synthetic-data-guardian"
```

### 2. Container Security Scanning Configuration

```yaml
# .github/workflows/container-security.yml
name: Container Security Scan

on:
  push:
    paths: 
      - 'Dockerfile*'
      - '.dockerignore'
  schedule:
    - cron: '0 6 * * *'

jobs:
  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build image
        run: docker build -t synthetic-guardian:latest .
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'synthetic-guardian:latest'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
      
      - name: Run Grype vulnerability scanner
        uses: anchore/grype-action@v3
        with:
          image: synthetic-guardian:latest
          fail-build: true
          severity-cutoff: high
```

## Runtime Security

### 3. Application Security Headers

```typescript
// src/middleware/security.ts
import helmet from 'helmet';
import { Request, Response, NextFunction } from 'express';

export const securityMiddleware = helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'"],
      fontSrc: ["'self'"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"],
    },
  },
  crossOriginEmbedderPolicy: true,
  crossOriginOpenerPolicy: { policy: "same-origin" },
  crossOriginResourcePolicy: { policy: "cross-origin" },
  dnsPrefetchControl: true,
  frameguard: { action: 'deny' },
  hidePoweredBy: true,
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true
  },
  ieNoOpen: true,
  noSniff: true,
  originAgentCluster: true,
  permittedCrossDomainPolicies: false,
  referrerPolicy: { policy: "no-referrer" },
  xssFilter: true,
});

export const rateLimitMiddleware = (req: Request, res: Response, next: NextFunction) => {
  // Implement rate limiting logic
  next();
};
```

### 4. Secrets Management Configuration

```yaml
# kubernetes/secrets-config.yaml
apiVersion: v1
kind: Secret
metadata:
  name: guardian-secrets
  namespace: synthetic-guardian
type: Opaque
data:
  # Base64 encoded secrets - managed via CI/CD
  database-url: ""
  jwt-secret: ""
  watermark-key: ""
  encryption-key: ""

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: guardian-config
  namespace: synthetic-guardian
data:
  NODE_ENV: "production"
  LOG_LEVEL: "info"
  RATE_LIMIT_WINDOW: "900000"
  RATE_LIMIT_MAX: "100"
```

## Supply Chain Security

### 5. Dependency Verification

```json
// .npmrc
audit-level=moderate
fund=false
save-exact=true
package-lock-only=true
```

```yaml
# .github/workflows/supply-chain.yml
name: Supply Chain Security

on:
  schedule:
    - cron: '0 8 * * MON'
  workflow_dispatch:

jobs:
  dependency-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18.x'
      
      - name: Audit npm dependencies
        run: |
          npm audit --audit-level high
          npm audit --json > npm-audit.json
      
      - name: Check for known vulnerabilities
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
      
      - name: Verify package signatures
        run: |
          npm audit signatures
      
      - name: Generate dependency tree
        run: |
          npm list --all --json > dependency-tree.json
      
      - name: Upload audit results
        uses: actions/upload-artifact@v3
        with:
          name: security-audit
          path: |
            npm-audit.json
            dependency-tree.json
```

### 6. SBOM (Software Bill of Materials) Generation

```yaml
# .github/workflows/sbom.yml
name: SBOM Generation

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  generate-sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate SBOM with Syft
        uses: anchore/sbom-action@v0
        with:
          format: spdx-json
          output-file: sbom.spdx.json
      
      - name: Generate CycloneDX SBOM
        run: |
          npm install -g @cyclonedx/cyclonedx-npm
          npx @cyclonedx/cyclonedx-npm --output-format JSON --output-file sbom.cyclonedx.json
      
      - name: Sign SBOM with Cosign
        uses: sigstore/cosign-installer@v3
        with:
          cosign-release: 'v2.1.1'
      
      - name: Upload SBOMs
        uses: actions/upload-artifact@v3
        with:
          name: sbom-files
          path: |
            sbom.spdx.json
            sbom.cyclonedx.json
```

## Data Protection

### 7. Encryption Configuration

```typescript
// src/utils/encryption.ts
import crypto from 'crypto';

export class DataEncryption {
  private readonly algorithm = 'aes-256-gcm';
  private readonly keyLength = 32;

  constructor(private readonly key: string) {
    if (!key || key.length !== this.keyLength * 2) {
      throw new Error('Invalid encryption key length');
    }
  }

  encrypt(data: string): { encrypted: string; iv: string; tag: string } {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher(this.algorithm, Buffer.from(this.key, 'hex'));
    cipher.setAAD(Buffer.from('synthetic-data-guardian'));
    
    let encrypted = cipher.update(data, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const tag = cipher.getAuthTag();
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      tag: tag.toString('hex')
    };
  }

  decrypt(encrypted: string, iv: string, tag: string): string {
    const decipher = crypto.createDecipher(this.algorithm, Buffer.from(this.key, 'hex'));
    decipher.setAAD(Buffer.from('synthetic-data-guardian'));
    decipher.setAuthTag(Buffer.from(tag, 'hex'));
    
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }
}
```

### 8. Audit Logging Configuration

```typescript
// src/middleware/audit.ts
import winston from 'winston';

const auditLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'synthetic-data-guardian' },
  transports: [
    new winston.transports.File({ 
      filename: 'logs/audit.log',
      level: 'info'
    })
  ]
});

export const auditMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const startTime = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - startTime;
    
    auditLogger.info('API Request', {
      method: req.method,
      url: req.originalUrl,
      userAgent: req.get('User-Agent'),
      ip: req.ip,
      statusCode: res.statusCode,
      duration,
      userId: req.user?.id,
      requestId: req.headers['x-request-id']
    });
  });
  
  next();
};
```

## Compliance Frameworks

### 9. SLSA (Supply-chain Levels for Software Artifacts) Configuration

```yaml
# .github/workflows/slsa.yml
name: SLSA Provenance

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      hashes: ${{ steps.hash.outputs.hashes }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Build artifacts
        run: |
          npm ci
          npm run build
          tar -czf synthetic-guardian.tar.gz dist/
      
      - name: Generate hashes
        shell: bash
        id: hash
        run: |
          echo "hashes=$(sha256sum synthetic-guardian.tar.gz | base64 -w0)" >> "$GITHUB_OUTPUT"
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: build-artifacts
          path: synthetic-guardian.tar.gz

  provenance:
    needs: [build]
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.7.0
    with:
      base64-subjects: "${{ needs.build.outputs.hashes }}"
      upload-assets: true
```

## Incident Response

### 10. Security Incident Response Plan

```yaml
# .github/ISSUE_TEMPLATE/security-incident.yml
name: Security Incident Report
description: Report a security incident or vulnerability
title: "[SECURITY] "
labels: ["security", "incident", "priority-high"]
assignees:
  - security-team

body:
  - type: markdown
    attributes:
      value: |
        **⚠️ For critical security issues, contact security@your-org.com immediately**
        
  - type: dropdown
    id: severity
    attributes:
      label: Severity Level
      description: How severe is this security incident?
      options:
        - Critical (Active exploit, data breach)
        - High (Potential for significant impact)
        - Medium (Limited impact, needs investigation)
        - Low (Minor security concern)
    validations:
      required: true
      
  - type: textarea
    id: description
    attributes:
      label: Incident Description
      description: Detailed description of the security incident
      placeholder: Describe what happened, when it was discovered, and potential impact
    validations:
      required: true
      
  - type: textarea
    id: affected-systems
    attributes:
      label: Affected Systems
      description: Which systems, services, or data are affected?
    validations:
      required: true
      
  - type: textarea
    id: immediate-actions
    attributes:
      label: Immediate Actions Taken
      description: What steps have already been taken to contain the incident?
```

## Security Monitoring

### 11. Runtime Security Monitoring

```typescript
// src/monitoring/security-monitor.ts
import { EventEmitter } from 'events';

export class SecurityMonitor extends EventEmitter {
  private readonly suspiciousPatterns = [
    /(\.\./.*){3,}/, // Path traversal attempts
    /<script.*>/i,  // XSS attempts
    /union.*select/i, // SQL injection attempts
    /exec\(|eval\(/i, // Code injection attempts
  ];

  monitorRequest(req: any): void {
    const suspicious = this.detectSuspiciousActivity(req);
    
    if (suspicious.length > 0) {
      this.emit('suspicious-activity', {
        ip: req.ip,
        userAgent: req.get('User-Agent'),
        patterns: suspicious,
        timestamp: new Date(),
        url: req.originalUrl,
        method: req.method
      });
    }
  }

  private detectSuspiciousActivity(req: any): string[] {
    const suspicious: string[] = [];
    const checkString = JSON.stringify(req.body) + req.url + JSON.stringify(req.query);
    
    for (const pattern of this.suspiciousPatterns) {
      if (pattern.test(checkString)) {
        suspicious.push(pattern.source);
      }
    }
    
    return suspicious;
  }
}
```

This comprehensive security configuration provides enterprise-grade protection appropriate for a synthetic data handling platform, implementing defense-in-depth strategies across the entire application lifecycle.