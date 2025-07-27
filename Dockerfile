# =============================================================================
# Synthetic Data Guardian - Production Dockerfile
# =============================================================================

# Multi-stage build for optimal image size and security
FROM node:18-alpine AS base

# Install security updates and required packages
RUN apk update && apk upgrade && \
    apk add --no-cache \
    python3 \
    py3-pip \
    make \
    g++ \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/cache/apk/*

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S synthetic -u 1001

# Set working directory
WORKDIR /app

# Copy package files for dependency installation
COPY package*.json ./
COPY requirements*.txt ./

# =============================================================================
# Dependencies stage
# =============================================================================
FROM base AS deps

# Install Node.js dependencies
RUN npm ci --only=production --silent && \
    npm cache clean --force

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# =============================================================================
# Development stage
# =============================================================================
FROM base AS development

# Install all dependencies (including dev dependencies)
RUN npm ci --silent

# Copy Python requirements and install
COPY requirements*.txt ./
RUN pip3 install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Set development environment
ENV NODE_ENV=development
ENV DEBUG=true

# Expose development ports
EXPOSE 8080 9229

# Set user
USER synthetic

# Development command
CMD ["npm", "run", "dev"]

# =============================================================================
# Build stage
# =============================================================================
FROM base AS builder

# Copy package files and install all dependencies
COPY package*.json ./
RUN npm ci --silent

# Copy source code
COPY . .

# Build the application
RUN npm run build && \
    npm prune --production

# =============================================================================
# Production stage
# =============================================================================
FROM node:18-alpine AS production

# Install security updates
RUN apk update && apk upgrade && \
    apk add --no-cache \
    python3 \
    py3-pip \
    ca-certificates \
    tini \
    && rm -rf /var/cache/apk/*

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S synthetic -u 1001

# Set working directory
WORKDIR /app

# Copy Python requirements and install
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Copy built application from builder stage
COPY --from=builder --chown=synthetic:nodejs /app/dist ./dist
COPY --from=builder --chown=synthetic:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=synthetic:nodejs /app/package.json ./package.json

# Copy configuration files
COPY --chown=synthetic:nodejs config/ ./config/
COPY --chown=synthetic:nodejs scripts/docker-entrypoint.sh ./scripts/

# Make entrypoint script executable
RUN chmod +x ./scripts/docker-entrypoint.sh

# Set production environment
ENV NODE_ENV=production
ENV PORT=8080

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

# Use tini for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]

# Set user
USER synthetic

# Expose port
EXPOSE 8080

# Production command
CMD ["./scripts/docker-entrypoint.sh"]

# =============================================================================
# Metadata
# =============================================================================
LABEL maintainer="Terragon Labs <info@terragonlabs.com>"
LABEL version="1.0.0"
LABEL description="Synthetic Data Guardian - Enterprise-grade synthetic data pipeline"
LABEL org.opencontainers.image.title="Synthetic Data Guardian"
LABEL org.opencontainers.image.description="Enterprise-grade synthetic data pipeline with built-in validation, watermarking, and auditable lineage tracking"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/your-org/synthetic-data-guardian"
LABEL org.opencontainers.image.documentation="https://docs.your-org.com/synthetic-guardian"