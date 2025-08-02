#!/bin/bash

# =============================================================================
# Synthetic Data Guardian - Docker Entrypoint Script
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [ "${DEBUG:-false}" = "true" ]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Environment validation
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Required environment variables
    local required_vars=(
        "NODE_ENV"
        "PORT"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        log_error "Please set these variables before starting the container"
        exit 1
    fi
    
    # Validate NODE_ENV
    case "${NODE_ENV}" in
        development|staging|production)
            log_info "Environment: ${NODE_ENV}"
            ;;
        *)
            log_error "Invalid NODE_ENV: ${NODE_ENV}. Must be development, staging, or production"
            exit 1
            ;;
    esac
    
    # Validate PORT
    if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || [ "${PORT}" -lt 1 ] || [ "${PORT}" -gt 65535 ]; then
        log_error "Invalid PORT: ${PORT}. Must be a number between 1 and 65535"
        exit 1
    fi
    
    log_info "Environment validation completed successfully"
}

# Database connectivity check
check_database() {
    if [ -n "${DATABASE_URL}" ]; then
        log_info "Checking database connectivity..."
        
        local max_attempts=30
        local attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            log_debug "Database connection attempt $attempt/$max_attempts"
            
            if node -e "
                const { Client } = require('pg');
                const client = new Client({ connectionString: process.env.DATABASE_URL });
                client.connect()
                    .then(() => {
                        console.log('Database connection successful');
                        client.end();
                        process.exit(0);
                    })
                    .catch((err) => {
                        console.error('Database connection failed:', err.message);
                        process.exit(1);
                    });
            " 2>/dev/null; then
                log_info "Database connection established"
                return 0
            fi
            
            if [ $attempt -eq $max_attempts ]; then
                log_error "Failed to connect to database after $max_attempts attempts"
                if [ "${NODE_ENV}" = "production" ]; then
                    exit 1
                else
                    log_warn "Continuing without database connection (non-production environment)"
                fi
            fi
            
            sleep 2
            attempt=$((attempt + 1))
        done
    else
        log_warn "DATABASE_URL not set, skipping database connectivity check"
    fi
}

# Redis connectivity check
check_redis() {
    if [ -n "${REDIS_URL}" ]; then
        log_info "Checking Redis connectivity..."
        
        local max_attempts=15
        local attempt=1
        
        while [ $attempt -le $max_attempts ]; do
            log_debug "Redis connection attempt $attempt/$max_attempts"
            
            if node -e "
                const redis = require('redis');
                const client = redis.createClient({ url: process.env.REDIS_URL });
                client.connect()
                    .then(() => {
                        console.log('Redis connection successful');
                        client.quit();
                        process.exit(0);
                    })
                    .catch((err) => {
                        console.error('Redis connection failed:', err.message);
                        process.exit(1);
                    });
            " 2>/dev/null; then
                log_info "Redis connection established"
                return 0
            fi
            
            if [ $attempt -eq $max_attempts ]; then
                log_warn "Failed to connect to Redis after $max_attempts attempts"
                log_warn "Continuing without Redis (caching will be disabled)"
            fi
            
            sleep 1
            attempt=$((attempt + 1))
        done
    else
        log_warn "REDIS_URL not set, skipping Redis connectivity check"
    fi
}

# Health check
health_check() {
    log_info "Performing application health check..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "http://localhost:${PORT}/api/v1/health" > /dev/null 2>&1; then
            log_info "Application health check passed"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            log_error "Application health check failed after $max_attempts attempts"
            return 1
        fi
        
        sleep 2
        attempt=$((attempt + 1))
    done
}

# Pre-start checks
pre_start_checks() {
    log_info "Running pre-start checks..."
    
    # Check if required files exist
    local required_files=(
        "package.json"
        "dist/index.js"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    
    # Check Node.js version
    local node_version=$(node --version | sed 's/v//')
    local required_version="18.0.0"
    
    if ! node -e "
        const current = process.version.slice(1).split('.').map(Number);
        const required = '$required_version'.split('.').map(Number);
        const isValid = current[0] > required[0] || 
                       (current[0] === required[0] && current[1] >= required[1]);
        process.exit(isValid ? 0 : 1);
    "; then
        log_error "Node.js version $node_version is not supported. Required: $required_version or higher"
        exit 1
    fi
    
    log_info "Pre-start checks completed successfully"
}

# Graceful shutdown handler
shutdown_handler() {
    log_info "Received shutdown signal, gracefully shutting down..."
    
    if [ -n "$APP_PID" ]; then
        log_info "Stopping application (PID: $APP_PID)..."
        kill -TERM "$APP_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        local timeout=30
        local count=0
        
        while kill -0 "$APP_PID" 2>/dev/null && [ $count -lt $timeout ]; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$APP_PID" 2>/dev/null; then
            log_warn "Application did not stop gracefully, force killing..."
            kill -KILL "$APP_PID" 2>/dev/null || true
        fi
    fi
    
    log_info "Shutdown completed"
    exit 0
}

# Set up signal handlers
trap shutdown_handler SIGTERM SIGINT

# Main execution
main() {
    log_info "Starting Synthetic Data Guardian..."
    log_info "Version: ${APP_VERSION:-unknown}"
    log_info "Environment: ${NODE_ENV}"
    log_info "Port: ${PORT}"
    
    # Run checks
    validate_environment
    pre_start_checks
    
    # Optional dependency checks (non-blocking in development)
    if [ "${NODE_ENV}" = "production" ]; then
        check_database
        check_redis
    else
        check_database || true
        check_redis || true
    fi
    
    # Start the application
    log_info "Starting application server..."
    
    if [ "${NODE_ENV}" = "development" ]; then
        # Development mode with hot reload
        exec npm run dev &
    else
        # Production mode
        exec node dist/index.js &
    fi
    
    APP_PID=$!
    log_info "Application started with PID: $APP_PID"
    
    # Wait for application to be ready
    sleep 5
    
    # Run health check in background for production
    if [ "${NODE_ENV}" = "production" ]; then
        health_check &
    fi
    
    # Wait for the application process
    wait $APP_PID
    
    # If we reach here, the application has exited
    local exit_code=$?
    log_info "Application exited with code: $exit_code"
    exit $exit_code
}

# Handle special commands
case "${1:-}" in
    --help|-h)
        echo "Synthetic Data Guardian Docker Entrypoint"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start          Start the application (default)"
        echo "  health         Run health check only"
        echo "  validate       Validate environment only"
        echo "  shell          Start interactive shell"
        echo "  --help, -h     Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  NODE_ENV       Application environment (development|staging|production)"
        echo "  PORT           Application port (default: 8080)"
        echo "  DATABASE_URL   PostgreSQL connection string"
        echo "  REDIS_URL      Redis connection string"
        echo "  DEBUG          Enable debug logging (true|false)"
        exit 0
        ;;
    health)
        validate_environment
        health_check
        exit $?
        ;;
    validate)
        validate_environment
        log_info "Environment validation successful"
        exit 0
        ;;
    shell)
        log_info "Starting interactive shell..."
        exec /bin/sh
        ;;
    start|"")
        main
        ;;
    *)
        # Pass through any other commands
        log_info "Executing command: $*"
        exec "$@"
        ;;
esac