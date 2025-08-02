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
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Health check function
health_check() {
    local service=$1
    local host=$2
    local port=$3
    local max_attempts=${4:-30}
    local attempt=1

    log_info "Waiting for $service to be ready at $host:$port..."

    while [ $attempt -le $max_attempts ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            log_success "$service is ready!"
            return 0
        fi

        log_warn "Attempt $attempt/$max_attempts: $service not ready, waiting..."
        sleep 2
        ((attempt++))
    done

    log_error "$service failed to become ready after $max_attempts attempts"
    return 1
}

# Database migration function
run_migrations() {
    log_info "Running database migrations..."
    
    if command -v npm >/dev/null 2>&1; then
        npm run db:migrate || {
            log_error "Database migration failed"
            exit 1
        }
    else
        log_warn "npm not found, skipping migrations"
    fi
    
    log_success "Database migrations completed"
}

# Environment validation
validate_environment() {
    log_info "Validating environment variables..."
    
    # Required environment variables
    local required_vars=(
        "NODE_ENV"
        "PORT"
        "DATABASE_URL"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -ne 0 ]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Security setup
setup_security() {
    log_info "Setting up security configurations..."
    
    # Set secure file permissions
    chmod 600 /app/.env 2>/dev/null || true
    chmod 600 /app/config/*.json 2>/dev/null || true
    
    # Remove any temporary files
    find /tmp -type f -name "*.tmp" -delete 2>/dev/null || true
    
    log_success "Security setup completed"
}

# Performance optimization
optimize_performance() {
    log_info "Applying performance optimizations..."
    
    # Set Node.js performance flags
    export NODE_OPTIONS="--max-old-space-size=2048 --optimize-for-size"
    
    # Set timezone
    export TZ=${TZ:-UTC}
    
    log_success "Performance optimizations applied"
}

# Monitoring setup
setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create logs directory
    mkdir -p /app/logs
    
    # Set log level based on environment
    if [ "$NODE_ENV" = "production" ]; then
        export LOG_LEVEL=${LOG_LEVEL:-info}
    else
        export LOG_LEVEL=${LOG_LEVEL:-debug}
    fi
    
    log_success "Monitoring setup completed"
}

# Graceful shutdown handler
shutdown_handler() {
    log_info "Received shutdown signal, gracefully shutting down..."
    
    # Kill the main process if it's running
    if [ -n "$main_pid" ]; then
        kill -TERM "$main_pid" 2>/dev/null || true
        wait "$main_pid" 2>/dev/null || true
    fi
    
    log_success "Shutdown completed"
    exit 0
}

# Set up signal handlers
trap shutdown_handler SIGTERM SIGINT

# Main execution
main() {
    log_info "Starting Synthetic Data Guardian..."
    log_info "Environment: $NODE_ENV"
    log_info "Version: $(cat package.json | grep version | cut -d '"' -f 4)"
    
    # Run startup procedures
    validate_environment
    setup_security
    optimize_performance
    setup_monitoring
    
    # Wait for dependencies if in production
    if [ "$NODE_ENV" = "production" ]; then
        # Parse DATABASE_URL to get host and port
        if [[ $DATABASE_URL =~ postgres://[^@]+@([^:]+):([0-9]+)/ ]]; then
            db_host="${BASH_REMATCH[1]}"
            db_port="${BASH_REMATCH[2]}"
            health_check "PostgreSQL" "$db_host" "$db_port"
        fi
        
        # Check Redis if URL is provided
        if [ -n "$REDIS_URL" ] && [[ $REDIS_URL =~ redis://([^:]+):([0-9]+) ]]; then
            redis_host="${BASH_REMATCH[1]}"
            redis_port="${BASH_REMATCH[2]}"
            health_check "Redis" "$redis_host" "$redis_port"
        fi
        
        # Check Neo4j if URL is provided
        if [ -n "$NEO4J_URI" ] && [[ $NEO4J_URI =~ bolt://([^:]+):([0-9]+) ]]; then
            neo4j_host="${BASH_REMATCH[1]}"
            neo4j_port="${BASH_REMATCH[2]}"
            health_check "Neo4j" "$neo4j_host" "$neo4j_port"
        fi
        
        # Run database migrations
        run_migrations
    fi
    
    log_info "Starting application server..."
    
    # Start the main application
    if [ "$NODE_ENV" = "development" ]; then
        # Development mode with hot reload
        exec npm run dev &
    else
        # Production mode
        exec npm start &
    fi
    
    main_pid=$!
    log_success "Application started with PID $main_pid"
    
    # Wait for the main process
    wait "$main_pid"
}

# Handle different command line arguments
case "${1:-}" in
    "")
        # Default behavior - start the application
        main
        ;;
    "bash"|"sh")
        # Shell access for debugging
        exec "$@"
        ;;
    "test")
        # Run tests
        log_info "Running tests..."
        exec npm test
        ;;
    "migrate")
        # Run migrations only
        validate_environment
        run_migrations
        ;;
    "health")
        # Health check
        log_info "Performing health check..."
        curl -f http://localhost:${PORT:-8080}/api/v1/health || exit 1
        log_success "Health check passed"
        ;;
    *)
        # Custom command
        log_info "Executing custom command: $*"
        exec "$@"
        ;;
esac