#!/bin/bash
set -e

echo "🌅 Running post-start setup for Synthetic Data Guardian..."

# Check if services are running
echo "🔍 Checking service health..."

# Function to check if a service is healthy
check_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            echo "❌ $service_name failed to start within timeout"
            return 1
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts - $service_name not ready yet..."
        sleep 2
        ((attempt++))
    done
}

# Start essential services if not running
if command -v docker-compose &> /dev/null; then
    echo "🐳 Ensuring Docker services are running..."
    docker-compose up -d db redis neo4j
    
    # Wait for services to be ready
    check_service "PostgreSQL" "postgresql://localhost:5432" || echo "⚠️ PostgreSQL may not be ready"
    check_service "Redis" "redis://localhost:6379" || echo "⚠️ Redis may not be ready"
    check_service "Neo4j" "http://localhost:7474" || echo "⚠️ Neo4j may not be ready"
fi

# Update dependencies if package files changed
echo "📦 Checking for dependency updates..."
if [ package.json -nt node_modules/.package-lock-time ] 2>/dev/null; then
    echo "🔄 package.json changed, updating Node.js dependencies..."
    npm install
    touch node_modules/.package-lock-time
fi

if [ requirements-dev.txt -nt .pip-install-time ] 2>/dev/null; then
    echo "🔄 requirements-dev.txt changed, updating Python dependencies..."
    pip install -r requirements-dev.txt
    touch .pip-install-time
fi

# Run pre-commit autoupdate periodically
if [ -f ".pre-commit-config.yaml" ]; then
    if [ ! -f ".pre-commit-last-update" ] || [ $(($(date +%s) - $(stat -c %Y .pre-commit-last-update 2>/dev/null || echo 0))) -gt 604800 ]; then
        echo "🔄 Updating pre-commit hooks..."
        pre-commit autoupdate || echo "⚠️ Pre-commit update failed"
        touch .pre-commit-last-update
    fi
fi

# Setup development database schema if needed
echo "🗄️ Checking database schema..."
if command -v npm &> /dev/null && npm run | grep -q "db:migrate"; then
    echo "🔄 Running database migrations..."
    npm run db:migrate || echo "⚠️ Database migration failed or not configured"
fi

# Start development services
echo "🚀 Starting development environment..."

# Show system status
echo ""
echo "📊 System Status:"
echo "=================="

# Check Node.js
if command -v node &> /dev/null; then
    echo "Node.js:    $(node --version)"
else
    echo "Node.js:    ❌ Not found"
fi

# Check Python
if command -v python &> /dev/null; then
    echo "Python:     $(python --version)"
else
    echo "Python:     ❌ Not found"
fi

# Check Docker
if command -v docker &> /dev/null; then
    echo "Docker:     $(docker --version | cut -d' ' -f3 | tr -d ',')"
else
    echo "Docker:     ❌ Not found"
fi

# Check Git
if command -v git &> /dev/null; then
    echo "Git:        $(git --version | cut -d' ' -f3)"
else
    echo "Git:        ❌ Not found"
fi

echo ""
echo "🔗 Available Services:"
echo "======================"

# Check service ports
services=(
    "8080:API Server"
    "3000:Frontend Dev"
    "5432:PostgreSQL"
    "6379:Redis"
    "7687:Neo4j"
    "9090:Prometheus"
    "3001:Grafana"
)

for service in "${services[@]}"; do
    port=$(echo $service | cut -d: -f1)
    name=$(echo $service | cut -d: -f2)
    
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo "$name: ✅ http://localhost:$port"
    else
        echo "$name: ⏳ Starting on port $port"
    fi
done

echo ""
echo "💡 Development Tips:"
echo "===================="
echo "• Use 'npm run dev' to start the development server"
echo "• Use 'npm run test:watch' for continuous testing"
echo "• Use 'docker-compose logs -f' to view service logs"
echo "• Use 'sg_health' alias to check API health"
echo "• Use 'sg_reset' alias to reset the development environment"
echo ""

# Start the development server in background if not already running
if ! pgrep -f "npm.*dev" > /dev/null; then
    echo "🎬 Starting development server..."
    nohup npm run dev > logs/dev-server.log 2>&1 &
    echo "📝 Development server logs: logs/dev-server.log"
fi

echo "✅ Post-start setup complete!"
echo "🎉 Synthetic Data Guardian development environment is ready!"