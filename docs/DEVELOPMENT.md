# 🚀 Development Guide

Comprehensive guide for setting up and developing the Synthetic Data Guardian project.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Database & Services](#database--services)
- [Debugging](#debugging)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### Required Tools

- **Node.js** 18+ and **npm** 8+
- **Python** 3.9+ (3.11+ recommended)
- **Docker** & **Docker Compose**
- **Git** 2.30+

### Optional Tools

- **VS Code** (recommended editor)
- **GitHub CLI** (`gh`)
- **k6** (for performance testing)
- **Neo4j Desktop** (for local graph database)

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **OS**: Linux, macOS, or Windows with WSL2

## ⚡ Quick Start

### Automated Setup

```bash
# Clone repository
git clone https://github.com/danieleschmidt/synthetic-data-guardian.git
cd synthetic-data-guardian

# Run automated setup script
./scripts/setup.sh
```

### Manual Setup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Install all dependencies
make install

# 3. Start development environment
make dev

# 4. Verify setup
make health-check
```

## 🏗️ Development Environment

### Environment Options

#### 1. Local Development (Recommended)

```bash
# Install dependencies
make install

# Start local services
make dev

# Start application
npm run dev  # Node.js API
# or
python -m synthetic_guardian.cli serve  # Python API
```

#### 2. DevContainer (VS Code)

1. Open project in VS Code
2. Click "Reopen in Container" when prompted
3. All dependencies automatically installed

#### 3. Docker Development

```bash
# Build and start all services
make docker-compose-up

# View logs
make docker-compose-logs

# Stop services
make docker-compose-down
```

### Environment Variables

Create `.env` file based on `.env.example`:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/synthetic_guardian
REDIS_URL=redis://localhost:6379
NEO4J_URI=bolt://localhost:7687

# API Keys (for external services)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Security
SECRET_KEY=your_secret_key
JWT_SECRET=your_jwt_secret

# Monitoring
PROMETHEUS_ENABLED=true
LANG_OBSERVATORY_URL=http://localhost:3001
```

## 📁 Project Structure

```
synthetic-data-guardian/
├── src/                          # Source code
│   ├── synthetic_guardian/       # Python package
│   │   ├── api/                  # FastAPI routes
│   │   ├── core/                 # Core business logic
│   │   ├── generators/           # Data generators
│   │   ├── validators/           # Validation frameworks
│   │   ├── watermarks/           # Watermarking systems
│   │   └── cli/                  # Command line interface
│   ├── middleware/               # Express middleware
│   └── routes/                   # Express routes
├── tests/                        # Test files
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── e2e/                      # End-to-end tests
│   ├── performance/              # Performance tests
│   └── fixtures/                 # Test data
├── docs/                         # Documentation
├── scripts/                      # Build and utility scripts
├── monitoring/                   # Monitoring configurations
└── .devcontainer/               # DevContainer configuration
```

## 🔄 Development Workflow

### 1. Feature Development

```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make changes and test
make test
make lint

# 3. Commit changes
git add .
git commit -m "feat: add your feature description"

# 4. Push and create PR
git push -u origin feature/your-feature-name
gh pr create
```

### 2. Code Standards

- **Python**: Follow PEP 8, use Black for formatting
- **TypeScript**: Follow project ESLint configuration
- **Commits**: Use [Conventional Commits](https://conventionalcommits.org/)
- **Documentation**: Update relevant docs with changes

### 3. Pre-commit Hooks

Automatically run on each commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🧪 Testing

### Test Categories

#### Unit Tests
```bash
# Python
pytest tests/unit/
pytest tests/unit/test_generators.py -v

# Node.js
npm run test
npm run test -- --testNamePattern="generator"
```

#### Integration Tests
```bash
# Python
pytest tests/integration/ -m integration

# Node.js
npm run test:integration
```

#### End-to-End Tests
```bash
# Playwright tests
npm run test:e2e

# Run specific test
npx playwright test tests/e2e/generation-pipeline.spec.ts
```

#### Performance Tests
```bash
# k6 load tests
npm run test:performance

# Python performance tests
pytest tests/performance/ -m performance
```

### Test Development

#### Writing Python Tests

```python
import pytest
from synthetic_guardian.generators import SDVGenerator

class TestSDVGenerator:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000]
        })
    
    def test_generator_initialization(self, sample_data):
        generator = SDVGenerator()
        generator.fit(sample_data)
        assert generator.is_fitted
```

#### Writing TypeScript Tests

```typescript
import { describe, it, expect } from '@jest/globals';
import { HealthService } from '../src/services/health';

describe('HealthService', () => {
  it('should return healthy status', async () => {
    const service = new HealthService();
    const status = await service.check();
    expect(status.healthy).toBe(true);
  });
});
```

## ✅ Code Quality

### Linting & Formatting

```bash
# Run all quality checks
make check

# Fix formatting issues
make lint-fix
make format

# Type checking
make typecheck
```

### Code Coverage

```bash
# Generate coverage report
make test-coverage

# View HTML report
open coverage_html/index.html  # macOS
xdg-open coverage_html/index.html  # Linux
```

### Security Scanning

```bash
# Run security audit
make security-audit

# Scan for vulnerabilities
make security-scan

# Check for secrets
truffleHog .
```

## 🗄️ Database & Services

### Local Services

Start required services:

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# Individual services
docker-compose -f docker-compose.dev.yml up -d postgres
docker-compose -f docker-compose.dev.yml up -d redis
docker-compose -f docker-compose.dev.yml up -d neo4j
```

### Database Migrations

```bash
# Python (Alembic)
alembic upgrade head
alembic revision --autogenerate -m "description"

# Node.js (if using TypeORM)
npm run db:migrate
npm run db:generate
```

### Seed Data

```bash
# Load test data
make db-seed

# Python
python scripts/seed_database.py

# Custom data
python -m synthetic_guardian.cli generate --config seed-config.yaml
```

## 🐛 Debugging

### VS Code Debugging

Launch configurations in `.vscode/launch.json`:

```json
{
  "type": "python",
  "request": "launch",
  "module": "synthetic_guardian.cli",
  "args": ["serve", "--debug"],
  "env": {"PYTHONPATH": "${workspaceFolder}/src"}
}
```

### Docker Debugging

```bash
# Inspect running container
make inspect

# Access container shell
docker exec -it synthetic-data-guardian /bin/bash

# View container logs
docker logs synthetic-data-guardian -f
```

### Log Analysis

```bash
# Application logs
tail -f logs/app.log

# Service logs
make dev-logs

# Structured log analysis
cat logs/app.log | jq '.level="ERROR"'
```

## ⚡ Performance

### Profiling

```bash
# Python profiling
python -m cProfile -o profile.stats -m synthetic_guardian.cli generate

# Analyze profile
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"

# Memory profiling
pip install memory-profiler
python -m memory_profiler synthetic_guardian/generators/sdv.py
```

### Benchmarking

```bash
# Run benchmarks
make benchmark

# Performance regression tests
pytest tests/performance/ --benchmark-only
```

### Monitoring

```bash
# Start monitoring stack
make monitor

# View metrics
open http://localhost:9090  # Prometheus
open http://localhost:3001  # Grafana
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Dependency Conflicts

```bash
# Clean and reinstall
make clean-all
make install

# Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
pip install -r requirements-dev.txt
```

#### 2. Docker Issues

```bash
# Clean Docker state
docker system prune -af
docker volume prune -f

# Rebuild containers
make docker-compose-down
make docker-compose-up
```

#### 3. Port Conflicts

```bash
# Check port usage
lsof -i :8080
lsof -i :5432

# Kill process using port
kill -9 $(lsof -t -i:8080)
```

#### 4. Database Connection Issues

```bash
# Check database status
docker-compose -f docker-compose.dev.yml ps postgres

# Reset database
make db-reset

# Check connection
psql $DATABASE_URL -c "SELECT 1"
```

### Getting Help

1. **Check logs**: `make dev-logs`
2. **Run health check**: `make health-check`
3. **Verify environment**: Check `.env` file
4. **Update dependencies**: `make update-deps`
5. **Ask for help**: Create GitHub issue with:
   - OS and version
   - Docker version
   - Error logs
   - Steps to reproduce

### Useful Commands

```bash
# Show all make targets
make help

# Check system status
make health-check

# View current version
make version

# Clean everything
make clean-all

# Update dependencies
make update-deps
```

## 📚 Documentation

- [Contributing Guide](../CONTRIBUTING.md) - Detailed contribution process
- [Getting Started](guides/getting-started.md) - User getting started guide
- [Architecture](../ARCHITECTURE.md) - System architecture overview
- [API Documentation](../api/) - API reference
- [Deployment Guide](deployment/) - Production deployment

For comprehensive development information, see [CONTRIBUTING.md](../CONTRIBUTING.md).