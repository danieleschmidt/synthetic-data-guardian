# 🔄 Migration Guide - SDLC Implementation

## Overview

This guide provides step-by-step instructions for migrating to the new SDLC implementation for the Synthetic Data Guardian project. Follow these procedures to ensure a smooth transition to the enhanced development lifecycle.

## 📋 Pre-Migration Checklist

### Prerequisites Verification
- [ ] Git repository access with appropriate permissions
- [ ] Node.js 18+ installed
- [ ] Python 3.9+ installed
- [ ] Docker and Docker Compose installed
- [ ] VS Code (recommended) installed
- [ ] GitHub CLI (gh) installed (optional but recommended)

### Backup Current State
```bash
# Create backup branch
git checkout main
git branch backup/pre-sdlc-migration
git push origin backup/pre-sdlc-migration

# Backup current configuration files
mkdir -p backup/configs
cp .gitignore backup/configs/ 2>/dev/null || true
cp package.json backup/configs/ 2>/dev/null || true
cp requirements.txt backup/configs/ 2>/dev/null || true
cp Dockerfile backup/configs/ 2>/dev/null || true
cp docker-compose.yml backup/configs/ 2>/dev/null || true
```

## 🚀 Migration Steps

### Step 1: Pull Latest SDLC Implementation
```bash
# Ensure you're on the main branch
git checkout main
git pull origin main

# Verify SDLC files are present
ls -la scripts/
ls -la automation/
ls -la .github/
ls -la docs/
```

### Step 2: Initial Environment Setup
```bash
# Run the automated setup
make setup

# If make is not available, run setup script directly
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Step 3: Verify Development Environment
```bash
# Test the development environment
make test-env

# Or manually verify components
node --version
python --version
docker --version
npm --version
pip --version
```

### Step 4: Configure Project-Specific Settings

#### Update package.json (if exists)
```bash
# Backup current package.json
cp package.json package.json.backup

# Merge SDLC-specific scripts (manual step required)
# Add the following scripts if not present:
```

```json
{
  "scripts": {
    "test": "jest",
    "test:coverage": "jest --coverage",
    "test:watch": "jest --watch",
    "test:e2e": "playwright test",
    "lint": "eslint src/ --ext .js,.ts,.tsx",
    "lint:fix": "eslint src/ --ext .js,.ts,.tsx --fix",
    "format": "prettier --write src/",
    "build": "webpack --mode production",
    "dev": "webpack serve --mode development",
    "security:audit": "npm audit --audit-level moderate"
  }
}
```

#### Update requirements.txt (if exists)
```bash
# Backup current requirements
cp requirements.txt requirements.txt.backup

# Add SDLC-specific Python packages
cat >> requirements.txt << 'EOF'

# SDLC Development Dependencies
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
flake8>=4.0.0
isort>=5.0.0
bandit>=1.7.0
safety>=2.0.0
EOF
```

### Step 5: Update Docker Configuration

#### Dockerfile Updates
```bash
# If you have an existing Dockerfile, compare with the SDLC version
if [ -f Dockerfile ]; then
    echo "Existing Dockerfile found. Please review the new Dockerfile.sdlc for improvements:"
    echo "Key improvements include:"
    echo "- Multi-stage builds"
    echo "- Security hardening"
    echo "- Health checks"
    echo "- Non-root user"
    echo ""
    echo "Consider merging improvements manually or renaming current Dockerfile:"
    echo "mv Dockerfile Dockerfile.legacy"
    echo "cp Dockerfile.sdlc Dockerfile"
fi
```

#### docker-compose.yml Updates
```bash
# Update Docker Compose configuration
if [ -f docker-compose.yml ]; then
    echo "Existing docker-compose.yml found."
    echo "The SDLC implementation includes enhanced docker-compose configuration."
    echo "Please review docker-compose.sdlc.yml for improvements."
fi
```

### Step 6: Configure CI/CD Workflows

#### GitHub Actions Setup
```bash
# Create GitHub Actions workflows directory if it doesn't exist
mkdir -p .github/workflows

# Copy example workflows from templates
cp docs/workflows/examples/dependency-update-workflow.yml .github/workflows/ 2>/dev/null || echo "Manual workflow setup required"
cp docs/workflows/examples/deployment-workflow.yml .github/workflows/ 2>/dev/null || echo "Manual workflow setup required"

echo "Please review the Comprehensive Workflow Guide for manual GitHub Actions setup:"
echo "docs/workflows/COMPREHENSIVE_WORKFLOW_GUIDE.md"
```

### Step 7: Database and Configuration Migration

#### Environment Variables
```bash
# Create environment configuration template
cat > .env.example << 'EOF'
# Application Configuration
NODE_ENV=development
PORT=8080
API_VERSION=v1

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/synthetic_guardian
REDIS_URL=redis://localhost:6379/0
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Security Configuration
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
WATERMARK_KEY=your-watermark-key-here

# External Services
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Monitoring Configuration
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000

# GitHub Configuration (for automation)
GITHUB_TOKEN=your-github-token
GITHUB_REPOSITORY=your-username/synthetic-data-guardian

# Notification Configuration
SLACK_WEBHOOK_URL=your-slack-webhook-url
EMAIL_SERVICE_API_KEY=your-email-service-key
EOF

echo "Created .env.example - please copy to .env and update with your values"
echo "cp .env.example .env"
```

### Step 8: Initialize Monitoring and Metrics

#### Repository Health Initialization
```bash
# Initialize repository health database
python automation/repository-health.py --repo-path . --verbose

# Run initial metrics collection
python scripts/metrics-collector.py --repo-path . --print

# Run initial technical debt analysis
python scripts/tech-debt-tracker.py --repo-path . --summary-only
```

#### Prometheus Configuration
```bash
# Set up Prometheus configuration
mkdir -p monitoring/prometheus
cp monitoring/prometheus/alert_rules.yml monitoring/prometheus/alert_rules.yml.example 2>/dev/null || echo "Prometheus setup complete"

echo "Prometheus configuration available in monitoring/prometheus/"
echo "Please review and customize alert rules for your environment"
```

### Step 9: Test Migration Success

#### Run Comprehensive Tests
```bash
# Test the build process
make build

# Run all tests
make test

# Run security scans
make security-scan

# Verify integration capabilities
python scripts/integration-tester.py --health-only

# Run performance benchmarks
python scripts/performance-benchmarker.py --build-only
```

#### Verify SDLC Components
```bash
# Check all SDLC scripts are executable
ls -la scripts/
ls -la automation/

# Verify configuration files
ls -la automation/*.json

# Check documentation
ls -la docs/
```

## 🔧 Configuration Customization

### Customize Metrics Collection
```bash
# Edit metrics configuration
cp automation/metrics-config.json automation/metrics-config.json.backup
nano automation/metrics-config.json

# Key customization points:
# - GitHub repository settings
# - Threshold values
# - Collection schedule
# - Alert configurations
```

### Customize Performance Benchmarking
```bash
# Edit performance configuration
cp automation/performance-config.json automation/performance-config.json.backup
nano automation/performance-config.json

# Key customization points:
# - API endpoints for testing
# - Performance thresholds
# - Load testing scenarios
# - Resource limits
```

### Customize Integration Testing
```bash
# Edit integration configuration
cp automation/integration-config.json automation/integration-config.json.backup
nano automation/integration-config.json

# Key customization points:
# - Service endpoints
# - Database connections
# - Test scenarios
# - Timeout values
```

## 🚨 Troubleshooting Migration Issues

### Common Issues and Solutions

#### Issue: Setup Script Fails
```bash
# Solution: Run with verbose output
chmod +x scripts/setup.sh
bash -x scripts/setup.sh

# Check for missing dependencies
which node npm python pip docker
```

#### Issue: Tests Fail After Migration
```bash
# Solution: Clear cache and reinstall
make clean
npm cache clean --force 2>/dev/null || true
pip cache purge 2>/dev/null || true
make setup
make test
```

#### Issue: Docker Build Fails
```bash
# Solution: Check Docker configuration
docker system prune -f
docker build --no-cache -t synthetic-data-guardian .

# Check for Docker-related issues
docker version
docker-compose version
```

#### Issue: Permission Errors
```bash
# Solution: Fix script permissions
chmod +x scripts/*.py
chmod +x scripts/*.sh
chmod +x automation/*.py

# Fix ownership if needed
sudo chown -R $USER:$USER .
```

#### Issue: Environment Variables Not Set
```bash
# Solution: Verify environment configuration
cp .env.example .env
nano .env

# Source environment variables
set -a
source .env
set +a
```

### Rollback Procedures

#### Quick Rollback
```bash
# If migration fails, rollback to previous state
git checkout backup/pre-sdlc-migration

# Restore original configuration files
cp backup/configs/* . 2>/dev/null || true

# Clean up SDLC files if needed
git clean -fd  # BE CAREFUL - this removes untracked files
```

#### Partial Rollback
```bash
# Rollback specific components
git checkout HEAD~1 -- scripts/  # Rollback scripts only
git checkout HEAD~1 -- automation/  # Rollback automation only
git checkout HEAD~1 -- .github/  # Rollback GitHub configs only
```

## 📊 Post-Migration Validation

### Validation Checklist
- [ ] All tests pass (`make test`)
- [ ] Build process works (`make build`)
- [ ] Security scans run without critical issues
- [ ] Development environment starts successfully
- [ ] Docker containers build and run
- [ ] Metrics collection works
- [ ] Performance benchmarks run
- [ ] Integration tests pass
- [ ] Documentation is accessible
- [ ] VS Code workspace loads correctly

### Health Check Commands
```bash
# Comprehensive health check
python automation/repository-health.py --report

# Quick validation script
cat > validate-migration.sh << 'EOF'
#!/bin/bash
echo "🔍 Validating SDLC Migration..."

# Check critical files
echo "✅ Checking critical files..."
test -f scripts/metrics-collector.py && echo "  ✓ Metrics collector found" || echo "  ✗ Metrics collector missing"
test -f automation/repository-health.py && echo "  ✓ Health monitor found" || echo "  ✗ Health monitor missing"
test -f .github/ISSUE_TEMPLATE/bug_report.md && echo "  ✓ Issue templates found" || echo "  ✗ Issue templates missing"

# Check executable permissions
echo "✅ Checking permissions..."
test -x scripts/metrics-collector.py && echo "  ✓ Scripts are executable" || echo "  ✗ Scripts need chmod +x"

# Check basic functionality
echo "✅ Testing basic functionality..."
make --version > /dev/null 2>&1 && echo "  ✓ Make available" || echo "  ✗ Make not available"
node --version > /dev/null 2>&1 && echo "  ✓ Node.js available" || echo "  ✗ Node.js not available"
python --version > /dev/null 2>&1 && echo "  ✓ Python available" || echo "  ✗ Python not available"

echo "🎉 Migration validation complete!"
EOF

chmod +x validate-migration.sh
./validate-migration.sh
```

## 📞 Getting Help

### Support Resources
- **Documentation:** Check `SDLC_IMPLEMENTATION_GUIDE.md`
- **Workflow Guide:** See `docs/workflows/COMPREHENSIVE_WORKFLOW_GUIDE.md`
- **GitHub Issues:** Create an issue for migration problems
- **Team Contact:** team@synthetic-guardian.com

### Emergency Contacts
- **Technical Issues:** Create GitHub issue with `migration` label
- **Urgent Problems:** Contact repository maintainers directly
- **Security Concerns:** Follow security disclosure procedures

## 📈 Next Steps

### After Successful Migration
1. **Team Training:** Share SDLC documentation with team
2. **Process Integration:** Integrate SDLC into daily workflows
3. **Monitoring Setup:** Configure monitoring dashboards
4. **Continuous Improvement:** Regular review of metrics and processes

### Recommended Timeline
- **Week 1:** Complete migration and basic validation
- **Week 2:** Team training and workflow integration
- **Week 3:** Monitor and fine-tune configurations
- **Week 4:** Full SDLC process adoption

---

**Migration Support:** Terragon Labs  
**Last Updated:** 2024-08-02  
**Version:** 1.0.0  
**Status:** Ready for Production