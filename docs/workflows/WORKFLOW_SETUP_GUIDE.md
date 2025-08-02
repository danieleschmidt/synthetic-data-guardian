# GitHub Actions Workflow Setup Guide

This guide provides step-by-step instructions for setting up all required GitHub Actions workflows for the Synthetic Data Guardian project.

## 🚨 Important Notice

**GitHub App Permission Limitation**: The automated setup of workflow files requires write access to the `.github/workflows/` directory. Repository administrators must manually create these workflow files using the templates provided below.

## Required Workflows

### 1. CI/CD Pipeline (`ci.yml`)

**File Location**: `.github/workflows/ci.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

env:
  NODE_VERSION: '18'
  PYTHON_VERSION: '3.11'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        test-type: [unit, integration, security]
    
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: ${{ env.NODE_VERSION }}
        cache: 'npm'
        
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
        
    - name: Install Dependencies
      run: |
        npm ci
        pip install -r requirements-dev.txt
        
    - name: Run ${{ matrix.test-type }} Tests
      run: |
        case "${{ matrix.test-type }}" in
          unit)
            npm run test:unit
            pytest tests/python -m unit
            ;;
          integration)
            npm run test:integration
            pytest tests/python -m integration
            ;;
          security)
            npm run test:security
            pytest tests/python -m security
            ;;
        esac
        
    - name: Upload Coverage
      if: matrix.test-type == 'unit'
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage/lcov.info,./coverage/python/coverage.xml

  lint-and-format:
    name: Code Quality
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: ${{ env.NODE_VERSION }}
        cache: 'npm'
        
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
        
    - name: Install Dependencies
      run: |
        npm ci
        pip install -r requirements-dev.txt
        
    - name: Run Linting
      run: |
        npm run lint
        flake8 src/ tests/python/
        
    - name: Check Formatting
      run: |
        npm run format:check
        black --check src/ tests/python/
        
    - name: Type Checking
      run: |
        npm run typecheck
        mypy src/

  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'trivy-results.sarif'
        
    - name: Run Snyk Security Scan
      uses: snyk/actions/node@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high
        
    - name: Run CodeQL Analysis
      uses: github/codeql-action/analyze@v3

  build:
    name: Build Application
    runs-on: ubuntu-latest
    needs: [test, lint-and-format, security-scan]
    
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: ${{ env.NODE_VERSION }}
        cache: 'npm'
        
    - name: Install Dependencies
      run: npm ci
      
    - name: Build Application
      run: npm run build
      
    - name: Upload Build Artifacts
      uses: actions/upload-artifact@v4
      with:
        name: build-files
        path: dist/
        
  docker-build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [build]
    if: github.event_name == 'push'
    
    permissions:
      contents: read
      packages: write
      
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [docker-build]
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Deploy to Staging
      run: |
        echo "Deploying to staging environment"
        # Add deployment commands here
```

### 2. Performance Testing (`performance.yml`)

**File Location**: `.github/workflows/performance.yml`

```yaml
name: Performance Testing

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:
  push:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'tests/performance/**'

jobs:
  performance-test:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
        cache: 'npm'
        
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
        
    - name: Install Dependencies
      run: |
        npm ci
        pip install -r requirements-dev.txt
        
    - name: Start Test Environment
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30  # Wait for services to be ready
        
    - name: Run Performance Tests
      run: |
        npm run test:performance
        pytest tests/python -m performance --benchmark-json=benchmark.json
        
    - name: Upload Performance Results
      uses: actions/upload-artifact@v4
      with:
        name: performance-results
        path: |
          test-results/performance/
          benchmark.json
          
    - name: Performance Regression Check
      run: |
        python scripts/check-performance-regression.py benchmark.json
        
    - name: Cleanup
      if: always()
      run: docker-compose -f docker-compose.test.yml down
```

### 3. Security Scanning (`security.yml`)

**File Location**: `.github/workflows/security.yml`

```yaml
name: Security Scanning

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday at 6 AM

jobs:
  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      
    - name: Run npm audit
      run: npm audit --audit-level moderate
      
    - name: Run Python safety check
      run: |
        pip install safety
        safety check -r requirements.txt
        
    - name: Run Snyk
      uses: snyk/actions/node@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      
    - name: Build Docker image
      run: docker build -t synthetic-guardian:scan .
      
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'synthetic-guardian:scan'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'trivy-results.sarif'

  secret-scan:
    name: Secret Detection
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      
    - name: Run GitGuardian scan
      uses: GitGuardian/ggshield/actions/secret@main
      env:
        GITHUB_PUSH_BEFORE_SHA: ${{ github.event.before }}
        GITHUB_PUSH_BASE_SHA: ${{ github.event.base }}
        GITHUB_DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}
        GITGUARDIAN_API_KEY: ${{ secrets.GITGUARDIAN_API_KEY }}

  code-analysis:
    name: Static Code Analysis
    runs-on: ubuntu-latest
    
    permissions:
      actions: read
      contents: read
      security-events: write
      
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: javascript, python
        
    - name: Autobuild
      uses: github/codeql-action/autobuild@v3
      
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
```

### 4. Release Automation (`release.yml`)

**File Location**: `.github/workflows/release.yml`

```yaml
name: Release

on:
  push:
    branches: [ main ]

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    if: "!contains(github.event.head_commit.message, 'skip ci')"
    
    permissions:
      contents: write
      packages: write
      issues: write
      pull-requests: write
      
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
        token: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
        cache: 'npm'
        
    - name: Install Dependencies
      run: npm ci
      
    - name: Build Application
      run: npm run build
      
    - name: Run Tests
      run: npm test
      
    - name: Semantic Release
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
      run: npx semantic-release
      
    - name: Build and Push Docker Images
      if: success()
      env:
        REGISTRY: ghcr.io
        IMAGE_NAME: ${{ github.repository }}
      run: |
        echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
        docker build -t $REGISTRY/$IMAGE_NAME:latest .
        docker push $REGISTRY/$IMAGE_NAME:latest
```

## Required Repository Secrets

Configure these secrets in your GitHub repository settings:

### Core Secrets
- `GITHUB_TOKEN` (automatically provided)
- `NPM_TOKEN` (if publishing to npm)

### Security Scanning
- `SNYK_TOKEN` (Snyk security scanning)
- `GITGUARDIAN_API_KEY` (Secret detection)

### External Services
- `OPENAI_API_KEY` (for tests requiring OpenAI)
- `ANTHROPIC_API_KEY` (for tests requiring Anthropic)

### Deployment
- `KUBECONFIG` (Kubernetes deployment)
- `DOCKER_REGISTRY_TOKEN` (if using external registry)

## Branch Protection Rules

Configure the following branch protection rules for the `main` branch:

### Required Status Checks
- `test (unit)`
- `test (integration)`
- `test (security)`
- `lint-and-format`
- `security-scan`
- `build`

### Protection Settings
- [x] Require a pull request before merging
- [x] Require approvals (minimum 1)
- [x] Dismiss stale reviews when new commits are pushed
- [x] Require review from code owners
- [x] Require status checks to pass before merging
- [x] Require branches to be up to date before merging
- [x] Require conversation resolution before merging
- [x] Include administrators

## Environment Setup

### Staging Environment
- **Name**: `staging`
- **URL**: `https://staging.your-domain.com`
- **Protection Rules**: None
- **Reviewers**: Development team

### Production Environment
- **Name**: `production`
- **URL**: `https://your-domain.com`
- **Protection Rules**: Required reviewers
- **Reviewers**: Senior developers, DevOps team

## Workflow Security Best Practices

### 1. Minimal Permissions
All workflows use the principle of least privilege:

```yaml
permissions:
  contents: read
  # Only add additional permissions when required
```

### 2. Pinned Dependencies
All actions are pinned to specific SHA or version:

```yaml
- uses: actions/checkout@v4  # Good: pinned to major version
- uses: actions/checkout@f43a0e5ff2bd294095638e18286ca9a3d1956744  # Better: pinned to SHA
```

### 3. Secret Management
- Never log secrets
- Use GitHub's encrypted secrets
- Rotate secrets regularly
- Use environment-specific secrets

### 4. Input Validation
Validate all inputs to prevent injection attacks:

```yaml
- name: Validate Input
  run: |
    if [[ ! "${{ github.event.inputs.environment }}" =~ ^(staging|production)$ ]]; then
      echo "Invalid environment"
      exit 1
    fi
```

## Monitoring and Alerting

### Workflow Notifications
Configure GitHub notifications for:
- Failed workflow runs
- Security alerts
- Deployment notifications

### External Monitoring
- Integrate with monitoring tools (Datadog, New Relic)
- Set up alerts for performance regressions
- Monitor deployment success/failure rates

## Troubleshooting

### Common Issues

1. **Workflow fails with permission error**
   - Check repository permissions
   - Verify GITHUB_TOKEN scope
   - Review branch protection rules

2. **Tests timeout**
   - Increase timeout values
   - Optimize test performance
   - Use test parallelization

3. **Docker build fails**
   - Check Dockerfile syntax
   - Verify base image availability
   - Review build context size

### Debugging Steps

1. Check workflow logs in Actions tab
2. Review failed step details
3. Reproduce issue locally
4. Check for recent changes in dependencies
5. Verify all required secrets are set

## Maintenance

### Regular Tasks
- [ ] Update action versions monthly
- [ ] Review and rotate secrets quarterly
- [ ] Monitor workflow performance
- [ ] Update security scanning rules
- [ ] Review branch protection settings

### Annual Reviews
- [ ] Audit all workflows for security
- [ ] Review and update permissions
- [ ] Optimize workflow performance
- [ ] Update documentation