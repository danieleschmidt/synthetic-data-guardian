# GitHub Actions Workflow Setup Instructions

This directory contains template workflows that need to be manually copied to enable full CI/CD automation.

## Quick Setup

To enable the complete SDLC automation infrastructure, copy these workflow templates to your `.github/workflows/` directory:

```bash
# Create workflows directory if it doesn't exist
mkdir -p .github/workflows/

# Copy workflow templates
cp docs/workflows/ci-workflow-template.yml .github/workflows/ci.yml
cp docs/workflows/security-workflow-template.yml .github/workflows/security.yml
cp docs/workflows/performance-workflow-template.yml .github/workflows/performance.yml
```

## Required GitHub Secrets

Configure these secrets in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

### Security Scanning
- `SNYK_TOKEN` - Snyk API token for vulnerability scanning
- `CODECOV_TOKEN` - Codecov token for coverage reporting

### Optional Integrations
- `K6_CLOUD_TOKEN` - K6 Cloud token for performance testing
- `SENTRY_DSN` - Sentry DSN for error monitoring

## Workflow Features

### CI Pipeline (`ci.yml`)
- ✅ Automated testing (unit, integration, e2e)
- ✅ Code quality checks (linting, formatting, type checking)
- ✅ Multi-language support (Node.js, Python)
- ✅ Database services (PostgreSQL, Redis)
- ✅ Docker image building
- ✅ Coverage reporting

### Security Pipeline (`security.yml`)
- 🔒 OWASP dependency checking
- 🔒 Container vulnerability scanning (Trivy)
- 🔒 Static code analysis (CodeQL)
- 🔒 Secret detection (TruffleHog)
- 🔒 SBOM generation
- 🔒 Weekly scheduled scans

### Performance Pipeline (`performance.yml`)
- 📊 Load testing with K6
- 📊 Lighthouse performance audits
- 📊 Memory profiling and leak detection
- 📊 Performance regression detection

## Repository Settings

### Branch Protection
Enable branch protection for `main` and `develop` branches:
1. Go to `Settings > Branches`
2. Add rule for `main` branch
3. Enable:
   - Require status checks to pass
   - Require branches to be up to date
   - Require review from code owners
   - Dismiss stale reviews

### Dependabot Configuration
The `renovate.json` configuration is already in place for automated dependency management. Alternatively, enable GitHub's Dependabot:
1. Go to `Settings > Security & analysis`
2. Enable "Dependabot alerts"
3. Enable "Dependabot security updates"

## Manual Workflow Creation Steps

Since automated workflow creation requires special permissions, follow these steps:

1. **Copy Templates**: Use the bash commands above to copy workflow templates
2. **Configure Secrets**: Add required secrets in GitHub settings
3. **Test Workflows**: Create a test PR to verify workflow execution
4. **Monitor Results**: Check Actions tab for workflow results and artifacts

## Troubleshooting

### Common Issues

**Permission Errors**: Ensure your GitHub token has `workflow` permissions
**Secret Not Found**: Verify secrets are configured in repository settings
**Service Connection**: Check if external services (Snyk, Codecov) are properly configured

### Workflow Status

After setup, you should see these workflows in the Actions tab:
- ✅ CI/CD Pipeline (runs on every push/PR)
- 🔒 Security Scanning (runs on push/PR + weekly)
- 📊 Performance Testing (runs on push/PR + weekly)

## Next Steps

1. **Copy workflow files** using the commands above
2. **Configure repository secrets** for external integrations
3. **Enable branch protection** rules
4. **Create a test PR** to verify workflow execution
5. **Monitor and tune** workflow performance

This setup provides enterprise-grade SDLC automation with comprehensive testing, security scanning, and performance monitoring.