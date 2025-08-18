# 📋 Manual Workflow Setup Guide

This guide provides step-by-step instructions for repository administrators to manually set up the required GitHub Actions workflows due to GitHub App permission limitations.

## 🚀 Quick Setup Checklist

- [ ] Branch protection rules configured
- [ ] Repository secrets added  
- [ ] Workflow files created from templates
- [ ] External integrations configured
- [ ] Security scanning enabled
- [ ] Deployment environments configured

---

## 1️⃣ Branch Protection Configuration

### Main Branch Protection
Navigate to **Settings → Branches → Add rule** and configure:

```yaml
Branch name pattern: main
Protection rules:
  ✅ Require a pull request before merging
    ✅ Require approvals: 2
    ✅ Dismiss stale PR approvals when new commits are pushed
    ✅ Require review from code owners
  ✅ Require status checks to pass before merging
    ✅ Require branches to be up to date before merging
    Required status checks:
      - build-and-test
      - security-scan
      - code-quality
      - performance-test
  ✅ Require conversation resolution before merging
  ✅ Require signed commits
  ✅ Require linear history
  ✅ Include administrators
  ✅ Restrict pushes that create files
```

### Development Branch Protection
For `develop` branch (if using GitFlow):

```yaml
Branch name pattern: develop
Protection rules:
  ✅ Require a pull request before merging
    ✅ Require approvals: 1
  ✅ Require status checks to pass before merging
    Required status checks:
      - build-and-test
      - security-scan
```

---

## 2️⃣ Repository Secrets Configuration

Navigate to **Settings → Secrets and variables → Actions** and add:

### Required Secrets

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `NPM_TOKEN` | NPM registry authentication | `npm_xxxxxxxxxxxx` |
| `DOCKER_USERNAME` | Docker Hub username | `your-dockerhub-user` |
| `DOCKER_PASSWORD` | Docker Hub password/token | `dckr_pat_xxxxxxxxxxxx` |
| `SONAR_TOKEN` | SonarQube/SonarCloud token | `sqp_xxxxxxxxxxxx` |
| `SNYK_TOKEN` | Snyk security scanning token | `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` |
| `CODECOV_TOKEN` | Codecov upload token | `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` |
| `SLACK_WEBHOOK_URL` | Slack notifications webhook | `https://hooks.slack.com/services/...` |

### Optional Secrets (for advanced features)

| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS deployment credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS deployment credentials |
| `GCP_SA_KEY` | Google Cloud service account key |
| `AZURE_CREDENTIALS` | Azure deployment credentials |
| `SENTRY_DSN` | Sentry error tracking DSN |
| `DATADOG_API_KEY` | DataDog monitoring API key |

---

## 3️⃣ Workflow Files Creation

Copy the template files from `docs/workflows/examples/` to `.github/workflows/`:

### Core Workflows

1. **CI Pipeline** (`ci.yml`)
   ```bash
   cp docs/workflows/examples/ci-workflow-template.yml .github/workflows/ci.yml
   ```

2. **Security Scanning** (`security.yml`)
   ```bash
   cp docs/workflows/examples/security-workflow-template.yml .github/workflows/security.yml
   ```

3. **Performance Testing** (`performance.yml`)
   ```bash
   cp docs/workflows/examples/performance-workflow-template.yml .github/workflows/performance.yml
   ```

4. **Deployment** (`deploy.yml`)
   ```bash
   cp docs/workflows/examples/deployment-workflow.yml .github/workflows/deploy.yml
   ```

5. **Dependency Updates** (`dependencies.yml`)
   ```bash
   cp docs/workflows/examples/dependency-update-workflow.yml .github/workflows/dependencies.yml
   ```

### Workflow Customization

After copying, customize each workflow:

1. **Update repository-specific values**:
   - Repository name and organization
   - Node.js and Python versions
   - Docker image names and tags
   - Deployment targets

2. **Configure environment-specific settings**:
   - Production vs staging environments
   - Database connection strings
   - API endpoints
   - Monitoring configurations

---

## 4️⃣ External Integrations Setup

### SonarQube/SonarCloud
1. Create account at [SonarCloud](https://sonarcloud.io/)
2. Import your repository
3. Generate authentication token
4. Add `SONAR_TOKEN` to repository secrets
5. Configure quality gate rules

### Snyk Security Scanning
1. Create account at [Snyk](https://snyk.io/)
2. Connect your GitHub repository
3. Generate API token
4. Add `SNYK_TOKEN` to repository secrets
5. Configure vulnerability policies

### Codecov Coverage Reporting
1. Create account at [Codecov](https://codecov.io/)
2. Connect your repository
3. Get upload token
4. Add `CODECOV_TOKEN` to repository secrets

### Docker Hub
1. Create account at [Docker Hub](https://hub.docker.com/)
2. Create repository for your project
3. Generate access token
4. Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` to secrets

---

## 5️⃣ Environment Configuration

### GitHub Environments
Navigate to **Settings → Environments** and create:

#### Production Environment
```yaml
Environment name: production
Protection rules:
  ✅ Required reviewers: 2
  ✅ Wait timer: 5 minutes
  ✅ Restrict deployments to protected branches: main
Environment secrets:
  - DATABASE_URL
  - REDIS_URL
  - API_KEYS
```

#### Staging Environment
```yaml
Environment name: staging
Protection rules:
  ✅ Required reviewers: 1
  ✅ Restrict deployments to protected branches: develop, main
Environment secrets:
  - STAGING_DATABASE_URL
  - STAGING_REDIS_URL
  - STAGING_API_KEYS
```

---

## 6️⃣ Monitoring & Alerting Setup

### GitHub Notifications
Configure in **Settings → Notifications**:
- Email notifications for workflow failures
- Slack/Teams integration for team notifications
- Mobile notifications for critical alerts

### External Monitoring
Set up integrations with:
- **DataDog**: Application performance monitoring
- **Sentry**: Error tracking and alerting
- **PagerDuty**: Incident management
- **Slack**: Team communication and alerts

---

## 7️⃣ Security Configuration

### Dependabot
Enable in **Settings → Code security and analysis**:
```yaml
Dependabot alerts: ✅ Enabled
Dependabot security updates: ✅ Enabled
Dependabot version updates: ✅ Enabled
```

Configure `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
  
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

### Code Scanning
Enable GitHub Advanced Security features:
- **Code scanning**: CodeQL analysis
- **Secret scanning**: Detect committed secrets
- **Dependency review**: PR-based dependency analysis

---

## 8️⃣ Verification & Testing

### Post-Setup Verification
1. **Create test PR** to verify all workflows trigger
2. **Check status checks** appear correctly
3. **Verify security scans** run and report results
4. **Test deployment pipeline** to staging environment
5. **Confirm notifications** are delivered to correct channels

### Troubleshooting Common Issues

#### Workflow Not Triggering
- Check workflow file syntax with [GitHub Actions validator](https://rhymelph.github.io/yaml-actions-online/)
- Verify branch protection rules are correctly configured
- Ensure required secrets are properly set

#### Failed Security Scans
- Review Snyk and CodeQL findings
- Update vulnerable dependencies
- Add security exceptions for false positives

#### Deployment Failures
- Verify environment secrets are correctly configured
- Check deployment target accessibility
- Review deployment logs for specific error messages

---

## 9️⃣ Maintenance & Updates

### Regular Maintenance Tasks
- **Weekly**: Review and merge Dependabot PRs
- **Monthly**: Update workflow actions to latest versions
- **Quarterly**: Review and update security policies
- **Annually**: Audit and rotate all secrets and tokens

### Workflow Updates
When updating workflows:
1. Test changes in a fork or feature branch
2. Use workflow dispatch for manual testing
3. Monitor first few runs after deployment
4. Document changes in CHANGELOG.md

---

## 🔗 Additional Resources

- [GitHub Actions Best Practices](https://docs.github.com/en/actions/guides)
- [Security Hardening Guide](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Repository Security Settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features)

---

## 📞 Support

If you encounter issues during setup:
1. Check the [troubleshooting section](#troubleshooting-common-issues)
2. Review GitHub Actions logs for specific error messages
3. Consult team documentation or reach out to DevOps team
4. Create an issue in the repository with detailed error information