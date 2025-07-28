# 🔧 Manual Setup Requirements

Items requiring elevated permissions or manual configuration.

## Repository Settings

### Branch Protection (Admin Required)
- Enable branch protection on `main` branch
- Require status checks before merging
- Require up-to-date branches before merging
- Require review from code owners

### GitHub Actions (Admin Required)
- Create workflow files in `.github/workflows/`
- Configure repository secrets for external services
- Set up deployment environments and approval processes

## External Services

### Monitoring & Observability
- Set up monitoring dashboards (Prometheus/Grafana)
- Configure log aggregation and alerting
- Integrate performance monitoring tools

### Security Services
- Enable vulnerability scanning (Snyk, SAST tools)
- Configure secret scanning baselines
- Set up security incident response procedures

## Automation Setup

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit
pre-commit install
```

### Husky Git Hooks
```bash
# Already configured via package.json
npx husky install
```

## Documentation

See [Workflow Requirements](workflows/README.md) for detailed workflow setup guidance.