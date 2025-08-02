# Deployment Guide

This guide covers deployment options and procedures for the Synthetic Data Guardian.

## Deployment Options

### 1. Docker Compose (Recommended for Development/Testing)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Kubernetes (Recommended for Production)

```bash
# Deploy to staging
make deploy-staging

# Deploy to production
make deploy-production

# Rollback if needed
make deploy-rollback
```

### 3. Single Docker Container

```bash
# Build image
make build-docker

# Run container
docker run -d \
  --name synthetic-guardian \
  -p 8080:8080 \
  -e DATABASE_URL="your-db-url" \
  synthetic-data-guardian:latest
```

## Environment Configuration

### Required Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_ENV` | Environment mode | `production` |
| `PORT` | Application port | `8080` |
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | Optional |
| `NEO4J_URI` | Neo4j connection string | Optional |

### Security Variables

| Variable | Description |
|----------|-------------|
| `JWT_SECRET` | JWT signing secret |
| `ENCRYPTION_KEY` | Data encryption key |
| `WATERMARK_KEY` | Watermarking secret |

### External APIs

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `STABILITY_AI_KEY` | Stability AI key |

## Health Checks

The application provides several health check endpoints:

- `/api/v1/health` - Basic health check
- `/api/v1/health/ready` - Readiness probe
- `/api/v1/health/live` - Liveness probe

## Monitoring

### Prometheus Metrics

Available at `/metrics` endpoint:

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration
- `generation_jobs_total` - Total generation jobs
- `generation_duration_seconds` - Generation duration

### Grafana Dashboards

Pre-configured dashboards available in `monitoring/grafana/dashboards/`:

- Application overview
- Performance metrics
- Error rates
- Resource usage

## Security Considerations

### Production Checklist

- [ ] Use HTTPS in production
- [ ] Set strong secrets for all keys
- [ ] Enable audit logging
- [ ] Configure proper firewall rules
- [ ] Use least privilege access
- [ ] Regular security updates
- [ ] Backup encryption keys

### SSL/TLS Configuration

```yaml
# nginx.conf example
server {
    listen 443 ssl http2;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        proxy_pass http://app:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Scaling

### Horizontal Scaling

The application is stateless and can be scaled horizontally:

```yaml
# kubernetes example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: synthetic-guardian
spec:
  replicas: 3
  selector:
    matchLabels:
      app: synthetic-guardian
  template:
    spec:
      containers:
      - name: app
        image: synthetic-guardian:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Vertical Scaling

Adjust resource limits based on workload:

- **Light workload**: 1 CPU, 2GB RAM
- **Medium workload**: 2 CPU, 4GB RAM
- **Heavy workload**: 4 CPU, 8GB RAM

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
./scripts/backup-db.sh
```

### Configuration Backup

Backup critical configuration files:
- Environment variables
- SSL certificates
- Application configuration
- Monitoring dashboards

## Troubleshooting

### Common Issues

#### Application Won't Start

1. Check environment variables
2. Verify database connectivity
3. Check application logs
4. Validate configuration files

#### High Memory Usage

1. Check for memory leaks in logs
2. Adjust Node.js heap size
3. Monitor garbage collection
4. Scale horizontally if needed

#### Performance Issues

1. Check database query performance
2. Monitor API response times
3. Verify Redis cache hit rates
4. Check resource utilization

### Diagnostic Commands

```bash
# Check container status
docker ps -a

# View application logs
docker logs synthetic-guardian -f

# Check resource usage
docker stats synthetic-guardian

# Execute commands in container
docker exec -it synthetic-guardian sh

# Check database connectivity
docker exec synthetic-guardian pg_isready -h db -p 5432
```

### Log Analysis

Important log patterns to monitor:

```bash
# Error patterns
grep "ERROR" /var/log/synthetic-guardian/app.log

# Performance patterns
grep "duration" /var/log/synthetic-guardian/app.log

# Security patterns
grep "authentication\|authorization" /var/log/synthetic-guardian/app.log
```

## Disaster Recovery

### Recovery Procedures

1. **Service Outage**
   - Check health endpoints
   - Restart failed containers
   - Scale up if needed
   - Investigate root cause

2. **Database Failure**
   - Restore from latest backup
   - Check data integrity
   - Update connection strings
   - Restart application

3. **Complete System Failure**
   - Deploy to backup environment
   - Restore all data from backups
   - Update DNS records
   - Notify stakeholders

### RTO/RPO Targets

- **Recovery Time Objective (RTO)**: 1 hour
- **Recovery Point Objective (RPO)**: 15 minutes

## CI/CD Integration

### GitHub Actions Deployment

```yaml
name: Deploy to Production
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build and Push
      run: |
        make build-docker
        make docker-push
    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f k8s/production/
```

### Deployment Strategies

1. **Blue-Green Deployment**: Zero-downtime deployments
2. **Rolling Updates**: Gradual service updates
3. **Canary Releases**: Test with subset of users

## Cost Optimization

### Resource Optimization

- Use appropriate instance sizes
- Implement auto-scaling
- Monitor and optimize database queries
- Use caching effectively
- Compress static assets

### Cloud Cost Management

- Use reserved instances for predictable workloads
- Implement cost monitoring and alerts
- Regular resource usage reviews
- Optimize data storage costs