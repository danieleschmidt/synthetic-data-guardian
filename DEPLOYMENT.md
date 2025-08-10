# Synthetic Data Guardian - Production Deployment Guide

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended
- 4+ CPU cores recommended
- 50GB+ disk space

### One-Command Production Deployment

```bash
# Clone and deploy
git clone <repository-url>
cd synthetic-data-guardian
docker-compose -f docker-compose.production.yml up -d
```

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer (NGINX)                    │
│                         Port 80/443                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────────┐
│                Synthetic Data Guardian                          │
│                    Port 8080/8081                              │
└─────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┘
      │         │         │         │         │         │
┌─────┴───┐ ┌───┴────┐ ┌──┴────┐ ┌──┴────┐ ┌──┴────┐ ┌──┴────┐
│ Redis   │ │Postgres│ │Prometheus│ │Grafana│ │Fluentd│ │ Models│
│ Cache   │ │   DB   │ │Metrics │ │Monitor│ │ Logs  │ │ Storage│
└─────────┘ └────────┘ └────────┘ └───────┘ └───────┘ └───────┘
```

## ⚙️ Configuration

### Environment Variables

```bash
# Application
NODE_ENV=production
LOG_LEVEL=info
LOG_FORMAT=json
PYTHONUNBUFFERED=1

# Database
DATABASE_URL=postgresql://guardian:guardian_pass@postgres:5432/guardian_db

# Cache
REDIS_URL=redis://redis:6379

# Performance
MAX_CONCURRENT_GENERATIONS=10
MAX_MEMORY_MB=4096

# Security
SECURITY_LEVEL=strict
JWT_SECRET=your-jwt-secret-here
ENCRYPTION_KEY=your-encryption-key-here

# Features
MONITORING_ENABLED=true
CACHE_ENABLED=true
WATERMARKING_ENABLED=true
VALIDATION_ENABLED=true
```

### Production Configuration Files

1. **Prometheus** (`config/prometheus.yml`)
2. **Grafana** (`config/grafana/`)
3. **NGINX** (`config/nginx/nginx.conf`)
4. **PostgreSQL** (`config/postgresql.conf`)
5. **Redis** (`config/redis.conf`)
6. **Fluentd** (`config/fluentd/fluent.conf`)

## 🔧 Deployment Options

### Option 1: Docker Compose (Recommended)

```bash
# Production deployment
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Scale application
docker-compose -f docker-compose.production.yml up -d --scale synthetic-guardian=3
```

### Option 2: Kubernetes

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n synthetic-guardian

# Scale deployment
kubectl scale deployment synthetic-guardian --replicas=3 -n synthetic-guardian
```

### Option 3: Standalone Docker

```bash
# Build image
docker build -t synthetic-guardian:latest .

# Run container
docker run -d \
  --name synthetic-guardian \
  -p 8080:8080 \
  -e NODE_ENV=production \
  -v guardian_data:/home/synthetic_guardian/data \
  synthetic-guardian:latest
```

## 📊 Monitoring & Health Checks

### Health Endpoints

- **Application Health**: `GET /health`
- **Detailed Health**: `GET /health/detailed`
- **Metrics**: `GET /metrics` (Prometheus format)
- **Version Info**: `GET /version`

### Monitoring Stack

1. **Prometheus**: Metrics collection (Port 9090)
2. **Grafana**: Dashboards (Port 3000)
   - Default login: `admin/admin_password_change_me`
3. **Application Logs**: Structured JSON logs via Fluentd

### Key Metrics to Monitor

- Generation success rate
- Average generation time
- Memory usage
- CPU utilization
- Cache hit rate
- Error rates
- Queue depths

## 🔒 Security

### Security Features

- **Input Validation**: Comprehensive input sanitization
- **Authentication**: JWT-based authentication
- **Authorization**: Role-based access control
- **Encryption**: Data encryption at rest and in transit
- **Audit Logging**: Complete audit trail
- **Rate Limiting**: Request rate limiting
- **Security Headers**: OWASP security headers

### Security Best Practices

1. **Change Default Passwords**
   ```bash
   # Update in docker-compose.production.yml
   - GF_SECURITY_ADMIN_PASSWORD=your-secure-password
   - POSTGRES_PASSWORD=your-secure-password
   ```

2. **Use Secrets Management**
   ```bash
   # Use Docker secrets or external secret managers
   docker secret create guardian_jwt_secret jwt_secret.txt
   ```

3. **Enable SSL/TLS**
   ```nginx
   # Configure SSL in config/nginx/nginx.conf
   listen 443 ssl http2;
   ssl_certificate /etc/nginx/ssl/cert.pem;
   ssl_certificate_key /etc/nginx/ssl/key.pem;
   ```

4. **Network Security**
   ```bash
   # Use private networks
   # Restrict external access to necessary ports only
   # Enable firewall rules
   ```

## 🚨 Backup & Disaster Recovery

### Automated Backups

```bash
# Database backup
docker exec synthetic-guardian-postgres pg_dump -U guardian guardian_db > backup.sql

# Data backup
docker run --rm -v guardian_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/data-backup-$(date +%Y%m%d).tar.gz /data

# Redis backup
docker exec synthetic-guardian-redis redis-cli BGSAVE
```

### Recovery Procedures

```bash
# Database restore
docker exec -i synthetic-guardian-postgres psql -U guardian guardian_db < backup.sql

# Data restore
docker run --rm -v guardian_data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/data-backup-20240101.tar.gz -C /
```

## 📈 Performance Tuning

### Resource Optimization

1. **Memory Management**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 4G
         cpus: '2.0'
   ```

2. **Caching Configuration**
   ```bash
   # Redis optimization
   maxmemory 1gb
   maxmemory-policy allkeys-lru
   ```

3. **Database Tuning**
   ```postgresql
   # PostgreSQL optimization
   shared_buffers = 256MB
   effective_cache_size = 1GB
   work_mem = 4MB
   ```

### Scaling Strategies

1. **Horizontal Scaling**
   ```bash
   # Scale application instances
   docker-compose up -d --scale synthetic-guardian=5
   ```

2. **Vertical Scaling**
   ```yaml
   # Increase resource limits
   resources:
     limits:
       memory: 8G
       cpus: '4.0'
   ```

3. **Auto-scaling** (Kubernetes)
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: synthetic-guardian-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: synthetic-guardian
     minReplicas: 2
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
   ```

## 🔧 Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats synthetic-guardian-app
   
   # Reduce batch sizes
   export MAX_BATCH_SIZE=1000
   ```

2. **Slow Generation**
   ```bash
   # Check system resources
   docker exec synthetic-guardian-app htop
   
   # Scale up resources
   docker-compose up -d --scale synthetic-guardian=3
   ```

3. **Database Connection Issues**
   ```bash
   # Check database health
   docker exec synthetic-guardian-postgres pg_isready
   
   # Check connection string
   docker logs synthetic-guardian-app | grep DATABASE_URL
   ```

### Log Analysis

```bash
# Application logs
docker logs synthetic-guardian-app --tail=100 -f

# Database logs
docker logs synthetic-guardian-postgres --tail=100 -f

# System metrics
docker exec synthetic-guardian-app ps aux
docker exec synthetic-guardian-app free -h
```

## 📚 API Documentation

### Core Endpoints

- `POST /api/v1/generate` - Generate synthetic data
- `POST /api/v1/validate` - Validate data quality
- `GET /api/v1/pipelines` - List available pipelines
- `POST /api/v1/pipelines` - Create new pipeline
- `GET /api/v1/lineage/{id}` - Get data lineage
- `GET /api/v1/health` - Health check

### Authentication

```bash
# Get JWT token
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{"pipeline": "tabular", "records": 100}'
```

## 📞 Support

### Getting Help

1. **Documentation**: Check README.md and API docs
2. **Logs**: Review application and system logs
3. **Health Checks**: Verify all services are healthy
4. **Metrics**: Check Grafana dashboards
5. **Community**: Join our Discord/Slack community

### Maintenance Tasks

```bash
# Daily health check
docker-compose -f docker-compose.production.yml ps
docker-compose -f docker-compose.production.yml logs --tail=50

# Weekly cleanup
docker system prune -f
docker volume prune -f

# Monthly backup verification
# Test restore procedures
# Update security patches
```

---

**Note**: This is a production-ready deployment guide. Always test in a staging environment before deploying to production. Customize configurations based on your specific requirements and security policies.