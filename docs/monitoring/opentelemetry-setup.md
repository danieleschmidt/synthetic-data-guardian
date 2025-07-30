# OpenTelemetry Observability Stack Setup

## Overview
Comprehensive observability implementation using OpenTelemetry for metrics, traces, and logs with Prometheus, Jaeger, and Grafana integration.

## Architecture Components

### Core Stack
- **OpenTelemetry SDK**: Instrumentation and data collection
- **Prometheus**: Metrics storage and alerting
- **Jaeger**: Distributed tracing backend
- **Grafana**: Unified dashboards and visualization
- **Loki**: Log aggregation and analysis

## OpenTelemetry Configuration

### Application Instrumentation
```typescript
// src/telemetry/instrumentation.ts
import { NodeSDK } from '@opentelemetry/sdk-node';
import { HttpInstrumentation } from '@opentelemetry/instrumentation-http';
import { ExpressInstrumentation } from '@opentelemetry/instrumentation-express';
import { PrometheusExporter } from '@opentelemetry/exporter-prometheus';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';

const sdk = new NodeSDK({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'synthetic-data-guardian',
    [SemanticResourceAttributes.SERVICE_VERSION]: process.env.npm_package_version,
    [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: process.env.NODE_ENV,
  }),
  
  traceExporter: new JaegerExporter({
    endpoint: process.env.JAEGER_ENDPOINT || 'http://localhost:14268/api/traces',
  }),
  
  metricReader: new PrometheusExporter({
    port: 9464,
    endpoint: '/metrics',
  }, {
    prefix: 'synthetic_data_',
  }),
  
  instrumentations: [
    new HttpInstrumentation({
      requestHook: (span, request) => {
        span.setAttributes({
          'http.request.size': request.socket?.bytesRead || 0,
          'http.user_agent': request.headers['user-agent'],
        });
      },
    }),
    new ExpressInstrumentation({
      requestHook: (span, info) => {
        span.setAttributes({
          'express.route': info.route?.path,
          'express.method': info.request.method,
        });
      },
    }),
  ],
});

sdk.start();
```

### Custom Metrics Collection
```typescript
// src/telemetry/metrics.ts
import { metrics } from '@opentelemetry/api';
import { MeterProvider } from '@opentelemetry/sdk-metrics';

const meter = metrics.getMeter('synthetic-data-guardian', '1.0.0');

// Business metrics
export const generationRequestsCounter = meter.createCounter('generation_requests_total', {
  description: 'Total number of synthetic data generation requests',
});

export const generationDurationHistogram = meter.createHistogram('generation_duration_seconds', {
  description: 'Time taken to generate synthetic data',
  boundaries: [0.1, 0.5, 1, 2, 5, 10, 30, 60, 120],
});

export const validationScoreGauge = meter.createUpDownCounter('validation_score', {
  description: 'Current validation score for generated data',
});

export const activeConnectionsGauge = meter.createUpDownCounter('active_connections', {
  description: 'Number of active client connections',
});

// System metrics
export const memoryUsageGauge = meter.createUpDownCounter('memory_usage_bytes', {
  description: 'Memory usage in bytes',
});

export const cpuUsageGauge = meter.createUpDownCounter('cpu_usage_percent', {
  description: 'CPU usage percentage',
});
```

### Distributed Tracing
```typescript
// src/telemetry/tracing.ts
import { trace, SpanStatusCode, SpanKind } from '@opentelemetry/api';

const tracer = trace.getTracer('synthetic-data-guardian', '1.0.0');

export function traceGenerationRequest(operation: string) {
  return tracer.startSpan(`generation.${operation}`, {
    kind: SpanKind.SERVER,
    attributes: {
      'operation.type': 'generation',
      'operation.name': operation,
    },
  });
}

export function traceValidationRequest(validatorType: string) {
  return tracer.startSpan(`validation.${validatorType}`, {
    kind: SpanKind.INTERNAL,
    attributes: {
      'validation.type': validatorType,
    },
  });
}

// Usage example in middleware
export function instrumentationMiddleware(req: any, res: any, next: any) {
  const span = tracer.startSpan(`http.${req.method} ${req.route?.path || req.path}`, {
    kind: SpanKind.SERVER,
    attributes: {
      'http.method': req.method,
      'http.url': req.url,
      'http.scheme': req.protocol,
      'http.host': req.get('host'),
      'user.id': req.user?.id,
    },
  });

  res.on('finish', () => {
    span.setAttributes({
      'http.status_code': res.statusCode,
      'http.response_size': res.get('content-length') || 0,
    });
    
    if (res.statusCode >= 400) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: `HTTP ${res.statusCode}` });
    }
    
    span.end();
  });

  next();
}
```

## Infrastructure Configuration

### Docker Compose Stack
```yaml
# docker-compose.observability.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/rules:/etc/prometheus/rules
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'

  grafana:
    image: grafana/grafana:10.0.0
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
      - grafana_data:/var/lib/grafana

  jaeger:
    image: jaegertracing/all-in-one:1.47
    container_name: jaeger
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  loki:
    image: grafana/loki:2.8.0
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - ./monitoring/loki/loki-config.yml:/etc/loki/local-config.yaml
      - loki_data:/loki

  promtail:
    image: grafana/promtail:2.8.0
    container_name: promtail
    volumes:
      - ./monitoring/promtail/promtail-config.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
      - ./logs:/app/logs:ro

volumes:
  prometheus_data:
  grafana_data:
  loki_data:
```

### Prometheus Configuration
```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'synthetic-data-guardian'
    static_configs:
      - targets: ['host.docker.internal:9464']
    scrape_interval: 10s
    metrics_path: '/metrics'
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
      
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

### Alerting Rules
```yaml
# monitoring/prometheus/rules/alerts.yml
groups:
  - name: synthetic-data-guardian
    rules:
      - alert: HighErrorRate
        expr: rate(synthetic_data_generation_requests_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in synthetic data generation"
          description: "Error rate is {{ $value }} requests per second"

      - alert: SlowGeneration
        expr: histogram_quantile(0.95, rate(synthetic_data_generation_duration_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow synthetic data generation"
          description: "95th percentile latency is {{ $value }} seconds"

      - alert: LowValidationScore
        expr: synthetic_data_validation_score < 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Low validation score detected"
          description: "Validation score is {{ $value }}"

      - alert: HighMemoryUsage
        expr: synthetic_data_memory_usage_bytes / (1024 * 1024 * 1024) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB"
```

## Dashboard Configuration

### Grafana Dashboard JSON
```json
{
  "dashboard": {
    "id": null,
    "title": "Synthetic Data Guardian - Overview",
    "tags": ["synthetic-data", "observability"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Generation Requests Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(synthetic_data_generation_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": { "mode": "thresholds" },
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 10 },
                { "color": "red", "value": 50 }
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Generation Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(synthetic_data_generation_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(synthetic_data_generation_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Validation Scores",
        "type": "graph",
        "targets": [
          {
            "expr": "synthetic_data_validation_score",
            "legendFormat": "{{validator_type}}"
          }
        ]
      }
    ]
  }
}
```

## Health Checks and SLIs

### Service Level Indicators
```typescript
// src/telemetry/sli.ts
export class SLIMetrics {
  private availabilityCounter = meter.createCounter('sli_availability_total');
  private latencyHistogram = meter.createHistogram('sli_latency_seconds');
  private errorRateCounter = meter.createCounter('sli_errors_total');

  recordAvailability(isAvailable: boolean) {
    this.availabilityCounter.add(1, { available: isAvailable.toString() });
  }

  recordLatency(duration: number, operation: string) {
    this.latencyHistogram.record(duration, { operation });
  }

  recordError(errorType: string) {
    this.errorRateCounter.add(1, { error_type: errorType });
  }
}

// SLO Definitions
export const SLO_TARGETS = {
  availability: 99.9, // 99.9% uptime
  latency_p95: 2000, // 95th percentile < 2 seconds
  error_rate: 0.1,   // < 0.1% error rate
};
```

### Health Check Endpoint
```typescript
// src/routes/health.ts
import { Router } from 'express';
import { SLIMetrics } from '../telemetry/sli';

const router = Router();
const sliMetrics = new SLIMetrics();

router.get('/health', async (req, res) => {
  const startTime = Date.now();
  
  try {
    // Check database connectivity
    await checkDatabase();
    
    // Check external dependencies
    await checkDependencies();
    
    const duration = (Date.now() - startTime) / 1000;
    sliMetrics.recordLatency(duration, 'health_check');
    sliMetrics.recordAvailability(true);
    
    res.status(200).json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      checks: {
        database: 'healthy',
        dependencies: 'healthy'
      }
    });
  } catch (error) {
    const duration = (Date.now() - startTime) / 1000;
    sliMetrics.recordLatency(duration, 'health_check');
    sliMetrics.recordAvailability(false);
    sliMetrics.recordError('health_check_failed');
    
    res.status(503).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: error.message
    });
  }
});
```

## Deployment and Setup

### Installation Script
```bash
#!/bin/bash
# scripts/setup-observability.sh

echo "🔧 Setting up OpenTelemetry observability stack..."

# Install OpenTelemetry dependencies
npm install @opentelemetry/sdk-node @opentelemetry/instrumentation-http @opentelemetry/instrumentation-express \
           @opentelemetry/exporter-prometheus @opentelemetry/exporter-jaeger @opentelemetry/resources \
           @opentelemetry/semantic-conventions

# Start observability stack
docker-compose -f docker-compose.observability.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Import Grafana dashboards
curl -X POST http://admin:admin123@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboards/overview.json

echo "✅ Observability stack ready!"
echo "📊 Grafana: http://localhost:3000 (admin/admin123)"
echo "📈 Prometheus: http://localhost:9090"
echo "🔍 Jaeger: http://localhost:16686"
```

This comprehensive observability setup provides real-time monitoring, distributed tracing, and alerting for the Synthetic Data Guardian application.