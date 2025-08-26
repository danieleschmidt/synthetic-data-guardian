#!/bin/bash

echo "🏥 Checking Terragon SDLC Health"
echo "================================"

# Check pod status
echo "📋 Pod Status:"
kubectl get pods -l app=terragon-sdlc

# Check service endpoints
echo "🔌 Service Endpoints:"
kubectl get endpoints terragon-sdlc-service

# Check HPA status
echo "📈 Auto-scaling Status:"
kubectl get hpa terragon-sdlc-hpa

# Check logs
echo "📄 Recent Logs:"
kubectl logs deployment/terragon-sdlc --tail=10

echo "✅ Health check completed"
