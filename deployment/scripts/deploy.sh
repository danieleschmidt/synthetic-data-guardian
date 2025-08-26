#!/bin/bash
set -e

echo "🚀 Deploying Terragon SDLC to Production"
echo "========================================"

# Build Docker image
echo "📦 Building Docker image..."
docker build -f deployment/Dockerfile.production -t terragon-sdlc:latest .

# Deploy to Kubernetes
echo "☸️  Deploying to Kubernetes..."
kubectl apply -f deployment/kubernetes/terragon-deployment.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to complete..."
kubectl rollout status deployment/terragon-sdlc --timeout=300s

# Check deployment status
echo "🔍 Checking deployment status..."
kubectl get pods -l app=terragon-sdlc
kubectl get services terragon-sdlc-service

echo "✅ Terragon SDLC deployment completed successfully!"
echo "🌐 Application should be accessible via the LoadBalancer service"
