#!/bin/bash

REVISION=${1:-""}

if [ -z "$REVISION" ]; then
    echo "Usage: $0 <revision_number>"
    echo "Available revisions:"
    kubectl rollout history deployment/terragon-sdlc
    exit 1
fi

echo "🔄 Rolling back Terragon SDLC to revision $REVISION..."
kubectl rollout undo deployment/terragon-sdlc --to-revision=$REVISION

echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/terragon-sdlc --timeout=300s

echo "✅ Rollback completed successfully!"
