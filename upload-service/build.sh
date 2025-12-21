#!/bin/bash

# Upload Service Build Script

set -e

echo "🚀 Building Lorekeeper Upload Service..."

# Build Docker image
echo "Building Docker image..."
docker build -t lorekeeper/upload-service:latest .

echo "Build completed successfully!"
echo ""
echo "Next steps:"
echo "1. Push to your registry: docker tag lorekeeper/upload-service:latest your-registry/lorekeeper/upload-service:latest"
echo "2. Deploy to Kubernetes: kubectl apply -f k8s/"
echo "3. Get service URL: kubectl get svc upload-service"