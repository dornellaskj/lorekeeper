#!/bin/bash

# Data Loader Build Script

set -e

echo "Building Lorekeeper Data Loader..."

# Build Docker image
echo "Building Docker image..."
docker build -f Dockerfile.loader -t lorekeeper/data-loader:latest .

echo "Build completed successfully!"
echo ""
echo "Next steps:"
echo "1. Push to your registry: docker tag lorekeeper/data-loader:latest your-registry/lorekeeper/data-loader:latest"
echo "2. Deploy to Kubernetes: kubectl apply -f k8s/"
echo "3. Monitor the job: kubectl logs -f job/data-loader"