# Qdrant Kubernetes Deployment

This directory contains the Kubernetes manifests for deploying Qdrant vector database.

## Files

- `qdrant-deployment.yaml` - Deployment for Qdrant
- `qdrant-service.yaml` - LoadBalancer service for external access
- `qdrant-pvc.yaml` - PersistentVolumeClaim for data storage
- `qdrant-config.yaml` - ConfigMap with Qdrant configuration

## Deployment

To deploy Qdrant to your Kubernetes cluster:

```bash
kubectl apply -f k8s/
```

Or deploy files individually in order:

```bash
kubectl apply -f k8s/qdrant-config.yaml
kubectl apply -f k8s/qdrant-pvc.yaml
kubectl apply -f k8s/qdrant-deployment.yaml
kubectl apply -f k8s/qdrant-service.yaml
```

## Accessing Qdrant

After deployment, Qdrant will be available:

- **Internal access**: `http://qdrant-service:6333` (from within the cluster)
- **External access**: Through the LoadBalancer service (get external IP with `kubectl get svc qdrant-service`)

## Configuration

- **Storage**: 10GB persistent volume
- **Resources**: 512Mi-1Gi memory, 250m-500m CPU
- **Ports**: 6333 (HTTP), 6334 (gRPC)
- **Replicas**: 1 (can be scaled by modifying the Deployment)

## Notes

- Adjust the `storageClassName` in the PVC based on your cluster's available storage classes
- The LoadBalancer service type might need to be changed to `ClusterIP` or `NodePort` depending on your cluster setup
- For production deployments, consider increasing replicas and resources