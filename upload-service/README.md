# Lorekeeper Upload Service

A web-based file upload service that allows you to easily upload documents from your local computer to the Kubernetes cluster for processing by the data loader.

## Features

- 🌐 **Web Interface**: Clean, modern web UI accessible from any browser
- 📤 **Drag & Drop**: Simply drag files to upload or click to select
- 📊 **Progress Tracking**: Real-time upload progress and file count
- 🗂️ **Multi-format Support**: Supports all data loader compatible file types
- 📦 **ZIP Support**: Automatically extracts ZIP files
- 🚀 **Integrated Processing**: Start data loader directly from the web interface
- 📱 **Responsive**: Works on desktop and mobile devices

## Quick Start

### 1. Build and Deploy

```bash
# Build the Docker image
cd upload-service
docker build -t lorekeeper/upload-service:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/
```

### 2. Get the Service URL

```bash
# Get the external IP (LoadBalancer)
kubectl get svc upload-service

# Or use port-forward for testing
kubectl port-forward svc/upload-service 8080:8080
```

### 3. Access the Web Interface

Open your browser and go to:
- LoadBalancer: `http://<EXTERNAL-IP>:8080`
- Port-forward: `http://localhost:8080`

## Usage

1. **Upload Files**:
   - Drag and drop files onto the upload area
   - Or click "Choose Files" to select files
   - Supported: .txt, .md, .rst, .log, .py, .js, .html, .css, .json, .xml, .yaml, .yml, .zip

2. **Monitor Uploads**:
   - Watch real-time progress
   - See file count and status
   - Check for any upload errors

3. **Start Processing**:
   - Click "Start Data Processing" to trigger the data loader
   - Monitor logs with: `kubectl logs -f job/data-loader`

## Architecture

```
[Your Computer] --HTTP--> [Upload Service] --PVC--> [Data Loader] ---> [Qdrant]
```

- **Upload Service**: Web UI + API for file uploads
- **Shared PVC**: Both services access the same persistent volume
- **Data Loader**: Processes uploaded files into Qdrant vectors

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `UPLOAD_DIR` | `/uploads` | Directory for uploaded files |
| `MAX_FILE_SIZE` | `104857600` | Maximum file size (100MB) |

## API Endpoints

- `GET /` - Web interface
- `POST /upload` - Upload a file
- `GET /files/count` - Get uploaded file count
- `GET /files/list` - List all uploaded files
- `POST /trigger-loader` - Start data processing job
- `DELETE /files/clear` - Clear all uploaded files

## Security Features

- File type validation
- File size limits
- ZIP extraction safety (prevents directory traversal)
- Kubernetes RBAC for job management

## Kubernetes Resources

The service creates:
- **Deployment**: Upload service pod
- **Service**: LoadBalancer for external access
- **ServiceAccount**: For Kubernetes API access
- **RBAC**: Permissions to manage data loader jobs
- **Shared PVC**: Uses the same PVC as data loader

## Troubleshooting

### Check Service Status
```bash
kubectl get pods -l app=upload-service
kubectl logs -l app=upload-service
```

### Check External Access
```bash
kubectl get svc upload-service
```

### Test Connectivity
```bash
curl http://<SERVICE-IP>:8080/files/count
```

### Manual File Access
```bash
# List files in the PVC
kubectl exec deployment/upload-service -- ls -la /uploads
```

This upload service eliminates the need for SSH and kubectl file transfers, making it easy to upload documents from any device with a web browser! 🚀