# Data Loader

This directory contains the data loading system for the Lorekeeper RAG application.

## Components

### Core Files
- `data_loader.py` - Main data loading script that processes documents and stores them in Qdrant
- `requirements-loader.txt` - Python dependencies for the data loader
- `Dockerfile.loader` - Container image for the data loader
- `test_loader.py` - Local testing script

### Kubernetes Manifests
- `k8s/data-loader-config.yaml` - ConfigMap with data loader settings
- `k8s/data-input-pvc.yaml` - PersistentVolumeClaim for document storage
- `k8s/data-loader-job.yaml` - Kubernetes Job to run the data loader

## Features

The data loader provides:

- **Multi-format Support**: Handles `.txt`, `.md`, `.rst`, `.py`, `.js`, `.html`, `.css`, `.json`, `.xml`, `.yaml`, `.yml` files
- **Intelligent Chunking**: Splits documents into overlapping chunks with sentence boundary detection
- **Vector Embeddings**: Uses SentenceTransformers to generate embeddings (default: `all-MiniLM-L6-v2`)
- **Batch Processing**: Efficiently processes large document collections
- **Error Handling**: Robust error handling with multiple encoding support
- **Progress Tracking**: Shows progress during processing

## Usage

### Local Testing

1. Install dependencies:
   ```bash
   pip install -r requirements-loader.txt
   ```

2. Start Qdrant locally (optional, for testing):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. Run the test script:
   ```bash
   python test_loader.py
   ```

### Kubernetes Deployment

1. **Build the Docker image**:
   ```bash
   cd dataloader
   docker build -f Dockerfile.loader -t lorekeeper/data-loader:latest .
   ```

2. **Push to your registry**:
   ```bash
   docker tag lorekeeper/data-loader:latest your-registry/lorekeeper/data-loader:latest
   docker push your-registry/lorekeeper/data-loader:latest
   ```

3. **Update the image reference** in `k8s/data-loader-job.yaml`

4. **Deploy the Qdrant database** (if not already deployed):
   ```bash
   kubectl apply -f ../k8s/qdrant-config.yaml
   kubectl apply -f ../k8s/qdrant-pvc.yaml
   kubectl apply -f ../k8s/qdrant-deployment.yaml
   kubectl apply -f ../k8s/qdrant-service.yaml
   ```

5. **Upload your documents** to the PVC. You can create a temporary pod to copy files:
   ```bash
   # Apply the PVC
   kubectl apply -f k8s/data-input-pvc.yaml
   
   # Create a temporary pod to upload documents
   kubectl run temp-pod --image=busybox --rm -it --restart=Never \
     --overrides='{"spec":{"containers":[{"name":"temp","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}],"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"data-input-pvc"}}]}}'
   
   # In another terminal, copy your documents
   kubectl cp ../DataInput temp-pod:/data/
   
   # Exit the temp pod (it will be deleted automatically)
   ```

6. **Run the data loader**:
   ```bash
   kubectl apply -f k8s/data-loader-config.yaml
   kubectl apply -f k8s/data-loader-job.yaml
   ```

7. **Monitor the job**:
   ```bash
   # Check job status
   kubectl get jobs
   
   # View logs
   kubectl logs -f job/data-loader
   
   # Clean up completed job (optional)
   kubectl delete job data-loader
   ```

## Configuration

The data loader can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `qdrant-service` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `documents` | Collection name in Qdrant |
| `DATA_FOLDER` | `/data` | Path to documents folder |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `CHUNK_SIZE` | `500` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `50` | Character overlap between chunks |

## Supported File Types

The data loader automatically detects and processes these file types:
- Text files (`.txt`)
- Markdown (`.md`)
- reStructuredText (`.rst`)
- Python (`.py`)
- JavaScript (`.js`)
- HTML (`.html`)
- CSS (`.css`)
- JSON (`.json`)
- XML (`.xml`)
- YAML (`.yaml`, `.yml`)
- Log files (`.log`)

## Document Processing

1. **File Discovery**: Recursively scans the input directory for supported file types
2. **Content Extraction**: Reads file content with multiple encoding fallbacks
3. **Chunking**: Splits content into overlapping chunks with intelligent sentence boundary detection
4. **Embedding**: Generates vector embeddings using SentenceTransformers
5. **Storage**: Uploads chunks with metadata to Qdrant vector database

Each chunk includes metadata:
- Original file path and name
- Chunk index and position
- File size
- Content boundaries

This enables precise source attribution in RAG responses.