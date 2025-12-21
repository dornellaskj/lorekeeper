FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including kubectl
RUN apt-get update && apt-get install -y \
    curl \
    && curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the upload service
COPY upload_service.py .

# Create directories
RUN mkdir -p /uploads /app/k8s

# Copy Kubernetes manifests
COPY k8s/ /app/k8s/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV UPLOAD_DIR=/uploads
ENV MAX_FILE_SIZE=104857600

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Run the upload service
CMD ["python", "upload_service.py"]