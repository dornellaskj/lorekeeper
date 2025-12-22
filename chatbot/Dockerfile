# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY chatbot_service.py .

# Expose port
EXPOSE 8081

# Create non-root user
RUN useradd -m -u 1000 chatbot && chown -R chatbot:chatbot /app
USER chatbot

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8081/health', timeout=5)"

# Run the application
CMD ["uvicorn", "chatbot_service:app", "--host", "0.0.0.0", "--port", "8081"]