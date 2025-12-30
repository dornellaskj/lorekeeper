# 🤖 Lorekeeper Chatbot Service

The chatbot service provides a conversational interface to query your document collection stored in Qdrant. It uses semantic search to find relevant documents and generates contextual answers.

## ✨ Features

- **💬 Interactive Chat Interface** - Clean web UI for natural conversations
- **🔍 Semantic Search** - Uses sentence transformers for intelligent document retrieval
- **📚 Source Citations** - Shows which documents were used to generate answers
- **⚡ Fast Responses** - Optimized vector search with similarity scoring
- **📊 Query Statistics** - Response times and relevance scores
- **🎯 Configurable Results** - Adjustable similarity thresholds and result limits

## 🏗️ Architecture

```
User Question → Embedding Model → Qdrant Search → Answer Generation → Response
```

## 🚀 Quick Start

### 1. Build the Docker Image
```bash
cd chatbot
docker build -t chatbot-service:latest .
```

### 2. Import to MicroK8s
```bash
docker save chatbot-service:latest > chatbot-service-latest.tar
microk8s ctr image import chatbot-service-latest.tar
```

### 3. Deploy to Kubernetes
```bash
kubectl apply -f k8s/chatbot-deployment.yaml
kubectl apply -f k8s/chatbot-service.yaml
```

### 4. Access the Chatbot
```bash
# Get the service URL
kubectl get svc chatbot-service

# Access at: http://<EXTERNAL-IP>:8081
```

## 🔧 Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `qdrant-service` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `COLLECTION_NAME` | `documents` | Collection name to search |
| `MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `MAX_RESULTS` | `5` | Maximum search results |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum similarity score |
| `OPENAI_API_KEY` | `` | OpenAI API key (for GPT synthesis) |
| `AZURE_OPENAI_API_KEY` | `` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | `` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4` | Azure OpenAI deployment name |
| `USE_AZURE_OPENAI` | `false` | Use Azure OpenAI instead of OpenAI |
| `LLM_MODEL` | `gpt-4-1106-preview` | OpenAI model to use |

## 🤖 LLM-Powered Answer Synthesis

The chatbot now supports **Copilot-style answer synthesis** using OpenAI GPT-4 (or Azure OpenAI). Instead of just returning retrieved chunks, it:

1. **Retrieves** the top 3 most relevant document chunks from Qdrant
2. **Sends** the chunks + user question to GPT-4/Azure OpenAI
3. **Synthesizes** a coherent, natural-language answer that weaves information from all sources

### Setup Options

#### Option 1: OpenAI API
```bash
export OPENAI_API_KEY="sk-your-api-key"
```

#### Option 2: Azure OpenAI
```bash
export USE_AZURE_OPENAI="true"
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="gpt-4"
```

### Fallback Behavior
If no API key is configured, the chatbot falls back to basic rule-based answer generation.

## 📡 API Endpoints

### GET `/`
Main chat interface (HTML page)

### POST `/chat`
Chat endpoint for asking questions
```json
{
  "question": "What is the purpose of this system?",
  "max_results": 5,
  "threshold": 0.3
}
```

Response:
```json
{
  "answer": "Generated answer based on documents",
  "sources": [
    {
      "filename": "document.txt",
      "content": "Relevant text excerpt...",
      "chunk_id": 0
    }
  ],
  "query_time": 0.234,
  "similarity_scores": [0.85, 0.72, 0.68]
}
```

### GET `/health`
Health check endpoint

### GET `/stats`
Collection statistics

## 🎨 Web Interface

The chatbot provides a modern, responsive web interface with:

- **Real-time Chat** - Messages appear instantly with timestamps
- **Source Display** - Shows document sources with similarity scores
- **Performance Stats** - Query time and result count
- **Keyboard Shortcuts** - Enter to send, clear chat button
- **Mobile Friendly** - Responsive design for all devices

## 🔍 How It Works

1. **User Input** - User types a question in the chat interface
2. **Embedding** - Question is converted to vector using sentence transformers
3. **Search** - Vector similarity search in Qdrant database
4. **Filtering** - Results filtered by similarity threshold
5. **Answer Generation** - Simple extractive summarization from top results
6. **Response** - Answer displayed with source citations and stats

## 📊 Answer Generation

The current implementation uses **extractive summarization**:
- Finds sentences containing question keywords
- Returns most relevant sentences as the answer
- Falls back to context excerpt if no direct matches

### 🚀 Future Enhancements

You can easily replace the simple answer generation with:
- **OpenAI GPT** integration for better responses
- **Local LLM** (like Ollama) for privacy
- **Custom fine-tuned models** for domain-specific answers

## 🛠️ Development

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (make sure Qdrant is accessible)
python chatbot_service.py
```

### Debugging
```bash
# Check logs
kubectl logs -f deployment/chatbot-service

# Check health
curl http://localhost:8081/health

# Check stats
curl http://localhost:8081/stats
```

## 🔧 Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   kubectl get svc qdrant-service  # Check Qdrant is running
   ```

2. **No Documents Found**
   ```bash
   curl http://qdrant-service:6333/collections/documents/points/count
   ```

3. **Slow Responses**
   - Increase similarity threshold
   - Reduce max_results
   - Check Qdrant performance

## 🎯 Usage Tips

- **Be Specific** - More specific questions get better answers
- **Use Keywords** - Include important terms from your documents
- **Check Sources** - Review similarity scores to gauge answer quality
- **Iterate** - Try rephrasing if the first answer isn't helpful

## 🔒 Security

- Runs as non-root user in container
- No external API calls (fully local)
- Input validation and error handling
- Health checks and resource limits

The Lorekeeper Chatbot completes your RAG system with an intelligent, user-friendly interface for document querying! 🎉