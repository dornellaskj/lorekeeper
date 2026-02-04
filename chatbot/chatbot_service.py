from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import asyncio
from datetime import datetime
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lorekeeper Chatbot", version="2.2.0")

# Version info for deployment verification
APP_VERSION = "2.2.0"
BUILD_DATE = "2026-02-04"
FEATURES = ["knowledge-graph-search", "graph-toggle-ui", "topic-clustering", "graph-first-search", "semantic-triples"]

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-service")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
GRAPH_COLLECTION = os.getenv("GRAPH_COLLECTION", "knowledge_graph")
TRIPLE_COLLECTION = os.getenv("TRIPLE_COLLECTION", "semantic_triples")
USE_KNOWLEDGE_GRAPH = os.getenv("USE_KNOWLEDGE_GRAPH", "true").lower() == "true"
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# OpenAI/Azure OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
USE_AZURE_OPENAI = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")  # Default to GPT-3.5 (widely available)

# Global variables
qdrant_client = None
embedding_model = None
openai_client = None

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = MAX_RESULTS
    threshold: Optional[float] = SIMILARITY_THRESHOLD
    use_knowledge_graph: Optional[bool] = None  # None = use server default
    search_mode: Optional[str] = None  # 'vector', 'graph', 'triples', or None for auto

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    query_time: float
    similarity_scores: List[float]
    related_topics: Optional[List[dict]] = None
    graph_enhanced: Optional[bool] = False
    search_mode: Optional[str] = None
    facts: Optional[List[dict]] = None  # For triple-based results

async def initialize_services():
    """Initialize Qdrant client, embedding model, and OpenAI client."""
    global qdrant_client, embedding_model, openai_client
    
    try:
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Test connection
        collections = qdrant_client.get_collections()
        logger.info(f"Connected to Qdrant. Available collections: {[c.name for c in collections.collections]}")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        embedding_model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded successfully")
        
        # Initialize OpenAI client
        if USE_AZURE_OPENAI and AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
            logger.info(f"Initializing Azure OpenAI client with endpoint: {AZURE_OPENAI_ENDPOINT}")
            openai_client = openai.AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version="2024-02-15-preview",
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            logger.info("Azure OpenAI client initialized successfully")
        elif OPENAI_API_KEY:
            logger.info("Initializing OpenAI client")
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")
        else:
            logger.warning("No OpenAI API key configured. LLM synthesis will be disabled.")
            openai_client = None
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("=" * 60)
    logger.info(f"🚀 LOREKEEPER CHATBOT v{APP_VERSION}")
    logger.info(f"📅 Build Date: {BUILD_DATE}")
    logger.info(f"✨ Features: {', '.join(FEATURES)}")
    logger.info("=" * 60)
    logger.info(f"Knowledge Graph Collection: {GRAPH_COLLECTION}")
    logger.info(f"Knowledge Graph Enabled (default): {USE_KNOWLEDGE_GRAPH}")
    logger.info("=" * 60)
    
    success = await initialize_services()
    if not success:
        logger.error("Failed to initialize services. Some endpoints may not work.")

@app.get("/", response_class=HTMLResponse)
async def chat_page():
    """Serve the chat interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Lorekeeper Chatbot</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                height: 80vh;
                display: flex;
                flex-direction: column;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 20px;
            }
            .chat-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-bottom: 20px;
                background: #fafafa;
            }
            .message {
                margin-bottom: 20px;
                padding: 15px;
                border-radius: 10px;
            }
            .user-message {
                background: #007bff;
                color: white;
                margin-left: 50px;
                text-align: right;
            }
            .bot-message {
                background: #e9ecef;
                color: #333;
                margin-right: 50px;
            }
            .message-time {
                font-size: 12px;
                opacity: 0.7;
                margin-top: 5px;
            }
            .sources {
                margin-top: 10px;
                padding: 10px;
                background: rgba(255,255,255,0.3);
                border-radius: 5px;
                font-size: 14px;
            }
            .source-item {
                margin: 5px 0;
                padding: 5px;
                background: rgba(255,255,255,0.5);
                border-radius: 3px;
            }
            .input-container {
                display: flex;
                gap: 10px;
            }
            #questionInput {
                flex: 1;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            #askButton {
                padding: 15px 30px;
                background: #007bff;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            #askButton:hover {
                background: #0056b3;
            }
            #askButton:disabled {
                background: #6c757d;
                cursor: not-allowed;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #666;
            }
            .stats {
                font-size: 12px;
                color: #666;
                margin-top: 5px;
            }
            .error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .clear-btn {
                background: #6c757d;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
                margin-bottom: 10px;
            }
            .clear-btn:hover {
                background: #545b62;
            }
            .toggle-container {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 15px;
                padding: 10px 15px;
                background: #f8f9fa;
                border-radius: 5px;
                border: 1px solid #e9ecef;
            }
            .toggle-label {
                font-size: 14px;
                color: #495057;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .toggle-switch {
                position: relative;
                width: 50px;
                height: 26px;
            }
            .toggle-switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }
            .toggle-slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #ccc;
                transition: 0.3s;
                border-radius: 26px;
            }
            .toggle-slider:before {
                position: absolute;
                content: "";
                height: 20px;
                width: 20px;
                left: 3px;
                bottom: 3px;
                background-color: white;
                transition: 0.3s;
                border-radius: 50%;
            }
            input:checked + .toggle-slider {
                background-color: #28a745;
            }
            input:checked + .toggle-slider:before {
                transform: translateX(24px);
            }
            .graph-badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 5px;
            }
            .badge-graph {
                background: #28a745;
                color: white;
            }
            .badge-standard {
                background: #6c757d;
                color: white;
            }
            .badge-triples {
                background: #9c27b0;
                color: white;
            }
            .related-topics {
                margin-top: 8px;
                padding: 8px;
                background: rgba(40, 167, 69, 0.1);
                border-radius: 5px;
                font-size: 13px;
            }
            .related-topics strong {
                color: #28a745;
            }
            .facts-list {
                margin-top: 8px;
                padding: 8px;
                background: rgba(156, 39, 176, 0.1);
                border-radius: 5px;
                font-size: 13px;
            }
            .facts-list strong {
                color: #9c27b0;
            }
            .fact-item {
                margin: 4px 0;
                padding: 4px 8px;
                background: rgba(255,255,255,0.5);
                border-radius: 3px;
                border-left: 3px solid #9c27b0;
            }
            .fact-subject {
                color: #d32f2f;
                font-weight: 500;
            }
            .fact-predicate {
                color: #1976d2;
                font-style: italic;
                margin: 0 4px;
            }
            .fact-object {
                color: #388e3c;
                font-weight: 500;
            }
            .mode-select {
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                background: white;
                cursor: pointer;
                min-width: 180px;
            }
            .mode-select:focus {
                outline: none;
                border-color: #007bff;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Lorekeeper Chatbot</h1>
            <p style="text-align: center; color: #666; margin-bottom: 20px;">
                Ask questions about your uploaded documents
            </p>
            
            <div class="chat-container">
                <div class="toggle-container">
                    <div class="toggle-label">
                        🔍 <span>Search Mode:</span>
                        <span id="modeBadge" class="graph-badge badge-graph">Graph</span>
                    </div>
                    <select id="searchModeSelect" class="mode-select" onchange="updateSearchMode()">
                        <option value="graph" selected>🌐 Knowledge Graph</option>
                        <option value="vector">📄 Standard Vector</option>
                        <option value="triples">🔗 Semantic Triples</option>
                    </select>
                </div>
                <button class="clear-btn" onclick="clearChat()">🗑️ Clear Chat</button>
                
                <div class="chat-messages" id="chatMessages">
                    <div class="message bot-message">
                        <strong>🤖 Lorekeeper:</strong> Hello! I'm ready to help you find information from your documents. Ask me anything!
                        <div class="message-time">Ready to chat</div>
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    🔍 Searching through your documents...
                </div>
                
                <div class="input-container">
                    <input type="text" id="questionInput" placeholder="Ask a question about your documents..." onkeypress="handleKeyPress(event)">
                    <button id="askButton" onclick="askQuestion()">Ask</button>
                </div>
            </div>
        </div>

        <script>
            const chatMessages = document.getElementById('chatMessages');
            const questionInput = document.getElementById('questionInput');
            const askButton = document.getElementById('askButton');
            const loading = document.getElementById('loading');
            const searchModeSelect = document.getElementById('searchModeSelect');
            const modeBadge = document.getElementById('modeBadge');

            function updateSearchMode() {
                const mode = searchModeSelect.value;
                if (mode === 'graph') {
                    modeBadge.textContent = 'Graph';
                    modeBadge.className = 'graph-badge badge-graph';
                } else if (mode === 'triples') {
                    modeBadge.textContent = 'Triples';
                    modeBadge.className = 'graph-badge badge-triples';
                } else {
                    modeBadge.textContent = 'Vector';
                    modeBadge.className = 'graph-badge badge-standard';
                }
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    askQuestion();
                }
            }

            async function askQuestion() {
                const question = questionInput.value.trim();
                if (!question) return;

                const searchMode = searchModeSelect.value;
                addMessage(question, 'user');
                
                // Clear input and disable button
                questionInput.value = '';
                askButton.disabled = true;
                loading.style.display = 'block';
                
                const loadingMessages = {
                    'graph': '🌐 Searching knowledge graph...',
                    'vector': '📄 Searching documents...',
                    'triples': '🔗 Querying semantic triples...'
                };
                loading.textContent = loadingMessages[searchMode] || '🔍 Searching...';

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ 
                            question: question,
                            search_mode: searchMode
                        })
                    });

                    const data = await response.json();
                    
                    if (response.ok) {
                        addBotResponse(data);
                    } else {
                        addMessage(`Error: ${data.detail}`, 'bot', true);
                    }
                } catch (error) {
                    addMessage(`Error: Failed to get response - ${error.message}`, 'bot', true);
                } finally {
                    askButton.disabled = false;
                    loading.style.display = 'none';
                    questionInput.focus();
                }
            }

            function addMessage(content, sender, isError = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message ${isError ? 'error' : ''}`;
                
                const time = new Date().toLocaleTimeString();
                const senderIcon = sender === 'user' ? '👤' : '🤖';
                const senderName = sender === 'user' ? 'You' : 'Lorekeeper';
                
                messageDiv.innerHTML = `
                    <strong>${senderIcon} ${senderName}:</strong> ${content}
                    <div class="message-time">${time}</div>
                `;
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function addBotResponse(data) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message bot-message';
                
                const time = new Date().toLocaleTimeString();
                const searchMode = data.search_mode || (data.graph_enhanced ? 'graph' : 'vector');
                
                let searchBadge;
                if (searchMode === 'graph') {
                    searchBadge = '<span class="graph-badge badge-graph">Graph</span>';
                } else if (searchMode === 'triples') {
                    searchBadge = '<span class="graph-badge badge-triples">Triples</span>';
                } else {
                    searchBadge = '<span class="graph-badge badge-standard">Vector</span>';
                }
                
                let relatedTopicsHtml = '';
                if (data.related_topics && data.related_topics.length > 0) {
                    const topicNames = data.related_topics.map(t => t.label).join(', ');
                    relatedTopicsHtml = `
                        <div class="related-topics">
                            <strong>🏷️ Related Topics:</strong> ${topicNames}
                        </div>
                    `;
                }
                
                // Display facts for triple search mode
                let factsHtml = '';
                if (data.facts && data.facts.length > 0) {
                    factsHtml = '<div class="facts-list"><strong>🔗 Semantic Facts:</strong>';
                    data.facts.slice(0, 10).forEach(fact => {
                        factsHtml += `
                            <div class="fact-item">
                                <span class="fact-subject">${fact.subject}</span>
                                <span class="fact-predicate">${fact.predicate}</span>
                                <span class="fact-object">${fact.object}</span>
                            </div>
                        `;
                    });
                    if (data.facts.length > 10) {
                        factsHtml += `<div class="fact-item"><em>...and ${data.facts.length - 10} more facts</em></div>`;
                    }
                    factsHtml += '</div>';
                }
                
                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = '<div class="sources"><strong>📚 Sources:</strong>';
                    data.sources.forEach((source, index) => {
                        const score = data.similarity_scores[index];
                        sourcesHtml += `
                            <div class="source-item">
                                📄 <strong>${source.filename}</strong> (similarity: ${score.toFixed(3)})
                                <br><small>${source.content.substring(0, 100)}...</small>
                            </div>
                        `;
                    });
                    sourcesHtml += '</div>';
                }
                
                messageDiv.innerHTML = `
                    <strong>🤖 Lorekeeper:</strong> ${searchBadge} ${data.answer}
                    ${relatedTopicsHtml}
                    ${factsHtml}
                    ${sourcesHtml}
                    <div class="stats">Query time: ${data.query_time.toFixed(2)}s | Found ${data.sources.length} relevant sources</div>
                    <div class="message-time">${time}</div>
                `;
                
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function clearChat() {
                chatMessages.innerHTML = `
                    <div class="message bot-message">
                        <strong>🤖 Lorekeeper:</strong> Chat cleared! Ask me anything about your documents.
                        <div class="message-time">Ready to chat</div>
                    </div>
                `;
            }

            // Focus on input when page loads
            window.onload = function() {
                questionInput.focus();
            };
        </script>
    </body>
    </html>
    """
    return html_content


@app.get("/graph/topics")
async def get_topics():
    """Get all topic clusters from the knowledge graph."""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Services not initialized.")
    
    try:
        collections = qdrant_client.get_collections()
        if GRAPH_COLLECTION not in [c.name for c in collections.collections]:
            return {"error": "Knowledge graph not found", "topics": []}
        
        topics = []
        offset = None
        
        while True:
            results, next_offset = qdrant_client.scroll(
                collection_name=GRAPH_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="node_type", match=MatchValue(value="cluster_topic"))]
                ),
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not results:
                break
            
            for point in results:
                topics.append({
                    'cluster_id': point.payload.get('cluster_id'),
                    'label': point.payload.get('label'),
                    'num_chunks': point.payload.get('num_chunks'),
                    'num_documents': point.payload.get('num_documents'),
                    'documents': point.payload.get('document_titles', [])[:10],
                    'sample_content': point.payload.get('sample_content', [])[:2]
                })
            
            if next_offset is None:
                break
            offset = next_offset
        
        return {"topics": topics, "count": len(topics)}
        
    except Exception as e:
        logger.error(f"Error fetching topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/stats")
async def get_graph_stats():
    """Get statistics about the knowledge graph."""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Services not initialized.")
    
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        stats = {
            'documents_collection': COLLECTION_NAME in collection_names,
            'graph_collection': GRAPH_COLLECTION in collection_names,
            'graph_enabled': USE_KNOWLEDGE_GRAPH
        }
        
        if COLLECTION_NAME in collection_names:
            doc_info = qdrant_client.get_collection(COLLECTION_NAME)
            stats['total_chunks'] = doc_info.points_count
        
        if GRAPH_COLLECTION in collection_names:
            graph_info = qdrant_client.get_collection(GRAPH_COLLECTION)
            stats['total_graph_nodes'] = graph_info.points_count
            
            # Count node types
            for node_type in ['cluster_topic', 'document', 'edge']:
                results, _ = qdrant_client.scroll(
                    collection_name=GRAPH_COLLECTION,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="node_type", match=MatchValue(value=node_type))]
                    ),
                    limit=1,
                    with_payload=False,
                    with_vectors=False
                )
                # Get count by scrolling (simplified - for accurate count would need aggregation)
                count_results, _ = qdrant_client.scroll(
                    collection_name=GRAPH_COLLECTION,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="node_type", match=MatchValue(value=node_type))]
                    ),
                    limit=10000,
                    with_payload=False,
                    with_vectors=False
                )
                stats[f'{node_type}_count'] = len(count_results)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/triples/stats")
async def get_triple_stats():
    """Get statistics about the semantic triple store."""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Services not initialized.")
    
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        stats = {
            'triple_collection_exists': TRIPLE_COLLECTION in collection_names,
            'triple_collection_name': TRIPLE_COLLECTION,
            'total_triples': 0,
            'triple_types': {}
        }
        
        if TRIPLE_COLLECTION in collection_names:
            triple_info = qdrant_client.get_collection(TRIPLE_COLLECTION)
            stats['total_triples'] = triple_info.points_count
            
            # Count by triple type
            for triple_type in ['definition', 'property', 'action', 'relationship', 'hierarchy', 'causation', 'location']:
                count_results, _ = qdrant_client.scroll(
                    collection_name=TRIPLE_COLLECTION,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="triple_type", match=MatchValue(value=triple_type))]
                    ),
                    limit=10000,
                    with_payload=False,
                    with_vectors=False
                )
                if count_results:
                    stats['triple_types'][triple_type] = len(count_results)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching triple stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def search_with_triples(question_embedding: list, max_results: int, threshold: float) -> dict:
    """
    Search using semantic triples for atomic fact retrieval.
    
    Returns structured facts from the triple store.
    """
    results = {
        'facts': [],
        'entities': set(),
        'sources': set(),
        'chunks': [],  # We'll convert facts to chunk-like format for compatibility
        'triple_enhanced': False
    }
    
    try:
        # Check if triple collection exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if TRIPLE_COLLECTION not in collection_names:
            logger.warning(f"Triple collection '{TRIPLE_COLLECTION}' not found.")
            return results
        
        results['triple_enhanced'] = True
        
        # Search for relevant triples
        triple_results = qdrant_client.search(
            collection_name=TRIPLE_COLLECTION,
            query_vector=question_embedding,
            limit=max_results * 3,
            score_threshold=threshold
        )
        
        logger.info(f"Triple search returned {len(triple_results)} results")
        
        for r in triple_results:
            payload = r.payload
            
            # Format as readable fact
            predicate_text = payload.get('predicate', '').replace('_', ' ')
            subject = payload.get('subject', '')
            obj = payload.get('object', '')
            
            fact_text = f"{subject} {predicate_text} {obj}"
            
            results['facts'].append({
                'fact': fact_text,
                'subject': subject,
                'predicate': payload.get('predicate', ''),
                'object': obj,
                'type': payload.get('triple_type', ''),
                'confidence': payload.get('confidence', 0.5),
                'score': r.score,
                'source': payload.get('source_file', ''),
                'source_text': payload.get('source_text', '')
            })
            
            results['entities'].add(subject)
            results['entities'].add(obj)
            results['sources'].add(payload.get('source_file', ''))
        
        # Sort by combined score
        results['facts'].sort(key=lambda x: x['score'] * x['confidence'], reverse=True)
        results['facts'] = results['facts'][:max_results * 2]
        
        # Get related triples for top entities (graph expansion)
        if results['facts']:
            top_entity = results['facts'][0]['subject']
            related = await _get_related_triples(top_entity, limit=3)
            results['related_facts'] = related
        
        # Convert to list for JSON serialization
        results['entities'] = list(results['entities'])
        results['sources'] = list(results['sources'])
        
        logger.info(f"Triple search found {len(results['facts'])} facts from {len(results['sources'])} sources")
        
    except Exception as e:
        logger.error(f"Error in triple search: {e}")
    
    return results


async def _get_related_triples(entity: str, limit: int = 5) -> List[dict]:
    """Get triples related to an entity."""
    related = []
    
    try:
        # Find triples where entity is subject
        results, _ = qdrant_client.scroll(
            collection_name=TRIPLE_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="subject", match=MatchValue(value=entity))]
            ),
            limit=limit,
            with_payload=True
        )
        
        for r in results:
            predicate_text = r.payload.get('predicate', '').replace('_', ' ')
            related.append({
                'fact': f"{r.payload.get('subject', '')} {predicate_text} {r.payload.get('object', '')}",
                'type': r.payload.get('triple_type', '')
            })
        
        # Find triples where entity is object
        results, _ = qdrant_client.scroll(
            collection_name=TRIPLE_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="object", match=MatchValue(value=entity))]
            ),
            limit=limit,
            with_payload=True
        )
        
        for r in results:
            predicate_text = r.payload.get('predicate', '').replace('_', ' ')
            related.append({
                'fact': f"{r.payload.get('subject', '')} {predicate_text} {r.payload.get('object', '')}",
                'type': r.payload.get('triple_type', '')
            })
    
    except Exception as e:
        logger.warning(f"Error getting related triples: {e}")
    
    return related[:limit]


async def search_with_knowledge_graph(question_embedding: list, max_results: int, threshold: float) -> dict:
    """
    Enhanced search using knowledge graph for better context retrieval.
    
    Strategy (Graph-First Approach):
    1. Search cluster topics to identify relevant topic areas
    2. Find related documents via graph (document nodes + edges)
    3. Retrieve chunks ONLY from those identified documents
    4. This gives more focused, topic-coherent results vs general vector search
    """
    results = {
        'chunks': [],
        'related_topics': [],
        'related_documents': [],
        'graph_enhanced': False
    }
    
    try:
        # Check if knowledge_graph collection exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if GRAPH_COLLECTION not in collection_names:
            logger.warning(f"Knowledge graph collection '{GRAPH_COLLECTION}' not found. Using standard search.")
            return results
        
        results['graph_enhanced'] = True
        
        # 1. Find relevant cluster topics
        topic_results = qdrant_client.search(
            collection_name=GRAPH_COLLECTION,
            query_vector=question_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value="cluster_topic"))]
            ),
            limit=3,
            score_threshold=0.25
        )
        
        relevant_cluster_ids = []
        cluster_doc_names = set()
        for topic in topic_results:
            cluster_id = topic.payload.get('cluster_id')
            if cluster_id is not None:
                relevant_cluster_ids.append(cluster_id)
                # Get documents from this cluster
                for doc in topic.payload.get('documents', []):
                    cluster_doc_names.add(doc)
                results['related_topics'].append({
                    'cluster_id': cluster_id,
                    'label': topic.payload.get('label', f'Topic {cluster_id}'),
                    'documents': topic.payload.get('document_titles', [])[:5],
                    'score': topic.score
                })
        
        logger.info(f"Found {len(relevant_cluster_ids)} relevant topic clusters: {relevant_cluster_ids}")
        logger.info(f"Cluster documents: {list(cluster_doc_names)[:5]}")
        
        # 2. Find related documents from graph (document nodes)
        doc_results = qdrant_client.search(
            collection_name=GRAPH_COLLECTION,
            query_vector=question_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value="document"))]
            ),
            limit=5,
            score_threshold=0.3
        )
        
        relevant_doc_names = set()
        for doc in doc_results:
            doc_name = doc.payload.get('file_name', '')
            if doc_name:
                relevant_doc_names.add(doc_name)
                results['related_documents'].append({
                    'file_name': doc_name,
                    'title': doc.payload.get('document_title', doc_name),
                    'cluster_id': doc.payload.get('primary_cluster', -1),
                    'score': doc.score
                })
        
        # 3. Find connected documents via edges (expand our document set)
        edge_results = qdrant_client.search(
            collection_name=GRAPH_COLLECTION,
            query_vector=question_embedding,
            query_filter=Filter(
                must=[FieldCondition(key="node_type", match=MatchValue(value="edge"))]
            ),
            limit=5,
            score_threshold=0.3
        )
        
        for edge in edge_results:
            # Add both connected documents
            source_file = edge.payload.get('source_file', '')
            target_file = edge.payload.get('target_file', '')
            if source_file:
                relevant_doc_names.add(source_file)
            if target_file:
                relevant_doc_names.add(target_file)
        
        # Combine with cluster documents
        all_relevant_docs = relevant_doc_names.union(cluster_doc_names)
        logger.info(f"Graph identified {len(all_relevant_docs)} relevant documents: {list(all_relevant_docs)[:5]}")
        
        # 4. Search chunks ONLY from graph-identified documents
        if all_relevant_docs:
            for doc_name in list(all_relevant_docs)[:8]:  # Limit to top 8 docs
                try:
                    doc_chunks = qdrant_client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=question_embedding,
                        query_filter=Filter(
                            must=[FieldCondition(key="file_name", match=MatchValue(value=doc_name))]
                        ),
                        limit=3,
                        score_threshold=threshold
                    )
                    results['chunks'].extend(doc_chunks)
                    logger.info(f"  Document '{doc_name}': found {len(doc_chunks)} chunks")
                except Exception as e:
                    logger.warning(f"Error searching document {doc_name}: {e}")
        
        # Deduplicate chunks by ID
        seen_ids = set()
        unique_chunks = []
        for chunk in results['chunks']:
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                unique_chunks.append(chunk)
        results['chunks'] = unique_chunks
        
        # Sort by score
        results['chunks'].sort(key=lambda x: x.score, reverse=True)
        
        # Trim to requested limit
        results['chunks'] = results['chunks'][:max_results * 2]
        
        logger.info(f"Graph-enhanced search returned {len(results['chunks'])} unique chunks from graph-identified documents")
        
    except Exception as e:
        logger.error(f"Error in graph-enhanced search: {e}")
        # Fall back to empty results, will use standard search
    
    return results


@app.post("/chat")
async def chat(request: QueryRequest):
    """Process a chat question and return an answer with sources."""
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Services not initialized. Please try again later.")
    
    start_time = datetime.now()
    
    try:
        # Generate embedding for the question
        question_embedding = embedding_model.encode(request.question).tolist()
        
        # Determine search mode
        # Priority: search_mode > use_knowledge_graph > server default
        search_mode = request.search_mode
        if search_mode is None:
            if request.use_knowledge_graph is not None:
                search_mode = 'graph' if request.use_knowledge_graph else 'vector'
            else:
                search_mode = 'graph' if USE_KNOWLEDGE_GRAPH else 'vector'
        
        logger.info(f"Using search mode: {search_mode}")
        
        # Initialize result variables
        search_results = []
        related_topics = None
        graph_enhanced = False
        triple_results = None
        facts = []
        
        if search_mode == 'triples':
            # Use semantic triple search
            logger.info("Using semantic triple search")
            triple_results = await search_with_triples(
                question_embedding,
                request.max_results,
                request.threshold
            )
            
            facts = triple_results.get('facts', [])
            
            # If we have triples with sources, get the actual document chunks
            if triple_results.get('sources'):
                source_files = list(triple_results['sources'])
                logger.info(f"Triple search found sources: {source_files}")
                
                # Search for chunks from the source files identified by triples
                search_results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=question_embedding,
                    limit=request.max_results * 2,
                    query_filter=Filter(
                        should=[
                            FieldCondition(key="file_name", match=MatchValue(value=src))
                            for src in source_files[:5]
                        ]
                    ),
                    score_threshold=max(request.threshold - 0.15, 0.15)
                )
                logger.info(f"Found {len(search_results)} chunks from triple sources")
            
            # Fallback to regular search if no chunks found
            if not search_results:
                search_results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=question_embedding,
                    limit=request.max_results * 2,
                    score_threshold=max(request.threshold - 0.1, 0.2)
                )
        
        elif search_mode == 'graph':
            # Use knowledge graph enhanced search
            logger.info("Using knowledge graph enhanced search")
            graph_results = await search_with_knowledge_graph(
                question_embedding, 
                request.max_results, 
                request.threshold
            )
            graph_enhanced = graph_results.get('graph_enhanced', False)
            related_topics = graph_results.get('related_topics', [])
            
            if graph_results['chunks']:
                search_results = graph_results['chunks']
                logger.info(f"Using graph-enhanced results: {len(search_results)} chunks")
            else:
                # Fall back to standard search
                logger.info("Graph search returned no results, falling back to standard search")
                search_results = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=question_embedding,
                    limit=min(request.max_results * 2, 10),
                    score_threshold=max(request.threshold - 0.1, 0.2)
                )
        
        else:
            # Standard vector search
            logger.info("Using standard vector search")
            search_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=question_embedding,
                limit=min(request.max_results * 2, 10),
                score_threshold=max(request.threshold - 0.1, 0.2)
            )
        
        # Filter and diversify results
        filtered_results = []
        seen_content = set()
        
        for result in search_results:
            content = result.payload.get("content", "").strip()
            
            # Skip very similar content (simple deduplication)
            content_start = content[:100].lower()
            if content_start not in seen_content:
                seen_content.add(content_start)
                filtered_results.append(result)
                
                # Stop when we have enough diverse results
                if len(filtered_results) >= request.max_results:
                    break
        
        search_results = filtered_results
        
        if not search_results:
            return ChatResponse(
                answer="I couldn't find any relevant information in your documents to answer that question. Try rephrasing your question or asking about different topics.",
                sources=[],
                query_time=(datetime.now() - start_time).total_seconds(),
                similarity_scores=[]
            )
        
        # Extract sources and create answer
        sources = []
        similarity_scores = []
        context_parts = []
        
        for result in search_results:
            payload = result.payload
            # Get document title (clean filename) if available, otherwise use filename
            doc_title = payload.get("document_title", payload.get("file_name", "Unknown"))
            content = payload.get("content", "")
            
            sources.append({
                "filename": payload.get("file_name", "Unknown"),
                "document_title": doc_title,
                "content": content,
                "chunk_id": payload.get("chunk_index", 0)
            })
            similarity_scores.append(result.score)
            # Include document title in context for better LLM synthesis
            context_parts.append(f"[From document: {doc_title}]\n{content}")
        
        # Use top 3 chunks for LLM synthesis
        top_chunks = context_parts[:3]
        
        # Debug: Print what we're working with
        print(f"\nDEBUG - Question: {request.question}")
        print(f"DEBUG - Search mode: {search_mode}")
        print(f"DEBUG - Top chunks count: {len(top_chunks)}")
        for i, chunk in enumerate(top_chunks):
            print(f"DEBUG - Chunk {i} (score: {similarity_scores[i]:.3f}): {chunk[:150]}...")
        
        # Add topic context to LLM prompt if we have related topics
        topic_context = ""
        if related_topics:
            topic_names = [t['label'] for t in related_topics[:3]]
            topic_context = f"\n\nRelated topic areas: {', '.join(topic_names)}"
        
        # Add semantic facts to context if we have triples
        fact_context = ""
        if facts:
            fact_text = [f"- {f['subject']} {f['predicate'].replace('_', ' ')} {f['object']}" for f in facts[:8]]
            fact_context = f"\n\nKnown facts from the knowledge base:\n" + "\n".join(fact_text)
        
        # Generate answer using LLM synthesis (Copilot-style)
        answer = await synthesize_answer_with_llm(request.question, top_chunks, topic_context + fact_context)
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        # Format facts for response
        response_facts = [
            {'subject': f['subject'], 'predicate': f['predicate'], 'object': f['object']}
            for f in facts[:15]
        ] if facts else None
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            query_time=query_time,
            similarity_scores=similarity_scores,
            related_topics=related_topics,
            graph_enhanced=graph_enhanced,
            search_mode=search_mode,
            facts=response_facts
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


async def synthesize_answer_with_llm(question: str, top_chunks: List[str], topic_context: str = "") -> str:
    """
    Use OpenAI/Azure OpenAI (Copilot) to synthesize an answer from the top chunks.
    This provides Copilot-style, coherent answers that weave together information from multiple sources.
    """
    global openai_client
    
    if not openai_client:
        logger.warning("OpenAI client not initialized. Falling back to basic answer generation.")
        return generate_fallback_answer(question, top_chunks)
    
    # Build context from top chunks
    context = "\n\n---\n\n".join([f"Source {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])
    
    # Add topic context if available
    if topic_context:
        context += f"\n\n---{topic_context}"
    
    system_prompt = """You are Lorekeeper, an expert assistant that helps users find and understand information from their documents. 

Your task is to synthesize a clear, helpful answer based on the provided context. Follow these guidelines:
- Combine information from all provided sources into a coherent, well-structured response
- Be concise but comprehensive - include all relevant details
- Use natural language - don't just copy text, synthesize and explain
- If the sources contain conflicting information, acknowledge it
- If the context doesn't fully answer the question, say what you can answer and what's missing
- Format your response with proper punctuation and structure
- Do not make up information not present in the sources"""

    user_prompt = f"""Based on the following context from the user's documents, please answer their question.

Context:
{context}

Question: {question}

Please provide a clear, synthesized answer:"""

    try:
        if USE_AZURE_OPENAI:
            response = openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=0.3,
            )
        else:
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1024,
                temperature=0.3,
            )
        
        answer = response.choices[0].message.content.strip()
        logger.info(f"LLM synthesis successful. Response length: {len(answer)}")
        return answer
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        logger.info("Falling back to basic answer generation.")
        return generate_fallback_answer(question, top_chunks)


def generate_fallback_answer(question: str, top_chunks: List[str]) -> str:
    """Fallback answer generation when LLM is not available."""
    if not top_chunks:
        return "I couldn't find relevant information to answer your question."
    
    # Simple fallback: return the most relevant chunk with context
    context = "\n\n".join(top_chunks)
    return generate_answer(question, context, [])


def generate_answer(question: str, context: str, search_results: list) -> str:
    """Generate an enhanced answer based on the question and context.
    This implementation provides more nuanced, analytical responses."""
    
    if not context.strip():
        return "I couldn't find relevant information to answer your question."
    
    question_lower = question.lower()
    
    # Parse markdown structure to understand sections
    def parse_markdown_sections(text):
        lines = text.split('\n')
        sections = []
        current_section = {'level': 0, 'title': '', 'content': []}
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Save previous section if it has content
                if current_section['title'] or current_section['content']:
                    sections.append(current_section)
                
                # Start new section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                current_section = {'level': level, 'title': title, 'content': []}
            elif line and not line.startswith('#'):
                current_section['content'].append(line)
        
        # Add final section
        if current_section['title'] or current_section['content']:
            sections.append(current_section)
        
        return sections
    
    sections = parse_markdown_sections(context)
    
    print(f"DEBUG - Parsed {len(sections)} sections:")
    for i, section in enumerate(sections):
        print(f"  Section {i}: '{section['title']}' (level {section['level']}) - {len(section['content'])} lines")
    
    # Extract keywords from question
    question_keywords = [word.lower() for word in question_lower.split() 
                        if len(word) > 2 and word not in [
                            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'tell', 'me',
                            'about', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                            'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did',
                            'has', 'have', 'had', 'this', 'that', 'these', 'those', 'a', 'an'
                        ]]
    
    print(f"DEBUG - Question keywords: {question_keywords}")
    
    # Enhanced response strategies based on question type
    
    # Strategy 1: Seasonal/temporal questions (high priority)
    if any(word in question_lower for word in ['seasonal', 'seasonally', 'season', 'spring', 'summer', 'fall', 'winter', 'holiday', 'timing']):
        return generate_seasonal_answer(sections, question_keywords, question)
    
    # Strategy 2: Categorical/types questions  
    elif any(word in question_lower for word in ['different', 'types', 'kinds', 'categories', 'compare', 'versus', 'options']):
        return generate_analytical_answer(sections, question_keywords, question)
        
    # Strategy 3: Retailer needs/wants questions
    elif any(word in question_lower for word in ['want', 'wants', 'need', 'needs', 'require', 'looking for', 'seeking']):
        return generate_needs_answer(sections, question_keywords, question)
        
    # Strategy 4: Challenge/problem questions
    elif any(word in question_lower for word in ['challenges', 'problems', 'issues', 'difficulties', 'pain points']):
        return generate_challenge_answer(sections, question_keywords, question)
        
    # Strategy 4: Comprehensive overview questions
    elif any(word in question_lower for word in ['about', 'overview', 'tell me', 'explain', 'describe', 'what are']):
        return generate_comprehensive_answer(sections, question_keywords, question)
    
    # Strategy 5: Specific detail questions  
    elif any(word in question_lower for word in ['how', 'why', 'when', 'which', 'specific']):
        return generate_detailed_answer(sections, question_keywords, question)
        
    # Strategy 6: Yes/No questions with elaboration
    elif any(indicator in question_lower.split()[:3] for indicator in ['is', 'are', 'does', 'do', 'can', 'will']):
        return generate_yesno_answer(sections, question_keywords, question)
    
    # Default: Contextual synthesis
    else:
        return generate_contextual_answer(sections, question_keywords, question)

def generate_seasonal_answer(sections, keywords, question):
    """Generate answer focused on seasonal/temporal aspects."""
    seasonal_sections = []
    
    for section in sections:
        title_lower = section['title'].lower()
        content = ' '.join(section['content']).lower()
        
        # Look for seasonal content
        seasonal_terms = ['seasonal', 'spring', 'summer', 'fall', 'winter', 'holiday', 'timing', 'calendar']
        seasonal_score = sum(1 for term in seasonal_terms if term in title_lower or term in content)
        
        if seasonal_score > 0:
            keyword_matches = sum(1 for keyword in keywords if keyword in f"{title_lower} {content}")
            seasonal_sections.append({
                'section': section,
                'score': keyword_matches + seasonal_score,
                'content': ' '.join(section['content']).strip()
            })
    
    if not seasonal_sections:
        return "I don't have specific seasonal information available in the current documents."
    
    # Sort by relevance
    seasonal_sections.sort(key=lambda x: x['score'], reverse=True)
    
    # Extract seasonal information from the best matching section
    main_section = seasonal_sections[0]
    content = main_section['content']
    
    # Look for structured seasonal content
    lines = content.split('\n')
    seasonal_info = []
    
    current_category = None
    for line in lines:
        line = line.strip()
        
        # Check for seasonal section headers
        if any(term in line.lower() for term in ['spring', 'summer', 'fall', 'winter', 'seasonal']):
            if line.startswith('###') or line.startswith('##'):
                current_category = line.replace('#', '').strip()
            elif line.startswith('- **') and ':' in line:
                # Extract bullet point information
                clean_line = line.replace('- **', '').replace('**:', ':')
                if current_category:
                    seasonal_info.append(f"{current_category}: {clean_line}")
                else:
                    seasonal_info.append(clean_line)
    
    if seasonal_info:
        # Format the seasonal information nicely
        if len(seasonal_info) <= 3:
            response = f"Seasonal operations include: {'; '.join(seasonal_info)}."
        else:
            response = f"Key seasonal aspects: {'; '.join(seasonal_info[:3])} (and {len(seasonal_info)-3} more considerations)."
    else:
        # Fallback to sentences containing seasonal terms
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        seasonal_sentences = []
        
        for sentence in sentences:
            if any(term in sentence.lower() for term in ['spring', 'summer', 'fall', 'winter', 'seasonal', 'holiday']):
                seasonal_sentences.append(sentence)
        
        if seasonal_sentences:
            response = f"Regarding seasonal operations: {'. '.join(seasonal_sentences[:2])}."
        else:
            response = f"From {main_section['section']['title']}: {sentences[0] if sentences else content}."
    
    return response

def generate_needs_answer(sections, keywords, question):
    """Generate answer focused on what retailers want/need."""
    needs_sections = []
    
    for section in sections:
        title_lower = section['title'].lower()
        content = ' '.join(section['content']).lower()
        
        # Look for needs/wants related content
        needs_terms = ['want', 'need', 'goal', 'objective', 'motivation', 'desire', 'require', 'seek']
        needs_score = sum(1 for term in needs_terms if term in title_lower or term in content)
        
        if needs_score > 0 or 'need' in title_lower or 'want' in title_lower:
            keyword_matches = sum(1 for keyword in keywords if keyword in f"{title_lower} {content}")
            needs_sections.append({
                'section': section,
                'score': keyword_matches + needs_score,
                'content': ' '.join(section['content']).strip()
            })
    
    if not needs_sections:
        return "I don't have specific information about retailer needs and wants in the current documents."
    
    # Sort by relevance
    needs_sections.sort(key=lambda x: x['score'], reverse=True)
    
    # Extract needs-focused information
    main_section = needs_sections[0]
    content = main_section['content']
    
    # Look for goal/need oriented sentences
    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
    need_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(term in sentence_lower for term in ['want', 'need', 'goal', 'seek', 'require', 'desire', 'motivation']):
            need_sentences.append(sentence)
    
    if need_sentences:
        response = f"Retailer needs and goals: {'. '.join(need_sentences[:2])}."
    else:
        # Look for any section about retailer objectives
        response = f"From {main_section['section']['title']}: {sentences[0] if sentences else content}."
    
    return response

def generate_challenge_answer(sections, keywords, question):
    """Generate answer focused on challenges and problems."""
    challenge_sections = []
    
    for section in sections:
        title_lower = section['title'].lower()
        content = ' '.join(section['content']).lower()
        
        # Look for challenge/problem content
        challenge_terms = ['challenge', 'problem', 'issue', 'difficulty', 'pain point', 'barrier', 'obstacle']
        if any(term in title_lower or term in content for term in challenge_terms):
            keyword_matches = sum(1 for keyword in keywords if keyword in f"{title_lower} {content}")
            challenge_sections.append({
                'section': section,
                'score': keyword_matches + 2,  # Bonus for challenge relevance
                'content': ' '.join(section['content'])
            })
    
    if not challenge_sections:
        return "I don't have specific information about challenges in the current documents."
    
    # Sort by relevance
    challenge_sections.sort(key=lambda x: x['score'], reverse=True)
    
    # Build challenge-focused response
    main_section = challenge_sections[0]
    content = main_section['content']
    
    # Extract bullet points about challenges
    lines = content.split('\n')
    challenge_points = []
    
    for line in lines:
        line_lower = line.lower()
        if any(term in line_lower for term in ['challenge', 'problem', 'pain point', 'difficulty']) and \
           any(marker in line for marker in ['- ', '• ', '* ', '-']):
            clean_line = line.strip().lstrip('- •*').strip()
            challenge_points.append(clean_line)
    
    if challenge_points:
        response = f"Key challenges include: {'; '.join(challenge_points[:3])}."
    else:
        # Fallback to relevant sentences
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
        challenge_sentences = []
        for sentence in sentences:
            if any(term in sentence.lower() for term in ['challenge', 'problem', 'difficulty', 'pain']):
                challenge_sentences.append(sentence)
        
        if challenge_sentences:
            response = f"Regarding challenges: {'. '.join(challenge_sentences[:2])}."
        else:
            response = f"From {main_section['section']['title']}: {sentences[0] if sentences else content}."
    
    return response

def generate_comprehensive_answer(sections, keywords, question):
    """Generate a comprehensive overview combining multiple relevant sections."""
    relevant_sections = []
    
    for section in sections:
        section_text = f"{section['title']} {' '.join(section['content'])}".lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in section_text)
        
        if keyword_matches > 0:
            relevant_sections.append({
                'section': section,
                'score': keyword_matches,
                'content': ' '.join(section['content']).strip()
            })
    
    if not relevant_sections:
        return "I found information about this topic, but it doesn't directly address your specific question."
    
    # Sort by relevance
    relevant_sections.sort(key=lambda x: x['score'], reverse=True)
    
    # Build comprehensive answer
    answer_parts = []
    
    # Primary definition/overview
    main_section = relevant_sections[0]
    content = main_section['content']
    
    # Get complete sentences, not fragments
    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
    if sentences:
        # Use first 1-2 complete sentences for overview
        overview = '. '.join(sentences[0:2])
        if not overview.endswith('.'):
            overview += '.'
        answer_parts.append(overview)
    
    # Add specific details from other relevant sections
    for section_info in relevant_sections[1:3]:  # Include up to 2 additional sections
        section = section_info['section']
        section_content = section_info['content']
        
        if section['title'] and section_content:
            # Extract most relevant complete sentence from this section
            section_sentences = [s.strip() for s in section_content.split('.') if len(s.strip()) > 25]
            if section_sentences:
                best_sentence = section_sentences[0]
                
                # Find sentence with most keyword matches
                for sentence in section_sentences[0:3]:  # Check first 3 sentences
                    sentence_matches = sum(1 for keyword in keywords if keyword in sentence.lower())
                    if sentence_matches > 0:
                        best_sentence = sentence
                        break
                
                if not best_sentence.endswith('.'):
                    best_sentence += '.'
                
                # Only add if it provides new information
                if best_sentence.lower() not in answer_parts[0].lower():
                    answer_parts.append(f"Additionally, {best_sentence}")
    
    return ' '.join(answer_parts)

def generate_detailed_answer(sections, keywords, question):
    """Generate detailed answer focusing on specific aspects."""
    question_lower = question.lower()
    
    # Find the most specific section
    best_section = None
    highest_specificity = 0
    
    for section in sections:
        section_text = f"{section['title']} {' '.join(section['content'])}".lower()
        
        # Calculate specificity score
        keyword_matches = sum(2 for keyword in keywords if keyword in section['title'].lower())
        keyword_matches += sum(1 for keyword in keywords if keyword in section_text)
        
        if keyword_matches > highest_specificity:
            highest_specificity = keyword_matches
            best_section = section
    
    if not best_section:
        return "I couldn't find specific details about that aspect in the available information."
    
    content = ' '.join(best_section['content'])
    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 15]
    
    # For "how" questions, look for process or method information
    if question_lower.startswith('how'):
        process_words = ['by', 'through', 'via', 'using', 'with', 'process', 'method', 'approach']
        for sentence in sentences:
            if any(word in sentence.lower() for word in process_words):
                return f"From {best_section['title']}: {sentence}."
    
    # For "why" questions, look for reasons or explanations
    elif question_lower.startswith('why'):
        reason_words = ['because', 'since', 'due to', 'reason', 'purpose', 'goal', 'objective']
        for sentence in sentences:
            if any(word in sentence.lower() for word in reason_words):
                return f"From {best_section['title']}: {sentence}."
    
    # Default detailed response
    if sentences:
        detailed_response = '. '.join(sentences[0:2])  # First two sentences for detail
        return f"From {best_section['title']}: {detailed_response}."
    
    return f"From {best_section['title']}: {content}"

def generate_analytical_answer(sections, keywords, question):
    """Generate analytical answer comparing different aspects or types."""
    question_lower = question.lower()
    
    # Look for sections that contain categories, types, or comparisons
    category_sections = []
    
    for section in sections:
        title_lower = section['title'].lower()
        content = ' '.join(section['content']).lower()
        
        # Check for categorical information or specific question matches
        is_categorical = any(word in title_lower for word in ['types', 'kinds', 'categories', 'different']) or \
                       any(word in content for word in ['include', 'such as', 'example', 'different types'])
        
        # Also check for seasonal, challenges, or other analytical topics
        is_analytical = any(word in title_lower for word in ['seasonal', 'challenges', 'considerations', 'factors', 'pain points']) or \
                       any(word in content for word in ['seasonal', 'challenges', 'problems', 'issues', 'difficulties'])
        
        if is_categorical or is_analytical:
            keyword_matches = sum(1 for keyword in keywords if keyword in f"{title_lower} {content}")
            if keyword_matches > 0:
                category_sections.append({
                    'section': section,
                    'score': keyword_matches,
                    'content': ' '.join(section['content']).strip()
                })
    
    if not category_sections:
        return generate_comprehensive_answer(sections, keywords, question)
    
    # Sort by relevance
    category_sections.sort(key=lambda x: x['score'], reverse=True)
    
    answer_parts = []
    
    for section_info in category_sections[:2]:  # Top 2 most relevant
        section = section_info['section']
        content = section_info['content']
        
        if section['title']:
            # Look for structured information (bullet points, lists, etc.)
            lines = content.split('\n')
            structured_items = []
            
            for line in lines:
                line = line.strip()
                if any(marker in line for marker in ['- **', '- ', '• ', '*']) and len(line) > 10:
                    # Clean up bullet point formatting
                    clean_item = line.lstrip('-*• ').strip()
                    if clean_item:
                        structured_items.append(clean_item)
                elif ':' in line and len(line) < 150 and len(line) > 20:  # Descriptive lines
                    structured_items.append(line)
            
            if structured_items:
                # Remove duplicates while preserving order
                unique_items = []
                seen = set()
                for item in structured_items:
                    item_key = item.lower().strip()
                    if item_key not in seen and len(item) > 10:
                        unique_items.append(item)
                        seen.add(item_key)
                
                # Format structured items nicely
                if len(unique_items) <= 3:
                    formatted_items = '; '.join(unique_items)
                else:
                    formatted_items = '; '.join(unique_items[:3]) + f" (and {len(unique_items)-3} more)"
                
                answer_parts.append(f"Regarding {section['title']}: {formatted_items}")
            else:
                # Fallback to sentences if no structured content
                sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 25]
                if sentences:
                    best_sentences = '. '.join(sentences[0:2])
                    if not best_sentences.endswith('.'):
                        best_sentences += '.'
                    answer_parts.append(f"Regarding {section['title']}: {best_sentences}")
    
    if answer_parts:
        return '. '.join(answer_parts)
    else:
        return generate_comprehensive_answer(sections, keywords, question)

def generate_yesno_answer(sections, keywords, question):
    """Generate yes/no answer with supporting explanation."""
    # Find most relevant section
    best_section = None
    highest_score = 0
    
    for section in sections:
        section_text = f"{section['title']} {' '.join(section['content'])}".lower()
        score = sum(1 for keyword in keywords if keyword in section_text)
        
        if score > highest_score:
            highest_score = score
            best_section = section
    
    if not best_section or highest_score == 0:
        return "I don't have enough information to definitively answer that yes/no question."
    
    content = ' '.join(best_section['content'])
    
    # Determine yes/no based on content
    question_lower = question.lower()
    positive_indicators = ['yes', 'true', 'can', 'will', 'does', 'is', 'are']
    
    # Simple heuristic: if the content contains information about the topic, it's likely "yes"
    if any(keyword in content.lower() for keyword in keywords):
        first_sentence = content.split('.')[0].strip()
        return f"Yes, according to the information about {best_section['title']}: {first_sentence}."
    else:
        return f"Based on the available information about {best_section['title']}, I cannot confirm this."

def generate_contextual_answer(sections, keywords, question):
    """Generate contextual answer by synthesizing information."""
    # Find all relevant sections
    relevant_info = []
    
    for section in sections:
        section_text = f"{section['title']} {' '.join(section['content'])}".lower()
        keyword_matches = sum(1 for keyword in keywords if keyword in section_text)
        
        if keyword_matches > 0:
            relevant_info.append({
                'title': section['title'],
                'content': ' '.join(section['content']).strip(),
                'matches': keyword_matches
            })
    
    if not relevant_info:
        return "I couldn't find information directly related to your question in the available documents."
    
    # Sort by relevance
    relevant_info.sort(key=lambda x: x['matches'], reverse=True)
    
    # Create contextual response
    main_info = relevant_info[0]
    
    # Start with most relevant information - use complete sentences
    sentences = [s.strip() for s in main_info['content'].split('.') if len(s.strip()) > 20]
    if sentences:
        main_sentence = sentences[0]
        if not main_sentence.endswith('.'):
            main_sentence += '.'
    else:
        main_sentence = main_info['content']
    
    if main_info['title']:
        response = f"Based on {main_info['title']}: {main_sentence}"
    else:
        response = main_sentence
    
    # Add context from other relevant sections if available and different
    if len(relevant_info) > 1:
        additional_info = relevant_info[1]
        additional_sentences = [s.strip() for s in additional_info['content'].split('.') if len(s.strip()) > 25]
        
        if additional_sentences:
            context_sentence = additional_sentences[0]
            if not context_sentence.endswith('.'):
                context_sentence += '.'
                
            # Only add if it provides genuinely different information
            if (context_sentence.lower() not in response.lower() and 
                len(set(context_sentence.lower().split()) & set(response.lower().split())) < len(context_sentence.split()) * 0.7):
                
                if additional_info['title']:
                    response += f" Additionally, regarding {additional_info['title']}: {context_sentence}"
                else:
                    response += f" {context_sentence}"
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if qdrant_client:
            collections = qdrant_client.get_collections()
            return {
                "status": "healthy",
                "qdrant_connected": True,
                "embedding_model_loaded": embedding_model is not None,
                "collections": [c.name for c in collections.collections]
            }
        else:
            return {
                "status": "unhealthy",
                "qdrant_connected": False,
                "embedding_model_loaded": embedding_model is not None
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "qdrant_connected": False,
            "embedding_model_loaded": embedding_model is not None
        }

@app.get("/stats")
async def get_stats():
    """Get collection statistics."""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not connected")
    
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "status": collection_info.status
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting collection stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)