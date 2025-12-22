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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lorekeeper Chatbot", version="1.0.0")

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-service")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# Global variables
qdrant_client = None
embedding_model = None

class QueryRequest(BaseModel):
    question: str
    max_results: Optional[int] = MAX_RESULTS
    threshold: Optional[float] = SIMILARITY_THRESHOLD

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    query_time: float
    similarity_scores: List[float]

async def initialize_services():
    """Initialize Qdrant client and embedding model."""
    global qdrant_client, embedding_model
    
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
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Lorekeeper Chatbot</h1>
            <p style="text-align: center; color: #666; margin-bottom: 20px;">
                Ask questions about your uploaded documents
            </p>
            
            <div class="chat-container">
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

            function handleKeyPress(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    askQuestion();
                }
            }

            async function askQuestion() {
                const question = questionInput.value.trim();
                if (!question) return;

                // Add user message
                addMessage(question, 'user');
                
                // Clear input and disable button
                questionInput.value = '';
                askButton.disabled = true;
                loading.style.display = 'block';

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: question })
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
                    <strong>🤖 Lorekeeper:</strong> ${data.answer}
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

@app.post("/chat")
async def chat(request: QueryRequest):
    """Process a chat question and return an answer with sources."""
    if not qdrant_client or not embedding_model:
        raise HTTPException(status_code=503, detail="Services not initialized. Please try again later.")
    
    start_time = datetime.now()
    
    try:
        # Generate embedding for the question
        question_embedding = embedding_model.encode(request.question).tolist()
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_embedding,
            limit=request.max_results,
            score_threshold=request.threshold
        )
        
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
            sources.append({
                "filename": payload.get("filename", "Unknown"),
                "content": payload.get("content", ""),
                "chunk_id": payload.get("chunk_id", 0)
            })
            similarity_scores.append(result.score)
            context_parts.append(payload.get("content", ""))
        
        # Generate answer based on context - use all relevant results
        context = "\n\n".join(context_parts)  # Use all results for better context
        answer = generate_answer(request.question, context, search_results)
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            query_time=query_time,
            similarity_scores=similarity_scores
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

def generate_answer(question: str, context: str, search_results: list) -> str:
    """Generate an answer based on the question and context.
    This is a simple implementation - you could replace with a proper LLM later."""
    
    if not context.strip():
        return "I couldn't find relevant information to answer your question."
    
    question_lower = question.lower()
    
    # Clean the context by removing markdown headers
    clean_context = context.replace('#', '').strip()
    
    # Split context into sentences using regex
    import re
    sentences = re.split(r'[.!?]+', clean_context)
    clean_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    # Handle Yes/No questions first
    yes_no_indicators = ['is', 'are', 'does', 'do', 'can', 'will', 'should', 'has', 'have']
    if any(indicator in question_lower.split()[:3] for indicator in yes_no_indicators):
        # For yes/no questions, look for confirming information
        question_keywords = [word.lower() for word in question_lower.split() 
                           if len(word) > 2 and word not in yes_no_indicators + ['a', 'an', 'the', 'it']]
        
        best_sentence = None
        highest_score = 0
        
        for sentence in clean_sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for keyword in question_keywords if keyword in sentence_lower)
            if score > highest_score:
                highest_score = score
                best_sentence = sentence
        
        if best_sentence and highest_score > 0:
            # Check if the sentence supports a "yes" answer
            sentence_lower = best_sentence.lower()
            # Look for affirming patterns
            if any(keyword in sentence_lower for keyword in question_keywords):
                # For B2B question: if sentence contains "business to business" -> "Yes"
                if 'b2b' in question_lower or 'business to business' in question_lower:
                    if 'business to business' in sentence_lower or 'b2b' in sentence_lower:
                        return f"Yes, {best_sentence.lower()}."
                # For other yes/no questions, provide context
                return f"Yes, {best_sentence.lower()}."
            else:
                return f"Based on the information: {best_sentence}."
    
    # Extract key terms from the question (excluding common words)
    question_keywords = [word.lower() for word in question_lower.split() 
                        if len(word) > 2 and word not in [
                            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'tell', 'me',
                            'about', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                            'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did',
                            'has', 'have', 'had', 'this', 'that', 'these', 'those', 'a', 'an'
                        ]]
    
    # Score sentences based on keyword relevance
    scored_sentences = []
    for sentence in clean_sentences:
        sentence_lower = sentence.lower()
        # Count keyword matches
        keyword_score = sum(1 for keyword in question_keywords if keyword in sentence_lower)
        
        # Bonus for exact phrase matches
        phrase_bonus = 0
        if len(question_keywords) >= 2:
            for i in range(len(question_keywords) - 1):
                phrase = f"{question_keywords[i]} {question_keywords[i+1]}"
                if phrase in sentence_lower:
                    phrase_bonus += 2
        
        total_score = keyword_score + phrase_bonus
        if total_score > 0:
            scored_sentences.append((total_score, sentence))
    
    if scored_sentences:
        # Sort by relevance and return the best sentence
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        best_sentence = scored_sentences[0][1]
        
        # Make sure sentence ends properly
        if not best_sentence.endswith(('.', '!', '?')):
            best_sentence += '.'
            
        return best_sentence
    
    # If no good matches, return the first substantial sentence
    for sentence in clean_sentences:
        if len(sentence) > 20:
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            return sentence
    
    # Final fallback - return cleaned context
    if len(clean_context) > 300:
        # Find a good breaking point
        cutoff = clean_context.find('. ', 250)
        if cutoff > 0:
            return clean_context[:cutoff + 1]
        else:
            return clean_context[:300] + '...'
    else:
        return clean_context

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