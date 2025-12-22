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
        
        # Debug: Print what we're working with
        print(f"\nDEBUG - Question: {request.question}")
        print(f"DEBUG - Context length: {len(context)}")
        print(f"DEBUG - Search results count: {len(search_results)}")
        for i, result in enumerate(search_results):
            text = result.payload.get("content", "No content")
            print(f"DEBUG - Result {i} (score: {result.score:.3f}): {text[:150]}...")
        print(f"DEBUG - Full context: {context[:800]}...")
        
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
        if section['content']:
            print(f"    Content preview: {' '.join(section['content'])[:100]}...")
    
    # Extract keywords from question
    question_keywords = [word.lower() for word in question_lower.split() 
                        if len(word) > 2 and word not in [
                            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'tell', 'me',
                            'about', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                            'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did',
                            'has', 'have', 'had', 'this', 'that', 'these', 'those', 'a', 'an'
                        ]]
    
    print(f"DEBUG - Question keywords: {question_keywords}")
    
    # Handle Yes/No questions first
    yes_no_indicators = ['is', 'are', 'does', 'do', 'can', 'will', 'should', 'has', 'have']
    if any(indicator in question_lower.split()[:3] for indicator in yes_no_indicators):
        print("DEBUG - Detected yes/no question")
        # Find the most relevant section for yes/no questions
        best_section = None
        highest_score = 0
        
        for section in sections:
            section_text = f"{section['title']} {' '.join(section['content'])}".lower()
            score = sum(1 for keyword in question_keywords if keyword in section_text)
            print(f"DEBUG - Section '{section['title']}' score: {score}")
            if score > highest_score:
                highest_score = score
                best_section = section
        
        if best_section and highest_score > 0:
            section_content = ' '.join(best_section['content'])
            
            # For B2B questions
            if 'b2b' in question_lower or 'business to business' in question_lower:
                if any(term in section_content.lower() for term in ['business to business', 'b2b', 'business']):
                    return f"Yes, based on the {best_section['title']} section: {section_content.split('.')[0]}."
            
            # General yes/no answer
            first_sentence = section_content.split('.')[0]
            if first_sentence:
                return f"Yes, according to the {best_section['title']}: {first_sentence}."
    
    # Special handling for technology/architecture questions
    if any(tech_word in question_lower for tech_word in ['technology', 'tech', 'framework', 'built', 'stack', 'architecture', 'development', 'programming', 'service', 'database', 'frontend', 'backend']):
        print("DEBUG - Detected technology question")
        # Look for architecture or tech-related sections
        tech_sections = []
        for section in sections:
            section_title_lower = section['title'].lower()
            section_content = ' '.join(section['content']).lower()
            
            print(f"DEBUG - Checking section '{section['title']}' for tech content")
            
            # Check for direct keyword matches in title first
            title_matches = sum(1 for keyword in question_keywords if keyword in section_title_lower)
            content_matches = sum(1 for keyword in question_keywords if keyword in section_content)
            
            # High priority for exact title matches (e.g., "Backend" section for "backend" question)
            if title_matches > 0:
                print(f"DEBUG - Direct title match in '{section['title']}' with score {title_matches + 10}")
                tech_sections.append((title_matches + 10, section))
            # Medium priority for content matches
            elif content_matches > 0:
                print(f"DEBUG - Content match in '{section['title']}' with score {content_matches + 5}")
                tech_sections.append((content_matches + 5, section))
            # Lower priority for general tech terms
            elif any(tech_term in section_title_lower or tech_term in section_content for tech_term in [
                'architecture', 'frontend', 'backend', 'technology', 'service', 'database',
                'application', 'layer', 'security', 'deployment', 'system', 'api',
                'javascript', 'react', 'dojo', 'framework', 'microservices'
            ]):
                print(f"DEBUG - General tech match in '{section['title']}' with score 1")
                tech_sections.append((1, section))
        
        if tech_sections:
            # Sort by priority and find the best match
            tech_sections.sort(reverse=True, key=lambda x: x[0])
            best_tech_section = tech_sections[0][1]
            
            print(f"DEBUG - Selected tech section: '{best_tech_section['title']}' with priority {tech_sections[0][0]}")
            
            if best_tech_section:
                content = ' '.join(best_tech_section['content'])
                # If asking specifically about backend/frontend, return the whole section content
                if any(specific in question_lower for specific in ['backend', 'frontend', 'technology stack']):
                    # Return the full content for specific tech questions
                    if content:
                        return f"From the {best_tech_section['title']}: {content}"
                
                # For other tech questions, return first relevant sentence
                sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10]
                if sentences:
                    return f"From the {best_tech_section['title']}: {sentences[0]}."
    
    # General question handling - find most relevant section
    best_section = None
    highest_score = 0
    
    for section in sections:
        section_text = f"{section['title']} {' '.join(section['content'])}".lower()
        
        # Score based on keyword matches
        keyword_score = sum(1 for keyword in question_keywords if keyword in section_text)
        
        # Bonus for title matches
        title_score = sum(2 for keyword in question_keywords if keyword in section['title'].lower())
        
        total_score = keyword_score + title_score
        if total_score > highest_score:
            highest_score = total_score
            best_section = section
    
    if best_section and highest_score > 0:
        content = ' '.join(best_section['content'])
        
        # Find most relevant sentence within the section
        sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 15]
        
        if sentences:
            # Score sentences within the section
            best_sentence = sentences[0]
            best_sentence_score = 0
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = sum(1 for keyword in question_keywords if keyword in sentence_lower)
                if score > best_sentence_score:
                    best_sentence_score = score
                    best_sentence = sentence
            
            # Return with section context
            if not best_sentence.endswith(('.', '!', '?')):
                best_sentence += '.'
            
            if best_section['title']:
                return f"From the {best_section['title']}: {best_sentence}"
            else:
                return best_sentence
    
    # Fallback to original logic if no sections found
    clean_context = context.replace('#', '').strip()
    sentences = [s.strip() for s in clean_context.split('.') if len(s.strip()) > 15]
    
    if sentences:
        return sentences[0] + ('.' if not sentences[0].endswith(('.', '!', '?')) else '')
    
    return clean_context[:300] + ('...' if len(clean_context) > 300 else '')

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