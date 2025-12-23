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
        
        # Search in Qdrant with improved strategy for diverse results
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=question_embedding,
            limit=min(request.max_results * 2, 10),  # Get more results initially
            score_threshold=max(request.threshold - 0.1, 0.2)  # Lower threshold for more diversity
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
    
    # Strategy 1: Comprehensive overview questions
    if any(word in question_lower for word in ['about', 'overview', 'tell me', 'explain', 'describe', 'what are']):
        return generate_comprehensive_answer(sections, question_keywords, question)
    
    # Strategy 2: Specific detail questions  
    elif any(word in question_lower for word in ['how', 'why', 'when', 'which', 'specific']):
        return generate_detailed_answer(sections, question_keywords, question)
        
    # Strategy 3: Comparative or analytical questions
    elif any(word in question_lower for word in ['different', 'types', 'kinds', 'compare', 'versus', 'options']):
        return generate_analytical_answer(sections, question_keywords, question)
        
    # Strategy 4: Yes/No questions with elaboration
    elif any(indicator in question_lower.split()[:3] for indicator in ['is', 'are', 'does', 'do', 'can', 'will']):
        return generate_yesno_answer(sections, question_keywords, question)
    
    # Default: Contextual synthesis
    else:
        return generate_contextual_answer(sections, question_keywords, question)

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
                'content': ' '.join(section['content'])
            })
    
    if not relevant_sections:
        return "I found information about this topic, but it doesn't directly address your specific question."
    
    # Sort by relevance
    relevant_sections.sort(key=lambda x: x['score'], reverse=True)
    
    # Build comprehensive answer
    answer_parts = []
    
    # Primary definition/overview
    main_section = relevant_sections[0]
    first_sentences = main_section['content'].split('.')[0:2]
    answer_parts.append(' '.join(first_sentences).strip() + '.')
    
    # Add specific details from other sections
    for section_info in relevant_sections[1:3]:  # Include up to 2 additional sections
        section = section_info['section']
        content = section_info['content']
        
        if section['title'] and content:
            # Extract most relevant sentence from this section
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
            if sentences:
                best_sentence = sentences[0]
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in keywords):
                        best_sentence = sentence
                        break
                
                answer_parts.append(f"In terms of {section['title'].lower()}: {best_sentence}.")
    
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
    # Look for sections that contain categories, types, or comparisons
    category_sections = []
    
    for section in sections:
        title_lower = section['title'].lower()
        content = ' '.join(section['content']).lower()
        
        # Check for categorical information
        if any(word in title_lower for word in ['types', 'kinds', 'categories', 'different']) or \
           any(word in content for word in ['include', 'such as', 'example', 'different types']):
            
            keyword_matches = sum(1 for keyword in keywords if keyword in f"{title_lower} {content}")
            if keyword_matches > 0:
                category_sections.append({
                    'section': section,
                    'score': keyword_matches,
                    'content': ' '.join(section['content'])
                })
    
    if not category_sections:
        return generate_comprehensive_answer(sections, keywords, question)
    
    # Sort by relevance
    category_sections.sort(key=lambda x: x['score'], reverse=True)
    
    answer_parts = []
    
    for section_info in category_sections[:3]:  # Top 3 most relevant
        section = section_info['section']
        content = section_info['content']
        
        if section['title']:
            # Extract bullet points or list items
            lines = content.split('\n')
            items = []
            
            for line in lines:
                if any(marker in line for marker in ['- **', '- ', '• ', '*']):
                    items.append(line.strip())
                elif ':' in line and len(line) < 100:  # Short descriptive lines
                    items.append(line.strip())
            
            if items:
                formatted_items = '; '.join(items[:3])  # Top 3 items
                answer_parts.append(f"{section['title']}: {formatted_items}")
            else:
                # Fallback to sentences
                sentences = content.split('.')[0:2]
                answer_parts.append(f"{section['title']}: {'. '.join(sentences)}.")
    
    return '. '.join(answer_parts) if answer_parts else generate_comprehensive_answer(sections, keywords, question)

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
                'content': ' '.join(section['content']),
                'matches': keyword_matches
            })
    
    if not relevant_info:
        return "I couldn't find information directly related to your question in the available documents."
    
    # Sort by relevance
    relevant_info.sort(key=lambda x: x['matches'], reverse=True)
    
    # Create contextual response
    main_info = relevant_info[0]
    
    # Start with most relevant information
    sentences = [s.strip() for s in main_info['content'].split('.') if len(s.strip()) > 15]
    main_sentence = sentences[0] if sentences else main_info['content']
    
    response = f"Based on the information about {main_info['title']}: {main_sentence}."
    
    # Add context from other relevant sections
    if len(relevant_info) > 1:
        additional_context = []
        for info in relevant_info[1:2]:  # Add one more piece of context
            context_sentence = info['content'].split('.')[0].strip()
            if context_sentence and len(context_sentence) > 20:
                additional_context.append(f"Additionally, regarding {info['title']}: {context_sentence}.")
        
        if additional_context:
            response += " " + " ".join(additional_context)
    
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