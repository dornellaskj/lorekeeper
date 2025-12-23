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
                "filename": payload.get("file_name", "Unknown"),  # Fixed: use file_name instead of filename
                "content": payload.get("content", ""),
                "chunk_id": payload.get("chunk_index", 0)  # Fixed: use chunk_index instead of chunk_id
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