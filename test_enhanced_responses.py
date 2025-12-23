#!/usr/bin/env python3
"""
Test script to demonstrate enhanced chatbot responses
"""

import requests
import json
import time

def test_chatbot_question(question, base_url="http://localhost:8081"):
    """Send a question to the chatbot and return the response"""
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"question": question},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}

def main():
    print("🤖 Testing Enhanced Lorekeeper Responses")
    print("=" * 50)
    
    # Test questions with expected improvements
    test_questions = [
        "Tell me about retailers",
        "What types of retailers are there?",
        "What do retailers want?",
        "How do retailers operate seasonally?", 
        "What are the different retailer categories?",
        "Tell me more about retailers",
        "What challenges do retailers face?",
        "Describe retailer business models"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        print("-" * 30)
        
        result = test_chatbot_question(question)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"🤖 Answer: {result['answer']}")
            print(f"⏱️  Query time: {result['query_time']:.2f}s")
            print(f"📚 Sources found: {len(result['sources'])}")
            
            # Show similarity scores to assess diversity
            if result.get('similarity_scores'):
                scores = [f"{score:.3f}" for score in result['similarity_scores']]
                print(f"🎯 Similarity scores: {', '.join(scores)}")
        
        time.sleep(1)  # Small delay between requests
    
    print(f"\n{'=' * 50}")
    print("✅ Testing completed!")
    print("\nExpected improvements:")
    print("- More comprehensive answers combining multiple sections")
    print("- Better analysis of retailer types and categories") 
    print("- More contextual and nuanced responses")
    print("- Less repetition across different questions")

if __name__ == "__main__":
    main()