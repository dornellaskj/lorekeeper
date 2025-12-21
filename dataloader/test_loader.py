#!/usr/bin/env python3
"""
Simple test script to verify the data loader functionality locally.
"""

import os
import tempfile
from pathlib import Path
from data_loader import DataLoader

def create_test_documents(test_dir: Path) -> None:
    """Create some test documents for testing."""
    
    # Create a simple text file
    (test_dir / "document1.txt").write_text("""
    This is the first test document. It contains information about 
    artificial intelligence and machine learning. AI systems can 
    process natural language and understand context.
    """)
    
    # Create a markdown file
    (test_dir / "document2.md").write_text("""
    # Knowledge Base
    
    ## Introduction
    This document describes our knowledge management system.
    
    ## Features
    - Document ingestion
    - Vector embeddings
    - Semantic search
    - Question answering
    
    The system uses Qdrant for vector storage and retrieval.
    """)
    
    # Create a Python file
    (test_dir / "example.py").write_text("""
    def greet(name):
        '''A simple greeting function.'''
        return f"Hello, {name}!"
    
    class DataProcessor:
        '''Processes data for the application.'''
        
        def __init__(self):
            self.data = []
        
        def add_data(self, item):
            self.data.append(item)
    """)

def main():
    """Run a local test of the data loader."""
    
    print("Testing Data Loader Locally")
    print("=" * 40)
    
    # Create temporary directory with test documents
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        print(f"Creating test documents in: {test_dir}")
        
        create_test_documents(test_dir)
        
        # List created files
        print("\nTest files created:")
        for file_path in test_dir.iterdir():
            if file_path.is_file():
                print(f"  - {file_path.name} ({file_path.stat().st_size} bytes)")
        
        # Initialize data loader (will try to connect to Qdrant)
        try:
            print(f"\nConnecting to Qdrant at localhost:6333")
            loader = DataLoader(
                qdrant_host="localhost",
                qdrant_port=6333,
                collection_name="test_documents",
                chunk_size=200,  # Smaller chunks for testing
                chunk_overlap=20
            )
            
            print("Loading documents...")
            loader.load_documents(str(test_dir))
            print("Test completed successfully!")
            
        except Exception as e:
            print(f"Error: {e}")
            print("\nNote: Make sure Qdrant is running locally on port 6333")
            print("You can start it with Docker: docker run -p 6333:6333 qdrant/qdrant")

if __name__ == "__main__":
    main()