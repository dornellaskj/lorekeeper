import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import hashlib
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, CollectionInfo
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    content: str
    file_path: str
    file_name: str
    chunk_index: int
    metadata: Dict[str, Any]

class DataLoader:
    """Data loader for ingesting documents into Qdrant vector database."""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,  # Increased from 500
        chunk_overlap: int = 200  # Increased from 50
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Test connection
        self._test_connection()
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        
        # Check for GPU availability
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")
        
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        logger.info(f"Embedding model loaded on {device.upper()}")
        logger.info(f"Vector dimension: {self.vector_size}")
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _test_connection(self):
        """Test connection to Qdrant server."""
        try:
            # Simple health check
            collections = self.qdrant_client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} collections.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
            logger.error(f"Connection error: {e}")
            raise ConnectionError(f"Cannot connect to Qdrant: {e}")
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            # Try to get collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists with {collection_info.points_count} points")
            return
        except Exception as e:
            logger.info(f"Collection '{self.collection_name}' does not exist or error occurred: {e}")
        
        try:
            # Try to create the collection
            logger.info(f"Creating collection '{self.collection_name}' with vector size {self.vector_size}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Successfully created collection '{self.collection_name}'")
        except Exception as e:
            # Check if the error is about collection already existing
            if "already exists" in str(e).lower() or "conflict" in str(e).lower():
                logger.info(f"Collection '{self.collection_name}' already exists (detected from error)")
            else:
                logger.error(f"Failed to create collection '{self.collection_name}': {e}")
                raise
    
    def _read_text_file(self, file_path: Path) -> str:
        """Read text content from a file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, read as binary and decode with errors='ignore'
            with open(file_path, 'rb') as f:
                return f.read().decode('utf-8', errors='ignore')
                
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def _chunk_text(self, text: str, file_path: str, file_name: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks with intelligent boundaries."""
        chunks = []
        
        # First, try to split by paragraphs (double newlines)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        start_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size and current_chunk:
                # Save current chunk
                if current_chunk.strip():
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        file_path=file_path,
                        file_name=file_name,
                        chunk_index=chunk_index,
                        metadata={
                            'file_size': len(text),
                            'chunk_start': start_pos,
                            'chunk_end': start_pos + len(current_chunk)
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap
                if len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    # Find a good break point in the overlap
                    good_breaks = ['. ', '! ', '? ', '\n']
                    best_break = 0
                    for i, char_pair in enumerate(zip(overlap_text, overlap_text[1:])):
                        if ''.join(char_pair) in good_breaks:
                            best_break = i + 2
                    
                    current_chunk = overlap_text[best_break:] + "\n\n" + para
                else:
                    current_chunk = para
                
                start_pos = start_pos + len(current_chunk) - len(para) - 2
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                file_path=file_path,
                file_name=file_name,
                chunk_index=chunk_index,
                metadata={
                    'file_size': len(text),
                    'chunk_start': start_pos,
                    'chunk_end': start_pos + len(current_chunk)
                }
            )
            chunks.append(chunk)
        
        # If we got no chunks (text might be too short or no paragraphs), fall back to simple chunking
        if not chunks:
            chunks = self._simple_chunk_text(text, file_path, file_name)
        
        return chunks
    
    def _simple_chunk_text(self, text: str, file_path: str, file_name: str) -> List[DocumentChunk]:
        """Fallback simple chunking method."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_content = text[start:end]
            
            # Try to end at a sentence boundary if possible
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                sentence_endings = ['. ', '! ', '? ', '\n\n', '\n']
                best_end = end
                
                for i in range(max(0, end - 200), end):
                    for ending in sentence_endings:
                        if text[i:i+len(ending)] == ending:
                            best_end = i + len(ending)
                
                chunk_content = text[start:best_end]
                end = best_end
            
            if chunk_content.strip():  # Only add non-empty chunks
                chunk = DocumentChunk(
                    content=chunk_content.strip(),
                    file_path=file_path,
                    file_name=file_name,
                    chunk_index=chunk_index,
                    metadata={
                        'file_size': len(text),
                        'chunk_start': start,
                        'chunk_end': end
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate a unique ID for a chunk based on file path, modification time, and content."""
        # Include file path and chunk index for uniqueness
        content = f"{chunk.file_path}_{chunk.chunk_index}_{chunk.content[:50]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_file_already_processed(self, file_path: Path) -> bool:
        """Check if a file has already been processed by looking for its chunks in Qdrant."""
        try:
            # Search for any chunk from this file
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [{
                        "key": "file_path",
                        "match": {"value": str(file_path)}
                    }]
                },
                limit=1
            )
            return len(search_result[0]) > 0
        except Exception as e:
            logger.warning(f"Could not check if file {file_path} is processed: {e}")
            return False
    
    def _get_processed_files_count(self) -> int:
        """Get count of unique files already processed."""
        try:
            # Get a sample of points to count unique files
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on your needs
                with_payload=True
            )
            
            unique_files = set()
            for point in scroll_result[0]:
                if point.payload and 'file_path' in point.payload:
                    unique_files.add(point.payload['file_path'])
            
            return len(unique_files)
        except Exception as e:
            logger.warning(f"Could not count processed files: {e}")
            return 0
        """Generate embeddings for chunks."""
        texts = [chunk.content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        
        # Generate embeddings in batches
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def load_documents(self, data_folder: str) -> None:
        """Load all documents from the specified folder."""
        data_path = Path(data_folder)
        
        if not data_path.exists():
            logger.error(f"Data folder {data_folder} does not exist")
            return
        
        # Find all text files
        text_extensions = {'.txt', '.md', '.rst', '.log', '.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml'}
        text_files = []
        
        for ext in text_extensions:
            text_files.extend(data_path.rglob(f'*{ext}'))
        
        if not text_files:
            logger.warning(f"No text files found in {data_folder}")
            return
        
        logger.info(f"Found {len(text_files)} text files")
        
        # Check how many files are already processed
        processed_files_count = self._get_processed_files_count()
        logger.info(f"Already processed files in database: {processed_files_count}")
        
        # Debug: Log all relevant environment variables
        logger.info(f"Environment variables:")
        logger.info(f"  FORCE_REPROCESS = '{os.getenv('FORCE_REPROCESS', 'NOT_SET')}'")
        logger.info(f"  SKIP_PROCESSED_FILES = '{os.getenv('SKIP_PROCESSED_FILES', 'NOT_SET')}'")
        logger.info(f"  CHUNK_SIZE = '{os.getenv('CHUNK_SIZE', 'NOT_SET')}'")
        logger.info(f"  CHUNK_OVERLAP = '{os.getenv('CHUNK_OVERLAP', 'NOT_SET')}'")
        
        # Check for force reprocess flag
        force_reprocess = os.getenv("FORCE_REPROCESS", "false").lower() == "true"
        skip_processed = os.getenv("SKIP_PROCESSED_FILES", "true").lower() == "true"
        
        logger.info(f"Parsed flags: force_reprocess={force_reprocess}, skip_processed={skip_processed}")
        
        # If force reprocess is enabled, clear existing data and process all files
        if force_reprocess:
            logger.info("FORCE_REPROCESS enabled - clearing existing collection data")
            try:
                # Delete and recreate collection to clear all data
                self.qdrant_client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection '{self.collection_name}'")
                self._ensure_collection_exists()
                logger.info(f"Recreated collection '{self.collection_name}'")
            except Exception as e:
                logger.warning(f"Could not clear collection (might not exist): {e}")
                self._ensure_collection_exists()
            
            files_to_process = text_files
            logger.info(f"Force processing all {len(files_to_process)} files")
        
        # Filter out already processed files (optional - can be disabled)
        elif not skip_processed:
            files_to_process = text_files
            logger.info(f"Processing all {len(files_to_process)} files (skip check disabled)")
        else:
            files_to_process = []
            for file_path in text_files:
                if self._is_file_already_processed(file_path):
                    logger.info(f"Skipping already processed file: {file_path.name}")
                else:
                    files_to_process.append(file_path)
            
            logger.info(f"Files to process: {len(files_to_process)} (skipped {len(text_files) - len(files_to_process)} already processed)")
        
        if not files_to_process:
            logger.info("No new files to process!")
            return
        
        all_chunks = []
        
        # Process each file
        for file_path in tqdm(files_to_process, desc="Processing files"):
            logger.info(f"Processing: {file_path}")
            
            content = self._read_text_file(file_path)
            if not content.strip():
                logger.warning(f"File is empty or unreadable: {file_path}")
                continue
            
            # Split into chunks
            chunks = self._chunk_text(
                content,
                str(file_path),
                file_path.name
            )
            
            all_chunks.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
        
        if not all_chunks:
            logger.warning("No chunks created from any files")
            return
        
        logger.info(f"Total chunks to process: {len(all_chunks)}")
        
        # Generate embeddings
        embeddings = self._embed_chunks(all_chunks)
        
        # Prepare points for Qdrant
        points = []
        for chunk, embedding in zip(all_chunks, embeddings):
            point = PointStruct(
                id=self._generate_chunk_id(chunk),
                vector=embedding.tolist(),
                payload={
                    'content': chunk.content,
                    'file_path': chunk.file_path,
                    'file_name': chunk.file_name,
                    'chunk_index': chunk.chunk_index,
                    **chunk.metadata
                }
            )
            points.append(point)
        
        # Upload to Qdrant in batches
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        logger.info(f"Uploading {len(points)} points to Qdrant in {total_batches} batches")
        
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i:i + batch_size]
            
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            except Exception as e:
                logger.error(f"Error uploading batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info("Data loading completed successfully!")
        
        # Print collection stats
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        logger.info(f"Collection '{self.collection_name}' now contains {collection_info.points_count} points")

def main():
    """Main entry point."""
    # Configuration from environment variables
    qdrant_host = os.getenv("QDRANT_HOST", "qdrant-service")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "documents")
    data_folder = os.getenv("DATA_FOLDER", "/data")
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    logger.info(f"Starting data loader with configuration:")
    logger.info(f"  Qdrant Host: {qdrant_host}:{qdrant_port}")
    logger.info(f"  Collection: {collection_name}")
    logger.info(f"  Data Folder: {data_folder}")
    logger.info(f"  Embedding Model: {embedding_model}")
    logger.info(f"  Chunk Size: {chunk_size}")
    logger.info(f"  Chunk Overlap: {chunk_overlap}")
    
    # Initialize and run data loader
    loader = DataLoader(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    loader.load_documents(data_folder)

if __name__ == "__main__":
    main()