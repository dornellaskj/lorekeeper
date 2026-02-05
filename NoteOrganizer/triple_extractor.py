"""
Semantic Triple Extractor - Extract knowledge graph triples from documents
Transforms unstructured text into (Subject, Predicate, Object) triples
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import hashlib

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SemanticTriple:
    """Represents a semantic triple (Subject, Predicate, Object)."""
    subject: str
    predicate: str
    object: str
    confidence: float
    source_file: str
    source_text: str
    triple_type: str  # 'definition', 'relationship', 'property', 'action', 'hierarchy'


class TripleExtractor:
    """Extracts semantic triples from text using pattern matching and NLP heuristics."""
    
    def __init__(self):
        # Patterns for extracting different types of triples
        self.patterns = {
            # Definitions: "X is Y", "X are Y", "X refers to Y"
            'definition': [
                r'(?P<subject>[A-Z][^.]*?)\s+(?:is|are|refers to|means|defines?)\s+(?:a|an|the)?\s*(?P<object>[^.]+)',
                r'(?P<subject>[A-Z][A-Za-z\s]+)\s*[-:]\s*(?P<object>[^.]+)',
                r'(?P<subject>[A-Z][A-Za-z\s]+)\s+(?:can be defined as|is defined as)\s+(?P<object>[^.]+)',
            ],
            # Properties: "X has Y", "X contains Y", "X includes Y"
            'property': [
                r'(?P<subject>[A-Z][^.]*?)\s+(?:has|have|contains?|includes?|consists? of)\s+(?P<object>[^.]+)',
                r'(?P<subject>[A-Z][^.]*?)\s+(?:is composed of|is made up of)\s+(?P<object>[^.]+)',
            ],
            # Actions: "X does Y", "X performs Y", "to X, do Y"
            'action': [
                r'(?:To|When)\s+(?P<subject>[^,]+),\s+(?P<object>[^.]+)',
                r'(?P<subject>[A-Z][^.]*?)\s+(?:should|must|will|can)\s+(?P<object>[^.]+)',
                r'(?P<subject>[A-Z][^.]*?)\s+(?:performs?|executes?|runs?|triggers?)\s+(?P<object>[^.]+)',
            ],
            # Relationships: "X relates to Y", "X connects to Y", "X depends on Y"
            'relationship': [
                r'(?P<subject>[A-Z][^.]*?)\s+(?:relates? to|connects? to|depends? on|requires?|needs?)\s+(?P<object>[^.]+)',
                r'(?P<subject>[A-Z][^.]*?)\s+(?:is related to|is connected to|is dependent on)\s+(?P<object>[^.]+)',
                r'(?P<subject>[A-Z][^.]*?)\s+(?:communicates? with|interacts? with|works? with)\s+(?P<object>[^.]+)',
            ],
            # Hierarchy: "X is a type of Y", "X is part of Y"
            'hierarchy': [
                r'(?P<subject>[A-Z][^.]*?)\s+(?:is a type of|is a kind of|is part of|belongs to)\s+(?P<object>[^.]+)',
                r'(?P<subject>[A-Z][^.]*?)\s+(?:inherits from|extends|derives from)\s+(?P<object>[^.]+)',
            ],
            # Causation: "X causes Y", "X leads to Y", "X results in Y"
            'causation': [
                r'(?P<subject>[A-Z][^.]*?)\s+(?:causes?|leads? to|results? in|triggers?)\s+(?P<object>[^.]+)',
                r'(?:If|When)\s+(?P<subject>[^,]+),\s+(?:then\s+)?(?P<object>[^.]+)',
            ],
            # Location/Context: "X is in Y", "X is located in Y", "X occurs in Y"
            'location': [
                r'(?P<subject>[A-Z][^.]*?)\s+(?:is in|is located in|occurs? in|happens? in|exists? in)\s+(?P<object>[^.]+)',
            ],
        }
        
        # Predicate mappings for each pattern type
        self.predicates = {
            'definition': 'is_defined_as',
            'property': 'has_property',
            'action': 'performs_action',
            'relationship': 'relates_to',
            'hierarchy': 'is_type_of',
            'causation': 'causes',
            'location': 'is_located_in',
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove markdown formatting
        text = re.sub(r'[#*_`]', '', text)
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        return text.strip()
    
    def clean_entity(self, entity: str) -> str:
        """Clean an extracted entity."""
        entity = entity.strip()
        # Remove leading articles
        entity = re.sub(r'^(a|an|the)\s+', '', entity, flags=re.IGNORECASE)
        # Remove trailing punctuation
        entity = re.sub(r'[.,;:!?]+$', '', entity)
        # Limit length
        if len(entity) > 100:
            entity = entity[:100] + '...'
        return entity.strip()
    
    def extract_from_sentence(self, sentence: str, source_file: str) -> List[SemanticTriple]:
        """Extract triples from a single sentence."""
        triples = []
        sentence = self.clean_text(sentence)
        
        if len(sentence) < 10:  # Skip very short sentences
            return triples
        
        for triple_type, patterns in self.patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        subject = self.clean_entity(match.group('subject'))
                        obj = self.clean_entity(match.group('object'))
                        
                        # Skip if subject or object is too short or too long
                        if len(subject) < 2 or len(obj) < 2:
                            continue
                        if len(subject) > 100 or len(obj) > 100:
                            continue
                        
                        # Calculate confidence based on pattern match quality
                        confidence = self._calculate_confidence(subject, obj, sentence)
                        
                        if confidence > 0.3:  # Minimum confidence threshold
                            triple = SemanticTriple(
                                subject=subject,
                                predicate=self.predicates[triple_type],
                                object=obj,
                                confidence=confidence,
                                source_file=source_file,
                                source_text=sentence[:200],
                                triple_type=triple_type
                            )
                            triples.append(triple)
                except Exception as e:
                    logger.debug(f"Pattern match error: {e}")
                    continue
        
        return triples
    
    def _calculate_confidence(self, subject: str, obj: str, sentence: str) -> float:
        """Calculate confidence score for an extracted triple."""
        confidence = 0.5
        
        # Boost confidence for proper nouns (capitalized)
        if subject[0].isupper():
            confidence += 0.1
        if obj and obj[0].isupper():
            confidence += 0.1
        
        # Boost for reasonable lengths
        if 3 < len(subject) < 50:
            confidence += 0.1
        if 3 < len(obj) < 50:
            confidence += 0.1
        
        # Penalize very long objects (likely extracted too much)
        if len(obj) > 80:
            confidence -= 0.2
        
        # Boost for technical terms (contains numbers, underscores, or camelCase)
        if re.search(r'[A-Z][a-z]+[A-Z]|_|\d', subject):
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def extract_from_text(self, text: str, source_file: str) -> List[SemanticTriple]:
        """Extract triples from a full text document."""
        triples = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences:
            sentence_triples = self.extract_from_sentence(sentence, source_file)
            triples.extend(sentence_triples)
        
        # Deduplicate triples
        seen = set()
        unique_triples = []
        for t in triples:
            key = (t.subject.lower(), t.predicate, t.object.lower())
            if key not in seen:
                seen.add(key)
                unique_triples.append(t)
        
        return unique_triples
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text (simple heuristic approach)."""
        entities = set()
        
        # Capitalized words/phrases (potential proper nouns)
        caps_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for match in re.finditer(caps_pattern, text):
            entity = match.group(1)
            if len(entity) > 2 and entity.lower() not in ['the', 'this', 'that', 'these', 'those']:
                entities.add(entity)
        
        # Technical terms (CamelCase, snake_case, with numbers)
        tech_pattern = r'\b([A-Z][a-z]+[A-Z][A-Za-z]*|[a-z]+_[a-z_]+|\w+\d+\w*)\b'
        for match in re.finditer(tech_pattern, text):
            entities.add(match.group(1))
        
        # Quoted terms
        quote_pattern = r'["\']([^"\']+)["\']'
        for match in re.finditer(quote_pattern, text):
            if len(match.group(1)) > 2:
                entities.add(match.group(1))
        
        return list(entities)


class SemanticTripleStore:
    """Store and query semantic triples in Qdrant."""
    
    def __init__(
        self,
        qdrant_host: str = "192.168.86.160",
        qdrant_port: int = 6333,
        collection_name: str = "semantic_triples",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize extractor
        self.extractor = TripleExtractor()
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' exists")
        except Exception:
            logger.info(f"Creating collection '{self.collection_name}'")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
    
    def _generate_triple_id(self, triple: SemanticTriple) -> str:
        """Generate unique ID for a triple."""
        content = f"{triple.subject}|{triple.predicate}|{triple.object}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _triple_to_text(self, triple: SemanticTriple) -> str:
        """Convert triple to natural language for embedding."""
        predicate_text = triple.predicate.replace('_', ' ')
        return f"{triple.subject} {predicate_text} {triple.object}"
    
    def process_document(self, text: str, source_file: str) -> List[SemanticTriple]:
        """Extract triples from a document and store them."""
        logger.info(f"Processing document: {source_file}")
        
        # Extract triples
        triples = self.extractor.extract_from_text(text, source_file)
        logger.info(f"Extracted {len(triples)} triples from {source_file}")
        
        if not triples:
            return []
        
        # Generate embeddings for triples
        triple_texts = [self._triple_to_text(t) for t in triples]
        embeddings = self.embedding_model.encode(triple_texts, show_progress_bar=False)
        
        # Create points
        points = []
        for triple, embedding in zip(triples, embeddings):
            point = PointStruct(
                id=self._generate_triple_id(triple),
                vector=embedding.tolist(),
                payload={
                    'subject': triple.subject,
                    'predicate': triple.predicate,
                    'object': triple.object,
                    'confidence': triple.confidence,
                    'source_file': triple.source_file,
                    'source_text': triple.source_text,
                    'triple_type': triple.triple_type,
                    'triple_text': self._triple_to_text(triple)
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(points)} triples in Qdrant")
        
        return triples
    
    def process_documents_folder(self, folder_path: str) -> int:
        """Process all documents in a folder."""
        from pathlib import Path
        
        folder = Path(folder_path)
        if not folder.exists():
            logger.error(f"Folder not found: {folder_path}")
            return 0
        
        total_triples = 0
        text_extensions = {'.txt', '.md', '.rst'}
        
        for file_path in folder.rglob('*'):
            if file_path.suffix.lower() in text_extensions:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                    
                    triples = self.process_document(text, str(file_path))
                    total_triples += len(triples)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Total triples extracted: {total_triples}")
        return total_triples
    
    def process_from_qdrant(self, source_collection: str = "documents") -> int:
        """
        Process documents directly from Qdrant vector database.
        
        This reads the 'content' field from each point in the source collection
        and extracts triples from it.
        
        Args:
            source_collection: Name of the Qdrant collection containing documents
            
        Returns:
            Total number of triples extracted
        """
        logger.info(f"Fetching documents from Qdrant collection: {source_collection}")
        
        # Verify collection exists
        try:
            collection_info = self.qdrant_client.get_collection(source_collection)
            total_points = collection_info.points_count
            logger.info(f"Collection '{source_collection}' has {total_points} points")
        except Exception as e:
            logger.error(f"Collection '{source_collection}' not found: {e}")
            return 0
        
        total_triples = 0
        processed_docs = set()  # Track unique documents to avoid duplicates
        offset = None
        batch_size = 100
        points_processed = 0
        
        while True:
            results, next_offset = self.qdrant_client.scroll(
                collection_name=source_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need vectors, just content
            )
            
            if not results:
                break
            
            for point in results:
                payload = point.payload
                content = payload.get('content', '')
                file_name = payload.get('file_name', f'chunk_{point.id}')
                chunk_index = payload.get('chunk_index', 0)
                
                # Create unique identifier for this chunk
                doc_id = f"{file_name}:chunk{chunk_index}"
                
                if doc_id in processed_docs:
                    continue
                processed_docs.add(doc_id)
                
                if not content or len(content.strip()) < 20:
                    continue
                
                try:
                    triples = self.process_document(content, file_name)
                    total_triples += len(triples)
                    points_processed += 1
                    
                    if points_processed % 50 == 0:
                        logger.info(f"Processed {points_processed} chunks, extracted {total_triples} triples...")
                        
                except Exception as e:
                    logger.error(f"Error processing {doc_id}: {e}")
            
            if next_offset is None:
                break
            offset = next_offset
        
        logger.info(f"Finished processing {points_processed} chunks from Qdrant")
        logger.info(f"Total triples extracted: {total_triples}")
        return total_triples
    
    def query_by_subject(self, subject: str, limit: int = 10) -> List[Dict]:
        """Find all triples with a given subject."""
        results, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="subject", match=MatchValue(value=subject))]
            ),
            limit=limit,
            with_payload=True
        )
        return [r.payload for r in results]
    
    def query_by_object(self, obj: str, limit: int = 10) -> List[Dict]:
        """Find all triples with a given object."""
        results, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="object", match=MatchValue(value=obj))]
            ),
            limit=limit,
            with_payload=True
        )
        return [r.payload for r in results]
    
    def query_semantic(self, query: str, limit: int = 10) -> List[Dict]:
        """Semantic search for triples."""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        return [
            {**r.payload, 'score': r.score}
            for r in results
        ]
    
    def get_entity_graph(self, entity: str, depth: int = 2) -> Dict:
        """Get the subgraph around an entity (breadth-first traversal)."""
        visited = set()
        nodes = []
        edges = []
        queue = [(entity, 0)]
        
        while queue:
            current_entity, current_depth = queue.pop(0)
            
            if current_entity in visited or current_depth > depth:
                continue
            
            visited.add(current_entity)
            nodes.append({'id': current_entity, 'depth': current_depth})
            
            # Find triples where entity is subject
            subject_triples = self.query_by_subject(current_entity, limit=20)
            for t in subject_triples:
                edges.append({
                    'source': t['subject'],
                    'target': t['object'],
                    'predicate': t['predicate'],
                    'type': t['triple_type']
                })
                if t['object'] not in visited:
                    queue.append((t['object'], current_depth + 1))
            
            # Find triples where entity is object
            object_triples = self.query_by_object(current_entity, limit=20)
            for t in object_triples:
                edges.append({
                    'source': t['subject'],
                    'target': t['object'],
                    'predicate': t['predicate'],
                    'type': t['triple_type']
                })
                if t['subject'] not in visited:
                    queue.append((t['subject'], current_depth + 1))
        
        return {'nodes': nodes, 'edges': edges}
    
    def get_stats(self) -> Dict:
        """Get statistics about the triple store."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            # Count by triple type
            type_counts = {}
            for triple_type in ['definition', 'property', 'action', 'relationship', 'hierarchy', 'causation', 'location']:
                results, _ = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="triple_type", match=MatchValue(value=triple_type))]
                    ),
                    limit=10000,
                    with_payload=False
                )
                type_counts[triple_type] = len(results)
            
            return {
                'total_triples': collection_info.points_count,
                'by_type': type_counts
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def export_to_json(self, output_path: str = "triples_export.json"):
        """Export all triples to JSON."""
        all_triples = []
        offset = None
        
        while True:
            results, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True
            )
            
            if not results:
                break
            
            for r in results:
                all_triples.append(r.payload)
            
            if next_offset is None:
                break
            offset = next_offset
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_triples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(all_triples)} triples to {output_path}")


def main():
    """Main entry point."""
    qdrant_host = os.getenv("QDRANT_HOST", "192.168.86.160")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("TRIPLE_COLLECTION", "semantic_triples")
    source_collection = os.getenv("SOURCE_COLLECTION", "documents")
    use_qdrant = os.getenv("USE_QDRANT_SOURCE", "true").lower() == "true"
    data_folder = os.getenv("DATA_FOLDER", "../DataInput/loaded data")
    
    logger.info("=" * 60)
    logger.info("SEMANTIC TRIPLE EXTRACTOR")
    logger.info("=" * 60)
    logger.info(f"Qdrant: {qdrant_host}:{qdrant_port}")
    logger.info(f"Triple Collection: {collection_name}")
    if use_qdrant:
        logger.info(f"Source: Qdrant collection '{source_collection}'")
    else:
        logger.info(f"Source: Local folder '{data_folder}'")
    
    # Initialize store
    store = SemanticTripleStore(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name
    )
    
    # Process documents from Qdrant or local folder
    if use_qdrant:
        total = store.process_from_qdrant(source_collection)
    else:
        total = store.process_documents_folder(data_folder)
    
    # Print stats
    stats = store.get_stats()
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total triples: {stats.get('total_triples', 0)}")
    print("\nBy type:")
    for t, count in stats.get('by_type', {}).items():
        print(f"  {t}: {count}")
    
    # Export
    store.export_to_json("triples_export.json")
    
    return store


if __name__ == "__main__":
    main()
