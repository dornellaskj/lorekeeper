"""
Triple-Based Query Engine
Queries semantic triples to answer questions with atomic facts
"""

import os
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripleQueryEngine:
    """Query engine that uses semantic triples for knowledge retrieval."""
    
    def __init__(
        self,
        qdrant_host: str = "192.168.86.160",
        qdrant_port: int = 6333,
        triple_collection: str = "semantic_triples",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.triple_collection = triple_collection
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Verify collection exists
        try:
            info = self.qdrant_client.get_collection(triple_collection)
            logger.info(f"Triple collection has {info.points_count} triples")
        except Exception as e:
            logger.warning(f"Triple collection not found: {e}")
    
    def query(self, question: str, limit: int = 10) -> Dict[str, Any]:
        """
        Query the triple store and return relevant facts.
        
        Returns structured response with:
        - Direct facts (high-confidence triples)
        - Related concepts (entities connected to the answer)
        - Source tracing (where each fact came from)
        """
        query_embedding = self.embedding_model.encode(question).tolist()
        
        # Search for relevant triples
        results = self.qdrant_client.query_points(
            collection_name=self.triple_collection,
            query=query_embedding,
            limit=limit * 2,  # Get more for filtering
            score_threshold=0.3
        ).points
        
        # Organize results
        facts = []
        entities = set()
        sources = set()
        
        for r in results:
            payload = r.payload
            
            # Format as readable fact
            predicate_text = payload['predicate'].replace('_', ' ')
            fact_text = f"{payload['subject']} {predicate_text} {payload['object']}"
            
            facts.append({
                'fact': fact_text,
                'subject': payload['subject'],
                'predicate': payload['predicate'],
                'object': payload['object'],
                'type': payload['triple_type'],
                'confidence': payload['confidence'],
                'score': r.score,
                'source': payload['source_file']
            })
            
            entities.add(payload['subject'])
            entities.add(payload['object'])
            sources.add(payload['source_file'])
        
        # Sort by combined score (relevance * confidence)
        facts.sort(key=lambda x: x['score'] * x['confidence'], reverse=True)
        facts = facts[:limit]
        
        # Get related facts for top entities (graph expansion)
        related_facts = []
        if facts:
            top_entities = [facts[0]['subject'], facts[0]['object']]
            for entity in top_entities[:2]:
                related = self._get_related_triples(entity, exclude_ids=set())
                related_facts.extend(related[:3])
        
        return {
            'facts': facts,
            'entities': list(entities),
            'sources': list(sources),
            'related_facts': related_facts
        }
    
    def _get_related_triples(self, entity: str, exclude_ids: set, limit: int = 5) -> List[Dict]:
        """Get triples related to an entity."""
        related = []
        
        # As subject
        results, _ = self.qdrant_client.scroll(
            collection_name=self.triple_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="subject", match=MatchValue(value=entity))]
            ),
            limit=limit,
            with_payload=True
        )
        
        for r in results:
            if r.id not in exclude_ids:
                predicate_text = r.payload['predicate'].replace('_', ' ')
                related.append({
                    'fact': f"{r.payload['subject']} {predicate_text} {r.payload['object']}",
                    'type': r.payload['triple_type']
                })
        
        # As object
        results, _ = self.qdrant_client.scroll(
            collection_name=self.triple_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="object", match=MatchValue(value=entity))]
            ),
            limit=limit,
            with_payload=True
        )
        
        for r in results:
            if r.id not in exclude_ids:
                predicate_text = r.payload['predicate'].replace('_', ' ')
                related.append({
                    'fact': f"{r.payload['subject']} {predicate_text} {r.payload['object']}",
                    'type': r.payload['triple_type']
                })
        
        return related
    
    def format_for_llm(self, query_result: Dict) -> str:
        """Format query results as context for LLM."""
        lines = ["## Relevant Facts:\n"]
        
        for i, fact in enumerate(query_result['facts'], 1):
            lines.append(f"{i}. {fact['fact']}")
            lines.append(f"   (Type: {fact['type']}, Confidence: {fact['confidence']:.2f})")
        
        if query_result.get('related_facts'):
            lines.append("\n## Related Information:")
            for fact in query_result['related_facts']:
                lines.append(f"- {fact['fact']}")
        
        if query_result.get('sources'):
            lines.append(f"\n## Sources: {', '.join(query_result['sources'][:3])}")
        
        return "\n".join(lines)
    
    def answer_question(self, question: str) -> str:
        """Generate a simple answer from triples (no LLM needed)."""
        result = self.query(question, limit=5)
        
        if not result['facts']:
            return "I don't have specific facts about that in my knowledge base."
        
        # Build answer from top facts
        answer_parts = []
        for fact in result['facts'][:3]:
            answer_parts.append(fact['fact'])
        
        answer = "Based on my knowledge:\n\n"
        answer += "\n".join(f"• {part}" for part in answer_parts)
        
        if result.get('related_facts'):
            answer += "\n\nRelated facts:\n"
            answer += "\n".join(f"• {f['fact']}" for f in result['related_facts'][:2])
        
        return answer


def main():
    """Test the query engine."""
    engine = TripleQueryEngine(
        qdrant_host=os.getenv("QDRANT_HOST", "192.168.86.160"),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333"))
    )
    
    # Interactive query mode
    print("\n" + "=" * 60)
    print("TRIPLE QUERY ENGINE")
    print("=" * 60)
    print("Enter questions to search the knowledge graph.")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("Question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        result = engine.query(question)
        
        print(f"\n📊 Found {len(result['facts'])} relevant facts:")
        print("-" * 40)
        
        for fact in result['facts']:
            print(f"  • {fact['fact']}")
            print(f"    [{fact['type']}] confidence: {fact['confidence']:.2f}, relevance: {fact['score']:.3f}")
        
        if result['related_facts']:
            print(f"\n🔗 Related facts:")
            for rf in result['related_facts'][:3]:
                print(f"  • {rf['fact']}")
        
        print()


if __name__ == "__main__":
    main()
