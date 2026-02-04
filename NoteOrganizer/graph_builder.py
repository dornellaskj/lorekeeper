"""
Knowledge Graph Builder - Create semantic graph in Qdrant from clustering results
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from itertools import combinations
import hashlib

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    Filter, FieldCondition, MatchValue,
    UpdateStatus, PayloadSchemaType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    metadata: Dict[str, Any]


class KnowledgeGraphBuilder:
    """Builds a semantic knowledge graph in Qdrant from clustering results."""
    
    def __init__(
        self,
        qdrant_host: str = "192.168.86.160",
        qdrant_port: int = 6333,
        source_collection: str = "documents",
        graph_collection: str = "knowledge_graph",
        cluster_results_path: str = "cluster_results.json"
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.source_collection = source_collection
        self.graph_collection = graph_collection
        self.cluster_results_path = cluster_results_path
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._test_connection()
        
        # Load cluster results
        self.cluster_data = self._load_cluster_results()
        
        # Storage
        self.points_by_id: Dict[str, Dict] = {}
        self.vectors_by_id: Dict[str, np.ndarray] = {}
        self.cluster_centroids: Dict[int, np.ndarray] = {}
    
    def _test_connection(self):
        """Test connection to Qdrant."""
        try:
            collections = self.qdrant_client.get_collections()
            logger.info(f"Connected to Qdrant. Found {len(collections.collections)} collections.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _load_cluster_results(self) -> Dict:
        """Load clustering results from JSON file."""
        try:
            with open(self.cluster_results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded cluster results: {len(data.get('points', []))} points")
            return data
        except FileNotFoundError:
            logger.error(f"Cluster results file not found: {self.cluster_results_path}")
            raise
    
    def _generate_edge_id(self, source_id: str, target_id: str, edge_type: str) -> str:
        """Generate unique ID for an edge."""
        # Sort IDs to ensure consistent edge ID regardless of direction
        sorted_ids = sorted([source_id, target_id])
        content = f"{sorted_ids[0]}_{sorted_ids[1]}_{edge_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_node_id(self, node_type: str, identifier: str) -> str:
        """Generate unique ID for a graph node."""
        content = f"{node_type}_{identifier}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def fetch_source_data(self):
        """Fetch all points from source collection."""
        logger.info(f"Fetching data from '{self.source_collection}'...")
        
        offset = None
        batch_size = 100
        
        while True:
            results, next_offset = self.qdrant_client.scroll(
                collection_name=self.source_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not results:
                break
            
            for point in results:
                self.points_by_id[str(point.id)] = point.payload
                self.vectors_by_id[str(point.id)] = np.array(point.vector)
            
            if next_offset is None:
                break
            offset = next_offset
        
        logger.info(f"Fetched {len(self.points_by_id)} points from source collection")
    
    def compute_cluster_centroids(self):
        """Compute centroid vectors for each cluster."""
        logger.info("Computing cluster centroids...")
        
        # Group points by cluster
        cluster_points: Dict[int, List[str]] = {}
        for point in self.cluster_data.get('points', []):
            cluster_id = point['cluster_id']
            if cluster_id not in cluster_points:
                cluster_points[cluster_id] = []
            cluster_points[cluster_id].append(point['point_id'])
        
        # Compute centroids
        for cluster_id, point_ids in cluster_points.items():
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            vectors = []
            for pid in point_ids:
                if pid in self.vectors_by_id:
                    vectors.append(self.vectors_by_id[pid])
            
            if vectors:
                centroid = np.mean(vectors, axis=0)
                self.cluster_centroids[cluster_id] = centroid
                logger.info(f"Cluster {cluster_id}: centroid computed from {len(vectors)} vectors")
        
        logger.info(f"Computed {len(self.cluster_centroids)} cluster centroids")
    
    def create_graph_collection(self, vector_size: int):
        """Create the knowledge graph collection."""
        try:
            # Check if collection exists
            self.qdrant_client.get_collection(self.graph_collection)
            logger.info(f"Collection '{self.graph_collection}' already exists, will add to it")
            return
        except Exception:
            pass
        
        logger.info(f"Creating graph collection '{self.graph_collection}'...")
        
        self.qdrant_client.create_collection(
            collection_name=self.graph_collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        
        logger.info(f"Created collection '{self.graph_collection}'")
    
    def build_graph(
        self,
        create_cluster_nodes: bool = True,
        create_document_nodes: bool = True,
        create_intra_cluster_edges: bool = True,
        similarity_threshold: float = 0.7,
        max_edges_per_node: int = 5
    ):
        """
        Build the knowledge graph with various node and edge types.
        
        Args:
            create_cluster_nodes: Create nodes representing cluster centroids
            create_document_nodes: Create nodes for unique documents
            create_intra_cluster_edges: Create edges between similar items in same cluster
            similarity_threshold: Min similarity for creating edges
            max_edges_per_node: Max number of edges per node
        """
        # Fetch source data first
        self.fetch_source_data()
        
        if not self.points_by_id:
            logger.error("No source data found!")
            return
        
        # Get vector size from first point
        vector_size = len(next(iter(self.vectors_by_id.values())))
        
        # Compute centroids
        self.compute_cluster_centroids()
        
        # Create graph collection
        self.create_graph_collection(vector_size)
        
        all_points = []
        
        # 1. Create cluster topic nodes (centroids)
        if create_cluster_nodes and self.cluster_centroids:
            logger.info("Creating cluster topic nodes...")
            cluster_points = self._create_cluster_nodes()
            all_points.extend(cluster_points)
        
        # 2. Create document nodes (aggregated by file)
        if create_document_nodes:
            logger.info("Creating document nodes...")
            doc_points = self._create_document_nodes()
            all_points.extend(doc_points)
        
        # 3. Create edge nodes (relationships)
        if create_intra_cluster_edges:
            logger.info("Creating relationship edges...")
            edge_points = self._create_relationship_edges(
                similarity_threshold=similarity_threshold,
                max_edges_per_node=max_edges_per_node
            )
            all_points.extend(edge_points)
        
        # Upload all points
        if all_points:
            self._upload_points(all_points)
        
        # Update source collection with cluster metadata
        self._update_source_with_clusters()
        
        logger.info("Knowledge graph building complete!")
        self._print_graph_stats()
    
    def _create_cluster_nodes(self) -> List[PointStruct]:
        """Create nodes representing cluster topics."""
        points = []
        summary = self.cluster_data.get('summary', {})
        
        for cluster_id, centroid in self.cluster_centroids.items():
            cluster_info = summary.get(str(cluster_id), {})
            
            # Get representative content from cluster
            cluster_docs = cluster_info.get('documents', [])
            cluster_titles = cluster_info.get('document_titles', [])
            
            point = PointStruct(
                id=self._generate_node_id('cluster', str(cluster_id)),
                vector=centroid.tolist(),
                payload={
                    'node_type': 'cluster_topic',
                    'cluster_id': cluster_id,
                    'num_chunks': cluster_info.get('num_points', 0),
                    'num_documents': cluster_info.get('num_documents', 0),
                    'documents': cluster_docs[:10],  # Limit for payload size
                    'document_titles': cluster_titles[:10],
                    'sample_content': cluster_info.get('sample_content', [])[:3],
                    'label': f"Topic Cluster {cluster_id}",
                    'description': f"Cluster containing {cluster_info.get('num_documents', 0)} documents about: {', '.join(cluster_titles[:3])}"
                }
            )
            points.append(point)
        
        logger.info(f"Created {len(points)} cluster topic nodes")
        return points
    
    def _create_document_nodes(self) -> List[PointStruct]:
        """Create nodes representing unique documents."""
        points = []
        
        # Group chunks by document
        doc_chunks: Dict[str, List[Tuple[str, np.ndarray, Dict]]] = {}
        
        for point_id, payload in self.points_by_id.items():
            file_name = payload.get('file_name', 'unknown')
            if file_name not in doc_chunks:
                doc_chunks[file_name] = []
            doc_chunks[file_name].append((
                point_id,
                self.vectors_by_id.get(point_id),
                payload
            ))
        
        # Create document nodes with averaged vectors
        for file_name, chunks in doc_chunks.items():
            vectors = [c[1] for c in chunks if c[1] is not None]
            if not vectors:
                continue
            
            # Average vector for document
            doc_vector = np.mean(vectors, axis=0)
            
            # Get cluster assignment (most common among chunks)
            chunk_clusters = []
            for point_data in self.cluster_data.get('points', []):
                if any(c[0] == point_data['point_id'] for c in chunks):
                    chunk_clusters.append(point_data['cluster_id'])
            
            primary_cluster = max(set(chunk_clusters), key=chunk_clusters.count) if chunk_clusters else -1
            
            # Get sample content
            sample_payload = chunks[0][2]
            
            point = PointStruct(
                id=self._generate_node_id('document', file_name),
                vector=doc_vector.tolist(),
                payload={
                    'node_type': 'document',
                    'file_name': file_name,
                    'file_path': sample_payload.get('file_path', ''),
                    'document_title': sample_payload.get('document_title', file_name),
                    'num_chunks': len(chunks),
                    'primary_cluster': primary_cluster,
                    'chunk_ids': [c[0] for c in chunks],
                    'label': sample_payload.get('document_title', file_name)
                }
            )
            points.append(point)
        
        logger.info(f"Created {len(points)} document nodes")
        return points
    
    def _create_relationship_edges(
        self,
        similarity_threshold: float = 0.7,
        max_edges_per_node: int = 5
    ) -> List[PointStruct]:
        """Create edge nodes representing relationships between documents."""
        points = []
        
        # Group documents by cluster
        cluster_docs: Dict[int, List[Dict]] = {}
        for point in self.cluster_data.get('points', []):
            cluster_id = point['cluster_id']
            if cluster_id == -1:  # Skip noise
                continue
            if cluster_id not in cluster_docs:
                cluster_docs[cluster_id] = []
            cluster_docs[cluster_id].append(point)
        
        # Create edges within clusters
        edges_created = set()
        
        for cluster_id, docs in cluster_docs.items():
            # Get unique documents in cluster
            doc_ids = list(set(d['point_id'] for d in docs))
            
            if len(doc_ids) < 2:
                continue
            
            # Compute pairwise similarities
            similarities = []
            for id1, id2 in combinations(doc_ids, 2):
                if id1 in self.vectors_by_id and id2 in self.vectors_by_id:
                    v1 = self.vectors_by_id[id1]
                    v2 = self.vectors_by_id[id2]
                    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if sim >= similarity_threshold:
                        similarities.append((id1, id2, float(sim)))
            
            # Sort by similarity and take top edges
            similarities.sort(key=lambda x: x[2], reverse=True)
            
            # Track edges per node
            node_edge_count: Dict[str, int] = {}
            
            for source_id, target_id, similarity in similarities:
                # Check edge limits
                if node_edge_count.get(source_id, 0) >= max_edges_per_node:
                    continue
                if node_edge_count.get(target_id, 0) >= max_edges_per_node:
                    continue
                
                edge_key = tuple(sorted([source_id, target_id]))
                if edge_key in edges_created:
                    continue
                
                edges_created.add(edge_key)
                node_edge_count[source_id] = node_edge_count.get(source_id, 0) + 1
                node_edge_count[target_id] = node_edge_count.get(target_id, 0) + 1
                
                # Create edge vector (average of connected nodes)
                edge_vector = (self.vectors_by_id[source_id] + self.vectors_by_id[target_id]) / 2
                
                # Get payloads
                source_payload = self.points_by_id.get(source_id, {})
                target_payload = self.points_by_id.get(target_id, {})
                
                point = PointStruct(
                    id=self._generate_edge_id(source_id, target_id, 'similar'),
                    vector=edge_vector.tolist(),
                    payload={
                        'node_type': 'edge',
                        'edge_type': 'semantic_similarity',
                        'source_id': source_id,
                        'target_id': target_id,
                        'source_file': source_payload.get('file_name', ''),
                        'target_file': target_payload.get('file_name', ''),
                        'source_title': source_payload.get('document_title', ''),
                        'target_title': target_payload.get('document_title', ''),
                        'similarity': similarity,
                        'cluster_id': cluster_id,
                        'label': f"{source_payload.get('document_title', '')[:20]} <-> {target_payload.get('document_title', '')[:20]}"
                    }
                )
                points.append(point)
        
        logger.info(f"Created {len(points)} relationship edges")
        return points
    
    def _update_source_with_clusters(self):
        """Update source collection points with cluster metadata."""
        logger.info("Updating source collection with cluster metadata...")
        
        # Build cluster lookup
        cluster_lookup = {}
        for point in self.cluster_data.get('points', []):
            cluster_lookup[point['point_id']] = {
                'cluster_id': point['cluster_id'],
                'membership_probability': point.get('membership_probability', 1.0)
            }
        
        # Update in batches
        batch_size = 100
        updates = []
        
        for point_id, cluster_info in cluster_lookup.items():
            updates.append({
                'id': point_id,
                'payload': {
                    'cluster_id': cluster_info['cluster_id'],
                    'cluster_probability': cluster_info['membership_probability']
                }
            })
            
            if len(updates) >= batch_size:
                self._batch_update_payloads(updates)
                updates = []
        
        if updates:
            self._batch_update_payloads(updates)
        
        logger.info("Source collection updated with cluster metadata")
    
    def _batch_update_payloads(self, updates: List[Dict]):
        """Update payloads in batch."""
        for update in updates:
            try:
                self.qdrant_client.set_payload(
                    collection_name=self.source_collection,
                    payload=update['payload'],
                    points=[update['id']]
                )
            except Exception as e:
                logger.warning(f"Failed to update point {update['id']}: {e}")
    
    def _upload_points(self, points: List[PointStruct]):
        """Upload points to graph collection."""
        batch_size = 100
        total = len(points)
        
        logger.info(f"Uploading {total} points to graph collection...")
        
        for i in range(0, total, batch_size):
            batch = points[i:i + batch_size]
            try:
                self.qdrant_client.upsert(
                    collection_name=self.graph_collection,
                    points=batch
                )
                logger.info(f"Uploaded {min(i + batch_size, total)}/{total} points")
            except Exception as e:
                logger.error(f"Failed to upload batch: {e}")
                raise
    
    def _print_graph_stats(self):
        """Print statistics about the created graph."""
        try:
            graph_info = self.qdrant_client.get_collection(self.graph_collection)
            source_info = self.qdrant_client.get_collection(self.source_collection)
            
            print("\n" + "=" * 60)
            print("KNOWLEDGE GRAPH STATISTICS")
            print("=" * 60)
            print(f"\nSource Collection: '{self.source_collection}'")
            print(f"  Total chunks: {source_info.points_count}")
            
            print(f"\nGraph Collection: '{self.graph_collection}'")
            print(f"  Total graph nodes: {graph_info.points_count}")
            
            # Count node types
            node_types = {'cluster_topic': 0, 'document': 0, 'edge': 0}
            
            offset = None
            while True:
                results, next_offset = self.qdrant_client.scroll(
                    collection_name=self.graph_collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not results:
                    break
                
                for point in results:
                    node_type = point.payload.get('node_type', 'unknown')
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                if next_offset is None:
                    break
                offset = next_offset
            
            print(f"\nNode Types:")
            print(f"  Cluster Topics: {node_types.get('cluster_topic', 0)}")
            print(f"  Documents: {node_types.get('document', 0)}")
            print(f"  Edges: {node_types.get('edge', 0)}")
            print("=" * 60)
            
        except Exception as e:
            logger.warning(f"Could not print stats: {e}")
    
    def query_related(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Query the knowledge graph for related content.
        
        This is a placeholder - you'd need an embedding model to encode the query.
        """
        logger.info(f"Query functionality requires embedding model integration")
        return []
    
    def export_graph_for_visualization(self, output_path: str = "graph_export.json"):
        """Export graph in a format suitable for visualization tools."""
        logger.info("Exporting graph for visualization...")
        
        nodes = []
        edges = []
        
        offset = None
        while True:
            results, next_offset = self.qdrant_client.scroll(
                collection_name=self.graph_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not results:
                break
            
            for point in results:
                payload = point.payload
                node_type = payload.get('node_type', 'unknown')
                
                if node_type == 'edge':
                    edges.append({
                        'id': str(point.id),
                        'source': payload.get('source_id'),
                        'target': payload.get('target_id'),
                        'type': payload.get('edge_type'),
                        'weight': payload.get('similarity', 1.0),
                        'label': payload.get('label', '')
                    })
                else:
                    nodes.append({
                        'id': str(point.id),
                        'type': node_type,
                        'label': payload.get('label', ''),
                        'cluster_id': payload.get('cluster_id', payload.get('primary_cluster', -1)),
                        'metadata': {
                            k: v for k, v in payload.items() 
                            if k not in ['node_type', 'label', 'chunk_ids']
                        }
                    })
            
            if next_offset is None:
                break
            offset = next_offset
        
        export_data = {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'total_nodes': len(nodes),
                'total_edges': len(edges)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Graph exported to {output_path}")
        logger.info(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")


def main():
    """Main entry point."""
    # Configuration
    qdrant_host = os.getenv("QDRANT_HOST", "192.168.86.160")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    source_collection = os.getenv("QDRANT_COLLECTION", "documents")
    graph_collection = os.getenv("GRAPH_COLLECTION", "knowledge_graph")
    cluster_results = os.getenv("CLUSTER_RESULTS", "cluster_results.json")
    
    # Graph building options
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    max_edges = int(os.getenv("MAX_EDGES_PER_NODE", "5"))
    
    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH BUILDER")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Qdrant: {qdrant_host}:{qdrant_port}")
    logger.info(f"  Source Collection: {source_collection}")
    logger.info(f"  Graph Collection: {graph_collection}")
    logger.info(f"  Cluster Results: {cluster_results}")
    logger.info(f"  Similarity Threshold: {similarity_threshold}")
    logger.info(f"  Max Edges Per Node: {max_edges}")
    
    # Build graph
    builder = KnowledgeGraphBuilder(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        source_collection=source_collection,
        graph_collection=graph_collection,
        cluster_results_path=cluster_results
    )
    
    builder.build_graph(
        create_cluster_nodes=True,
        create_document_nodes=True,
        create_intra_cluster_edges=True,
        similarity_threshold=similarity_threshold,
        max_edges_per_node=max_edges
    )
    
    # Export for visualization
    builder.export_graph_for_visualization("graph_export.json")
    
    return builder


if __name__ == "__main__":
    main()
