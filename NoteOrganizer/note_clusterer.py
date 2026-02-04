"""
Note Organizer - Cluster notes from Qdrant using HDBSCAN
"""

import os
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import hdbscan
from qdrant_client import QdrantClient
from sklearn.preprocessing import StandardScaler
import umap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Represents the result of clustering a document."""
    point_id: str
    cluster_id: int
    content: str
    file_name: str
    file_path: str
    document_title: str
    membership_probability: float


class NoteClusterer:
    """Clusters notes from Qdrant using HDBSCAN."""
    
    def __init__(
        self,
        qdrant_host: str = "192.168.86.160",
        qdrant_port: int = 6333,
        collection_name: str = "documents",
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean"
    ):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = collection_name
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._test_connection()
        
        # Storage for results
        self.points_data: List[Dict[str, Any]] = []
        self.vectors: np.ndarray = None
        self.cluster_labels: np.ndarray = None
        self.clusterer: hdbscan.HDBSCAN = None
        self.reduced_vectors: np.ndarray = None
    
    def _test_connection(self):
        """Test connection to Qdrant server."""
        try:
            collections = self.qdrant_client.get_collections()
            logger.info(f"Connected to Qdrant. Found {len(collections.collections)} collections.")
            
            # Verify our collection exists
            collection_names = [c.name for c in collections.collections]
            if self.collection_name not in collection_names:
                raise ValueError(f"Collection '{self.collection_name}' not found. Available: {collection_names}")
            
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' has {collection_info.points_count} points")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def fetch_all_points(self) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Fetch all points from the collection with vectors and payloads.
        
        Returns:
            Tuple of (list of point data dicts, numpy array of vectors)
        """
        logger.info(f"Fetching all points from collection '{self.collection_name}'...")
        
        all_points = []
        all_vectors = []
        
        # Use scroll to fetch all points (handles pagination automatically)
        offset = None
        batch_size = 100
        
        while True:
            results, next_offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            if not results:
                break
            
            for point in results:
                point_data = {
                    'id': point.id,
                    'payload': point.payload,
                    'content': point.payload.get('content', ''),
                    'file_name': point.payload.get('file_name', ''),
                    'file_path': point.payload.get('file_path', ''),
                    'document_title': point.payload.get('document_title', ''),
                    'chunk_index': point.payload.get('chunk_index', 0)
                }
                all_points.append(point_data)
                all_vectors.append(point.vector)
            
            logger.info(f"Fetched {len(all_points)} points so far...")
            
            if next_offset is None:
                break
            offset = next_offset
        
        self.points_data = all_points
        self.vectors = np.array(all_vectors)
        
        logger.info(f"Total points fetched: {len(all_points)}")
        logger.info(f"Vector shape: {self.vectors.shape}")
        
        return all_points, self.vectors
    
    def reduce_dimensions(self, n_components: int = 10, n_neighbors: int = 15) -> np.ndarray:
        """
        Reduce vector dimensions using UMAP for better clustering.
        
        HDBSCAN works better on lower-dimensional data.
        
        Args:
            n_components: Target dimensionality
            n_neighbors: UMAP neighborhood size
            
        Returns:
            Reduced dimension vectors
        """
        if self.vectors is None:
            raise ValueError("No vectors loaded. Call fetch_all_points() first.")
        
        logger.info(f"Reducing dimensions from {self.vectors.shape[1]} to {n_components} using UMAP...")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        self.reduced_vectors = reducer.fit_transform(self.vectors)
        logger.info(f"Reduced vectors shape: {self.reduced_vectors.shape}")
        
        return self.reduced_vectors
    
    def cluster(self, use_reduced: bool = True) -> np.ndarray:
        """
        Run HDBSCAN clustering on the vectors.
        
        Args:
            use_reduced: Whether to use dimension-reduced vectors
            
        Returns:
            Array of cluster labels (-1 = noise/unclustered)
        """
        if self.vectors is None:
            raise ValueError("No vectors loaded. Call fetch_all_points() first.")
        
        vectors_to_cluster = self.reduced_vectors if (use_reduced and self.reduced_vectors is not None) else self.vectors
        
        logger.info(f"Running HDBSCAN clustering on {vectors_to_cluster.shape[0]} points...")
        logger.info(f"Parameters: min_cluster_size={self.min_cluster_size}, min_samples={self.min_samples}")
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            cluster_selection_method='eom',  # Excess of Mass - good for varied density clusters
            prediction_data=True  # Enable soft clustering
        )
        
        self.cluster_labels = self.clusterer.fit_predict(vectors_to_cluster)
        
        # Statistics
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        logger.info(f"Clustering complete!")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Noise points (unclustered): {n_noise}")
        logger.info(f"  Clustered points: {len(self.cluster_labels) - n_noise}")
        
        return self.cluster_labels
    
    def get_cluster_results(self) -> List[ClusterResult]:
        """
        Get detailed clustering results for each point.
        
        Returns:
            List of ClusterResult objects
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering done. Call cluster() first.")
        
        results = []
        probabilities = self.clusterer.probabilities_ if self.clusterer else np.ones(len(self.cluster_labels))
        
        for i, (point_data, label) in enumerate(zip(self.points_data, self.cluster_labels)):
            result = ClusterResult(
                point_id=str(point_data['id']),
                cluster_id=int(label),
                content=point_data['content'][:200] + '...' if len(point_data['content']) > 200 else point_data['content'],
                file_name=point_data['file_name'],
                file_path=point_data['file_path'],
                document_title=point_data['document_title'],
                membership_probability=float(probabilities[i])
            )
            results.append(result)
        
        return results
    
    def get_cluster_summary(self) -> Dict[int, Dict[str, Any]]:
        """
        Get a summary of each cluster.
        
        Returns:
            Dictionary mapping cluster_id to cluster info
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering done. Call cluster() first.")
        
        cluster_summary = {}
        
        for cluster_id in set(self.cluster_labels):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_points = [p for p, m in zip(self.points_data, cluster_mask) if m]
            
            # Get unique documents in this cluster
            unique_docs = set(p['file_name'] for p in cluster_points)
            unique_titles = set(p['document_title'] for p in cluster_points)
            
            # Sample content from cluster
            sample_contents = [p['content'][:100] for p in cluster_points[:3]]
            
            cluster_summary[int(cluster_id)] = {
                'cluster_id': int(cluster_id),
                'is_noise': bool(cluster_id == -1),
                'num_points': int(sum(cluster_mask)),
                'num_documents': len(unique_docs),
                'documents': list(unique_docs),
                'document_titles': list(unique_titles),
                'sample_content': sample_contents
            }
        
        return cluster_summary
    
    def print_cluster_report(self):
        """Print a human-readable cluster report."""
        summary = self.get_cluster_summary()
        
        print("\n" + "=" * 80)
        print("CLUSTER ANALYSIS REPORT")
        print("=" * 80)
        
        # Sort clusters by size (excluding noise)
        sorted_clusters = sorted(
            [(k, v) for k, v in summary.items() if k != -1],
            key=lambda x: x[1]['num_points'],
            reverse=True
        )
        
        # Print noise first if it exists
        if -1 in summary:
            noise = summary[-1]
            print(f"\n🔸 UNCLUSTERED (Noise): {noise['num_points']} chunks from {noise['num_documents']} documents")
            print(f"   Documents: {', '.join(noise['documents'][:5])}{'...' if len(noise['documents']) > 5 else ''}")
        
        # Print each cluster
        for cluster_id, info in sorted_clusters:
            print(f"\n📁 CLUSTER {cluster_id}: {info['num_points']} chunks from {info['num_documents']} documents")
            print(f"   Documents: {', '.join(info['document_titles'][:5])}{'...' if len(info['document_titles']) > 5 else ''}")
            print(f"   Sample content:")
            for i, sample in enumerate(info['sample_content'][:2]):
                clean_sample = sample.replace('\n', ' ')[:80]
                print(f"     [{i+1}] \"{clean_sample}...\"")
        
        print("\n" + "=" * 80)
    
    def export_results(self, output_path: str = "cluster_results.json"):
        """Export clustering results to JSON file."""
        summary = self.get_cluster_summary()
        results = self.get_cluster_results()
        
        export_data = {
            'summary': summary,
            'config': {
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'metric': self.metric,
                'total_points': len(self.points_data),
                'vector_dimensions': self.vectors.shape[1] if self.vectors is not None else None
            },
            'points': [
                {
                    'point_id': r.point_id,
                    'cluster_id': r.cluster_id,
                    'file_name': r.file_name,
                    'document_title': r.document_title,
                    'membership_probability': r.membership_probability
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results exported to {output_path}")
    
    def create_2d_visualization_data(self) -> pd.DataFrame:
        """
        Create 2D UMAP projection for visualization.
        
        Returns:
            DataFrame with x, y coordinates and cluster info
        """
        logger.info("Creating 2D UMAP projection for visualization...")
        
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        
        coords_2d = reducer_2d.fit_transform(self.vectors)
        
        df = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'cluster': self.cluster_labels,
            'file_name': [p['file_name'] for p in self.points_data],
            'document_title': [p['document_title'] for p in self.points_data],
            'content_preview': [p['content'][:100] for p in self.points_data]
        })
        
        return df


def main():
    """Main entry point."""
    # Configuration from environment variables
    qdrant_host = os.getenv("QDRANT_HOST", "192.168.86.160")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = os.getenv("QDRANT_COLLECTION", "documents")
    min_cluster_size = int(os.getenv("MIN_CLUSTER_SIZE", "3"))
    min_samples = int(os.getenv("MIN_SAMPLES", "2"))
    
    logger.info("=" * 60)
    logger.info("NOTE ORGANIZER - HDBSCAN Clustering")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Qdrant: {qdrant_host}:{qdrant_port}")
    logger.info(f"  Collection: {collection_name}")
    logger.info(f"  Min Cluster Size: {min_cluster_size}")
    logger.info(f"  Min Samples: {min_samples}")
    
    # Initialize clusterer
    clusterer = NoteClusterer(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    
    # Fetch all points
    clusterer.fetch_all_points()
    
    # Reduce dimensions for better clustering
    clusterer.reduce_dimensions(n_components=10)
    
    # Run clustering
    clusterer.cluster(use_reduced=True)
    
    # Print report
    clusterer.print_cluster_report()
    
    # Export results
    clusterer.export_results("cluster_results.json")
    
    logger.info("Clustering complete! Results saved to cluster_results.json")
    
    return clusterer


if __name__ == "__main__":
    main()
