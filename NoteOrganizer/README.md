# Note Organizer

Cluster your notes from Qdrant using HDBSCAN unsupervised learning to discover natural topic groupings.

## Overview

This tool:
1. Queries all document vectors and payloads from your Qdrant collection
2. Uses UMAP for dimensionality reduction (improves clustering quality)
3. Applies HDBSCAN clustering to find natural topic groupings
4. Generates a report showing how your notes cluster together

## Installation

```bash
cd NoteOrganizer
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python note_clusterer.py
```

### Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `documents` | Collection name to cluster |
| `MIN_CLUSTER_SIZE` | `3` | Minimum points to form a cluster |
| `MIN_SAMPLES` | `2` | Core samples for density estimation |

### Example with Custom Settings

```bash
# Windows PowerShell
$env:QDRANT_HOST = "localhost"
$env:MIN_CLUSTER_SIZE = "5"
python note_clusterer.py

# Linux/Mac
QDRANT_HOST=localhost MIN_CLUSTER_SIZE=5 python note_clusterer.py
```

## Output

The tool generates:

1. **Console Report** - Human-readable cluster summary
2. **cluster_results.json** - Detailed JSON export with:
   - Cluster summaries (documents per cluster, sample content)
   - Per-point assignments with membership probabilities
   - Configuration used

## Understanding HDBSCAN Parameters

- **min_cluster_size**: Minimum number of points to form a cluster. Larger = fewer, bigger clusters
- **min_samples**: How conservative the clustering is. Larger = more points marked as noise

### Tuning Tips

- If you get too few clusters: decrease `min_cluster_size`
- If you get too many noise points: decrease `min_samples`
- If clusters seem too broad: increase `min_cluster_size`

## Programmatic Usage

```python
from note_clusterer import NoteClusterer

# Initialize
clusterer = NoteClusterer(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="documents",
    min_cluster_size=3
)

# Run clustering pipeline
clusterer.fetch_all_points()
clusterer.reduce_dimensions(n_components=10)
clusterer.cluster(use_reduced=True)

# Get results
summary = clusterer.get_cluster_summary()
results = clusterer.get_cluster_results()

# Export
clusterer.export_results("my_clusters.json")
```

## Next Steps

After clustering, you can:
1. Review the cluster report to understand topic groupings
2. Use cluster assignments to build a semantic graph in Qdrant
3. Create topic labels based on cluster content
4. Identify outlier notes that don't fit any category
