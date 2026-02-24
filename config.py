"""
GO2Vec pipeline configuration: paths and parameters.
Import this in each step script so paths stay consistent.
"""
from pathlib import Path

# Base paths (project root = directory containing config.py)
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Input files
EDGES_TSV = INPUT_DIR / "edges.tsv"
GO_OBO_PATH = DATA_DIR / "go-basic.obo"
GAF_PATH = DATA_DIR / "goa_human.gaf"

# Output files (intermediates and final)
EDGE_LIST_PARQUET = OUTPUT_DIR / "edge_list.parquet"
NODE_MAP_PARQUET = OUTPUT_DIR / "node_map.parquet"
WALKS_PARQUET = OUTPUT_DIR / "walks.parquet"
EMBEDDINGS_PARQUET = OUTPUT_DIR / "embeddings.parquet"
LAYOUT_TSV = OUTPUT_DIR / "layout.tsv"

# Node2vec parameters (approaches.md: 100 dims, 20 walks per node, 100 length per walk)
WALK_LENGTH = 100
WALKS_PER_NODE = 20
NODE2VEC_P = 1.0
NODE2VEC_Q = 1.0

# Embedding
EMBED_DIM = 100
SKIPGRAM_WINDOW = 5
SKIPGRAM_EPOCHS = 5
SKIPGRAM_BATCH_SIZE = 4096
SKIPGRAM_NEGATIVE_SAMPLES = 5
SKIPGRAM_LEARNING_RATE = 0.025
# Cap training pairs to avoid OOM (None = use all)
SKIPGRAM_MAX_PAIRS = 50_000_000

# 3D projection (cuML UMAP)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# Data URLs (for download step)
GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
# GAF: Gene Ontology Annotation (human) from GO consortium
GAF_URL = "https://geneontology.org/gene-associations/goa_human.gaf.gz"


def ensure_dirs():
    """Create data and output directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
