from pathlib import Path

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

# Node2vec parameters (100 dims, 20 walks per node, 100 length per walk; see README)
# For de-compression: q > 1 favors structural equivalence (similar roles), reducing hub crowding
WALK_LENGTH = 100
WALKS_PER_NODE = 20
NODE2VEC_P = 1.0
NODE2VEC_Q = 1.5   # 1.0 = community; 1.5–2.0 = structural equivalence (better for layout spread?)

# Embedding - hopefully optimized for RTX 5090
EMBED_DIM = 100
SKIPGRAM_WINDOW = 5
SKIPGRAM_EPOCHS = 10
SKIPGRAM_BATCH_SIZE = 131072  
SKIPGRAM_NEGATIVE_SAMPLES = 5
SKIPGRAM_LEARNING_RATE = 0.001 
# 32gb vram, should work, .. maybe
SKIPGRAM_MAX_PAIRS = 50_000_000

# 3D projection (cuML UMAP)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# Data URLs
GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
# GAF: Gene Ontology Annotation (human) from GO consortium
GAF_URL = "https://geneontology.org/gene-associations/goa_human.gaf.gz"


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
