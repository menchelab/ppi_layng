"""
Multi-method 3D projection: 20 UMAP (cuML) variants + 3 PaCMAP variants.
Loads protein embeddings, runs each method, names columns x_<method>, y_<method>, z_<method>,
saves combined layout to output/layout_multi.tsv.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config_tune import (
    EMBEDDINGS_PARQUET,
    NODE_MAP_PARQUET,
    OUTPUT_DIR,
    UMAP_RANDOM_STATE,
)
from gpu_check import require_rapids_gpu

LAYOUT_MULTI_TSV = OUTPUT_DIR / "layout_multi.tsv"

# 20 UMAP parameter variants (n_neighbors, min_dist)
UMAP_VARIANTS = [
    (5, 0.1), (10, 0.1), (15, 0.1), (20, 0.1), (30, 0.1), (50, 0.1),
    (15, 0.0), (15, 0.25), (15, 0.5), (10, 0.25), (20, 0.25),
    (5, 0.5), (50, 0.0), (8, 0.15), (25, 0.15), (40, 0.2),
    (12, 0.05), (18, 0.3), (35, 0.05), (45, 0.35),
]

# 3 PaCMAP variants (n_neighbors, MN_ratio)
PACMAP_VARIANTS = [
    (10, 0.5),
    (25, 0.5),
    (10, 0.2),
]


def _to_cpu(coords):
    """Copy GPU result to numpy (avoid CUDA context loss in WSL2)."""
    if hasattr(coords, "to_pandas"):
        return np.asarray(coords.to_pandas().values, dtype=np.float64)
    if hasattr(coords, "get"):
        return np.asarray(coords.get(), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64)


def run_umap_variants(emb_protein):
    """Run 20 cuML UMAP variants on GPU; return dict method_name -> (n, 3) array."""
    require_rapids_gpu()
    import cudf
    from cuml.manifold import UMAP

    emb_gpu = cudf.DataFrame(emb_protein)
    out = {}
    for i, (n_neighbors, min_dist) in enumerate(tqdm(UMAP_VARIANTS, desc="UMAP variants", unit=" run")):
        name = f"umap_{i + 1}"
        umap = UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=UMAP_RANDOM_STATE,
        )
        coords = umap.fit_transform(emb_gpu)
        out[name] = _to_cpu(coords)
    return out


def run_pacmap_variants(emb_protein):
    """Run 3 PaCMAP variants (CPU)."""
    import pacmap

    out = {}
    for i, (n_neighbors, mn_ratio) in enumerate(tqdm(PACMAP_VARIANTS, desc="PaCMAP variants", unit=" run")):
        name = f"pacmap_{i + 1}"
        reducer = pacmap.PaCMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            MN_ratio=mn_ratio,
            random_state=UMAP_RANDOM_STATE,
        )
        coords = reducer.fit_transform(emb_protein)
        out[name] = np.asarray(coords, dtype=np.float64)
    return out


def main():
    # Load embeddings
    npy_path = EMBEDDINGS_PARQUET.with_suffix(".npy")
    if npy_path.exists():
        emb = np.load(npy_path).astype(np.float32)
    else:
        df = pd.read_parquet(EMBEDDINGS_PARQUET)
        cols = [c for c in df.columns if c.startswith("dim_")]
        emb = df[cols].values.astype(np.float32)
    print(f"Embeddings shape: {emb.shape}")

    node_map = pd.read_parquet(NODE_MAP_PARQUET)
    protein_mask = (node_map["node_type"] == "protein").values
    protein_indices = np.where(protein_mask)[0]
    protein_str_ids = node_map.loc[protein_mask, "str_id"].values
    print(f"Protein nodes: {len(protein_indices)}")

    emb_protein = emb[protein_indices]

    # Collect all layouts: method_name -> (n, 3)
    layouts = {}
    layouts.update(run_umap_variants(emb_protein))
    layouts.update(run_pacmap_variants(emb_protein))

    # Build one DataFrame: node_id, x_umap_1, y_umap_1, z_umap_1, ...
    data = {"node_id": protein_str_ids}
    for name, coords in layouts.items():
        data[f"x_{name}"] = coords[:, 0]
        data[f"y_{name}"] = coords[:, 1]
        data[f"z_{name}"] = coords[:, 2]

    layout_df = pd.DataFrame(data)
    LAYOUT_MULTI_TSV.parent.mkdir(parents=True, exist_ok=True)
    layout_df.to_csv(LAYOUT_MULTI_TSV, sep="\t", index=False)
    print(f"Wrote {LAYOUT_MULTI_TSV} ({len(layout_df)} rows, {len(layout_df.columns)} columns)")
    print("Step 5 (multi) done.")


if __name__ == "__main__":
    main()
