"""
Step 5: Project protein embeddings to 3D with cuML UMAP; write layout TSV.

this is without decompression, see 6.

Inputs: output/embeddings.npy (or .parquet), output/node_map.parquet.
Outputs: output/layout.tsv (node_id, x, y, z) for proteins only.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from config_tune import (
    EMBEDDINGS_PARQUET,
    NODE_MAP_PARQUET,
    LAYOUT_TSV,
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST,
    UMAP_RANDOM_STATE,
)
from gpu_check import require_rapids_gpu
from utils.normalize_layout import normalize_to_cube


def main():
    require_rapids_gpu()
    from cuml.manifold import UMAP

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

    # Subset to protein embeddings; move to GPU for cuML
    emb_protein = emb[protein_indices]
    import cudf
    emb_gpu = cudf.DataFrame(emb_protein)

    # cuML UMAP on GPU
    umap = UMAP(
        n_components=3,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    with tqdm(total=1, desc="UMAP 3D projection", unit=" run", bar_format="{desc}: {bar}{postfix}") as pbar:
        pbar.set_postfix_str(f"n={len(emb_protein)}")
        coords_3d = umap.fit_transform(emb_gpu)
        pbar.update(1)
    # fit_transform may return cudf DataFrame, cupy, or numpy
    if hasattr(coords_3d, "to_pandas"):
        coords_3d = np.asarray(coords_3d.to_pandas().values, dtype=np.float64)
    elif hasattr(coords_3d, "get"):
        coords_3d = np.asarray(coords_3d.get(), dtype=np.float64)
    else:
        coords_3d = np.asarray(coords_3d, dtype=np.float64)

    coords_3d = normalize_to_cube(coords_3d, method="percentile", low=0.5, high=99.5)

    # Write layout TSV: node_id, x, y, z
    layout_df = pd.DataFrame({
        "node_id": protein_str_ids,
        "x": coords_3d[:, 0],
        "y": coords_3d[:, 1],
        "z": coords_3d[:, 2],
    })
    layout_df.to_csv(LAYOUT_TSV, sep="\t", index=False)
    print(f"Wrote {LAYOUT_TSV} ({len(layout_df)} rows)")
    print("Step 5 done.")


if __name__ == "__main__":
    main()
