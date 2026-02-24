"""
Step 10: Add a generic adjacency-based UMAP layout to a final TSV.

Builds a protein-protein adjacency matrix from `edge_list.parquet`, reduces it with
TruncatedSVD, runs UMAP in 3D, and appends:
  x_umap_adjacency, y_umap_adjacency, z_umap_adjacency
to the target layout TSV.

Why SVD first?
Running UMAP directly on a 20k x 20k sparse adjacency can be very slow. SVD keeps
most structure while making UMAP much faster and more stable.

Usage:
  python 10_add_generic_umap_layout.py [layout_in.tsv] [layout_out.tsv]
  python 10_add_generic_umap_layout.py --svd-dim 64 --no-seed
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from config_tune import (
    EDGE_LIST_PARQUET,
    NODE_MAP_PARQUET,
    OUTPUT_DIR,
    LAYOUT_TSV,
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST,
    UMAP_RANDOM_STATE,
)

LAYOUT_MULTI_TSV = OUTPUT_DIR / "layout_multi.tsv"
LAYOUT_DECOMPRESSION_TSV = OUTPUT_DIR / "layout_decompression.tsv"


def choose_layout_path() -> Path:
    if LAYOUT_DECOMPRESSION_TSV.exists():
        return LAYOUT_DECOMPRESSION_TSV
    if LAYOUT_MULTI_TSV.exists():
        return LAYOUT_MULTI_TSV
    return LAYOUT_TSV


def build_protein_adjacency():
    node_map = pd.read_parquet(NODE_MAP_PARQUET)
    edges = pd.read_parquet(EDGE_LIST_PARQUET)

    protein_df = node_map.loc[node_map["node_type"] == "protein", ["int_id", "str_id"]].copy()
    protein_ids = protein_df["int_id"].to_numpy(dtype=np.int64)
    protein_str_ids = protein_df["str_id"].astype(str).to_numpy()
    n_prot = len(protein_df)

    idx_map = {nid: i for i, nid in enumerate(protein_ids)}

    src = edges["src"].to_numpy(dtype=np.int64)
    dst = edges["dst"].to_numpy(dtype=np.int64)
    src_in = np.isin(src, protein_ids)
    dst_in = np.isin(dst, protein_ids)
    mask = src_in & dst_in
    src_p = src[mask]
    dst_p = dst[mask]

    # Map global protein int_id -> local row index
    rows = np.fromiter((idx_map[s] for s in src_p), dtype=np.int64, count=len(src_p))
    cols = np.fromiter((idx_map[d] for d in dst_p), dtype=np.int64, count=len(dst_p))
    data = np.ones(len(rows), dtype=np.float32)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_prot, n_prot)).tocsr()
    # Ensure binary + no diagonal self-loop
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, protein_str_ids


def run_umap_on_adjacency(
    A: sparse.csr_matrix,
    svd_dim: int = 128,
    use_seed: bool = True,
):
    try:
        import umap
    except ImportError as e:
        raise RuntimeError(
            "This script requires 'umap-learn' for sparse adjacency UMAP.\n"
            "Install with: pip install umap-learn"
        ) from e

    n_nodes = A.shape[0]
    svd_dim = max(8, min(int(svd_dim), n_nodes - 1))
    print(f"Reducing sparse adjacency with TruncatedSVD -> {svd_dim} dims...")
    t0 = time.time()
    X = TruncatedSVD(
        n_components=svd_dim,
        random_state=UMAP_RANDOM_STATE if use_seed else None,
    ).fit_transform(A)
    print(f"SVD done in {time.time() - t0:.1f}s, shape={X.shape}")

    random_state = UMAP_RANDOM_STATE if use_seed else None
    if random_state is None:
        print("UMAP random_state=None (faster, multi-threaded, non-deterministic).")
    else:
        print("UMAP random_state fixed (deterministic, often slower).")

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        n_epochs=300,
        low_memory=True,
        verbose=True,
        random_state=random_state,
    )
    t1 = time.time()
    coords = reducer.fit_transform(X)
    print(f"UMAP done in {time.time() - t1:.1f}s")
    return np.asarray(coords, dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append adjacency-based 3D UMAP columns to final TSV."
    )
    parser.add_argument("layout_in", nargs="?", default=None)
    parser.add_argument("layout_out", nargs="?", default=None)
    parser.add_argument(
        "--svd-dim",
        type=int,
        default=128,
        help="TruncatedSVD dimension before UMAP (default: 128).",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Disable fixed random_state for faster multi-threaded UMAP.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    layout_in = Path(args.layout_in) if args.layout_in else choose_layout_path()
    layout_out = Path(args.layout_out) if args.layout_out else layout_in

    if not layout_in.exists():
        print(f"Layout input not found: {layout_in}", file=sys.stderr)
        sys.exit(1)

    print("Building protein adjacency matrix...")
    A, protein_str_ids = build_protein_adjacency()
    print(f"Adjacency shape: {A.shape}, nnz={A.nnz}")

    print("Running generic UMAP on adjacency features (SVD -> UMAP)...")
    coords = run_umap_on_adjacency(A, svd_dim=args.svd_dim, use_seed=not args.no_seed)

    adj_df = pd.DataFrame(
        {
            "node_id": protein_str_ids,
            "x_umap_adjacency": coords[:, 0],
            "y_umap_adjacency": coords[:, 1],
            "z_umap_adjacency": coords[:, 2],
        }
    )

    base_df = pd.read_csv(layout_in, sep="\t")
    # Replace if already present
    for c in ["x_umap_adjacency", "y_umap_adjacency", "z_umap_adjacency"]:
        if c in base_df.columns:
            base_df = base_df.drop(columns=[c])

    out_df = base_df.merge(adj_df, on="node_id", how="left")
    out_df.to_csv(layout_out, sep="\t", index=False)
    print(f"Wrote {layout_out} with adjacency UMAP columns.")


if __name__ == "__main__":
    main()

