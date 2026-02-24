"""
Step 11: Add a generic adjacency-based PaCMAP layout to a final TSV.

Builds a protein-protein adjacency matrix from `edge_list.parquet`, reduces it with
TruncatedSVD, runs PaCMAP in 3D, and appends:
  x_pacmap_adjacency, y_pacmap_adjacency, z_pacmap_adjacency
to the target layout TSV.

Usage:
  python 11_add_generic_pacmap_layout.py [layout_in.tsv] [layout_out.tsv]
  python 11_add_generic_pacmap_layout.py --svd-dim 64 --n-neighbors 15
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from config_tune import EDGE_LIST_PARQUET, NODE_MAP_PARQUET, OUTPUT_DIR, LAYOUT_TSV

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
    mask = np.isin(src, protein_ids) & np.isin(dst, protein_ids)
    src_p = src[mask]
    dst_p = dst[mask]

    rows = np.fromiter((idx_map[s] for s in src_p), dtype=np.int64, count=len(src_p))
    cols = np.fromiter((idx_map[d] for d in dst_p), dtype=np.int64, count=len(dst_p))
    data = np.ones(len(rows), dtype=np.float32)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_prot, n_prot)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, protein_str_ids


def run_pacmap_on_adjacency(
    A: sparse.csr_matrix,
    svd_dim: int = 128,
    n_neighbors: int = 15,
    mn_ratio: float = 0.5,
    fp_ratio: float = 1.0,
    n_iters: int = 450,
    random_state: int = 42,
):
    try:
        import pacmap
    except ImportError as e:
        raise RuntimeError(
            "This script requires 'pacmap'. Install with: pip install pacmap"
        ) from e

    n_nodes = A.shape[0]
    svd_dim = max(8, min(int(svd_dim), n_nodes - 1))
    print(f"Reducing sparse adjacency with TruncatedSVD -> {svd_dim} dims...")
    t0 = time.time()
    X = TruncatedSVD(n_components=svd_dim, random_state=random_state).fit_transform(A)
    print(f"SVD done in {time.time() - t0:.1f}s, shape={X.shape}")

    print("Running PaCMAP (regular baseline, non-expansion)...")
    reducer = pacmap.PaCMAP(
        n_components=3,
        n_neighbors=int(n_neighbors),
        MN_ratio=float(mn_ratio),
        FP_ratio=float(fp_ratio),
        num_iters=int(n_iters),
        random_state=int(random_state),
    )
    t1 = time.time()
    coords = reducer.fit_transform(X)
    print(f"PaCMAP done in {time.time() - t1:.1f}s")
    return np.asarray(coords, dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append adjacency-based 3D PaCMAP columns to final TSV."
    )
    parser.add_argument("layout_in", nargs="?", default=None)
    parser.add_argument("layout_out", nargs="?", default=None)
    parser.add_argument(
        "--svd-dim",
        type=int,
        default=128,
        help="TruncatedSVD dimension before PaCMAP (default: 128).",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="PaCMAP n_neighbors (default: 15).",
    )
    parser.add_argument(
        "--mn-ratio",
        type=float,
        default=0.5,
        help="PaCMAP MN_ratio (default: 0.5).",
    )
    parser.add_argument(
        "--fp-ratio",
        type=float,
        default=1.0,
        help="PaCMAP FP_ratio (default: 1.0).",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=450,
        help="PaCMAP num_iters (default: 450).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for SVD and PaCMAP (default: 42).",
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

    coords = run_pacmap_on_adjacency(
        A=A,
        svd_dim=args.svd_dim,
        n_neighbors=args.n_neighbors,
        mn_ratio=args.mn_ratio,
        fp_ratio=args.fp_ratio,
        n_iters=args.n_iters,
        random_state=args.random_state,
    )

    pac_df = pd.DataFrame(
        {
            "node_id": protein_str_ids,
            "x_pacmap_adjacency": coords[:, 0],
            "y_pacmap_adjacency": coords[:, 1],
            "z_pacmap_adjacency": coords[:, 2],
        }
    )

    base_df = pd.read_csv(layout_in, sep="\t")
    for c in ["x_pacmap_adjacency", "y_pacmap_adjacency", "z_pacmap_adjacency"]:
        if c in base_df.columns:
            base_df = base_df.drop(columns=[c])

    out_df = base_df.merge(pac_df, on="node_id", how="left")
    out_df.to_csv(layout_out, sep="\t", index=False)
    print(f"Wrote {layout_out} with PaCMAP adjacency columns.")


if __name__ == "__main__":
    main()

