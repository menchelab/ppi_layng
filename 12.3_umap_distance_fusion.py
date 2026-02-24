"""
12.3 - UMAP on blended precomputed distances.

D = (1-w)*D_emb + w*D_adj_svd, where both are cosine distances.
Then UMAP(metric='precomputed') in 3D.

Note: This is memory-heavy for ~20k proteins (dense NxN matrix).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances

from utils.hybrid_layout_common import (
    adjacency_svd,
    append_layout_columns,
    build_protein_adjacency,
    choose_layout_path,
    ensure_same_order,
    l2_normalize_rows,
    load_protein_embeddings,
    run_umap_precomputed,
    standardize_dense,
    validate_weight,
    weight_tag,
)


def parse_args():
    p = argparse.ArgumentParser(description="12.3 precomputed-distance fusion layout.")
    p.add_argument("layout_in", nargs="?", default=None)
    p.add_argument("layout_out", nargs="?", default=None)
    p.add_argument("--weight", type=float, default=0.20, help="Adjacency distance weight in [0,1].")
    p.add_argument("--svd-dim", type=int, default=128, help="SVD dims for adjacency view.")
    p.add_argument("--no-seed", action="store_true", help="Disable fixed random seed.")
    return p.parse_args()


def main():
    args = parse_args()
    w = validate_weight(args.weight)
    layout_in = Path(args.layout_in) if args.layout_in else choose_layout_path()
    layout_out = Path(args.layout_out) if args.layout_out else layout_in

    A, ids_adj = build_protein_adjacency()
    E, ids_emb = load_protein_embeddings()
    ensure_same_order(ids_adj, ids_emb)

    rs = None if args.no_seed else 42
    E_feat = l2_normalize_rows(standardize_dense(E))
    A_feat = l2_normalize_rows(standardize_dense(adjacency_svd(A, svd_dim=args.svd_dim, random_state=rs)))

    print("Computing pairwise cosine distances for embeddings...")
    D_emb = pairwise_distances(E_feat, metric="cosine", n_jobs=-1).astype(np.float32)
    print("Computing pairwise cosine distances for adjacency-SVD...")
    D_adj = pairwise_distances(A_feat, metric="cosine", n_jobs=-1).astype(np.float32)
    D = ((1.0 - w) * D_emb + w * D_adj).astype(np.float32)

    coords = run_umap_precomputed(D, use_seed=not args.no_seed)
    tag = weight_tag(w)
    prefix = f"umap_fused_distance_{tag}"
    append_layout_columns(layout_in, layout_out, ids_adj, coords, prefix=prefix)
    print(f"Wrote {layout_out} with x/y/z_{prefix}")


if __name__ == "__main__":
    main()

