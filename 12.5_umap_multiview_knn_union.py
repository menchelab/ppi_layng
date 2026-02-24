"""
12.5 - UMAP from multi-view kNN graph union.

Build kNN affinities in two views:
- Embedding view
- Adjacency-SVD view

Union/fuse:
G = (1-w) * G_emb + w * G_adj
Then reduce G with SVD and run UMAP in 3D.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils.hybrid_layout_common import (
    adjacency_svd,
    append_layout_columns,
    build_protein_adjacency,
    build_sparse_knn_affinity,
    choose_layout_path,
    ensure_same_order,
    load_protein_embeddings,
    run_umap_dense,
    standardize_dense,
    validate_weight,
    weight_tag,
)


def parse_args():
    p = argparse.ArgumentParser(description="12.5 multi-view kNN union fusion layout.")
    p.add_argument("layout_in", nargs="?", default=None)
    p.add_argument("layout_out", nargs="?", default=None)
    p.add_argument("--weight", type=float, default=0.30, help="Adjacency-view graph weight in [0,1].")
    p.add_argument("--svd-dim", type=int, default=128, help="SVD dims for adjacency view.")
    p.add_argument("--knn", type=int, default=30, help="k for each view kNN graph.")
    p.add_argument("--graph-svd-dim", type=int, default=64, help="SVD dims on fused graph before UMAP.")
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
    E_feat = standardize_dense(E)
    A_feat = standardize_dense(adjacency_svd(A, svd_dim=args.svd_dim, random_state=rs))

    G_emb = build_sparse_knn_affinity(E_feat, k=args.knn, metric="cosine")
    G_adj = build_sparse_knn_affinity(A_feat, k=args.knn, metric="cosine")
    G = ((1.0 - w) * G_emb + w * G_adj).tocsr()

    G_feat = adjacency_svd(G, svd_dim=args.graph_svd_dim, random_state=rs)
    coords = run_umap_dense(G_feat, use_seed=not args.no_seed)

    tag = weight_tag(w)
    prefix = f"umap_multiview_knn_union_{tag}_k{int(args.knn)}"
    append_layout_columns(layout_in, layout_out, ids_adj, coords, prefix=prefix)
    print(f"Wrote {layout_out} with x/y/z_{prefix}")


if __name__ == "__main__":
    main()

