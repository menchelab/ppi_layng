"""
12.1 - UMAP hybrid layout via weighted feature concatenation.

X = [sqrt(1-w) * Z(embeddings), sqrt(w) * Z(SVD(adjacency))]
Then UMAP(3D) on X.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.hybrid_layout_common import (
    adjacency_svd,
    append_layout_columns,
    build_protein_adjacency,
    choose_layout_path,
    ensure_same_order,
    load_protein_embeddings,
    run_umap_dense,
    standardize_dense,
    validate_weight,
    weight_tag,
)


def parse_args():
    p = argparse.ArgumentParser(description="12.1 weighted concat fusion (adjacency + embeddings).")
    p.add_argument("layout_in", nargs="?", default=None)
    p.add_argument("layout_out", nargs="?", default=None)
    p.add_argument("--weight", type=float, default=0.10, help="Adjacency weight in [0,1].")
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
    A_svd = adjacency_svd(A, svd_dim=args.svd_dim, random_state=rs)
    E_z = standardize_dense(E)
    A_z = standardize_dense(A_svd)

    X = np.concatenate(
        [np.sqrt(1.0 - w, dtype=np.float32) * E_z, np.sqrt(w, dtype=np.float32) * A_z],
        axis=1,
    ).astype(np.float32)

    coords = run_umap_dense(X, use_seed=not args.no_seed)
    tag = weight_tag(w)
    prefix = f"umap_fused_concat_{tag}"
    append_layout_columns(layout_in, layout_out, ids_adj, coords, prefix=prefix)
    print(f"Wrote {layout_out} with x/y/z_{prefix}")


if __name__ == "__main__":
    main()

