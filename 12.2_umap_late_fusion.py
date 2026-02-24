"""
12.2 - UMAP late fusion of two independent layouts.

1) UMAP on embeddings
2) UMAP on adjacency-SVD features
3) Align adjacency layout to embedding layout (orthogonal Procrustes)
4) Blend in 3D: L = (1-w)*L_emb + w*L_adj_aligned
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

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
    p = argparse.ArgumentParser(description="12.2 late-fusion layout blend.")
    p.add_argument("layout_in", nargs="?", default=None)
    p.add_argument("layout_out", nargs="?", default=None)
    p.add_argument("--weight", type=float, default=0.20, help="Adjacency-layout blend weight in [0,1].")
    p.add_argument("--svd-dim", type=int, default=128, help="SVD dims for adjacency view.")
    p.add_argument("--no-seed", action="store_true", help="Disable fixed random seed.")
    return p.parse_args()


def align_to_reference(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    mov = np.asarray(moving, dtype=np.float64)
    ref_mu = ref.mean(axis=0, keepdims=True)
    mov_mu = mov.mean(axis=0, keepdims=True)
    ref0 = ref - ref_mu
    mov0 = mov - mov_mu

    R, _ = orthogonal_procrustes(mov0, ref0)
    mov_aligned = mov0 @ R
    ref_std = float(np.std(ref0))
    mov_std = float(np.std(mov_aligned))
    if mov_std > 0:
        mov_aligned = mov_aligned * (ref_std / mov_std)
    mov_aligned = mov_aligned + ref_mu
    return mov_aligned


def main():
    args = parse_args()
    w = validate_weight(args.weight)
    layout_in = Path(args.layout_in) if args.layout_in else choose_layout_path()
    layout_out = Path(args.layout_out) if args.layout_out else layout_in

    A, ids_adj = build_protein_adjacency()
    E, ids_emb = load_protein_embeddings()
    ensure_same_order(ids_adj, ids_emb)

    rs = None if args.no_seed else 42
    E_z = standardize_dense(E)
    A_z = standardize_dense(adjacency_svd(A, svd_dim=args.svd_dim, random_state=rs))

    L_emb = run_umap_dense(E_z, use_seed=not args.no_seed)
    L_adj = run_umap_dense(A_z, use_seed=not args.no_seed)
    L_adj_aligned = align_to_reference(L_emb, L_adj)

    L = ((1.0 - w) * L_emb + w * L_adj_aligned).astype(np.float64)
    tag = weight_tag(w)
    prefix = f"umap_fused_lateblend_{tag}"
    append_layout_columns(layout_in, layout_out, ids_adj, L, prefix=prefix)
    print(f"Wrote {layout_out} with x/y/z_{prefix}")


if __name__ == "__main__":
    main()

