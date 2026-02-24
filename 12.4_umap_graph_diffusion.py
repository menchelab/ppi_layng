"""
12.4 - UMAP on graph-diffused embeddings.

Iterative diffusion:
E_t+1 = beta * E_0 + (1-beta) * (A_row_norm @ E_t)
Then UMAP in 3D on E_t.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.hybrid_layout_common import (
    append_layout_columns,
    build_protein_adjacency,
    choose_layout_path,
    ensure_same_order,
    load_protein_embeddings,
    row_normalize_sparse,
    run_umap_dense,
    standardize_dense,
)


def parse_args():
    p = argparse.ArgumentParser(description="12.4 graph-diffused embedding layout.")
    p.add_argument("layout_in", nargs="?", default=None)
    p.add_argument("layout_out", nargs="?", default=None)
    p.add_argument("--beta", type=float, default=0.80, help="Retain-original factor in [0,1].")
    p.add_argument("--steps", type=int, default=2, help="Diffusion steps.")
    p.add_argument("--no-seed", action="store_true", help="Disable fixed random seed.")
    return p.parse_args()


def diffuse_embeddings(E: np.ndarray, A_row_norm, beta: float, steps: int) -> np.ndarray:
    b = float(beta)
    if not (0.0 <= b <= 1.0):
        raise ValueError(f"beta must be in [0,1], got {beta}")
    t = max(1, int(steps))
    E0 = np.asarray(E, dtype=np.float32)
    Et = E0.copy()
    for _ in range(t):
        Et = b * E0 + (1.0 - b) * (A_row_norm @ Et)
        Et = np.asarray(Et, dtype=np.float32)
    return Et


def main():
    args = parse_args()
    layout_in = Path(args.layout_in) if args.layout_in else choose_layout_path()
    layout_out = Path(args.layout_out) if args.layout_out else layout_in

    A, ids_adj = build_protein_adjacency()
    E, ids_emb = load_protein_embeddings()
    ensure_same_order(ids_adj, ids_emb)

    A_row = row_normalize_sparse(A)
    E_diff = diffuse_embeddings(E, A_row, beta=args.beta, steps=args.steps)
    E_diff = standardize_dense(E_diff)

    coords = run_umap_dense(E_diff, use_seed=not args.no_seed)
    btag = int(round(float(args.beta) * 100))
    stag = max(1, int(args.steps))
    prefix = f"umap_graph_diffusion_b{btag:03d}_s{stag:02d}"
    append_layout_columns(layout_in, layout_out, ids_adj, coords, prefix=prefix)
    print(f"Wrote {layout_out} with x/y/z_{prefix}")


if __name__ == "__main__":
    main()

