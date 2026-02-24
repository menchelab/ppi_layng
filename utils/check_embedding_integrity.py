"""
Embedding integrity checks.

Checks:
- finite values
- L2 norm distribution
- per-dimension variance (near-constant dims)
- sampled cosine similarity distribution
- protein-only subset stats

Usage:
  python -m utils.check_embedding_integrity
  python -m utils.check_embedding_integrity /path/to/embeddings.npy
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from config_tune import EMBEDDINGS_PARQUET, NODE_MAP_PARQUET


def load_embeddings(path: Path):
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    if path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        cols = [c for c in df.columns if c.startswith("dim_")]
        return df[cols].to_numpy(dtype=np.float64)
    raise ValueError(f"Unsupported embedding path: {path}")


def sampled_cosine(X: np.ndarray, n_pairs: int = 30000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    v1 = X[i]
    v2 = X[j]
    n1 = np.linalg.norm(v1, axis=1) + 1e-12
    n2 = np.linalg.norm(v2, axis=1) + 1e-12
    return np.sum(v1 * v2, axis=1) / (n1 * n2)


def summary(name: str, X: np.ndarray):
    norms = np.linalg.norm(X, axis=1)
    dim_std = X.std(axis=0)
    near_zero_dims = int(np.sum(dim_std < 1e-6))
    cos = sampled_cosine(X)
    print(f"\n{name}")
    print("-" * max(20, len(name)))
    print(f"shape={X.shape} finite={np.isfinite(X).all()}")
    print(
        f"norms mean/median/p5/p95/min/max: "
        f"{norms.mean():.4f}/{np.median(norms):.4f}/{np.percentile(norms,5):.4f}/"
        f"{np.percentile(norms,95):.4f}/{norms.min():.4f}/{norms.max():.4f}"
    )
    print(
        f"dim std min/median/max: {dim_std.min():.6f}/{np.median(dim_std):.6f}/{dim_std.max():.6f} "
        f"| near-zero dims (<1e-6): {near_zero_dims}"
    )
    if cos.size:
        print(
            f"sampled cosine mean/median/p5/p95/min/max: "
            f"{cos.mean():.4f}/{np.median(cos):.4f}/{np.percentile(cos,5):.4f}/"
            f"{np.percentile(cos,95):.4f}/{cos.min():.4f}/{cos.max():.4f}"
        )


def main():
    if len(sys.argv) > 1:
        epath = Path(sys.argv[1])
    else:
        npy = EMBEDDINGS_PARQUET.with_suffix(".npy")
        epath = npy if npy.exists() else EMBEDDINGS_PARQUET

    if not epath.exists():
        raise FileNotFoundError(f"Embedding file not found: {epath}")

    X = load_embeddings(epath)
    node_map = pd.read_parquet(NODE_MAP_PARQUET)
    protein_mask = (node_map["node_type"] == "protein").to_numpy()

    print("Embedding integrity report")
    print("=" * 80)
    print(f"Path: {epath}")
    summary("All nodes", X)
    if len(protein_mask) == X.shape[0]:
        summary("Protein nodes only", X[protein_mask])
    else:
        print("\nWARNING: node_map length does not match embedding rows.")


if __name__ == "__main__":
    main()

