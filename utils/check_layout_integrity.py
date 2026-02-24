"""
Layout integrity checks for single- or multi-method layout TSVs.

Checks per method:
- range and std per axis
- center-mass fraction (points within radii 0.2 and 0.5 from origin)
- nearest-neighbor distance summary (sampled)

Usage:
  python -m utils.check_layout_integrity
  python -m utils.check_layout_integrity /path/to/layout_decompression.tsv
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from config_tune import OUTPUT_DIR, LAYOUT_TSV
from utils.distribution_analysis import discover_methods, compute_axis_stats


def default_layout_path():
    p1 = OUTPUT_DIR / "layout_decompression.tsv"
    p2 = OUTPUT_DIR / "layout_multi.tsv"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return LAYOUT_TSV


def sampled_nn_dist(X: np.ndarray, sample_n: int = 4000, seed: int = 42):
    n = X.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample_n, n), replace=False)
    S = X[idx]
    # O(sample_n * n), manageable for sample_n ~ 4k
    dists = np.sqrt(((S[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    # self-distance at original index if sampled same row
    # set exact 0 distances to inf, then take nearest
    dists[dists == 0] = np.inf
    nn = np.min(dists, axis=1)
    nn = nn[np.isfinite(nn)]
    return nn


def method_coords(df: pd.DataFrame):
    methods = discover_methods(df)
    if methods:
        for m in methods:
            yield m, np.column_stack([
                df[f"x_{m}"].to_numpy(dtype=np.float64),
                df[f"y_{m}"].to_numpy(dtype=np.float64),
                df[f"z_{m}"].to_numpy(dtype=np.float64),
            ])
    else:
        # single layout fallback
        if {"x", "y", "z"}.issubset(df.columns):
            arr = np.column_stack([
                df["x"].to_numpy(dtype=np.float64),
                df["y"].to_numpy(dtype=np.float64),
                df["z"].to_numpy(dtype=np.float64),
            ])
            yield "single_layout", arr


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_layout_path()
    if not path.exists():
        raise FileNotFoundError(f"Layout not found: {path}")

    df = pd.read_csv(path, sep="\t")
    print("Layout integrity report")
    print("=" * 100)
    print(f"Path: {path}")
    print(f"{'method':<35} {'std_x':>9} {'std_y':>9} {'std_z':>9} {'r<=0.2':>9} {'r<=0.5':>9} {'nn_med':>9} {'nn_p95':>9}")
    print("-" * 100)

    for name, X in method_coords(df):
        X = X[np.isfinite(X).all(axis=1)]
        if X.size == 0:
            continue
        sx = compute_axis_stats(X[:, 0], "x")["std"]
        sy = compute_axis_stats(X[:, 1], "y")["std"]
        sz = compute_axis_stats(X[:, 2], "z")["std"]
        r = np.linalg.norm(X, axis=1)
        c02 = float(np.mean(r <= 0.2))
        c05 = float(np.mean(r <= 0.5))
        nn = sampled_nn_dist(X)
        nn_med = float(np.median(nn)) if nn.size else np.nan
        nn_p95 = float(np.percentile(nn, 95)) if nn.size else np.nan
        print(f"{name:<35} {sx:>9.4f} {sy:>9.4f} {sz:>9.4f} {c02:>9.2%} {c05:>9.2%} {nn_med:>9.4f} {nn_p95:>9.4f}")


if __name__ == "__main__":
    main()

