"""
Build UMAP and PaCMAP layouts from a nodes TSV using a comma-separated location column.

Input:
  nodes_tsv with columns: id, <location_column> (e.g. locations_a)
  The location column has comma-separated strings (0-5 typical, can be more) or null.

Output:
  - nodes.tsv: all original data + layout columns
      loc1_01..loc1_23 (Approach 1: location-weighted, 20 UMAP + 3 PaCMAP)
      loc3_01..loc3_13 (Approach 3: pure location, 10 UMAP + 3 PaCMAP)
  - loc_runs.tsv: one row per run with approach, algo, params

Note: PaCMAP may fail on very small datasets (< ~30 nodes); those layout columns
will be omitted. UMAP handles small datasets.

Usage:
  python -m utils.umap_from_locations_tsv /path/to/nodes.tsv --column locations_a
  python -m utils.umap_from_locations_tsv /path/to/nodes.tsv -c locations_a --output /path/to/out.tsv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create location-based 3D layouts from nodes TSV."
    )
    parser.add_argument(
        "nodes_tsv",
        type=str,
        help="TSV with id and a comma-separated location column.",
    )
    parser.add_argument(
        "-c",
        "--column",
        type=str,
        default="locations_a",
        help="Column name containing comma-separated locations (default: locations_a).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output nodes TSV path. Default: same dir as input, nodes_loc.tsv",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Disable random_state for faster non-deterministic runs.",
    )
    return parser.parse_args()


def parse_locations(series: pd.Series) -> tuple[list[list[str]], list[str]]:
    """
    Parse comma-separated location strings from a Series.
    Returns (list of lists per row, sorted unique locations).
    """
    node_locations: list[list[str]] = []
    all_locations: set[str] = set()

    for val in series:
        if pd.isna(val) or val is None or str(val).strip() == "":
            node_locations.append([])
            continue
        parts = [x.strip() for x in str(val).split(",") if x.strip()]
        node_locations.append(parts)
        all_locations.update(parts)

    unique = sorted(all_locations)
    return node_locations, unique


def encode_multihot(
    node_locations: list[list[str]],
    unique_locations: list[str],
    add_unknown: bool = True,
) -> np.ndarray:
    """
    Multi-hot encode node locations. Returns (n_nodes, n_dims) float32 array.
    """
    n = len(node_locations)
    loc_to_idx = {loc: i for i, loc in enumerate(unique_locations)}
    n_loc = len(unique_locations)
    n_dims = n_loc + (1 if add_unknown else 0)
    unknown_idx = n_loc if add_unknown else -1

    X = np.zeros((n, n_dims), dtype=np.float32)
    for i, locs in enumerate(node_locations):
        if not locs:
            if add_unknown:
                X[i, unknown_idx] = 1.0
        else:
            for loc in locs:
                if loc in loc_to_idx:
                    X[i, loc_to_idx[loc]] = 1.0
    return X


def apply_location_weight(X: np.ndarray, weight: int) -> np.ndarray:
    """
    Approach 1: Repeat location columns `weight` times to emphasize them
    in distance computation. weight=1 means no change.
    """
    if weight <= 1:
        return X
    return np.hstack([X] * weight)


# --- Approach 1 variants: 20 UMAP + 3 PaCMAP ---
APPROACH1_UMAP_VARIANTS = [
    # Weight sweep (1,2,3,5,10)
    {"weight": 1, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"weight": 3, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"weight": 5, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"weight": 10, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    # n_neighbors sweep
    {"weight": 2, "n_neighbors": 5, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 10, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 20, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 30, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 50, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    # min_dist sweep
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.0, "spread": 0.8, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.005, "spread": 0.8, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.05, "spread": 0.8, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.1, "spread": 0.8, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.2, "spread": 0.8, "metric": "cosine"},
    # spread sweep
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.5, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.01, "spread": 1.0, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.01, "spread": 1.2, "metric": "cosine"},
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.01, "spread": 1.5, "metric": "cosine"},
    # Jaccard metric
    {"weight": 2, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "jaccard"},
]

APPROACH1_PACMAP_VARIANTS = [
    {"weight": 1, "n_neighbors": 10, "mn_ratio": 0.5, "fp_ratio": 2.0, "n_iters": 450},
    {"weight": 2, "n_neighbors": 15, "mn_ratio": 0.5, "fp_ratio": 1.5, "n_iters": 600},
    {"weight": 5, "n_neighbors": 20, "mn_ratio": 0.5, "fp_ratio": 2.0, "n_iters": 800},
]

# --- Approach 3 variants: 10 UMAP + 3 PaCMAP (pure location, no weighting) ---
APPROACH3_UMAP_VARIANTS = [
    {"n_neighbors": 5, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"n_neighbors": 10, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"n_neighbors": 20, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"n_neighbors": 30, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"n_neighbors": 15, "min_dist": 0.0, "spread": 0.8, "metric": "cosine"},
    {"n_neighbors": 15, "min_dist": 0.05, "spread": 0.8, "metric": "cosine"},
    {"n_neighbors": 15, "min_dist": 0.01, "spread": 0.5, "metric": "cosine"},
    {"n_neighbors": 15, "min_dist": 0.01, "spread": 1.2, "metric": "cosine"},
    {"n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "jaccard"},
]

APPROACH3_PACMAP_VARIANTS = [
    {"n_neighbors": 10, "mn_ratio": 0.5, "fp_ratio": 2.0, "n_iters": 450},
    {"n_neighbors": 15, "mn_ratio": 0.5, "fp_ratio": 1.5, "n_iters": 600},
    {"n_neighbors": 20, "mn_ratio": 0.5, "fp_ratio": 2.0, "n_iters": 800},
]


def run_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.01,
    spread: float = 1.0,
    metric: str = "cosine",
    n_epochs: int = 5000,
    repulsion_strength: float = 1.0,
    negative_sample_rate: int = 10,
    random_state: int | None = 42,
):
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("Please install umap-learn: pip install umap-learn") from e

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        spread=float(spread),
        metric=str(metric),
        n_epochs=int(n_epochs),
        repulsion_strength=float(repulsion_strength),
        negative_sample_rate=int(negative_sample_rate),
        init="spectral",
        low_memory=True,
        random_state=random_state,
        verbose=True,
    )
    t0 = time.time()
    coords = reducer.fit_transform(np.asarray(X, dtype=np.float32))
    return np.asarray(coords, dtype=np.float64), time.time() - t0


def run_pacmap(
    X: np.ndarray,
    n_neighbors: int = 10,
    mn_ratio: float = 0.5,
    fp_ratio: float = 2.0,
    n_iters: int = 450,
    random_state: int | None = 42,
):
    try:
        import pacmap
    except ImportError as e:
        raise RuntimeError("Please install pacmap: pip install pacmap") from e

    rs = 42 if random_state is None else int(random_state)
    n = X.shape[0]
    nn = min(int(n_neighbors), max(2, n - 1))
    fp = float(fp_ratio)
    kwargs: dict = {
        "n_components": 3,
        "n_neighbors": nn,
        "MN_ratio": float(mn_ratio),
        "FP_ratio": fp,
        "num_iters": int(n_iters),
        "random_state": rs,
    }
    reducer = pacmap.PaCMAP(**kwargs)
    t0 = time.time()
    coords = reducer.fit_transform(np.asarray(X, dtype=np.float32))
    return np.asarray(coords, dtype=np.float64), time.time() - t0


def _add_jitter_if_needed(X: np.ndarray) -> np.ndarray:
    """Add tiny jitter if all rows are identical (e.g. all null locations)."""
    if X.shape[0] < 2:
        return X
    if np.allclose(X, X[0:1], atol=1e-9):
        rng = np.random.default_rng(42)
        X = X + rng.normal(0, 1e-6, X.shape).astype(np.float32)
    return X


def main():
    args = parse_args()
    nodes_path = Path(args.nodes_tsv).expanduser().resolve()
    if not nodes_path.exists():
        raise FileNotFoundError(f"Input not found: {nodes_path}")

    col = args.column
    df = pd.read_csv(nodes_path, sep="\t")
    if "id" not in df.columns:
        raise ValueError(f"{nodes_path} must have column: id")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    random_state = None if args.no_seed else 42
    out_path = Path(args.output).expanduser().resolve() if args.output else nodes_path.parent / "nodes_loc.tsv"
    runs_path = out_path.parent / "loc_runs.tsv"

    # Parse and encode
    node_locations, unique_locations = parse_locations(df[col])
    X_loc = encode_multihot(node_locations, unique_locations, add_unknown=True)
    X_loc = _add_jitter_if_needed(X_loc)

    n_nodes = len(df)
    n_unique = len(unique_locations)
    print(f"Nodes: {n_nodes} | Unique locations: {n_unique} | Feature dim: {X_loc.shape[1]}")

    run_rows = []

    # --- Approach 1: 20 UMAP + 3 PaCMAP ---
    for i, spec in enumerate(APPROACH1_UMAP_VARIANTS, start=1):
        suffix = f"loc1_{i:02d}"
        print(f"\n=== Approach 1 UMAP {suffix} ===")
        X = apply_location_weight(X_loc, spec["weight"])
        try:
            coords, sec = run_umap(
                X,
                n_neighbors=spec["n_neighbors"],
                min_dist=spec["min_dist"],
                spread=spec["spread"],
                metric=spec["metric"],
                random_state=random_state,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        df[f"x_{suffix}"] = coords[:, 0]
        df[f"y_{suffix}"] = coords[:, 1]
        df[f"z_{suffix}"] = coords[:, 2]
        run_rows.append({
            "approach": 1,
            "algo": "umap",
            "suffix": suffix,
            "location_weight": spec["weight"],
            "n_neighbors": spec["n_neighbors"],
            "min_dist": spec["min_dist"],
            "spread": spec["spread"],
            "metric": spec["metric"],
            "layout_seconds": round(sec, 3),
        })

    for i, spec in enumerate(APPROACH1_PACMAP_VARIANTS, start=len(APPROACH1_UMAP_VARIANTS) + 1):
        suffix = f"loc1_{i:02d}"
        print(f"\n=== Approach 1 PaCMAP {suffix} ===")
        X = apply_location_weight(X_loc, spec["weight"])
        try:
            coords, sec = run_pacmap(
                X,
                n_neighbors=spec["n_neighbors"],
                mn_ratio=spec["mn_ratio"],
                fp_ratio=spec["fp_ratio"],
                n_iters=spec["n_iters"],
                random_state=random_state,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        df[f"x_{suffix}"] = coords[:, 0]
        df[f"y_{suffix}"] = coords[:, 1]
        df[f"z_{suffix}"] = coords[:, 2]
        run_rows.append({
            "approach": 1,
            "algo": "pacmap",
            "suffix": suffix,
            "location_weight": spec["weight"],
            "n_neighbors": spec["n_neighbors"],
            "mn_ratio": spec["mn_ratio"],
            "fp_ratio": spec["fp_ratio"],
            "n_iters": spec["n_iters"],
            "layout_seconds": round(sec, 3),
        })

    # --- Approach 3: 10 UMAP + 3 PaCMAP (pure location) ---
    for i, spec in enumerate(APPROACH3_UMAP_VARIANTS, start=1):
        suffix = f"loc3_{i:02d}"
        print(f"\n=== Approach 3 UMAP {suffix} ===")
        try:
            coords, sec = run_umap(
                X_loc,
                n_neighbors=spec["n_neighbors"],
                min_dist=spec["min_dist"],
                spread=spec["spread"],
                metric=spec["metric"],
                random_state=random_state,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        df[f"x_{suffix}"] = coords[:, 0]
        df[f"y_{suffix}"] = coords[:, 1]
        df[f"z_{suffix}"] = coords[:, 2]
        run_rows.append({
            "approach": 3,
            "algo": "umap",
            "suffix": suffix,
            "location_weight": 1,
            "n_neighbors": spec["n_neighbors"],
            "min_dist": spec["min_dist"],
            "spread": spec["spread"],
            "metric": spec["metric"],
            "layout_seconds": round(sec, 3),
        })

    for i, spec in enumerate(APPROACH3_PACMAP_VARIANTS, start=len(APPROACH3_UMAP_VARIANTS) + 1):
        suffix = f"loc3_{i:02d}"
        print(f"\n=== Approach 3 PaCMAP {suffix} ===")
        try:
            coords, sec = run_pacmap(
                X_loc,
                n_neighbors=spec["n_neighbors"],
                mn_ratio=spec["mn_ratio"],
                fp_ratio=spec["fp_ratio"],
                n_iters=spec["n_iters"],
                random_state=random_state,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        df[f"x_{suffix}"] = coords[:, 0]
        df[f"y_{suffix}"] = coords[:, 1]
        df[f"z_{suffix}"] = coords[:, 2]
        run_rows.append({
            "approach": 3,
            "algo": "pacmap",
            "suffix": suffix,
            "location_weight": 1,
            "n_neighbors": spec["n_neighbors"],
            "mn_ratio": spec["mn_ratio"],
            "fp_ratio": spec["fp_ratio"],
            "n_iters": spec["n_iters"],
            "layout_seconds": round(sec, 3),
        })

    df.to_csv(out_path, sep="\t", index=False)
    pd.DataFrame(run_rows).to_csv(runs_path, sep="\t", index=False)

    print(f"\nWrote {out_path}")
    print(f"Wrote {runs_path}")


if __name__ == "__main__":
    main()
