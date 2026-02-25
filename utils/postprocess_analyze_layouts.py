"""
Analyze which 3D layouts would benefit from ForceAtlas-like postprocessing.

Input:
  argv1: nodes.tsv / layout.tsv with columns:
    - id or node_id
    - x_*, y_*, z_* layout triplets (or single x,y,z)

Output:
  - Prints ranked recommendations to stdout
  - Writes TSV report (default: <input_dir>/postprocess_forceatlas_recommendations.tsv)

This tool does NOT overwrite input layouts. It only recommends parameter profiles.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from utils.distribution_analysis import discover_methods
from utils.force_directed_3d import refine_3d_repulsion


def parse_args():
    p = argparse.ArgumentParser(description="Recommend ForceAtlas-like postprocessing params per layout.")
    p.add_argument("layout_tsv", type=str, help="Path to nodes/layout TSV.")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output TSV path (default: input_dir/postprocess_forceatlas_recommendations.tsv)",
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=2400,
        help="Sample size per layout for parameter sweep (default: 1200).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return p.parse_args()


def _get_layout_arrays(df: pd.DataFrame):
    methods = discover_methods(df)
    if methods:
        for m in methods:
            yield m, np.column_stack(
                [
                    df[f"x_{m}"].to_numpy(dtype=np.float64),
                    df[f"y_{m}"].to_numpy(dtype=np.float64),
                    df[f"z_{m}"].to_numpy(dtype=np.float64),
                ]
            )
        return
    if {"x", "y", "z"}.issubset(df.columns):
        yield "single_layout", np.column_stack(
            [
                df["x"].to_numpy(dtype=np.float64),
                df["y"].to_numpy(dtype=np.float64),
                df["z"].to_numpy(dtype=np.float64),
            ]
        )


def _sample_rows(X: np.ndarray, n: int, rng: np.random.Generator):
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) <= n:
        return X
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


def _sample_nn(X: np.ndarray):
    from sklearn.neighbors import NearestNeighbors

    if len(X) < 2:
        return np.array([], dtype=np.float64)
    n_neighbors = min(2, len(X))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
    d, _ = nn.kneighbors(X)
    if d.shape[1] < 2:
        return np.array([], dtype=np.float64)
    return d[:, 1]


def _center_fraction(X: np.ndarray, radius: float):
    # Normalized radial concentration around centroid (scale-invariant per-axis)
    X0 = X - np.median(X, axis=0, keepdims=True)
    s = np.std(X0, axis=0)
    s[s == 0] = 1.0
    Z = X0 / s
    r = np.linalg.norm(Z, axis=1)
    return float(np.mean(r <= radius))


def _pair_distance_rank_corr(X0: np.ndarray, X1: np.ndarray, rng: np.random.Generator, pair_n: int = 20000):
    n = len(X0)
    if n < 3:
        return 1.0
    i = rng.integers(0, n, size=pair_n, endpoint=False)
    j = rng.integers(0, n, size=pair_n, endpoint=False)
    m = i != j
    i = i[m]
    j = j[m]
    if len(i) == 0:
        return 1.0
    d0 = np.linalg.norm(X0[i] - X0[j], axis=1)
    d1 = np.linalg.norm(X1[i] - X1[j], axis=1)
    corr = spearmanr(d0, d1).correlation
    if corr is None or not np.isfinite(corr):
        return 0.0
    return float(corr)


def _evaluate_forceatlas_candidate(X: np.ndarray, cfg: dict, seed: int, rng: np.random.Generator):
    nn0 = _sample_nn(X)
    nn0_med = float(np.median(nn0)) if len(nn0) else np.nan
    c0 = _center_fraction(X, radius=1.0)

    X1 = refine_3d_repulsion(
        X,
        iterations=int(cfg["iterations"]),
        repulsion_strength=float(cfg["repulsion_strength"]),
        min_dist=1e-3,  # safer for stability
        use_knn=True,
        k_repel=int(cfg["k_repel"]),
        random_state=seed,
    )

    if not np.isfinite(X1).all():
        return None

    nn1 = _sample_nn(X1)
    nn1_med = float(np.median(nn1)) if len(nn1) else np.nan
    c1 = _center_fraction(X1, radius=1.0)
    rank_corr = _pair_distance_rank_corr(X, X1, rng=rng)

    # Scale-shift penalty to discourage "exploding" layouts.
    r0 = np.linalg.norm(X - np.median(X, axis=0, keepdims=True), axis=1)
    r1 = np.linalg.norm(X1 - np.median(X1, axis=0, keepdims=True), axis=1)
    r0_med = float(np.median(r0)) if len(r0) else np.nan
    r1_med = float(np.median(r1)) if len(r1) else np.nan
    if np.isfinite(r0_med) and r0_med > 0 and np.isfinite(r1_med):
        scale_ratio = float(r1_med / r0_med)
    else:
        scale_ratio = 1.0
    log_scale_shift = float(abs(np.log(max(scale_ratio, 1e-12))))

    # Bound spread-improvement so huge expansions do not dominate score.
    if not np.isfinite(nn0_med) or nn0_med <= 0 or not np.isfinite(nn1_med) or nn1_med <= 0:
        spread_improve = 0.0
    else:
        spread_improve = float(np.log(nn1_med / nn0_med))
    spread_improve = float(np.clip(spread_improve, -1.0, 1.0))

    center_relief = c0 - c1

    # Higher is better:
    # - reduce center crowding
    # - modestly improve nearest-neighbor spacing
    # - preserve pairwise rank structure
    # - penalize large global scale distortions
    score = (
        0.40 * center_relief
        + 0.25 * spread_improve
        + 0.35 * rank_corr
        - 0.30 * log_scale_shift
    )
    return {
        "score": float(score),
        "nn_med_before": float(nn0_med) if np.isfinite(nn0_med) else np.nan,
        "nn_med_after": float(nn1_med) if np.isfinite(nn1_med) else np.nan,
        "center_before_r1.0": float(c0),
        "center_after_r1.0": float(c1),
        "center_relief": float(center_relief),
        "spread_improve_log": float(spread_improve),
        "scale_ratio_median_radius": float(scale_ratio),
        "log_scale_shift": float(log_scale_shift),
        "rank_corr": float(rank_corr),
    }


def main():
    args = parse_args()
    path = Path(args.layout_tsv)
    if not path.exists():
        raise FileNotFoundError(f"Layout file not found: {path}")
    out_path = Path(args.out) if args.out else (path.parent / "postprocess_forceatlas_recommendations.tsv")

    df = pd.read_csv(path, sep="\t")
    layouts = list(_get_layout_arrays(df))
    if not layouts:
        raise ValueError("No layout columns found. Expected x_*/y_*/z_* or x/y/z.")

    rng = np.random.default_rng(args.seed)
    grid = [
        {"profile": "mild", "iterations": 40, "repulsion_strength": 0.12, "k_repel": 25},
        {"profile": "balanced", "iterations": 60, "repulsion_strength": 0.20, "k_repel": 40},
        {"profile": "spread", "iterations": 90, "repulsion_strength": 0.30, "k_repel": 60},
        {"profile": "strong", "iterations": 120, "repulsion_strength": 0.40, "k_repel": 80},
        {"profile": "aggressive", "iterations": 160, "repulsion_strength": 0.55, "k_repel": 110},
    ]

    rows = []
    layout_pbar = tqdm(layouts, desc="Layouts", unit=" layout")
    for name, Xfull in layout_pbar:
        layout_pbar.set_postfix_str(name)
        X = _sample_rows(Xfull, n=max(200, int(args.sample_n)), rng=rng)
        baseline_center = _center_fraction(X, radius=1.0)
        baseline_nn = _sample_nn(X)
        baseline_nn_med = float(np.median(baseline_nn)) if len(baseline_nn) else np.nan

        best = None
        # No-op baseline: explicit option to keep original layout
        best = {
            "layout": name,
            "sample_n": len(X),
            "baseline_center_r1.0": baseline_center,
            "baseline_nn_med": baseline_nn_med,
            "profile": "none",
            "iterations": 0,
            "repulsion_strength": 0.0,
            "k_repel": 0,
            "score": 0.0,
            "nn_med_before": baseline_nn_med,
            "nn_med_after": baseline_nn_med,
            "center_before_r1.0": baseline_center,
            "center_after_r1.0": baseline_center,
            "center_relief": 0.0,
            "spread_improve_log": 0.0,
            "scale_ratio_median_radius": 1.0,
            "log_scale_shift": 0.0,
            "rank_corr": 1.0,
        }
        cfg_pbar = tqdm(grid, desc=f"{name}: sweep", unit=" cfg", leave=False)
        for cfg in cfg_pbar:
            cfg_pbar.set_postfix_str(
                f"{cfg['profile']} it={cfg['iterations']} r={cfg['repulsion_strength']} k={cfg['k_repel']}"
            )
            res = _evaluate_forceatlas_candidate(X, cfg, seed=args.seed, rng=rng)
            if res is None:
                continue
            cand = {
                "layout": name,
                "sample_n": len(X),
                "baseline_center_r1.0": baseline_center,
                "baseline_nn_med": baseline_nn_med,
                "profile": cfg["profile"],
                "iterations": cfg["iterations"],
                "repulsion_strength": cfg["repulsion_strength"],
                "k_repel": cfg["k_repel"],
                **res,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
                cfg_pbar.set_postfix_str(
                    f"best={cfg['profile']} score={best['score']:.4f}"
                )

        if best is not None:
            # Only recommend when improvement is meaningful and not just rescaling.
            best["recommend_postprocess"] = bool(
                (best["profile"] != "none")
                and (best["score"] > 0.08)
                and (best["center_relief"] > 0.01)
                and (best["log_scale_shift"] < 0.8)
            )
            rows.append(best)

    if not rows:
        raise RuntimeError("No valid recommendations generated.")

    out_df = pd.DataFrame(rows).sort_values(["recommend_postprocess", "score"], ascending=[False, False])
    out_df.to_csv(out_path, sep="\t", index=False)

    print("ForceAtlas postprocess recommendation report")
    print("=" * 90)
    print(f"Input: {path}")
    print(f"Output: {out_path}")
    print(f"{'layout':<30} {'rec?':<6} {'score':>8} {'profile':<10} {'iters':>6} {'repel':>8} {'k':>5}")
    print("-" * 90)
    for _, r in out_df.iterrows():
        rec = "yes" if bool(r["recommend_postprocess"]) else "no"
        print(
            f"{str(r['layout']):<30} {rec:<6} {float(r['score']):>8.4f} "
            f"{str(r['profile']):<10} {int(r['iterations']):>6} "
            f"{float(r['repulsion_strength']):>8.3f} {int(r['k_repel']):>5}"
        )


if __name__ == "__main__":
    main()

