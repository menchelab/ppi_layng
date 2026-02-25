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
        help="Sample size per layout for parameter sweep (default: 2400).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--seed-list",
        type=str,
        default="42,52,62",
        help="Comma-separated seeds for robust candidate evaluation (default: 42,52,62).",
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


def _pair_distance_rank_corr(
    X0: np.ndarray,
    X1: np.ndarray,
    rng: np.random.Generator,
    pair_n: int = 12000,
):
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


def _evaluate_forceatlas_once(X: np.ndarray, cfg: dict, seed: int, rng: np.random.Generator):
    nn0 = _sample_nn(X)
    nn0_med = float(np.median(nn0)) if len(nn0) else np.nan
    c0 = _center_fraction(X, radius=1.0)

    X1 = refine_3d_repulsion(
        X,
        iterations=int(cfg["iterations"]),
        repulsion_strength=float(cfg["repulsion_strength"]),
        step_size=float(cfg.get("step_size", 0.1)),
        min_dist=1e-3,  # safer for stability
        use_knn=True,
        k_repel=int(cfg["k_repel"]),
        anchor_strength=float(cfg.get("anchor_strength", 0.0)),
        max_step_factor=float(cfg.get("max_step_factor", 0.0)),
        dense_only_quantile=cfg.get("dense_only_quantile", None),
        two_phase=bool(cfg.get("two_phase", False)),
        phase_split=float(cfg.get("phase_split", 0.35)),
        phase1_strength_mult=float(cfg.get("phase1_strength_mult", 1.5)),
        phase2_strength_mult=float(cfg.get("phase2_strength_mult", 0.6)),
        early_stop_patience=int(cfg.get("early_stop_patience", 0)),
        early_stop_min_improve=float(cfg.get("early_stop_min_improve", 1e-4)),
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
        0.10 * center_relief
        + 0.40 * spread_improve
        + 0.40 * rank_corr
        - 0.15 * log_scale_shift
    )
    return {
        "score_single": float(score),
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


def _evaluate_forceatlas_candidate_multi(
    X: np.ndarray,
    cfg: dict,
    seeds: list[int],
    base_seed: int,
):
    runs = []
    for s in seeds:
        rng = np.random.default_rng(base_seed + int(s))
        res = _evaluate_forceatlas_once(X, cfg, seed=int(s), rng=rng)
        if res is not None:
            runs.append(res)
    if not runs:
        return None

    score_vals = np.array([r["score_single"] for r in runs], dtype=np.float64)
    score_mean = float(np.mean(score_vals))
    score_std = float(np.std(score_vals))
    # Stability-aware objective: prioritize robust candidates over brittle lucky ones.
    score_stable = score_mean - 0.5 * score_std

    def _mean_key(k: str) -> float:
        vals = np.array([r[k] for r in runs], dtype=np.float64)
        return float(np.nanmean(vals))

    out = {
        "score": score_stable,
        "score_mean": score_mean,
        "score_std": score_std,
        "n_seed_runs": int(len(runs)),
        "nn_med_before": _mean_key("nn_med_before"),
        "nn_med_after": _mean_key("nn_med_after"),
        "center_before_r1.0": _mean_key("center_before_r1.0"),
        "center_after_r1.0": _mean_key("center_after_r1.0"),
        "center_relief": _mean_key("center_relief"),
        "spread_improve_log": _mean_key("spread_improve_log"),
        "scale_ratio_median_radius": _mean_key("scale_ratio_median_radius"),
        "log_scale_shift": _mean_key("log_scale_shift"),
        "rank_corr": _mean_key("rank_corr"),
    }
    return out


def _build_fine_grid(top_cfgs: list[dict]):
    fine = []
    for cfg in top_cfgs:
        it = int(cfg["iterations"])
        rs = float(cfg["repulsion_strength"])
        kk = int(cfg["k_repel"])
        anchor = float(cfg.get("anchor_strength", 0.0))
        msf = float(cfg.get("max_step_factor", 0.0))
        dense_q = cfg.get("dense_only_quantile", None)
        step_size = float(cfg.get("step_size", 0.1))
        two_phase = bool(cfg.get("two_phase", False))
        phase_split = float(cfg.get("phase_split", 0.35))
        p1 = float(cfg.get("phase1_strength_mult", 1.5))
        p2 = float(cfg.get("phase2_strength_mult", 0.6))
        esp = int(cfg.get("early_stop_patience", 0))
        esi = float(cfg.get("early_stop_min_improve", 1e-4))
        for it2 in [max(20, int(it * 0.8)), it, int(it * 1.25)]:
            for rs2 in [max(0.03, rs * 0.8), rs, rs * 1.2]:
                for k2 in [max(10, int(kk * 0.8)), kk, int(kk * 1.2)]:
                    fine.append(
                        {
                            "profile": f"fine_{cfg['profile']}",
                            "iterations": int(it2),
                            "repulsion_strength": float(rs2),
                            "k_repel": int(k2),
                            "step_size": step_size,
                            "anchor_strength": anchor,
                            "max_step_factor": msf,
                            "dense_only_quantile": dense_q,
                            "two_phase": two_phase,
                            "phase_split": phase_split,
                            "phase1_strength_mult": p1,
                            "phase2_strength_mult": p2,
                            "early_stop_patience": esp,
                            "early_stop_min_improve": esi,
                        }
                    )
    # Deduplicate configs
    seen = set()
    out = []
    for c in fine:
        key = (
            c["iterations"],
            round(c["repulsion_strength"], 6),
            c["k_repel"],
            round(float(c.get("anchor_strength", 0.0)), 6),
            round(float(c.get("max_step_factor", 0.0)), 6),
            c.get("dense_only_quantile", None),
            bool(c.get("two_phase", False)),
        )
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _global_drift_risk(row: pd.Series) -> str:
    """Simple traffic-light risk based on scale shift + rank-structure preservation."""
    ls = float(row.get("log_scale_shift", 0.0))
    rc = float(row.get("rank_corr", 1.0))
    if ls >= 0.55 or rc < 0.90:
        return "high"
    if ls >= 0.35 or rc < 0.95:
        return "med"
    return "low"


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
    seeds = [int(x) for x in str(args.seed_list).split(",") if str(x).strip()]
    if not seeds:
        seeds = [42, 52, 62]

    rows = []
    layout_pbar = tqdm(layouts, desc="Layouts", unit=" layout")
    for name, Xfull in layout_pbar:
        layout_pbar.set_postfix_str(name)
        X = _sample_rows(Xfull, n=max(200, int(args.sample_n)), rng=rng)
        baseline_center = _center_fraction(X, radius=1.0)
        baseline_nn = _sample_nn(X)
        baseline_nn_med = float(np.median(baseline_nn)) if len(baseline_nn) else np.nan

        # Adaptive coarse grid by baseline density.
        if np.isfinite(baseline_nn_med) and baseline_nn_med < 0.05:
            coarse_grid = [
                {"profile": "mild", "iterations": 45, "repulsion_strength": 0.08, "k_repel": 18, "step_size": 0.08, "anchor_strength": 0.18, "max_step_factor": 0.015, "dense_only_quantile": 0.20, "two_phase": True, "phase_split": 0.35, "phase1_strength_mult": 1.4, "phase2_strength_mult": 0.7, "early_stop_patience": 8, "early_stop_min_improve": 2e-4},
                {"profile": "balanced", "iterations": 70, "repulsion_strength": 0.12, "k_repel": 28, "step_size": 0.08, "anchor_strength": 0.22, "max_step_factor": 0.018, "dense_only_quantile": 0.25, "two_phase": True, "phase_split": 0.35, "phase1_strength_mult": 1.5, "phase2_strength_mult": 0.65, "early_stop_patience": 10, "early_stop_min_improve": 2e-4},
                {"profile": "spread", "iterations": 90, "repulsion_strength": 0.16, "k_repel": 40, "step_size": 0.09, "anchor_strength": 0.20, "max_step_factor": 0.020, "dense_only_quantile": 0.30, "two_phase": True, "phase_split": 0.4, "phase1_strength_mult": 1.6, "phase2_strength_mult": 0.6, "early_stop_patience": 10, "early_stop_min_improve": 2e-4},
                {"profile": "strong", "iterations": 120, "repulsion_strength": 0.22, "k_repel": 55, "step_size": 0.10, "anchor_strength": 0.16, "max_step_factor": 0.025, "dense_only_quantile": 0.35, "two_phase": True, "phase_split": 0.45, "phase1_strength_mult": 1.7, "phase2_strength_mult": 0.55, "early_stop_patience": 12, "early_stop_min_improve": 2e-4},
            ]
        elif np.isfinite(baseline_nn_med) and baseline_nn_med < 0.20:
            coarse_grid = [
                {"profile": "mild", "iterations": 35, "repulsion_strength": 0.06, "k_repel": 16, "step_size": 0.07, "anchor_strength": 0.25, "max_step_factor": 0.012, "dense_only_quantile": 0.18, "two_phase": True, "phase_split": 0.35, "phase1_strength_mult": 1.35, "phase2_strength_mult": 0.75, "early_stop_patience": 8, "early_stop_min_improve": 2e-4},
                {"profile": "balanced", "iterations": 50, "repulsion_strength": 0.10, "k_repel": 24, "step_size": 0.08, "anchor_strength": 0.24, "max_step_factor": 0.015, "dense_only_quantile": 0.22, "two_phase": True, "phase_split": 0.35, "phase1_strength_mult": 1.45, "phase2_strength_mult": 0.7, "early_stop_patience": 10, "early_stop_min_improve": 2e-4},
                {"profile": "spread", "iterations": 70, "repulsion_strength": 0.14, "k_repel": 34, "step_size": 0.08, "anchor_strength": 0.20, "max_step_factor": 0.018, "dense_only_quantile": 0.28, "two_phase": True, "phase_split": 0.4, "phase1_strength_mult": 1.55, "phase2_strength_mult": 0.65, "early_stop_patience": 10, "early_stop_min_improve": 2e-4},
                {"profile": "strong", "iterations": 95, "repulsion_strength": 0.18, "k_repel": 44, "step_size": 0.09, "anchor_strength": 0.18, "max_step_factor": 0.020, "dense_only_quantile": 0.32, "two_phase": True, "phase_split": 0.4, "phase1_strength_mult": 1.65, "phase2_strength_mult": 0.6, "early_stop_patience": 12, "early_stop_min_improve": 2e-4},
            ]
        else:
            coarse_grid = [
                {"profile": "mild", "iterations": 20, "repulsion_strength": 0.04, "k_repel": 12, "step_size": 0.06, "anchor_strength": 0.30, "max_step_factor": 0.010, "dense_only_quantile": 0.15, "two_phase": False, "early_stop_patience": 6, "early_stop_min_improve": 2e-4},
                {"profile": "balanced", "iterations": 30, "repulsion_strength": 0.07, "k_repel": 18, "step_size": 0.07, "anchor_strength": 0.28, "max_step_factor": 0.012, "dense_only_quantile": 0.20, "two_phase": True, "phase_split": 0.35, "phase1_strength_mult": 1.35, "phase2_strength_mult": 0.75, "early_stop_patience": 8, "early_stop_min_improve": 2e-4},
                {"profile": "spread", "iterations": 45, "repulsion_strength": 0.10, "k_repel": 26, "step_size": 0.08, "anchor_strength": 0.24, "max_step_factor": 0.015, "dense_only_quantile": 0.25, "two_phase": True, "phase_split": 0.4, "phase1_strength_mult": 1.45, "phase2_strength_mult": 0.7, "early_stop_patience": 10, "early_stop_min_improve": 2e-4},
            ]

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
            "score_mean": 0.0,
            "score_std": 0.0,
            "n_seed_runs": len(seeds),
            "step_size": 0.0,
            "anchor_strength": 0.0,
            "max_step_factor": 0.0,
            "dense_only_quantile": "",
            "two_phase": False,
            "phase_split": 0.35,
            "phase1_strength_mult": 1.5,
            "phase2_strength_mult": 0.6,
            "early_stop_patience": 0,
            "early_stop_min_improve": 0.0,
        }
        all_candidates = []
        cfg_pbar = tqdm(coarse_grid, desc=f"{name}: coarse", unit=" cfg", leave=False)
        for cfg in cfg_pbar:
            cfg_pbar.set_postfix_str(
                f"{cfg['profile']} it={cfg['iterations']} r={cfg['repulsion_strength']} k={cfg['k_repel']}"
            )
            res = _evaluate_forceatlas_candidate_multi(
                X, cfg, seeds=seeds, base_seed=args.seed
            )
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
                "step_size": cfg.get("step_size", 0.1),
                "anchor_strength": cfg.get("anchor_strength", 0.0),
                "max_step_factor": cfg.get("max_step_factor", 0.0),
                "dense_only_quantile": cfg.get("dense_only_quantile", ""),
                "two_phase": cfg.get("two_phase", False),
                "phase_split": cfg.get("phase_split", 0.35),
                "phase1_strength_mult": cfg.get("phase1_strength_mult", 1.5),
                "phase2_strength_mult": cfg.get("phase2_strength_mult", 0.6),
                "early_stop_patience": cfg.get("early_stop_patience", 0),
                "early_stop_min_improve": cfg.get("early_stop_min_improve", 1e-4),
                **res,
            }
            all_candidates.append(cand)
            if best is None or cand["score"] > best["score"]:
                best = cand
                cfg_pbar.set_postfix_str(
                    f"best={cfg['profile']} score={best['score']:.4f}±{best['score_std']:.4f}"
                )

        # Fine search around top 2 coarse candidates.
        top_coarse = sorted(all_candidates, key=lambda x: x["score"], reverse=True)[:2]
        fine_grid = _build_fine_grid(top_coarse)
        if fine_grid:
            fine_pbar = tqdm(fine_grid, desc=f"{name}: fine", unit=" cfg", leave=False)
            for cfg in fine_pbar:
                fine_pbar.set_postfix_str(
                    f"{cfg['profile']} it={cfg['iterations']} r={cfg['repulsion_strength']:.3f} k={cfg['k_repel']}"
                )
                res = _evaluate_forceatlas_candidate_multi(
                    X, cfg, seeds=seeds, base_seed=args.seed
                )
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
                    "step_size": cfg.get("step_size", 0.1),
                    "anchor_strength": cfg.get("anchor_strength", 0.0),
                    "max_step_factor": cfg.get("max_step_factor", 0.0),
                    "dense_only_quantile": cfg.get("dense_only_quantile", ""),
                    "two_phase": cfg.get("two_phase", False),
                    "phase_split": cfg.get("phase_split", 0.35),
                    "phase1_strength_mult": cfg.get("phase1_strength_mult", 1.5),
                    "phase2_strength_mult": cfg.get("phase2_strength_mult", 0.6),
                    "early_stop_patience": cfg.get("early_stop_patience", 0),
                    "early_stop_min_improve": cfg.get("early_stop_min_improve", 1e-4),
                    **res,
                }
                if best is None or cand["score"] > best["score"]:
                    best = cand
                    fine_pbar.set_postfix_str(
                        f"best={cfg['profile']} score={best['score']:.4f}±{best['score_std']:.4f}"
                    )

        if best is not None:
            # Only recommend when improvement is meaningful and not just rescaling.
            best["recommend_postprocess"] = bool(
                (best["profile"] != "none")
                and (best["score"] > 0.10)
                and (best["center_relief"] > 0.01)
                and (best["log_scale_shift"] < 0.55)
                and (best["rank_corr"] > 0.90)
                and (best["score_std"] < 0.08)
            )
            best["seed_list"] = ",".join(str(s) for s in seeds)
            rows.append(best)

    if not rows:
        raise RuntimeError("No valid recommendations generated.")

    out_df = pd.DataFrame(rows).sort_values(["recommend_postprocess", "score"], ascending=[False, False])
    out_df["global_drift_risk"] = out_df.apply(_global_drift_risk, axis=1)
    out_df.to_csv(out_path, sep="\t", index=False)

    print("ForceAtlas postprocess recommendation report")
    print("=" * 90)
    print(f"Input: {path}")
    print(f"Output: {out_path}")
    print(
        f"{'layout':<30} {'rec?':<6} {'risk':<5} {'score':>8} {'profile':<10} {'iters':>6} {'repel':>8} {'k':>5}"
    )
    print("-" * 90)
    for _, r in out_df.iterrows():
        rec = "yes" if bool(r["recommend_postprocess"]) else "no"
        print(
            f"{str(r['layout']):<30} {rec:<6} {str(r['global_drift_risk']):<5} {float(r['score']):>8.4f} "
            f"{str(r['profile']):<10} {int(r['iterations']):>6} "
            f"{float(r['repulsion_strength']):>8.3f} {int(r['k_repel']):>5}"
        )


if __name__ == "__main__":
    main()

