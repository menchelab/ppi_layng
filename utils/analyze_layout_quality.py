"""
Analyze multi-layout nodes.tsv and suggest improvement directions.

This is a read-only analytics helper for files like temp/ikersub_v2/nodes.tsv.
It computes per-layout quality metrics on normalized coordinates so different
methods are comparable, ranks layouts, and (optionally) links to umap_runs.tsv
to surface parameter tendencies.

Usage:
  python -m utils.analyze_layout_quality temp/ikersub_v2/nodes.tsv
  python -m utils.analyze_layout_quality temp/ikersub_v2/nodes.tsv --runs temp/ikersub_v2/umap_runs.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Analyze layout quality and suggest improvements.")
    p.add_argument("nodes_tsv", type=str, help="nodes.tsv / layout TSV with x_* y_* z_* columns")
    p.add_argument(
        "--runs",
        type=str,
        default=None,
        help="Optional runs metadata TSV (default: sibling umap_runs.tsv if exists).",
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=3000,
        help="Max nodes sampled per layout for expensive metrics (default: 3000).",
    )
    return p.parse_args()


def discover_layouts(df: pd.DataFrame):
    out = []
    for c in df.columns:
        if not c.startswith("x_"):
            continue
        base = c[2:]
        y = f"y_{base}"
        z = f"z_{base}"
        if y in df.columns and z in df.columns:
            out.append(base)
    return out


def sampled_nn(X: np.ndarray):
    from sklearn.neighbors import NearestNeighbors

    if len(X) < 2:
        return np.array([], dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, 1]


def normalize_layout(X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    X0 = X - np.median(X, axis=0, keepdims=True)
    s = np.std(X0, axis=0)
    s[s == 0] = 1.0
    return X0 / s


def metric_row(name: str, X: np.ndarray):
    Xn = normalize_layout(X)
    r = np.linalg.norm(Xn, axis=1)
    nn = sampled_nn(Xn)

    nn_med = float(np.median(nn)) if len(nn) else np.nan
    nn_p95 = float(np.percentile(nn, 95)) if len(nn) else np.nan
    nn_cv = float(np.std(nn) / np.mean(nn)) if len(nn) and np.mean(nn) > 0 else np.nan

    center_r08 = float(np.mean(r <= 0.8))
    center_r10 = float(np.mean(r <= 1.0))
    outlier_ratio = float(np.percentile(r, 99) / max(np.percentile(r, 50), 1e-9))
    axis_std = np.std(X, axis=0)
    axis_balance = float(np.min(axis_std) / max(np.max(axis_std), 1e-12))

    # Bounded quality components (higher is better)
    crowd_score = 1.0 - np.clip(center_r10, 0.0, 1.0)
    sep_score = float(np.tanh(max(nn_med, 0.0)))
    outlier_pen = np.clip((outlier_ratio - 4.0) / 4.0, 0.0, 1.0)
    balance_score = np.clip(axis_balance, 0.0, 1.0)

    quality = (
        0.40 * crowd_score
        + 0.30 * sep_score
        + 0.20 * (1.0 - outlier_pen)
        + 0.10 * balance_score
    )

    suggestions = []
    if center_r10 > 0.30:
        suggestions.append("crowded: try mild anchored postprocess")
    if nn_med < 0.10:
        suggestions.append("low local separation: prefer larger n_neighbors")
    if outlier_ratio > 6.0:
        suggestions.append("outlier-heavy: reduce repulsion/epochs")
    if axis_balance < 0.20:
        suggestions.append("anisotropic: consider metric/cfg with better balance")
    if not suggestions:
        suggestions.append("looks stable: keep as candidate baseline")

    return {
        "layout": name,
        "quality_score": float(quality),
        "center_frac_r0.8": center_r08,
        "center_frac_r1.0": center_r10,
        "nn_med_norm": nn_med,
        "nn_p95_norm": nn_p95,
        "nn_cv_norm": nn_cv,
        "outlier_ratio_r99_r50": outlier_ratio,
        "axis_balance_min_over_max_std": axis_balance,
        "suggestion": "; ".join(suggestions),
    }


def parse_layout_run_id(layout_name: str):
    # adju10 / adjp4 / adjt2 / adjh3 / adjf4 / adjs2
    for pref, algo in [
        ("adju", "umap"),
        ("adjp", "pacmap"),
        ("adjt", "trimap"),
        ("adjh", "phate"),
        ("adjf", "forceatlas2"),
        ("adjs", "spring"),
    ]:
        if layout_name.startswith(pref):
            rid = layout_name[len(pref):]
            if rid.isdigit():
                return algo, int(rid)
    return None, None


def append_param_tendencies(report_df: pd.DataFrame, runs_df: pd.DataFrame):
    merged = report_df.copy()
    algo = []
    run_id = []
    for l in merged["layout"]:
        a, r = parse_layout_run_id(str(l))
        algo.append(a)
        run_id.append(r)
    merged["algo"] = algo
    merged["run_id"] = run_id

    runs = runs_df.copy()
    if "run_id" in runs.columns:
        runs["run_id"] = pd.to_numeric(runs["run_id"], errors="coerce")
    if "algo" not in runs.columns:
        runs["algo"] = "unknown"

    joined = merged.merge(runs, on=["algo", "run_id"], how="left", suffixes=("", "_run"))

    tendencies = []
    for a, g in joined.groupby("algo", dropna=True):
        if a == "unknown" or len(g) < 3:
            continue
        tendencies.append(
            {
                "algo": a,
                "n_layouts": int(len(g)),
                "best_layout": g.sort_values("quality_score", ascending=False).iloc[0]["layout"],
                "best_quality_score": float(g["quality_score"].max()),
                "median_quality_score": float(g["quality_score"].median()),
            }
        )
    return joined, pd.DataFrame(tendencies)


def main():
    args = parse_args()
    nodes_path = Path(args.nodes_tsv)
    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes TSV not found: {nodes_path}")

    runs_path = Path(args.runs) if args.runs else (nodes_path.parent / "umap_runs.tsv")
    use_runs = runs_path.exists()

    df = pd.read_csv(nodes_path, sep="\t")
    layouts = discover_layouts(df)
    if not layouts:
        raise ValueError("No x_*/y_*/z_* layouts found in TSV.")

    rng = np.random.default_rng(42)
    rows = []
    for name in layouts:
        X = np.column_stack(
            [
                df[f"x_{name}"].to_numpy(dtype=np.float64),
                df[f"y_{name}"].to_numpy(dtype=np.float64),
                df[f"z_{name}"].to_numpy(dtype=np.float64),
            ]
        )
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) > args.sample_n:
            idx = rng.choice(len(X), size=args.sample_n, replace=False)
            X = X[idx]
        rows.append(metric_row(name, X))

    report = pd.DataFrame(rows).sort_values("quality_score", ascending=False)
    out_report = nodes_path.parent / "layout_quality_report.tsv"
    report.to_csv(out_report, sep="\t", index=False)

    out_summary = nodes_path.parent / "layout_quality_summary.md"
    lines = []
    lines.append("# Layout Quality Summary")
    lines.append("")
    lines.append(f"- Input: `{nodes_path}`")
    lines.append(f"- Layouts analyzed: **{len(report)}**")
    lines.append("")
    lines.append("## Top 8 Layouts")
    lines.append("")
    top = report.head(8)
    for _, r in top.iterrows():
        lines.append(
            f"- `{r['layout']}` | score={r['quality_score']:.4f} | center@1.0={r['center_frac_r1.0']:.2%} | "
            f"nn_med={r['nn_med_norm']:.4f} | outlier_ratio={r['outlier_ratio_r99_r50']:.2f}"
        )
    lines.append("")

    if use_runs:
        runs_df = pd.read_csv(runs_path, sep="\t")
        joined, tend = append_param_tendencies(report, runs_df)
        out_joined = nodes_path.parent / "layout_quality_with_runs.tsv"
        joined.to_csv(out_joined, sep="\t", index=False)
        lines.append("## Algorithm Tendencies")
        lines.append("")
        if len(tend):
            for _, t in tend.sort_values("median_quality_score", ascending=False).iterrows():
                lines.append(
                    f"- `{t['algo']}`: best `{t['best_layout']}` (score={t['best_quality_score']:.4f}), "
                    f"median score={t['median_quality_score']:.4f} over {int(t['n_layouts'])} layouts"
                )
        else:
            lines.append("- Not enough run metadata overlap for tendencies.")
        lines.append("")
        lines.append(f"- Detailed merged table: `{out_joined}`")
    else:
        lines.append("- No runs metadata found (`umap_runs.tsv`) to compute parameter tendencies.")

    lines.append("")
    lines.append("## Suggested Next Actions")
    lines.append("")
    lines.append("- Start from top 2-3 layouts by `quality_score` and inspect visually.")
    lines.append("- Only apply postprocess to layouts with high center fraction and low nn_med.")
    lines.append("- Avoid aggressive postprocess when outlier ratio is already high.")
    out_summary.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_report}")
    print(f"Wrote {out_summary}")
    if use_runs:
        print(f"Used runs metadata: {runs_path}")


if __name__ == "__main__":
    main()

