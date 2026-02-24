"""
Distribution analysis for layout (x, y, z) data.
Supports TSV path, DataFrame, or (N, 3) numpy array.
Multi-method layout TSVs (x_*, y_*, z_*) can be analyzed per method with ball detection.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Flag as "ball" (collapsed layout) when range or std is below this
BALL_RANGE_THRESHOLD = 0.05
BALL_STD_THRESHOLD = 0.02


def discover_methods(df):
    """Return list of method names that have x_*, y_*, z_* numeric columns (in order)."""
    methods = []
    for c in df.columns:
        if c.startswith("x_") and c[2:] not in methods:
            base = c[2:]
            if f"y_{base}" in df.columns and f"z_{base}" in df.columns:
                xcol = df[c]
                if pd.api.types.is_numeric_dtype(xcol):
                    methods.append(base)
    return methods


def get_xyz_columns(df):
    """
    Detect x, y, z columns from a DataFrame.
    Supports "x", "y", "z" or last 3 numeric columns.
    Returns list of 3 column names.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    lower = {c.lower(): c for c in df.columns}
    if "x" in lower and "y" in lower and "z" in lower:
        return [lower["x"], lower["y"], lower["z"]]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) < 3:
        raise ValueError(
            f"Need at least 3 numeric columns for x,y,z; found {len(numeric)}"
        )
    return numeric[-3:]


def compute_axis_stats(arr, name):
    """
    Compute statistics for a 1d array.
    Returns dict with min, max, mean, std, median, q1, q3, iqr,
    and percentiles p0.5, p1, p5, p25, p50, p75, p95, p99, p99.5.
    """
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.ravel(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan,
            "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan,
            **{f"p{p}": np.nan for p in [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]},
        }
    percentiles = [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5]
    pvals = np.percentile(arr, percentiles)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        **{f"p{p}": float(pvals[i]) for i, p in enumerate(percentiles)},
    }


def analyze_layout(source):
    """
    Analyze layout from source (path to TSV, DataFrame, or (N,3) numpy array).
    Returns dict mapping axis name -> stats (from compute_axis_stats).
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source, sep="\t")
        cols = get_xyz_columns(df)
        axes = ["x", "y", "z"]
        return {
            axes[i]: compute_axis_stats(df[cols[i]].values, axes[i])
            for i in range(3)
        }
    if isinstance(source, pd.DataFrame):
        cols = get_xyz_columns(source)
        axes = ["x", "y", "z"]
        return {
            axes[i]: compute_axis_stats(source[cols[i]].values, axes[i])
            for i in range(3)
        }
    if isinstance(source, np.ndarray) and source.ndim == 2 and source.shape[1] == 3:
        axes = ["x", "y", "z"]
        return {
            axes[i]: compute_axis_stats(source[:, i], axes[i]) for i in range(3)
        }
    raise TypeError(
        "source must be a path (str/Path), DataFrame, or (N,3) numpy array"
    )


def print_report(stats, title="Layout distribution"):
    """Print a readable summary of axis stats."""
    print(f"\n{title}")
    print("=" * (len(title) + 2))
    for axis in ["x", "y", "z"]:
        s = stats.get(axis, {})
        if not s:
            continue
        print(f"\n  {axis}:")
        print(f"    min={s['min']:.4f}, max={s['max']:.4f}")
        print(f"    mean={s['mean']:.4f}, std={s['std']:.4f}, median={s['median']:.4f}")
        print(f"    q1={s['q1']:.4f}, q3={s['q3']:.4f}, iqr={s['iqr']:.4f}")
        print(
            f"    percentiles: p0.5={s['p0.5']:.4f}, p1={s['p1']:.4f}, p5={s['p5']:.4f}, "
            f"p25={s['p25']:.4f}, p50={s['p50']:.4f}, p75={s['p75']:.4f}, "
            f"p95={s['p95']:.4f}, p99={s['p99']:.4f}, p99.5={s['p99.5']:.4f}"
        )
    print()


def analyze_layout_multi(source):
    """
    Analyze a multi-method layout (TSV or DataFrame with x_*, y_*, z_* columns).
    Returns list of (method_name, stats_dict) where stats_dict has keys "x", "y", "z"
    each with the result of compute_axis_stats.
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source, sep="\t")
    elif isinstance(source, pd.DataFrame):
        df = source
    else:
        raise TypeError("source must be a path or DataFrame")
    methods = discover_methods(df)
    if not methods:
        return []
    out = []
    for method in methods:
        x = df[f"x_{method}"].astype(float).values
        y = df[f"y_{method}"].astype(float).values
        z = df[f"z_{method}"].astype(float).values
        out.append((
            method,
            {
                "x": compute_axis_stats(x, "x"),
                "y": compute_axis_stats(y, "y"),
                "z": compute_axis_stats(z, "z"),
            },
        ))
    return out


def print_multi_report(multi_stats, title="Layout distributions (multi-method)", ball_range_thresh=BALL_RANGE_THRESHOLD, ball_std_thresh=BALL_STD_THRESHOLD):
    """
    Print a table of per-method stats and flag methods that look like a annoying "ball" (collapsed).
    """
    print(f"\n{title}")
    print("=" * min(100, len(title) + 2))
    print(f"{'method':<35} {'range_x':>10} {'range_y':>10} {'range_z':>10} {'std_x':>10} {'std_y':>10} {'std_z':>10}  BALL?")
    print("-" * 100)
    for method, stats in multi_stats:
        rx = stats["x"]["max"] - stats["x"]["min"]
        ry = stats["y"]["max"] - stats["y"]["min"]
        rz = stats["z"]["max"] - stats["z"]["min"]
        sx = stats["x"]["std"]
        sy = stats["y"]["std"]
        sz = stats["z"]["std"]
        ball = (
            (rx < ball_range_thresh and ry < ball_range_thresh and rz < ball_range_thresh)
            or (sx < ball_std_thresh and sy < ball_std_thresh and sz < ball_std_thresh)
        )
        flag = "  *** BALL (collapsed) ***" if ball else ""
        print(f"{method:<35} {rx:>10.4f} {ry:>10.4f} {rz:>10.4f} {sx:>10.4f} {sy:>10.4f} {sz:>10.4f}  {flag}")
    print()


def run_analysis_multi(source, print_to_stdout=True, title=None):
    """
    Run analyze_layout_multi on path or DataFrame; optionally print table + ball flags.
    Returns list of (method_name, stats_dict).
    """
    multi_stats = analyze_layout_multi(source)
    if print_to_stdout and multi_stats:
        if title is None:
            title = f"Layout distributions: {source}" if isinstance(source, (str, Path)) else "Layout distributions (multi-method)"
        print_multi_report(multi_stats, title=title)
    return multi_stats


def run_analysis(layout_path, print_to_stdout=True):
    """
    Load TSV at layout_path. If it has multiple x_*/y_*/z_* methods, run analyze_layout_multi;
    else run single analyze_layout. Optionally print report.
    Returns either the multi-method list or the single stats dict.
    """
    path = Path(layout_path) if isinstance(layout_path, str) else layout_path
    df = pd.read_csv(path, sep="\t")
    methods = discover_methods(df)
    if methods:
        return run_analysis_multi(df, print_to_stdout=print_to_stdout, title=f"Layout distributions: {path}")
    stats = analyze_layout(df)
    if print_to_stdout:
        print_report(stats, title=f"Layout distribution: {path}")
    return stats


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/layout.tsv"
    run_analysis(path, print_to_stdout=True)
