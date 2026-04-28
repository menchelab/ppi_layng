"""
Normalize layout coordinates to 0.0–1.0 per layout.

For each layout (x_<name>, y_<name>, z_<name>), computes a single min/max
across all coordinates (x, y, z) and scales the entire layout into [0, 1].

Usage:
  python -m utils.normalize_layouts input/nodes.tsv output/nodes.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def discover_layout_names(df: pd.DataFrame) -> list[str]:
    """Return layout names that have x_*, y_*, z_* columns, in column order."""
    names = []
    for c in df.columns:
        if c.startswith("x_") and c[2:] not in names:
            base = c[2:]
            if f"y_{base}" in df.columns and f"z_{base}" in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    names.append(base)
    return names


def normalize_layout(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Normalize x, y, z to 0–1 using a single min/max across all coordinates.
    Returns (x_norm, y_norm, z_norm, prev_min, prev_max).
    """
    coords = np.concatenate([np.ravel(x), np.ravel(y), np.ravel(z)])
    valid = np.isfinite(coords)
    if not np.any(valid):
        return x, y, z, np.nan, np.nan
    prev_min = float(np.nanmin(coords))
    prev_max = float(np.nanmax(coords))
    span = prev_max - prev_min
    if span <= 0:
        return x, y, z, prev_min, prev_max
    x_norm = np.asarray(x, dtype=np.float64)
    y_norm = np.asarray(y, dtype=np.float64)
    z_norm = np.asarray(z, dtype=np.float64)
    x_norm = np.where(np.isfinite(x_norm), (x_norm - prev_min) / span, np.nan)
    y_norm = np.where(np.isfinite(y_norm), (y_norm - prev_min) / span, np.nan)
    z_norm = np.where(np.isfinite(z_norm), (z_norm - prev_min) / span, np.nan)
    return x_norm, y_norm, z_norm, prev_min, prev_max


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize layout coordinates to 0–1 per layout (min/max across x,y,z)."
    )
    parser.add_argument("input_tsv", type=str, help="Input nodes TSV (with x_*, y_*, z_* layout columns).")
    parser.add_argument("output_path", type=str, help="Output path (TSV or TXT).")
    args = parser.parse_args()

    inp = Path(args.input_tsv)
    out = Path(args.output_path)
    if not inp.exists():
        print(f"Error: input not found: {inp}", file=sys.stderr)
        return 1

    df = pd.read_csv(inp, sep="\t")
    names = discover_layout_names(df)
    if not names:
        print("No layouts found (need x_*, y_*, z_* for each).", file=sys.stderr)
        return 1

    out_df = df.copy()
    stats: list[dict] = []

    for name in names:
        xcol, ycol, zcol = f"x_{name}", f"y_{name}", f"z_{name}"
        x = df[xcol].to_numpy(dtype=np.float64)
        y = df[ycol].to_numpy(dtype=np.float64)
        z = df[zcol].to_numpy(dtype=np.float64)

        x_n, y_n, z_n, prev_min, prev_max = normalize_layout(x, y, z)
        out_df[xcol] = x_n
        out_df[ycol] = y_n
        out_df[zcol] = z_n

        span = prev_max - prev_min if np.isfinite(prev_min) and np.isfinite(prev_max) else np.nan
        stats.append({
            "layout": name,
            "prev_min": prev_min,
            "prev_max": prev_max,
            "prev_span": span,
            "n_points": int(np.sum(np.isfinite(x) & np.isfinite(y) & np.isfinite(z))),
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if out.suffix.lower() in (".tsv", ".txt", "") else ","
    out_df.to_csv(out, sep=sep, index=False)

    print("Layout normalization stats:")
    print("-" * 70)
    for s in stats:
        pct = ""
        if s["prev_span"] and s["prev_span"] > 0:
            mid = (s["prev_min"] + s["prev_max"]) / 2
            pct = f"  center={mid:.4f}"
        print(
            f"  {s['layout']:<12}  prev_min={s['prev_min']:>12.6f}  prev_max={s['prev_max']:>12.6f}  "
            f"span={s['prev_span']:>10.4f}  n={s['n_points']:>6}{pct}"
        )
    print("-" * 70)
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
