"""
Step 8: Read final layout TSV and create distribution charts per method.
One row per method, three columns (x, y, z). Saves to output/distribution_charts.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_tune import OUTPUT_DIR

LAYOUT_DECOMPRESSION_TSV = OUTPUT_DIR / "layout_decompression.tsv"
LAYOUT_MULTI_TSV = OUTPUT_DIR / "layout_multi.tsv"
CHARTS_PATH = OUTPUT_DIR / "distribution_charts.png"


def get_layout_path():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if LAYOUT_DECOMPRESSION_TSV.exists():
        return LAYOUT_DECOMPRESSION_TSV
    if LAYOUT_MULTI_TSV.exists():
        return LAYOUT_MULTI_TSV
    return OUTPUT_DIR / "layout.tsv"


def discover_methods(df):
    """Return list of method names that have x_*, y_*, z_* columns (in order)."""
    methods = []
    for c in df.columns:
        if c.startswith("x_") and c[2:] not in methods:
            base = c[2:]
            if f"y_{base}" in df.columns and f"z_{base}" in df.columns:
                xcol = df[c]
                if pd.api.types.is_numeric_dtype(xcol):
                    methods.append(base)
    return methods


def main():
    layout_path = get_layout_path()
    if not layout_path.exists():
        print(f"Layout not found: {layout_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(layout_path, sep="\t")
    methods = discover_methods(df)
    if not methods:
        print("No x_*/y_*/z_* method columns found.", file=sys.stderr)
        sys.exit(1)

    n_methods = len(methods)
    fig, axes = plt.subplots(
        n_methods, 3,
        figsize=(10, 2.2 * n_methods),
        sharex="col",
        squeeze=False,
    )
    fig.suptitle(f"Layout distributions: {layout_path.name}", fontsize=12)

    for i, method in enumerate(methods):
        x = df[f"x_{method}"].astype(float).values
        y = df[f"y_{method}"].astype(float).values
        z = df[f"z_{method}"].astype(float).values
        for j, (arr, label) in enumerate(zip([x, y, z], ["x", "y", "z"])):
            ax = axes[i, j]
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            ax.hist(arr, bins=min(80, max(20, len(arr) // 200)), density=True, alpha=0.8, color="steelblue", edgecolor="none")
            ax.set_ylabel("density", fontsize=8)
            if i == 0:
                ax.set_title(label, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{method}\ndensity", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    plt.setp(axes[-1, :], xlabel="value")
    plt.tight_layout()
    CHARTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {CHARTS_PATH} ({n_methods} methods, 3 cols each)")


if __name__ == "__main__":
    main()
