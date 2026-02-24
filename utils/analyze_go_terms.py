"""
Analyze GO term usage across proteins and plot top terms.

Outputs:
- A TSV with GO term frequencies (count + fraction of proteins).
- A top-N horizontal bar chart (default top 100).

By default it reads GO annotations from the unified graph via:
  output/edge_list.parquet + output/node_map.parquet
If a layout TSV contains a `go_terms` column, you can use it instead.

Examples:
  python -m utils.analyze_go_terms
  python -m utils.analyze_go_terms --layout output/layout_decompression.tsv
  python -m utils.analyze_go_terms --top-k 100 --out-prefix output/go_terms
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config_tune import EDGE_LIST_PARQUET, GO_OBO_PATH, NODE_MAP_PARQUET, OUTPUT_DIR
from utils.go_terms_per_node import build_protein_to_go_terms, load_go_id_to_name


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze GO-term usage and create top-terms chart.")
    parser.add_argument(
        "--layout",
        type=str,
        default=None,
        help="Optional layout TSV with a `go_terms` column. If omitted, uses graph files.",
    )
    parser.add_argument(
        "--go-col",
        type=str,
        default="go_terms",
        help="GO-term column name when --layout is used (default: go_terms).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of terms to plot in bar chart (default: 100).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=str(OUTPUT_DIR / "go_terms"),
        help="Output prefix for files (default: output/go_terms).",
    )
    return parser.parse_args()


def _series_from_layout(layout_path: Path, go_col: str) -> pd.Series:
    df = pd.read_csv(layout_path, sep="\t")
    if go_col not in df.columns:
        raise ValueError(f"Column `{go_col}` not found in {layout_path}")
    s = df[go_col].fillna("").astype(str)
    if "node_id" in df.columns:
        s.index = df["node_id"].astype(str).values
    return s


def _explode_go_series(go_series: pd.Series) -> pd.DataFrame:
    # go_series values are comma-separated IDs without spaces by project convention.
    exploded = (
        go_series.astype(str)
        .str.split(",")
        .explode()
        .astype(str)
        .str.strip()
    )
    exploded = exploded[exploded != ""]
    term_counts = exploded.value_counts()
    n_proteins = int(len(go_series))
    out = pd.DataFrame(
        {
            "go_id": term_counts.index.astype(str),
            "protein_count": term_counts.values.astype(int),
        }
    )
    out["protein_fraction"] = out["protein_count"] / max(n_proteins, 1)
    out["protein_percent"] = 100.0 * out["protein_fraction"]
    return out


def _plot_top_terms(df_terms: pd.DataFrame, k: int, out_png: Path):
    top = df_terms.head(k).iloc[::-1]  # reverse so largest appears at top in barh
    if top.empty:
        raise ValueError("No GO terms available to plot.")

    fig_h = max(10.0, min(40.0, 0.28 * len(top) + 2.0))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.barh(top["go_label"], top["protein_count"])
    ax.set_xlabel("Protein count")
    ax.set_ylabel("GO term")
    ax.set_title(f"Top {len(top)} most widely used GO terms")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.layout:
        layout_path = Path(args.layout)
        print(f"Reading GO terms from layout: {layout_path}")
        go_series = _series_from_layout(layout_path, args.go_col)
    else:
        print("Building protein->GO terms from graph files...")
        go_series = build_protein_to_go_terms(EDGE_LIST_PARQUET, NODE_MAP_PARQUET)

    if len(go_series) == 0:
        raise ValueError("No GO annotations found.")

    print(f"Proteins with GO terms: {len(go_series)}")
    df_terms = _explode_go_series(go_series)

    id_to_name = load_go_id_to_name(GO_OBO_PATH)
    df_terms["go_name"] = df_terms["go_id"].map(id_to_name).fillna(df_terms["go_id"])
    df_terms["go_label"] = df_terms["go_id"] + " | " + df_terms["go_name"]

    top_k = max(1, int(args.top_k))
    out_tsv = out_prefix.with_name(out_prefix.name + "_term_frequency.tsv")
    out_png = out_prefix.with_name(out_prefix.name + f"_top{top_k}_barchart.png")

    df_terms.to_csv(out_tsv, sep="\t", index=False)
    _plot_top_terms(df_terms, k=top_k, out_png=out_png)

    print(f"Wrote frequency table: {out_tsv} ({len(df_terms)} terms)")
    print(f"Wrote top-{top_k} chart: {out_png}")
    print(f"\nTop {top_k} most widely used GO terms:")
    print(
        df_terms.head(top_k)[["go_id", "go_name", "protein_count", "protein_percent"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

