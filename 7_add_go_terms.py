"""
Step 7: Add a column 'go_terms' to the final layout TSV.
GO terms are comma-separated; any commas inside term names are removed first
so the TSV stays well-formed.

Inputs: output/edge_list.parquet, output/node_map.parquet, and a layout TSV.
Output: same layout TSV with new column go_terms (overwrites by default).
Usage: python 7_add_go_terms.py [layout.tsv] [out.tsv]
  If layout.tsv is omitted, uses layout_decompression.tsv > layout_multi.tsv > layout.tsv.
  If out.tsv is omitted, overwrites layout.tsv.
"""
import sys
from pathlib import Path

import pandas as pd

from config_tune import (
    EDGE_LIST_PARQUET,
    NODE_MAP_PARQUET,
    GO_OBO_PATH,
    OUTPUT_DIR,
    LAYOUT_TSV,
)
from utils.go_terms_per_node import (
    build_protein_to_go_terms,
    add_go_terms_column,
    load_go_id_to_name,
    go_series_to_readable,
)

LAYOUT_MULTI_TSV = OUTPUT_DIR / "layout_multi.tsv"
LAYOUT_DECOMPRESSION_TSV = OUTPUT_DIR / "layout_decompression.tsv"


def choose_layout_path() -> Path:
    """Prefer decompression > multi > single layout."""
    if LAYOUT_DECOMPRESSION_TSV.exists():
        return LAYOUT_DECOMPRESSION_TSV
    if LAYOUT_MULTI_TSV.exists():
        return LAYOUT_MULTI_TSV
    return LAYOUT_TSV


def main():
    layout_path = Path(sys.argv[1]) if len(sys.argv) > 1 else choose_layout_path()
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else layout_path
    if not layout_path.exists():
        print(f"Layout not found: {layout_path}", file=sys.stderr)
        print("Run a layout step (5_project_3d, 5_multiproject_3d, or 6_decompression_layouts) first.")
        sys.exit(1)

    print("Building protein -> GO terms from graph...")
    go_series = build_protein_to_go_terms(EDGE_LIST_PARQUET, NODE_MAP_PARQUET)
    print(f"GO terms for {len(go_series)} proteins")

    go_readable_series = None
    if GO_OBO_PATH.exists():
        print("Loading GO term names from OBO...")
        id_to_name = load_go_id_to_name(GO_OBO_PATH)
        go_readable_series = go_series_to_readable(go_series, id_to_name)
        print(f"Names for {len(id_to_name)} GO terms")
    else:
        print("OBO not found; skipping go_terms_readable column.")

    print(f"Reading {layout_path}...")
    layout_df = pd.read_csv(layout_path, sep="\t")
    for col in ("go_terms", "go_terms_readable"):
        if col in layout_df.columns:
            layout_df = layout_df.drop(columns=[col])
    layout_df = add_go_terms_column(
        layout_df,
        go_series,
        node_id_column="node_id",
        go_column="go_terms",
        go_readable_series=go_readable_series,
        go_readable_column="go_terms_readable",
    )

    layout_df.to_csv(out_path, sep="\t", index=False)
    cols = "go_terms" + (" + go_terms_readable" if go_readable_series is not None else "")
    print(f"Wrote {out_path} with {cols} ({len(layout_df)} rows)")
    print("Step 7 done.")


if __name__ == "__main__":
    main()
