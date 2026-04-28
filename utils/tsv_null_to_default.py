"""
Replace NaN/null values in a TSV column with a default value. Overwrites the file in place.

Usage:
  python -m utils.tsv_null_to_default file.tsv column_name default_value
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace NaN/null in a TSV column with a default value. Saves to same file."
    )
    parser.add_argument("tsv_path", type=str, help="Input/output TSV file (overwritten).")
    parser.add_argument("column", type=str, help="Column to fill.")
    parser.add_argument("default", type=str, help="Default value for null/NaN (string, parsed if numeric).")
    args = parser.parse_args()

    path = Path(args.tsv_path)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path, sep="\t", na_values=["", "null", "nan", "NA", "None", "NULL", "NaN"])
    if args.column not in df.columns:
        print(f"Error: column '{args.column}' not found. Columns: {list(df.columns)}", file=sys.stderr)
        return 1

    default = args.default
    n_null = int(df[args.column].isna().sum())
    col = df[args.column].astype(object)
    df[args.column] = col.where(col.notna(), default)

    df.to_csv(path, sep="\t", index=False)
    print(f"Replaced {n_null} null/NaN in '{args.column}' with {repr(default)}. Saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
