"""
Merge two TSVs: add columns from the second file to the first by matching an id column.

Row count is always that of main (argv1). Extra (argv2) is used only for lookup;
duplicate ids in extra are dropped (first kept) so the merge never adds rows.

Usage:
  python -m utils.tsv_merge [--prepend PREFIX | -p PREFIX] <main.tsv> <extra.tsv> <output.tsv> <id_column_main> <id_column_extra> [col1 col2 ...]
  If no columns specified: merge all columns from extra.
  If columns specified: merge only those columns from extra.
  --prepend PREFIX  (or -p): prepend PREFIX to every column name added from extra.
"""
import sys
import pandas as pd


def main():
    # Parse optional --prepend / -p
    prepend = None
    args = sys.argv[1:]
    positional = []
    i = 0
    while i < len(args):
        if args[i] in ("--prepend", "-p"):
            if i + 1 >= len(args):
                print("Error: --prepend requires a value", file=sys.stderr)
                sys.exit(1)
            prepend = args[i + 1]
            i += 2
        else:
            positional.append(args[i])
            i += 1

    if len(positional) < 5:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)

    path_main = positional[0]
    path_extra = positional[1]
    path_out = positional[2]
    id_main = positional[3]
    id_extra = positional[4]
    cols_extra = positional[5:] if len(positional) > 5 else None

    df_main = pd.read_csv(path_main, sep="\t")
    df_extra = pd.read_csv(path_extra, sep="\t")

    if id_main not in df_main.columns:
        print(f"Error: column '{id_main}' not in {path_main}", file=sys.stderr)
        sys.exit(1)
    if id_extra not in df_extra.columns:
        print(f"Error: column '{id_extra}' not in {path_extra}", file=sys.stderr)
        sys.exit(1)

    if cols_extra is not None and len(cols_extra) > 0:
        for c in cols_extra:
            if c not in df_extra.columns:
                print(f"Error: column '{c}' not in {path_extra}", file=sys.stderr)
                sys.exit(1)
        cols = [id_extra] + [c for c in cols_extra if c != id_extra]
        df_extra = df_extra[cols]

    # Dedupe extra on id so left merge never adds rows (one row per main id).
    df_extra = df_extra.drop_duplicates(subset=[id_extra], keep="first")

    n_main = len(df_main)
    merged = df_main.merge(df_extra, left_on=id_main, right_on=id_extra, how="left")
    if id_extra in merged.columns and id_extra != id_main:
        merged = merged.drop(columns=[id_extra])
    if prepend:
        extra_cols_in_merged = [c for c in merged.columns if c in df_extra.columns]
        merged = merged.rename(columns={c: prepend + c for c in extra_cols_in_merged})
    n_out = len(merged)
    merged.to_csv(path_out, sep="\t", index=False)
    print(f"Main: {n_main} rows, extra: {len(df_extra)} rows (after dedupe) -> output: {n_out} rows, {len(merged.columns)} columns")
    if n_out != n_main:
        print(f"Warning: output row count ({n_out}) != main row count ({n_main})", file=sys.stderr)
    print(f"Wrote {path_out}")


if __name__ == "__main__":
    main()
