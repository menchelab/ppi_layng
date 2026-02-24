"""
Merge two TSVs: add all extra columns from the second file to the first by matching an id column.
Usage: python tsv_merge.py <main.tsv> <extra.tsv> <output.tsv> <id_column_main> <id_column_extra> [postfix]
  id_column_main: column name in main.tsv to join on
  id_column_extra: column name in extra.tsv to join on
  postfix: optional string to append to extra file's column names (except the id column)
"""
import sys
import pandas as pd


def main():
    if len(sys.argv) < 6:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(1)

    path_main = sys.argv[1]
    path_extra = sys.argv[2]
    path_out = sys.argv[3]
    id_main = sys.argv[4]
    id_extra = sys.argv[5]
    postfix = sys.argv[6] if len(sys.argv) > 6 else None

    df_main = pd.read_csv(path_main, sep="\t")
    df_extra = pd.read_csv(path_extra, sep="\t")

    if id_main not in df_main.columns:
        print(f"Error: column '{id_main}' not in {path_main}", file=sys.stderr)
        sys.exit(1)
    if id_extra not in df_extra.columns:
        print(f"Error: column '{id_extra}' not in {path_extra}", file=sys.stderr)
        sys.exit(1)

    if postfix:
        rename = {c: c + postfix for c in df_extra.columns}
        df_extra = df_extra.rename(columns=rename)
        right_key = id_extra + postfix
    else:
        right_key = id_extra

    merged = df_main.merge(df_extra, left_on=id_main, right_on=right_key, how="left")
    merged = merged.drop(columns=[right_key])
    merged.to_csv(path_out, sep="\t", index=False)
    print(f"Wrote {path_out} ({len(merged)} rows, {len(merged.columns)} columns)")


if __name__ == "__main__":
    main()
