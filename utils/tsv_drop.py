#!/usr/bin/env python3
"""
Drop named columns from a TSV file.

Reads argv[1] as TSV, drops all columns named by argv[3], argv[4], ...,
and writes the result to argv[2]. Column names that are not present in
the file are ignored (no error).

Column names support wildcards: * matches any substring, ? matches any
single character. E.g. asdf* drops columns that start with "asdf".

Usage: python tsv_drop.py <input.tsv> <output.tsv> <col1> [<col2> ...]

Example: python tsv_drop.py nodes.tsv nodes_slim.tsv gene_id gene_name
Example: python tsv_drop.py nodes.tsv nodes_slim.tsv asdf*
"""

import sys
import csv
import fnmatch


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python tsv_drop.py <input.tsv> <output.tsv> <col1> [<col2> ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    drop_specs = sys.argv[3:]

    try:
        with open(in_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            headers = reader.fieldnames
            if not headers:
                print("Error: input file has no header row", file=sys.stderr)
                sys.exit(1)

            to_drop = set()
            for spec in drop_specs:
                if "*" in spec or "?" in spec:
                    to_drop.update(h for h in headers if fnmatch.fnmatch(h, spec))
                else:
                    to_drop.add(spec)

            new_headers = [h for h in headers if h not in to_drop]
            if len(new_headers) == 0:
                print("Error: would drop all columns; aborting", file=sys.stderr)
                sys.exit(1)

            rows = list(reader)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_headers, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    except FileNotFoundError:
        print(f"Error: file not found: {in_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    dropped = [h for h in headers if h in to_drop]
    if dropped:
        print(f"Dropped {len(dropped)} column(s): {', '.join(dropped)}")
    print(f"Wrote {len(rows)} rows, {len(new_headers)} columns -> {out_path}")


if __name__ == "__main__":
    main()
