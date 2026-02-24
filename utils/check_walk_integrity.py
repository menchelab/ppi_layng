"""
Walk integrity checks for node2vec outputs.

Checks:
- shape and dtype
- invalid token count (-1 padding)
- node 0 frequency (to catch padding contamination)
- unique node coverage
- top frequent node IDs
- invalid ratio per walk position

Usage:
  python -m utils.check_walk_integrity
  python -m utils.check_walk_integrity /path/to/walks.npy
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from config_tune import WALKS_PARQUET, NODE_MAP_PARQUET


def load_walks(path: Path):
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        cols = [c for c in df.columns if c.startswith("step_")]
        return df[cols].to_numpy(dtype=np.int64)
    raise ValueError(f"Unsupported walks path: {path}")


def main():
    if len(sys.argv) > 1:
        wpath = Path(sys.argv[1])
    else:
        npy = WALKS_PARQUET.with_suffix(".npy")
        wpath = npy if npy.exists() else WALKS_PARQUET

    if not wpath.exists():
        raise FileNotFoundError(f"Walk file not found: {wpath}")

    walks = load_walks(wpath).astype(np.int64, copy=False)
    n_walks, walk_len = walks.shape
    invalid = walks < 0
    valid = ~invalid
    total = walks.size
    n_invalid = int(invalid.sum())
    n_valid = int(valid.sum())

    valid_vals = walks[valid]
    uniq = np.unique(valid_vals) if n_valid > 0 else np.array([], dtype=np.int64)
    node0_count = int(np.sum(valid_vals == 0)) if n_valid > 0 else 0

    node_map = pd.read_parquet(NODE_MAP_PARQUET)
    if "int_id" in node_map.columns:
        id_to_str = dict(zip(node_map["int_id"].to_numpy(), node_map["str_id"].astype(str).to_numpy()))
    else:
        id_to_str = {}

    print("\nWalk integrity report")
    print("=" * 80)
    print(f"Path: {wpath}")
    print(f"Shape: {walks.shape} | dtype={walks.dtype}")
    print(f"Tokens total: {total:,} | valid: {n_valid:,} | invalid(-1): {n_invalid:,} ({n_invalid / max(1,total):.2%})")
    print(f"Unique valid node IDs: {len(uniq):,}")
    print(f"Node 0 valid frequency: {node0_count:,} ({node0_count / max(1,n_valid):.2%} of valid tokens)")

    if n_valid > 0:
        counts = np.bincount(valid_vals)
        top = np.argsort(counts)[::-1][:15]
        print("\nTop node frequencies:")
        for nid in top:
            cnt = int(counts[nid])
            if cnt <= 0:
                continue
            sid = id_to_str.get(int(nid), str(int(nid)))
            print(f"  {nid:>6}  {sid:<18}  count={cnt:,}")

    invalid_by_pos = invalid.mean(axis=0)
    print("\nInvalid ratio by walk position:")
    print(f"  min={invalid_by_pos.min():.4f} mean={invalid_by_pos.mean():.4f} max={invalid_by_pos.max():.4f}")
    print(f"  first10={np.array2string(invalid_by_pos[:10], precision=4, separator=', ')}")
    print(f"  last10 ={np.array2string(invalid_by_pos[-10:], precision=4, separator=', ')}")

    if node0_count > 0 and node0_count / max(1, n_valid) > 0.10:
        print("\nWARNING: node 0 is very frequent; check padding handling.")


if __name__ == "__main__":
    main()

"""
Walk integrity checks for node2vec outputs.

Checks:
- shape and dtype
- invalid token count (-1 padding)
- node 0 frequency (to catch padding contamination)
- unique node coverage
- top frequent node IDs
- invalid ratio per walk position

Usage:
  python -m utils.check_walk_integrity
  python -m utils.check_walk_integrity /path/to/walks.npy
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from config_tune import WALKS_PARQUET, NODE_MAP_PARQUET


def load_walks(path: Path):
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
        cols = [c for c in df.columns if c.startswith("step_")]
        return df[cols].to_numpy(dtype=np.int64)
    raise ValueError(f"Unsupported walks path: {path}")


def main():
    if len(sys.argv) > 1:
        wpath = Path(sys.argv[1])
    else:
        npy = WALKS_PARQUET.with_suffix(".npy")
        wpath = npy if npy.exists() else WALKS_PARQUET

    if not wpath.exists():
        raise FileNotFoundError(f"Walk file not found: {wpath}")

    walks = load_walks(wpath).astype(np.int64, copy=False)
    n_walks, walk_len = walks.shape
    invalid = walks < 0
    valid = ~invalid
    total = walks.size
    n_invalid = int(invalid.sum())
    n_valid = int(valid.sum())

    valid_vals = walks[valid]
    uniq = np.unique(valid_vals) if n_valid > 0 else np.array([], dtype=np.int64)
    node0_count = int(np.sum(valid_vals == 0)) if n_valid > 0 else 0

    node_map = pd.read_parquet(NODE_MAP_PARQUET)
    if "int_id" in node_map.columns:
        id_to_str = dict(zip(node_map["int_id"].to_numpy(), node_map["str_id"].astype(str).to_numpy()))
    else:
        id_to_str = {}

    print("\nWalk integrity report")
    print("=" * 80)
    print(f"Path: {wpath}")
    print(f"Shape: {walks.shape} | dtype={walks.dtype}")
    print(f"Tokens total: {total:,} | valid: {n_valid:,} | invalid(-1): {n_invalid:,} ({n_invalid / max(1,total):.2%})")
    print(f"Unique valid node IDs: {len(uniq):,}")
    print(f"Node 0 valid frequency: {node0_count:,} ({node0_count / max(1,n_valid):.2%} of valid tokens)")

    if n_valid > 0:
        counts = np.bincount(valid_vals)
        top = np.argsort(counts)[::-1][:15]
        print("\nTop node frequencies:")
        for nid in top:
            cnt = int(counts[nid])
            if cnt <= 0:
                continue
            sid = id_to_str.get(int(nid), str(int(nid)))
            print(f"  {nid:>6}  {sid:<18}  count={cnt:,}")

    invalid_by_pos = invalid.mean(axis=0)
    print("\nInvalid ratio by walk position:")
    print(f"  min={invalid_by_pos.min():.4f} mean={invalid_by_pos.mean():.4f} max={invalid_by_pos.max():.4f}")
    print(f"  first10={np.array2string(invalid_by_pos[:10], precision=4, separator=', ')}")
    print(f"  last10 ={np.array2string(invalid_by_pos[-10:], precision=4, separator=', ')}")

    if node0_count > 0 and node0_count / max(1, n_valid) > 0.10:
        print("\nWARNING: node 0 is very frequent; check padding handling.")


if __name__ == "__main__":
    main()

