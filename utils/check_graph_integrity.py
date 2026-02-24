"""
Graph integrity checks for the unified GOA graph.

Checks:
- node counts by type
- edge counts by type (directed and undirected unique)
- proteins with GO annotations
- degree distribution summary + top-degree nodes

Usage:
  python -m utils.check_graph_integrity
  python -m utils.check_graph_integrity /path/to/edge_list.parquet /path/to/node_map.parquet
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from config_tune import EDGE_LIST_PARQUET, NODE_MAP_PARQUET


def _load(edge_path: Path, node_path: Path):
    edges = pd.read_parquet(edge_path)
    nodes = pd.read_parquet(node_path)
    if "int_id" in nodes.columns:
        nodes = nodes.sort_values("int_id")
    else:
        nodes = nodes.reset_index().rename(columns={"index": "int_id"})
    return edges, nodes


def _build_type_arrays(nodes: pd.DataFrame):
    max_id = int(nodes["int_id"].max())
    type_arr = np.full(max_id + 1, -1, dtype=np.int8)  # 0 protein, 1 term
    s = nodes["node_type"].astype(str).str.lower()
    type_arr[nodes.loc[s == "protein", "int_id"].to_numpy(dtype=np.int64)] = 0
    type_arr[nodes.loc[s == "term", "int_id"].to_numpy(dtype=np.int64)] = 1
    id_to_str = dict(zip(nodes["int_id"].to_numpy(), nodes["str_id"].astype(str).to_numpy()))
    return type_arr, id_to_str


def _edge_type_counts(src_t: np.ndarray, dst_t: np.ndarray):
    pp = int(np.sum((src_t == 0) & (dst_t == 0)))
    pt = int(np.sum((src_t == 0) & (dst_t == 1)))
    tp = int(np.sum((src_t == 1) & (dst_t == 0)))
    tt = int(np.sum((src_t == 1) & (dst_t == 1)))
    unknown = int(np.sum((src_t < 0) | (dst_t < 0)))
    return {"pp": pp, "pt": pt, "tp": tp, "tt": tt, "unknown": unknown}


def main():
    edge_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EDGE_LIST_PARQUET
    node_path = Path(sys.argv[2]) if len(sys.argv) > 2 else NODE_MAP_PARQUET
    if not edge_path.exists() or not node_path.exists():
        raise FileNotFoundError(f"Missing inputs: {edge_path} / {node_path}")

    edges, nodes = _load(edge_path, node_path)
    type_arr, id_to_str = _build_type_arrays(nodes)
    src = edges["src"].to_numpy(dtype=np.int64)
    dst = edges["dst"].to_numpy(dtype=np.int64)
    src_t = type_arr[src]
    dst_t = type_arr[dst]

    # Directed edge counts (as stored)
    d = _edge_type_counts(src_t, dst_t)

    # Undirected unique edges
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    base = int(nodes["int_id"].max()) + 1
    keys = lo * base + hi
    uniq_keys = np.unique(keys)
    u = uniq_keys // base
    v = uniq_keys % base
    ut = type_arr[u]
    vt = type_arr[v]
    ud = _edge_type_counts(ut, vt)

    # Annotation coverage (protein-term unique undirected)
    annot_mask = ((ut == 0) & (vt == 1)) | ((ut == 1) & (vt == 0))
    annot_u = u[annot_mask]
    annot_v = v[annot_mask]
    proteins_with_go = np.unique(
        np.concatenate([annot_u[type_arr[annot_u] == 0], annot_v[type_arr[annot_v] == 0]])
    )

    # Degree on undirected unique graph
    deg = np.zeros(len(type_arr), dtype=np.int64)
    np.add.at(deg, u, 1)
    np.add.at(deg, v, 1)
    prot_ids = nodes.loc[nodes["node_type"] == "protein", "int_id"].to_numpy(dtype=np.int64)
    term_ids = nodes.loc[nodes["node_type"] == "term", "int_id"].to_numpy(dtype=np.int64)

    def topk(ids, k=10):
        ids = ids[np.argsort(deg[ids])[::-1][:k]]
        return [(int(i), id_to_str.get(int(i), str(i)), int(deg[i])) for i in ids]

    print("\nGraph integrity report")
    print("=" * 80)
    print(f"Nodes total: {len(nodes):,} | proteins: {len(prot_ids):,} | terms: {len(term_ids):,}")
    print(f"Edges stored (directed pairs): {len(edges):,}")
    print(f"Undirected unique edges: {len(uniq_keys):,}")
    print("\nEdge types (stored/directed):")
    print(f"  protein-protein: {d['pp']:,}")
    print(f"  protein-term   : {d['pt']:,}")
    print(f"  term-protein   : {d['tp']:,}")
    print(f"  term-term      : {d['tt']:,}")
    print(f"  unknown        : {d['unknown']:,}")
    print("\nEdge types (undirected unique):")
    print(f"  protein-protein: {ud['pp']:,}")
    print(f"  protein-term   : {ud['pt'] + ud['tp']:,}")
    print(f"  term-term      : {ud['tt']:,}")

    proteins_total = len(prot_ids)
    proteins_annot = len(proteins_with_go)
    print("\nAnnotation coverage:")
    print(f"  proteins with >=1 GO term: {proteins_annot:,} / {proteins_total:,} ({proteins_annot / max(1, proteins_total):.2%})")
    print(f"  proteins without GO terms: {proteins_total - proteins_annot:,}")

    print("\nDegree summary (undirected unique graph):")
    print(
        f"  proteins mean/median/p95/max: "
        f"{deg[prot_ids].mean():.2f} / {np.median(deg[prot_ids]):.0f} / {np.percentile(deg[prot_ids],95):.0f} / {deg[prot_ids].max():.0f}"
    )
    print(
        f"  terms    mean/median/p95/max: "
        f"{deg[term_ids].mean():.2f} / {np.median(deg[term_ids]):.0f} / {np.percentile(deg[term_ids],95):.0f} / {deg[term_ids].max():.0f}"
    )

    print("\nTop 10 protein degrees:")
    for i, sid, dval in topk(prot_ids, k=10):
        print(f"  {i:>6}  {sid:<18}  deg={dval}")


if __name__ == "__main__":
    main()

