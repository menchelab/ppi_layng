"""
Build UMAP and PaCMAP layouts from edges TSV + node attributes (location column).

Combines graph structure (adjacency SVD) with attribute features (multi-hot locations),
with configurable attribute weight per variant. High emphasis on attributes by default.

Input:
  edges_tsv: source, target
  nodes_tsv: id, <location_column> (e.g. locations_a) - comma-separated strings or null

Output (to --output directory):
  - nodes.tsv: all original node columns + x_locc_01..locc_NN (combined layout columns)
  - edges.tsv: copy of input edges
  - loc_comb_runs.tsv: run parameters

Usage:
  python -m utils.umap_from_edges_and_locations_tsv /path/to/edges.tsv --nodes /path/to/nodes.tsv -o /path/to/output
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create combined graph+attribute 3D layouts from edges and nodes TSV."
    )
    parser.add_argument("edges_tsv", type=str, help="TSV with columns source, target.")
    parser.add_argument(
        "--nodes",
        type=str,
        default=None,
        help="Nodes TSV with id and location column. Default: nodes_orig.tsv in edges dir.",
    )
    parser.add_argument(
        "-c",
        "--column",
        type=str,
        default="locations_a",
        help="Column name for comma-separated locations (default: locations_a).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory. Creates nodes.tsv, edges.tsv, loc_comb_runs.tsv.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Disable random_state for faster non-deterministic runs.",
    )
    return parser.parse_args()


def build_adjacency(edges_tsv: Path) -> tuple[sparse.csr_matrix, np.ndarray]:
    df = pd.read_csv(edges_tsv, sep="\t")
    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError(f"{edges_tsv} must contain columns: source, target")
    src = df["source"].astype(str).to_numpy()
    dst = df["target"].astype(str).to_numpy()
    node_ids = np.unique(np.concatenate([src, dst]))
    node_ids = np.sort(node_ids)
    n = len(node_ids)
    idx = {node_id: i for i, node_id in enumerate(node_ids)}
    rows = np.fromiter((idx[s] for s in src), dtype=np.int64, count=len(src))
    cols = np.fromiter((idx[t] for t in dst), dtype=np.int64, count=len(dst))
    rows_ud = np.concatenate([rows, cols])
    cols_ud = np.concatenate([cols, rows])
    vals = np.ones(len(rows_ud), dtype=np.float32)
    A = sparse.coo_matrix((vals, (rows_ud, cols_ud)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, node_ids


def _load_extra_nodes(edges_tsv: Path, node_ids_from_edges: np.ndarray) -> np.ndarray:
    path = edges_tsv.parent / "nodes_orig.tsv"
    if not path.exists():
        return np.array([], dtype=object)
    df = pd.read_csv(path, sep="\t")
    if "id" not in df.columns:
        return np.array([], dtype=object)
    ids_set = set(node_ids_from_edges.astype(str))
    extra = np.array([x for x in df["id"].astype(str).unique() if x not in ids_set], dtype=object)
    return np.sort(extra)


def _expand_adjacency(
    A: sparse.csr_matrix, node_ids_edges: np.ndarray, node_ids_extra: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    if len(node_ids_extra) == 0:
        return A, node_ids_edges
    n_total = len(node_ids_edges) + len(node_ids_extra)
    node_ids_all = np.concatenate([node_ids_edges, node_ids_extra])
    Ac = A.tocoo()
    A_exp = sparse.coo_matrix(
        (Ac.data, (Ac.row, Ac.col)), shape=(n_total, n_total), dtype=A.dtype
    ).tocsr()
    A_exp.eliminate_zeros()
    return A_exp, node_ids_all


def _load_location_series(nodes_path: Path, node_ids: np.ndarray, col: str) -> pd.Series:
    """Load location column for node_ids order. Missing nodes get null."""
    df = pd.read_csv(nodes_path, sep="\t")
    if "id" not in df.columns or col not in df.columns:
        raise ValueError(f"{nodes_path} must have 'id' and '{col}' columns")
    id_to_val = dict(zip(df["id"].astype(str), df[col]))
    return pd.Series([id_to_val.get(str(nid), None) for nid in node_ids], index=range(len(node_ids)))


def parse_locations(series: pd.Series) -> tuple[list[list[str]], list[str]]:
    node_locations: list[list[str]] = []
    all_locations: set[str] = set()
    for val in series:
        if pd.isna(val) or val is None or str(val).strip() == "":
            node_locations.append([])
            continue
        parts = [x.strip() for x in str(val).split(",") if x.strip()]
        node_locations.append(parts)
        all_locations.update(parts)
    return node_locations, sorted(all_locations)


def encode_multihot(
    node_locations: list[list[str]],
    unique_locations: list[str],
    add_unknown: bool = True,
) -> np.ndarray:
    n = len(node_locations)
    loc_to_idx = {loc: i for i, loc in enumerate(unique_locations)}
    n_loc = len(unique_locations)
    n_dims = n_loc + (1 if add_unknown else 0)
    unknown_idx = n_loc if add_unknown else -1
    X = np.zeros((n, n_dims), dtype=np.float32)
    for i, locs in enumerate(node_locations):
        if not locs:
            if add_unknown:
                X[i, unknown_idx] = 1.0
        else:
            for loc in locs:
                if loc in loc_to_idx:
                    X[i, loc_to_idx[loc]] = 1.0
    return X


def _place_isolated_ring(coords: np.ndarray, n_isolated: int) -> np.ndarray:
    if n_isolated == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(coords) == 0:
        angles = np.linspace(0, 2 * np.pi, n_isolated, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n_isolated)]).astype(np.float64)
    cen = np.mean(coords, axis=0)
    radii = np.linalg.norm(coords - cen, axis=1)
    radius = float(np.percentile(radii[radii > 0] if np.any(radii > 0) else [1.0], 95)) * 1.2
    angles = np.linspace(0, 2 * np.pi, n_isolated, endpoint=False)
    ring = np.column_stack([
        cen[0] + radius * np.cos(angles),
        cen[1] + radius * np.sin(angles),
        np.full(n_isolated, cen[2]),
    ])
    return ring.astype(np.float64)


def _merge_layout_with_ring(
    coords_connected: np.ndarray,
    connected_mask: np.ndarray,
    isolated_mask: np.ndarray,
) -> np.ndarray:
    n_total = len(connected_mask)
    n_isolated = int(isolated_mask.sum())
    full = np.zeros((n_total, 3), dtype=np.float64)
    full[connected_mask] = coords_connected
    if n_isolated > 0:
        full[isolated_mask] = _place_isolated_ring(coords_connected, n_isolated)
    return full


def _add_jitter_if_needed(X: np.ndarray) -> np.ndarray:
    if X.shape[0] < 2:
        return X
    if np.allclose(X, X[0:1], atol=1e-9):
        rng = np.random.default_rng(42)
        X = X + rng.normal(0, 1e-6, X.shape).astype(np.float32)
    return X


def combine_features(
    X_attr: np.ndarray,
    X_graph: np.ndarray,
    attr_weight: float,
) -> np.ndarray:
    """
    Combine attribute and graph features: X = [X_attr * sqrt(w), X_graph * sqrt(1-w)].
    attr_weight in (0, 1); higher = attributes dominate.
    """
    w = float(attr_weight)
    w = max(0.0, min(1.0, w))
    if w >= 1.0:
        return X_attr.astype(np.float32)
    if w <= 0.0:
        return X_graph.astype(np.float32)
    a = np.sqrt(w)
    b = np.sqrt(1.0 - w)
    return np.hstack([X_attr * a, X_graph * b]).astype(np.float32)


# --- Combined layout variants: high attr_weight (0.5-0.99) ---
COMBINED_UMAP_VARIANTS = [
    {"attr_weight": 0.3, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"attr_weight": 0.5, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"attr_weight": 0.7, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"attr_weight": 0.8, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"attr_weight": 0.9, "n_neighbors": 15, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},

    {"attr_weight": 0.2, "n_neighbors": 10, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"attr_weight": 0.4, "n_neighbors": 10, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"attr_weight": 0.6, "n_neighbors": 10, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"attr_weight": 0.8, "n_neighbors": 10, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
    {"attr_weight": 1.0, "n_neighbors": 10, "min_dist": 0.01, "spread": 0.8, "metric": "cosine"},
]

COMBINED_PACMAP_VARIANTS = [
    {"attr_weight": 0.4, "n_neighbors": 10, "mn_ratio": 0.5, "fp_ratio": 2.0, "n_iters": 600},
    {"attr_weight": 0.5, "n_neighbors": 10, "mn_ratio": 0.5, "fp_ratio": 1.0, "n_iters": 600},
    {"attr_weight": 0.7, "n_neighbors": 10, "mn_ratio": 0.5, "fp_ratio": 2.0, "n_iters": 450},
    {"attr_weight": 0.9, "n_neighbors": 10, "mn_ratio": 0.5, "fp_ratio": 1.5, "n_iters": 600},
    {"attr_weight": 0.95, "n_neighbors": 10, "mn_ratio": 0.5, "fp_ratio": 2.0, "n_iters": 800},

    {"attr_weight": 0.4, "n_neighbors": 15, "mn_ratio": 0.5, "fp_ratio": 2.0, "n_iters": 600},
    {"attr_weight": 0.5, "n_neighbors": 15, "mn_ratio": 0.5, "fp_ratio": 1.0, "n_iters": 600},
    {"attr_weight": 0.7, "n_neighbors": 15, "mn_ratio": 0.5, "fp_ratio": 2.0, "n_iters": 450},
    {"attr_weight": 0.9, "n_neighbors": 15, "mn_ratio": 0.5, "fp_ratio": 1.5, "n_iters": 600},
    {"attr_weight": 0.95, "n_neighbors": 15, "mn_ratio": 0.5, "fp_ratio": 2.0, "n_iters": 800},
]


def run_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.01,
    spread: float = 0.8,
    metric: str = "cosine",
    n_epochs: int = 5000,
    random_state: int | None = 42,
):
    import umap
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        spread=float(spread),
        metric=str(metric),
        n_epochs=int(n_epochs),
        init="spectral",
        low_memory=True,
        random_state=random_state,
        verbose=True,
    )
    t0 = time.time()
    coords = reducer.fit_transform(np.asarray(X, dtype=np.float32))
    return np.asarray(coords, dtype=np.float64), time.time() - t0


def run_pacmap(
    X: np.ndarray,
    n_neighbors: int = 10,
    mn_ratio: float = 0.5,
    fp_ratio: float = 2.0,
    n_iters: int = 450,
    random_state: int | None = 42,
):
    import pacmap
    rs = 42 if random_state is None else int(random_state)
    n = X.shape[0]
    nn = min(int(n_neighbors), max(2, n - 1))
    kwargs = {
        "n_components": 3,
        "n_neighbors": nn,
        "MN_ratio": float(mn_ratio),
        "FP_ratio": float(fp_ratio),
        "num_iters": int(n_iters),
        "random_state": rs,
    }
    reducer = pacmap.PaCMAP(**kwargs)
    t0 = time.time()
    coords = reducer.fit_transform(np.asarray(X, dtype=np.float32))
    return np.asarray(coords, dtype=np.float64), time.time() - t0


def main():
    args = parse_args()
    edges_path = Path(args.edges_tsv).expanduser().resolve()
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges not found: {edges_path}")

    nodes_path = Path(args.nodes).expanduser().resolve() if args.nodes else edges_path.parent / "nodes_orig.tsv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes not found: {nodes_path} (use --nodes to specify)")

    col = args.column
    random_state = None if args.no_seed else 42

    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_out = out_dir / "nodes.tsv"
    edges_out = out_dir / "edges.tsv"
    runs_out = out_dir / "loc_comb_runs.tsv"
    print(f"Output directory: {out_dir}")

    # Build graph
    print(f"Reading edges: {edges_path}")
    A, node_ids = build_adjacency(edges_path)
    extra = _load_extra_nodes(edges_path, node_ids)
    if len(extra) > 0:
        print(f"  + {len(extra)} extra nodes from nodes_orig.tsv")
        A, node_ids = _expand_adjacency(A, node_ids, extra)

    n_total = len(node_ids)
    degree = np.array(A.sum(axis=1)).flatten()
    connected_mask = degree > 0
    isolated_mask = ~connected_mask
    n_connected = int(connected_mask.sum())
    n_isolated = int(isolated_mask.sum())
    if n_isolated > 0:
        print(f"Nodes: {n_total} (connected: {n_connected}, isolated: {n_isolated})")
    else:
        print(f"Nodes: {n_total}")

    A_layout = A[connected_mask][:, connected_mask] if n_isolated > 0 else A
    all_isolated = n_connected == 0

    # Load locations
    loc_series = _load_location_series(nodes_path, node_ids, col)
    node_locations, unique_locations = parse_locations(loc_series)
    X_attr_full = encode_multihot(node_locations, unique_locations, add_unknown=True)
    X_attr_full = _add_jitter_if_needed(X_attr_full)
    X_attr = X_attr_full[connected_mask] if n_isolated > 0 else X_attr_full
    print(f"Unique locations: {len(unique_locations)} | attr dim: {X_attr.shape[1]}")

    # Graph SVD (connected only)
    if all_isolated:
        X_graph = np.zeros((0, 8), dtype=np.float32)
        d_svd = 8
        svd_sec = 0.0
    else:
        d_svd = min(
            max(8, min(384, A_layout.shape[0] - 1)),
            A_layout.shape[0],
            A_layout.shape[1],
        )
        t0 = time.time()
        X_graph = TruncatedSVD(n_components=d_svd, random_state=random_state).fit_transform(
            A_layout
        ).astype(np.float32)
        svd_sec = time.time() - t0
        print(f"SVD: dim={d_svd}, {svd_sec:.3f}s")

    def _get_layout_coords(coords_connected: np.ndarray) -> np.ndarray:
        if all_isolated:
            return _place_isolated_ring(np.zeros((0, 3)), n_total)
        return _merge_layout_with_ring(coords_connected, connected_mask, isolated_mask)

    nodes_df = pd.DataFrame({"id": node_ids})
    run_rows = []

    # --- Combined UMAP variants ---
    for i, spec in enumerate(COMBINED_UMAP_VARIANTS, start=1):
        suffix = f"locc_u{i:02d}"
        print(f"\n=== Combined UMAP {suffix} (attr_weight={spec['attr_weight']}) ===")
        if all_isolated:
            coords = _get_layout_coords(None)
            sec = 0.0
        else:
            X_comb = combine_features(X_attr, X_graph, spec["attr_weight"])
            try:
                coords, sec = run_umap(
                    X_comb,
                    n_neighbors=spec["n_neighbors"],
                    min_dist=spec["min_dist"],
                    spread=spec["spread"],
                    metric=spec["metric"],
                    random_state=random_state,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            coords = _get_layout_coords(coords)
        nodes_df[f"x_{suffix}"] = coords[:, 0]
        nodes_df[f"y_{suffix}"] = coords[:, 1]
        nodes_df[f"z_{suffix}"] = coords[:, 2]
        run_rows.append({
            "approach": "combined",
            "algo": "umap",
            "suffix": suffix,
            "attr_weight": spec["attr_weight"],
            "n_neighbors": spec["n_neighbors"],
            "min_dist": spec["min_dist"],
            "spread": spec["spread"],
            "metric": spec["metric"],
            "svd_dim": d_svd,
            "layout_seconds": round(sec, 3),
        })

    # --- Combined PaCMAP variants ---
    for i, spec in enumerate(COMBINED_PACMAP_VARIANTS, start=1):
        suffix = f"locc_p{i:02d}"
        print(f"\n=== Combined PaCMAP {suffix} (attr_weight={spec['attr_weight']}) ===")
        if all_isolated:
            coords = _get_layout_coords(None)
            sec = 0.0
        else:
            X_comb = combine_features(X_attr, X_graph, spec["attr_weight"])
            try:
                coords, sec = run_pacmap(
                    X_comb,
                    n_neighbors=spec["n_neighbors"],
                    mn_ratio=spec["mn_ratio"],
                    fp_ratio=spec["fp_ratio"],
                    n_iters=spec["n_iters"],
                    random_state=random_state,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            coords = _get_layout_coords(coords)
        nodes_df[f"x_{suffix}"] = coords[:, 0]
        nodes_df[f"y_{suffix}"] = coords[:, 1]
        nodes_df[f"z_{suffix}"] = coords[:, 2]
        run_rows.append({
            "approach": "combined",
            "algo": "pacmap",
            "suffix": suffix,
            "attr_weight": spec["attr_weight"],
            "n_neighbors": spec["n_neighbors"],
            "mn_ratio": spec["mn_ratio"],
            "fp_ratio": spec["fp_ratio"],
            "n_iters": spec["n_iters"],
            "svd_dim": d_svd,
            "layout_seconds": round(sec, 3),
        })

    # Merge: original node columns (from nodes file) + layout columns, in node_ids order
    nodes_orig = pd.read_csv(nodes_path, sep="\t")
    out_nodes = nodes_orig.set_index("id").reindex(node_ids.astype(str)).reset_index()
    layout_cols = [c for c in nodes_df.columns if c != "id"]
    for c in layout_cols:
        out_nodes[c] = nodes_df[c].values
    out_nodes.to_csv(nodes_out, sep="\t", index=False)

    shutil.copy2(edges_path, edges_out)

    pd.DataFrame(run_rows).to_csv(runs_out, sep="\t", index=False)
    print(f"\nWrote {nodes_out}")
    print(f"Wrote {edges_out}")
    print(f"Wrote {runs_out}")


if __name__ == "__main__":
    main()
