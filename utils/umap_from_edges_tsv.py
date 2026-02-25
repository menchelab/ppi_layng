"""
Build multiple UMAP layouts directly from an edges TSV.

Input:
  argv[1] edges TSV path with columns: source, target

Output (same directory as input edges TSV):
  - nodes.tsv:
      id + x_adju<n>, y_adju<n>, z_adju<n> (UMAP runs)
         + x_adjp<n>, y_adjp<n>, z_adjp<n> (PaCMAP runs)
         + x_adjt<n>, y_adjt<n>, z_adjt<n> (TriMAP runs)
         + x_adjh<n>, y_adjh<n>, z_adjh<n> (PHATE runs)
         + x_adjf<n>, y_adjf<n>, z_adjf<n> (ForceAtlas2 runs)
         + x_adjs<n>, y_adjs<n>, z_adjs<n> (spring runs)
  - umap_runs.tsv: one row per run with algo + parameter settings

Usage:
  python -m utils.umap_from_edges_tsv /path/to/edges.tsv
  python -m utils.umap_from_edges_tsv /path/to/edges.tsv --no-seed
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import trustworthiness


def parse_args():
    parser = argparse.ArgumentParser(description="Create 10 adjacency UMAP layouts from edges TSV.")
    parser.add_argument("edges_tsv", type=str, help="TSV with columns source,target.")
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Disable random_state for faster non-deterministic runs.",
    )
    parser.add_argument(
        "--no-phate",
        action="store_true",
        help="Skip PHATE (adjh) layout variants.",
    )
    parser.add_argument(
        "--no-trimap",
        action="store_true",
        help="Skip TriMAP (adjt) layout variants.",
    )
    parser.add_argument(
        "--no-forceatlas2",
        action="store_true",
        help="Skip ForceAtlas2 (adjf) layout variants.",
    )
    parser.add_argument(
        "--no-spring",
        action="store_true",
        help="Skip spring (adjs) layout variants.",
    )
    return parser.parse_args()


def build_adjacency(edges_tsv: Path):
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

    # Undirected graph by adding reverse edges
    rows_ud = np.concatenate([rows, cols])
    cols_ud = np.concatenate([cols, rows])
    vals = np.ones(len(rows_ud), dtype=np.float32)
    A = sparse.coo_matrix((vals, (rows_ud, cols_ud)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, node_ids


def _load_extra_nodes_from_nodes_orig(edges_tsv: Path, node_ids_from_edges: np.ndarray) -> np.ndarray:
    """Load nodes from nodes_orig.tsv in same dir; return extra ids not in edges. Empty if no file."""
    nodes_orig_path = edges_tsv.parent / "nodes_orig.tsv"
    if not nodes_orig_path.exists():
        return np.array([], dtype=node_ids_from_edges.dtype)
    df = pd.read_csv(nodes_orig_path, sep="\t")
    if "id" not in df.columns:
        return np.array([], dtype=node_ids_from_edges.dtype)
    ids_orig = df["id"].astype(str).to_numpy()
    ids_set = set(node_ids_from_edges.astype(str))
    extra = np.array([x for x in np.unique(ids_orig) if x not in ids_set], dtype=object)
    return extra


def _expand_adjacency_for_extra_nodes(
    A: sparse.csr_matrix, node_ids_edges: np.ndarray, node_ids_extra: np.ndarray
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Expand A to include extra nodes (no edges). Returns (A_expanded, node_ids_all)."""
    if len(node_ids_extra) == 0:
        return A, node_ids_edges
    n_edges = len(node_ids_edges)
    n_total = n_edges + len(node_ids_extra)
    node_ids_all = np.concatenate([node_ids_edges, np.sort(node_ids_extra)])
    # Expand sparse matrix: keep original entries, new rows/cols stay zero
    Ac = A.tocoo()
    A_exp = sparse.coo_matrix(
        (Ac.data, (Ac.row, Ac.col)), shape=(n_total, n_total), dtype=A.dtype
    ).tocsr()
    A_exp.eliminate_zeros()
    return A_exp, node_ids_all


def _place_isolated_ring(coords_connected: np.ndarray, n_isolated: int) -> np.ndarray:
    """Place n_isolated points on a ring around the centroid of coords_connected."""
    if n_isolated == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if len(coords_connected) == 0:
        angles = np.linspace(0, 2 * np.pi, n_isolated, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n_isolated)]).astype(np.float64)
    cen = np.mean(coords_connected, axis=0)
    radii = np.linalg.norm(coords_connected - cen, axis=1)
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
    """Merge layout coords (connected only) with ring positions for isolated nodes."""
    n_total = len(connected_mask)
    n_isolated = int(isolated_mask.sum())
    full = np.zeros((n_total, 3), dtype=np.float64)
    full[connected_mask] = coords_connected
    if n_isolated > 0:
        full[isolated_mask] = _place_isolated_ring(coords_connected, n_isolated)
    return full


def _merge_force_layout_concentric(
    coords_connected: np.ndarray,
    A_layout: sparse.csr_matrix,
    connected_mask: np.ndarray,
    isolated_mask: np.ndarray,
    inner_radius_frac: float = 0.85,
    outer_radius_frac: float = 1.3,
) -> np.ndarray:
    """
    Merge force-directed layout with concentric rings: main component center,
    isolated nodes on inner ring, small connected components on outer ring.
    """
    n_connected = coords_connected.shape[0]
    n_total = len(connected_mask)
    n_isolated = int(isolated_mask.sum())
    full = np.zeros((n_total, 3), dtype=np.float64)

    if n_connected == 0:
        if n_isolated > 0:
            full[isolated_mask] = _place_isolated_ring(np.zeros((0, 3)), n_isolated)
        return full

    # Find connected components (in A_layout index space: 0..n_connected-1)
    n_comp, labels = csgraph.connected_components(A_layout, directed=False)
    comp_ids = [np.where(labels == c)[0] for c in range(n_comp)]
    comp_sizes = [len(ids) for ids in comp_ids]
    main_idx = int(np.argmax(comp_sizes))
    main_ids = comp_ids[main_idx]
    small_comp_ids = [ids for i, ids in enumerate(comp_ids) if i != main_idx and len(ids) > 0]

    # Base radius from main component only
    main_coords = coords_connected[main_ids]
    cen = np.mean(main_coords, axis=0)
    radii = np.linalg.norm(main_coords - cen, axis=1)
    base_radius = float(np.percentile(radii[radii > 0] if np.any(radii > 0) else [1.0], 95))
    if base_radius < 1e-6:
        base_radius = 1.0
    r_inner = base_radius * inner_radius_frac
    r_outer = base_radius * outer_radius_frac

    # Main component: keep layout, optionally recenter
    full[connected_mask] = coords_connected

    # Small components: place each centroid on outer ring, translate
    if small_comp_ids:
        n_small = len(small_comp_ids)
        angles = np.linspace(0, 2 * np.pi, n_small, endpoint=False)
        for k, ids in enumerate(small_comp_ids):
            comp_coords = coords_connected[ids]
            comp_cen = np.mean(comp_coords, axis=0)
            target = cen + np.array([
                r_outer * np.cos(angles[k]),
                r_outer * np.sin(angles[k]),
                0.0,
            ])
            # Translate component so centroid -> target
            shift = target - comp_cen
            full[np.flatnonzero(connected_mask)[ids]] = comp_coords + shift

    # Isolated: inner ring (same plane as main)
    if n_isolated > 0:
        angles = np.linspace(0, 2 * np.pi, n_isolated, endpoint=False)
        inner_ring = np.column_stack([
            cen[0] + r_inner * np.cos(angles),
            cen[1] + r_inner * np.sin(angles),
            np.full(n_isolated, cen[2]),
        ]).astype(np.float64)
        full[isolated_mask] = inner_ring

    return full


def run_umap_variant(
    X: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    n_epochs: int,
    repulsion_strength: float,
    negative_sample_rate: int,
    random_state: int | None,
):
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("Please install umap-learn: pip install umap-learn") from e

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        n_epochs=int(n_epochs),
        repulsion_strength=float(repulsion_strength),
        negative_sample_rate=int(negative_sample_rate),
        init="spectral",
        low_memory=True,
        random_state=random_state,
        verbose=True,
    )
    t1 = time.time()
    coords = reducer.fit_transform(np.asarray(X, dtype=np.float32))
    layout_seconds = time.time() - t1
    return np.asarray(coords, dtype=np.float64), layout_seconds


def run_pacmap_variant(
    X: np.ndarray,
    n_neighbors: int,
    mn_ratio: float,
    fp_ratio: float,
    n_iters: int,
    random_state: int | None,
):
    try:
        import pacmap
    except ImportError as e:
        raise RuntimeError("Please install pacmap: pip install pacmap") from e

    rs = 42 if random_state is None else int(random_state)
    reducer = pacmap.PaCMAP(
        n_components=3,
        n_neighbors=int(n_neighbors),
        MN_ratio=float(mn_ratio),
        FP_ratio=float(fp_ratio),
        num_iters=int(n_iters),
        random_state=rs,
    )
    t0 = time.time()
    coords = reducer.fit_transform(np.asarray(X, dtype=np.float32))
    layout_seconds = time.time() - t0
    return np.asarray(coords, dtype=np.float64), layout_seconds


def run_trimap_variant(
    X: np.ndarray,
    n_inliers: int,
    n_outliers: int,
    n_random: int,
    n_iters: int,
    random_state: int | None,
):
    try:
        import trimap
    except ImportError as e:
        raise RuntimeError("Please install trimap: pip install trimap") from e

    if random_state is not None:
        np.random.seed(int(random_state))
    reducer = trimap.TRIMAP(
        n_dims=3,
        n_inliers=int(n_inliers),
        n_outliers=int(n_outliers),
        n_random=int(n_random),
        n_iters=int(n_iters),
        verbose=False,
    )
    t0 = time.time()
    coords = reducer.fit_transform(np.asarray(X, dtype=np.float32))
    layout_seconds = time.time() - t0
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"TriMAP returned shape {coords.shape}, expected (N, 3). "
            "Check trimap version compatibility."
        )
    return coords, layout_seconds


def run_phate_variant(
    X: np.ndarray,
    knn: int,
    t: int | str,
    decay: int,
    gamma: float,
    random_state: int | None,
    seed_candidates: list[int] | None = None,
    jitter: float = 1e-6,
):
    try:
        import phate
    except ImportError as e:
        raise RuntimeError("Please install phate: pip install phate") from e

    X = np.asarray(X, dtype=np.float32)
    # Remove exact duplicate rows to avoid zero-distance graph warnings.
    X_unique, inv = np.unique(X, axis=0, return_inverse=True)
    if jitter > 0:
        # Tiny jitter to break tie distances while preserving manifold structure.
        rng_j = np.random.default_rng(1234)
        X_unique = X_unique + (rng_j.normal(size=X_unique.shape).astype(np.float32) * float(jitter))

    if seed_candidates is None:
        if random_state is None:
            seed_candidates = [42, 52, 62]
        else:
            base = int(random_state)
            seed_candidates = [base, base + 10, base + 20]

    best = None
    total_seconds = 0.0
    for seed in seed_candidates:
        kwargs = dict(
            n_components=3,
            knn=int(knn),
            t=t,
            decay=int(decay),
            gamma=float(gamma),
            random_state=int(seed),
            mds="metric",
            mds_solver="smacof",
            verbose=False,
        )
        try:
            reducer = phate.PHATE(**kwargs)
        except TypeError:
            # Older PHATE versions may not expose mds/mds_solver.
            kwargs.pop("mds", None)
            kwargs.pop("mds_solver", None)
            reducer = phate.PHATE(**kwargs)

        t0 = time.time()
        coords_u = reducer.fit_transform(X_unique)
        run_seconds = time.time() - t0
        total_seconds += run_seconds
        coords_u = np.asarray(coords_u, dtype=np.float64)
        coords = coords_u[inv]

        # Higher trustworthiness is better local-neighborhood preservation.
        tw = float(trustworthiness(X, coords, n_neighbors=min(15, max(2, len(X) - 1))))
        cand = {
            "coords": coords,
            "seconds": run_seconds,
            "seed": int(seed),
            "trustworthiness": tw,
        }
        if best is None or cand["trustworthiness"] > best["trustworthiness"]:
            best = cand

    assert best is not None
    return best["coords"], total_seconds, best["seed"], best["trustworthiness"], len(X_unique)


def _build_nx_graph_from_adjacency(A: sparse.csr_matrix) -> "nx.Graph":
    """Build undirected NetworkX graph from sparse adjacency matrix."""
    import networkx as nx

    G = nx.Graph()
    n = A.shape[0]
    G.add_nodes_from(range(n))
    rows, cols = A.nonzero()
    for i, j in zip(rows, cols):
        if i < j:  # undirected: add each edge once
            G.add_edge(i, j, weight=float(A[i, j]))
    return G


def run_forceatlas2_variant(
    A: sparse.csr_matrix,
    max_iter: int,
    gravity: float,
    scaling_ratio: float,
    random_state: int | None,
) -> tuple[np.ndarray, float]:
    """Run ForceAtlas2 layout on adjacency matrix. Returns (coords, seconds)."""
    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError("Please install networkx: pip install networkx") from e

    G = _build_nx_graph_from_adjacency(A)
    if G.number_of_edges() == 0:
        # Disconnected/empty: return random init
        rng = np.random.default_rng(42 if random_state is None else random_state)
        n = A.shape[0]
        coords = rng.standard_normal((n, 3)).astype(np.float64)
        return coords, 0.0

    rs = None if random_state is None else int(random_state)
    t0 = time.time()
    pos = nx.forceatlas2_layout(G, dim=3, max_iter=max_iter, gravity=gravity, scaling_ratio=scaling_ratio, seed=rs)
    layout_s = time.time() - t0
    n = G.number_of_nodes()
    coords = np.array([[pos[i][0], pos[i][1], pos[i][2]] for i in range(n)], dtype=np.float64)
    return coords, layout_s


def run_spring_variant(
    A: sparse.csr_matrix,
    iterations: int,
    k: float | None,
    random_state: int | None,
) -> tuple[np.ndarray, float]:
    """Run Fruchterman-Reingold spring layout on adjacency matrix. Returns (coords, seconds)."""
    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError("Please install networkx: pip install networkx") from e

    G = _build_nx_graph_from_adjacency(A)
    if G.number_of_edges() == 0:
        rng = np.random.default_rng(42 if random_state is None else random_state)
        n = A.shape[0]
        coords = rng.standard_normal((n, 3)).astype(np.float64)
        return coords, 0.0

    rs = None if random_state is None else int(random_state)
    t0 = time.time()
    pos = nx.spring_layout(G, dim=3, iterations=iterations, k=k, seed=rs)
    layout_s = time.time() - t0
    n = G.number_of_nodes()
    coords = np.array([[pos[i][0], pos[i][1], pos[i][2]] for i in range(n)], dtype=np.float64)
    return coords, layout_s


def main():
    args = parse_args()
    edges_tsv = Path(args.edges_tsv).expanduser().resolve()
    if not edges_tsv.exists():
        raise FileNotFoundError(f"Input not found: {edges_tsv}")

    out_dir = edges_tsv.parent
    nodes_out = out_dir / "nodes.tsv"
    runs_out = out_dir / "umap_runs.tsv"

    use_seed = not args.no_seed
    random_state = 42 if use_seed else None

    BOLD_RED = "\033[1;31m"
    RESET = "\033[0m"

    print(f"Reading edges: {edges_tsv}")
    A, node_ids = build_adjacency(edges_tsv)
    extra_ids = _load_extra_nodes_from_nodes_orig(edges_tsv, node_ids)
    if len(extra_ids) > 0:
        print(f"{BOLD_RED}nodes_orig.tsv: {len(extra_ids)} extra nodes (no edges) added{RESET}")
        A, node_ids = _expand_adjacency_for_extra_nodes(A, node_ids, extra_ids)
    print(f"Nodes: {len(node_ids)} | A shape: {A.shape} | nnz: {A.nnz}")
    if random_state is None:
        print("Running without fixed seed (faster, non-deterministic).")
    else:
        print(f"Running with fixed seed random_state={random_state}.")

    # Split connected vs isolated (degree 0). Run layout only on connected; place isolated on ring.
    degree = np.array(A.sum(axis=1)).flatten()
    connected_mask = degree > 0
    isolated_mask = ~connected_mask
    n_connected = int(connected_mask.sum())
    n_isolated = int(isolated_mask.sum())
    if n_isolated > 0:
        print(f"Layout on {n_connected} connected nodes; {n_isolated} isolated → ring around layout")
        A_layout = A[connected_mask][:, connected_mask]
    else:
        A_layout = A

    fixed_svd_dim = 384
    all_isolated = n_connected == 0
    if all_isolated:
        X_svd = np.zeros((0, 8), dtype=np.float32)
        svd_seconds_shared = 0.0
        d = 8
        print("All nodes isolated; placing on ring (no layout).")
    else:
        d = min(
            max(8, min(int(fixed_svd_dim), A_layout.shape[0] - 1)),
            A_layout.shape[0],
            A_layout.shape[1],
        )
        t_svd0 = time.time()
        X_svd = TruncatedSVD(n_components=d, random_state=random_state).fit_transform(A_layout).astype(np.float32)
        svd_seconds_shared = time.time() - t_svd0
        print(f"Shared SVD done: dim={d}, seconds={svd_seconds_shared:.3f}")

    # Longer runs for quality: n_epochs roughly 2x previous defaults.
    variants = [
        {"run_id": 1, "svd_dim": fixed_svd_dim, "n_neighbors": 20, "min_dist": 0.01, "metric": "cosine", "n_epochs": 2400, "repulsion_strength": 1.5, "negative_sample_rate": 12},
        {"run_id": 2, "svd_dim": fixed_svd_dim, "n_neighbors": 35, "min_dist": 0.02, "metric": "cosine", "n_epochs": 3000, "repulsion_strength": 1.7, "negative_sample_rate": 15},
        {"run_id": 3, "svd_dim": fixed_svd_dim, "n_neighbors": 50, "min_dist": 0.03, "metric": "cosine", "n_epochs": 3600, "repulsion_strength": 1.9, "negative_sample_rate": 18},
        {"run_id": 4, "svd_dim": fixed_svd_dim, "n_neighbors": 70, "min_dist": 0.05, "metric": "cosine", "n_epochs": 4400, "repulsion_strength": 2.0, "negative_sample_rate": 20},
        {"run_id": 5, "svd_dim": fixed_svd_dim, "n_neighbors": 90, "min_dist": 0.08, "metric": "cosine", "n_epochs": 5200, "repulsion_strength": 2.2, "negative_sample_rate": 24},
        {"run_id": 6, "svd_dim": fixed_svd_dim, "n_neighbors": 120, "min_dist": 0.12, "metric": "cosine", "n_epochs": 6000, "repulsion_strength": 2.4, "negative_sample_rate": 28},
        {"run_id": 7, "svd_dim": fixed_svd_dim, "n_neighbors": 40, "min_dist": 0.02, "metric": "euclidean", "n_epochs": 3600, "repulsion_strength": 1.8, "negative_sample_rate": 16},
        {"run_id": 8, "svd_dim": fixed_svd_dim, "n_neighbors": 80, "min_dist": 0.08, "metric": "euclidean", "n_epochs": 4800, "repulsion_strength": 2.0, "negative_sample_rate": 22},
        {"run_id": 9, "svd_dim": fixed_svd_dim, "n_neighbors": 140, "min_dist": 0.15, "metric": "euclidean", "n_epochs": 6400, "repulsion_strength": 2.4, "negative_sample_rate": 30},
        {"run_id": 10, "svd_dim": fixed_svd_dim, "n_neighbors": 200, "min_dist": 0.20, "metric": "cosine", "n_epochs": 7200, "repulsion_strength": 2.6, "negative_sample_rate": 35},
    ]
    pacmap_variants = [
        {"run_id": 1, "n_neighbors": 20, "mn_ratio": 0.5, "fp_ratio": 1.5, "n_iters": 1600},
        {"run_id": 2, "n_neighbors": 40, "mn_ratio": 0.6, "fp_ratio": 2.0, "n_iters": 2000},
        {"run_id": 3, "n_neighbors": 60, "mn_ratio": 0.7, "fp_ratio": 2.0, "n_iters": 2400},
        {"run_id": 4, "n_neighbors": 80, "mn_ratio": 0.8, "fp_ratio": 2.5, "n_iters": 3000},
        {"run_id": 5, "n_neighbors": 120, "mn_ratio": 0.9, "fp_ratio": 3.0, "n_iters": 3600},
    ]
    trimap_variants = [
        {"run_id": 1, "n_inliers": 10, "n_outliers": 5, "n_random": 5, "n_iters": 1600},
        {"run_id": 2, "n_inliers": 20, "n_outliers": 10, "n_random": 10, "n_iters": 2400},
        {"run_id": 3, "n_inliers": 30, "n_outliers": 15, "n_random": 10, "n_iters": 3200},
    ]
    # Quality-focused PHATE sweep. Low t causes line collapse; higher t + gamma=0.5 gives 3D spread.
    # adjh1/adjh2: raised t and gamma=0.5 to avoid "everything on one line".
    # adjh3: raised t to spread hotspots; postprocess_apply_layouts can further refine.
    phate_variants = [
        {"run_id": 1, "knn": 30, "t": 50, "decay": 60, "gamma": 0.5},
        {"run_id": 2, "knn": 60, "t": 65, "decay": 80, "gamma": 0.5},
        {"run_id": 3, "knn": 100, "t": 100, "decay": 100, "gamma": 0.5},
    ]
    forceatlas2_variants = [
        {"run_id": 1, "max_iter": 200, "gravity": 0.5, "scaling_ratio": 2.0},
        {"run_id": 2, "max_iter": 500, "gravity": 1.0, "scaling_ratio": 2.0},
        {"run_id": 3, "max_iter": 1000, "gravity": 0.5, "scaling_ratio": 2.5},
        {"run_id": 4, "max_iter": 1500, "gravity": 1.0, "scaling_ratio": 2.5},
    ]
    spring_variants = [
        {"run_id": 1, "iterations": 200, "k": None},
        {"run_id": 2, "iterations": 500, "k": None},
    ]

    nodes_df = pd.DataFrame({"id": node_ids})
    run_rows = []

    def _get_layout_coords(coords_connected: np.ndarray) -> np.ndarray:
        """Merge layout with ring for isolated nodes; handle all-isolated case."""
        if all_isolated:
            return _place_isolated_ring(np.zeros((0, 3)), len(node_ids))
        return _merge_layout_with_ring(coords_connected, connected_mask, isolated_mask)

    for v in variants:
        run_id = int(v["run_id"])
        print(f"\n=== Run {run_id} ===")
        print(v)
        if all_isolated:
            coords = _get_layout_coords(None)
            layout_s = 0.0
        else:
            coords, layout_s = run_umap_variant(
                X=X_svd,
                n_neighbors=v["n_neighbors"],
                min_dist=v["min_dist"],
                metric=v["metric"],
                n_epochs=v["n_epochs"],
                repulsion_strength=v["repulsion_strength"],
                negative_sample_rate=v["negative_sample_rate"],
                random_state=random_state,
            )
            coords = _get_layout_coords(coords)

        nodes_df[f"x_adju{run_id}"] = coords[:, 0]
        nodes_df[f"y_adju{run_id}"] = coords[:, 1]
        nodes_df[f"z_adju{run_id}"] = coords[:, 2]

        run_rows.append(
            {
                "algo": "umap",
                "run_id": run_id,
                "svd_dim": d,
                "n_neighbors": int(v["n_neighbors"]),
                "min_dist": float(v["min_dist"]),
                "metric": str(v["metric"]),
                "n_epochs": int(v["n_epochs"]),
                "repulsion_strength": float(v["repulsion_strength"]),
                "negative_sample_rate": int(v["negative_sample_rate"]),
                "random_state": "" if random_state is None else int(random_state),
                "svd_seconds": round(float(svd_seconds_shared), 3),
                "layout_seconds": round(float(layout_s), 3),
            }
        )

    for v in pacmap_variants:
        run_id = int(v["run_id"])
        print(f"\n=== PaCMAP Run {run_id} ===")
        print(v)
        if all_isolated:
            coords = _get_layout_coords(None)
            layout_s = 0.0
        else:
            coords, layout_s = run_pacmap_variant(
                X=X_svd,
                n_neighbors=v["n_neighbors"],
                mn_ratio=v["mn_ratio"],
                fp_ratio=v["fp_ratio"],
                n_iters=v["n_iters"],
                random_state=random_state,
            )
            coords = _get_layout_coords(coords)
        nodes_df[f"x_adjp{run_id}"] = coords[:, 0]
        nodes_df[f"y_adjp{run_id}"] = coords[:, 1]
        nodes_df[f"z_adjp{run_id}"] = coords[:, 2]
        run_rows.append(
            {
                "algo": "pacmap",
                "run_id": run_id,
                "svd_dim": d,
                "n_neighbors": int(v["n_neighbors"]),
                "mn_ratio": float(v["mn_ratio"]),
                "fp_ratio": float(v["fp_ratio"]),
                "n_iters": int(v["n_iters"]),
                "random_state": "" if random_state is None else int(random_state),
                "svd_seconds": round(float(svd_seconds_shared), 3),
                "layout_seconds": round(float(layout_s), 3),
            }
        )

    if not args.no_trimap:
        for v in trimap_variants:
            run_id = int(v["run_id"])
            print(f"\n=== TriMAP Run {run_id} ===")
            print(v)
            if all_isolated:
                coords = _get_layout_coords(None)
                layout_s = 0.0
            else:
                coords, layout_s = run_trimap_variant(
                    X=X_svd,
                    n_inliers=v["n_inliers"],
                    n_outliers=v["n_outliers"],
                    n_random=v["n_random"],
                    n_iters=v["n_iters"],
                    random_state=random_state,
                )
                coords = _get_layout_coords(coords)
            nodes_df[f"x_adjt{run_id}"] = coords[:, 0]
            nodes_df[f"y_adjt{run_id}"] = coords[:, 1]
            nodes_df[f"z_adjt{run_id}"] = coords[:, 2]
            run_rows.append(
                {
                    "algo": "trimap",
                    "run_id": run_id,
                    "svd_dim": d,
                    "n_inliers": int(v["n_inliers"]),
                    "n_outliers": int(v["n_outliers"]),
                    "n_random": int(v["n_random"]),
                    "n_iters": int(v["n_iters"]),
                    "random_state": "" if random_state is None else int(random_state),
                    "svd_seconds": round(float(svd_seconds_shared), 3),
                    "layout_seconds": round(float(layout_s), 3),
                }
            )

    if not args.no_phate:
        for v in phate_variants:
            run_id = int(v["run_id"])
            print(f"\n=== PHATE Run {run_id} ===")
            print(v)
            if all_isolated:
                coords = _get_layout_coords(None)
                layout_s = 0.0
                best_seed = 42
                best_tw = 0.0
                unique_n = 0
            else:
                coords, layout_s, best_seed, best_tw, unique_n = run_phate_variant(
                    X=X_svd,
                    knn=v["knn"],
                    t=v["t"],
                    decay=v["decay"],
                    gamma=v["gamma"],
                    random_state=random_state,
                    seed_candidates=[42, 52, 62, 72, 82],
                    jitter=1e-6,
                )
                coords = _get_layout_coords(coords)
            nodes_df[f"x_adjh{run_id}"] = coords[:, 0]
            nodes_df[f"y_adjh{run_id}"] = coords[:, 1]
            nodes_df[f"z_adjh{run_id}"] = coords[:, 2]
            run_rows.append(
                {
                    "algo": "phate",
                    "run_id": run_id,
                    "svd_dim": d,
                    "knn": int(v["knn"]),
                    "t": str(v["t"]),
                    "decay": int(v["decay"]),
                    "gamma": float(v["gamma"]),
                    "random_state": "" if random_state is None else int(random_state),
                    "seed_candidates": "42,52,62,72,82",
                    "best_seed": int(best_seed),
                    "best_trustworthiness": float(best_tw),
                    "n_unique_input": int(unique_n),
                    "svd_seconds": round(float(svd_seconds_shared), 3),
                    "layout_seconds": round(float(layout_s), 3),
                }
            )

    if not args.no_forceatlas2:
        for v in forceatlas2_variants:
            run_id = int(v["run_id"])
            print(f"\n=== ForceAtlas2 Run {run_id} ===")
            print(v)
            if all_isolated:
                coords = _get_layout_coords(None)
                layout_s = 0.0
            else:
                coords, layout_s = run_forceatlas2_variant(
                    A=A_layout,
                    max_iter=v["max_iter"],
                    gravity=v["gravity"],
                    scaling_ratio=v["scaling_ratio"],
                    random_state=random_state,
                )
                coords = _merge_force_layout_concentric(
                    coords, A_layout, connected_mask, isolated_mask
                )
            nodes_df[f"x_adjf{run_id}"] = coords[:, 0]
            nodes_df[f"y_adjf{run_id}"] = coords[:, 1]
            nodes_df[f"z_adjf{run_id}"] = coords[:, 2]
            run_rows.append(
                {
                    "algo": "forceatlas2",
                    "run_id": run_id,
                    "max_iter": int(v["max_iter"]),
                    "gravity": float(v["gravity"]),
                    "scaling_ratio": float(v["scaling_ratio"]),
                    "random_state": "" if random_state is None else int(random_state),
                    "layout_seconds": round(float(layout_s), 3),
                }
            )

    if not args.no_spring:
        for v in spring_variants:
            run_id = int(v["run_id"])
            print(f"\n=== Spring Run {run_id} ===")
            print(v)
            if all_isolated:
                coords = _get_layout_coords(None)
                layout_s = 0.0
            else:
                coords, layout_s = run_spring_variant(
                    A=A_layout,
                    iterations=v["iterations"],
                    k=v.get("k"),
                    random_state=random_state,
                )
                coords = _merge_force_layout_concentric(
                    coords, A_layout, connected_mask, isolated_mask
                )
            nodes_df[f"x_adjs{run_id}"] = coords[:, 0]
            nodes_df[f"y_adjs{run_id}"] = coords[:, 1]
            nodes_df[f"z_adjs{run_id}"] = coords[:, 2]
            run_rows.append(
                {
                    "algo": "spring",
                    "run_id": run_id,
                    "iterations": int(v["iterations"]),
                    "k": v.get("k"),
                    "random_state": "" if random_state is None else int(random_state),
                    "layout_seconds": round(float(layout_s), 3),
                }
            )

    nodes_df.to_csv(nodes_out, sep="\t", index=False)
    pd.DataFrame(run_rows).sort_values(["algo", "run_id"]).to_csv(runs_out, sep="\t", index=False)

    print(f"\nWrote {nodes_out}")
    print(f"Wrote {runs_out}")


if __name__ == "__main__":
    main()

