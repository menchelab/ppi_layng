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

    print(f"Reading edges: {edges_tsv}")
    A, node_ids = build_adjacency(edges_tsv)
    print(f"Nodes: {len(node_ids)} | A shape: {A.shape} | nnz: {A.nnz}")
    if random_state is None:
        print("Running without fixed seed (faster, non-deterministic).")
    else:
        print(f"Running with fixed seed random_state={random_state}.")

    # Highest-quality visual comparison sweep:
    # Keep SVD fixed (same adjacency features for all runs) and vary manifold params.
    fixed_svd_dim = 384
    d = max(8, min(int(fixed_svd_dim), A.shape[0] - 1))
    t_svd0 = time.time()
    X_svd = TruncatedSVD(n_components=d, random_state=random_state).fit_transform(A).astype(np.float32)
    svd_seconds_shared = time.time() - t_svd0
    print(f"Shared SVD done: dim={d}, seconds={svd_seconds_shared:.3f}")

    variants = [
        {"run_id": 1, "svd_dim": fixed_svd_dim, "n_neighbors": 20, "min_dist": 0.01, "metric": "cosine", "n_epochs": 1200, "repulsion_strength": 1.5, "negative_sample_rate": 12},
        {"run_id": 2, "svd_dim": fixed_svd_dim, "n_neighbors": 35, "min_dist": 0.02, "metric": "cosine", "n_epochs": 1500, "repulsion_strength": 1.7, "negative_sample_rate": 15},
        {"run_id": 3, "svd_dim": fixed_svd_dim, "n_neighbors": 50, "min_dist": 0.03, "metric": "cosine", "n_epochs": 1800, "repulsion_strength": 1.9, "negative_sample_rate": 18},
        {"run_id": 4, "svd_dim": fixed_svd_dim, "n_neighbors": 70, "min_dist": 0.05, "metric": "cosine", "n_epochs": 2200, "repulsion_strength": 2.0, "negative_sample_rate": 20},
        {"run_id": 5, "svd_dim": fixed_svd_dim, "n_neighbors": 90, "min_dist": 0.08, "metric": "cosine", "n_epochs": 2600, "repulsion_strength": 2.2, "negative_sample_rate": 24},
        {"run_id": 6, "svd_dim": fixed_svd_dim, "n_neighbors": 120, "min_dist": 0.12, "metric": "cosine", "n_epochs": 3000, "repulsion_strength": 2.4, "negative_sample_rate": 28},
        {"run_id": 7, "svd_dim": fixed_svd_dim, "n_neighbors": 40, "min_dist": 0.02, "metric": "euclidean", "n_epochs": 1800, "repulsion_strength": 1.8, "negative_sample_rate": 16},
        {"run_id": 8, "svd_dim": fixed_svd_dim, "n_neighbors": 80, "min_dist": 0.08, "metric": "euclidean", "n_epochs": 2400, "repulsion_strength": 2.0, "negative_sample_rate": 22},
        {"run_id": 9, "svd_dim": fixed_svd_dim, "n_neighbors": 140, "min_dist": 0.15, "metric": "euclidean", "n_epochs": 3200, "repulsion_strength": 2.4, "negative_sample_rate": 30},
        {"run_id": 10, "svd_dim": fixed_svd_dim, "n_neighbors": 200, "min_dist": 0.20, "metric": "cosine", "n_epochs": 3600, "repulsion_strength": 2.6, "negative_sample_rate": 35},
    ]
    pacmap_variants = [
        {"run_id": 1, "n_neighbors": 20, "mn_ratio": 0.5, "fp_ratio": 1.5, "n_iters": 800},
        {"run_id": 2, "n_neighbors": 40, "mn_ratio": 0.6, "fp_ratio": 2.0, "n_iters": 1000},
        {"run_id": 3, "n_neighbors": 60, "mn_ratio": 0.7, "fp_ratio": 2.0, "n_iters": 1200},
        {"run_id": 4, "n_neighbors": 80, "mn_ratio": 0.8, "fp_ratio": 2.5, "n_iters": 1500},
        {"run_id": 5, "n_neighbors": 120, "mn_ratio": 0.9, "fp_ratio": 3.0, "n_iters": 1800},
    ]
    trimap_variants = [
        {"run_id": 1, "n_inliers": 10, "n_outliers": 5, "n_random": 5, "n_iters": 800},
        {"run_id": 2, "n_inliers": 20, "n_outliers": 10, "n_random": 10, "n_iters": 1200},
        {"run_id": 3, "n_inliers": 30, "n_outliers": 15, "n_random": 10, "n_iters": 1600},
    ]
    # Quality-focused PHATE sweep with explicit diffusion scales.
    phate_variants = [
        {"run_id": 1, "knn": 30, "t": 20, "decay": 60, "gamma": 1.0},
        {"run_id": 2, "knn": 60, "t": 40, "decay": 80, "gamma": 1.0},
        {"run_id": 3, "knn": 100, "t": 80, "decay": 100, "gamma": 0.5},
    ]

    nodes_df = pd.DataFrame({"id": node_ids})
    run_rows = []

    for v in variants:
        run_id = int(v["run_id"])
        print(f"\n=== Run {run_id} ===")
        print(v)
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
        coords, layout_s = run_pacmap_variant(
            X=X_svd,
            n_neighbors=v["n_neighbors"],
            mn_ratio=v["mn_ratio"],
            fp_ratio=v["fp_ratio"],
            n_iters=v["n_iters"],
            random_state=random_state,
        )
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

    for v in trimap_variants:
        run_id = int(v["run_id"])
        print(f"\n=== TriMAP Run {run_id} ===")
        print(v)
        coords, layout_s = run_trimap_variant(
            X=X_svd,
            n_inliers=v["n_inliers"],
            n_outliers=v["n_outliers"],
            n_random=v["n_random"],
            n_iters=v["n_iters"],
            random_state=random_state,
        )
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

    for v in phate_variants:
        run_id = int(v["run_id"])
        print(f"\n=== PHATE Run {run_id} ===")
        print(v)
        coords, layout_s, best_seed, best_tw, unique_n = run_phate_variant(
            X=X_svd,
            knn=v["knn"],
            t=v["t"],
            decay=v["decay"],
            gamma=v["gamma"],
            random_state=random_state,
            seed_candidates=[42, 52, 62],
            jitter=1e-6,
        )
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
                "seed_candidates": "42,52,62",
                "best_seed": int(best_seed),
                "best_trustworthiness": float(best_tw),
                "n_unique_input": int(unique_n),
                "svd_seconds": round(float(svd_seconds_shared), 3),
                "layout_seconds": round(float(layout_s), 3),
            }
        )

    nodes_df.to_csv(nodes_out, sep="\t", index=False)
    pd.DataFrame(run_rows).sort_values("run_id").to_csv(runs_out, sep="\t", index=False)

    print(f"\nWrote {nodes_out}")
    print(f"Wrote {runs_out}")


if __name__ == "__main__":
    main()

