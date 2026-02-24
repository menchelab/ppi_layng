"""
Shared helpers for hybrid adjacency+embedding layout experiments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config_tune import (
    EDGE_LIST_PARQUET,
    EMBEDDINGS_PARQUET,
    LAYOUT_TSV,
    NODE_MAP_PARQUET,
    OUTPUT_DIR,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
)

LAYOUT_MULTI_TSV = OUTPUT_DIR / "layout_multi.tsv"
LAYOUT_DECOMPRESSION_TSV = OUTPUT_DIR / "layout_decompression.tsv"


def choose_layout_path() -> Path:
    if LAYOUT_DECOMPRESSION_TSV.exists():
        return LAYOUT_DECOMPRESSION_TSV
    if LAYOUT_MULTI_TSV.exists():
        return LAYOUT_MULTI_TSV
    return LAYOUT_TSV


def weight_tag(weight: float) -> str:
    return f"w{int(round(float(weight) * 100)):03d}"


def validate_weight(weight: float) -> float:
    w = float(weight)
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"weight must be in [0,1], got {weight}")
    return w


def build_protein_adjacency():
    node_map = pd.read_parquet(NODE_MAP_PARQUET)
    edges = pd.read_parquet(EDGE_LIST_PARQUET)

    protein_df = node_map.loc[node_map["node_type"] == "protein", ["int_id", "str_id"]].copy()
    protein_ids = protein_df["int_id"].to_numpy(dtype=np.int64)
    protein_str_ids = protein_df["str_id"].astype(str).to_numpy()
    n_prot = len(protein_df)

    idx_map = {nid: i for i, nid in enumerate(protein_ids)}
    src = edges["src"].to_numpy(dtype=np.int64)
    dst = edges["dst"].to_numpy(dtype=np.int64)
    mask = np.isin(src, protein_ids) & np.isin(dst, protein_ids)
    src_p = src[mask]
    dst_p = dst[mask]

    rows = np.fromiter((idx_map[s] for s in src_p), dtype=np.int64, count=len(src_p))
    cols = np.fromiter((idx_map[d] for d in dst_p), dtype=np.int64, count=len(dst_p))
    data = np.ones(len(rows), dtype=np.float32)

    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_prot, n_prot)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A, protein_str_ids


def load_protein_embeddings():
    npy_path = EMBEDDINGS_PARQUET.with_suffix(".npy")
    if npy_path.exists():
        emb = np.load(npy_path).astype(np.float32)
    else:
        df = pd.read_parquet(EMBEDDINGS_PARQUET)
        cols = [c for c in df.columns if c.startswith("dim_")]
        emb = df[cols].values.astype(np.float32)

    node_map = pd.read_parquet(NODE_MAP_PARQUET)
    protein_mask = (node_map["node_type"] == "protein").values
    protein_indices = np.where(protein_mask)[0]
    protein_str_ids = node_map.loc[protein_mask, "str_id"].astype(str).to_numpy()
    emb_protein = emb[protein_indices]
    return emb_protein, protein_str_ids


def ensure_same_order(ids_a, ids_b):
    a = np.asarray(ids_a).astype(str)
    b = np.asarray(ids_b).astype(str)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise ValueError("Protein order mismatch between adjacency and embedding views.")


def standardize_dense(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(X, dtype=np.float32)).astype(np.float32)


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n


def adjacency_svd(A: sparse.csr_matrix, svd_dim: int, random_state: int | None) -> np.ndarray:
    n_nodes = A.shape[0]
    d = max(8, min(int(svd_dim), n_nodes - 1))
    return TruncatedSVD(n_components=d, random_state=random_state).fit_transform(A).astype(np.float32)


def row_normalize_sparse(A: sparse.csr_matrix) -> sparse.csr_matrix:
    A = A.tocsr(copy=True)
    rs = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    rs[rs == 0] = 1.0
    Dinv = sparse.diags(1.0 / rs)
    return Dinv @ A


def build_sparse_knn_affinity(X: np.ndarray, k: int, metric: str = "cosine") -> sparse.csr_matrix:
    X = np.asarray(X, dtype=np.float32)
    k = max(2, int(k))
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    n = X.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = ind.reshape(-1)
    d = dist.reshape(-1)

    sigma = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    if sigma <= 0:
        sigma = 1.0
    w = np.exp(-d / sigma).astype(np.float32)
    G = sparse.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = ((G + G.T) * 0.5).tocsr()
    G.setdiag(0.0)
    G.eliminate_zeros()
    return G


def run_umap_dense(X: np.ndarray, use_seed: bool = True) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("Install umap-learn: pip install umap-learn") from e

    rs = UMAP_RANDOM_STATE if use_seed else None
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        n_epochs=300,
        low_memory=True,
        random_state=rs,
        verbose=True,
    )
    return np.asarray(reducer.fit_transform(np.asarray(X, dtype=np.float32)), dtype=np.float64)


def run_umap_precomputed(D: np.ndarray, use_seed: bool = True) -> np.ndarray:
    try:
        import umap
    except ImportError as e:
        raise RuntimeError("Install umap-learn: pip install umap-learn") from e

    rs = UMAP_RANDOM_STATE if use_seed else None
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="precomputed",
        n_epochs=300,
        low_memory=True,
        random_state=rs,
        verbose=True,
    )
    return np.asarray(reducer.fit_transform(np.asarray(D, dtype=np.float32)), dtype=np.float64)


def append_layout_columns(layout_in: Path, layout_out: Path, node_ids, coords, prefix: str):
    xcol, ycol, zcol = f"x_{prefix}", f"y_{prefix}", f"z_{prefix}"
    add_df = pd.DataFrame(
        {
            "node_id": np.asarray(node_ids).astype(str),
            xcol: np.asarray(coords)[:, 0],
            ycol: np.asarray(coords)[:, 1],
            zcol: np.asarray(coords)[:, 2],
        }
    )
    base_df = pd.read_csv(layout_in, sep="\t")
    for c in [xcol, ycol, zcol]:
        if c in base_df.columns:
            base_df = base_df.drop(columns=[c])
    out_df = base_df.merge(add_df, on="node_id", how="left")
    out_df.to_csv(layout_out, sep="\t", index=False)
    return xcol, ycol, zcol

