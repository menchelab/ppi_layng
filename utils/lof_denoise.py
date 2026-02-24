"""
Local Outlier Factor (LOF) denoising for high-dimensional embeddings.

"Moonlight" proteins (e.g. with many GO terms) can blur cluster boundaries.
LOF identifies the most ambiguous nodes (high reachability / outlier score).
Options: (1) exclude them during projection then place back; (2) down-weight
their influence in the distance matrix (here we provide the mask/weights).
"""
import numpy as np


def lof_scores(X: np.ndarray, n_neighbors: int = 20) -> np.ndarray:
    """
    Compute LOF (negative) score per sample. Higher score = more "outlier-like"
    (ambiguous). sklearn's score_samples is opposite: more negative = more outlier;
    we return negative so that high = ambiguous.

    Parameters
    ----------
    X : (N, D) array
        High-dimensional embeddings.
    n_neighbors : int
        LOF k-neighborhood size.

    Returns
    -------
    (N,) array
        Per-node score; higher = more ambiguous (candidate for exclusion).
    """
    from sklearn.neighbors import LocalOutlierFactor

    X = np.asarray(X, dtype=np.float64)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    lof.fit(X)
    # decision_function: negative = outlier; we want "ambiguity" so use -decision_function
    scores = -lof.negative_outlier_factor_
    return np.asarray(scores, dtype=np.float64)


def ambiguous_mask(
    X: np.ndarray,
    top_percent: float = 1.0,
    n_neighbors: int = 20,
) -> np.ndarray:
    """
    Boolean mask: True = keep for "core" projection, False = ambiguous (top % by LOF).

    Parameters
    ----------
    X : (N, D) array
        Embeddings.
    top_percent : float
        Fraction (0–100) of nodes to label as ambiguous (exclude from core layout).
    n_neighbors : int
        LOF n_neighbors.

    Returns
    -------
    (N,) bool array
        True where node is kept (non-ambiguous), False for ambiguous.
    """
    scores = lof_scores(X, n_neighbors=n_neighbors)
    n = len(scores)
    k = max(1, int(round(n * top_percent / 100.0)))
    # Top k highest scores = most ambiguous
    thresh = np.partition(scores, n - k)[n - k]
    return scores < thresh


def place_outliers_back(
    coords_core: np.ndarray,
    mask_keep: np.ndarray,
    X_highd: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """
    Assign 3D positions to ambiguous (outlier) nodes: place each at centroid
    of its k nearest *kept* nodes in high-D space (outliers have no 3D yet).

    Parameters
    ----------
    coords_core : (n_keep, 3) array
        3D positions for kept nodes only; row j corresponds to kept_idx[j].
    mask_keep : (N,) bool
        True for kept nodes (same order as full node list).
    X_highd : (N, D) array
        High-dimensional embeddings; used to find k-NN kept neighbors per outlier.
    k : int
        Number of nearest kept neighbors (in high-D) for centroid.

    Returns
    -------
    (N, 3) array
        Full layout: coords_core at kept indices, centroid-of-k-NN at outlier indices.
    """
    from sklearn.neighbors import NearestNeighbors

    X_highd = np.asarray(X_highd, dtype=np.float64)
    kept_idx = np.where(mask_keep)[0]
    out_idx = np.where(~mask_keep)[0]
    n_keep = len(kept_idx)
    coords_all = np.zeros((len(mask_keep), 3), dtype=np.float64)
    coords_all[kept_idx] = coords_core

    if len(out_idx) == 0:
        return coords_all

    # k-NN in high-D: fit on kept nodes only; for each outlier query k nearest kept
    k_use = min(k, n_keep)
    X_kept = X_highd[kept_idx]
    nbrs = NearestNeighbors(n_neighbors=k_use, algorithm="auto").fit(X_kept)
    for i in out_idx:
        _, inds = nbrs.kneighbors(X_highd[i : i + 1])
        # inds are indices into X_kept/coords_core (same order as kept_idx)
        coords_all[i] = coords_core[inds[0]].mean(axis=0)
    return coords_all
