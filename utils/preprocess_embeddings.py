"""
Hub-effect pre-processing for GO2Vec embeddings.

In PPI networks, high-degree hub proteins pull everything toward the center.
Apply log-normalization or z-score scaling to embeddings before 3D projection
so that distance metrics are not dominated by global structure.

Use these before projection (e.g. PaCMAP/UMAP) to help "de-compress" the manifold.
"""
import numpy as np


def zscore_scale(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Z-score normalization per dimension: (X - mean) / std.
    Reduces dominance of high-variance dimensions (often hub-related).

    Parameters
    ----------
    X : (N, D) array
        Embedding matrix.
    axis : int
        0 = normalize across samples (per dimension); 1 = per node (per sample).

    Returns
    -------
    (N, D) array, float64
    """
    X = np.asarray(X, dtype=np.float64)
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    std = np.where(std <= 0, 1.0, std)
    return (X - mean) / std


def log1p_scale(X: np.ndarray) -> np.ndarray:
    """
    Log(1 + |x|) * sign(x) per dimension. Compresses large values (hubs)
    while preserving sign and zero.

    Parameters
    ----------
    X : (N, D) array
        Embedding matrix.

    Returns
    -------
    (N, D) array, float64
    """
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.log1p(np.abs(X))


def robust_scale(X: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Robust scaling per dimension: (X - median) / IQR.
    Less sensitive to outliers than z-score.

    Parameters
    ----------
    X : (N, D) array
        Embedding matrix.
    axis : int
        0 = scale across samples (per dimension).

    Returns
    -------
    (N, D) array, float64
    """
    X = np.asarray(X, dtype=np.float64)
    med = np.median(X, axis=axis, keepdims=True)
    q1 = np.percentile(X, 25, axis=axis, keepdims=True)
    q3 = np.percentile(X, 75, axis=axis, keepdims=True)
    iqr = np.where(q3 - q1 <= 0, 1.0, q3 - q1)
    return (X - med) / iqr


def apply_preprocess(X: np.ndarray, method: str) -> np.ndarray:
    """
    Apply one of the supported pre-processing methods.

    Parameters
    ----------
    X : (N, D) array
        Embedding matrix.
    method : str
        One of: "raw", "zscore", "lognorm", "robust".

    Returns
    -------
    (N, D) array
    """
    if method == "raw":
        return np.asarray(X, dtype=np.float64)
    if method == "zscore":
        return zscore_scale(X, axis=0)
    if method == "lognorm":
        return log1p_scale(X)
    if method == "robust":
        return robust_scale(X, axis=0)
    raise ValueError(f"Unknown method: {method!r}. Use raw, zscore, lognorm, or robust.")
