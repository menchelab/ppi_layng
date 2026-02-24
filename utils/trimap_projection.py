"""
TriMAP 3D projection for embeddings.

TriMAP uses triplet constraints and preserves global structure better than
t-SNE/UMAP in some settings. Optional dependency: pip install trimap.
"""
import numpy as np


def trimap_3d(
    X: np.ndarray,
    n_components: int = 3,
    n_inliers: int = 10,
    n_outliers: int = 5,
    n_random: int = 5,
    n_iters: int = 400,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project (N, D) to 3D using TriMAP.

    Parameters
    ----------
    X : (N, D) array
        High-dimensional embeddings.
    n_components : int
        3 for x, y, z.
    n_inliers, n_outliers, n_random : int
        Triplet sampling (see trimap docs).
    n_iters : int
        Optimization iterations.
    random_state : int
        For reproducibility.

    Returns
    -------
    (N, 3) array, float64
    """
    import trimap

    X = np.asarray(X, dtype=np.float64)
    # TRIMAP does not expose random_state; set numpy seed for reproducibility
    np.random.seed(random_state)
    embedding = trimap.TRIMAP(
        n_dims=n_components,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        n_iters=n_iters,
    )
    return np.asarray(embedding.fit_transform(X), dtype=np.float64)
