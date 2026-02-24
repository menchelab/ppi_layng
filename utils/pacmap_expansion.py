"""
PaCMAP projection with expansion-oriented parameters for de-compressing
crowded high-dimensional embeddings (e.g. GO2Vec).

Uses neighbor + mid-near + further pairs; higher n_neighbors and FP_ratio
help push non-related clusters apart and preserve global structure.
"""
import numpy as np


def pacmap_3d(
    X: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 70,
    MN_ratio: float = 0.6,
    FP_ratio: float = 2.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project (N, D) embeddings to 3D using PaCMAP with expansion-friendly settings.

    Parameters
    ----------
    X : (N, D) array
        High-dimensional embeddings (e.g. 100D GO2Vec).
    n_components : int
        3 for x, y, z.
    n_neighbors : int
        Increase (50–100) to capture broader biological context.
    MN_ratio : float
        Mid-near ratio; ~0.6 preserves more global structure.
    FP_ratio : float
        Further-pair ratio; ~2.0 aggressively pushes non-related clusters apart.
    random_state : int
        For reproducibility.

    Returns
    -------
    (N, 3) array, float64
    """
    import pacmap

    X = np.asarray(X, dtype=np.float64)
    reducer = pacmap.PaCMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(X), dtype=np.float64)
