"""
Force-directed refinement for 3D layouts.

Takes initial (x,y,z) from e.g. PaCMAP/UMAP and runs a short repulsion-only
simulation (e.g. 50–100 iterations) with F ~ 1/d^2 to "unfold" tight clusters
without destroying functional grouping. Uses a simple O(N^2) repulsion with
distance cutoff for large N, or k-NN repulsion for speed.
"""
import numpy as np


def refine_3d_repulsion(
    coords: np.ndarray,
    iterations: int = 75,
    repulsion_strength: float = 1.0,
    min_dist: float = 1e-6,
    max_dist: float = np.inf,
    use_knn: bool = True,
    k_repel: int = 50,
    random_state: int = 42,
) -> np.ndarray:
    """
    Refine 3D coordinates by repulsive forces 1/d^2 between points.

    Parameters
    ----------
    coords : (N, 3) array
        Initial 3D layout.
    iterations : int
        Number of refinement steps (50–100 typical).
    repulsion_strength : float
        Scale of repulsion step.
    min_dist : float
        Clamp distances below this to avoid explosion.
    max_dist : float
        Ignore pairs beyond this (for O(N^2) variant); not used if use_knn.
    use_knn : bool
        If True, each point is repelled only by its k_repel nearest neighbors (faster).
    k_repel : int
        Number of neighbors to repulse from when use_knn=True.
    random_state : int
        For reproducibility (e.g. shuffling update order).

    Returns
    -------
    (N, 3) array, float64
    """
    X = np.asarray(coords, dtype=np.float64).copy()
    n = X.shape[0]
    rng = np.random.default_rng(random_state)

    for _ in range(iterations):
        if use_knn and n > k_repel:
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=k_repel + 1, algorithm="auto").fit(X)
            # include self so we get k_repel others
            _, indices = nbrs.kneighbors(X)
            # indices[:, 0] is self
            displacement = np.zeros_like(X)
            for i in range(n):
                neighbors = indices[i, 1:]  # exclude self
                d_vec = X[i] - X[neighbors]
                d = np.linalg.norm(d_vec, axis=1, keepdims=True)
                d = np.maximum(d, min_dist)
                # F ~ 1/d^2, direction away from neighbors
                unit = d_vec / d
                force = repulsion_strength / (d ** 2)
                displacement[i] += (unit * force).sum(axis=0)
            X += displacement * 0.1  # step size
        else:
            displacement = np.zeros_like(X)
            order = rng.permutation(n)
            for i in order:
                j_mask = np.ones(n, dtype=bool)
                j_mask[i] = False
                d_vec = X[i] - X[j_mask]
                d = np.linalg.norm(d_vec, axis=1, keepdims=True)
                d = np.maximum(d, min_dist)
                if np.isfinite(max_dist):
                    valid = (d.ravel() <= max_dist).reshape(-1, 1)
                    d_vec = np.where(valid, d_vec, 0.0)
                    d = np.where(valid, d, np.nan)
                    d = np.nan_to_num(d, nan=min_dist, posinf=min_dist)
                unit = d_vec / d
                force = repulsion_strength / (d ** 2)
                displacement[i] = (unit * force).sum(axis=0)
            X += displacement * 0.1
    return X
