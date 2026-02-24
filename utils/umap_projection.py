"""
UMAP 3D projection (cuML on GPU).
Returns (N, 3) numpy array. Requires RAPIDS/cuML.
"""
import numpy as np


def umap_3d(
    X: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Project (N, D) to 3D using cuML UMAP. X must fit in GPU memory.

    Returns
    -------
    (N, 3) array, float64
    """
    from gpu_check import require_rapids_gpu
    import cudf
    from cuml.manifold import UMAP

    require_rapids_gpu()
    X = np.asarray(X, dtype=np.float32)
    emb_gpu = cudf.DataFrame(X)
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(emb_gpu)
    if hasattr(coords, "to_pandas"):
        coords = np.asarray(coords.to_pandas().values, dtype=np.float64)
    elif hasattr(coords, "get"):
        coords = np.asarray(coords.get(), dtype=np.float64)
    else:
        coords = np.asarray(coords, dtype=np.float64)
    return coords
