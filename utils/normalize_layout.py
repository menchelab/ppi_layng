"""
Normalize (N, 3) layout coordinates into a cube (e.g. [-1, 1]^3).
"""
import numpy as np


def normalize_to_cube(X, method="percentile", low=0.5, high=99.5):
    """
    Map (N, 3) float array into [-1, 1]^3 in a smart way.

    Parameters
    ----------
    X : (N, 3) array-like
        Layout coordinates.
    method : str
        - "percentile": for each axis, use percentiles at `low` and `high`;
          linear map that interval to [-1, 1], clip outside to -1 or 1.
        - "robust": use median and IQR; map (median - k*IQR, median + k*IQR)
          to [-1, 1] with k so most points fit (default k=3), then clip.
        - "minmax": simple min-max scale to [-1, 1].
    low, high : float
        Used only for method "percentile" (percentile values, e.g. 0.5 and 99.5).

    Returns
    -------
    out : (N, 3) np.ndarray
        Values in [-1, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X must be (N, 3)")
    out = np.empty_like(X)
    for i in range(3):
        col = X[:, i]
        col = np.ravel(col)
        valid = np.isfinite(col)
        if not np.any(valid):
            out[:, i] = 0.0
            continue
        v = col[valid]
        if method == "minmax":
            lo, hi = np.min(v), np.max(v)
            if hi <= lo:
                out[:, i] = 0.0
                continue
            scaled = 2.0 * (col - lo) / (hi - lo) - 1.0
        elif method == "percentile":
            lo = np.percentile(v, low)
            hi = np.percentile(v, high)
            if hi <= lo:
                out[:, i] = 0.0
                continue
            scaled = 2.0 * (col - lo) / (hi - lo) - 1.0
            scaled = np.clip(scaled, -1.0, 1.0)
        elif method == "robust":
            med = np.median(v)
            q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
            iqr = q3 - q1
            k = 3.0
            if iqr <= 0:
                out[:, i] = 0.0
                continue
            lo = med - k * iqr
            hi = med + k * iqr
            scaled = 2.0 * (col - lo) / (hi - lo) - 1.0
            scaled = np.clip(scaled, -1.0, 1.0)
        else:
            raise ValueError(
                f"method must be 'percentile', 'robust', or 'minmax'; got {method!r}"
            )
        out[:, i] = np.where(valid, scaled, 0.0)
    return out
