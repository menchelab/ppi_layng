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
    step_size: float = 0.1,
    min_dist: float = 1e-6,
    max_dist: float = np.inf,
    use_knn: bool = True,
    k_repel: int = 50,
    anchor_strength: float = 0.0,
    max_step_factor: float = 0.0,
    dense_only_quantile: float | None = None,
    two_phase: bool = False,
    phase_split: float = 0.35,
    phase1_strength_mult: float = 1.5,
    phase2_strength_mult: float = 0.6,
    early_stop_patience: int = 0,
    early_stop_min_improve: float = 1e-4,
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
    step_size : float
        Global step size multiplier.
    min_dist : float
        Clamp distances below this to avoid explosion.
    max_dist : float
        Ignore pairs beyond this (for O(N^2) variant); not used if use_knn.
    use_knn : bool
        If True, each point is repelled only by its k_repel nearest neighbors (faster).
    k_repel : int
        Number of neighbors to repulse from when use_knn=True.
    anchor_strength : float
        Pullback force to original coordinates (0 disables). Higher preserves global layout.
    max_step_factor : float
        Per-iteration displacement clipping as factor * median nearest-neighbor distance.
        0 disables clipping.
    dense_only_quantile : float | None
        If set in (0,1), only nodes with nearest-neighbor distance <= this quantile are
        actively repelled; others only get anchor pull.
    two_phase : bool
        If True, use stronger repulsion early and gentler repulsion later.
    phase_split : float
        Fraction of iterations in phase 1 when two_phase=True.
    phase1_strength_mult, phase2_strength_mult : float
        Multipliers applied to repulsion_strength for phase 1 / phase 2.
    early_stop_patience : int
        Stop early if mean step norm improvement stalls for this many iterations (0 disables).
    early_stop_min_improve : float
        Minimum absolute improvement in mean step norm to reset early-stop counter.
    random_state : int
        For reproducibility (e.g. shuffling update order).

    Returns
    -------
    (N, 3) array, float64
    """
    X = np.asarray(coords, dtype=np.float64).copy()
    X0 = X.copy()
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    stall_count = 0
    prev_mean_step = None

    for it in range(iterations):
        if two_phase:
            split_idx = max(1, int(iterations * float(phase_split)))
            strength_mult = phase1_strength_mult if it < split_idx else phase2_strength_mult
        else:
            strength_mult = 1.0
        eff_repulsion = float(repulsion_strength) * float(strength_mult)

        if use_knn and n > k_repel:
            from sklearn.neighbors import NearestNeighbors

            nbrs = NearestNeighbors(n_neighbors=k_repel + 1, algorithm="auto").fit(X)
            # include self so we get k_repel others
            d_knn, indices = nbrs.kneighbors(X)
            # indices[:, 0] is self
            displacement = np.zeros_like(X)
            nn_dist = d_knn[:, 1] if d_knn.shape[1] > 1 else np.full(n, np.inf)
            if dense_only_quantile is not None and 0.0 < float(dense_only_quantile) < 1.0:
                thr = np.quantile(nn_dist[np.isfinite(nn_dist)], float(dense_only_quantile))
                active = nn_dist <= thr
            else:
                active = np.ones(n, dtype=bool)
            for i in range(n):
                if not active[i]:
                    continue
                neighbors = indices[i, 1:]  # exclude self
                d_vec = X[i] - X[neighbors]
                d = np.linalg.norm(d_vec, axis=1, keepdims=True)
                d = np.maximum(d, min_dist)
                # F ~ 1/d^2, direction away from neighbors
                unit = d_vec / d
                force = eff_repulsion / (d ** 2)
                displacement[i] += (unit * force).sum(axis=0)

            if anchor_strength > 0:
                displacement -= float(anchor_strength) * (X - X0)

            step = displacement * float(step_size)
            if max_step_factor > 0:
                finite_nn = nn_dist[np.isfinite(nn_dist) & (nn_dist > 0)]
                local_scale = float(np.median(finite_nn)) if finite_nn.size else 1.0
                max_step = float(max_step_factor) * local_scale
                norms = np.linalg.norm(step, axis=1, keepdims=True)
                scale = np.minimum(1.0, max_step / np.maximum(norms, 1e-12))
                step = step * scale
            X += step
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
                force = eff_repulsion / (d ** 2)
                displacement[i] = (unit * force).sum(axis=0)

            if anchor_strength > 0:
                displacement -= float(anchor_strength) * (X - X0)
            step = displacement * float(step_size)
            if max_step_factor > 0:
                # Fallback local scale estimate from current cloud radius
                local_scale = float(np.median(np.linalg.norm(X - np.median(X, axis=0), axis=1)))
                local_scale = max(local_scale, 1e-6)
                max_step = float(max_step_factor) * local_scale
                norms = np.linalg.norm(step, axis=1, keepdims=True)
                scale = np.minimum(1.0, max_step / np.maximum(norms, 1e-12))
                step = step * scale
            X += step

        if early_stop_patience > 0:
            mean_step = float(np.mean(np.linalg.norm(step, axis=1)))
            if prev_mean_step is not None and abs(prev_mean_step - mean_step) < float(early_stop_min_improve):
                stall_count += 1
            else:
                stall_count = 0
            prev_mean_step = mean_step
            if stall_count >= int(early_stop_patience):
                break
    return X
