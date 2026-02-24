"""
De-compression 3D layouts: combine hub pre-processing, UMAP, PaCMAP expansion,
LOF denoising, and force-directed refinement into one TSV with columns
x_<approach>, y_<approach>, z_<approach>.

Addresses the crowding problem where 99.9% of nodes sit in a dense ball
by: (1) pre-processing embeddings, (2) UMAP or PaCMAP with expansion params,
(3) optional LOF denoise, (4) optional force-directed refinement.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from config_tune import (
    EMBEDDINGS_PARQUET,
    NODE_MAP_PARQUET,
    OUTPUT_DIR,
    UMAP_RANDOM_STATE,
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST,
)
from utils.preprocess_embeddings import apply_preprocess
from utils.pacmap_expansion import pacmap_3d
from utils.lof_denoise import ambiguous_mask, place_outliers_back
from utils.force_directed_3d import refine_3d_repulsion
from utils.normalize_layout import normalize_to_cube

try:
    from utils.umap_projection import umap_3d
    HAS_UMAP = True
except (ImportError, OSError):
    HAS_UMAP = False
try:
    from utils.trimap_projection import trimap_3d
    HAS_TRIMAP = True
except ImportError:
    HAS_TRIMAP = False

LAYOUT_DECOMPRESSION_TSV = OUTPUT_DIR / "layout_decompression.tsv"

# Pre-processing variants for hub effect
PREPROCESS_METHODS = ["raw", "zscore", "lognorm"]

# PaCMAP expansion params (from user protocol)
PACMAP_NEIGHBORS = 70
PACMAP_MN_RATIO = 0.6
PACMAP_FP_RATIO = 2.0

# LOF: top percent to treat as ambiguous
LOF_TOP_PERCENT = 1.0

# Force-directed refinement
FD_ITERATIONS = 75
FD_K_REPEL = 50


def load_protein_embeddings():
    """Load full embeddings and return protein subset and str_ids."""
    npy_path = EMBEDDINGS_PARQUET.with_suffix(".npy")
    if npy_path.exists():
        emb = np.load(npy_path).astype(np.float64)
    else:
        df = pd.read_parquet(EMBEDDINGS_PARQUET)
        cols = [c for c in df.columns if c.startswith("dim_")]
        emb = df[cols].values.astype(np.float64)
    node_map = pd.read_parquet(NODE_MAP_PARQUET)
    protein_mask = (node_map["node_type"] == "protein").values
    protein_indices = np.where(protein_mask)[0]
    protein_str_ids = node_map.loc[protein_mask, "str_id"].values
    emb_protein = emb[protein_indices]
    return emb_protein, protein_str_ids


def main():
    emb_protein, protein_str_ids = load_protein_embeddings()
    n = len(protein_str_ids)
    print(f"Protein embeddings: {emb_protein.shape} ({n} nodes)")

    data = {"node_id": protein_str_ids}

    # --- Preprocess variants ---
    preprocessed = {}
    for method in tqdm(PREPROCESS_METHODS, desc="Preprocessing", unit=" method"):
        preprocessed[method] = apply_preprocess(emb_protein, method)

    # --- UMAP per preprocess (GPU) ---
    if HAS_UMAP:
        for method in tqdm(PREPROCESS_METHODS, desc="UMAP 3D", unit=" method"):
            X = preprocessed[method]
            name = f"umap_{method}"
            coords = umap_3d(
                X,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST,
                random_state=UMAP_RANDOM_STATE,
            )
            coords = normalize_to_cube(coords, method="percentile", low=0.5, high=99.5)
            data[f"x_{name}"] = coords[:, 0]
            data[f"y_{name}"] = coords[:, 1]
            data[f"z_{name}"] = coords[:, 2]

    # --- PaCMAP expansion per preprocess ---
    for method in tqdm(PREPROCESS_METHODS, desc="PaCMAP expansion", unit=" method"):
        X = preprocessed[method]
        name = f"pacmap_expansion_{method}"
        coords = pacmap_3d(
            X,
            n_neighbors=PACMAP_NEIGHBORS,
            MN_ratio=PACMAP_MN_RATIO,
            FP_ratio=PACMAP_FP_RATIO,
            random_state=UMAP_RANDOM_STATE,
        )
        coords = normalize_to_cube(coords, method="percentile", low=0.5, high=99.5)
        data[f"x_{name}"] = coords[:, 0]
        data[f"y_{name}"] = coords[:, 1]
        data[f"z_{name}"] = coords[:, 2]

    # --- LOF denoise: project core only then place outliers back ---
    method_lof = "zscore"
    X_lof = preprocessed[method_lof]
    mask_keep = ambiguous_mask(X_lof, top_percent=LOF_TOP_PERCENT)
    n_keep = mask_keep.sum()
    X_core = X_lof[mask_keep]
    coords_core = pacmap_3d(
        X_core,
        n_neighbors=PACMAP_NEIGHBORS,
        MN_ratio=PACMAP_MN_RATIO,
        FP_ratio=PACMAP_FP_RATIO,
        random_state=UMAP_RANDOM_STATE,
    )
    coords_lof = place_outliers_back(coords_core, mask_keep, X_lof, k=5)
    coords_lof = normalize_to_cube(coords_lof, method="percentile", low=0.5, high=99.5)
    name_lof = "pacmap_expansion_lof"
    data[f"x_{name_lof}"] = coords_lof[:, 0]
    data[f"y_{name_lof}"] = coords_lof[:, 1]
    data[f"z_{name_lof}"] = coords_lof[:, 2]
    print(f"LOF (PaCMAP): kept {n_keep} core, placed {n - n_keep} outliers back")
    if HAS_UMAP:
        coords_core_umap = umap_3d(
            X_core,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=UMAP_RANDOM_STATE,
        )
        coords_umap_lof = place_outliers_back(coords_core_umap, mask_keep, X_lof, k=5)
        coords_umap_lof = normalize_to_cube(coords_umap_lof, method="percentile", low=0.5, high=99.5)
        name_umap_lof = "umap_lof"
        data[f"x_{name_umap_lof}"] = coords_umap_lof[:, 0]
        data[f"y_{name_umap_lof}"] = coords_umap_lof[:, 1]
        data[f"z_{name_umap_lof}"] = coords_umap_lof[:, 2]
        print("UMAP + LOF added.")

    # --- Optional: TriMAP (global structure preservation) ---
    if HAS_TRIMAP:
        X_raw = preprocessed["raw"]
        coords_tm = trimap_3d(X_raw, random_state=UMAP_RANDOM_STATE)
        coords_tm = normalize_to_cube(coords_tm, method="percentile", low=0.5, high=99.5)
        data["x_trimap_3d"] = coords_tm[:, 0]
        data["y_trimap_3d"] = coords_tm[:, 1]
        data["z_trimap_3d"] = coords_tm[:, 2]
        print("TriMAP 3D added.")

    # --- Force-directed refinement on selected layouts ---
    for method in ["zscore", "lognorm"]:
        for proj in ["pacmap_expansion", "umap"]:
            name_base = f"{proj}_{method}"
            if f"x_{name_base}" not in data:
                continue
            coords = np.column_stack([
                data[f"x_{name_base}"],
                data[f"y_{name_base}"],
                data[f"z_{name_base}"],
            ])
            coords_fd = refine_3d_repulsion(
                coords,
                iterations=FD_ITERATIONS,
                use_knn=True,
                k_repel=FD_K_REPEL,
                random_state=UMAP_RANDOM_STATE,
            )
            coords_fd = normalize_to_cube(coords_fd, method="percentile", low=0.5, high=99.5)
            name_fd = f"{name_base}_fd"
            data[f"x_{name_fd}"] = coords_fd[:, 0]
            data[f"y_{name_fd}"] = coords_fd[:, 1]
            data[f"z_{name_fd}"] = coords_fd[:, 2]

    layout_df = pd.DataFrame(data)
    LAYOUT_DECOMPRESSION_TSV.parent.mkdir(parents=True, exist_ok=True)
    layout_df.to_csv(LAYOUT_DECOMPRESSION_TSV, sep="\t", index=False)
    print(f"Wrote {LAYOUT_DECOMPRESSION_TSV} ({len(layout_df)} rows, {len(layout_df.columns)} columns)")
    print("De-compression layouts done.")


if __name__ == "__main__":
    main()
