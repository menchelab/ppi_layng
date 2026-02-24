# VERY EXPERIMENTAL - PPI 3D layout w/ GO Ontology

Pipeline for 3D embeddings of large PPI networks (~20k nodes) by fusing the interaction graph with Gene Ontology: unified graph → node2vec walks → skip-gram embeddings → manifold projection (UMAP/PaCMAP) with optional de-compression (hub scaling, LOF denoise, force-directed refinement).

**Input:** `input/edges.tsv` (columns `source`, `target`; protein IDs).  
**Data:** `data/go-basic.obo`, `data/goa_human.gaf` (step 1 downloads if missing).  
**Env:** RAPIDS (cudf, cugraph, cuml), PyTorch+CUDA, `pip install -r requirements.txt`. Optional: `trimap` for TriMAP in step 6.

---

## Process files (run in order)

**`1_download_data.py`**  
Downloads `go-basic.obo` and `goa_human.gaf.gz` to `data/` (skips if present). Decompresses GAF. Uses `config` URLs.

**`2_build_graph.py`**  
Builds a single undirected graph: (1) GO term–term edges from OBO (`is_a`, `part_of` via obonet); (2) protein–GO edges from GAF, restricted to PPI proteins and OBO terms; (3) PPI edges from `edges.tsv`. Nodes get a linear index; node types (protein vs term) stored. Writes `output/edge_list.parquet` (int `src`, `dst`) and `output/node_map.parquet` (`int_id`, `str_id`, `node_type`). Uses cuGraph for construction.

**`3_run_walks.py`**  
Loads the graph with cugraph, runs node2vec (biased random walks with return/in-out params `p`, `q`). One walk per (node × `WALKS_PER_NODE`), length `WALK_LENGTH`. Outputs `output/walks.parquet` and `.npy` (shape num_walks × walk_length). `q > 1` (config) favours structural equivalence.

**`4_train_embeddings.py`**  
Reads walks, extracts (center, context) pairs with a fixed window; negative sampling. Trains a skip-gram model (PyTorch, GPU) to embed node IDs. Writes `output/embeddings.parquet` and `.npy` (all nodes × `EMBED_DIM`). Only protein rows are used later for layout.

**`5_project_3d.py`**  
Subsets embeddings to proteins, runs cuML UMAP (3 components, config `n_neighbors`, `min_dist`). Writes `output/layout.tsv`: `node_id`, `x`, `y`, `z`.

**`5_multiproject_3d.py`** (alternative to 5)  
Same inputs; runs 20 UMAP parameter variants + 3 PaCMAP variants; each gets columns `x_<name>`, `y_<name>`, `z_<name>`. Writes `output/layout_multi.tsv`.

**`6_decompression_layouts.py`**  
Loads protein embeddings. Applies three preprocesses (raw, zscore, lognorm) to reduce hub dominance. For each: (1) UMAP 3D (if cuML available); (2) PaCMAP 3D with expansion params (n_neighbors=70, MN_ratio=0.6, FP_ratio=2.0). LOF: top 1% “ambiguous” nodes (by LOF in high-D) excluded from projection, then placed back at k-NN centroid in high-D (PaCMAP and UMAP). Optional TriMAP on raw. Force-directed refinement (k-NN repulsion, ~75 iters) on selected PaCMAP and UMAP layouts. Single TSV: `output/layout_decompression.tsv` with one `x_<approach>`, `y_<approach>`, `z_<approach>` per method (e.g. `umap_zscore`, `pacmap_expansion_lof`, `umap_zscore_fd`).

**`7_add_go_terms.py`**  
From `edge_list` + `node_map`, infers protein → GO term IDs from annotation edges. Optionally loads OBO for ID→name; builds comma-separated `go_terms` (IDs) and `go_terms_readable` (names, commas stripped inside names). Merges into the chosen layout TSV (prefer `layout_decompression.tsv` > `layout_multi.tsv` > `layout.tsv`). Usage: `python 7_add_go_terms.py [layout.tsv] [out.tsv]`.

**`8_distribution_charts.py`**  
Reads the final layout TSV (same precedence as step 7), discovers all `x_*`, `y_*`, `z_*` method columns, and plots one row per method with three density histograms (x, y, z). Writes `output/distribution_charts.png`. Usage: `python 8_distribution_charts.py [layout.tsv]`.

**`10_add_generic_umap_layout.py`**  
Builds a protein-protein adjacency matrix from `edge_list.parquet`, runs a generic 3D UMAP on that adjacency-feature space, and appends `x_umap_adjacency`, `y_umap_adjacency`, `z_umap_adjacency` to the final TSV for side-by-side comparison. Usage: `python 10_add_generic_umap_layout.py [layout_in.tsv] [layout_out.tsv]`.

**`11_add_generic_pacmap_layout.py`**  
Builds a protein-protein adjacency matrix from `edge_list.parquet`, runs a regular 3D PaCMAP baseline (non-expansion) on adjacency-derived features, and appends `x_pacmap_adjacency`, `y_pacmap_adjacency`, `z_pacmap_adjacency` to the final TSV. Usage: `python 11_add_generic_pacmap_layout.py [layout_in.tsv] [layout_out.tsv]`.

**`12.1_umap_weighted_concat.py`**  
Hybrid feature fusion baseline: concatenate standardized model embeddings with standardized adjacency-SVD features using a tunable weight, then run UMAP. Appends `x/y/z_umap_fused_concat_*`. Usage: `python 12.1_umap_weighted_concat.py --weight 0.10`.

**`12.2_umap_late_fusion.py`**  
Late fusion baseline: run UMAP separately on embeddings and adjacency-SVD, align layouts (Procrustes), then blend in 3D with weight `w`. Appends `x/y/z_umap_fused_lateblend_*`. Usage: `python 12.2_umap_late_fusion.py --weight 0.20`.

**`12.3_umap_distance_fusion.py`**  
Distance fusion baseline: blend cosine distance matrices from embeddings and adjacency-SVD, then run UMAP with `metric=precomputed`. Appends `x/y/z_umap_fused_distance_*`. Usage: `python 12.3_umap_distance_fusion.py --weight 0.20` (memory-heavy on large N).

**`12.4_umap_graph_diffusion.py`**  
Topology smoothing baseline: diffuse embeddings over row-normalized PPI adjacency for `steps`, then UMAP. Appends `x/y/z_umap_graph_diffusion_*`. Usage: `python 12.4_umap_graph_diffusion.py --beta 0.80 --steps 2`.

**`12.5_umap_multiview_knn_union.py`**  
Multi-view graph baseline: build kNN affinity graphs from embeddings and adjacency-SVD, fuse them by weight, reduce via SVD, and run UMAP. Appends `x/y/z_umap_multiview_knn_union_*`. Usage: `python 12.5_umap_multiview_knn_union.py --weight 0.30 --knn 30`.

Coordinate normalization is **config-driven** in `config_tune.py` (`LAYOUT_NORMALIZE_COORDS`, default `False`). Keep it off to preserve native manifold geometry; enable it only for cross-method visual comparability. To check per-method spread and flag collapsed layouts, run `python -m utils.distribution_analysis output/layout_decompression.tsv`.

---

## Integrity helper scripts

- `python -m utils.check_graph_integrity` — validates node/edge type counts, annotation coverage, and degree structure from `edge_list` + `node_map`.
- `python -m utils.check_walk_integrity` — validates walk padding, node-0 frequency, unique-node coverage, and per-position invalid rates.
- `python -m utils.check_embedding_integrity` — checks embedding norms, near-constant dimensions, and sampled cosine-similarity spread.
- `python -m utils.check_layout_integrity [layout.tsv]` — compares per-method spread, center mass, and sampled nearest-neighbor distances.
- `python -m utils.pipeline_diagnostics [output/diagnostics_report.md]` — runs all major checks and writes one markdown report.

---

## Outputs

| File | Contents |
|------|----------|
| `output/layout.tsv` | `node_id`, `x`, `y`, `z` (single UMAP). |
| `output/layout_multi.tsv` | `node_id` + many `x_*/y_*/z_*` (UMAP/PaCMAP sweeps). |
| `output/layout_decompression.tsv` | `node_id` + `x_*/y_*/z_*` for umap_*, pacmap_expansion_*, *_fd, optional trimap_3d. |
| After step 7 | Same layout + `go_terms`, `go_terms_readable`. |
| `output/distribution_charts.png` | One row per layout method, 3 cols (x, y, z distributions); from step 8. |

**Config:** `config.py` (paths, graph/embed params), `config_tune.py` (tuned training and projection defaults).
