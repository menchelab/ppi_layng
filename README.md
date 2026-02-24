# PPI 3D layout (GO2Vec)

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

Layout coordinates from steps 5 and 6 are **normalized to [-1, 1]³** (percentile-based per axis) so all methods use the same scale and collapsed “ball” layouts are stretched for comparison. To check per-method spread and flag collapsed layouts, run `python -m utils.distribution_analysis output/layout_decompression.tsv`.

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
