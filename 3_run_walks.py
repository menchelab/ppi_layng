"""
Step 3: Run node2vec random walks on the unified graph (cuGraph).
Inputs: output/edge_list.parquet, output/node_map.parquet.
Outputs: output/walks.parquet (num_walks x walk_length).
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from config_tune import (
    EDGE_LIST_PARQUET,
    NODE_MAP_PARQUET,
    WALKS_PARQUET,
    WALK_LENGTH,
    WALKS_PER_NODE,
    NODE2VEC_P,
    NODE2VEC_Q,
)
from gpu_check import require_rapids_gpu


def main():
    require_rapids_gpu()
    import cudf
    import cugraph

    # Load edge list and node map
    edge_df = cudf.read_parquet(EDGE_LIST_PARQUET)
    node_map = cudf.read_parquet(NODE_MAP_PARQUET)
    n_nodes = len(node_map)

    # Build graph (renumber=False since we already have 0..N-1)
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(edge_df, source="src", destination="dst", renumber=False)

    # Start vertices: each node repeated WALKS_PER_NODE times
    start_list = []
    for _ in range(WALKS_PER_NODE):
        start_list.extend(range(n_nodes))
    start_vertices = cudf.Series(start_list, dtype="int32")
    num_walks = len(start_vertices)

    # Run node2vec (one walk per start vertex). Returns (vertex_paths, edge_weight_paths).
    with tqdm(total=1, desc="Node2vec random walks", unit=" run", bar_format="{desc}: {bar}{postfix}") as pbar:
        pbar.set_postfix_str(f"{num_walks} walks × {WALK_LENGTH}")
        # is this different or at a different place in *every* fsckinhg version????
        # jhgufztdtrsdt trdersarfgh asdf asdf
        result = cugraph.node2vec_random_walks(
            G,
            start_vertices=start_vertices,
            max_depth=WALK_LENGTH,
            p=NODE2VEC_P,
            q=NODE2VEC_Q,
    # not in this env this version ... FML
    #        compress_result=False,
           random_state=42
        )
        pbar.update(1)
    if isinstance(result, tuple):
        paths = result[0]
    elif hasattr(result, "vertex_paths"):
        paths = result["vertex_paths"]
    else:
        paths = result

    # Copy to CPU immediately to avoid CUDA context loss before host copy (WSL2)
    paths_cpu = paths.to_pandas()
    del paths

    # Convert to 2D array (num_walks, walk_length) using CPU data only
    if isinstance(paths_cpu, pd.DataFrame):
        col_names = [c for c in paths_cpu.columns if "vertex" in c.lower() or "path" in c.lower() or c.isdigit()]
        if not col_names:
            col_names = list(paths_cpu.columns)[: WALK_LENGTH + 1]
        mat = paths_cpu[col_names[: WALK_LENGTH]].values
        walks_arr = np.asarray(mat, dtype=np.int32)
    else:
        raw = paths_cpu.values
        if hasattr(raw, "dtype") and np.issubdtype(raw.dtype, np.object_):
            walk_lengths = [len(r) if hasattr(r, "__len__") else 0 for r in raw]
            max_len = min(max(walk_lengths, default=0), WALK_LENGTH) or WALK_LENGTH
            arr = np.full((num_walks, max_len), -1, dtype=np.int32)
            for i, r in enumerate(tqdm(raw, desc="Converting paths", unit=" walks")):
                if hasattr(r, "__len__"):
                    arr[i, : min(len(r), max_len)] = r[:max_len]
                else:
                    arr[i, 0] = int(r)
            walks_arr = arr
            if walks_arr.shape[1] < WALK_LENGTH:
                extra = np.full((num_walks, WALK_LENGTH - walks_arr.shape[1]), -1, dtype=np.int32)
                walks_arr = np.concatenate([walks_arr, extra], axis=1)
        else:
            mat = np.asarray(raw, dtype=np.int32)
            if mat.ndim == 1:
                stride = WALK_LENGTH + 1
                nw = (len(mat) + stride - 1) // stride
                pad = stride * nw - len(mat)
                if pad > 0:
                    mat = np.concatenate([mat, np.full(pad, -1, dtype=np.int32)])
                walks_arr = mat.reshape(nw, stride)[:, :WALK_LENGTH]
            else:
                walks_arr = mat[:, :WALK_LENGTH]

    # Ensure shape (num_walks, WALK_LENGTH)
    if walks_arr.shape[0] != num_walks:
        # Truncate or pad
        if walks_arr.shape[0] > num_walks:
            walks_arr = walks_arr[:num_walks]
        else:
            pad = np.full((num_walks - walks_arr.shape[0], WALK_LENGTH), -1, dtype=np.int32)
            walks_arr = np.concatenate([walks_arr, pad], axis=0)
    if walks_arr.shape[1] > WALK_LENGTH:
        walks_arr = walks_arr[:, :WALK_LENGTH]

    # Keep -1 padding as invalid marker. Step 4 must skip invalid centers/contexts.
    # Converting -1 to 0 would inject synthetic occurrences of real node 0.

    # Save as parquet (DataFrame with columns step_0 .. step_99)
    out_df = cudf.DataFrame({f"step_{i}": walks_arr[:, i] for i in tqdm(range(walks_arr.shape[1]), desc="Writing walk columns", unit=" cols")})
    out_df.to_parquet(WALKS_PARQUET, index=False)
    # Also save as npy for step 4 (simpler to load)
    np.save(WALKS_PARQUET.with_suffix(".npy"), walks_arr)
    print(f"Walks shape: {walks_arr.shape}, saved to {WALKS_PARQUET} and .npy")
    print("Step 3 done.")


if __name__ == "__main__":
    main()
