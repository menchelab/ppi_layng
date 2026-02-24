"""
Step 2: Build unified GOA graph from OBO, GAF, and edges.tsv.
Outputs: output/edge_list.parquet (int src, dst), output/node_map.parquet (int_id, str_id, node_type).
"""
import pandas as pd
from tqdm import tqdm

from config_tune import (
    EDGES_TSV,
    GO_OBO_PATH,
    GAF_PATH,
    EDGE_LIST_PARQUET,
    NODE_MAP_PARQUET,
    OUTPUT_DIR,
    GRAPH_EXCLUDE_TOP_K_GO_TERMS,
    ensure_dirs,
)
from gpu_check import require_rapids_gpu


def load_ppi_proteins():
    """Load unique protein IDs from edges.tsv."""
    df = pd.read_csv(EDGES_TSV, sep="\t", usecols=["source", "target"])
    proteins = set(df["source"].astype(str)) | set(df["target"].astype(str))
    return proteins


def load_ontology_edges(obo_path):
    """Load GO term-term edges (is_a, part_of) from go-basic.obo. Returns set of (child, parent) as str."""
    import obonet
    graph = obonet.read_obo(obo_path)
    edges = set()
    for u, v, key in tqdm(graph.edges(keys=True), desc="Loading ontology edges", unit=" edges"):
        if key in ("is_a", "part_of"):
            edges.add((str(u), str(v)))
            edges.add((str(v), str(u)))  # undirected
    return edges


def load_ontology_terms(obo_path):
    """All GO term IDs in the OBO graph."""
    import obonet
    graph = obonet.read_obo(obo_path)
    return set(str(n) for n in graph.nodes())


def load_annotation_edges(gaf_path, protein_set, go_terms):
    """Load (protein_id, go_id) from GAF; filter to protein_set and go_terms."""
    # GAF 2.2: col 2 (0-indexed 1) = DB Object ID, col 5 (0-indexed 4) = GO ID
    # (col 1 = DB e.g. UniProtKB, col 2 = accession). Protein IDs must match edges.tsv.
    COL_DB_OBJECT_ID = 1
    COL_GO_ID = 4
    edges = set()
    with open(gaf_path, "r", encoding="utf-8", errors="replace") as f:
        lines = tqdm(f, desc="Reading GAF", unit=" lines")
        for line in lines:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) <= max(COL_DB_OBJECT_ID, COL_GO_ID):
                continue
            pid = parts[COL_DB_OBJECT_ID].strip()
            go_id = parts[COL_GO_ID].strip()
            if pid in protein_set and go_id in go_terms:
                edges.add((pid, go_id))
                edges.add((go_id, pid))
    return edges


def top_k_go_terms_in_gaf(gaf_path, protein_set, go_terms, k):
    """
    Return set of top-k most frequent GO IDs among proteins in `protein_set`.
    Frequency is counted from GAF protein->GO records after GO-term validity filtering.
    """
    if int(k) <= 0:
        return set()
    COL_DB_OBJECT_ID = 1
    COL_GO_ID = 4
    counts = {}
    with open(gaf_path, "r", encoding="utf-8", errors="replace") as f:
        lines = tqdm(f, desc=f"Counting GO frequencies (top {k})", unit=" lines")
        for line in lines:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) <= max(COL_DB_OBJECT_ID, COL_GO_ID):
                continue
            pid = parts[COL_DB_OBJECT_ID].strip()
            go_id = parts[COL_GO_ID].strip()
            if pid in protein_set and go_id in go_terms:
                counts[go_id] = counts.get(go_id, 0) + 1
    if not counts:
        return set()
    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[: int(k)]
    return {go_id for go_id, _ in top}


def filter_annotation_edges_excluding_go_terms(ann_edges, excluded_go_terms):
    """
    Remove protein<->term annotation edges for GO terms in `excluded_go_terms`.
    `ann_edges` is an undirected edge set containing both (protein, term) and (term, protein).
    """
    if not excluded_go_terms:
        return ann_edges
    excluded = set(excluded_go_terms)
    out = set()
    for a, b in ann_edges:
        if a in excluded or b in excluded:
            continue
        out.add((a, b))
    return out


def load_ppi_edges(edges_tsv):
    """Load (source, target) from edges.tsv as undirected pairs."""
    df = pd.read_csv(edges_tsv, sep="\t", usecols=["source", "target"])
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    edges = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading PPI edges", unit=" edges"):
        edges.add((row["source"], row["target"]))
        edges.add((row["target"], row["source"]))
    return edges


def build_unified_graph():
    ensure_dirs()
    require_rapids_gpu()

    proteins = load_ppi_proteins()
    print(f"PPI proteins: {len(proteins)}")

    go_terms = load_ontology_terms(GO_OBO_PATH)
    print(f"GO terms in OBO: {len(go_terms)}")

    ont_edges = load_ontology_edges(GO_OBO_PATH)
    print(f"Ontology edges (undirected pairs): {len(ont_edges)}")

    ann_edges = load_annotation_edges(GAF_PATH, proteins, go_terms)  # has internal tqdm
    print(f"Annotation edges (protein-term, filtered): {len(ann_edges)}")

    top_k = int(GRAPH_EXCLUDE_TOP_K_GO_TERMS)
    if top_k > 0:
        excluded_go = top_k_go_terms_in_gaf(GAF_PATH, proteins, go_terms, top_k)
        before = len(ann_edges)
        ann_edges = filter_annotation_edges_excluding_go_terms(ann_edges, excluded_go)
        removed = before - len(ann_edges)
        print(
            f"Excluded top-{top_k} generic GO terms: {len(excluded_go)} terms, "
            f"removed {removed} annotation edges"
        )
    else:
        print("GO generic-term exclusion disabled (GRAPH_EXCLUDE_TOP_K_GO_TERMS=0).")

    ppi_edges = load_ppi_edges(EDGES_TSV)
    print(f"PPI edges (undirected pairs): {len(ppi_edges)}")

    # All nodes that appear in any edge set
    all_nodes = set()
    for a, b in ont_edges:
        all_nodes.add(a)
        all_nodes.add(b)
    for a, b in ann_edges:
        all_nodes.add(a)
        all_nodes.add(b)
    for a, b in ppi_edges:
        all_nodes.add(a)
        all_nodes.add(b)

    # Node mapping: str_id -> int_id, and node_type (protein vs term)
    # GO terms have format GO:nnnnnnn; proteins are UniProt IDs
    node_list = sorted(all_nodes)
    str_to_int = {s: i for i, s in enumerate(node_list)}
    N = len(node_list)

    def is_protein(s):
        return s in proteins

    # Build edge list (int, int)
    edge_tuples = []
    seen = set()
    for a, b in tqdm(ont_edges, desc="Adding ontology edges", unit=" edges"):
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if a in str_to_int and b in str_to_int:
            edge_tuples.append((str_to_int[a], str_to_int[b]))
    for a, b in tqdm(ann_edges, desc="Adding annotation edges", unit=" edges"):
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if a in str_to_int and b in str_to_int:
            edge_tuples.append((str_to_int[a], str_to_int[b]))
    for a, b in tqdm(ppi_edges, desc="Adding PPI edges", unit=" edges"):
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if a in str_to_int and b in str_to_int:
            edge_tuples.append((str_to_int[a], str_to_int[b]))

    # Node map: int_id, str_id, node_type
    node_type = ["protein" if is_protein(s) else "term" for s in node_list]
    node_map_df = pd.DataFrame({
        "int_id": range(N),
        "str_id": node_list,
        "node_type": node_type,
    })

    # Edge list as cudf for parquet (GPU-friendly)
    import cudf
    edge_df = cudf.DataFrame({
        "src": [e[0] for e in edge_tuples],
        "dst": [e[1] for e in edge_tuples],
    })
    # Ensure int32 for cugraph compatibility
    edge_df["src"] = edge_df["src"].astype("int32")
    edge_df["dst"] = edge_df["dst"].astype("int32")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    edge_df.to_parquet(EDGE_LIST_PARQUET, index=False)
    node_map_df.to_parquet(NODE_MAP_PARQUET, index=False)
    print(f"Nodes: {N}, Edges: {len(edge_tuples)}")
    print(f"Wrote {EDGE_LIST_PARQUET} and {NODE_MAP_PARQUET}")


def main():
    build_unified_graph()
    print("Step 2 done.")


if __name__ == "__main__":
    main()
