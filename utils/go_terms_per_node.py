"""
Build protein (node_id) -> list of GO terms from the unified graph.
Uses edge_list.parquet + node_map.parquet: annotation edges are
protein-term links; we collect all term str_ids for each protein.
Human-readable names come from go-basic.obo (obonet).
"""
import pandas as pd
from pathlib import Path


def load_go_id_to_name(obo_path: Path) -> dict:
    """
    Load GO term ID -> human-readable name from go-basic.obo.
    Returns dict; missing names fall back to the ID. Names are sanitized (comma -> space).
    """
    obo_path = Path(obo_path)
    if not obo_path.exists():
        return {}
    import obonet
    graph = obonet.read_obo(obo_path)
    out = {}
    for go_id, data in graph.nodes(data=True):
        name = data.get("name") or go_id
        if isinstance(name, (list, tuple)):
            name = name[0] if name else str(go_id)
        name = str(name).strip().replace(",", " ")
        out[str(go_id)] = name
    return out


def _sanitize_term(t: str) -> str:
    """Remove commas from a single GO term so comma-separated joining is safe."""
    if not isinstance(t, str):
        return str(t)
    return t.strip().replace(",", "")


def build_protein_to_go_terms(
    edge_list_parquet: Path,
    node_map_parquet: Path,
) -> pd.Series:
    """
    Build a Series index by protein str_id (node_id), value = comma-separated
    GO terms (with commas removed from each term).

    Returns
    -------
    pd.Series
        Index: protein str_id. Values: str of GO terms joined by ",".
    """
    edges = pd.read_parquet(edge_list_parquet)
    node_map = pd.read_parquet(node_map_parquet)
    # node_map: int_id, str_id, node_type (int_id = 0..N-1 from 2_build_graph)
    id_col = node_map.columns[0] if "int_id" not in node_map.columns else "int_id"
    str_col = "str_id" if "str_id" in node_map.columns else node_map.columns[1]
    type_col = "node_type" if "node_type" in node_map.columns else node_map.columns[2]
    id_to_str = dict(zip(node_map[id_col], node_map[str_col]))
    id_to_type = dict(zip(node_map[id_col], node_map[type_col]))

    # Edges: src, dst (int). Find (protein, term) pairs.
    protein_terms = {}  # protein_str_id -> set of term str_ids
    for _, row in edges.iterrows():
        a, b = int(row["src"]), int(row["dst"])
        ta, tb = id_to_type.get(a), id_to_type.get(b)
        if ta == "protein" and tb == "term":
            pid = id_to_str.get(a)
            go_id = id_to_str.get(b)
            if pid is not None and go_id is not None:
                protein_terms.setdefault(pid, set()).add(_sanitize_term(go_id))
        elif ta == "term" and tb == "protein":
            pid = id_to_str.get(b)
            go_id = id_to_str.get(a)
            if pid is not None and go_id is not None:
                protein_terms.setdefault(pid, set()).add(_sanitize_term(go_id))

    # Sort for reproducibility; join with comma
    out = pd.Series(
        index=list(protein_terms.keys()),
        data=[",".join(sorted(s)) for s in protein_terms.values()],
    )
    return out


def go_series_to_readable(go_series: pd.Series, id_to_name: dict, sep: str = ",") -> pd.Series:
    """
    Map a Series of comma-separated GO IDs to comma-separated human-readable names.
    id_to_name: dict from load_go_id_to_name(obo_path). sep joins multiple terms (no spaces).
    """
    def _map_row(s):
        if pd.isna(s) or not str(s).strip():
            return ""
        ids = [x.strip() for x in str(s).split(",") if x.strip()]
        names = [id_to_name.get(i, i) for i in ids]
        return sep.join(names)
    return go_series.apply(_map_row)


def add_go_terms_column(
    layout_df: pd.DataFrame,
    go_series: pd.Series,
    node_id_column: str = "node_id",
    go_column: str = "go_terms",
    go_readable_series: pd.Series | None = None,
    go_readable_column: str = "go_terms_readable",
) -> pd.DataFrame:
    """
    Add column(s) to layout_df with GO terms per node (IDs and optionally names).
    Nodes without annotations get empty string.

    Parameters
    ----------
    layout_df : DataFrame
        Must have column node_id_column (protein str_id).
    go_series : Series
        Index = protein str_id, value = "GO:1,GO:2,...".
    node_id_column : str
        Name of node identifier column in layout_df.
    go_column : str
        Name of the new column for IDs.
    go_readable_series : Series, optional
        If given, index = protein str_id, value = human-readable names (e.g. " | "-separated).
    go_readable_column : str
        Name of the human-readable column when go_readable_series is provided.

    Returns
    -------
    DataFrame
        layout_df with new column(s) (copy).
    """
    out = layout_df.copy()
    out[go_column] = out[node_id_column].map(go_series).fillna("")
    if go_readable_series is not None:
        out[go_readable_column] = out[node_id_column].map(go_readable_series).fillna("")
    return out
