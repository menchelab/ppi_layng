"""
End-to-end diagnostics for GO2Vec pipeline artifacts.

Writes a single markdown report with:
1) graph integrity
2) walk integrity
3) protein-centered context composition
4) embedding integrity
5) layout integrity (per method)

Usage:
  python -m utils.pipeline_diagnostics
  python -m utils.pipeline_diagnostics output/diagnostics_report.md
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from config_tune import (
    EDGE_LIST_PARQUET,
    NODE_MAP_PARQUET,
    WALKS_PARQUET,
    EMBEDDINGS_PARQUET,
    OUTPUT_DIR,
    SKIPGRAM_WINDOW,
)
from utils.distribution_analysis import discover_methods


def _read_walks() -> np.ndarray:
    npy = WALKS_PARQUET.with_suffix(".npy")
    if npy.exists():
        return np.load(npy).astype(np.int64, copy=False)
    df = pd.read_parquet(WALKS_PARQUET)
    cols = [c for c in df.columns if c.startswith("step_")]
    return df[cols].to_numpy(dtype=np.int64)


def _read_embeddings() -> np.ndarray:
    npy = EMBEDDINGS_PARQUET.with_suffix(".npy")
    if npy.exists():
        return np.load(npy).astype(np.float64, copy=False)
    df = pd.read_parquet(EMBEDDINGS_PARQUET)
    cols = [c for c in df.columns if c.startswith("dim_")]
    return df[cols].to_numpy(dtype=np.float64)


def _sampled_cosine(X: np.ndarray, n_pairs: int = 30000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    v1 = X[i]
    v2 = X[j]
    n1 = np.linalg.norm(v1, axis=1) + 1e-12
    n2 = np.linalg.norm(v2, axis=1) + 1e-12
    return np.sum(v1 * v2, axis=1) / (n1 * n2)


def graph_section(edge_df: pd.DataFrame, node_df: pd.DataFrame) -> str:
    nodes = node_df.copy()
    if "int_id" not in nodes.columns:
        nodes = nodes.reset_index().rename(columns={"index": "int_id"})
    nodes = nodes.sort_values("int_id")
    t = nodes["node_type"].astype(str).str.lower()
    n_prot = int((t == "protein").sum())
    n_term = int((t == "term").sum())

    max_id = int(nodes["int_id"].max())
    type_arr = np.full(max_id + 1, -1, dtype=np.int8)
    type_arr[nodes.loc[t == "protein", "int_id"].to_numpy(dtype=np.int64)] = 0
    type_arr[nodes.loc[t == "term", "int_id"].to_numpy(dtype=np.int64)] = 1

    src = edge_df["src"].to_numpy(dtype=np.int64)
    dst = edge_df["dst"].to_numpy(dtype=np.int64)
    st = type_arr[src]
    dt = type_arr[dst]
    pp = int(np.sum((st == 0) & (dt == 0)))
    pt = int(np.sum((st == 0) & (dt == 1)))
    tp = int(np.sum((st == 1) & (dt == 0)))
    tt = int(np.sum((st == 1) & (dt == 1)))

    base = max_id + 1
    undirected_keys = np.unique(np.minimum(src, dst) * base + np.maximum(src, dst))
    u = undirected_keys // base
    v = undirected_keys % base
    ut = type_arr[u]
    vt = type_arr[v]
    pp_u = int(np.sum((ut == 0) & (vt == 0)))
    pt_u = int(np.sum(((ut == 0) & (vt == 1)) | ((ut == 1) & (vt == 0))))
    tt_u = int(np.sum((ut == 1) & (vt == 1)))

    annot_mask = ((ut == 0) & (vt == 1)) | ((ut == 1) & (vt == 0))
    annot_u = u[annot_mask]
    annot_v = v[annot_mask]
    proteins_with_go = np.unique(
        np.concatenate([annot_u[type_arr[annot_u] == 0], annot_v[type_arr[annot_v] == 0]])
    )

    return (
        "## Graph integrity\n"
        f"- Nodes: **{len(nodes):,}** (proteins {n_prot:,}, terms {n_term:,})\n"
        f"- Edges stored (directed pairs): **{len(edge_df):,}**\n"
        f"- Edges undirected unique: **{len(undirected_keys):,}**\n"
        f"- Directed type counts: pp={pp:,}, pt={pt:,}, tp={tp:,}, tt={tt:,}\n"
        f"- Undirected type counts: pp={pp_u:,}, pt={pt_u:,}, tt={tt_u:,}\n"
        f"- Proteins with >=1 GO term: **{len(proteins_with_go):,}/{n_prot:,} ({len(proteins_with_go)/max(1,n_prot):.2%})**\n"
    )


def walk_section(walks: np.ndarray, node_df: pd.DataFrame) -> tuple[str, dict]:
    invalid = walks < 0
    valid = ~invalid
    total = walks.size
    valid_vals = walks[valid]
    node0 = int(np.sum(valid_vals == 0))
    uniq = int(np.unique(valid_vals).size) if valid_vals.size else 0

    id_to_str = {}
    if {"int_id", "str_id"}.issubset(node_df.columns):
        id_to_str = dict(zip(node_df["int_id"].to_numpy(), node_df["str_id"].astype(str).to_numpy()))

    top_txt = []
    if valid_vals.size:
        counts = np.bincount(valid_vals)
        top = np.argsort(counts)[::-1][:15]
        for nid in top:
            cnt = int(counts[nid])
            if cnt <= 0:
                continue
            top_txt.append(f"  - {id_to_str.get(int(nid), str(int(nid)))}: {cnt:,}")

    sec = (
        "## Walk integrity\n"
        f"- Shape: **{walks.shape}**\n"
        f"- Valid tokens: **{int(valid.sum()):,}/{total:,}**, invalid(-1): **{int(invalid.sum()):,} ({invalid.mean():.4%})**\n"
        f"- Unique valid node IDs: **{uniq:,}**\n"
        f"- Node 0 frequency: **{node0:,} ({node0/max(1,valid_vals.size):.4%})**\n"
        "- Top token frequencies:\n"
        + "\n".join(top_txt)
        + "\n"
    )
    return sec, id_to_str


def context_section(walks: np.ndarray, node_df: pd.DataFrame, id_to_str: dict) -> str:
    types = node_df["node_type"].astype(str).to_numpy()
    rng = np.random.default_rng(42)
    n = 300_000
    wi = rng.integers(0, walks.shape[0], n)
    wj = rng.integers(0, walks.shape[1], n)
    centers = walks[wi, wj]
    m = centers >= 0
    wi, wj, centers = wi[m], wj[m], centers[m]

    pm = types[centers] == "protein"
    wi, wj, centers = wi[pm], wj[pm], centers[pm]
    if len(centers) == 0:
        return "## Protein-centered context composition\n- No valid protein centers sampled.\n"

    ctx = []
    W = SKIPGRAM_WINDOW
    for off in range(-W, W + 1):
        if off == 0:
            continue
        jj = wj + off
        ok = (jj >= 0) & (jj < walks.shape[1])
        vals = walks[wi[ok], jj[ok]]
        vals = vals[vals >= 0]
        ctx.append(vals)
    ctx = np.concatenate(ctx) if ctx else np.array([], dtype=np.int64)
    if ctx.size == 0:
        return "## Protein-centered context composition\n- No valid contexts sampled.\n"

    term_ratio = float(np.mean(types[ctx] == "term"))
    term_ctx = ctx[types[ctx] == "term"]
    top_txt = []
    if term_ctx.size:
        counts = np.bincount(term_ctx, minlength=len(types))
        top = np.argsort(counts)[::-1][:20]
        for tid in top:
            cnt = int(counts[tid])
            if cnt <= 0:
                continue
            top_txt.append(f"  - {id_to_str.get(int(tid), str(int(tid)))}: {cnt:,}")

    return (
        "## Protein-centered context composition\n"
        f"- Sampled protein centers: **{len(centers):,}**\n"
        f"- Term ratio in contexts: **{term_ratio:.4%}**\n"
        "- Top GO terms in protein-centered contexts:\n"
        + "\n".join(top_txt)
        + "\n"
    )


def embedding_section(X: np.ndarray, node_df: pd.DataFrame) -> str:
    norms = np.linalg.norm(X, axis=1)
    dim_std = X.std(axis=0)
    cos_all = _sampled_cosine(X)

    prot_mask = (node_df["node_type"].astype(str).to_numpy() == "protein")
    Xp = X[prot_mask] if len(prot_mask) == X.shape[0] else None
    if Xp is not None:
        norms_p = np.linalg.norm(Xp, axis=1)
        dim_std_p = Xp.std(axis=0)
        cos_p = _sampled_cosine(Xp)
    else:
        norms_p = dim_std_p = cos_p = None

    sec = (
        "## Embedding integrity\n"
        f"- Shape: **{X.shape}**, finite: **{np.isfinite(X).all()}**\n"
        f"- Norms (all): mean/median/p95/max = {norms.mean():.4f}/{np.median(norms):.4f}/{np.percentile(norms,95):.4f}/{norms.max():.4f}\n"
        f"- Dim std (all): min/median/max = {dim_std.min():.6f}/{np.median(dim_std):.6f}/{dim_std.max():.6f}\n"
        f"- Cosine sampled (all): mean/median/p5/p95 = {cos_all.mean():.4f}/{np.median(cos_all):.4f}/{np.percentile(cos_all,5):.4f}/{np.percentile(cos_all,95):.4f}\n"
    )
    if norms_p is not None:
        sec += (
            f"- Norms (proteins): mean/median/p95/max = {norms_p.mean():.4f}/{np.median(norms_p):.4f}/{np.percentile(norms_p,95):.4f}/{norms_p.max():.4f}\n"
            f"- Dim std (proteins): min/median/max = {dim_std_p.min():.6f}/{np.median(dim_std_p):.6f}/{dim_std_p.max():.6f}\n"
            f"- Cosine sampled (proteins): mean/median/p5/p95 = {cos_p.mean():.4f}/{np.median(cos_p):.4f}/{np.percentile(cos_p,5):.4f}/{np.percentile(cos_p,95):.4f}\n"
        )
    return sec


def layout_section() -> str:
    lp = OUTPUT_DIR / "layout_decompression.tsv"
    if not lp.exists():
        lp = OUTPUT_DIR / "layout_multi.tsv"
    if not lp.exists():
        lp = OUTPUT_DIR / "layout.tsv"
    if not lp.exists():
        return "## Layout integrity\n- No layout file found in output/.\n"

    df = pd.read_csv(lp, sep="\t")
    methods = discover_methods(df)
    if not methods and {"x", "y", "z"}.issubset(df.columns):
        methods = ["single"]
    rows = []
    for m in methods:
        if m == "single":
            X = np.column_stack([df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy()]).astype(np.float64)
        else:
            X = np.column_stack([
                df[f"x_{m}"].to_numpy(),
                df[f"y_{m}"].to_numpy(),
                df[f"z_{m}"].to_numpy(),
            ]).astype(np.float64)
        X = X[np.isfinite(X).all(axis=1)]
        if X.size == 0:
            continue
        r = np.linalg.norm(X, axis=1)
        rows.append(
            (
                m,
                float(X[:, 0].std()),
                float(X[:, 1].std()),
                float(X[:, 2].std()),
                float(np.mean(r <= 0.2)),
                float(np.mean(r <= 0.5)),
            )
        )

    lines = [
        "## Layout integrity",
        f"- Layout path: **{lp}**",
        "",
        "| method | std_x | std_y | std_z | r<=0.2 | r<=0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, sx, sy, sz, c02, c05 in rows:
        lines.append(f"| {m} | {sx:.4f} | {sy:.4f} | {sz:.4f} | {c02:.2%} | {c05:.2%} |")
    return "\n".join(lines) + "\n"


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (OUTPUT_DIR / "diagnostics_report.md")
    edge_df = pd.read_parquet(EDGE_LIST_PARQUET)
    node_df = pd.read_parquet(NODE_MAP_PARQUET)
    walks = _read_walks()
    emb = _read_embeddings()

    sections = [
        "# GO2Vec Pipeline Diagnostics",
        "",
        graph_section(edge_df, node_df),
    ]
    walk_sec, id_to_str = walk_section(walks, node_df)
    sections.append(walk_sec)
    sections.append(context_section(walks, node_df, id_to_str))
    sections.append(embedding_section(emb, node_df))
    sections.append(layout_section())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"Wrote diagnostics report: {out_path}")


if __name__ == "__main__":
    main()

