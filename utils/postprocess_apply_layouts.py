"""
Apply ForceAtlas-like postprocessing to selected layouts in a nodes/layout TSV.

Usage:
  python -m utils.postprocess_apply_layouts <nodes.tsv> <recommendations.tsv> [layout1 layout2 ...]

Positional args:
  argv1: nodes/layout TSV containing x_<layout>, y_<layout>, z_<layout> columns.
  argv2: recommendation TSV from utils.postprocess_analyze_layouts.py
  argv3+: layout names to postprocess (e.g. adju1 adju2 adjp3). If omitted, runs for all layouts in recommendations.

Behavior:
  - For each selected layout name L, reads recommended parameters from recommendations TSV
    (matching row where layout == L), runs postprocessing on x_L/y_L/z_L, and appends:
      x_Lp, y_Lp, z_Lp
  - Writes back to the input TSV by default (or optional --out path).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.force_directed_3d import refine_3d_repulsion


def parse_args():
    p = argparse.ArgumentParser(description="Apply recommended postprocessing to selected layouts.")
    p.add_argument("nodes_tsv", type=str, help="Input nodes/layout TSV.")
    p.add_argument("recommendations_tsv", type=str, help="TSV from postprocess_analyze_layouts.py")
    p.add_argument(
        "layouts",
        nargs="*",
        help="Layout names to process (without x_/y_/z_ prefix). If omitted, process all layouts in recommendations.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output TSV path (default: overwrite input nodes_tsv).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for postprocessing (default: 42).",
    )
    p.add_argument(
        "--force-apply",
        action="store_true",
        help="Apply even when recommend_postprocess is false.",
    )
    return p.parse_args()


def _required_columns_present(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def _get_rec_row(rec_df: pd.DataFrame, layout_name: str):
    m = rec_df.loc[rec_df["layout"] == layout_name]
    if m.empty:
        return None
    return m.iloc[0]


def _rec_val(rec, key: str, default):
    if key not in rec.index:
        return default
    v = rec[key]
    if pd.isna(v):
        return default
    return v


def _to_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return True
    if s in {"0", "false", "no", "n", "f", ""}:
        return False
    return default


def _to_optional_float(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"", "none", "nan"}:
        return None
    x = float(v)
    if not np.isfinite(x):
        return None
    return x


def main():
    args = parse_args()
    nodes_path = Path(args.nodes_tsv)
    rec_path = Path(args.recommendations_tsv)
    out_path = Path(args.out) if args.out else nodes_path

    if not nodes_path.exists():
        raise FileNotFoundError(f"Input nodes/layout TSV not found: {nodes_path}")
    if not rec_path.exists():
        raise FileNotFoundError(f"Recommendations TSV not found: {rec_path}")

    df = pd.read_csv(nodes_path, sep="\t")
    rec_df = pd.read_csv(rec_path, sep="\t")
    if "layout" not in rec_df.columns:
        raise ValueError(f"{rec_path} must contain a 'layout' column.")

    layouts = args.layouts if args.layouts else rec_df["layout"].unique().tolist()

    applied = 0
    skipped = []

    pbar = tqdm(layouts, desc="Applying postprocess", unit=" layout")
    for layout_name in pbar:
        pbar.set_postfix_str(layout_name)
        xcol, ycol, zcol = f"x_{layout_name}", f"y_{layout_name}", f"z_{layout_name}"
        if not _required_columns_present(df, [xcol, ycol, zcol]):
            skipped.append((layout_name, "missing layout columns"))
            continue

        rec = _get_rec_row(rec_df, layout_name)
        if rec is None:
            skipped.append((layout_name, "no recommendation row"))
            continue

        if not args.force_apply and "recommend_postprocess" in rec.index:
            if not bool(rec["recommend_postprocess"]):
                skipped.append((layout_name, "recommend_postprocess=false"))
                continue

        for required_param in ["iterations", "repulsion_strength", "k_repel"]:
            if required_param not in rec.index:
                raise ValueError(
                    f"Recommendation row for {layout_name} missing '{required_param}'."
                )

        X = np.column_stack(
            [
                df[xcol].to_numpy(dtype=np.float64),
                df[ycol].to_numpy(dtype=np.float64),
                df[zcol].to_numpy(dtype=np.float64),
            ]
        )

        Xp = refine_3d_repulsion(
            X,
            iterations=int(rec["iterations"]),
            repulsion_strength=float(rec["repulsion_strength"]),
            step_size=float(_rec_val(rec, "step_size", 0.1)),
            min_dist=1e-3,
            use_knn=True,
            k_repel=int(rec["k_repel"]),
            anchor_strength=float(_rec_val(rec, "anchor_strength", 0.0)),
            max_step_factor=float(_rec_val(rec, "max_step_factor", 0.0)),
            dense_only_quantile=_to_optional_float(_rec_val(rec, "dense_only_quantile", None)),
            two_phase=_to_bool(_rec_val(rec, "two_phase", False)),
            phase_split=float(_rec_val(rec, "phase_split", 0.35)),
            phase1_strength_mult=float(_rec_val(rec, "phase1_strength_mult", 1.5)),
            phase2_strength_mult=float(_rec_val(rec, "phase2_strength_mult", 0.6)),
            early_stop_patience=int(_rec_val(rec, "early_stop_patience", 0)),
            early_stop_min_improve=float(_rec_val(rec, "early_stop_min_improve", 1e-4)),
            random_state=int(args.seed),
        )
        if not np.isfinite(Xp).all():
            skipped.append((layout_name, "non-finite output"))
            continue

        df[f"x_{layout_name}p"] = Xp[:, 0]
        df[f"y_{layout_name}p"] = Xp[:, 1]
        df[f"z_{layout_name}p"] = Xp[:, 2]
        applied += 1

    df.to_csv(out_path, sep="\t", index=False)

    print("Postprocess apply report")
    print("=" * 80)
    print(f"Input TSV: {nodes_path}")
    print(f"Recommendations: {rec_path}")
    print(f"Output TSV: {out_path}")
    print(f"Applied: {applied} layout(s)")
    if skipped:
        print("Skipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()

