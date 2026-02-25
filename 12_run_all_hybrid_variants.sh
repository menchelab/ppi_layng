#!/usr/bin/env bash
set -euo pipefail

# Run 5 variants for each hybrid metric (12.1 ... 12.5)
# and append all generated columns into one output TSV.
#
# Usage:
#   bash 12_run_all_hybrid_variants.sh [layout_in.tsv] [layout_out.tsv]
#
# Defaults:
#   layout_in.tsv  -> output/layout_decompression.tsv
#   layout_out.tsv -> output/layout_decompression.tsv (in-place)

LAYOUT_IN="${1:-output/layout_decompression.tsv}"
LAYOUT_OUT="${2:-$LAYOUT_IN}"

echo "Hybrid sweep input : ${LAYOUT_IN}"
echo "Hybrid sweep output: ${LAYOUT_OUT}"
echo

run_step() {
  echo ">>> $*"
  "$@"
  echo
}

if [[ ! -f "${LAYOUT_IN}" ]]; then
  echo "ERROR: layout input not found: ${LAYOUT_IN}" >&2
  exit 1
fi

# 12.1 weighted concat (5 weights)
for w in 0.05 0.10 0.20 0.35 0.50; do
  run_step python 12.1_umap_weighted_concat.py "${LAYOUT_IN}" "${LAYOUT_OUT}" --weight "${w}" --svd-dim 128
done

# 12.2 late fusion blend (5 weights)
for w in 0.05 0.10 0.20 0.35 0.50; do
  run_step python 12.2_umap_late_fusion.py "${LAYOUT_IN}" "${LAYOUT_OUT}" --weight "${w}" --svd-dim 128
done

# 12.3 distance fusion (5 weights) -- heavy
for w in 0.05 0.10 0.20 0.35 0.50; do
  run_step python 12.3_umap_distance_fusion.py "${LAYOUT_IN}" "${LAYOUT_OUT}" --weight "${w}" --svd-dim 96
done

# 12.4 graph diffusion (5 beta/step variants)
run_step python 12.4_umap_graph_diffusion.py "${LAYOUT_IN}" "${LAYOUT_OUT}" --beta 0.95 --steps 1
run_step python 12.4_umap_graph_diffusion.py "${LAYOUT_IN}" "${LAYOUT_OUT}" --beta 0.90 --steps 1
run_step python 12.4_umap_graph_diffusion.py "${LAYOUT_IN}" "${LAYOUT_OUT}" --beta 0.85 --steps 2
run_step python 12.4_umap_graph_diffusion.py "${LAYOUT_IN}" "${LAYOUT_OUT}" --beta 0.80 --steps 2
run_step python 12.4_umap_graph_diffusion.py "${LAYOUT_IN}" "${LAYOUT_OUT}" --beta 0.75 --steps 3

# 12.5 multi-view kNN union (5 weights)
for w in 0.10 0.20 0.30 0.40 0.50; do
  run_step python 12.5_umap_multiview_knn_union.py "${LAYOUT_IN}" "${LAYOUT_OUT}" --weight "${w}" --svd-dim 128 --knn 30 --graph-svd-dim 64
done

echo "Done. All hybrid variants appended to: ${LAYOUT_OUT}"

