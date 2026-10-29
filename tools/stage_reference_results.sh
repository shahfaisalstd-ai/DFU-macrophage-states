#!/usr/bin/env bash
# Copy the stored original-pipeline outputs (results/original_pipeline) into the
# working folder, so that the revision analyses (run_revision.sh) can be run
# without re-running the single-cell pipeline. The two AnnData files
# 03_myeloid.h5ad and 12_macrophage_trajectory.h5ad are too large for GitHub;
# produce them with run_original_pipeline.sh (steps 01-03 and 12) or obtain them
# from the authors, and place them in work/original_pipeline/02_results/.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DFU_ORIGINAL_DIR:-$REPO/work/original_pipeline}"
mkdir -p "$DEST/02_results" "$DEST/04_raw_data_by_figure"
for sub in 02_results 04_raw_data_by_figure; do
  for f in "$REPO"/results/original_pipeline/$sub/*; do
    name="$(basename "$f")"
    case "$name" in
      *.csv.gz) out="${name%.gz}"; [ -e "$DEST/$sub/$out" ] || gunzip -c "$f" > "$DEST/$sub/$out" ;;   # stored compressed (GitHub size limit)
      *)        [ -e "$DEST/$sub/$name" ] || cp "$f" "$DEST/$sub/" ;;                               # never overwrite your own outputs
    esac
  done
done
for f in 03_myeloid.h5ad 12_macrophage_trajectory.h5ad; do
  [ -s "$DEST/02_results/$f" ] || echo "MISSING: $DEST/02_results/$f (needed by the revision scripts R2_00, R2_06 and R2_10)"
done
echo "staged into $DEST"
