#!/usr/bin/env bash
# Revision analyses R2_00-R2_14: from the GEO matrices and the original-pipeline
# outputs to every result table and Figures 1, 2 and 4-17 (with source data).
# Environment: environment/revision_requirements.txt (Python 3.13).
# Prerequisites (see README.md):
#   work/original_pipeline/00_inputs/raw_data/            GEO files (data/download_geo_data.sh)
#   work/original_pipeline/00_inputs/raw_data/GSE165816/  the 54 per-library count matrices
#   work/original_pipeline/02_results/                    original-pipeline outputs, incl.
#                                                         03_myeloid.h5ad and 12_macrophage_trajectory.h5ad
#   work/original_pipeline/04_raw_data_by_figure/         original-pipeline figure data (steps 08a, 08b, 19)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DFU_ORIGINAL_DIR="${DFU_ORIGINAL_DIR:-$REPO/work/original_pipeline}"
export R2IN="${R2IN:-$REPO/work/revision_results}"
export R2OUT="${R2OUT:-$R2IN}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GEO_SC="$DFU_ORIGINAL_DIR/00_inputs/raw_data/GSE165816"
LOG="$R2OUT/logs"; mkdir -p "$R2IN" "$R2OUT" "$LOG"
cd "$REPO/revision"
run () { local name=$1; shift; echo "[$(date +%H:%M:%S)] $name"; "${PYTHON:-python3}" "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED: $name (see $LOG/$name.log)"; exit 1; }; }

run R2_00 R2_00_export_h5ad_metadata.py                         # cell metadata + myeloid cell list from the .h5ad files
run R2_01 R2_01_library_pseudobulk.py      "$GEO_SC" "$R2IN"    # per-cell QC, whole-library pseudobulk (54 libraries)
run R2_02 R2_02_myeloid_raw_pseudobulk.py  "$GEO_SC" "$R2IN"    # raw-count macrophage / state pseudobulk
# per-cell XIST / Y check: the two libraries with mixed whole-library signal (G2, G13)
# and four single-sex reference libraries (G33, G6, G49 male; G9 female)
run R2_02b R2_02b_percell_sex_check.py     "$GEO_SC" "$R2IN" G2 G13 G33 G9 G6 G49
run R2_03 R2_03_validate_deseq_engine.py   # PyDESeq2 0.5.2 here vs PyDESeq2 0.5.2 of the original environment
run R2_04 R2_04_sample_tables.py           # sample tables, cell reconciliation, sex, donor structure
run R2_05 R2_05_covariate_models.py        # composition, programme, pseudotime and pseudobulk DE models
run R2_06 R2_06_classifier_CIs.py          # per-cell / library / donor classifier AUCs with CIs
run R2_07 R2_07_bulk_CIs.py                # bulk classifier CIs, external transfer, replication, covariate-adjusted bulk DE
run R2_08 R2_08_networks_and_panel.py      # co-expression networks, 129-gene panel table
run R2_09 R2_09_effect_CIs.py              # effect sizes with CIs
run R2_10 R2_10_figures.py                 # Figures 1, 2, 4-17 + Figure_source_data
run R2_14 R2_14_crosscheck_figure_data.py  # figure data vs original-pipeline figure data
echo "done: results in $R2OUT"
