#!/usr/bin/env bash
# Original pipeline (steps 01-19): single-cell atlas, annotation, trajectory,
# communication, bulk analyses, machine learning and the original figures.
# Environment: environment/original_pipeline_environment.yml (Python 3.10, scanpy 1.11.0).
# Inputs: data/download_geo_data.sh puts the GEO files in work/original_pipeline/00_inputs/raw_data.
# Step 14 queries the Enrichr web service through GSEApy (internet access needed;
# results depend on the Enrichr library versions served at run time).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DFU_ORIGINAL_DIR="${DFU_ORIGINAL_DIR:-$REPO/work/original_pipeline}"
export PYTHONHASHSEED=0
mkdir -p "$DFU_ORIGINAL_DIR"/{02_results,03_figures,04_raw_data_by_figure,logs}
cd "$REPO/original_pipeline"
run () { local s=$1; echo "[$(date +%H:%M:%S)] $s"; "${PYTHON:-python3}" "$s.py" > "$DFU_ORIGINAL_DIR/logs/$s.log" 2>&1 || { echo "FAILED: $s"; exit 1; }; }
# order matters: 17 -> 17b -> 17c (17b adds to 17_summary.json, 17c rewrites 17_bulk_feature_importance.csv)
for s in 01_build_atlas 02_integrate 03_annotate_myeloid 04_pseudobulk 05_deconvolution 06_bulk_validation \
         07_machine_learning 07b_ml_robustness 08a_figures 08b_figures 12_trajectory 13_communication \
         14_bulk_landscape 15_networks 16_confounders 16b_bulk_confounders 17_bulk_ml 17b_external_transfer \
         17c_feature_importance 18_perturbation 19_figures; do run $s; done
echo "done: results in $DFU_ORIGINAL_DIR/02_results"
