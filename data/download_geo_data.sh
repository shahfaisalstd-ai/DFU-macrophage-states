#!/usr/bin/env bash
# Download the public GEO data used by both pipelines into the layout the code expects:
#   $DFU_ORIGINAL_DIR/00_inputs/raw_data/            (default: <repo>/work/original_pipeline)
#   $DFU_ORIGINAL_DIR/00_inputs/raw_data/GSE165816/  54 per-library count matrices (from GSE165816_RAW.tar)
# and check them against the checksums of the files used for the published analysis.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DFU_ORIGINAL_DIR:-$REPO/work/original_pipeline}/00_inputs/raw_data"
mkdir -p "$DEST"; cd "$DEST"
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series
CNT="https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&format=file"
get () { [ -s "$2" ] && { echo "exists: $2"; return; }; echo "downloading $2"; curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }

# single-cell RNA-seq, Theocharidis et al. 2022 (54 libraries)
get "$GEO/GSE165nnn/GSE165816/suppl/GSE165816_RAW.tar"                     GSE165816_RAW.tar
get "$GEO/GSE165nnn/GSE165816/matrix/GSE165816_series_matrix.txt.gz"       GSE165816_series_matrix.txt.gz
# bulk RNA-seq: NCBI-generated raw counts (GRCh38.p13) and series matrices
get "$CNT&acc=GSE134431&file=GSE134431_raw_counts_GRCh38.p13_NCBI.tsv.gz"  GSE134431_raw_counts_GRCh38.p13_NCBI.tsv.gz
get "$GEO/GSE134nnn/GSE134431/matrix/GSE134431_series_matrix.txt.gz"       GSE134431_series_matrix.txt.gz
get "$CNT&acc=GSE143735&file=GSE143735_raw_counts_GRCh38.p13_NCBI.tsv.gz"  GSE143735_raw_counts_GRCh38.p13_NCBI.tsv.gz
get "$GEO/GSE143nnn/GSE143735/matrix/GSE143735_series_matrix.txt.gz"       GSE143735_series_matrix.txt.gz
get "$CNT&file=Human.GRCh38.p13.annot.tsv.gz"                              Human.GRCh38.p13.annot.tsv.gz

# curated library annotation (gid, tissue, group) taken from the GEO sample characteristics
cp "$REPO/data/metadata/GSE165816_phenotype.csv" .

mkdir -p GSE165816 && tar -xf GSE165816_RAW.tar -C GSE165816
echo "checking checksums"
grep -v '^#' "$REPO/data/SHA256SUMS_geo.txt" | sha256sum -c - || echo "WARNING: a GEO file differs from the one used for the published analysis (see data/README.md)"
( cd GSE165816 && grep -v '^#' "$REPO/data/SHA256SUMS_GSE165816_matrices.txt" | sha256sum -c --quiet - && echo "GSE165816: all 54 matrices match" ) \
  || echo "WARNING: GSE165816 matrices differ from the published analysis"
