# Input data

All input data are public NCBI GEO records. Nothing here is patient-identifying.

`download_geo_data.sh` downloads them into `work/original_pipeline/00_inputs/raw_data/` (or `$DFU_ORIGINAL_DIR/00_inputs/raw_data/`), unpacks the 54 single-cell count matrices of GSE165816 and checks every file against the checksums of the files used for the published analysis.

| File | Source | Used for |
|---|---|---|
| `GSE165816_RAW.tar` (54 `GSM*counts.csv.gz` matrices) | GEO supplementary file | single-cell atlas, raw-count pseudobulk |
| `GSE165816_series_matrix.txt.gz` | GEO series matrix | library annotation (tissue, healing group) |
| `GSE134431_raw_counts_GRCh38.p13_NCBI.tsv.gz`, `GSE134431_series_matrix.txt.gz` | NCBI-generated counts, GEO series matrix | bulk ulcer contrast |
| `GSE143735_raw_counts_GRCh38.p13_NCBI.tsv.gz`, `GSE143735_series_matrix.txt.gz` | NCBI-generated counts, GEO series matrix | external forearm cohort |
| `Human.GRCh38.p13.annot.tsv.gz` | NCBI gene annotation distributed with the GEO count matrices | gene ID to symbol mapping |

## Checksums

* `SHA256SUMS_geo.txt` covers the downloaded files.
* `SHA256SUMS_GSE165816_matrices.txt` covers the 54 per-library matrices.

GEO occasionally regenerates series-matrix and NCBI-count files. A checksum mismatch is therefore reported as a warning, not an error. The 54 single-cell matrices are the authors' original submissions and should always match. If an NCBI count file differs, the gene-level results may differ slightly from the reference results.

If `curl` cannot reach NCBI (for example behind a firewall), download the files listed above by hand from the GEO pages of GSE165816, GSE134431 and GSE143735 and place them in the folder named above. Then re-run the script, which skips files that already exist.

## Curated metadata (`metadata/`)

* `GSE165816_phenotype.csv` gives the GEO sample, library ID (`gid`), tissue site, condition and healing group of each of the 54 libraries. `tools/make_phenotype_table.py` rebuilds it from the series matrix and checks that the two are identical.
* `consensus_novel_panel.csv` lists the 6,043 genes nominated by at least one of the six discovery evidence sets, with a 0/1 column for each set. The 129-gene candidate panel (Supplementary Table 5) is the subset with `n_analyses` ≥ 3.
