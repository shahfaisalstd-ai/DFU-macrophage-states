# Macrophage states in healing and non-healing diabetic foot ulcers: analysis code

This repository holds the complete analysis code for the article

> **Integrative single-cell and bulk transcriptomic analysis reveals a macrophage state imbalance distinguishing non-healing diabetic foot ulcers**
> Piniel Alphayo Kambey\*, Shah Faisal\*, Muhammad Adeel Ejaz, Abdul Malik, Yin-Xiong Li.
> *World Journal of Diabetes*, manuscript 124795 (in revision). \*Co-first authors. Corresponding author: Piniel Alphayo Kambey (pinielalphayo@yahoo.com).

It contains the code, the software environments, the reference results and the source data for every figure. The code runs from the public GEO accessions to every result table and figure reported in the manuscript and its Supplementary Tables 1-14.

## Quick start

| I want to... | Do this |
|---|---|
| look at the numbers behind a figure | open `results/figures/Figure_source_data/` (one CSV per panel) |
| look at a result table | open `results/revision/` (revision analyses) or `results/original_pipeline/02_results/` |
| re-run the analyses that the manuscript reports | follow **Running the analysis** below, then run `python tools/verify_results.py` |
| re-run everything from raw data | run all four steps under **Running the analysis** |

## Data

All data are public (NCBI GEO):

| Accession | Content | Role in the study |
|---|---|---|
| [GSE165816](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165816) | scRNA-seq, 54 libraries (44 skin, 10 PBMC), Theocharidis *et al.* 2022 | single-cell atlas; healing vs non-healing contrast (19 skin libraries) |
| [GSE134431](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134431) | bulk RNA-seq of foot ulcers, Sawaya *et al.* 2020 | bulk contrast; panel construction |
| [GSE143735](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE143735) | bulk RNA-seq of forearm skin, Theocharidis *et al.* 2020 | external transportability test |

`data/download_geo_data.sh` downloads the data and checks them against the checksums of the files used for the published analysis (`data/SHA256SUMS_*.txt`). It also unpacks the 54 per-library count matrices.

Two small curated inputs are under `data/metadata/`:

* `GSE165816_phenotype.csv` holds the library ID, tissue and healing group of each GEO sample. `tools/make_phenotype_table.py` rebuilds it from the GEO series matrix and checks that the two are identical.
* `consensus_novel_panel.csv` is the 129-gene candidate panel. It lists genes recurring in at least three of six discovery evidence sets of the earlier analysis (`n_analyses` ≥ 3).

## Repository layout

```
original_pipeline/   steps 01-19: atlas, integration, annotation, trajectory, communication,
                     bulk DE, deconvolution, networks, machine learning, original figures
revision/            R2_00-R2_14: raw-count pseudobulk, sample structure, covariate models,
                     confidence intervals, networks/panel, Figures 1, 2, 4-17
vendor/              pyfive (pure-Python HDF5 reader) and PyDESeq2 0.5.2, with patches and licences
environment/         software environments of the two pipelines
data/                download script, checksums, curated metadata (see data/README.md)
results/             reference outputs (original_pipeline/, revision/) and figures with source data
                     (03_myeloid_subcluster_markers.csv is stored gzip-compressed to stay under GitHub's 25 MB web-upload limit)
tools/               verification and helper scripts
run_original_pipeline.sh, run_revision.sh
```

## Running the analysis

There are two environments because the analysis was run in two stages:

| Stage | Environment | Main packages |
|---|---|---|
| `original_pipeline/` | `environment/original_pipeline_environment.yml` (Python 3.10.20) | scanpy 1.11.0, anndata 0.11.4, harmonypy 2.0.0, leidenalg 0.10.2, PyDESeq2 0.5.2, scikit-learn 1.5.2, statsmodels 0.14.6 |
| `revision/` | `environment/revision_requirements.txt` (Python 3.13) | numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2 |

The revision stage needs neither R, h5py, anndata nor scanpy. The `.h5ad` files are read with the vendored pyfive. Differential expression runs the vendored PyDESeq2 0.5.2, the same version as the original stage, through `revision/deseq_engine.py`.

```bash
# 0. get the code
git clone https://github.com/shahfaisalstd-ai/DFU-macrophage-states.git
cd DFU-macrophage-states

# 1. data
bash data/download_geo_data.sh

# 2. original pipeline (single-cell atlas etc.; needs the Python 3.10 environment)
conda env create -f environment/original_pipeline_environment.yml
conda activate dfu-original
bash run_original_pipeline.sh                 # writes work/original_pipeline/{02_results,03_figures,04_raw_data_by_figure}

# 3. revision analyses (Python 3.13 environment)
python3.13 -m venv .venv-revision && source .venv-revision/bin/activate
pip install -r environment/revision_requirements.txt
bash run_revision.sh                          # writes work/revision_results (tables, figures/, figures/Figure_source_data/)

# 4. compare the re-run with the reference results
python tools/verify_results.py work/revision_results
```

**Running only the revision stage.** `tools/stage_reference_results.sh` copies the stored original-pipeline outputs (`results/original_pipeline`) into the working folder.

The revision scripts also read two AnnData files, `03_myeloid.h5ad` (100 MB) and `12_macrophage_trajectory.h5ad` (70 MB). They exceed GitHub's file-size limit, so they are not in this repository. Produce them with `run_original_pipeline.sh` (steps 01–03 and 12), or ask the authors for them. Then place them in `work/original_pipeline/02_results/`.

All paths are relative to the repository. You can redirect them with environment variables: `DFU_ORIGINAL_DIR`, `R2IN`, `R2OUT`, `H5AD_DIR`, `PANEL_CSV` and `R2FIG` (see `revision/r2_common.py`).

Runtime for the revision stage is about 40 minutes on 2 CPU cores with 8 GB RAM.

## Where each result comes from

| Manuscript item | Script(s) | Output |
|---|---|---|
| Figure 1 (workflow), Figure 2 (QC, cell recovery) | `R2_10_figures.py` (data: `01_build_atlas.py`, `R2_01`, `R2_04`) | `figures/Figure1_*`, `Figure2_*` |
| Figure 3 (integrated atlas) | `02_integrate.py`, `03_annotate_myeloid.py`, `08a_figures.py` | `results/figures/Figure3_integrated_atlas.png` |
| Figures 4–5 (myeloid states, programs) | `03_annotate_myeloid.py`, `R2_05`, `R2_10` | `Figure4_*`, `Figure5_*` |
| Figure 6, Suppl. Tables 12–14 (library-level models, pseudobulk DE) | `R2_02`, `R2_05`, `R2_10` | `R2_05_*` |
| Figure 7 (pseudotime) | `12_trajectory.py`, `R2_05` | `12_*`, `Figure7_*` |
| Figure 8 (ligand–receptor) | `13_communication.py`, `R2_09` | `13_*`, `R2_09_*` |
| Figures 9–12 (bulk, deconvolution, networks, bulk ML) | `05`, `06`, `14`, `15`, `17*`, `R2_07`, `R2_08` | `R2_07_*`, `R2_08_*` |
| Figure 13 (macrophage classifier) | `07_machine_learning.py`, `07b`, `R2_06` | `R2_06_*` |
| Figure 14 (external replication) | `06_bulk_validation.py`, `R2_07` | `R2_07_directional_replication_CIs.csv` |
| Figure 15, Suppl. Tables 1–3 (sex, sample structure) | `R2_01`, `R2_02b`, `R2_04`, `R2_05` | `R2_04_*` |
| Figures 16–17 (prioritized genes, perturbation) | `18_perturbation.py`, `R2_09`, `R2_10` | `18_*`, `Figure16_*`, `Figure17_*` |
| Supplementary Table 10b (DE engine check) | `R2_03_validate_deseq_engine.py` | `R2_03_*` |

Figure 3 is drawn by the original pipeline. Every other figure, and the numbers behind each panel, is in `results/figures/` and `results/figures/Figure_source_data/`.

## Verification

* **Reproducibility.** Running `run_revision.sh` from the GEO matrices and the stored original-pipeline outputs reproduces every table in `results/revision` byte for byte, and every figure pixel for pixel. This was confirmed on a fresh run in the revision environment.
  * Steps R2_00 to R2_02b were re-run from the 54 GEO count matrices.
  * Their per-library cell counts match the original pipeline for all 54 libraries: 174,962 cells profiled, 123,884 passing QC and 97,677 in the 44 skin libraries.
* **Differential-expression engine.** `R2_03` runs PyDESeq2 0.5.2 in the revision environment on the bulk input of the original pipeline. It reproduces the PyDESeq2 0.5.2 results of the original environment:
  * median ratio of standard errors 1.00;
  * Spearman correlation of P values > 0.9999;
  * identical sets of 50 (GSE134431) and 9 (GSE143735) genes at FDR < 0.05.
* **Determinism.** Seeds are fixed (123), and the revision stage runs with fixed thread settings (`run_revision.sh`).

## Notes and known limitations

These points are stated in the manuscript or follow from the design. They are listed here so that users can judge the results.

* **Original-pipeline pseudobulk.** In the original pipeline, step 04 summed log-normalized values (the stored AnnData objects carry no raw-count layer). Its pseudobulk outputs (`04_macrophage_pseudobulk_counts.csv` and dependent steps 07 pseudobulk classifier, 14, 15 and 16) are superseded by the raw-count pseudobulk of `R2_02`/`R2_05`/`R2_06`/`R2_08`. The manuscript reports the raw-count versions.
* **Ligand–receptor table.** `lr_database.tsv` is a curated list of 205 ligand–receptor pairs. The PTHLH entry is misspelled, so 204 pairs are tested.
* **Enrichr results.** Step 14 queries the Enrichr web service (GO Biological Process 2021, KEGG 2021). Results depend on the library versions the service provides at run time.
* **Thread counts.** Single-cell steps 01–03 and 12 fix thread counts. Exact Leiden clusters can depend on the scanpy, igraph and BLAS builds, so a re-run in a different environment may relabel a few cells.
* **Sample structure.** GEO releases no subject identifiers. The 19 healing and non-healing libraries come from 11 patients. Donor groups in `R2_04` are inferred from germline-variable expression markers, are conservative, and forearm libraries are attached to the nearest same-group foot donor. All of this is reported as inferred.
* **Panel classifier.** The 129-gene panel was assembled partly from GSE134431, so the internal bulk-classifier AUC in GSE134431 is optimistic. The external test in GSE143735 is the relevant estimate.
* **Gene-level confidence intervals.** Confidence intervals for gene-level concordance (Spearman, binomial) treat genes as independent and are therefore narrow.

## Contact

Questions about the code or requests for the two `.h5ad` files: open an issue in this repository, or contact Piniel Alphayo Kambey (pinielalphayo@yahoo.com).

## Citation and licence

Please cite the article (see `CITATION.cff`; GitHub shows a "Cite this repository" button). Code: MIT licence (`LICENSE`). Vendored third-party code keeps its own licence: pyfive (BSD 3-Clause) and PyDESeq2 (MIT).
