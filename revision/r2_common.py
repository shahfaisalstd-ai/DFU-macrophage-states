"""Shared paths and loaders for the revision analyses (R2_*).

All locations are relative to the repository root and can be overridden with
environment variables:
  DFU_ORIGINAL_DIR / KHAN2  working folder of the original pipeline (00_inputs/raw_data, 02_results, 04_raw_data_by_figure)
  R2IN                      outputs of R2_00-R2_02b (cell lists, raw-count pseudobulk)
  R2OUT                     outputs of R2_03 onwards
  R2AUX                     cell metadata exported by R2_00 (default R2IN/cell_metadata)
  H5AD_DIR                  03_myeloid.h5ad and 12_macrophage_trajectory.h5ad (default: original 02_results)
  PANEL_CSV                 129-gene candidate panel (default data/metadata/consensus_novel_panel.csv)
"""
import os, sys, gzip
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
import numpy as np, pandas as pd
KHAN = os.environ.get("KHAN2", os.environ.get("DFU_ORIGINAL_DIR", os.path.join(REPO, "work", "original_pipeline")))
RES = f"{KHAN}/02_results"
RAWD = f"{KHAN}/00_inputs/raw_data"
R2IN = os.environ.get("R2IN", os.path.join(REPO, "work", "revision_results"))
OUT = os.environ.get("R2OUT", R2IN)
PANEL_CSV = os.environ.get("PANEL_CSV", os.path.join(REPO, "data", "metadata", "consensus_novel_panel.csv"))
AUX = os.environ.get("R2AUX", os.path.join(R2IN, "cell_metadata"))   # cell metadata exported from the .h5ad files, atlas_genes.csv
H5AD_DIR = os.environ.get("H5AD_DIR", os.path.join(KHAN, "02_results"))  # 03_myeloid.h5ad, 12_macrophage_trajectory.h5ad
SEED = 123
os.makedirs(OUT, exist_ok=True); os.makedirs(AUX, exist_ok=True)

def gene_annotation():
    a = pd.read_csv(f"{RAWD}/Human.GRCh38.p13.annot.tsv.gz", sep="\t",
                    usecols=["GeneID", "Symbol", "GeneType"], low_memory=False)
    return a.dropna(subset=["Symbol"]).drop_duplicates("GeneID").set_index("GeneID")

def series_meta(path):
    rows = {}
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("!Sample_"): continue
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            rows.setdefault(parts[0], []).append(parts[1:])
    gsm = rows["!Sample_geo_accession"][0]
    out = pd.DataFrame(index=gsm)
    out["title"] = rows["!Sample_title"][0]
    for i, vals in enumerate(rows.get("!Sample_characteristics_ch1", [])):
        tags = [v.split(":", 1)[0].strip() for v in vals if v and ":" in v]
        tag = tags[0] if tags else f"char{i}"
        out[tag] = [v.split(":", 1)[1].strip() if ":" in v else np.nan for v in vals]
    for k in ["!Sample_source_name_ch1", "!Sample_instrument_model", "!Sample_platform_id"]:
        if k in rows: out[k.replace("!Sample_", "")] = rows[k][0]
    return out

def load_counts(prefix, ann=None):
    """Same as original_pipeline/khan_bulk.load_counts: NCBI raw counts, protein-coding, symbol-collapsed."""
    C = pd.read_csv(f"{RAWD}/{prefix}_raw_counts_GRCh38.p13_NCBI.tsv.gz", sep="\t", index_col=0)
    C.index.name = "GeneID"
    ann = ann if ann is not None else gene_annotation()
    C = C.join(ann[["Symbol", "GeneType"]], how="inner")
    C = C[C.GeneType == "protein-coding"].drop(columns=["GeneType"]).groupby("Symbol").sum()
    meta = series_meta(f"{RAWD}/{prefix}_series_matrix.txt.gz").loc[C.columns]
    return C, meta

def dfu_groups(prefix, meta):
    if prefix == "GSE134431":
        g = meta["healer_or_nonhealer"].fillna("Skin").replace({"Healer": "Healing", "Nonhealer": "Nonhealing"})
        g[meta["ulcer_or_skin"] == "Skin"] = "DiabeticSkin"
    else:
        g = meta["group"].map({"Diabetes - No Ulcer": "DiabeticSkin", "Diabetes - Ulcer - Healer": "Healing",
                               "Diabetes - Ulcer - Non Healer": "Nonhealing"})
    return g.rename("group")

def cpm_log(C, prior=1.0):
    return np.log2(C.div(C.sum(0), axis=1) * 1e6 + prior)

LABEL_FIX = {"unadjusted (previous analysis)": "group only (unadjusted)", " (previous revision)": "",
             "(previous revision)": "", "previous analysis": "group-only model"}
def neutral_labels(df):
    """presentation copy of a result table with history wording removed from labels"""
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("previous_")])
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            s = df[c].astype(str)
            for a, b in LABEL_FIX.items(): s = s.str.replace(a, b, regex=False)
            df[c] = s.where(df[c].notna(), None)
    return df
