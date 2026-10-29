"""Shared loaders for the bulk cohorts used in the Khan 2 revision."""
import os, gzip, re
import numpy as np, pandas as pd

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RAWD = f"{BASE}/00_inputs/raw_data"


def gene_annotation():
    a = pd.read_csv(f"{RAWD}/Human.GRCh38.p13.annot.tsv.gz", sep="\t",
                    usecols=["GeneID", "Symbol", "GeneType"], low_memory=False)
    return a.dropna(subset=["Symbol"]).drop_duplicates("GeneID").set_index("GeneID")


def _series_meta(path):
    """Parse a GEO series matrix into a sample x characteristic frame."""
    rows = {}
    with gzip.open(path, "rt", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("!Sample_"):
                continue
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            key, vals = parts[0], parts[1:]
            rows.setdefault(key, []).append(vals)
    gsm = rows["!Sample_geo_accession"][0]
    out = pd.DataFrame(index=gsm)
    out["title"] = rows["!Sample_title"][0]
    for i, vals in enumerate(rows.get("!Sample_characteristics_ch1", [])):
        tags = [v.split(":", 1)[0].strip() if ":" in v else f"char{i}" for v in vals if v]
        tag = tags[0] if tags else f"char{i}"
        out[tag] = [v.split(":", 1)[1].strip() if ":" in v else np.nan for v in vals]
    return out


def load_counts(prefix, annot=None):
    """Return (counts_symbol_level DataFrame genes x samples, metadata)."""
    cf = f"{RAWD}/{prefix}_raw_counts_GRCh38.p13_NCBI.tsv.gz"
    C = pd.read_csv(cf, sep="\t", index_col=0)
    C.index.name = "GeneID"
    ann = annot if annot is not None else gene_annotation()
    C = C.join(ann[["Symbol", "GeneType"]], how="inner")
    C = C[C.GeneType == "protein-coding"].drop(columns=["GeneType"])
    C = C.groupby("Symbol").sum()                     # collapse to symbols
    meta = _series_meta(f"{RAWD}/{prefix}_series_matrix.txt.gz").loc[C.columns]
    return C, meta


def dfu_groups(prefix, meta):
    """Normalise the healer / non-healer labels of each cohort."""
    if prefix == "GSE134431":
        g = meta["healer_or_nonhealer"].fillna("Skin")
        g = g.replace({"Healer": "Healing", "Nonhealer": "Nonhealing"})
        g[meta["ulcer_or_skin"] == "Skin"] = "DiabeticSkin"
    else:  # GSE143735
        g = meta["group"].map({
            "Diabetes - No Ulcer": "DiabeticSkin",
            "Diabetes - Ulcer - Healer": "Healing",
            "Diabetes - Ulcer - Non Healer": "Nonhealing"})
    return g.rename("group")


def cpm_log(C, prior=1.0):
    lib = C.sum(0)
    return np.log2(C.div(lib, axis=1) * 1e6 + prior)
