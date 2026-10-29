#!/usr/bin/env python
"""
Khan 2 revision -- Step 01
Build the FULL GSE165816 single-cell atlas (no sub-sampling).

Addresses Reviewer 2, comment 4 (subsampling criterion / myeloid recovery) and
supplies the "full-scale single-cell analysis of all cells" the original
manuscript listed as its main future direction.

Reads all 54 GEO libraries, applies the manuscript's QC thresholds
(200-5000 genes/cell, <10% mitochondrial), and additionally records a
threshold-sensitivity sweep so the effect of the gene-count cut-offs on
myeloid recovery can be quantified rather than assumed.

Output: 02_results/01_atlas_raw.h5ad + per-sample and sweep QC tables.
"""
import os, sys, gzip, time, warnings
import numpy as np, pandas as pd, scipy.sparse as sp
import anndata as ad
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore")

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RAW  = f"{BASE}/00_inputs/raw_data/GSE165816"
RES  = f"{BASE}/02_results"
os.makedirs(RES, exist_ok=True)

PHENO = pd.read_csv(f"{BASE}/00_inputs/raw_data/GSE165816_phenotype.csv")
PHENO["file"] = None

# QC thresholds as published
MIN_GENES, MAX_GENES, MAX_MT = 200, 5000, 10.0
# sweep grid for the reviewer-requested sensitivity analysis
SWEEP = [(100, 8000), (200, 5000), (200, 8000), (300, 5000), (500, 6000)]


def load_one(args):
    """Read one GEO CSV (genes x cells) -> (AnnData cells x genes, qc rows)."""
    gsm, gid, tissue, condition, group, path = args
    t0 = time.time()
    df = pd.read_csv(path, index_col=0)          # genes x cells
    genes = df.index.astype(str).values
    cells = df.columns.astype(str).values
    X = sp.csr_matrix(df.values.T.astype(np.float32))   # cells x genes
    del df

    mt = np.char.startswith(np.char.upper(genes.astype(str)), "MT-")
    nCount   = np.asarray(X.sum(1)).ravel()
    nFeature = np.asarray((X > 0).sum(1)).ravel()
    mt_sum   = np.asarray(X[:, mt].sum(1)).ravel() if mt.sum() else np.zeros(X.shape[0])
    pctMT    = np.divide(mt_sum, nCount, out=np.zeros_like(nCount), where=nCount > 0) * 100

    obs = pd.DataFrame(
        dict(gsm=gsm, gid=gid, tissue=tissue, condition=condition, group=group,
             nCount=nCount, nFeature=nFeature, pctMT=pctMT),
        index=pd.Index([f"{gid}_{b}" for b in cells], name="cell"))
    A = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes, name="gene")))

    # threshold sweep -- how many cells survive under each rule
    sweep = [dict(gsm=gsm, gid=gid, tissue=tissue, group=group,
                  min_genes=lo, max_genes=hi,
                  n_pass=int(((nFeature >= lo) & (nFeature <= hi) & (pctMT < MAX_MT)).sum()))
             for lo, hi in SWEEP]

    keep = (nFeature >= MIN_GENES) & (nFeature <= MAX_GENES) & (pctMT < MAX_MT)
    qc = dict(gsm=gsm, gid=gid, tissue=tissue, condition=condition, group=group,
              n_cells_raw=int(A.n_obs), n_cells_pass=int(keep.sum()),
              n_genes_detected=int((np.asarray((X > 0).sum(0)).ravel() > 0).sum()),
              median_nFeature=float(np.median(nFeature)),
              median_nCount=float(np.median(nCount)),
              median_pctMT=float(np.median(pctMT)),
              secs=round(time.time() - t0, 1))
    A = A[keep].copy()
    print(f"  {gid:5s} {tissue:8s} {group:11s} {qc['n_cells_raw']:6d} -> {qc['n_cells_pass']:6d} "
          f"({qc['secs']}s)", flush=True)
    return A, qc, sweep


def main():
    files = {f.split("_")[0]: os.path.join(RAW, f) for f in os.listdir(RAW) if f.endswith(".csv.gz")}
    jobs = []
    for _, r in PHENO.iterrows():
        p = files.get(r.gsm)
        if p is None:
            print("MISSING", r.gsm); continue
        jobs.append((r.gsm, r.gid, r.tissue, r.condition, r.group, p))
    print(f"Loading {len(jobs)} libraries with 10 workers ...", flush=True)

    adatas, qcs, sweeps = [], [], []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for A, qc, sw in ex.map(load_one, jobs):
            adatas.append(A); qcs.append(qc); sweeps.extend(sw)

    qc_df = pd.DataFrame(qcs).sort_values("gid")
    qc_df.to_csv(f"{RES}/01_persample_QC.csv", index=False)
    pd.DataFrame(sweeps).to_csv(f"{RES}/01_QC_threshold_sweep.csv", index=False)

    print("\nConcatenating ...", flush=True)
    A = ad.concat(adatas, join="outer", fill_value=0, index_unique=None)
    A.var_names_make_unique()
    A.X = sp.csr_matrix(A.X)
    A.layers["counts"] = A.X.copy()
    A.write_h5ad(f"{RES}/01_atlas_raw.h5ad", compression="gzip")

    tot_raw, tot_pass = int(qc_df.n_cells_raw.sum()), int(qc_df.n_cells_pass.sum())
    skin = qc_df[qc_df.tissue != "PBMC"]
    print(f"\n=== FULL ATLAS BUILT ===")
    print(f"libraries          : {len(qc_df)} (skin {len(skin)}, PBMC {len(qc_df)-len(skin)})")
    print(f"cells raw          : {tot_raw:,}")
    print(f"cells passing QC   : {tot_pass:,}")
    print(f"skin cells post-QC : {int(skin.n_cells_pass.sum()):,}")
    print(f"genes              : {A.n_vars:,}")
    print(f"object             : {RES}/01_atlas_raw.h5ad")

if __name__ == "__main__":
    main()
