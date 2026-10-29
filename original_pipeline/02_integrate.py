#!/usr/bin/env python
"""
Khan 2 revision -- Step 02
Harmony integration + Leiden clustering of the FULL skin atlas.

The published atlas used 13,200 cells (300 per library) because the original
analysis ran in a 2.8 GB sandbox. Here all 97,677 QC-passing skin cells from
the same 44 libraries are integrated, so cluster structure and myeloid
recovery no longer depend on a sub-sampling step (Reviewer 2, comment 4).

Pipeline follows the published Methods: normalise to 1e4, log1p, 3000 HVGs,
scale with regression of mitochondrial fraction and cell-cycle scores,
PCA, Harmony over library, Leiden at several resolutions.
"""
import os, time, warnings
import numpy as np, pandas as pd, scanpy as sc
warnings.filterwarnings("ignore")

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES  = f"{BASE}/02_results"
sc.settings.n_jobs = 16
sc.settings.verbosity = 2
SEED = 123
np.random.seed(SEED)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

def harmony(A, key, basis="X_pca", out="X_pca_harmony", n_pcs=None, iters=20):
    """Harmony integration. harmonypy 2.0 returns Z_corr as (d, N); scanpy's
    wrapper mis-handles that, so we call the library directly and orient the
    corrected embedding ourselves."""
    import harmonypy
    Z = A.obsm[basis]
    if n_pcs: Z = Z[:, :n_pcs]
    ho = harmonypy.run_harmony(np.asarray(Z, dtype=np.float64), A.obs, [key],
                               max_iter_harmony=iters, random_state=SEED)
    Zc = np.asarray(ho.Z_corr)
    if Zc.shape[0] != A.n_obs:
        Zc = Zc.T
    assert Zc.shape[0] == A.n_obs, f"harmony shape {Zc.shape} vs {A.n_obs} cells"
    A.obsm[out] = np.ascontiguousarray(Zc, dtype=np.float32)
    return A


A = sc.read_h5ad(f"{RES}/01_atlas_raw.h5ad")
log(f"loaded {A.shape}")

# skin only -- PBMC libraries are not part of the tissue atlas
A = A[A.obs.tissue != "PBMC"].copy()
log(f"skin subset {A.shape} from {A.obs.gid.nunique()} libraries")

sc.pp.filter_genes(A, min_cells=3)
log(f"gene filter -> {A.shape}")

A.layers["counts"] = A.X.copy()
sc.pp.normalize_total(A, target_sum=1e4)
sc.pp.log1p(A)
A.raw = A

sc.pp.highly_variable_genes(A, n_top_genes=3000, batch_key="gid")
log(f"HVGs {int(A.var.highly_variable.sum())}")

# cell-cycle scores (regressed out, as published)
s_genes = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7","DTL",
           "PRIM1","UHRF1","CENPU","HELLS","RFC2","RPA2","NASP","RAD51AP1","GMNN","WDR76","SLBP",
           "CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN",
           "DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]
g2m_genes = ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2","CKS1B",
             "MKI67","TMPO","CENPF","TACC3","PIMREG","SMC4","CCNB2","CKAP2L","CKAP2","AURKB",
             "BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","CDCA3","JPT1","CDC20",
             "TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23",
             "HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA"]
sc.tl.score_genes_cell_cycle(A, s_genes=[g for g in s_genes if g in A.var_names],
                             g2m_genes=[g for g in g2m_genes if g in A.var_names])
log("cell-cycle scored")

A = A[:, A.var.highly_variable].copy()
sc.pp.regress_out(A, ["pctMT", "S_score", "G2M_score"], n_jobs=16)
sc.pp.scale(A, max_value=10)
log("scaled + regressed")

sc.tl.pca(A, n_comps=50, svd_solver="arpack", random_state=SEED)
log("PCA done")
A.write_h5ad(f"{RES}/02_ckpt_pca.h5ad", compression="gzip")
log("checkpoint written")

harmony(A, "gid")
log("Harmony done")

sc.pp.neighbors(A, n_neighbors=15, n_pcs=50, use_rep="X_pca_harmony", random_state=SEED)
log("neighbours done")

for r in [0.3, 0.5, 0.8, 1.0, 1.5]:
    sc.tl.leiden(A, resolution=r, key_added=f"leiden_{r}", flavor="igraph",
                 n_iterations=2, directed=False, random_state=SEED)
    log(f"leiden r={r}: {A.obs[f'leiden_{r}'].nunique()} clusters")

A.obs["leiden"] = A.obs["leiden_1.0"]
sc.tl.umap(A, random_state=SEED)
log("UMAP done")

# restore full log-normalised matrix for downstream marker work
full = A.raw.to_adata()
full.obs = A.obs.copy()
full.obsm = A.obsm.copy()
full.obsp = A.obsp.copy()
full.uns["neighbors"] = A.uns["neighbors"]
full.write_h5ad(f"{RES}/02_atlas_integrated.h5ad", compression="gzip")

A.obs.to_csv(f"{RES}/02_cell_metadata.csv")
pd.DataFrame({"resolution": [0.3,0.5,0.8,1.0,1.5],
              "n_clusters": [A.obs[f"leiden_{r}"].nunique() for r in [0.3,0.5,0.8,1.0,1.5]]}
             ).to_csv(f"{RES}/02_resolution_scan.csv", index=False)

log(f"=== INTEGRATED ATLAS: {full.n_obs:,} cells x {full.n_vars:,} genes, "
    f"{A.obs.leiden.nunique()} clusters at r=1.0 ===")
