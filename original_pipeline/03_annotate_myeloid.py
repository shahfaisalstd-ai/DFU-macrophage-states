#!/usr/bin/env python
"""
Khan 2 revision -- Step 03
Cell-type annotation, myeloid extraction, and a functional macrophage-state
analysis that goes beyond the M1/M2 dichotomy.

Reviewer 1, comment 1: "macrophage states are highly heterogeneous, and the
classical M1/M2 classification may oversimplify single-cell macrophage
phenotypes. The authors should provide additional functional marker analysis
and avoid directly equating M2-like states with impaired wound healing."

Here each macrophage is scored against 13 independently curated functional
programmes (resident, lipid-associated, inflammatory, classical-M1,
alternative-M2, angiogenic, efferocytotic, antigen-presenting, complement,
interferon-stimulated, matrix-remodelling, proliferating, monocyte) so the
non-healing skew can be described in functional terms rather than as an
"M2 = impaired healing" equation.
"""
import os, time, warnings, json
import numpy as np, pandas as pd, scanpy as sc
warnings.filterwarnings("ignore")

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES  = f"{BASE}/02_results"
sc.settings.n_jobs = 16
SEED = 123; np.random.seed(SEED)
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


# ---------------------------------------------------------------- lineage sets
LINEAGE = {
 "Keratinocyte":       ["KRT1","KRT10","KRT14","KRT5","KRT6A","KRTDAP","DMKN","SBSN","LORICRIN","FLG"],
 "Fibroblast":         ["COL1A1","COL1A2","COL3A1","DCN","LUM","PDGFRA","MMP2","POSTN"],
 "Endothelial":        ["PECAM1","VWF","CDH5","CLDN5","EGFL7","RAMP2"],
 "Lymphatic EC":       ["PROX1","PDPN","CCL21","LYVE1","FLT4"],
 "Smooth muscle":      ["ACTA2","TAGLN","MYH11","DES","MYL9"],
 "Pericyte":           ["RGS5","PDGFRB","NOTCH3","KCNJ8","HIGD1B"],
 "Melanocyte":         ["PMEL","MLANA","TYRP1","DCT","TYR"],
 "Mast cell":          ["TPSAB1","TPSB2","CPA3","MS4A2","KIT"],
 "Myeloid":            ["LYZ","CD68","CD14","AIF1","ITGAM","C1QA","C1QB","FCGR3A","CD163","MNDA"],
 "T cell":             ["CD3D","CD3E","CD2","IL7R","TRAC","CD8A","CD40LG"],
 "NK cell":            ["NKG7","GNLY","KLRD1","PRF1","KLRF1"],
 "B/Plasma cell":      ["MS4A1","CD79A","JCHAIN","MZB1","IGHG1"],
 "Sweat gland":        ["DCD","SCGB2A2","MUCL1","PIP","SCGB1B2P"],
 "Schwann cell":       ["PLP1","S100B","MPZ","NRXN1"],
 "Erythrocyte":        ["HBB","HBA1","HBA2"],
}

# ------------------------------------------------- functional macrophage sets
# Each programme is drawn from an independent published source so that the
# scores are not circularly defined by the M1/M2 axis.
MACSTATE = {
 "Resident_tissue":      ["FOLR2","LYVE1","MRC1","SELENOP","STAB1","F13A1","CD163","MAF","IGF1"],
 "Lipid_associated_LAM": ["TREM2","APOE","APOC1","LPL","CD9","GPNMB","FABP5","ACP5","CTSD","LIPA"],
 "Inflammatory_mono":    ["S100A8","S100A9","S100A12","FCN1","VCAN","SELL","CD52","LYZ"],
 "Classical_M1":         ["CD80","CD86","NOS2","IL1B","TNF","CXCL9","CXCL10","CXCL11","IDO1","SOCS3"],
 "Alternative_M2":       ["MRC1","CD163","IL10","TGFB1","CCL18","CCL22","MSR1","CLEC7A","CD209"],
 "Angiogenic":           ["VEGFA","VEGFB","ADM","THBS1","HIF1A","CXCL8","EREG"],
 "Efferocytotic":        ["MERTK","GAS6","AXL","TIMD4","ITGAV","ITGB5","CD36","ANXA1"],
 "Antigen_presenting":   ["HLA-DRA","HLA-DRB1","HLA-DPA1","HLA-DPB1","CD74","CIITA","HLA-DQA1"],
 "Complement":           ["C1QA","C1QB","C1QC","C3","CFD","VSIG4","C2"],
 "Interferon_stim":      ["ISG15","IFIT1","IFIT3","MX1","STAT1","IFI6","OAS1"],
 "Matrix_remodelling":   ["SPP1","MMP9","MMP12","MMP19","TIMP1","CTSK"],
 "Proliferating":        ["MKI67","TOP2A","STMN1","TUBB","CDK1"],
 "Phagocytic_lysosomal": ["CD68","LAMP1","CTSB","CTSS","CTSL","NPC2","GRN"],
}

def score_sets(A, sets, prefix=""):
    """Add one gene-set score per programme; returns list of score column names."""
    cols = []
    for name, genes in sets.items():
        present = [g for g in genes if g in A.var_names]
        if len(present) < 2:
            log(f"  skip {name}: only {len(present)} genes present"); continue
        key = f"{prefix}{name}"
        sc.tl.score_genes(A, present, score_name=key, random_state=SEED)
        cols.append(key)
    return cols

# ==================================================================== annotate
A = sc.read_h5ad(f"{RES}/02_atlas_integrated.h5ad")
log(f"loaded integrated atlas {A.shape}")

lin_cols = score_sets(A, LINEAGE, prefix="sc_")
log(f"scored {len(lin_cols)} lineage programmes")

# assign each Leiden cluster to the lineage with the highest mean z-scored score
cl = A.obs.groupby("leiden", observed=True)[lin_cols].mean()
clz = (cl - cl.mean()) / cl.std(ddof=0)
assign = clz.idxmax(axis=1).str.replace("sc_", "", regex=False)
A.obs["cell_type"] = A.obs["leiden"].map(assign).astype(str)
clz.to_csv(f"{RES}/03_cluster_lineage_zscores.csv")
log("cell types assigned:\n" + A.obs.cell_type.value_counts().to_string())

A.obs[["gsm","gid","tissue","condition","group","leiden","cell_type",
       "nCount","nFeature","pctMT"]].to_csv(f"{RES}/03_cell_metadata_annotated.csv")

comp = (pd.crosstab(A.obs.gid, A.obs.cell_type, normalize="index") * 100)
comp.to_csv(f"{RES}/03_celltype_proportions_per_sample.csv")
pd.crosstab(A.obs.group, A.obs.cell_type).to_csv(f"{RES}/03_celltype_by_group_counts.csv")

# ============================================================ myeloid analysis
mye = A[A.obs.cell_type == "Myeloid"].copy()
log(f"myeloid cells recovered: {mye.n_obs:,} "
    f"({100*mye.n_obs/A.n_obs:.2f}% of atlas) from {mye.obs.gid.nunique()} libraries")

# re-cluster the myeloid compartment on its own
sc.pp.highly_variable_genes(mye, n_top_genes=2000, batch_key="gid")
mye.raw = mye
m = mye[:, mye.var.highly_variable].copy()
sc.pp.scale(m, max_value=10)
sc.tl.pca(m, n_comps=30, svd_solver="arpack", random_state=SEED)
harmony(m, "gid")
sc.pp.neighbors(m, n_neighbors=15, n_pcs=30, use_rep="X_pca_harmony", random_state=SEED)
sc.tl.leiden(m, resolution=0.8, key_added="mye_leiden", flavor="igraph",
             n_iterations=2, directed=False, random_state=SEED)
sc.tl.umap(m, random_state=SEED)
mye.obs["mye_leiden"] = m.obs["mye_leiden"].values
mye.obsm["X_umap"] = m.obsm["X_umap"]
mye.obsm["X_pca_harmony"] = m.obsm["X_pca_harmony"]
log(f"myeloid subclusters: {mye.obs.mye_leiden.nunique()}")

# functional programme scores on every myeloid cell
mac_cols = score_sets(mye, MACSTATE, prefix="ms_")
log(f"scored {len(mac_cols)} functional macrophage programmes")

# myeloid subtype label from canonical myeloid markers
MYE_SUB = {
 "Resident M2-like macrophage": ["FOLR2","LYVE1","C1QA","C1QB","C1QC","MRC1","SELENOP","CD163"],
 "Inflammatory monocyte-macrophage": ["S100A8","S100A9","S100A12","FCN1","VCAN","IL1B"],
 "Lipid-associated macrophage": ["TREM2","APOE","APOC1","GPNMB","CD9","SPP1"],
 "cDC1": ["CLEC9A","XCR1","BATF3","IDO1"],
 "cDC2": ["CD1C","FCER1A","CLEC10A","CD1E"],
 "LC/Langerhans": ["CD207","CD1A","EPCAM"],
 "pDC": ["LILRA4","GZMB","IRF7","JCHAIN"],
 "Mature/migratory DC": ["LAMP3","CCR7","CCL19","FSCN1"],
 "Proliferating myeloid": ["MKI67","TOP2A","STMN1"],
}
sub_cols = score_sets(mye, MYE_SUB, prefix="sub_")
sc_ = mye.obs.groupby("mye_leiden", observed=True)[sub_cols].mean()
scz = (sc_ - sc_.mean()) / sc_.std(ddof=0)
sub_assign = scz.idxmax(axis=1).str.replace("sub_", "", regex=False)
mye.obs["myeloid_subtype"] = mye.obs["mye_leiden"].map(sub_assign).astype(str)
scz.to_csv(f"{RES}/03_myeloid_subcluster_zscores.csv")
log("myeloid subtypes:\n" + mye.obs.myeloid_subtype.value_counts().to_string())

# per-cluster markers (full myeloid compartment)
sc.tl.rank_genes_groups(mye, "mye_leiden", method="wilcoxon", use_raw=True)
pd.concat([sc.get.rank_genes_groups_df(mye, group=g).assign(cluster=g)
           for g in mye.obs.mye_leiden.cat.categories]
          ).to_csv(f"{RES}/03_myeloid_subcluster_markers.csv", index=False)

mye.obs.to_csv(f"{RES}/03_myeloid_cell_metadata.csv")
mye.write_h5ad(f"{RES}/03_myeloid.h5ad", compression="gzip")
A.write_h5ad(f"{RES}/03_atlas_annotated.h5ad", compression="gzip")

summary = dict(atlas_cells=int(A.n_obs), atlas_genes=int(A.n_vars),
               n_libraries=int(A.obs.gid.nunique()),
               myeloid_cells=int(mye.n_obs),
               myeloid_pct=round(100*mye.n_obs/A.n_obs, 3),
               myeloid_subclusters=int(mye.obs.mye_leiden.nunique()),
               n_clusters=int(A.obs.leiden.nunique()))
json.dump(summary, open(f"{RES}/03_summary.json","w"), indent=2)
log("=== STEP 03 COMPLETE ===\n" + json.dumps(summary, indent=2))
