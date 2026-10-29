#!/usr/bin/env python
"""
Khan 2 revision -- Step 16
Confounder audit of the patient-level comparison.

Motivated by an independent re-analysis of the same accession (Yao et al. 2026),
which reported that sex-linked genes reached significance in a DFU-versus-control
macrophage contrast and warned that part of the transcriptional signal may
reflect demographic imbalance rather than biology.

This script therefore checks, for the healing-versus-non-healing comparison used
throughout the revision:
  (a) inferred sex of every library from XIST / Y-chromosome gene expression
  (b) whether sex is imbalanced between the healing groups
  (c) whether sex-linked genes appear among the macrophage pseudobulk hits
  (d) whether the leading pseudobulk genes survive adjustment for inferred sex
  (e) whether library size / cell number differs between groups
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from scipy import stats
warnings.filterwarnings("ignore")

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES = f"{BASE}/02_results"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

XIST = ["XIST"]
YGEN = ["RPS4Y1", "DDX3Y", "UTY", "USP9Y", "KDM5D", "EIF1AY", "NLGN4Y", "ZFY"]

pb = pd.read_csv(f"{RES}/04_macrophage_pseudobulk_counts.csv", index_col=0)
pbm = pd.read_csv(f"{RES}/04_pseudobulk_sample_metadata.csv", index_col=0)
cpm = np.log2(pb.div(pb.sum(1), axis=0) * 1e6 + 1)

# --------------------------------------------------- (a) infer sex per library
# Sex is called from the WHOLE library, not from its macrophages. A macrophage
# pseudobulk built from a few dozen cells drops Y transcripts often enough that
# the call becomes threshold-dependent; summing all cells in the library gives
# an unambiguous, bimodal signal.
A = sc.read_h5ad(f"{RES}/03_atlas_annotated.h5ad")
Xall = sp.csr_matrix(A.X)                       # log1p-normalised
lin = Xall.copy(); lin.data = np.expm1(lin.data)
gidx = {g: i for i, g in enumerate(A.var_names)}
have_y = [g for g in YGEN if g in gidx]
have_x = [g for g in XIST if g in gidx]
rows = {}
for gid, idx in A.obs.groupby("gid", observed=True).indices.items():
    sub = lin[idx]
    tot = float(sub.sum()) + 1e-9
    ymean = float(np.log2(sub[:, [gidx[g] for g in have_y]].sum() / tot * 1e6 + 1))
    xmean = (float(np.log2(sub[:, [gidx[g] for g in have_x]].sum() / tot * 1e6 + 1))
             if have_y and have_x else np.nan)
    rows[gid] = dict(Y_genes=ymean, XIST=xmean, n_cells=len(idx))
whole = pd.DataFrame(rows).T

Y_THRESHOLD = 1.0
whole["inferred_sex"] = np.where(whole.Y_genes > Y_THRESHOLD, "male", "female")
log(f"whole-library Y range {whole.Y_genes.min():.2f}-{whole.Y_genes.max():.2f}; "
    f"cut-off {Y_THRESHOLD} log2 CPM")
log("XIST vs Y separation check (should be anti-correlated):")
if have_x:
    log(f"  Spearman rho(XIST, Y) = {stats.spearmanr(whole.XIST, whole.Y_genes)[0]:.2f}")

sex = whole.reindex(pbm.index)[["XIST", "Y_genes", "inferred_sex"]].copy()
sex["group"] = pbm.group
sex.to_csv(f"{RES}/16_inferred_sex.csv")
log("inferred sex per library:\n" + sex.groupby(["group", "inferred_sex"]).size().to_string())

# ------------------------------------------------- (b) imbalance between groups
ul = sex[sex.group.isin(["Healing", "Nonhealing"])]
tab = pd.crosstab(ul.group, ul.inferred_sex)
if tab.shape == (2, 2):
    odds, p_fisher = stats.fisher_exact(tab.values)
else:
    odds, p_fisher = np.nan, 1.0
log(f"\nsex x healing-group contingency:\n{tab.to_string()}")
log(f"Fisher exact P = {p_fisher:.3f} (odds ratio {odds:.2f})")

# --------------------------------- (c) sex-linked genes among pseudobulk hits
deg = pd.read_csv(f"{RES}/04_pseudobulk_DESeq2_NHvsH.csv", index_col=0)
sexg = [g for g in XIST + YGEN if g in deg.index]
hits = deg.loc[sexg, ["log2FoldChange", "pvalue", "padj"]].sort_values("pvalue")
hits.to_csv(f"{RES}/16_sexlinked_in_pseudobulk.csv")
log("\nsex-linked genes in the macrophage pseudobulk contrast:\n" + hits.to_string())
n_sex_nominal = int((hits.pvalue < 0.05).sum())
log(f"sex-linked genes at nominal P < 0.05: {n_sex_nominal} of {len(hits)}")

# ------------------------------- (d) do the leading genes survive sex adjustment?
sub = pbm[pbm.group.isin(["Healing", "Nonhealing"])].index
X = pb.loc[sub].round().astype(int)
X = X.loc[:, (X > 0).sum(0) >= max(3, int(0.2 * len(sub)))]
cd = pd.DataFrame({
    "group": pd.Categorical(pbm.loc[sub, "group"], categories=["Healing", "Nonhealing"]),
    "sex": pd.Categorical(sex.loc[sub, "inferred_sex"])}, index=sub)
log(f"\nre-running DESeq2 with ~sex+group on {X.shape[0]} samples x {X.shape[1]} genes")

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
if cd.sex.nunique() > 1:
    dds = DeseqDataSet(counts=X, metadata=cd, design="~sex+group", refit_cooks=True, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["group", "Nonhealing", "Healing"], quiet=True)
    st.summary()
    adj = st.results_df.sort_values("pvalue")
    adj.to_csv(f"{RES}/16_pseudobulk_DESeq2_sex_adjusted.csv")
    comp = pd.DataFrame({
        "log2FC_unadjusted": deg.log2FoldChange, "p_unadjusted": deg.pvalue,
        "padj_unadjusted": deg.padj,
        "log2FC_sex_adjusted": adj.log2FoldChange, "p_sex_adjusted": adj.pvalue,
        "padj_sex_adjusted": adj.padj}).dropna()
    comp.to_csv(f"{RES}/16_sex_adjustment_comparison.csv")
    r, _ = stats.spearmanr(comp.log2FC_unadjusted, comp.log2FC_sex_adjusted)
    log(f"DESeq2 with sex: {int((adj.padj<0.05).sum())} genes FDR<0.05 "
        f"(unadjusted: {int((deg.padj<0.05).sum())}); "
        f"log2FC correlation {r:.3f}")
    log("top 10 after sex adjustment:\n" +
        adj.head(10)[["log2FoldChange","pvalue","padj"]].to_string())
else:
    adj = None
    log("only one inferred sex present; adjustment not applicable")

# -------------------------------------- (e) technical covariates between groups
tech = pbm.loc[sub].copy()
tech["total_counts"] = pb.loc[sub].sum(1).values
tech["genes_detected"] = (pb.loc[sub] > 0).sum(1).values
rows = []
for c in ["n_cells", "total_counts", "genes_detected"]:
    a = tech.loc[tech.group == "Nonhealing", c].astype(float)
    b = tech.loc[tech.group == "Healing", c].astype(float)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rows.append(dict(covariate=c, median_nonhealing=float(a.median()),
                     median_healing=float(b.median()), p=float(p)))
T = pd.DataFrame(rows)
T.to_csv(f"{RES}/16_technical_covariates.csv", index=False)
log("\n=== TECHNICAL COVARIATES BY GROUP ===\n" + T.to_string(index=False))

json.dump(dict(
    n_male=int((ul.inferred_sex == "male").sum()),
    n_female=int((ul.inferred_sex == "female").sum()),
    sex_by_group={k: v.to_dict() for k, v in tab.iterrows()} if tab.shape == (2, 2) else {},
    fisher_p=float(p_fisher),
    n_sexlinked_nominal=n_sex_nominal,
    n_FDR05_unadjusted=int((deg.padj < 0.05).sum()),
    n_FDR05_sex_adjusted=int((adj.padj < 0.05).sum()) if adj is not None else None,
    technical_covariates_min_p=float(T.p.min())),
    open(f"{RES}/16_summary.json", "w"), indent=2)
log("=== STEP 16 COMPLETE ===")
