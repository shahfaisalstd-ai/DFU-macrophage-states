#!/usr/bin/env python
"""
Khan 2 revision -- Step 04
Patient-level (pseudobulk) analysis of the myeloid compartment.

Reviewer 1, comment 2: "the actual biological replicates are determined by the
number of patients rather than the number of cells. Patient-level or pseudobulk
analysis is recommended to improve statistical robustness."

Every test below therefore uses the LIBRARY (patient sample) as the unit of
analysis, n = number of samples, not number of cells:
  (a) DESeq2 (PyDESeq2) on macrophage pseudobulk counts, non-healing vs healing
  (b) patient-level functional-programme scores (mean per sample) compared with
      Mann-Whitney U + Benjamini-Hochberg, with Hodges-Lehmann effect sizes
  (c) patient-level macrophage-state composition (proportion per sample)
  (d) a cell-level-vs-patient-level comparison that quantifies how much the
      original per-cell testing inflated significance (pseudoreplication check)
"""
import os, time, json, warnings
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES  = f"{BASE}/02_results"
SEED = 123; np.random.seed(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

MIN_CELLS = 10          # minimum macrophage cells for a sample to enter pseudobulk
GROUPS = ["Nonhealing", "Healing"]

mye = sc.read_h5ad(f"{RES}/03_myeloid.h5ad")
log(f"myeloid object {mye.shape}")

# macrophage/monocyte compartment only (exclude DC and proliferating)
is_mac = ~mye.obs.myeloid_subtype.str.contains("DC|Langerhans|Proliferating", regex=True)
mac = mye[is_mac].copy()
log(f"macrophage/monocyte cells: {mac.n_obs:,} of {mye.n_obs:,} myeloid "
    f"across {mac.obs.gid.nunique()} libraries")

counts = mac.layers["counts"] if "counts" in mac.layers else mac.raw.X
counts = sp.csr_matrix(counts)
genes  = mac.raw.var_names if mac.raw is not None else mac.var_names
genes  = pd.Index(genes)

# ------------------------------------------------------- (a) pseudobulk DESeq2
rows, meta = [], []
for gid, idx in mac.obs.groupby("gid", observed=True).indices.items():
    if len(idx) < MIN_CELLS:
        continue
    rows.append(np.asarray(counts[idx].sum(0)).ravel())
    r = mac.obs.iloc[idx[0]]
    meta.append(dict(gid=gid, gsm=r.gsm, tissue=r.tissue, group=r.group,
                     condition=r.condition, n_cells=len(idx)))
pb = pd.DataFrame(np.vstack(rows), index=[m["gid"] for m in meta], columns=genes)
pbm = pd.DataFrame(meta).set_index("gid")
pb.to_csv(f"{RES}/04_macrophage_pseudobulk_counts.csv")
pbm.to_csv(f"{RES}/04_pseudobulk_sample_metadata.csv")
log(f"pseudobulk matrix: {pb.shape[0]} samples x {pb.shape[1]} genes "
    f"(>= {MIN_CELLS} cells/sample)")
log("samples per group:\n" + pbm.group.value_counts().to_string())

sub = pbm[pbm.group.isin(GROUPS)]
X = pb.loc[sub.index].round().astype(int)
X = X.loc[:, (X > 0).sum(0) >= max(3, int(0.2 * len(sub)))]     # expressed filter
log(f"DESeq2 input: {X.shape[0]} samples ({sub.group.value_counts().to_dict()}) "
    f"x {X.shape[1]} genes")

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
cd = sub[["group"]].copy()
cd["group"] = pd.Categorical(cd["group"], categories=["Healing", "Nonhealing"])
dds = DeseqDataSet(counts=X, metadata=cd, design="~group", refit_cooks=True, quiet=True)
dds.deseq2()
st = DeseqStats(dds, contrast=["group", "Nonhealing", "Healing"], quiet=True)
st.summary()
deg = st.results_df.sort_values("pvalue")
deg.to_csv(f"{RES}/04_pseudobulk_DESeq2_NHvsH.csv")
nsig = int((deg.padj < 0.05).sum())
log(f"DESeq2: {nsig} genes at FDR<0.05; {(deg.pvalue<0.05).sum()} at nominal p<0.05")
log("top 15 by p:\n" + deg.head(15)[["log2FoldChange","pvalue","padj"]].to_string())

# ------------------------------- (b) patient-level functional programme scores
ms_cols = [c for c in mac.obs.columns if c.startswith("ms_")]
per_cell   = mac.obs[["gid","group"] + ms_cols]
per_sample = per_cell.groupby(["gid","group"], observed=True)[ms_cols].mean().reset_index()
per_sample.to_csv(f"{RES}/04_programme_scores_per_sample.csv", index=False)

def hodges_lehmann(a, b):
    return float(np.median([x - y for x in a for y in b]))

res = []
for c in ms_cols:
    a = per_sample.loc[per_sample.group == "Nonhealing", c].dropna().values
    b = per_sample.loc[per_sample.group == "Healing", c].dropna().values
    if len(a) < 3 or len(b) < 3:
        continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # rank-biserial correlation as a bounded effect size
    rbc = 1 - 2 * u / (len(a) * len(b))
    ca = per_cell.loc[per_cell.group == "Nonhealing", c].dropna().values
    cb = per_cell.loc[per_cell.group == "Healing", c].dropna().values
    _, p_cell = stats.mannwhitneyu(ca, cb, alternative="two-sided")
    res.append(dict(programme=c.replace("ms_", ""),
                    n_nonhealing=len(a), n_healing=len(b),
                    mean_nonhealing=float(a.mean()), mean_healing=float(b.mean()),
                    hodges_lehmann=hodges_lehmann(a, b),
                    rank_biserial=float(-rbc),
                    p_patient_level=float(p),
                    p_cell_level=float(p_cell),
                    n_cells_nonhealing=len(ca), n_cells_healing=len(cb)))
r = pd.DataFrame(res)
r["padj_patient_level"] = multipletests(r.p_patient_level, method="fdr_bh")[1]
r["padj_cell_level"]    = multipletests(r.p_cell_level, method="fdr_bh")[1]
r["log10_p_inflation"]  = np.log10(r.p_patient_level) - np.log10(r.p_cell_level)
r = r.sort_values("p_patient_level")
r.to_csv(f"{RES}/04_programme_patient_vs_cell_level.csv", index=False)
log("\n=== FUNCTIONAL PROGRAMMES, PATIENT-LEVEL (n samples) ===")
log(r[["programme","mean_nonhealing","mean_healing","rank_biserial",
       "p_patient_level","padj_patient_level","p_cell_level"]].to_string(index=False))

# ------------------------------------ (c) patient-level state composition
comp = pd.crosstab(mac.obs.gid, mac.obs.myeloid_subtype, normalize="index") * 100
comp = comp.join(mac.obs.groupby("gid", observed=True).group.first())
comp.to_csv(f"{RES}/04_macrophage_state_composition_per_sample.csv")
crows = []
for st_ in [c for c in comp.columns if c != "group"]:
    a = comp.loc[comp.group == "Nonhealing", st_].dropna().values
    b = comp.loc[comp.group == "Healing", st_].dropna().values
    if len(a) < 3 or len(b) < 3: continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    crows.append(dict(state=st_, n_nonhealing=len(a), n_healing=len(b),
                      pct_nonhealing=float(a.mean()), pct_healing=float(b.mean()),
                      rank_biserial=float(2*u/(len(a)*len(b))-1), p=float(p)))
cdf = pd.DataFrame(crows)
if len(cdf):
    cdf["padj"] = multipletests(cdf.p, method="fdr_bh")[1]
    cdf = cdf.sort_values("p")
    cdf.to_csv(f"{RES}/04_state_composition_tests.csv", index=False)
    log("\n=== MACROPHAGE STATE COMPOSITION, PATIENT-LEVEL ===")
    log(cdf.to_string(index=False))

json.dump(dict(macrophage_cells=int(mac.n_obs),
               pseudobulk_samples=int(pb.shape[0]),
               deseq2_samples=int(X.shape[0]),
               deseq2_genes=int(X.shape[1]),
               deseq2_sig_FDR05=nsig,
               groups=sub.group.value_counts().to_dict()),
          open(f"{RES}/04_summary.json","w"), indent=2)
log("=== STEP 04 COMPLETE ===")
