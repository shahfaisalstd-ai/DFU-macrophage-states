#!/usr/bin/env python
"""
Khan 2 revision -- Step 06
Bulk differential expression (GSE134431) and external validation (GSE143735),
reported so that directional support and statistical validation are never
conflated.

Reviewer 2, comment 2: "Claiming 'validation' based only on directional
concordance (rather than effect size or statistical significance) may be overly
optimistic. Clearly distinguish 'directional support' from 'statistical
validation'."

For the prioritised panel this script reports, side by side:
  - the discovery effect (log2FC, CI, p, FDR) in foot DFU
  - the replication effect (log2FC, CI, p, FDR) in forearm DFU
  - the sign agreement, tested against chance with an exact binomial test
  - how many genes meet a genuine statistical replication bar (same sign AND
    p < 0.05 in the validation cohort) as opposed to sign agreement alone
  - a permutation null for the concordance rate of the panel
  - a site-effect analysis quantifying how much variance anatomical site
    explains relative to healing status
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from khan_bulk import load_counts, dfu_groups, cpm_log, gene_annotation

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES  = f"{BASE}/02_results"
SEED = 123; rng = np.random.default_rng(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def run_deseq(C, groups, keep=("Healing", "Nonhealing")):
    sub = groups[groups.isin(keep)]
    X = C[sub.index].T.round().astype(int)
    X = X.loc[:, (X > 0).sum(0) >= max(3, int(0.2 * len(sub)))]
    cd = pd.DataFrame({"group": pd.Categorical(sub.values, categories=list(keep))}, index=sub.index)
    dds = DeseqDataSet(counts=X, metadata=cd, design="~group", refit_cooks=True, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["group", keep[1], keep[0]], quiet=True)
    st.summary()
    r = st.results_df.copy()
    # Wald CI on the log2 scale
    r["ci_low"]  = r.log2FoldChange - 1.96 * r.lfcSE
    r["ci_high"] = r.log2FoldChange + 1.96 * r.lfcSE
    return r, sub


ann = gene_annotation()
C134, m134 = load_counts("GSE134431", ann); g134 = dfu_groups("GSE134431", m134)
C143, m143 = load_counts("GSE143735", ann); g143 = dfu_groups("GSE143735", m143)
log(f"discovery GSE134431 {C134.shape} {g134.value_counts().to_dict()}")
log(f"validation GSE143735 {C143.shape} {g143.value_counts().to_dict()}")

disc, sub134 = run_deseq(C134, g134)
disc.to_csv(f"{RES}/06_discovery_DESeq2_GSE134431.csv")
log(f"discovery: {(disc.padj<0.05).sum()} genes FDR<0.05, "
    f"{(disc.pvalue<0.05).sum()} nominal p<0.05  (n={len(sub134)})")

vald, sub143 = run_deseq(C143, g143)
vald.to_csv(f"{RES}/06_validation_DESeq2_GSE143735.csv")
log(f"validation: {(vald.padj<0.05).sum()} genes FDR<0.05, "
    f"{(vald.pvalue<0.05).sum()} nominal p<0.05  (n={len(sub143)})")

# ------------------------------------------------ the prioritised panel
# The published consensus panel is the set of genes recurring in >= 3 of the
# six discovery analyses (n_analyses column), which is exactly 129 genes.
PANEL_CSV = os.environ.get("PANEL_CSV", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "metadata", "consensus_novel_panel.csv"))
pan = pd.read_csv(PANEL_CSV)
panel_levels = {f"consensus panel (>={t} analyses)":
                pan.loc[pan.n_analyses >= t, "gene"].astype(str).unique().tolist()
                for t in (3, 4, 5)}
panel = panel_levels["consensus panel (>=3 analyses)"]
log(f"consensus panel loaded: " +
    ", ".join(f"{k} = {len(v)}" for k, v in panel_levels.items()))

common = disc.index.intersection(vald.index)
panel = [g for g in panel if g in common]
panel_levels = {k: [g for g in v if g in common] for k, v in panel_levels.items()}
log(f"panel genes testable in both cohorts: {len(panel)}")

# --------------------------------------- directional vs statistical replication
J = pd.DataFrame({
    "disc_log2FC": disc.loc[common, "log2FoldChange"],
    "disc_ci_low": disc.loc[common, "ci_low"], "disc_ci_high": disc.loc[common, "ci_high"],
    "disc_p": disc.loc[common, "pvalue"], "disc_padj": disc.loc[common, "padj"],
    "val_log2FC": vald.loc[common, "log2FoldChange"],
    "val_ci_low": vald.loc[common, "ci_low"], "val_ci_high": vald.loc[common, "ci_high"],
    "val_p": vald.loc[common, "pvalue"], "val_padj": vald.loc[common, "padj"],
}).dropna(subset=["disc_log2FC", "val_log2FC"])
J["same_sign"] = np.sign(J.disc_log2FC) == np.sign(J.val_log2FC)
J["in_panel"] = J.index.isin(panel)
J["statistically_replicated"] = J.same_sign & (J.val_p < 0.05)
J["ci_overlap"] = (J.disc_ci_low <= J.val_ci_high) & (J.val_ci_low <= J.disc_ci_high)
J.to_csv(f"{RES}/06_discovery_vs_validation_joint.csv")


def concordance(sub, label):
    n = len(sub); k = int(sub.same_sign.sum())
    bp = stats.binomtest(k, n, 0.5, alternative="greater").pvalue if n else np.nan
    lo, hi = stats.binomtest(k, n, 0.5).proportion_ci(0.95) if n else (np.nan, np.nan)
    rep = int(sub.statistically_replicated.sum())
    r, pr = (stats.spearmanr(sub.disc_log2FC, sub.val_log2FC) if n > 3 else (np.nan, np.nan))
    return dict(set=label, n_genes=n,
                directional_concordance_pct=round(100*k/n, 1) if n else np.nan,
                concordance_ci_low_pct=round(100*lo,1), concordance_ci_high_pct=round(100*hi,1),
                binomial_p_vs_chance=bp,
                statistically_replicated_n=rep,
                statistically_replicated_pct=round(100*rep/n, 1) if n else np.nan,
                spearman_rho=float(r), spearman_p=float(pr))

sets = {
    "whole transcriptome": J,
    "nominal discovery p<0.05": J[J.disc_p < 0.05],
    "discovery FDR<0.05": J[J.disc_padj < 0.05],
}
for lbl, gl in panel_levels.items():
    sets[lbl] = J[J.index.isin(gl)]
sets["consensus panel (>=3), |log2FC|>1"] = J[J.in_panel & (J.disc_log2FC.abs() > 1)]
conc = pd.DataFrame([concordance(v, k) for k, v in sets.items()])
conc.to_csv(f"{RES}/06_concordance_by_evidence_level.csv", index=False)
log("\n=== DIRECTIONAL SUPPORT vs STATISTICAL REPLICATION ===")
log(conc.to_string(index=False))

# permutation null for the panel's concordance
obs = float(J[J.in_panel].same_sign.mean()) if J.in_panel.any() else np.nan
null = np.array([J.same_sign.sample(int(J.in_panel.sum()), random_state=int(s)).mean()
                 for s in rng.integers(0, 1e6, 2000)])
perm_p = float((null >= obs).mean())
json.dump(dict(panel_concordance=obs, permutation_p=perm_p,
               null_mean=float(null.mean()), null_sd=float(null.std())),
          open(f"{RES}/06_panel_permutation.json","w"), indent=2)
log(f"panel concordance {obs:.3f}; permutation p = {perm_p:.4f} "
    f"(null {null.mean():.3f} +/- {null.std():.3f})")

# ------------------------------------------------- anatomical site variance
L134, L143 = cpm_log(C134), cpm_log(C143)
shared = L134.index.intersection(L143.index)
M = pd.concat([L134.loc[shared], L143.loc[shared]], axis=1)
site = pd.Series(["Foot"]*L134.shape[1] + ["Forearm"]*L143.shape[1], index=M.columns)
heal = pd.concat([g134, g143]).loc[M.columns]
keep = heal.isin(["Healing", "Nonhealing"])
Mk, sk, hk = M.loc[:, keep], site[keep], heal[keep]
Mk = Mk.loc[Mk.var(1) > 0]

def eta_sq(y, f):
    grand = y.mean()
    ss_b = sum(len(y[f == l]) * (y[f == l].mean() - grand) ** 2 for l in f.unique())
    ss_t = ((y - grand) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else np.nan

sub_ix = Mk.var(1).sort_values(ascending=False).head(4000).index
var_df = pd.DataFrame({
    "eta2_site":    [eta_sq(Mk.loc[g], sk) for g in sub_ix],
    "eta2_healing": [eta_sq(Mk.loc[g], hk) for g in sub_ix]}, index=sub_ix)
var_df.to_csv(f"{RES}/06_variance_site_vs_healing.csv")
site_note = dict(n_genes=len(sub_ix),
                 median_eta2_site=float(var_df.eta2_site.median()),
                 median_eta2_healing=float(var_df.eta2_healing.median()),
                 ratio=float(var_df.eta2_site.median() / max(var_df.eta2_healing.median(), 1e-9)),
                 pct_genes_site_dominant=float((var_df.eta2_site > var_df.eta2_healing).mean()*100))
json.dump(site_note, open(f"{RES}/06_site_variance_summary.json","w"), indent=2)
log("\n=== ANATOMICAL SITE vs HEALING STATUS VARIANCE ===\n" + json.dumps(site_note, indent=2))

pd.DataFrame({"gene": panel}).to_csv(f"{RES}/06_panel_genes_used.csv", index=False)
log("=== STEP 06 COMPLETE ===")
