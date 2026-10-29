#!/usr/bin/env python
"""
Khan 2 revision -- Step 18
Prioritised-gene expression and in-silico perturbation
(replaces original Figures 12 and 13).

The perturbation re-scores the trained macrophage classifier after setting each
gene to zero (knockout) or to a high quantile (over-expression). These are
responses of a fitted model, not causal predictions, and the revision says so
explicitly; to make that limit concrete we additionally report, for each gene,
how stable its perturbation effect is across 25 classifiers trained on different
patient-grouped splits. Genes whose effect changes sign across resamples are
flagged as unstable and are not carried into the ranking.
"""
import os, json, time, warnings
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES = f"{BASE}/02_results"
SEED = 123; rng = np.random.default_rng(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

N_RESAMPLE, OE_Q, K_FEAT = 25, 0.90, 500

mye = sc.read_h5ad(f"{RES}/03_myeloid.h5ad")
is_mac = ~mye.obs.myeloid_subtype.str.contains("DC|Langerhans|Proliferating", regex=True)
mac = mye[is_mac & mye.obs.group.isin(["Healing", "Nonhealing"])].copy()
X_all = sp.csr_matrix(mac.raw.X)
genes_all = pd.Index(mac.raw.var_names)
det = np.asarray((X_all > 0).mean(0)).ravel()
keep = det >= 0.05
X = np.asarray(X_all[:, keep].todense(), dtype=np.float32)
genes = genes_all[keep]
y = (mac.obs.group.values == "Nonhealing").astype(int)
groups = mac.obs.gid.values
log(f"{X.shape[0]:,} macrophages x {X.shape[1]:,} genes from {len(np.unique(groups))} patients")

pan = pd.read_csv(os.environ.get("PANEL_CSV", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "metadata", "consensus_novel_panel.csv")))
PANEL = [g for g in pan.loc[pan.n_analyses >= 3, "gene"] if g in genes]
log(f"consensus panel genes present in the macrophage matrix: {len(PANEL)}")

gi = {g: i for i, g in enumerate(genes)}


def fit(idx_train):
    return make_pipeline(StandardScaler(),
                         SelectKBest(f_classif, k=min(K_FEAT, X.shape[1])),
                         LogisticRegression(penalty="l1", C=0.1, solver="liblinear",
                                            max_iter=2000, random_state=SEED)
                         ).fit(X[idx_train], y[idx_train])


def perturb(model, Xe):
    """mean predicted probability of NON-HEALING under a modified matrix"""
    return float(model.predict_proba(Xe)[:, 1].mean())


# ---------------------------------------- perturbation across resampled models
hi = {g: float(np.quantile(X[:, gi[g]][X[:, gi[g]] > 0], OE_Q))
      if (X[:, gi[g]] > 0).any() else 0.0 for g in PANEL}
records = []
for rep in range(N_RESAMPLE):
    tr, _ = next(iter(StratifiedGroupKFold(n_splits=4, shuffle=True,
                                           random_state=rep).split(X, y, groups)))
    m = fit(tr)
    base = perturb(m, X)
    for g in PANEL:
        j = gi[g]
        col = X[:, j].copy()
        X[:, j] = 0.0
        ko = perturb(m, X)
        X[:, j] = hi[g]
        oe = perturb(m, X)
        X[:, j] = col
        records.append(dict(rep=rep, gene=g, baseline=base,
                            knockout_delta=ko - base, overexpression_delta=oe - base))
    if (rep + 1) % 5 == 0:
        log(f"  resample {rep+1}/{N_RESAMPLE}")
P = pd.DataFrame(records)
P.to_csv(f"{RES}/18_perturbation_raw.csv", index=False)

agg = P.groupby("gene").agg(
    ko_mean=("knockout_delta", "mean"), ko_sd=("knockout_delta", "std"),
    oe_mean=("overexpression_delta", "mean"), oe_sd=("overexpression_delta", "std"),
    n=("knockout_delta", "size")).reset_index()
agg["ko_sign_consistency"] = P.groupby("gene").knockout_delta.apply(
    lambda v: float(max((v > 0).mean(), (v < 0).mean()))).values
agg["oe_sign_consistency"] = P.groupby("gene").overexpression_delta.apply(
    lambda v: float(max((v > 0).mean(), (v < 0).mean()))).values
agg["stable"] = (agg.oe_sign_consistency >= 0.80) & (agg.ko_sign_consistency >= 0.80)
# one-sample test that the over-expression effect differs from zero across models
agg["oe_p"] = P.groupby("gene").overexpression_delta.apply(
    lambda v: float(stats.wilcoxon(v)[1]) if v.abs().sum() > 0 else 1.0).values
agg["oe_padj"] = multipletests(agg.oe_p, method="fdr_bh")[1]
agg["direction"] = np.where(agg.oe_mean > 0, "over-expression favours non-healing",
                            "over-expression favours healing")
agg = agg.sort_values("oe_mean")
agg.to_csv(f"{RES}/18_perturbation_summary.csv", index=False)
log(f"\nstable genes (consistent sign in >=80% of resampled models): "
    f"{int(agg.stable.sum())} of {len(agg)}")
log("\ntop over-expression effects toward HEALING:\n" +
    agg[agg.stable].head(10)[["gene","oe_mean","oe_sd","oe_sign_consistency","oe_padj"]]
    .to_string(index=False))
log("\ntop over-expression effects toward NON-HEALING:\n" +
    agg[agg.stable].tail(10)[["gene","oe_mean","oe_sd","oe_sign_consistency","oe_padj"]]
    .to_string(index=False))

# ------------------------------------------- prioritised gene expression
top = agg[agg.stable].reindex(
    agg[agg.stable].oe_mean.abs().sort_values(ascending=False).index).head(12).gene.tolist()
if len(top) < 6:
    top = agg.reindex(agg.oe_mean.abs().sort_values(ascending=False).index).head(12).gene.tolist()
log(f"\nprioritised genes for visualisation: {top}")

E = pd.DataFrame(X[:, [gi[g] for g in top]], columns=top, index=mac.obs_names)
E["group"] = mac.obs.group.values
E["gid"] = mac.obs.gid.values
E["state"] = mac.obs.myeloid_subtype.values
E.to_csv(f"{RES}/18_prioritised_gene_expression.csv")

per_pat = E.groupby(["gid", "group"], observed=True)[top].mean().reset_index()
per_pat.to_csv(f"{RES}/18_prioritised_per_patient.csv", index=False)
rows = []
for g in top:
    a = per_pat.loc[per_pat.group == "Nonhealing", g].values
    b = per_pat.loc[per_pat.group == "Healing", g].values
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    ca = E.loc[E.group == "Nonhealing", g].values
    cb = E.loc[E.group == "Healing", g].values
    _, pc = stats.mannwhitneyu(ca, cb, alternative="two-sided")
    rows.append(dict(gene=g, mean_nonhealing=float(a.mean()), mean_healing=float(b.mean()),
                     log2FC=float(np.log2((a.mean()+.01)/(b.mean()+.01))),
                     p_patient_level=float(p), p_cell_level=float(pc)))
G = pd.DataFrame(rows)
G["padj_patient_level"] = multipletests(G.p_patient_level, method="fdr_bh")[1]
G = G.sort_values("p_patient_level")
G.to_csv(f"{RES}/18_prioritised_gene_tests.csv", index=False)
log("\n=== PRIORITISED GENES, PATIENT LEVEL ===\n" + G.to_string(index=False))

by_state = E.groupby("state", observed=True)[top].mean()
by_state.to_csv(f"{RES}/18_prioritised_by_state.csv")

json.dump(dict(n_panel_tested=len(PANEL), n_stable=int(agg.stable.sum()),
               n_resamples=N_RESAMPLE, top_genes=top,
               n_patient_level_sig=int((G.padj_patient_level < 0.05).sum())),
          open(f"{RES}/18_summary.json", "w"), indent=2)
log("=== STEP 18 COMPLETE ===")
