"""
R2_07 -- Bulk cohorts: confidence intervals for classifier AUCs and directional
replication (Editor comment 8), covariate-adjusted bulk differential expression
(comment 3), and the external transportability test.

* Bulk classifiers re-run exactly as original-pipeline step 17/17b (log2 CPM; feature
  selection inside folds; L1 logistic C=0.5 etc.); leave-one-out predictions are
  kept so DeLong CIs can be computed; external test = train on all GSE134431,
  apply untouched to GSE143735.
* Directional concordance and statistical replication rates with exact
  (Clopper-Pearson) 95% CIs, recomputed from the DESeq2 tables.
* Bulk DE with available covariates: GSE134431 ~ biopsy timepoint + group
  (week 0 / week 4 / not stated); GSE143735 ~ age + sex + group.
"""
import sys, os, json; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from r2_common import *
import deseq_engine


def delong_ci(yt, s, alpha=0.05):
    yt = np.asarray(yt); s = np.asarray(s, float)
    pos, neg = s[yt == 1], s[yt == 0]; m, n = len(pos), len(neg)
    V10 = np.array([(np.sum(p > neg) + 0.5 * np.sum(p == neg)) / n for p in pos])
    V01 = np.array([(np.sum(pos > q) + 0.5 * np.sum(pos == q)) / m for q in neg])
    auc = V10.mean(); se = np.sqrt(V10.var(ddof=1) / m + V01.var(ddof=1) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(auc), float(max(0, auc - z * se)), float(min(1, auc + z * se))

ann = gene_annotation()
C1, m1 = load_counts("GSE134431", ann); g1 = dfu_groups("GSE134431", m1)
C2, m2 = load_counts("GSE143735", ann); g2 = dfu_groups("GSE143735", m2)
u1 = g1[g1.isin(["Healing", "Nonhealing"])].index; u2 = g2[g2.isin(["Healing", "Nonhealing"])].index
L1 = cpm_log(C1)[u1].T; y1 = (g1.loc[u1] == "Nonhealing").astype(int).values
L1 = L1.loc[:, L1.var() > 0]
L2 = cpm_log(C2)[u2].T; y2 = (g2.loc[u2] == "Nonhealing").astype(int).values
pan = pd.read_csv(PANEL_CSV)
PANEL = [x for x in pan.loc[pan.n_analyses >= 3, "gene"] if x in L1.columns]
CANON = [x for x in ["COL1A1","COL3A1","FN1","VEGFA","TGFB1","MMP1","MMP2","MMP9","TIMP1","IL1B","IL6","TNF","CXCL8",
                     "PDGFB","FGF2","EGF","HIF1A","ACTA2","CD68","ARG1","NOS2","MRC1","CCL2","SPP1","LOX","ELN","KRT14","KRT10"]
         if x in L1.columns]
MODELS = {"L1 logistic": lambda: LogisticRegression(penalty="l1", C=0.5, solver="liblinear", max_iter=5000, random_state=SEED),
          "L2 logistic": lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=SEED),
          "Random forest": lambda: RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=2),
          "Gradient boosting": lambda: GradientBoostingClassifier(random_state=SEED)}
def pipe(model, k=None):
    steps = [StandardScaler()] + ([SelectKBest(f_classif, k=k)] if k else []) + [MODELS[model]()]
    return make_pipeline(*steps)
SETS = {"all genes (k=200 selected in-fold)": (L1, 200), "consensus panel": (L1[PANEL], None),
        "canonical wound genes": (L1[CANON], None)}
rows = []
prev = pd.read_csv(f"{RES}/17_bulk_ml_performance.csv")
for label, (Xd, k) in SETS.items():
    for model in MODELS:
        Xv = Xd.values
        p_loo = cross_val_predict(pipe(model, k), Xv, y1, cv=LeaveOneOut(), method="predict_proba")[:, 1]
        p_cv = cross_val_predict(pipe(model, k), Xv, y1, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        a, lo, hi = delong_ci(y1, p_loo); a5, lo5, hi5 = delong_ci(y1, p_cv)
        pr = prev[(prev.feature_set == label) & (prev.model == model)]
        rows.append(dict(cohort="GSE134431 (internal, n=13)", feature_set=label, model=model,
                         cv5_auc=round(a5, 3), cv5_ci_low=round(lo5, 3), cv5_ci_high=round(hi5, 3),
                         loo_auc=round(a, 3), loo_ci_low=round(lo, 3), loo_ci_high=round(hi, 3),
                         previous_loo_auc=round(float(pr.leaveoneout_auc.iloc[0]), 3) if len(pr) else np.nan))
        print(rows[-1], flush=True)
ML = pd.DataFrame(rows); ML.to_csv(f"{OUT}/R2_07_bulk_classifier_AUC_CIs.csv", index=False)

ext = []
for label, (Xtr, k) in SETS.items():
    cols = [c for c in Xtr.columns if c in L2.columns]
    steps = [StandardScaler()] + ([SelectKBest(f_classif, k=min(k, len(cols)))] if k else []) + \
            [LogisticRegression(penalty="l1", C=0.5, solver="liblinear", max_iter=5000, random_state=SEED)]
    mdl = make_pipeline(*steps).fit(Xtr[cols].values, y1)
    a, lo, hi = delong_ci(y2, mdl.predict_proba(L2[cols].values)[:, 1])
    ext.append(dict(feature_set=label, n_shared_genes=len(cols), external_auc=round(a, 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
                    note="trained on all 13 GSE134431 ulcers; applied without retraining to 9 GSE143735 forearm samples (5 healers, 4 non-healers)"))
    print(ext[-1], flush=True)
EXT = pd.DataFrame(ext); EXT.to_csv(f"{OUT}/R2_07_external_transfer_AUC_CIs.csv", index=False)

# -------------------------------------------------- directional replication CIs
disc = pd.read_csv(f"{RES}/06_discovery_DESeq2_GSE134431.csv", index_col=0)
vald = pd.read_csv(f"{RES}/06_validation_DESeq2_GSE143735.csv", index_col=0)
J = pd.read_csv(f"{RES}/06_discovery_vs_validation_joint.csv", index_col=0)
levels = {"whole transcriptome": J, "nominal discovery P<0.05": J[J.disc_p < 0.05], "discovery FDR<0.05": J[J.disc_padj < 0.05]}
for t in (3, 4, 5):
    gl = pan.loc[pan.n_analyses >= t, "gene"]
    levels[f"consensus panel (>={t} analyses)"] = J[J.index.isin(gl)]
levels["consensus panel (>=3), |discovery log2FC|>1"] = J[J.index.isin(pan.loc[pan.n_analyses >= 3, "gene"]) & (J.disc_log2FC.abs() > 1)]
rr = []
for lbl, d in levels.items():
    n = len(d); k = int(d.same_sign.sum()); r = int(d.statistically_replicated.sum())
    ci_k = stats.binomtest(k, n).proportion_ci(0.95, "exact") if n else (np.nan, np.nan)
    ci_r = stats.binomtest(r, n).proportion_ci(0.95, "exact") if n else (np.nan, np.nan)
    rr.append(dict(gene_set=lbl, n_genes=n, same_direction_n=k, same_direction_pct=round(100 * k / n, 1),
                   same_direction_ci_low=round(100 * ci_k.low, 1), same_direction_ci_high=round(100 * ci_k.high, 1),
                   binomial_p_vs_50pct=stats.binomtest(k, n, 0.5, alternative="greater").pvalue,
                   replicated_same_direction_and_P05_n=r, replicated_pct=round(100 * r / n, 1),
                   replicated_ci_low=round(100 * ci_r.low, 1), replicated_ci_high=round(100 * ci_r.high, 1),
                   expected_replicated_pct_by_chance=2.5))
RR = pd.DataFrame(rr); RR.to_csv(f"{OUT}/R2_07_directional_replication_CIs.csv", index=False)
print(RR.to_string(index=False))

# ---------------------------------------- covariate-adjusted bulk DE (comment 3)
def de(C, sub, Xd, name):
    K = C[sub].T.round().astype(int)
    K = K.loc[:, (K > 0).sum(0) >= max(3, int(0.2 * len(sub)))]
    r = deseq_engine.deseq(K.T.values, Xd, coef=Xd.shape[1] - 1)
    T = pd.DataFrame({k: r[k] for k in ["baseMean", "log2FoldChange", "lfcSE", "pvalue", "padj"]}, index=K.columns).sort_values("pvalue")
    T["ci_low"] = T.log2FoldChange - 1.96 * T.lfcSE; T["ci_high"] = T.log2FoldChange + 1.96 * T.lfcSE
    T.to_csv(f"{OUT}/R2_07_bulk_DE_{name}.csv"); return T
tp = m1.loc[u1, "title"].map(lambda t: "W0" if "W0" in t else ("W4" if "W4" in t else "P2"))
yv1 = (g1.loc[u1] == "Nonhealing").astype(float).values
X_unadj1 = np.column_stack([np.ones(len(u1)), yv1])
X_tp = np.column_stack([np.ones(len(u1)), (tp == "W4").astype(float).values, (tp == "P2").astype(float).values, yv1])
T1u = de(C1, u1, X_unadj1, "GSE134431_unadjusted"); T1a = de(C1, u1, X_tp, "GSE134431_timepoint_adjusted")
age = pd.to_numeric(m2.loc[u2, "age"]).values; sexm = (m2.loc[u2, "Sex"] == "M").astype(float).values
yv2 = (g2.loc[u2] == "Nonhealing").astype(float).values
T2u = de(C2, u2, np.column_stack([np.ones(len(u2)), yv2]), "GSE143735_unadjusted")
T2a = de(C2, u2, np.column_stack([np.ones(len(u2)), (age - age.mean()) / age.std(), sexm, yv2]), "GSE143735_age_sex_adjusted")
def cmp(a, b):
    j = a.join(b, rsuffix="_adj", how="inner").dropna(subset=["log2FoldChange", "log2FoldChange_adj"])
    j = j[(j.log2FoldChange.abs() < 10) & (j.log2FoldChange_adj.abs() < 10)]
    return float(stats.spearmanr(j.log2FoldChange, j.log2FoldChange_adj)[0])
tp_tab = pd.crosstab(g1.loc[u1], tp).to_dict()
BS = dict(GSE134431=dict(timepoint_by_group=tp_tab, n_FDR05_unadjusted=int((T1u.padj < 0.05).sum()),
                         n_FDR05_timepoint_adjusted=int((T1a.padj < 0.05).sum()), lfc_spearman=cmp(T1u, T1a),
                         overlap=int(len(set(T1u.index[T1u.padj < 0.05]) & set(T1a.index[T1a.padj < 0.05])))),
          GSE143735=dict(age_healing=age[yv2 == 0].tolist(), age_nonhealing=age[yv2 == 1].tolist(),
                         age_mannwhitney_p=float(stats.mannwhitneyu(age[yv2 == 0], age[yv2 == 1]).pvalue),
                         n_FDR05_unadjusted=int((T2u.padj < 0.05).sum()), n_FDR05_age_sex_adjusted=int((T2a.padj < 0.05).sum()),
                         lfc_spearman=cmp(T2u, T2a)))
# panel replication after covariate adjustment in both cohorts
jj = pd.DataFrame({"d": T1a.log2FoldChange, "v": T2a.log2FoldChange, "vp": T2a.pvalue}).dropna()
pp = jj[jj.index.isin(pan.loc[pan.n_analyses >= 3, "gene"])]
k = int((np.sign(pp.d) == np.sign(pp.v)).sum()); ci = stats.binomtest(k, len(pp)).proportion_ci(0.95, "exact")
rep_adj = int(((np.sign(pp.d) == np.sign(pp.v)) & (pp.vp < 0.05)).sum())
BS["panel_after_adjustment"] = dict(n=len(pp), same_direction=k, pct=round(100 * k / len(pp), 1), ci=[round(100 * ci.low, 1), round(100 * ci.high, 1)],
                                    binomial_p=float(stats.binomtest(k, len(pp), 0.5, alternative="greater").pvalue),
                                    replicated_P05=rep_adj)
json.dump(BS, open(f"{OUT}/R2_07_bulk_covariate_summary.json", "w"), indent=2)
print(json.dumps(BS, indent=1)); print(ML.to_string(index=False)); print(EXT.to_string(index=False))
