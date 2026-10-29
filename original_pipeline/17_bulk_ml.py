#!/usr/bin/env python
"""
Khan 2 revision -- Step 17
Bulk and feature-set machine learning (replaces original Figure 9).

Reproduces the published comparison -- apparent versus cross-validated versus
leakage-free performance, and the novel panel against a canonical wound-gene set
-- on the bulk cohort, with feature selection performed strictly inside each
training fold and a label-permutation null for every estimate.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from khan_bulk import load_counts, dfu_groups, cpm_log, gene_annotation

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES = f"{BASE}/02_results"
SEED = 123; rng = np.random.default_rng(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
N_PERM = 500

ann = gene_annotation()
C, meta = load_counts("GSE134431", ann)
g = dfu_groups("GSE134431", meta)
ulcer = g[g.isin(["Healing", "Nonhealing"])].index
L = cpm_log(C)[ulcer].T
y = (g.loc[ulcer] == "Nonhealing").astype(int).values
L = L.loc[:, L.var() > 0]
log(f"bulk matrix {L.shape}; {y.sum()} non-healing / {len(y)-y.sum()} healing")

pan = pd.read_csv(os.environ.get("PANEL_CSV", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "metadata", "consensus_novel_panel.csv")))
PANEL = [x for x in pan.loc[pan.n_analyses >= 3, "gene"] if x in L.columns]
CANON = [x for x in ["COL1A1","COL3A1","FN1","VEGFA","TGFB1","MMP1","MMP2","MMP9","TIMP1",
                     "IL1B","IL6","TNF","CXCL8","PDGFB","FGF2","EGF","HIF1A","ACTA2",
                     "CD68","ARG1","NOS2","MRC1","CCL2","SPP1","LOX","ELN","KRT14","KRT10"]
         if x in L.columns]
log(f"consensus panel genes present: {len(PANEL)}; canonical wound genes present: {len(CANON)}")

MODELS = {
 "L1 logistic": LogisticRegression(penalty="l1", C=0.5, solver="liblinear",
                                   max_iter=5000, random_state=SEED),
 "L2 logistic": LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=SEED),
 "Random forest": RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=8),
 "Gradient boosting": GradientBoostingClassifier(random_state=SEED),
}


def pipe(model, k=None):
    steps = [StandardScaler()]
    if k: steps.append(SelectKBest(f_classif, k=k))
    steps.append(MODELS[model])
    return make_pipeline(*steps)


def evaluate(X, yy, model, k=None, label=""):
    """apparent (resubstitution), 5-fold CV, and leave-one-out -- all with any
    feature selection inside the fold."""
    Xv = X.values if hasattr(X, "values") else X
    m = pipe(model, k).fit(Xv, yy)
    apparent = roc_auc_score(yy, m.predict_proba(Xv)[:, 1])
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p_cv = cross_val_predict(pipe(model, k), Xv, yy, cv=cv, method="predict_proba")[:, 1]
    auc_cv = roc_auc_score(yy, p_cv)
    p_loo = cross_val_predict(pipe(model, k), Xv, yy, cv=LeaveOneOut(), method="predict_proba")[:, 1]
    auc_loo = roc_auc_score(yy, p_loo)
    return dict(feature_set=label, model=model, n_features=Xv.shape[1],
                apparent_auc=float(apparent), cv5_auc=float(auc_cv),
                leaveoneout_auc=float(auc_loo)), p_loo


rows, roc_store = [], {}
SETS = {"all genes (k=200 selected in-fold)": (L, 200),
        "consensus panel": (L[PANEL], None),
        "canonical wound genes": (L[CANON], None)}
for label, (X, k) in SETS.items():
    for model in MODELS:
        r, p_loo = evaluate(X, y, model, k, label)
        rows.append(r)
        roc_store[(label, model)] = p_loo
        log(f"  {label:34s} {model:18s} apparent {r['apparent_auc']:.2f}  "
            f"5-fold {r['cv5_auc']:.2f}  LOO {r['leaveoneout_auc']:.2f}")
perf = pd.DataFrame(rows)
perf.to_csv(f"{RES}/17_bulk_ml_performance.csv", index=False)

# ------------------------------------------------------- permutation nulls
log(f"\npermutation null ({N_PERM} label shuffles) for the leave-one-out estimates ...")
nulls = {}
for label, (X, k) in SETS.items():
    Xv = X.values
    best = perf[perf.feature_set == label].nlargest(1, "leaveoneout_auc").iloc[0]
    obs = best.leaveoneout_auc
    d = []
    for i in range(N_PERM):
        yp = rng.permutation(y)
        try:
            pp = cross_val_predict(pipe(best.model, k), Xv, yp, cv=LeaveOneOut(),
                                   method="predict_proba")[:, 1]
            d.append(roc_auc_score(yp, pp))
        except Exception:
            continue
    d = np.array(d)
    nulls[label] = dict(model=best.model, observed=float(obs),
                        null_mean=float(d.mean()), null_sd=float(d.std()),
                        p=float((d >= obs).mean()))
    log(f"  {label:34s} {best.model:18s} LOO {obs:.2f} vs null "
        f"{d.mean():.2f}+/-{d.std():.2f}, P = {nulls[label]['p']:.3f}")
json.dump(nulls, open(f"{RES}/17_permutation_nulls.json", "w"), indent=2)

# --------------------------------------------------------------- ROC + importance
best_overall = perf.nlargest(1, "leaveoneout_auc").iloc[0]
p_loo = roc_store[(best_overall.feature_set, best_overall.model)]
fpr, tpr, _ = roc_curve(y, p_loo)
pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(f"{RES}/17_bulk_roc.csv", index=False)

Xb, kb = SETS[best_overall.feature_set]
mfit = pipe(best_overall.model, kb).fit(Xb.values, y)
names = np.array(Xb.columns)
if kb:
    names = names[mfit.named_steps["selectkbest"].get_support()]
clf = list(mfit.named_steps.values())[-1]
imp = getattr(clf, "feature_importances_", None)
if imp is None:
    imp = np.abs(clf.coef_[0])
FI = pd.DataFrame({"gene": names, "importance": imp})
FI["in_consensus_panel"] = FI.gene.isin(PANEL)
FI["in_canonical_set"] = FI.gene.isin(CANON)
FI = FI[FI.importance > 0].sort_values("importance", ascending=False)
FI.to_csv(f"{RES}/17_bulk_feature_importance.csv", index=False)
log(f"\nbest configuration: {best_overall.feature_set} / {best_overall.model} "
    f"(LOO AUC {best_overall.leaveoneout_auc:.2f})")
log("top 15 features:\n" + FI.head(15).to_string(index=False))

json.dump(dict(n_samples=int(len(y)), n_nonhealing=int(y.sum()),
               n_panel=len(PANEL), n_canonical=len(CANON),
               best_feature_set=best_overall.feature_set,
               best_model=best_overall.model,
               best_apparent=float(best_overall.apparent_auc),
               best_cv5=float(best_overall.cv5_auc),
               best_loo=float(best_overall.leaveoneout_auc),
               permutation=nulls),
          open(f"{RES}/17_summary.json", "w"), indent=2)
log("=== STEP 17 COMPLETE ===")
