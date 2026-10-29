#!/usr/bin/env python
"""
Khan 2 revision -- Step 07b
Stability of the per-cell macrophage classifier.

Step 07 returned a pooled out-of-fold AUC well below the published 0.88 and a
non-significant permutation test. Because that is a substantive change to a
headline claim, this script re-estimates it several independent ways before the
number is used anywhere:

  A. repeated stratified grouped CV over 25 random seeds (distribution of AUC)
  B. leave-one-patient-out CV (the most conservative grouped scheme)
  C. the same classifier evaluated WITHOUT grouping (naive cell-level split),
     which reproduces the optimistic estimate and shows where 0.88 comes from
  D. sensitivity to the classifier family and to the number of features
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     LeaveOneGroupOut)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
warnings.filterwarnings("ignore")

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES  = f"{BASE}/02_results"
SEED = 123
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

mye = sc.read_h5ad(f"{RES}/03_myeloid.h5ad")
is_mac = ~mye.obs.myeloid_subtype.str.contains("DC|Langerhans|Proliferating", regex=True)
mac = mye[is_mac & mye.obs.group.isin(["Healing", "Nonhealing"])].copy()
Xall = sp.csr_matrix(mac.raw.X if mac.raw is not None else mac.X)
gsym = pd.Index(mac.raw.var_names if mac.raw is not None else mac.var_names)
det = np.asarray((Xall > 0).mean(0)).ravel()
keep = det >= 0.05
X = np.asarray(Xall[:, keep].todense(), dtype=np.float32)
y = (mac.obs.group.values == "Nonhealing").astype(int)
groups = mac.obs.gid.values
log(f"{X.shape[0]:,} macrophages x {X.shape[1]:,} genes, "
    f"{len(np.unique(groups))} patients ({y.sum()} NH cells / {len(y)-y.sum()} H cells)")


def mk(model="l1", k=500):
    clf = {"l1": LogisticRegression(penalty="l1", C=0.1, solver="liblinear",
                                    max_iter=2000, random_state=SEED),
           "l2": LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=SEED),
           "rf": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=8)}[model]
    return make_pipeline(StandardScaler(), SelectKBest(f_classif, k=min(k, X.shape[1])), clf)


def grouped_auc(seed, model="l1", k=500, n_splits=4):
    oof = np.full(len(y), np.nan)
    try:
        for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                           random_state=seed).split(X, y, groups):
            oof[te] = mk(model, k).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    except Exception:
        return np.nan
    ok = ~np.isnan(oof)
    return roc_auc_score(y[ok], oof[ok]) if len(np.unique(y[ok])) == 2 else np.nan


# ---------------------------------------- A. repeated grouped CV over 25 seeds
A_ = np.array([grouped_auc(s) for s in range(25)])
A_ = A_[~np.isnan(A_)]
log(f"A. repeated grouped CV (25 seeds): mean AUC {A_.mean():.3f}, sd {A_.std():.3f}, "
    f"range {A_.min():.3f}-{A_.max():.3f}, median {np.median(A_):.3f}")

# ------------------------------------------------ B. leave-one-patient-out CV
oof = np.full(len(y), np.nan)
for tr, te in LeaveOneGroupOut().split(X, y, groups):
    if len(np.unique(y[tr])) < 2: continue
    oof[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
ok = ~np.isnan(oof)
B_cell = roc_auc_score(y[ok], oof[ok])
pat = pd.DataFrame({"gid": groups[ok], "p": oof[ok], "y": y[ok]}).groupby("gid").agg(
    mean_p=("p", "mean"), y=("y", "first"))
B_pat = roc_auc_score(pat.y, pat.mean_p)
log(f"B. leave-one-patient-out: per-cell AUC {B_cell:.3f}; "
    f"patient-level AUC from the same predictions {B_pat:.3f} (n={len(pat)})")
pat.to_csv(f"{RES}/07b_loo_patient_predictions.csv")

# --------------------------------- C. ungrouped (naive) cell-level split
oof_n = np.full(len(y), np.nan)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
    oof_n[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
C_ = roc_auc_score(y, oof_n)
log(f"C. NAIVE cell-level CV (cells from one patient in both train and test): "
    f"AUC {C_:.3f}  <-- the optimistic estimate")

# ------------------------------------------- D. model / feature sensitivity
rows = []
for model in ["l1", "l2", "rf"]:
    for k in [100, 500, 2000]:
        a = np.array([grouped_auc(s, model, k) for s in range(5)])
        a = a[~np.isnan(a)]
        if len(a):
            rows.append(dict(model=model, k_features=k, mean_auc=float(a.mean()),
                             sd_auc=float(a.std()), n_seeds=len(a)))
            log(f"D. {model:3s} k={k:5d}: AUC {a.mean():.3f} +/- {a.std():.3f}")
D_ = pd.DataFrame(rows)
D_.to_csv(f"{RES}/07b_model_sensitivity.csv", index=False)

pd.DataFrame({"seed": range(len(A_)), "auc": A_}).to_csv(
    f"{RES}/07b_repeated_grouped_cv.csv", index=False)

summary = dict(
    n_macrophages=int(X.shape[0]), n_patients=int(len(np.unique(groups))),
    repeated_grouped_cv_mean=float(A_.mean()), repeated_grouped_cv_sd=float(A_.std()),
    repeated_grouped_cv_median=float(np.median(A_)),
    repeated_grouped_cv_min=float(A_.min()), repeated_grouped_cv_max=float(A_.max()),
    leave_one_patient_out_percell_auc=float(B_cell),
    leave_one_patient_out_patient_auc=float(B_pat),
    naive_ungrouped_auc=float(C_),
    published_auc=0.88)
json.dump(summary, open(f"{RES}/07b_summary.json", "w"), indent=2)
log("\n=== STEP 07b COMPLETE ===\n" + json.dumps(summary, indent=2))
