"""
R2_06 -- Confidence intervals for every classifier AUC (Editor comment 8) and a
donor-aware re-evaluation of the per-cell macrophage classifier (comment 1).

Re-runs the per-cell classifier of original-pipeline step 07 exactly (StandardScaler ->
SelectKBest(f_classif, k=500) -> L1 logistic C=0.1, liblinear; 4-fold
StratifiedGroupKFold, seeds 0-4; genes detected in >=5% of macrophages; input =
log-normalised expression from 03_myeloid.h5ad) and adds
  * library-cluster bootstrap 95% CIs for per-cell AUCs (2,000 resamples of
    libraries within group) -- cells are not independent, so cells are never
    resampled on their own
  * DeLong 95% CIs and stratified bootstrap CIs for library-level AUCs
  * donor-grouped CV: folds built on putative donors so that two libraries from
    one person can never be split between training and test
  * foot-only evaluation
  * the pseudobulk library-level classifier rebuilt from RAW-count pseudobulk
"""
import sys, os, json; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from r2_common import *
import h5ad_lite as H

rng = np.random.default_rng(SEED)
f = H.open_h5ad(os.environ.get("MYELOID_H5AD", f"{H5AD_DIR}/03_myeloid.h5ad"))
obs = pd.read_csv(f"{AUX}/myeloid_obs_with_umap.csv", index_col=0)
Xr = H.read_csr(f["raw"]["X"], shape=(len(obs), 25683))
is_mac = ~obs.myeloid_subtype.str.contains("DC|Langerhans|Proliferating", regex=True)
sel = (is_mac & obs.group.isin(["Healing", "Nonhealing"])).values
Xall = Xr[sel]; o = obs[sel]
det = np.asarray((Xall > 0).mean(0)).ravel()
X = np.asarray(Xall[:, det >= 0.05].todense(), dtype=np.float32)
y = (o.group.values == "Nonhealing").astype(int)
lib = o.gid.values
print("matrix", X.shape, "cells per group", np.bincount(y), flush=True)

cov = pd.read_csv(f"{OUT}/R2_04_library_covariates.csv", index_col=0)
dsum = json.load(open(f"{OUT}/R2_04_summary.json"))
donor = {g: v.split(" ")[0] for g, v in dsum["donor_groups"].items()}
donor_arr = np.array([donor[g] for g in lib])

def pipe():
    return make_pipeline(StandardScaler(), SelectKBest(f_classif, k=min(500, X.shape[1])),
                         LogisticRegression(penalty="l1", C=0.1, solver="liblinear", max_iter=2000, random_state=SEED))

def grouped_oof(groups, seed, rows=None):
    rows = np.arange(len(y)) if rows is None else rows
    o_ = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(X[rows], y[rows], groups[rows]):
        o_[rows[te]] = pipe().fit(X[rows[tr]], y[rows[tr]]).predict_proba(X[rows[te]])[:, 1]
    return o_

# ---------------------------------------------------------------- AUC CIs
def delong_ci(yt, s, alpha=0.05):
    yt = np.asarray(yt); s = np.asarray(s, float)
    pos, neg = s[yt == 1], s[yt == 0]; m, n = len(pos), len(neg)
    V10 = np.array([(np.sum(p > neg) + 0.5 * np.sum(p == neg)) / n for p in pos])
    V01 = np.array([(np.sum(pos > q) + 0.5 * np.sum(pos == q)) / m for q in neg])
    auc = V10.mean()
    se = np.sqrt(V10.var(ddof=1) / m + V01.var(ddof=1) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(auc), float(max(0, auc - z * se)), float(min(1, auc + z * se))

def strat_boot_ci(yt, s, B=2000):
    yt = np.asarray(yt); s = np.asarray(s); i1, i0 = np.where(yt == 1)[0], np.where(yt == 0)[0]
    a = []
    for _ in range(B):
        ii = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        a.append(roc_auc_score(yt[ii], s[ii]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def cluster_boot_ci(score, groups=lib, B=2000):
    """resample libraries (within outcome group), keep all their cells"""
    ok = ~np.isnan(score)
    gl = pd.Series(y[ok], index=np.arange(ok.sum())).groupby(groups[ok]).first()
    pos_l, neg_l = gl.index[gl == 1].values, gl.index[gl == 0].values
    idx = {g: np.where(groups[ok] == g)[0] for g in gl.index}
    ys, ss = y[ok], score[ok]
    a = []
    for _ in range(B):
        chosen = np.concatenate([rng.choice(pos_l, len(pos_l)), rng.choice(neg_l, len(neg_l))])
        ii = np.concatenate([idx[g] for g in chosen])
        a.append(roc_auc_score(ys[ii], ss[ii]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

rows = []
def record(name, unit, auc, lo, hi, method, n_lib, note=""):
    rows.append(dict(evaluation=name, unit=unit, auc=round(auc, 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
                     ci_method=method, n_libraries=n_lib, note=note))
    print(rows[-1], flush=True)

# (1) per-cell, library-grouped 4-fold CV, seeds 0-4 (published estimator)
oofs, seed_auc = [], []
for sd in range(5):
    o_ = grouped_oof(lib, sd); oofs.append(o_); k = ~np.isnan(o_)
    seed_auc.append(roc_auc_score(y[k], o_[k]))
oof = np.nanmean(np.vstack(oofs), 0)
lo, hi = cluster_boot_ci(oof)
record("per-cell, library-grouped 4-fold CV (mean of 5 split seeds)", "cell", float(np.mean(seed_auc)), lo, hi,
       "library-cluster bootstrap (2,000)", 19, f"seed AUCs {np.round(seed_auc, 3).tolist()}; CI computed on the seed-averaged out-of-fold scores")

# library-level aggregation of those predictions
P = pd.DataFrame({"lib": lib, "p": oof, "y": y}).groupby("lib").agg(p=("p", "mean"), y=("y", "first"))
a, l1, h1 = delong_ci(P.y, P.p); bl, bh_ = strat_boot_ci(P.y.values, P.p.values)
record("library outcome from aggregated per-cell predictions", "library", a, l1, h1, "DeLong", 19, f"stratified bootstrap CI {bl:.2f}-{bh_:.2f}")

# (2) leave-one-library-out
o_lolo = np.full(len(y), np.nan)
for tr, te in LeaveOneGroupOut().split(X, y, lib):
    o_lolo[te] = pipe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
lo, hi = cluster_boot_ci(o_lolo)
record("per-cell, leave-one-library-out", "cell", roc_auc_score(y, o_lolo), lo, hi, "library-cluster bootstrap (2,000)", 19)
PL = pd.DataFrame({"lib": lib, "p": o_lolo, "y": y}).groupby("lib").agg(p=("p", "mean"), y=("y", "first"))
a, l1, h1 = delong_ci(PL.y, PL.p); bl, bh_ = strat_boot_ci(PL.y.values, PL.p.values)
record("library outcome, leave-one-library-out", "library", a, l1, h1, "DeLong", 19, f"stratified bootstrap CI {bl:.2f}-{bh_:.2f}")

# (3) donor-grouped CV (two libraries of one putative donor never split)
oofs_d, aucs_d = [], []
for sd in range(5):
    o_ = grouped_oof(donor_arr, sd); oofs_d.append(o_); k = ~np.isnan(o_)
    aucs_d.append(roc_auc_score(y[k], o_[k]))
oof_d = np.nanmean(np.vstack(oofs_d), 0)
lo, hi = cluster_boot_ci(oof_d, groups=donor_arr)
record("per-cell, putative-donor-grouped 4-fold CV (5 seeds)", "cell", float(np.mean(aucs_d)), lo, hi,
       "donor-cluster bootstrap (2,000)", 19, f"{len(set(donor_arr))} putative donors; seed AUCs {np.round(aucs_d, 3).tolist()}")
PD = pd.DataFrame({"d": donor_arr, "p": oof_d, "y": y}).groupby("d").agg(p=("p", "mean"), y=("y", "first"))
a, l1, h1 = delong_ci(PD.y, PD.p)
record("donor outcome from aggregated per-cell predictions (donor-grouped CV)", "putative donor", a, l1, h1, "DeLong", 19, f"{len(PD)} putative donors")

# (4) foot libraries only, library-grouped
foot = np.where(o.tissue.values == "Foot")[0]
oofs_f, aucs_f = [], []
for sd in range(5):
    o_ = grouped_oof(lib, sd, rows=foot); oofs_f.append(o_); k = ~np.isnan(o_)
    aucs_f.append(roc_auc_score(y[k], o_[k]))
oof_f = np.nanmean(np.vstack(oofs_f), 0)
lo, hi = cluster_boot_ci(oof_f)
record("per-cell, foot libraries only, library-grouped CV (5 seeds)", "cell", float(np.mean(aucs_f)), lo, hi,
       "library-cluster bootstrap (2,000)", 14, f"seed AUCs {np.round(aucs_f, 3).tolist()}")

# (5) repeated grouped CV from the previous revision (25 splits): empirical interval
rep = pd.read_csv(f"{RES}/07b_repeated_grouped_cv.csv")
col = [c for c in rep.columns if "auc" in c.lower()][0]
record("per-cell, 25 repeated library-grouped splits (previous revision)", "cell", float(rep[col].mean()),
       float(rep[col].quantile(0.025)), float(rep[col].quantile(0.975)), "2.5th-97.5th percentile of the 25 splits", 19,
       "split-to-split variability only; not a sampling CI")
naive = pd.read_csv(f"{RES}/07b_summary.json".replace(".csv", "")) if False else json.load(open(f"{RES}/07b_summary.json"))
# (6) pseudobulk library-level classifier, rebuilt on RAW counts
PBR = pd.read_csv(f"{R2IN}/R2_02_raw_pseudobulk_macrophage.csv", index_col=0)
libs19 = cov.index[cov.group.isin(["Healing", "Nonhealing"]) & (cov.site != "PBMC") & (cov.n_mac >= 10)]
Pb = PBR[libs19].T
Pb = np.log2(Pb.div(Pb.sum(1), axis=0) * 1e6 + 1)
Pb = Pb.loc[:, Pb.var() > 0]
yb = (cov.loc[libs19, "group"] == "Nonhealing").astype(int).values
def loo_auc(M, yy, k=200):
    pr = np.zeros(len(yy))
    for tr, te in LeaveOneOut().split(M):
        p = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=min(k, M.shape[1] - 1)),
                          LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=SEED)).fit(M[tr], yy[tr])
        pr[te] = p.predict_proba(M[te])[:, 1]
    return roc_auc_score(yy, pr), pr
auc_pb, pr_pb = loo_auc(Pb.values, yb)
null = np.array([loo_auc(Pb.values, rng.permutation(yb))[0] for _ in range(200)])
a, l1, h1 = delong_ci(yb, pr_pb)
record("library-level pseudobulk classifier (raw-count pseudobulk), leave-one-out", "library", a, l1, h1, "DeLong", 19,
       f"label-permutation P = {(null >= auc_pb).mean():.3f} (200 permutations; null mean {null.mean():.2f})")

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/R2_06_classifier_AUC_CIs.csv", index=False)
pd.DataFrame({"cell": o.index, "library": lib, "putative_donor": donor_arr, "y_nonhealing": y,
              "oof_library_grouped": oof, "oof_leave_one_library_out": o_lolo, "oof_donor_grouped": oof_d}
             ).to_csv(f"{OUT}/R2_06_per_cell_out_of_fold_predictions.csv.gz", index=False)
print(R.to_string(index=False))
