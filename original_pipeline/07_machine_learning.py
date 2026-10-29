#!/usr/bin/env python
"""
Khan 2 revision -- Step 07
Macrophage-state classification, evaluated so that a cell-level result is never
presented as a clinical prediction.

Reviewer 1, comment 4: "The current model mainly reflects macrophage state
classification at the cellular level. The authors should avoid overinterpreting
it as a clinical prediction model."

Four evaluations are reported together:
  (1) per-cell, sample-grouped CV  -- what the published AUC 0.88 measures:
      can a macrophage's transcriptome be assigned to a healing class
  (2) patient-level aggregation of those same per-cell predictions -- the
      clinically relevant question, evaluated on n patients
  (3) patient-level pseudobulk classifier with leave-one-out CV
  (4) a label-permutation null for each, plus a learning curve over the number
      of training patients, showing how far this cohort is from supporting a
      deployable predictor
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
warnings.filterwarnings("ignore")

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES  = f"{BASE}/02_results"
SEED = 123; rng = np.random.default_rng(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

N_PERM, K_FEAT = 200, 500

mye = sc.read_h5ad(f"{RES}/03_myeloid.h5ad")
is_mac = ~mye.obs.myeloid_subtype.str.contains("DC|Langerhans|Proliferating", regex=True)
mac = mye[is_mac & mye.obs.group.isin(["Healing", "Nonhealing"])].copy()
log(f"macrophages in healing/non-healing samples: {mac.n_obs:,} "
    f"from {mac.obs.gid.nunique()} libraries")
log("cells per group: " + str(mac.obs.group.value_counts().to_dict()))
log("libraries per group: " +
    str(mac.obs.groupby('group', observed=True).gid.nunique().to_dict()))

Xall = mac.raw.X if mac.raw is not None else mac.X
Xall = sp.csr_matrix(Xall)
genes = pd.Index(mac.raw.var_names if mac.raw is not None else mac.var_names)
# keep genes detected in >=5% of cells to bound the feature space
det = np.asarray((Xall > 0).mean(0)).ravel()
keep = det >= 0.05
X = np.asarray(Xall[:, keep].todense(), dtype=np.float32)
genes = genes[keep]
y = (mac.obs.group.values == "Nonhealing").astype(int)
groups = mac.obs.gid.values
log(f"feature matrix {X.shape}")

def pipe():
    return make_pipeline(StandardScaler(),
                         SelectKBest(f_classif, k=min(K_FEAT, X.shape[1])),
                         LogisticRegression(penalty="l1", C=0.1, solver="liblinear",
                                            max_iter=2000, random_state=SEED))

# ------------------------------- (1) per-cell, sample-grouped cross-validation
# With 19 libraries a single fold assignment is unstable, so the estimate is
# averaged over SEEDS_CV independent grouped splits and the spread reported.
SEEDS_CV = 5

def grouped_oof(seed):
    o = np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(n_splits=4, shuffle=True,
                                       random_state=seed).split(X, y, groups):
        o[te] = pipe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return o

oof_list, seed_auc = [], []
for sd in range(SEEDS_CV):
    o = grouped_oof(sd)
    k = ~np.isnan(o)
    oof_list.append(o)
    seed_auc.append(roc_auc_score(y[k], o[k]))
oof = np.nanmean(np.vstack(oof_list), axis=0)
ok = ~np.isnan(oof)
cell_auc = float(np.mean(seed_auc))
log(f"(1) per-cell sample-grouped CV AUC = {cell_auc:.3f} "
    f"(sd {np.std(seed_auc):.3f} over {SEEDS_CV} split seeds: "
    f"{np.round(seed_auc,3).tolist()})")
fold_auc = seed_auc

# ------------------------- (2) patient-level aggregation of per-cell predictions
pat = pd.DataFrame({"gid": groups[ok], "p": oof[ok], "y": y[ok]}).groupby("gid").agg(
    mean_p=("p", "mean"), median_p=("p", "median"),
    frac_nonhealing_called=("p", lambda v: float((v > 0.5).mean())),
    n_cells=("p", "size"), y=("y", "first"))
pat.to_csv(f"{RES}/07_patient_level_aggregated_predictions.csv")
pat_auc = roc_auc_score(pat.y, pat.mean_p)
pat_auc_frac = roc_auc_score(pat.y, pat.frac_nonhealing_called)
log(f"(2) patient-level AUC from aggregated per-cell predictions = {pat_auc:.3f} "
    f"(n = {len(pat)} patients; fraction-called variant {pat_auc_frac:.3f})")

# ------------------------------------ (3) pseudobulk patient-level classifier
pb = pd.read_csv(f"{RES}/04_macrophage_pseudobulk_counts.csv", index_col=0)
pbm = pd.read_csv(f"{RES}/04_pseudobulk_sample_metadata.csv", index_col=0)
sel = pbm[pbm.group.isin(["Healing", "Nonhealing"])].index
Pb = pb.loc[sel]
Pb = np.log2(Pb.div(Pb.sum(1), axis=0) * 1e6 + 1)
yb = (pbm.loc[sel, "group"].values == "Nonhealing").astype(int)
log(f"(3) pseudobulk matrix {Pb.shape}, {yb.sum()} non-healing / {len(yb)-yb.sum()} healing")

def loo_auc(M, yy, k=200):
    preds = np.zeros(len(yy))
    for tr, te in LeaveOneOut().split(M):
        p = make_pipeline(StandardScaler(),
                          SelectKBest(f_classif, k=min(k, M.shape[1]-1)),
                          LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                             random_state=SEED)).fit(M[tr], yy[tr])
        preds[te] = p.predict_proba(M[te])[:, 1]
    return roc_auc_score(yy, preds), preds

pb_auc, pb_pred = loo_auc(Pb.values, yb)
log(f"(3) pseudobulk leave-one-out AUC = {pb_auc:.3f}")

# --------------------------------------------------- (4) permutation nulls
def perm_null_cell(n=N_PERM):
    """Null for the SAME statistic: mean grouped-CV AUC over SEEDS_CV splits,
    with labels permuted between patients (within-patient structure kept)."""
    out = []
    uniq = np.unique(groups)
    ypat = pd.Series(y, index=groups).groupby(level=0).first()
    for i in range(n):
        ymap = dict(zip(uniq, rng.permutation(ypat.loc[uniq].values)))
        yp = np.array([ymap[g] for g in groups])
        aucs = []
        for sd in range(SEEDS_CV):
            try:
                o = np.full(len(yp), np.nan)
                for tr, te in StratifiedGroupKFold(n_splits=4, shuffle=True,
                                                   random_state=sd).split(X, yp, groups):
                    o[te] = pipe().fit(X[tr], yp[tr]).predict_proba(X[te])[:, 1]
                k = ~np.isnan(o)
                if len(np.unique(yp[k])) == 2:
                    aucs.append(roc_auc_score(yp[k], o[k]))
            except Exception:
                continue
        if aucs:
            out.append(np.mean(aucs))
    return np.array(out)

null_cell = perm_null_cell(40)
p_cell = float((null_cell >= cell_auc).mean()) if len(null_cell) else float("nan")
log(f"(4) per-cell permutation null AUC {null_cell.mean():.3f} +/- {null_cell.std():.3f}; "
    f"p = {p_cell:.4f}")

null_pb = []
for i in range(N_PERM):
    a, _ = loo_auc(Pb.values, rng.permutation(yb))
    null_pb.append(a)
null_pb = np.array(null_pb)
p_pb = float((null_pb >= pb_auc).mean())
log(f"(4) pseudobulk permutation null AUC {null_pb.mean():.3f} +/- {null_pb.std():.3f}; "
    f"p = {p_pb:.4f}")

# ------------------------------------------------ learning curve over patients
uniq_g = pd.Series(y, index=groups).groupby(level=0).first()
curve = []
for n_train in range(4, len(uniq_g)):
    aucs = []
    for rep in range(15):
        tr_g = uniq_g.sample(n_train, random_state=rep*7+n_train)
        if tr_g.nunique() < 2: continue
        te_g = uniq_g.drop(tr_g.index)
        if te_g.nunique() < 2: continue
        tr = np.isin(groups, tr_g.index); te = np.isin(groups, te_g.index)
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2: continue
        try:
            m = pipe().fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        except Exception: pass
    if aucs:
        curve.append(dict(n_train_patients=n_train, mean_auc=float(np.mean(aucs)),
                          sd_auc=float(np.std(aucs)), n_reps=len(aucs)))
cur = pd.DataFrame(curve)
cur.to_csv(f"{RES}/07_learning_curve_patients.csv", index=False)
log("\n=== LEARNING CURVE (per-cell AUC vs number of training patients) ===")
log(cur.to_string(index=False))

# ------------------------------------------------------------ feature weights
final = pipe().fit(X, y)
kb = final.named_steps["selectkbest"]
lr = final.named_steps["logisticregression"]
imp = pd.DataFrame({"gene": genes[kb.get_support()], "coef": lr.coef_[0]})
imp["abs"] = imp.coef.abs()
imp = imp[imp.coef != 0].sort_values("abs", ascending=False)
imp.to_csv(f"{RES}/07_macrophage_classifier_weights.csv", index=False)
log(f"non-zero features: {len(imp)}; top 15:\n" + imp.head(15).to_string(index=False))

fpr_c, tpr_c, _ = roc_curve(y, oof)
pd.DataFrame({"fpr": fpr_c, "tpr": tpr_c}).to_csv(f"{RES}/07_roc_percell.csv", index=False)
fpr_p, tpr_p, _ = roc_curve(pat.y, pat.mean_p)
pd.DataFrame({"fpr": fpr_p, "tpr": tpr_p}).to_csv(f"{RES}/07_roc_patient.csv", index=False)

summary = dict(
    n_macrophages=int(mac.n_obs), n_patients=int(len(pat)),
    patients_per_group=mac.obs.groupby("group", observed=True).gid.nunique().to_dict(),
    percell_grouped_auc=float(cell_auc), percell_seed_aucs=list(map(float, seed_auc)),
    percell_auc_sd=float(np.std(seed_auc)),
    percell_permutation_p=p_cell, percell_null_mean=float(null_cell.mean()),
    patient_aggregated_auc=float(pat_auc), patient_aggregated_auc_fraction=float(pat_auc_frac),
    pseudobulk_loo_auc=float(pb_auc), pseudobulk_permutation_p=p_pb,
    pseudobulk_null_mean=float(null_pb.mean()),
    n_nonzero_features=int(len(imp)))
json.dump(summary, open(f"{RES}/07_summary.json", "w"), indent=2)
log("\n=== STEP 07 COMPLETE ===\n" + json.dumps(summary, indent=2))
