#!/usr/bin/env python
"""
Khan 2 revision -- Step 13
Intercellular communication across the full atlas (replaces original Figure 5).

Same estimator as the published analysis -- mean ligand expression in the sender
multiplied by mean receptor expression in the receiver, with a label-permutation
test for specificity -- but computed over all 97,677 cells rather than 13,200,
and with the myeloid compartment split into its resolved states so that
macrophage-centred signalling can be attributed to a specific programme.

The healing-versus-non-healing contrast is additionally recomputed with each
library scored separately, so the comparison is supported by patient-level
variation rather than by pooled cells (Reviewer 1, comment 2).
"""
import os, time, json, warnings
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES  = f"{BASE}/02_results"
SEED = 123; rng = np.random.default_rng(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

N_PERM   = 200
MIN_CELLS = 25          # a cell type must have this many cells to be a sender/receiver
MIN_FRAC  = 0.10        # a gene must be detected in this fraction to count as expressed

LR = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lr_database.tsv"), sep="\t")
log(f"ligand-receptor database: {len(LR)} pairs")

A   = sc.read_h5ad(f"{RES}/03_atlas_annotated.h5ad")
mye = sc.read_h5ad(f"{RES}/03_myeloid.h5ad")
lab = A.obs.cell_type.astype(str).copy()
sub = mye.obs.myeloid_subtype.astype(str)
# resolve the myeloid compartment into its states, collapse the DC subsets
mac_map = {"Resident M2-like macrophage": "Macrophage: resident",
           "Inflammatory monocyte-macrophage": "Macrophage: inflammatory",
           "Lipid-associated macrophage": "Macrophage: lipid-assoc."}
for k, v in mac_map.items():
    lab.loc[lab.index.intersection(sub[sub == k].index)] = v
lab.loc[lab.index.intersection(sub[sub.str.contains("DC|Langerhans")].index)] = "Dendritic cell"
lab.loc[lab.index.intersection(sub[sub == "Proliferating myeloid"].index)] = "Dendritic cell"
A.obs["ct"] = lab.values

counts = A.obs.ct.value_counts()
types = sorted(counts[counts >= MIN_CELLS].index)
A = A[A.obs.ct.isin(types)].copy()
log(f"{len(types)} sender/receiver classes over {A.n_obs:,} cells")
log(A.obs.ct.value_counts().to_string())

genes = pd.Index(A.var_names)
pairs = LR[LR.ligand.isin(genes) & LR.receptor.isin(genes)].reset_index(drop=True)
log(f"{len(pairs)} pairs with both partners in the atlas")
pairs["pair"] = pairs.ligand + "-" + pairs.receptor
pair_names = pairs["pair"].tolist()
use = pd.Index(sorted(set(pairs.ligand) | set(pairs.receptor)))
X = sp.csr_matrix(A[:, use].X)
gpos = {g: i for i, g in enumerate(use)}
ct_codes = pd.Categorical(A.obs.ct, categories=types).codes


def group_means(codes):
    """mean expression and detection fraction per class -> (n_types, n_genes)"""
    M = np.zeros((len(types), X.shape[1]), dtype=np.float32)
    F = np.zeros_like(M)
    for i in range(len(types)):
        idx = np.where(codes == i)[0]
        if len(idx) == 0: continue
        sub_ = X[idx]
        M[i] = np.asarray(sub_.mean(0)).ravel()
        F[i] = np.asarray((sub_ > 0).mean(0)).ravel()
    return M, F


M, F = group_means(ct_codes)
li = np.array([gpos[g] for g in pairs.ligand])
ri = np.array([gpos[g] for g in pairs.receptor])
# score[s, r, p] = mean ligand in sender s  x  mean receptor in receiver r
S = M[:, li][:, None, :] * M[None, :, ri]
expressed = (F[:, li][:, None, :] >= MIN_FRAC) & (F[None, :, ri] >= MIN_FRAC)
log(f"score tensor {S.shape}; {int(expressed.sum()):,} sender-receiver-pair combinations "
    f"pass the detection filter")

log(f"permuting cell-type labels {N_PERM} times ...")
ge = np.zeros_like(S, dtype=np.int32)
for k in range(N_PERM):
    Mp, _ = group_means(rng.permutation(ct_codes))
    Sp = Mp[:, li][:, None, :] * Mp[None, :, ri]
    ge += (Sp >= S)
    if (k + 1) % 50 == 0: log(f"  {k+1}/{N_PERM}")
P = (ge + 1) / (N_PERM + 1)

rows = []
for a in range(len(types)):
    for b in range(len(types)):
        m = expressed[a, b]
        if not m.any(): continue
        for p in np.where(m)[0]:
            rows.append((types[a], types[b], pairs.ligand[p], pairs.receptor[p],
                         float(S[a, b, p]), float(P[a, b, p])))
comm = pd.DataFrame(rows, columns=["sender", "receiver", "ligand", "receptor",
                                   "score", "pval"])
comm["padj"] = multipletests(comm.pval, method="fdr_bh")[1]
comm["pair"] = comm.ligand + "-" + comm.receptor
comm = comm.sort_values("score", ascending=False)
comm.to_csv(f"{RES}/13_communication_all.csv", index=False)
sig = comm[comm.padj < 0.05]
log(f"{len(comm):,} tested interactions, {len(sig):,} significant at FDR < 0.05")

net = sig.groupby(["sender", "receiver"]).size().unstack(fill_value=0).reindex(
    index=types, columns=types, fill_value=0)
net.to_csv(f"{RES}/13_interaction_counts.csv")
act = pd.DataFrame({"outgoing": sig.groupby("sender").score.sum(),
                    "incoming": sig.groupby("receiver").score.sum()}).reindex(types).fillna(0)
act.to_csv(f"{RES}/13_signalling_activity.csv")
log("top senders by outgoing strength:\n" +
    act.sort_values("outgoing", ascending=False).head(6).to_string())

# ------------------------------- macrophage-centred contrast, per library
mac_types = [t for t in types if t.startswith("Macrophage")]
keep = A.obs.group.isin(["Healing", "Nonhealing"])
sub_ad = A[keep].copy()
log(f"\nper-library contrast over {sub_ad.n_obs:,} cells from "
    f"{sub_ad.obs.gid.nunique()} libraries")

Xs = sp.csr_matrix(sub_ad[:, use].X)
recs = []
for gid, idx in sub_ad.obs.groupby("gid", observed=True).indices.items():
    grp = sub_ad.obs.group.iloc[idx[0]]
    codes = pd.Categorical(sub_ad.obs.ct.iloc[idx], categories=types).codes
    Ml = np.zeros((len(types), Xs.shape[1]), dtype=np.float32)
    ok = np.zeros(len(types), dtype=bool)
    for i in range(len(types)):
        sel = np.asarray(idx)[codes == i]
        if len(sel) < 5: continue
        Ml[i] = np.asarray(Xs[sel].mean(0)).ravel(); ok[i] = True
    for mt in mac_types:
        mi = types.index(mt)
        if not ok[mi]: continue
        for a in range(len(types)):
            if not ok[a]: continue
            inc = Ml[a, li] * Ml[mi, ri]        # macrophage receiving
            out = Ml[mi, li] * Ml[a, ri]        # macrophage sending
            for p in range(len(pairs)):
                nm = pair_names[p]
                if inc[p] > 0:
                    recs.append((gid, grp, mt, types[a], nm, "incoming", float(inc[p])))
                if out[p] > 0:
                    recs.append((gid, grp, mt, types[a], nm, "outgoing", float(out[p])))
per_lib = pd.DataFrame(recs, columns=["gid", "group", "macrophage_state", "partner",
                                      "pair", "direction", "score"])
per_lib.to_csv(f"{RES}/13_macrophage_communication_per_library.csv", index=False)
log(f"per-library macrophage interaction records: {len(per_lib):,}")

agg = per_lib.groupby(["gid", "group", "pair", "direction"], observed=True).score.sum().reset_index()
tests = []
for (pair, direction), g in agg.groupby(["pair", "direction"], observed=True):
    a = g.loc[g.group == "Nonhealing", "score"].values
    b = g.loc[g.group == "Healing", "score"].values
    if len(a) < 4 or len(b) < 4: continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    tests.append(dict(pair=pair, direction=direction,
                      n_nonhealing=len(a), n_healing=len(b),
                      mean_nonhealing=float(a.mean()), mean_healing=float(b.mean()),
                      log2_ratio=float(np.log2((a.mean()+1e-9)/(b.mean()+1e-9))),
                      rank_biserial=float(2*u/(len(a)*len(b))-1), p=float(p)))
T = pd.DataFrame(tests)
if len(T):
    T["padj"] = multipletests(T.p, method="fdr_bh")[1]
    T = T.sort_values("p")
    T.to_csv(f"{RES}/13_macrophage_contrast_patient_level.csv", index=False)
    log(f"\npatient-level macrophage interaction tests: {len(T)}; "
        f"{int((T.padj<0.05).sum())} at FDR < 0.05, {int((T.p<0.05).sum())} at nominal P < 0.05")
    log("top 12 by P:\n" + T.head(12)[["pair","direction","mean_nonhealing","mean_healing",
                                       "log2_ratio","p","padj"]].to_string(index=False))

json.dump(dict(n_types=len(types), n_cells=int(A.n_obs), n_pairs=int(len(pairs)),
               n_tested=int(len(comm)), n_significant=int(len(sig)),
               n_patient_level_tests=int(len(T)),
               n_patient_level_FDR05=int((T.padj < 0.05).sum()) if len(T) else 0),
          open(f"{RES}/13_summary.json","w"), indent=2)
log("=== STEP 13 COMPLETE ===")
