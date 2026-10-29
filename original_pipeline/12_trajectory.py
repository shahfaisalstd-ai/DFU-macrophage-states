#!/usr/bin/env python
"""
Khan 2 revision -- Step 12
Monocyte-to-macrophage trajectory at full scale (replaces original Figure 4).

The published trajectory was built from 1,342 myeloid cells and ordered
sub-cluster centroids with a minimum-spanning tree. With the full compartment
(7,756 myeloid cells, 4,969 macrophage/monocyte) the trajectory can be computed
per cell by diffusion pseudotime, rooted in the inflammatory monocyte state.

Consistent with the rest of the revision, the group comparison of pseudotime is
reported BOTH per cell and per patient, and only the patient-level value is
interpreted.
"""
import os, time, json, warnings
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES = f"{BASE}/02_results"
sc.settings.n_jobs = 16
SEED = 123; np.random.seed(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

N_BINS = 20

mye = sc.read_h5ad(f"{RES}/03_myeloid.h5ad")
is_mac = ~mye.obs.myeloid_subtype.str.contains("DC|Langerhans|Proliferating", regex=True)
mac = mye[is_mac].copy()
log(f"macrophage/monocyte compartment: {mac.n_obs:,} cells, {mac.obs.gid.nunique()} libraries")

# recompute the graph on this compartment so the diffusion map is local to it
sc.pp.highly_variable_genes(mac, n_top_genes=2000, batch_key="gid")
mac.raw = mac
m = mac[:, mac.var.highly_variable].copy()
sc.pp.scale(m, max_value=10)
sc.tl.pca(m, n_comps=30, svd_solver="arpack", random_state=SEED)

import harmonypy
ho = harmonypy.run_harmony(np.asarray(m.obsm["X_pca"], dtype=np.float64), m.obs, ["gid"],
                           max_iter_harmony=20, random_state=SEED)
Z = np.asarray(ho.Z_corr)
if Z.shape[0] != m.n_obs: Z = Z.T
m.obsm["X_pca_harmony"] = np.ascontiguousarray(Z, dtype=np.float32)
sc.pp.neighbors(m, n_neighbors=15, use_rep="X_pca_harmony", random_state=SEED)
sc.tl.diffmap(m, random_state=SEED)
sc.tl.umap(m, random_state=SEED)
log("diffusion map computed")

# Root the trajectory at the monocyte end, as pre-specified: monocytes are the
# developmental precursor of tissue macrophages. Concretely, the root is the
# inflammatory monocyte cell lying furthest in diffusion space from the centroid
# of the differentiated (resident and lipid-associated) macrophage states, so
# the root is at the monocyte extreme rather than merely inside that cluster.
infl = (m.obs.myeloid_subtype == "Inflammatory monocyte-macrophage").values
mature = m.obs.myeloid_subtype.isin(
    ["Resident M2-like macrophage", "Lipid-associated macrophage"]).values
DM = m.obsm["X_diffmap"][:, 1:11]
centroid = DM[mature].mean(0)
dist = np.linalg.norm(DM - centroid, axis=1)
root = int(np.where(infl)[0][np.argmax(dist[infl])])
m.uns["iroot"] = int(root)
sc.tl.dpt(m, n_dcs=10)
pt = m.obs["dpt_pseudotime"].values
if np.isinf(pt).any():
    pt = np.where(np.isinf(pt), np.nanmax(pt[~np.isinf(pt)]), pt)
mac.obs["pseudotime"] = pt
mac.obsm["X_umap"] = m.obsm["X_umap"]
log(f"pseudotime assigned (root = inflammatory monocyte cell {root}, "
    f"the monocyte-extreme cell in diffusion space)")
log("mean pseudotime by state:\n" +
    mac.obs.groupby("myeloid_subtype", observed=True).pseudotime.mean()
       .sort_values().to_string())

# ---------------------------------------------- group comparison, both levels
d = mac.obs[["gid", "group", "pseudotime"]]
d = d[d.group.isin(["Healing", "Nonhealing"])]
per_pat = d.groupby(["gid", "group"], observed=True).pseudotime.mean().reset_index()
a = per_pat.loc[per_pat.group == "Nonhealing", "pseudotime"].values
b = per_pat.loc[per_pat.group == "Healing", "pseudotime"].values
u_p, p_pat = stats.mannwhitneyu(a, b, alternative="two-sided")
ca = d.loc[d.group == "Nonhealing", "pseudotime"].values
cb = d.loc[d.group == "Healing", "pseudotime"].values
_, p_cell = stats.mannwhitneyu(ca, cb, alternative="two-sided")
pt_res = dict(n_patients_nonhealing=len(a), n_patients_healing=len(b),
              mean_pseudotime_nonhealing=float(a.mean()), mean_pseudotime_healing=float(b.mean()),
              rank_biserial=float(2*u_p/(len(a)*len(b))-1),
              p_patient_level=float(p_pat), p_cell_level=float(p_cell),
              n_cells_nonhealing=len(ca), n_cells_healing=len(cb))
# ---- rooting sensitivity -------------------------------------------------
# Pseudotime ordering depends on where the trajectory is rooted, so the group
# comparison is repeated with the root placed at the opposite (differentiated)
# extreme. The conclusion should not hinge on that choice.
alt_root = int(np.where(mature)[0][np.argmax(
    np.linalg.norm(DM - DM[infl].mean(0), axis=1)[mature])])
m.uns["iroot"] = alt_root
sc.tl.dpt(m, n_dcs=10)
alt = m.obs["dpt_pseudotime"].values.copy()
alt = np.where(np.isinf(alt), np.nanmax(alt[~np.isinf(alt)]), alt)
mac.obs["pseudotime_mature_rooted"] = alt
da = mac.obs[["gid", "group", "pseudotime_mature_rooted"]]
da = da[da.group.isin(["Healing", "Nonhealing"])]
pa = da.groupby(["gid", "group"], observed=True).pseudotime_mature_rooted.mean().reset_index()
aa = pa.loc[pa.group == "Nonhealing", "pseudotime_mature_rooted"].values
bb = pa.loc[pa.group == "Healing", "pseudotime_mature_rooted"].values
_, p_alt = stats.mannwhitneyu(aa, bb, alternative="two-sided")
pt_res["alt_root_mean_nonhealing"] = float(aa.mean())
pt_res["alt_root_mean_healing"] = float(bb.mean())
pt_res["alt_root_p_patient_level"] = float(p_alt)
log(f"rooting sensitivity: monocyte-rooted P = {p_pat:.3f}, "
    f"differentiated-rooted P = {p_alt:.3f} (patient level)")
# restore the pre-specified, monocyte-rooted ordering
m.uns["iroot"] = int(root)
sc.tl.dpt(m, n_dcs=10)

json.dump(pt_res, open(f"{RES}/12_pseudotime_group_test.json", "w"), indent=2)
per_pat.to_csv(f"{RES}/12_pseudotime_per_patient.csv", index=False)
log("\n=== PSEUDOTIME BY GROUP ===\n" + json.dumps(pt_res, indent=2))

# ------------------------------------------ genes changing along the trajectory
X = sp.csr_matrix(mac.raw.X)
genes = pd.Index(mac.raw.var_names)
det = np.asarray((X > 0).mean(0)).ravel()
keep = det >= 0.05
X = X[:, keep]; genes = genes[keep]
log(f"testing {len(genes):,} detected genes across {N_BINS} pseudotime bins")

bins = pd.qcut(mac.obs.pseudotime, N_BINS, labels=False, duplicates="drop")
groups_idx = [np.where(bins == b)[0] for b in sorted(pd.unique(bins))]
Xd = np.asarray(X.todense())
stat, pvals = [], []
for j in range(Xd.shape[1]):
    v = Xd[:, j]
    try:
        h, p = stats.kruskal(*[v[g] for g in groups_idx])
    except ValueError:
        h, p = 0.0, 1.0
    stat.append(h); pvals.append(p)
tr = pd.DataFrame({"gene": genes, "kruskal_H": stat, "pvalue": pvals})
tr["padj"] = multipletests(tr.pvalue, method="fdr_bh")[1]
binmeans = pd.DataFrame(np.vstack([Xd[g].mean(0) for g in groups_idx]).T,
                        index=genes, columns=[f"bin{b}" for b in range(len(groups_idx))])
tr["peak_bin"] = binmeans.values.argmax(1)
tr["amplitude"] = binmeans.max(1).values - binmeans.min(1).values
tr = tr.sort_values("pvalue")
tr.to_csv(f"{RES}/12_trajectory_genes.csv", index=False)
binmeans.to_csv(f"{RES}/12_trajectory_bin_means.csv")
nsig = int((tr.padj < 0.05).sum())
log(f"trajectory-associated genes: {nsig:,} at FDR < 0.05")
log("top 15:\n" + tr.head(15)[["gene","kruskal_H","padj","peak_bin"]].to_string(index=False))

# ---------------------------------------------------- expression pattern groups
top = tr[(tr.padj < 0.05)].nlargest(300, "amplitude").gene
Z = binmeans.loc[top]
Z = Z.sub(Z.mean(1), axis=0).div(Z.std(1) + 1e-9, axis=0)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(Z.values)
pat = pd.DataFrame({"gene": Z.index, "pattern": km.labels_})
pat.to_csv(f"{RES}/12_trajectory_gene_patterns.csv", index=False)
Z.to_csv(f"{RES}/12_trajectory_heatmap_matrix.csv")
log("pattern sizes: " + str(pat.pattern.value_counts().to_dict()))

mac.obs[["gsm","gid","group","myeloid_subtype","pseudotime"]].to_csv(
    f"{RES}/12_macrophage_pseudotime_metadata.csv")
mac.write_h5ad(f"{RES}/12_macrophage_trajectory.h5ad", compression="gzip")
json.dump(dict(n_cells=int(mac.n_obs), n_libraries=int(mac.obs.gid.nunique()),
               n_trajectory_genes_FDR05=nsig, root_cell=int(root)),
          open(f"{RES}/12_summary.json","w"), indent=2)
log("=== STEP 12 COMPLETE ===")
