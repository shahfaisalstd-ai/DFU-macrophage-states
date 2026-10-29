#!/usr/bin/env python
"""
Khan 2 revision -- Step 15
Weighted co-expression networks and cross-analysis integration
(replaces original Figure 8).

WGCNA in the strict sense is an R package; what is implemented here is the same
formulation -- signed correlation raised to a soft-thresholding power, converted
to topological overlap, clustered, and summarised by module eigengenes -- run
independently on the bulk cohort and on single-cell pseudobulk. Reviewer 2 asked
us not to imply equivalence we have not shown, so the Methods now state this is
an equivalent formulation rather than the same software, and the scale-free fit
used to pick the soft power is reported so the choice is checkable.
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from khan_bulk import load_counts, dfu_groups, cpm_log, gene_annotation

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES = f"{BASE}/02_results"
SEED = 123; rng = np.random.default_rng(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

N_GENES, MIN_MODULE = 3000, 30
POWERS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]


def soft_power(C, powers=POWERS):
    """Pick the lowest power whose connectivity distribution is approximately
    scale-free (signed R^2 of log10 p(k) on log10 k >= 0.80), as in WGCNA."""
    out = []
    for b in powers:
        A = np.abs(C) ** b
        np.fill_diagonal(A, 0)
        k = A.sum(1)
        if k.max() <= 0: continue
        hist, edges = np.histogram(k, bins=10)
        centres = (edges[:-1] + edges[1:]) / 2
        m = (hist > 0) & (centres > 0)
        if m.sum() < 4: continue
        x, y = np.log10(centres[m]), np.log10(hist[m])
        sl, ic, r, p, se = stats.linregress(x, y)
        out.append(dict(power=b, r2=float(r**2), slope=float(sl),
                        signed_r2=float(r**2 * (-1 if sl > 0 else 1)),
                        mean_k=float(k.mean())))
    S = pd.DataFrame(out)
    ok = S[S.signed_r2 >= 0.80]
    return (int(ok.power.iloc[0]) if len(ok) else int(S.loc[S.signed_r2.idxmax(), "power"])), S


def build_modules(M, label):
    """M: samples x genes (log scale). Returns module assignment + eigengenes."""
    M = M.loc[:, M.var() > 0]
    keep = M.var().nlargest(min(N_GENES, M.shape[1])).index
    M = M[keep]
    C = np.corrcoef(M.values.T)
    C = np.nan_to_num(C)
    beta, scan = soft_power(C)
    log(f"  [{label}] soft power {beta} "
        f"(signed R2 {scan.loc[scan.power==beta,'signed_r2'].iloc[0]:.2f})")

    A = np.abs(C) ** beta
    np.fill_diagonal(A, 0)
    # topological overlap matrix
    L = A @ A
    k = A.sum(1)
    denom = np.minimum.outer(k, k) + 1 - A
    TOM = (L + A) / np.maximum(denom, 1e-12)
    np.fill_diagonal(TOM, 1.0)
    D = 1 - TOM
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    labs = fcluster(Z, t=0.985 * D.max(), criterion="distance")
    s = pd.Series(labs, index=M.columns)
    sizes = s.value_counts()
    good = sizes[sizes >= MIN_MODULE].index
    s = s.where(s.isin(good), 0)
    # rename modules by size
    order = s[s > 0].value_counts().index
    remap = {old: f"{label}_M{i+1}" for i, old in enumerate(order)}
    s = s.map(lambda v: remap.get(v, "grey"))
    log(f"  [{label}] {len(order)} modules (>= {MIN_MODULE} genes), "
        f"{int((s=='grey').sum())} genes unassigned")

    eig = {}
    for mod in s[s != "grey"].unique():
        g = s[s == mod].index
        Xm = M[g].values
        Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9)
        u, sv, vt = np.linalg.svd(Xm, full_matrices=False)
        eig[mod] = u[:, 0] * sv[0]
    E = pd.DataFrame(eig, index=M.index)
    return s, E, scan, beta


# ------------------------------------------------------------------ bulk net
ann = gene_annotation()
C134, m134 = load_counts("GSE134431", ann)
g134 = dfu_groups("GSE134431", m134)
ulcer = g134[g134.isin(["Healing", "Nonhealing"])].index
Lb = cpm_log(C134)[ulcer].T                       # samples x genes
log(f"bulk network input {Lb.shape}")
bulk_mod, bulk_eig, bulk_scan, bulk_beta = build_modules(Lb, "bulk")

# --------------------------------------------------------- single-cell net
pb = pd.read_csv(f"{RES}/04_macrophage_pseudobulk_counts.csv", index_col=0)
pbm = pd.read_csv(f"{RES}/04_pseudobulk_sample_metadata.csv", index_col=0)
sel = pbm[pbm.group.isin(["Healing", "Nonhealing"])].index
Ps = pb.loc[sel]
Ps = np.log2(Ps.div(Ps.sum(1), axis=0) * 1e6 + 1)
log(f"single-cell pseudobulk network input {Ps.shape}")
sc_mod, sc_eig, sc_scan, sc_beta = build_modules(Ps, "sc")

# ------------------------------------------------------------ module-trait
def module_trait(E, groups, label):
    y = (groups.loc[E.index] == "Nonhealing").astype(int).values
    rows = []
    for m in E.columns:
        r, p = stats.pearsonr(E[m].values, y)
        rows.append(dict(module=m, n_genes=int((bulk_mod if label == "bulk" else sc_mod).eq(m).sum()),
                         pearson_r=float(r), p=float(p)))
    T = pd.DataFrame(rows)
    T["padj"] = multipletests(T.p, method="fdr_bh")[1]
    return T.sort_values("p")

bt = module_trait(bulk_eig, g134, "bulk")
st_ = module_trait(sc_eig, pbm.group, "sc")
bt.to_csv(f"{RES}/15_bulk_module_trait.csv", index=False)
st_.to_csv(f"{RES}/15_sc_module_trait.csv", index=False)
log("\n=== BULK MODULE-TRAIT ===\n" + bt.to_string(index=False))
log("\n=== SINGLE-CELL MODULE-TRAIT ===\n" + st_.to_string(index=False))

# --------------------------------------------------- cross-modality overlap
bm = {m: set(bulk_mod[bulk_mod == m].index) for m in bulk_mod.unique() if m != "grey"}
sm = {m: set(sc_mod[sc_mod == m].index) for m in sc_mod.unique() if m != "grey"}
universe = set(bulk_mod.index) & set(sc_mod.index)
rows = []
for b, gb in bm.items():
    for s_, gs in sm.items():
        A_, B_ = gb & universe, gs & universe
        if not A_ or not B_: continue
        inter = len(A_ & B_)
        jac = inter / len(A_ | B_)
        p = stats.hypergeom.sf(inter - 1, len(universe), len(A_), len(B_))
        rows.append(dict(bulk_module=b, sc_module=s_, n_bulk=len(A_), n_sc=len(B_),
                         overlap=inter, jaccard=float(jac), p=float(p)))
O = pd.DataFrame(rows)
if len(O):
    O["padj"] = multipletests(O.p, method="fdr_bh")[1]
    O = O.sort_values("p")
    O.to_csv(f"{RES}/15_module_overlap.csv", index=False)
    log(f"\nmodule pairs tested: {len(O)}; significant at FDR<0.05: "
        f"{int((O.padj<0.05).sum())}; at P<0.001: {int((O.p<0.001).sum())}")
    log(O.head(10).to_string(index=False))

# ------------------------------------------------------- consensus panel
PANEL_CSV = os.environ.get("PANEL_CSV", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "metadata", "consensus_novel_panel.csv"))
pan = pd.read_csv(PANEL_CSV)
panel = pan[pan.n_analyses >= 3].copy()
# annotate each panel gene with what the full-scale re-analysis now says
sc_deg = pd.read_csv(f"{RES}/04_pseudobulk_DESeq2_NHvsH.csv", index_col=0)
bulk = pd.read_csv(f"{RES}/06_discovery_DESeq2_GSE134431.csv", index_col=0)
traj = pd.read_csv(f"{RES}/12_trajectory_genes.csv").set_index("gene")
panel["sc_pseudobulk_p"] = panel.gene.map(sc_deg.pvalue)
panel["sc_pseudobulk_log2FC"] = panel.gene.map(sc_deg.log2FoldChange)
panel["bulk_p"] = panel.gene.map(bulk.pvalue)
panel["bulk_log2FC"] = panel.gene.map(bulk.log2FoldChange)
panel["trajectory_padj"] = panel.gene.map(traj.padj)
panel["in_bulk_module"] = panel.gene.map(bulk_mod).fillna("not in network")
panel["in_sc_module"] = panel.gene.map(sc_mod).fillna("not in network")
panel["recovered_full_scale"] = (
    (panel.sc_pseudobulk_p < 0.05).fillna(False) |
    (panel.bulk_p < 0.05).fillna(False) |
    (panel.trajectory_padj < 0.05).fillna(False))
panel = panel.sort_values("n_analyses", ascending=False)
panel.to_csv(f"{RES}/15_consensus_panel_annotated.csv", index=False)
log(f"\nconsensus panel: {len(panel)} genes; "
    f"{int(panel.recovered_full_scale.sum())} recovered by at least one full-scale analysis "
    f"({100*panel.recovered_full_scale.mean():.0f}%)")

pd.concat([bulk_scan.assign(network="bulk"), sc_scan.assign(network="single-cell")]
          ).to_csv(f"{RES}/15_soft_power_scan.csv", index=False)
bulk_mod.rename("module").to_csv(f"{RES}/15_bulk_module_assignment.csv")
sc_mod.rename("module").to_csv(f"{RES}/15_sc_module_assignment.csv")
bulk_eig.to_csv(f"{RES}/15_bulk_eigengenes.csv"); sc_eig.to_csv(f"{RES}/15_sc_eigengenes.csv")
json.dump(dict(bulk_power=bulk_beta, sc_power=sc_beta,
               n_bulk_modules=len(bm), n_sc_modules=len(sm),
               n_module_pairs=int(len(O)),
               n_pairs_p001=int((O.p < 0.001).sum()) if len(O) else 0,
               n_pairs_FDR05=int((O.padj < 0.05).sum()) if len(O) else 0,
               panel_size=int(len(panel)),
               panel_recovered=int(panel.recovered_full_scale.sum())),
          open(f"{RES}/15_summary.json","w"), indent=2)
log("=== STEP 15 COMPLETE ===")
