"""
R2_08 -- (i) co-expression networks and bulk-vs-single-cell concordance re-run on
the RAW-count macrophage pseudobulk (the single-cell network and the concordance
of the previous revision used the log-sum pseudobulk); (ii) the 129-gene
candidate panel with explicit selection criteria, per-analysis evidence, effect
estimates with 95% CIs, adjusted P values and external-validation results
(Editor comment 5).
Network method identical to original-pipeline step 15 (3,000 most variable genes, |r|^beta
adjacency, beta = lowest power with signed scale-free R^2 >= 0.80, TOM, average
linkage, cut at 0.985 x max distance, modules >= 30 genes, eigengene = PC1).
"""
import sys, os, json; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from r2_common import *
import deseq_engine

N_GENES, MIN_MODULE = 3000, 30
POWERS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
def soft_power(C):
    out = []
    for b in POWERS:
        A = np.abs(C) ** b; np.fill_diagonal(A, 0); k = A.sum(1)
        hist, edges = np.histogram(k, bins=10); cen = (edges[:-1] + edges[1:]) / 2
        m = (hist > 0) & (cen > 0)
        if m.sum() < 4: continue
        sl, ic, r, p, se = stats.linregress(np.log10(cen[m]), np.log10(hist[m]))
        out.append(dict(power=b, r2=r ** 2, slope=sl, signed_r2=r ** 2 * (-1 if sl > 0 else 1), mean_k=k.mean()))
    S = pd.DataFrame(out); ok = S[S.signed_r2 >= 0.80]
    return (int(ok.power.iloc[0]) if len(ok) else int(S.loc[S.signed_r2.idxmax(), "power"])), S
def build_modules(M, label):
    M = M.loc[:, M.var() > 0]; M = M[M.var().nlargest(min(N_GENES, M.shape[1])).index]
    C = np.nan_to_num(np.corrcoef(M.values.T)); beta, scan = soft_power(C)
    A = np.abs(C) ** beta; np.fill_diagonal(A, 0)
    L = A @ A; k = A.sum(1); TOM = (L + A) / np.maximum(np.minimum.outer(k, k) + 1 - A, 1e-12)
    np.fill_diagonal(TOM, 1.0); D = 1 - TOM; np.fill_diagonal(D, 0); D = (D + D.T) / 2
    labs = fcluster(linkage(squareform(D, checks=False), method="average"), t=0.985 * D.max(), criterion="distance")
    s = pd.Series(labs, index=M.columns); sizes = s.value_counts(); s = s.where(s.isin(sizes[sizes >= MIN_MODULE].index), 0)
    order = s[s > 0].value_counts().index; remap = {o: f"{label}_M{i+1}" for i, o in enumerate(order)}
    s = s.map(lambda v: remap.get(v, "grey"))
    eig = {}
    for mod in s[s != "grey"].unique():
        Xm = M[s[s == mod].index].values; Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9)
        u, sv, vt = np.linalg.svd(Xm, full_matrices=False); e = u[:, 0] * sv[0]
        if np.corrcoef(e, Xm.mean(1))[0, 1] < 0: e = -e
        eig[mod] = e
    return s, pd.DataFrame(eig, index=M.index), scan, beta
def module_trait(E, y, mods, label):
    rows = []
    for m in E.columns:
        r, p = stats.pearsonr(E[m].values, y); n = len(y); z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
        rows.append(dict(network=label, module=m, n_genes=int((mods == m).sum()), pearson_r_with_nonhealing=r,
                         ci_low=np.tanh(z - 1.96 * se), ci_high=np.tanh(z + 1.96 * se), p=p))
    T = pd.DataFrame(rows); T["padj"] = deseq_engine.bh(T.p.values); return T.sort_values("p")

ann = gene_annotation()
C134, m134 = load_counts("GSE134431", ann); g134 = dfu_groups("GSE134431", m134)
ulcer = g134[g134.isin(["Healing", "Nonhealing"])].index
Lb = cpm_log(C134)[ulcer].T
bmod, beig, bscan, bbeta = build_modules(Lb, "bulk")
cov = pd.read_csv(f"{OUT}/R2_04_library_covariates.csv", index_col=0)
libs19 = cov.index[cov.group.isin(["Healing", "Nonhealing"]) & (cov.site != "PBMC") & (cov.n_mac >= 10)]
PBR = pd.read_csv(f"{R2IN}/R2_02_raw_pseudobulk_macrophage.csv", index_col=0)
atlas_genes = pd.read_csv(f"{AUX}/atlas_genes.csv").gene
PBR = PBR.loc[PBR.index.intersection(atlas_genes)]
Ps = PBR[libs19].T; Ps = Ps.loc[:, (Ps > 0).sum() >= 4]
Ps = np.log2(Ps.div(Ps.sum(1), axis=0) * 1e6 + 1)
smod, seig, sscan, sbeta = build_modules(Ps, "sc")
BT = module_trait(beig, (g134.loc[beig.index] == "Nonhealing").astype(int).values, bmod, "bulk GSE134431")
ST = module_trait(seig, (cov.loc[seig.index, "group"] == "Nonhealing").astype(int).values, smod, "single-cell pseudobulk (raw counts)")
MT = pd.concat([BT, ST]); MT.to_csv(f"{OUT}/R2_08_module_trait.csv", index=False)
bm = {m: set(bmod[bmod == m].index) for m in bmod.unique() if m != "grey"}
sm = {m: set(smod[smod == m].index) for m in smod.unique() if m != "grey"}
U = set(bmod.index) & set(smod.index); orow = []
for b, gb in bm.items():
    for s_, gs in sm.items():
        A_, B_ = gb & U, gs & U
        if not A_ or not B_: continue
        inter = len(A_ & B_)
        orow.append(dict(bulk_module=b, sc_module=s_, n_bulk=len(A_), n_sc=len(B_), overlap=inter,
                         jaccard=inter / len(A_ | B_), p=stats.hypergeom.sf(inter - 1, len(U), len(A_), len(B_))))
O = pd.DataFrame(orow); O["padj"] = deseq_engine.bh(O.p.values); O = O.sort_values("p"); O.to_csv(f"{OUT}/R2_08_module_overlap.csv", index=False)
pd.concat([bscan.assign(network="bulk"), sscan.assign(network="single-cell")]).to_csv(f"{OUT}/R2_08_soft_power_scan.csv", index=False)
pd.concat([bmod.rename("module").to_frame().assign(network="bulk"), smod.rename("module").to_frame().assign(network="single-cell")]).to_csv(f"{OUT}/R2_08_module_assignment.csv")

# bulk vs single-cell concordance (raw-count pseudobulk, unadjusted and adjusted)
disc = pd.read_csv(f"{RES}/06_discovery_DESeq2_GSE134431.csv", index_col=0)
conc = {}
for m in ["M0", "M2"]:
    sc_ = pd.read_csv(f"{OUT}/R2_05_pseudobulk_DE_{m}.csv", index_col=0)
    j = pd.DataFrame({"bulk": disc.log2FoldChange, "bulk_p": disc.pvalue, "sc": sc_.log2FoldChange, "sc_p": sc_.pvalue}).dropna()
    j = j[(j.bulk.abs() < 10) & (j.sc.abs() < 10)]
    r, p = stats.spearmanr(j.bulk, j.sc); lead = j[(j.bulk_p < 0.05) & (j.sc_p < 0.05)]
    rl, pl = stats.spearmanr(lead.bulk, lead.sc) if len(lead) > 5 else (np.nan, np.nan)
    def rci(r, n): z = np.arctanh(r); se = 1 / np.sqrt(n - 3); return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]
    conc[m] = dict(n_genes=len(j), spearman_all=float(r), ci_all=rci(r, len(j)), p_all=float(p),
                   n_leading=len(lead), spearman_leading=float(rl), ci_leading=rci(rl, len(lead)) if len(lead) > 5 else None,
                   p_leading=float(pl), pct_same_sign_leading=float((np.sign(lead.bulk) == np.sign(lead.sc)).mean() * 100) if len(lead) else None)
    if m == "M0": j.assign(leading=j.index.isin(lead.index)).to_csv(f"{OUT}/R2_08_bulk_vs_sc_concordance.csv")
summ = dict(bulk_power=bbeta, sc_power=sbeta, n_bulk_modules=len(bm), n_sc_modules=len(sm), n_pairs=len(O),
            n_pairs_FDR05=int((O.padj < 0.05).sum()), n_pairs_p001=int((O.p < 0.001).sum()),
            bulk_modules_p05=int((BT.p < 0.05).sum()), sc_modules_p05=int((ST.p < 0.05).sum()),
            bulk_modules_FDR05=int((BT.padj < 0.05).sum()), sc_modules_FDR05=int((ST.padj < 0.05).sum()), concordance=conc)
json.dump(summ, open(f"{OUT}/R2_08_networks_summary.json", "w"), indent=2)
print(json.dumps(summ, indent=1)); print(MT.round(3).to_string(index=False)); print(O.head(5).to_string(index=False))

# ======================================================= 129-gene panel table
pan = pd.read_csv(PANEL_CSV); P = pan[pan.n_analyses >= 3].copy().reset_index(drop=True)
assert len(P) == 129
CRIT = {
 "P3_sc_macroDEG": "Macrophage DEG, non-healing vs healing (Wilcoxon per cell, BH-FDR < 0.05, detected in >= 10% of cells; 12 canonical macrophage genes excluded) -- original sub-sampled atlas",
 "P4_trajectory": "Monocyte-to-macrophage trajectory gene (Kruskal-Wallis across pseudotime bins, BH-FDR < 0.05; detected in >= 10 cells) -- original sub-sampled atlas",
 "P5_LRpairs": "Ligand or receptor in a significant cell-cell interaction (permutation P < 0.05, expressed in >= 10% of sender/receiver cells; 6 canonical growth-factor pairs excluded)",
 "P6_bulkDEG": "Bulk DEG in GSE134431 (limma-trend on log2 RPKM, nominal P < 0.01 and |log2FC| > 1; 19 canonical wound genes excluded)",
 "P7_ML_top30": "Top-30 random-forest feature importance in the GSE134431 bulk healing classifier",
 "P8_hubs": "Top-10 intramodular hub gene (kME) of a bulk or single-cell co-expression module",
}
P["selection_rule"] = "member of >= 3 of the 6 discovery evidence sets (P3-P8)"
P["evidence_sets"] = P.apply(lambda r: "; ".join(k for k in CRIT if r[k] == 1), axis=1)
P["uses_GSE134431_labels"] = (P.P6_bulkDEG == 1) | (P.P7_ML_top30 == 1)
P["per_cell_evidence_only"] = (P[["P5_LRpairs", "P6_bulkDEG", "P7_ML_top30", "P8_hubs"]].sum(1) == 0)
def take(df, g, cols):
    return [df.loc[g, c] if g in df.index else np.nan for c in cols]
sc0 = pd.read_csv(f"{OUT}/R2_05_pseudobulk_DE_M0.csv", index_col=0)
sc2 = pd.read_csv(f"{OUT}/R2_05_pseudobulk_DE_M2.csv", index_col=0)
vald = pd.read_csv(f"{RES}/06_validation_DESeq2_GSE143735.csv", index_col=0)
v_adj = pd.read_csv(f"{OUT}/R2_07_bulk_DE_GSE143735_age_sex_adjusted.csv", index_col=0)
d_tp = pd.read_csv(f"{OUT}/R2_07_bulk_DE_GSE134431_timepoint_adjusted.csv", index_col=0)
traj = pd.read_csv(f"{RES}/12_trajectory_genes.csv").set_index("gene")
pert = pd.read_csv(f"{RES}/18_perturbation_summary.csv")
pert = pert.set_index(pert.columns[0]) if "gene" not in pert.columns else pert.set_index("gene")
wts = pd.read_csv(f"{RES}/07_macrophage_classifier_weights.csv").set_index("gene")
rows = []
for _, r in P.iterrows():
    g = r.gene; d = dict(gene=g, n_analyses=int(r.n_analyses), evidence_sets=r.evidence_sets,
                         **{k: int(r[k]) for k in CRIT}, uses_GSE134431_labels=bool(r.uses_GSE134431_labels))
    for tag, df in [("sc_pseudobulk_unadj", sc0), ("sc_pseudobulk_sex_site_adj", sc2)]:
        lfc, se, p, q = take(df, g, ["log2FoldChange", "lfcSE", "pvalue", "padj"])
        d.update({f"{tag}_log2FC": lfc, f"{tag}_ci_low": lfc - 1.96 * se if pd.notna(se) else np.nan,
                  f"{tag}_ci_high": lfc + 1.96 * se if pd.notna(se) else np.nan, f"{tag}_P": p, f"{tag}_FDR": q})
    d["trajectory_FDR_full_scale"] = traj.loc[g, "padj"] if g in traj.index else np.nan
    lfc, lo, hi, p, q = take(disc, g, ["log2FoldChange", "ci_low", "ci_high", "pvalue", "padj"])
    d.update(bulk_GSE134431_log2FC=lfc, bulk_GSE134431_ci_low=lo, bulk_GSE134431_ci_high=hi, bulk_GSE134431_P=p, bulk_GSE134431_FDR=q)
    lfc, p, q = take(d_tp, g, ["log2FoldChange", "pvalue", "padj"])
    d.update(bulk_GSE134431_timepoint_adj_log2FC=lfc, bulk_GSE134431_timepoint_adj_P=p, bulk_GSE134431_timepoint_adj_FDR=q)
    lfc, lo, hi, p, q = take(vald, g, ["log2FoldChange", "ci_low", "ci_high", "pvalue", "padj"])
    d.update(external_GSE143735_log2FC=lfc, external_GSE143735_ci_low=lo, external_GSE143735_ci_high=hi,
             external_GSE143735_P=p, external_GSE143735_FDR=q)
    lfc, p, q = take(v_adj, g, ["log2FoldChange", "pvalue", "padj"])
    d.update(external_age_sex_adj_log2FC=lfc, external_age_sex_adj_P=p, external_age_sex_adj_FDR=q)
    b, v = d["bulk_GSE134431_log2FC"], d["external_GSE143735_log2FC"]
    d["external_same_direction"] = (bool(np.sign(b) == np.sign(v)) if pd.notna(b) and pd.notna(v) else np.nan)
    d["external_replicated_same_direction_P05"] = (bool(d["external_same_direction"] and d["external_GSE143735_P"] < 0.05)
                                                   if pd.notna(b) and pd.notna(v) else np.nan)
    d["external_replicated_FDR05"] = (bool(d["external_same_direction"] and d["external_GSE143735_FDR"] < 0.05)
                                      if pd.notna(b) and pd.notna(v) and pd.notna(d["external_GSE143735_FDR"]) else False)
    d["sc_bulk_same_direction"] = (bool(np.sign(d["sc_pseudobulk_unadj_log2FC"]) == np.sign(b))
                                   if pd.notna(b) and pd.notna(d["sc_pseudobulk_unadj_log2FC"]) else np.nan)
    d["in_macrophage_classifier_nonzero"] = g in wts.index
    stab = [c for c in pert.columns if "stable" in c.lower()]
    d["perturbation_sign_stable"] = bool(pert.loc[g, stab[0]]) if stab and g in pert.index else np.nan
    rows.append(d)
TAB = pd.DataFrame(rows).sort_values(["n_analyses", "external_replicated_same_direction_P05", "bulk_GSE134431_P"],
                                     ascending=[False, False, True])
TAB.to_csv(f"{OUT}/R2_08_TableS_panel_129.csv", index=False)
crit = pd.DataFrame([dict(code=k, criterion=v, n_panel_genes=int(P[k].sum())) for k, v in CRIT.items()])
crit.to_csv(f"{OUT}/R2_08_panel_selection_criteria.csv", index=False)
ps = dict(n=len(TAB), testable_external=int(TAB.external_same_direction.notna().sum()),
          same_direction=int(TAB.external_same_direction.fillna(False).sum()),
          replicated_P05=int(TAB.external_replicated_same_direction_P05.fillna(False).sum()),
          replicated_FDR05=int(TAB.external_replicated_FDR05.sum()),
          sc_FDR05_unadj=int((TAB.sc_pseudobulk_unadj_FDR < 0.05).sum()), sc_FDR05_adj=int((TAB.sc_pseudobulk_sex_site_adj_FDR < 0.05).sum()),
          bulk_FDR05=int((TAB.bulk_GSE134431_FDR < 0.05).sum()), uses_GSE134431=int(TAB.uses_GSE134431_labels.sum()),
          per_cell_only=int(P.per_cell_evidence_only.sum()),
          sc_bulk_same_direction=int(TAB.sc_bulk_same_direction.fillna(False).sum()),
          sc_bulk_testable=int(TAB.sc_bulk_same_direction.notna().sum()))
json.dump(ps, open(f"{OUT}/R2_08_panel_summary.json", "w"), indent=2)
print(json.dumps(ps, indent=1)); print(crit.to_string(index=False))
print(TAB.head(8).T.to_string())
