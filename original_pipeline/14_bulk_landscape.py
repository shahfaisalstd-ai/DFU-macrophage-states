#!/usr/bin/env python
"""
Khan 2 revision -- Step 14
Bulk transcriptomic landscape (replaces original Figure 6).

The published analysis used moderated linear modelling on log-RPKM and reported
that no gene survived genome-wide correction. Re-running on raw counts with
PyDESeq2 -- the appropriate model for count data, and the one Reviewer 2 asked
us to name explicitly -- recovers a set of genes at FDR < 0.05, so the claim in
the manuscript is updated.

Also computes pathway over-representation and the gene-level concordance between
the bulk contrast and the full-scale single-cell macrophage contrast.
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from khan_bulk import load_counts, dfu_groups, cpm_log, gene_annotation

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES = f"{BASE}/02_results"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

disc = pd.read_csv(f"{RES}/06_discovery_DESeq2_GSE134431.csv", index_col=0)
log(f"bulk DESeq2 loaded: {disc.shape[0]:,} genes, "
    f"{int((disc.padj<0.05).sum())} at FDR<0.05, {int((disc.pvalue<0.05).sum())} nominal")
disc.to_csv(f"{RES}/14_bulk_volcano.csv")

# ------------------------------------------------------- pathway enrichment
import gseapy as gp
up_healing = disc[(disc.pvalue < 0.05) & (disc.log2FoldChange < 0)].index.tolist()
up_nonheal = disc[(disc.pvalue < 0.05) & (disc.log2FoldChange > 0)].index.tolist()
log(f"genes higher in healing: {len(up_healing)}; higher in non-healing: {len(up_nonheal)}")

SETS = ["GO_Biological_Process_2021", "KEGG_2021_Human"]
enr_all = []
for name, genes in [("higher in healing", up_healing), ("higher in non-healing", up_nonheal)]:
    if len(genes) < 15: continue
    try:
        e = gp.enrichr(gene_list=genes, gene_sets=SETS,
                       organism="human", outdir=None, no_plot=True).results
        e["direction"] = name
        enr_all.append(e)
        log(f"  {name}: {len(e)} terms, {int((e['Adjusted P-value']<0.05).sum())} at FDR<0.05")
    except Exception as ex:
        log(f"  enrichr failed for {name}: {ex}")
if enr_all:
    E = pd.concat(enr_all)
    E.to_csv(f"{RES}/14_pathway_enrichment.csv", index=False)
    top = E[E.direction == "higher in healing"].nsmallest(15, "Adjusted P-value")
    log("\ntop pathways higher in healing:\n" +
        top[["Term", "Adjusted P-value", "Overlap"]].to_string(index=False))
else:
    E = pd.DataFrame()

# --------------------------------------- bulk vs single-cell concordance
sc_deg = pd.read_csv(f"{RES}/04_pseudobulk_DESeq2_NHvsH.csv", index_col=0)
common = disc.index.intersection(sc_deg.index)
C = pd.DataFrame({
    "bulk_log2FC": disc.loc[common, "log2FoldChange"],
    "bulk_p": disc.loc[common, "pvalue"],
    "sc_log2FC": sc_deg.loc[common, "log2FoldChange"],
    "sc_p": sc_deg.loc[common, "pvalue"]}).dropna()
r_all, p_all = stats.spearmanr(C.bulk_log2FC, C.sc_log2FC)
lead = C[(C.bulk_p < 0.05) & (C.sc_p < 0.05)]
r_lead, p_lead = (stats.spearmanr(lead.bulk_log2FC, lead.sc_log2FC)
                  if len(lead) > 5 else (np.nan, np.nan))
C["same_sign"] = np.sign(C.bulk_log2FC) == np.sign(C.sc_log2FC)
C["leading"] = C.index.isin(lead.index)
C.to_csv(f"{RES}/14_bulk_sc_concordance.csv")
conc = dict(n_genes=int(len(C)), spearman_all=float(r_all), p_all=float(p_all),
            n_leading=int(len(lead)), spearman_leading=float(r_lead),
            p_leading=float(p_lead),
            pct_same_sign_all=float(100*C.same_sign.mean()),
            pct_same_sign_leading=float(100*C.loc[C.leading, "same_sign"].mean())
                                   if len(lead) else np.nan)
json.dump(conc, open(f"{RES}/14_concordance_summary.json","w"), indent=2)
log("\n=== BULK vs SINGLE-CELL CONCORDANCE ===\n" + json.dumps(conc, indent=2))

# ------------------------------------------ representative genes in bulk
ann = gene_annotation()
C134, m134 = load_counts("GSE134431", ann)
g134 = dfu_groups("GSE134431", m134)
L = cpm_log(C134)
PANELS = {
 "myeloid activation": ["S100A8","S100A9","FCN1","VCAN","IL1B","CXCL8","LYZ","AIF1"],
 "resident / complement": ["FOLR2","LYVE1","C1QA","C1QB","C1QC","MRC1","CD163","VSIG4"],
 "lipid metabolism": ["APOE","APOC1","TREM2","LPL","GPNMB","CD9","FABP5","PLIN2"],
}
rows = []
ulcer = g134[g134.isin(["Healing","Nonhealing"])].index
for panel, genes in PANELS.items():
    for g in genes:
        if g not in L.index: continue
        a = L.loc[g, [i for i in ulcer if g134[i]=="Nonhealing"]]
        b = L.loc[g, [i for i in ulcer if g134[i]=="Healing"]]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(dict(panel=panel, gene=g,
                         mean_nonhealing=float(a.mean()), mean_healing=float(b.mean()),
                         log2FC=float(a.mean()-b.mean()), p=float(p)))
rep = pd.DataFrame(rows)
rep.to_csv(f"{RES}/14_representative_genes_bulk.csv", index=False)
log(f"\nrepresentative genes measured in bulk: {len(rep)}")
log(rep.groupby("panel").log2FC.mean().to_string())

Lz = L.loc[[g for p in PANELS.values() for g in p if g in L.index], ulcer]
Lz = Lz.sub(Lz.mean(1), axis=0).div(Lz.std(1)+1e-9, axis=0)
Lz.to_csv(f"{RES}/14_representative_heatmap.csv")
pd.Series(g134.loc[ulcer]).to_csv(f"{RES}/14_bulk_ulcer_groups.csv")

json.dump(dict(n_FDR05=int((disc.padj<0.05).sum()),
               n_nominal=int((disc.pvalue<0.05).sum()),
               n_pathways_FDR05=int((E["Adjusted P-value"]<0.05).sum()) if len(E) else 0),
          open(f"{RES}/14_summary.json","w"), indent=2)
log("=== STEP 14 COMPLETE ===")
