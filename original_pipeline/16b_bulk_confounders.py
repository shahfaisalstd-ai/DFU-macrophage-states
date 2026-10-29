#!/usr/bin/env python
"""
Khan 2 revision -- Step 16b
Confounder audit of the two bulk cohorts.

The single-cell comparison turned out to be confounded by sex (step 16). The
same question has to be asked of the bulk cohorts, because the discovery
contrast in GSE134431 is what the prioritised panel and the machine-learning
results rest on. GSE143735 reports sex directly in its series matrix; for
GSE134431 sex is inferred from XIST and Y-chromosome expression as before.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from khan_bulk import load_counts, dfu_groups, cpm_log, gene_annotation

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES = f"{BASE}/02_results"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

YGEN = ["RPS4Y1", "DDX3Y", "UTY", "USP9Y", "KDM5D", "EIF1AY", "NLGN4Y", "ZFY"]
ann = gene_annotation()
out = {}

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

for acc in ["GSE134431", "GSE143735"]:
    C, meta = load_counts(acc, ann)
    g = dfu_groups(acc, meta)
    ulcer = g[g.isin(["Healing", "Nonhealing"])].index
    L = cpm_log(C)[ulcer]
    y_ = np.log2(C[ulcer].div(C[ulcer].sum(0), axis=1) * 1e6 + 1)

    # Call sex from Y-chromosome expression only. XIST is a lncRNA and is removed
    # by the protein-coding filter these matrices use, so an XIST-versus-Y rule
    # would classify every sample as male.
    xi = y_.loc["XIST"] if "XIST" in y_.index else pd.Series(np.nan, index=ulcer)
    yy = y_.loc[[x for x in YGEN if x in y_.index]].mean(0)
    # Absolute cut-off (see step 16): a data-driven midpoint would split an
    # all-male cohort in half and manufacture a confounder.
    Y_THRESHOLD = 1.0
    sex = pd.DataFrame({"XIST": xi, "Y_genes": yy})
    sex["inferred_sex"] = np.where(sex.Y_genes > Y_THRESHOLD, "male", "female")
    both = (yy < Y_THRESHOLD).any() and (yy > Y_THRESHOLD).any()
    log(f"[{acc}] Y-gene range {yy.min():.2f}-{yy.max():.2f}; "
        f"cut-off {Y_THRESHOLD}; both sexes present: {both}")
    # GSE143735 states sex explicitly; use it where available
    if "Sex" in meta.columns:
        sex["reported_sex"] = meta.loc[ulcer, "Sex"].map({"M": "male", "F": "female"})
        agree = float((sex.reported_sex == sex.inferred_sex).mean())
        log(f"[{acc}] inferred vs reported sex agreement: {agree*100:.0f}%")
    sex["group"] = g.loc[ulcer]
    sex.to_csv(f"{RES}/16b_{acc}_inferred_sex.csv")

    tab = pd.crosstab(sex.group, sex.inferred_sex)
    p_f = stats.fisher_exact(tab.values)[1] if tab.shape == (2, 2) else np.nan
    log(f"\n[{acc}] sex x group:\n{tab.to_string()}\nFisher P = {p_f:.3f}")

    # sex-linked genes in the group contrast
    disc = pd.read_csv(f"{RES}/06_discovery_DESeq2_GSE134431.csv", index_col=0) if acc == "GSE134431" \
        else pd.read_csv(f"{RES}/06_validation_DESeq2_GSE143735.csv", index_col=0)
    sg = [x for x in ["XIST"] + YGEN if x in disc.index]
    sh = disc.loc[sg, ["log2FoldChange", "pvalue", "padj"]].sort_values("pvalue")
    n_sig = int((sh.padj < 0.05).sum())
    log(f"[{acc}] sex-linked genes at FDR<0.05 in the contrast: {n_sig} of {len(sh)}")
    log(sh.to_string())

    # re-run with sex in the design
    if tab.shape == (2, 2) and tab.min().min() >= 1 and both:
        X = C[ulcer].T.round().astype(int)
        X = X.loc[:, (X > 0).sum(0) >= max(3, int(0.2 * len(ulcer)))]
        cd = pd.DataFrame({
            "group": pd.Categorical(g.loc[ulcer], categories=["Healing", "Nonhealing"]),
            "sex": pd.Categorical(sex.inferred_sex)}, index=ulcer)
        dds = DeseqDataSet(counts=X, metadata=cd, design="~sex+group", refit_cooks=True, quiet=True)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["group", "Nonhealing", "Healing"], quiet=True)
        st.summary()
        adj = st.results_df.sort_values("pvalue")
        adj.to_csv(f"{RES}/16b_{acc}_DESeq2_sex_adjusted.csv")
        n_un = int((disc.padj < 0.05).sum()); n_ad = int((adj.padj < 0.05).sum())
        common = disc.index.intersection(adj.index)
        r = stats.spearmanr(disc.loc[common, "log2FoldChange"],
                            adj.loc[common, "log2FoldChange"])[0]
        keep = set(disc[disc.padj < 0.05].index) & set(adj[adj.padj < 0.05].index)
        log(f"[{acc}] FDR<0.05: {n_un} unadjusted -> {n_ad} sex-adjusted "
            f"({len(keep)} shared); log2FC rho = {r:.3f}")
        out[acc] = dict(sex_table={k: v.to_dict() for k, v in tab.iterrows()},
                        fisher_p=float(p_f), n_sexlinked_FDR05=n_sig,
                        n_FDR05_unadjusted=n_un, n_FDR05_sex_adjusted=n_ad,
                        n_shared=len(keep), log2FC_spearman=float(r))
    else:
        out[acc] = dict(sex_table={k: v.to_dict() for k, v in tab.iterrows()},
                        fisher_p=float(p_f), n_sexlinked_FDR05=n_sig,
                        note="sex adjustment not estimable")
        log(f"[{acc}] sex adjustment not estimable from this design")

json.dump(out, open(f"{RES}/16b_summary.json", "w"), indent=2)
log("\n=== STEP 16b COMPLETE ===\n" + json.dumps(out, indent=2))
