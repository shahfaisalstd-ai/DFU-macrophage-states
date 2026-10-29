"""R2_03: cross-environment check of the differential-expression engine.
deseq_engine.py runs PyDESeq2 0.5.2 (vendored) in the revision environment
(Python 3.13, numpy 2 / pandas 3). Here it is run on exactly the bulk input of the
original pipeline (GSE134431 and GSE143735, ~group) and compared with the
PyDESeq2 0.5.2 results stored by that pipeline (Python 3.10, numpy 1.26 / pandas 1.5)."""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats
from r2_common import *
import deseq_engine
ann = gene_annotation(); rep = {}
for prefix, ref in [("GSE134431", "06_discovery_DESeq2_GSE134431.csv"), ("GSE143735", "06_validation_DESeq2_GSE143735.csv")]:
    C, m = load_counts(prefix, ann); g = dfu_groups(prefix, m)
    sub = g[g.isin(["Healing", "Nonhealing"])]
    X = C[sub.index].T.round().astype(int)
    X = X.loc[:, (X > 0).sum(axis=0) >= max(3, int(0.2 * len(sub)))]
    D = np.column_stack([np.ones(len(sub)), (sub.values == "Nonhealing").astype(float)])
    r = deseq_engine.deseq(X.T.values, D, coef=1, gene_names=X.columns)
    res = pd.DataFrame({k: r[k] for k in ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]}, index=X.columns)
    res.to_csv(f"{OUT}/R2_03_deseq_engine_{prefix}.csv")
    py = pd.read_csv(f"{RES}/{ref}", index_col=0)
    j = res.join(py, rsuffix="_py", how="inner").dropna(subset=["pvalue", "pvalue_py"])
    se = (j.lfcSE / j.lfcSE_py)
    rep[prefix] = dict(n_genes=len(j),
        lfc_pearson=float(np.corrcoef(j.log2FoldChange, j.log2FoldChange_py)[0, 1]),
        lfc_median_abs_diff=float((j.log2FoldChange - j.log2FoldChange_py).abs().median()),
        se_ratio_median=float(se.median()),
        logp_spearman=float(stats.spearmanr(j.pvalue, j.pvalue_py)[0]),
        pvalue_max_abs_diff=float((j.pvalue - j.pvalue_py).abs().max()),
        genes_pvalue_diff_gt_1e3=int(((j.pvalue - j.pvalue_py).abs() > 1e-3).sum()),
        n_FDR05_engine=int((res.padj < 0.05).sum()), n_FDR05_pydeseq2=int((py.padj < 0.05).sum()),
        n_nominal_engine=int((res.pvalue < 0.05).sum()), n_nominal_pydeseq2=int((py.pvalue < 0.05).sum()),
        overlap_FDR05=int(len(set(res.index[res.padj < 0.05]) & set(py.index[py.padj < 0.05]))),
        identical_FDR05_set=bool(set(res.index[res.padj < 0.05]) == set(py.index[py.padj < 0.05])),
        top20_overlap=int(len(set(j.nsmallest(20, "pvalue").index) & set(j.nsmallest(20, "pvalue_py").index))))
    print(prefix, json.dumps(rep[prefix], indent=1), flush=True)
json.dump(rep, open(f"{OUT}/R2_03_deseq_engine_vs_original_pydeseq2.json", "w"), indent=2)
