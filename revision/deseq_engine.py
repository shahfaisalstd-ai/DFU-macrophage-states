"""
deseq_engine.py -- negative-binomial Wald test for the revision-2 analyses,
run with PyDESeq2 0.5.2 itself (the version used by the original pipeline).

PyDESeq2 0.5.2 is vendored in vendor/pydeseq2 (MIT licence; upstream commit
12afe9d, tag v0.5.2) with one dtype patch for pandas >= 3 (vendor/patches/pydeseq2_0.5.2_pandas3.diff).
anndata/formulaic are not needed: vendor/stubs provides a minimal in-memory
AnnData and the design is passed as a design-matrix DataFrame.

deseq(K, X, coef) keeps the interface of the earlier nbdeseq.py:
  K    genes x samples raw counts
  X    samples x p design matrix (first column = intercept)
  coef index of the tested column; the Wald test uses the unit contrast on it.
All DESeq2 defaults are kept (ratio size factors, parametric trend, MAP
dispersions, Cook's distance refit with >= 7 replicates, Cook's filtering,
independent filtering with alpha = 0.05). Single-threaded for determinism.
"""
import os, sys, warnings, contextlib, io
HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(os.path.dirname(HERE), "vendor")          # <repo>/vendor
for p in (f"{VENDOR}/stubs", f"{VENDOR}/pydeseq2"):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np, pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference


def deseq(K, X, coef, gene_names=None, alpha_fdr=0.05):
    K = np.asarray(K); X = np.asarray(X, dtype=float)
    G, n = K.shape
    genes = pd.Index([f"g{i}" for i in range(G)] if gene_names is None else list(map(str, gene_names)))
    samples = pd.Index([f"s{i}" for i in range(n)])
    counts = pd.DataFrame(np.rint(K).astype(int).T, index=samples, columns=genes)
    cols = ["Intercept"] + [f"x{i}" for i in range(1, X.shape[1])]
    design = pd.DataFrame(X, index=samples, columns=cols)
    contrast = np.zeros(X.shape[1]); contrast[coef] = 1.0
    inf = DefaultInference(n_cpus=1)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts, metadata=pd.DataFrame(index=samples), design=design,
                           refit_cooks=True, quiet=True, inference=inf)
        dds.deseq2()
        st = DeseqStats(dds, contrast=contrast, alpha=alpha_fdr, cooks_filter=True,
                        independent_filter=True, quiet=True, inference=inf)
        st.summary()
    R = st.results_df.reindex(genes)
    out = {k: R[k].values.astype(float) for k in ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]}
    var = dds.var.reindex(genes)
    for k_out, k_in in [("dispGeneEst", "genewise_dispersions"), ("dispFit", "fitted_dispersions"), ("dispersion", "dispersions")]:
        out[k_out] = var[k_in].values.astype(float) if k_in in var else np.full(G, np.nan)
    out["size_factors"] = np.asarray(dds.obs["size_factors"].values, float)
    out["n_cooks_flagged"] = int(np.asarray(dds.cooks_outlier()).sum())
    out["n_cooks_replaced"] = int(dds.var["replaced"].sum()) if "replaced" in dds.var else 0
    conv = var["_LFC_converged"] if "_LFC_converged" in var else pd.Series(True, index=genes)
    out["n_separated"] = int((conv == False).sum())          # rows whose IRLS/L-BFGS fit did not converge
    return out


def bh(p):
    """Benjamini-Hochberg adjusted P values; NaN entries are left as NaN."""
    p = np.asarray(p, float); q = np.full_like(p, np.nan)
    ok = np.isfinite(p); pv = p[ok]; n = len(pv)
    if n == 0: return q
    o = np.argsort(pv); r = pv[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]; r = np.minimum(r, 1)
    tmp = np.empty(n); tmp[o] = r; q[ok] = tmp
    return q
