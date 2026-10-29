#!/usr/bin/env python
"""
Khan 2 revision -- Step 05
Deconvolution of the bulk cohort, benchmarked against synthetic mixtures of
known composition, with a direct comparison of the enlarged and the original
reference.

Reviewer 2, comment 3: "Deconvolution accuracy depends heavily on the quality
of the single cell reference signature. The authors used only 1,342 myeloid
cells from 44 samples, and the number of myeloid cells per sample may be very
limited."

Rather than assert that the reference is adequate, this script measures it:

  (1) BENCHMARK. Cells are split in half. One half builds the signature; the
      other is combined into 200 synthetic bulk mixtures of known composition.
      Deconvolution accuracy is then measured directly (RMSE, Pearson r,
      per-cell-type error), and the ridge penalty is chosen on this benchmark
      rather than by eye.
  (2) REFERENCE SIZE. The identical benchmark is repeated with a reference
      down-sampled to the original 300-cells-per-library design, so the gain
      from the full atlas is quantified rather than assumed.
  (3) APPLICATION. The selected settings are applied to GSE134431, with a
      bootstrap confidence interval on every fraction.
  (4) ORTHOGONAL CHECK. A reference-free macrophage marker score is computed on
      the same samples.
  (5) POWER. The smallest macrophage-fraction difference this cohort could have
      detected is reported alongside the observed one.
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
from scipy.optimize import nnls
from scipy import stats
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from khan_bulk import load_counts, dfu_groups, cpm_log, gene_annotation

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES  = f"{BASE}/02_results"
SEED = 123; rng = np.random.default_rng(SEED)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

N_MARKER, N_BOOT, N_MIX = 100, 60, 200
LAMBDAS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# ------------------------------------------------------------------ reference
A   = sc.read_h5ad(f"{RES}/03_atlas_annotated.h5ad")
mye = sc.read_h5ad(f"{RES}/03_myeloid.h5ad")
lab = A.obs.cell_type.astype(str).copy()
sub = mye.obs.myeloid_subtype.astype(str)
lab.loc[lab.index.intersection(sub[~sub.str.contains("DC|Langerhans")].index)] = "Macrophage"
lab.loc[lab.index.intersection(sub[sub.str.contains("DC|Langerhans")].index)]  = "Dendritic cell"
A.obs["ref_type"] = lab.values
labels = A.obs.ref_type.astype(str)
log("reference classes:\n" + labels.value_counts().to_string())

Xlog = sp.csr_matrix(A.X)                 # log1p(CP10K)
X = Xlog.copy(); X.data = np.expm1(X.data)   # linear CP10K -- fractions are additive here
genes = pd.Index(A.var_names)
bad = (genes.str.startswith(("MT-", "RPL", "RPS", "MTRNR"))
       | genes.isin(["MALAT1", "NEAT1", "HBB", "HBA1", "HBA2", "HBD", "HBG1", "HBG2"]))
good = np.where(~bad)[0]
G = genes[good]
log(f"marker-eligible genes: {len(good):,}")


def build_signature(cell_idx, n_marker=N_MARKER):
    lb = labels.iloc[cell_idx]
    lin, lg = {}, {}
    for ct, idx in lb.groupby(lb, observed=True).indices.items():
        rows = np.asarray(cell_idx)[idx]
        lin[ct] = np.asarray(X[rows][:, good].mean(0)).ravel()
        lg[ct]  = np.asarray(Xlog[rows][:, good].mean(0)).ravel()
    P, LP = pd.DataFrame(lin, index=G), pd.DataFrame(lg, index=G)
    marks = set()
    for ct in LP.columns:
        marks.update((LP[ct] - LP.drop(columns=[ct]).mean(1))
                     .sort_values(ascending=False).head(n_marker).index)
    return P.loc[sorted(marks)]


def deconvolve(sig, bulk_linear, lam):
    """Ridge-regularised NNLS on the linear scale (see docstring at top)."""
    common = sig.index.intersection(bulk_linear.index)
    S = sig.loc[common].values.astype(float)
    B = bulk_linear.loc[common].values.astype(float)
    rs = np.linalg.norm(S, axis=1, keepdims=True); rs[rs == 0] = 1.0
    S, B = S / rs, B / rs
    cs = np.linalg.norm(S, axis=0, keepdims=True); cs[cs == 0] = 1.0
    Sn = S / cs
    k = Sn.shape[1]
    Sa = np.vstack([Sn, np.sqrt(lam) * np.eye(k)]) if lam > 0 else Sn
    out = []
    for j in range(B.shape[1]):
        ba = np.concatenate([B[:, j], np.zeros(k)]) if lam > 0 else B[:, j]
        v, _ = nnls(Sa, ba)
        w = v / cs.ravel()
        out.append(w / w.sum() if w.sum() > 0 else w)
    return pd.DataFrame(out, index=bulk_linear.columns, columns=sig.columns), len(common)


# ============================================ (1) benchmark on known mixtures
half = rng.permutation(A.n_obs)
ref_half, mix_half = half[:A.n_obs // 2], half[A.n_obs // 2:]
types = sorted(labels.unique())
mix_pool = {ct: np.array([i for i in mix_half if labels.iloc[i] == ct]) for ct in types}
mix_pool = {k: v for k, v in mix_pool.items() if len(v) >= 20}
log(f"benchmark: {len(mix_pool)} cell types with >=20 cells in the mixing half")

def make_mixtures(n_mix=N_MIX, n_cells=2000):
    """Synthetic bulk profiles with known cell-type proportions."""
    tl = list(mix_pool)
    props, profs = [], []
    for _ in range(n_mix):
        # Dirichlet skewed toward the structural cell types, as real skin is
        alpha = np.array([5.0 if t in ("Keratinocyte", "Fibroblast") else 0.5 for t in tl])
        p = rng.dirichlet(alpha)
        counts = rng.multinomial(n_cells, p)
        chosen = np.concatenate([rng.choice(mix_pool[t], c, replace=True)
                                 for t, c in zip(tl, counts) if c > 0])
        profs.append(np.asarray(X[chosen][:, good].mean(0)).ravel())
        props.append(p)
    Bm = pd.DataFrame(np.vstack(profs).T, index=G,
                      columns=[f"mix{i}" for i in range(n_mix)])
    Pm = pd.DataFrame(np.vstack(props), index=Bm.columns, columns=tl)
    return Bm, Pm

Bmix, Ptrue = make_mixtures()
log(f"built {Bmix.shape[1]} synthetic mixtures")

def score(est, truth):
    cols = [c for c in truth.columns if c in est.columns]
    e, t = est[cols].values, truth[cols].values
    return dict(rmse=float(np.sqrt(((e - t) ** 2).mean())),
                mae=float(np.abs(e - t).mean()),
                pearson=float(np.corrcoef(e.ravel(), t.ravel())[0, 1]),
                mac_rmse=float(np.sqrt(((est["Macrophage"] - truth["Macrophage"]) ** 2).mean()))
                          if "Macrophage" in cols else np.nan,
                mac_pearson=float(stats.pearsonr(est["Macrophage"], truth["Macrophage"])[0])
                          if "Macrophage" in cols else np.nan)

sig_bench = build_signature(ref_half)
rows = []
for lam in LAMBDAS:
    est, _ = deconvolve(sig_bench, Bmix, lam)
    s = score(est, Ptrue); s["lambda"] = lam; rows.append(s)
    log(f"  lambda={lam:5.1f}  RMSE={s['rmse']:.4f}  r={s['pearson']:.3f}  "
        f"macrophage RMSE={s['mac_rmse']:.4f} r={s['mac_pearson']:.3f}")
bench = pd.DataFrame(rows)[["lambda","rmse","mae","pearson","mac_rmse","mac_pearson"]]
bench["benchmark"] = "matched platform"

# The mixtures above are built from the same single-cell data as the reference,
# so they measure the estimator under ideal conditions. Real bulk RNA-seq is
# generated on a different platform from 10x single-cell data, which introduces
# gene-wise capture bias. The second benchmark reproduces that mismatch, and it
# is the one used to choose the ridge penalty actually applied to the bulk.
def platform_perturb(Bm, bias_sd=0.6, noise_sd=0.25):
    bias  = np.exp(rng.normal(0, bias_sd, (Bm.shape[0], 1)))
    noise = np.exp(rng.normal(0, noise_sd, Bm.shape))
    return pd.DataFrame(Bm.values * bias * noise, index=Bm.index, columns=Bm.columns)

Bmix_x = platform_perturb(Bmix)
rows_x = []
for lam in LAMBDAS:
    est, _ = deconvolve(sig_bench, Bmix_x, lam)
    sx = score(est, Ptrue); sx["lambda"] = lam; rows_x.append(sx)
    log(f"  [cross-platform] lambda={lam:5.1f}  RMSE={sx['rmse']:.4f}  "
        f"r={sx['pearson']:.3f}  macrophage r={sx['mac_pearson']:.3f}  "
        f"types recovered={int((est.mean(0) > 1e-4).sum())}/{est.shape[1]}")
bench_x = pd.DataFrame(rows_x)[["lambda","rmse","mae","pearson","mac_rmse","mac_pearson"]]
bench_x["benchmark"] = "cross platform"

pd.concat([bench, bench_x]).to_csv(f"{RES}/05_benchmark_lambda.csv", index=False)
BEST_LAM_MATCHED = float(bench.loc[bench.rmse.idxmin(), "lambda"])
BEST_LAM = float(bench_x.loc[bench_x.rmse.idxmin(), "lambda"])
log(f"lambda minimising RMSE: matched platform {BEST_LAM_MATCHED}, "
    f"cross platform {BEST_LAM} <- applied to the bulk cohort")

# ==================================== (2) does reference size change accuracy?
orig_idx = []
for gid, idx in A.obs.groupby("gid", observed=True).indices.items():
    pool = np.intersect1d(idx, ref_half)
    if len(pool) == 0: continue
    orig_idx.extend(rng.choice(pool, min(300, len(pool)), replace=False))
orig_idx = np.array(sorted(orig_idx))
sig_orig_bench = build_signature(orig_idx)
est_full, _ = deconvolve(sig_bench,      Bmix, BEST_LAM)
est_orig, _ = deconvolve(sig_orig_bench, Bmix, BEST_LAM)
sf, so = score(est_full, Ptrue), score(est_orig, Ptrue)
ref_cmp = pd.DataFrame([dict(reference="full atlas", n_cells=len(ref_half),
                             n_macrophages=int((labels.iloc[ref_half]=="Macrophage").sum()), **sf),
                        dict(reference="original 300/library design", n_cells=len(orig_idx),
                             n_macrophages=int((labels.iloc[orig_idx]=="Macrophage").sum()), **so)])
ref_cmp.to_csv(f"{RES}/05_reference_size_benchmark.csv", index=False)
log("\n=== ACCURACY vs REFERENCE SIZE (identical synthetic mixtures) ===")
log(ref_cmp.to_string(index=False))

# ================================================= (3) apply to the real bulk
ann = gene_annotation()
C134, m134 = load_counts("GSE134431", ann)
g134 = dfu_groups("GSE134431", m134)
L134   = cpm_log(C134)
Lin134 = C134.div(C134.sum(0), axis=1) * 1e6
log(f"\nGSE134431 {C134.shape}; groups {g134.value_counts().to_dict()}")

sig_full = build_signature(np.arange(A.n_obs))
frac_full, ngene = deconvolve(sig_full, Lin134, BEST_LAM)
log(f"signature {sig_full.shape[0]} markers, {ngene} shared with bulk; "
    f"macrophages in reference = {int((labels=='Macrophage').sum()):,}")
log(f"cell types with non-zero fraction in real bulk: {int((frac_full.mean(0) > 1e-4).sum())}/{frac_full.shape[1]}")
log("mean composition (%):\n" +
    (frac_full.mean(0)*100).round(2).sort_values(ascending=False).to_string())
frac_full.assign(group=g134).to_csv(f"{RES}/05_deconvolution_full_reference.csv")

orig_all = []
for gid, idx in A.obs.groupby("gid", observed=True).indices.items():
    orig_all.extend(rng.choice(idx, min(300, len(idx)), replace=False))
sig_orig = build_signature(np.array(sorted(orig_all)))
frac_orig, _ = deconvolve(sig_orig, Lin134, BEST_LAM)
frac_orig.assign(group=g134).to_csv(f"{RES}/05_deconvolution_original_design_reference.csv")

boot = []
for b in range(N_BOOT):
    bidx = rng.choice(A.n_obs, size=A.n_obs, replace=True)
    try:
        f, _ = deconvolve(build_signature(bidx, n_marker=60), Lin134, BEST_LAM)
        boot.append(f.reindex(columns=frac_full.columns).fillna(0))
    except Exception:
        continue
    if (b+1) % 25 == 0: log(f"  bootstrap {b+1}/{N_BOOT}")
Bt = np.stack([f.values for f in boot])
for nm, q in [("low", 2.5), ("high", 97.5)]:
    pd.DataFrame(np.percentile(Bt, q, 0), index=frac_full.index,
                 columns=frac_full.columns).assign(group=g134).to_csv(
        f"{RES}/05_deconvolution_bootstrap_CI_{nm}.csv")
log(f"bootstrap complete ({len(boot)} resamples)")

# ------------------------------------------- (4) orthogonal marker-based score
MAC = ["CD68","CD163","MRC1","MSR1","AIF1","C1QA","C1QB","C1QC","ITGAM",
       "FCGR1A","FCGR3A","CSF1R","MARCO","VSIG4","LYZ","TYROBP","FCER1G"]
pres = [g for g in MAC if g in L134.index]
z = L134.loc[pres].sub(L134.loc[pres].mean(1), axis=0).div(L134.loc[pres].std(1)+1e-9, axis=0)
mac_score = z.mean(0).rename("macrophage_marker_score")
log(f"marker score from {len(pres)} genes")

# --------------------------------------------------- comparisons and power
ulcer = g134[g134.isin(["Healing","Nonhealing"])].index
def compare(v, name):
    a = v.loc[[i for i in ulcer if g134[i]=="Nonhealing"]].astype(float)
    b = v.loc[[i for i in ulcer if g134[i]=="Healing"]].astype(float)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    t, pt = stats.ttest_ind(a, b, equal_var=False)
    sd = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return dict(measure=name, n_nonhealing=len(a), n_healing=len(b),
                mean_nonhealing=float(a.mean()), mean_healing=float(b.mean()),
                cohens_d=float((a.mean()-b.mean())/(sd+1e-12)),
                p_mannwhitney=float(p), p_welch=float(pt))

rows  = [compare(frac_full[c], f"NNLS_full_ref::{c}") for c in frac_full.columns]
rows += [compare(frac_orig[c], f"NNLS_orig_ref::{c}") for c in frac_orig.columns]
rows += [compare(mac_score, "marker_score::Macrophage")]
cmp_df = pd.DataFrame(rows).sort_values("p_mannwhitney")
cmp_df.to_csv(f"{RES}/05_abundance_group_tests.csv", index=False)
log("\n=== ABUNDANCE TESTS (non-healing vs healing ulcers) ===")
log(cmp_df[cmp_df.measure.str.contains("Macrophage|Dendritic|T cell|Keratinocyte|Fibroblast")]
    .to_string(index=False))

from statsmodels.stats.power import TTestIndPower
n1 = int((g134.loc[ulcer]=="Nonhealing").sum()); n2 = int((g134.loc[ulcer]=="Healing").sum())
mde = TTestIndPower().solve_power(effect_size=None, nobs1=n1, alpha=0.05, power=0.80,
                                  ratio=n2/n1, alternative="two-sided")
macf = frac_full.loc[ulcer, "Macrophage"]
sd = float(macf.std(ddof=1))
obs = float(macf[[i for i in ulcer if g134[i]=="Nonhealing"]].mean()
            - macf[[i for i in ulcer if g134[i]=="Healing"]].mean())
power = dict(n_nonhealing=n1, n_healing=n2, min_detectable_cohens_d=float(mde),
             macrophage_fraction_sd=sd,
             min_detectable_fraction_difference_pct=float(mde*sd*100),
             observed_fraction_difference_pct=float(obs*100),
             mean_macrophage_fraction_pct=float(macf.mean()*100))
json.dump(power, open(f"{RES}/05_power_analysis.json","w"), indent=2)
log("\n=== POWER FOR MACROPHAGE ABUNDANCE ===\n" + json.dumps(power, indent=2))

r, pr   = stats.pearsonr(frac_full["Macrophage"], frac_orig["Macrophage"])
sr, psr = stats.spearmanr(frac_full["Macrophage"], frac_orig["Macrophage"])
conc = dict(lambda_selected=BEST_LAM, lambda_matched_benchmark=BEST_LAM_MATCHED, pearson_r=float(r), pearson_p=float(pr),
            spearman_rho=float(sr), spearman_p=float(psr),
            n_macrophage_full=int((labels=="Macrophage").sum()),
            n_macrophage_original_design=int((labels.iloc[np.array(sorted(orig_all))]=="Macrophage").sum()),
            benchmark_rmse_full=sf["rmse"], benchmark_rmse_original=so["rmse"],
            benchmark_macrophage_r_full=sf["mac_pearson"],
            benchmark_macrophage_r_original=so["mac_pearson"])
json.dump(conc, open(f"{RES}/05_reference_concordance.json","w"), indent=2)
log("\n=== FULL vs ORIGINAL-DESIGN REFERENCE ===\n" + json.dumps(conc, indent=2))

pd.concat([frac_full.add_prefix("full::"), frac_orig.add_prefix("orig::"),
           mac_score, g134], axis=1).to_csv(f"{RES}/05_deconvolution_combined.csv")
Ptrue.to_csv(f"{RES}/05_benchmark_true_proportions.csv")
est_full.to_csv(f"{RES}/05_benchmark_estimated_full.csv")
est_orig.to_csv(f"{RES}/05_benchmark_estimated_original.csv")
log("=== STEP 05 COMPLETE ===")
