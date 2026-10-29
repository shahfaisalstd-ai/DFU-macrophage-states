"""
R2_09 -- 95% confidence intervals for every group-effect estimate and state
proportion quoted in the manuscript (Editor comment 8), computed with the
library as the unit of analysis.
  * group means of state proportions: bootstrap over libraries (10,000)
  * difference in means: stratified bootstrap (10,000)
  * Hodges-Lehmann shift: distribution-free (Moses) CI from the Mann-Whitney
    critical value; Mann-Whitney P reported alongside
Covers: macrophage-state composition, functional programmes, pseudotime,
deconvolution / marker-score abundance, patient-level communication contrasts,
prioritised-gene per-library expression.
"""
import sys, os, json; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats
from r2_common import *
rng = np.random.default_rng(SEED)

def boot_mean(v, B=10000):
    v = np.asarray(v, float); m = rng.choice(v, (B, len(v))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def boot_diff(a, b, B=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = rng.choice(a, (B, len(a))).mean(1) - rng.choice(b, (B, len(b))).mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
def hl_moses(a, b, alpha=0.05):
    a, b = np.asarray(a, float), np.asarray(b, float); n1, n2 = len(a), len(b)
    D = np.sort((a[:, None] - b[None, :]).ravel())
    # exact lower critical value of U (scipy's exact null distribution)
    from scipy.stats import mannwhitneyu
    Ks = np.arange(0, n1 * n2 // 2 + 1)
    # P(U <= k) under H0 via scipy exact cdf
    dist = stats._mannwhitneyu._MWU(n1, n2)
    cdf = np.array([dist.cdf(k) for k in Ks])
    K = int(Ks[cdf <= alpha / 2].max()) if (cdf <= alpha / 2).any() else 0
    lo, hi = D[K], D[n1 * n2 - K - 1]
    return float(np.median(D)), float(lo), float(hi)
def row(name, endpoint, a, b, unit="library", scale=1.0, extra=None):
    a, b = np.asarray(a, float) * scale, np.asarray(b, float) * scale
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    ma, mb = a.mean(), b.mean(); la, ha = boot_mean(a); lb, hb = boot_mean(b); ld, hd = boot_diff(a, b)
    hl, hll, hlh = hl_moses(a, b); p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    d = dict(analysis=name, endpoint=endpoint, unit=unit, n_nonhealing=len(a), n_healing=len(b),
             mean_nonhealing=ma, ci_nonhealing_low=la, ci_nonhealing_high=ha,
             mean_healing=mb, ci_healing_low=lb, ci_healing_high=hb,
             difference_NH_minus_H=ma - mb, diff_ci_low=ld, diff_ci_high=hd,
             hodges_lehmann=hl, hl_ci_low=hll, hl_ci_high=hlh, p_mannwhitney=float(p))
    if extra: d.update(extra)
    return d

rows = []
L = pd.read_csv(f"{OUT}/R2_05_library_level_data.csv", index_col=0)
NH, H = L[L.group == "Nonhealing"], L[L.group == "Healing"]
for s, col in [("resident", "n_resident"), ("inflammatory", "n_inflammatory"), ("lipid-associated", "n_lipid")]:
    rows.append(row("macrophage-state composition (% of macrophage/monocyte cells)", s, NH[col] / NH.n_mac, H[col] / H.n_mac, scale=100))
    f = L[L.site == "Foot"]; fn, fh = f[f.group == "Nonhealing"], f[f.group == "Healing"]
    rows.append(row("macrophage-state composition, foot libraries only (%)", s, fn[col] / fn.n_mac, fh[col] / fh.n_mac, scale=100))
for c in [c for c in L.columns if c.startswith("ms_")]:
    rows.append(row("functional programme score (library mean)", c.replace("ms_", ""), NH[c], H[c]))
rows.append(row("mean pseudotime (monocyte-rooted)", "pseudotime", NH.pseudotime, H.pseudotime))
dc = pd.read_csv(f"{RES}/05_deconvolution_combined.csv", index_col=0)
grp = dc.iloc[:, -1]
for c in ["full::Macrophage", "macrophage_marker_score"]:
    rows.append(row("bulk GSE134431 abundance", c, dc.loc[grp == "Nonhealing", c], dc.loc[grp == "Healing", c],
                    unit="bulk sample", scale=100 if c.startswith("full") else 1))
pp = pd.read_csv(f"{RES}/18_prioritised_per_patient.csv")
for g in [c for c in pp.columns if c not in ("gid", "group")]:
    rows.append(row("prioritised gene, mean log-expression per library", g, pp.loc[pp.group == "Nonhealing", g], pp.loc[pp.group == "Healing", g]))
cm = pd.read_csv(f"{RES}/13_macrophage_communication_per_library.csv")
top = pd.read_csv(f"{RES}/13_macrophage_contrast_patient_level.csv").nsmallest(6, "p")
for _, t in top.iterrows():
    s = cm[(cm.pair == t.pair) & (cm.direction == t.direction)].groupby(["gid", "group"]).score.sum().reset_index()
    rows.append(row(f"macrophage communication score, {t.direction}", t.pair, s.loc[s.group == "Nonhealing", "score"],
                    s.loc[s.group == "Healing", "score"], extra=dict(previous_padj=t.padj)))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/R2_09_effect_estimates_CIs.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(3).iloc[:, :18].to_string(index=False))
