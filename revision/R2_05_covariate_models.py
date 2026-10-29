"""
R2_05 -- Primary single-cell healing contrast re-analysed with sex and the other
available covariates in the model (Editor comment 3), with confidence intervals
(comment 8) and sensitivity analyses for the independence structure (comment 1).

Unit of analysis: library. Endpoints
  (a) macrophage-state composition: quasi-binomial logistic GLM on cell counts
      (state cells / macrophage-monocyte cells), OR non-healing vs healing
  (b) 13 functional-programme scores (library means): OLS, difference in means
  (c) mean pseudotime per library: OLS
  (d) macrophage pseudobulk differential expression on RAW UMI counts
      (rebuilt in R2_02; previous revision summed log-normalised values),
      negative-binomial Wald test (PyDESeq2 0.5.2 via deseq_engine.py)
Models
  M0  ~ group                                   (19 libraries; previous analysis)
  M1  ~ sex + group
  M2  ~ sex + site + group                       PRIMARY adjusted model
  M3  ~ sex + site + log(median UMI) + median %MT + group   (technical)
  S1  foot libraries only, ~ sex + group         (14)
  S2  foot only, aggregated to putative donors (conservative pairs), ~ sex + group (12)
  S3  M2 excluding the two libraries with mixed XIST/Y signal (G2, G13) (17)
  S4  M2 with opposite-sex (contaminating) macrophages removed (composition only)
P values: model-based (t / Wald) and, for (a)-(c), a permutation test that
permutes the healing label within sex x site strata (exact enumeration).
"""
import sys, os, json, itertools; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats
from r2_common import *
import deseq_engine

cov = pd.read_csv(f"{OUT}/R2_04_library_covariates.csv", index_col=0)
mye = pd.read_csv(f"{AUX}/myeloid_obs_with_umap.csv", index_col=0)
odd = pd.read_csv(f"{R2IN}/R2_02b_opposite_sex_cells.csv")
odd_ids = set(odd.gid + "_" + odd.barcode)
STATE = {"Resident M2-like macrophage": "resident", "Inflammatory monocyte-macrophage": "inflammatory",
         "Lipid-associated macrophage": "lipid"}
mye["state"] = mye.myeloid_subtype.map(STATE)
mac = mye[mye.state.notna()].copy()
mac["odd"] = mac.index.isin(odd_ids)

base = cov[cov.group.isin(["Healing", "Nonhealing"]) & (cov.site != "PBMC") & (cov.n_mac >= 10)].copy()
assert len(base) == 19, len(base)
base["y"] = (base.group == "Nonhealing").astype(float)
base["male"] = (base.sex == "male").astype(float)
base["forearm"] = (base.site == "Forearm").astype(float)
base["log_umi"] = np.log(base.median_umi)
# conservative donor key for S2: merge only the pairs that passed the calibrated threshold
pairs = pd.read_csv(f"{OUT}/R2_04_donor_pair_distances.csv")
donor_key = {g: g for g in base.index}
for _, r in pairs[pairs.below_different_donor_1pct & (pairs.tissue_a == "Foot") & (pairs.tissue_b == "Foot")].iterrows():
    donor_key[r.lib_b] = donor_key[r.lib_a]
base["donor_key"] = [donor_key[g] for g in base.index]

MODELS = {
    "M0": dict(rows=lambda d: d.index, terms=[]),
    "M1": dict(rows=lambda d: d.index, terms=["male"]),
    "M2": dict(rows=lambda d: d.index, terms=["male", "forearm"]),
    "M3": dict(rows=lambda d: d.index, terms=["male", "forearm", "log_umi", "median_pctMT"]),
    "S1": dict(rows=lambda d: d.index[d.site == "Foot"], terms=["male"]),
    "S2": dict(rows=None, terms=["male"]),                      # donor-aggregated, built below
    "S3": dict(rows=lambda d: d.index[~d.index.isin(["G2", "G13"])], terms=["male", "forearm"]),
}
LABEL = {"M0": "unadjusted (previous analysis), 19 libraries",
         "M1": "+ sex, 19 libraries", "M2": "+ sex + tissue site (primary), 19 libraries",
         "M3": "+ sex + site + sequencing depth + %mito, 19 libraries",
         "S1": "foot libraries only, + sex, 14 libraries",
         "S2": "foot only, aggregated to putative donors, + sex, 12 donors",
         "S3": "primary model excluding mixed-sex libraries G2/G13, 17 libraries",
         "S4": "primary model with opposite-sex macrophages removed, 19 libraries"}

def design(d, terms):
    return np.column_stack([np.ones(len(d))] + [d[t].values for t in terms] + [d["y"].values])

def strata_perms(d, terms, max_n=50000, rng=np.random.default_rng(SEED)):
    """All (or sampled) label arrangements that permute the healing label within
    strata defined by the categorical covariates in the model."""
    cat = [t for t in terms if t in ("male", "forearm")]
    key = d[cat].astype(int).astype(str).agg("".join, axis=1) if cat else pd.Series("all", index=d.index)
    groups = [np.where(key.values == k)[0] for k in key.unique()]
    y = d["y"].values
    per_stratum = []
    for idx in groups:
        k = int(y[idx].sum())
        per_stratum.append([np.array(c) for c in itertools.combinations(idx, k)])
    total = np.prod([len(p) for p in per_stratum])
    def make(choice):
        yy = np.zeros(len(y))
        for c in choice: yy[c] = 1.0
        return yy
    if total <= max_n:
        for choice in itertools.product(*per_stratum):
            yield make(choice)
    else:
        for _ in range(max_n):
            yield make([p[rng.integers(len(p))] for p in per_stratum])

# ------------------------------------------------------------ estimators
def binom_glm(succ, trials, X, n_iter=100):
    yprop = succ / trials
    b = np.zeros(X.shape[1]); b[0] = np.log((yprop.mean() + 1e-3) / (1 - yprop.mean() + 1e-3))
    for _ in range(n_iter):
        eta = X @ b; p = 1 / (1 + np.exp(-eta))
        w = trials * p * (1 - p) + 1e-10
        z = eta + (yprop - p) / (p * (1 - p) + 1e-10)
        bn = np.linalg.solve(X.T @ (w[:, None] * X) + 1e-8 * np.eye(X.shape[1]), X.T @ (w * z))
        if np.max(np.abs(bn - b)) < 1e-10: b = bn; break
        b = bn
    p = 1 / (1 + np.exp(-(X @ b)))
    w = trials * p * (1 - p)
    covb = np.linalg.inv(X.T @ (w[:, None] * X))
    n, k = X.shape
    phi = max(float(np.sum((succ - trials * p) ** 2 / (trials * p * (1 - p) + 1e-12)) / (n - k)), 1.0)
    return b, covb * phi, n - k

def ols(yv, X):
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r = yv - X @ b; n, k = X.shape
    s2 = r @ r / (n - k)
    return b, s2 * np.linalg.inv(X.T @ X), n - k

def summarise(est, se, df, log_scale):
    t = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(est / se), df)
    if log_scale:
        return dict(effect=float(np.exp(est)), ci_low=float(np.exp(est - t * se)), ci_high=float(np.exp(est + t * se)), p_model=float(p))
    return dict(effect=float(est), ci_low=float(est - t * se), ci_high=float(est + t * se), p_model=float(p))

# ----------------------------------------- donor-aggregated table (S2)
def aggregate_donors(d, value_cols=None, count_cols=None):
    f = d[d.site == "Foot"].copy()
    agg = {"y": "first", "male": "first"}
    if count_cols:
        for c in count_cols: agg[c] = "sum"
    g = f.groupby("donor_key").agg(agg)
    if value_cols:   # cell-weighted means
        for c in value_cols:
            g[c] = f.groupby("donor_key").apply(lambda x: np.average(x[c], weights=x["n_mac"]), include_groups=False)
    return g

# =============================================== (a) state composition
comp_rows = []
SKIP = os.environ.get('R2_DE_ONLY') == '1'
for state in ([] if SKIP else ["resident", "inflammatory", "lipid"]):
    base[f"n_{state}_clean"] = mac[(mac.state == state) & ~mac.odd].groupby("gid").size().reindex(base.index).fillna(0)
    base["n_mac_clean"] = mac[~mac.odd].groupby("gid").size().reindex(base.index).fillna(0)
    for m in list(MODELS) + ["S4"]:
        if m == "S2":
            d = aggregate_donors(base, count_cols=[f"n_{state}", "n_mac"])
        elif m == "S4":
            d = base.copy(); d[f"n_{state}"] = d[f"n_{state}_clean"]; d["n_mac"] = d["n_mac_clean"]
        else:
            d = base.loc[MODELS[m]["rows"](base)]
        terms = MODELS["M2"]["terms"] if m == "S4" else MODELS[m]["terms"]
        X = design(d, terms)
        succ, tri = d[f"n_{state}"].values.astype(float), d["n_mac"].values.astype(float)
        b, cb, df = binom_glm(succ, tri, X)
        res = summarise(b[-1], np.sqrt(cb[-1, -1]), df, True)
        obs = b[-1]; ge = n = 0
        for yy in strata_perms(d, terms):
            Xp = X.copy(); Xp[:, -1] = yy
            bp, _, _ = binom_glm(succ, tri, Xp)
            ge += abs(bp[-1]) >= abs(obs) - 1e-12; n += 1
        res.update(state=state, model=m, model_label=LABEL[m], n_units=len(d),
                   n_nonhealing=int(d.y.sum()), p_permutation=ge / n, n_permutations=n,
                   pct_nonhealing_crude=float((succ[d.y == 1] / tri[d.y == 1]).mean() * 100),
                   pct_healing_crude=float((succ[d.y == 0] / tri[d.y == 0]).mean() * 100))
        comp_rows.append(res)
    print("composition done:", state, flush=True)
COMP = pd.read_csv(f"{OUT}/R2_05_state_composition_models.csv") if SKIP else pd.DataFrame(comp_rows)[["state", "model", "model_label", "n_units", "n_nonhealing", "pct_nonhealing_crude",
                                "pct_healing_crude", "effect", "ci_low", "ci_high", "p_model", "p_permutation", "n_permutations"]]
if not SKIP: COMP = COMP.rename(columns={"effect": "odds_ratio_NH_vs_H"})
if not SKIP: COMP.to_csv(f"{OUT}/R2_05_state_composition_models.csv", index=False)

# =============================================== (b) programme scores, (c) pseudotime
ms = [c for c in mac.columns if c.startswith("ms_")]
prog = mac.groupby("gid")[ms].mean().reindex(base.index)
for c in ms: base[c] = prog[c]
pt = pd.read_csv(f"{RES}/12_macrophage_pseudotime_metadata.csv", index_col=0)
base["pseudotime"] = pt.groupby("gid").pseudotime.mean().reindex(base.index)
cont_rows = []
for v in ([] if SKIP else ms + ["pseudotime"]):
    for m in MODELS:
        if m == "S2":
            d = aggregate_donors(base, value_cols=[v], count_cols=["n_mac"])
        else:
            d = base.loc[MODELS[m]["rows"](base)]
        terms = MODELS[m]["terms"]
        X = design(d, terms); yv = d[v].values
        b, cb, df = ols(yv, X)
        res = summarise(b[-1], np.sqrt(cb[-1, -1]), df, False)
        ge = n = 0
        for yy in strata_perms(d, terms):
            Xp = X.copy(); Xp[:, -1] = yy
            bp, *_ = np.linalg.lstsq(Xp, yv, rcond=None)
            ge += abs(bp[-1]) >= abs(b[-1]) - 1e-12; n += 1
        res.update(endpoint=v.replace("ms_", ""), model=m, model_label=LABEL[m], n_units=len(d),
                   n_nonhealing=int(d.y.sum()), p_permutation=ge / n, n_permutations=n,
                   mean_nonhealing=float(yv[d.y == 1].mean()), mean_healing=float(yv[d.y == 0].mean()))
        cont_rows.append(res)
CONT = pd.read_csv(f"{OUT}/R2_05_programme_and_pseudotime_models.csv") if SKIP else pd.DataFrame(cont_rows).rename(columns={"effect": "difference_NH_minus_H"})
if not SKIP: CONT["q_BH_within_model"] = np.nan
for m in ([] if SKIP else CONT.model.unique()):
    k = (CONT.model == m) & (CONT.endpoint != "pseudotime")
    CONT.loc[k, "q_BH_within_model"] = deseq_engine.bh(CONT.loc[k, "p_model"].values)
if not SKIP: CONT = CONT[["endpoint", "model", "model_label", "n_units", "n_nonhealing", "mean_nonhealing", "mean_healing",
             "difference_NH_minus_H", "ci_low", "ci_high", "p_model", "q_BH_within_model", "p_permutation", "n_permutations"]]
if not SKIP: CONT.to_csv(f"{OUT}/R2_05_programme_and_pseudotime_models.csv", index=False)
print("programmes done", flush=True)

# =============================================== (d) raw-count pseudobulk DE
PBR = pd.read_csv(f"{R2IN}/R2_02_raw_pseudobulk_macrophage.csv", index_col=0)
atlas_genes = pd.read_csv(f"{AUX}/atlas_genes.csv").gene
PBR = PBR.loc[PBR.index.intersection(atlas_genes)]
# audit: raw pseudobulk cell totals vs the annotated macrophage numbers
de_rows, de_tables = [], {}
def run_de(name, cols, terms, counts=PBR, d=None):
    d = base.loc[cols] if d is None else d
    K = counts[d.index]
    keep = (K > 0).sum(1) >= max(3, int(0.2 * len(d)))
    K = K[keep]
    X = design(d, terms)
    r = deseq_engine.deseq(K.values, X, coef=X.shape[1] - 1)
    T = pd.DataFrame({k: r[k] for k in ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]}, index=K.index)
    T["ci_low"] = T.log2FoldChange - 1.96 * T.lfcSE; T["ci_high"] = T.log2FoldChange + 1.96 * T.lfcSE
    T = T.sort_values("pvalue")
    T.to_csv(f"{OUT}/R2_05_pseudobulk_DE_{name}.csv")
    de_tables[name] = T
    de_rows.append(dict(model=name, model_label=LABEL.get(name, name), n_units=len(d), n_nonhealing=int(d.y.sum()),
                        n_genes_tested=int(T.pvalue.notna().sum()), n_FDR05=int((T.padj < 0.05).sum()),
                        n_nominal05=int((T.pvalue < 0.05).sum()),
                        expected_nominal_under_null=round(0.05 * T.pvalue.notna().sum()),
                        top_genes=", ".join(T.index[:10]), FDR05_genes=", ".join(T.index[T.padj < 0.05][:40]),
                        cooks_flagged=r["n_cooks_flagged"], separated_NA=r["n_separated"]))
    print(name, de_rows[-1]["n_FDR05"], de_rows[-1]["top_genes"], flush=True)
for m in ["M0", "M1", "M2", "S1", "S3"]:
    run_de(m, MODELS[m]["rows"](base), MODELS[m]["terms"])
# donor-aggregated foot libraries: sum raw counts within putative donor
f = base[base.site == "Foot"]
dk = f.groupby("donor_key").agg(y=("y", "first"), male=("male", "first"))
Kd = pd.DataFrame({k: PBR[f.index[f.donor_key == k]].sum(1) for k in dk.index})
run_de("S2", None, ["male"], counts=Kd, d=dk)
DE = pd.DataFrame(de_rows)
DE.to_csv(f"{OUT}/R2_05_pseudobulk_DE_summary.csv", index=False)

# concordance with the previous (log-sum) analysis
old = pd.read_csv(f"{RES}/04_pseudobulk_DESeq2_NHvsH.csv", index_col=0)
j = de_tables["M0"].join(old, rsuffix="_previous", how="inner").dropna(subset=["log2FoldChange", "log2FoldChange_previous"])
conc = dict(n=len(j), lfc_spearman_raw_vs_previous=float(stats.spearmanr(j.log2FoldChange, j.log2FoldChange_previous)[0]),
            previous_FDR05_genes=old.index[old.padj < 0.05].tolist(),
            previous_FDR05_gene_now=de_tables["M0"].loc[old.index[old.padj < 0.05].intersection(de_tables["M0"].index), ["log2FoldChange", "pvalue", "padj"]].to_dict("index"))
j2 = de_tables["M2"].join(de_tables["M0"], rsuffix="_M0", how="inner").dropna(subset=["log2FoldChange", "log2FoldChange_M0"])
conc["lfc_spearman_M2_vs_M0"] = float(stats.spearmanr(j2.log2FoldChange, j2.log2FoldChange_M0)[0])
json.dump(conc, open(f"{OUT}/R2_05_pseudobulk_concordance.json", "w"), indent=2, default=str)

base.to_csv(f"{OUT}/R2_05_library_level_data.csv")
print(COMP[COMP.state == "resident"].round(3).to_string(index=False))
print(CONT[CONT.model == "M2"].round(3).to_string(index=False))
print(DE.to_string(index=False)); print(json.dumps(conc, indent=1, default=str)[:1500])
