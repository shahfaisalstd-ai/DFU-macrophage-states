"""
R2_04 -- Sample-level supplementary table (Editor comment 1) and cell-count
reconciliation (Editor comment 2), built from the raw GEO matrices.

Inputs : GEO series matrices + phenotype table (original pipeline 00_inputs/raw_data),
         R2_01 per-cell QC and whole-library pseudobulk (raw GEO counts),
         R2_02b per-cell sex check, myeloid cell annotations (03_myeloid.h5ad,
         exported to myeloid_obs_with_umap.csv), original-pipeline per-library QC table.
Outputs: R2_04_TableS1_GSE165816_libraries.csv, R2_04_TableS1_bulk_samples.csv,
         R2_04_TableS2_cell_reconciliation.csv, R2_04_library_covariates.csv,
         R2_04_donor_proxy_markers.csv, R2_04_summary.json
"""
import sys, os, json; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from r2_common import *

ph = pd.read_csv(f"{RAWD}/GSE165816_phenotype.csv").set_index("gid")
gm = series_meta(f"{RAWD}/GSE165816_series_matrix.txt.gz")
qc = pd.read_csv(f"{R2IN}/R2_01_cell_qc.csv.gz")
PB = pd.read_csv(f"{R2IN}/R2_01_library_pseudobulk_allcells.csv", index_col=0)
k2qc = pd.read_csv(f"{RES}/01_persample_QC.csv").set_index("gid")
mye = pd.read_csv(f"{AUX}/myeloid_obs_with_umap.csv", index_col=0)
sexcheck = pd.read_csv(f"{R2IN}/R2_02b_percell_sex_check.csv").set_index("gid")
oddcells = pd.read_csv(f"{R2IN}/R2_02b_opposite_sex_cells.csv")

# ------------------------------------------------------------- per-library QC
qc["pass"] = (qc.nFeature >= 200) & (qc.nFeature <= 5000) & (qc.pctMT < 10)
L = qc.groupby("gid").agg(gsm=("gsm", "first"), n_cells_geo=("barcode", "size"),
                          n_cells_qc=("pass", "sum"),
                          median_umi_all=("nCount", "median"))
L["median_umi_qc"] = qc[qc["pass"]].groupby("gid").nCount.median()
L["median_genes_qc"] = qc[qc["pass"]].groupby("gid").nFeature.median()
L["median_pctMT_qc"] = qc[qc["pass"]].groupby("gid").pctMT.median()
# audit: identical to the original-pipeline QC table?
L["audit_raw_matches_khan2"] = L.n_cells_geo == k2qc.loc[L.index, "n_cells_raw"]
L["audit_qc_matches_khan2"] = L.n_cells_qc == k2qc.loc[L.index, "n_cells_pass"]

# --------------------------------------------------------------- sex (all 54)
cpm = PB / PB.sum() * 1e6
Yg = ["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "USP9Y", "ZFY"]
L["XIST_log2cpm"] = np.log2(cpm.loc["XIST", L.index] + 1)
L["Ygenes_log2cpm"] = np.log2(cpm.loc[[g for g in Yg if g in cpm.index], L.index].sum() + 1)
L["inferred_sex"] = np.where(L.Ygenes_log2cpm > 1, "male", "female")
L["mixed_sex_signal"] = (L.Ygenes_log2cpm > 1) & (L.XIST_log2cpm > 2.5)

# ------------------------------------------------- germline (genotype) proxies
def lr(a, b): return np.log2((cpm.loc[a] + 5) / (cpm.loc[b] + 5))
G = pd.DataFrame({"ERAP2_over_ERAP1": lr("ERAP2", "ERAP1"),
                  "GSTM1_over_GSTM2": lr("GSTM1", "GSTM2"),
                  "HLA_DRB5_over_DRB1": lr("HLA-DRB5", "HLA-DRB1"),
                  "CCL3L1_over_CCL3": lr("CCL3L1", "CCL3"),
                  "CCL4L2_over_CCL4": lr("CCL4L2", "CCL4"),
                  "KANSL1AS1_over_KANSL1": lr("KANSL1-AS1", "KANSL1"),
                  "UGT2B17_log2cpm": np.log2(cpm.loc["UGT2B17"] + 1)}).loc[L.index]
G["inferred_sex"] = L.inferred_sex
G.to_csv(f"{OUT}/R2_04_donor_proxy_markers.csv")

# Putative donor structure among the DFU skin libraries. Pairs are called only
# when sex agrees and every proxy marker agrees within a tolerance that no pair
# of libraries from different-sex donors meets (calibrated below), so the
# grouping is conservative; it is reported as inferred, not as ground truth.
dfu = L.index[ph.loc[L.index, "group"].isin(["Healing", "Nonhealing"]) & (ph.loc[L.index, "tissue"] != "PBMC")]
cont = ["ERAP2_over_ERAP1", "GSTM1_over_GSTM2", "HLA_DRB5_over_DRB1", "CCL3L1_over_CCL3",
        "CCL4L2_over_CCL4", "KANSL1AS1_over_KANSL1", "UGT2B17_log2cpm"]
Z = (G[cont] - G.loc[ph.index[ph.tissue != "PBMC"], cont].mean()) / G.loc[ph.index[ph.tissue != "PBMC"], cont].std()
def dist(a, b): return float(np.sqrt(((Z.loc[a] - Z.loc[b]) ** 2).mean()))
skin = [g for g in L.index if ph.loc[g, "tissue"] != "PBMC"]
# calibration: distances between libraries known to come from different people
# (opposite inferred sex, mixed-signal libraries excluded)
cal = [dist(a, b) for i, a in enumerate(skin) for b in skin[i + 1:]
       if G.loc[a, "inferred_sex"] != G.loc[b, "inferred_sex"]
       and not L.loc[a, "mixed_sex_signal"] and not L.loc[b, "mixed_sex_signal"]]
thr = float(np.percentile(cal, 1))
pairs = []
for i, a in enumerate(dfu):
    for b in dfu[i + 1:]:
        if ph.loc[a, "group"] != ph.loc[b, "group"]: continue
        if G.loc[a, "inferred_sex"] != G.loc[b, "inferred_sex"]: continue
        d = dist(a, b)
        pairs.append(dict(lib_a=a, lib_b=b, tissue_a=ph.loc[a, "tissue"], tissue_b=ph.loc[b, "tissue"],
                          group=ph.loc[a, "group"], sex=G.loc[a, "inferred_sex"], proxy_distance=d,
                          below_different_donor_1pct=d < thr))
pairs = pd.DataFrame(pairs).sort_values("proxy_distance")
pairs["calibration_threshold_1pct_of_different_donor_pairs"] = thr
pairs.to_csv(f"{OUT}/R2_04_donor_pair_distances.csv", index=False)

# donor groups: connected components of foot-foot pairs below threshold;
# forearm libraries attached to their nearest same-group, same-sex foot donor
foot = [g for g in dfu if ph.loc[g, "tissue"] == "Foot"]
parent = {g: g for g in dfu}
def find(x):
    while parent[x] != x: x = parent[x]
    return x
for _, r in pairs[pairs.below_different_donor_1pct & (pairs.tissue_a == "Foot") & (pairs.tissue_b == "Foot")].iterrows():
    parent[find(r.lib_b)] = find(r.lib_a)
donor = {}
roots = sorted({find(g) for g in foot}, key=lambda r: (ph.loc[r, "group"], int(r[1:])))
cnt = {"Healing": 0, "Nonhealing": 0}
for r in roots:
    grp = ph.loc[r, "group"]; cnt[grp] += 1
    lab = ("H" if grp == "Healing" else "NH") + f"-D{cnt[grp]}"
    for g in foot:
        if find(g) == r: donor[g] = lab
forearm_match = {}
for g in [x for x in dfu if ph.loc[x, "tissue"] == "Forearm"]:
    cands = [(dist(g, f), f) for f in foot if ph.loc[f, "group"] == ph.loc[g, "group"]
             and G.loc[f, "inferred_sex"] == G.loc[g, "inferred_sex"]]
    cands.sort()
    best = cands[0] if cands else (np.nan, None)
    second = cands[1][0] if len(cands) > 1 else np.nan
    forearm_match[g] = dict(nearest_foot=best[1], distance=best[0], second_distance=second,
                            unambiguous=bool(best[1] is not None and (np.isnan(second) or second - best[0] > 0.25)))
    donor[g] = (donor[best[1]] + " (forearm, inferred)") if best[1] else "unassigned"

# ------------------------------------------------------- myeloid / macrophage
mac_states = {"Resident M2-like macrophage": "resident", "Inflammatory monocyte-macrophage": "inflammatory",
              "Lipid-associated macrophage": "lipid_associated"}
mye["state"] = mye.myeloid_subtype.map(mac_states)
L["n_atlas_cells"] = 0
skin_ix = [g for g in L.index if ph.loc[g, "tissue"] != "PBMC"]
L.loc[skin_ix, "n_atlas_cells"] = L.loc[skin_ix, "n_cells_qc"]
L["n_myeloid"] = mye.groupby("gid").size().reindex(L.index).fillna(0).astype(int)
L["n_macrophage_monocyte"] = mye[mye.state.notna()].groupby("gid").size().reindex(L.index).fillna(0).astype(int)
for s in ["resident", "inflammatory", "lipid_associated"]:
    L[f"n_{s}"] = mye[mye.state == s].groupby("gid").size().reindex(L.index).fillna(0).astype(int)
odd = set(oddcells.gid + "_" + oddcells.barcode)
mye["opposite_sex_cell"] = mye.index.isin(odd)
L["n_macrophage_opposite_sex"] = mye[mye.state.notna() & mye.opposite_sex_cell].groupby("gid").size().reindex(L.index).fillna(0).astype(int)

# --------------------------------------------------------------- the table
T = pd.DataFrame(index=L.index)
T["GEO_series"] = "GSE165816"
T["GEO_sample_GSM"] = L.gsm
T["library_id"] = L.index
T["GEO_title"] = gm.loc[L.gsm, "title"].values
T["patient_identifier"] = "not released in GEO (source study: 27 subjects, 54 libraries)"
T["putative_donor_group"] = [donor.get(g, "not inferred (not a DFU skin library)") for g in L.index]
T["tissue_site"] = ph.loc[L.index, "tissue"]
T["GEO_disease_label"] = gm.loc[L.gsm, "disease"].values
T["analysis_group"] = ph.loc[L.index, "group"]
T["healing_group_definition"] = np.where(T.analysis_group.isin(["Healing", "Nonhealing"]),
    "Healer/non-healer at 12-week follow-up (source study)", "not applicable")
T["sex_inferred"] = L.inferred_sex
T["sex_source"] = "inferred from XIST and Y-chromosome transcripts (not reported in GEO)"
T["XIST_log2CPM"] = L.XIST_log2cpm.round(2); T["Y_genes_log2CPM"] = L.Ygenes_log2cpm.round(2)
T["mixed_XIST_Y_signal"] = L.mixed_sex_signal
T["clinical_covariates_available"] = "none at sample level (age, HbA1c, BMI, ulcer duration/size, infection, ischaemia, treatment not released)"
T["sequencing_platform"] = "Illumina NovaSeq 6000 (GPL24676)"
T["library_chemistry"] = "10x Genomics Chromium 3' v2 or v3 (per-library version not specified in GEO)"
T["cells_in_GEO_matrix"] = L.n_cells_geo
T["cells_passing_QC"] = L.n_cells_qc
T["median_UMI_per_cell_QC"] = L.median_umi_qc
T["median_genes_per_cell_QC"] = L.median_genes_qc
T["median_pct_mito_QC"] = L.median_pctMT_qc.round(2)
T["included_in_skin_atlas"] = T.tissue_site != "PBMC"
T["exclusion_reason"] = np.where(T.tissue_site == "PBMC", "PBMC library: not skin tissue", "")
T["myeloid_cells"] = L.n_myeloid
T["macrophage_monocyte_cells"] = L.n_macrophage_monocyte
T["resident_state_cells"] = L.n_resident
T["inflammatory_state_cells"] = L.n_inflammatory
T["lipid_associated_state_cells"] = L.n_lipid_associated
T["in_macrophage_pseudobulk_(>=10_cells)"] = T.included_in_skin_atlas & (L.n_macrophage_monocyte >= 10)
T["in_primary_healing_contrast"] = T.analysis_group.isin(["Healing", "Nonhealing"]) & T["in_macrophage_pseudobulk_(>=10_cells)"]
T["in_foot_only_sensitivity"] = T.in_primary_healing_contrast & (T.tissue_site == "Foot")
T["opposite_sex_cells_in_library"] = [int(sexcheck.loc[g, "xist_only"]) if g in sexcheck.index and L.loc[g, "inferred_sex"] == "male" else (0 if g in sexcheck.index else np.nan) for g in L.index]
T["opposite_sex_macrophages"] = L.n_macrophage_opposite_sex
def indep(g):
    t, grp = T.loc[g, "tissue_site"], T.loc[g, "analysis_group"]
    if grp in ("Healing", "Nonhealing"):
        if t == "Foot":
            return ("Foot ulcer/peri-wound library. Source study: 14 foot libraries from 11 DFU patients (7 healers, 4 non-healers); "
                    "libraries sharing a putative donor group are probably wound and peri-wound tissue of one patient.")
        if t == "Forearm":
            return "Forearm skin from a DFU patient who also contributed foot tissue (source study); not independent of that patient's foot library."
        return "PBMC from a DFU patient who also contributed skin; excluded from all analyses."
    if t == "PBMC":
        return "PBMC; excluded."
    return "Control library (non-diabetic or diabetic without DFU); foot/forearm pairs from one subject possible; not used in the healing contrast."
T["independence_structure"] = [indep(g) for g in T.index]
order = sorted(T.index, key=lambda g: (int(''.join(c for c in g[1:] if c.isdigit())), g))
T = T.loc[order]
T.to_csv(f"{OUT}/R2_04_TableS1_GSE165816_libraries.csv", index=False)

# library-level covariates used by the models
cov = pd.DataFrame({"group": T.analysis_group, "site": T.tissue_site, "sex": T.sex_inferred,
                    "donor": T.putative_donor_group, "mixed_sex": T.mixed_XIST_Y_signal,
                    "median_umi": T.median_UMI_per_cell_QC, "median_pctMT": T.median_pct_mito_QC,
                    "n_cells_qc": T.cells_passing_QC, "n_mac": T.macrophage_monocyte_cells,
                    "n_resident": T.resident_state_cells, "n_inflammatory": T.inflammatory_state_cells,
                    "n_lipid": T.lipid_associated_state_cells,
                    "n_mac_opposite_sex": T.opposite_sex_macrophages})
cov.to_csv(f"{OUT}/R2_04_library_covariates.csv")

# ------------------------------------------------------------ bulk cohorts
ann = gene_annotation()
brow = []
for prefix, site, ref in [("GSE134431", "Foot (ulcer edge/bed or intact foot skin)", "Sawaya et al. 2020, Nat Commun (PMID 32938916)"),
                          ("GSE143735", "Volar forearm skin (non-ulcerated)", "Theocharidis et al. 2020, Diabetes (PMID 32763913)")]:
    C, m = load_counts(prefix, ann); g = dfu_groups(prefix, m)
    sexf = pd.read_csv(f"{RES}/16b_{prefix}_inferred_sex.csv", index_col=0)
    for s in m.index:
        tp = ""
        if prefix == "GSE134431":
            t = m.loc[s, "title"]
            tp = "week 0" if "W0" in t else ("week 4" if "W4" in t else ("not stated (P2 cohort)" if "P2#" in t else ""))
        brow.append(dict(GEO_series=prefix, GEO_sample_GSM=s, sample_title=m.loc[s, "title"],
                         patient_identifier=m.loc[s, "title"].split("Patient ")[-1].split(", ")[-1],
                         analysis_group=g[s], tissue_site=site,
                         age_years=m.loc[s, "age"] if "age" in m.columns else "not reported",
                         sex_reported=m.loc[s, "Sex"] if "Sex" in m.columns else "not reported",
                         sex_inferred=sexf.loc[s, sexf.columns[-1]] if s in sexf.index else "",
                         biopsy_timepoint=tp,
                         sequencing_platform=f"{m.loc[s, 'instrument_model']} ({m.loc[s, 'platform_id']})",
                         library_size_protein_coding=int(C[s].sum()),
                         in_healing_contrast=g[s] in ("Healing", "Nonhealing"),
                         role=("bulk corroboration / panel construction (not independent of the panel)" if prefix == "GSE134431"
                               else "external transportability test (different anatomical site)"),
                         independence_structure="one sample per patient (distinct patient IDs)",
                         source_publication=ref))
B = pd.DataFrame(brow)
B.to_csv(f"{OUT}/R2_04_TableS1_bulk_samples.csv", index=False)

# ------------------------------------------------- reconciliation (comment 2)
tis = ph.loc[L.index, "tissue"]
def s(col, mask): return int(L.loc[mask, col].sum())
allm = np.ones(len(L), bool); skinm = (tis != "PBMC").values; pbmc = (tis == "PBMC").values
footm = (tis == "Foot").values; armm = (tis == "Forearm").values
dfu_skin = skinm & ph.loc[L.index, "group"].isin(["Healing", "Nonhealing"]).values
rec = [
 ("Cells reported by the source study, all 54 libraries", 174962, "54 libraries (44 skin + 10 PBMC)", "Theocharidis et al. 2022: 94,325 foot + 37,182 forearm + 43,455 PBMC"),
 ("Cells reported by the source study, 44 skin libraries", 131507, "44 skin libraries", "94,325 foot + 37,182 forearm"),
 ("Cells in the GEO count matrices, all 54 libraries", s("n_cells_geo", allm), "54", "sum of matrix columns (R2_01)"),
 ("Cells in the GEO count matrices, 44 skin libraries", s("n_cells_geo", skinm), "44", "foot %d + forearm %d" % (s("n_cells_geo", footm), s("n_cells_geo", armm))),
 ("Cells in the GEO count matrices, 10 PBMC libraries", s("n_cells_geo", pbmc), "10", ""),
 ("Cells passing QC (200-5,000 genes, <10% mitochondrial), all 54 libraries", s("n_cells_qc", allm), "54", "includes PBMC; not analysed"),
 ("Cells passing QC, 44 skin libraries = integrated atlas", s("n_cells_qc", skinm), "44", "all retained; no sub-sampling"),
 ("Cells passing QC, 10 PBMC libraries (excluded)", s("n_cells_qc", pbmc), "10", ""),
 ("Myeloid cells in the skin atlas", int(L.n_myeloid.sum()), "44", ""),
 ("Macrophage/monocyte cells (resident, inflammatory, lipid-associated)", int(L.n_macrophage_monocyte.sum()), "44", ""),
 ("Macrophage/monocyte cells in healing + non-healing skin libraries", int(L.loc[dfu_skin, "n_macrophage_monocyte"].sum()), "19", "12 healing + 7 non-healing libraries"),
 ("  of which foot libraries", int(L.loc[dfu_skin & footm, "n_macrophage_monocyte"].sum()), "14", "9 healing + 5 non-healing"),
 ("  of which forearm libraries", int(L.loc[dfu_skin & armm, "n_macrophage_monocyte"].sum()), "5", "3 healing + 2 non-healing"),
]
R = pd.DataFrame(rec, columns=["quantity", "cells", "libraries", "note"])
R.to_csv(f"{OUT}/R2_04_TableS2_cell_reconciliation.csv", index=False)

summ = dict(
    n_libraries=len(L), cells_geo_all=s("n_cells_geo", allm), cells_geo_skin=s("n_cells_geo", skinm),
    cells_geo_foot=s("n_cells_geo", footm), cells_geo_forearm=s("n_cells_geo", armm), cells_geo_pbmc=s("n_cells_geo", pbmc),
    cells_qc_all=s("n_cells_qc", allm), cells_qc_skin=s("n_cells_qc", skinm), cells_qc_pbmc=s("n_cells_qc", pbmc),
    audit_all_raw_match=bool(L.audit_raw_matches_khan2.all()), audit_all_qc_match=bool(L.audit_qc_matches_khan2.all()),
    myeloid=int(L.n_myeloid.sum()), macrophage=int(L.n_macrophage_monocyte.sum()),
    macrophage_dfu_skin=int(L.loc[dfu_skin, "n_macrophage_monocyte"].sum()),
    donor_threshold=thr, donor_groups=donor, forearm_match=forearm_match,
    n_healing_foot_donors=len([r for r in roots if ph.loc[r, "group"] == "Healing"]),
    n_nonhealing_foot_donors=len([r for r in roots if ph.loc[r, "group"] == "Nonhealing"]),
    mixed_sex_libraries=L.index[L.mixed_sex_signal].tolist(),
    sex_by_group_contrast=pd.crosstab(T.loc[T.in_primary_healing_contrast, "analysis_group"], T.loc[T.in_primary_healing_contrast, "sex_inferred"]).to_dict())
json.dump(summ, open(f"{OUT}/R2_04_summary.json", "w"), indent=2, default=str)
print(json.dumps(summ, indent=1, default=str))
print(R.to_string(index=False))
print(pairs.head(12).to_string(index=False))
