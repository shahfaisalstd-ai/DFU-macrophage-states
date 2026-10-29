"""
R2_14 -- cross-check the source data of the revised figures against the raw
data deposited with the previous revision (original pipeline 04_raw_data_by_figure) and
against the original result tables (original pipeline 02_results).  For each pair of files
the shared numeric columns are aligned on their key and the maximum absolute
difference is reported.  Output: R2_14_figure_data_crosscheck.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from r2_common import KHAN, RES, OUT
O = f"{KHAN}/04_raw_data_by_figure"; N = os.path.join(os.environ.get("R2FIG", f"{OUT}/figures"), "Figure_source_data")
rd = lambda p: pd.read_csv(p).drop(columns=[c for c in pd.read_csv(p, nrows=0).columns if c.startswith("Unnamed")], errors="ignore")

def compare(old, new, key, cols=None, note=""):
    A, B = rd(old), rd(new)
    if key: A, B = A.set_index(key), B.set_index(key)
    norm = lambda ix: ix.map(lambda v: v.replace("\u2013", "-") if isinstance(v, str) else v) if not isinstance(ix, pd.MultiIndex) else ix
    A.index, B.index = norm(A.index), norm(B.index)
    B = B.loc[[k for k in A.index if k in B.index]]
    A = A.loc[B.index]
    cols = cols or [c for c in A.columns if c in B.columns and A[c].dtype.kind in "fi" and B[c].dtype.kind in "fi"]
    diff = max(float(np.nanmax(np.abs(A[c].values.astype(float) - B[c].values.astype(float)))) for c in cols) if cols else np.nan
    return dict(previous_file=os.path.relpath(old, KHAN), revised_figure_source=os.path.basename(new), key=str(key), rows_compared=len(A),
                columns_compared=", ".join(map(str, cols)), max_abs_difference=diff, identical=bool(diff < 1e-9) if cols else False, note=note)

R = []
R.append(compare(f"{O}/FigureR1b_QC_threshold_sensitivity.csv", f"{N}/Figure2B_QC_threshold_grid.csv", "rule", ["n_pass"]))
R.append(compare(f"{O}/FigureR1c_myeloid_recovery.csv", f"{N}/Figure2C_myeloid_recovery.csv", "analysis"))
R.append(compare(f"{O}/FigureR3b_myeloid_composition.csv", f"{N}/Figure4B_myeloid_composition_pooled.csv", "group"))
R.append(compare(f"{O}/FigureR4b_M1_M2_scores.csv", f"{N}/Figure5B_M1_M2_per_cell.csv", "cell"))
R.append(compare(f"{O}/FigureR5a_patient_vs_cell_level_P.csv", f"{N}/Figure6A_cell_vs_library_P.csv", "programme"))
R.append(compare(f"{O}/FigureR6b_lambda_benchmark.csv", f"{N}/Figure10B_lambda.csv", ["benchmark", "lambda"]))
R.append(compare(f"{O}/FigureR6c_reference_size_benchmark.csv", f"{N}/Figure10C_reference_size.csv", "reference"))
R.append(compare(f"{O}/FigureR7c_variance_site_vs_healing.csv", f"{N}/Figure14C_site_vs_healing_variance.csv", "Symbol"))
R.append(compare(f"{O}/FigureR8d_learning_curve.csv", f"{N}/Figure13D_learning_curve.csv", "n_train_patients"))
R.append(compare(f"{O}/FigureR11d_representative_genes.csv", f"{N}/Figure9D_representative_genes.csv", "gene", ["mean_nonhealing", "mean_healing", "log2FC", "p"]))
# programme heatmap: previous file is unscaled means; revised panel is the same means z-scored across populations
a = rd(f"{O}/FigureR4a_programme_by_subtype.csv").set_index("myeloid_subtype"); a = (a - a.mean()) / a.std(ddof=0)
b = pd.read_csv(f"{N}/Figure5A_programme_heatmap.csv", index_col=0)
b = b.rename(index={"Resident (FOLR2+ LYVE1+ C1Q+) state": "Resident M2-like macrophage", "Inflammatory monocyte-derived state": "Inflammatory monocyte-macrophage",
                    "Lipid-associated (TREM2+ APOE+) state": "Lipid-associated macrophage", "Langerhans cell": "LC/Langerhans"})
b.columns = a.columns[:b.shape[1]] if b.shape[1] == a.shape[1] else b.columns
common = [i for i in a.index if i in b.index]
d = float(np.nanmax(np.abs(a.loc[common].values - b.loc[common].values))) if common else np.nan
R.append(dict(previous_file="04_raw_data_by_figure/FigureR4a_programme_by_subtype.csv", revised_figure_source="Figure5A_programme_heatmap.csv", key="myeloid population",
              rows_compared=len(common), columns_compared="13 programmes (previous means z-scored across populations)", max_abs_difference=d, identical=bool(d < 1e-6),
              note="revised panel shows z-scores of the identical means; populations renamed as transcriptional states"))
# library programme scores: revised panel summarises the same per-library values (group means)
a = rd(f"{O}/FigureR4c_patient_level_programme_scores.csv"); b = rd(f"{N}/Figure5C_library_programme_scores.csv")
lut = {"Resident/tissue": "ms_Resident_tissue", "Lipid-associated": "ms_Lipid_associated_LAM", "Inflammatory monocyte": "ms_Inflammatory_mono",
       "Classical activation (M1 gene set)": "ms_Classical_M1", "Alternative activation (M2 gene set)": "ms_Alternative_M2"}
dd = []
for _, rr in b.iterrows():
    col = lut.get(rr.programme, "ms_" + str(rr.programme).replace(" ", "_"))
    if col in a.columns: dd.append(abs(a.loc[a.group == rr.group, col].mean() - rr["mean"]))
R.append(dict(previous_file="04_raw_data_by_figure/FigureR4c_patient_level_programme_scores.csv", revised_figure_source="Figure5C_library_programme_scores.csv", key="programme x group",
              rows_compared=len(dd), columns_compared="group mean of library scores", max_abs_difference=float(max(dd)) if dd else np.nan,
              identical=bool(dd and max(dd) < 1e-9), note="revised panel adds bootstrap CIs to the same library values"))
# trajectory patterns: same 300 genes and pattern assignment underlie the revised profiles
a = rd(f"{O}/FigureR9e_trajectory_gene_patterns.csv"); b = rd(f"{RES}/12_trajectory_gene_patterns.csv")
k = [c for c in b.columns if c in ("gene",)][0]; m = a.merge(b, on="gene", suffixes=("_prev", "_res"))
pc = [c for c in m.columns if c.startswith("pattern")]
same = int((m[pc[0]] == m[pc[1]]).sum()) if len(pc) == 2 else len(m)
R.append(dict(previous_file="04_raw_data_by_figure/FigureR9e_trajectory_gene_patterns.csv", revised_figure_source="Figure7D/7E (from 02_results/12_trajectory_gene_patterns.csv)",
              key="gene", rows_compared=len(m), columns_compared="pattern assignment", max_abs_difference=float(len(m) - same), identical=bool(same == len(m) == len(a)),
              note="number of genes whose pattern differs"))
# macrophage abundance: revised 10D plots the marker score from 05_deconvolution_combined.csv
a = rd(f"{O}/FigureR6d_macrophage_abundance.csv").set_index("index"); b = pd.read_csv(f"{N}/Figure10D_marker_score.csv", index_col=0)
common = [i for i in a.index if i in b.index]
d = float(np.nanmax(np.abs(a.loc[common, "macrophage_marker_score"] - b.loc[common, "macrophage_marker_score"]))) if "macrophage_marker_score" in a.columns else np.nan
R.append(dict(previous_file="04_raw_data_by_figure/FigureR6d_macrophage_abundance.csv", revised_figure_source="Figure10D_marker_score.csv", key="GSM", rows_compared=len(common),
              columns_compared="macrophage_marker_score", max_abs_difference=d, identical=bool(d < 1e-9), note=""))

# ---- remaining raw-data files of the previous revision -------------------------------------------
def row(prev, new, key, n, d, ident, note=""):
    R.append(dict(previous_file=prev, revised_figure_source=new, key=key, rows_compared=n, columns_compared="", max_abs_difference=d, identical=ident, note=note))
# per-library cell numbers (Figure 2A) against the previous per-sample QC table
a = rd(f"{O}/FigureR1a_persample_QC.csv").set_index("gid"); b = rd(f"{N}/Figure2A_cells_per_library.csv").set_index("library_id")
c = [g for g in a.index if g in b.index]
d = float(max((a.loc[c, "n_cells_raw"] - b.loc[c, "cells_in_GEO_matrix"]).abs().max(), (a.loc[c, "n_cells_pass"] - b.loc[c, "cells_passing_QC"]).abs().max()))
row("04_raw_data_by_figure/FigureR1a_persample_QC.csv", "Figure2A_cells_per_library.csv", "library", len(c), d, d == 0, "cells in GEO matrix and after QC (recounted from GEO in R2_01/R2_04)")
# atlas cell-type counts and composition (Figure 3) against the original result table
k = rd(f"{RES}/03_celltype_by_group_counts.csv").set_index("group")
a = rd(f"{O}/FigureR2a_celltype_counts.csv").set_index("cell_type").n_cells
d = float((a - k.sum(axis=0).reindex(a.index)).abs().max())
row("04_raw_data_by_figure/FigureR2a_celltype_counts.csv", "Figure 3A (02_results/03_celltype_by_group_counts.csv)", "cell type", len(a), d, d == 0,
    f"total {int(a.sum())} cells = atlas size")
a = rd(f"{O}/FigureR2c_composition_by_group.csv").set_index("group"); pc = k.div(k.sum(axis=1), axis=0) * 100
d = float((a - pc.loc[a.index, a.columns]).abs().max().max())
row("04_raw_data_by_figure/FigureR2c_composition_by_group.csv", "Figure 3C (02_results/03_celltype_by_group_counts.csv)", "group x cell type", a.size, d, d < 1e-9)
# marker means (Figure 4C)
a = rd(f"{O}/FigureR3c_marker_mean_expression.csv").set_index("state"); b = rd(f"{N}/Figure4C_marker_dotplot.csv")
diffs = [abs(a.loc[r.state, r.gene] - r["mean"]) for _, r in b.iterrows() if r.state in a.index and r.gene in a.columns]
row("04_raw_data_by_figure/FigureR3c_marker_mean_expression.csv", "Figure4C_marker_dotplot.csv", "population x gene", len(diffs), float(max(diffs)), max(diffs) < 1e-5,
    "mean log-expression; revised panel adds detection fraction")
# directional concordance (Figure 14A)
a = rd(f"{O}/FigureR7a_concordance_by_evidence.csv"); b = rd(f"{N}/Figure14A_replication_CIs.csv")
a["k"] = a.set.str.lower(); b["k"] = b.gene_set.str.lower()
m = a.merge(b, on="k")
cols = [("n_genes_x", "n_genes_y"), ("directional_concordance_pct", "same_direction_pct"), ("concordance_ci_low_pct", "same_direction_ci_low"),
        ("concordance_ci_high_pct", "same_direction_ci_high"), ("statistically_replicated_n", "replicated_same_direction_and_P05_n")]
d = float(max((m[x] - m[y]).abs().max() for x, y in cols))
row("04_raw_data_by_figure/FigureR7a_concordance_by_evidence.csv", "Figure14A_replication_CIs.csv", "gene set", len(m), d, d == 0, "revised panel adds CIs for the replication rate")
# classifier AUCs (Figure 13A): same estimates except the pseudobulk classifier, rebuilt from raw counts
a = rd(f"{O}/FigureR8a_auc_by_evaluation.csv"); a["evaluation"] = a.evaluation.str.replace("\n", " ", regex=False)
au = rd(f"{OUT}/R2_06_classifier_AUC_CIs.csv").set_index("evaluation").auc; s7 = __import__("json").load(open(f"{RES}/07b_summary.json"))
pairs = {"cell naive": s7["naive_ungrouped_auc"], "cell grouped": au["per-cell, 25 repeated library-grouped splits (previous revision)"],
         "cell LOPO": au["per-cell, leave-one-library-out"], "patient aggregated": au["library outcome, leave-one-library-out"],
         "patient pseudobulk": au["library-level pseudobulk classifier (raw-count pseudobulk), leave-one-out"]}
for _, r_ in a.iterrows():
    v = pairs[r_.evaluation]; dd = abs(r_.auc - v)
    row("04_raw_data_by_figure/FigureR8a_auc_by_evaluation.csv", f"Figure13A ({r_.evaluation})", "evaluation", 1, dd, dd < 1e-3,
        "differs by design: previous pseudobulk summed log-normalised values; revised one sums raw counts (0.536 -> 0.524)" if dd >= 1e-3 else ("equal within the 3-decimal rounding of R2_06_classifier_AUC_CIs.csv" if dd > 1e-9 else ""))

T = pd.DataFrame(R); T.to_csv(f"{OUT}/R2_14_figure_data_crosscheck.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 45)
print(T[["previous_file", "revised_figure_source", "rows_compared", "max_abs_difference", "identical", "note"]].to_string(index=False))
