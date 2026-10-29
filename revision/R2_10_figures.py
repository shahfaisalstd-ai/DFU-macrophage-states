"""
R2_10 -- Revised figures (Editor comments 7, 8, 9, 12).
* New Figure 1: graphical workflow (discovery / internal sensitivity /
  external transportability / hypothesis-generating in-silico).
* Figures 2 and 4-17 redrawn from the stored results with neutral, descriptive
  panel titles; macrophage populations named as transcriptional states;
  'library' rather than 'patient' as the unit; 95% CIs wherever an effect, AUC,
  proportion or replication rate is shown. Figure 3 (atlas embedding) is the
  previous image, whose panel titles were already descriptive.
Every panel's numbers are written to Figure_source_data/ (one CSV per panel).
Output: 600-dpi PNG + LZW TIFF + vector PDF, 183 mm wide.
"""
import sys, os, json, textwrap; sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy import stats
from r2_common import *
import h5ad_lite as H

FIG = os.environ.get("R2FIG", f"{OUT}/figures"); SRC = f"{FIG}/Figure_source_data"
os.makedirs(FIG, exist_ok=True); os.makedirs(SRC, exist_ok=True)
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "pdf.fonttype": 42, "axes.titlelocation": "left", "axes.titleweight": "normal"})
W = 183 / 25.4
CH, CNH = "#0072B2", "#D55E00"            # healing / non-healing (Okabe-Ito)
GREY, LGREY = "#7a7a7a", "#c8c8c8"
STATE_NAME = {"Resident M2-like macrophage": "Resident (FOLR2+ LYVE1+ C1Q+) state",
              "Inflammatory monocyte-macrophage": "Inflammatory monocyte-derived state",
              "Lipid-associated macrophage": "Lipid-associated (TREM2+ APOE+) state",
              "cDC1": "cDC1", "cDC2": "cDC2", "LC/Langerhans": "Langerhans cell", "pDC": "pDC",
              "Mature/migratory DC": "Mature/migratory DC", "Proliferating myeloid": "Proliferating myeloid"}
SHORT = {"Resident M2-like macrophage": "Resident", "Inflammatory monocyte-macrophage": "Inflammatory",
         "Lipid-associated macrophage": "Lipid-associated"}
STATE_COL = {"Resident M2-like macrophage": "#009E73", "Inflammatory monocyte-macrophage": "#E69F00",
             "Lipid-associated macrophage": "#56B4E9", "cDC2": "#CC79A7", "Mature/migratory DC": "#999999",
             "cDC1": "#0072B2", "LC/Langerhans": "#F0E442", "pDC": "#D55E00", "Proliferating myeloid": "#000000"}
PROG = {"Resident_tissue": "Resident/tissue", "Lipid_associated_LAM": "Lipid-associated", "Inflammatory_mono": "Inflammatory monocyte",
        "Classical_M1": "Classical activation (M1 gene set)", "Alternative_M2": "Alternative activation (M2 gene set)",
        "Angiogenic": "Angiogenic", "Efferocytotic": "Efferocytotic", "Antigen_presenting": "Antigen presentation",
        "Complement": "Complement", "Interferon_stim": "Interferon-stimulated", "Matrix_remodelling": "Matrix remodeling",
        "Proliferating": "Proliferating", "Phagocytic_lysosomal": "Phagocytic/lysosomal"}
def lab(ax, s, *a, **k):
    ax.set_title(r"$\bf{" + s + r"}$   " + ax.get_title(loc="left"), loc="left")
def save(fig, name):
    for ext, kw in [("png", dict(dpi=600)), ("pdf", {}), ("tif", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"}))]:
        fig.savefig(f"{FIG}/{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig); print("saved", name, flush=True)
def src(df, name): neutral_labels(df).to_csv(f"{SRC}/{name}.csv", index=True)
def ci_point(ax, x, m, lo, hi, color, w=0.18):
    ax.plot([x - w, x + w], [m, m], color="black", lw=1.2, zorder=4)
    ax.plot([x, x], [lo, hi], color="black", lw=0.9, zorder=4)
def strip(ax, x, v, color, seed=0):
    r = np.random.default_rng(seed); ax.scatter(x + r.uniform(-0.12, 0.12, len(v)), v, s=14, color=color, ec="white", lw=0.4, zorder=3)
def legend_groups(ax, loc="below", off=-0.3):
    h = [Line2D([], [], marker="o", ls="", color=CH, label="Healing", ms=4), Line2D([], [], marker="o", ls="", color=CNH, label="Non-healing", ms=4)]
    if loc == "below": ax.legend(handles=h, frameon=False, loc="upper center", bbox_to_anchor=(0.5, off), ncol=2)
    else: ax.legend(handles=h, frameon=False, loc=loc)
SHORTN = {"Resident M2-like macrophage": "Resident state", "Inflammatory monocyte-macrophage": "Inflammatory state",
          "Lipid-associated macrophage": "Lipid-associated state"}
def sn(k): return SHORTN.get(k, STATE_NAME.get(k, k))

T1 = pd.read_csv(f"{OUT}/R2_04_TableS1_GSE165816_libraries.csv").set_index("library_id")
cov = pd.read_csv(f"{OUT}/R2_04_library_covariates.csv", index_col=0)
EFF = pd.read_csv(f"{OUT}/R2_09_effect_estimates_CIs.csv")
def eff(analysis_prefix, endpoint):
    r = EFF[EFF.analysis.str.startswith(analysis_prefix) & (EFF.endpoint == endpoint)].iloc[0]; return r

# =========================================================== Figure 1 workflow
def figure1():
    fig, ax = plt.subplots(figsize=(W, 5.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    tiers = [
     ("Discovery", "#0072B2", 76, 98,
      ["GSE165816 scRNA-seq: 54 libraries from 27 people; 44 skin libraries, 131,507 cells",
       "Uniform QC (200-5,000 genes, <10% mito): 97,677 skin cells, no sub-sampling",
       "Harmony integration, Leiden clustering, post hoc annotation; myeloid re-clustering (7,756 cells)",
       "Primary contrast: macrophage-state composition and program scores, 12 healing vs 7 non-healing libraries",
       "Library = unit of analysis; raw-count pseudobulk; model ~ sex + tissue site + group (Figs 2-6)"]),
     ("Internal sensitivity analyses", "#009E73", 52, 74,
      ["Cell-level vs library-level testing; QC-threshold grid; trajectory rooted at either end",
       "Covariate models (sex, site, depth, %mito); foot-only (14 libraries); putative-donor aggregation (12 donors)",
       "Exclusion of libraries with mixed XIST/Y signal; removal of opposite-sex cells",
       "Classifier: library-grouped, leave-one-library-out and donor-grouped CV; permutation nulls; bootstrap CIs",
       "Deconvolution benchmark on 200 synthetic mixtures; power calculation (Figs 6, 10, 13, 15)"]),
     ("External transportability testing", "#E69F00", 30, 50,
      ["GSE134431 foot ulcers (13; bulk): corroboration only - used in panel construction, not independent",
       "GSE143735 forearm skin (9; bulk): untouched external test of direction, effect size and classifier transfer",
       "Directional concordance vs statistical replication (same sign and P < 0.05), with exact 95% CIs (Figs 9, 12, 14)"]),
     ("Hypothesis-generating in-silico analyses", "#CC79A7", 4, 28,
      ["Ligand-receptor communication; trajectory gene waves; co-expression networks (Figs 7, 8, 11)",
       "129-gene candidate panel = genes recurring in >= 3 of 6 discovery evidence sets (Supplementary Table 5)",
       "In-silico perturbation of the fitted classifier: associative model responses, no causal meaning (Figs 16, 17)",
       "Not inferential evidence; candidates for foot-matched multiplex IF, spatial, flow and functional assays"])]
    for name, col, y0, y1, items in tiers:
        ax.add_patch(FancyBboxPatch((1, y0), 98, y1 - y0, boxstyle="round,pad=0.4,rounding_size=1.5", fc=col, alpha=0.10, ec=col, lw=1.0))
        ax.text(2.5, y1 - 2.4, name, fontsize=8.5, fontweight="bold", color="black", va="top")
        for i, t in enumerate(items):
            ax.text(4, y1 - 6.4 - i * 3.35, "• " + t, fontsize=6.3, va="top")
    for ya, yb in [(76, 74.2), (52, 50.2), (30, 28.2)]:
        ax.annotate("", xy=(50, yb - 0.6), xytext=(50, ya + 0.4), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0))
    ax.text(99, 1.0, "Interpretation: exploratory; directionally consistent findings are not treated as established patient-level differences",
            fontsize=6, ha="right", style="italic", color="#333333")
    save(fig, "Figure1_workflow")
    src(pd.DataFrame([(t[0], i) for t in tiers for i in t[4]], columns=["tier", "item"]), "Figure1_workflow_items")

# =========================================================== Figure 2 QC
def figure2():
    t = T1[T1.included_in_skin_atlas].copy(); t["group"] = t.analysis_group
    order = ["Diabetic", "Healing", "Healthy", "Nonhealing"]; gcol = {"Healthy": "#009E73", "Diabetic": "#E69F00", "Healing": CH, "Nonhealing": CNH}
    t = t.sort_values(["group", "tissue_site"], key=lambda s: s.map({g: i for i, g in enumerate(order)}) if s.name == "group" else s)
    fig = plt.figure(figsize=(W, 4.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.75, wspace=0.35)
    ax = fig.add_subplot(gs[0, :]); x = np.arange(len(t))
    ax.bar(x - 0.2, t.cells_in_GEO_matrix, 0.4, color=LGREY, ec="none", label="cells in GEO matrix")
    ax.bar(x + 0.2, t.cells_passing_QC, 0.4, color=[gcol[g] for g in t.group], ec="none")
    ax.set_xticks(x); ax.set_xticklabels([f"{g}{'*' if s=='Forearm' else ''}" for g, s in zip(t.index, t.tissue_site)], rotation=90, fontsize=5)
    ax.set_ylabel("cells per library")
    ax.set_title(f"Cells per skin library before and after QC (44 libraries: {int(t.cells_in_GEO_matrix.sum()):,} cells in GEO matrices, {int(t.cells_passing_QC.sum()):,} after QC; * forearm)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=LGREY, label="before QC")] +
              [plt.Rectangle((0, 0), 1, 1, color=gcol[g], label=f"after QC: {'Non-healing' if g=='Nonhealing' else g}") for g in order],
              ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.28)); lab(ax, "A", -0.06, 1.12)
    src(t[["analysis_group", "tissue_site", "cells_in_GEO_matrix", "cells_passing_QC"]], "Figure2A_cells_per_library")
    sw = pd.read_csv(f"{RES}/01_QC_threshold_sweep.csv"); sw = sw[sw.tissue != "PBMC"]
    s = sw.groupby(["min_genes", "max_genes"]).n_pass.sum().reset_index(); ref = s[(s.min_genes == 200) & (s.max_genes == 5000)].n_pass.iloc[0]
    s["pct_of_applied"] = 100 * s.n_pass / ref; s["rule"] = s.min_genes.astype(str) + "-" + s.max_genes.astype(str)
    ax = fig.add_subplot(gs[1, 0]); y = np.arange(len(s))
    ax.barh(y, s.n_pass, color=[CNH if r == "200-5000" else LGREY for r in s.rule], height=0.6)
    for i, (n, p) in enumerate(zip(s.n_pass, s.pct_of_applied)): ax.text(n + 1500, i, f"{n:,} ({p:.0f}%)", va="center", fontsize=5.5)
    ax.set_yticks(y); ax.set_yticklabels(s.rule); ax.set_ylabel("genes per cell (min-max)"); ax.set_xlabel("skin cells passing QC")
    ax.set_xlim(0, s.n_pass.max() * 1.35); ax.set_title("Cells passing QC under five gene-count rules\n(applied rule in orange; mitochondrial cut-off 10%)"); lab(ax, "B")
    src(s, "Figure2B_QC_threshold_grid")
    r = pd.read_csv(f"{KHAN}/04_raw_data_by_figure/FigureR1c_myeloid_recovery.csv")
    ax = fig.add_subplot(gs[1, 1]); ax.bar([0, 1], r.myeloid_cells, color=[LGREY, CH], width=0.55)
    for i, (m, a) in enumerate(zip(r.myeloid_cells, r.atlas_cells)): ax.text(i, m + 150, f"{m:,} of {a:,}\n({100*m/a:.1f}%)", ha="center", fontsize=5.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["sub-sample of 300\ncells per library", "all quality-\ncontrolled cells"])
    ax.set_ylabel("myeloid cells"); ax.set_ylim(0, r.myeloid_cells.max() * 1.3); ax.set_title("Myeloid cells available for sub-clustering"); lab(ax, "C")
    src(r, "Figure2C_myeloid_recovery"); save(fig, "Figure2_QC_and_cell_recovery")

# =========================================================== Figure 4 myeloid
mye = pd.read_csv(f"{AUX}/myeloid_obs_with_umap.csv", index_col=0)
def figure4():
    fig = plt.figure(figsize=(W, 2.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.75, 1.55], wspace=0.6)
    ax = fig.add_subplot(gs[0]); r = np.random.default_rng(1).permutation(len(mye))
    m = mye.iloc[r]; ax.scatter(m.umap1, m.umap2, s=0.4, c=[STATE_COL[s] for s in m.myeloid_subtype], rasterized=True, lw=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    cnt = mye.myeloid_subtype.value_counts()
    ax.set_title(f"{len(mye):,} myeloid cells, {mye.mye_leiden.nunique()} subclusters,\n9 annotated populations")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=STATE_COL[k], ms=3, label=f"{STATE_NAME[k]} ({cnt[k]:,})") for k in cnt.index],
              frameon=False, loc="upper left", bbox_to_anchor=(-0.05, -0.08), ncol=1, fontsize=5.3); lab(ax, "A", -0.1)
    src(mye[["gid", "group", "myeloid_subtype", "umap1", "umap2"]], "Figure4A_myeloid_embedding")
    comp = pd.crosstab(mye.group, mye.myeloid_subtype, normalize="index") * 100
    comp = comp.loc[["Healthy", "Diabetic", "Healing", "Nonhealing"], cnt.index]
    ax = fig.add_subplot(gs[1]); left = np.zeros(4)
    for k in comp.columns:
        ax.barh(range(4), comp[k], left=left, color=STATE_COL[k], height=0.65, ec="white", lw=0.4); left += comp[k].values
    ax.set_yticks(range(4)); ax.set_yticklabels(["Healthy", "Diabetic", "Healing", "Non-healing"]); ax.set_xlabel("% of myeloid cells (pooled)")
    ax.set_title("Composition by group\n(pooled cells; descriptive)"); lab(ax, "B", -0.35)
    src(comp, "Figure4B_myeloid_composition_pooled")
    # dot plot from the stored log-normalised expression
    f = H.open_h5ad(f"{H5AD_DIR}/03_myeloid.h5ad"); X = H.read_csr(f["raw"]["X"], shape=(len(mye), 25683))
    genes = pd.read_csv(f"{AUX}/atlas_genes.csv").gene.values; gi = {g: i for i, g in enumerate(genes)}
    MK = ["FOLR2", "LYVE1", "C1QA", "CD163", "SELENOP", "S100A8", "S100A9", "FCN1", "VCAN", "TREM2", "APOE", "GPNMB", "CD9",
          "CLEC9A", "CD1C", "FCER1A", "CD207", "LILRA4", "LAMP3", "MKI67"]
    sub = X[:, [gi[g] for g in MK]].toarray(); rows = []
    for k in cnt.index:
        s_ = sub[(mye.myeloid_subtype == k).values]; rows += [(k, g, s_[:, j].mean(), (s_[:, j] > 0).mean()) for j, g in enumerate(MK)]
    D = pd.DataFrame(rows, columns=["state", "gene", "mean", "frac"])
    D["z"] = D.groupby("gene")["mean"].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    ax = fig.add_subplot(gs[2]); yo = {k: i for i, k in enumerate(cnt.index[::-1])}; xo = {g: i for i, g in enumerate(MK)}
    sc_ = ax.scatter(D.gene.map(xo), D.state.map(yo), s=D.frac * 40, c=D.z, cmap="RdBu_r", vmin=-2, vmax=2, ec="#555555", lw=0.2)
    ax.set_xticks(range(len(MK))); ax.set_xticklabels(MK, rotation=90, fontsize=5.5); ax.set_yticks(range(len(yo)))
    ax.set_yticklabels([sn(k) for k in cnt.index[::-1]], fontsize=5.5); ax.set_title("Canonical myeloid markers by population")
    cb = plt.colorbar(sc_, ax=ax, fraction=0.03, pad=0.02); cb.set_label("scaled mean expression", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    for fr in (0.25, 0.5, 1.0): ax.scatter([], [], s=fr * 40, c="grey", label=f"{int(fr*100)}%")
    ax.legend(title="cells expressing", frameon=False, loc="upper left", bbox_to_anchor=(0, -0.42), ncol=3, fontsize=5, title_fontsize=5.5); lab(ax, "C", -0.55)
    src(D, "Figure4C_marker_dotplot"); save(fig, "Figure4_myeloid_compartment")

# =========================================================== Figure 5 programmes
def figure5():
    A = pd.read_csv(f"{KHAN}/04_raw_data_by_figure/FigureR4a_programme_by_subtype.csv", index_col=0)
    A.columns = [PROG[c.replace("ms_", "")] for c in A.columns]
    Z = (A - A.mean()) / A.std(ddof=0); Z.index = [STATE_NAME[i] for i in Z.index]
    fig = plt.figure(figsize=(W, 2.9)); gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.95, 1.25], wspace=0.75)
    ax = fig.add_subplot(gs[0]); im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(Z.shape[1])); ax.set_xticklabels(Z.columns, rotation=90, fontsize=5.3); ax.set_yticks(range(len(Z))); ax.set_yticklabels([sn(k) for k in A.index], fontsize=5.3)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("z-scored program score", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title("Mean score of 13 gene-set programs\nby myeloid population"); lab(ax, "A", -0.9); src(Z, "Figure5A_programme_heatmap")
    B = pd.read_csv(f"{KHAN}/04_raw_data_by_figure/FigureR4b_M1_M2_scores.csv")
    ax = fig.add_subplot(gs[1]); r = np.random.default_rng(2).permutation(len(B)); B2 = B.iloc[r]
    ax.scatter(B2.ms_Classical_M1, B2.ms_Alternative_M2, s=1.2, c=[CH if g == "Healing" else CNH for g in B2.group], alpha=0.5, lw=0, rasterized=True)
    ax.set_xlabel("M1 gene-set score"); ax.set_ylabel("M2 gene-set score")
    ax.set_title(f"Per-cell M1 and M2 gene-set\nscores ({len(B):,} cells)"); legend_groups(ax, off=-0.22); lab(ax, "B", -0.3); src(B, "Figure5B_M1_M2_per_cell")
    ax = fig.add_subplot(gs[2]); L = pd.read_csv(f"{OUT}/R2_05_library_level_data.csv", index_col=0)
    progs = ["Resident_tissue", "Inflammatory_mono", "Complement", "Lipid_associated_LAM"]; rows = []
    for i, p in enumerate(progs):
        for j, (g, c) in enumerate([("Healing", CH), ("Nonhealing", CNH)]):
            v = L.loc[L.group == g, f"ms_{p}"]; x = i + (j - 0.5) * 0.36; strip(ax, x, v, c, i * 2 + j)
            e = eff("functional programme", p)
            m, lo, hi = (e.mean_healing, e.ci_healing_low, e.ci_healing_high) if g == "Healing" else (e.mean_nonhealing, e.ci_nonhealing_low, e.ci_nonhealing_high)
            ci_point(ax, x, m, lo, hi, c, 0.1); rows.append((p, g, m, lo, hi))
    M2 = pd.read_csv(f"{OUT}/R2_05_programme_and_pseudotime_models.csv"); M2 = M2[M2.model == "M2"].set_index("endpoint")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{ {'Resident_tissue':'Resident','Inflammatory_mono':'Inflamm.','Complement':'Complement','Lipid_associated_LAM':'Lipid-assoc.'}[p]}\nadj. P\n= {M2.loc[p, 'p_model']:.2f}" for p in progs], fontsize=4.8)
    ax.set_ylabel("program score (library mean)"); ax.set_title("Library-level program scores\n(mean and 95% CI)")
    legend_groups(ax, off=-0.3); lab(ax, "C", -0.25)
    src(pd.DataFrame(rows, columns=["programme", "group", "mean", "ci_low", "ci_high"]), "Figure5C_library_programme_scores")
    save(fig, "Figure5_functional_programmes")

# =========================================================== Figure 6 inference + covariates
def figure6():
    fig = plt.figure(figsize=(W, 5.0)); gs = fig.add_gridspec(2, 2, hspace=0.6, wspace=0.42)
    A = pd.read_csv(f"{KHAN}/04_raw_data_by_figure/FigureR5a_patient_vs_cell_level_P.csv")
    ax = fig.add_subplot(gs[0, 0]); ax.scatter(-np.log10(A.p_cell_level), -np.log10(A.p_patient_level), s=18, color=CH, ec="white", lw=0.4)
    for _, r in A.iterrows():
        if -np.log10(r.p_cell_level) > 100 or r.p_patient_level < 0.1: ax.annotate(PROG[r.programme], (-np.log10(r.p_cell_level), -np.log10(r.p_patient_level)), fontsize=5, xytext=(3, {"Angiogenic": -8, "Matrix_remodelling": 6}.get(r.programme, 3)), textcoords="offset points")
    ax.axhline(-np.log10(0.05), ls=":", color=GREY, lw=0.8); ax.set_xlabel("-log10 P, cells as replicates"); ax.set_ylabel("-log10 P, libraries as replicates")
    ax.set_title("Program comparisons tested per cell and per library"); lab(ax, "A"); src(A, "Figure6A_cell_vs_library_P")
    D = pd.read_csv(f"{OUT}/R2_05_pseudobulk_DE_M2.csv", index_col=0).dropna(subset=["pvalue"])
    ax = fig.add_subplot(gs[0, 1]); sig = D.padj < 0.05
    ax.scatter(D.log2FoldChange.clip(-8, 8), -np.log10(D.pvalue), s=2, color=LGREY, rasterized=True, lw=0)
    ax.scatter(D.log2FoldChange[sig].clip(-8, 8), -np.log10(D.pvalue[sig]), s=10, color=CNH, ec="white", lw=0.3)
    if sig.any():
        for k, (g, r) in enumerate(D[sig].sort_values("pvalue").iterrows()): ax.annotate(g, (np.clip(r.log2FoldChange, -8, 8), -np.log10(r.pvalue)), fontsize=5, xytext=(3, [2, -7, 6, -3, 9][k % 5]), textcoords="offset points")
    else:   # no gene at FDR < 0.05: mark and label the five genes with the smallest P values
        top = D.nsmallest(5, "pvalue")
        ax.scatter(top.log2FoldChange.clip(-8, 8), -np.log10(top.pvalue), s=8, color=GREY, ec="white", lw=0.3)
        for k, (g, r) in enumerate(top.iterrows()):
            ax.annotate(g, (np.clip(r.log2FoldChange, -8, 8), -np.log10(r.pvalue)), fontsize=5, textcoords="offset points",
                        xytext=[(4, 1), (4, 0), (4, 1), (-26, -3), (4, -5)][k])
    ax.set_xlabel("log2 fold change (non-healing / healing)"); ax.set_ylabel("-log10 P")
    ax.set_title(f"Macrophage pseudobulk from raw counts, ~ sex + site + group\n(19 libraries; {int(sig.sum())} genes at FDR < 0.05)")
    lab(ax, "B"); src(D[["baseMean", "log2FoldChange", "lfcSE", "ci_low", "ci_high", "pvalue", "padj"]], "Figure6B_pseudobulk_DE_adjusted")
    ax = fig.add_subplot(gs[1, 0]); L = pd.read_csv(f"{OUT}/R2_05_library_level_data.csv", index_col=0); rows = []
    for i, (st, col, nm) in enumerate([("resident", "n_resident", "Resident"), ("inflammatory", "n_inflammatory", "Inflammatory"), ("lipid-associated", "n_lipid", "Lipid-associated")]):
        e = eff("macrophage-state composition (%", st)
        for j, (g, c) in enumerate([("Healing", CH), ("Nonhealing", CNH)]):
            v = 100 * L.loc[L.group == g, col] / L.loc[L.group == g, "n_mac"]; x = i + (j - 0.5) * 0.36; strip(ax, x, v, c, i * 3 + j)
            m, lo, hi = (e.mean_healing, e.ci_healing_low, e.ci_healing_high) if g == "Healing" else (e.mean_nonhealing, e.ci_nonhealing_low, e.ci_nonhealing_high)
            ci_point(ax, x, m, lo, hi, c, 0.1); rows.append((st, g, m, lo, hi))
    C = pd.read_csv(f"{OUT}/R2_05_state_composition_models.csv").set_index(["state", "model"])
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{n}\nadj. OR {C.loc[(s,'M2'),'odds_ratio_NH_vs_H']:.2f}\n({C.loc[(s,'M2'),'ci_low']:.2f}-{C.loc[(s,'M2'),'ci_high']:.2f})"
                                                for s, n in [("resident", "Resident"), ("inflammatory", "Inflammatory"), ("lipid", "Lipid-associated")]], fontsize=5.3)
    ax.set_ylabel("% of the library's macrophages"); ax.set_title("Macrophage-state composition per library\n(mean and 95% CI)")
    legend_groups(ax, off=-0.33); lab(ax, "C"); src(pd.DataFrame(rows, columns=["state", "group", "mean_pct", "ci_low", "ci_high"]), "Figure6C_state_composition")
    ax = fig.add_subplot(gs[1, 1]); R = pd.read_csv(f"{OUT}/R2_05_state_composition_models.csv"); R = R[R.state == "resident"].reset_index(drop=True)
    y = np.arange(len(R))[::-1]
    for yy, (_, r) in zip(y, R.iterrows()):
        ax.plot([r.ci_low, r.ci_high], [yy, yy], color="black", lw=0.9); ax.scatter(r.odds_ratio_NH_vs_H, yy, s=22, color=CNH if r.model == "M2" else GREY, zorder=3)
        ax.text(60, yy, f"P = {r.p_model:.3f} / {r.p_permutation:.3f}", va="center", fontsize=5)
    ax.axvline(1, ls="--", color=GREY, lw=0.7); ax.set_xscale("log"); ax.set_xlim(0.15, 400)
    SL = {"M0": "M0 unadjusted (19)", "M1": "M1 + sex (19)", "M2": "M2 + sex + site, primary (19)", "M3": "M3 + depth, %mito (19)", "S1": "S1 foot only (14)", "S2": "S2 putative donors (12)", "S3": "S3 excl. G2, G13 (17)", "S4": "S4 excl. opposite-sex cells (19)"}
    ax.set_yticks(y); ax.set_yticklabels([SL[m] for m in R.model], fontsize=5.3)
    ax.set_xlabel("odds ratio, resident state, non-healing vs healing (95% CI)")
    ax.set_title("Resident-state proportion under alternative models\n(model-based P / stratified permutation P)"); lab(ax, "D", -0.95)
    src(R, "Figure6D_resident_state_models"); save(fig, "Figure6_library_level_inference")

# =========================================================== Figure 7 trajectory
def figure7():
    T = pd.read_csv(f"{AUX}/trajectory_obs_with_umap.csv", index_col=0); T["pct"] = T.pseudotime.rank(pct=True)
    fig = plt.figure(figsize=(W, 4.6)); gs = fig.add_gridspec(2, 3, hspace=0.7, wspace=0.8, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, 0]); s_ = ax.scatter(T.umap1, T.umap2, c=T.pct, s=0.6, cmap="viridis", rasterized=True, lw=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_title(f"{len(T):,} macrophages/monocytes\n(pseudotime percentile)")
    cb = plt.colorbar(s_, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=5); lab(ax, "A", -0.1)
    src(T[["gid", "group", "myeloid_subtype", "pseudotime", "pct", "umap1", "umap2"]], "Figure7A_trajectory_embedding")
    ax = fig.add_subplot(gs[0, 1]); states = ["Inflammatory monocyte-macrophage", "Lipid-associated macrophage", "Resident M2-like macrophage"]
    for i, s in enumerate(states):
        v = T.loc[T.myeloid_subtype == s, "pct"].values; r = np.random.default_rng(i)
        ax.scatter(v, i + r.uniform(-0.25, 0.25, len(v)), s=0.5, color=STATE_COL[s], alpha=0.5, rasterized=True, lw=0)
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color="black", lw=1.4)
    ax.set_yticks(range(3)); ax.set_yticklabels([SHORT[s] for s in states]); ax.set_xlabel("pseudotime percentile")
    ax.set_title("Pseudotime by macrophage state\n(median marked)"); lab(ax, "B", -0.4)
    ax = fig.add_subplot(gs[0, 2]); L = pd.read_csv(f"{OUT}/R2_05_library_level_data.csv", index_col=0); e = eff("mean pseudotime", "pseudotime")
    for j, (g, c) in enumerate([("Healing", CH), ("Nonhealing", CNH)]):
        strip(ax, j, L.loc[L.group == g, "pseudotime"], c, j)
        m, lo, hi = (e.mean_healing, e.ci_healing_low, e.ci_healing_high) if g == "Healing" else (e.mean_nonhealing, e.ci_nonhealing_low, e.ci_nonhealing_high)
        ci_point(ax, j, m, lo, hi, c)
    pt = json.load(open(f"{RES}/12_pseudotime_group_test.json")); M2 = pd.read_csv(f"{OUT}/R2_05_programme_and_pseudotime_models.csv").set_index(["endpoint", "model"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healing", "Non-healing"]); ax.set_ylabel("mean pseudotime per library")
    ax.set_title(f"Library means (mean and 95% CI)\nlibrary-level P = {pt['p_patient_level']:.2f}; adjusted P = {M2.loc[('pseudotime','M2'),'p_model']:.2f}\n(cells as replicates: P = {pt['p_cell_level']:.0e})", fontsize=6.3)
    lab(ax, "C", -0.35); src(L[["group", "pseudotime"]], "Figure7C_library_pseudotime")
    Hm = pd.read_csv(f"{RES}/12_trajectory_heatmap_matrix.csv", index_col=0); Hm = Hm.iloc[Hm.values.argmax(1).argsort(kind="stable")]
    ax = fig.add_subplot(gs[1, :2]); im = ax.imshow(Hm.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
    ax.set_yticks([]); ax.set_ylabel(f"{len(Hm)} trajectory genes"); ax.set_xlabel("pseudotime bin (20 quantile bins)")
    ax.set_title("300 trajectory-associated genes with the largest amplitude,\nordered by peak bin (cell-level association test)")
    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01); cb.set_label("scaled", fontsize=5.5); cb.ax.tick_params(labelsize=5); lab(ax, "D", -0.05)
    src(Hm, "Figure7D_trajectory_heatmap")
    P = pd.read_csv(f"{RES}/12_trajectory_gene_patterns.csv").set_index("gene"); Hz = pd.read_csv(f"{RES}/12_trajectory_heatmap_matrix.csv", index_col=0)
    ax = fig.add_subplot(gs[1, 2]); cols = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]; rows = []
    for k in sorted(P.pattern.unique()):
        g = P.index[P.pattern == k]; mprof = Hz.loc[g].mean(); ax.plot(range(len(mprof)), mprof, color=cols[k], lw=1.2, label=f"pattern {k+1} ({len(g)})")
        rows.append(mprof.rename(f"pattern{k+1}"))
    ax.set_xlabel("pseudotime bin"); ax.set_ylabel("mean scaled\nexpression"); ax.legend(frameon=False, fontsize=5); ax.set_title("Mean profile of four gene patterns"); lab(ax, "E", -0.3)
    src(pd.concat(rows, axis=1), "Figure7E_pattern_profiles"); save(fig, "Figure7_trajectory")

# =========================================================== Figure 8 communication
def figure8():
    fig = plt.figure(figsize=(W, 6.4)); gs = fig.add_gridspec(2, 2, wspace=0.55, hspace=0.75)
    N = pd.read_csv(f"{RES}/13_interaction_counts.csv", index_col=0)
    ax = fig.add_subplot(gs[0, 0]); im = ax.imshow(N.values, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(N))); ax.set_xticklabels(N.columns, rotation=90, fontsize=4.6); ax.set_yticks(range(len(N))); ax.set_yticklabels(N.index, fontsize=4.6)
    ax.set_xlabel("receiver"); ax.set_ylabel("sender"); ax.set_title("Interactions at FDR < 0.05 (cell-label\npermutation; pooled cells)")
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=5); lab(ax, "A", -0.6); src(N, "Figure8A_interaction_counts")
    S = pd.read_csv(f"{RES}/13_signalling_activity.csv", index_col=0)
    ax = fig.add_subplot(gs[0, 1]); ax.scatter(S.outgoing, S.incoming, s=14, color=CH, ec="white", lw=0.3)
    for k, r in S.nlargest(6, "outgoing").iterrows(): ax.annotate(k, (r.outgoing, r.incoming), fontsize=4.6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("total outgoing score"); ax.set_ylabel("total incoming score"); ax.set_title("Outgoing and incoming\nsignaling per population"); lab(ax, "B", -0.4)
    src(S, "Figure8B_signalling_activity")
    Cm = pd.read_csv(f"{RES}/13_communication_all.csv"); Cm = Cm[Cm.padj < 0.05].sort_values("score", ascending=False).drop_duplicates("pair").head(14)
    ax = fig.add_subplot(gs[1, 0]); ax.barh(range(len(Cm))[::-1], Cm.score, color=CH, height=0.65)
    ax.set_yticks(range(len(Cm))[::-1]); ax.set_yticklabels([p.replace("-", " → ") for p in Cm.pair], fontsize=5)
    ax.set_xlabel("interaction score"); ax.set_title("Fourteen highest-scoring pairs\n(best sender-receiver)"); lab(ax, "C", -0.75); src(Cm, "Figure8C_top_pairs")
    V = pd.read_csv(f"{RES}/13_macrophage_contrast_patient_level.csv")
    ax = fig.add_subplot(gs[1, 1]); nom = V.p < 0.05
    ax.scatter(V.log2_ratio, -np.log10(V.p), s=7, color=LGREY, lw=0); ax.scatter(V.log2_ratio[nom], -np.log10(V.p[nom]), s=10, color=CNH, ec="white", lw=0.3)
    for _, r in V.nsmallest(5, "p").iterrows(): ax.annotate(r.pair, (r.log2_ratio, -np.log10(r.p)), fontsize=4.6, xytext=(2, 2), textcoords="offset points")
    ax.axhline(-np.log10(0.05), ls=":", color=GREY, lw=0.7); ax.axvline(0, color=LGREY, lw=0.6)
    ax.set_xlabel("log2 ratio (non-healing / healing)"); ax.set_ylabel("-log10 P (library level)")
    ax.set_title(f"Macrophage interactions per library\n({len(V)} tests; {int(nom.sum())} nominal, 0 at FDR < 0.05)"); lab(ax, "D", -0.35)
    src(V, "Figure8D_macrophage_contrast_library_level"); save(fig, "Figure8_communication")

# =========================================================== Figure 9 bulk
def figure9():
    fig = plt.figure(figsize=(W, 5.6)); gs = fig.add_gridspec(2, 2, wspace=0.75, hspace=0.6)
    D = pd.read_csv(f"{RES}/06_discovery_DESeq2_GSE134431.csv", index_col=0).dropna(subset=["pvalue"])
    ax = fig.add_subplot(gs[0, 0]); s = D.padj < 0.05
    ax.scatter(D.log2FoldChange.clip(-6, 6), -np.log10(D.pvalue), s=1.5, color=LGREY, rasterized=True, lw=0)
    ax.scatter(D.log2FoldChange[s].clip(-6, 6), -np.log10(D.pvalue[s]), s=7, color=CNH, ec="white", lw=0.3)
    for k, (g, r) in enumerate(D.nsmallest(6, "pvalue").iterrows()): ax.annotate(g, (np.clip(r.log2FoldChange, -6, 6), -np.log10(r.pvalue)), fontsize=5, xytext=(3, [2, 2, 8, -9, 14, -3][k]), textcoords="offset points", arrowprops=dict(arrowstyle="-", lw=0.3, color=GREY))
    ax.set_xlabel("log2 FC (non-healing / healing)"); ax.set_ylabel("-log10 P"); ax.set_title(f"GSE134431 ulcers (7 vs 6),\nDESeq2: {int(s.sum())} genes at FDR < 0.05"); lab(ax, "A", -0.35)
    src(D, "Figure9A_bulk_DE")
    E = pd.read_csv(f"{RES}/14_pathway_enrichment.csv"); E = E[E.direction == "higher in healing"].nsmallest(10, "Adjusted P-value")
    ax = fig.add_subplot(gs[0, 1]); ax.barh(range(10)[::-1], -np.log10(E["Adjusted P-value"]), color=CH, height=0.65)
    ax.set_yticks(range(10)[::-1]); ax.set_yticklabels([textwrap.shorten(t.split(" (GO")[0], 42, placeholder="...") for t in E.Term], fontsize=5)
    ax.axvline(-np.log10(0.05), ls=":", color=GREY, lw=0.7); ax.set_xlabel("-log10 FDR"); ax.set_title("Gene sets over-represented among genes\nnominally higher in healing ulcers"); lab(ax, "B", -0.9)
    src(E, "Figure9B_pathways")
    C = pd.read_csv(f"{OUT}/R2_08_bulk_vs_sc_concordance.csv", index_col=0); cs = json.load(open(f"{OUT}/R2_08_networks_summary.json"))["concordance"]["M0"]
    ax = fig.add_subplot(gs[1, 0]); ax.scatter(C.bulk.clip(-4, 4), C.sc.clip(-4, 4), s=1.5, color=LGREY, rasterized=True, lw=0)
    ax.scatter(C.bulk[C.leading].clip(-4, 4), C.sc[C.leading].clip(-4, 4), s=7, color=CH, ec="white", lw=0.3)
    ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5); ax.set_xlabel("bulk log2 FC"); ax.set_ylabel("macrophage pseudobulk log2 FC")
    ax.set_title(f"Bulk vs single-cell (raw-count pseudobulk)\nrho {cs['spearman_all']:.2f} ({cs['ci_all'][0]:.2f}-{cs['ci_all'][1]:.2f}); nominal in both:\n"
                 f"rho {cs['spearman_leading']:.2f} ({cs['ci_leading'][0]:.2f}-{cs['ci_leading'][1]:.2f}), n = {cs['n_leading']}", fontsize=6)
    lab(ax, "C", -0.4); src(C, "Figure9C_bulk_vs_singlecell")
    R = pd.read_csv(f"{KHAN}/04_raw_data_by_figure/FigureR11d_representative_genes.csv"); R = R[R.gene.isin(D.index)]
    R = R.assign(lfc=D.loc[R.gene, "log2FoldChange"].values, lo=D.loc[R.gene, "ci_low"].values, hi=D.loc[R.gene, "ci_high"].values, q=D.loc[R.gene, "padj"].values).sort_values("lfc")
    pc = {"myeloid activation": CH, "resident / complement": "#009E73", "lipid metabolism": "#E69F00"}
    ax = fig.add_subplot(gs[1, 1]); y = np.arange(len(R))
    for yy, (_, r) in zip(y, R.iterrows()):
        ax.plot([r.lo, r.hi], [yy, yy], color=pc.get(r.panel, GREY), lw=0.8); ax.scatter(r.lfc, yy, s=10, color=pc.get(r.panel, GREY), zorder=3)
    ax.axvline(0, color=GREY, lw=0.6); ax.set_yticks(y); ax.set_yticklabels(R.gene, fontsize=4.8); ax.set_xlabel("log2 FC (95% CI)")
    ax.set_title("Selected immune, complement and\nlipid genes in bulk ulcers")
    ax.legend(handles=[Line2D([], [], color=c, marker="o", ms=3, label=k) for k, c in pc.items()], frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3); lab(ax, "D", -0.45)
    src(R, "Figure9D_representative_genes"); save(fig, "Figure9_bulk_landscape")

# =========================================================== Figure 10 deconvolution
def figure10():
    fig = plt.figure(figsize=(W, 4.9)); gs = fig.add_gridspec(2, 2, wspace=0.4, hspace=0.75)
    Tt = pd.read_csv(f"{RES}/05_benchmark_true_proportions.csv", index_col=0); Ee = pd.read_csv(f"{RES}/05_benchmark_estimated_full.csv", index_col=0)
    cols = [c for c in Tt.columns if c in Ee.columns]; r = np.corrcoef(Tt[cols].values.ravel(), Ee[cols].values.ravel())[0, 1]
    ax = fig.add_subplot(gs[0, 0]); ax.scatter(100 * Tt[cols].values.ravel(), 100 * Ee[cols].values.ravel(), s=1.5, color=CH, alpha=0.4, rasterized=True, lw=0)
    ax.plot([0, 70], [0, 70], ls="--", color=GREY, lw=0.7); ax.set_xlabel("true fraction (%)"); ax.set_ylabel("estimated fraction (%)")
    ax.set_title(f"200 synthetic mixtures of known\ncomposition (r = {r:.3f})"); lab(ax, "A", -0.3)
    src(pd.DataFrame({"true": Tt[cols].values.ravel(), "estimated": Ee[cols].values.ravel()}), "Figure10A_benchmark")
    B = pd.read_csv(f"{KHAN}/04_raw_data_by_figure/FigureR6b_lambda_benchmark.csv")
    ax = fig.add_subplot(gs[0, 1])
    for (bm, d), c in zip(B.groupby("benchmark"), [CNH, CH]): ax.plot(range(len(d)), d.rmse, marker="o", ms=3, color=c, lw=1, label=bm)
    ax.set_xticks(range(len(d))); ax.set_xticklabels([f"{v:g}" for v in d["lambda"]], fontsize=5); ax.set_xlabel("ridge penalty (lambda)"); ax.set_ylabel("RMSE")
    ax.legend(frameon=False); ax.set_title("Estimation error by ridge penalty\n(lambda = 0 selected)"); lab(ax, "B", -0.35); src(B, "Figure10B_lambda")
    Rf = pd.read_csv(f"{KHAN}/04_raw_data_by_figure/FigureR6c_reference_size_benchmark.csv")
    ax = fig.add_subplot(gs[1, 0]); ax.bar([0, 1], Rf.mac_pearson, color=[CH, LGREY], width=0.55)
    for i, (rr, n) in enumerate(zip(Rf.mac_pearson, Rf.n_macrophages)): ax.text(i, rr + 0.02, f"r = {rr:.3f}\n{n:,} macrophages", ha="center", fontsize=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["full reference", "300 cells/library"], fontsize=5.5); ax.set_ylim(0, 1.25); ax.set_ylabel("macrophage-fraction accuracy (r)")
    ax.set_title("Accuracy with full vs reduced\nreference (same mixtures)"); lab(ax, "C", -0.35); src(Rf, "Figure10C_reference_size")
    dc = pd.read_csv(f"{RES}/05_deconvolution_combined.csv", index_col=0); g = dc.iloc[:, -1]; e = eff("bulk GSE134431 abundance", "macrophage_marker_score")
    pw = json.load(open(f"{RES}/05_power_analysis.json"))
    ax = fig.add_subplot(gs[1, 1])
    for j, (gg, c) in enumerate([("Healing", CH), ("Nonhealing", CNH)]):
        strip(ax, j, dc.loc[g == gg, "macrophage_marker_score"], c, j)
        m, lo, hi = (e.mean_healing, e.ci_healing_low, e.ci_healing_high) if gg == "Healing" else (e.mean_nonhealing, e.ci_nonhealing_low, e.ci_nonhealing_high)
        ci_point(ax, j, m, lo, hi, c)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healing", "Non-healing"]); ax.set_ylabel("macrophage marker score")
    ax.set_title(f"Reference-free macrophage marker score (bulk)\nHodges-Lehmann shift {e.hodges_lehmann:.2f} (95% CI {e.hl_ci_low:.2f} to {e.hl_ci_high:.2f}); P = {e.p_mannwhitney:.2f}\n80% power only for Cohen's d > {pw['min_detectable_cohens_d']:.2f}", fontsize=6)
    lab(ax, "D", -0.4); src(dc[["macrophage_marker_score"]].assign(group=g), "Figure10D_marker_score"); save(fig, "Figure10_deconvolution_power")

# =========================================================== Figure 11 networks
def figure11():
    MT = pd.read_csv(f"{OUT}/R2_08_module_trait.csv"); O = pd.read_csv(f"{OUT}/R2_08_module_overlap.csv"); ns = json.load(open(f"{OUT}/R2_08_networks_summary.json"))
    fig = plt.figure(figsize=(W, 5.6)); gs = fig.add_gridspec(2, 2, wspace=0.5, hspace=0.6)
    for k, (net, ttl) in enumerate([("bulk GSE134431", "Bulk network"), ("single-cell pseudobulk (raw counts)", "Single-cell pseudobulk network")]):
        d = MT[MT.network == net].sort_values("pearson_r_with_nonhealing"); ax = fig.add_subplot(gs[0, k]); y = np.arange(len(d))
        for yy, (_, r) in zip(y, d.iterrows()):
            ax.plot([r.ci_low, r.ci_high], [yy, yy], color=GREY, lw=0.8); ax.scatter(r.pearson_r_with_nonhealing, yy, s=12, color=CNH if r.p < 0.05 else CH, zorder=3)
        ax.axvline(0, color=GREY, lw=0.6); ax.set_yticks(y); ax.set_yticklabels([f"{m} ({n})" for m, n in zip(d.module, d.n_genes)], fontsize=5)
        ax.set_xlabel("r with non-healing (95% CI)"); ax.set_title(f"{ttl}\n({len(d)} modules; 0 at FDR < 0.05)"); lab(ax, "AB"[k], -0.6)
        src(d, f"Figure11{'AB'[k]}_module_trait")
    piv = O.pivot(index="bulk_module", columns="sc_module", values="jaccard").fillna(0)
    piv = piv[sorted(piv.columns, key=lambda c: int(c.split("M")[-1]))]; piv = piv.loc[sorted(piv.index, key=lambda c: int(c.split("M")[-1]))]
    ax = fig.add_subplot(gs[1, 0]); im = ax.imshow(piv.values, cmap="magma", aspect="auto")
    for _, r in O[O.padj < 0.05].iterrows(): ax.text(list(piv.columns).index(r.sc_module), list(piv.index).index(r.bulk_module), "*", ha="center", va="center", color="white", fontsize=8)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=90, fontsize=5); ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=5)
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("Jaccard overlap", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title(f"Module overlap between networks\n(* FDR < 0.05; {ns['n_pairs_FDR05']} of {ns['n_pairs']} pairs)"); lab(ax, "C", -0.4); src(piv, "Figure11C_module_overlap")
    A = pd.read_csv(f"{RES}/15_consensus_panel_annotated.csv"); A = A[A.n_analyses >= 3]
    ct = A.groupby("n_analyses").agg(panel=("gene", "size"), recovered=("recovered_full_scale", "sum"))
    ax = fig.add_subplot(gs[1, 1]); x = np.arange(len(ct)); ax.bar(x, ct.panel, color=LGREY, width=0.6, label="panel genes"); ax.bar(x, ct.recovered, color=CH, width=0.4, label="recovered in the full dataset")
    for i, (p, rr) in enumerate(zip(ct.panel, ct.recovered)): ax.text(i, p + 2, f"{rr}/{p}", ha="center", fontsize=5.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{i}" for i in ct.index]); ax.set_xlabel("discovery evidence sets\nsupporting the gene"); ax.set_ylabel("genes")
    ax.legend(frameon=False, fontsize=5); ax.set_title(f"129-gene candidate panel\n({int(ct.recovered.sum())} recovered in the full dataset)"); lab(ax, "D", -0.45)
    src(ct, "Figure11D_panel_evidence"); save(fig, "Figure11_networks")

# =========================================================== Figure 12 bulk ML
def figure12():
    ML = pd.read_csv(f"{OUT}/R2_07_bulk_classifier_AUC_CIs.csv"); EX = pd.read_csv(f"{OUT}/R2_07_external_transfer_AUC_CIs.csv").set_index("feature_set")
    NU = json.load(open(f"{RES}/17_permutation_nulls.json")); FI = pd.read_csv(f"{RES}/17_bulk_feature_importance.csv").head(15)
    sets = ["all genes (k=200 selected in-fold)", "consensus panel", "canonical wound genes"]; short = ["all genes\n(k = 200 in-fold)", "consensus\npanel", "canonical\nwound genes"]
    fig = plt.figure(figsize=(W, 5.0)); gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.75)
    ax = fig.add_subplot(gs[0, 0]); L1 = ML[ML.model == "L1 logistic"].set_index("feature_set")
    for i, s in enumerate(sets):
        for j, (m, lo, hi, c, nm) in enumerate([("cv5_auc", "cv5_ci_low", "cv5_ci_high", "#56B4E9", "5-fold CV"), ("loo_auc", "loo_ci_low", "loo_ci_high", CH, "leave-one-out")]):
            x = i + (j - 0.5) * 0.3; ax.bar(x, L1.loc[s, m], 0.28, color=c, label=nm if i == 0 else None)
            ax.plot([x, x], [L1.loc[s, lo], L1.loc[s, hi]], color="black", lw=0.8)
    ax.axhline(0.5, ls="--", color=GREY, lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=5.3); ax.set_ylim(0, 1.32); ax.set_ylabel("AUC (95% CI)")
    ax.legend(frameon=False, fontsize=5, loc="upper right", ncol=2); ax.set_title("L1-logistic, GSE134431 (n = 13),\nselection inside folds"); lab(ax, "A", -0.35)
    src(L1, "Figure12A_internal_AUC")
    ax = fig.add_subplot(gs[0, 1])
    for i, s in enumerate(sets):
        n = NU[s]; ax.bar(i, n["observed"], 0.55, color=CH); ax.errorbar(i, n["null_mean"], yerr=n["null_sd"], fmt="o", color="black", ms=3, lw=0.8)
        ax.text(i, n["observed"] + 0.04, f"P = {n['p']:.3f}", ha="center", fontsize=5.3)
    ax.axhline(0.5, ls="--", color=GREY, lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=5.3); ax.set_ylim(0, 1.15); ax.set_ylabel("leave-one-out AUC")
    ax.set_title("Observed vs label-permutation null\n(500 permutations; mean +/- SD)"); lab(ax, "B", -0.35); src(pd.DataFrame(NU).T, "Figure12B_permutation")
    ax = fig.add_subplot(gs[1, 0])
    for i, s in enumerate(sets):
        for j, (v, lo, hi, c, nm) in enumerate([(L1.loc[s, "loo_auc"], L1.loc[s, "loo_ci_low"], L1.loc[s, "loo_ci_high"], CH, "internal LOO (GSE134431)"),
                                                (EX.loc[s, "external_auc"], EX.loc[s, "ci_low"], EX.loc[s, "ci_high"], CNH, "external (GSE143735, n = 9)")]):
            x = i + (j - 0.5) * 0.3; ax.bar(x, v, 0.28, color=c, label=nm if i == 0 else None); ax.plot([x, x], [lo, hi], color="black", lw=0.8)
    ax.axhline(0.5, ls="--", color=GREY, lw=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(short, fontsize=5.3); ax.set_ylim(0, 1.32); ax.set_ylabel("AUC (95% CI)")
    ax.legend(frameon=False, fontsize=5, loc="upper right", ncol=2); ax.set_title("Internal vs external performance;\npanel was built using GSE134431 labels"); lab(ax, "C", -0.35)
    src(EX, "Figure12C_external_transfer")
    ax = fig.add_subplot(gs[1, 1]); y = np.arange(len(FI))[::-1]
    ax.barh(y, FI.importance, color=[CNH if v else CH for v in FI.l1_nonzero], height=0.65); ax.set_yticks(y); ax.set_yticklabels(FI.gene, fontsize=5)
    ax.set_xlabel("random-forest importance"); ax.set_title("Fifteen leading panel genes\n(orange: retained by L1)"); lab(ax, "D", -0.45)
    src(FI, "Figure12D_feature_importance"); save(fig, "Figure12_bulk_machine_learning")

# =========================================================== Figure 13 per-cell classifier
def figure13():
    R = pd.read_csv(f"{OUT}/R2_06_classifier_AUC_CIs.csv"); s7 = json.load(open(f"{RES}/07b_summary.json"))
    keep = ["per-cell, library-grouped 4-fold CV (mean of 5 split seeds)", "per-cell, leave-one-library-out", "per-cell, putative-donor-grouped 4-fold CV (5 seeds)",
            "per-cell, foot libraries only, library-grouped CV (5 seeds)", "library outcome from aggregated per-cell predictions", "library outcome, leave-one-library-out",
            "donor outcome from aggregated per-cell predictions (donor-grouped CV)", "library-level pseudobulk classifier (raw-count pseudobulk), leave-one-out"]
    nm = ["cell, library-grouped", "cell, leave-one-library-out", "cell, donor-grouped", "cell, foot only", "library, aggregated", "library, LOLO", "donor, aggregated", "library, pseudobulk"]
    Rk = R.set_index("evaluation").loc[keep].reset_index()
    fig = plt.figure(figsize=(W, 5.4)); gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.8)
    ax = fig.add_subplot(gs[0, 0]); x = np.arange(len(Rk) + 1)
    ax.bar(0, s7["naive_ungrouped_auc"], 0.6, color=LGREY); ax.text(0, s7["naive_ungrouped_auc"] + 0.03, f"{s7['naive_ungrouped_auc']:.2f}", ha="center", fontsize=5)
    for i, (_, r) in enumerate(Rk.iterrows(), start=1):
        c = CH if r.unit == "cell" else CNH; ax.bar(i, r.auc, 0.6, color=c); ax.plot([i, i], [r.ci_low, r.ci_high], color="black", lw=0.8)
    ax.axhline(0.5, ls="--", color=GREY, lw=0.6); ax.set_xticks(x); ax.set_xticklabels(["cell, ungrouped\n(leaks libraries)"] + nm, rotation=50, ha="right", fontsize=5)
    ax.set_ylim(0, 1.32); ax.set_ylabel("AUC (95% CI)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CH, label="per-cell state"), plt.Rectangle((0, 0), 1, 1, color=CNH, label="library / donor outcome"),
                       plt.Rectangle((0, 0), 1, 1, color=LGREY, label="not grouped (optimistic)")], frameon=False, fontsize=5, loc="upper right", ncol=3)
    ax.set_title("Macrophage classifier under eight evaluation schemes\n(cluster-bootstrap or DeLong 95% CI)"); lab(ax, "A", -0.2)
    src(R, "Figure13A_AUC_with_CIs")
    P = pd.read_csv(f"{OUT}/R2_06_per_cell_out_of_fold_predictions.csv.gz")
    from sklearn.metrics import roc_curve
    ax = fig.add_subplot(gs[0, 1]); f1, t1, _ = roc_curve(P.y_nonhealing, P.oof_library_grouped)
    Pl = P.groupby("library").agg(p=("oof_library_grouped", "mean"), y=("y_nonhealing", "first")); f2, t2, _ = roc_curve(Pl.y, Pl.p)
    ax.plot(f1, t1, color=CH, lw=1.2, label=f"per cell (AUC {Rk.auc[0]:.2f})"); ax.step(f2, t2, color=CNH, lw=1.2, where="post", label=f"per library (AUC {Rk.auc[4]:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=0.6); ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate"); ax.legend(frameon=False, fontsize=5, loc="lower right")
    ax.set_title("Same predictions scored per cell\nand per library"); lab(ax, "B", -0.35); src(Pl, "Figure13B_library_predictions")
    rep = pd.read_csv(f"{RES}/07b_repeated_grouped_cv.csv"); ax = fig.add_subplot(gs[1, 0])
    ax.hist(rep.auc, bins=12, color=CH, ec="white"); ax.axvline(rep.auc.mean(), color="black", lw=1)
    ax.set_xlabel("per-cell AUC"); ax.set_ylabel("splits"); ax.set_title(f"25 library-grouped splits\nmean {rep.auc.mean():.2f}; 2.5-97.5%: {rep.auc.quantile(.025):.2f}-{rep.auc.quantile(.975):.2f}")
    lab(ax, "C", -0.35); src(rep, "Figure13C_repeated_splits")
    lc = pd.read_csv(f"{KHAN}/04_raw_data_by_figure/FigureR8d_learning_curve.csv"); ax = fig.add_subplot(gs[1, 1])
    ax.errorbar(lc.n_train_patients, lc.mean_auc, yerr=lc.sd_auc, color=CH, marker="o", ms=3, lw=1, ecolor=LGREY, capsize=1.5)
    ax.axhline(0.5, ls="--", color=GREY, lw=0.6); ax.set_xlabel("training libraries"); ax.set_ylabel("per-cell AUC, held-out libraries")
    ax.set_title("Performance vs number of\ntraining libraries (mean +/- SD)"); lab(ax, "D", -0.4); src(lc, "Figure13D_learning_curve")
    save(fig, "Figure13_macrophage_classifier")

# =========================================================== Figure 14 replication
def figure14():
    RR = pd.read_csv(f"{OUT}/R2_07_directional_replication_CIs.csv")
    J = pd.read_csv(f"{RES}/06_discovery_vs_validation_joint.csv", index_col=0); V = pd.read_csv(f"{KHAN}/04_raw_data_by_figure/FigureR7c_variance_site_vs_healing.csv", index_col=0)
    sv = json.load(open(f"{RES}/06_site_variance_summary.json"))
    fig = plt.figure(figsize=(W, 5.4)); gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.75)
    ax = fig.add_subplot(gs[0, :]); x = np.arange(len(RR))
    ax.bar(x - 0.18, RR.same_direction_pct, 0.34, color=CH, label="same direction")
    ax.bar(x + 0.18, RR.replicated_pct, 0.34, color=CNH, label="same direction and P < 0.05")
    for i, r in RR.iterrows():
        ax.plot([i - 0.18] * 2, [r.same_direction_ci_low, r.same_direction_ci_high], color="black", lw=0.8)
        ax.plot([i + 0.18] * 2, [r.replicated_ci_low, r.replicated_ci_high], color="black", lw=0.8)
    ax.axhline(50, ls="--", color=GREY, lw=0.6); ax.axhline(2.5, ls=":", color=GREY, lw=0.6)
    labels = ["whole\ntranscriptome", "nominal\nP < 0.05", "FDR < 0.05", "panel\n>= 3", "panel\n>= 4", "panel\n>= 5", "panel >= 3,\n|log2FC| > 1"]
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, RR.n_genes)], fontsize=5.5); ax.set_ylabel("% of genes in GSE143735 (95% CI)"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left"); ax.set_title("Directional concordance and statistical replication in the forearm cohort GSE143735\n(exact 95% CI; dashed line: 50% expected by chance; dotted line: 2.5% expected by chance)"); lab(ax, "A", -0.15)
    src(RR, "Figure14A_replication_CIs")
    ax = fig.add_subplot(gs[1, 0]); ax.scatter(J.disc_log2FC.clip(-3, 3), J.val_log2FC.clip(-3, 3), s=1.2, color=LGREY, rasterized=True, lw=0)
    p = J[J.in_panel]; ax.scatter(p.disc_log2FC.clip(-3, 3), p.val_log2FC.clip(-3, 3), s=8, color=CH, ec="white", lw=0.3, label=f"panel genes (n = {len(p)})")
    rho = stats.spearmanr(p.disc_log2FC, p.val_log2FC)[0]
    ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5); ax.set_xlabel("log2 FC, foot ulcers (GSE134431)"); ax.set_ylabel("log2 FC, forearm skin (GSE143735)")
    ax.legend(frameon=False, fontsize=5); ax.set_title(f"Discovery vs external effect sizes\n(panel genes: rho = {rho:.2f})"); lab(ax, "B", -0.3)
    src(J[["disc_log2FC", "val_log2FC", "in_panel", "same_sign", "statistically_replicated"]], "Figure14B_effect_sizes")
    ax = fig.add_subplot(gs[1, 1]); bins = np.linspace(0, 1, 41)
    ax.hist(V.eta2_site, bins=bins, color="#E69F00", alpha=0.8, label="anatomical site"); ax.hist(V.eta2_healing, bins=bins, color=CH, alpha=0.8, label="healing status")
    ax.set_xlabel("variance explained (eta squared)"); ax.set_ylabel("genes"); ax.legend(frameon=False, fontsize=5)
    ax.set_title(f"Variance explained in the 4,000 most\nvariable genes (medians {sv['median_eta2_site']:.2f} vs {sv['median_eta2_healing']:.3f})"); lab(ax, "C", -0.3)
    src(V, "Figure14C_site_vs_healing_variance"); save(fig, "Figure14_external_replication")

# =========================================================== Figure 15 confounding / independence
def figure15():
    fig = plt.figure(figsize=(W, 5.6)); gs = fig.add_gridspec(2, 2, wspace=0.45, hspace=0.85, width_ratios=[1, 1.15])
    ax = fig.add_subplot(gs[0, 0]); t = T1
    for tis, mk in [("Foot", "o"), ("Forearm", "s"), ("PBMC", "^")]:
        d = t[t.tissue_site == tis]; ax.scatter(d.XIST_log2CPM, d.Y_genes_log2CPM, s=12, marker=mk, color=[CNH if m else CH for m in d.mixed_XIST_Y_signal], ec="white", lw=0.3, label=tis)
    for g in t.index[t.mixed_XIST_Y_signal]: ax.annotate(g, (t.loc[g, "XIST_log2CPM"], t.loc[g, "Y_genes_log2CPM"]), fontsize=5, xytext=(2, -6), textcoords="offset points")
    ax.set_xlabel("XIST (log2 CPM)"); ax.set_ylabel("Y-chromosome genes (log2 CPM)"); ax.legend(frameon=False, fontsize=5, loc="center right")
    ax.set_title("Sex inferred for all 54 libraries\n(orange: mixed XIST/Y signal)"); lab(ax, "A", -0.35); src(t[["XIST_log2CPM", "Y_genes_log2CPM", "sex_inferred", "mixed_XIST_Y_signal", "tissue_site"]], "Figure15A_sex_inference")
    c = cov[cov.group.isin(["Healing", "Nonhealing"]) & (cov.site != "PBMC")]; tab = pd.crosstab(c.group, c.sex)
    ax = fig.add_subplot(gs[0, 1]); ax.bar([0, 1], tab["female"], color="#CC79A7", width=0.55, label="female"); ax.bar([0, 1], tab["male"], bottom=tab["female"], color="#56B4E9", width=0.55, label="male")
    for i, g in enumerate(tab.index): ax.text(i, tab.loc[g, "female"] / 2, tab.loc[g, "female"], ha="center", fontsize=6); ax.text(i, tab.loc[g, "female"] + tab.loc[g, "male"] / 2, tab.loc[g, "male"], ha="center", fontsize=6)
    from scipy.stats import fisher_exact
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healing", "Non-\nhealing"]); ax.set_ylabel("libraries"); ax.legend(frameon=False, fontsize=5, loc="upper right")
    ax.set_title(f"Inferred sex by group\n(Fisher P = {fisher_exact(tab.values)[1]:.2f})"); lab(ax, "B", -0.6); src(tab, "Figure15B_sex_by_group")
    G = pd.read_csv(f"{OUT}/R2_04_donor_proxy_markers.csv", index_col=0)
    dfu = T1.index[T1.analysis_group.isin(["Healing", "Nonhealing"]) & T1.included_in_skin_atlas]
    mk = ["ERAP2_over_ERAP1", "GSTM1_over_GSTM2", "HLA_DRB5_over_DRB1", "CCL3L1_over_CCL3", "CCL4L2_over_CCL4", "KANSL1AS1_over_KANSL1", "UGT2B17_log2cpm"]
    skin = T1.index[T1.included_in_skin_atlas]; Z = (G.loc[dfu, mk] - G.loc[skin, mk].mean()) / G.loc[skin, mk].std()
    Z["donor"] = T1.loc[dfu, "putative_donor_group"].str.split(" ").str[0]; Z = Z.sort_values(["donor"])
    ax = fig.add_subplot(gs[1, 0]); im = ax.imshow(Z[mk].T.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(Z))); ax.set_xticklabels([f"{g} {T1.loc[g,'tissue_site'][:2]} {Z.loc[g,'donor']}" for g in Z.index], rotation=90, fontsize=4.8)
    ax.set_yticks(range(len(mk))); ax.set_yticklabels(["ERAP2/ERAP1", "GSTM1/GSTM2", "HLA-DRB5/DRB1", "CCL3L1/CCL3", "CCL4L2/CCL4", "KANSL1-AS1/KANSL1", "UGT2B17"], fontsize=5)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("z score", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title("Germline-variable expression markers in the 19 DFU skin libraries\n(columns grouped by putative donor; Fo = foot, Fa = forearm)"); lab(ax, "C", -0.2)
    src(Z, "Figure15C_donor_proxy_markers")
    a = pd.read_csv(f"{OUT}/R2_05_pseudobulk_DE_M0.csv", index_col=0); b = pd.read_csv(f"{OUT}/R2_05_pseudobulk_DE_M2.csv", index_col=0)
    j = pd.DataFrame({"u": -np.log10(a.pvalue), "a": -np.log10(b.pvalue), "qu": a.padj, "qa": b.padj}).dropna()
    ax = fig.add_subplot(gs[1, 1]); ax.scatter(j.u, j.a, s=1.5, color=LGREY, rasterized=True, lw=0)
    ax.scatter(j.u[j.qu < 0.05], j.a[j.qu < 0.05], s=6, color=CNH, lw=0, label=f"FDR < 0.05 unadjusted ({int((j.qu<0.05).sum())})")
    ax.scatter(j.u[j.qa < 0.05], j.a[j.qa < 0.05], s=10, facecolor="none", ec=CH, lw=0.7, label=f"FDR < 0.05 adjusted ({int((j.qa<0.05).sum())})")
    mx = max(j.u.max(), j.a.max()); ax.plot([0, mx], [0, mx], ls="--", color=GREY, lw=0.6)
    ax.set_xlabel("-log10 P, ~ group"); ax.set_ylabel("-log10 P, ~ sex + site + group"); ax.legend(frameon=False, fontsize=4.8, loc="upper left")
    ax.set_title("Raw-count pseudobulk before and\nafter covariate adjustment"); lab(ax, "D", -0.35)
    src(j, "Figure15D_adjustment_comparison"); save(fig, "Figure15_confounding_and_independence")

# =========================================================== Figure 16 prioritised genes
def figure16():
    S = pd.read_csv(f"{RES}/18_prioritised_by_state.csv", index_col=0); Z = (S - S.mean()) / S.std(ddof=0)
    Z.index = [SHORT.get(i, i) for i in Z.index]
    fig = plt.figure(figsize=(W, 3.0)); gs = fig.add_gridspec(1, 3, wspace=0.75, width_ratios=[1.1, 1.5, 0.9])
    ax = fig.add_subplot(gs[0]); im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(Z.shape[1])); ax.set_xticklabels(Z.columns, rotation=90, fontsize=5.3); ax.set_yticks(range(3)); ax.set_yticklabels([f"{i}" for i in Z.index], fontsize=5.5)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("z-scored mean expression", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title("Prioritized genes across the three\nmacrophage states"); lab(ax, "A", -0.45); src(Z, "Figure16A_by_state")
    P = pd.read_csv(f"{RES}/18_prioritised_per_patient.csv"); genes = ["S100A9", "HIF1A", "GK", "COL3A1", "COL1A1", "CD99"]
    ax = fig.add_subplot(gs[1]); rows = []
    for i, g in enumerate(genes):
        e = eff("prioritised gene", g)
        for j, (gg, c) in enumerate([("Healing", CH), ("Nonhealing", CNH)]):
            x = i + (j - 0.5) * 0.36; strip(ax, x, P.loc[P.group == gg, g], c, i * 2 + j)
            m, lo, hi = (e.mean_healing, e.ci_healing_low, e.ci_healing_high) if gg == "Healing" else (e.mean_nonhealing, e.ci_nonhealing_low, e.ci_nonhealing_high)
            ci_point(ax, x, m, lo, hi, c, 0.1); rows.append((g, gg, m, lo, hi, e.p_mannwhitney))
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{g}\nP =\n{eff('prioritised gene', g).p_mannwhitney:.2f}" for g in genes], fontsize=5)
    ax.set_ylabel("mean log-expression"); ax.set_title("Library-level expression of six genes\n(mean, 95% CI; Mann-Whitney P, unadjusted)")
    legend_groups(ax, off=-0.3); lab(ax, "B", -0.2); src(pd.DataFrame(rows, columns=["gene", "group", "mean", "ci_low", "ci_high", "p"]), "Figure16B_library_expression")
    Tt = pd.read_csv(f"{RES}/18_prioritised_gene_tests.csv"); ax = fig.add_subplot(gs[2])
    ax.scatter(-np.log10(Tt.p_cell_level), -np.log10(Tt.p_patient_level), s=14, color=CH, ec="white", lw=0.3)
    for _, r in Tt.iterrows(): ax.annotate(r.gene, (-np.log10(r.p_cell_level), -np.log10(r.p_patient_level)), fontsize=4.6, xytext=(3, {"COL3A1": -8, "COL1A1": -8, "TGFB1": 3}.get(r.gene, 2)), textcoords="offset points")
    ax.axhline(-np.log10(0.05), ls=":", color=GREY, lw=0.7); ax.set_xlabel("-log10 P, cells as replicates"); ax.set_ylabel("-log10 P, libraries as replicates")
    ax.set_title(f"Cell- vs library-level P\n({int((Tt.padj_patient_level<0.05).sum())} of {len(Tt)} at FDR < 0.05, library level)"); lab(ax, "C", -0.35)
    src(Tt, "Figure16C_cell_vs_library"); save(fig, "Figure16_prioritised_genes")

# =========================================================== Figure 17 perturbation
def figure17():
    Pz = pd.read_csv(f"{RES}/18_perturbation_summary.csv"); st = Pz[Pz.stable]
    fig = plt.figure(figsize=(W * 0.8, 2.7)); gs = fig.add_gridspec(1, 2, wspace=0.45)
    ax = fig.add_subplot(gs[0]); ax.scatter(1e3 * Pz.ko_mean, 1e3 * Pz.oe_mean, s=12, color=LGREY, ec="#999999", lw=0.3, label="sign not stable across models")
    ax.scatter(1e3 * st.ko_mean, 1e3 * st.oe_mean, s=16, color=[CNH if v > 0 else CH for v in st.oe_mean], ec="white", lw=0.3)
    for _, r in st.iterrows(): ax.annotate(r.gene, (1e3 * r.ko_mean, 1e3 * r.oe_mean), fontsize=5, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5)
    ax.set_xlabel("change in predicted P(non-healing), knock-out (x10^-3)"); ax.set_ylabel("change, over-expression (x10^-3)"); ax.legend(frameon=False, fontsize=5, loc="lower left")
    ax.set_title(f"In-silico perturbation of the fitted classifier\n({len(Pz)} panel genes; {len(st)} sign-stable in 25 models)"); lab(ax, "A", -0.2)
    src(Pz, "Figure17A_perturbation")
    s2 = st.sort_values("oe_mean"); ax = fig.add_subplot(gs[1]); y = np.arange(len(s2))
    ax.barh(y, 1e3 * s2.oe_mean, color=[CNH if v > 0 else CH for v in s2.oe_mean], height=0.6, xerr=1e3 * s2.oe_sd, error_kw=dict(lw=0.7, ecolor=GREY))
    ax.set_yticks(y); ax.set_yticklabels(s2.gene); ax.axvline(0, color="black", lw=0.6); ax.set_xlabel("change on over-expression (x10^-3), mean +/- SD")
    ax.set_title("Sign-stable genes: model response only,\nno causal interpretation"); lab(ax, "B", -0.25)
    src(s2, "Figure17B_stable_genes"); save(fig, "Figure17_in_silico_perturbation")

if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"]
    for w in which: globals()[f"figure{w}"]()
