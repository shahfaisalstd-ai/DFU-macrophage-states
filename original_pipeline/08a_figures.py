#!/usr/bin/env python
"""
Khan 2 revision -- Step 08a
Revised Figures R1-R4, redrawn to journal standard.

R1  full-scale QC and cell recovery            (Reviewer 2, comment 4)
R2  the complete 97,677-cell atlas             (Reviewer 2, comment 4)
R3  the myeloid compartment at full scale      (Reviewer 2, comment 4)
R4  functional macrophage programmes beyond M1/M2 (Reviewer 1, comment 1)

Every panel's underlying numbers are written to 04_raw_data_by_figure/.
"""
import os, sys, warnings, json
import numpy as np, pandas as pd, scanpy as sc
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from khan_style import (W1, W15, W2, GROUP, STATE, OKABE_ITO, SEQ, DIV,
                        panel, tidy, save, source_data)

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES, FIG, RAW = f"{BASE}/02_results", f"{BASE}/03_figures", f"{BASE}/04_raw_data_by_figure"
os.makedirs(FIG, exist_ok=True); os.makedirs(RAW, exist_ok=True)
ORDER = ["Healthy", "Diabetic", "Healing", "Nonhealing"]

print("loading objects ...", flush=True)
A   = sc.read_h5ad(f"{RES}/03_atlas_annotated.h5ad")
mye = sc.read_h5ad(f"{RES}/03_myeloid.h5ad")
qc  = pd.read_csv(f"{RES}/01_persample_QC.csv")
sw  = pd.read_csv(f"{RES}/01_QC_threshold_sweep.csv")
print(f"atlas {A.shape}, myeloid {mye.shape}", flush=True)


# =========================================================== FIGURE R1
def figure_R1():
    skin = qc[qc.tissue != "PBMC"].sort_values(["group", "gid"])
    fig = plt.figure(figsize=(W2, 0.62 * W2))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.25,
                          height_ratios=[1, 1])

    # (a) cells per library, raw vs post-QC
    ax = fig.add_subplot(gs[0, :]); tidy(ax, "y")
    x = np.arange(len(skin))
    ax.bar(x - 0.21, skin.n_cells_raw, 0.42, color="#CCCCCC",
           edgecolor="black", lw=0.3, label="raw")
    ax.bar(x + 0.21, skin.n_cells_pass, 0.42,
           color=[GROUP[g] for g in skin.group], edgecolor="black", lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels(skin.gid, rotation=90, fontsize=4.5)
    ax.set_ylabel("cells per library"); ax.set_xlim(-0.8, len(skin) - 0.2)
    ax.legend(handles=[Patch(fc="#CCCCCC", ec="black", lw=.3, label="raw")] +
                      [Patch(fc=GROUP[g], ec="black", lw=.3, label=f"post-QC, {g}")
                       for g in ORDER if g in set(skin.group)],
              loc="upper left", ncol=5, fontsize=5.5, bbox_to_anchor=(0, 1.02))
    ax.set_title(f"All {len(skin)} skin libraries analysed in full "
                 f"({skin.n_cells_raw.sum():,} raw → {skin.n_cells_pass.sum():,} post-QC)",
                 fontsize=6.5, loc="left")
    panel(ax, "a", dx=-0.045, dy=1.20)

    # (b) QC threshold sensitivity
    ax = fig.add_subplot(gs[1, 0]); tidy(ax, "y")
    sk = sw[sw.tissue != "PBMC"]
    tot = sk.groupby(["min_genes", "max_genes"], as_index=False).n_pass.sum()
    tot["rule"] = tot.min_genes.astype(str) + "–" + tot.max_genes.astype(str)
    ref = float(tot.loc[(tot.min_genes == 200) & (tot.max_genes == 5000), "n_pass"].iloc[0])
    tot["pct"] = 100 * tot.n_pass / ref
    tot = tot.sort_values("n_pass")
    cols = ["#D55E00" if r == "200–5000" else "#BBBBBB" for r in tot.rule]
    ax.barh(tot.rule, tot.n_pass, color=cols, edgecolor="black", lw=0.3)
    for i, (n, p) in enumerate(zip(tot.n_pass, tot.pct)):
        ax.text(n, i, f" {p:.1f}%", va="center", fontsize=5.5)
    ax.set_xlabel("cells passing QC"); ax.set_ylabel("genes per cell (min–max)")
    ax.set_xlim(0, tot.n_pass.max() * 1.16)
    ax.set_title("Cell yield is insensitive to the gene-count thresholds\n"
                 "(published rule in orange)", fontsize=6.5, loc="left")
    panel(ax, "b")
    source_data(tot[["rule", "n_pass", "pct"]], RAW, "FigureR1b_QC_threshold_sensitivity")

    # (c) myeloid recovery, published sub-sample vs full atlas
    ax = fig.add_subplot(gs[1, 1]); tidy(ax, "y")
    n_mye = int((A.obs.cell_type == "Myeloid").sum())
    dat = pd.DataFrame({
        "analysis": ["published\nsub-sample", "this revision\nfull atlas"],
        "atlas_cells": [13200, int(A.n_obs)],
        "myeloid_cells": [1342, n_mye]})
    dat["pct"] = 100 * dat.myeloid_cells / dat.atlas_cells
    b = ax.bar(dat.analysis, dat.pct, 0.55, color=["#BBBBBB", "#0072B2"],
               edgecolor="black", lw=0.4)
    for r, m, t in zip(b, dat.myeloid_cells, dat.atlas_cells):
        ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.15,
                f"{m:,} myeloid\nof {t:,}", ha="center", fontsize=5.5)
    ax.set_ylabel("myeloid cells (% of analysed atlas)")
    ax.set_ylim(0, max(dat.pct) * 1.42)
    ax.set_title("Myeloid representation is normal for skin at full scale",
                 fontsize=6.5, loc="left")
    panel(ax, "c")
    source_data(dat, RAW, "FigureR1c_myeloid_recovery")

    source_data(skin, RAW, "FigureR1a_persample_QC")
    save(fig, FIG, "FigureR1_revised_QC_and_recovery")


# =========================================================== FIGURE R2
def figure_R2():
    U = A.obsm["X_umap"]
    ct = A.obs.cell_type.astype(str)
    order = ct.value_counts().index.tolist()
    cmap = {c: OKABE_ITO[i % len(OKABE_ITO)] for i, c in enumerate(order)}
    # widen the palette so 15 lineages stay distinguishable
    extra = plt.get_cmap("tab20").colors
    cmap = {c: (OKABE_ITO[i] if i < 8 else extra[(i - 8) % 20]) for i, c in enumerate(order)}

    fig = plt.figure(figsize=(W2, 0.55 * W2))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

    ax = fig.add_subplot(gs[:, :2]); tidy(ax)
    for c in order:
        m = (ct == c).values
        ax.scatter(U[m, 0], U[m, 1], s=0.6, lw=0, c=[cmap[c]], rasterized=True,
                   label=f"{c} ({m.sum():,})")
        ax.text(np.median(U[m, 0]), np.median(U[m, 1]), c, fontsize=5,
                ha="center", va="center",
                path_effects=None, bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.6))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"Complete atlas: {A.n_obs:,} skin cells, {A.obs.gid.nunique()} libraries, "
                 f"{A.obs.leiden.nunique()} clusters", fontsize=7, loc="left")
    panel(ax, "a", dx=-0.03)

    ax = fig.add_subplot(gs[0, 2]); tidy(ax)
    # shuffle so that no clinical group is systematically drawn on top
    rs = np.random.RandomState(123).permutation(A.n_obs)
    gcol = A.obs.group.astype(str).map(GROUP).values
    ax.scatter(U[rs, 0], U[rs, 1], s=0.3, lw=0, c=gcol[rs], rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, mfc=GROUP[g],
                              mec="none", label=g) for g in ORDER],
              loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("clinical group", fontsize=6.5, loc="left")
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 2]); tidy(ax, "x")
    comp = pd.crosstab(A.obs.group, ct, normalize="index").loc[ORDER] * 100
    comp = comp[order]
    left = np.zeros(len(comp))
    for c in order:
        ax.barh(comp.index, comp[c], left=left, color=cmap[c],
                edgecolor="white", lw=0.2, height=0.72)
        left += comp[c].values
    ax.set_xlabel("% of cells"); ax.set_xlim(0, 100)
    ax.set_title("composition by group", fontsize=6.5, loc="left")
    ax.legend(handles=[Patch(fc=cmap[c], ec="white", lw=.2, label=c) for c in order],
              loc="upper left", bbox_to_anchor=(1.02, 1.05), ncol=1, fontsize=5)
    panel(ax, "c")
    source_data(comp.reset_index(), RAW, "FigureR2c_composition_by_group")
    source_data(ct.value_counts().rename_axis("cell_type").reset_index(name="n_cells"),
                RAW, "FigureR2a_celltype_counts")
    save(fig, FIG, "FigureR2_revised_full_atlas")


# =========================================================== FIGURE R3
def figure_R3():
    U = mye.obsm["X_umap"]
    st = mye.obs.myeloid_subtype.astype(str)
    order = st.value_counts().index.tolist()
    fig = plt.figure(figsize=(W2, 0.46 * W2))
    gs = fig.add_gridspec(1, 3, wspace=0.36, width_ratios=[1.25, 1, 1.15])

    ax = fig.add_subplot(gs[0]); tidy(ax)
    for c in order:
        m = (st == c).values
        ax.scatter(U[m, 0], U[m, 1], s=1.6, lw=0,
                   c=STATE.get(c, "#999999"), rasterized=True, label=f"{c} ({m.sum():,})")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=4, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=2, fontsize=5)
    ax.set_title(f"{mye.n_obs:,} myeloid cells, {mye.obs.mye_leiden.nunique()} subclusters",
                 fontsize=6.5, loc="left")
    panel(ax, "a")

    ax = fig.add_subplot(gs[1]); tidy(ax, "x")
    comp = pd.crosstab(mye.obs.group, st, normalize="index").loc[ORDER] * 100
    left = np.zeros(len(comp))
    for c in order:
        ax.barh(comp.index, comp[c], left=left, color=STATE.get(c, "#999999"),
                edgecolor="white", lw=0.25, height=0.7)
        left += comp[c].values
    ax.set_xlabel("% of myeloid cells"); ax.set_xlim(0, 100)
    ax.set_title("myeloid composition by group", fontsize=6.5, loc="left")
    panel(ax, "b")
    source_data(comp.reset_index(), RAW, "FigureR3b_myeloid_composition")

    # (c) canonical markers by subtype
    ax = fig.add_subplot(gs[2])
    MK = ["FOLR2","LYVE1","C1QA","CD163","SELENOP","S100A8","S100A9","FCN1","VCAN",
          "TREM2","APOE","GPNMB","CD9","CLEC9A","CD1C","FCER1A","CD207","LILRA4","LAMP3","MKI67"]
    MK = [g for g in MK if g in mye.raw.var_names]
    sub = mye.raw[:, MK].X
    df = pd.DataFrame(np.asarray(sub.todense()), columns=MK, index=mye.obs_names)
    df["state"] = st.values
    mean = df.groupby("state")[MK].mean().loc[[c for c in order if c in df.state.unique()]]
    frac = df.groupby("state")[MK].apply(lambda d: (d > 0).mean()).loc[mean.index]
    z = (mean - mean.mean()) / (mean.std(ddof=0) + 1e-9)
    ys, xs, ss, cs = [], [], [], []
    for i, s in enumerate(mean.index):
        for j, g in enumerate(MK):
            ys.append(i); xs.append(j)
            ss.append(frac.loc[s, g] * 26); cs.append(z.loc[s, g])
    sc_ = ax.scatter(xs, ys, s=ss, c=cs, cmap=DIV, vmin=-2, vmax=2,
                     edgecolors="black", linewidths=0.15)
    ax.set_xticks(range(len(MK))); ax.set_xticklabels(MK, rotation=90, fontsize=5)
    ax.set_yticks(range(len(mean.index)))
    ax.set_yticklabels(mean.index, fontsize=5)
    ax.set_xlim(-0.7, len(MK) - 0.3); ax.set_ylim(-0.7, len(mean.index) - 0.3)
    ax.invert_yaxis()
    cb = fig.colorbar(sc_, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label("scaled mean expression", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    h = [Line2D([], [], marker="o", ls="", mfc="grey", mec="black", mew=0.2,
                ms=np.sqrt(f*26), label=f"{int(f*100)}%") for f in (0.25, 0.5, 1.0)]
    ax.legend(handles=h, title="% expressing", loc="upper center",
              bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=5, title_fontsize=5.5)
    ax.set_title("canonical myeloid markers", fontsize=6.5, loc="left")
    panel(ax, "c")
    source_data(mean.reset_index(), RAW, "FigureR3c_marker_mean_expression")
    save(fig, FIG, "FigureR3_revised_myeloid_compartment")


# =========================================================== FIGURE R4
def figure_R4():
    ms = [c for c in mye.obs.columns if c.startswith("ms_")]
    names = [c.replace("ms_", "").replace("_", " ") for c in ms]
    st = mye.obs.myeloid_subtype.astype(str)
    order = [c for c in st.value_counts().index if c in STATE]

    fig = plt.figure(figsize=(W2, 0.50 * W2))
    gs = fig.add_gridspec(1, 3, wspace=0.72, width_ratios=[1.15, 1.0, 1.0])

    # (a) programme x subtype heatmap
    ax = fig.add_subplot(gs[0])
    M = mye.obs.groupby(st, observed=True)[ms].mean().loc[order]
    Z = (M - M.mean()) / (M.std(ddof=0) + 1e-9)
    im = ax.imshow(Z.values, cmap=DIV, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(ms))); ax.set_xticklabels(names, rotation=90, fontsize=5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=5)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("z-scored programme score", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title("Thirteen functional programmes separate myeloid states\n"
                 "that M1/M2 alone does not", fontsize=6.5, loc="left")
    panel(ax, "a")
    source_data(Z.reset_index(), RAW, "FigureR4a_programme_by_subtype")

    # (b) M1 vs M2 fails to separate; resident vs inflammatory does
    macs = ~st.str.contains("DC|Langerhans|Proliferating", regex=True)
    d = mye.obs.loc[macs, ["ms_Classical_M1", "ms_Alternative_M2",
                           "ms_Resident_tissue", "ms_Inflammatory_mono", "group"]].copy()
    d = d[d.group.isin(["Healing", "Nonhealing"])]
    ax = fig.add_subplot(gs[1]); tidy(ax)
    for g in ["Healing", "Nonhealing"]:
        s = d[d.group == g]
        ax.scatter(s.ms_Classical_M1, s.ms_Alternative_M2, s=1.2, lw=0,
                   c=GROUP[g], alpha=0.45, rasterized=True, label=g)
    ax.set_xlabel("classical M1 score"); ax.set_ylabel("alternative M2 score")
    ax.legend(markerscale=5, loc="upper right")
    ax.set_title("M1 and M2 axes overlap between groups", fontsize=6.5, loc="left")
    panel(ax, "b")
    source_data(d.reset_index(), RAW, "FigureR4b_M1_M2_scores")

    # (c) resident vs inflammatory, per patient
    ax = fig.add_subplot(gs[2]); tidy(ax, "y")
    ps = pd.read_csv(f"{RES}/04_programme_scores_per_sample.csv")
    ps = ps[ps.group.isin(["Healing", "Nonhealing"])]
    pairs = [("ms_Resident_tissue", "resident"), ("ms_Inflammatory_mono", "inflammatory"),
             ("ms_Complement", "complement"), ("ms_Lipid_associated_LAM", "lipid-assoc.")]
    centres, tests = [], pd.read_csv(f"{RES}/04_programme_patient_vs_cell_level.csv")
    tmap = dict(zip(tests.programme, tests.p_patient_level))
    for i, (col, lbl) in enumerate(pairs):
        for j, g in enumerate(["Healing", "Nonhealing"]):
            v = ps.loc[ps.group == g, col].dropna().values
            x = i * 2.6 + j * 0.95
            ax.scatter(np.full(len(v), x) + np.random.RandomState(1).uniform(-.16, .16, len(v)),
                       v, s=7, c=GROUP[g], edgecolors="black", linewidths=0.2, zorder=3)
            ax.hlines(np.median(v), x - 0.34, x + 0.34, color="black", lw=1.1, zorder=4)
        centres.append(i * 2.6 + 0.475)
    ax.set_xticks(centres)
    ax.set_xticklabels([f"{l}\nP = {tmap.get(c.replace('ms_',''), float('nan')):.2f}"
                        for c, l in pairs], fontsize=5.5)
    ax.set_ylabel("programme score (mean per patient)")
    ax.set_title("Patient-level scores: direction preserved,\nno difference reaches significance",
                 fontsize=6.5, loc="left")
    ax.legend(handles=[Patch(fc=GROUP[g], ec="black", lw=.3, label=g)
                       for g in ["Healing", "Nonhealing"]],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    panel(ax, "c")
    source_data(ps, RAW, "FigureR4c_patient_level_programme_scores")
    save(fig, FIG, "FigureR4_revised_functional_programmes")


for fn in (figure_R1, figure_R2, figure_R3, figure_R4):
    print(f"\n>>> {fn.__name__}", flush=True)
    fn()
print("\n=== STEP 08a COMPLETE ===", flush=True)
