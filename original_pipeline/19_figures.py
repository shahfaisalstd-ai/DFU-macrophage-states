#!/usr/bin/env python
"""
Khan 2 revision -- Step 19
Revised Figures R9-R16, completing the redraw of every figure in the manuscript.

R9   monocyte-to-macrophage trajectory        (replaces Figure 4)
R10  intercellular communication              (replaces Figure 5)
R11  bulk transcriptomic landscape            (replaces Figure 6)
R12  co-expression networks and integration   (replaces Figure 8)
R13  bulk and feature-based machine learning  (replaces Figure 9)
R14  prioritised genes in macrophages         (replaces Figure 12)
R15  in-silico perturbation                   (replaces Figure 13)
R16  confounder audit                         (new, added in revision)
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from adjustText import adjust_text
from khan_style import (W1, W15, W2, GROUP, STATE, OKABE_ITO, SEQ, DIV,
                        panel, tidy, save, source_data)

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES, FIG, RAW = f"{BASE}/02_results", f"{BASE}/03_figures", f"{BASE}/04_raw_data_by_figure"
R = lambda f: pd.read_csv(f"{RES}/{f}")
J = lambda f: json.load(open(f"{RES}/{f}"))
ORDER = ["Healthy", "Diabetic", "Healing", "Nonhealing"]


# =========================================================== FIGURE R9
def figure_R9():
    import scanpy as sc
    mac = sc.read_h5ad(f"{RES}/12_macrophage_trajectory.h5ad")
    pt_res = J("12_pseudotime_group_test.json")
    per_pat = R("12_pseudotime_per_patient.csv")
    Z = pd.read_csv(f"{RES}/12_trajectory_heatmap_matrix.csv", index_col=0)
    patt = R("12_trajectory_gene_patterns.csv")
    U = mac.obsm["X_umap"]; pt = mac.obs.pseudotime.values
    # Diffusion pseudotime is strongly right-skewed here, so the raw values
    # compress almost every cell into one colour. Panels a and b therefore show
    # the pseudotime RANK (percentile); all statistics use the raw values.
    rank = pd.Series(pt).rank(pct=True).values

    fig = plt.figure(figsize=(W2, 0.56 * W2))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.50)

    ax = fig.add_subplot(gs[0, 0]); tidy(ax)
    sca = ax.scatter(U[:, 0], U[:, 1], c=rank, s=1.6, lw=0, cmap=SEQ, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    cb = fig.colorbar(sca, ax=ax, pad=0.02, fraction=0.035)
    cb.set_label("percentile", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title(f"{mac.n_obs:,} macrophages ordered\nfrom the monocyte root",
                 fontsize=6.5, loc="left")
    panel(ax, "a", dy=1.14)

    ax = fig.add_subplot(gs[0, 1]); tidy(ax, "x")
    mac.obs["_rank"] = rank
    states = mac.obs.groupby("myeloid_subtype", observed=True)._rank.median().sort_values().index
    for i, st in enumerate(states):
        v = mac.obs.loc[mac.obs.myeloid_subtype == st, "_rank"].values
        ax.scatter(v, np.full(len(v), i) + np.random.RandomState(0).uniform(-.22, .22, len(v)),
                   s=1.0, lw=0, c=STATE.get(st, "#999"), rasterized=True, alpha=.45)
        ax.plot(np.median(v), i, marker="|", ms=12, c="black", mew=1.6, zorder=5)
    SHORT = {"Resident M2-like macrophage": "resident",
             "Lipid-associated macrophage": "lipid-assoc.",
             "Inflammatory monocyte-macrophage": "inflammatory"}
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([SHORT.get(s, s) for s in states], fontsize=5.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(states) - 0.4)
    ax.set_xlabel("pseudotime percentile")
    ax.set_title("states occupy successive\npositions", fontsize=6.5, loc="left")
    panel(ax, "b", dx=-0.42, dy=1.14)

    ax = fig.add_subplot(gs[0, 2]); tidy(ax, "y")
    for j, g in enumerate(["Healing", "Nonhealing"]):
        v = per_pat.loc[per_pat.group == g, "pseudotime"].values
        ax.scatter(np.full(len(v), j) + np.random.RandomState(1).uniform(-.12, .12, len(v)),
                   v, s=16, c=GROUP[g], edgecolors="black", linewidths=0.3, zorder=3)
        ax.hlines(np.median(v), j - .26, j + .26, color="black", lw=1.2, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healing", "Non-healing"], fontsize=6)
    ax.set_ylabel("mean pseudotime per patient")
    ax.set_title(f"Patient level: P = {pt_res['p_patient_level']:.2f}\n"
                 f"(cell level P = {pt_res['p_cell_level']:.0e})", fontsize=6.5, loc="left")
    panel(ax, "c", dy=1.14)
    source_data(per_pat, RAW, "FigureR9c_pseudotime_per_patient")

    ax = fig.add_subplot(gs[1, :2])
    ordr = Z.values.argmax(1).argsort()
    im = ax.imshow(Z.values[ordr], aspect="auto", cmap=DIV, vmin=-2, vmax=2)
    ax.set_xlabel("pseudotime bin"); ax.set_ylabel(f"{Z.shape[0]} trajectory genes")
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    cb.set_label("row-scaled", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title("Trajectory-associated genes, ordered by peak", fontsize=6.5, loc="left")
    panel(ax, "d", dx=-0.05)

    ax = fig.add_subplot(gs[1, 2]); tidy(ax, "y")
    for p in sorted(patt.pattern.unique()):
        g = patt.loc[patt.pattern == p, "gene"]
        m = Z.loc[Z.index.intersection(g)].mean(0)
        ax.plot(range(len(m)), m.values, lw=1.2, color=OKABE_ITO[p % len(OKABE_ITO)],
                label=f"pattern {p+1} (n={len(g)})")
    ax.set_xlabel("pseudotime bin")
    ax.set_ylabel("mean scaled expression", labelpad=3)
    ax.legend(fontsize=4.8, loc="upper left", bbox_to_anchor=(0, 1.02))
    ax.set_title("expression patterns", fontsize=6.5, loc="left")
    panel(ax, "e", dx=-0.22)
    source_data(patt, RAW, "FigureR9e_trajectory_gene_patterns")
    save(fig, FIG, "FigureR9_revised_trajectory")


# =========================================================== FIGURE R10
def figure_R10():
    net = pd.read_csv(f"{RES}/13_interaction_counts.csv", index_col=0)
    act = pd.read_csv(f"{RES}/13_signalling_activity.csv", index_col=0)
    comm = R("13_communication_all.csv")
    T = R("13_macrophage_contrast_patient_level.csv")

    fig = plt.figure(figsize=(W2, 0.50 * W2))
    gs = fig.add_gridspec(1, 4, wspace=0.60, width_ratios=[1.25, 1, 1, 1])

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(net.values, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(net.columns)))
    ax.set_xticklabels(net.columns, rotation=90, fontsize=4.5)
    ax.set_yticks(range(len(net.index)))
    ax.set_yticklabels(net.index, fontsize=4.5)
    ax.set_xlabel("receiver"); ax.set_ylabel("sender")
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("significant interactions", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title("Interactions across the full atlas", fontsize=6.5, loc="left")
    panel(ax, "a", dx=-0.35)
    source_data(net.reset_index(), RAW, "FigureR10a_interaction_counts")

    ax = fig.add_subplot(gs[1]); tidy(ax)
    ax.scatter(act.outgoing, act.incoming, s=18, c="#0072B2", edgecolors="black", linewidths=.3)
    txt = [ax.text(r.outgoing, r.incoming, i, fontsize=4.5)
           for i, r in act.nlargest(8, "outgoing").iterrows()]
    adjust_text(txt, ax=ax, arrowprops=dict(arrowstyle="-", lw=.3, color="grey"), lim=150)
    ax.set_xlabel("outgoing signalling"); ax.set_ylabel("incoming signalling")
    ax.set_title("Fibroblasts and stroma\ndominate as senders", fontsize=6.5, loc="left")
    panel(ax, "b")

    ax = fig.add_subplot(gs[2]); tidy(ax, "x")
    top = comm.nlargest(14, "score")
    lab = top.ligand + "→" + top.receptor
    ax.barh(range(len(top))[::-1], top.score, color="#D55E00", edgecolor="black", lw=.3)
    ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels(lab, fontsize=4.8)
    ax.set_xlabel("interaction score")
    ax.set_title("Leading ligand–receptor pairs", fontsize=6.5, loc="left")
    panel(ax, "c", dx=-0.55)

    ax = fig.add_subplot(gs[3]); tidy(ax)
    T2 = T.dropna(subset=["log2_ratio", "p"]).copy()
    T2["y"] = -np.log10(T2.p)
    ax.scatter(T2.log2_ratio, T2.y, s=10, lw=.3, c=np.where(T2.p < 0.05, "#D55E00", "#BBBBBB"),
               edgecolors="black")
    ax.axhline(-np.log10(0.05), ls=":", lw=.6, c="grey"); ax.axvline(0, lw=.4, c="grey")
    txt = [ax.text(r.log2_ratio, r.y, r.pair, fontsize=4.2)
           for _, r in T2.nsmallest(6, "p").iterrows()]
    adjust_text(txt, ax=ax, arrowprops=dict(arrowstyle="-", lw=.3, color="grey"), lim=150)
    ax.set_xlabel("log$_2$ ratio (non-healing / healing)")
    ax.set_ylabel("$-\\log_{10}$ P (patients)")
    ax.set_title(f"Macrophage interactions,\npatient level: 0 at FDR < 0.05", fontsize=6.5, loc="left")
    panel(ax, "d")
    source_data(T, RAW, "FigureR10d_macrophage_contrast_patient_level")
    save(fig, FIG, "FigureR10_revised_communication")


# =========================================================== FIGURE R11
def figure_R11():
    v = pd.read_csv(f"{RES}/14_bulk_volcano.csv", index_col=0)
    conc = J("14_concordance_summary.json")
    C = pd.read_csv(f"{RES}/14_bulk_sc_concordance.csv", index_col=0)
    rep = R("14_representative_genes_bulk.csv")
    try:
        E = R("14_pathway_enrichment.csv")
    except Exception:
        E = pd.DataFrame()

    fig = plt.figure(figsize=(W2, 0.48 * W2))
    gs = fig.add_gridspec(1, 4, wspace=0.52)

    ax = fig.add_subplot(gs[0]); tidy(ax)
    d = v.dropna(subset=["pvalue", "log2FoldChange"]).copy(); d["y"] = -np.log10(d.pvalue)
    sig = d.padj < 0.05
    ax.scatter(d.loc[~sig, "log2FoldChange"], d.loc[~sig, "y"], s=1.5, lw=0, c="#DDDDDD", rasterized=True)
    ax.scatter(d.loc[sig, "log2FoldChange"], d.loc[sig, "y"], s=7, lw=.3, c="#D55E00",
               edgecolors="black", zorder=3)
    ax.set_xlim(-np.nanpercentile(d.log2FoldChange.abs(), 99.5),
                np.nanpercentile(d.log2FoldChange.abs(), 99.5))
    ax.set_ylim(0, d.y.max() * 1.1)
    txt = [ax.text(r.log2FoldChange, r.y, g, fontsize=4.5) for g, r in d.nsmallest(8, "pvalue").iterrows()]
    adjust_text(txt, ax=ax, arrowprops=dict(arrowstyle="-", lw=.3, color="grey"), lim=150)
    ax.axvline(0, lw=.4, c="grey")
    ax.set_xlabel("log$_2$ FC (non-healing / healing)"); ax.set_ylabel("$-\\log_{10}$ P")
    ax.set_title(f"GSE134431, PyDESeq2\n{int(sig.sum())} genes at FDR < 0.05", fontsize=6.5, loc="left")
    panel(ax, "a")

    ax = fig.add_subplot(gs[1]); tidy(ax, "x")
    if len(E):
        t = E[E.direction == "higher in healing"].nsmallest(10, "Adjusted P-value")
        lab = [x[:44] + ("…" if len(x) > 44 else "") for x in t.Term]
        ax.barh(range(len(t))[::-1], -np.log10(t["Adjusted P-value"]),
                color="#0072B2", edgecolor="black", lw=.3)
        ax.set_yticks(range(len(t))[::-1]); ax.set_yticklabels(lab, fontsize=4.5)
        ax.axvline(-np.log10(0.05), ls=":", lw=.6, c="red")
        ax.set_xlabel("$-\\log_{10}$ FDR")
        source_data(E, RAW, "FigureR11b_pathway_enrichment")
    ax.set_title("Pathways higher in healing", fontsize=6.5, loc="left")
    panel(ax, "b", dx=-0.85)

    ax = fig.add_subplot(gs[2]); tidy(ax)
    ax.scatter(C.bulk_log2FC, C.sc_log2FC, s=1.5, lw=0, c="#DDDDDD", rasterized=True)
    lead = C[C.leading]
    ax.scatter(lead.bulk_log2FC, lead.sc_log2FC, s=8, lw=.3, c="#0072B2", edgecolors="black")
    ax.axhline(0, lw=.4, c="grey"); ax.axvline(0, lw=.4, c="grey")
    lim = np.nanpercentile(np.abs(C[["bulk_log2FC", "sc_log2FC"]].values), 99)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("bulk log$_2$ FC"); ax.set_ylabel("macrophage pseudobulk log$_2$ FC")
    ax.set_title(f"Bulk vs single cell\n$\\rho$ = {conc['spearman_all']:.2f} "
                 f"(leading genes {conc['spearman_leading']:.2f})", fontsize=6.5, loc="left")
    panel(ax, "c")

    ax = fig.add_subplot(gs[3]); tidy(ax, "y")
    pal = {"myeloid activation": "#0072B2", "resident / complement": "#D55E00",
           "lipid metabolism": "#E69F00"}
    xs, cols, labs = [], [], []
    for i, (p, gsub) in enumerate(rep.groupby("panel", observed=True)):
        for _, r in gsub.iterrows():
            xs.append(r.log2FC); cols.append(pal.get(p, "#999")); labs.append(r.gene)
    order = np.argsort(xs)
    ax.barh(range(len(xs)), np.array(xs)[order], color=np.array(cols)[order],
            edgecolor="black", lw=.25)
    ax.set_yticks(range(len(xs))); ax.set_yticklabels(np.array(labs)[order], fontsize=4.2)
    ax.axvline(0, lw=.5, c="black")
    ax.set_xlabel("log$_2$ FC (non-healing / healing)")
    ax.legend(handles=[Patch(fc=c, ec="black", lw=.3, label=k) for k, c in pal.items()],
              fontsize=4.5, loc="lower left")
    ax.set_title("Myeloid signal is reduced\nin non-healing bulk tissue", fontsize=6.5, loc="left")
    panel(ax, "d", dx=-0.42)
    source_data(rep, RAW, "FigureR11d_representative_genes")
    save(fig, FIG, "FigureR11_revised_bulk_landscape")


# =========================================================== FIGURE R12
def figure_R12():
    bt = R("15_bulk_module_trait.csv"); st_ = R("15_sc_module_trait.csv")
    O = R("15_module_overlap.csv"); pan = R("15_consensus_panel_annotated.csv")
    s15 = J("15_summary.json")

    fig = plt.figure(figsize=(W2, 0.46 * W2))
    gs = fig.add_gridspec(1, 4, wspace=0.55)

    for i, (T, ttl, lt) in enumerate([(bt, "Bulk network", "a"), (st_, "Single-cell network", "b")]):
        ax = fig.add_subplot(gs[i]); tidy(ax, "x")
        T = T.sort_values("pearson_r")
        c = ["#D55E00" if p < 0.05 else "#BBBBBB" for p in T.p]
        ax.barh(range(len(T)), T.pearson_r, color=c, edgecolor="black", lw=.3)
        ax.set_yticks(range(len(T)))
        ax.set_yticklabels([f"{m} ({n})" for m, n in zip(T.module, T.n_genes)], fontsize=4.5)
        ax.axvline(0, lw=.5, c="black")
        ax.set_xlabel("correlation with non-healing")
        ax.set_title(f"{ttl}\n({len(T)} modules)", fontsize=6.5, loc="left")
        panel(ax, lt, dx=-0.55)
    source_data(bt, RAW, "FigureR12a_bulk_module_trait")
    source_data(st_, RAW, "FigureR12b_sc_module_trait")

    ax = fig.add_subplot(gs[2])
    M = O.pivot_table(index="bulk_module", columns="sc_module", values="jaccard", fill_value=0)
    im = ax.imshow(M.values, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(M.columns))); ax.set_xticklabels(M.columns, rotation=90, fontsize=4.5)
    ax.set_yticks(range(len(M.index))); ax.set_yticklabels(M.index, fontsize=4.5)
    P = O.pivot_table(index="bulk_module", columns="sc_module", values="p", fill_value=1)
    for a in range(len(M.index)):
        for b in range(len(M.columns)):
            if P.values[a, b] < 0.001:
                ax.text(b, a, "*", ha="center", va="center", fontsize=7, c="white")
    cb = fig.colorbar(im, ax=ax, pad=.02, fraction=.04)
    cb.set_label("Jaccard overlap", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title(f"Cross-modality overlap\n{s15['n_pairs_FDR05']} pair(s) at FDR < 0.05",
                 fontsize=6.5, loc="left")
    panel(ax, "c", dx=-0.30)
    source_data(O, RAW, "FigureR12c_module_overlap")

    ax = fig.add_subplot(gs[3]); tidy(ax, "y")
    cnt = pan.groupby("n_analyses").recovered_full_scale.agg(["sum", "size"])
    x = np.arange(len(cnt))
    ax.bar(x, cnt["size"], .6, color="#BBBBBB", edgecolor="black", lw=.3, label="panel genes")
    ax.bar(x, cnt["sum"], .6, color="#0072B2", edgecolor="black", lw=.3,
           label="recovered at full scale")
    ax.set_xticks(x); ax.set_xticklabels([f"≥{i}" for i in cnt.index], fontsize=5.5)
    ax.set_xlabel("analyses supporting the gene"); ax.set_ylabel("genes")
    ax.legend(fontsize=5)
    ax.set_title(f"{s15['panel_recovered']}/{s15['panel_size']} panel genes recovered",
                 fontsize=6.5, loc="left")
    panel(ax, "d")
    source_data(pan, RAW, "FigureR12d_consensus_panel_annotated")
    save(fig, FIG, "FigureR12_revised_networks")


# =========================================================== FIGURE R13
def figure_R13():
    perf = R("17_bulk_ml_performance.csv"); nulls = J("17_permutation_nulls.json")
    roc = R("17_bulk_roc.csv"); FI = R("17_bulk_feature_importance.csv")
    try:
        EXT = R("17_external_transfer.csv")
    except Exception:
        EXT = pd.DataFrame()

    fig = plt.figure(figsize=(W2, 0.48 * W2))
    gs = fig.add_gridspec(1, 4, wspace=0.55)

    ax = fig.add_subplot(gs[0]); tidy(ax, "y")
    sets = perf.feature_set.unique()
    x = np.arange(len(sets)); w = 0.26
    for k, (col, lab, c) in enumerate([("apparent_auc", "apparent", "#BBBBBB"),
                                       ("cv5_auc", "5-fold CV", "#56B4E9"),
                                       ("leaveoneout_auc", "leave-one-out", "#0072B2")]):
        vals = [perf[perf.feature_set == s][col].max() for s in sets]
        ax.bar(x + (k - 1) * w, vals, w, color=c, edgecolor="black", lw=.3, label=lab)
    ax.axhline(0.5, ls="--", lw=.6, c="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" (k=200 selected in-fold)", "").replace(" ", "\n") for s in sets],
                       fontsize=5)
    ax.set_ylabel("AUC"); ax.set_ylim(0, 1.15); ax.legend(fontsize=5, loc="lower left")
    ax.set_title("Apparent performance is\nlargely overfitting", fontsize=6.5, loc="left")
    panel(ax, "a", dx=-0.22)
    source_data(perf, RAW, "FigureR13a_bulk_ml_performance")

    ax = fig.add_subplot(gs[1]); tidy(ax, "y")
    ks = list(nulls)
    x = np.arange(len(ks))
    ax.bar(x, [nulls[k]["observed"] for k in ks], .5, color="#0072B2",
           edgecolor="black", lw=.3, label="observed")
    ax.errorbar(x, [nulls[k]["null_mean"] for k in ks],
                yerr=[nulls[k]["null_sd"] for k in ks], fmt="o", ms=3,
                color="black", lw=.8, capsize=2, label="permuted labels")
    for i, k in enumerate(ks):
        ax.text(i, nulls[k]["observed"] + .04, f"P={nulls[k]['p']:.3f}", ha="center", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace(" (k=200 selected in-fold)", "").replace(" ", "\n") for k in ks],
                       fontsize=5)
    ax.set_ylabel("leave-one-out AUC"); ax.set_ylim(0, 1.2); ax.legend(fontsize=5, loc="lower left")
    ax.set_title("Against a permutation null", fontsize=6.5, loc="left")
    panel(ax, "b", dx=-0.22)

    ax = fig.add_subplot(gs[2]); tidy(ax, "y")
    if len(EXT):
        x = np.arange(len(EXT)); w = 0.36
        ax.bar(x - w/2, EXT.internal_loo_auc, w, color="#0072B2", edgecolor="black", lw=.3,
               label="internal (GSE134431)")
        ax.bar(x + w/2, EXT.external_auc, w, color="#D55E00", edgecolor="black", lw=.3,
               label="external (GSE143735)")
        ax.axhline(0.5, ls="--", lw=.6, c="grey")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" (k=200 selected in-fold)", "").replace(" ", "\n")
                            for s in EXT.feature_set], fontsize=5)
        ax.set_ylabel("AUC"); ax.set_ylim(0, 1.15); ax.legend(fontsize=5, loc="lower left")
        source_data(EXT, RAW, "FigureR13c_external_transfer")
    ax.set_title("The panel was built using the\ninternal cohort; only the external\ntest is uncontaminated",
                 fontsize=6.2, loc="left")
    panel(ax, "c", dx=-0.22)

    ax = fig.add_subplot(gs[3]); tidy(ax, "x")
    t = FI.head(15).iloc[::-1]
    nz = t.l1_nonzero if "l1_nonzero" in t.columns else pd.Series(True, index=t.index)
    c = ["#D55E00" if v else "#0072B2" for v in nz]
    ax.barh(range(len(t)), t.importance, color=c, edgecolor="black", lw=.3)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.gene, fontsize=5)
    ax.set_xlabel("random-forest importance")
    n_nz = int(nz.sum()) if "l1_nonzero" in FI.columns else 0
    tot_nz = int(FI.l1_nonzero.sum()) if "l1_nonzero" in FI.columns else 0
    ax.legend(handles=[Patch(fc="#D55E00", ec="black", lw=.3, label="retained by L1"),
                       Patch(fc="#0072B2", ec="black", lw=.3, label="dropped by L1")],
              fontsize=5, loc="lower right")
    ax.set_title(f"Top predictive genes\n(L1 retains only {tot_nz} of {len(FI)})",
                 fontsize=6.5, loc="left")
    panel(ax, "d", dx=-0.40)
    source_data(FI, RAW, "FigureR13d_bulk_feature_importance")
    save(fig, FIG, "FigureR13_revised_bulk_ml")


# =========================================================== FIGURE R14 + R15
def figure_R14_R15():
    E = pd.read_csv(f"{RES}/18_prioritised_gene_expression.csv", index_col=0)
    per_pat = R("18_prioritised_per_patient.csv")
    G = R("18_prioritised_gene_tests.csv")
    by_state = pd.read_csv(f"{RES}/18_prioritised_by_state.csv", index_col=0)
    agg = R("18_perturbation_summary.csv")
    s18 = J("18_summary.json")
    top = s18["top_genes"]

    # ---- R14
    fig = plt.figure(figsize=(W2, 0.44 * W2))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.2, 1.1, 1])

    ax = fig.add_subplot(gs[0])
    Z = by_state[[g for g in top if g in by_state.columns]]
    Z = (Z - Z.mean()) / (Z.std(ddof=0) + 1e-9)
    im = ax.imshow(Z.values, cmap=DIV, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(Z.shape[1])); ax.set_xticklabels(Z.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(Z.shape[0])); ax.set_yticklabels(Z.index, fontsize=5)
    cb = fig.colorbar(im, ax=ax, pad=.02, fraction=.04)
    cb.set_label("z-scored mean expression", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    ax.set_title("Prioritised genes across\nmacrophage states", fontsize=6.5, loc="left")
    panel(ax, "a", dx=-0.30)
    source_data(by_state.reset_index(), RAW, "FigureR14a_prioritised_by_state")

    ax = fig.add_subplot(gs[1]); tidy(ax, "y")
    centres = []
    for i, g in enumerate(top[:6]):
        for j, grp in enumerate(["Healing", "Nonhealing"]):
            v = per_pat.loc[per_pat.group == grp, g].values
            x = i * 2.6 + j * .95
            ax.scatter(np.full(len(v), x) + np.random.RandomState(2).uniform(-.15, .15, len(v)),
                       v, s=7, c=GROUP[grp], edgecolors="black", linewidths=.2, zorder=3)
            ax.hlines(np.median(v), x - .33, x + .33, color="black", lw=1.1, zorder=4)
        centres.append(i * 2.6 + .475)
    pm = dict(zip(G.gene, G.p_patient_level))
    ax.set_xticks(centres)
    ax.set_xticklabels([f"{g}\nP={pm.get(g, np.nan):.2f}" for g in top[:6]], fontsize=5)
    ax.set_ylabel("mean expression per patient")
    ax.legend(handles=[Patch(fc=GROUP[g], ec="black", lw=.3, label=g)
                       for g in ["Healing", "Nonhealing"]], fontsize=5,
              loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2)
    ax.set_title("Patient-level expression", fontsize=6.5, loc="left")
    panel(ax, "b")
    source_data(per_pat, RAW, "FigureR14b_prioritised_per_patient")

    ax = fig.add_subplot(gs[2]); tidy(ax)
    ax.scatter(-np.log10(G.p_cell_level.clip(lower=1e-300)), -np.log10(G.p_patient_level),
               s=18, c="#0072B2", edgecolors="black", linewidths=.3)
    ax.axhline(-np.log10(0.05), ls=":", lw=.6, c="#D55E00")
    lim = (-np.log10(G.p_cell_level.clip(lower=1e-300))).max() * 1.08
    ax.set_xlim(0, lim); ax.set_ylim(0, max(2, (-np.log10(G.p_patient_level)).max() * 1.25))
    txt = [ax.text(-np.log10(max(r.p_cell_level, 1e-300)), -np.log10(r.p_patient_level),
                   r.gene, fontsize=4.5) for _, r in G.head(6).iterrows()]
    adjust_text(txt, ax=ax, arrowprops=dict(arrowstyle="-", lw=.3, color="grey"), lim=150)
    ax.set_xlabel("$-\\log_{10}$ P, cells"); ax.set_ylabel("$-\\log_{10}$ P, patients")
    ax.set_title(f"{int(s18['n_patient_level_sig'])} of {len(G)} significant\nat patient level",
                 fontsize=6.5, loc="left")
    panel(ax, "c")
    source_data(G, RAW, "FigureR14c_prioritised_gene_tests")
    save(fig, FIG, "FigureR14_revised_prioritised_genes")

    # ---- R15
    fig = plt.figure(figsize=(W15, 0.52 * W15))
    gs = fig.add_gridspec(1, 2, wspace=0.42)

    ax = fig.add_subplot(gs[0]); tidy(ax)
    c = np.where(agg.stable, np.where(agg.oe_mean > 0, "#D55E00", "#0072B2"), "#DDDDDD")
    ax.scatter(agg.ko_mean, agg.oe_mean, s=16, c=c, edgecolors="black", linewidths=.3)
    ax.axhline(0, lw=.4, c="grey"); ax.axvline(0, lw=.4, c="grey")
    sel = agg[agg.stable].reindex(agg[agg.stable].oe_mean.abs().sort_values(ascending=False).index).head(8)
    txt = [ax.text(r.ko_mean, r.oe_mean, r.gene, fontsize=4.5) for _, r in sel.iterrows()]
    adjust_text(txt, ax=ax, arrowprops=dict(arrowstyle="-", lw=.3, color="grey"), lim=150)
    ax.set_xlabel("Δ predicted non-healing probability, knockout")
    ax.set_ylabel("Δ predicted non-healing probability, over-expression")
    ax.legend(handles=[Patch(fc="#D55E00", ec="black", lw=.3, label="favours non-healing"),
                       Patch(fc="#0072B2", ec="black", lw=.3, label="favours healing"),
                       Patch(fc="#DDDDDD", ec="black", lw=.3, label="unstable across models")],
              fontsize=5, loc="upper left")
    ax.set_title("In-silico perturbation of the macrophage classifier", fontsize=6.5, loc="left")
    panel(ax, "a")

    ax = fig.add_subplot(gs[1]); tidy(ax, "x")
    sel = agg[agg.stable].copy()
    sel = pd.concat([sel.nsmallest(10, "oe_mean"), sel.nlargest(10, "oe_mean")]).drop_duplicates("gene")
    sel = sel.sort_values("oe_mean")
    ax.barh(range(len(sel)), sel.oe_mean, xerr=sel.oe_sd,
            color=np.where(sel.oe_mean > 0, "#D55E00", "#0072B2"),
            edgecolor="black", lw=.3, error_kw=dict(lw=.4, ecolor="#666666"))
    ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel.gene, fontsize=5)
    ax.axvline(0, lw=.5, c="black")
    ax.set_xlabel("Δ predicted non-healing probability on over-expression")
    ax.set_title(f"Ranked effect, mean ± SD over {s18['n_resamples']} models\n"
                 f"({s18['n_stable']}/{s18['n_panel_tested']} genes sign-stable)",
                 fontsize=6.5, loc="left")
    panel(ax, "b", dx=-0.30)
    source_data(agg, RAW, "FigureR15_perturbation_summary")
    save(fig, FIG, "FigureR15_revised_perturbation")


# =========================================================== FIGURE R16
def figure_R16():
    sex = pd.read_csv(f"{RES}/16_inferred_sex.csv", index_col=0)
    s16 = J("16_summary.json")
    hits = pd.read_csv(f"{RES}/16_sexlinked_in_pseudobulk.csv", index_col=0)
    cmp_ = pd.read_csv(f"{RES}/16_sex_adjustment_comparison.csv", index_col=0)
    tech = R("16_technical_covariates.csv")

    fig = plt.figure(figsize=(W2, 0.44 * W2))
    gs = fig.add_gridspec(1, 4, wspace=0.55)

    ax = fig.add_subplot(gs[0]); tidy(ax)
    ul = sex[sex.group.isin(["Healing", "Nonhealing"])]
    for g in ["Healing", "Nonhealing"]:
        s = ul[ul.group == g]
        ax.scatter(s.XIST, s.Y_genes, s=26, c=GROUP[g], edgecolors="black",
                   linewidths=.4, label=g, zorder=3)
    lim = max(ul.XIST.max(), ul.Y_genes.max()) * 1.1
    ax.plot([0, lim], [0, lim], ls="--", lw=.6, c="grey")
    ax.set_xlabel("XIST (log$_2$ CPM)"); ax.set_ylabel("Y-chromosome genes (log$_2$ CPM)")
    ax.legend(fontsize=5, loc="upper right")
    ax.set_title("Sex inferred from every library", fontsize=6.5, loc="left")
    panel(ax, "a")
    source_data(sex.reset_index(), RAW, "FigureR16a_inferred_sex")

    ax = fig.add_subplot(gs[1]); tidy(ax, "y")
    tab = pd.crosstab(ul.group, ul.inferred_sex).reindex(["Healing", "Nonhealing"])
    x = np.arange(len(tab)); bottom = np.zeros(len(tab))
    for i, c in enumerate(tab.columns):
        ax.bar(x, tab[c], .55, bottom=bottom, color=["#CC79A7", "#56B4E9"][i],
               edgecolor="black", lw=.3, label=c)
        for j, v in enumerate(tab[c]):
            if v: ax.text(j, bottom[j] + v/2, int(v), ha="center", va="center", fontsize=6)
        bottom += tab[c].values
    ax.set_xticks(x); ax.set_xticklabels(tab.index, fontsize=6)
    ax.set_ylabel("libraries"); ax.legend(fontsize=5)
    sig = "significantly " if s16["fisher_p"] < 0.05 else ""
    ax.set_title(f"Sex is unevenly distributed but not\n{sig}imbalanced "
                 f"(Fisher P = {s16['fisher_p']:.2f})", fontsize=6.5, loc="left")
    panel(ax, "b")

    ax = fig.add_subplot(gs[2]); tidy(ax, "x")
    h = hits.sort_values("pvalue", ascending=False)
    ax.barh(range(len(h)), -np.log10(h.pvalue),
            color=np.where(h.pvalue < 0.05, "#D55E00", "#BBBBBB"), edgecolor="black", lw=.3)
    ax.set_yticks(range(len(h))); ax.set_yticklabels(h.index, fontsize=5)
    ax.axvline(-np.log10(0.05), ls=":", lw=.6, c="red")
    ax.set_xlabel("$-\\log_{10}$ P in the macrophage contrast")
    ax.set_title("Sex-linked genes rank among\nthe leading hits", fontsize=6.5, loc="left")
    panel(ax, "c", dx=-0.35)
    source_data(hits.reset_index(), RAW, "FigureR16c_sexlinked_genes")

    ax = fig.add_subplot(gs[3]); tidy(ax)
    ax.scatter(-np.log10(cmp_.p_unadjusted), -np.log10(cmp_.p_sex_adjusted),
               s=1.5, lw=0, c="#BBBBBB", rasterized=True)
    m = cmp_.padj_unadjusted < 0.05
    ax.scatter(-np.log10(cmp_.loc[m, "p_unadjusted"]), -np.log10(cmp_.loc[m, "p_sex_adjusted"]),
               s=18, c="#D55E00", edgecolors="black", linewidths=.3, zorder=3,
               label="FDR < 0.05 before adjustment")
    lim = max((-np.log10(cmp_.p_unadjusted)).max(), (-np.log10(cmp_.p_sex_adjusted)).max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", lw=.6, c="grey")
    ax.set_xlabel("$-\\log_{10}$ P, group only"); ax.set_ylabel("$-\\log_{10}$ P, adjusted for sex")
    ax.legend(fontsize=5, loc="lower right")
    ax.set_title(f"After adjustment: {s16['n_FDR05_sex_adjusted']} genes\nat FDR < 0.05 "
                 f"(was {s16['n_FDR05_unadjusted']})", fontsize=6.5, loc="left")
    panel(ax, "d")
    source_data(tech, RAW, "FigureR16d_technical_covariates")
    save(fig, FIG, "FigureR16_confounder_audit")


for fn in (figure_R9, figure_R10, figure_R11, figure_R12, figure_R13,
           figure_R14_R15, figure_R16):
    print(f"\n>>> {fn.__name__}", flush=True)
    try:
        fn()
    except Exception as e:
        import traceback; traceback.print_exc()
print("\n=== STEP 19 COMPLETE ===", flush=True)
