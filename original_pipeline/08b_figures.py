#!/usr/bin/env python
"""
Khan 2 revision -- Step 08b
Revised Figures R5-R8.

R5  patient-level versus cell-level inference   (Reviewer 1, comment 2)
R6  deconvolution accuracy and statistical power(Reviewer 2, comment 3)
R7  directional support versus statistical replication (Reviewer 2, comment 2)
R8  what the classifier does and does not predict(Reviewer 1, comment 4)
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from adjustText import adjust_text
from khan_style import W1, W15, W2, GROUP, OKABE_ITO, SEQ, DIV, panel, tidy, save, source_data

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES, FIG, RAW = f"{BASE}/02_results", f"{BASE}/03_figures", f"{BASE}/04_raw_data_by_figure"
os.makedirs(FIG, exist_ok=True); os.makedirs(RAW, exist_ok=True)
R = lambda f: pd.read_csv(f"{RES}/{f}")
J = lambda f: json.load(open(f"{RES}/{f}"))


# =========================================================== FIGURE R5
def figure_R5():
    pv = R("04_programme_patient_vs_cell_level.csv")
    deg = pd.read_csv(f"{RES}/04_pseudobulk_DESeq2_NHvsH.csv", index_col=0)
    comp = pd.read_csv(f"{RES}/04_macrophage_state_composition_per_sample.csv", index_col=0)

    fig = plt.figure(figsize=(W2, 0.48 * W2))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.05, 1.0, 1.0])

    # (a) the pseudoreplication inflation
    ax = fig.add_subplot(gs[0]); tidy(ax, "both")
    xc = -np.log10(pv.p_cell_level.clip(lower=1e-300))
    yp = -np.log10(pv.p_patient_level)
    ax.scatter(xc, yp, s=16, c="#0072B2", edgecolors="black", linewidths=0.3, zorder=3)
    lim = max(xc.max(), yp.max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", lw=0.6, c="grey", zorder=1)
    ax.axhline(-np.log10(0.05), ls=":", lw=0.6, c="#D55E00", zorder=2)
    ax.text(lim*0.98, -np.log10(0.05), " P = 0.05", fontsize=5, va="bottom", ha="right",
            color="#D55E00")
    ax.set_xlabel("$-\\log_{10}$ P, cells as replicates")
    ax.set_ylabel("$-\\log_{10}$ P, patients as replicates")
    # limits must be final before the labels are repelled, otherwise adjustText
    # places them against the autoscaled axes and they end up outside the panel
    ax.set_xlim(0, lim); ax.set_ylim(0, max(yp.max()*1.25, 2))
    txt = [ax.text(-np.log10(max(r.p_cell_level, 1e-300)), -np.log10(r.p_patient_level),
                   r.programme.replace("_", " "), fontsize=5)
           for _, r in pv.nsmallest(5, "p_patient_level").iterrows()]
    adjust_text(txt, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.3, color="grey"),
                expand_points=(1.3, 1.3), only_move={"text": "xy"},
                lim=200, avoid_self=False)
    ax.set_title("Treating cells as replicates inflates significance\n"
                 "by up to 220 orders of magnitude", fontsize=6.5, loc="left")
    panel(ax, "a")
    source_data(pv, RAW, "FigureR5a_patient_vs_cell_level_P")

    # (b) pseudobulk volcano
    ax = fig.add_subplot(gs[1]); tidy(ax)
    d = deg.dropna(subset=["pvalue", "log2FoldChange"]).copy()
    d["y"] = -np.log10(d.pvalue)
    sig = d.padj < 0.05
    ax.scatter(d.loc[~sig, "log2FoldChange"], d.loc[~sig, "y"], s=2, lw=0,
               c="#BBBBBB", rasterized=True)
    ax.scatter(d.loc[sig, "log2FoldChange"], d.loc[sig, "y"], s=8, lw=0.3,
               c="#D55E00", edgecolors="black", zorder=3)
    ax.axhline(-np.log10(0.05), ls=":", lw=0.6, c="grey")
    ax.axvline(0, ls="-", lw=0.4, c="grey")
    ax.set_xlim(*ax.get_xlim()); ax.set_ylim(0, d.y.max() * 1.12)
    txt = [ax.text(r.log2FoldChange, r.y, g, fontsize=5)
           for g, r in d.nsmallest(10, "pvalue").iterrows()]
    adjust_text(txt, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.3, color="grey"),
                expand_points=(1.3, 1.3), only_move={"text": "xy"},
                lim=200, avoid_self=False)
    ax.set_xlabel("log$_2$ fold change (non-healing / healing)")
    ax.set_ylabel("$-\\log_{10}$ P")
    ax.set_title(f"Macrophage pseudobulk, DESeq2\n"
                 f"n = 19 patients; {int(sig.sum())} gene(s) at FDR < 0.05",
                 fontsize=6.5, loc="left")
    panel(ax, "b")
    source_data(d.reset_index().rename(columns={"index": "gene"}).head(2000),
                RAW, "FigureR5b_pseudobulk_volcano")

    # (c) state composition per patient
    ax = fig.add_subplot(gs[2]); tidy(ax, "y")
    ct = R("04_state_composition_tests.csv")
    states = ct.state.tolist()
    centres = []
    for i, st in enumerate(states):
        for j, g in enumerate(["Healing", "Nonhealing"]):
            v = comp.loc[comp.group == g, st].dropna().values
            x = i * 2.6 + j * 0.95
            ax.scatter(np.full(len(v), x) + np.random.RandomState(2).uniform(-.16, .16, len(v)),
                       v, s=7, c=GROUP[g], edgecolors="black", linewidths=0.2, zorder=3)
            ax.hlines(np.median(v), x-0.34, x+0.34, color="black", lw=1.1, zorder=4)
        centres.append(i * 2.6 + 0.475)
    SHORT = {"Resident M2-like macrophage": "resident\nM2-like",
             "Lipid-associated macrophage": "lipid-\nassociated",
             "Inflammatory monocyte-macrophage": "inflammatory\nmonocyte"}
    ax.set_xticks(centres)
    ax.set_xticklabels([f"{SHORT.get(s, s)}\nP = {p:.2f}" for s, p in zip(ct.state, ct.p)],
                       fontsize=5)
    ax.set_ylabel("% of macrophages in the patient")
    ax.legend(handles=[Patch(fc=GROUP[g], ec="black", lw=.3, label=g)
                       for g in ["Healing", "Nonhealing"]],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("Macrophage state composition, one point per patient",
                 fontsize=6.5, loc="left")
    panel(ax, "c")
    source_data(comp.reset_index(), RAW, "FigureR5c_state_composition_per_patient")
    save(fig, FIG, "FigureR5_revised_patient_level_inference")


# =========================================================== FIGURE R6
def figure_R6():
    bench = R("05_benchmark_lambda.csv")
    refc  = R("05_reference_size_benchmark.csv")
    comb  = pd.read_csv(f"{RES}/05_deconvolution_combined.csv", index_col=0)
    power = J("05_power_analysis.json")
    conc  = J("05_reference_concordance.json")
    Pt = pd.read_csv(f"{RES}/05_benchmark_true_proportions.csv", index_col=0)
    Pe = pd.read_csv(f"{RES}/05_benchmark_estimated_full.csv", index_col=0)

    fig = plt.figure(figsize=(W2, 0.50 * W2))
    gs = fig.add_gridspec(1, 4, wspace=0.50)

    # (a) estimated vs true on known mixtures
    ax = fig.add_subplot(gs[0]); tidy(ax)
    cols = [c for c in Pt.columns if c in Pe.columns]
    for i, c in enumerate(cols):
        ax.scatter(Pt[c]*100, Pe[c]*100, s=2, lw=0, alpha=0.5,
                   c=OKABE_ITO[i % len(OKABE_ITO)], rasterized=True)
    m = max(Pt[cols].values.max(), Pe[cols].values.max())*100
    ax.plot([0, m], [0, m], ls="--", lw=0.6, c="black")
    ax.set_xlabel("true fraction (%)"); ax.set_ylabel("estimated fraction (%)")
    r = np.corrcoef(Pt[cols].values.ravel(), Pe[cols].values.ravel())[0, 1]
    ax.set_title(f"Benchmark on 200 mixtures\nof known composition (r = {r:.3f})",
                 fontsize=6.5, loc="left")
    panel(ax, "a")

    # (b) lambda selection under both benchmarks
    ax = fig.add_subplot(gs[1]); tidy(ax, "y")
    for bm, c, mk in [("matched platform", "#0072B2", "o"),
                      ("cross platform", "#D55E00", "s")]:
        b = bench[bench.benchmark == bm]
        if not len(b): continue
        ax.plot(b["lambda"], b.rmse, marker=mk, ms=3, lw=1, color=c, label=bm)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("ridge penalty $\\lambda$"); ax.set_ylabel("RMSE of estimated fractions")
    ax.legend(loc="best", fontsize=5.5)
    ax.set_title("Penalty chosen on a benchmark that\nreproduces platform mismatch",
                 fontsize=6.5, loc="left")
    panel(ax, "b")
    source_data(bench, RAW, "FigureR6b_lambda_benchmark")

    # (c) does reference size matter?
    ax = fig.add_subplot(gs[2]); tidy(ax, "y")
    x = np.arange(len(refc))
    ax.bar(x, refc.mac_pearson, 0.5, color=["#0072B2", "#BBBBBB"],
           edgecolor="black", lw=0.4)
    for i, (v, n) in enumerate(zip(refc.mac_pearson, refc.n_macrophages)):
        ax.text(i, v + 0.012, f"r = {v:.3f}\n{n:,} macrophages", ha="center", fontsize=5.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["full atlas\nreference", "original design\n(300 cells/library)"],
                       fontsize=5.5)
    ax.set_ylabel("macrophage-fraction accuracy (r)")
    ax.set_ylim(0, 1.16)
    ax.set_title("A four-fold larger reference does not\nchange deconvolution accuracy",
                 fontsize=6.5, loc="left")
    panel(ax, "c")
    source_data(refc, RAW, "FigureR6c_reference_size_benchmark")

    # (d) macrophage abundance with the detectable effect marked
    ax = fig.add_subplot(gs[3]); tidy(ax, "y")
    sub = comb[comb.group.isin(["Healing", "Nonhealing"])]
    for j, g in enumerate(["Healing", "Nonhealing"]):
        v = sub.loc[sub.group == g, "macrophage_marker_score"].values
        ax.scatter(np.full(len(v), j) + np.random.RandomState(3).uniform(-.12, .12, len(v)),
                   v, s=14, c=GROUP[g], edgecolors="black", linewidths=0.3, zorder=3)
        ax.hlines(np.median(v), j-0.26, j+0.26, color="black", lw=1.2, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Healing", "Non-healing"], fontsize=6)
    ax.set_ylabel("macrophage marker score\n(reference-free)")
    mw = pd.read_csv(f"{RES}/05_abundance_group_tests.csv")
    pm = float(mw.loc[mw.measure == "marker_score::Macrophage", "p_mannwhitney"].iloc[0])
    ax.set_title(f"Orthogonal abundance check\nP = {pm:.2f}; detectable only if d > "
                 f"{power['min_detectable_cohens_d']:.2f}", fontsize=6.5, loc="left")
    panel(ax, "d")
    source_data(sub.reset_index(), RAW, "FigureR6d_macrophage_abundance")
    save(fig, FIG, "FigureR6_revised_deconvolution_and_power")


# =========================================================== FIGURE R7
def figure_R7():
    conc = R("06_concordance_by_evidence_level.csv")
    joint = pd.read_csv(f"{RES}/06_discovery_vs_validation_joint.csv", index_col=0)
    var = pd.read_csv(f"{RES}/06_variance_site_vs_healing.csv", index_col=0)
    site = J("06_site_variance_summary.json")

    fig = plt.figure(figsize=(W2, 0.48 * W2))
    gs = fig.add_gridspec(1, 3, wspace=0.42, width_ratios=[1.25, 1.0, 0.95])

    # (a) directional vs statistical replication
    ax = fig.add_subplot(gs[0]); tidy(ax, "y")
    c = conc.copy()
    lab = [s.replace("consensus panel ", "panel ").replace(" analyses", " an.")
           for s in c.set]
    x = np.arange(len(c)); w = 0.38
    err = np.vstack([c.directional_concordance_pct - c.concordance_ci_low_pct,
                     c.concordance_ci_high_pct - c.directional_concordance_pct])
    ax.bar(x - w/2, c.directional_concordance_pct, w, yerr=err, capsize=1.6,
           color="#0072B2", edgecolor="black", lw=0.4, label="same direction",
           error_kw=dict(lw=0.5))
    ax.bar(x + w/2, c.statistically_replicated_pct, w, color="#D55E00",
           edgecolor="black", lw=0.4, label="same direction AND P < 0.05")
    ax.axhline(50, ls="--", lw=0.6, c="grey")
    ax.text(len(c)-0.4, 51, "chance", fontsize=5, color="grey", ha="right")
    for i, p in enumerate(c.binomial_p_vs_chance):
        if p < 0.05:
            ax.text(i - w/2, c.concordance_ci_high_pct.iloc[i] + 2.5,
                    "*" if p >= 1e-3 else "**", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=28, ha="right", fontsize=5)
    ax.set_ylabel("% of genes replicating in GSE143735")
    ax.set_ylim(0, 100); ax.legend(loc="upper left", fontsize=5.5)
    ax.set_title("Directional support is real; statistical replication is not",
                 fontsize=6.5, loc="left")
    panel(ax, "a", dx=-0.10)
    source_data(conc, RAW, "FigureR7a_concordance_by_evidence")

    # (b) discovery vs validation effect sizes
    ax = fig.add_subplot(gs[1]); tidy(ax)
    o = joint[~joint.in_panel]
    p_ = joint[joint.in_panel]
    ax.scatter(o.disc_log2FC, o.val_log2FC, s=1.2, lw=0, c="#DDDDDD", rasterized=True)
    ax.scatter(p_.disc_log2FC, p_.val_log2FC, s=8, lw=0.3, c="#0072B2",
               edgecolors="black", zorder=3, label="consensus panel")
    ax.axhline(0, lw=0.4, c="grey"); ax.axvline(0, lw=0.4, c="grey")
    lim = np.nanpercentile(np.abs(joint[["disc_log2FC", "val_log2FC"]].values), 99)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("log$_2$ FC, foot DFU (GSE134431)")
    ax.set_ylabel("log$_2$ FC, forearm DFU (GSE143735)")
    rho = float(conc.loc[conc.set.str.contains(">=3"), "spearman_rho"].iloc[0])
    ax.legend(loc="upper left", fontsize=5.5)
    ax.set_title(f"Panel genes cluster in the concordant\nquadrants ($\\rho$ = {rho:.2f})",
                 fontsize=6.5, loc="left")
    panel(ax, "b")

    # (c) anatomical site dominates the variance
    ax = fig.add_subplot(gs[2]); tidy(ax, "y")
    ax.hist(var.eta2_site, bins=40, color="#D55E00", alpha=0.75, label="anatomical site")
    ax.hist(var.eta2_healing, bins=40, color="#0072B2", alpha=0.75, label="healing status")
    ax.set_xlabel("variance explained ($\\eta^2$)"); ax.set_ylabel("genes")
    ax.legend(loc="upper right", fontsize=5.5)
    ax.set_title(f"Site explains {site['ratio']:.0f}x more variance than healing;\n"
                 f"{site['pct_genes_site_dominant']:.0f}% of genes site-dominant",
                 fontsize=6.5, loc="left")
    panel(ax, "c")
    source_data(var.reset_index(), RAW, "FigureR7c_variance_site_vs_healing")
    save(fig, FIG, "FigureR7_revised_external_validation")


# =========================================================== FIGURE R8
def figure_R8():
    s7  = J("07_summary.json")
    s7b = J("07b_summary.json")
    rep = R("07b_repeated_grouped_cv.csv")
    sens = R("07b_model_sensitivity.csv")
    roc_c = R("07_roc_percell.csv"); roc_p = R("07_roc_patient.csv")
    cur = R("07_learning_curve_patients.csv")

    fig = plt.figure(figsize=(W2, 0.48 * W2))
    gs = fig.add_gridspec(1, 4, wspace=0.55)

    # (a) AUC by evaluation scheme
    ax = fig.add_subplot(gs[0]); tidy(ax, "y")
    items = [("cell\nnaive", s7b["naive_ungrouped_auc"], "#BBBBBB"),
             ("cell\ngrouped", s7b["repeated_grouped_cv_mean"], "#0072B2"),
             ("cell\nLOPO", s7b["leave_one_patient_out_percell_auc"], "#0072B2"),
             ("patient\naggregated", s7b["leave_one_patient_out_patient_auc"], "#D55E00"),
             ("patient\npseudobulk", s7["pseudobulk_loo_auc"], "#D55E00")]
    x = np.arange(len(items))
    ax.bar(x, [i[1] for i in items], 0.62, color=[i[2] for i in items],
           edgecolor="black", lw=0.4)
    ax.errorbar(1, s7b["repeated_grouped_cv_mean"], yerr=s7b["repeated_grouped_cv_sd"],
                fmt="none", ecolor="black", lw=0.6, capsize=1.8)
    for i, (_, v, _) in enumerate(items):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=5.5)
    ax.axhline(0.5, ls="--", lw=0.6, c="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([i[0].replace("\n", " ") for i in items], rotation=38,
                       ha="right", fontsize=5)
    ax.set_ylabel("AUC"); ax.set_ylim(0, 1.16)
    ax.legend(handles=[Patch(fc="#0072B2", ec="black", lw=.3, label="cell state"),
                       Patch(fc="#D55E00", ec="black", lw=.3, label="patient outcome"),
                       Patch(fc="#BBBBBB", ec="black", lw=.3, label="leaky")],
              loc="lower left", fontsize=5, ncol=1)
    ax.set_title("Cell state is classifiable;\npatient outcome is not", fontsize=6.5, loc="left")
    panel(ax, "a", dx=-0.20, dy=1.20)
    source_data(pd.DataFrame(items, columns=["evaluation", "auc", "colour"]),
                RAW, "FigureR8a_auc_by_evaluation")

    # (b) ROC curves
    ax = fig.add_subplot(gs[1]); tidy(ax)
    ax.plot(roc_c.fpr, roc_c.tpr, lw=1.1, c="#0072B2",
            label=f"per cell (AUC {s7['percell_grouped_auc']:.2f})")
    ax.plot(roc_p.fpr, roc_p.tpr, lw=1.1, c="#D55E00",
            label=f"per patient (AUC {s7['patient_aggregated_auc']:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.6, c="grey")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.legend(loc="lower right", fontsize=5.5)
    ax.set_title("Same model,\ntwo questions", fontsize=6.5, loc="left")
    panel(ax, "b", dx=-0.18, dy=1.20)

    # (c) stability across resampling
    ax = fig.add_subplot(gs[2]); tidy(ax, "y")
    ax.hist(rep.auc, bins=12, color="#0072B2", edgecolor="black", lw=0.3)
    ax.axvline(rep.auc.mean(), color="black", lw=1)
    ax.axvline(0.88, color="#D55E00", lw=1, ls="--")
    ax.text(0.88, ax.get_ylim()[1]*0.92, " published\n 0.88", fontsize=5, color="#D55E00")
    ax.set_xlabel("AUC across 25 grouped-CV splits"); ax.set_ylabel("splits")
    ax.set_title(f"Per-cell AUC {rep.auc.mean():.2f} $\\pm$ {rep.auc.std():.2f}\n"
                 f"across 25 splits", fontsize=6.5, loc="left")
    panel(ax, "c")
    source_data(rep, RAW, "FigureR8c_repeated_grouped_cv")

    # (d) learning curve
    ax = fig.add_subplot(gs[3]); tidy(ax, "y")
    ax.errorbar(cur.n_train_patients, cur.mean_auc, yerr=cur.sd_auc, marker="o",
                ms=2.6, lw=1, capsize=1.6, color="#0072B2", ecolor="#999999",
                elinewidth=0.5)
    ax.axhline(0.5, ls="--", lw=0.6, c="grey")
    ax.set_xlabel("training patients"); ax.set_ylabel("per-cell AUC (held-out patients)")
    ax.set_title("Cohort size bounds\nwhat can be claimed", fontsize=6.5, loc="left")
    panel(ax, "d", dx=-0.20, dy=1.20)
    source_data(cur, RAW, "FigureR8d_learning_curve")
    save(fig, FIG, "FigureR8_revised_machine_learning")


for fn in (figure_R5, figure_R6, figure_R7, figure_R8):
    print(f"\n>>> {fn.__name__}", flush=True)
    fn()
print("\n=== STEP 08b COMPLETE ===", flush=True)
