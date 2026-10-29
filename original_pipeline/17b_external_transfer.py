#!/usr/bin/env python
"""
Khan 2 revision -- Step 17b
External transfer test for the bulk classifier.

The consensus panel was assembled partly FROM the GSE134431 cohort, so its
leave-one-out performance on that same cohort is optimistic no matter how the
folds are drawn: the labels of those samples informed which genes entered the
panel in the first place. The only uncontaminated test is a different cohort,
so each classifier is trained on all of GSE134431 and applied, untouched, to
the forearm cohort GSE143735.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from khan_bulk import load_counts, dfu_groups, cpm_log, gene_annotation

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline"))
RES = f"{BASE}/02_results"
SEED = 123

ann = gene_annotation()
C1, m1 = load_counts("GSE134431", ann); g1 = dfu_groups("GSE134431", m1)
u1 = g1[g1.isin(["Healing", "Nonhealing"])].index
L1 = cpm_log(C1)[u1].T; y1 = (g1.loc[u1] == "Nonhealing").astype(int).values

C2, m2 = load_counts("GSE143735", ann); g2 = dfu_groups("GSE143735", m2)
u2 = g2[g2.isin(["Healing", "Nonhealing"])].index
L2 = cpm_log(C2)[u2].T; y2 = (g2.loc[u2] == "Nonhealing").astype(int).values
print(f"train GSE134431 n={len(y1)} ({y1.sum()} NH) -> test GSE143735 n={len(y2)} ({y2.sum()} NH)",
      flush=True)

pan = pd.read_csv(os.environ.get("PANEL_CSV", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "metadata", "consensus_novel_panel.csv")))
PANEL = [x for x in pan.loc[pan.n_analyses >= 3, "gene"] if x in L1.columns]
CANON = [x for x in ["COL1A1","COL3A1","FN1","VEGFA","TGFB1","MMP1","MMP2","MMP9","TIMP1",
                     "IL1B","IL6","TNF","CXCL8","PDGFB","FGF2","EGF","HIF1A","ACTA2",
                     "CD68","ARG1","NOS2","MRC1","CCL2","SPP1","LOX","ELN","KRT14","KRT10"]
         if x in L1.columns]
perf = pd.read_csv(f"{RES}/17_bulk_ml_performance.csv")
SETS = {"all genes (k=200 selected in-fold)": (L1, 200),
        "consensus panel": (L1[PANEL], None),
        "canonical wound genes": (L1[CANON], None)}

rows = []
for label, (Xtr, k) in SETS.items():
    cols = [c for c in Xtr.columns if c in L2.columns]
    if len(cols) < 5:
        continue
    steps = [StandardScaler()]
    if k: steps.append(SelectKBest(f_classif, k=min(k, len(cols))))
    steps.append(LogisticRegression(penalty="l1", C=0.5, solver="liblinear",
                                    max_iter=5000, random_state=SEED))
    m = make_pipeline(*steps).fit(Xtr[cols].values, y1)
    auc = roc_auc_score(y2, m.predict_proba(L2[cols].values)[:, 1])
    internal = float(perf[(perf.feature_set == label) &
                          (perf.model == "L1 logistic")].leaveoneout_auc.iloc[0])
    rows.append(dict(feature_set=label, n_shared_genes=len(cols),
                     internal_loo_auc=internal, external_auc=float(auc)))
    print(f"  {label:34s} shared {len(cols):5d}  internal LOO {internal:.2f}  "
          f"->  external {auc:.2f}", flush=True)

EXT = pd.DataFrame(rows)
EXT.to_csv(f"{RES}/17_external_transfer.csv", index=False)
s = json.load(open(f"{RES}/17_summary.json"))
s["external_transfer"] = EXT.to_dict("records")
json.dump(s, open(f"{RES}/17_summary.json", "w"), indent=2)
print("=== STEP 17b COMPLETE ===")
