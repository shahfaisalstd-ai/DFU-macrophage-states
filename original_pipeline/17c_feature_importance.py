#!/usr/bin/env python
"""
Khan 2 revision -- Step 17c
Feature importances for the bulk classifier.

The L1-penalised model retains only three non-zero coefficients on the consensus
panel, which is itself informative but makes a poor ranking. Importances are
therefore also computed from a random forest, which scores every gene, and both
are saved so the sparsity of the L1 fit is visible rather than hidden.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from khan_bulk import load_counts, dfu_groups, cpm_log, gene_annotation

BASE = os.environ.get("DFU_ORIGINAL_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "original_pipeline")); RES = f"{BASE}/02_results"
SEED = 123
ann = gene_annotation()
C, meta = load_counts("GSE134431", ann); g = dfu_groups("GSE134431", meta)
u = g[g.isin(["Healing", "Nonhealing"])].index
L = cpm_log(C)[u].T; y = (g.loc[u] == "Nonhealing").astype(int).values

pan = pd.read_csv(os.environ.get("PANEL_CSV", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "metadata", "consensus_novel_panel.csv")))
PANEL = [x for x in pan.loc[pan.n_analyses >= 3, "gene"] if x in L.columns]
X = L[PANEL]

rf = make_pipeline(StandardScaler(),
                   RandomForestClassifier(n_estimators=2000, random_state=SEED,
                                          n_jobs=8)).fit(X.values, y)
l1 = make_pipeline(StandardScaler(),
                   LogisticRegression(penalty="l1", C=0.5, solver="liblinear",
                                      max_iter=5000, random_state=SEED)).fit(X.values, y)
FI = pd.DataFrame({
    "gene": PANEL,
    "rf_importance": rf.named_steps["randomforestclassifier"].feature_importances_,
    "l1_coefficient": l1.named_steps["logisticregression"].coef_[0]})
FI["importance"] = FI.rf_importance
FI["in_consensus_panel"] = True
FI["in_canonical_set"] = False
FI["l1_nonzero"] = FI.l1_coefficient != 0
FI = FI.sort_values("rf_importance", ascending=False)
FI.to_csv(f"{RES}/17_bulk_feature_importance.csv", index=False)
print(f"consensus panel genes: {len(PANEL)}; "
      f"non-zero L1 coefficients: {int(FI.l1_nonzero.sum())}")
print(FI.head(15).to_string(index=False))
