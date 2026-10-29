"""
R2_00 -- export, from the stored AnnData objects of the original pipeline
(original pipeline 02_results/03_myeloid.h5ad and 12_macrophage_trajectory.h5ad), the
cell-level metadata used by the revision scripts:
  myeloid_obs_with_umap.csv        obs of the 7756 myeloid cells + UMAP coordinates
  trajectory_obs_with_umap.csv     4969 macrophage/monocyte cells: pseudotime + UMAP
  atlas_genes.csv                  the 25683 genes of raw.X (column order of raw counts)
  R2_00_myeloid_cells_for_raw_extraction.csv   cell list used by R2_02 (written to R2IN) on the GEO matrices
Uses the pure-Python reader h5ad_lite (pyfive) because h5py/anndata were not
installable in the revision environment.
Usage: python R2_00_export_h5ad_metadata.py   (paths from r2_common: H5AD_DIR -> AUX)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import h5ad_lite as H
from r2_common import AUX, H5AD_DIR, R2IN

f = H.open_h5ad(f"{H5AD_DIR}/03_myeloid.h5ad")
obs = H.read_df(f["obs"]); um = np.asarray(f["obsm"]["X_umap"][:])
obs["umap1"], obs["umap2"] = um[:, 0], um[:, 1]
obs.to_csv(f"{AUX}/myeloid_obs_with_umap.csv")
raw_var = H.read_df(f["raw"]["var"]) if "raw" in f else H.read_df(f["var"])
pd.DataFrame({"gene": raw_var.index}).to_csv(f"{AUX}/atlas_genes.csv", index=False)
cl = obs[["gid", "myeloid_subtype", "group", "tissue"]].copy()
cl["barcode"] = [c.split("_", 1)[1] for c in cl.index]          # cell id = <library>_<10x barcode>
cl.to_csv(f"{R2IN}/R2_00_myeloid_cells_for_raw_extraction.csv")   # read by R2_02 from <results folder>

g = H.open_h5ad(f"{H5AD_DIR}/12_macrophage_trajectory.h5ad")
t = H.read_df(g["obs"]); ut = np.asarray(g["obsm"]["X_umap"][:])
T = pd.DataFrame({"gsm": t.gsm, "gid": t.gid, "group": t.group, "myeloid_subtype": t.myeloid_subtype,
                  "pseudotime": t.pseudotime, "umap1": ut[:, 0], "umap2": ut[:, 1]}, index=t.index)
T.index.name = "cell"; T.to_csv(f"{AUX}/trajectory_obs_with_umap.csv")
print(len(obs), "myeloid cells;", len(T), "trajectory cells;", len(raw_var), "genes")
