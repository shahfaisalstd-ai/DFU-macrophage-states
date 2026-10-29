#!/usr/bin/env python
"""
Revision 2 -- step R2_02 (runs on the raw GEO matrices, low memory, resumable)
Rebuilds the macrophage pseudobulk from RAW UMI counts.

Audit finding: the previous revision summed log1p-normalised values (the h5ad
had no raw-count layer) and passed those sums to DESeq2. Here, for every
library, the raw GEO counts of the cells annotated as myeloid (cell list from
03_myeloid.h5ad, exported as R2_00_myeloid_cells_for_raw_extraction.csv) are
summed per macrophage state and for the macrophage/monocyte compartment.
Output: R2_02_raw_pseudobulk_<unit>.csv (genes x libraries) for unit in
macrophage, resident, inflammatory, lipid.
Usage: python R2_02_myeloid_raw_pseudobulk.py <GSE165816 folder> <results folder>
"""
import os, sys, time, gzip, numpy as np, pandas as pd
raw, out = sys.argv[1], sys.argv[2]
T0 = time.time(); BUDGET = float(os.environ.get("R2_BUDGET", "1e9"))
cells = pd.read_csv(f"{out}/R2_00_myeloid_cells_for_raw_extraction.csv", index_col=0)
STATE = {"Resident M2-like macrophage": "resident",
         "Inflammatory monocyte-macrophage": "inflammatory",
         "Lipid-associated macrophage": "lipid"}
cells["unit"] = cells.myeloid_subtype.map(STATE)          # NaN for DC / LC / proliferating
files = {f.split("_")[1].replace("counts.csv.gz", ""): f for f in os.listdir(raw) if f.endswith("counts.csv.gz")}
units = ["macrophage", "resident", "inflammatory", "lipid"]
state_file = f"{out}/R2_02_progress.txt"
done = set(open(state_file).read().split()) if os.path.exists(state_file) else set()
acc = {u: (pd.read_csv(f"{out}/R2_02_raw_pseudobulk_{u}.csv", index_col=0) if done else None) for u in units}
acc = {u: ({c: v[c] for c in v.columns} if v is not None else {}) for u, v in acc.items()}
for gid in sorted(cells.gid.unique()):
    if gid in done: continue
    if time.time() - T0 > BUDGET: print("budget reached; re-run to resume", flush=True); break
    sub = cells[(cells.gid == gid) & cells.unit.notna()]
    with gzip.open(f"{raw}/{files[gid]}", "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
    bcs = header[1:] if header[0] == "" else header
    pos = {b: i for i, b in enumerate(bcs)}
    want = {u: np.array([pos[b] for b in sub.barcode[(sub.unit == u) if u != "macrophage" else slice(None)]], dtype=int)
            for u in units}
    missing = [b for b in sub.barcode if b not in pos]
    sums = {u: [] for u in units}
    for ch in pd.read_csv(f"{raw}/{files[gid]}", index_col=0, chunksize=2000):
        v = ch.values
        for u in units:
            sums[u].append(pd.Series(v[:, want[u]].sum(1) if len(want[u]) else np.zeros(len(ch), dtype=int),
                                     index=ch.index.astype(str)))
    for u in units:
        acc[u][gid] = pd.concat(sums[u]).groupby(level=0).sum()
    for u in units:
        P = pd.DataFrame(acc[u]).fillna(0).astype(np.int64)
        P.to_csv(f"{out}/R2_02_raw_pseudobulk_{u}.csv.tmp"); os.replace(f"{out}/R2_02_raw_pseudobulk_{u}.csv.tmp", f"{out}/R2_02_raw_pseudobulk_{u}.csv")
    done.add(gid); open(state_file, "w").write("\n".join(sorted(done)))
    print(gid, {u: len(want[u]) for u in units}, "missing", len(missing), flush=True)
print("DONE" if len(done) == cells.gid.nunique() else "PARTIAL", len(done), "of", cells.gid.nunique(), flush=True)
