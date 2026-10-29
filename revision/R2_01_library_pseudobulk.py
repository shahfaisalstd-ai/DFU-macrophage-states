#!/usr/bin/env python
"""
Revision 2 -- step R2_01 (runs on the raw GEO matrices, low memory)
For every one of the 54 GSE165816 libraries: per-cell QC statistics, whole-
library pseudobulk counts (all cells), and the list of cell barcodes.
Used for: the sample-level table (Table S1), the cell-count reconciliation
(Table S2), donor-structure (genotype-proxy) analysis and chemistry inference.
Usage: python R2_01_library_pseudobulk.py <GSE165816 csv.gz folder> <out folder>
"""
import os, sys, gzip, time, numpy as np, pandas as pd
T0 = time.time(); BUDGET = float(os.environ.get("R2_BUDGET", "1e9"))  # seconds; lets the job run in resumable slices
raw, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
files = sorted(f for f in os.listdir(raw) if f.endswith("counts.csv.gz"))
pb, cellqc = {}, []
done = set()
if os.path.exists(f"{out}/R2_01_library_pseudobulk_allcells.csv"):
    prev = pd.read_csv(f"{out}/R2_01_library_pseudobulk_allcells.csv", index_col=0)
    pb = {c: prev[c] for c in prev.columns}; done = set(prev.columns)
    cellqc = [pd.read_csv(f"{out}/R2_01_cell_qc.csv.gz")]
for f in files:
    gsm, gid = f.split("_")[0], f.split("_")[1].replace("counts.csv.gz", "")
    if gid in done: continue
    if time.time() - T0 > BUDGET: print("budget reached; re-run to resume", flush=True); break
    with gzip.open(f"{raw}/{f}", "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
    barcodes = header[1:] if header[0] == "" else header
    ncell = len(barcodes)
    nCount = np.zeros(ncell); nFeat = np.zeros(ncell); mt = np.zeros(ncell)
    sums = []
    for ch in pd.read_csv(f"{raw}/{f}", index_col=0, chunksize=1500, dtype={b: np.int32 for b in barcodes} if False else None):
        v = ch.values
        nCount += v.sum(0); nFeat += (v > 0).sum(0)
        ism = ch.index.astype(str).str.upper().str.startswith("MT-")
        if ism.any(): mt += v[ism].sum(0)
        sums.append(pd.Series(v.sum(1), index=ch.index.astype(str)))
    s = pd.concat(sums); s = s.groupby(level=0).sum()
    pb[gid] = s
    pct = np.where(nCount > 0, mt / np.maximum(nCount, 1) * 100, 0)
    cellqc.append(pd.DataFrame(dict(gsm=gsm, gid=gid, barcode=barcodes, nCount=nCount.astype(int),
                                    nFeature=nFeat.astype(int), pctMT=np.round(pct, 3))))
    P = pd.DataFrame(pb).fillna(0).astype(np.int64)
    P.to_csv(f"{out}/R2_01_library_pseudobulk_allcells.csv.tmp")           # atomic writes
    pd.concat(cellqc).to_csv(f"{out}/R2_01_cell_qc.tmp.csv.gz", index=False)
    os.replace(f"{out}/R2_01_library_pseudobulk_allcells.csv.tmp", f"{out}/R2_01_library_pseudobulk_allcells.csv")
    os.replace(f"{out}/R2_01_cell_qc.tmp.csv.gz", f"{out}/R2_01_cell_qc.csv.gz")
    print(gid, ncell, flush=True)
print("DONE" if len(pb) == len(files) else "PARTIAL", len(pb), "of", len(files), flush=True)
