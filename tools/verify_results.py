"""Compare a re-run of the revision analyses with the reference results in results/revision.

Every reference table (CSV, JSON, gzipped CSV) is compared with the re-run file of
the same name: first byte for byte (gzip compared after decompression), then, if the
bytes differ, numerically (relative tolerance --rtol). Figures are compared pixel by
pixel (PNG). Exit status 0 means every file matched.
Reference: results/revision (tables) and results/figures (figures and source data).
Usage: python tools/verify_results.py [rerun_folder] [--rtol 1e-6]
"""
import argparse, gzip, hashlib, io, json, os, sys
import numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("rerun", nargs="?", default=os.environ.get("R2OUT", os.path.join(REPO, "work", "revision_results")))
ap.add_argument("--rtol", type=float, default=1e-6)
a = ap.parse_args()

def raw(p):
    b = open(p, "rb").read()
    return gzip.decompress(b) if p.endswith(".gz") else b

def numeric_equal(p, q):
    if p.endswith(".json"):
        A, B = json.load(open(p)), json.load(open(q))
        def walk(x, y):
            if isinstance(x, dict): return isinstance(y, dict) and x.keys() == y.keys() and all(walk(x[k], y[k]) for k in x)
            if isinstance(x, list): return isinstance(y, list) and len(x) == len(y) and all(walk(i, j) for i, j in zip(x, y))
            if isinstance(x, (int, float)) and isinstance(y, (int, float)): return bool(np.isclose(x, y, rtol=a.rtol, atol=1e-12, equal_nan=True))
            return x == y
        return walk(A, B)
    A = pd.read_csv(io.BytesIO(raw(p)), low_memory=False); B = pd.read_csv(io.BytesIO(raw(q)), low_memory=False)
    if A.shape != B.shape or list(A.columns) != list(B.columns): return False
    for c in A.columns:
        x, y = A[c], B[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.astype(float), y.astype(float), rtol=a.rtol, atol=1e-12, equal_nan=True): return False
        elif not x.astype(str).equals(y.astype(str)): return False
    return True

def png_equal(p, q):
    from matplotlib.image import imread
    x, y = imread(p), imread(q)
    return x.shape == y.shape and np.array_equal(x, y)

rows = []
pairs = [(os.path.join(REPO, "results", "revision"), a.rerun, ""),
         (os.path.join(REPO, "results", "figures"), os.path.join(a.rerun, "figures"), "figures/")]
for refdir, newdir, tag in pairs:
  for root, _, files in os.walk(refdir):
    for f in sorted(files):
        ref = os.path.join(root, f); rel = tag + os.path.relpath(ref, refdir); new = os.path.join(newdir, os.path.relpath(ref, refdir))
        if not f.endswith((".csv", ".json", ".gz", ".png")) or "/logs/" in ref: continue
        if f.startswith("Figure3"): continue            # Figure 3 is drawn by the original pipeline (08a_figures.py)
        if not os.path.exists(new): rows.append((rel, "MISSING")); continue
        if f.endswith(".png"): rows.append((rel, "identical" if png_equal(ref, new) else "DIFFERENT")); continue
        if raw(ref) == raw(new): rows.append((rel, "identical")); continue
        rows.append((rel, f"equal within rtol={a.rtol}" if numeric_equal(ref, new) else "DIFFERENT"))
R = pd.DataFrame(rows, columns=["file", "status"])
bad = R[~R.status.str.startswith(("identical", "equal"))]
print(R.status.value_counts().to_string())
if len(bad): print(bad.to_string(index=False))
print("ALL FILES MATCH" if bad.empty else f"{len(bad)} file(s) do not match")
sys.exit(0 if bad.empty else 1)
