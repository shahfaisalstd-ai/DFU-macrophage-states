"""Minimal read-only .h5ad loader built on pyfive (pure-Python HDF5).
Used because h5py/anndata could not be installed in the revision environment.
pyfive is vendored in vendor/pyfive (upstream commit 7d2bfa7 + vendor/patches/pyfive_7d2bfa7_local.diff).
Reads X / raw.X (CSR), obs (incl. categoricals), var names and obsm arrays."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(os.path.dirname(HERE), "vendor")          # <repo>/vendor
sys.path[:0] = [f"{VENDOR}/stubs", f"{VENDOR}/pyfive"]
import numpy as np, pandas as pd, scipy.sparse as sp
import pyfive

def _s(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(a)])

def read_df(g):
    idx_key = g.attrs.get("_index", "_index")
    if isinstance(idx_key, bytes): idx_key = idx_key.decode()
    index = _s(g[idx_key][:])
    order = g.attrs.get("column-order", [])
    cols = {}
    for c in [x.decode() if isinstance(x, bytes) else x for x in order]:
        o = g[c]
        if hasattr(o, "keys"):  # categorical
            cats = _s(o["categories"][:]); codes = np.asarray(o["codes"][:])
            cols[c] = pd.Categorical.from_codes(codes, cats)
        else:
            v = o[:]
            cols[c] = _s(v) if v.dtype.kind in "OSU" else np.asarray(v)
    return pd.DataFrame(cols, index=index)

def read_csr(g, shape=None):
    data = np.asarray(g["data"][:]); ind = np.asarray(g["indices"][:]); ptr = np.asarray(g["indptr"][:])
    shp = tuple(g.attrs["shape"]) if shape is None else shape
    enc = g.attrs.get("encoding-type", b"csr_matrix")
    enc = enc.decode() if isinstance(enc, bytes) else enc
    M = sp.csr_matrix((data, ind, ptr), shape=shp) if "csr" in enc else sp.csc_matrix((data, ind, ptr), shape=shp).tocsr()
    return M

def open_h5ad(path):
    return pyfive.File(path)
