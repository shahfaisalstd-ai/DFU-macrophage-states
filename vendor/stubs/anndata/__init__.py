"""Minimal in-memory stand-in for anndata.AnnData, sufficient for PyDESeq2 0.5.2
(DeseqDataSet / DeseqStats). Supports X, obs, var, layers, obsm, varm, uns,
column slicing obj[:, mask] and copy(). No file I/O."""
import numpy as np, pandas as pd, copy as _copy

class _AxisDict(dict):
    pass

class AnnData:
    def __init__(self, X=None, obs=None, var=None, layers=None, obsm=None, varm=None, uns=None):
        if isinstance(X, pd.DataFrame):
            obs_names, var_names, X = X.index, X.columns, X.values
        else:
            obs_names = obs.index if obs is not None else pd.RangeIndex(X.shape[0]).astype(str)
            var_names = var.index if var is not None else pd.RangeIndex(X.shape[1]).astype(str)
        self.X = np.asarray(X)
        self.obs = obs.copy() if obs is not None else pd.DataFrame(index=obs_names)
        self.obs.index = pd.Index(obs_names).astype(str)
        self.var = var.copy() if var is not None else pd.DataFrame(index=pd.Index(var_names).astype(str))
        self.layers = dict(layers or {}); self.obsm = dict(obsm or {}); self.varm = dict(varm or {}); self.uns = dict(uns or {})

    @property
    def n_obs(self): return self.X.shape[0]
    @property
    def n_vars(self): return self.X.shape[1]
    @property
    def obs_names(self): return self.obs.index
    @property
    def var_names(self): return self.var.index
    @property
    def shape(self): return self.X.shape

    def _col_index(self, key):
        if isinstance(key, (pd.Series, pd.Index)): key = key.values
        key = np.asarray(key)
        if key.dtype == bool: return np.where(key)[0]
        if key.dtype.kind in "OUS": return self.var.index.get_indexer(key)
        return key

    def __getitem__(self, idx):
        rows, cols = idx if isinstance(idx, tuple) else (idx, slice(None))
        if not (isinstance(rows, slice) and rows == slice(None)):
            raise NotImplementedError("row slicing not supported in the stub")
        ci = np.arange(self.n_vars) if (isinstance(cols, slice) and cols == slice(None)) else self._col_index(cols)
        new = AnnData.__new__(AnnData)
        new.X = self.X[:, ci]
        new.obs = self.obs.copy(); new.var = self.var.iloc[ci].copy()
        new.layers = {k: v[:, ci] for k, v in self.layers.items()}
        new.obsm = {k: _copy.deepcopy(v) for k, v in self.obsm.items()}
        new.varm = {k: (v.iloc[ci] if isinstance(v, (pd.DataFrame, pd.Series)) else np.asarray(v)[ci]) for k, v in self.varm.items()}
        new.uns = _copy.deepcopy(self.uns)
        return new

    def copy(self):
        return _copy.deepcopy(self)

    def to_df(self):
        return pd.DataFrame(self.X, index=self.obs_names, columns=self.var_names)
