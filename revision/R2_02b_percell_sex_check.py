"""R2_02b: per-cell XIST / Y-gene check for libraries whose whole-library profile
shows both XIST and Y transcripts (audit of possible multi-donor content).
Usage: python R2_02b_percell_sex_check.py <GSE165816 folder> <results folder> G2 G13 G33 G9 ..."""
import sys, os, pandas as pd, numpy as np
raw, out, libs = sys.argv[1], sys.argv[2], sys.argv[3:]
Y = ["RPS4Y1", "DDX3Y", "KDM5D", "UTY", "EIF1AY", "USP9Y", "ZFY"]
files = {f.split("_")[1].replace("counts.csv.gz", ""): f for f in os.listdir(raw) if f.endswith("counts.csv.gz")}
qc = pd.read_csv(f"{out}/R2_01_cell_qc.csv.gz")
rows = []; flagged = []
for gid in libs:
    keep = []
    for ch in pd.read_csv(f"{raw}/{files[gid]}", index_col=0, chunksize=3000):
        keep.append(ch.loc[ch.index.intersection(["XIST"] + Y)])
    d = pd.concat(keep)
    x = d.loc["XIST"] if "XIST" in d.index else pd.Series(0, index=d.columns)
    y = d.loc[d.index.intersection(Y)].sum()
    q = qc[qc.gid == gid].set_index("barcode")
    ok = (q.nFeature >= 200) & (q.nFeature <= 5000) & (q.pctMT < 10)
    x, y = x[ok[ok].index], y[ok[ok].index]
    rows.append(dict(gid=gid, n_qc_cells=int(ok.sum()),
                     xist_only=int(((x > 0) & (y == 0)).sum()), y_only=int(((y > 0) & (x == 0)).sum()),
                     both=int(((x > 0) & (y > 0)).sum()), neither=int(((x == 0) & (y == 0)).sum()),
                     pct_xist_pos=round(100 * float((x > 0).mean()), 1), pct_y_pos=round(100 * float((y > 0).mean()), 1)))
    print(rows[-1], flush=True)
    # cells carrying the opposite-sex signal of the library majority
    maj_male = (y > 0).mean() > (x > 0).mean()
    odd = x.index[(x > 0) & (y == 0)] if maj_male else y.index[(y > 0) & (x == 0)]
    flagged += [dict(gid=gid, barcode=b, signal="XIST-only in male library" if maj_male else "Y-only in female library") for b in odd]
pd.DataFrame(rows).to_csv(f"{out}/R2_02b_percell_sex_check.csv", index=False)
pd.DataFrame(flagged).to_csv(f"{out}/R2_02b_opposite_sex_cells.csv", index=False)
