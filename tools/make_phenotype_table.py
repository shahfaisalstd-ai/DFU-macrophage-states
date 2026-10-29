"""Rebuild data/metadata/GSE165816_phenotype.csv from the GEO series matrix and
check that it is identical to the copy used by the analysis.
Usage: python tools/make_phenotype_table.py [path/to/GSE165816_series_matrix.txt.gz]"""
import gzip, os, sys
import pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default = os.path.join(os.environ.get("DFU_ORIGINAL_DIR", os.path.join(REPO, "work", "original_pipeline")),
                       "00_inputs", "raw_data", "GSE165816_series_matrix.txt.gz")
path = sys.argv[1] if len(sys.argv) > 1 else default
rows = {}
with gzip.open(path, "rt", errors="ignore") as fh:
    for line in fh:
        if line.startswith("!Sample_"):
            parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")]
            rows.setdefault(parts[0], []).append(parts[1:])
chars = {v.split(":", 1)[0].strip(): [x.split(":", 1)[1].strip() for x in vals]
         for vals in rows["!Sample_characteristics_ch1"] for v in vals[:1]}
T = pd.DataFrame({"gsm": rows["!Sample_geo_accession"][0],
                  "gid": [t.split(":")[0] for t in rows["!Sample_title"][0]],
                  "tissue": pd.Series(chars["tissue"]).map({"Foot skin": "Foot", "Forearm skin": "Forearm", "PBMCs": "PBMC"}),
                  "condition": pd.Series(chars["disease"]).map({"DFU-healer": "Healing DFU", "DFU-nonhealer": "Non-healing DFU",
                                                               "Non-DFU Diabetic": "Diabetic without DFU", "Non-diabetic": "Healthy non-diabetic"})})
T["group"] = T.condition.map({"Healing DFU": "Healing", "Non-healing DFU": "Nonhealing",
                              "Diabetic without DFU": "Diabetic", "Healthy non-diabetic": "Healthy"})
assert T.notna().all().all(), "unmapped GEO characteristic"
ref = pd.read_csv(os.path.join(REPO, "data", "metadata", "GSE165816_phenotype.csv"))
same = T.set_index("gsm").sort_index().equals(ref.set_index("gsm").sort_index())
print(f"{len(T)} libraries; identical to data/metadata/GSE165816_phenotype.csv: {same}")
sys.exit(0 if same else 1)
