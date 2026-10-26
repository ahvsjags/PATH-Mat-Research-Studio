#!/usr/bin/env python3
"""Reproduce the V238 DOI-held-out within-formulation ZeoSyn analysis.

The published dataset is source-adapted and does not evaluate frozen PATH-Mat.
All model settings, folds, endpoints and cluster bootstrap seeds are fixed here.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs" / "ZEOSYN.xlsx"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 20261016
BOOT = 2000

df = pd.read_excel(INPUT, sheet_name="Sheet1")
df = df.loc[df["doi"].notna() & df["normed"].astype(str).str.lower().eq("yes")].copy()
prod = df["product1"].astype(str).str.strip().str.lower()
code_present = df["Code1"].notna() & df["Code1"].astype(str).str.strip().ne("")
y0 = np.where(code_present, 1, np.where(prod.isin({"amorphous", "dense"}), 0, -1))
df = df.loc[y0 >= 0].copy().reset_index(drop=True)
df["y"] = y0[y0 >= 0].astype(int)
n_endpoint_rows = int(len(df))

# Exact reported gel composition and solvent/OSDA identity form the matched cell.
chem_num = [c for c in df.columns[1:52] if c != "solvent"]
chem_cat = ["solvent", "osda1 smiles", "osda2 smiles", "osda3 smiles"]
chem = chem_num + chem_cat
sig = df[chem].astype("object").where(df[chem].notna(), "__MISSING__").apply(lambda r: tuple(str(x) for x in r), axis=1)
df["_cell"] = df["doi"].astype(str) + "|" + sig.astype(str)
cell_n = df.groupby("_cell").size()
df = df.loc[df["_cell"].map(cell_n) >= 2].copy().reset_index(drop=True)

process = [c for c in ["aging_time", "aging_temp", "cryst_time", "cryst_temp", "react_vol", "seed"] if c in df and pd.api.types.is_numeric_dtype(df[c])]
for c in process:
    df[c + "_within"] = df[c] - df.groupby("_cell")[c].transform("mean")
df["_coverage"] = df[process].notna().sum(axis=1)
df["_coverage_within"] = df["_coverage"] - df.groupby("_cell")["_coverage"].transform("mean")
features = {
    "reporting_only": ["_coverage_within"],
    "process_only": [c + "_within" for c in process],
    "process_plus_reporting": [c + "_within" for c in process] + ["_coverage_within"],
}
y = df["y"].to_numpy()
groups = df["doi"].astype(str).to_numpy()
preds = {name: np.full(len(df), np.nan) for name in features}
cv = GroupKFold(n_splits=5)
for train, test in cv.split(df, y, groups):
    assert set(groups[train]).isdisjoint(set(groups[test]))
    for name, cols in features.items():
        model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=SEED),
        )
        model.fit(df.iloc[train][cols], y[train])
        preds[name][test] = model.predict_proba(df.iloc[test][cols])[:, 1]

cells = []
for cell, g in df.groupby("_cell", sort=False):
    if g["y"].nunique() < 2:
        continue
    pos = g.index[g["y"] == 1].to_numpy()
    neg = g.index[g["y"] == 0].to_numpy()
    n_pairs = len(pos) * len(neg)
    rec = {"cell": cell, "doi": str(g["doi"].iloc[0]), "n_rows": len(g), "n_positive_negative_pairs": n_pairs}
    for name, p in preds.items():
        delta = p[pos, None] - p[neg][None, :]
        rec[name] = float((np.sum(delta > 1e-12) + 0.5 * np.sum(np.abs(delta) <= 1e-12)) / n_pairs)
    cells.append(rec)
cell_df = pd.DataFrame(cells)
unique_doi = cell_df["doi"].unique()
by_doi = {doi: cell_df.loc[cell_df["doi"] == doi] for doi in unique_doi}
rng = np.random.default_rng(SEED)
boot = {k: [] for k in ["reporting_only", "process_only", "process_plus_reporting", "process_minus_reporting", "combined_minus_reporting", "combined_minus_process"]}
for _ in range(BOOT):
    draw = rng.choice(unique_doi, size=len(unique_doi), replace=True)
    sample = pd.concat([by_doi[doi] for doi in draw], ignore_index=True)
    means = {name: float(np.average(sample[name], weights=sample["n_positive_negative_pairs"])) for name in features}
    boot["reporting_only"].append(means["reporting_only"])
    boot["process_only"].append(means["process_only"])
    boot["process_plus_reporting"].append(means["process_plus_reporting"])
    boot["process_minus_reporting"].append(means["process_only"] - means["reporting_only"])
    boot["combined_minus_reporting"].append(means["process_plus_reporting"] - means["reporting_only"])
    boot["combined_minus_process"].append(means["process_plus_reporting"] - means["process_only"])

out = {
    "schema": "v238_zeosyn_within_formulation_doi_heldout",
    "source": "ZeoSyn public dataset; official repository; DOI-grouped source-adapted analysis",
    "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    "endpoint": "positive: standardized Code1 IZA framework present; negative: product1 explicitly amorphous or dense; ambiguous outcomes excluded",
    "cohort": {"n_endpoint_rows_before_repeated_cell_filter": n_endpoint_rows, "n_rows_in_repeated_cells": int(len(df)), "n_mixed_doi_formulation_cells": int(len(cell_df)), "n_mixed_dois": int(cell_df["doi"].nunique()), "n_positive_negative_pairs": int(cell_df["n_positive_negative_pairs"].sum())},
    "features": {"chemical_cell": chem, "process_centered": process, "reporting_control": "within-cell centered count of nonmissing process features"},
    "model": "L2 logistic regression C=1.0; five-fold GroupKFold by DOI; all features centered/imputed without outcome access; no hyperparameter selection",
    "pair_weighted_concordance": {name: float(np.average(cell_df[name], weights=cell_df["n_positive_negative_pairs"])) for name in features},
    "macro_cell_concordance": {name: float(cell_df[name].mean()) for name in features},
    "doi_cluster_bootstrap": {"draws": BOOT, "seed": SEED, "unit": "DOI", "ci95_percentile": {name: [float(x) for x in np.quantile(values, [0.025, 0.975])] for name, values in boot.items()}},
    "boundary": "Conditional within-formulation ranking among cells with discordant source labels; not full-cohort impurity prediction, frozen PATH-Mat zero-shot transfer, sequence validation, or a causal estimate. Cell centering uses covariates only and requires multiple candidate routes per held-out cell.",
}
(RESULTS / "V238_ZEOSYN_WITHIN_CELL_BLOCK_COMPARISON.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
cell_df.to_csv(RESULTS / "V238_ZEOSYN_WITHIN_CELL_BLOCK_CONCORDANCE.csv", index=False)
print(json.dumps({"cohort": out["cohort"], "pair_weighted_concordance": out["pair_weighted_concordance"], "doi_cluster_bootstrap": out["doi_cluster_bootstrap"]}, indent=2))
