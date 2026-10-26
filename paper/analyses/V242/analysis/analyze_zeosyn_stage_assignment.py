#!/usr/bin/env python3
"""Test stage-assigned aging/crystallization conditions against an unordered bag.

This is a source-adapted conditional ranking analysis. It cannot establish a
PATH-Mat zero-shot effect or the causal effect of experimentally reordering a
route. DOI groups are held out throughout; paired route comparisons are made
only within the same DOI and exact reported formulation.
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
OUT = ROOT / "results"
SEED = 20261016
N_SPLITS = 5
N_BOOT = 2000

raw = pd.read_excel(INPUT, sheet_name="Sheet1")
df = raw.loc[raw["doi"].notna() & raw["normed"].astype(str).str.lower().eq("yes")].copy()
prod = df["product1"].astype(str).str.strip().str.lower()
pos = df["Code1"].notna() & df["Code1"].astype(str).str.strip().ne("")
neg = prod.isin({"amorphous", "dense"})
df = df.loc[pos | neg].copy().reset_index(drop=True)
df["y"] = pos[pos | neg].astype(int).to_numpy()

chem = [c for c in df.columns[1:52] if c != "solvent"] + [
    "solvent", "osda1 smiles", "osda2 smiles", "osda3 smiles"
]
sig = df[chem].astype("object").where(df[chem].notna(), "__MISSING__").apply(
    lambda r: tuple(str(x) for x in r), axis=1
)
df["_cell"] = df["doi"].astype(str) + "|" + sig.astype(str)
df["_coverage"] = df[["aging_time", "aging_temp", "cryst_time", "cryst_temp", "react_vol", "seed"]].notna().sum(axis=1)

def pair_concordance(group: pd.DataFrame, score: np.ndarray) -> tuple[float, int]:
    ix = group.index.to_numpy()
    yy = group["y"].to_numpy()
    p = score[ix]
    posix, negix = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
    d = p[posix, None] - p[negix][None, :]
    n = int(d.size)
    return float(((d > 1e-12).sum() + 0.5 * (np.abs(d) <= 1e-12).sum()) / n), n

def run_variant(label: str, required: list[str], stage_features: list[str], bag_features: list[str]) -> dict:
    sub = df.dropna(subset=required).copy().reset_index(drop=True)
    sizes = sub.groupby("_cell").size()
    sub = sub.loc[sub["_cell"].map(sizes) >= 2].copy().reset_index(drop=True)
    # Keep matched cells with positive and explicit negative outcomes.
    cell_stats = sub.groupby("_cell")["y"].nunique()
    mixed_cells = set(cell_stats[cell_stats == 2].index)
    sub["_mixed"] = sub["_cell"].isin(mixed_cells)
    # Each unordered bag is the sorted multiset of the same values; no outcome
    # is used in construction. Stage-specific and bag features have equal size.
    if label == "time_pair":
        sub["bag_low"] = np.minimum(sub["aging_time"], sub["cryst_time"])
        sub["bag_high"] = np.maximum(sub["aging_time"], sub["cryst_time"])
        stage_base = ["aging_time", "cryst_time"]
        bag_base = ["bag_low", "bag_high"]
    elif label == "temperature_pair":
        sub["bag_low"] = np.minimum(sub["aging_temp"], sub["cryst_temp"])
        sub["bag_high"] = np.maximum(sub["aging_temp"], sub["cryst_temp"])
        stage_base = ["aging_temp", "cryst_temp"]
        bag_base = ["bag_low", "bag_high"]
    elif label == "joint_time_temperature":
        sub["time_low"] = np.minimum(sub["aging_time"], sub["cryst_time"])
        sub["time_high"] = np.maximum(sub["aging_time"], sub["cryst_time"])
        sub["temp_low"] = np.minimum(sub["aging_temp"], sub["cryst_temp"])
        sub["temp_high"] = np.maximum(sub["aging_temp"], sub["cryst_temp"])
        stage_base = ["aging_time", "cryst_time", "aging_temp", "cryst_temp"]
        bag_base = ["time_low", "time_high", "temp_low", "temp_high"]
    else:
        raise ValueError(label)
    stage_cols, bag_cols, count_cols = [], [], []
    for c in stage_base:
        cc = c + "_within"
        sub[cc] = sub[c] - sub.groupby("_cell")[c].transform("mean")
        stage_cols.append(cc)
    for c in bag_base:
        cc = c + "_within"
        sub[cc] = sub[c] - sub.groupby("_cell")[c].transform("mean")
        bag_cols.append(cc)
    sub["coverage_within"] = sub["_coverage"] - sub.groupby("_cell")["_coverage"].transform("mean")
    count_cols = ["coverage_within"]

    y = sub["y"].to_numpy()
    groups = sub["doi"].astype(str).to_numpy()
    pred = {"stage_assigned": np.full(len(sub), np.nan), "unordered_bag": np.full(len(sub), np.nan), "report_count": np.full(len(sub), np.nan)}
    cv = GroupKFold(n_splits=N_SPLITS)
    folds = []
    for fold, (tr, te) in enumerate(cv.split(sub, y, groups), 1):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        for name, cols in [("stage_assigned", stage_cols), ("unordered_bag", bag_cols), ("report_count", count_cols)]:
            model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=SEED))
            model.fit(sub.iloc[tr][cols], y[tr])
            pred[name][te] = model.predict_proba(sub.iloc[te][cols])[:, 1]
        folds.append({"fold": fold, "n_train": int(len(tr)), "n_test": int(len(te)), "n_test_doi": int(len(set(groups[te])))})

    cells = []
    mixed = sub.loc[sub["_mixed"]]
    for cell, g in mixed.groupby("_cell", sort=False):
        rec = {"cell": cell, "doi": str(g["doi"].iloc[0]), "n_rows": int(len(g))}
        total = int((g["y"].sum()) * ((1-g["y"]).sum()))
        rec["n_pairs"] = total
        for name, score in pred.items():
            rec[name], _ = pair_concordance(g, score)
        cells.append(rec)
    cell_df = pd.DataFrame(cells)
    unique_doi = cell_df["doi"].unique()
    by_doi = {doi: cell_df.loc[cell_df["doi"] == doi] for doi in unique_doi}
    rng = np.random.default_rng(SEED)
    keys = ["stage_assigned", "unordered_bag", "report_count"]
    draws = {k: [] for k in keys + ["stage_minus_bag", "stage_minus_count", "bag_minus_count"]}
    for _ in range(N_BOOT):
        draw = rng.choice(unique_doi, size=len(unique_doi), replace=True)
        sample = pd.concat([by_doi[d] for d in draw], ignore_index=True)
        means = {k: float(np.average(sample[k], weights=sample["n_pairs"])) for k in keys}
        for k in keys: draws[k].append(means[k])
        draws["stage_minus_bag"].append(means["stage_assigned"] - means["unordered_bag"])
        draws["stage_minus_count"].append(means["stage_assigned"] - means["report_count"])
        draws["bag_minus_count"].append(means["unordered_bag"] - means["report_count"])
    summary = {
        "analysis": label,
        "n_rows_complete_and_cell_repeated": int(len(sub)),
        "n_mixed_cells": int(len(cell_df)),
        "n_mixed_doi": int(cell_df["doi"].nunique()),
        "n_positive_negative_pairs": int(cell_df["n_pairs"].sum()),
        "doi_grouped_cv": True,
        "folds": folds,
        "pair_weighted_concordance": {k: float(np.average(cell_df[k], weights=cell_df["n_pairs"])) for k in keys},
        "macro_cell_concordance": {k: float(cell_df[k].mean()) for k in keys},
        "doi_cluster_bootstrap_95ci": {k: [float(v) for v in np.quantile(vals, [0.025, 0.975])] for k, vals in draws.items()},
        "bootstrap_draws": N_BOOT,
        "seed": SEED,
        "feature_sets": {"stage_assigned": stage_cols, "unordered_bag": bag_cols, "report_count": count_cols},
    }
    return summary, cell_df

variants = {
    "time_pair": (["aging_time", "cryst_time"], ["aging_time", "cryst_time"], ["bag_low", "bag_high"]),
    "temperature_pair": (["aging_temp", "cryst_temp"], ["aging_temp", "cryst_temp"], ["bag_low", "bag_high"]),
    "joint_time_temperature": (["aging_time", "aging_temp", "cryst_time", "cryst_temp"], ["aging_time", "cryst_time", "aging_temp", "cryst_temp"], ["time_low", "time_high", "temp_low", "temp_high"]),
}
results = []
for label, (required, stage, bag) in variants.items():
    summary, cells = run_variant(label, required, stage, bag)
    results.append(summary)
    cells.to_csv(OUT / f"V238_ZEOSYN_STAGE_{label.upper()}_CELL_CONCORDANCE.csv", index=False)

payload = {
    "schema": "v238_zeosyn_stage_assignment_vs_unordered_bag",
    "source": "ZeoSyn public zeolite synthesis corpus; source-adapted DOI-held-out evaluation",
    "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    "endpoint": "standardized framework present vs explicitly amorphous/dense product; ambiguous outcomes excluded",
    "design": "Within exact DOI and exact formulation, predict positive-vs-negative pairs from either stage-assigned aging/crystallization values or the sorted unordered multiset of the exact same values. All values are centered within matched formulation cells without using outcomes; DOI-disjoint 5-fold OOF scoring; DOI-cluster bootstrap.",
    "boundary": "A positive stage-assignment contrast would show that the semantic assignment of reported conditions to aging versus crystallization stages carries predictive information in this source. It is not a PATH-Mat frozen test, not a randomized route-order intervention, and not a causal estimate.",
    "analyses": results,
}
(OUT / "V238_ZEOSYN_STAGE_ASSIGNMENT_AUDIT.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
