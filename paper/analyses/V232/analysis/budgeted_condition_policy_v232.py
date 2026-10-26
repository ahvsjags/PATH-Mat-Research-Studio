#!/usr/bin/env python3
"""Nested, held-metal evaluation of budgeted condition-ranking policies.

This is an exploratory extension of the V231 top-1 policy analysis. The
condition model/specification is selected only inside each outer training
fold using the original nested top-1 criterion. Test outcomes are never used
to choose features, regularization, or the candidate budget. The analysis
compares top-k model-selected conditions with a training-only global recipe
ranking and the exact random-without-replacement expectation on complete
metal×linker grids.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs" / "rare_earth_mof_campaign.csv"
OUT_JSON = ROOT / "results" / "V232_RE_MOF_BUDGETED_POLICY.json"
OUT_CSV = ROOT / "results" / "V232_RE_MOF_BUDGETED_POLICY_BY_GRID.csv"
SEED = 20261016
BOOT = 20000
NO_PRODUCT = {"no solid product", "no crystalline product"}
RADIUS = {"Sc": .870, "Y": 1.019, "La": 1.160, "Ce": 1.143, "Pr": 1.126,
          "Nd": 1.109, "Sm": 1.079, "Eu": 1.066, "Gd": 1.053, "Tb": 1.040,
          "Dy": 1.027, "Ho": 1.015, "Er": 1.004, "Tm": .994, "Yb": .985, "Lu": .977}
Z = {"Sc": 21, "Y": 39, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62,
     "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69,
     "Yb": 70, "Lu": 71}
CARBOXYLATES = {"BDC": 2, "Phthalic": 2, "Isophthalic": 2, "BTC": 3,
                "Trimellitic": 3, "Pyromellitic": 4, "Mellitic": 6}
MODEL_SPECS = [(v, c) for v in ["composition", "poly", "categorical"] for c in [.1, 1.0, 10.0]]
BUDGETS = [1, 2, 3, 4, 6]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load() -> pd.DataFrame:
    d = pd.read_csv(INPUT)
    d = d[~d.Metal.astype(str).str.contains(r"\+", regex=True)].copy()
    d["Primary Phase"] = d["Primary Phase"].fillna("").astype(str).str.strip()
    p = d["Primary Phase"].str.lower()
    d = d[p.isin(NO_PRODUCT) | (~p.isin({"", "unknown phase"}))].copy()
    p = d["Primary Phase"].str.lower()
    d["y"] = (~p.isin(NO_PRODUCT)).astype(int)
    d["run_id"] = d.SampleID.astype(str).str.extract(r"MG2-(\d+)-", expand=False)
    d["radius"] = d.Metal.map(RADIUS)
    d["atomic_number"] = d.Metal.map(Z)
    d["topology"] = d.Linker.map(CARBOXYLATES)
    if d[["run_id", "radius", "atomic_number", "topology"]].isna().any().any():
        raise ValueError("Unmapped source record or chemistry descriptor")
    d["Temp"] = d.Temp.astype(float)
    d["AA Conc"] = d["AA Conc"].astype(float)
    return d.reset_index(drop=True)


def make_features(d: pd.DataFrame, variant: str) -> pd.DataFrame:
    x = pd.DataFrame({"radius": d.radius, "atomic_number": d.atomic_number}, index=d.index)
    for l in sorted(CARBOXYLATES):
        x[f"linker={l}"] = (d.Linker == l).astype(float)
    t = (d.Temp - 60.0) / 20.0
    a = d["AA Conc"] / 5.0
    r = (d.radius - 1.03) / .1
    q = (d.topology - 3.0) / 2.0
    if variant == "poly":
        x["temp"] = t
        x["temp2"] = t * t
        x["acid"] = a
        x["acid2"] = a * a
        x["radius_temp"] = r * t
        x["radius_acid"] = r * a
        x["topology_temp"] = q * t
        x["topology_acid"] = q * a
    elif variant == "categorical":
        for v in sorted(d.Temp.unique())[1:]:
            tc = (d.Temp == v).astype(float)
            x[f"temp={v:g}"] = tc
            x[f"radius_temp={v:g}"] = r * tc
            x[f"topology_temp={v:g}"] = q * tc
        for v in sorted(d["AA Conc"].unique())[1:]:
            ac = (d["AA Conc"] == v).astype(float)
            x[f"acid={v:g}"] = ac
            x[f"radius_acid={v:g}"] = r * ac
            x[f"topology_acid={v:g}"] = q * ac
        for tv in sorted(d.Temp.unique())[1:]:
            for av in sorted(d["AA Conc"].unique())[1:]:
                x[f"temp={tv:g}:acid={av:g}"] = ((d.Temp == tv) & (d["AA Conc"] == av)).astype(float)
    elif variant != "composition":
        raise ValueError(variant)
    return x.astype(float)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, variant: str, c: float) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, solver="lbfgs"))
    model.fit(make_features(train, variant), train.y.astype(int))
    return model.predict_proba(make_features(test, variant))[:, 1]


def selected_topk_metrics(g: pd.DataFrame, score_col: str, k: int) -> tuple[float, float]:
    """Expected precision@k and hit@k under uniform tie-breaking at the boundary."""
    s = g[score_col].to_numpy(float)
    y = g.y.to_numpy(int)
    order = np.argsort(-s, kind="stable")
    ss, yy = s[order], y[order]
    if k > len(g):
        raise ValueError("budget exceeds candidate count")
    cutoff = ss[k - 1]
    higher = ss > cutoff + 1e-12
    tied = np.abs(ss - cutoff) <= 1e-12
    n_hi = int(higher.sum())
    n_slots = k - n_hi
    y_hi = int(yy[higher].sum())
    n_tie = int(tied.sum())
    y_tie = int(yy[tied].sum())
    precision = (y_hi + n_slots * y_tie / n_tie) / k
    if y_hi > 0:
        hit = 1.0
    else:
        n_tie_neg = n_tie - y_tie
        hit = 1.0 - (math.comb(n_tie_neg, n_slots) / math.comb(n_tie, n_slots) if n_tie_neg >= n_slots else 0.0)
    return float(precision), float(hit)


def random_metrics(g: pd.DataFrame, k: int) -> tuple[float, float]:
    n = len(g)
    positives = int(g.y.sum())
    precision = positives / n
    hit = 1.0 - (math.comb(n - positives, k) / math.comb(n, k) if n - positives >= k else 0.0)
    return float(precision), float(hit)


def select_global_conditions(train: pd.DataFrame, kmax: int) -> list[tuple[float, float]]:
    rates = train.groupby(["Temp", "AA Conc"], sort=True).y.mean().sort_values(ascending=False, kind="stable")
    # Stable tie-break is inherited from sorted group keys; no test outcomes are used.
    return [(float(t), float(a)) for t, a in rates.index[:kmax]]


def bootstrap_ci(values: np.ndarray, seed: int) -> list[float]:
    a = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, len(a), size=(BOOT, len(a)))].mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def main() -> None:
    d = load()
    sizes = d.groupby(["Metal", "Linker"]).size()
    complete = set(sizes[sizes == 12].index)
    d["complete_grid"] = [(m, l) in complete for m, l in zip(d.Metal, d.Linker)]
    rows: list[dict] = []
    fold_manifest: list[dict] = []

    for outer_metal in sorted(d.Metal.unique()):
        test_all = d[d.Metal == outer_metal].copy()
        test_runs = set(test_all.run_id)
        test = test_all[test_all.complete_grid].copy()
        train_outer = d[(d.Metal != outer_metal) & (~d.run_id.isin(test_runs))].copy()
        if not len(test) or train_outer.y.nunique() != 2:
            continue
        inner_scores = {spec: [] for spec in MODEL_SPECS}
        for inner_metal in sorted(train_outer.Metal.unique()):
            val_all = train_outer[train_outer.Metal == inner_metal].copy()
            val = val_all[val_all.complete_grid].copy()
            inner_runs = set(val_all.run_id)
            train_inner = train_outer[(train_outer.Metal != inner_metal) & (~train_outer.run_id.isin(inner_runs))].copy()
            if not len(val) or train_inner.y.nunique() != 2:
                continue
            for spec in MODEL_SPECS:
                v, c = spec
                val["score"] = fit_predict(train_inner, val, v, c)
                gains = [selected_topk_metrics(g, "score", 1)[0] - float(g.y.mean())
                         for _, g in val.groupby(["Metal", "Linker"], sort=True)]
                if gains:
                    inner_scores[spec].append(float(np.mean(gains)))
        mean_inner = {k: (float(np.mean(v)) if v else float("-inf")) for k, v in inner_scores.items()}
        chosen = max(MODEL_SPECS, key=lambda k: (mean_inner[k], -MODEL_SPECS.index(k)))
        fold_manifest.append({"held_out_metal": outer_metal, "selected_model": chosen[0],
                              "selected_C": chosen[1], "inner_mean_top1_gain": mean_inner[chosen]})
        test["score_selected"] = fit_predict(train_outer, test, chosen[0], chosen[1])
        test["score_composition"] = fit_predict(train_outer, test, "composition", 1.0)
        global_conditions = select_global_conditions(train_outer, max(BUDGETS))
        for (metal, linker), g in test.groupby(["Metal", "Linker"], sort=True):
            for k in BUDGETS:
                p_model, h_model = selected_topk_metrics(g, "score_selected", k)
                p_comp, h_comp = selected_topk_metrics(g, "score_composition", k)
                p_rand, h_rand = random_metrics(g, k)
                selected_global = global_conditions[:k]
                ys = [int(((g.Temp == t) & (g["AA Conc"] == a)).any() and
                          g.loc[(g.Temp == t) & (g["AA Conc"] == a), "y"].iloc[0]) for t, a in selected_global]
                p_global = float(np.mean(ys))
                h_global = float(any(ys))
                rows.append({"held_out_metal": outer_metal, "linker": linker, "n_candidates": int(len(g)),
                             "budget_k": k, "model_variant": chosen[0], "model_C": chosen[1],
                             "model_precision_at_k": p_model, "model_hit_at_k": h_model,
                             "composition_precision_at_k": p_comp, "composition_hit_at_k": h_comp,
                             "random_expected_precision_at_k": p_rand, "random_expected_hit_at_k": h_rand,
                             "global_train_selected_precision_at_k": p_global,
                             "global_train_selected_hit_at_k": h_global})

    out = pd.DataFrame(rows)
    if out.empty or out.n_candidates.nunique() != 1 or out.n_candidates.iloc[0] != 12:
        raise RuntimeError("Expected complete 12-condition grids")
    # Aggregate the 91 grids within each metal, then bootstrap the 16 metals.
    summaries = {}
    for k in BUDGETS:
        sub = out[out.budget_k == k]
        metrics = ["precision_at_k", "hit_at_k"]
        summaries[str(k)] = {}
        for metric in metrics:
            cols = {"model": "model_" + metric,
                    "composition": "composition_" + metric,
                    "random": "random_expected_" + metric,
                    "global_recipe": "global_train_selected_" + metric}
            values_by_metal = {name: sub.groupby("held_out_metal")[col].mean().sort_index() for name, col in cols.items()}
            summaries[str(k)][metric] = {}
            for name, vals in values_by_metal.items():
                summaries[str(k)][metric][name] = {"mean": float(vals.mean()), "n_metals": int(vals.size)}
            for baseline in ["random", "global_recipe", "composition"]:
                delta = values_by_metal["model"] - values_by_metal[baseline]
                summaries[str(k)][metric]["model_minus_" + baseline] = {
                    "mean": float(delta.mean()), "metal_cluster_bootstrap_95ci": bootstrap_ci(delta.to_numpy(), SEED + 100*k + len(baseline)),
                    "metals_positive": int((delta > 0).sum()), "metals_tied": int((delta == 0).sum()),
                    "metals_negative": int((delta < 0).sum())}
    result = {
        "schema": "pathmat_v232_budgeted_condition_policy",
        "analysis": "Nested outer leave-one-metal-and-associated-run-block-out top-k policy replay on complete metal×linker condition grids",
        "status": "post hoc, source-adapted offline policy replay; not a frozen PATH-Mat transfer, prospective deployment, randomized intervention, or causal effect",
        "source": "Gaidimas et al., Chemical Science 2026, DOI 10.1039/D5SC09992G; Zenodo 10.5281/zenodo.17902549",
        "input_sha256": sha256(INPUT),
        "cohort": {"n_grids": int(out[out.budget_k == 1].shape[0]), "n_metals": int(out.held_out_metal.nunique()),
                    "n_linkers": int(out.linker.nunique()), "conditions_per_grid": 12,
                    "n_rows": int(out[out.budget_k == 1].shape[0] * 12),
                    "prevalence": float(out[out.budget_k == 1].random_expected_precision_at_k.mean())},
        "outer_fold_selection": fold_manifest,
        "budgets": BUDGETS,
        "summaries": summaries,
        "interpretation_limits": [
            "The endpoint is source-assigned formation of a named crystalline primary product, not target-specific purity or yield.",
            "Models are refit within this single campaign; this is not zero-shot PATH-Mat transfer.",
            "All candidate conditions have observed outcomes in the complete-grid subset, enabling offline policy replay but not prospective causal identification.",
            "The complete-grid subset excludes formulations with unknown or missing primary-phase labels.",
            "Model selection is nested using training folds; the final analysis was nevertheless designed after earlier inspection of this campaign and is exploratory."
        ]}
    out.to_csv(OUT_CSV, index=False)
    result["output_csv"] = OUT_CSV.name
    result["output_csv_sha256"] = sha256(OUT_CSV)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(OUT_JSON), "output_csv": str(OUT_CSV),
                      "cohort": result["cohort"], "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
