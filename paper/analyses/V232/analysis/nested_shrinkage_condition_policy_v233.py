#!/usr/bin/env python3
"""Nested shrinkage of chemistry-aware scores toward a training-only recipe prior.

Exploratory, held-metal replay on the rare-earth MOF campaign. Candidate score
blends model probabilities with beta-smoothed condition-specific success rates
estimated only in the current training fold. Model family, regularization and
shrinkage are selected in inner held-metal folds; no outer test outcomes enter
selection. This can test a legitimate robustness improvement to the prior
budgeted policy, but does not create independent-source or causal evidence.
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
OUT_JSON = ROOT / "results" / "V233_RE_MOF_NESTED_SHRINKAGE_POLICY.json"
OUT_CSV = ROOT / "results" / "V233_RE_MOF_NESTED_SHRINKAGE_POLICY_BY_GRID.csv"
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
ALPHAS = [0.0, .25, .5, .75, 1.0]


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
    t, a = (d.Temp - 60.0) / 20.0, d["AA Conc"] / 5.0
    r, q = (d.radius - 1.03) / .1, (d.topology - 3.0) / 2.0
    if variant == "poly":
        x["temp"], x["temp2"] = t, t * t
        x["acid"], x["acid2"] = a, a * a
        x["radius_temp"], x["radius_acid"] = r * t, r * a
        x["topology_temp"], x["topology_acid"] = q * t, q * a
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


def condition_prior(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    # Beta(1,1) smoothing, estimated only from the current training rows.
    counts = train.groupby(["Temp", "AA Conc"]).y.agg(["sum", "count"])
    return np.array([(float(counts.loc[(r.Temp, r["AA Conc"]), "sum"]) + 1.0) /
                     (float(counts.loc[(r.Temp, r["AA Conc"]), "count"]) + 2.0)
                     for _, r in test.iterrows()], dtype=float)


def top1(g: pd.DataFrame, col: str) -> float:
    s = g[col].to_numpy(float)
    y = g.y.to_numpy(int)
    cutoff = np.max(s)
    ties = np.isclose(s, cutoff, rtol=0.0, atol=1e-12)
    return float(y[ties].mean())


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
    out_rows: list[dict] = []
    fold_manifest: list[dict] = []

    for outer_metal in sorted(d.Metal.unique()):
        test_all = d[d.Metal == outer_metal].copy()
        test_runs = set(test_all.run_id)
        test = test_all[test_all.complete_grid].copy()
        train_outer = d[(d.Metal != outer_metal) & (~d.run_id.isin(test_runs))].copy()
        if not len(test) or train_outer.y.nunique() != 2:
            continue
        inner_values = {(*spec, alpha): [] for spec in MODEL_SPECS for alpha in ALPHAS}

        for inner_metal in sorted(train_outer.Metal.unique()):
            val_all = train_outer[train_outer.Metal == inner_metal].copy()
            val = val_all[val_all.complete_grid].copy()
            inner_runs = set(val_all.run_id)
            train_inner = train_outer[(train_outer.Metal != inner_metal) & (~train_outer.run_id.isin(inner_runs))].copy()
            if not len(val) or train_inner.y.nunique() != 2:
                continue
            prior = condition_prior(train_inner, val)
            base = float("nan")
            for variant, c in MODEL_SPECS:
                p_model = fit_predict(train_inner, val, variant, c)
                for alpha in ALPHAS:
                    val["score"] = alpha * p_model + (1.0 - alpha) * prior
                    gains = [top1(g, "score") - float(g.y.mean())
                             for _, g in val.groupby(["Metal", "Linker"], sort=True)]
                    if gains:
                        inner_values[(variant, c, alpha)].append(float(np.mean(gains)))

        mean_inner = {k: float(np.mean(v)) if v else float("-inf") for k, v in inner_values.items()}
        chosen = max(mean_inner, key=lambda k: (mean_inner[k], -list(mean_inner).index(k)))
        variant, c, alpha = chosen
        fold_manifest.append({"held_out_metal": outer_metal, "selected_variant": variant,
                              "selected_C": c, "selected_alpha_model_weight": alpha,
                              "inner_top1_gain_over_grid_prevalence": mean_inner[chosen]})

        p_model = fit_predict(train_outer, test, variant, c)
        prior = condition_prior(train_outer, test)
        p_comp = fit_predict(train_outer, test, "composition", 1.0)
        test["score_nested_shrinkage"] = alpha * p_model + (1.0 - alpha) * prior
        test["score_raw_model"] = p_model
        test["score_composition"] = p_comp
        test["score_global_prior"] = prior
        # The baseline chooses a single global condition with the largest training-only success rate.
        global_counts = train_outer.groupby(["Temp", "AA Conc"]).y.agg(["sum", "count"])
        global_rates = ((global_counts["sum"] + 1.0) / (global_counts["count"] + 2.0)).sort_values(ascending=False, kind="stable")
        best_global = global_rates.index[0]
        for (metal, linker), g in test.groupby(["Metal", "Linker"], sort=True):
            global_mask = (g.Temp == best_global[0]) & (g["AA Conc"] == best_global[1])
            if global_mask.sum() != 1:
                raise RuntimeError("Global condition absent or duplicated in a complete grid")
            out_rows.append({"held_out_metal": outer_metal, "linker": linker,
                             "variant": variant, "C": c, "alpha": alpha,
                             "nested_policy_top1": top1(g, "score_nested_shrinkage"),
                             "raw_model_top1": top1(g, "score_raw_model"),
                             "composition_top1": top1(g, "score_composition"),
                             "global_recipe_top1": float(g.loc[global_mask, "y"].iloc[0]),
                             "random_expected_top1": float(g.y.mean()),
                             "n_conditions": int(len(g)), "prevalence": float(g.y.mean())})

    out = pd.DataFrame(out_rows)
    if out.empty or out.n_conditions.nunique() != 1 or out.n_conditions.iloc[0] != 12:
        raise RuntimeError("Expected only complete 12-condition grids")
    metrics = ["nested_policy_top1", "raw_model_top1", "composition_top1", "global_recipe_top1", "random_expected_top1"]
    by_metal = {m: out.groupby("held_out_metal")[m].mean().sort_index() for m in metrics}
    summary = {m: float(by_metal[m].mean()) for m in metrics}
    deltas = {}
    for baseline in ["global_recipe_top1", "random_expected_top1", "composition_top1"]:
        diff = by_metal["nested_policy_top1"] - by_metal[baseline]
        deltas[baseline] = {"mean_delta": float(diff.mean()),
                            "metal_cluster_bootstrap_95ci": bootstrap_ci(diff.to_numpy(), SEED + len(baseline)),
                            "metals_positive": int((diff > 0).sum()),
                            "metals_tied": int((diff == 0).sum()),
                            "metals_negative": int((diff < 0).sum())}
    result = {
        "schema": "pathmat_v233_nested_shrinkage_condition_policy",
        "analysis": "Nested outer held-metal and associated-run-block-out policy replay on complete metal×linker 12-condition grids; inner folds select model family, regularization, and shrinkage toward training-only condition priors.",
        "status": "post hoc, single-source, source-adapted offline replay; not a frozen transfer, independent replication, prospective deployment, randomized intervention, or causal estimate",
        "source": "Gaidimas et al., Chemical Science 2026, DOI 10.1039/D5SC09992G; Zenodo 10.5281/zenodo.17902549",
        "input_sha256": sha256(INPUT),
        "cohort": {"n_grids": int(len(out)), "n_metals": int(out.held_out_metal.nunique()),
                    "n_linkers": int(out.linker.nunique()), "conditions_per_grid": 12,
                    "n_rows": int(len(out) * 12), "grid_weighted_prevalence": float(out.random_expected_top1.mean())},
        "alpha_candidates": ALPHAS,
        "fold_selection": fold_manifest,
        "selected_alpha_counts": {str(a): int(sum(x["selected_alpha_model_weight"] == a for x in fold_manifest)) for a in ALPHAS},
        "macro_metal_means": summary,
        "nested_policy_minus_baseline": deltas,
        "interpretation_limits": [
            "The complete-grid subset excludes unknown or missing phase labels.",
            "The endpoint is source-assigned primary crystalline product, not target-specific phase purity or yield.",
            "The source-adapted model/prior policy is evaluated on metals held out within one campaign; it is not zero-shot PATH-Mat transfer.",
            "All candidate conditions have observed outcomes, so this is retrospective policy replay rather than prospective utility or causal intervention.",
            "The analysis was designed after prior inspection of this campaign and is exploratory despite nested selection."
        ]}
    out.to_csv(OUT_CSV, index=False)
    result["output_csv"] = OUT_CSV.name
    result["output_csv_sha256"] = sha256(OUT_CSV)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"cohort": result["cohort"], "macro_metal_means": summary,
                      "deltas": deltas, "selected_alpha_counts": result["selected_alpha_counts"]}, indent=2))


if __name__ == "__main__":
    main()
