#!/usr/bin/env python3
"""Nested cross-fitted ranking of condition choices in the RE-MOF factorial grid.

This is a post hoc, source-adapted decision-value analysis. Each outer fold
holds out one metal and purges every source run block containing that metal.
The model/feature recipe is selected using only inner metal-and-run-purged
folds. Policy value is evaluated only on complete 12-condition metal×linker
grids, so every candidate has a known source-assigned endpoint. This analysis
does not establish randomized causal effects or prospective PATH-Mat utility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs" / "rare_earth_mof_campaign.csv"
PREDICTIONS = ROOT / "results" / "V229_RE_MOF_DOUBLE_HOLDOUT_PREDICTIONS.csv"
OUT_JSON = ROOT / "results" / "V230_RE_MOF_NESTED_CONDITION_POLICY.json"
OUT_CSV = ROOT / "results" / "V230_RE_MOF_NESTED_CONDITION_POLICY_BY_FORMULATION.csv"
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
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
        tv = sorted(d.Temp.unique())
        av = sorted(d["AA Conc"].unique())
        tcols = {}
        acols = {}
        for v in tv[1:]:
            tcols[v] = (d.Temp == v).astype(float)
            x[f"temp={v:g}"] = tcols[v]
            x[f"radius_temp={v:g}"] = r * tcols[v]
            x[f"topology_temp={v:g}"] = q * tcols[v]
        for v in av[1:]:
            acols[v] = (d["AA Conc"] == v).astype(float)
            x[f"acid={v:g}"] = acols[v]
            x[f"radius_acid={v:g}"] = r * acols[v]
            x[f"topology_acid={v:g}"] = q * acols[v]
        for tvv, tc in tcols.items():
            for avv, ac in acols.items():
                x[f"temp={tvv:g}:acid={avv:g}"] = tc * ac
    elif variant != "composition":
        raise ValueError(variant)
    return x.astype(float)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, variant: str, c: float) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000, solver="lbfgs"))
    model.fit(make_features(train, variant), train.y.astype(int))
    return model.predict_proba(make_features(test, variant))[:, 1]


def top_expected_y(g: pd.DataFrame, score_col: str) -> float:
    mx = g[score_col].max()
    top = g[np.isclose(g[score_col].to_numpy(float), float(mx), rtol=0, atol=1e-12)]
    return float(top.y.mean())


def select_global_condition(train: pd.DataFrame) -> tuple[float, float]:
    rates = train.groupby(["Temp", "AA Conc"], sort=True).y.mean().sort_values(ascending=False, kind="stable")
    temp, acid = rates.index[0]
    return float(temp), float(acid)


def metal_bootstrap(metal_values: pd.Series, seed: int = SEED) -> list[float]:
    a = metal_values.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, len(a), size=(BOOT, len(a)))].mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def main() -> None:
    d = load()
    # Restrict policy evaluation to complete factorial grids. A selected test
    # condition therefore never becomes unevaluable because its outcome was
    # missing; this is a label-availability cohort, not an outcome-selected set.
    sizes = d.groupby(["Metal", "Linker"]).size()
    complete = set(sizes[sizes == 12].index)
    d["complete_grid"] = [(m, l) in complete for m, l in zip(d.Metal, d.Linker)]
    d = d.reset_index(drop=True)
    model_specs = [(v, c) for v in ["composition", "poly", "categorical"] for c in [.1, 1.0, 10.0]]
    outer_records = []
    selected_models = []
    for outer_metal in sorted(d.Metal.unique()):
        test = d[(d.Metal == outer_metal) & d.complete_grid].copy()
        test_all = d[d.Metal == outer_metal].copy()
        test_runs = set(test_all.run_id)
        train_outer = d[(d.Metal != outer_metal) & (~d.run_id.isin(test_runs))].copy()
        if not len(test) or train_outer.y.nunique() != 2:
            continue
        # The candidate model is chosen only in inner held-out-metal folds.
        inner_scores = {spec: [] for spec in model_specs}
        for inner_metal in sorted(train_outer.Metal.unique()):
            val = train_outer[(train_outer.Metal == inner_metal) & train_outer.complete_grid].copy()
            val_all = train_outer[train_outer.Metal == inner_metal]
            if not len(val):
                continue
            inner_runs = set(val_all.run_id)
            train_inner = train_outer[(train_outer.Metal != inner_metal) & (~train_outer.run_id.isin(inner_runs))].copy()
            if train_inner.y.nunique() != 2:
                continue
            for variant, c in model_specs:
                p = fit_predict(train_inner, val, variant, c)
                val["score"] = p
                gains = []
                for _, g in val.groupby(["Metal", "Linker"], sort=True):
                    gains.append(top_expected_y(g, "score") - float(g.y.mean()))
                if gains:
                    inner_scores[(variant, c)].append(float(np.mean(gains)))
        mean_inner = {k: (float(np.mean(v)) if v else float("-inf")) for k, v in inner_scores.items()}
        chosen = max(model_specs, key=lambda k: (mean_inner[k], -model_specs.index(k)))
        selected_models.append({"outer_metal": outer_metal, "selected_variant": chosen[0], "selected_C": chosen[1],
                                "inner_mean_top1_gain": mean_inner[chosen]})
        for variant, c in model_specs:
            test[f"score_{variant}_C{c:g}"] = fit_predict(train_outer, test, variant, c)
        test["score_nested_selected"] = fit_predict(train_outer, test, chosen[0], chosen[1])
        glob_temp, glob_acid = select_global_condition(train_outer)
        test["global_recipe_y"] = ((test.Temp == glob_temp) & (test["AA Conc"] == glob_acid)).astype(int) * test.y
        # For each complete formulation, compute expected selected outcome
        # across exact score ties, plus the within-formulation random baseline.
        for (metal, linker), g in test.groupby(["Metal", "Linker"], sort=True):
            rec = {"metal": metal, "linker": linker, "n_candidates": int(len(g)),
                   "random_rate": float(g.y.mean()), "global_recipe": f"{glob_temp:g}C/{glob_acid:g}M",
                   "global_recipe_y": float(g.loc[(g.Temp == glob_temp) & (g["AA Conc"] == glob_acid), "y"].mean())}
            for model_name in ["composition_C0.1", "composition_C1", "composition_C10",
                               "poly_C0.1", "poly_C1", "poly_C10", "categorical_C0.1",
                               "categorical_C1", "categorical_C10", "nested_selected"]:
                col = "score_" + model_name.replace("_C", "_C") if model_name != "nested_selected" else "score_nested_selected"
                rec[model_name + "_top_y"] = top_expected_y(g, col)
            rec["outer_selected_variant"] = chosen[0]
            rec["outer_selected_C"] = chosen[1]
            outer_records.append(rec)
    out = pd.DataFrame(outer_records)
    if out.empty or out.n_candidates.nunique() != 1 or out.n_candidates.iloc[0] != 12:
        raise RuntimeError("Policy evaluation did not retain complete 12-condition grids")
    model_cols = [c for c in out.columns if c.endswith("_top_y")]
    summary = {
        "analysis": "Nested outer leave-one-metal-and-associated-run-block-out policy ranking on complete source factorial grids",
        "status": "post hoc; source-adapted; no model used the outer test metal or its run blocks; not randomized, prospective, frozen PATH-Mat, or causal",
        "source": "Gaidimas et al., Chemical Science 2026, DOI 10.1039/D5SC09992G; Zenodo 10.5281/zenodo.17902549",
        "input_sha256": sha256(INPUT), "oof_reference_predictions_sha256": sha256(PREDICTIONS),
        "cohort": {"complete_metal_linker_grids": int(out.shape[0]), "condition_cells_per_grid": 12,
                   "metals": int(out.metal.nunique()), "linkers": int(out.linker.nunique()),
                   "rows": int(out.shape[0] * 12), "prevalence": float(out.random_rate.mean())},
        "policy": "Within each held-out metal×linker grid, select the candidate with the highest training-only predicted source endpoint; score ties by their expected observed outcome. Compare to random choice within the same grid and a single temperature×acid recipe chosen from each outer training fold.",
        "nested_selection": {"inner_group": "metal with all associated run blocks purged", "candidate_features": ["composition", "polynomial conditions and prespecified interactions", "categorical conditions and prespecified interactions"], "C_values": [.1, 1.0, 10.0], "objective": "mean within-formulation top-1 observed endpoint gain over the group's random-choice mean", "selected_by_outer_fold": selected_models},
        "policies": {},
        "limitations": ["The endpoint is source-assigned presence of a named crystalline primary product, not a target-specific purity or yield measure.",
                        "The design is a public retrospective factorial campaign; policy selections were not executed after model predictions were frozen.",
                        "A top-ranked condition's observed outcome is an empirical decision-value estimate for these grids, not a randomized causal treatment effect.",
                        "The complete-grid cohort excludes 21 of 112 metal×linker formulations with one or more unknown primary labels; results may not represent these missing-label strata.",
                        "All models are refitted on this campaign's training metals; this does not evaluate frozen PATH-Mat weights or generalize across independent campaigns."]}
    for col in model_cols:
        delta = out[col] - out.random_rate
        metal_delta = out.assign(delta=delta).groupby("metal").delta.mean()
        summary["policies"][col.removesuffix("_top_y")] = {
            "mean_top1_outcome_rate": float(out[col].mean()),
            "mean_within_grid_random_rate": float(out.random_rate.mean()),
            "top1_gain_percentage_points": float(100 * delta.mean()),
            "metal_cluster_bootstrap_95ci_gain_percentage_points": [100*x for x in metal_bootstrap(metal_delta)],
            "metals_with_positive_gain": int((metal_delta > 0).sum()),
            "metals_with_zero_gain": int((metal_delta == 0).sum()),
            "metals_with_negative_gain": int((metal_delta < 0).sum()),
        }
    for baseline in ["random_rate", "global_recipe_y"]:
        delta = out.nested_selected_top_y - out[baseline]
        md = out.assign(delta=delta).groupby("metal").delta.mean()
        summary["policies"]["nested_selected_vs_" + baseline] = {
            "mean_difference_percentage_points": float(100 * delta.mean()),
            "metal_cluster_bootstrap_95ci_percentage_points": [100*x for x in metal_bootstrap(md, SEED+1)],
            "metals_with_positive_difference": int((md > 0).sum()),
            "metals_with_zero_difference": int((md == 0).sum()),
            "metals_with_negative_difference": int((md < 0).sum()),
        }
    # The best single recipe is selected per outer fold from training data only.
    delta_global = out.global_recipe_y - out.random_rate
    mdg = out.assign(delta=delta_global).groupby("metal").delta.mean()
    summary["policies"]["train_selected_single_global_recipe_vs_random_rate"] = {
        "mean_global_recipe_outcome_rate": float(out.global_recipe_y.mean()),
        "mean_within_grid_random_rate": float(out.random_rate.mean()),
        "gain_percentage_points": float(100*delta_global.mean()),
        "metal_cluster_bootstrap_95ci_gain_percentage_points": [100*x for x in metal_bootstrap(mdg, SEED+2)],
        "selected_recipe_counts": out.global_recipe.value_counts().to_dict(),
        "metals_with_positive_gain": int((mdg > 0).sum()),
        "metals_with_zero_gain": int((mdg == 0).sum()),
        "metals_with_negative_gain": int((mdg < 0).sum()),
    }
    out.to_csv(OUT_CSV, index=False)
    summary["output_csv"] = OUT_CSV.name
    summary["output_csv_sha256"] = sha256(OUT_CSV)
    summary["script_sha256"] = sha256(Path(__file__))
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cohort": summary["cohort"], "policies": summary["policies"],
                      "selected_models": pd.Series([x["selected_variant"] for x in selected_models]).value_counts().to_dict()},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
