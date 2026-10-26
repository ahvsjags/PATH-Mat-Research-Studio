#!/usr/bin/env python3
"""Independent chemistry-and-run holdout for the public RE-MOF campaign.

The estimand is source-adapted prediction of an explicitly source-reported
PXRD primary-product endpoint. It is not a frozen Lee-model transfer or a
causal treatment-effect estimate.
"""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "inputs" / "rare_earth_mof_campaign.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
RADIUS = {"Sc": .870, "Y": 1.019, "La": 1.160, "Ce": 1.143, "Pr": 1.126,
          "Nd": 1.109, "Sm": 1.079, "Eu": 1.066, "Gd": 1.053, "Tb": 1.040,
          "Dy": 1.027, "Ho": 1.015, "Er": 1.004, "Tm": .994, "Yb": .985, "Lu": .977}
Z = {"Sc": 21, "Y": 39, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62,
     "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69,
     "Yb": 70, "Lu": 71}
CARBOXYLATES = {"BDC": 2, "Phthalic": 2, "Isophthalic": 2, "BTC": 3,
                "Trimellitic": 3, "Pyromellitic": 4, "Mellitic": 6}
NO_PRODUCT = {"no solid product", "no crystalline product"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load() -> pd.DataFrame:
    d = pd.read_csv(INP)
    for c in ["Metal", "Linker"]:
        d[c] = d[c].astype(str).str.strip()
    d["Primary Phase"] = d["Primary Phase"].fillna("").astype(str).str.strip()
    # Binary-metal experiments and ambiguous phase labels are excluded.
    d = d[~d.Metal.str.contains(r"\+", regex=True)].copy()
    p = d["Primary Phase"].str.lower()
    d = d[p.isin(NO_PRODUCT) | (~p.isin({"", "unknown phase"}))].copy()
    p = d["Primary Phase"].str.lower()
    d["y"] = (~p.isin(NO_PRODUCT)).astype(int)
    d["radius_8coord"] = d.Metal.map(RADIUS)
    d["atomic_number"] = d.Metal.map(Z)
    d["linker_carboxylates"] = d.Linker.map(CARBOXYLATES)
    d["run_id"] = d.SampleID.astype(str).str.extract(r"MG2-(\d+)-", expand=False)
    if d[["radius_8coord", "atomic_number", "linker_carboxylates", "run_id"]].isna().any().any():
        raise ValueError("Unmapped chemistry descriptor or sample run identifier")
    return d.reset_index(drop=True)

def features(d: pd.DataFrame, variant: str) -> pd.DataFrame:
    x = pd.DataFrame({"radius_8coord": d.radius_8coord, "atomic_number": d.atomic_number}, index=d.index)
    for linker in sorted(d.Linker.unique()):
        x[f"linker={linker}"] = (d.Linker == linker).astype(float)
    t = (d.Temp.astype(float) - 60.0) / 20.0
    a = d["AA Conc"].astype(float) / 5.0
    r = (d.radius_8coord - 1.03) / .1
    q = (d.linker_carboxylates - 3.0) / 2.0
    if variant in {"temperature", "full"}:
        x["temperature"] = t
        x["temperature_squared"] = t * t
    if variant in {"acid", "full"}:
        x["acetic_acid"] = a
        x["acetic_acid_squared"] = a * a
    if variant == "full":
        x["radius_x_temperature"] = r * t
        x["radius_x_acid"] = r * a
        x["linker_topology_x_temperature"] = q * t
        x["linker_topology_x_acid"] = q * a
    return x.astype(float)

def fit_oof(d: pd.DataFrame, scheme: str) -> tuple[dict[str, np.ndarray], list[dict]]:
    groups = sorted(d.Metal.unique()) if scheme == "metal_and_run_purged" else sorted(d.run_id.unique())
    pred = {v: np.full(len(d), np.nan) for v in ["composition", "temperature", "acid", "full"]}
    folds = []
    for group in groups:
        if scheme == "metal_and_run_purged":
            test = (d.Metal == group).to_numpy()
            test_runs = set(d.loc[test, "run_id"])
            train = (~test) & (~d.run_id.isin(test_runs).to_numpy())
            fold = {"held_out_metal": str(group), "held_out_run_ids": sorted(test_runs)}
        else:
            test = (d.run_id == group).to_numpy()
            train = ~test
            fold = {"held_out_run_id": str(group)}
        if np.any(train & test) or not np.any(test) or not np.any(train):
            raise ValueError(f"Invalid split: {scheme} {group}")
        if scheme == "metal_and_run_purged":
            overlap_runs = set(d.loc[train, "run_id"]) & set(d.loc[test, "run_id"])
            if overlap_runs or np.any(d.loc[train, "Metal"].to_numpy() == group):
                raise ValueError("Chemistry/run overlap in purged split")
            fold["train_test_run_overlap"] = sorted(overlap_runs)
            fold["train_test_metal_overlap"] = int((d.loc[train, "Metal"] == group).sum())
        for variant in pred:
            x = features(d, variant)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs"))
            model.fit(x.loc[train], d.y.to_numpy()[train])
            pred[variant][test] = model.predict_proba(x.loc[test])[:, 1]
        fold["n_train"] = int(train.sum())
        fold["n_test"] = int(test.sum())
        folds.append(fold)
    if any(np.isnan(v).any() for v in pred.values()):
        raise ValueError("Incomplete out-of-fold scores")
    return pred, folds

def metrics(y: np.ndarray, pred: dict[str, np.ndarray]) -> dict:
    return {name: {"average_precision": float(average_precision_score(y, s)),
                   "auroc": float(roc_auc_score(y, s)),
                   "brier": float(brier_score_loss(y, s))} for name, s in pred.items()}

def paired_cluster_bootstrap(d: pd.DataFrame, pred: dict[str, np.ndarray], cluster: str,
                             draws: int = 5000, seed: int = 20261016) -> dict:
    rng = np.random.default_rng(seed)
    groups = sorted(d[cluster].unique())
    ids = {g: np.flatnonzero(d[cluster].to_numpy() == g) for g in groups}
    out = {name: {metric: [] for metric in ["average_precision", "auroc", "brier_improvement"]}
           for name in ["temperature", "acid", "full"]}
    baseline = pred["composition"]
    y0 = d.y.to_numpy()
    for _ in range(draws):
        chosen = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([ids[groups[j]] for j in chosen])
        y = y0[idx]
        if np.unique(y).size < 2:
            continue
        ap0 = average_precision_score(y, baseline[idx])
        auc0 = roc_auc_score(y, baseline[idx])
        br0 = brier_score_loss(y, baseline[idx])
        for model in out:
            s = pred[model][idx]
            out[model]["average_precision"].append(float(average_precision_score(y, s) - ap0))
            out[model]["auroc"].append(float(roc_auc_score(y, s) - auc0))
            out[model]["brier_improvement"].append(float(br0 - brier_score_loss(y, s)))
    return {m: {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                for k, v in vals.items()} for m, vals in out.items()}

def matched_cell_concordance(d: pd.DataFrame, pred: dict[str, np.ndarray], bootstrap_draws: int = 20000) -> dict:
    z = d.copy()
    z["p_composition"] = pred["composition"]
    z["p_full"] = pred["full"]
    cells = []
    for (metal, linker), g in z.groupby(["Metal", "Linker"], sort=True):
        y = g.y.to_numpy(int)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if not len(pos) or not len(neg):
            continue
        row = {"metal": metal, "linker": linker, "n": int(len(g)), "n_pos_neg_pairs": int(len(pos) * len(neg))}
        for label, col in [("composition", "p_composition"), ("composition_plus_conditions", "p_full")]:
            scores = g[col].to_numpy(float)
            delta = scores[pos[:, None]] - scores[neg][None, :]
            row[label] = float(((delta > 1e-10).sum() + .5 * (np.abs(delta) <= 1e-10).sum()) / delta.size)
        cells.append(row)
    c = pd.DataFrame(cells)
    metals = sorted(c.metal.unique())
    by_metal = np.array([(c.loc[c.metal == m, "composition_plus_conditions"].mean() - c.loc[c.metal == m, "composition"].mean()) for m in metals])
    rng = np.random.default_rng(20261017)
    boot = by_metal[rng.integers(0, len(metals), size=(bootstrap_draws, len(metals)))].mean(axis=1)
    return {"within_cell_definition": "positive–negative pairs within each exact Metal × Linker formulation; process model scores come from folds that held out the metal and purged every source run containing that metal; composition-only scores are tied within each exact formulation (ties count 0.5).",
            "n_mixed_outcome_metal_linker_cells": int(len(c)),
            "n_positive_negative_pairs": int(c.n_pos_neg_pairs.sum()),
            "composition_macro_concordance": float(c.composition.mean()),
            "conditions_macro_concordance": float(c.composition_plus_conditions.mean()),
            "macro_difference": float((c.composition_plus_conditions - c.composition).mean()),
            "metal_cluster_bootstrap_95ci_difference": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "metal_clusters_with_any_mixed_cells": int(len(metals)),
            "per_cell": cells}

def main() -> None:
    d = load()
    summary = {"source": "Gaidimas et al., Chemical Science 2026, DOI 10.1039/D5SC09992G; Zenodo DOI 10.5281/zenodo.17902549",
               "analysis_status": "post hoc source-adapted, cross-validation sensitivity; not a frozen PATH-Mat zero-shot test",
               "cohort": {"n": int(len(d)), "positive": int(d.y.sum()), "negative": int((1-d.y).sum()),
                          "prevalence": float(d.y.mean()), "metals": int(d.Metal.nunique()),
                          "linkers": int(d.Linker.nunique()), "source_run_ids": int(d.run_id.nunique())},
               "endpoint": "named source-assigned crystalline primary phase = 1; explicit no-solid/no-crystalline-product = 0; unknown/missing primary phase and bimetallic reactions excluded",
               "models": "L2 logistic regression (C=1, fixed; no tuning), standardized predictors; composition = Shannon 8-coordinate ionic radius + atomic number + linker identity; temperature and acetic-acid blocks are added as coded below",
               "feature_blocks": {"temperature": ["linear temperature", "quadratic temperature"], "acid": ["linear acetic-acid concentration", "quadratic concentration"], "full_interactions": ["metal-radius × temperature", "metal-radius × acid", "linker-carboxylate-count × temperature", "linker-carboxylate-count × acid"]},
               "splits": {}, "input_sha256": sha256(INP)}
    y = d.y.to_numpy(int)
    for scheme in ["metal_and_run_purged", "leave_one_run_out"]:
        pred, folds = fit_oof(d, scheme)
        cluster = "Metal" if scheme == "metal_and_run_purged" else "run_id"
        summary["splits"][scheme] = {"n_folds": len(folds), "group_cluster_bootstrap_draws": 5000,
                                     "metrics": metrics(y, pred),
                                     "paired_bootstrap_95ci_of_difference_vs_composition": paired_cluster_bootstrap(d, pred, cluster),
                                     "folds": folds}
        if scheme == "metal_and_run_purged":
            summary["splits"][scheme]["within_exact_formulation_pairwise_concordance"] = matched_cell_concordance(d, pred)
            pred_frame = d[["Sample#", "SampleID", "Metal", "Linker", "AA Conc", "Temp", "Primary Phase", "run_id", "y"]].copy()
            for name, scores in pred.items():
                pred_frame[f"oof_{name}"] = scores
            path = OUT / "V229_RE_MOF_DOUBLE_HOLDOUT_PREDICTIONS.csv"
            pred_frame.to_csv(path, index=False)
            summary["splits"][scheme]["prediction_csv"] = path.name
            summary["splits"][scheme]["prediction_csv_sha256"] = sha256(path)
    out = OUT / "V229_RE_MOF_DOUBLE_HOLDOUT_AUDIT.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["script_sha256"] = sha256(Path(__file__))
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cohort": summary["cohort"], "splits": {k: {"metrics": v["metrics"], "ci": v["paired_bootstrap_95ci_of_difference_vs_composition"], "within_pair": v.get("within_exact_formulation_pairwise_concordance", {})} for k, v in summary["splits"].items()}}, indent=2))

if __name__ == "__main__":
    main()
