#!/usr/bin/env python3
"""Exploratory target-formula novelty stratification of the ASTRAL challenge.

Reads an authorized local Lee source archive but exports only aggregate metrics
and overlap counts. Formula novelty is exact reduced-formula novelty, not
independent source or family novelty. ASTRAL outcomes were already seen before
PATH-Mat frozen scoring; this is diagnostic, never confirmatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = {
    "B0": "B0_composition_risk_score",
    "B1": "B1_process_bag_risk_score",
    "B2": "B2_ordered_process_risk_score",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reduced(formula: object) -> str | None:
    try:
        return Composition(str(formula)).reduced_formula
    except Exception:
        return None


def clustered_ci(frame: pd.DataFrame, metric, cluster: str, draws: int, seed: int) -> list[float] | None:
    groups = frame[cluster].astype(str).unique()
    if len(groups) < 2:
        return None
    by = {g: frame.loc[frame[cluster].astype(str) == g] for g in groups}
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(draws):
        sample = pd.concat([by[g] for g in rng.choice(groups, size=len(groups), replace=True)], ignore_index=True)
        value = metric(sample)
        if value is not None and np.isfinite(value):
            vals.append(float(value))
    if len(vals) < max(100, draws // 10):
        return None
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def metrics(frame: pd.DataFrame, draws: int, seed: int) -> dict:
    out = {"rows": int(len(frame)), "target_formulae": int(frame.target_formula.nunique())}
    for i, (model, score_col) in enumerate(MODELS.items()):
        def rho(x):
            if x.target_phase_fraction_reported.nunique() < 2 or x[score_col].nunique() < 2:
                return None
            return float(spearmanr(x[score_col], x.target_phase_fraction_reported).statistic)
        out[model] = {"spearman_risk_score_vs_target_phase_fraction": rho(frame),
                      "target_cluster_ci95_spearman": clustered_ci(frame, rho, "target_formula", draws, seed + i)}
    binary = frame.loc[frame.target_phase_fraction_reported.lt(.2) | frame.target_phase_fraction_reported.gt(.5)].copy()
    if binary.empty or binary.target_phase_fraction_reported.lt(.2).nunique() < 2:
        out["low_vs_high_purity"] = {"rows": int(len(binary)), "status": "insufficient_class_variation"}
    else:
        binary["low_purity_positive"] = binary.target_phase_fraction_reported.lt(.2).astype(int)
        out["low_vs_high_purity"] = {"rows": int(len(binary)),
                                     "low_purity_positive": int(binary.low_purity_positive.sum()),
                                     "prevalence_ap_baseline": float(binary.low_purity_positive.mean()),
                                     "moderate_rows_excluded": int(len(frame) - len(binary))}
        for i, (model, score_col) in enumerate(MODELS.items()):
            y = binary.low_purity_positive.to_numpy()
            s = binary[score_col].to_numpy()
            ap = float(average_precision_score(y, s))
            auc = float(roc_auc_score(y, s))
            out["low_vs_high_purity"][model] = {
                "average_precision": ap, "auroc": auc,
                "target_cluster_ci95_ap": clustered_ci(
                    binary, lambda x: float(average_precision_score(x.low_purity_positive, x[score_col]))
                    if x.low_purity_positive.nunique() > 1 else None,
                    "target_formula", draws, seed + 100 + i),
            }
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--lee-archive", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=17304)
    a = p.parse_args()
    data = pd.read_csv(a.predictions)
    needed = {"target_formula", "target_phase_fraction_reported", *MODELS.values()}
    if not needed.issubset(data.columns):
        raise SystemExit(f"STOP: missing fields: {sorted(needed - set(data.columns))}")
    source_formulae: set[str] = set()
    source_rows = 0
    with a.lee_archive.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            source = row.get("source")
            sid = source.get("source_id") if isinstance(source, dict) else source
            if sid == "lee2025_impurities":
                source_rows += 1
                rf = reduced(row.get("target_formula"))
                if rf:
                    source_formulae.add(rf)
    data["_reduced"] = data.target_formula.map(reduced)
    data["_formula_seen_in_lee_source_archive"] = data._reduced.isin(source_formulae)
    strata = {}
    strata_specs = [
        ("exact_formula_seen_in_lee_source_archive", data._formula_seen_in_lee_source_archive, 0),
        ("exact_formula_not_seen_in_lee_source_archive", ~data._formula_seen_in_lee_source_archive, 500),
    ]
    for name, mask, offset in strata_specs:
        strata[name] = metrics(data.loc[mask].copy(), a.bootstrap, a.seed + offset)
    result = {
        "schema_version": "pathmat_astral_formula_novelty_performance_v173",
        "status": "exploratory_post_outcome_formula_novelty_diagnostic",
        "source": {"prediction_file": a.predictions.name, "prediction_sha256": sha256(a.predictions),
                   "lee_archive_sha256": sha256(a.lee_archive), "lee_source_rows_scanned": source_rows,
                   "formula_normalization": "pymatgen Composition(...).reduced_formula exact equality",
                   "outcome_access": "ASTRAL target-phase-fraction outcomes were known before frozen scoring"},
        "strata": strata,
        "interpretation": "Exact formula novelty is not independent source or chemical-family novelty. The 9-formula unseen stratum is small and descriptive; all ASTRAL metrics are post-outcome exploratory and cannot confirm generalization.",
        "privacy": "No DOI, precursor string, target formula, row-level join key, or row-level score is emitted.",
        "bootstrap": {"unit": "target formula", "draws": a.bootstrap, "seed": a.seed},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {"rows": v["rows"], "target_formulae": v["target_formulae"],
                         "low_vs_high_purity": v.get("low_vs_high_purity", {})} for k, v in strata.items()},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
