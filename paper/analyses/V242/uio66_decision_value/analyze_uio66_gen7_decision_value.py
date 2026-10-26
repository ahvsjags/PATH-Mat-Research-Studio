#!/usr/bin/env python3
"""Post hoc decision-value summary for fixed UiO-66 Gen7 out-of-generation scores.

No model is fit, calibrated, thresholded, or selected here. The existing Gen7
out-of-fold scores come from the fixed Gen0-Gen6 fit in V189. Outcomes are read
only after the score file and model result hashes have been verified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
PRED = HERE / "UIO66_GENERATION_GROUPED_OOF_PREDICTIONS.csv"
PARENT = HERE / "UIO66_GEN7_ADAPTATION_RESULTS_parent.json"
OUT_JSON = HERE / "UIO66_GEN7_DECISION_VALUE_V240.json"
OUT_CSV = HERE / "UIO66_GEN7_DECISION_VALUE_V240.csv"
EXPECTED_PRED_SHA256 = "efe5d6eee65d8105deb9baf0e45de806ac6064b2b685f7203c7ebbf81f6205dd"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> list[float]:
    """Two-sided exact binomial interval; descriptive for these selected sets."""
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return [lo, hi]


def main() -> None:
    pred_sha = sha256(PRED)
    if pred_sha != EXPECTED_PRED_SHA256:
        raise RuntimeError(f"OOF prediction input hash mismatch: {pred_sha}")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    df = pd.read_csv(PRED)
    test = df.loc[df["generation"].astype(str).eq("Gen7")].copy()
    required = {"sample_id", "impurity", "A_N_score", "A_N_S_score"}
    if not required.issubset(test.columns):
        raise RuntimeError(f"missing required fields: {sorted(required - set(test.columns))}")
    if len(test) != 43 or int(test["impurity"].sum()) != 22:
        raise RuntimeError("Gen7 cohort does not match the locked parent audit")
    if test["A_N_score"].nunique() != 1:
        raise RuntimeError("expected the count-augmented baseline to be tied in Gen7")
    if test["A_N_S_score"].nunique() < 2:
        raise RuntimeError("expanded process scores are unexpectedly tied")

    n_all = int(len(test))
    impure_all = int(test["impurity"].sum())
    pure_all = n_all - impure_all
    prevalence = impure_all / n_all
    pure_rate = pure_all / n_all
    rows = []
    for coverage in (0.10, 0.20, 0.25, 0.50):
        target_n = int(np.ceil(n_all * coverage))
        # Keep every sample tied at a score boundary; actual tail sizes can exceed
        # the nominal capacity fraction by a few rows.
        hi_cut = float(test["A_N_S_score"].nlargest(target_n).min())
        lo_cut = float(test["A_N_S_score"].nsmallest(target_n).max())
        hi = test.loc[test["A_N_S_score"] >= hi_cut]
        lo = test.loc[test["A_N_S_score"] <= lo_cut]
        n_hi, n_lo = int(len(hi)), int(len(lo))
        hi_pos = int(hi["impurity"].sum())
        lo_pure = int(n_lo - lo["impurity"].sum())
        rows.append({
            "nominal_coverage_fraction": coverage,
            "high_risk_n": n_hi,
            "high_risk_effective_coverage": n_hi / n_all,
            "high_risk_impure_n": hi_pos,
            "high_risk_impure_fraction": hi_pos / n_hi,
            "high_risk_impure_exact95_low": clopper_pearson(hi_pos, n_hi)[0],
            "high_risk_impure_exact95_high": clopper_pearson(hi_pos, n_hi)[1],
            "recall_of_all_impure": hi_pos / impure_all,
            "low_risk_n": n_lo,
            "low_risk_effective_coverage": n_lo / n_all,
            "low_risk_source_pure_n": lo_pure,
            "low_risk_source_pure_fraction": lo_pure / n_lo,
            "low_risk_pure_exact95_low": clopper_pearson(lo_pure, n_lo)[0],
            "low_risk_pure_exact95_high": clopper_pearson(lo_pure, n_lo)[1],
            "fraction_of_all_source_pure_captured": lo_pure / pure_all,
            "baseline_impurity_prevalence": prevalence,
            "baseline_source_pure_fraction": pure_rate,
        })
    result = {
        "status": "completed_posthoc_offline_decision_value_summary",
        "source": "Holz, UiO-66 Benchmark Dataset v1.0, Zenodo 10.5281/zenodo.17211171",
        "scope": "Gen0-Gen6-fitted same-material/same-lab model scores evaluated on Gen7; source XRD-linked binary Phase Purity endpoint; not frozen cross-source zero-shot and not a prospective intervention",
        "parent_result_sha256": sha256(PARENT),
        "parent_prediction_csv_sha256": pred_sha,
        "parent_split": {"fit_generations": "Gen0-Gen6", "test_generation": "Gen7", "fit_rows": 192, "test_rows": n_all},
        "endpoint": "impurity = 1 - source Phase Purity; source-pure is the complementary source label, not an independent re-refinement",
        "baseline": {"n": n_all, "impure": impure_all, "source_pure": pure_all, "impurity_prevalence": prevalence, "source_pure_fraction": pure_rate},
        "policy": "Rank existing A_N_S Gen7 scores; high-risk tail is largest predicted impurity score and low-risk tail is smallest. A_N (frozen composition score plus action count) is tied within Gen7 and has no ranking. No model refit, score calibration, threshold, or cutoff selection was performed in this analysis.",
        "coverage_sensitivity": rows,
        "posthoc_status": "The archive and labels were previously inspected. Capacity fractions are descriptive sensitivity points, not preregistered operating thresholds. Exact intervals are per-selected-set binomial intervals and do not include model-fit, generation, or laboratory-batch uncertainty.",
        "model_parent_metrics": parent.get("models", {}).get("A_N_S", {}),
        "interpretation": "This is an offline ranking/triage value analysis on a later labelled generation after same-source supervision. It cannot establish universal zero-shot generalization, causal optimization, or prospective utility.",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(json.dumps({"baseline": result["baseline"], "coverage_sensitivity": rows, "output_sha256": {OUT_JSON.name: sha256(OUT_JSON), OUT_CSV.name: sha256(OUT_CSV)}}, indent=2))


if __name__ == "__main__":
    main()
