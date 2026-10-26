#!/usr/bin/env python3
"""Score the partial ASTRAL table with the frozen V136 PATH-Mat estimators.

Important: this is post-hoc exploratory evaluation because ASTRAL outcomes had
already been transcribed before this prediction run. It must not be described
as a confirmatory, blind or universal-generalization test.

The scorer reads only learned vectorizer vocabulary, logistic coefficients,
intercepts and isotonic thresholds from the frozen pickle files. This permits
version-independent prediction and is cross-checked against the original V136
Precursor Genome prediction-lock CSV when supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
import sklearn


MODELS = ("B0_composition", "B1_process_bag", "B2_ordered_process")
EXPECTED_V136_MODEL_SHA256 = {
    "B0_composition": "4a1ccd2ae6f6a51cf743a41da5b79ff7cec6f8688dbc47e001ab4dbfdd59d19e",
    "B1_process_bag": "364c85b24bc0c4d6d3a5651969bfc6bf3e0ad59e4af4a1fc905e5e3a9ae0ad3e",
    "B2_ordered_process": "f401a42cdfe20e71d3a879d961312934b13edcd82848071a7d2817ad36b1e0f6",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def actions(row: pd.Series) -> list[dict]:
    # ASTRAL article Methods: powder dispensing, ball milling (15 h),
    # vacuum drying (80 C, 2 h), then calcination in air (8 h at row T).
    # Cooling is not added because its time/temperature was not row-resolved.
    return [
        {"name": "Weighing", "ordinal": 0},
        {"name": "Ball milling", "ordinal": 1, "duration_h": 15.0},
        {"name": "Drying", "ordinal": 2, "temperature_C": 80.0, "duration_h": 2.0},
        {"name": "Calcination", "ordinal": 3,
         "temperature_C": float(row["calcination_temperature_C"]), "duration_h": 8.0},
    ]


def manual_score(obj: dict, records: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    names = list(obj["feature_names"])
    index = {name: i for i, name in enumerate(names)}
    feature_fn = obj["feature_function"]
    from PATH_FEATURES import FEATURES
    matrix = np.zeros((len(records), len(names)), dtype=np.float64)
    missing = set()
    present = set()
    for ri, record in enumerate(records):
        for key, value in FEATURES[feature_fn](record).items():
            if key in index:
                matrix[ri, index[key]] += float(value)
                present.add(key)
            else:
                missing.add(key)
    model = obj["model"]
    logits = matrix @ model.coef_[0] + float(model.intercept_[0])
    logits = np.clip(logits, -700.0, 700.0)
    raw = 1.0 / (1.0 + np.exp(-logits))
    calibrator = obj["calibrator"]
    calibrated = np.interp(raw, calibrator.X_thresholds_, calibrator.y_thresholds_)
    support = {
        "model_feature_count": int(len(names)),
        "raw_feature_keys_seen": int(len(present | missing)),
        "raw_feature_keys_matched_to_model_vocabulary": int(len(present)),
        "raw_feature_key_vocabulary_coverage": float(len(present) / max(1, len(present | missing))),
        "unmatched_raw_feature_keys": sorted(missing),
    }
    return raw, calibrated, support


def clustered_bootstrap(data: pd.DataFrame, metric_fn, group: str, draws: int,
                        seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = data[group].astype(str).unique()
    by_group = {g: data.loc[data[group].astype(str) == g] for g in groups}
    vals = []
    for _ in range(draws):
        selected = rng.choice(groups, size=len(groups), replace=True)
        pieces = [by_group[g] for g in selected]
        sample = pd.concat(pieces, ignore_index=True)
        value = metric_fn(sample)
        if value is not None and np.isfinite(value):
            vals.append(float(value))
    return np.asarray(vals, dtype=float)


def ap_low_purity(frame: pd.DataFrame, score: str) -> float | None:
    # Publisher-defined categories: low <0.2; moderate 0.2-0.5; high >0.5.
    keep = frame["target_phase_fraction_reported"].lt(0.2) | frame["target_phase_fraction_reported"].gt(0.5)
    sub = frame.loc[keep]
    y = sub["target_phase_fraction_reported"].lt(0.2).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, sub[score].to_numpy()))


def pair_concordance(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    p = frame.pivot_table(
        index=["target_formula", "calcination_temperature_C"],
        columns="precursor_strategy", values=["target_phase_fraction_reported", score],
        aggfunc="first",
    ).dropna()
    if not {"Traditional", "Predicted"}.issubset(set(p.columns.get_level_values(1))):
        return pd.DataFrame()
    dr = p[(score, "Predicted")] - p[(score, "Traditional")]
    dy = p[("target_phase_fraction_reported", "Predicted")] - p[("target_phase_fraction_reported", "Traditional")]
    correct = (np.sign(-dr) == np.sign(dy)).astype(float)
    correct[(dr == 0) | (dy == 0)] = 0.5
    return pd.DataFrame({
        "target_formula": p.index.get_level_values(0).astype(str),
        "risk_score_difference_predicted_minus_traditional": dr.to_numpy(),
        "phase_fraction_difference_predicted_minus_traditional": dy.to_numpy(),
        "rank_concordance": correct.to_numpy(),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17302)
    parser.add_argument("--pg-lock", type=Path, default=None,
                        help="optional original V136 PG prediction lock for exact scorer reproduction")
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(args.model_dir.resolve()))
    from PATH_FEATURES import FEATURES

    data = pd.read_csv(args.table)
    needed = {"target_formula", "precursor_strategy", "precursors_as_reported",
              "calcination_temperature_C", "target_phase_fraction_reported"}
    if not needed.issubset(data.columns):
        raise SystemExit(f"STOP: missing fields {sorted(needed - set(data.columns))}")
    if data[list(needed)].isna().any().any():
        raise SystemExit("STOP: required ASTRAL feature/outcome fields contain missing values")
    if data["target_phase_fraction_reported"].lt(0).any() or data["target_phase_fraction_reported"].gt(1).any():
        raise SystemExit("STOP: phase fraction outside [0,1]")

    records = []
    for _, row in data.iterrows():
        records.append({
            "target_formula": str(row["target_formula"]),
            "precursor_formulas": [s.strip() for s in str(row["precursors_as_reported"]).split(",") if s.strip()],
            "actions": actions(row),
        })
    unique_order_signatures = len({tuple(x.get("name") for x in r["actions"]) for r in records})
    scores = data.copy()
    model_hashes = {}
    feature_support = {}
    model_objects = {}
    for name in MODELS:
        model_path = args.model_dir / f"{name}.pkl"
        if not model_path.is_file():
            raise SystemExit(f"STOP: frozen model not found: {model_path}")
        model_hashes[name] = sha256(model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = pickle.loads(model_path.read_bytes())
        if model_hashes[name] != EXPECTED_V136_MODEL_SHA256[name]:
            raise SystemExit(f"STOP: {name} hash does not match the frozen V136 lock")
        model_objects[name] = obj
        raw, calibrated, support = manual_score(obj, records)
        scores[f"{name}_raw_score"] = raw
        scores[f"{name}_risk_score"] = calibrated
        feature_support[name] = support

    pg_reproduction = {"status": "not_run"}
    if args.pg_lock is not None:
        pg = pd.read_csv(args.pg_lock)
        max_deltas = {name: 0.0 for name in MODELS}
        for _, row in pg.iterrows():
            rec = {
                "target_formula": str(row["target_formula"]),
                "precursor_formulas": json.loads(row["precursor_formulas"]),
                "actions": [
                    {"name": "Weighing", "ordinal": 0},
                    {"name": "Mixing", "ordinal": 1, "duration_h": 0.5},
                    {"name": "Heating", "ordinal": 2,
                     "temperature_C": float(row["heating_temperature_C"]),
                     "duration_h": float(row["heating_time_min"]) / 60.0},
                    {"name": "Cooling", "ordinal": 3},
                    {"name": "XRD", "ordinal": 4},
                ],
            }
            for name in MODELS:
                _, cal, _ = manual_score(model_objects[name], [rec])
                key = {"B0_composition": "b0_probability", "B1_process_bag": "b1_probability", "B2_ordered_process": "b2_probability"}[name]
                max_deltas[name] = max(max_deltas[name], abs(float(cal[0]) - float(row[key])))
        pg_reproduction = {
            "status": "passed" if len(pg) == 981 and max(max_deltas.values()) <= 1e-10 else "failed",
            "rows": int(len(pg)), "expected_rows": 981,
            "maximum_absolute_prediction_difference_by_model": max_deltas,
            "pg_lock_sha256": sha256(args.pg_lock),
        }
        if pg_reproduction["status"] != "passed":
            raise SystemExit(f"STOP: V136 score-reproduction gate failed: {pg_reproduction}")

    summary = {
        "schema_version": "pathmat_astral_frozen_v136_exploratory_v173",
        "status": "exploratory_frozen_model_cross_source_challenge_outcome_previously_seen",
        "source": {
            "source_table": args.table.name,
            "source_table_sha256": sha256(args.table),
            "n_rows": int(len(data)), "n_target_formulae": int(data["target_formula"].nunique()),
            "article_reported_reactions": 224,
            "records_in_reconciled_local_transcription": int(len(data)),
            "unresolved_count_difference": int(224 - len(data)),
            "endpoint": "reported target-phase fraction estimated from integrated XRD target and residual intensities",
            "independent_laboratory": "ASTRAL at Samsung; independent of the Lee literature source used for PATH-Mat V136 estimators",
        },
        "freeze": {
            "model_sha256": model_hashes,
            "model_hashes_checked_against_frozen_v136_lock": True,
            "model_versions": "V136 frozen estimators; no fitting, recalibration or threshold change performed",
            "feature_mapping": "copied unchanged from V136 PATH_FEATURES.py",
            "feature_support": feature_support,
            "operation_order_signatures_in_astral_table": unique_order_signatures,
            "order_interpretation": "All evaluated samples use the same recorded operation-name order. This dataset cannot test a variable chronology effect; unsupported action classes are reported as out-of-vocabulary features.",
            "precursor_strategy_is_not_PATH_MAT_model_prediction": "ASTRAL labels Traditional/Predicted refer to the source authors' precursor-selection strategies, not PATH-Mat outputs.",
        },
        "evaluation": {
            "blindness": "not blind; source outcomes had been transcribed before frozen-model scoring",
            "confirmatory": False,
            "bootstrap_unit": "target formula; resample full target clusters",
            "bootstrap_draws": int(args.bootstrap),
            "random_seed": int(args.seed),
            "raw_vs_calibrated": "Both uncalibrated model probabilities and the original Lee-fitted isotonic scores are preserved; no ASTRAL recalibration.",
        },
        "metrics": {},
        "paired_precursor_strategy_comparison": {},
        "v136_original_prediction_reproduction": pg_reproduction,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "interpretation": "This is an exploratory, partial external-source challenge on a distinct physical target-phase-fraction endpoint. It is not confirmatory, does not establish universal zero-shot transfer, and does not estimate a causal effect. Only B0/B1's feature and score behavior is interpreted as a stress test; B2 cannot identify order benefit because recorded order is invariant and several order tokens are unsupported by the frozen vocabulary.",
    }

    target_groups = data["target_formula"].astype(str).to_numpy()
    scores["target_formula"] = target_groups
    for name in MODELS:
        calibrated = f"{name}_risk_score"
        raw = f"{name}_raw_score"
        observed = data["target_phase_fraction_reported"].to_numpy(dtype=float)
        base = {
            "calibrated_score_spearman_vs_target_fraction": float(spearmanr(scores[calibrated], observed).statistic),
            "raw_score_spearman_vs_target_fraction": float(spearmanr(scores[raw], observed).statistic),
            "mean_calibrated_score": float(scores[calibrated].mean()),
            "calibrated_score_sd": float(scores[calibrated].std(ddof=1)),
            "unique_calibrated_scores": int(scores[calibrated].nunique()),
            "target_cluster_bootstrap_ci95_calibrated_spearman": [],
            "low_purity_vs_high_purity": {},
        }
        boot_rho = clustered_bootstrap(
            scores, lambda b: float(spearmanr(b[calibrated], b["target_phase_fraction_reported"]).statistic),
            "target_formula", args.bootstrap, args.seed + MODELS.index(name) * 100,
        )
        base["target_cluster_bootstrap_ci95_calibrated_spearman"] = [float(np.quantile(boot_rho, .025)), float(np.quantile(boot_rho, .975))]
        sub = scores.loc[(scores["target_phase_fraction_reported"] < .2) | (scores["target_phase_fraction_reported"] > .5)]
        y = sub["target_phase_fraction_reported"].lt(.2).astype(int).to_numpy()
        base["low_purity_vs_high_purity"] = {
            "endpoint_rule": "published ASTRAL bins: low <0.2; high >0.5; moderate 0.2-0.5 excluded",
            "n": int(len(sub)), "low_purity_positive": int(y.sum()),
            "high_purity_negative": int(len(y) - y.sum()), "prevalence_baseline_ap": float(y.mean()),
            "calibrated_score_ap": float(average_precision_score(y, sub[calibrated])) if len(np.unique(y)) > 1 else None,
            "calibrated_score_auroc": float(roc_auc_score(y, sub[calibrated])) if len(np.unique(y)) > 1 else None,
            "moderate_purity_rows_excluded": int(len(scores) - len(sub)),
        }
        boot_ap = clustered_bootstrap(
            scores, lambda b: ap_low_purity(b, calibrated), "target_formula",
            args.bootstrap, args.seed + 1000 + MODELS.index(name) * 100,
        )
        base["low_purity_vs_high_purity"]["target_cluster_bootstrap_ci95_calibrated_ap"] = [float(np.quantile(boot_ap, .025)), float(np.quantile(boot_ap, .975))]
        summary["metrics"][name] = base

        pair = pair_concordance(scores, calibrated)
        if not pair.empty:
            pair_boot = clustered_bootstrap(
                pair, lambda b: float(b["rank_concordance"].mean()), "target_formula",
                args.bootstrap, args.seed + 2000 + MODELS.index(name) * 100,
            )
            summary["paired_precursor_strategy_comparison"][name] = {
                "matched_target_temperature_pairs": int(len(pair)),
                "unique_targets": int(pair["target_formula"].nunique()),
                "concordance": float(pair["rank_concordance"].mean()),
                "target_cluster_bootstrap_ci95": [float(np.quantile(pair_boot, .025)), float(np.quantile(pair_boot, .975))],
                "chance_reference": 0.5,
                "correct_direction_definition": "within a target and calcination temperature, model assigns lower risk to the precursor strategy with the higher measured target-phase fraction",
            }

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "ASTRAL_FROZEN_V136_PREDICTIONS_V173.csv"
    scores.to_csv(pred_path, index=False)
    summary["prediction_file"] = pred_path.name
    summary["prediction_file_sha256"] = sha256(pred_path)
    report_path = out_dir / "ASTRAL_FROZEN_V136_AUDIT_V173.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"], "n": summary["source"]["n_rows"],
        "targets": summary["source"]["n_target_formulae"],
        "metrics": {k: {"rho": v["calibrated_score_spearman_vs_target_fraction"],
                         "rho_ci": v["target_cluster_bootstrap_ci95_calibrated_spearman"],
                         "low_vs_high_ap": v["low_purity_vs_high_purity"]["calibrated_score_ap"],
                         "ap_ci": v["low_purity_vs_high_purity"]["target_cluster_bootstrap_ci95_calibrated_ap"]}
                    for k, v in summary["metrics"].items()},
        "paired": summary["paired_precursor_strategy_comparison"],
        "pg_model_reproduction": pg_reproduction,
        "prediction_sha256": summary["prediction_file_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
