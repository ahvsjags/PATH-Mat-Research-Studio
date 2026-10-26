"""Outcome-visible frozen PATH-Mat evaluation on the public ZIF-8 coverage data.

This script uses locked V136 B0/B1/B2 estimators without refitting,
recalibration, threshold selection, or target-label access during scoring. It
evaluates their rank ordering against the continuous SEM-coverage endpoint
reported by an independent study and separately reports the study's released
High/Low class. The adapter uses one documented solution-treatment event; no
within-route chronology is available or inferred.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "frozen_v136_models"
DATA = ROOT / "zif8_synthesis_coverage_s1_v210.csv"
PRED_OUT = ROOT / "DEDECKER_FROZEN_PATHMAT_PREDICTIONS_V210.csv"
JSON_OUT = ROOT / "DEDECKER_FROZEN_PATHMAT_AUDIT_V210.json"
sys.path.insert(0, str(MODEL_DIR))
from PATH_FEATURES import FEATURES  # noqa: E402

MODEL_FILES = {
    "B0_composition": "B0_composition.pkl",
    "B1_process_bag": "B1_process_bag.pkl",
    "B2_ordered_process": "B2_ordered_process.pkl",
}
EXPECTED_SHA256 = {
    "B0_composition": "4a1ccd2ae6f6a51cf743a41da5b79ff7cec6f8688dbc47e001ab4dbfdd59d19e",
    "B1_process_bag": "364c85b24bc0c4d6d3a5651969bfc6bf3e0ad59e4af4a1fc905e5e3a9ae0ad3e",
    "B2_ordered_process": "f401a42cdfe20e71d3a879d961312934b13edcd82848071a7d2817ad36b1e0f6",
}
BOOTSTRAPS = 10000
SEED = 20261016

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def score_model(obj: dict, records: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    names = list(obj["feature_names"])
    idx = {n: i for i, n in enumerate(names)}
    fn = FEATURES[obj["feature_function"]]
    matrix = np.zeros((len(records), len(names)), dtype=np.float64)
    seen, matched, missing, row_cov, raw_rows = set(), set(), set(), [], []
    for ri, rec in enumerate(records):
        raw = fn(rec)
        raw_rows.append(raw)
        for key, val in raw.items():
            seen.add(key)
            if key in idx:
                matched.add(key)
                matrix[ri, idx[key]] += float(val)
            else:
                missing.add(key)
        row_cov.append(sum(k in idx for k in raw) / max(1, len(raw)))
    estimator = obj["model"]
    logits = matrix @ estimator.coef_[0] + float(estimator.intercept_[0])
    raw_risk = 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))
    cal = obj["calibrator"]
    calibrated_risk = np.interp(raw_risk, cal.X_thresholds_, cal.y_thresholds_)
    support = {
        "model_features": len(names), "distinct_raw_keys": len(seen),
        "matched_raw_keys": len(matched),
        "distinct_key_coverage": len(matched) / max(1, len(seen)),
        "mean_row_key_coverage": float(np.mean(row_cov)),
        "unmatched_raw_keys": sorted(missing),
    }
    varying = sorted({k for row in raw_rows for k in row
                      if len({float(r.get(k, 0.0)) for r in raw_rows}) > 1})
    support["varying_raw_features_and_frozen_coefficients"] = [
        {"feature": k, "matched": k in idx,
         "coefficient": float(estimator.coef_[0][idx[k]]) if k in idx else None,
         "observed_min": float(min(r.get(k, 0.0) for r in raw_rows)),
         "observed_max": float(max(r.get(k, 0.0) for r in raw_rows))}
        for k in varying
    ]
    return raw_risk, calibrated_risk, support

def safe_spearman(y: np.ndarray, utility: np.ndarray) -> float:
    if len(np.unique(y)) < 2 or len(np.unique(utility)) < 2:
        return float("nan")
    return float(spearmanr(y, utility).statistic)

def continuous_concordance(y: np.ndarray, utility: np.ndarray) -> float:
    # Fraction of unequal-coverage pairs ordered in the correct direction;
    # ties in prediction receive half credit, outcome ties are excluded.
    yi, yj = np.triu_indices(len(y), k=1)
    dy = y[yi] - y[yj]
    keep = dy != 0
    if not keep.any():
        return float("nan")
    du = utility[yi[keep]] - utility[yj[keep]]
    correct = (du * dy[keep] > 0).astype(float) + 0.5 * (du == 0)
    return float(np.mean(correct))

def continuous_metrics(y: np.ndarray, risk: np.ndarray) -> dict:
    utility = -risk
    rho = safe_spearman(y, utility)
    return {
        "n": int(len(y)),
        "coverage_spearman_utility_minus_risk": rho if np.isfinite(rho) else None,
        "coverage_pairwise_concordance": continuous_concordance(y, utility),
        "raw_risk_unique_scores": int(len(np.unique(risk))),
        "raw_risk_sd": float(np.std(risk, ddof=1)),
    }

def binary_metrics(y_low: np.ndarray, risk: np.ndarray) -> dict:
    if len(np.unique(y_low)) < 2:
        return {"n": int(len(y_low)), "prevalence": float(y_low.mean()), "AP": None, "AUROC": None}
    return {
        "n": int(len(y_low)), "prevalence": float(y_low.mean()),
        "AP": float(average_precision_score(y_low, risk)),
        "AUROC": float(roc_auc_score(y_low, risk)),
    }

def main() -> None:
    df = pd.read_csv(DATA)
    assert len(df) == 62 and df.recipe_group.nunique() == 62
    assert df.coverage_pct.notna().all() and df.Quality.notna().all()
    # Source-defined target and actual documented process inputs. The adapter
    # represents one ALD-ZnO + 2-methylimidazole solution-conversion treatment;
    # solvent composition is not represented by this locked model vocabulary.
    records = []
    for r in df.itertuples(index=False):
        records.append({
            "target_formula": "C8H10N4Zn",  # Zn(2-methylimidazolate)2, ZIF-8
            "precursor_formulas": ["ZnO", "C4H6N2"],
            "actions": [{
                "name": "solution treatment",
                "temperature_C": float(r.Temperature),
                "duration_h": float(r.Duration),
            }],
        })

    models, hashes, supports = {}, {}, {}
    scores = {}
    for name, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        digest = sha256(path)
        if digest != EXPECTED_SHA256[name]:
            raise ValueError(f"Frozen model hash mismatch for {name}: {digest}")
        hashes[name] = digest
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = pickle.loads(path.read_bytes())
        models[name] = obj
        raw_risk, calibrated_risk, supports[name] = score_model(obj, records)
        scores[name] = (raw_risk, calibrated_risk)

    y_cov = df.coverage_pct.to_numpy(float)
    y_low = (df.Quality.eq("Low")).to_numpy(int)
    rows = []
    for i, rec in enumerate(records):
        row = {
            "recipe_group": str(df.recipe_group.iloc[i]),
            "coverage_pct": float(y_cov[i]), "source_quality": str(df.Quality.iloc[i]),
            "low_source_class": int(y_low[i]),
            "target_formula_adapter": rec["target_formula"],
            "precursor_formulas_adapter": ";".join(rec["precursor_formulas"]),
            "process_adapter": "one solution-treatment event; observed temperature and duration; solvent inputs omitted by frozen feature schema",
        }
        for name, (raw, calibrated) in scores.items():
            row[f"{name}_raw_risk"] = float(raw[i])
            row[f"{name}_calibrated_risk"] = float(calibrated[i])
        rows.append(row)
    pred = pd.DataFrame(rows)
    pred.to_csv(PRED_OUT, index=False)

    metrics = {}
    for name, (raw, calibrated) in scores.items():
        metrics[name] = {
            "continuous_coverage_primary_raw_risk": continuous_metrics(y_cov, raw),
            "binary_low_coverage_secondary_raw_risk": binary_metrics(y_low, raw),
            "binary_low_coverage_calibrated_risk_sensitivity": binary_metrics(y_low, calibrated),
        }
    transfer_gate = {
        name: ("RANKABLE" if len(np.unique(raw)) > 1 else "ABSTAIN_CONSTANT_SCORE")
        for name, (raw, _) in scores.items()
    }
    for name, status in transfer_gate.items():
        pred[f"{name}_transfer_status"] = status
    pred.to_csv(PRED_OUT, index=False)

    # Paired cluster bootstrap by unique recipe groups. For binary metrics,
    # stratify by the released Low/High class to avoid class-absent replicates;
    # continuous metrics use an unstratified group bootstrap.
    rng = np.random.default_rng(SEED)
    idx_low = np.flatnonzero(y_low == 1)
    idx_high = np.flatnonzero(y_low == 0)
    boot = {m: [] for m in models}
    boot_binary = {m: [] for m in models}
    paired = {k: [] for k in ["B1_minus_B0_spearman", "B2_minus_B0_spearman", "B2_minus_B1_spearman",
                              "B1_minus_B0_concordance", "B2_minus_B0_concordance", "B2_minus_B1_concordance",
                              "B1_minus_B0_AP", "B2_minus_B0_AP", "B2_minus_B1_AP",
                              "B1_minus_B0_AUROC", "B2_minus_B0_AUROC", "B2_minus_B1_AUROC"]}
    for _ in range(BOOTSTRAPS):
        ix_cont = rng.integers(0, len(df), size=len(df))
        for name, (raw, _) in scores.items():
            boot[name].append((
                safe_spearman(y_cov[ix_cont], -raw[ix_cont]),
                continuous_concordance(y_cov[ix_cont], -raw[ix_cont]),
            ))
        ix_bin = np.concatenate([
            rng.choice(idx_low, size=len(idx_low), replace=True),
            rng.choice(idx_high, size=len(idx_high), replace=True),
        ])
        bin_vals = {}
        for name, (raw, _) in scores.items():
            ap = average_precision_score(y_low[ix_bin], raw[ix_bin])
            auc = roc_auc_score(y_low[ix_bin], raw[ix_bin])
            bin_vals[name] = (float(ap), float(auc))
            boot_binary[name].append(bin_vals[name])
        # continuous paired contrasts use the exact same resampled recipe set
        # and compare each frozen score vector without refitting.
        cont_vals = {name: boot[name][-1] for name in models}
        for left, right in [("B1_process_bag", "B0_composition"),
                            ("B2_ordered_process", "B0_composition"),
                            ("B2_ordered_process", "B1_process_bag")]:
            key = f"{left.split('_')[0]}_minus_{right.split('_')[0]}"
            paired[f"{key}_spearman"].append(cont_vals[left][0] - cont_vals[right][0])
            paired[f"{key}_concordance"].append(cont_vals[left][1] - cont_vals[right][1])
            paired[f"{key}_AP"].append(bin_vals[left][0] - bin_vals[right][0])
            paired[f"{key}_AUROC"].append(bin_vals[left][1] - bin_vals[right][1])

    def ci(a):
        arr = np.asarray(a, dtype=float)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            return {"median": None, "q025": None, "q975": None, "valid_draws": 0}
        return {"median": float(np.median(arr)), "q025": float(np.quantile(arr, .025)),
                "q975": float(np.quantile(arr, .975)), "valid_draws": int(len(arr))}

    bootstrap = {
        "draws_requested": BOOTSTRAPS,
        "unit": "62 exact recipe groups; continuous metrics unstratified, binary metrics stratified by source High/Low label",
        "caveat": "conditional bootstrap of fixed frozen predictions; it contains no refitting, split-selection, or model uncertainty",
        "continuous_by_model": {m: {"Spearman": ci([x[0] for x in boot[m]]),
                                     "pairwise_concordance": ci([x[1] for x in boot[m]])} for m in models},
        "binary_by_model": {m: {"AP": ci([x[0] for x in boot_binary[m]]),
                                "AUROC": ci([x[1] for x in boot_binary[m]])} for m in models},
        "paired_differences": {k: ci(v) for k, v in paired.items()},
    }
    report = {
        "analysis": "outcome-visible frozen zero-fit evaluation of Lee-trained PATH-Mat V136 on an independent ZIF-8 membrane study",
        "source_article": "Dedecker et al., ACS Applied Materials & Interfaces 18, 12199-12211 (2026), DOI 10.1021/acsami.5c24910",
        "source_url": "https://doi.org/10.1021/acsami.5c24910",
        "data_file": DATA.name,
        "data_sha256": sha256(DATA),
        "models": {m: {"file": MODEL_FILES[m], "sha256": hashes[m]} for m in models},
        "feature_builder_sha256": sha256(MODEL_DIR / "PATH_FEATURES.py"),
        "source_sample": {"rows": int(len(df)), "unique_recipes": int(df.recipe_group.nunique()),
                          "coverage_min_max_pct": [float(y_cov.min()), float(y_cov.max())],
                          "binary_low_n": int(y_low.sum()), "binary_high_n": int((1-y_low).sum())},
        "adapter": {
            "target_formula": "C8H10N4Zn (ZIF-8)",
            "precursors": ["ZnO ALD layer", "2-methylimidazole (C4H6N2)"],
            "action": "one solution-treatment event with source-reported temperature and duration",
            "unencoded_fields": ["solvent identities", "solvent ratio", "substrate/support", "all reagent concentrations"],
            "sequence": "one event for every row; no variable sequence or chronological order is available",
        },
        "freeze": {"fit": False, "recalibration": False, "threshold_transfer": False,
                   "outcome_labels_used_for_inference": False,
                   "outcome_labels_used_only_for_posthoc_metric_calculation": True,
                   "positive direction": "lower Lee-trained impurity-risk score indicates higher predicted quality; continuous endpoint rank uses negative raw risk"},
        "feature_support": supports,
        "metrics": metrics,
        "score_variance_gate": {
            "rule": "abstain from ranking when the frozen raw score has fewer than two unique values",
            "result": transfer_gate,
            "dataset_decision": "ABSTAIN_CONSTANT_SCORE" if all(x != "RANKABLE" for x in transfer_gate.values()) else "PARTIAL_RANKABILITY",
        },
        "paired_cluster_bootstrap": bootstrap,
        "interpretation": "Any association is frozen transfer to one independent source and one different image-derived endpoint within one material system. It is not universal zero-shot generalization. Because action semantics/order are constant across rows and solvent identities/ratios are unsupported by this model vocabulary, any predictive signal cannot establish generic process semantics or an operation-order effect. It also does not validate PATH-Mat-guided causal optimization.",
        "prediction_file": PRED_OUT.name,
        "software": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    JSON_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
