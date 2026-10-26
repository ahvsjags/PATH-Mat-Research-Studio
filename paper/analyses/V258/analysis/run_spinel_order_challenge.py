from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import fisher_exact
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "inputs" / "gremlin"
EXPECTED_COMMIT = "4e4a30b4d12b0e3f55f0990bb2edaaff0c34bfe6"
EXPECTED_CSV_SHA256 = "46ecb303d0163fb54d1a2260119e4c19850d8dea18e4ad20b6ef2d3c98e00807"
REPO: Path
CSV: Path
OUT: Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / den
    return float(mid - half), float(mid + half)


def newcombe_diff_ci(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float]:
    """Newcombe unpaired difference interval from independent Wilson intervals."""
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    d = p1 - p0
    lo = d - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return float(lo), float(hi)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, pred)),
        "average_precision": float(average_precision_score(y, pred)),
        "brier": float(brier_score_loss(y, pred)),
    }


def main() -> None:
    global REPO, CSV, OUT
    parser = argparse.ArgumentParser(description="Reproduce the independent spinel order-feature challenge.")
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_REPO, help="Local checkout of the GREMLIN source repository.")
    parser.add_argument("--source-csv", type=Path, default=None, help="Optional path to K2CO3-all.csv; defaults to <source-repo>/data/K2CO3-all.csv.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results", help="Directory for audit JSON and OOF predictions.")
    args = parser.parse_args()
    REPO = args.source_repo.resolve()
    CSV = (args.source_csv or (REPO / "data" / "K2CO3-all.csv")).resolve()
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if not CSV.is_file():
        raise FileNotFoundError(f"Source CSV not found: {CSV}")
    if not (REPO / ".git").exists():
        raise FileNotFoundError(f"Source repository checkout not found: {REPO}")
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    data_digest = sha256(CSV)
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"Unexpected GREMLIN commit: {commit}; expected {EXPECTED_COMMIT}")
    if data_digest != EXPECTED_CSV_SHA256:
        raise ValueError(f"Unexpected K2CO3-all.csv SHA-256: {data_digest}; expected {EXPECTED_CSV_SHA256}")
    df = pd.read_csv(CSV)
    # The repository codes normal precipitation as 0 and reverse order as 1.
    df["singlephase"] = df["singlephase"].astype(str).str.lower().map({"true": 1, "false": 0})
    if df["singlephase"].isna().any():
        raise ValueError("Unexpected missing or non-boolean phase labels")
    if set(df["precipitation_order"].unique()) != {0, 1}:
        raise ValueError("Unexpected precipitation_order coding")

    normal = df[df.precipitation_order == 0]
    reverse = df[df.precipitation_order == 1]
    n0, k0 = len(normal), int(normal.singlephase.sum())
    n1, k1 = len(reverse), int(reverse.singlephase.sum())
    rd = k1 / n1 - k0 / n0
    rd_ci = newcombe_diff_ci(k1, n1, k0, n0)
    odds_ratio, fisher_p = fisher_exact([[k1, n1 - k1], [k0, n0 - k0]], alternative="two-sided")

    exact_key = ["metal_initial_conc", "metal_mmol", "K2CO3_initial_conc", "rate"]
    strata = df.groupby(exact_key + ["precipitation_order"], dropna=False).size().unstack("precipitation_order", fill_value=0)
    n_matched_strata = int(((strata.get(0, 0) > 0) & (strata.get(1, 0) > 0)).sum())

    # A deliberately simple out-of-fold descriptive benchmark. This is not a causal estimator.
    y = df.singlephase.to_numpy(dtype=int)
    base_features = ["metal_initial_conc", "metal_mmol", "K2CO3_initial_conc", "rate"]
    models = {
        "numeric_conditions": base_features,
        "numeric_conditions_plus_order": base_features + ["precipitation_order"],
    }
    cv_results: dict[str, dict] = {}
    pred_table = df.copy()
    loo_predictions: dict[str, np.ndarray] = {}
    for name, features in models.items():
        X = df[features].to_numpy(dtype=float)
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", max_iter=500))
        pred = cross_val_predict(estimator, X, y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
        loo_predictions[name] = pred
        pred_table[name + "_loo_probability"] = pred
        cv_results[name] = {
            "auroc": float(roc_auc_score(y, pred)),
            "average_precision": float(average_precision_score(y, pred)),
            "brier": float(brier_score_loss(y, pred)),
            "probability_unique_n": int(np.unique(np.round(pred, 12)).size),
        }

    # Paired bootstrap of the already generated LOO predictions. This interval is
    # conditional on those predictions and does not include model-refit uncertainty.
    rng = np.random.default_rng(244)
    idx0, idx1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    boot_delta = []
    for _ in range(1_000):
        ix = np.concatenate((rng.choice(idx0, len(idx0), replace=True), rng.choice(idx1, len(idx1), replace=True)))
        m0 = metrics(y[ix], loo_predictions["numeric_conditions"][ix])
        m1 = metrics(y[ix], loo_predictions["numeric_conditions_plus_order"][ix])
        boot_delta.append({key: m1[key] - m0[key] for key in m0})
    conditional_bootstrap = {}
    for key in ("auroc", "average_precision", "brier"):
        values = np.asarray([row[key] for row in boot_delta], dtype=float)
        conditional_bootstrap[key] = {
            "point_difference_order_added_minus_baseline": cv_results["numeric_conditions_plus_order"][key] - cv_results["numeric_conditions"][key],
            "conditional_stratified_bootstrap_95ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "interpretation": "conditional on paired LOO predictions; not a full model-refit or external-sampling interval",
        }

    # Hold out the four-character destination identifier prefix as a conservative
    # source-record grouping sensitivity. It is not assigned a physical batch/date
    # meaning because the public metadata does not document that interpretation.
    prefix = df["destination"].astype(str).str[:4].to_numpy()
    logo_results = {}
    logo_predictions = {}
    for name, features in models.items():
        X = df[features].to_numpy(dtype=float)
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="liblinear", max_iter=500))
        pred = cross_val_predict(estimator, X, y, groups=prefix, cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
        logo_predictions[name] = pred
        logo_results[name] = metrics(y, pred)
        pred_table[name + "_destination_prefix_heldout_probability"] = pred
    logo_delta = {key: logo_results["numeric_conditions_plus_order"][key] - logo_results["numeric_conditions"][key] for key in ("auroc", "average_precision", "brier")}
    destination_group_summary = []
    for group in sorted(np.unique(prefix)):
        mask = prefix == group
        destination_group_summary.append({
            "destination_prefix": str(group),
            "n": int(mask.sum()),
            "single_phase_n": int(y[mask].sum()),
            "normal_order_n": int((df.loc[mask, "precipitation_order"].to_numpy() == 0).sum()),
            "reverse_order_n": int((df.loc[mask, "precipitation_order"].to_numpy() == 1).sum()),
        })
    positive_groups = [g for g in np.unique(prefix) if int(y[prefix == g].sum()) > 0]
    zero_positive_groups = [g for g in np.unique(prefix) if int(y[prefix == g].sum()) == 0]
    group_boot_delta = []
    group_rng = np.random.default_rng(2441)
    for _ in range(1_000):
        sampled_groups = list(group_rng.choice(positive_groups, len(positive_groups), replace=True))
        sampled_groups += list(group_rng.choice(zero_positive_groups, len(zero_positive_groups), replace=True))
        ix = np.concatenate([np.flatnonzero(prefix == g) for g in sampled_groups])
        m0 = metrics(y[ix], logo_predictions["numeric_conditions"][ix])
        m1 = metrics(y[ix], logo_predictions["numeric_conditions_plus_order"][ix])
        group_boot_delta.append({key: m1[key] - m0[key] for key in m0})
    group_bootstrap = {}
    for key in ("auroc", "average_precision", "brier"):
        values = np.asarray([row[key] for row in group_boot_delta], dtype=float)
        group_bootstrap[key] = {
            "point_difference_order_added_minus_baseline": logo_delta[key],
            "conditional_destination_group_bootstrap_95ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "groups_resampled": len(positive_groups) + len(zero_positive_groups),
            "resampling_strata": {"groups_with_positive": len(positive_groups), "zero_positive_groups": len(zero_positive_groups)},
            "interpretation": "conditional on leave-destination-prefix-out predictions; only seven source-ID groups, not a population-level external interval",
        }

    try:
        data_file = CSV.relative_to(REPO).as_posix()
    except ValueError:
        data_file = CSV.name

    audit = {
        "schema": "pathmat_v244_gremlin_order_challenge",
        "question": "Does an independent public mixed-metal-oxide HTE/XRD dataset provide evidence for an order effect under PATH-Mat-relevant interpretation?",
        "dataset_scope": "Repository analysis CSV (70 model-ready experiments); the article reports 79 total samples, of which 70 were used in the classifier. No rows were fabricated or imputed.",
        "source": {
            "paper_doi": "10.1038/s43246-025-00867-3",
            "repository": "https://github.com/AccelerationConsortium/gremlin",
            "repo_commit": commit,
            "data_file": data_file,
            "data_sha256": data_digest,
        },
        "outcome": "source-reported high-throughput XRD single-phase label for Fe2(ZnCo)O4",
        "order_coding": {"0": "normal: precipitant added into metal precursor", "1": "reverse: metal precursor added into precipitant"},
        "descriptive_order_contrast": {
            "normal_n": n0,
            "normal_single_phase_n": k0,
            "normal_rate": k0 / n0,
            "normal_rate_wilson_95ci": wilson(k0, n0),
            "reverse_n": n1,
            "reverse_single_phase_n": k1,
            "reverse_rate": k1 / n1,
            "reverse_rate_wilson_95ci": wilson(k1, n1),
            "reverse_minus_normal_risk_difference": rd,
            "risk_difference_newcombe_95ci": rd_ci,
            "reverse_vs_normal_odds_ratio": float(odds_ratio),
            "fisher_exact_two_sided_p": float(fisher_p),
            "interpretation": "unadjusted association only; order allocation was not documented as randomized",
        },
        "common_support": {
            "exact_matching_fields": exact_key,
            "exact_covariate_strata_with_both_order_arms": n_matched_strata,
            "exactly_matched_pairwise_causal_effect_estimable": n_matched_strata > 0,
            "note": "No exact condition strata contain both order arms; the marginal order contrast is confounded by the experimental design distribution.",
        },
        "loo_predictive_sensitivity": {
            "models": cv_results,
            "paired_oof_conditional_bootstrap": conditional_bootstrap,
            "interpretation": "exploratory within-cohort prediction comparison; not a causal effect, external validation, or PATH-Mat frozen-model test",
        },
        "destination_prefix_group_holdout_sensitivity": {
            "group_definition": "first four characters of the public destination identifier; held out as a group without assigning the identifier a documented physical meaning",
            "groups": destination_group_summary,
            "models": logo_results,
            "order_added_minus_numeric_conditions": logo_delta,
            "paired_conditional_group_bootstrap": group_bootstrap,
            "interpretation": "additional internal group-holdout sensitivity; not an untouched external test, and groups are only as independent as the source identifiers warrant",
        },
        "causal_pathmat_optimization_estimable": False,
        "reason": "No PATH-Mat policy assigned treatments; no randomized policy-versus-control experiment is present.",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (OUT / "V256_SPINEL_ORDER_CHALLENGE_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pred_table.to_csv(OUT / "V256_SPINEL_ORDER_CHALLENGE_PREDICTIONS.csv", index=False, encoding="utf-8")
    print(json.dumps({
        "rows": len(df),
        "normal": {"n": n0, "positive": k0, "rate": k0 / n0, "ci": wilson(k0, n0)},
        "reverse": {"n": n1, "positive": k1, "rate": k1 / n1, "ci": wilson(k1, n1)},
        "risk_difference_reverse_minus_normal": rd,
        "newcombe_ci": rd_ci,
        "odds_ratio": odds_ratio,
        "fisher_p": fisher_p,
        "exact_matched_strata": n_matched_strata,
        "loo_models": cv_results,
        "loo_conditional_bootstrap": conditional_bootstrap,
        "destination_prefix_group_holdout": {"models": logo_results, "delta": logo_delta, "paired_conditional_group_bootstrap": group_bootstrap},
        "data_sha256": sha256(CSV),
    }, indent=2))


if __name__ == "__main__":
    main()
