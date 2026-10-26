"""Reproduce a static, retrospective candidate-prioritization replay on AIRES.

This is a source-native auxiliary analysis, not a PATH-Mat frozen-weight test.
The public labels were inspected before this analysis and therefore the replay
is exploratory. Unknown labels remain unknown and are excluded from metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261016
BOOTSTRAPS = 10_000
TIE_REPLICATES = 1_000
RANDOM_POLICY_REPLICATES = 50_000
BUDGETS = [10, 25, 50, 100, 200, 400]
EXPECTED_SHA256 = "1a0aa2ef26aeeff0150e4d1d24d49f9f8bb6d1c5097a99fd8b78e583fd9043d0"
EXPECTED_MD5 = "e260eb1ebcf2b507658517cb0cb0c055"
EXPECTED_SIZE = 2_162_393


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discovery_curve(y: np.ndarray, groups: np.ndarray, scores: np.ndarray | None,
                    budgets: list[int], rng: np.random.Generator) -> dict[int, int]:
    """Static row ranking; after a hit, later trials for that linker are skipped."""
    if scores is None:
        order = rng.permutation(len(y))
    else:
        order = np.lexsort((rng.random(len(y)), -scores))
    discovered: set[str] = set()
    attempts = hits = 0
    result: dict[int, int] = {}
    for idx in order:
        linker = str(groups[idx])
        if linker in discovered:
            continue
        attempts += 1
        if int(y[idx]) == 1:
            hits += 1
            discovered.add(linker)
        if attempts in budgets:
            result[attempts] = hits
        if attempts >= max(budgets):
            break
    for budget in budgets:
        result.setdefault(budget, hits)
    return result


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    n_top5 = max(1, int(np.ceil(.05 * len(y))))
    n_top10 = max(1, int(np.ceil(.10 * len(y))))
    order = np.argsort(-p, kind="stable")
    prevalence = float(np.mean(y))
    return {
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "prevalence": prevalence,
        "average_precision": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "precision_top_5pct": float(np.mean(y[order[:n_top5]])),
        "lift_top_5pct": float(np.mean(y[order[:n_top5]]) / prevalence),
        "recall_top_5pct": float(np.sum(y[order[:n_top5]]) / np.sum(y)),
        "precision_top_10pct": float(np.mean(y[order[:n_top10]])),
        "lift_top_10pct": float(np.mean(y[order[:n_top10]]) / prevalence),
        "recall_top_10pct": float(np.sum(y[order[:n_top10]]) / np.sum(y)),
    }


def paired_linker_bootstrap(y: np.ndarray, linker_ids: np.ndarray,
                            score_map: dict[str, np.ndarray]) -> dict:
    linkers = np.unique(linker_ids)
    rows_by_linker = {linker: np.flatnonzero(linker_ids == linker) for linker in linkers}
    rng = np.random.default_rng(SEED)
    comparisons = [
        ("combined_minus_chemistry", "chemistry_plus_conditions", "chemistry_only"),
        ("combined_minus_conditions", "chemistry_plus_conditions", "conditions_only"),
    ]
    result = {}
    for name, a, b in comparisons:
        values = []
        attempts = 0
        while len(values) < BOOTSTRAPS and attempts < BOOTSTRAPS * 2:
            attempts += 1
            sampled = rng.choice(linkers, size=len(linkers), replace=True)
            idx = np.concatenate([rows_by_linker[linker] for linker in sampled])
            if len(np.unique(y[idx])) < 2:
                continue
            values.append(average_precision_score(y[idx], score_map[a][idx]) -
                          average_precision_score(y[idx], score_map[b][idx]))
        arr = np.asarray(values)
        result[name] = {
            "point_delta_average_precision": float(average_precision_score(y, score_map[a]) -
                                                     average_precision_score(y, score_map[b])),
            "cluster_unit": "held-out linker, retaining all labeled reaction conditions as a block",
            "bootstrap_replicates": int(len(arr)),
            "percentile_95_ci": [float(np.quantile(arr, .025)), float(np.quantile(arr, .975))],
            "fraction_bootstrap_at_or_below_zero": float(np.mean(arr <= 0)),
        }
    return result


def top_recommendations(test: pd.DataFrame, y: np.ndarray, linker_ids: np.ndarray,
                        scores: np.ndarray, seed: int) -> pd.DataFrame:
    """Emit one deterministic 10-trial static replay, preserving policy updates."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(scores)), -scores))
    discovered: set[str] = set()
    chosen: list[int] = []
    for idx in order:
        linker = str(linker_ids[idx])
        if linker in discovered:
            continue
        chosen.append(int(idx))
        if int(y[idx]) == 1:
            discovered.add(linker)
        if len(chosen) == 10:
            break
    cond = [c for c in test.columns if c.startswith("rxn_")]
    out = test.iloc[chosen][["LinkerCanonicalSmiles", *cond]].copy().reset_index(drop=True)
    out.insert(0, "trial_rank", np.arange(1, len(out) + 1))
    out["predicted_success_probability"] = scores[chosen]
    out["observed_source_label"] = y[chosen]
    out["interpretation"] = ["success" if x == 1 else "non-success" for x in y[chosen]]
    out["policy_note"] = "static score order; skip further trials for a linker after first observed success"
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to the source AIRES_ML_discovery.zip archive")
    args = parser.parse_args()
    input_path = args.input.resolve()
    if input_path.stat().st_size != EXPECTED_SIZE:
        raise ValueError(f"Unexpected source archive size: {input_path.stat().st_size}")
    if sha256(input_path) != EXPECTED_SHA256 or md5(input_path) != EXPECTED_MD5:
        raise ValueError("Source archive digest mismatch; expected the cited Zenodo ML_discovery.zip file")
    with zipfile.ZipFile(input_path) as archive:
        train = pd.read_csv(archive.open("ML_discovery/data/ZIF_training_set.csv"))
        all_test = pd.read_csv(archive.open("ML_discovery/data/ZIF_test_set.csv"))

    target = "Crystal_clf"
    linker_col = "LinkerCanonicalSmiles"
    unknown_n = int((all_test[target] == -1).sum())
    test = all_test.loc[all_test[target].isin([0, 1])].copy().reset_index(drop=True)
    feature_cols = [c for c in train.columns if c.startswith("feat_") or c.startswith("rxn_")]
    chemistry_cols = [c for c in feature_cols if c.startswith("feat_")]
    condition_cols = [c for c in feature_cols if c.startswith("rxn_")]
    if train.columns.tolist() != all_test.columns.tolist():
        raise ValueError("Training and test schema differ")
    if set(train[linker_col]).intersection(set(test[linker_col])):
        raise ValueError("Linker overlap between author-provided training and test split")
    if train.groupby(linker_col).size().nunique() != 1 or train.groupby(linker_col).size().iloc[0] != 336:
        raise ValueError("Unexpected training grid: expected 336 trials per linker")
    if all_test.groupby(linker_col).size().nunique() != 1 or all_test.groupby(linker_col).size().iloc[0] != 36:
        raise ValueError("Unexpected test grid: expected 36 trials per linker")

    feature_sets = {
        "chemistry_only": chemistry_cols,
        "conditions_only": condition_cols,
        "chemistry_plus_conditions": chemistry_cols + condition_cols,
    }
    y_train = train[target].astype(int).to_numpy()
    y = test[target].astype(int).to_numpy()
    linker_ids = test[linker_col].astype(str).to_numpy()
    score_map: dict[str, np.ndarray] = {}
    for name, cols in feature_sets.items():
        model = RandomForestClassifier(random_state=SEED, n_jobs=-1)
        model.fit(train[cols].to_numpy(dtype=float), y_train)
        score_map[name] = model.predict_proba(test[cols].to_numpy(dtype=float))[:, 1]

    # Low-budget discovery curves. Random rows receive the same remove-after-hit rule.
    policy: dict[str, dict] = {}
    for name, p in score_map.items():
        rng = np.random.default_rng(SEED + 11)
        draws = {budget: [] for budget in BUDGETS}
        for _ in range(TIE_REPLICATES):
            curve = discovery_curve(y, linker_ids, p, BUDGETS, rng)
            for budget in BUDGETS:
                draws[budget].append(curve[budget])
        policy[name] = {
            str(budget): {
                "mean_unique_successful_linkers": float(np.mean(draws[budget])),
                "q025": float(np.quantile(draws[budget], .025)),
                "q975": float(np.quantile(draws[budget], .975)),
            } for budget in BUDGETS
        }
    rng_random = np.random.default_rng(SEED + 12)
    random_draws = {budget: [] for budget in BUDGETS}
    for _ in range(RANDOM_POLICY_REPLICATES):
        curve = discovery_curve(y, linker_ids, None, BUDGETS, rng_random)
        for budget in BUDGETS:
            random_draws[budget].append(curve[budget])
    policy["random_exploration"] = {
        str(budget): {
            "mean_unique_successful_linkers": float(np.mean(random_draws[budget])),
            "q025": float(np.quantile(random_draws[budget], .025)),
            "q975": float(np.quantile(random_draws[budget], .975)),
        } for budget in BUDGETS
    }
    for name in score_map:
        for budget in BUDGETS:
            avg = policy[name][str(budget)]["mean_unique_successful_linkers"]
            policy[name][str(budget)]["ratio_to_random_mean"] = (
                float(avg / policy["random_exploration"][str(budget)]["mean_unique_successful_linkers"])
                if policy["random_exploration"][str(budget)]["mean_unique_successful_linkers"] else None)
            policy[name][str(budget)]["random_tail_frequency_at_least_model_mean"] = float(
                np.mean(np.asarray(random_draws[budget]) >= avg))

    top10 = top_recommendations(test, y, linker_ids, score_map["chemistry_plus_conditions"], SEED + 13)
    top10.to_csv(OUT / "V257_AIRES_TOP10_RECOMMENDATIONS.csv", index=False)
    result = {
        "analysis": "Source-native static candidate-prioritization replay on the AIRES author-provided unseen-linker split",
        "status": "exploratory_retrospective; public source split and outcomes inspected; not a PATH-Mat frozen-weight validation or prospective experiment",
        "source_article": "Rong et al., Nature Synthesis 5, 409–419 (2026), DOI 10.1038/s44160-025-00939-9",
        "source_data_record": "Zenodo DOI 10.5281/zenodo.17489399; linked from article data availability",
        "input": {"basename": input_path.name, "md5": md5(input_path),
                  "sha256": sha256(input_path), "size_bytes": input_path.stat().st_size},
        "protocol": {
            "training_linkers": int(train[linker_col].nunique()),
            "training_rows": int(len(train)),
            "heldout_linkers": int(test[linker_col].nunique()),
            "heldout_rows_all": int(len(all_test)),
            "heldout_labeled_rows": int(len(test)),
            "unknown_label_rows_excluded_without_relabeling": unknown_n,
            "positive_labeled_rows": int(y.sum()),
            "labeled_test_prevalence": float(y.mean()),
            "endpoint": "source-defined Crystal_clf; -1 remains unknown; 1 marks source-defined successful ZIF single-crystal formation",
            "model": "RandomForestClassifier defaults with fixed random_state=20261016; no hyperparameter tuning",
            "feature_sets": feature_sets,
            "metrics_uncertainty_unit": "held-out linker; all labeled reaction conditions remain grouped",
            "discovery_policy": "rank candidate rows by fixed predicted probability; count each attempted reaction toward budget; after a successful linker, skip its remaining rows; failed linkers remain eligible at other conditions; random ties",
            "limitations": [
                "public test outcomes were inspected before the present reanalysis; exploratory, not confirmatory",
                "same publication and experimental platform in train and test; unseen-linker holdout, not independent-source transfer",
                "endpoint and features differ from PATH-Mat primary task; no PATH-Mat weights are evaluated",
                "no synthesis is initiated by this replay; positive rows are previously published outcomes, not new discoveries",
                "discovery advantage is concentrated at the 10-trial budget and is not sustained at larger budgets",
            ],
        },
        "metrics": {name: metrics(y, scores) for name, scores in score_map.items()},
        "paired_heldout_linker_bootstrap": paired_linker_bootstrap(y, linker_ids, score_map),
        "budgeted_discovery_replay": policy,
        "deterministic_top10_replay": {
            "tie_break_seed": SEED + 13,
            "unique_successful_linkers": int(top10.loc[top10["observed_source_label"] == 1, "LinkerCanonicalSmiles"].nunique()),
            "rows": len(top10),
            "output_csv": "V257_AIRES_TOP10_RECOMMENDATIONS.csv",
        },
        "software": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    import sklearn
    result["software"]["scikit_learn"] = sklearn.__version__
    (OUT / "V257_AIRES_CANDIDATE_DISCOVERY_AUDIT.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": result["metrics"],
                      "combined_minus_chemistry": result["paired_heldout_linker_bootstrap"]["combined_minus_chemistry"],
                      "budget10": {k: v["10"] for k, v in policy.items()},
                      "top10_success_linkers": result["deterministic_top10_replay"]["unique_successful_linkers"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
