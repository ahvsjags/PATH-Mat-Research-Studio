"""Paired condition-group uncertainty for the fixed ZIF-8 grouped-CV audit.

Resamples exact recipe groups, stratified by their single source label, and
reuses the same sampled groups for each model. Intervals are conditional on the
repeat-0 out-of-fold predictions and therefore do not include model/refit or
partition uncertainty. This is an exploratory, single-publication analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
PRED = pd.read_csv(ROOT / "ZIF8_GROUPED_CV_PREDICTIONS_V208.csv")
OUT = ROOT / "ZIF8_PAIRED_GROUP_BOOTSTRAP_V208.json"
SEED = 20261016
DRAWS = 10000
MODELS = ["all_process", "solvent_identity_only", "numeric_conditions_only"]

pred = PRED[(PRED.repeat == 0) & PRED.model.isin(MODELS)].copy()
groups = sorted(pred.condition_group.unique())
base = pred[pred.model == MODELS[0]].sort_values("row_id").reset_index(drop=True)
row_ids = base.row_id.to_numpy()
y_all = base.y_true_low.to_numpy()
row_index = {int(row_id): i for i, row_id in enumerate(row_ids)}
group_labels = base.groupby("condition_group").y_true_low.first().to_dict()
group_indices = {
    g: np.array([row_index[int(i)] for i in base.loc[base.condition_group == g, "row_id"]], dtype=int)
    for g in groups
}
scores = {
    model: pred[pred.model == model].set_index("row_id").loc[row_ids].score_low.to_numpy()
    for model in MODELS
}
class_groups = {
    label: [g for g in groups if int(group_labels[g]) == label]
    for label in (0, 1)
}

def metrics(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.array([
        average_precision_score(y, s),
        roc_auc_score(y, s),
        brier_score_loss(y, s),
    ])

point = {m: metrics(y_all, scores[m]) for m in MODELS}
metric_names = ("AP_low", "AUROC_low", "Brier_low")
contrasts = {
    "all_process_minus_solvent_identity_only": ("all_process", "solvent_identity_only"),
    "all_process_minus_numeric_conditions_only": ("all_process", "numeric_conditions_only"),
}
global_prevalence = float(y_all.mean())
constant_point = np.array([
    global_prevalence,
    0.5,
    global_prevalence * (1.0 - global_prevalence),
])
point["constant_prevalence"] = constant_point
contrasts["all_process_minus_constant_prevalence"] = ("all_process", "constant_prevalence")
rng = np.random.default_rng(SEED)
draws: dict[str, list[np.ndarray]] = {k: [] for k in contrasts}
for _ in range(DRAWS):
    selected = []
    for label in (0, 1):
        selected.extend(rng.choice(class_groups[label], size=len(class_groups[label]), replace=True))
    sampled_rows = np.concatenate([group_indices[g] for g in selected])
    boot_metrics = {}
    for model in MODELS:
        boot_metrics[model] = metrics(y_all[sampled_rows], scores[model][sampled_rows])
    boot_prevalence = float(y_all[sampled_rows].mean())
    boot_metrics["constant_prevalence"] = np.array([
        boot_prevalence,
        0.5,
        boot_prevalence * (1.0 - boot_prevalence),
    ])
    for name, (a, b) in contrasts.items():
        draws[name].append(boot_metrics[a] - boot_metrics[b])

out = {
    "description": "Paired exact-condition-group bootstrap of repeat-0 OOF predictions; conditional on fitted folds, not a refit/split interval.",
    "seed": SEED,
    "bootstrap_draws": DRAWS,
    "unit": "exact solvent pair + ratio + temperature + duration condition group; stratified by group-level low-quality label",
    "n_rows_repeat0": int(pred[pred.model == MODELS[0]].shape[0]),
    "n_condition_groups": len(groups),
    "point_metrics": {
        m: {metric: float(value) for metric, value in zip(("AP_low", "AUROC_low", "Brier_low"), point[m])}
        for m in MODELS
    },
    "constant_prevalence_reference": {
        metric: float(value) for metric, value in zip(metric_names, constant_point)
    },
    "paired_contrasts": {},
    "repeated_partition_robustness": {},
    "interpretation_boundary": "Evidence that measured process inputs rank the source-defined low-coverage class within this one material system/source; not a frozen PATH-Mat transfer, cross-material generalization, phase-purity validation, or causal intervention estimate.",
}
for name, (a, b) in contrasts.items():
    arr = np.asarray(draws[name])
    out["paired_contrasts"][name] = {
        metric: {
            "point": float(point[a][j] - point[b][j]),
            "conditional_group_bootstrap_95CI": [float(np.quantile(arr[:, j], .025)), float(np.quantile(arr[:, j], .975))],
            "bootstrap_fraction_above_zero": float(np.mean(arr[:, j] > 0)),
        }
        for j, metric in enumerate(metric_names)
    }

# Repeated-partition differences pair models within each identical split seed.
rows = []
for repeat, block in PRED[PRED.model.isin(MODELS)].groupby("repeat"):
    repeat_base = block[block.model == MODELS[0]].sort_values("row_id")
    repeat_y = repeat_base.y_true_low.to_numpy()
    repeat_ids = repeat_base.row_id.to_numpy()
    per_model = {
        m: metrics(repeat_y, block[block.model == m].set_index("row_id").loc[repeat_ids].score_low.to_numpy())
        for m in MODELS
    }
    row = {"repeat": int(repeat)}
    for name, (a, b) in contrasts.items():
        if b == "constant_prevalence":
            continue
        d = per_model[a] - per_model[b]
        row[name] = {metric: float(v) for metric, v in zip(metric_names, d)}
    rows.append(row)
for name in contrasts:
    out["repeated_partition_robustness"][name] = {}
    for metric in metric_names:
        if name == "all_process_minus_constant_prevalence":
            continue
        vals = np.array([row[name][metric] for row in rows])
        out["repeated_partition_robustness"][name][metric] = {
            "median_across_50_partitions": float(np.median(vals)),
            "q025_q975_partition_range": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))],
            "fraction_partitions_positive": float(np.mean(vals > 0)),
        }

OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
