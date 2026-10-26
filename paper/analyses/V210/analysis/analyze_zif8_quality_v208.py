"""Exploratory, leakage-aware analysis of the public 68-row ZnO-to-ZIF-8 dataset.

This is an orthogonal single-system physical-quality dataset, not a PATH-Mat
zero-shot evaluation. All model choices and diagnostics are fixed in this file;
grouping is by exact experimental input tuple so duplicate condition rows never
cross train/test folds. Outcomes are parsed from the public source snapshot.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "zif8_synthesis_data_source.csv"
OUT_CSV = ROOT / "ZIF8_GROUPED_CV_PREDICTIONS_V208.csv"
OUT_JSON = ROOT / "ZIF8_GROUPED_CV_AUDIT_V208.json"
SEED = 20261016
FOLDS = 5
REPEATS = 50
BOOTSTRAPS = 1000

df = pd.read_csv(SOURCE)
required = {"Solvent_1", "Solvent_2", "Ratio", "Temperature", "Duration", "Quality"}
assert required.issubset(df.columns)
assert set(df.Quality) == {"High", "Low"}
df["low_quality"] = (df.Quality == "Low").astype(int)
cat = ["Solvent_1", "Solvent_2"]
num = ["Ratio", "Temperature", "Duration"]
all_x = cat + num
group_cols = all_x
groups = df[group_cols].astype(str).agg("|".join, axis=1)

# Exact duplicate recipes have a single class in the source; never split such
# rows across folds. They remain separate records only where the source presents
# them separately, and are not claimed to be independent experimental batches.
conflicting_duplicate_groups = int(
    df.assign(_group=groups).groupby("_group").low_quality.nunique().gt(1).sum()
)
assert conflicting_duplicate_groups == 0

def make_pipe(categorical, numeric):
    transformers = []
    if categorical:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if numeric:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    prep = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=1.0, max_iter=5000, solver="liblinear"))])

model_specs = {
    "all_process": (cat, num),
    "solvent_identity_only": (cat, []),
    "numeric_conditions_only": ([], num),
}

predictions = []
fold_metrics = []
for rep in range(REPEATS):
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + rep)
    for fold, (train, test) in enumerate(splitter.split(df[all_x], df.low_quality, groups=groups)):
        y_train = df.low_quality.iloc[train].to_numpy()
        y_test = df.low_quality.iloc[test].to_numpy()
        # prevalence-only comparator fitted on training data, no tuned threshold
        prevalence = float(y_train.mean())
        base_scores = np.full(len(test), prevalence)
        predictions.extend({
            "repeat": rep, "fold": fold, "row_id": int(i), "model": "training_prevalence",
            "y_true_low": int(df.low_quality.iloc[i]), "score_low": prevalence,
            "condition_group": str(groups.iloc[i])
        } for i in test)
        for name, (cats, nums) in model_specs.items():
            pipe = make_pipe(cats, nums)
            pipe.fit(df.iloc[train][cats + nums], y_train)
            classes = list(pipe.named_steps["model"].classes_)
            low_idx = classes.index(1)
            scores = pipe.predict_proba(df.iloc[test][cats + nums])[:, low_idx]
            predictions.extend({
                "repeat": rep, "fold": fold, "row_id": int(i), "model": name,
                "y_true_low": int(df.low_quality.iloc[i]), "score_low": float(s),
                "condition_group": str(groups.iloc[i])
            } for i, s in zip(test, scores))
            if len(np.unique(y_test)) == 2:
                fold_metrics.append({
                    "repeat": rep, "fold": fold, "model": name,
                    "n_test": int(len(test)), "low_test": int(y_test.sum()),
                    "AP_low": float(average_precision_score(y_test, scores)),
                    "AUROC_low": float(roc_auc_score(y_test, scores)),
                    "Brier_low": float(brier_score_loss(y_test, scores)),
                })

pred = pd.DataFrame(predictions)
pred.to_csv(OUT_CSV, index=False)

# Summarize each complete repeated out-of-fold prediction vector. For a
# prevalence-only score, AP equals prevalence exactly; AUROC is 0.5 when both
# labels are present. Report distributions across fixed-seed grouped partitions.
summary = {}
for name, block in pred.groupby("model", sort=False):
    rep_rows = []
    for rep, r in block.groupby("repeat"):
        y = r.y_true_low.to_numpy()
        s = r.score_low.to_numpy()
        rep_rows.append({
            "repeat": int(rep),
            "AP_low": float(average_precision_score(y, s)),
            "AUROC_low": float(roc_auc_score(y, s)),
            "Brier_low": float(brier_score_loss(y, s)),
        })
    x = pd.DataFrame(rep_rows)
    summary[name] = {metric: {
        "median": float(x[metric].median()),
        "q025": float(x[metric].quantile(.025)),
        "q975": float(x[metric].quantile(.975)),
        "mean": float(x[metric].mean()),
    } for metric in ["AP_low", "AUROC_low", "Brier_low"]}

# Explicit temperature-held-out stress test: each temperature is wholly
# excluded from fitting; the pooled metric is descriptive because score scales
# from separate training sets are not guaranteed to be calibrated alike.
temp_out = []
for temp in sorted(df.Temperature.unique()):
    test = np.flatnonzero(df.Temperature.to_numpy() == temp)
    train = np.flatnonzero(df.Temperature.to_numpy() != temp)
    if len(np.unique(df.low_quality.iloc[train])) < 2:
        continue
    pipe = make_pipe(cat, num)
    pipe.fit(df.iloc[train][all_x], df.low_quality.iloc[train])
    cls = list(pipe.named_steps["model"].classes_).index(1)
    scores = pipe.predict_proba(df.iloc[test][all_x])[:, cls]
    y = df.low_quality.iloc[test].to_numpy()
    temp_out.extend({"row_id": int(i), "held_out_temperature": float(temp),
                     "y_true_low": int(yi), "score_low": float(si)}
                    for i, yi, si in zip(test, y, scores))
temp_df = pd.DataFrame(temp_out)
if len(temp_df) and len(np.unique(temp_df.y_true_low)) == 2:
    temp_summary = {
        "n": int(len(temp_df)), "low_n": int(temp_df.y_true_low.sum()),
        "AP_low_pooled_descriptive": float(average_precision_score(temp_df.y_true_low, temp_df.score_low)),
        "AUROC_low_pooled_descriptive": float(roc_auc_score(temp_df.y_true_low, temp_df.score_low)),
        "Brier_low_pooled_descriptive": float(brier_score_loss(temp_df.y_true_low, temp_df.score_low)),
        "by_temperature": []
    }
    for temp, g in temp_df.groupby("held_out_temperature"):
        row = {"temperature": float(temp), "n": int(len(g)), "low_n": int(g.y_true_low.sum())}
        if g.y_true_low.nunique() == 2:
            row.update(AP_low=float(average_precision_score(g.y_true_low, g.score_low)),
                       AUROC_low=float(roc_auc_score(g.y_true_low, g.score_low)))
        temp_summary["by_temperature"].append(row)
else:
    temp_summary = {"n": int(len(temp_df)), "status": "no usable pooled temperature-held-out predictions"}

# Cluster-bootstrap one complete OOF repeat by exact condition group. This is
# conditional on the fold-fitted models and is reported separately from the
# repeated-partition range; it is not a full refitting/split uncertainty interval.
boot_rng = np.random.default_rng(SEED)
boot_base = pred[(pred.repeat == 0) & pred.model.isin(list(model_specs))].copy()
boot_summary = {}
group_label = df.assign(_group=groups).groupby("_group").low_quality.first()
for name, block in boot_base.groupby("model"):
    by_group = {g: x for g, x in block.groupby("condition_group", sort=False)}
    class_groups = {
        lab: [g for g in by_group if int(group_label.loc[g]) == lab]
        for lab in (0, 1)
    }
    vals = []
    for _ in range(BOOTSTRAPS):
        chosen = []
        for lab in (0, 1):
            ids = boot_rng.choice(class_groups[lab], size=len(class_groups[lab]), replace=True)
            chosen.extend(by_group[g] for g in ids)
        b = pd.concat(chosen, ignore_index=True)
        y, s = b.y_true_low.to_numpy(), b.score_low.to_numpy()
        vals.append((average_precision_score(y, s), roc_auc_score(y, s), brier_score_loss(y, s)))
    vals = np.asarray(vals)
    boot_summary[name] = {
        metric: {"q025": float(np.quantile(vals[:, j], .025)),
                 "q975": float(np.quantile(vals[:, j], .975))}
        for j, metric in enumerate(["AP_low", "AUROC_low", "Brier_low"])
    }

global_prev = float(df.low_quality.mean())
baseline = {"AP_low": global_prev, "AUROC_low": 0.5,
            "Brier_low": global_prev * (1 - global_prev)}

source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
report = {
    "analysis": "exploratory orthogonal measured-quality audit; not a PATH-Mat frozen transfer",
    "source": "https://github.com/adphan35/Small-language-models-for-ZIF-8-synthesis/blob/main/zif8_synthesis_data.csv",
    "source_commit_file_sha": "1848ce79ff50e68e0253d6e3d89310917e0a5379",
    "local_sha256": source_sha,
    "rows": int(len(df)), "class_counts": {k:int(v) for k,v in df.Quality.value_counts().items()},
    "low_quality_prevalence": float(df.low_quality.mean()),
    "unique_exact_condition_groups": int(groups.nunique()),
    "conflicting_duplicate_condition_groups": conflicting_duplicate_groups,
    "duplicate_rows_not_interpreted_as_independent_batch_replicates": True,
    "target_material": "single ZnO-to-ZIF-8 membrane system",
    "endpoint": "source-defined High/Low membrane coverage class (High >=85% coverage per public README; raw continuous coverage unavailable in CSV)",
    "validation": {"outer_split": "5-fold StratifiedGroupKFold", "group": "exact solvent pair + ratio + temperature + duration tuple", "repeats": REPEATS, "seed_base": SEED, "models": "fixed logistic regression C=1; one-hot solvents; standardized numeric fields; no hyperparameter search", "positive_class": "Low quality"},
    "metrics": summary,
    "global_prevalence_constant_baseline": baseline,
    "cluster_bootstrap_conditional_on_repeat0_oof": {
        "draws": BOOTSTRAPS,
        "unit": "exact condition group, stratified by group-level low/high class",
        "caveat": "intervals condition on the fixed repeat-0 OOF predictions and fitted fold models; they exclude fitting and partition uncertainty",
        "metrics": boot_summary
    },
    "temperature_held_out": temp_summary,
    "software": {"python": "3.11", "scikit_learn": sklearn.__version__, "pandas": pd.__version__},
    "prediction_file": OUT_CSV.name,
    "fold_metric_rows": int(len(fold_metrics)),
    "interpretation_boundary": "Provides an orthogonal test of process-to-measured-quality association within one fixed material system and one publication; it neither validates PATH-Mat weights nor establishes cross-material/source zero-shot generalization or causality."
}
OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
