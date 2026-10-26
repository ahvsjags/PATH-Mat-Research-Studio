"""Grouped continuous-coverage reanalysis of the public ZnO-to-ZIF-8 dataset.

The 68 continuous ImageJ coverage values are transcribed in the Table S1 row
order of Dedecker et al. (2026), then joined to the public 68-row recipe table.
Exact recipe tuples are collapsed to one analysis unit before model fitting.
This is an exploratory, single-paper/single-material workflow analysis; it is
not a PATH-Mat frozen transfer or a causal intervention study.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "zif8_synthesis_data_source.csv"
DATA_OUT = ROOT / "zif8_synthesis_coverage_s1_v210.csv"
PRED_OUT = ROOT / "ZIF8_CONTINUOUS_COVERAGE_OOF_V210.csv"
JSON_OUT = ROOT / "ZIF8_CONTINUOUS_COVERAGE_AUDIT_V210.json"
SEED = 20261016
FOLDS = 5
REPEATS = 100
BOOTSTRAPS = 10000

# Table S1 image-derived coverage percentages, in the corresponding public
# Table S2/source CSV recipe-row order. Values are retained exactly as printed.
COVERAGE = [
    98,91,97,94,99,84,98,98,83,93,89,73,91,99,98,99,95,95,91,92,91,84,94,99,
    93,88,96,100,81,93,94,94,99,97,97,100,99,99,62,95,92,99,99,96,98,95,98,
    94,100,96,99,99,99,98,99,99,84,80,78,13,37,11,100,99,0,0,97,17
]

df0 = pd.read_csv(SOURCE)
assert len(df0) == len(COVERAGE) == 68
df0["coverage_pct"] = COVERAGE
cat = ["Solvent_1", "Solvent_2"]
num = ["Ratio", "Temperature", "Duration"]
features = cat + num
recipe = df0[features].astype(str).agg("|".join, axis=1)
df0["recipe_group"] = recipe

# Join audit: source row ordering must map each vector entry to the exact recipe
# shown on the same row in the public source snapshot. Duplicate exact recipes
# are aggregated and treated as one condition, not independent replicates.
group_counts = df0.groupby("recipe_group", sort=False).size()
group_labels = df0.groupby("recipe_group", sort=False).Quality.nunique()
assert int(group_labels.gt(1).sum()) == 0
agg = df0.groupby("recipe_group", sort=False).agg(
    Solvent_1=("Solvent_1", "first"), Solvent_2=("Solvent_2", "first"),
    Ratio=("Ratio", "first"), Temperature=("Temperature", "first"),
    Duration=("Duration", "first"), coverage_pct=("coverage_pct", "mean"),
    n_source_rows=("coverage_pct", "size"), Quality=("Quality", "first")
).reset_index()
agg.to_csv(DATA_OUT, index=False)

# Independent reconstruction check against the source High/Low labels. The
# paper's stated cutoff is >=85% coverage for High; structural-defect overrides
# are possible. We report, rather than suppress, any disagreement.
df0["coverage_label"] = np.where(df0.coverage_pct >= 85, "High", "Low")
mismatch = df0.loc[df0.coverage_label != df0.Quality,
                   ["Solvent_1", "Solvent_2", "Ratio", "Temperature", "Duration",
                    "coverage_pct", "Quality", "coverage_label"]]

unique = agg.copy()
X_all = unique[features]
y = unique.coverage_pct.to_numpy(dtype=float)
model_specs = {
    "all_process": (cat, num),
    "numeric_conditions_only": ([], num),
    "solvent_identity_only": (cat, []),
}

def make_model(cats, nums):
    transformers = []
    if cats:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cats))
    if nums:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), nums))
    return Pipeline([
        ("prep", ColumnTransformer(transformers, remainder="drop")),
        ("model", Ridge(alpha=1.0)),
    ])

pred_rows = []
repeat_summaries = {name: [] for name in [*model_specs, "training_mean"]}
for rep in range(REPEATS):
    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED + rep)
    oof = {name: np.full(len(unique), np.nan) for name in [*model_specs, "training_mean"]}
    oof_raw = {name: np.full(len(unique), np.nan) for name in [*model_specs, "training_mean"]}
    for fold, (train, test) in enumerate(splitter.split(unique)):
        y_train = y[train]
        oof_raw["training_mean"][test] = float(np.mean(y_train))
        for name, (cats, nums) in model_specs.items():
            cols = cats + nums
            model = make_model(cats, nums)
            model.fit(unique.iloc[train][cols], y_train)
            oof_raw[name][test] = model.predict(unique.iloc[test][cols])
    # Coverage is physically bounded. Preserve raw estimates, but apply the
    # fixed [0, 100] projection before calculating reported prediction metrics.
    # This bound is outcome-independent and no parameter is fit on the test fold.
    for name in oof:
        oof[name] = np.clip(oof_raw[name], 0.0, 100.0)
    for name, scores in oof.items():
        assert np.isfinite(scores).all()
        repeat_summaries[name].append({
            "repeat": rep,
            "MAE": float(mean_absolute_error(y, scores)),
            "RMSE": float(np.sqrt(mean_squared_error(y, scores))),
            "R2": float(r2_score(y, scores)),
            "Spearman": float(spearmanr(y, scores).statistic),
        })
        for idx, s in enumerate(scores):
            pred_rows.append({"repeat": rep, "row_id": int(idx),
                              "recipe_group": unique.recipe_group.iloc[idx],
                              "coverage_true_pct": float(y[idx]),
                              "coverage_pred_raw_pct": float(oof_raw[name][idx]),
                              "coverage_pred_pct": float(s), "model": name})

pred = pd.DataFrame(pred_rows)
pred.to_csv(PRED_OUT, index=False)

summary = {}
for name, rows in repeat_summaries.items():
    r = pd.DataFrame(rows)
    summary[name] = {m: {
        "median": float(r[m].median()),
        "q025_split_range": float(r[m].quantile(.025)),
        "q975_split_range": float(r[m].quantile(.975)),
        "mean": float(r[m].mean()),
    } for m in ["MAE", "RMSE", "R2", "Spearman"]}

# Paired group bootstrap on the repeat-0 OOF predictions. It is conditional on
# this fixed set of fold-fitted models and does not include refit/split variance.
boot_rng = np.random.default_rng(SEED)
boot_df = pred[pred.repeat.eq(0)]
wide = boot_df.pivot(index="row_id", columns="model", values="coverage_pred_pct")
boot_contrasts = {}
for comparator in ["training_mean", "numeric_conditions_only", "solvent_identity_only"]:
    vals = []
    for _ in range(BOOTSTRAPS):
        ix = boot_rng.integers(0, len(unique), len(unique))
        yt = y[ix]
        p_all = wide["all_process"].to_numpy()[ix]
        p_cmp = wide[comparator].to_numpy()[ix]
        vals.append((
            mean_absolute_error(yt, p_cmp) - mean_absolute_error(yt, p_all),
            np.sqrt(mean_squared_error(yt, p_cmp)) - np.sqrt(mean_squared_error(yt, p_all)),
            r2_score(yt, p_all) - r2_score(yt, p_cmp),
        ))
    a = np.asarray(vals)
    boot_contrasts[comparator] = {
        metric: {"median": float(np.median(a[:, j])),
                 "q025": float(np.quantile(a[:, j], .025)),
                 "q975": float(np.quantile(a[:, j], .975))}
        for j, metric in enumerate(["MAE_reduction_pp", "RMSE_reduction_pp", "R2_increase"])
    }

raw_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
data_sha = hashlib.sha256(DATA_OUT.read_bytes()).hexdigest()
report = {
    "analysis": "exploratory continuous ImageJ coverage prediction within one public ZIF-8 membrane study",
    "article": "Dedecker et al., ACS Appl. Mater. Interfaces 18, 12199-12211 (2026), DOI 10.1021/acsami.5c24910",
    "source_url": "https://doi.org/10.1021/acsami.5c24910",
    "source_csv_url": "https://github.com/adphan35/Small-language-models-for-ZIF-8-synthesis/blob/main/zif8_synthesis_data.csv",
    "endpoint_source": "Supporting Information Table S1 continuous ImageJ-derived coverage; values transcribed in displayed row order and joined to the associated 68-row public recipe table",
    "endpoint_definition": "percentage coverage from SEM images; source paper averaged two sample locations and used one operator for manual thresholding; not an independent phase-purity or PXRD endpoint",
    "source_csv_sha256": raw_sha,
    "derived_coverage_csv_sha256": data_sha,
    "rows": int(len(df0)),
    "unique_exact_recipe_groups": int(len(unique)),
    "duplicated_recipe_rows": int((group_counts - 1).clip(lower=0).sum()),
    "conflicting_source_class_groups": int(group_labels.gt(1).sum()),
    "coverage_min_max": [float(df0.coverage_pct.min()), float(df0.coverage_pct.max())],
    "source_class_counts": {str(k): int(v) for k, v in df0.Quality.value_counts().items()},
    "threshold_reconstruction_cutoff_pct": 85,
    "threshold_reconstruction_mismatch_rows": int(len(mismatch)),
    "threshold_reconstruction_mismatches": mismatch.to_dict(orient="records"),
    "validation": {
        "unit": "unique exact recipe tuple; duplicate source rows collapsed to their mean coverage before fitting",
        "outer_split": "5-fold shuffled KFold over unique recipe tuples, repeated 100 times",
        "seed_base": SEED,
        "models": "fixed Ridge(alpha=1), standardized numeric inputs, one-hot solvents; no tuning or feature selection; predictions projected to the physical [0,100]% coverage range before metrics, with raw values also saved",
        "comparators": ["numeric conditions only", "solvent identity only", "training-fold mean"],
        "metrics": "MAE, RMSE, R2, Spearman; split ranges describe repeated partitions, not confidence intervals",
    },
    "repeated_grouped_cv_metrics": summary,
    "prediction_bound_audit": {
        "projection": "clip predicted coverage to [0,100]%; fixed physical outcome range, not fitted",
        "raw_out_of_range_prediction_count": int(((pred.coverage_pred_raw_pct < 0) | (pred.coverage_pred_raw_pct > 100)).sum()),
        "raw_prediction_min_max": [float(pred.coverage_pred_raw_pct.min()), float(pred.coverage_pred_raw_pct.max())],
        "bounded_prediction_min_max": [float(pred.coverage_pred_pct.min()), float(pred.coverage_pred_pct.max())],
    },
    "paired_bootstrap_repeat0_oof": {
        "draws": BOOTSTRAPS,
        "unit": "unique exact recipe groups resampled with replacement",
        "caveat": "conditional on repeat-0 out-of-fold predictions and fitted fold models; excludes refit and split uncertainty",
        "all_process_improvement_vs_comparator": boot_contrasts,
    },
    "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    "prediction_file": PRED_OUT.name,
    "derived_data_file": DATA_OUT.name,
    "interpretation_boundary": "Continuous within-study coverage adds a stronger, non-thresholded outcome check for process-condition association in this single ZnO-to-ZIF-8 workflow. The same paper/source and narrow experimental domain were used; no frozen PATH-Mat weights, new-source transfer, operation-order effect, phase-purity gold standard, or causal route recommendation is established.",
}
JSON_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
