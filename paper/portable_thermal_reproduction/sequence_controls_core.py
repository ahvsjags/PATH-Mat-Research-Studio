"""Exploratory LOPO controls for chronology in measured furnace profiles.

This reuses the V148 fixed cohort, endpoint, feature construction, model family,
and leave-one-precursor-out folds. It adds direction reversal and per-record
random permutations of the *same 32 measured values*. This is a sensitivity
analysis on an already inspected archive, not a confirmatory external test.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
V148_DIR = ROOT / "PATH-Mat_Evidence_Strengthening_V148"
sys.path.insert(0, str(V148_DIR))
import analysis_core as v

OUT = pathlib.Path(__file__).resolve().parent
N_PERM = 10
BOOTSTRAPS = 3000
SEED = 15029


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def extract_rows():
    ledger = json.loads(v.LEDGER.read_text(encoding="utf-8"))
    rows = []
    for s in ledger["samples"]:
        case = v.active_case(s)
        if case is None:
            continue
        ver = case.get("verification") or {}
        if ver.get("is_accepted") is not True or ver.get("human_quality_score") not in (1, 2):
            continue
        pm = v.phase_metrics(s, case)
        prof = v.get_profile(s)
        if pm is None or not s.get("target_compound"):
            continue
        precs = s.get("precursors") or []
        preforms = [v.reduced(p.get("formula_clean") or p.get("formula")) for p in precs]
        preforms = sorted(set(x for x in preforms if x))
        if len(preforms) < 2:
            continue
        target = v.frac(s.get("target_compound"))
        pf = [v.frac(p.get("formula_clean") or p.get("formula")) for p in precs]
        pre = np.sum(pf, axis=0) if pf else np.zeros(len(v.ELEMENTS))
        pre = pre / (pre.sum() + 1e-12)
        syn = s.get("synthesis") or {}
        cond = [syn.get("heating_temperature"), syn.get("heating_time")]
        if prof:
            ordered, sorted_values, summary, _ = prof
        else:
            ordered = np.full(32, np.nan)
            sorted_values = np.full(32, np.nan)
            summary = np.full(12, np.nan)
        rows.append(
            {
                "group": v.reduced(s.get("target_compound")) or str(s.get("target_compound")),
                "precs": preforms,
                "y": float(pm[0]),
                "base": np.r_[target, pre, v.input_chemistry(s), cond, summary].astype(float),
                "ordered": np.asarray(ordered, dtype=float),
                "sorted": np.asarray(sorted_values, dtype=float),
            }
        )
    return rows


def make_folds(rows):
    folds = []
    all_idx = np.arange(len(rows))
    for precursor in sorted({p for r in rows for p in r["precs"]}):
        test = np.asarray([i for i, r in enumerate(rows) if precursor in r["precs"]], dtype=int)
        test_groups = {rows[i]["group"] for i in test}
        train = np.asarray(
            [i for i in all_idx if precursor not in rows[i]["precs"] and rows[i]["group"] not in test_groups],
            dtype=int,
        )
        if len(test) and len(train) >= 40:
            folds.append({"precursor": precursor, "train": train, "test": test})
    return folds


def build_variants(rows, n_perm):
    base = np.vstack([r["base"] for r in rows])
    actual = np.vstack([r["ordered"] for r in rows])
    sorted_values = np.vstack([r["sorted"] for r in rows])
    variants = {
        "sorted": np.c_[base, sorted_values],
        "ordered_forward": np.c_[base, actual],
        "ordered_reversed": np.c_[base, actual[:, ::-1]],
    }
    # Each row's observed temperatures are permuted independently. Each replicate
    # preserves that row's exact empirical temperature values and missingness.
    rng = np.random.default_rng(SEED)
    for j in range(n_perm):
        permuted = np.empty_like(actual)
        for i, curve in enumerate(actual):
            if np.isfinite(curve).all():
                permuted[i] = curve[rng.permutation(curve.size)]
            else:
                permuted[i] = curve
        variants[f"random_permutation_{j+1:02d}"] = np.c_[base, permuted]
    return variants


def estimator(name):
    if name == "ridge":
        model = Ridge(alpha=10.0)
    else:
        model = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=1,
            random_state=v.SEED,
        )
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), model)


def group_bootstrap_gain(y, groups, pred_control, pred_ordered, seed):
    unique = np.asarray(sorted(set(groups)), dtype=object)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    vals = np.empty(BOOTSTRAPS, dtype=float)
    for b in range(BOOTSTRAPS):
        draw = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([index[g] for g in draw])
        vals[b] = mean_absolute_error(y[ix], pred_control[ix]) - mean_absolute_error(y[ix], pred_ordered[ix])
    return [float(x) for x in np.quantile(vals, [0.025, 0.975])]

