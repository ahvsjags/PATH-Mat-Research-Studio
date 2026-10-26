#!/usr/bin/env python
"""Matched chronological-versus-order-erased signal benchmark (V161).

Reads the public Zenodo HDF5 using h5py and a restricted NumPy-only unpickler
because pandas/PyTables is not installed. No source identifiers or raw traces
are exported; only date-clustered metrics and OOF predictions are written.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
import platform
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
DATA = HERE / "perovskite_in_situ_data.h5"
PROTOCOL = HERE / "PEROVSKITE_MEASURED_CHRONOLOGY_PROTOCOL_V161.md"
CHANNELS = ["ND", "LP725", "LP780", "SP775"]
GRID_N = 32
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20261006


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Allow only NumPy array reconstruction globals used by pandas fixed HDF."""
    def find_class(self, module, name):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "scalar"): np._core.multiarray.scalar,
            ("numpy._core.multiarray", "scalar"): np._core.multiarray.scalar,
        }
        key = (module, name)
        if key not in allowed:
            raise pickle.UnpicklingError(f"blocked non-NumPy global: {module}.{name}")
        return allowed[key]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hdf(path: Path):
    with h5py.File(path, "r") as f:
        cols0 = [x.decode("utf-8") for x in f["df/block0_items"][()]]
        vals0 = f["df/block0_values"][()]
        cols1 = [x.decode("utf-8") for x in f["df/block1_items"][()]]
        vals1 = f["df/block1_values"][()]
        cols2 = [x.decode("utf-8") for x in f["df/block2_items"][()]]
        payload = bytes(f["df/block2_values"][0])
    obj = RestrictedNumpyUnpickler(io.BytesIO(payload)).load()
    if not isinstance(obj, np.ndarray) or obj.dtype != object or obj.ndim != 2:
        raise ValueError("Unexpected object-array layout in source HDF5")
    if obj.shape[0] != vals0.shape[0] or obj.shape[1] != len(cols2):
        raise ValueError("HDF5 block row/column mismatch")
    num = {c: vals0[:, i].astype(float) for i, c in enumerate(cols0)}
    num.update({c: vals1[:, i] for i, c in enumerate(cols1)})
    obj_cols = {c: obj[:, i] for i, c in enumerate(cols2)}
    return num, obj_cols


def normalized_trace(row):
    grid = np.linspace(0.0, 1.0, GRID_N)
    sampled = []
    for channel in CHANNELS:
        raw = np.asarray(row[channel], dtype=float).reshape(-1)
        if len(raw) < 2 or not np.isfinite(raw).all():
            return None
        raw_t = np.linspace(0.0, 1.0, len(raw))
        sampled.append(np.interp(grid, raw_t, raw))
    return np.concatenate(sampled)


def build_cohort(num, obj, selector):
    y0 = num["PCE_backward"]
    dates0 = num["date"].astype(int)
    ratio = num["ratio"]
    molarity = num["molarity"]
    duration = num["evac_duration"]
    finite_meta = np.isfinite(ratio) & np.isfinite(molarity) & np.isfinite(duration)
    if selector == "full":
        mask = np.isfinite(y0) & finite_meta
    elif selector == "source_protocol":
        mask = (
            np.isfinite(y0) & finite_meta & (dates0 > 20230724)
            & (ratio == 1.0) & (molarity == 0.67) & (y0 > 1.0)
        )
    else:
        raise ValueError(selector)

    X_order, X_bag, y, dates, row_ids = [], [], [], [], []
    dropped = 0
    for i in np.flatnonzero(mask):
        row = {ch: obj[ch][i] for ch in CHANNELS}
        trace = normalized_trace(row)
        if trace is None:
            dropped += 1
            continue
        # The sorted control is generated from the exact same sampled values.
        bag = np.concatenate([np.sort(trace[j * GRID_N:(j + 1) * GRID_N]) for j in range(len(CHANNELS))])
        chron = trace
        if not np.array_equal(np.sort(chron.reshape(len(CHANNELS), GRID_N), axis=1), bag.reshape(len(CHANNELS), GRID_N)):
            raise AssertionError("Order-erased control does not preserve sampled multiset")
        meta = np.asarray([ratio[i], molarity[i], duration[i]], dtype=float)
        X_order.append(np.concatenate([meta, chron]))
        X_bag.append(np.concatenate([meta, bag]))
        y.append(float(y0[i]))
        dates.append(int(dates0[i]))
        row_ids.append(int(i))

    return {
        "X_order": np.asarray(X_order, dtype=float),
        "X_bag": np.asarray(X_bag, dtype=float),
        "y": np.asarray(y, dtype=float),
        "dates": np.asarray(dates, dtype=int),
        "row_ids": np.asarray(row_ids, dtype=int),
        "pre_trace_rows": int(mask.sum()),
        "dropped_invalid_trace_rows": dropped,
    }


def make_model(name):
    if name == "ExtraTrees_primary":
        return ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            max_features=1.0,
            random_state=20261006,
            n_jobs=1,
        )
    if name == "Ridge_sensitivity":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    raise ValueError(name)


def leave_one_date_out(X, y, dates, model_name):
    groups = np.unique(dates)
    pred = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    for date in groups:
        test = np.flatnonzero(dates == date)
        train = np.flatnonzero(dates != date)
        model = make_model(model_name)
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
        fold_rows.append({
            "date": int(date),
            "n_test": int(len(test)),
            "mae": float(mean_absolute_error(y[test], pred[test])),
            "rmse": float(mean_squared_error(y[test], pred[test]) ** 0.5),
            "delta_mae_sorted_minus_chron": None,
        })
    if not np.isfinite(pred).all():
        raise RuntimeError("Missing leave-one-date-out predictions")
    return pred, fold_rows


def date_deltas(y, pred_bag, pred_order, dates):
    rows = []
    for date in np.unique(dates):
        idx = dates == date
        rows.append({
            "date": int(date),
            "n_rows": int(idx.sum()),
            "mae_sorted": float(mean_absolute_error(y[idx], pred_bag[idx])),
            "mae_chronological": float(mean_absolute_error(y[idx], pred_order[idx])),
            "delta_mae_sorted_minus_chronological": float(
                mean_absolute_error(y[idx], pred_bag[idx]) - mean_absolute_error(y[idx], pred_order[idx])
            ),
        })
    return rows


def bootstrap_by_date(y, pred_bag, pred_order, dates, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(dates)
    members = {d: np.flatnonzero(dates == d) for d in uniq}
    deltas = np.empty(BOOTSTRAPS, dtype=float)
    for b in range(BOOTSTRAPS):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[d] for d in sampled])
        deltas[b] = mean_absolute_error(y[idx], pred_bag[idx]) - mean_absolute_error(y[idx], pred_order[idx])
    return float(np.mean(deltas)), np.quantile(deltas, [0.025, 0.975]).tolist(), deltas


def exact_sign_test(date_table):
    ds = [r["delta_mae_sorted_minus_chronological"] for r in date_table]
    n = len(ds)
    pos = sum(d > 0 for d in ds)
    neg = sum(d < 0 for d in ds)
    # One-sided sign test under independent date-level signs, positive favors chronology.
    from math import comb
    p_one = sum(comb(n, k) for k in range(pos, n + 1)) / (2 ** n)
    p_two = min(1.0, 2.0 * min(p_one, sum(comb(n, k) for k in range(0, neg + 1)) / (2 ** n)))
    return {"n_dates": n, "dates_favoring_chronology": pos, "dates_favoring_sorted": neg,
            "one_sided_exact_sign_p": p_one, "two_sided_exact_sign_p": p_two}


def analyze(cohort_name, cohort, model_name):
    y, dates = cohort["y"], cohort["dates"]
    bag_pred, _ = leave_one_date_out(cohort["X_bag"], y, dates, model_name)
    order_pred, _ = leave_one_date_out(cohort["X_order"], y, dates, model_name)
    date_table = date_deltas(y, bag_pred, order_pred, dates)
    mean_delta, ci, boot = bootstrap_by_date(y, bag_pred, order_pred, dates)
    return {
        "cohort": cohort_name,
        "model": model_name,
        "n_rows": int(len(y)),
        "n_dates": int(len(np.unique(dates))),
        "n_features_each_representation": int(cohort["X_bag"].shape[1]),
        "mae_sorted": float(mean_absolute_error(y, bag_pred)),
        "mae_chronological": float(mean_absolute_error(y, order_pred)),
        "delta_mae_sorted_minus_chronological": float(mean_absolute_error(y, bag_pred) - mean_absolute_error(y, order_pred)),
        "rmse_sorted": float(mean_squared_error(y, bag_pred) ** 0.5),
        "rmse_chronological": float(mean_squared_error(y, order_pred) ** 0.5),
        "r2_sorted": float(r2_score(y, bag_pred)),
        "r2_chronological": float(r2_score(y, order_pred)),
        "date_cluster_bootstrap_mean_delta": mean_delta,
        "date_cluster_bootstrap_ci95": ci,
        "date_sign_test": exact_sign_test(date_table),
        "date_specific": date_table,
        "oof_predictions": {"bag": bag_pred, "order": order_pred},
    }


def main():
    t0 = time.time()
    num, obj = load_hdf(DATA)
    cohorts = {
        "full_finite_PCE": build_cohort(num, obj, "full"),
        "source_protocol_sensitivity": build_cohort(num, obj, "source_protocol"),
    }
    output = {
        "dataset": "Laufer et al. in-situ photoluminescence dataset for blade-coated perovskite photovoltaics",
        "zenodo_doi": "10.5281/zenodo.14609789",
        "source_paper_doi": "10.1039/D4EE03445G",
        "license": "CC BY 4.0",
        "data_sha256": sha256(DATA),
        "protocol_sha256": sha256(PROTOCOL),
        "analysis_script_sha256": sha256(Path(__file__)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "raw_rows": int(len(num["date"])),
        "raw_unique_dates": int(len(np.unique(num["date"]))),
        "raw_unique_expIDs": int(len(np.unique(obj["expID"]))),
        "channels": CHANNELS,
        "grid_points_per_channel": GRID_N,
        "primary_split": "Leave-one-acquisition-date-out; no date appears in both train and test.",
        "primary_metric": "PCE_backward MAE; positive sorted-minus-chronological delta favors chronology.",
        "primary_control": "Per-channel sorted version of the exact same 32 interpolated values; same covariates.",
        "models": {},
        "cohort_flow": {},
        "runtime_seconds": None,
    }
    for name, cohort in cohorts.items():
        output["cohort_flow"][name] = {
            "n_rows_after_metadata_filter_before_trace_check": cohort["pre_trace_rows"],
            "n_rows_dropped_for_invalid_trace": cohort["dropped_invalid_trace_rows"],
            "n_rows_analyzed": int(len(cohort["y"])),
            "n_dates": int(len(np.unique(cohort["dates"]))),
            "n_expIDs": int(len(np.unique(obj["expID"][cohort["row_ids"]]))),
            "date_rows": {str(int(d)): int(np.sum(cohort["dates"] == d)) for d in np.unique(cohort["dates"])},
            "pce_backward_summary": [float(x) for x in np.quantile(cohort["y"], [0, .25, .5, .75, 1])],
        }
        output["models"][name] = {}
        for model_name in ["ExtraTrees_primary", "Ridge_sensitivity"]:
            result = analyze(name, cohort, model_name)
            preds = result.pop("oof_predictions")
            output["models"][name][model_name] = result
            if model_name == "ExtraTrees_primary":
                ppath = HERE / f"PEROVSKITE_OOF_{name}_V161.csv"
                with ppath.open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=["source_row_index", "date", "PCE_backward", "pred_sorted_control", "pred_chronological"])
                    writer.writeheader()
                    for j, ri in enumerate(cohort["row_ids"]):
                        writer.writerow({"source_row_index": int(ri), "date": int(cohort["dates"][j]),
                                         "PCE_backward": float(cohort["y"][j]),
                                         "pred_sorted_control": float(preds["bag"][j]),
                                         "pred_chronological": float(preds["order"][j])})
    output["runtime_seconds"] = round(time.time() - t0, 2)
    out_json = HERE / "PEROVSKITE_MEASURED_CHRONOLOGY_RESULTS_V161.json"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for cohort, models in output["models"].items():
        for model, r in models.items():
            rows.append({
                "cohort": cohort, "model": model, "n_rows": r["n_rows"], "n_dates": r["n_dates"],
                "mae_sorted": r["mae_sorted"], "mae_chronological": r["mae_chronological"],
                "delta_mae_sorted_minus_chronological": r["delta_mae_sorted_minus_chronological"],
                "bootstrap_ci95_low": r["date_cluster_bootstrap_ci95"][0],
                "bootstrap_ci95_high": r["date_cluster_bootstrap_ci95"][1],
                "dates_favoring_chronology": r["date_sign_test"]["dates_favoring_chronology"],
                "dates_favoring_sorted": r["date_sign_test"]["dates_favoring_sorted"],
                "exact_sign_test_two_sided_p": r["date_sign_test"]["two_sided_exact_sign_p"],
                "rmse_sorted": r["rmse_sorted"], "rmse_chronological": r["rmse_chronological"],
                "r2_sorted": r["r2_sorted"], "r2_chronological": r["r2_chronological"],
            })
    out_csv = HERE / "Supplementary_Data_22_Perovskite_Measured_Chronology_V161.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    date_rows = []
    for cohort, models in output["models"].items():
        for model, r in models.items():
            for dr in r["date_specific"]:
                date_rows.append({"cohort": cohort, "model": model, **dr})
    out_date = HERE / "Supplementary_Data_23_Perovskite_Date_Specific_V161.csv"
    with out_date.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(date_rows[0]))
        writer.writeheader()
        writer.writerows(date_rows)
    print(json.dumps({
        "raw_rows": output["raw_rows"],
        "cohort_flow": output["cohort_flow"],
        "models": {c: {m: {k: v for k, v in r.items() if k not in {"date_specific"}}
                       for m, r in ms.items()} for c, ms in output["models"].items()},
        "runtime_seconds": output["runtime_seconds"],
        "outputs": [str(out_json), str(out_csv), str(out_date)],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
