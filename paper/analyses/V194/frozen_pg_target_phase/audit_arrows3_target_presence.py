#!/usr/bin/env python3
"""Stress-test frozen PATH-Mat target-phase ranking on existing ARROWS3 scores.

Uses the raw-XRD-available subset and never fits or calibrates models. The endpoint
is target-phase detection at a fixed 1 wt% cutoff, with threshold sensitivities.
This is a post-hoc cross-archive sensitivity on previously inspected outcomes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
V194 = ROOT / "PATH-Mat_Evidence_Strengthening_V194"
SOURCE = V194 / "results" / "ARROWS3_FROZEN_XRD_PREDICTIONS_INTERNAL.csv"
OUT = V194 / "results" / "frozen_pg_target_phase"
DOMAINS = ("YBCO", "NTMO", "LTOPO")
VARIANTS = ("C", "C_N", "C_N_S", "C_N_S_O")
THRESHOLDS = (0.0, 1.0, 5.0, 10.0)
N_BOOT = 10_000
SEED = 1941502


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_ap_bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], group_idx: np.ndarray,
                        n_groups: int, rng: np.random.Generator) -> tuple[dict, dict]:
    """Vectorized cluster bootstrap AP, preserving ties as score thresholds."""
    counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=N_BOOT)
    row_weights = counts[:, group_idx]
    valid = (row_weights @ y > 0) & (row_weights @ (1 - y) > 0)
    ap_draws: dict[str, np.ndarray] = {}
    for variant, raw_score in scores.items():
        # Success is ranked by negative risk, so low raw risk is visited first.
        order = np.argsort(raw_score, kind="stable")
        sorted_score = raw_score[order]
        sorted_y = y[order]
        sorted_w = row_weights[:, order]
        tp_cum = np.cumsum(sorted_w * sorted_y[None, :], axis=1)
        fp_cum = np.cumsum(sorted_w * (1 - sorted_y)[None, :], axis=1)
        # AP advances only after each distinct score; all ties enter together.
        ends = np.flatnonzero(np.r_[sorted_score[1:] != sorted_score[:-1], True])
        tp = tp_cum[:, ends]
        fp = fp_cum[:, ends]
        prior_tp = np.c_[np.zeros(len(tp)), tp[:, :-1]]
        increments = tp - prior_tp
        total_pos = tp[:, -1]
        precision = tp / np.maximum(tp + fp, 1)
        ap_draws[variant] = np.sum(increments * precision / np.maximum(total_pos[:, None], 1), axis=1)
    contrast = {
        "C_N_minus_C": ap_draws["C_N"] - ap_draws["C"],
        "C_N_S_minus_C_N": ap_draws["C_N_S"] - ap_draws["C_N"],
        "C_N_S_O_minus_C_N_S": ap_draws["C_N_S_O"] - ap_draws["C_N_S"],
        "C_N_S_O_minus_C": ap_draws["C_N_S_O"] - ap_draws["C"],
    }
    ap_summary = {v: {"ci95": np.quantile(a[valid], [.025, .975]).tolist(), "valid_draws": int(valid.sum())}
                  for v, a in ap_draws.items()}
    contrast_summary = {k: {"estimate": None, "ci95": np.quantile(a[valid], [.025, .975]).tolist(),
                             "valid_draws": int(valid.sum())} for k, a in contrast.items()}
    return ap_summary, contrast_summary


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        source_rows = list(csv.DictReader(f))
    raw = [r for r in source_rows if r["cohort"] == "raw_xrd_available"]
    output = {
        "schema_version": "pathmat_v194_frozen_arrows3_target_presence_sensitivity_v1",
        "status": "COMPLETED_POSTHOC_CROSS_ARCHIVE_SENSITIVITY_ON_PREEXISTING_FROZEN_SCORES",
        "source_file": str(SOURCE),
        "source_file_sha256": sha(SOURCE),
        "cohort": "ARROWS3 raw-XRD-available records; unchanged Lee-trained V178 scores in V193 prediction table",
        "endpoint": "reported exact target phase detected above threshold wt%; thresholds 0, 1, 5 and 10 wt%",
        "score_direction": "negative impurity-risk score ranks target-phase presence",
        "models": {},
        "bootstrap": {"draws": N_BOOT, "seed": SEED, "unit": "exact precursor-set cluster", "paired": True},
        "interpretation": [
            "This applies pre-existing frozen scores; it is not a new training or calibration run.",
            "The data and endpoints were previously inspected; this is exploratory rather than confirmatory.",
            "ARROWS3 is one experimental program represented by three target chemistries, not three independent laboratories.",
            "The source campaigns use a common mix-to-heat order, so the evaluation cannot identify a route-order effect.",
        ],
    }
    table = []
    for domain in DOMAINS:
        base = [r for r in raw if r["domain"] == domain and r["variant"] == VARIANTS[0]]
        if not base:
            output["models"][domain] = {"status": "no_rows"}
            continue
        # The CSV is emitted once per route x variant; ensure sample order agrees.
        route_key = lambda r: (r["precursor_set"], r["temperature_C"])
        row_map = {v: {route_key(r): r for r in raw if r["domain"] == domain and r["variant"] == v} for v in VARIANTS}
        keys = [route_key(r) for r in base]
        if any(set(row_map[v]) != set(keys) for v in VARIANTS):
            raise RuntimeError(f"Route alignment failed for {domain}")
        scores = {
            v: np.asarray([float(row_map[v][key]["impurity_risk_score"]) for key in keys])
            for v in VARIANTS
        }
        groups = np.asarray([r["precursor_set"] for r in base], dtype=str)
        unique = np.unique(groups)
        group_idx = np.searchsorted(unique, groups)
        ywt = np.asarray([float(r["target_phase_wt_pct"]) for r in base])
        n_rows = len(base)
        domain_result = {"n": n_rows, "precursor_set_groups": int(len(unique)), "thresholds": []}
        for threshold in THRESHOLDS:
            y = (ywt > threshold).astype(np.int8)
            if y.sum() == 0 or y.sum() == len(y):
                domain_result["thresholds"].append({"threshold_wt_pct": threshold, "n_positive": int(y.sum()), "metrics": "not_estimable"})
                continue
            ap = {v: float(average_precision_score(y, -scores[v])) for v in VARIANTS}
            auc = {v: float(roc_auc_score(y, -scores[v])) for v in VARIANTS}
            contrast_pairs = (("C_N_minus_C", "C_N", "C"), ("C_N_S_minus_C_N", "C_N_S", "C_N"),
                              ("C_N_S_O_minus_C_N_S", "C_N_S_O", "C_N_S"), ("C_N_S_O_minus_C", "C_N_S_O", "C"))
            contrast_summary = {name: {"estimate": ap[a] - ap[b]} for name, a, b in contrast_pairs}
            ap_ci, contrast_ci = None, None
            if threshold == 1.0:
                rng = np.random.default_rng(SEED + len(domain))
                ap_ci, contrast_ci = paired_ap_bootstrap(y, scores, group_idx, len(unique), rng)
                for name, a, b in contrast_pairs:
                    contrast_ci[name]["estimate"] = ap[a] - ap[b]
                contrast_summary = contrast_ci
            result = {
                "threshold_wt_pct": threshold,
                "n": n_rows,
                "positive_n": int(y.sum()),
                "prevalence": float(y.mean()),
                "ap_by_model": ap,
                "auroc_by_model": auc,
                "paired_ap_contrasts": contrast_summary,
                "ap_ci95_precursor_cluster_bootstrap": ap_ci,
            }
            domain_result["thresholds"].append(result)
            for v in VARIANTS:
                table.append({"domain": domain, "threshold_wt_pct": threshold, "n": n_rows,
                              "positive_n": int(y.sum()), "prevalence": float(y.mean()), "model": v,
                              "average_precision": ap[v], "auroc": auc[v]})
        output["models"][domain] = domain_result

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FROZEN_ARROWS3_TARGET_PHASE_RESULTS.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with (OUT / "FROZEN_ARROWS3_TARGET_PHASE_METRICS.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(json.dumps({d: {"n": output["models"][d].get("n"), "groups": output["models"][d].get("precursor_set_groups"),
                         "primary_1wt": next((x for x in output["models"][d].get("thresholds", []) if x.get("threshold_wt_pct") == 1.0), None)} for d in DOMAINS}, indent=2))


if __name__ == "__main__":
    main()
