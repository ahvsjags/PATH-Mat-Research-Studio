#!/usr/bin/env python
"""Fixed-cutoff retrieval audit for the already frozen Precursor Genome scores.

No model fitting, score calibration, or cutoff selection is performed here.
Boundary ties are handled fractionally so results do not depend on row ordering.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
INFILE = ROOT / "results" / "frozen_pg_target_phase" / "FROZEN_PG_TARGET_PHASE_ROW_AUDIT_INTERNAL.csv"
OUTJSON = ROOT / "results" / "frozen_pg_target_phase" / "FROZEN_PG_TARGET_RETRIEVAL_RESULTS.json"
OUTCSV = ROOT / "results" / "frozen_pg_target_phase" / "FROZEN_PG_TARGET_RETRIEVAL_SUMMARY.csv"
SEED = 1_941_502
DRAWS = 10_000
FRACTIONS = (0.05, 0.10, 0.20)
MODELS = ("B0", "B1", "B2")


def top_fraction_metrics(y: np.ndarray, score: np.ndarray, weight: np.ndarray, frac: float) -> tuple[float, float]:
    """Return weighted precision and recall at frac, allocating cutoff ties fractionally."""
    total = float(weight.sum())
    positives = float(np.dot(y, weight))
    budget = frac * total
    # Aggregate equal scores first, preserving the expected result for ties.
    unique_score, inverse = np.unique(score, return_inverse=True)
    group_w = np.bincount(inverse, weights=weight, minlength=len(unique_score))[::-1]
    group_hits = np.bincount(inverse, weights=weight * y, minlength=len(unique_score))[::-1]
    cumulative = np.cumsum(group_w)
    boundary = min(int(np.searchsorted(cumulative, budget, side="left")), len(group_w) - 1)
    weight_before = float(cumulative[boundary - 1]) if boundary else 0.0
    hits_before = float(group_hits[:boundary].sum())
    take_at_boundary = max(0.0, budget - weight_before)
    boundary_hit_rate = float(group_hits[boundary] / group_w[boundary]) if group_w[boundary] else 0.0
    hits = hits_before + take_at_boundary * boundary_hit_rate
    precision = hits / budget if budget else float("nan")
    recall = hits / positives if positives else float("nan")
    return precision, recall


def main() -> None:
    df = pd.read_csv(INFILE)
    y = (df["target_phase_wt_pct"].to_numpy(float) > 1.0).astype(int)
    components = df["dependency_component"].astype(str).to_numpy()
    unique_components, comp_idx = np.unique(components, return_inverse=True)
    ncomp = len(unique_components)
    scores = {m: -df[f"{m.lower()}_probability"].to_numpy(float) for m in MODELS}
    base = {m: {"average_precision": float(average_precision_score(y, s))} for m, s in scores.items()}
    for m in MODELS:
        base[m]["top_fraction"] = {}
        for frac in FRACTIONS:
            p, r = top_fraction_metrics(y, scores[m], np.ones(len(y)), frac)
            base[m]["top_fraction"][str(frac)] = {"precision": p, "recall": r, "lift": p / float(y.mean())}

    rng = np.random.default_rng(SEED)
    boot = {f"{m1}-{m0}": {str(frac): [] for frac in FRACTIONS} for m1, m0 in (("B2", "B0"), ("B2", "B1"))}
    ap_boot = {k: [] for k in boot}
    for _ in range(DRAWS):
        sampled = rng.integers(0, ncomp, ncomp)
        freq = np.bincount(sampled, minlength=ncomp)
        w = freq[comp_idx].astype(float)
        if np.dot(w, y) == 0 or np.dot(w, 1-y) == 0:
            continue
        aps = {m: float(average_precision_score(y, s, sample_weight=w)) for m, s in scores.items()}
        for key, (m1, m0) in zip(boot, (("B2", "B0"), ("B2", "B1"))):
            ap_boot[key].append(aps[m1] - aps[m0])
        for frac in FRACTIONS:
            ps = {m: top_fraction_metrics(y, s, w, frac)[0] for m, s in scores.items()}
            for key, (m1, m0) in zip(boot, (("B2", "B0"), ("B2", "B1"))):
                boot[key][str(frac)].append(ps[m1] - ps[m0])

    contrasts = {}
    for key in boot:
        contrasts[key] = {
            "average_precision_difference": float(base[key.split("-")[0]]["average_precision"] - base[key.split("-")[1]]["average_precision"]),
            "average_precision_difference_ci95": np.quantile(ap_boot[key], [0.025, 0.975]).tolist(),
            "top_fraction_precision_difference": {
                f: {"estimate": float(base[key.split("-")[0]]["top_fraction"][f]["precision"] - base[key.split("-")[1]]["top_fraction"][f]["precision"]),
                    "ci95": np.quantile(vals, [0.025, 0.975]).tolist()}
                for f, vals in boot[key].items()
            },
        }
    result = {
        "status": "POSTHOC_FIXED_CUTOFF_RETRIEVAL_AUDIT_ON_PREEXISTING_FROZEN_SCORES",
        "endpoint": "source-designated active human-accepted target phase present above 1 wt%",
        "n": int(len(df)), "positive_n": int(y.sum()), "prevalence": float(y.mean()),
        "components": int(ncomp), "draws": DRAWS, "seed": SEED,
        "score_semantics": "negative frozen impurity-risk score; higher ranks target-phase detections earlier",
        "cutoffs": list(FRACTIONS), "boundary_ties": "fractional allocation; invariant to row order",
        "models": base, "paired_cluster_bootstrap_contrasts": contrasts,
        "limitations": [
            "One previously analyzed external archive and a post hoc secondary endpoint; not a confirmatory replication.",
            "Only 22 positive target-phase detections; top-fraction estimates are consequently imprecise.",
            "Ranking enrichment does not imply calibrated probabilities or a usable decision threshold.",
            "All records share a common reported powder-dosing/heating order; this analysis cannot isolate route-order effects or causal optimization."
        ],
    }
    OUTJSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    for m in MODELS:
        for frac in FRACTIONS:
            met = base[m]["top_fraction"][str(frac)]
            rows.append({"model": m, "top_fraction": frac, "precision": met["precision"], "recall": met["recall"], "lift_vs_prevalence": met["lift"], "average_precision": base[m]["average_precision"]})
    pd.DataFrame(rows).to_csv(OUTCSV, index=False)
    print(json.dumps({"n": len(df), "positive_n": int(y.sum()), "prevalence": float(y.mean()), "models": base, "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
