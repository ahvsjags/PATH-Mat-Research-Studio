#!/usr/bin/env python
"""Post-hoc quality-tier and single-component influence checks on locked scores."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
INFILE = ROOT / "results" / "frozen_pg_target_phase" / "FROZEN_PG_TARGET_PHASE_ROW_AUDIT_INTERNAL.csv"
OUT = ROOT / "results" / "frozen_pg_target_phase" / "FROZEN_PG_QUALITY_INFLUENCE_RESULTS.json"
DRAWS = 10_000
SEED = 1_941_501


def summarize(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    y = (frame.target_phase_wt_pct.to_numpy(float) > 1).astype(int)
    sc = np.column_stack([-frame[f"{m.lower()}_probability"].to_numpy(float) for m in ("B0", "B1", "B2")])
    # Preserve the integer union-find component IDs used by the primary audit;
    # numeric sorting is part of deterministic bootstrap reproducibility.
    g = frame.dependency_component.to_numpy(dtype=np.int64)
    ug = np.unique(g)
    ap = np.array([average_precision_score(y, sc[:, j]) for j in range(3)])
    valid = []
    idx_by_group = [np.flatnonzero(g == x) for x in ug]
    for _ in range(DRAWS):
        take = rng.integers(0, len(ug), len(ug))
        counts = np.bincount(take, minlength=len(ug))
        w = np.zeros(len(y), dtype=np.int32)
        for i, count in enumerate(counts):
            if count:
                w[idx_by_group[i]] = count
        keep = w > 0
        yy = y[keep]
        if yy.min() == yy.max():
            continue
        a = [average_precision_score(yy, sc[keep, j], sample_weight=w[keep]) for j in range(3)]
        valid.append([a[1]-a[0], a[2]-a[1], a[2]-a[0]])
    boot = np.asarray(valid)
    # Leave-one-component-out influence range; fixed scores, no refit.
    loo = []
    for component in ug:
        keep = g != component
        yy = y[keep]
        if yy.min() == yy.max():
            continue
        a = [average_precision_score(yy, sc[keep, j]) for j in range(3)]
        loo.append([a[1]-a[0], a[2]-a[1], a[2]-a[0]])
    loo = np.asarray(loo)
    keys = ("B1_minus_B0", "B2_minus_B1", "B2_minus_B0")
    observed_diffs = (ap[1]-ap[0], ap[2]-ap[1], ap[2]-ap[0])
    return {
        "n": len(y), "positive_n": int(y.sum()), "prevalence": float(y.mean()), "components": len(ug),
        "ap": dict(zip(("B0", "B1", "B2"), map(float, ap))),
        "paired_ap_differences": {
            k: {"estimate": float(observed_diffs[i]),
                "component_bootstrap_ci95": np.quantile(boot[:, i], [.025, .975]).tolist(),
                "loo_component_range": [float(loo[:, i].min()), float(loo[:, i].max())],
                "loo_components_evaluated": int(len(loo))}
            for i, k in enumerate(keys)
        },
        "draws": DRAWS, "seed": SEED,
        "caution": "Post-hoc endpoint/subgroup sensitivity; the quality-1 subset has few positives and remains from the same previously inspected archive."
    }


def main() -> None:
    df = pd.read_csv(INFILE)
    rng = np.random.default_rng(SEED)
    result = {
        "status": "POSTHOC_QUALITY_TIER_AND_INFLUENCE_AUDIT_ON_PREEXISTING_FROZEN_SCORES",
        "endpoint": "target phase present at >1 wt% in the source-designated active refinement",
        "full_accepted_tiers_1_2": summarize(df, rng),
        "highest_quality_tier_1_only": summarize(df[df.accepted_quality_score == 1].copy(), rng),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
