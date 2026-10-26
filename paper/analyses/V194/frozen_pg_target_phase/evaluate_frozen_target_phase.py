#!/usr/bin/env python3
"""Post-hoc endpoint audit using the already locked Lee->Precursor Genome scores.

No model fitting, score transformation other than sign reversal for success ranking,
calibration, threshold selection, or sample selection based on prediction is performed.
The endpoint is presence/weight of the intended target composition in the source-
designated active, human-accepted refinement.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from pymatgen.core import Composition
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
V194 = ROOT / "PATH-Mat_Evidence_Strengthening_V194"
V136 = ROOT / "PATH-Mat_Evidence_Expansion_V136"
LEDGER = V136 / "online_sources" / "ledger_precursor_genome.json"
LOCK = V136 / "physical_xrd_holdout_v136" / "PHYSICAL_XRD_HOLDOUT_PREDICTION_LOCK_V136.csv"
OUT = V194 / "results" / "frozen_pg_target_phase"
PRED_COLS = ("b0_probability", "b1_probability", "b2_probability")
MODEL_NAMES = ("B0", "B1", "B2")
THRESHOLDS = (0.0, 1.0, 5.0, 10.0)
N_BOOT = 10_000
SEED = 1941501


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reduced_composition(formula: str) -> tuple[tuple[str, float], ...]:
    amounts = Composition(formula).get_el_amt_dict()
    total = float(sum(amounts.values()))
    return tuple(sorted((el, round(float(v) / total, 6)) for el, v in amounts.items()))


def phase_formula(name: str) -> str:
    """Read formula prefix from source CIF labels, removing dataset annotations."""
    value = str(name).removesuffix(".cif").split("_")[0]
    return re.sub(r"\(\d+\)$", "", value)


def full_profile(sample: dict) -> tuple | None:
    p = (sample.get("synthesis") or {}).get("temperature_profile") or {}
    t = np.asarray(p.get("time_minutes") or [], dtype=float)
    y = np.asarray(p.get("temperature_celsius") or [], dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 2:
        return None
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]
    tu, inv = np.unique(t, return_inverse=True)
    yu = np.asarray([np.mean(y[inv == j]) for j in range(len(tu))])
    if len(tu) < 2:
        return None
    return tuple(zip(tu.tolist(), yu.tolist()))


def connected_components(rows: list[dict]) -> np.ndarray:
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for field in ("target_formula", "profile"):
        seen: dict[object, int] = {}
        for i, row in enumerate(rows):
            key = row[field]
            if key is None:
                continue
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    return np.asarray([find(i) for i in range(len(rows))])


def bootstrap_contrasts(y: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict:
    unique = np.unique(groups)
    row_by_group = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(SEED)
    estimates = []
    valid = 0
    for _ in range(N_BOOT):
        chosen = rng.integers(0, len(unique), size=len(unique))
        counts = np.bincount(chosen, minlength=len(unique))
        weights = np.zeros(len(y), dtype=np.int32)
        for j, n in enumerate(counts):
            if n:
                weights[row_by_group[j]] = n
        keep = weights > 0
        yy, ww = y[keep], weights[keep]
        if not np.any(yy == 0) or not np.any(yy == 1):
            continue
        ap = [average_precision_score(yy, -scores[keep, j], sample_weight=ww) for j in range(3)]
        estimates.append([ap[1] - ap[0], ap[2] - ap[1], ap[2] - ap[0]])
        valid += 1
    arr = np.asarray(estimates, dtype=float)
    pairs = (("B1_minus_B0", 0), ("B2_minus_B1", 1), ("B2_minus_B0", 2))
    return {
        name: {
            "ci95_cluster_bootstrap": np.quantile(arr[:, j], [0.025, 0.975]).tolist(),
            "valid_draws": valid,
            "draws_requested": N_BOOT,
            "seed": SEED,
            "resampling_unit": "connected components over exact target formula or exact recorded thermal profile",
        }
        for name, j in pairs
    }


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    by_id = {x["sample_id"]: x for x in ledger["samples"]}
    with LOCK.open(encoding="utf-8-sig", newline="") as f:
        locked = list(csv.DictReader(f))
    if len(locked) != 981 or len({r["sample_id"] for r in locked}) != len(locked):
        raise RuntimeError("Prediction lock reconciliation failed")

    audit = []
    quality = Counter()
    skipped = Counter()
    for pred in locked:
        sample = by_id[pred["sample_id"]]
        xrd = (sample.get("characterization") or {}).get("xrd") or {}
        scans = xrd.get("scans") or []
        scan_i = sample.get("active_scan_index")
        if not isinstance(scan_i, int) or not (0 <= scan_i < len(scans)):
            skipped["no_source_designated_active_scan"] += 1
            continue
        scan = scans[scan_i]
        case_i = scan.get("active_case_index")
        cases = scan.get("refinement_cases") or []
        if not isinstance(case_i, int) or not (0 <= case_i < len(cases)):
            skipped["no_source_designated_active_refinement"] += 1
            continue
        case = cases[case_i]
        verification = case.get("verification") or {}
        quality[(verification.get("is_accepted"), verification.get("human_quality_score"))] += 1
        if verification.get("is_accepted") is not True:
            skipped["active_refinement_not_human_accepted"] += 1
            continue

        target = str(sample.get("target_compound") or "")
        target_key = reduced_composition(target)
        target_fraction = 0.0
        for label, fraction in (case.get("phase_weights") or {}).items():
            formula = phase_formula(label)
            try:
                if reduced_composition(formula) == target_key:
                    target_fraction += float(fraction)
            except Exception:
                skipped["unparsed_phase_label"] += 1
        row = {
            "sample_id": pred["sample_id"],
            "target_formula": target,
            "target_phase_fraction": target_fraction,
            "target_phase_wt_pct": 100.0 * target_fraction,
            "accepted_quality_score": verification.get("human_quality_score"),
            "reaction_category": (sample.get("outcome") or {}).get("reaction_category"),
            "profile": full_profile(sample),
            "scores": [float(pred[c]) for c in PRED_COLS],
            "model_bundle_sha256": pred["model_bundle_sha256"],
            "prediction_lock_utc": pred["prediction_lock_utc"],
        }
        audit.append(row)

    if len(audit) != 952:
        raise RuntimeError(f"Accepted active-refinement cohort mismatch: {len(audit)} != 952")
    groups = connected_components(audit)
    if len(np.unique(groups)) != 244:
        raise RuntimeError(f"Dependency-component reconciliation failed: {len(np.unique(groups))} != 244")
    scores = np.asarray([r["scores"] for r in audit], dtype=float)
    target_pct = np.asarray([r["target_phase_wt_pct"] for r in audit], dtype=float)
    thresholds = []
    for threshold in THRESHOLDS:
        y = (target_pct > threshold).astype(np.int8)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        ap = [float(average_precision_score(y, -scores[:, j])) for j in range(3)]
        auc = [float(roc_auc_score(y, -scores[:, j])) for j in range(3)]
        contrasts = {
            "B1_minus_B0": ap[1] - ap[0],
            "B2_minus_B1": ap[2] - ap[1],
            "B2_minus_B0": ap[2] - ap[0],
        }
        thresholds.append({
            "target_phase_detection_threshold_wt_pct": threshold,
            "n": len(y),
            "positive_n": int(y.sum()),
            "prevalence": float(y.mean()),
            "average_precision_by_model_using_negative_risk": dict(zip(MODEL_NAMES, ap)),
            "auroc_by_model_using_negative_risk": dict(zip(MODEL_NAMES, auc)),
            "paired_ap_contrasts": contrasts,
            "paired_contrast_intervals": bootstrap_contrasts(y, scores, groups),
        })

    continuous = {
        model: {
            "spearman_negative_risk_vs_target_phase_wt_pct": float(spearmanr(-scores[:, j], target_pct).statistic),
        }
        for j, model in enumerate(MODEL_NAMES)
    }
    output = {
        "schema_version": "pathmat_v194_frozen_pg_accepted_target_phase_audit_v1",
        "status": "COMPLETED_POSTHOC_SECONDARY_ENDPOINT_ON_PREEXISTING_LOCKED_SCORES",
        "source": {
            "name": "Precursor Genome",
            "doi": "10.5281/zenodo.21285546",
            "license": "CC BY 4.0",
            "ledger_path": str(LEDGER),
            "ledger_sha256": sha(LEDGER),
            "prediction_lock_path": str(LOCK),
            "prediction_lock_sha256": sha(LOCK),
            "source_schema": "source-designated active scan -> active refinement; accepted flag required",
            "accepted_refinement_quality_score_counts": {str(k): int(v) for k, v in quality.items()},
        },
        "predictions": {
            "provenance": "V136 previously frozen Lee-trained B0/B1/B2 scores; this analysis did not refit, recalibrate, or threshold them",
            "model_bundle_sha256": sorted({r["model_bundle_sha256"] for r in audit}),
            "prediction_lock_utc": sorted({r["prediction_lock_utc"] for r in audit}),
            "positive_score_semantics": "impurity/competing-phase risk; sign reversed only for ranking target-phase presence",
        },
        "cohort": {
            "prediction_lock_rows": len(locked),
            "accepted_active_refinement_rows": len(audit),
            "connected_components": int(len(np.unique(groups))),
            "quality_1_n": int(sum(r["accepted_quality_score"] == 1 for r in audit)),
            "quality_2_n": int(sum(r["accepted_quality_score"] == 2 for r in audit)),
            "target_phase_present_any_weight_n": int(np.sum(target_pct > 0)),
            "target_phase_wt_pct_quantiles": np.quantile(target_pct, [0, .5, .9, .95, .99, 1]).tolist(),
            "skipped_reason_counts": dict(skipped),
        },
        "endpoint": {
            "target_formula_matching": "Reduced elemental composition of the target_compound is matched to the formula prefix of each active refinement phase label; weights of matching phases are summed.",
            "primary_binary_outcome": "target phase present at >1 wt% in active, human-accepted refinement",
            "threshold_sensitivity": [0, 1, 5, 10],
            "continuous_outcome": "source-reported active-refinement target-phase weight percentage",
            "status": "exploratory endpoint defined after prior access to this archive and related outcomes; not confirmatory or preregistered",
        },
        "threshold_results": thresholds,
        "continuous_result": continuous,
        "bootstrap": {
            "draws": N_BOOT,
            "seed": SEED,
            "unit": "connected components over exact target formula OR exact full recorded temperature profile",
            "type": "paired percentile bootstrap on fixed predictions",
        },
        "interpretation": [
            "This analysis tests a distinct, physically measured target-phase endpoint in one external experimental archive; it is not an independent new archive because Precursor Genome was previously analyzed.",
            "The workflow records the same powder-dosing/heating sequence for samples, so the external test does not contain route-order variation and cannot identify an order effect.",
            "The accepted refinement is expert-accepted source output, not an independent re-refinement by this study.",
            "A positive ranking result can support source-specific frozen transfer for target-phase detection only; it cannot establish universal zero-shot generalization, calibrated probabilities, or causal optimization.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "FROZEN_PG_TARGET_PHASE_RESULTS.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with (OUT / "FROZEN_PG_TARGET_PHASE_ROW_AUDIT_INTERNAL.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["sample_id", "target_formula", "target_phase_wt_pct", "accepted_quality_score", "reaction_category", "b0_probability", "b1_probability", "b2_probability", "dependency_component"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row, group in zip(audit, groups):
            w.writerow({"sample_id": row["sample_id"], "target_formula": row["target_formula"],
                        "target_phase_wt_pct": row["target_phase_wt_pct"],
                        "accepted_quality_score": row["accepted_quality_score"],
                        "reaction_category": row["reaction_category"],
                        "b0_probability": row["scores"][0], "b1_probability": row["scores"][1], "b2_probability": row["scores"][2],
                        "dependency_component": int(group)})
    print(json.dumps({"cohort": output["cohort"], "threshold_results": thresholds,
                      "continuous_result": continuous}, indent=2))


if __name__ == "__main__":
    main()
