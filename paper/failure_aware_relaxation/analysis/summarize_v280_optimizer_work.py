from __future__ import annotations

"""Post hoc, hash-verified accounting of optimizer work in locked V280 run.

This diagnostic does not alter the locked primary endpoint or policy. It compares
the trajectory-step and wall-clock totals implied by the four assigned slots per
formula in each arm. Steps are an optimizer-work proxy; wall time is descriptive.
"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]
RUN = PROJECT / "runs" / "holdout" / "v280_failure_repair_fresh_b4b4_v1" / "scored"
PROTOCOL = PROJECT / "protocol" / "v280_failure_repair_confirmation_v1.json"
OUT = RUN / "optimizer_work_diagnostic_v280.json"
REPORT = PROJECT / "reports" / "V280_OPTIMIZER_WORK_DIAGNOSTIC.md"
DRAWS = 50_000
PAIR_SEED = 28034
TWO_WAY_SEED = 28035
BASE = "baseline_static_0123"
CAND = "candidate_failure_triggered_lbfgs"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def qci(samples: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def pair_bootstrap(matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_pair = matrix.shape[0]
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        idx = rng.integers(0, n_pair, size=n_pair)
        out[i] = float(matrix[idx, :].mean())
    return out


def two_way_bootstrap(matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_pair, n_a = matrix.shape
    out = np.empty(DRAWS)
    for i in range(DRAWS):
        pi = rng.integers(0, n_pair, size=n_pair)
        ai = rng.integers(0, n_a, size=n_a)
        pw = np.bincount(pi, minlength=n_pair)
        aw = np.bincount(ai, minlength=n_a)
        w = pw[:, None] * aw[None, :]
        out[i] = float(np.sum(matrix * w) / w.sum())
    return out


def main() -> None:
    completion_path = RUN / "run_completed.json"
    calls_path = RUN / "oracle_calls.jsonl"
    assignments_path = RUN / "arm_assignments.jsonl"
    if not completion_path.is_file() or not (RUN / "run_completed.json").is_file():
        raise FileNotFoundError("V280 locked run is incomplete")
    completion = read_json(completion_path)
    protocol = read_json(PROTOCOL)
    if completion.get("protocol_sha256") != sha(PROTOCOL):
        raise RuntimeError("protocol hash differs from the locked run")
    if completion.get("oracle_calls_sha256") != sha(calls_path):
        raise RuntimeError("raw call log hash differs from completion manifest")
    if completion.get("arm_assignments_sha256") != sha(assignments_path):
        raise RuntimeError("assignment log hash differs from completion manifest")
    calls = read_jsonl(calls_path)
    assignments = read_jsonl(assignments_path)
    by_call = {row["call_id"]: row for row in calls}
    if len(by_call) != len(calls):
        raise RuntimeError("duplicate call ID in raw call log")
    if len(assignments) != 432 * 8:
        raise RuntimeError("expected exactly eight assigned slots per formula")
    by_fa: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in assignments:
        by_fa[(row["formula"], row["arm"])].append(row)
    formulas = sorted({row["formula"] for row in assignments})
    if len(formulas) != 432:
        raise RuntimeError("formula count differs from frozen grid")
    protocol_rows = {row["formula"]: row for row in protocol["candidate_records"]}
    if set(formulas) != set(protocol_rows):
        raise RuntimeError("formula identity set differs from frozen grid")

    rows = []
    for formula in formulas:
        by_arm = {}
        for arm in (BASE, CAND):
            slots = sorted(by_fa[(formula, arm)], key=lambda r: r["slot"])
            if [r["slot"] for r in slots] != [1, 2, 3, 4]:
                raise RuntimeError(f"slot assignment incomplete: {formula}/{arm}")
            call_rows = [by_call[r["call_id"]] for r in slots]
            if any(row.get("steps") is None or row.get("elapsed_seconds") is None for row in call_rows):
                raise RuntimeError(f"missing optimizer work fields: {formula}/{arm}")
            by_arm[arm] = {
                "steps": sum(int(row["steps"]) for row in call_rows),
                "wall_seconds": sum(float(row["elapsed_seconds"]) for row in call_rows),
                "methods": [r["method"] for r in slots],
            }
        rows.append({
            "formula": formula,
            "a_site": protocol_rows[formula]["a_site"],
            "pair_cluster": protocol_rows[formula]["pair_cluster"],
            "baseline_steps": by_arm[BASE]["steps"],
            "candidate_steps": by_arm[CAND]["steps"],
            "candidate_minus_baseline_steps": by_arm[CAND]["steps"] - by_arm[BASE]["steps"],
            "baseline_wall_seconds": by_arm[BASE]["wall_seconds"],
            "candidate_wall_seconds": by_arm[CAND]["wall_seconds"],
            "candidate_minus_baseline_wall_seconds": by_arm[CAND]["wall_seconds"] - by_arm[BASE]["wall_seconds"],
            "candidate_fourth_method": by_arm[CAND]["methods"][3],
        })

    delta_steps = np.asarray([r["candidate_minus_baseline_steps"] for r in rows], dtype=float)
    delta_wall = np.asarray([r["candidate_minus_baseline_wall_seconds"] for r in rows], dtype=float)
    pair_ids = sorted({r["pair_cluster"] for r in rows})
    a_sites = sorted({r["a_site"] for r in rows})
    index_p = {x: i for i, x in enumerate(pair_ids)}
    index_a = {x: i for i, x in enumerate(a_sites)}
    step_matrix = np.zeros((len(pair_ids), len(a_sites)))
    wall_matrix = np.zeros_like(step_matrix)
    for r in rows:
        step_matrix[index_p[r["pair_cluster"]], index_a[r["a_site"]]] = r["candidate_minus_baseline_steps"]
        wall_matrix[index_p[r["pair_cluster"]], index_a[r["a_site"]]] = r["candidate_minus_baseline_wall_seconds"]
    if step_matrix.shape != (72, 6):
        raise RuntimeError(f"expected 72x6 matched layout, got {step_matrix.shape}")

    step_pair = pair_bootstrap(step_matrix, PAIR_SEED)
    step_twoway = two_way_bootstrap(step_matrix, TWO_WAY_SEED)
    wall_pair = pair_bootstrap(wall_matrix, PAIR_SEED + 2)
    wall_twoway = two_way_bootstrap(wall_matrix, TWO_WAY_SEED + 2)
    baseline_total_steps = int(sum(r["baseline_steps"] for r in rows))
    candidate_total_steps = int(sum(r["candidate_steps"] for r in rows))
    baseline_total_wall = float(sum(r["baseline_wall_seconds"] for r in rows))
    candidate_total_wall = float(sum(r["candidate_wall_seconds"] for r in rows))
    result = {
        "status": "COMPLETED_V280_POSTHOC_HASH_VERIFIED_OPTIMIZER_WORK_DIAGNOSTIC",
        "analyzed_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script_sha256": sha(HERE),
        "protocol_sha256": sha(PROTOCOL),
        "run_completed_sha256": sha(completion_path),
        "raw_calls_sha256": sha(calls_path),
        "assignments_sha256": sha(assignments_path),
        "analysis_classification": "post hoc resource-accounting diagnostic; not a preregistered primary endpoint",
        "work_measure_definition": "sum of recorded optimizer trajectory energy entries across the four assigned slots; a proxy for relaxation work, not a direct count of CHGNet forward evaluations",
        "wall_time_definition": "sum of recorded per-call elapsed_seconds; descriptive because execution was sequential and wall-clock timing is hardware/runtime dependent",
        "n_formula": len(rows),
        "n_pair_cluster": len(pair_ids),
        "n_a_site": len(a_sites),
        "trajectory_steps": {
            "baseline_total": baseline_total_steps,
            "candidate_total": candidate_total_steps,
            "candidate_minus_baseline_total": candidate_total_steps - baseline_total_steps,
            "relative_change_percent": 100.0 * (candidate_total_steps / baseline_total_steps - 1.0),
            "mean_paired_candidate_minus_baseline_per_formula": float(delta_steps.mean()),
            "pair_cluster_bootstrap_95_ci_per_formula": qci(step_pair),
            "two_way_a_site_x_pair_cluster_bootstrap_95_ci_per_formula": qci(step_twoway),
            "formulas_fewer_steps_equal_steps_more_steps": [int((delta_steps < 0).sum()), int((delta_steps == 0).sum()), int((delta_steps > 0).sum())],
        },
        "wall_time_seconds": {
            "baseline_total": baseline_total_wall,
            "candidate_total": candidate_total_wall,
            "candidate_minus_baseline_total": candidate_total_wall - baseline_total_wall,
            "relative_change_percent": 100.0 * (candidate_total_wall / baseline_total_wall - 1.0),
            "mean_paired_candidate_minus_baseline_per_formula": float(delta_wall.mean()),
            "pair_cluster_bootstrap_95_ci_per_formula": qci(wall_pair),
            "two_way_a_site_x_pair_cluster_bootstrap_95_ci_per_formula": qci(wall_twoway),
        },
        "candidate_fourth_slot_method_counts": {
            method: sum(r["candidate_fourth_method"] == method for r in rows)
            for method in sorted({r["candidate_fourth_method"] for r in rows})
        },
        "per_formula": rows,
    }
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("refusing to overwrite an existing work diagnostic")
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    s = result["trajectory_steps"]
    w = result["wall_time_seconds"]
    report = [
        "# V280 optimizer-work diagnostic (post hoc)",
        "",
        "> This resource-accounting analysis was written after the locked confirmation scores were visible. It is supplementary and exploratory; the locked accepted-basin endpoint remains primary.",
        "",
        "## Hash-verified comparison",
        "",
        f"- Grid: {len(rows)} paired formulae, {len(pair_ids)} B/B′ pair clusters, {len(a_sites)} A-site cations.",
        f"- Candidate fourth-slot allocations: {result['candidate_fourth_slot_method_counts']}.",
        f"- Recorded optimizer trajectory steps: baseline {baseline_total_steps:,}; failure-aware policy {candidate_total_steps:,}; relative change {s['relative_change_percent']:+.2f}%.",
        f"- Mean paired difference: {s['mean_paired_candidate_minus_baseline_per_formula']:+.2f} steps per formula; pair-cluster 95% CI {s['pair_cluster_bootstrap_95_ci_per_formula']}; two-way A-site × pair-cluster 95% CI {s['two_way_a_site_x_pair_cluster_bootstrap_95_ci_per_formula']}.",
        f"- Formulae with fewer/same/more steps under the candidate: {s['formulas_fewer_steps_equal_steps_more_steps']}.",
        f"- Recorded wall time: baseline {baseline_total_wall:,.1f} s; candidate {candidate_total_wall:,.1f} s; descriptive change {w['relative_change_percent']:+.2f}%.",
        "",
        "## Interpretation",
        "",
        "Trajectory-step counts are a proxy for optimization work, not an instrumented count of every CHGNet forward evaluation. Wall time is descriptive and depends on the sequential run environment. The result can support a claim of lower observed optimizer-step burden only after independent V280 audit passes; it should not be presented as a preregistered or hardware-independent runtime guarantee.",
        "",
        f"The primary result and its preregistered cluster-aware confidence intervals remain in `{RUN.name}/analysis_v280_confirmation.json`. This file does not change that endpoint or the frozen strategy.",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "trajectory_steps", "wall_time_seconds", "candidate_fourth_slot_method_counts")}, indent=2))


if __name__ == "__main__":
    main()
