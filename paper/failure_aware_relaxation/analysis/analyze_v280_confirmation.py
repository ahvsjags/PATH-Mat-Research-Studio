from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]
HOLD = PROJECT / "runs" / "holdout" / "v280_failure_repair_fresh_b4b4_v1"
RUN = HOLD / "scored"
PROTOCOL = PROJECT / "protocol" / "v280_failure_repair_confirmation_v1.json"
LOCK = PROJECT / "protocol" / "v280_failure_repair_confirmation_v1.lock.json"
OUT = RUN / "analysis_v280_confirmation.json"
REPORT = PROJECT / "reports" / "V280_FAILURE_REPAIR_CONFIRMATION.md"
BASELINE = "baseline_static_0123"
CANDIDATE = "candidate_failure_triggered_lbfgs"
DRAWS = 50_000
FORMULA_SEED = 28031
PAIR_SEED = 28032
TWO_WAY_SEED = 28033
MATCHER = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0,
                           primitive_cell=True, scale=True,
                           attempt_supercell=True, allow_subset=False)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def formula_bootstrap(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(DRAWS)
    n = len(values)
    for i in range(DRAWS):
        samples[i] = float(np.mean(values[rng.integers(0, n, size=n)]))
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def pair_cluster_bootstrap(matrix: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    pair_count = matrix.shape[0]
    samples = np.empty(DRAWS)
    for i in range(DRAWS):
        draw = rng.integers(0, pair_count, size=pair_count)
        samples[i] = float(np.mean(matrix[draw, :]))
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def two_way_bootstrap(matrix: np.ndarray, seed: int) -> tuple[list[float], int]:
    rng = np.random.default_rng(seed)
    pair_count, a_count = matrix.shape
    samples = np.empty(DRAWS)
    empty = 0
    for i in range(DRAWS):
        pair_draw = rng.integers(0, pair_count, size=pair_count)
        a_draw = rng.integers(0, a_count, size=a_count)
        pc = np.bincount(pair_draw, minlength=pair_count)
        ac = np.bincount(a_draw, minlength=a_count)
        weights = pc[:, None] * ac[None, :]
        total = int(weights.sum())
        if total == 0:
            empty += 1
            samples[i] = float(np.mean(matrix))
        else:
            samples[i] = float(np.sum(matrix * weights) / total)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])], empty


def main() -> None:
    if not (RUN / "run_completed.json").is_file():
        raise FileNotFoundError("locked scoring run is not complete")
    completion = json.loads((RUN / "run_completed.json").read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if completion.get("protocol_sha256") != sha(PROTOCOL) or lock.get("protocol_sha256") != sha(PROTOCOL):
        raise RuntimeError("run, protocol and lock do not match")
    calls_path = RUN / "oracle_calls.jsonl"
    assignments_path = RUN / "arm_assignments.jsonl"
    calls = read_jsonl(calls_path)
    assignments = read_jsonl(assignments_path)
    if sha(calls_path) != completion.get("oracle_calls_sha256") or sha(assignments_path) != completion.get("arm_assignments_sha256"):
        raise RuntimeError("raw scoring logs changed after completion")
    by_call = {row["call_id"]: row for row in calls}
    if len(by_call) != len(calls) or len(assignments) != 432 * 8:
        raise RuntimeError("raw calls or assignments are incomplete/duplicated")
    by_formula_arm: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {BASELINE: [], CANDIDATE: []})
    for assignment in assignments:
        if assignment["call_id"] not in by_call:
            raise RuntimeError("assignment references missing call")
        by_formula_arm[assignment["formula"]][assignment["arm"]].append(assignment)
    records = {x["formula"]: x for x in protocol["candidate_records"]}
    if set(by_formula_arm) != set(records):
        raise RuntimeError("scored formula set differs from frozen grid")

    rows = []
    for formula in sorted(records):
        spec = records[formula]
        counts = {}
        for arm in (BASELINE, CANDIDATE):
            arm_rows = sorted(by_formula_arm[formula][arm], key=lambda x: x["slot"])
            if len(arm_rows) != 4 or [x["slot"] for x in arm_rows] != [1, 2, 3, 4]:
                raise RuntimeError(f"four assigned calls missing for {formula}/{arm}")
            structures = []
            for assignment in arm_rows:
                call = by_call[assignment["call_id"]]
                if (assignment["output_sha256"] != call["output_sha256"]
                        or assignment["accepted"] != call["accepted"]
                        or assignment["v0_status"] != call["v0_status"]
                        or assignment["serialized_force_eV_A"] != call["max_force_serialized_eV_A"]):
                    raise RuntimeError(f"assignment/call evidence mismatch: {formula}/{arm}/{assignment['slot']}")
                if not call["accepted"]:
                    continue
                out = (RUN / call["output_path"]).resolve(strict=True)
                out.relative_to((RUN / "relaxed").resolve(strict=True))
                if sha(out) != call["output_sha256"]:
                    raise RuntimeError(f"accepted output hash mismatch: {call['call_id']}")
                structure = Structure.from_file(out, primitive=False)
                if structure.composition.reduced_formula != formula:
                    raise RuntimeError(f"accepted relaxed composition mismatch: {call['call_id']}")
                structures.append(structure)
            counts[arm] = len(MATCHER.group_structures(structures)) if structures else 0
        rows.append({
            "formula": formula,
            "a_site": spec["a_site"],
            "pair_cluster": spec["pair_cluster"],
            "baseline_unique_accepted_basins": counts[BASELINE],
            "candidate_unique_accepted_basins": counts[CANDIDATE],
            "delta_candidate_minus_baseline": counts[CANDIDATE] - counts[BASELINE],
            "candidate_fourth_call_method": next(x["method"] for x in by_formula_arm[formula][CANDIDATE] if x["slot"] == 4),
        })

    deltas = np.asarray([x["delta_candidate_minus_baseline"] for x in rows], dtype=float)
    a_sites = sorted({x["a_site"] for x in rows})
    pair_ids = sorted({x["pair_cluster"] for x in rows})
    matrix = np.zeros((len(pair_ids), len(a_sites)), dtype=float)
    pair_index = {x: i for i, x in enumerate(pair_ids)}
    a_index = {x: i for i, x in enumerate(a_sites)}
    for row in rows:
        matrix[pair_index[row["pair_cluster"]], a_index[row["a_site"]]] = row["delta_candidate_minus_baseline"]
    if matrix.shape != (72, 6):
        raise RuntimeError(f"expected 72 x 6 complete paired grid, got {matrix.shape}")
    mean = float(np.mean(deltas))
    formula_ci = formula_bootstrap(deltas, FORMULA_SEED)
    pair_ci = pair_cluster_bootstrap(matrix, PAIR_SEED)
    two_way_ci, empty = two_way_bootstrap(matrix, TWO_WAY_SEED)
    positive = int(np.sum(deltas > 0))
    negative = int(np.sum(deltas < 0))
    tied = int(np.sum(deltas == 0))
    method_counts = defaultdict(int)
    for row in rows:
        method_counts[row["candidate_fourth_call_method"]] += 1
    class_call_summary = {}
    for cls in (0, 1, 2, 3):
        class_rows = [x for x in calls if x.get("call_type") == "catalog_fire" and x.get("ordering_class") == cls]
        class_call_summary[str(cls)] = {
            "calls": len(class_rows),
            "accepted": sum(bool(x.get("accepted")) for x in class_rows),
            "v0_pass_force_only_failure": sum(
                x.get("error") is None and x.get("v0_status") == "pass"
                and x.get("force_pass_serialized") is False for x in class_rows
            ),
            "runtime_or_pipeline_failure": sum(x.get("error") is not None for x in class_rows),
        }
    repair_calls = [x for x in calls if x.get("call_type") == "force_only_lbfgs_continuation"]
    mean_baseline_yield = float(np.mean([x["baseline_unique_accepted_basins"] for x in rows]))
    mean_candidate_yield = float(np.mean([x["candidate_unique_accepted_basins"] for x in rows]))
    candidate_fourth = [x for x in assignments if x["arm"] == CANDIDATE and x["slot"] == 4]
    result = {
        "status": "COMPLETED_V280_FRESH_B4B4_FAILURE_REPAIR_CONFIRMATION_ANALYSIS",
        "analyzed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(PROTOCOL),
        "analysis_script_sha256": sha(HERE),
        "run_completed_sha256": sha(RUN / "run_completed.json"),
        "raw_calls_sha256": sha(calls_path),
        "raw_assignments_sha256": sha(assignments_path),
        "raw_trace_sha256": sha(RUN / "agent_trace.jsonl"),
        "formula_count": len(rows), "pair_cluster_count": len(pair_ids),
        "a_site_count": len(a_sites), "unique_oracle_calls": len(calls),
        "assignments_per_arm": {arm: sum(x["arm"] == arm for x in assignments) for arm in (BASELINE, CANDIDATE)},
        "primary_endpoint": "candidate minus baseline unique accepted, serialized-force-passing relaxed basins per formula",
        "mean_delta_candidate_minus_baseline_per_formula": mean,
        "pair_cluster_bootstrap_95_ci_primary": pair_ci,
        "two_way_a_site_x_pair_cluster_bootstrap_95_ci_primary": two_way_ci,
        "formula_iid_bootstrap_95_ci_secondary": formula_ci,
        "bootstrap_draws": DRAWS,
        "bootstrap_seeds": {"formula_iid": FORMULA_SEED, "pair_cluster": PAIR_SEED, "two_way": TWO_WAY_SEED},
        "two_way_bootstrap_empty_weight_draws": empty,
        "paired_formula_results": {"positive": positive, "negative": negative, "tied": tied},
        "candidate_fourth_slot_methods": dict(method_counts),
        "mean_unique_accepted_basins_per_formula": {
            "baseline_static_0123": mean_baseline_yield,
            "candidate_failure_triggered_lbfgs": mean_candidate_yield,
        },
        "class_0_to_3_fire_call_summary": class_call_summary,
        "repair_summary": {
            "triggered_formula_count": len(repair_calls),
            "repair_calls_accepted": sum(bool(x.get("accepted")) for x in repair_calls),
            "repair_calls_failed": sum(not bool(x.get("accepted")) for x in repair_calls),
            "candidate_fourth_slot_lbfgs_count": sum(x.get("method") == "LBFGS_force_only_continuation" for x in candidate_fourth),
            "candidate_fourth_slot_class3_fallback_count": sum(x.get("method") == "FIRE_catalog_fallback_class3" for x in candidate_fourth),
            "repair_recovery_fraction": (sum(bool(x.get("accepted")) for x in repair_calls) / len(repair_calls)) if repair_calls else None,
        },
        "dual_primary_interval_lower_bounds_above_zero": bool(pair_ci[0] > 0 and two_way_ci[0] > 0),
        "per_formula": rows,
        "interpretation_scope": "Matched-budget local CHGNet relaxation yield on the locked fresh A2+ B4+ B4+ double-perovskite grid; not thermodynamic stability, synthesis feasibility, or experimental yield.",
    }
    if OUT.exists() or REPORT.exists():
        raise FileExistsError("V280 analysis outputs already exist; refusing to overwrite")
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = [
        "# V280 locked fresh-grid confirmation: failure-triggered continuation",
        "",
        "## Frozen comparison",
        "",
        "- Candidate: run ordering classes 0/1/2; when a hash-verified, V0-passing force-only failure occurs, spend the fourth call on an LBFGS continuation from the lowest-index failed class; otherwise run class 3.",
        "- Comparator: static classes 0/1/2/3 with FIRE.",
        f"- Fresh grid: {len(rows)} formulae, {len(pair_ids)} previously unscored B/B′ pair clusters, {len(a_sites)} matched A-site cations; four calls per arm per formula.",
        "",
        "## Primary result",
        "",
        f"- Mean change in unique accepted basins per formula: **{mean:+.4f}**.",
        f"- Pair-cluster bootstrap 95% CI: **[{pair_ci[0]:+.4f}, {pair_ci[1]:+.4f}]**.",
        f"- Two-way A-site × B/B′-pair bootstrap 95% CI: **[{two_way_ci[0]:+.4f}, {two_way_ci[1]:+.4f}]**.",
        f"- Paired formula outcomes: {positive} positive, {negative} negative, {tied} tied.",
        f"- Mean unique accepted basins per formula: baseline {mean_baseline_yield:.3f}; candidate {mean_candidate_yield:.3f}.",
        f"- Triggered continuations: {len(repair_calls)}; accepted {sum(bool(x.get('accepted')) for x in repair_calls)}.",
        f"- Prespecified dual-interval criterion met: **{result['dual_primary_interval_lower_bounds_above_zero']}**.",
        "",
        "## Provenance",
        "",
        f"- Frozen protocol SHA-256: `{result['protocol_sha256']}`.",
        f"- Raw calls SHA-256: `{result['raw_calls_sha256']}`.",
        f"- Raw assignments SHA-256: `{result['raw_assignments_sha256']}`.",
        "- No previously scored outcomes were used to select this grid or tune the policy.",
        "- This result requires independent force/V0/structure/statistic replay before manuscript use.",
        "",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "status", "formula_count", "pair_cluster_count", "unique_oracle_calls",
        "mean_delta_candidate_minus_baseline_per_formula", "pair_cluster_bootstrap_95_ci_primary",
        "two_way_a_site_x_pair_cluster_bootstrap_95_ci_primary", "paired_formula_results",
        "candidate_fourth_slot_methods", "dual_primary_interval_lower_bounds_above_zero",
    )}, indent=2))


if __name__ == "__main__":
    main()
