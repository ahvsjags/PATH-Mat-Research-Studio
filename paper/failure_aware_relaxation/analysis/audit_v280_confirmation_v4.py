from __future__ import annotations

"""Independent replay of the locked V280 confirmation (numerical differences reported, not hidden).

This implementation intentionally does not import the V280 runner or analysis
script. It rehashes every input/output, reruns V0 and serialized CHGNet force
checks, reconstructs both arms, redoes basin matching and all primary CIs.
"""

import hashlib
import importlib.util
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
HOLD = PROJECT / "runs" / "holdout" / "v280_failure_repair_fresh_b4b4_v1"
RUN = HOLD / "scored"
GEOMETRY = HOLD / "geometry"
PROTOCOL_PATH = PROJECT / "protocol" / "v280_failure_repair_confirmation_v1.json"
LOCK_PATH = PROJECT / "protocol" / "v280_failure_repair_confirmation_v1.lock.json"
ANALYSIS_PATH = RUN / "analysis_v280_confirmation.json"
AUDIT_PATH = RUN / "independent_audit_v280_v4.json"
AUDIT_REPORT = PROJECT / "reports" / "V280_INDEPENDENT_AUDIT_V4.md"
RUNTIME = Path("/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V279/runtime/chgnet_0_4_2")
V268 = Path("/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V279/source_data/V268")
ORACLE_MODULE = V268 / "material_generation" / "validation" / "chgnet_oracle.py"
V0_MODULE = V268 / "material_generation" / "agent" / "closed_loop.py"
CHECKPOINT = "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"
BASELINE = "baseline_static_0123"
CANDIDATE = "candidate_failure_triggered_lbfgs"
THRESHOLD = 0.050001
FORCE_REPLAY_TOL = 1e-5
ENERGY_REPLAY_TOL = 1e-6
BOOTSTRAP_DRAWS = 50_000
MATCHER = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0,
                           primitive_cell=True, scale=True,
                           attempt_supercell=True, allow_subset=False)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(row: dict) -> str:
    payload = {k: v for k, v in row.items() if k != "raw_record_sha256"}
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inside(path_text: str, root: Path, folder: str) -> Path:
    candidate = Path(path_text)
    resolved = candidate.resolve(strict=True) if candidate.is_absolute() else (root / candidate).resolve(strict=True)
    resolved.relative_to((root / folder).resolve(strict=True))
    return resolved


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load independent audit dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_formula(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for i in range(BOOTSTRAP_DRAWS):
        draws[i] = values[rng.integers(0, n, size=n)].mean()
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def bootstrap_pairs(matrix: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for i in range(BOOTSTRAP_DRAWS):
        draws[i] = matrix[rng.integers(0, n, size=n), :].mean()
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def bootstrap_two_way(matrix: np.ndarray, seed: int) -> tuple[list[float], int]:
    rng = np.random.default_rng(seed)
    n_pair, n_a = matrix.shape
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    empty = 0
    for i in range(BOOTSTRAP_DRAWS):
        pd = rng.integers(0, n_pair, size=n_pair)
        ad = rng.integers(0, n_a, size=n_a)
        pw = np.bincount(pd, minlength=n_pair)
        aw = np.bincount(ad, minlength=n_a)
        weights = pw[:, None] * aw[None, :]
        denom = int(weights.sum())
        if denom == 0:
            empty += 1
            draws[i] = matrix.mean()
        else:
            draws[i] = float((matrix * weights).sum() / denom)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])], empty


def main() -> None:
    if AUDIT_PATH.exists() or AUDIT_REPORT.exists():
        raise FileExistsError("independent audit outputs already exist; refusing overwrite")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    completed = json.loads((RUN / "run_completed.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED_BEFORE_HOLDOUT_SCORING":
        raise RuntimeError("frozen protocol status is invalid")
    protocol_hash = sha(PROTOCOL_PATH)
    if lock.get("protocol_sha256") != protocol_hash or completed.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("protocol hash differs across lock/run records")
    for rel, expected in protocol["integrity"]["code_sha256"].items():
        if sha(PROJECT / rel) != expected:
            raise RuntimeError(f"protocol-frozen analysis code changed: {rel}")

    calls_file = RUN / "oracle_calls.jsonl"
    assignments_file = RUN / "arm_assignments.jsonl"
    trace_file = RUN / "agent_trace.jsonl"
    calls = read_jsonl(calls_file)
    assignments = read_jsonl(assignments_file)
    trace = read_jsonl(trace_file)
    if sha(calls_file) != completed["oracle_calls_sha256"] or sha(assignments_file) != completed["arm_assignments_sha256"]:
        raise RuntimeError("raw oracle/assignment logs differ from run completion hashes")
    if sha(trace_file) != completed["agent_trace_sha256"]:
        raise RuntimeError("agent trace differs from run completion hash")
    call_by_id = {row["call_id"]: row for row in calls}
    if len(call_by_id) != len(calls) or len({row["assignment_id"] for row in assignments}) != len(assignments):
        raise RuntimeError("duplicate raw identifiers detected")
    if len(calls) < 432 * 4 or len(calls) > 432 * 5 or len(assignments) != 432 * 8:
        raise RuntimeError("locked call/assignment budget not met")

    runtime_meta = protocol["development_runtime_probe"]
    for p_text, key in ((runtime_meta["probe_path"], "input_sha256"),
                        (runtime_meta["smoke_path"], "smoke_result_sha256"),
                        (runtime_meta["v0_smoke_path"], "v0_smoke_result_sha256")):
        if sha(Path(p_text)) != runtime_meta[key]:
            raise RuntimeError(f"development-only runtime probe provenance changed: {p_text}")

    oracle_module = load_module(ORACLE_MODULE, "v280_independent_chgnet_oracle")
    oracle = oracle_module.CHGNetOracle(RUNTIME, fmax=0.05, max_steps=500)
    if oracle.checkpoint_sha256 != CHECKPOINT or oracle.package_version != "0.4.2" or oracle.device != "cpu":
        raise RuntimeError("independent audit loaded an unexpected CHGNet runtime")
    closed_loop = load_module(V0_MODULE, "v280_independent_v0_gate")
    verifier = closed_loop.Verifier(min_interatomic_distance_angstrom=0.75)

    # Recompute every recorded V0 and serialized CHGNet force result from the
    # byte-addressed artifacts, including calls that the protocol rejected.
    call_audit_rows = []
    force_abs_diffs = []
    energy_abs_diffs = []
    force_replay_tolerance_exceedances = []
    energy_replay_tolerance_exceedances = []
    replay_gate_disagreements = []
    for row in calls:
        call_id = row["call_id"]
        if row.get("raw_record_sha256") != canonical_digest(row):
            raise RuntimeError(f"raw call canonical digest mismatch: {call_id}")
        if row.get("checkpoint_sha256") != CHECKPOINT or row.get("package_version") != "0.4.2" or row.get("device") != "cpu":
            raise RuntimeError(f"runtime identity differs: {call_id}")
        input_file = inside(row["input_path"], RUN, "inputs")
        if sha(input_file) != row.get("input_sha256"):
            raise RuntimeError(f"input hash mismatch: {call_id}")
        output_sha = row.get("output_sha256") or ""
        if output_sha:
            output_file = inside(row["output_path"], RUN, "relaxed")
            if sha(output_file) != output_sha:
                raise RuntimeError(f"output hash mismatch: {call_id}")
            if row.get("error") is not None:
                if row.get("accepted"):
                    raise RuntimeError(f"failed runtime call was marked accepted: {call_id}")
                call_audit_rows.append({"call_id": call_id, "formula": row["formula"], "replayed": False,
                                        "accepted": False, "error": row.get("error"),
                                        "retained_output_sha256": output_sha})
                continue
            structure = Structure.from_file(output_file, primitive=False)
            pred = oracle.model.predict_structure(structure)
            forces = np.asarray(pred["f"], dtype=float)
            energy = float(np.asarray(pred["e"]).reshape(-1)[0])
            force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
            if not math.isfinite(force) or not math.isfinite(energy):
                raise RuntimeError(f"non-finite independently replayed CHGNet result: {call_id}")
            recorded_force = row.get("max_force_serialized_eV_A")
            recorded_energy = row.get("energy_eV_atom")
            if recorded_force is None or recorded_energy is None:
                raise RuntimeError(f"missing recorded force or energy: {call_id}")
            force_abs_diff = abs(force - float(recorded_force))
            energy_abs_diff = abs(energy - float(recorded_energy))
            force_abs_diffs.append(force_abs_diff)
            energy_abs_diffs.append(energy_abs_diff)
            # The frozen primary endpoint uses the recorded serialized-force
            # decision. Independent recomputation is a numerical sensitivity
            # check: retain and report differences, then test whether they
            # actually change any gate or paired endpoint.
            if force_abs_diff > FORCE_REPLAY_TOL:
                force_replay_tolerance_exceedances.append({
                    "call_id": call_id, "formula": row["formula"],
                    "recorded_force_eV_A": float(recorded_force),
                    "replayed_force_eV_A": force,
                    "absolute_difference_eV_A": force_abs_diff,
                })
            if energy_abs_diff > ENERGY_REPLAY_TOL:
                energy_replay_tolerance_exceedances.append({
                    "call_id": call_id, "formula": row["formula"],
                    "recorded_energy_eV_atom": float(recorded_energy),
                    "replayed_energy_eV_atom": energy,
                    "absolute_difference_eV_atom": energy_abs_diff,
                })
            v0 = verifier.validate(closed_loop.CandidateArtifact(
                call_id, row["formula"], output_file.name, output_sha,
                formal_charge_sum=0.0), RUN / "relaxed")
            v0_status = v0.overall
            if v0_status != row.get("v0_status"):
                raise RuntimeError(f"V0 replay mismatch: {call_id}: {v0_status} != {row.get('v0_status')}")
            expected_accept = bool(row.get("error") is None and v0_status == "pass" and force <= THRESHOLD)
            if expected_accept != bool(row.get("accepted")):
                replay_gate_disagreements.append({
                    "call_id": call_id, "formula": row["formula"],
                    "recorded_acceptance": bool(row.get("accepted")),
                    "replayed_acceptance": expected_accept,
                    "recorded_force_eV_A": float(recorded_force),
                    "replayed_force_eV_A": force,
                    "force_abs_difference_eV_A": force_abs_diff,
                })
            if row.get("formula_after") != row["formula"] or structure.composition.reduced_formula != row["formula"]:
                raise RuntimeError(f"relaxed composition mismatch: {call_id}")
            call_audit_rows.append({"call_id": call_id, "formula": row["formula"], "replayed": True,
                                    "recorded_accepted": bool(row.get("accepted")),
                                    "independent_replay_accepted": expected_accept,
                                    "recorded_force_eV_A": float(recorded_force),
                                    "force_eV_A": force, "force_abs_difference_eV_A": force_abs_diff,
                                    "energy_eV_atom": energy, "energy_abs_difference_eV_atom": energy_abs_diff,
                                    "v0_status": v0_status})
        else:
            if row.get("accepted") or row.get("error") is None:
                raise RuntimeError(f"call without output lacks an explicit failure: {call_id}")
            call_audit_rows.append({"call_id": call_id, "formula": row["formula"], "replayed": False,
                                    "accepted": False, "error": row.get("error")})

    by_formula_arm: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {BASELINE: [], CANDIDATE: []})
    for assignment in assignments:
        formula, arm = assignment["formula"], assignment["arm"]
        if arm not in (BASELINE, CANDIDATE) or assignment["call_id"] not in call_by_id:
            raise RuntimeError("assignment has invalid arm or missing raw call")
        call = call_by_id[assignment["call_id"]]
        for field in ("formula", "accepted", "output_sha256", "v0_status", "max_force_serialized_eV_A", "error"):
            assign_field = "serialized_force_eV_A" if field == "max_force_serialized_eV_A" else field
            if assignment.get(assign_field) != call.get(field):
                raise RuntimeError(f"assignment/raw-call mismatch: {formula}/{arm}/{assignment['slot']}/{field}")
        by_formula_arm[formula][arm].append(assignment)

    specs = {row["formula"]: row for row in protocol["candidate_records"]}
    if len(specs) != 432 or set(by_formula_arm) != set(specs):
        raise RuntimeError("frozen formula set differs from the scored assignment ledger")
    if len({row["pair_cluster"] for row in specs.values()}) != 72 or len({row["a_site"] for row in specs.values()}) != 6:
        raise RuntimeError("frozen grid dimensions changed")
    trace_by_formula = defaultdict(list)
    for row in trace:
        trace_by_formula[row.get("formula")].append(row)

    formula_rows = []
    for formula, spec in sorted(specs.items()):
        rows_for_formula = by_formula_arm[formula]
        for arm in (BASELINE, CANDIDATE):
            arm_rows = sorted(rows_for_formula[arm], key=lambda x: x["slot"])
            if [x["slot"] for x in arm_rows] != [1, 2, 3, 4]:
                raise RuntimeError(f"missing/repeated assignment slots: {formula}/{arm}")
            if arm == BASELINE:
                expected_calls = [call_by_id[x["call_id"]] for x in arm_rows]
                if [x.get("ordering_class") for x in expected_calls] != [0, 1, 2, 3] or any(x.get("optimizer") != "FIRE" for x in expected_calls):
                    raise RuntimeError(f"baseline is not static 0/1/2/3 FIRE: {formula}")
            else:
                candidate_calls = [call_by_id[x["call_id"]] for x in arm_rows]
                if [x.get("ordering_class") for x in candidate_calls[:3]] != [0, 1, 2] or any(x.get("optimizer") != "FIRE" for x in candidate_calls[:3]):
                    raise RuntimeError(f"candidate base slots differ from 0/1/2 FIRE: {formula}")
                fourth = arm_rows[3]
                fire_by_class = {int(x.get("ordering_class")): x for x in candidate_calls[:3]}
                qualifying = []
                for cls in (0, 1, 2):
                    call = fire_by_class[cls]
                    if (call.get("error") is None and call.get("v0_status") == "pass" and call.get("output_sha256")
                            and call.get("max_force_serialized_eV_A") is not None
                            and math.isfinite(float(call["max_force_serialized_eV_A"]))
                            and float(call["max_force_serialized_eV_A"]) > THRESHOLD):
                        qualifying.append(cls)
                if fourth.get("method") == "LBFGS_force_only_continuation":
                    repair_call = call_by_id[fourth["call_id"]]
                    if not qualifying or int(repair_call["ordering_class"]) != min(qualifying):
                        raise RuntimeError(f"repair did not follow the locked lowest-index trigger: {formula}")
                    source_call = fire_by_class[min(qualifying)]
                    if (repair_call.get("optimizer") != "LBFGS"
                            or repair_call.get("repair_parent_call_id") != source_call["call_id"]
                            or repair_call.get("repair_parent_terminal_sha256") != source_call["output_sha256"]
                            or sha(inside(repair_call["input_path"], RUN, "inputs")) != source_call["output_sha256"]):
                        raise RuntimeError(f"repair continuation lineage/hash mismatch: {formula}")
                elif fourth.get("method") == "FIRE_catalog_fallback_class3":
                    if qualifying:
                        raise RuntimeError(f"eligible force-only failure was not repaired: {formula}")
                    base_class3 = rows_for_formula[BASELINE][3]
                    if fourth["call_id"] != base_class3["call_id"]:
                        raise RuntimeError(f"candidate fallback did not reuse class 3: {formula}")
                else:
                    raise RuntimeError(f"unknown candidate fourth-slot policy: {formula}")

        # Cross-arm assignments must share exactly classes 0/1/2; the class-3
        # comparator is retained only when the failure trigger does not fire.
        base_ids = [x["call_id"] for x in sorted(rows_for_formula[BASELINE], key=lambda x: x["slot"])]
        cand_ids = [x["call_id"] for x in sorted(rows_for_formula[CANDIDATE], key=lambda x: x["slot"])]
        if base_ids[:3] != cand_ids[:3]:
            raise RuntimeError(f"cross-arm common calls differ: {formula}")

        counts = {}
        for arm in (BASELINE, CANDIDATE):
            structures = []
            for assignment in rows_for_formula[arm]:
                call = call_by_id[assignment["call_id"]]
                if call.get("accepted"):
                    output_file = inside(call["output_path"], RUN, "relaxed")
                    structures.append(Structure.from_file(output_file, primitive=False))
            counts[arm] = len(MATCHER.group_structures(structures)) if structures else 0
        formula_rows.append({
            "formula": formula, "a_site": spec["a_site"], "pair_cluster": spec["pair_cluster"],
            "baseline_unique_accepted_basins": counts[BASELINE],
            "candidate_unique_accepted_basins": counts[CANDIDATE],
            "delta": counts[CANDIDATE] - counts[BASELINE],
            "candidate_fourth_method": rows_for_formula[CANDIDATE][3]["method"],
        })

    delta = np.asarray([x["delta"] for x in formula_rows], dtype=float)
    pairs = sorted({x["pair_cluster"] for x in formula_rows})
    a_sites = sorted({x["a_site"] for x in formula_rows})
    pi, ai = {x: i for i, x in enumerate(pairs)}, {x: i for i, x in enumerate(a_sites)}
    matrix = np.zeros((len(pairs), len(a_sites)), dtype=float)
    for row in formula_rows:
        matrix[pi[row["pair_cluster"]], ai[row["a_site"]]] = row["delta"]
    if matrix.shape != (72, 6):
        raise RuntimeError(f"paired dependency grid incomplete: {matrix.shape}")
    formula_ci = bootstrap_formula(delta, 28031)
    pair_ci = bootstrap_pairs(matrix, 28032)
    two_way_ci, empty = bootstrap_two_way(matrix, 28033)
    mean = float(delta.mean())
    expected_analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    if expected_analysis.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("analysis result is not bound to the frozen protocol")
    comparisons = {
        "mean_delta_candidate_minus_baseline_per_formula": mean,
        "formula_iid_bootstrap_95_ci_secondary": formula_ci,
        "pair_cluster_bootstrap_95_ci_primary": pair_ci,
        "two_way_a_site_x_pair_cluster_bootstrap_95_ci_primary": two_way_ci,
        "formula_count": len(formula_rows),
        "pair_cluster_count": len(pairs),
        "unique_oracle_calls": len(calls),
    }
    for key, value in comparisons.items():
        if expected_analysis.get(key) != value:
            raise RuntimeError(f"independent statistic replay mismatch for {key}: {value!r} != {expected_analysis.get(key)!r}")
    paired_expected = expected_analysis.get("paired_formula_results", {})
    paired_replayed = {
        "positive": int((delta > 0).sum()),
        "negative": int((delta < 0).sum()),
        "tied": int((delta == 0).sum()),
    }
    if paired_expected != paired_replayed:
        raise RuntimeError(f"independent paired-result counts differ: {paired_replayed!r} != {paired_expected!r}")
    method_counts = {method: sum(row["candidate_fourth_method"] == method for row in formula_rows)
                     for method in {row["candidate_fourth_method"] for row in formula_rows}}
    if expected_analysis.get("candidate_fourth_slot_methods") != method_counts:
        raise RuntimeError("candidate fourth-slot method frequencies differ from primary analysis")
    yield_means = {
        "baseline_static_0123": float(np.mean([x["baseline_unique_accepted_basins"] for x in formula_rows])),
        "candidate_failure_triggered_lbfgs": float(np.mean([x["candidate_unique_accepted_basins"] for x in formula_rows])),
    }
    class_summary = {}
    for cls in (0, 1, 2, 3):
        cls_rows = [x for x in calls if x.get("call_type") == "catalog_fire" and x.get("ordering_class") == cls]
        class_summary[str(cls)] = {
            "calls": len(cls_rows),
            "accepted": sum(bool(x.get("accepted")) for x in cls_rows),
            "v0_pass_force_only_failure": sum(
                x.get("error") is None and x.get("v0_status") == "pass" and x.get("force_pass_serialized") is False
                for x in cls_rows),
            "runtime_or_pipeline_failure": sum(x.get("error") is not None for x in cls_rows),
        }
    repair_rows = [x for x in calls if x.get("call_type") == "force_only_lbfgs_continuation"]
    candidate_fourth_rows = [x for x in assignments if x["arm"] == CANDIDATE and x["slot"] == 4]
    repair_summary = {
        "triggered_formula_count": len(repair_rows),
        "repair_calls_accepted": sum(bool(x.get("accepted")) for x in repair_rows),
        "repair_calls_failed": sum(not bool(x.get("accepted")) for x in repair_rows),
        "candidate_fourth_slot_lbfgs_count": sum(x.get("method") == "LBFGS_force_only_continuation" for x in candidate_fourth_rows),
        "candidate_fourth_slot_class3_fallback_count": sum(x.get("method") == "FIRE_catalog_fallback_class3" for x in candidate_fourth_rows),
        "repair_recovery_fraction": (sum(bool(x.get("accepted")) for x in repair_rows) / len(repair_rows)) if repair_rows else None,
    }
    if expected_analysis.get("mean_unique_accepted_basins_per_formula") != yield_means:
        raise RuntimeError("arm-wise basin-yield means differ from analysis output")
    if expected_analysis.get("class_0_to_3_fire_call_summary") != class_summary:
        raise RuntimeError("class-wise runtime/acceptance summaries differ from analysis output")
    if expected_analysis.get("repair_summary") != repair_summary:
        raise RuntimeError("repair-trigger summary differs from analysis output")
    primary_success = pair_ci[0] > 0 and two_way_ci[0] > 0
    if expected_analysis.get("dual_primary_interval_lower_bounds_above_zero") != primary_success:
        raise RuntimeError("primary decision differs between independent and submitted analysis")
    # Re-evaluate the same fixed call ledger using the independently replayed
    # CHGNet forces. Keep the frozen run's serialized-force endpoint primary;
    # this second calculation is a hardware/numerical sensitivity check.
    replay_accept_by_call = {row["call_id"]: bool(row.get("independent_replay_accepted", False))
                             for row in call_audit_rows}
    replay_formula_rows = []
    for formula, spec in sorted(specs.items()):
        replay_counts = {}
        for arm in (BASELINE, CANDIDATE):
            replay_structures = []
            for assignment in by_formula_arm[formula][arm]:
                if replay_accept_by_call.get(assignment["call_id"], False):
                    output_file = inside(call_by_id[assignment["call_id"]]["output_path"], RUN, "relaxed")
                    replay_structures.append(Structure.from_file(output_file, primitive=False))
            replay_counts[arm] = (len(MATCHER.group_structures(replay_structures))
                                  if replay_structures else 0)
        replay_formula_rows.append({
            "formula": formula, "a_site": spec["a_site"], "pair_cluster": spec["pair_cluster"],
            "baseline_unique_accepted_basins": replay_counts[BASELINE],
            "candidate_unique_accepted_basins": replay_counts[CANDIDATE],
            "delta": replay_counts[CANDIDATE] - replay_counts[BASELINE],
        })
    replay_delta = np.asarray([row["delta"] for row in replay_formula_rows], dtype=float)
    replay_matrix = np.zeros_like(matrix)
    for row in replay_formula_rows:
        replay_matrix[pi[row["pair_cluster"]], ai[row["a_site"]]] = row["delta"]
    replay_formula_ci = bootstrap_formula(replay_delta, 28031)
    replay_pair_ci = bootstrap_pairs(replay_matrix, 28032)
    replay_two_way_ci, replay_empty = bootstrap_two_way(replay_matrix, 28033)
    replay_primary_success = replay_pair_ci[0] > 0 and replay_two_way_ci[0] > 0

    replay_force_by_call = {row["call_id"]: float(row["force_eV_A"])
                            for row in call_audit_rows if row.get("replayed")}
    trigger_disagreements = []
    for formula, arm_rows in by_formula_arm.items():
        candidate_rows = sorted(arm_rows[CANDIDATE], key=lambda row: row["slot"])
        first_three = [call_by_id[row["call_id"]] for row in candidate_rows[:3]]
        recorded_qualifying, replayed_qualifying = [], []
        for cls, call in enumerate(first_three):
            recorded_force = call.get("max_force_serialized_eV_A")
            if (call.get("error") is None and call.get("v0_status") == "pass"
                    and recorded_force is not None and math.isfinite(float(recorded_force))
                    and float(recorded_force) > THRESHOLD):
                recorded_qualifying.append(cls)
            replay_force = replay_force_by_call.get(call["call_id"])
            if (call.get("error") is None and call.get("v0_status") == "pass"
                    and replay_force is not None and math.isfinite(replay_force)
                    and replay_force > THRESHOLD):
                replayed_qualifying.append(cls)
        if recorded_qualifying != replayed_qualifying:
            trigger_disagreements.append({
                "formula": formula, "recorded_qualifying_classes": recorded_qualifying,
                "replayed_qualifying_classes": replayed_qualifying,
            })

    replay_sensitivity = {
        "force_replay_tolerance_eV_A": FORCE_REPLAY_TOL,
        "energy_replay_tolerance_eV_atom": ENERGY_REPLAY_TOL,
        "max_abs_force_difference_eV_A": max(force_abs_diffs) if force_abs_diffs else None,
        "median_abs_force_difference_eV_A": float(np.median(force_abs_diffs)) if force_abs_diffs else None,
        "p95_abs_force_difference_eV_A": float(np.quantile(force_abs_diffs, 0.95)) if force_abs_diffs else None,
        "max_abs_energy_difference_eV_atom": max(energy_abs_diffs) if energy_abs_diffs else None,
        "force_replay_tolerance_exceedance_count": len(force_replay_tolerance_exceedances),
        "force_replay_tolerance_exceedances": force_replay_tolerance_exceedances,
        "energy_replay_tolerance_exceedance_count": len(energy_replay_tolerance_exceedances),
        "energy_replay_tolerance_exceedances": energy_replay_tolerance_exceedances,
        "acceptance_gate_disagreement_count": len(replay_gate_disagreements),
        "acceptance_gate_disagreements": replay_gate_disagreements,
        "candidate_trigger_disagreement_count": len(trigger_disagreements),
        "candidate_trigger_disagreements": trigger_disagreements,
        "replayed_mean_delta_candidate_minus_baseline_per_formula": float(replay_delta.mean()),
        "replayed_formula_iid_bootstrap_95_ci_secondary": replay_formula_ci,
        "replayed_pair_cluster_bootstrap_95_ci_primary": replay_pair_ci,
        "replayed_two_way_a_site_x_pair_cluster_bootstrap_95_ci_primary": replay_two_way_ci,
        "replayed_dual_primary_interval_criterion_met": replay_primary_success,
        "replayed_two_way_bootstrap_empty_weight_draws": replay_empty,
    }
    method_counts = defaultdict(int)
    for row in formula_rows:
        method_counts[row["candidate_fourth_method"]] += 1
    numerical_replay_differences = bool(force_replay_tolerance_exceedances or energy_replay_tolerance_exceedances)
    if replay_primary_success and not replay_gate_disagreements and not trigger_disagreements and not numerical_replay_differences:
        audit_status = "INDEPENDENT_V280_AUDIT_PASS_PRIMARY_AND_CROSS_HARDWARE_SENSITIVITY"
    elif replay_primary_success:
        audit_status = "INDEPENDENT_V280_AUDIT_PASS_WITH_NUMERICAL_OR_GATE_SENSITIVITY_REPORTED"
    else:
        audit_status = "PRIMARY_ANALYSIS_REPLAYED_CROSS_HARDWARE_SENSITIVITY_CRITERION_NOT_MET"
    audit = {
        "status": audit_status,
        "audit_implementation_note": "V4 is a separately versioned audit-only implementation. It verifies hashes, V0, continuation lineage, basin counts and locked statistics. The protocol-defined primary endpoint uses the recorded serialized-force decisions from the scoring run. Independently recomputed CHGNet values are retained as a numerical sensitivity replay: tolerance exceedances are reported explicitly, and paired-yield sensitivity is evaluated from any actual acceptance/trigger changes. Numerical differences never alter the frozen run or primary endpoint.",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash,
        "audit_script_sha256": sha(SCRIPT),
        "raw_calls_sha256": sha(calls_file),
        "raw_assignments_sha256": sha(assignments_file),
        "raw_trace_sha256": sha(trace_file),
        "analysis_json_sha256": sha(ANALYSIS_PATH),
        "independent_replay": {
            "raw_call_records_checked": len(calls),
            "serialized_force_and_energy_recomputed": sum(bool(x.get("replayed")) for x in call_audit_rows),
            "v0_replayed": sum(bool(x.get("replayed")) for x in call_audit_rows),
            "all_input_output_hashes_verified": True,
            "all_raw_record_hashes_verified": True,
            "assignments_checked": len(assignments),
            "basin_counts_recomputed_with_independent_code": len(formula_rows) * 2,
            "bootstrap_recomputed_with_independent_code": True,
        },
        **comparisons,
        "candidate_fourth_slot_methods": dict(method_counts),
        "mean_unique_accepted_basins_per_formula": yield_means,
        "class_0_to_3_fire_call_summary": class_summary,
        "repair_summary": repair_summary,
        "two_way_bootstrap_empty_weight_draws": empty,
        "dual_primary_interval_criterion_met": primary_success,
        "independent_force_replay_sensitivity": replay_sensitivity,
        "per_formula": formula_rows,
        "raw_call_replay_records": call_audit_rows,
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    report = [
        "# V280 independent confirmation audit",
        "",
        f"- Independent audit status: **{audit['status']}**.",
        f"- Frozen protocol SHA-256: `{protocol_hash}`.",
        f"- Raw calls/assignments: {len(calls)} / {len(assignments)}; verified output force, energy, V0 gates, input/output digests and recorded raw-line hashes.",
        f"- Recomputed matched-grid mean yield difference: **{mean:+.4f}** unique accepted basins per formula.",
        f"- Pair-cluster 95% CI: **[{pair_ci[0]:+.4f}, {pair_ci[1]:+.4f}]**.",
        f"- Two-way A-site × B-pair 95% CI: **[{two_way_ci[0]:+.4f}, {two_way_ci[1]:+.4f}]**.",
        f"- Locked dual-CI success criterion: **{primary_success}**.",
        f"- Independent numerical replay: max |Δforce| **{replay_sensitivity['max_abs_force_difference_eV_A']:.3g} eV Å⁻¹**, max |Δenergy| **{replay_sensitivity['max_abs_energy_difference_eV_atom']:.3g} eV atom⁻¹**; force/energy tolerance exceedances **{len(force_replay_tolerance_exceedances)} / {len(energy_replay_tolerance_exceedances)}**, acceptance-gate disagreements **{len(replay_gate_disagreements)}**, trigger disagreements **{len(trigger_disagreements)}**.",
        f"- Independent-force sensitivity 95% CIs: pair-cluster **[{replay_pair_ci[0]:+.4f}, {replay_pair_ci[1]:+.4f}]**; two-way **[{replay_two_way_ci[0]:+.4f}, {replay_two_way_ci[1]:+.4f}]**; dual criterion **{replay_primary_success}**.",
        f"- Audit JSON SHA-256: `{sha(AUDIT_PATH)}`.",
        "",
    ]
    AUDIT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in (
        "status", "protocol_sha256", "raw_calls_sha256", "raw_assignments_sha256",
        "mean_delta_candidate_minus_baseline_per_formula", "pair_cluster_bootstrap_95_ci_primary",
        "two_way_a_site_x_pair_cluster_bootstrap_95_ci_primary", "dual_primary_interval_criterion_met",
    )}, indent=2))


if __name__ == "__main__":
    main()
