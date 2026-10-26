from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]
HOLD = PROJECT / "runs" / "holdout" / "v280_failure_repair_fresh_b4b4_v1"
GEOMETRY = HOLD / "geometry"
PREFLIGHT = GEOMETRY / "preflight.json"
PREFLIGHT_MANIFEST = GEOMETRY / "preflight_manifest.json"
SCAN = PROJECT / "source_data" / "prior_identity_scan_v280.json"
CONFIG = PROJECT / "material_generation" / "config" / "v280_repair_fresh_b4b4_v1.yaml"
PROTOCOL = PROJECT / "protocol" / "v280_failure_repair_confirmation_v1.json"
LOCK = PROJECT / "protocol" / "v280_failure_repair_confirmation_v1.lock.json"
RUN = HOLD / "scored"
V268 = Path("/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V279/source_data/V268")
RUNTIME = Path("/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V279/runtime/chgnet_0_4_2")
DEV_SMOKE = Path("/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V276/runs/development/v276_runtime_smoke/smoke.json")
DEV_V0_SMOKE = Path("/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V276/runs/development/v276_v0_development_smoke.json")
DEV_PROBE = V268 / "development" / "v268_pair_policy_development" / "parents" / "Ba2MnMoO6_parent.cif"
ORACLE_MODULE = V268 / "material_generation" / "validation" / "chgnet_oracle.py"
V0_MODULE = V268 / "material_generation" / "agent" / "closed_loop.py"
CHECKPOINT = "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"
FMAX = 0.05
MAX_STEPS = 500
SIGMA = 0.03
DEVICE = "cpu"
SEEDS_BY_CLASS = {0: 101, 1: 211, 2: 307, 3: 401, 4: 503, 5: 607}
BASELINE_ARM = "baseline_static_0123"
CANDIDATE_ARM = "candidate_failure_triggered_lbfgs"
BASELINE_CLASSES = (0, 1, 2, 3)
CANDIDATE_BASE_CLASSES = (0, 1, 2)
THRESHOLD = FMAX + 1e-6


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def record_digest(record: dict) -> str:
    payload = {k: v for k, v in record.items() if k != "raw_record_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def displaced(structure: Structure, seed: int) -> Structure:
    out = structure.copy()
    rng = np.random.default_rng(seed)
    for idx, vector in enumerate(rng.normal(0.0, SIGMA, size=(len(out), 3))):
        out.translate_sites([idx], vector, frac_coords=False, to_unit_cell=True)
    return out


def verify_protocol(protocol: dict[str, Any], lock: dict[str, Any]) -> None:
    if protocol.get("status") != "LOCKED_BEFORE_HOLDOUT_SCORING":
        raise RuntimeError("protocol is not frozen")
    if sha(PROTOCOL) != lock.get("protocol_sha256"):
        raise RuntimeError("protocol lock hash mismatch")
    hold = protocol["holdout"]
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    if (scan.get("identity_only") is not True or scan.get("scores_read") is not False
            or scan.get("outcomes_read") is not False
            or scan.get("versions_requested") != list(range(268, 280))):
        raise RuntimeError("freshness scan is not complete and identity-only")
    checks = {
        "preflight_sha256": sha(PREFLIGHT),
        "preflight_manifest_sha256": sha(PREFLIGHT_MANIFEST),
        "identity_scan_sha256": sha(SCAN),
        "config_sha256": sha(CONFIG),
    }
    for key, value in checks.items():
        if value != hold.get(key):
            raise RuntimeError(f"frozen input mismatch: {key}")
    for relative, expected in protocol["integrity"]["code_sha256"].items():
        path = (PROJECT / relative).resolve(strict=True)
        if sha(path) != expected:
            raise RuntimeError(f"frozen code changed: {relative}")
    for evidence in protocol["development_boundary"]["evidence_sha256"].values():
        path = (PROJECT / evidence["path"]).resolve(strict=True)
        path.relative_to(PROJECT.resolve(strict=True))
        if sha(path) != evidence["sha256"]:
            raise RuntimeError(f"development evidence changed after protocol lock: {evidence['path']}")
    records = protocol.get("candidate_records", [])
    if len(records) != 432 or len({x["formula"] for x in records}) != 432:
        raise RuntimeError("formula count or uniqueness mismatch")
    pairs = {x["pair_cluster"] for x in records}
    if len(pairs) != 72 or hold.get("formula_count") != 432 or hold.get("pair_cluster_count") != 72:
        raise RuntimeError("fresh holdout size mismatch")
    if protocol["arms"][BASELINE_ARM]["classes"] != list(BASELINE_CLASSES):
        raise RuntimeError("baseline arm changed")
    if protocol["arms"][CANDIDATE_ARM]["base_classes"] != list(CANDIDATE_BASE_CLASSES):
        raise RuntimeError("candidate base arm changed")
    if protocol["evaluation"].get("assignments_per_formula_per_arm") != 4:
        raise RuntimeError("matched budget changed")
    if protocol["evaluation"].get("serialized_force_threshold_eV_A") != FMAX:
        raise RuntimeError("force gate changed")
    for spec in records:
        if spec.get("formal_charge_sum") != 0 or not spec.get("preflight_checks", {}).get("neutral_parent"):
            raise RuntimeError(f"neutrality gate failed: {spec.get('formula')}")
        parent = (GEOMETRY / spec["parent_cif"]).resolve(strict=True)
        parent.relative_to(GEOMETRY.resolve(strict=True))
        if sha(parent) != spec["parent_sha256"]:
            raise RuntimeError(f"parent CIF hash mismatch: {spec['formula']}")
        if len(spec.get("class_records", [])) != 6:
            raise RuntimeError(f"six-class catalog missing: {spec['formula']}")
        for cls in spec["class_records"]:
            path = (GEOMETRY / cls["class_cif"]).resolve(strict=True)
            path.relative_to(GEOMETRY.resolve(strict=True))
            if sha(path) != cls["class_cif_sha256"]:
                raise RuntimeError(f"catalog CIF hash mismatch: {spec['formula']}/class{cls['ordering_class']}")


def contained(path_text: str, root: Path, subdir: str) -> Path:
    path = Path(path_text)
    resolved = (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
    resolved.relative_to((root / subdir).resolve(strict=True))
    return resolved


def main(preflight_only: bool = False, resume: bool = False) -> None:
    if not PROTOCOL.exists() or not LOCK.exists():
        raise FileNotFoundError("freeze V280 protocol before scoring")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    verify_protocol(protocol, lock)
    if protocol["evaluation"].get("device") != DEVICE:
        raise RuntimeError("device differs from frozen CPU protocol")
    dev_meta = protocol["development_runtime_probe"]
    if (sha(DEV_PROBE) != dev_meta["input_sha256"]
            or sha(DEV_SMOKE) != dev_meta["smoke_result_sha256"]
            or sha(DEV_V0_SMOKE) != dev_meta["v0_smoke_result_sha256"]):
        raise RuntimeError("development-only runtime probe files changed")
    v0smoke = json.loads(DEV_V0_SMOKE.read_text(encoding="utf-8"))
    if sha(DEV_V0_SMOKE) != dev_meta["v0_smoke_result_sha256"] or v0smoke.get("status") != "V0_DEVELOPMENT_SMOKE_PASS":
        raise RuntimeError("V268-only V0 smoke evidence mismatch")
    oracle_mod = load_module(ORACLE_MODULE, "v280_chgnet_oracle")
    oracle = oracle_mod.CHGNetOracle(RUNTIME, fmax=FMAX, max_steps=MAX_STEPS)
    if oracle.checkpoint_sha256 != CHECKPOINT or oracle.package_version != "0.4.2" or oracle.device != DEVICE:
        raise RuntimeError("pinned CHGNet runtime validation failed")
    if preflight_only:
        if RUN.exists():
            raise RuntimeError("pre-score check must run before holdout output directory exists")
        probe = Structure.from_file(DEV_PROBE, primitive=False)
        pred = oracle.model.predict_structure(probe)
        energy = float(np.asarray(pred["e"]).reshape(-1)[0])
        forces = np.asarray(pred["f"], dtype=float)
        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
        if not math.isfinite(energy) or not math.isfinite(max_force):
            raise RuntimeError("development-only forward pass returned non-finite values")
        print(json.dumps({
            "status": "PRE_SCORE_PROTOCOL_AND_RUNTIME_CHECK_PASS",
            "protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(HERE),
            "checkpoint_sha256": oracle.checkpoint_sha256,
            "device": oracle.device, "package_version": oracle.package_version,
            "development_probe_sha256": sha(DEV_PROBE),
            "development_probe_energy_eV_atom": energy,
            "development_probe_max_force_eV_A": max_force,
            "formula_count": len(protocol["candidate_records"]),
            "assignments_per_arm": len(protocol["candidate_records"]) * 4,
            "holdout_scoring_performed": False,
        }, indent=2))
        return

    from chgnet.model import StructOptimizer
    sys.path.insert(0, str(V268))
    from material_generation.agent.closed_loop import CandidateArtifact, Verifier
    verifier = Verifier(min_interatomic_distance_angstrom=0.75)
    fire_optimizer = StructOptimizer(model=oracle.model, use_device=DEVICE, optimizer_class="FIRE")

    if resume:
        if not RUN.is_dir() or (RUN / "run_completed.json").exists():
            raise RuntimeError("resume requires an incomplete existing run")
        started = json.loads((RUN / "run_started.json").read_text(encoding="utf-8"))
        if started.get("protocol_sha256") != sha(PROTOCOL) or not started.get("scoring_started"):
            raise RuntimeError("partial run does not match locked protocol")
    else:
        if RUN.exists():
            raise FileExistsError(f"refusing to overwrite existing scored run: {RUN}")
        RUN.mkdir(parents=True)
        for name in ("parents", "catalog", "inputs", "relaxed"):
            (RUN / name).mkdir()
        for name in ("oracle_calls.jsonl", "arm_assignments.jsonl", "agent_trace.jsonl"):
            (RUN / name).write_text("", encoding="utf-8")
        (RUN / "run_started.json").write_text(json.dumps({
            "status": "RUNNING_LOCKED_V280_FAILURE_REPAIR_CONFIRMATION",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha(PROTOCOL), "protocol_lock_sha256": sha(LOCK),
            "runner_sha256": sha(HERE), "checkpoint_sha256": CHECKPOINT,
            "device": DEVICE, "scoring_started": True,
            "holdout_outcomes_used_for_selection": False,
        }, indent=2) + "\n", encoding="utf-8")

    calls_path = RUN / "oracle_calls.jsonl"
    assignments_path = RUN / "arm_assignments.jsonl"
    trace_path = RUN / "agent_trace.jsonl"
    calls = jsonl(calls_path)
    assignments = jsonl(assignments_path)
    call_by_id = {row["call_id"]: row for row in calls}
    if len(call_by_id) != len(calls):
        raise RuntimeError("duplicate call identifiers in raw log")

    def force_saved(structure: Structure) -> float:
        pred = oracle.model.predict_structure(structure)
        forces = np.asarray(pred["f"], dtype=float)
        return float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")

    def v0_replay(output: Path, call_id: str, formula: str, digest: str) -> tuple[str, list[dict]]:
        report = verifier.validate(CandidateArtifact(call_id, formula, output.name, digest,
                                                     formal_charge_sum=0.0), RUN / "relaxed")
        return report.overall, [x.__dict__ for x in report.checks]

    def verify_logged_call(record: dict, expected_formula: str) -> None:
        if record.get("raw_record_sha256") != record_digest(record):
            raise RuntimeError(f"raw record digest mismatch: {record.get('call_id')}")
        if record.get("formula") != expected_formula or record.get("checkpoint_sha256") != CHECKPOINT:
            raise RuntimeError(f"call identity/checkpoint mismatch: {record.get('call_id')}")
        if not record.get("output_sha256"):
            return
        output = contained(record["output_path"], RUN, "relaxed")
        if sha(output) != record["output_sha256"]:
            raise RuntimeError(f"saved call output hash mismatch: {record['call_id']}")

    def execute_fire(spec: dict, cls: int) -> dict:
        formula = spec["formula"]
        class_spec = next(x for x in spec["class_records"] if int(x["ordering_class"]) == cls)
        source = (GEOMETRY / class_spec["class_cif"]).resolve(strict=True)
        catalog_dst = RUN / "catalog" / formula
        catalog_dst.mkdir(exist_ok=True)
        dest = catalog_dst / source.name
        if not dest.exists():
            dest.write_bytes(source.read_bytes())
        if sha(dest) != class_spec["class_cif_sha256"]:
            raise RuntimeError(f"copied catalog CIF hash mismatch: {formula}/class{cls}")
        call_id = hashlib.sha256(f"{formula}\0class\0{cls}\0{SEEDS_BY_CLASS[cls]}\0{sha(dest)}\0FIRE".encode()).hexdigest()
        if call_id in call_by_id:
            row = call_by_id[call_id]
            verify_logged_call(row, formula)
            return row
        inp = RUN / "inputs" / f"{formula}_class{cls}_seed{SEEDS_BY_CLASS[cls]}_FIRE.cif"
        out = RUN / "relaxed" / f"{formula}_class{cls}_seed{SEEDS_BY_CLASS[cls]}_FIRE.cif"
        loaded = displaced(Structure.from_file(dest, primitive=False), SEEDS_BY_CLASS[cls])
        loaded.to(filename=inp, fmt="cif")
        start = time.perf_counter()
        rec: dict[str, Any] = {
            "call_id": call_id, "formula": formula, "family_id": spec["family_id"],
            "pair_cluster": spec["pair_cluster"], "b_site_pair": spec["b_site_pair"],
            "a_site": spec["a_site"], "call_type": "catalog_fire",
            "ordering_class": cls, "seed": SEEDS_BY_CLASS[cls], "optimizer": "FIRE",
            "source_catalog_sha256": sha(dest), "input_path": inp.relative_to(RUN).as_posix(),
            "input_sha256": sha(inp), "checkpoint_sha256": CHECKPOINT,
            "package_version": "0.4.2", "device": DEVICE, "fmax_eV_A": FMAX,
            "max_steps": MAX_STEPS, "formal_charge_sum": 0,
        }
        warning_sample = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                relaxed = fire_optimizer.relax(Structure.from_file(inp, primitive=False), fmax=FMAX,
                    steps=MAX_STEPS, relax_cell=True, verbose=False, assign_magmoms=True)
                final = relaxed["final_structure"]
                pred = oracle.model.predict_structure(final)
                forces = np.asarray(pred["f"], dtype=float)
                inmem_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
                energy = float(np.asarray(pred["e"]).reshape(-1)[0])
                final.to(filename=out, fmt="cif")
                warning_sample = [f"{w.category.__name__}: {str(w.message)[:180]}" for w in caught[:6]]
            digest = sha(out)
            v0_status, v0_checks = v0_replay(out, call_id, formula, digest)
            serialized_force = force_saved(Structure.from_file(out, primitive=False))
            energy_saved = finite_or_none(energy)
            inmem_force_saved = finite_or_none(inmem_force)
            serialized_force_saved = finite_or_none(serialized_force)
            rec.update({
                "output_path": out.relative_to(RUN).as_posix(), "output_sha256": digest,
                "formula_after": final.composition.reduced_formula,
                "energy_eV_atom": energy_saved, "max_force_in_memory_eV_A": inmem_force_saved,
                "max_force_serialized_eV_A": serialized_force_saved,
                "force_pass_serialized": bool(serialized_force_saved is not None and serialized_force_saved <= THRESHOLD),
                "v0_status": v0_status, "v0_checks": v0_checks,
                "steps": len(relaxed["trajectory"].energies),
                "runtime_warning_count": len(caught), "runtime_warning_sample": warning_sample,
                "elapsed_seconds": time.perf_counter() - start, "error": None,
            })
        except Exception as exc:
            rec.update({
                "output_path": out.relative_to(RUN).as_posix(),
                "output_sha256": sha(out) if out.exists() else "",
                "formula_after": None, "energy_eV_atom": None,
                "max_force_in_memory_eV_A": None, "max_force_serialized_eV_A": None,
                "force_pass_serialized": False, "v0_status": "unknown", "v0_checks": [],
                "steps": None, "runtime_warning_count": None, "runtime_warning_sample": warning_sample,
                "elapsed_seconds": time.perf_counter() - start,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            })
        rec["accepted"] = bool(rec["error"] is None and rec["force_pass_serialized"] and rec["v0_status"] == "pass")
        rec["raw_record_sha256"] = record_digest(rec)
        with calls_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(rec, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        with trace_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "formula": formula,
                "agent": "CandidateEvaluator", "decision": "completed_fire_relaxation",
                "call_id": call_id, "ordering_class": cls, "accepted": rec["accepted"],
                "error": rec["error"],
            }, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        call_by_id[call_id] = rec
        print(json.dumps({"formula": formula, "call": "FIRE", "class": cls,
                          "accepted": rec["accepted"], "error": rec["error"]}, ensure_ascii=False), flush=True)
        return rec

    def verified_force_only_failure(spec: dict, cls: int, record: dict) -> dict | None:
        if record.get("error") is not None or record.get("v0_status") != "pass" or not record.get("output_sha256"):
            return None
        if record.get("formula") != spec["formula"] or record.get("ordering_class") != cls:
            raise RuntimeError("failure record identity differs from planned class")
        if record.get("raw_record_sha256") != record_digest(record):
            raise RuntimeError("failure record hash mismatch at repair gate")
        output = contained(record["output_path"], RUN, "relaxed")
        digest = sha(output)
        if digest != record["output_sha256"]:
            raise RuntimeError("force-failure terminal artifact hash mismatch")
        structure = Structure.from_file(output, primitive=False)
        if structure.composition.reduced_formula != spec["formula"]:
            raise RuntimeError("force-failure terminal formula mismatch")
        v0_status, _ = v0_replay(output, record["call_id"], spec["formula"], digest)
        recomputed = force_saved(structure)
        recorded = record.get("max_force_serialized_eV_A")
        if v0_status != "pass" or recorded is None or not math.isfinite(float(recorded)):
            return None
        if abs(recomputed - float(recorded)) > 1e-6:
            raise RuntimeError("serialized force failed to reproduce at repair gate")
        if recomputed <= THRESHOLD:
            return None
        return {"class": cls, "call_id": record["call_id"], "terminal_path": str(output),
                "terminal_sha256": digest, "serialized_force_eV_A": recomputed,
                "v0_status": v0_status}

    def execute_lbfgs_repair(spec: dict, source: dict) -> dict:
        formula = spec["formula"]
        source_path = Path(source["terminal_path"]).resolve(strict=True)
        source_path.relative_to((RUN / "relaxed").resolve(strict=True))
        repair_id = hashlib.sha256((formula + "\0LBFGS_CONTINUATION\0" + source["call_id"] + "\0" + source["terminal_sha256"]).encode()).hexdigest()
        if repair_id in call_by_id:
            rec = call_by_id[repair_id]
            verify_logged_call(rec, formula)
            return rec
        inp = RUN / "inputs" / f"{formula}_class{source['class']}_LBFGS_from_{source['call_id'][:12]}.cif"
        out = RUN / "relaxed" / f"{formula}_class{source['class']}_LBFGS_repair.cif"
        # Preserve the exact hash-verified terminal artifact as the continuation input;
        # parsing and reserializing a CIF is not byte-preserving.
        if not inp.exists():
            inp.write_bytes(source_path.read_bytes())
        if sha(inp) != source["terminal_sha256"]:
            raise RuntimeError("LBFGS input is not the exact verified FIRE terminal artifact")
        optimizer = StructOptimizer(model=oracle.model, use_device=DEVICE, optimizer_class="LBFGS")
        start = time.perf_counter()
        rec: dict[str, Any] = {
            "call_id": repair_id, "formula": formula, "family_id": spec["family_id"],
            "pair_cluster": spec["pair_cluster"], "b_site_pair": spec["b_site_pair"],
            "a_site": spec["a_site"], "call_type": "force_only_lbfgs_continuation",
            "ordering_class": int(source["class"]), "seed": None, "optimizer": "LBFGS",
            "repair_parent_call_id": source["call_id"],
            "repair_parent_terminal_sha256": source["terminal_sha256"],
            "input_path": inp.relative_to(RUN).as_posix(), "input_sha256": sha(inp),
            "checkpoint_sha256": CHECKPOINT, "package_version": "0.4.2",
            "device": DEVICE, "fmax_eV_A": FMAX, "max_steps": MAX_STEPS,
            "formal_charge_sum": 0,
        }
        warning_sample = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                relaxed = optimizer.relax(Structure.from_file(inp, primitive=False), fmax=FMAX,
                    steps=MAX_STEPS, relax_cell=True, verbose=False, assign_magmoms=True)
                final = relaxed["final_structure"]
                pred = oracle.model.predict_structure(final)
                forces = np.asarray(pred["f"], dtype=float)
                inmem_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
                energy = float(np.asarray(pred["e"]).reshape(-1)[0])
                final.to(filename=out, fmt="cif")
                warning_sample = [f"{w.category.__name__}: {str(w.message)[:180]}" for w in caught[:6]]
            digest = sha(out)
            v0_status, v0_checks = v0_replay(out, repair_id, formula, digest)
            serialized_force = force_saved(Structure.from_file(out, primitive=False))
            energy_saved = finite_or_none(energy)
            inmem_force_saved = finite_or_none(inmem_force)
            serialized_force_saved = finite_or_none(serialized_force)
            rec.update({
                "output_path": out.relative_to(RUN).as_posix(), "output_sha256": digest,
                "formula_after": final.composition.reduced_formula,
                "energy_eV_atom": energy_saved, "max_force_in_memory_eV_A": inmem_force_saved,
                "max_force_serialized_eV_A": serialized_force_saved,
                "force_pass_serialized": bool(serialized_force_saved is not None and serialized_force_saved <= THRESHOLD),
                "v0_status": v0_status, "v0_checks": v0_checks,
                "steps": len(relaxed["trajectory"].energies),
                "runtime_warning_count": len(caught), "runtime_warning_sample": warning_sample,
                "elapsed_seconds": time.perf_counter() - start, "error": None,
            })
        except Exception as exc:
            rec.update({
                "output_path": out.relative_to(RUN).as_posix(),
                "output_sha256": sha(out) if out.exists() else "",
                "formula_after": None, "energy_eV_atom": None,
                "max_force_in_memory_eV_A": None, "max_force_serialized_eV_A": None,
                "force_pass_serialized": False, "v0_status": "unknown", "v0_checks": [],
                "steps": None, "runtime_warning_count": None, "runtime_warning_sample": warning_sample,
                "elapsed_seconds": time.perf_counter() - start,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            })
        rec["accepted"] = bool(rec["error"] is None and rec["force_pass_serialized"] and rec["v0_status"] == "pass")
        rec["raw_record_sha256"] = record_digest(rec)
        with calls_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(rec, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        with trace_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "formula": formula,
                "agent": "CriticRepairAgent", "decision": "lbfgs_force_failure_continuation",
                "source_call_id": source["call_id"], "repair_call_id": repair_id,
                "source_class": source["class"], "accepted": rec["accepted"], "error": rec["error"],
            }, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        call_by_id[repair_id] = rec
        print(json.dumps({"formula": formula, "call": "LBFGS_REPAIR", "source_class": source["class"],
                          "accepted": rec["accepted"], "error": rec["error"]}, ensure_ascii=False), flush=True)
        return rec

    assignment_ids = {row["assignment_id"] for row in assignments}
    for spec in protocol["candidate_records"]:
        formula = spec["formula"]
        expected_assignment_count = 8
        formula_assignments = [row for row in assignments if row["formula"] == formula]
        if len(formula_assignments) == expected_assignment_count:
            slots_by_arm = {arm: sorted(x["slot"] for x in formula_assignments if x["arm"] == arm)
                            for arm in (BASELINE_ARM, CANDIDATE_ARM)}
            if slots_by_arm != {BASELINE_ARM: [1, 2, 3, 4], CANDIDATE_ARM: [1, 2, 3, 4]}:
                raise RuntimeError(f"completed formula assignment slots mismatch: {formula}")
            for prior in formula_assignments:
                call = call_by_id.get(prior["call_id"])
                if call is None:
                    raise RuntimeError(f"completed assignment points to missing raw call: {formula}")
                verify_logged_call(call, formula)
            continue
        elif len(formula_assignments) > expected_assignment_count:
            raise RuntimeError(f"too many assignments for formula {formula}")

        # Four shared/canonical FIRE calls form the comparator and the candidate's
        # no-repair path. Calls are content-addressed and safely reused on resume.
        fire_calls = {cls: execute_fire(spec, cls) for cls in BASELINE_CLASSES}
        repairable = []
        for cls in CANDIDATE_BASE_CLASSES:
            source = verified_force_only_failure(spec, cls, fire_calls[cls])
            if source is not None:
                repairable.append(source)
        repairable.sort(key=lambda row: row["class"])
        repair_source = repairable[0] if repairable else None
        repair_call = execute_lbfgs_repair(spec, repair_source) if repair_source else None

        plan: list[dict] = []
        for slot, cls in enumerate(BASELINE_CLASSES, 1):
            plan.append({"formula": formula, "pair_cluster": spec["pair_cluster"],
                "a_site": spec["a_site"], "arm": BASELINE_ARM, "slot": slot,
                "ordering_class": cls, "method": "FIRE_catalog", "call": fire_calls[cls],
                "reused_cross_arm": cls in CANDIDATE_BASE_CLASSES,
                "repair_source_call_id": None})
        for slot, cls in enumerate(CANDIDATE_BASE_CLASSES, 1):
            plan.append({"formula": formula, "pair_cluster": spec["pair_cluster"],
                "a_site": spec["a_site"], "arm": CANDIDATE_ARM, "slot": slot,
                "ordering_class": cls, "method": "FIRE_catalog", "call": fire_calls[cls],
                "reused_cross_arm": True, "repair_source_call_id": None})
        if repair_call:
            candidate_fourth = {"call": repair_call, "ordering_class": repair_source["class"],
                                "method": "LBFGS_force_only_continuation",
                                "repair_source_call_id": repair_source["call_id"]}
        else:
            candidate_fourth = {"call": fire_calls[3], "ordering_class": 3,
                                "method": "FIRE_catalog_fallback_class3",
                                "repair_source_call_id": None}
        plan.append({"formula": formula, "pair_cluster": spec["pair_cluster"],
            "a_site": spec["a_site"], "arm": CANDIDATE_ARM, "slot": 4,
            **candidate_fourth,
            "reused_cross_arm": candidate_fourth["call"]["call_id"] == fire_calls[3]["call_id"],
        })
        for row in plan:
            call = row.pop("call")
            verify_logged_call(call, formula)
            assignment_id = hashlib.sha256(
                f"{formula}\0{row['arm']}\0{row['slot']}\0{call['call_id']}\0V280_locked_four-call-policy".encode()
            ).hexdigest()
            outrow = {
                "assignment_id": assignment_id,
                **row,
                "call_id": call["call_id"], "accepted": bool(call["accepted"]),
                "output_path": call["output_path"], "output_sha256": call["output_sha256"],
                "serialized_force_eV_A": call["max_force_serialized_eV_A"],
                "v0_status": call["v0_status"], "error": call["error"],
            }
            if assignment_id in assignment_ids:
                old = next(x for x in assignments if x["assignment_id"] == assignment_id)
                if old != outrow:
                    raise RuntimeError(f"resume assignment reconciliation failed: {assignment_id}")
                continue
            with assignments_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(outrow, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            assignments.append(outrow)
            assignment_ids.add(assignment_id)
        with trace_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "formula": formula,
                "agent": "PolicyController", "decision": "matched_four_call_allocation",
                "arm_budget_per_formula": 4,
                "verified_force_only_failure_count_classes_0_2": len(repairable),
                "repair_triggered": repair_call is not None,
                "repair_source_call_id": repair_source["call_id"] if repair_source else None,
                "candidate_fourth_method": candidate_fourth["method"],
            }, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        print(json.dumps({"formula": formula, "unique_calls": len(call_by_id),
                          "repair_triggered": repair_call is not None,
                          "assignments_total": len(assignments)}, ensure_ascii=False), flush=True)

    calls = jsonl(calls_path)
    assignments = jsonl(assignments_path)
    if not (432 * 4 <= len(calls) <= 432 * 5) or len(assignments) != 432 * 8:
        raise RuntimeError(f"incomplete raw run: calls={len(calls)}, assignments={len(assignments)}")
    if len({row["call_id"] for row in calls}) != len(calls):
        raise RuntimeError("duplicate oracle call IDs in final raw log")
    if len({row["assignment_id"] for row in assignments}) != len(assignments):
        raise RuntimeError("duplicate assignment IDs in final assignment ledger")
    formula_set = {row["formula"] for row in protocol["candidate_records"]}
    if {row["formula"] for row in assignments} != formula_set:
        raise RuntimeError("assignment formula set does not match the frozen grid")
    for call in calls:
        if call.get("raw_record_sha256") != record_digest(call):
            raise RuntimeError(f"final raw call digest mismatch: {call.get('call_id')}")
        verify_logged_call(call, call["formula"])
        if call.get("input_path") and call.get("input_sha256"):
            input_path = contained(call["input_path"], RUN, "inputs")
            if sha(input_path) != call["input_sha256"]:
                raise RuntimeError(f"final call input hash mismatch: {call['call_id']}")
    arm_count = {arm: sum(row["arm"] == arm for row in assignments) for arm in (BASELINE_ARM, CANDIDATE_ARM)}
    if arm_count != {BASELINE_ARM: 432 * 4, CANDIDATE_ARM: 432 * 4}:
        raise RuntimeError(f"matched four-call budget violated: {arm_count}")
    for formula in formula_set:
        for arm in (BASELINE_ARM, CANDIDATE_ARM):
            slots = sorted(row["slot"] for row in assignments if row["formula"] == formula and row["arm"] == arm)
            if slots != [1, 2, 3, 4]:
                raise RuntimeError(f"final assignment slots incomplete: {formula}/{arm}/{slots}")
    completed = {
        "status": "COMPLETED_LOCKED_V280_FAILURE_REPAIR_CONFIRMATION",
        "started_utc": json.loads((RUN / "run_started.json").read_text(encoding="utf-8"))["started_utc"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha(PROTOCOL), "protocol_lock_sha256": sha(LOCK),
        "runner_sha256": sha(HERE), "preflight_sha256": sha(PREFLIGHT),
        "config_sha256": sha(CONFIG), "identity_scan_sha256": sha(SCAN),
        "oracle_calls_sha256": sha(calls_path), "arm_assignments_sha256": sha(assignments_path),
        "agent_trace_sha256": sha(trace_path), "formula_count": 432,
        "pair_cluster_count": 72, "unique_oracle_calls": len(calls),
        "assignment_rows": len(assignments), "assignments_per_arm": arm_count,
        "repair_triggered_formula_count": sum(1 for x in assignments if x["arm"] == CANDIDATE_ARM and x["slot"] == 4 and x["method"] == "LBFGS_force_only_continuation"),
        "class_seed_map": {str(k): v for k, v in SEEDS_BY_CLASS.items()},
        "checkpoint_sha256": oracle.checkpoint_sha256, "package_version": oracle.package_version,
        "device": oracle.device, "max_serialized_force_threshold_eV_A": FMAX,
        "V0_minimum_distance_angstrom": 0.75, "optimizer_baseline": "FIRE",
        "candidate_repair_optimizer": "LBFGS", "holdout_outcomes_used_for_selection": False,
    }
    (RUN / "run_completed.json").write_text(json.dumps(completed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(completed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.preflight_only, args.resume)
