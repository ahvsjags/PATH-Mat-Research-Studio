"""Runnable interface joining PATH-Mat's candidate-design agent components."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import yaml

from agent.closed_loop import CandidateArtifact, VerifiedMemory, sha256_file
from agent.closed_loop_controller import run_candidate
from agent.ordered_double_perovskite_agent import _candidate_id, build_structure, propose
from agent.pathmat_design_agent import AgentTrace, PathMatRouteAgent


def _json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_workflow(config_path: Path, output_root: Path, *, split: str = "development",
                 inject_byte_fault: bool = False, memory_seed: Path | None = None,
                 route_model: Path | None = None, route_vendor: Path | None = None) -> dict[str, Any]:
    """Generate, check, challenge and restore candidate artifacts.

    Route hypotheses are always generated. Frozen route-model scoring is
    optional and requires its explicit model and vendor paths. No MLIP or DFT
    evaluation is launched by this artifact-level entry point.
    """
    if split not in {"development", "confirmatory"}:
        raise ValueError("split must be development or confirmatory")
    if (route_model is None) != (route_vendor is None):
        raise ValueError("route-model and route-vendor must be supplied together")
    config_path, output_root = config_path.resolve(strict=True), output_root.resolve()
    if output_root.exists():
        raise FileExistsError("use a new output directory; workflow outputs are not overwritten")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if cfg.get("schema_version") != 1:
        raise ValueError("config must declare schema_version 1")
    task = cfg["design_task"]
    if task.get("prototype") != "rocksalt_ordered_double_perovskite_A2BBpO6":
        raise ValueError("the packaged native generator uses the A2BBpO6 prototype")
    output_root.mkdir(parents=True)
    for folder in ["parents", "active", "repaired", "traces", "routes"]:
        (output_root / folder).mkdir()
    shutil.copy2(config_path, output_root / "design_config.yaml")
    memory_path = output_root / "verified_memory.jsonl"
    if memory_seed is not None:
        shutil.copy2(memory_seed.resolve(strict=True), memory_path)
    memory_before = sha256_file(memory_path) if memory_path.exists() else None
    trace = AgentTrace(output_root / "workflow_trace.jsonl")
    trace.emit("Controller", "begin_workflow", "started", split=split,
               config_sha256=sha256_file(config_path), numerical_oracle_calls=0)
    t_low, t_high = map(float, task["tolerance_factor_window"])
    mu_low, mu_high = map(float, task["octahedral_factor_window"])
    records, exclusions, routes, seen = [], [], [], set()
    fault_injected = False
    for proposal in propose(cfg):
        if proposal["proposal_status"] != "charge_balanced":
            exclusions.append({**proposal, "reason": "formal_charge_imbalance"})
            continue
        try:
            structure, geom = build_structure(proposal, task)
        except (ValueError, KeyError) as exc:
            exclusions.append({**proposal, "reason": "radius_or_geometry_unavailable",
                               "exception": str(exc)})
            continue
        if not (t_low <= geom["tolerance_factor"] <= t_high
                and mu_low <= geom["b_oct_factor"] <= mu_high
                and mu_low <= geom["b_prime_oct_factor"] <= mu_high):
            exclusions.append({**proposal, **geom, "reason": "outside_geometry_window"})
            continue
        if proposal["formula"] in seen:
            exclusions.append({**proposal, "reason": "duplicate_formula_within_run"})
            continue
        seen.add(proposal["formula"])
        cid = _candidate_id(cfg["design_id"], proposal["formula"], structure)
        filename = cid + ".cif"
        parent = output_root / "parents" / filename
        structure.to(filename=str(parent), fmt="cif")
        digest = sha256_file(parent)
        active = output_root / "active" / filename
        shutil.copy2(parent, active)
        is_fault = inject_byte_fault and not fault_injected
        if is_fault:
            with active.open("ab") as stream:
                stream.write(b"\n# controlled integration-test byte fault\n")
            fault_injected = True
        artifact = CandidateArtifact(cid, proposal["formula"], filename, digest,
                                     proposal["formal_charge_sum"],
                                     origin="controlled_byte_fault" if is_fault else "generated_candidate",
                                     registered_source_path=filename).with_identity()
        experience = VerifiedMemory(memory_path).retrieve("artifact_sha256", target_split=split)
        validation = run_candidate(artifact, active_root=output_root / "active",
            recovery_root=output_root / "parents", output_root=output_root / "repaired",
            trace_path=output_root / "traces" / (cid + ".jsonl"), memory_path=memory_path,
            split=split, learning_allowed=(split == "development"), max_rounds=2, max_repairs=2)
        final = (output_root / "repaired" / f"{cid}_{proposal['formula']}_restored.cif"
                 if validation["repaired"] else active)
        precursors, reaction = PathMatRouteAgent.precursor_recipe(proposal)
        row = {**proposal, **geom, "candidate_id": cid,
               "parent_cif": parent.relative_to(output_root).as_posix(),
               "parent_sha256": digest,
               "final_cif": final.relative_to(output_root).as_posix(),
               "final_sha256": sha256_file(final),
               "controlled_fault_injected": is_fault,
               "retrieved_development_experiences": len(experience),
               "balanced_reaction": reaction, "precursor_formulas": precursors,
               "validation": validation}
        records.append(row)
        for schedule in PathMatRouteAgent.DEFAULT_SCHEDULES:
            routes.append({"candidate_id": cid, "schedule": dict(schedule),
                           "balanced_reaction": reaction,
                           "record": PathMatRouteAgent._record(proposal, schedule, precursors)})
        trace.emit("Controller", "candidate_completed", validation["final_validation"],
                   candidate_id=cid, formula=proposal["formula"], repaired=validation["repaired"])
    _json(output_root / "candidate_records.json", records)
    _json(output_root / "excluded_proposals.json", exclusions)
    _json(output_root / "routes" / "route_hypotheses.json", routes)
    route_scoring = {"status": "not_configured", "route_hypothesis_count": len(routes)}
    if route_model is not None:
        route_scoring = PathMatRouteAgent(route_model.resolve(strict=True),
            route_vendor.resolve(strict=True), trace).score_candidates(records, output_root / "routes")
    memory_after = sha256_file(memory_path) if memory_path.exists() else None
    if split == "confirmatory" and memory_before != memory_after:
        raise RuntimeError("confirmatory workflow changed its frozen memory")
    summary = {"workflow": "PATH-Mat candidate design, critique, verification and repair",
        "split": split, "generated_candidates": len(records), "excluded_proposals": len(exclusions),
        "final_integrity_passes": sum(r["validation"]["final_validation"] == "pass" for r in records),
        "restored_candidates": sum(r["validation"]["repaired"] for r in records),
        "controlled_byte_faults": sum(r["controlled_fault_injected"] for r in records),
        "route_hypotheses": len(routes), "route_scoring": route_scoring,
        "memory_sha256_before": memory_before, "memory_sha256_after": memory_after,
        "confirmatory_memory_unchanged": (memory_before == memory_after) if split == "confirmatory" else None,
        "numerical_oracle_calls": 0, "new_scientific_performance_evaluation": False,
        "python_version": platform.python_version(),
        "package_versions": {x: importlib.metadata.version(x) for x in ["pymatgen", "numpy", "PyYAML"]}}
    _json(output_root / "workflow_summary.json", summary)
    trace.emit("Controller", "end_workflow", "completed", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "confirmatory"], default="development")
    parser.add_argument("--inject-byte-fault", action="store_true",
                        help="controlled artifact-recovery demonstration only")
    parser.add_argument("--memory-seed", type=Path)
    parser.add_argument("--route-model", type=Path)
    parser.add_argument("--route-vendor", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_workflow(args.config, args.output, split=args.split,
        inject_byte_fault=args.inject_byte_fault, memory_seed=args.memory_seed,
        route_model=args.route_model, route_vendor=args.route_vendor), indent=2))


if __name__ == "__main__":
    main()
