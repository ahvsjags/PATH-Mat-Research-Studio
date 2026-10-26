"""Bounded orchestration of proposal evidence, critique, verification and repair."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .closed_loop import (BudgetController, CandidateArtifact, RepairAgent, SkepticAgent,
                          ValidationReport, VerifiedMemory, Verifier, checks_as_dicts, utc_now)


def run_candidate(artifact: CandidateArtifact, *, active_root: Path, recovery_root: Path,
                  output_root: Path, trace_path: Path, memory_path: Path,
                  split: str = "exploratory", learning_allowed: bool = True,
                  max_rounds: int = 2, max_repairs: int = 2) -> dict[str, Any]:
    """Run one artifact through a deterministic V0 loop.

    This V0 loop restores only a byte-verified original artifact. It does not
    change composition/coordinates or claim improved material properties.
    """
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    verifier, skeptic, repairer = Verifier(), SkepticAgent(), RepairAgent()
    budget = BudgetController(max_rounds=max_rounds, max_repairs=max_repairs)
    memory = VerifiedMemory(memory_path)
    trace: list[dict[str, Any]] = []

    def emit(event: str, **fields: Any) -> None:
        trace.append({"timestamp_utc": utc_now(), "event": event, **fields})
        with trace_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(trace[-1], sort_keys=True, default=str) + "\n")

    artifact = artifact.with_identity()
    emit("candidate_received", candidate_id=artifact.candidate_id, artifact_id=artifact.artifact_id,
         sha256=artifact.expected_sha256, split=split)
    initial_report = None
    issues = []
    pending_memory_issue = None
    current_artifact = artifact
    current_root = active_root
    while budget.begin_round():
        report = verifier.validate(current_artifact, current_root)
        emit("validation", round=budget.rounds_used, overall=report.overall,
             checks=checks_as_dicts(report), limitations=report.limitations)
        if initial_report is None:
            initial_report = report
        if report.overall == "pass":
            emit("candidate_passes_v0_checks", artifact_id=current_artifact.artifact_id)
            break
        critiques = skeptic.challenge(current_artifact, report)
        emit("critiques_proposed", round=budget.rounds_used,
             critiques=[asdict(c) for c in critiques])
        confirmed = []
        for critique in critiques:
            issue = verifier.verify_critique(critique, current_artifact, report)
            issues.append(issue)
            emit("critique_verified", issue=asdict(issue))
            if issue.status == "confirmed":
                confirmed.append(issue)
        # Restore a corrupt serialization once; all other issues remain explicit
        # failures requiring a new design or expert action.
        recoverable = next((i for i in confirmed if i.check_id in {
            "artifact_sha256", "artifact_exists", "cif_parse", "formula_match",
            "positive_finite_volume", "site_occupancy", "periodic_minimum_distance"}), None)
        if recoverable is None or not budget.consume_repair():
            emit("stop_no_authorized_repair_or_budget", round=budget.rounds_used)
            break
        repaired = repairer.repair(current_artifact, recoverable,
                                   recovery_root=recovery_root, output_root=output_root)
        emit("artifact_repaired_from_verified_parent", parent_artifact_id=current_artifact.artifact_id,
             artifact_id=repaired.artifact_id, output_sha256=repaired.expected_sha256,
             repair_operator="restore_verified_artifact")
        pending_memory_issue = recoverable
        current_artifact = repaired
        current_root = output_root

    final_report = verifier.validate(current_artifact, current_root)
    if pending_memory_issue is not None and final_report.overall == "pass":
        if split in {"test", "heldout", "confirmatory"} or not learning_allowed:
            emit("verified_memory_update_suppressed", split=split,
                 issue=asdict(pending_memory_issue), reason="read-only or learning-disabled evaluation")
        else:
            memory_event = memory.commit(pending_memory_issue, "restore_verified_artifact", "restored_and_revalidated_pass",
                                         split=split, source_hash=artifact.expected_sha256,
                                         learning_allowed=learning_allowed)
            emit("verified_memory_commit", memory_event=memory_event)
    emit("final_validation", overall=final_report.overall,
         checks=checks_as_dicts(final_report), rounds_used=budget.rounds_used,
         repairs_used=budget.repairs_used)
    result = {
        "candidate_id": artifact.candidate_id,
        "original_artifact_id": artifact.artifact_id,
        "final_artifact_id": current_artifact.artifact_id,
        "initial_validation": initial_report.overall if initial_report else "not_run",
        "final_validation": final_report.overall,
        "repaired": current_artifact.artifact_id != artifact.artifact_id,
        "rounds_used": budget.rounds_used,
        "repairs_used": budget.repairs_used,
        "verified_issues": [asdict(i) for i in issues],
        "final_checks": checks_as_dicts(final_report),
        "scientific_property_status": "not_evaluated_by_V0_integrity_loop",
        "synthesis_status": "not_evaluated",
        "trace_path": str(trace_path),
    }
    return result
