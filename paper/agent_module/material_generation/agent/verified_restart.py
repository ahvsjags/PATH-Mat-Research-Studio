"""Evidence-bound critique, verification and continuation repair for failures.

All scientific observations come from injected deterministic tools (force
recomputation, V0 validation and the pinned relaxer). This module binds those
observations to immutable input/output artifacts and never treats prose as
evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pymatgen.core import Structure


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(path_text: str, base: Path, required_root: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    path = path.resolve(strict=True)
    path.relative_to(required_root.resolve(strict=True))
    return path


@dataclass(frozen=True)
class RestartHypothesis:
    candidate_id: str
    formula: str
    input_sha256: str
    terminal_sha256: str
    checkpoint_sha256: str
    claim: str


@dataclass(frozen=True)
class VerifiedRestartIssue:
    candidate_id: str
    status: str  # confirmed | rejected | unresolved
    repairable: bool
    evidence_sha256: str
    recomputed_force_eV_A: float | None
    v0_status: str
    reason: str


class RestartSkeptic:
    """Flag failures and a fixed guard band where serialization can flip status."""

    def challenge(self, row: dict[str, Any], *, force_guard_band_eV_A: float = 0.001) -> RestartHypothesis | None:
        result = row.get("result", {})
        failed_force = result.get("force_converged") is False
        failed_v0 = row.get("v0_overall") == "fail"
        threshold = result.get("fmax_eV_A")
        force = result.get("max_force_eV_A")
        near_threshold = False
        try:
            near_threshold = math.isfinite(float(force)) and math.isfinite(float(threshold)) and \
                float(force) >= float(threshold) - float(force_guard_band_eV_A)
        except (TypeError, ValueError):
            near_threshold = True
        if force_guard_band_eV_A < 0:
            raise ValueError("force_guard_band_eV_A must be non-negative")
        if not (failed_force or failed_v0 or near_threshold):
            return None
        return RestartHypothesis(
            candidate_id=str(row["candidate_id"]),
            formula=str(row["formula"]),
            input_sha256=str(row.get("input_sha256", "")),
            terminal_sha256=str(result.get("output_sha256", "")),
            checkpoint_sha256=str(result.get("checkpoint_sha256", "")),
            claim="The saved result failed or lies within a frozen force guard band; independently replay its hash-bound artifact before any repair.",
        )


class RestartVerifier:
    """Recompute force and V0 on the exact serialized terminal structure."""

    def verify(
        self,
        hypothesis: RestartHypothesis,
        row: dict[str, Any],
        *,
        run_root: Path,
        expected_checkpoint_sha256: str,
        expected_fmax_eV_A: float,
        recompute_force: Callable[[Structure], float],
        replay_v0: Callable[[Path, str, str], str],
    ) -> VerifiedRestartIssue:
        evidence: dict[str, Any] = {
            "candidate_id": hypothesis.candidate_id,
            "formula": hypothesis.formula,
            "expected_checkpoint_sha256": expected_checkpoint_sha256,
            "expected_fmax_eV_A": float(expected_fmax_eV_A),
        }
        force: float | None = None
        v0_status = "unknown"
        try:
            if row.get("candidate_id") != hypothesis.candidate_id or row.get("formula") != hypothesis.formula:
                raise ValueError("candidate identity mismatch")
            root = run_root.resolve(strict=True)
            input_root = (root / "inputs").resolve(strict=True)
            output_root = (root / "relaxed").resolve(strict=True)
            input_path = _contained(str(row["input_path"]), root, input_root)
            result = row["result"]
            output_path = _contained(str(result["output_path"]), root, output_root)
            input_hash, terminal_hash = sha256_file(input_path), sha256_file(output_path)
            structure = Structure.from_file(output_path, primitive=False)
            parsed_formula = structure.composition.reduced_formula
            force = float(recompute_force(structure))
            v0_status = str(replay_v0(output_path, hypothesis.formula, terminal_hash))
            threshold = float(expected_fmax_eV_A)
            identity_ok = (
                input_hash == hypothesis.input_sha256 == row.get("input_sha256")
                and terminal_hash == hypothesis.terminal_sha256 == result.get("output_sha256")
                and result.get("checkpoint_sha256") == hypothesis.checkpoint_sha256 == expected_checkpoint_sha256
                and parsed_formula == hypothesis.formula == result.get("formula")
                and math.isfinite(force)
                and math.isfinite(threshold)
                and abs(float(result.get("fmax_eV_A")) - threshold) <= 1e-12
            )
            computed_failed = force > threshold + 1e-6
            recorded_force = float(result["max_force_eV_A"])
            recorded_failed = recorded_force > threshold + 1e-6
            evidence.update({
                "input_sha256": input_hash,
                "terminal_sha256": terminal_hash,
                "parsed_formula": parsed_formula,
                "recorded_force_eV_A": recorded_force,
                "recomputed_serialized_force_eV_A": force,
                "recorded_failure": recorded_failed,
                "serialized_failure": computed_failed,
                "recorded_v0_status": row.get("v0_overall"),
                "replayed_v0_status": v0_status,
                "identity_ok": identity_ok,
            })
            if not identity_ok or v0_status not in {"pass", "fail", "unknown"}:
                status, reason = "unresolved", "artifact identity or verifier output is invalid"
            elif row.get("v0_overall") != v0_status:
                status, reason = "unresolved", "V0 replay does not match the saved record"
            elif v0_status == "unknown":
                status, reason = "unresolved", "V0 status is unknown"
            elif v0_status == "fail":
                status, reason = "confirmed", "serialized artifact independently fails V0; force continuation is not the appropriate repair"
                repairable = False
            elif computed_failed:
                status, reason = "confirmed", "serialized artifact passes V0 but independently exceeds the force threshold"
                repairable = True
            else:
                status, reason = "rejected", "serialized artifact passes the frozen checks"
                repairable = False
            if status in {"unresolved"}:
                repairable = False
        except Exception as exc:
            evidence.update({"exception": type(exc).__name__, "detail": str(exc)[:500]})
            status, reason, repairable = "unresolved", f"verification could not be completed: {type(exc).__name__}", False
        digest = hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        return VerifiedRestartIssue(hypothesis.candidate_id, status, repairable, digest, force, v0_status, reason)


@dataclass(frozen=True)
class RestartRepairRecord:
    candidate_id: str
    status: str  # performed | not_performed | unresolved
    parent_terminal_sha256: str
    repaired_output_sha256: str | None
    formula_before: str
    formula_after: str | None
    tool_record: dict[str, Any]
    reason: str


class ContinuationRepairer:
    """Run one bounded continuation from a confirmed, hash-bound terminal CIF."""

    def repair_once(
        self,
        issue: VerifiedRestartIssue,
        hypothesis: RestartHypothesis,
        row: dict[str, Any],
        *,
        run_root: Path,
        output_path: Path,
        continue_relaxation: Callable[[Structure, Path], dict[str, Any]],
    ) -> RestartRepairRecord:
        if issue.status != "confirmed" or not issue.repairable or issue.candidate_id != hypothesis.candidate_id:
            return RestartRepairRecord(hypothesis.candidate_id, "not_performed",
                hypothesis.terminal_sha256, None, hypothesis.formula, None, {},
                "repair requires a matching independently confirmed issue")
        try:
            root = run_root.resolve(strict=True)
            output_root = (root / "relaxed").resolve(strict=True)
            parent_path = _contained(str(row["result"]["output_path"]), root, output_root)
            parent_hash = sha256_file(parent_path)
            if parent_hash != hypothesis.terminal_sha256 or parent_hash != row["result"].get("output_sha256"):
                raise ValueError("terminal CIF changed after verification")
            target = output_path.resolve()
            target.relative_to(output_root)
            if target.exists():
                raise FileExistsError("repair output already exists; refusing overwrite")
            parent = Structure.from_file(parent_path, primitive=False)
            if parent.composition.reduced_formula != hypothesis.formula:
                raise ValueError("parent CIF formula does not match the verified candidate")
            tool_record = continue_relaxation(parent, target)
            target = target.resolve(strict=True)
            target.relative_to(output_root)
            repaired = Structure.from_file(target, primitive=False)
            formula_after = repaired.composition.reduced_formula
            if formula_after != hypothesis.formula:
                raise ValueError("continuation changed the composition")
            digest = sha256_file(target)
            recorded = str(tool_record.get("output_sha256", ""))
            if recorded != digest:
                raise ValueError("relaxer output hash does not match the saved CIF")
            return RestartRepairRecord(hypothesis.candidate_id, "performed", parent_hash,
                digest, hypothesis.formula, formula_after, tool_record,
                "continued from the exact verified terminal CIF; the original artifact was preserved")
        except Exception as exc:
            return RestartRepairRecord(hypothesis.candidate_id, "unresolved",
                hypothesis.terminal_sha256, None, hypothesis.formula, None, {},
                f"repair could not be verified: {type(exc).__name__}: {str(exc)[:300]}")
