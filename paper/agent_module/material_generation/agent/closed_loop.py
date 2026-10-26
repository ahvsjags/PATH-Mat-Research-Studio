"""Evidence-bound, budget-limited critique/repair loop for PATH-Mat candidates.

This first implementation deliberately makes no property or synthesis-success
claims.  LLM text is not used as ground truth: the skeptic emits hypotheses,
the verifier re-runs deterministic checks, and repair is allowed only for a
verified, recoverable artifact defect backed by a hash-registered parent CIF.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymatgen.core import Composition, Structure
import pymatgen


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_composition(formula: str) -> dict[str, float]:
    comp = Composition(formula).fractional_composition
    return {el.symbol: float(amount) for el, amount in sorted(comp.items(), key=lambda p: p[0].symbol)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CandidateArtifact:
    candidate_id: str
    formula: str
    cif_path: str
    expected_sha256: str
    formal_charge_sum: float | None = None
    artifact_id: str = ""
    parent_artifact_id: str | None = None
    origin: str = "frozen_baseline"
    registered_source_path: str | None = None

    def with_identity(self) -> "CandidateArtifact":
        if self.artifact_id:
            return self
        seed = f"{self.candidate_id}\0{self.expected_sha256}\0{self.parent_artifact_id or ''}"
        return replace(self, artifact_id=hashlib.sha256(seed.encode()).hexdigest()[:20])


@dataclass
class CheckResult:
    check_id: str
    status: str  # pass | fail | unknown
    observed: Any
    expected: Any
    evidence_sha256: str
    tool: str
    detail: str = ""


@dataclass
class ValidationReport:
    candidate_id: str
    artifact_id: str
    formula: str
    checks: list[CheckResult]
    overall: str
    structure_sha256: str | None
    limitations: list[str] = field(default_factory=list)

    def check(self, check_id: str) -> CheckResult | None:
        return next((c for c in self.checks if c.check_id == check_id), None)


@dataclass
class Critique:
    critique_id: str
    candidate_id: str
    artifact_id: str
    check_id: str
    claim: str
    proposed_action: str
    status: str = "hypothesis"
    evidence_sha256: str = ""


@dataclass
class VerifiedIssue:
    critique_id: str
    candidate_id: str
    artifact_id: str
    check_id: str
    status: str  # confirmed | rejected | unresolved
    evidence_sha256: str
    verifier: str
    verified_at: str
    details: dict[str, Any]


class Verifier:
    """Deterministic checks only; a passing report is not a stability proof."""

    def __init__(self, min_interatomic_distance_angstrom: float = 0.75):
        self.min_distance = float(min_interatomic_distance_angstrom)
        if self.min_distance <= 0:
            raise ValueError("min_interatomic_distance_angstrom must be positive")

    @staticmethod
    def _evidence_hash(payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _check(self, check_id: str, status: str, observed: Any, expected: Any, detail: str = "") -> CheckResult:
        payload = {"check_id": check_id, "status": status, "observed": observed, "expected": expected, "detail": detail}
        return CheckResult(check_id, status, observed, expected, self._evidence_hash(payload),
                           f"Verifier/pymatgen-{getattr(pymatgen, '__version__', 'unknown')}", detail)

    def validate(self, artifact: CandidateArtifact, candidate_root: Path) -> ValidationReport:
        artifact = artifact.with_identity()
        root = candidate_root.resolve(strict=True)
        path = (root / artifact.cif_path).resolve()
        try:
            path.relative_to(root)
            inside_root = True
        except ValueError:
            inside_root = False
        checks: list[CheckResult] = []
        if not inside_root:
            checks.append(self._check("path_containment", "fail", str(path), str(root), "artifact path escapes candidate root"))
            return self._report(artifact, checks, None)
        if not path.is_file():
            checks.append(self._check("artifact_exists", "fail", False, True, "candidate CIF is missing"))
            return self._report(artifact, checks, None)

        actual_hash = sha256_file(path)
        checks.append(self._check("artifact_sha256", "pass" if actual_hash == artifact.expected_sha256 else "fail",
                                  actual_hash, artifact.expected_sha256, "byte-level artifact integrity"))
        try:
            structure = Structure.from_file(path, primitive=False)
        except Exception as exc:  # parser exceptions are recorded as evidence
            checks.append(self._check("cif_parse", "fail", type(exc).__name__, "parseable CIF", str(exc)[:500]))
            return self._report(artifact, checks, actual_hash)

        actual_comp = canonical_composition(structure.composition.reduced_formula)
        expected_comp = canonical_composition(artifact.formula)
        comp_match = actual_comp.keys() == expected_comp.keys() and all(
            abs(actual_comp[k] - expected_comp[k]) <= 1e-6 for k in expected_comp
        )
        checks.append(self._check("formula_match", "pass" if comp_match else "fail", actual_comp, expected_comp,
                                  "reduced elemental composition comparison"))
        volume = float(structure.volume)
        volume_ok = np.isfinite(volume) and volume > 0
        checks.append(self._check("positive_finite_volume", "pass" if volume_ok else "fail", volume,
                                  "> 0 Å^3 and finite"))
        occupancies = [float(sum(site.species.values())) for site in structure]
        occupancy_ok = bool(occupancies) and all(np.isfinite(x) and abs(x - 1.0) <= 1e-6 for x in occupancies)
        checks.append(self._check("site_occupancy", "pass" if occupancy_ok else "fail",
                                  {"min": min(occupancies) if occupancies else None,
                                   "max": max(occupancies) if occupancies else None},
                                  "every site sums to 1.0"))
        distance_matrix = structure.distance_matrix
        nonzero = distance_matrix[distance_matrix > 1e-5]
        min_dist = float(nonzero.min()) if nonzero.size else None
        overlap_ok = min_dist is not None and np.isfinite(min_dist) and min_dist >= self.min_distance
        checks.append(self._check("periodic_minimum_distance", "pass" if overlap_ok else "fail", min_dist,
                                  f">= {self.min_distance:g} Å", "coarse catastrophic-overlap screen only"))
        if artifact.formal_charge_sum is not None:
            charge_ok = abs(float(artifact.formal_charge_sum)) <= 1e-8
            checks.append(self._check("formal_charge_balance", "pass" if charge_ok else "fail",
                                      float(artifact.formal_charge_sum), 0.0,
                                      "input oxidation-state bookkeeping, not measured oxidation state"))
        return self._report(artifact, checks, actual_hash)

    def validate_oracle_record(self, candidate_id: str, formula: str, result: dict[str, Any],
                               output_root: Path, input_root: Path, *,
                               expected_checkpoint_sha256: str) -> ValidationReport:
        """Check an MLIP relaxation record against its saved artifact and pinned model.

        A non-converged relaxation is a confirmed screen failure, not a malformed
        structure. It must be retained and may only be rerun under an explicit
        additional oracle budget; the automatic byte-restoration repair does not
        apply to scientific outputs.
        """
        root = output_root.resolve(strict=True)
        path_text = result.get("output_path")
        path = Path(path_text).resolve() if path_text else root / "__missing_output__"
        try:
            path.relative_to(root)
            inside_root = True
        except ValueError:
            inside_root = False
        identity_artifact = CandidateArtifact(candidate_id, formula, path.name,
                                              result.get("output_sha256", ""))
        artifact_id = identity_artifact.with_identity().artifact_id
        checks: list[CheckResult] = []
        in_root = input_root.resolve(strict=True)
        input_text = result.get("input_path")
        input_path = Path(input_text).resolve() if input_text else in_root / "__missing_input__"
        try:
            input_inside = input_path.is_relative_to(in_root)
        except AttributeError:  # Python versions before Path.is_relative_to
            try:
                input_path.relative_to(in_root)
                input_inside = True
            except ValueError:
                input_inside = False
        checks.append(self._check("oracle_input_containment", "pass" if input_inside else "fail",
                                  str(input_path), str(in_root), "oracle input must remain inside the registered input directory"))
        if input_inside:
            input_exists = input_path.is_file()
            checks.append(self._check("oracle_input_exists", "pass" if input_exists else "fail", input_exists, True,
                                      "exact input structure bytes must remain available"))
            if input_exists:
                actual_input_hash = sha256_file(input_path)
                checks.append(self._check("oracle_input_sha256",
                    "pass" if actual_input_hash == result.get("input_sha256") else "fail",
                    actual_input_hash, result.get("input_sha256"), "input structure digest must match the oracle record"))
        checks.append(self._check("oracle_output_containment", "pass" if inside_root else "fail",
                                  str(path), str(root), "oracle output must remain inside the registered run directory"))
        if not inside_root:
            return self._report(CandidateArtifact(candidate_id, formula, "", "", artifact_id=artifact_id),
                                checks, None)
        exists = path.is_file()
        checks.append(self._check("oracle_output_exists", "pass" if exists else "fail", exists, True,
                                  "relaxed structure artifact must be persisted"))
        if exists:
            actual_hash = sha256_file(path)
            checks.append(self._check("oracle_output_sha256", "pass" if actual_hash == result.get("output_sha256") else "fail",
                                      actual_hash, result.get("output_sha256"), "saved output must match the result record"))
            try:
                structure = Structure.from_file(path, primitive=False)
                actual_formula = structure.composition.reduced_formula
                formula_ok = actual_formula == formula and result.get("formula") == formula
                checks.append(self._check("oracle_formula_preserved", "pass" if formula_ok else "fail",
                                          {"parsed": actual_formula, "recorded": result.get("formula")}, formula,
                                          "composition is checked both in the output CIF and the result record"))
            except Exception as exc:
                checks.append(self._check("oracle_output_parse", "fail", type(exc).__name__, "parseable CIF", str(exc)[:500]))
        for name, value in (("energy_eV_atom", result.get("energy_eV_atom")),
                            ("max_force_eV_A", result.get("max_force_eV_A"))):
            valid = False
            try:
                valid = bool(np.isfinite(float(value)))
            except (TypeError, ValueError):
                pass
            checks.append(self._check(f"oracle_{name}_finite", "pass" if valid else "fail", value,
                                      "finite numeric value", "non-finite or missing MLIP output is unusable"))
        try:
            force = float(result.get("max_force_eV_A"))
            threshold = float(result.get("fmax_eV_A"))
            recorded_converged = bool(result.get("force_converged"))
            computed_converged = bool(np.isfinite(force) and force <= threshold + 1e-6)
            convergence_consistent = recorded_converged == computed_converged and computed_converged
        except (TypeError, ValueError):
            force, threshold, recorded_converged, computed_converged = None, result.get("fmax_eV_A"), None, False
            convergence_consistent = False
        checks.append(self._check("mlip_force_convergence", "pass" if convergence_consistent else "fail",
                                  {"max_force_eV_A": force, "fmax_eV_A": threshold,
                                   "recorded_force_converged": recorded_converged,
                                   "computed_force_converged": computed_converged},
                                  "recorded and recomputed convergence both true",
                                  "a valid but non-converged relaxation remains a failed screening result"))
        checkpoint_ok = result.get("checkpoint_sha256") == expected_checkpoint_sha256
        checks.append(self._check("oracle_checkpoint_sha256", "pass" if checkpoint_ok else "fail",
                                  result.get("checkpoint_sha256"), expected_checkpoint_sha256,
                                  "model checkpoint must match the frozen evaluation"))
        try:
            steps = int(result.get("steps"))
            max_steps = int(result.get("max_steps"))
            steps_ok = 1 <= steps <= max_steps
        except (TypeError, ValueError):
            steps, max_steps, steps_ok = result.get("steps"), result.get("max_steps"), False
        checks.append(self._check("oracle_optimizer_steps", "pass" if steps_ok else "fail",
                                  steps, f"integer in [1, {max_steps}]", "relaxation step count must be recorded"))
        artifact = CandidateArtifact(candidate_id, formula, path.name if exists else "", result.get("output_sha256", ""),
                                     artifact_id=artifact_id)
        return self._report(artifact, checks, result.get("output_sha256"))

    @staticmethod
    def _report(artifact: CandidateArtifact, checks: list[CheckResult], structure_hash: str | None) -> ValidationReport:
        states = {c.status for c in checks}
        overall = "fail" if "fail" in states else ("unknown" if "unknown" in states else "pass")
        return ValidationReport(artifact.candidate_id, artifact.artifact_id, artifact.formula, checks,
                                overall, structure_hash,
                                ["Does not establish thermodynamic/phonon stability, target properties, synthesizability, or experimental realization."])

    def verify_critique(self, critique: Critique, artifact: CandidateArtifact,
                        report: ValidationReport) -> VerifiedIssue:
        check = report.check(critique.check_id)
        if (critique.candidate_id != artifact.candidate_id or critique.artifact_id != artifact.with_identity().artifact_id
                or report.candidate_id != artifact.candidate_id
                or report.artifact_id != artifact.with_identity().artifact_id
                or check is None or not critique.evidence_sha256 or critique.evidence_sha256 != check.evidence_sha256):
            status, detail = "unresolved", {"reason": "candidate/artifact/check binding mismatch"}
            evidence_hash = self._evidence_hash(detail)
        else:
            status = "confirmed" if check.status == "fail" else ("rejected" if check.status == "pass" else "unresolved")
            detail = {"observed": check.observed, "expected": check.expected, "check_status": check.status,
                      "tool": check.tool, "detail": check.detail}
            evidence_hash = check.evidence_sha256
        return VerifiedIssue(critique.critique_id, artifact.candidate_id, artifact.with_identity().artifact_id,
                             critique.check_id, status, evidence_hash, "Verifier/recheck-v1", utc_now(), detail)


class SkepticAgent:
    """Converts failures/unknowns into testable hypotheses; never confirms them."""

    ACTIONS = {
        "artifact_sha256": ("Recheck the registered parent bytes and restore only from a matching source hash.", "restore_verified_artifact"),
        "artifact_exists": ("Resolve the registered source artifact and confirm its digest before recovery.", "restore_verified_artifact"),
        "path_containment": ("Reject the path and re-resolve through the registered candidate root.", "reject_path"),
        "cif_parse": ("Check whether the registered parent CIF can be restored without changing candidate chemistry.", "restore_verified_artifact"),
        "formula_match": ("Compare parsed composition with the design specification; do not rewrite the label to fit the CIF.", "restore_verified_artifact"),
        "positive_finite_volume": ("Reconstruct only from a registered parent design recipe and rerun the same validator.", "restore_verified_artifact"),
        "site_occupancy": ("Inspect occupancies and reconstruct from the registered parent if the serialized file is corrupt.", "restore_verified_artifact"),
        "periodic_minimum_distance": ("Reinitialize from the registered parent structure, then evaluate with the same checks.", "restore_verified_artifact"),
        "formal_charge_balance": ("Recompute the stated formal-charge bookkeeping from registered oxidation-state inputs.", "reject_candidate"),
        "mlip_force_convergence": ("Recheck the saved force threshold and convergence flag; if confirmed, preserve the failed relaxation and request an explicitly budgeted new start.", "defer_additional_oracle_budget"),
        "oracle_checkpoint_sha256": ("Resolve the checkpoint from the locked model registry and compare its byte digest.", "reject_model_mismatch"),
        "oracle_output_sha256": ("Recompute the saved relaxation artifact digest and compare it with the result row.", "restore_verified_artifact"),
        "oracle_input_sha256": ("Recompute the registered input structure digest and compare it with the oracle record.", "reject_input_provenance_mismatch"),
        "oracle_formula_preserved": ("Parse the relaxed CIF and compare its composition with the input formula.", "reject_candidate"),
        "oracle_energy_eV_atom_finite": ("Recheck the recorded energy unit and finite numerical result.", "defer_additional_oracle_budget"),
        "oracle_max_force_eV_A_finite": ("Recheck the recorded force unit and finite numerical result.", "defer_additional_oracle_budget"),
    }

    def challenge(self, artifact: CandidateArtifact, report: ValidationReport) -> list[Critique]:
        critiques: list[Critique] = []
        for check in report.checks:
            if check.status == "pass":
                continue
            claim, action = self.ACTIONS.get(check.check_id,
                                              ("Inspect this unresolved validation result with a deterministic check.", "defer"))
            critiques.append(Critique(uuid.uuid4().hex, artifact.candidate_id, artifact.with_identity().artifact_id,
                                      check.check_id, claim, action, evidence_sha256=check.evidence_sha256))
        return critiques


class RepairAgent:
    """Bounded repair; scientific content is restored, never silently edited."""

    def repair(self, artifact: CandidateArtifact, issue: VerifiedIssue, *,
               recovery_root: Path, output_root: Path) -> CandidateArtifact:
        if issue.status != "confirmed":
            raise ValueError("repair requires a verifier-confirmed issue")
        if issue.candidate_id != artifact.candidate_id or issue.artifact_id != artifact.with_identity().artifact_id:
            raise ValueError("repair issue is not bound to this candidate artifact")
        if issue.check_id not in {"artifact_sha256", "artifact_exists", "cif_parse", "formula_match",
                                  "positive_finite_volume", "site_occupancy", "periodic_minimum_distance"}:
            raise ValueError(f"no authorized automatic repair for {issue.check_id}")
        source_root = recovery_root.resolve(strict=True)
        source_rel = artifact.registered_source_path or artifact.cif_path
        source = (source_root / source_rel).resolve(strict=True)
        source.relative_to(source_root)
        source_hash = sha256_file(source)
        if source_hash != artifact.expected_sha256:
            raise ValueError("registered recovery source does not match the frozen expected hash")
        output_root.mkdir(parents=True, exist_ok=True)
        target_name = f"{artifact.candidate_id}_{artifact.formula}_restored.cif"
        target = output_root / target_name
        if target.exists():
            raise FileExistsError(f"repair output already exists; refusing to overwrite: {target}")
        shutil.copyfile(source, target)
        restored_hash = sha256_file(target)
        if restored_hash != artifact.expected_sha256:
            target.unlink(missing_ok=True)
            raise IOError("restored artifact digest differs from frozen source")
        next_id = hashlib.sha256(f"{artifact.artifact_id}\0restore_verified_artifact\0{restored_hash}".encode()).hexdigest()[:20]
        return CandidateArtifact(artifact.candidate_id, artifact.formula, target.name, restored_hash,
                                 artifact.formal_charge_sum, next_id, artifact.artifact_id,
                                 "restored_from_frozen_parent", target.name)


class VerifiedMemory:
    """Append-only memory; confirmation requires bound verifier evidence."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def commit(self, issue: VerifiedIssue, repair: str, outcome: str, *, split: str,
               source_hash: str, learning_allowed: bool) -> dict[str, Any]:
        if issue.status != "confirmed" or not issue.evidence_sha256:
            raise ValueError("only evidence-bound confirmed issues can enter verified memory")
        details = issue.details
        if details.get("check_status") != "fail":
            raise ValueError("memory requires a failed deterministic verifier check")
        evidence_payload = {"check_id": issue.check_id, "status": "fail",
                            "observed": details.get("observed"), "expected": details.get("expected"),
                            "detail": details.get("detail", "")}
        if Verifier._evidence_hash(evidence_payload) != issue.evidence_sha256:
            raise ValueError("verified-memory evidence digest does not match the check payload")
        if split in {"test", "heldout", "confirmatory"}:
            raise ValueError("held-out/confirmatory evidence is read-only and cannot be written to memory")
        event = {"event_id": uuid.uuid4().hex, "timestamp_utc": utc_now(), "candidate_id": issue.candidate_id,
                 "issue_code": issue.check_id, "verified_evidence_sha256": issue.evidence_sha256,
                 "verifier": issue.verifier, "repair_operator": repair, "repair_outcome": outcome,
                 "split": split, "source_hash": source_hash, "learning_allowed": bool(learning_allowed)}
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
        return event

    def retrieve(self, issue_code: str, *, target_split: str) -> list[dict[str, Any]]:
        """Retrieve only verified development experience; holdout cannot train itself."""
        if target_split in {"test", "heldout", "confirmatory"}:
            allowed_source_splits = {"development", "exploratory_smoke"}
        else:
            allowed_source_splits = {"development", "exploratory_smoke", "development_smoke"}
        if not self.path.exists():
            return []
        found = []
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (row.get("issue_code") == issue_code and row.get("learning_allowed") is True
                        and row.get("split") in allowed_source_splits and row.get("verified_evidence_sha256")):
                    found.append(row)
        return found


class BudgetController:
    """A hard-cap controller for a single candidate; no retry can exceed cap."""

    def __init__(self, max_rounds: int = 2, max_repairs: int = 2):
        if not 0 <= max_rounds <= 2 or not 0 <= max_repairs <= 2:
            raise ValueError("VCR pilot hard-caps rounds and repairs at two")
        self.max_rounds = max_rounds
        self.max_repairs = max_repairs
        self.rounds_used = 0
        self.repairs_used = 0

    def begin_round(self) -> bool:
        if self.rounds_used >= self.max_rounds:
            return False
        self.rounds_used += 1
        return True

    def consume_repair(self) -> bool:
        if self.repairs_used >= self.max_repairs:
            return False
        self.repairs_used += 1
        return True


def artifact_from_ledger_row(row: dict[str, Any]) -> CandidateArtifact:
    charge = row.get("formal_charge_sum", "")
    return CandidateArtifact(candidate_id=str(row["candidate_id"]), formula=str(row["formula"]),
                             cif_path=str(row["cif_path"]), expected_sha256=str(row["cif_sha256"]),
                             formal_charge_sum=float(charge) if charge not in (None, "") else None,
                             registered_source_path=str(row["cif_path"])).with_identity()


def checks_as_dicts(report: ValidationReport) -> list[dict[str, Any]]:
    return [asdict(check) for check in report.checks]
