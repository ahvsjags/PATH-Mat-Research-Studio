from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest

from agent.closed_loop import (BudgetController, CandidateArtifact, Critique, RepairAgent,
                               SkepticAgent, VerifiedIssue, VerifiedMemory, Verifier)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "p1_smoke"


def load_first_artifact() -> CandidateArtifact:
    row = json.loads((FIXTURES / "fixture_manifest.json").read_text(encoding="utf-8"))["candidates"][0]
    return CandidateArtifact(row["candidate_id"], row["formula"], row["cif_path"], row["cif_sha256"],
                             row["formal_charge_sum"], registered_source_path=row["registered_source_path"]).with_identity()


def make_confirmed_integrity_issue(tmp_path: Path) -> VerifiedIssue:
    parent = load_first_artifact()
    active_path = tmp_path / "memory_fault.cif"
    active_path.write_bytes((FIXTURES / parent.cif_path).read_bytes() + b"\n# controlled fault\n")
    active = CandidateArtifact(parent.candidate_id, parent.formula, active_path.name, parent.expected_sha256,
        parent.formal_charge_sum, parent_artifact_id=parent.artifact_id, origin="fault_injection",
        registered_source_path=parent.registered_source_path).with_identity()
    report = Verifier().validate(active, tmp_path)
    critique = next(c for c in SkepticAgent().challenge(active, report) if c.check_id == "artifact_sha256")
    return Verifier().verify_critique(critique, active, report)


def test_six_frozen_candidate_structures_pass_v0():
    verifier = Verifier()
    rows = json.loads((FIXTURES / "fixture_manifest.json").read_text(encoding="utf-8"))["candidates"]
    assert len(rows) == 6
    for row in rows:
        item = CandidateArtifact(row["candidate_id"], row["formula"], row["cif_path"], row["cif_sha256"],
                                 row["formal_charge_sum"]).with_identity()
        report = verifier.validate(item, FIXTURES)
        assert report.overall == "pass", (row["candidate_id"], report.checks)
        assert report.check("formula_match").status == "pass"
        assert report.check("positive_finite_volume").status == "pass"


def test_critic_hypothesis_must_be_rechecked_and_can_be_rejected(tmp_path):
    verifier = Verifier()
    item = load_first_artifact()
    report = verifier.validate(item, FIXTURES)
    assert report.overall == "pass"
    skeptic = SkepticAgent()
    # A fabricated complaint has no authority; verifier reruns the indicated check.
    fake = Critique("fake", item.candidate_id, item.artifact_id, "positive_finite_volume",
                    "The volume is invalid", "reject_candidate",
                    evidence_sha256=report.check("positive_finite_volume").evidence_sha256)
    issue = verifier.verify_critique(fake, item, report)
    assert issue.status == "rejected"


def test_critique_with_unbound_evidence_is_unresolved():
    verifier = Verifier()
    item = load_first_artifact()
    report = verifier.validate(item, FIXTURES)
    critique = Critique("fake", item.candidate_id, item.artifact_id, "positive_finite_volume",
                        "The volume is invalid", "reject_candidate", evidence_sha256="0" * 64)
    assert verifier.verify_critique(critique, item, report).status == "unresolved"


def test_repair_requires_confirmed_and_hash_matching_parent(tmp_path):
    verifier, skeptic, repairer = Verifier(), SkepticAgent(), RepairAgent()
    parent = load_first_artifact()
    source = FIXTURES / parent.cif_path
    fault = tmp_path / "corrupt.cif"
    fault.write_bytes(source.read_bytes() + b"\n# injected byte corruption\n")
    active = CandidateArtifact(parent.candidate_id, parent.formula, fault.name, parent.expected_sha256,
                               parent.formal_charge_sum, parent_artifact_id=parent.artifact_id,
                               origin="fault_injection", registered_source_path=parent.registered_source_path).with_identity()
    report = verifier.validate(active, tmp_path)
    assert report.check("artifact_sha256").status == "fail"
    critique = next(c for c in skeptic.challenge(active, report) if c.check_id == "artifact_sha256")
    issue = verifier.verify_critique(critique, active, report)
    assert issue.status == "confirmed"
    with pytest.raises(ValueError):
        repairer.repair(active, VerifiedIssue(**{**issue.__dict__, "status": "rejected"}),
                        recovery_root=FIXTURES, output_root=tmp_path / "repaired")
    restored = repairer.repair(active, issue, recovery_root=FIXTURES, output_root=tmp_path / "repaired")
    assert restored.expected_sha256 == parent.expected_sha256
    assert Verifier().validate(restored, tmp_path / "repaired").overall == "pass"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        repairer.repair(active, issue, recovery_root=FIXTURES, output_root=tmp_path / "repaired")


def test_wrong_recovery_digest_fails_closed(tmp_path):
    parent = load_first_artifact()
    fake_root = tmp_path / "recovery"
    fake_root.mkdir()
    (fake_root / parent.registered_source_path).parent.mkdir(parents=True, exist_ok=True)
    (fake_root / parent.registered_source_path).write_text("not the frozen CIF", encoding="utf-8")
    issue = VerifiedIssue("x", parent.candidate_id, parent.artifact_id, "artifact_sha256", "confirmed",
                          "a" * 64, "test-verifier", "2026-10-18T00:00:00+00:00", {})
    with pytest.raises(ValueError, match="does not match"):
        RepairAgent().repair(parent, issue, recovery_root=fake_root, output_root=tmp_path / "out")


def test_formula_charge_and_path_faults_are_verified_separately(tmp_path):
    verifier, skeptic = Verifier(), SkepticAgent()
    parent = load_first_artifact()

    wrong_formula = CandidateArtifact(parent.candidate_id, "Ca2ScNbO6", parent.cif_path,
                                      parent.expected_sha256, 0.0,
                                      registered_source_path=parent.registered_source_path).with_identity()
    report = verifier.validate(wrong_formula, FIXTURES)
    assert report.check("formula_match").status == "fail"
    critique = next(c for c in skeptic.challenge(wrong_formula, report) if c.check_id == "formula_match")
    assert verifier.verify_critique(critique, wrong_formula, report).status == "confirmed"

    charge_error = CandidateArtifact(parent.candidate_id, parent.formula, parent.cif_path,
                                     parent.expected_sha256, 1.0,
                                     registered_source_path=parent.registered_source_path).with_identity()
    report = verifier.validate(charge_error, FIXTURES)
    assert report.check("formal_charge_balance").status == "fail"
    critique = next(c for c in skeptic.challenge(charge_error, report) if c.check_id == "formal_charge_balance")
    issue = verifier.verify_critique(critique, charge_error, report)
    assert issue.status == "confirmed"
    with pytest.raises(ValueError, match="no authorized automatic repair"):
        RepairAgent().repair(charge_error, issue, recovery_root=FIXTURES, output_root=tmp_path / "charge_repair")

    escaped = CandidateArtifact(parent.candidate_id, parent.formula, "../outside.cif",
                                parent.expected_sha256, parent.formal_charge_sum,
                                registered_source_path=parent.registered_source_path).with_identity()
    report = verifier.validate(escaped, FIXTURES)
    assert report.check("path_containment").status == "fail"
    critique = next(c for c in skeptic.challenge(escaped, report) if c.check_id == "path_containment")
    assert verifier.verify_critique(critique, escaped, report).status == "confirmed"


def test_confirmatory_memory_is_read_only(tmp_path):
    issue = make_confirmed_integrity_issue(tmp_path)
    path = tmp_path / "memory.jsonl"
    with pytest.raises(ValueError, match="read-only"):
        VerifiedMemory(path).commit(issue, "restore", "pass", split="confirmatory",
                                    source_hash="b" * 64, learning_allowed=True)
    with pytest.raises(ValueError, match="read-only"):
        VerifiedMemory(path).commit(issue, "restore", "pass", split="confirmatory",
                                    source_hash="b" * 64, learning_allowed=False)
    assert not path.exists()


def test_unverified_issue_cannot_enter_memory(tmp_path):
    issue = VerifiedIssue("x", "candidate", "artifact", "artifact_sha256", "unresolved",
                          "a" * 64, "test-verifier", "2026-10-18T00:00:00+00:00", {})
    with pytest.raises(ValueError, match="only evidence-bound"):
        VerifiedMemory(tmp_path / "memory.jsonl").commit(issue, "restore", "pass", split="development",
                                                           source_hash="b" * 64, learning_allowed=True)


def test_verified_development_memory_is_retrievable_but_holdout_is_not_written(tmp_path):
    issue = make_confirmed_integrity_issue(tmp_path)
    memory = VerifiedMemory(tmp_path / "memory.jsonl")
    memory.commit(issue, "restore", "pass", split="development", source_hash="b" * 64, learning_allowed=True)
    assert len(memory.retrieve("artifact_sha256", target_split="confirmatory")) == 1
    assert memory.retrieve("different_issue", target_split="development") == []
    with pytest.raises(ValueError, match="read-only"):
        memory.commit(issue, "restore", "pass", split="heldout", source_hash="c" * 64, learning_allowed=False)


def test_forged_confirmed_issue_cannot_enter_memory(tmp_path):
    issue = VerifiedIssue("x", "candidate", "artifact", "artifact_sha256", "confirmed",
                          "a" * 64, "Verifier/recheck-v1", "2026-10-18T00:00:00+00:00",
                          {"check_status": "fail", "observed": "bad", "expected": "good", "detail": "x"})
    with pytest.raises(ValueError, match="digest does not match"):
        VerifiedMemory(tmp_path / "memory.jsonl").commit(issue, "restore", "pass", split="development",
                                                            source_hash="b" * 64, learning_allowed=True)


def test_budget_hard_cap():
    budget = BudgetController(max_rounds=2, max_repairs=1)
    assert [budget.begin_round(), budget.begin_round(), budget.begin_round()] == [True, True, False]
    assert [budget.consume_repair(), budget.consume_repair()] == [True, False]
    with pytest.raises(ValueError):
        BudgetController(max_rounds=3)


def _valid_oracle_record(tmp_path, *, formula: str | None = None, energy: float = -5.0,
                         force: float = 0.02, converged: bool = True, checkpoint: str = "f" * 64):
    artifact = load_first_artifact()
    output_root = tmp_path / "oracle"
    output_root.mkdir(exist_ok=True)
    input_root = tmp_path / "inputs"
    input_root.mkdir(exist_ok=True)
    input_path = input_root / "input.cif"
    output = output_root / "relaxed.cif"
    import shutil
    shutil.copyfile(FIXTURES / artifact.cif_path, output)
    shutil.copyfile(FIXTURES / artifact.cif_path, input_path)
    record = {"input_path": str(input_path), "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "output_path": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "formula": formula or artifact.formula, "energy_eV_atom": energy, "max_force_eV_A": force,
              "fmax_eV_A": 0.05, "force_converged": converged, "steps": 10, "max_steps": 100,
              "checkpoint_sha256": checkpoint}
    return artifact, output_root, input_root, record


def test_oracle_result_verifier_accepts_valid_control_and_detects_faults(tmp_path):
    import math
    from agent.closed_loop import CandidateArtifact
    verifier = Verifier()
    skeptic = SkepticAgent()
    input_artifact, root, input_root, valid = _valid_oracle_record(tmp_path)
    artifact = CandidateArtifact(input_artifact.candidate_id, input_artifact.formula, "relaxed.cif",
                                 valid["output_sha256"]).with_identity()
    good = verifier.validate_oracle_record(artifact.candidate_id, artifact.formula, valid, root, input_root,
                                           expected_checkpoint_sha256="f" * 64)
    assert good.overall == "pass"
    assert skeptic.challenge(artifact, good) == []

    bad_energy = dict(valid, energy_eV_atom=math.nan)
    report = verifier.validate_oracle_record(artifact.candidate_id, artifact.formula, bad_energy, root, input_root,
                                             expected_checkpoint_sha256="f" * 64)
    assert report.check("oracle_energy_eV_atom_finite").status == "fail"
    assert verifier.verify_critique(skeptic.challenge(artifact, report)[0], artifact, report).status == "confirmed"

    bad_force = dict(valid, force_converged=False)
    report = verifier.validate_oracle_record(artifact.candidate_id, artifact.formula, bad_force, root, input_root,
                                             expected_checkpoint_sha256="f" * 64)
    critique = next(c for c in skeptic.challenge(artifact, report) if c.check_id == "mlip_force_convergence")
    issue = verifier.verify_critique(critique, artifact, report)
    assert issue.status == "confirmed"
    with pytest.raises(ValueError, match="no authorized automatic repair"):
        RepairAgent().repair(artifact, issue, recovery_root=FIXTURES, output_root=tmp_path / "bad_repair")

    bad_formula = dict(valid, formula="Ca2ScNbO6")
    report = verifier.validate_oracle_record(artifact.candidate_id, artifact.formula, bad_formula, root, input_root,
                                             expected_checkpoint_sha256="f" * 64)
    assert report.check("oracle_formula_preserved").status == "fail"

    bad_checkpoint = dict(valid, checkpoint_sha256="0" * 64)
    report = verifier.validate_oracle_record(artifact.candidate_id, artifact.formula, bad_checkpoint, root, input_root,
                                             expected_checkpoint_sha256="f" * 64)
    assert report.check("oracle_checkpoint_sha256").status == "fail"

    tampered = dict(valid, output_sha256="0" * 64)
    report = verifier.validate_oracle_record(artifact.candidate_id, artifact.formula, tampered, root, input_root,
                                             expected_checkpoint_sha256="f" * 64)
    assert report.check("oracle_output_sha256").status == "fail"

    bad_input = dict(valid, input_sha256="0" * 64)
    report = verifier.validate_oracle_record(artifact.candidate_id, artifact.formula, bad_input, root, input_root,
                                             expected_checkpoint_sha256="f" * 64)
    assert report.check("oracle_input_sha256").status == "fail"

    escaped_input = dict(valid, input_path=str((tmp_path / "outside_input.cif").resolve()))
    report = verifier.validate_oracle_record(artifact.candidate_id, artifact.formula, escaped_input, root, input_root,
                                             expected_checkpoint_sha256="f" * 64)
    assert report.check("oracle_input_containment").status == "fail"
