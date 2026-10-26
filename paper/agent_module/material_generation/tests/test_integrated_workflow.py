from __future__ import annotations

from pathlib import Path

import pytest

from agent.relaxation_controller import FailureAwareController, VerifiedStart
from agent.verified_restart import RestartHypothesis, VerifiedRestartIssue
from workflow import run_workflow


CONFIG = Path(__file__).resolve().parents[1] / "config" / "integration_demo.yaml"


def test_generated_candidates_and_balanced_routes(tmp_path):
    result = run_workflow(CONFIG, tmp_path / "design")
    assert result["generated_candidates"] == result["final_integrity_passes"] == 2
    assert result["route_hypotheses"] == 6
    assert result["restored_candidates"] == 0
    assert result["numerical_oracle_calls"] == 0


def test_development_repair_then_read_only_evaluation(tmp_path):
    dev = tmp_path / "development"
    first = run_workflow(CONFIG, dev, inject_byte_fault=True)
    assert first["restored_candidates"] == 1
    assert first["memory_sha256_after"]
    held = run_workflow(CONFIG, tmp_path / "heldout", split="confirmatory",
                        inject_byte_fault=True, memory_seed=dev / "verified_memory.jsonl")
    assert held["restored_candidates"] == 1
    assert held["confirmatory_memory_unchanged"] is True
    assert held["memory_sha256_before"] == first["memory_sha256_after"]


def test_workflow_never_overwrites_a_run(tmp_path):
    output = tmp_path / "design"
    run_workflow(CONFIG, output)
    with pytest.raises(FileExistsError):
        run_workflow(CONFIG, output)


def start(index, *, confirmed=True, force=0.06, v0="pass", repairable=True):
    cid = "start" + str(index)
    h = RestartHypothesis(cid, "Ba2ScNbO6", "a" * 64, "b" * 64, "c" * 64, "force check")
    i = VerifiedRestartIssue(cid, "confirmed" if confirmed else "unresolved", repairable,
                             "d" * 64, force, v0, "checked")
    return VerifiedStart(index, h, i)


def test_fourth_slot_chooses_lowest_verified_failed_class():
    d = FailureAwareController().decide([start(2), start(0), start(1)])
    assert d.optimizer == "LBFGS" and d.ordering_class == 0


@pytest.mark.parametrize("entry", [start(0,confirmed=False), start(0,force=0.05),
                                    start(0,v0="fail"), start(0,repairable=False)])
def test_unverified_or_ineligible_issues_use_class3(entry):
    d = FailureAwareController().decide([entry])
    assert d.optimizer == "FIRE" and d.ordering_class == 3


def test_failed_continuation_does_not_allocate_a_fifth_call():
    calls = []
    def continuation(decision):
        calls.append("LBFGS")
        return {"accepted": False}
    def fallback(decision):
        calls.append("FIRE")
        return {"accepted": True}
    result = FailureAwareController().dispatch([start(0)],
        continue_verified=continuation, run_class3=fallback)
    assert calls == ["LBFGS"]
    assert result["additional_slots_allocated"] == 1


def test_duplicate_classes_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        FailureAwareController().decide([start(0),start(0)])
