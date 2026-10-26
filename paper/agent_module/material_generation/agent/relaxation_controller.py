"""Public interface for the frozen, four-slot failure-aware relaxation rule.

This adapter exposes the existing policy as a workflow component. Its unit
tests check dispatch behaviour; the original confirmation runner remains the
source of the published numerical result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from .verified_restart import RestartHypothesis, VerifiedRestartIssue


@dataclass(frozen=True)
class VerifiedStart:
    ordering_class: int
    hypothesis: RestartHypothesis
    issue: VerifiedRestartIssue


@dataclass(frozen=True)
class FourthSlotDecision:
    method: str
    optimizer: str
    ordering_class: int
    source_candidate_id: str | None
    evidence_sha256: str | None


class FailureAwareController:
    """Allocate one fourth slot; a failed continuation never earns a fifth call."""

    ACCEPTANCE_FORCE_EV_A = 0.050001

    def decide(self, starts: Iterable[VerifiedStart]) -> FourthSlotDecision:
        starts = list(starts)
        classes = [x.ordering_class for x in starts]
        if len(classes) != len(set(classes)):
            raise ValueError("duplicate ordering classes in verified starts")
        eligible = []
        for start in starts:
            issue, hypothesis = start.issue, start.hypothesis
            if start.ordering_class not in {0, 1, 2}:
                raise ValueError("the shared starts are restricted to classes 0, 1 and 2")
            if (issue.candidate_id == hypothesis.candidate_id
                    and issue.status == "confirmed" and issue.repairable
                    and issue.v0_status == "pass" and issue.evidence_sha256
                    and issue.recomputed_force_eV_A is not None
                    and issue.recomputed_force_eV_A > self.ACCEPTANCE_FORCE_EV_A):
                eligible.append(start)
        if not eligible:
            return FourthSlotDecision("FIRE_catalog_fallback_class3", "FIRE", 3, None, None)
        source = min(eligible, key=lambda x: x.ordering_class)
        return FourthSlotDecision("LBFGS_force_only_continuation", "LBFGS",
                                  source.ordering_class, source.hypothesis.candidate_id,
                                  source.issue.evidence_sha256)

    def dispatch(self, starts: Iterable[VerifiedStart], *,
                 continue_verified: Callable[[FourthSlotDecision], dict[str, Any]],
                 run_class3: Callable[[FourthSlotDecision], dict[str, Any]]) -> dict[str, Any]:
        decision = self.decide(starts)
        result = (continue_verified(decision) if decision.optimizer == "LBFGS"
                  else run_class3(decision))
        return {"decision": asdict(decision), "fourth_slot_result": result,
                "additional_slots_allocated": 1}
