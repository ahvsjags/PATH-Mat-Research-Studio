from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from pathmat.agents import AGENT_ROLES


@dataclass(frozen=True)
class AgentContract:
    role: str
    goal: str
    required_inputs: tuple[str, ...]
    output_fields: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    may_approve: bool = False
    may_veto: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_agent_contracts() -> dict[str, AgentContract]:
    contracts = {
        "Scout Agent": AgentContract("Scout Agent", "Find relevant admitted evidence and conservatively triage source metadata for future ingestion.", ("objective",), ("source_ids", "evidence_gaps", "candidate_triage"), ("crossref.snapshot", "optimade.snapshot", "source.triage", "process_memory.retrieve", "evidence.nrel_htem.retrieve", "evidence.nrel_htem.similarity", "evidence.nrel_htem.measurements.retrieve", "evidence.oc20_e1.status", "evidence.dash_e2.performance.retrieve")),
        "License-Provenance Agent": AgentContract("License-Provenance Agent", "Check provenance, license and version boundaries.", ("source_ids",), ("license_status", "provenance_findings"), ("crossref.snapshot",)),
        "Curator Agent": AgentContract("Curator Agent", "Normalize records and detect duplicates or leakage groups.", ("records",), ("canonical_records", "group_ids"), ("formula.parse",)),
        "Extraction Agent": AgentContract("Extraction Agent", "Extract structured recipe and evidence fields with abstention.", ("source_text_or_record",), ("recipe_fields", "abstain_reason"), (), False, False),
        "Physics Agent": AgentContract("Physics Agent", "Map candidate paths to world-state and physical constraints.", ("recipe_fields", "world_state"), ("path_edges", "physics_constraints"), ("formula.parse", "evidence.oc20_e1.status")),
        "Hypothesis Agent": AgentContract("Hypothesis Agent", "Generate competing hypotheses and falsification tests.", ("evidence", "world_state"), ("hypotheses", "falsification_tests"), (), False, False),
        "Synthesis Planner Agent": AgentContract("Synthesis Planner Agent", "Propose bounded candidate actions without external actuation.", ("hypotheses", "path_scores"), ("candidate_actions", "resource_budget"), ("process_memory.retrieve", "evidence.nrel_htem.retrieve", "evidence.nrel_htem.similarity", "evidence.nrel_htem.measurements.retrieve"), False, False),
        "Characterization Agent": AgentContract("Characterization Agent", "Specify observations needed to distinguish candidate paths.", ("candidate_actions",), ("observation_plan", "uncertainty_targets"), (), False, False),
        "Adversarial Critic Agent": AgentContract("Adversarial Critic Agent", "Search for unsupported claims, leakage and counterexamples.", ("decision_package",), ("critic_decision", "counterexamples"), (), False, True),
        "Statistician Agent": AgentContract("Statistician Agent", "Audit splits, calibration, uncertainty and comparison design.", ("evaluation_data",), ("metrics", "calibration_findings"), ("evidence.dash_e2.performance.retrieve",), False, False),
        "Safety Agent": AgentContract("Safety Agent", "Block unsafe, unlicensed or physically unauthorized actions.", ("candidate_actions", "device_limits"), ("safety_decision", "risk_reasons"), (), False, True),
        "Orchestrator Agent": AgentContract("Orchestrator Agent", "Coordinate worker contracts and preserve event lineage.", ("task", "policy"), ("event_ids", "final_status"), (), False, False),
    }
    if set(contracts) != set(AGENT_ROLES):
        raise AssertionError("agent contract registry does not cover exactly the configured roles")
    return contracts
