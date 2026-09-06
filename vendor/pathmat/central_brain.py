from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from pathmat.agent_contracts import AgentContract, default_agent_contracts
from pathmat.core import EvidencePackage, EvidenceLevel, critic_check
from pathmat.memory import MemoryStore, canonical_json
from pathmat.world_model import ReactionEdge, WorldState, relative_path_action, target_probability_from_actions


OPTIMADE_SNAPSHOT_SOURCE_IDS = frozenset({
    "materials_project",
    "oqmd",
    "nomad",
    "cod",
    "alexandria_pbesol",
})
TEMPORAL_GATE_POLICY_VERSION = "PATH-Mat-v3-temporal-two-track-gate-v1"
TEMPORAL_SOURCE_SCOPES = frozenset({"source_matched_retrospective", "unseen_or_shifted_source"})


class Planner(Protocol):
    def propose(self, task: "BrainTask", world_state: WorldState, memory: MemoryStore, cycle: int) -> list["BrainAction"]:
        ...


@dataclass(frozen=True)
class BrainTask:
    task_id: str
    objective: str
    target: str
    source_ids: tuple[str, ...]
    initial_state: dict[str, Any]
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrainAction:
    name: str
    authority_level: str
    payload: dict[str, Any]
    source_ids: tuple[str, ...] = ()
    rationale: str = ""
    requires_review: bool = True


@dataclass(frozen=True)
class BrainEvent:
    event_id: str
    task_id: str
    cycle: int
    stage: str
    status: str
    payload: dict[str, Any]
    event_hash: str

    @classmethod
    def build(cls, task_id: str, cycle: int, stage: str, status: str, payload: dict[str, Any], sequence: int) -> "BrainEvent":
        event_id = f"{task_id}:event:{sequence:06d}"
        body = {"event_id": event_id, "task_id": task_id, "cycle": cycle, "stage": stage, "status": status, "payload": payload}
        event_hash = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        return cls(event_id, task_id, cycle, stage, status, payload, event_hash)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrainResult:
    task_id: str
    status: str
    answer: dict[str, Any]
    world_state: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    checkpoint: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeterministicPathPlanner:
    """Offline planner used for replay and benchmark; an LLM can implement the same protocol."""

    def propose(self, task: BrainTask, world_state: WorldState, memory: MemoryStore, cycle: int) -> list[BrainAction]:
        if cycle > 0:
            return []
        edges = task.metadata.get("edges", [])
        source_candidate = task.metadata.get("source_candidate")
        process_memory_query = task.metadata.get("process_memory_query")
        nrel_htem_query = task.metadata.get("nrel_htem_query")
        nrel_htem_similarity_query = task.metadata.get("nrel_htem_similarity_query")
        nrel_htem_measurement_query = task.metadata.get("nrel_htem_measurement_query")
        oc20_e1_query = task.metadata.get("oc20_e1_query")
        dash_e2_query = task.metadata.get("dash_e2_query")
        optimade_snapshot_query = task.metadata.get("optimade_snapshot_query")
        if not edges and isinstance(optimade_snapshot_query, dict):
            return [BrainAction(
                "tool_call",
                "read_only_evidence",
                {"tool": "optimade.snapshot", "payload": optimade_snapshot_query},
                task.source_ids,
                "Retrieve the frozen, hashed OPTIMADE snapshot metadata only; preserve provider identity and license boundaries, and do not write world state, infer a synthesis outcome, or request physical action.",
                True,
            )]
        if not edges and isinstance(dash_e2_query, dict):
            return [BrainAction(
                "tool_call",
                "read_only_evidence",
                {"tool": "evidence.dash_e2.performance.retrieve", "payload": dash_e2_query},
                task.source_ids,
                "Retrieve one frozen external DASH process-performance record only; do not use a performance value as a phase outcome, write world state, train a synthesis-outcome model, or request physical action.",
                True,
            )]
        if not edges and isinstance(oc20_e1_query, dict):
            return [BrainAction(
                "tool_call",
                "read_only_evidence",
                {"tool": "evidence.oc20_e1.status", "payload": oc20_e1_query},
                task.source_ids,
                "Retrieve the frozen external-DFT OC20 cache manifest only; do not extract archives, rerun DFT, write world state, infer a synthesis outcome, or request physical action.",
                True,
            )]
        if not edges and isinstance(nrel_htem_measurement_query, dict):
            return [BrainAction(
                "tool_call",
                "read_only_evidence",
                {"tool": "evidence.nrel_htem.measurements.retrieve", "payload": nrel_htem_measurement_query},
                task.source_ids,
                "Retrieve frozen official NREL source-native measurements only; do not promote measured properties into an outcome, write world state, or request physical action.",
                True,
            )]
        if not edges and isinstance(nrel_htem_similarity_query, dict):
            return [BrainAction(
                "tool_call",
                "read_only_evidence",
                {"tool": "evidence.nrel_htem.similarity", "payload": nrel_htem_similarity_query},
                task.source_ids,
                "Retrieve frozen official NREL outcome-free XRD similarity pointers only; do not infer phase or outcome, write world state, or request physical action.",
                True,
            )]
        if not edges and isinstance(nrel_htem_query, dict):
            return [BrainAction(
                "tool_call",
                "read_only_evidence",
                {"tool": "evidence.nrel_htem.retrieve", "payload": nrel_htem_query},
                task.source_ids,
                "Retrieve frozen official NREL measurement pointers only; do not infer an outcome, write world state, or request physical action.",
                True,
            )]
        if not edges and isinstance(process_memory_query, str):
            return [BrainAction(
                "tool_call",
                "read_only_evidence",
                {"tool": "process_memory.retrieve", "payload": {"query": process_memory_query, "limit": task.metadata.get("process_memory_limit", 5)}},
                task.source_ids,
                "Retrieve source-licensed historical process context only; do not convert literature text into a world-state observation or physical action.",
                True,
            )]
        if not edges and isinstance(source_candidate, dict):
            return [BrainAction(
                "tool_call",
                "read_only_evidence",
                {"tool": "source.triage", "payload": {"candidate": source_candidate}},
                task.source_ids,
                "Conservatively triage source-declared metadata; do not download, extract, or promote a record.",
                True,
            )]
        if not edges:
            return [BrainAction("abstain", "local_reversible_compute", {"reason": "no_candidate_path_edges"}, task.source_ids, "Evidence is insufficient to construct a path.", False)]
        return [BrainAction("score_path", "local_reversible_compute", {"edges": edges, "temperature_K": task.metadata.get("temperature_K", 1000.0)}, task.source_ids, "Score the candidate reaction path with the PATH-Mat world model.", True)]


def default_world_state(initial: dict[str, Any] | None = None) -> WorldState:
    initial = dict(initial or {})
    state = WorldState(
        conserved_elements={str(k): float(v) for k, v in initial.get("conserved_elements", {"A": 1.0}).items()},
        phase_fractions={str(k): float(v) for k, v in initial.get("phase_fractions", {"target": 0.5, "competitor": 0.5}).items()},
        local_structure=dict(initial.get("local_structure", {})),
        defects={str(k): float(v) for k, v in initial.get("defects", {}).items()},
        microstructure={str(k): float(v) for k, v in initial.get("microstructure", {}).items()},
        environment={str(k): float(v) for k, v in initial.get("environment", {"temperature_K": 1000.0}).items()},
        history=tuple(str(item) for item in initial.get("history", [])),
    )
    state.validate()
    return state


class CentralBrainAgent:
    """Bounded event-driven central controller for the PATH-Mat domain model."""

    STAGES = ("INGEST", "RETRIEVE", "RECONSTRUCT", "PROPOSE", "SCORE", "DEBATE", "GATE", "ACT", "OBSERVE", "DECIDE")
    TERMINAL_STATES = {"completed", "abstained", "budget_exhausted", "external_dependency", "policy_blocked"}

    def __init__(
        self,
        *,
        planner: Planner | None = None,
        memory: MemoryStore | None = None,
        max_planning_cycles: int = 8,
        max_delegations: int = 24,
        max_tool_calls: int = 40,
        use_world_model: bool = True,
        use_critic: bool = True,
        use_safety: bool = True,
        tool_call: Callable[[str, dict[str, Any], bool], dict[str, Any]] | None = None,
        contracts: dict[str, AgentContract] | None = None,
    ) -> None:
        self.planner = planner or DeterministicPathPlanner()
        self.memory = memory or MemoryStore()
        self.max_planning_cycles = max_planning_cycles
        self.max_delegations = max_delegations
        self.max_tool_calls = max_tool_calls
        self.use_world_model = use_world_model
        self.use_critic = use_critic
        self.use_safety = use_safety
        self.tool_call = tool_call
        self.contracts = contracts or default_agent_contracts()

    def _emit(self, events: list[BrainEvent], task_id: str, cycle: int, stage: str, status: str, payload: dict[str, Any]) -> BrainEvent:
        event = BrainEvent.build(task_id, cycle, stage, status, payload, len(events))
        events.append(event)
        self.memory.append("episodic_memory", event.to_dict(), record_id=event.event_id)
        return event

    def _score_action(self, task: BrainTask, action: BrainAction) -> dict[str, Any]:
        """Return deterministic SCORE-stage metadata for an action.

        Subclasses use this hook to bind a domain model into SCORE without
        giving that model ownership of state or execution authority.
        """
        return {"world_model_enabled": self.use_world_model}

    @staticmethod
    def _state_dict(state: WorldState) -> dict[str, Any]:
        return asdict(state)

    @staticmethod
    def _parse_edges(raw_edges: Iterable[dict[str, Any]]) -> list[ReactionEdge]:
        return [ReactionEdge(**dict(edge)) for edge in raw_edges]

    @classmethod
    def _validate_path_payload(cls, action: BrainAction) -> None:
        """Validate path schema/physical fields independently of model execution.

        The safety gate must still reject malformed paths in the no-world-model
        ablation; otherwise that ablation conflates removal of scoring with
        removal of input safety.
        """
        if action.name != "score_path":
            return
        raw_edges = action.payload.get("edges")
        if not isinstance(raw_edges, list) or not raw_edges:
            raise ValueError("path must contain at least one reaction edge")
        temperature = float(action.payload.get("temperature_K", 1000.0))
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                raise TypeError("reaction edge must be an object")
            ReactionEdge(**dict(raw_edge)).to_path_step(temperature)

    @staticmethod
    def _temporal_policy_errors(task: BrainTask) -> list[str]:
        """Validate the explicit policy contract for a temporal/blind task.

        Tasks without ``temporal_gate`` retain their normal controller path. A
        temporal task, however, must declare whether it is source-matched or
        unseen/shifted and prove that blind labels are not used for policy.
        This keeps a generic LLM planner from silently converting a diagnostic
        into a temporal-transfer or discovery action.
        """
        gate = task.metadata.get("temporal_gate")
        if gate is None:
            return []
        if not isinstance(gate, dict):
            return ["temporal_gate_invalid"]
        reasons: list[str] = []
        if gate.get("policy_version") != TEMPORAL_GATE_POLICY_VERSION:
            reasons.append("temporal_gate_policy_version_invalid")
        if gate.get("source_scope") not in TEMPORAL_SOURCE_SCOPES:
            reasons.append("temporal_source_scope_invalid")
        if gate.get("blind_labels_used_for_policy") is not False:
            reasons.append("temporal_blind_labels_policy_forbidden")
        return reasons

    @staticmethod
    def _temporal_source_scope(task: BrainTask) -> str | None:
        gate = task.metadata.get("temporal_gate")
        return gate.get("source_scope") if isinstance(gate, dict) else None

    def _gate(self, task: BrainTask, action: BrainAction) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        temporal_errors = self._temporal_policy_errors(task)
        reasons.extend(temporal_errors)
        temporal_scope = self._temporal_source_scope(task)
        if temporal_scope == "unseen_or_shifted_source" and action.name != "abstain":
            reasons.append("temporal_unseen_source_must_abstain")
        if temporal_scope and action.name == "publish_claim":
            reasons.append("temporal_claim_publication_disallowed")
        if action.authority_level == "physical_actuation":
            reasons.append("physical_actuation_disabled")
        if action.authority_level == "expensive_simulation":
            reasons.append("expensive_simulation_plan_only")
        if self.use_safety and task.metadata.get("adversarial_input"):
            reasons.append("adversarial_input_blocked_by_safety")
        if action.authority_level not in {"read_only_evidence", "local_reversible_compute", "expensive_simulation", "physical_actuation"}:
            reasons.append("unknown_authority_level")
        if self.use_safety and action.name != "abstain" and not action.source_ids:
            reasons.append("action_source_ids_missing")
        if self.use_safety and action.source_ids and not self.memory.has_sources(action.source_ids):
            reasons.append("action_sources_not_in_memory")
        if (
            self.use_safety
            and action.name == "tool_call"
            and action.payload.get("tool") in {"evidence.nrel_htem.retrieve", "evidence.nrel_htem.similarity", "evidence.nrel_htem.measurements.retrieve"}
            and "nrel_htem_official_nlr_api_snapshot_v1" not in action.source_ids
        ):
            reasons.append("nrel_htem_source_binding_missing")
        if (
            self.use_safety
            and action.name == "tool_call"
            and action.payload.get("tool") == "optimade.snapshot"
            and not (OPTIMADE_SNAPSHOT_SOURCE_IDS & set(action.source_ids))
        ):
            reasons.append("optimade_snapshot_source_binding_missing")
        if (
            self.use_safety
            and action.name == "tool_call"
            and action.payload.get("tool") == "evidence.dash_e2.performance.retrieve"
            and "dash_author_github_external_oxide_process_performance_e2_v1" not in action.source_ids
        ):
            reasons.append("dash_e2_source_binding_missing")
        if (
            self.use_safety
            and action.name == "tool_call"
            and action.payload.get("tool") == "evidence.oc20_e1.status"
            and "open_catalyst_2020_official_e1_v1" not in action.source_ids
        ):
            reasons.append("oc20_e1_source_binding_missing")
        if (
            self.use_safety
            and action.name == "tool_call"
            and action.payload.get("tool") == "process_memory.retrieve"
            and "hf_omg_2024" not in action.source_ids
        ):
            reasons.append("process_memory_source_binding_missing")
        if self.use_safety and action.name == "score_path":
            try:
                self._validate_path_payload(action)
            except (TypeError, ValueError, KeyError) as exc:
                reasons.append(f"path_input_invalid:{type(exc).__name__}")
        if self.use_critic and action.name == "publish_claim" and action.payload.get("claim_status") not in {"verified", "proxy"}:
            reasons.append("critic_unsupported_claim_status")
        return not reasons, reasons

    def _execute(self, action: BrainAction) -> dict[str, Any]:
        if action.name == "abstain":
            return {"status": "abstained", "reason": action.payload.get("reason", "planner_abstained")}
        if action.name == "score_path":
            try:
                edges = self._parse_edges(action.payload["edges"])
                if self.use_world_model:
                    temperature = float(action.payload.get("temperature_K", 1000.0))
                    score = relative_path_action(edges, temperature)
                else:
                    score = None
            except (TypeError, ValueError, KeyError) as exc:
                if not self.use_safety:
                    return {"status": "completed", "path_score": None, "model": "unsafe_ablation_invalid_input_accepted", "invalid_input_accepted": True, "error": type(exc).__name__}
                return {"status": "abstained", "reason": f"invalid_path_input:{type(exc).__name__}"}
            if not self.use_world_model:
                return {"status": "completed", "path_score": None, "model": "ablation_no_world_model"}
            probability = target_probability_from_actions(score, [score + 1.0])
            return {"status": "completed", "path_score": score, "target_probability": probability, "model": "path_world_model"}
        if action.name == "tool_call":
            if self.tool_call is None:
                return {"status": "external_dependency", "reason": "tool_executor_not_configured"}
            tool = str(action.payload["tool"])
            tool_result = self.tool_call(tool, dict(action.payload.get("payload", {})), True)
            if not isinstance(tool_result, dict):
                return {"status": "external_dependency", "reason": "tool_executor_invalid_response", "tool": tool}
            if not tool_result.get("allowed", False):
                return {"status": "policy_blocked", "reason": tool_result.get("reasons", ["tool_call_denied"]), "tool": tool}
            result = tool_result.get("result", {})
            if not isinstance(result, dict):
                return {"status": "external_dependency", "reason": "tool_executor_result_invalid", "tool": tool}
            if result.get("status") == "tool_error":
                return {
                    "status": "tool_error",
                    "reason": result.get("reason", "tool_reported_error"),
                    "tool": tool,
                    "retryable": bool(result.get("retryable", True)),
                }
            if tool == "optimade.snapshot":
                providers = result.get("providers")
                required_boundary = (
                    result.get("status") == "retrieved"
                    and isinstance(providers, list)
                    and bool(providers)
                    and result.get("coverage_required_fields_complete") is True
                    and result.get("retrieval_only") is True
                    and result.get("world_state_write_eligible") is False
                    and result.get("strong_pairing") is False
                    and result.get("training_eligible") is False
                    and result.get("pathmat_owned_physical_experiment") is False
                    and all(
                        isinstance(provider, dict)
                        and provider.get("source_id") in OPTIMADE_SNAPSHOT_SOURCE_IDS
                        and isinstance(provider.get("snapshot_sha256"), str)
                        and len(provider["snapshot_sha256"]) == 64
                        for provider in providers
                    )
                )
                if not required_boundary:
                    return {"status": "policy_blocked", "reason": "optimade_snapshot_contract_violation", "tool": tool}
            if tool == "process_memory.retrieve":
                required_boundary = (
                    result.get("retrieval_only") is True
                    and result.get("world_state_write_eligible") is False
                    and result.get("strong_pairing") is False
                    and result.get("training_eligible") is False
                )
                if not required_boundary:
                    return {"status": "policy_blocked", "reason": "process_memory_contract_violation", "tool": tool}
            if tool == "evidence.nrel_htem.retrieve":
                required_boundary = (
                    result.get("retrieval_only") is True
                    and result.get("measurement_pointer_only") is True
                    and result.get("world_state_write_eligible") is False
                    and result.get("strong_pairing") is False
                    and result.get("training_eligible") is False
                    and result.get("pathmat_owned_physical_experiment") is False
                    and result.get("source_reported_outcome_available") is False
                )
                if not required_boundary:
                    return {"status": "policy_blocked", "reason": "nrel_htem_evidence_contract_violation", "tool": tool}
            if tool == "evidence.nrel_htem.similarity":
                required_boundary = (
                    result.get("retrieval_only") is True
                    and result.get("representation_only") is True
                    and result.get("measurement_pointer_only") is True
                    and result.get("world_state_write_eligible") is False
                    and result.get("strong_pairing") is False
                    and result.get("training_eligible") is False
                    and result.get("pathmat_owned_physical_experiment") is False
                    and result.get("source_reported_outcome_available") is False
                )
                if not required_boundary:
                    return {"status": "policy_blocked", "reason": "nrel_htem_similarity_contract_violation", "tool": tool}
            if tool == "evidence.nrel_htem.measurements.retrieve":
                required_boundary = (
                    result.get("retrieval_only") is True
                    and result.get("measurement_only") is True
                    and result.get("measurement_pointer_only") is True
                    and result.get("world_state_write_eligible") is False
                    and result.get("strong_pairing") is False
                    and result.get("training_eligible") is False
                    and result.get("pathmat_owned_physical_experiment") is False
                    and result.get("source_reported_outcome_available") is False
                )
                if not required_boundary:
                    return {"status": "policy_blocked", "reason": "nrel_htem_measurement_contract_violation", "tool": tool}
            if tool == "evidence.oc20_e1.status":
                required_boundary = (
                    result.get("retrieval_only") is True
                    and result.get("simulation_only") is True
                    and result.get("external_dft_only") is True
                    and result.get("world_state_write_eligible") is False
                    and result.get("strong_pairing") is False
                    and result.get("supervised_path_outcome_training_eligible") is False
                    and result.get("pathmat_owned_physical_experiment") is False
                    and result.get("source_reported_outcome_available") is False
                    and result.get("physical_action_eligible") is False
                )
                if not required_boundary:
                    return {"status": "policy_blocked", "reason": "oc20_e1_contract_violation", "tool": tool}
            if tool == "evidence.dash_e2.performance.retrieve":
                required_boundary = (
                    result.get("retrieval_only") is True
                    and result.get("external_source_native_performance_only") is True
                    and result.get("world_state_write_eligible") is False
                    and result.get("strong_pairing") is False
                    and result.get("strict_da3_multimodal_pair_eligible") is False
                    and result.get("world_model_synthesis_outcome_supervision_eligible") is False
                    and result.get("pathmat_owned_physical_experiment") is False
                    and result.get("source_reported_phase_or_impurity_outcome_available") is False
                    and result.get("physical_action_eligible") is False
                )
                if not required_boundary:
                    return {"status": "policy_blocked", "reason": "dash_e2_contract_violation", "tool": tool}
            return {"status": "completed", "tool": tool, "tool_result": result, "claim_status": "candidate_only"}
        if action.name == "publish_claim":
            return {"status": "completed", "claim": action.payload.get("claim"), "claim_status": action.payload.get("claim_status"), "model": "claim_ablation"}
        return {"status": "policy_blocked", "reason": "unknown_action"}

    def run(self, task: BrainTask, *, resume: dict[str, Any] | None = None) -> BrainResult:
        events: list[BrainEvent] = []
        state = default_world_state(task.initial_state)
        start_cycle = 0
        if resume:
            if resume.get("task_id") != task.task_id:
                raise ValueError("checkpoint task_id does not match task")
            state = default_world_state(resume["world_state"])
            start_cycle = int(resume.get("next_cycle", 0))
            for raw in resume.get("events", []):
                events.append(BrainEvent(**raw))
        self._emit(events, task.task_id, start_cycle, "INGEST", "ok", {"objective": task.objective, "target": task.target})
        if not task.source_ids:
            self._emit(events, task.task_id, start_cycle, "RETRIEVE", "abstain", {"reason": "task_source_ids_missing"})
            return self._result(task, "abstained", {"reason": "task_source_ids_missing"}, state, events, start_cycle + 1)
        if not self.memory.has_sources(task.source_ids):
            self._emit(events, task.task_id, start_cycle, "RETRIEVE", "abstain", {"reason": "task_sources_not_in_memory", "source_ids": task.source_ids})
            return self._result(task, "abstained", {"reason": "task_sources_not_in_memory"}, state, events, start_cycle + 1)
        self._emit(events, task.task_id, start_cycle, "RETRIEVE", "ok", {"source_ids": task.source_ids})
        self._emit(events, task.task_id, start_cycle, "RECONSTRUCT", "ok", {"world_state": self._state_dict(state)})
        temporal_errors = self._temporal_policy_errors(task)
        if temporal_errors:
            self._emit(events, task.task_id, start_cycle, "GATE", "deny", {"reasons": temporal_errors, "action": "temporal_policy"})
            self._emit(events, task.task_id, start_cycle, "DECIDE", "policy_blocked", {"reasons": temporal_errors})
            return self._result(task, "policy_blocked", {"reason": temporal_errors}, state, events, start_cycle + 1)
        if self._temporal_source_scope(task) == "unseen_or_shifted_source":
            reason = "temporal_unseen_source_must_abstain"
            self._emit(events, task.task_id, start_cycle, "GATE", "abstain", {"reasons": [reason], "action": "temporal_policy"})
            self._emit(events, task.task_id, start_cycle, "DECIDE", "abstained", {"reason": reason})
            return self._result(task, "abstained", {"reason": reason}, state, events, start_cycle + 1)
        for cycle in range(start_cycle, self.max_planning_cycles):
            if len(events) > self.max_delegations:
                self._emit(events, task.task_id, cycle, "DECIDE", "budget_exhausted", {"reason": "max_delegations"})
                return self._result(task, "budget_exhausted", {"reason": "max_delegations"}, state, events, cycle + 1)
            actions = self.planner.propose(task, state, self.memory, cycle)
            self._emit(events, task.task_id, cycle, "PROPOSE", "ok" if actions else "empty", {"actions": [asdict(action) for action in actions]})
            if not actions:
                self._emit(events, task.task_id, cycle, "DECIDE", "budget_exhausted", {"reason": "planner_returned_no_action"})
                return self._result(task, "budget_exhausted", {"reason": "planner_returned_no_action"}, state, events, cycle + 1)
            for action in actions[: self.max_tool_calls]:
                score_payload = {"action": asdict(action), **self._score_action(task, action)}
                self._emit(events, task.task_id, cycle, "SCORE", "ok", score_payload)
                if self.use_critic:
                    self._emit(events, task.task_id, cycle, "DEBATE", "pass", {"critic": "no_unsupported_claim_in_local_action"})
                allowed, reasons = self._gate(task, action)
                self._emit(events, task.task_id, cycle, "GATE", "allow" if allowed else "deny", {"reasons": reasons, "action": action.name})
                if not allowed:
                    status = "policy_blocked" if any("disabled" in reason or "unknown" in reason for reason in reasons) else "abstained"
                    self._emit(events, task.task_id, cycle, "DECIDE", status, {"reasons": reasons})
                    return self._result(task, status, {"reason": reasons}, state, events, cycle + 1)
                output = self._execute(action)
                self._emit(events, task.task_id, cycle, "ACT", "ok", {"action": action.name, "authority_level": action.authority_level})
                self._emit(events, task.task_id, cycle, "OBSERVE", output.get("status", "ok"), output)
                if output.get("status") in self.TERMINAL_STATES:
                    self._emit(events, task.task_id, cycle, "DECIDE", output["status"], {"answer": output})
                    return self._result(task, output["status"], output, state, events, cycle + 1)
            self._emit(events, task.task_id, cycle, "DECIDE", "checkpoint", {"next_cycle": cycle + 1})
        self._emit(events, task.task_id, self.max_planning_cycles, "DECIDE", "budget_exhausted", {"reason": "max_planning_cycles"})
        return self._result(task, "budget_exhausted", {"reason": "max_planning_cycles"}, state, events, self.max_planning_cycles)

    def _result(self, task: BrainTask, status: str, answer: dict[str, Any], state: WorldState, events: list[BrainEvent], next_cycle: int) -> BrainResult:
        checkpoint = {"task_id": task.task_id, "next_cycle": next_cycle, "world_state": self._state_dict(state), "events": [event.to_dict() for event in events]}
        return BrainResult(task.task_id, status, answer, self._state_dict(state), tuple(event.to_dict() for event in events), {"event_count": len(events), "role_count": len(self.contracts), "world_model_enabled": self.use_world_model, "critic_enabled": self.use_critic, "safety_enabled": self.use_safety}, checkpoint)

    @staticmethod
    def replay(result: BrainResult | dict[str, Any]) -> dict[str, Any]:
        """Verify event-chain hashes and return a deterministic replay summary."""
        payload = result.to_dict() if isinstance(result, BrainResult) else result
        events = payload.get("events", [])
        failures: list[str] = []
        for index, raw in enumerate(events):
            expected_id = f"{payload['task_id']}:event:{index:06d}"
            body = {key: raw[key] for key in ("event_id", "task_id", "cycle", "stage", "status", "payload")}
            expected_hash = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
            if raw.get("event_id") != expected_id:
                failures.append(f"event_id:{index}")
            if raw.get("event_hash") != expected_hash:
                failures.append(f"event_hash:{index}")
        return {"valid": not failures, "event_count": len(events), "failures": failures, "final_status": payload.get("status")}
