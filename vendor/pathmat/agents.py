from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pathmat.core import EvidencePackage, GateDecision, critic_check, safety_gate

AGENT_ROLES = (
    "Scout Agent",
    "License-Provenance Agent",
    "Curator Agent",
    "Extraction Agent",
    "Physics Agent",
    "Hypothesis Agent",
    "Synthesis Planner Agent",
    "Characterization Agent",
    "Adversarial Critic Agent",
    "Statistician Agent",
    "Safety Agent",
    "Orchestrator Agent",
)

REQUIRED_PROTOCOL_CHECKS = (
    "precursor_exists",
    "purity_checked",
    "device_compatible",
    "phase_checked",
    "volatility_checked",
    "toxicity_checked",
    "pressure_checked",
    "atmosphere_checked",
    "uncertainty_checked",
    "alternative_path_checked",
)

CODEX_REVIEW_ROLES = ("text_extractor", "workflow_builder", "conservative_auditor")


def codex_review_gate(review: dict[str, Any] | None) -> GateDecision:
    """Validate a structured Codex review record before recipe gating."""

    if not review:
        return GateDecision(False, ["codex_review_missing"])
    reasons: list[str] = []
    if review.get("review_mode") != "ai_assisted_multi_agent":
        reasons.append("codex_review_mode_invalid")
    if set(review.get("roles", [])) != set(CODEX_REVIEW_ROLES):
        reasons.append("codex_review_roles_incomplete")
    if review.get("status") not in {"AI_ASSISTED_CONSENSUS", "AI_ASSISTED_APPROVED"}:
        reasons.append("codex_review_not_approved")
    if review.get("abstained") is True:
        reasons.append("codex_review_abstained")
    if not review.get("evidence_checked", False):
        reasons.append("codex_review_evidence_unchecked")
    return GateDecision(not reasons, reasons)


@dataclass(frozen=True)
class DeviceLimits:
    """Explicit facility limits used before a recipe can enter an experiment queue."""

    min_temperature_C: float = 0.0
    max_temperature_C: float = 1800.0
    max_duration_h: float = 168.0
    allowed_atmospheres: tuple[str, ...] = ("air", "inert", "vacuum")


def device_gate(conditions: dict[str, Any] | None, limits: DeviceLimits | None = None) -> GateDecision:
    limits = limits or DeviceLimits()
    reasons: list[str] = []
    if not conditions:
        return GateDecision(False, ["device_constraints_missing"])
    try:
        minimum = float(conditions.get("temperature_C_min", conditions.get("temperature_C", 0.0)))
        maximum = float(conditions.get("temperature_C_max", conditions.get("temperature_C", minimum)))
        duration = float(conditions.get("duration_h", 0.0))
    except (TypeError, ValueError):
        return GateDecision(False, ["device_condition_not_numeric"])
    if minimum < limits.min_temperature_C or maximum > limits.max_temperature_C or minimum > maximum:
        reasons.append("temperature_outside_device_limits")
    if duration < 0 or duration > limits.max_duration_h:
        reasons.append("duration_outside_device_limits")
    atmosphere = str(conditions.get("atmosphere", "")).lower()
    if atmosphere not in {value.lower() for value in limits.allowed_atmospheres}:
        reasons.append("atmosphere_not_approved")
    return GateDecision(not reasons, reasons)


def protocol_gate(checks: dict[str, Any] | None) -> GateDecision:
    """Require structured evidence for every pre-experiment safety/provenance check."""

    if not checks:
        return GateDecision(False, ["protocol_checks_missing"])
    reasons = [f"protocol_check_failed:{key}" for key in REQUIRED_PROTOCOL_CHECKS if checks.get(key) is not True]
    return GateDecision(not reasons, reasons)


class Orchestrator:
    def __init__(self, roles: tuple[str, ...] = AGENT_ROLES) -> None:
        self.roles = roles
        self.events: list[dict[str, Any]] = []

    def evaluate_recipe(self, package: EvidencePackage, reactants: list[tuple[float, str]], products: list[tuple[float, str]], codex_review: dict[str, Any] | None = None, device_conditions: dict[str, Any] | None = None, device_limits: DeviceLimits | None = None, protocol_checks: dict[str, Any] | None = None) -> dict[str, Any]:
        critic = critic_check(package)
        codex = codex_review_gate(codex_review)
        safety = safety_gate(reactants, products, package.safety_constraints, codex_approved=codex.allowed)
        device = device_gate(device_conditions, device_limits)
        protocol = protocol_gate(protocol_checks)
        event = {"allowed": critic.allowed and codex.allowed and safety.allowed and device.allowed and protocol.allowed, "critic": asdict(critic), "codex_review": asdict(codex), "safety": asdict(safety), "device": asdict(device), "protocol": asdict(protocol), "target": package.target, "evidence_level": package.evidence_level.value, "approval_mode": "codex_agent"}
        self.events.append(event)
        return event

    def write_events(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"roles": list(self.roles), "events": self.events}, ensure_ascii=False, indent=2), encoding="utf-8")
