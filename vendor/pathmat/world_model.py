"""Equation-level interfaces for the PATH-Mat path world model.

This module implements the invariant-preserving and bookkeeping parts of the
handbook equations. It is deliberately explicit about the fact that learned
DFT/MLIP/NEB/MD parameters and physical observations are external inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from pathmat.core import PathStep, normalize_simplex, path_action_score, target_probability


BOLTZMANN_EV_PER_K = 8.617333262145e-5


@dataclass(frozen=True)
class WorldState:
    """State vector matching the handbook's n, phi, q, d, m, c and history."""

    conserved_elements: dict[str, float]
    phase_fractions: dict[str, float]
    local_structure: dict[str, Any]
    defects: dict[str, float]
    microstructure: dict[str, float]
    environment: dict[str, float]
    history: tuple[str, ...] = ()

    def validate(self, tolerance: float = 1e-8) -> None:
        if not self.conserved_elements or any(not np.isfinite(v) or v < 0 for v in self.conserved_elements.values()):
            raise ValueError("conserved elements must be finite and non-negative")
        if not self.phase_fractions:
            raise ValueError("phase simplex is empty")
        if any(not np.isfinite(v) or v < -tolerance for v in self.phase_fractions.values()):
            raise ValueError("phase fractions must be finite and non-negative")
        if abs(sum(self.phase_fractions.values()) - 1.0) > tolerance:
            raise ValueError("phase fractions must sum to one")
        if any(not np.isfinite(v) or v < 0 for v in self.defects.values()):
            raise ValueError("defect values must be finite and non-negative")
        if any(not np.isfinite(v) for v in self.microstructure.values()):
            raise ValueError("microstructure values must be finite")
        if any(not np.isfinite(v) for v in self.environment.values()):
            raise ValueError("environment values must be finite")
        if self.environment.get("temperature_K", 1.0) <= 0:
            raise ValueError("temperature_K must be positive")


@dataclass(frozen=True)
class ReactionEdge:
    """One edge in a reaction hypergraph for the relative path-action score."""

    name: str
    barrier_eV: float
    driving_eV: float
    diffusion_length_nm: float
    diffusivity_nm2_s: float
    duration_s: float
    topology_mismatch: float
    volatility_risk: float
    source_id: str = "unverified"
    evidence_level: str = "E0"

    def to_path_step(self, temperature_K: float) -> PathStep:
        if temperature_K <= 0:
            raise ValueError("temperature_K must be positive")
        values = [
            self.barrier_eV,
            self.driving_eV,
            self.diffusion_length_nm,
            self.diffusivity_nm2_s,
            self.duration_s,
            self.topology_mismatch,
            self.volatility_risk,
        ]
        if not np.isfinite(values).all():
            raise ValueError("reaction-edge values must be finite")
        if self.barrier_eV < 0 or self.diffusivity_nm2_s <= 0 or self.duration_s <= 0 or self.topology_mismatch < 0 or self.volatility_risk < 0:
            raise ValueError("invalid reaction-edge physical terms")
        kT = BOLTZMANN_EV_PER_K * temperature_K
        return PathStep(
            barrier_over_kT=self.barrier_eV / kT,
            driving_over_kT=self.driving_eV / kT,
            diffusion_penalty=(self.diffusion_length_nm**2) / (self.diffusivity_nm2_s * self.duration_s),
            topology_penalty=self.topology_mismatch,
            volatility_penalty=self.volatility_risk,
        )


def relative_path_action(edges: list[ReactionEdge], temperature_K: float) -> float:
    """Compute the handbook's candidate action S_pi for a path."""
    if not edges:
        raise ValueError("path must contain at least one reaction edge")
    return path_action_score([edge.to_path_step(temperature_K) for edge in edges])


def target_probability_from_actions(target_action: float, competitor_actions: list[float]) -> float:
    """Softmax probability exp(-S_target)/sum_k exp(-S_k)."""
    scores = np.asarray([target_action, *competitor_actions], dtype=float)
    if scores.size < 2 or not np.isfinite(scores).all():
        raise ValueError("at least one finite competitor action is required")
    return target_probability(float(target_action), competitor_actions)


def euler_stochastic_step(
    state: WorldState,
    phase_drift: Mapping[str, float],
    dt: float,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> WorldState:
    """Apply one Euler--Maruyama step while preserving the phase simplex.

    The drift/noise are supplied model outputs; this function does not claim
    that they are learned from physical trajectories.
    """
    state.validate()
    if dt <= 0 or noise_scale < 0 or not np.isfinite([dt, noise_scale]).all():
        raise ValueError("dt must be positive and noise_scale must be non-negative")
    rng = np.random.default_rng(seed)
    phases = dict(state.phase_fractions)
    keys = tuple(phases)
    delta = np.asarray([float(phase_drift.get(key, 0.0)) for key in keys], dtype=float) * dt
    if noise_scale:
        delta += rng.normal(0.0, noise_scale * np.sqrt(dt), size=len(keys))
    updated = normalize_simplex({key: max(0.0, phases[key] + change) for key, change in zip(keys, delta)})
    result = WorldState(
        conserved_elements=dict(state.conserved_elements),
        phase_fractions=updated,
        local_structure=dict(state.local_structure),
        defects=dict(state.defects),
        microstructure=dict(state.microstructure),
        environment=dict(state.environment),
        history=(*state.history, f"euler_maruyama:{seed}"),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class MultiFidelityObservation:
    low_value: float
    high_value: float
    rho: float
    discrepancy: float
    source_id: str
    evidence_level: str

    @property
    def reconstructed_high(self) -> float:
        return self.rho * self.low_value + self.discrepancy

    def validate(self) -> None:
        values = np.asarray([self.low_value, self.high_value, self.rho, self.discrepancy], dtype=float)
        if not np.isfinite(values).all() or not self.source_id or self.evidence_level not in {"E0", "E1", "E2", "E3", "E4", "E5"}:
            raise ValueError("invalid multi-fidelity observation")


def physics_constraint_penalty(
    predicted_elements: Mapping[str, float],
    expected_elements: Mapping[str, float],
    predicted_phases: Mapping[str, float],
) -> dict[str, float]:
    """Return explicit conservation/simplex loss terms from the handbook."""
    all_elements = set(predicted_elements) | set(expected_elements)
    element_residual = float(sum((float(predicted_elements.get(k, 0.0)) - float(expected_elements.get(k, 0.0))) ** 2 for k in all_elements))
    negative_phase = float(sum(max(0.0, -float(value)) ** 2 for value in predicted_phases.values()))
    simplex = float((sum(float(value) for value in predicted_phases.values()) - 1.0) ** 2)
    return {"element_conservation": element_residual, "negative_phase": negative_phase, "simplex": simplex, "total": element_residual + negative_phase + simplex}


def world_model_loss(components: Mapping[str, float], weights: Mapping[str, float] | None = None) -> float:
    """Weighted bookkeeping loss for PES, dynamics, observations and calibration."""
    weights = dict(weights or {})
    total = 0.0
    for name, value in components.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"loss component must be finite and non-negative: {name}")
        total += float(weights.get(name, 1.0)) * float(value)
    return total
