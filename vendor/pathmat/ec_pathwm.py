"""Evidence-conditioned competing-path contracts for PATH-Mat v4.

This module is an auditable implementation boundary, not a claim that a
learned or physical world model already exists.  It separates observed route
facts from model outputs and forces abstention when provenance or OOD coverage
is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import math
import numpy as np


EVIDENCE_LEVELS = {"E0", "E1", "E2", "E3", "E4", "E5"}
OBSERVED_OUTCOMES = {"reported_synthesis", "reported_failure", "reported_impurity", "inconclusive", "missing"}
ALLOWED_ACTIONS = {
    "Mixing", "SolutionMixing", "LiquidGrinding", "HeatingOperation",
    "DryingOperation", "ShapingOperation", "QuenchingOperation", "Wait",
    "AtmosphereSwitch", "Grinding", "Unknown",
}

PHYSICAL_DESCRIPTOR_FIELDS = (
    "reaction_energy_eV",
    "convex_hull_distance_eV",
    "competitive_phase_driving_eV",
    "diffusion_distance_A",
    "precursor_decomposition_temperature_C",
    "gas_release_window_C",
    "parallel_reaction_count",
    "competition_path_count",
    "path_bottleneck_eV",
    "interface_compatibility",
)


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


@dataclass(frozen=True)
class EvidenceBoundary:
    source_id: str
    source_locator: str
    content_hash: str
    license_status: str
    cutoff: str
    evidence_level: str = "E2"
    source_doi: str | None = None
    publication_year: int | None = None
    publication_year_dataset: int | None = None
    measurement_status: str = "not_available"
    observed: bool = True
    is_simulated: bool = False
    is_pathmat_owned_experiment: bool = False
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.source_id or not self.source_locator or not self.content_hash:
            raise ValueError("evidence boundary requires source identity, locator and hash")
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ValueError("invalid evidence level")
        if self.is_pathmat_owned_experiment:
            raise ValueError("PATH-Mat-owned experiment is outside the v4 public-data contract")
        if self.is_simulated and self.measurement_status == "measured":
            raise ValueError("simulated evidence cannot be marked measured")


@dataclass(frozen=True)
class ProcessAction:
    name: str
    ordinal: int
    temperature_C: float | None = None
    duration_h: float | None = None
    atmosphere: str | None = None
    heating_rate_C_min: float | None = None
    pressure_kPa: float | None = None
    observed: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.name not in ALLOWED_ACTIONS:
            raise ValueError(f"unsupported synthesis action: {self.name}")
        if self.ordinal < 0:
            raise ValueError("action ordinal must be non-negative")
        for value in (self.temperature_C, self.duration_h, self.heating_rate_C_min, self.pressure_kPa):
            if value is not None and (_finite(value) is None or float(value) < 0):
                raise ValueError("process attributes must be finite and non-negative")


@dataclass(frozen=True)
class PhysicalDescriptorBundle:
    """Optional thermo--kinetic/path descriptors with explicit provenance."""

    reaction_energy_eV: float | None = None
    convex_hull_distance_eV: float | None = None
    competitive_phase_driving_eV: float | None = None
    diffusion_distance_A: float | None = None
    precursor_decomposition_temperature_C: float | None = None
    gas_release_window_C: float | None = None
    parallel_reaction_count: int | None = None
    competition_path_count: int | None = None
    path_bottleneck_eV: float | None = None
    interface_compatibility: float | None = None
    availability: str = "missing"
    calculation_method: str | None = None
    source_id: str | None = None
    claim_status: str = "proxy"

    def validate(self) -> None:
        if self.availability not in {"missing", "observed", "computed", "proxy"}:
            raise ValueError("invalid physical descriptor availability")
        if self.claim_status not in {"proxy", "observed", "computed"}:
            raise ValueError("invalid physical descriptor claim status")
        values = {
            "reaction_energy_eV": self.reaction_energy_eV,
            "convex_hull_distance_eV": self.convex_hull_distance_eV,
            "competitive_phase_driving_eV": self.competitive_phase_driving_eV,
            "diffusion_distance_A": self.diffusion_distance_A,
            "precursor_decomposition_temperature_C": self.precursor_decomposition_temperature_C,
            "gas_release_window_C": self.gas_release_window_C,
            "path_bottleneck_eV": self.path_bottleneck_eV,
            "interface_compatibility": self.interface_compatibility,
        }
        for name, value in values.items():
            if value is not None and _finite(value) is None:
                raise ValueError(f"physical descriptor must be finite: {name}")
        for name, value in (("parallel_reaction_count", self.parallel_reaction_count), ("competition_path_count", self.competition_path_count)):
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(f"physical descriptor count must be a non-negative integer: {name}")
        if self.interface_compatibility is not None and not 0.0 <= float(self.interface_compatibility) <= 1.0:
            raise ValueError("interface_compatibility must lie in [0,1]")
        present = any(value is not None for value in values.values()) or self.parallel_reaction_count is not None or self.competition_path_count is not None
        if present and self.availability == "missing":
            raise ValueError("present physical descriptors cannot be marked missing")
        if present and not self.source_id and self.availability in {"observed", "computed"}:
            raise ValueError("observed/computed descriptors require source_id")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {field: getattr(self, field) for field in PHYSICAL_DESCRIPTOR_FIELDS} | {
            "availability": self.availability,
            "calculation_method": self.calculation_method,
            "source_id": self.source_id,
            "claim_status": self.claim_status,
        }


@dataclass(frozen=True)
class RouteTrajectory:
    route_id: str
    target_formula: str | None
    precursor_formulas: tuple[str, ...]
    actions: tuple[ProcessAction, ...]
    source: EvidenceBoundary
    observed_phases: tuple[str, ...] = ()
    competing_phases: tuple[str, ...] = ()
    reported_outcome: str = "missing"
    chemistry_family: str | None = None
    structure_prototype: str | None = None
    latent_state_mask: tuple[str, ...] = ()
    counterexample_of: str | None = None
    raw_record_hash: str | None = None
    physical_descriptors: PhysicalDescriptorBundle | None = None

    def validate(self) -> None:
        if not self.route_id:
            raise ValueError("route_id is required")
        self.source.validate()
        if self.reported_outcome not in OBSERVED_OUTCOMES:
            raise ValueError("reported_outcome is not an allowed observed label")
        ordinals = [action.ordinal for action in self.actions]
        if ordinals != sorted(set(ordinals)):
            raise ValueError("action ordinals must be unique and monotonic")
        for action in self.actions:
            action.validate()
        if self.physical_descriptors is not None:
            self.physical_descriptors.validate()
        if set(self.observed_phases) & set(self.latent_state_mask):
            raise ValueError("a phase cannot be both observed and latent")
        if self.source.is_simulated and self.reported_outcome in {"reported_synthesis", "reported_failure", "reported_impurity"}:
            raise ValueError("simulated records cannot carry experimental outcome labels")

    @property
    def route_available(self) -> bool:
        return bool(self.target_formula and self.precursor_formulas and self.actions)

    @property
    def has_competitor_label(self) -> bool:
        return bool(self.competing_phases) or self.reported_outcome == "reported_impurity"

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "route_id": self.route_id,
            "target_formula": self.target_formula,
            "precursor_formulas": list(self.precursor_formulas),
            "actions": [
                {"name": a.name, "ordinal": a.ordinal, "temperature_C": a.temperature_C,
                 "duration_h": a.duration_h, "atmosphere": a.atmosphere,
                 "heating_rate_C_min": a.heating_rate_C_min, "pressure_kPa": a.pressure_kPa,
                 "observed": a.observed, "metadata": dict(a.metadata)} for a in self.actions
            ],
            "source": {"source_id": self.source.source_id, "source_locator": self.source.source_locator,
                       "content_hash": self.source.content_hash, "license_status": self.source.license_status,
                       "cutoff": self.source.cutoff, "evidence_level": self.source.evidence_level,
                       "source_doi": self.source.source_doi, "publication_year": self.source.publication_year,
                       "publication_year_dataset": self.source.publication_year_dataset,
                       "measurement_status": self.source.measurement_status, "observed": self.source.observed,
                       "is_simulated": self.source.is_simulated, "is_pathmat_owned_experiment": self.source.is_pathmat_owned_experiment,
                       "notes": list(self.source.notes)},
            "observed_phases": list(self.observed_phases),
            "competing_phases": list(self.competing_phases),
            "reported_outcome": self.reported_outcome,
            "chemistry_family": self.chemistry_family,
            "structure_prototype": self.structure_prototype,
            "latent_state_mask": list(self.latent_state_mask),
            "counterexample_of": self.counterexample_of,
            "raw_record_hash": self.raw_record_hash,
            "physical_descriptors": self.physical_descriptors.to_dict() if self.physical_descriptors is not None else None,
            "route_available": self.route_available,
        }


@dataclass(frozen=True)
class PathPrediction:
    target_probability: float
    competitor_probabilities: Mapping[str, float]
    uncertainty_aleatoric: float
    uncertainty_epistemic: float
    evidence_risk: float
    path_score: float
    abstain: bool
    abstention_reasons: tuple[str, ...] = ()
    claim_level: str = "S0"
    model_version: str = "EC-PathWM-contract-v1"

    def validate(self) -> None:
        values = [self.target_probability, self.uncertainty_aleatoric, self.uncertainty_epistemic, self.evidence_risk, self.path_score]
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("prediction values must be finite")
        if not 0 <= self.target_probability <= 1:
            raise ValueError("target probability must be in [0,1]")
        if any(not 0 <= float(value) <= 1 for value in self.competitor_probabilities.values()):
            raise ValueError("competitor probabilities must be in [0,1]")
        if self.claim_level not in {"S0", "S1", "S2", "S3"}:
            raise ValueError("prediction cannot emit S4 physical discovery")
        if not self.abstain and self.abstention_reasons:
            raise ValueError("non-abstaining prediction cannot have abstention reasons")


def compute_path_score(target_probability: float, competitor_probabilities: Mapping[str, float], uncertainty: float, evidence_risk: float) -> float:
    """Compute a ranking score from model outputs; it does not fit a model."""
    if not 0 < target_probability <= 1:
        raise ValueError("target_probability must be in (0,1]")
    if any(value < 0 or value > 1 for value in competitor_probabilities.values()):
        raise ValueError("competitor probabilities must be in [0,1]")
    competitor_mass = sum(float(value) for value in competitor_probabilities.values())
    return float(math.log(target_probability) - math.log(max(competitor_mass, 1e-6)) - uncertainty - evidence_risk)


def risk_controlled_prediction(*, target_probability: float, competitor_probabilities: Mapping[str, float], uncertainty_aleatoric: float,
                               uncertainty_epistemic: float, evidence_risk: float, seen_source: bool,
                               seen_family: bool, source_coverage: float, minimum_coverage: float = 0.8,
                               claim_level: str = "S0") -> PathPrediction:
    """Create a conservative prediction from externally supplied model outputs.

    The function intentionally does not manufacture probabilities.  Training,
    calibration and the six OOD splits are upstream responsibilities.
    """
    reasons: list[str] = []
    if not seen_source:
        reasons.append("unseen_source")
    if not seen_family:
        reasons.append("unseen_chemistry_family")
    if source_coverage < minimum_coverage:
        reasons.append("insufficient_conformal_coverage")
    if uncertainty_epistemic > 0.35:
        reasons.append("high_epistemic_uncertainty")
    if evidence_risk > 0.35:
        reasons.append("high_evidence_risk")
    score = compute_path_score(target_probability, competitor_probabilities, uncertainty_aleatoric + uncertainty_epistemic, evidence_risk)
    prediction = PathPrediction(float(target_probability), dict(competitor_probabilities), float(uncertainty_aleatoric),
                                float(uncertainty_epistemic), float(evidence_risk), score, bool(reasons), tuple(reasons), claim_level)
    prediction.validate()
    return prediction
