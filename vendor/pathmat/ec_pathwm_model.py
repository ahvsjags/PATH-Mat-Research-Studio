"""CPU reference implementation of the evidence-conditioned PathWM heads.

The implementation is intentionally honest about the current public-data
boundary: it learns explicit competitor/outcome labels and observed next
actions.  It does not manufacture intermediate phases or next-material-state
labels.  The class is a deployable reference head for the v4 contract, not a
claim of a validated physical transition law.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pathmat.ec_pathwm import PHYSICAL_DESCRIPTOR_FIELDS, compute_path_score


_ELEMENT = re.compile(r"[A-Z][a-z]?")


def _tokens(value: Any) -> list[str]:
    return _ELEMENT.findall(str(value or ""))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def explicit_competitor_label(record: Mapping[str, Any]) -> int:
    evidence_label = record.get("evidence_conditioned_competitor_label")
    if evidence_label is not None:
        # This is a DOI-level public text-mined weak label.  It is allowed as
        # an observed training target, but is never promoted to physical truth.
        return int(bool(evidence_label))
    return int(bool(record.get("competing_phases")) or record.get("reported_outcome") == "reported_impurity")


def outcome_label(record: Mapping[str, Any]) -> int | None:
    outcome = record.get("reported_outcome")
    if outcome == "reported_synthesis":
        return 0
    if outcome in {"reported_impurity", "reported_failure"}:
        return 1
    return None


def _actions(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in (record.get("actions") or []) if isinstance(item, Mapping)]


def physical_descriptor_features(record: Mapping[str, Any]) -> dict[str, float]:
    """Expose optional descriptors with explicit missingness indicators."""
    bundle = record.get("physical_descriptors") or record.get("descriptors") or {}
    if not isinstance(bundle, Mapping):
        bundle = {}
    output: dict[str, float] = {"physical_descriptor_bundle_present": float(bool(bundle))}
    for field in PHYSICAL_DESCRIPTOR_FIELDS:
        value = bundle.get(field)
        output[f"physical_descriptor_present={field}"] = float(value is not None)
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                scale = 1000.0 if field.endswith("_C") else 10.0 if field.endswith("_eV") else 100.0 if field.endswith("_A") else 1.0
                output[f"physical_descriptor={field}"] = numeric / scale
    return output


def structure_and_route_proxy_features(record: Mapping[str, Any]) -> dict[str, float]:
    """Use public structure/route geometry proxies without treating them as physics.

    These features are derived from formula-matched OPTIMADE structures and
    observed operation metadata.  They contain no reaction energies,
    convex-hull distances, decomposition temperatures, diffusion barriers, or
    outcome labels.
    """
    output: dict[str, float] = {}
    structure = record.get("structure_proxy") or {}
    if not isinstance(structure, Mapping):
        structure = {}
    output["structure_proxy_target_match"] = float(structure.get("target_match_count", 0) > 0)
    output["structure_proxy_source_count"] = float(len(structure.get("target_sources") or []))
    geometry = structure.get("target_geometry") or {}
    if not isinstance(geometry, Mapping):
        geometry = {}
    scales = {
        "volume_per_atom_A3": 100.0,
        "density_g_cm3": 10.0,
        "cell_anisotropy": 10.0,
        "nsites": 100.0,
    }
    for field, scale in scales.items():
        value = geometry.get(field)
        output[f"structure_proxy_present={field}"] = float(value is not None)
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                output[f"structure_proxy={field}"] = numeric / scale

    route_proxy = record.get("route_geometry_proxy") or {}
    if not isinstance(route_proxy, Mapping):
        route_proxy = {}
    numeric_route_fields = {
        "action_count": 20.0,
        "heating_event_count": 10.0,
        "mixing_or_grinding_event_count": 10.0,
        "observed_temperature_count": 10.0,
        "observed_duration_count": 20.0,
        "max_temperature_C": 1500.0,
        "temperature_range_C": 1500.0,
        "total_observed_duration_h": 100.0,
        "atmosphere_observed_fraction": 1.0,
    }
    for field, scale in numeric_route_fields.items():
        value = route_proxy.get(field)
        output[f"route_proxy_present={field}"] = float(value is not None)
        if value is not None:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                output[f"route_proxy={field}"] = numeric / scale
    order = route_proxy.get("operation_order_proxy")
    if order:
        output[f"route_order_proxy={order}"] = 1.0
    return output


def route_features(record: Mapping[str, Any], *, include_process: bool = True, include_evidence: bool = True) -> dict[str, float]:
    """Build sparse, provenance-aware features without using target labels."""
    output: dict[str, float] = {}
    target = str(record.get("target_formula") or "")
    target_tokens = _tokens(target)
    for token in sorted(set(target_tokens)):
        output[f"target_element={token}"] = 1.0
    output[f"target_formula_hash={_hash(target)}"] = 1.0
    output["target_atom_token_count"] = float(len(target_tokens))
    precursors = record.get("precursor_formulas") or []
    output["precursor_count"] = float(len(precursors))
    for precursor in precursors:
        for token in sorted(set(_tokens(precursor))):
            output[f"precursor_element={token}"] = output.get(f"precursor_element={token}", 0.0) + 1.0
    actions = _actions(record)
    output["route_available"] = float(bool(record.get("route_available")))
    output["action_count"] = float(len(actions))
    output["latent_state_count"] = float(len(record.get("latent_state_mask") or []))
    output[f"family={record.get('chemistry_family') or 'unknown'}"] = 1.0
    if include_process:
        for index, action in enumerate(actions):
            name = str(action.get("name") or "Unknown")
            output[f"action={name}"] = output.get(f"action={name}", 0.0) + 1.0
            output[f"action_order_{min(index, 7)}={name}"] = 1.0
        temperatures = [float(item["temperature_C"]) for item in actions if item.get("temperature_C") is not None]
        durations = [float(item["duration_h"]) for item in actions if item.get("duration_h") is not None]
        output["temperature_observed"] = float(bool(temperatures))
        output["duration_observed"] = float(bool(durations))
        output["temperature_mean_norm"] = float(np.mean(temperatures) / 1000.0) if temperatures else 0.0
        output["temperature_range_norm"] = float((max(temperatures) - min(temperatures)) / 1000.0) if len(temperatures) > 1 else 0.0
        output["duration_mean_norm"] = float(np.mean(durations) / 10.0) if durations else 0.0
        output["has_atmosphere"] = float(any(item.get("atmosphere") for item in actions))
        for item in actions:
            atmosphere = str(item.get("atmosphere") or "").strip().lower()
            if atmosphere:
                output[f"atmosphere={atmosphere}"] = output.get(f"atmosphere={atmosphere}", 0.0) + 1.0
    if include_evidence:
        source = record.get("source") or {}
        output[f"source_id={source.get('source_id') or 'unknown'}"] = 1.0
        output[f"evidence_level={source.get('evidence_level') or 'unknown'}"] = 1.0
        output[f"measurement_status={source.get('measurement_status') or 'unknown'}"] = 1.0
        output["publication_year_norm"] = float((source.get("publication_year") or 0) / 2025.0)
        output["source_observed"] = float(bool(source.get("observed")))
        output["source_simulated"] = float(bool(source.get("is_simulated")))
        evidence = record.get("evidence_conditioned_competitor") or {}
        if isinstance(evidence, Mapping):
            status = str(evidence.get("status") or "unavailable")
            # Expose only evidence availability/scope.  Do not expose the
            # positive/negative semantic label as a feature of the competing-
            # phase target; that would be circular leakage.
            scope = "matched" if status in {"explicit_competitor_report", "explicit_no_competitor_report"} else "conflict" if status == "conflicting_doi_evidence_abstain" else "unmatched"
            output[f"competitor_evidence_scope={scope}"] = 1.0
            output["competitor_evidence_present"] = float(scope == "matched")
    output.update(physical_descriptor_features(record))
    output.update(structure_and_route_proxy_features(record))
    return output


@dataclass(frozen=True)
class ECPathWMConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    min_source_coverage: float = 0.8
    max_epistemic_uncertainty: float = 0.35
    max_evidence_risk: float = 0.35
    loss_weights: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 0.5, 0.5)


class ECPathWM:
    """Reference multi-head estimator for route-level v4 public evidence."""

    def __init__(self, config: ECPathWMConfig | None = None):
        self.config = config or ECPathWMConfig()
        self.vectorizer: DictVectorizer | None = None
        self.composition_vectorizer: DictVectorizer | None = None
        self.competitor_models: list[LogisticRegression] = []
        self.competitor_calibrators: list[IsotonicRegression | None] = []
        self.outcome_models: list[LogisticRegression] = []
        self.outcome_calibrators: list[IsotonicRegression | None] = []
        self.next_action_vectorizer: DictVectorizer | None = None
        self.next_action_models: list[LogisticRegression] = []
        self.source_coverage: dict[str, float] = {}
        self.family_coverage: dict[str, float] = {}
        self.metadata: dict[str, Any] = {
            "model_name": "EC-PathWM",
            "state_transition_scope": "observed_next_action_only",
            "next_material_state_labels": "unavailable_and_not_imputed",
            "intermediate_phase_policy": "latent_mask_only",
            "modules": [
                "composition_phase_graph_encoder_proxy",
                "precursor_competitor_graph_proxy",
                "process_event_sequence_head",
                "source_family_evidence_conditioning",
                "thermo_kinetic_path_descriptor_adapter_missingness_safe",
                "public_structure_geometry_and_observed_route_proxy_adapter",
                "competing_risk_head",
                "calibration_and_abstention_head",
            ],
            "physical_descriptor_status": "optional_contract_missing_in_current_public_route_layer",
            "structure_proxy_status": "optional_public_OPTIMADE_geometry_only_no_energy_or_kinetic_claim",
            "loss_weights": {
                "next_state": self.config.loss_weights[0],
                "competitor": self.config.loss_weights[1],
                "outcome": self.config.loss_weights[2],
                "calibration": self.config.loss_weights[3],
                "path": self.config.loss_weights[4],
            },
        }

    @staticmethod
    def _fit_binary_models(x, y: np.ndarray, seeds: Iterable[int]) -> list[LogisticRegression]:
        models: list[LogisticRegression] = []
        if len(np.unique(y)) < 2:
            return models
        for seed in seeds:
            model = LogisticRegression(max_iter=300, solver="liblinear", random_state=int(seed), class_weight="balanced")
            model.fit(x, y)
            models.append(model)
        return models

    @staticmethod
    def _compact_matrix(matrix):
        matrix = matrix.tocsr()
        matrix.indices = matrix.indices.astype(np.int32, copy=False)
        matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
        return matrix

    def fit(self, records: list[Mapping[str, Any]]) -> "ECPathWM":
        if not records:
            raise ValueError("at least one route record is required")
        fit_records = []
        calibration_records = []
        for record in records:
            key = str(record.get("route_id") or record.get("raw_record_hash") or "")
            bucket = int(_hash(key)[:8], 16) / float(16**8)
            (fit_records if bucket < 0.8 else calibration_records).append(record)
        if len(fit_records) < 20:
            fit_records, calibration_records = list(records), []
        self.vectorizer = DictVectorizer(sparse=True)
        x = self._compact_matrix(self.vectorizer.fit_transform([route_features(record) for record in fit_records]))
        competitor_y = np.asarray([explicit_competitor_label(record) for record in fit_records], dtype=int)
        self.competitor_models = self._fit_binary_models(x, competitor_y, self.config.seeds)
        if calibration_records:
            calibration_x = self._compact_matrix(self.vectorizer.transform([route_features(record) for record in calibration_records]))
            calibration_y = np.asarray([explicit_competitor_label(record) for record in calibration_records], dtype=int)
            self.competitor_calibrators = self._fit_calibrators(self.competitor_models, calibration_x, calibration_y)

        outcome_records = [(record, outcome_label(record)) for record in fit_records]
        outcome_records = [(record, label) for record, label in outcome_records if label is not None]
        if outcome_records:
            outcome_x = self._compact_matrix(self.vectorizer.transform([route_features(record) for record, _ in outcome_records]))
            outcome_y = np.asarray([label for _, label in outcome_records], dtype=int)
            self.outcome_models = self._fit_binary_models(outcome_x, outcome_y, self.config.seeds)
            if calibration_records:
                calibrated_outcomes = [(record, outcome_label(record)) for record in calibration_records]
                calibrated_outcomes = [(record, label) for record, label in calibrated_outcomes if label is not None]
                if calibrated_outcomes:
                    calibration_outcome_x = self._compact_matrix(self.vectorizer.transform([route_features(record) for record, _ in calibrated_outcomes]))
                    calibration_outcome_y = np.asarray([label for _, label in calibrated_outcomes], dtype=int)
                    self.outcome_calibrators = self._fit_calibrators(self.outcome_models, calibration_outcome_x, calibration_outcome_y)

        source_counts: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        for record in fit_records:
            source = str((record.get("source") or {}).get("source_id") or "unknown")
            family = str(record.get("chemistry_family") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            family_counts[family] = family_counts.get(family, 0) + 1
        total = float(len(records))
        self.source_coverage = {key: value / total for key, value in source_counts.items()}
        self.family_coverage = {key: value / total for key, value in family_counts.items()}

        transitions: list[tuple[dict[str, float], str]] = []
        for record in fit_records:
            actions = _actions(record)
            for index in range(len(actions) - 1):
                prefix = dict(record)
                prefix["actions"] = actions[: index + 1]
                transitions.append((route_features(prefix), str(actions[index + 1].get("name") or "Unknown")))
        if transitions:
            self.next_action_vectorizer = DictVectorizer(sparse=True)
            transition_x = self._compact_matrix(self.next_action_vectorizer.fit_transform([item[0] for item in transitions]))
            transition_y = np.asarray([item[1] for item in transitions])
            if len(np.unique(transition_y)) > 1:
                self.next_action_models = []
                for seed in self.config.seeds:
                    model = LogisticRegression(max_iter=300, solver="lbfgs", random_state=int(seed))
                    model.fit(transition_x, transition_y)
                    self.next_action_models.append(model)
        self.metadata.update({
            "fit_record_count": len(fit_records),
            "calibration_record_count": len(calibration_records),
            "competitor_label_count": int(competitor_y.sum()),
            "outcome_labeled_count": len(outcome_records),
            "observed_transition_count": len(transitions),
            "source_coverage": self.source_coverage,
            "family_coverage": self.family_coverage,
        })
        return self

    @staticmethod
    def _ensemble_probability(models: list[LogisticRegression], x) -> tuple[float, float]:
        if not models:
            return 0.5, 1.0
        values = np.asarray([float(model.predict_proba(x)[0, 1]) for model in models], dtype=float)
        return float(values.mean()), float(values.std(ddof=0))

    def predict_condition_score(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Return the pre-calibration signal used only to compare candidate conditions.

        The calibrated public-record probability can be flat across a narrow
        condition grid because isotonic calibration is piecewise constant.  A
        flat calibrated value is appropriate for route-level reporting but is
        not suitable for ordering nearby temperature/time candidates.  This
        method exposes the underlying ensemble signal without relabeling it as
        an experimental success probability.
        """
        if self.vectorizer is None:
            raise RuntimeError("fit must be called before predict_condition_score")
        x = self._compact_matrix(self.vectorizer.transform([route_features(record)]))
        competitor_signal, ensemble_spread = self._ensemble_probability(self.competitor_models, x)
        vocabulary = self.vectorizer.vocabulary_

        def coefficient_supported(feature: str) -> bool:
            index = vocabulary.get(feature)
            if index is None:
                return False
            return any(abs(float(model.coef_[0, index])) > 1e-12 for model in self.competitor_models)

        atmosphere_features = [name for name in vocabulary if str(name).startswith("atmosphere=")]
        return {
            "condition_target_score": float(max(1e-6, 1.0 - competitor_signal)),
            "condition_competitor_score": float(competitor_signal),
            "condition_margin": float(1.0 - 2.0 * competitor_signal),
            "ensemble_spread": float(ensemble_spread),
            "feature_support": {
                "temperature": coefficient_supported("temperature_mean_norm"),
                "duration": coefficient_supported("duration_mean_norm"),
                "atmosphere_category": bool(atmosphere_features),
            },
            "score_kind": "pre_calibration_candidate_ranking_signal",
            "calibrated_probability": False,
            "claim_boundary": "within-batch candidate ranking only; not an experimental success probability",
        }

    @staticmethod
    def _fit_calibrators(models: list[LogisticRegression], x, y: np.ndarray) -> list[IsotonicRegression | None]:
        calibrators: list[IsotonicRegression | None] = []
        if len(np.unique(y)) < 2 or len(y) < 20:
            return [None for _ in models]
        for model in models:
            raw = model.predict_proba(x)[:, 1]
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw, y)
            calibrators.append(calibrator)
        return calibrators

    @staticmethod
    def _ensemble_calibrated_probability(models: list[LogisticRegression], calibrators: list[IsotonicRegression | None], x) -> tuple[float, float]:
        if not models:
            return 0.5, 1.0
        values = []
        for index, model in enumerate(models):
            raw = float(model.predict_proba(x)[0, 1])
            calibrator = calibrators[index] if index < len(calibrators) else None
            values.append(float(calibrator.predict([raw])[0]) if calibrator is not None else raw)
        values = np.asarray(values, dtype=float)
        return float(values.mean()), float(values.std(ddof=0))

    def predict(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if self.vectorizer is None:
            raise RuntimeError("fit must be called before predict")
        x = self._compact_matrix(self.vectorizer.transform([route_features(record)]))
        competitor_probability, epistemic = self._ensemble_calibrated_probability(self.competitor_models, self.competitor_calibrators, x)
        outcome_probability, outcome_uncertainty = self._ensemble_calibrated_probability(self.outcome_models, self.outcome_calibrators, x)
        source = str((record.get("source") or {}).get("source_id") or "unknown")
        family = str(record.get("chemistry_family") or "unknown")
        source_coverage = float(self.source_coverage.get(source, 0.0))
        family_coverage = float(self.family_coverage.get(family, 0.0))
        evidence_risk = float(max(0.0, 1.0 - min(1.0, source_coverage + family_coverage)))
        reasons: list[str] = []
        if source not in self.source_coverage:
            reasons.append("unseen_source")
        if family not in self.family_coverage:
            reasons.append("unseen_family")
        if source_coverage < self.config.min_source_coverage / max(len(self.source_coverage), 1):
            reasons.append("low_source_coverage")
        if epistemic > self.config.max_epistemic_uncertainty:
            reasons.append("high_epistemic_uncertainty")
        if evidence_risk > self.config.max_evidence_risk:
            reasons.append("high_evidence_risk")
        target_probability = float(max(1e-6, 1.0 - competitor_probability))
        competitor_map = {"explicit_or_reported_competitor": competitor_probability}
        score = compute_path_score(target_probability, competitor_map, epistemic + outcome_uncertainty, evidence_risk)
        next_action = self.predict_next_action(record)
        return {
            "target_probability_proxy": target_probability,
            "competitor_probabilities": competitor_map,
            "outcome_impurity_probability_proxy": outcome_probability,
            "uncertainty_aleatoric_proxy": outcome_uncertainty,
            "uncertainty_epistemic_proxy": epistemic,
            "evidence_risk": evidence_risk,
            "path_score": score,
            "next_action_probability": next_action,
            "abstain": bool(reasons),
            "abstention_reasons": reasons,
            "claim_level": "S0" if reasons else "S1",
            "claim_boundary": "retrospective public-record proxy; no next-material-state or causal claim",
        }

    def predict_next_action(self, record: Mapping[str, Any]) -> dict[str, float]:
        """Predict the next reported operation for an observed route prefix."""
        if self.next_action_vectorizer is None or not self.next_action_models:
            return {}
        prefix_x = self._compact_matrix(self.next_action_vectorizer.transform([route_features(record)]))
        matrices = [model.predict_proba(prefix_x)[0] for model in self.next_action_models]
        mean = np.mean(np.asarray(matrices), axis=0)
        return {str(label): float(probability) for label, probability in zip(self.next_action_models[0].classes_, mean)}
