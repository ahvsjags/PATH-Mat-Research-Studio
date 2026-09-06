"""Learnable, calibrated proxy world model for PATH-Mat v3.

The module contains two dependency-light representation encoders: an analytic
E(3)-equivariant vector-message encoder for Cartesian structures and a
sequence-aware process-event encoder.  The current fitted backend learns a
calibrated classifier on their fixed summaries plus text features; it is not an
end-to-end equivariant neural model and keeps the proxy claim boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


NUMERIC_COLUMNS = [
    "temperature_C", "duration_h", "process_action_count", "precursor_count", "has_oxygen", "has_phosphorus",
    "process_event_count", "process_duration_h", "process_temperature_mean", "process_action_hash_mean", "process_order_mean",
    "structure_mean_atomic_number", "structure_atomic_number_std", "structure_mean_vector_norm", "structure_density_mean",
    "target_formula_atom_count", "target_formula_unique_elements", "target_formula_mean_atomic_number",
    "target_formula_atomic_number_std", "target_formula_heavy_atom_fraction", "target_formula_oxygen_fraction",
    "target_formula_transition_metal_fraction", "precursor_formula_atom_count", "precursor_formula_unique_elements",
    "precursor_formula_mean_atomic_number", "precursor_formula_atomic_number_std", "precursor_formula_heavy_atom_fraction",
    "precursor_formula_oxygen_fraction", "precursor_formula_transition_metal_fraction",
]


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


_ELEMENT_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_ATOMIC_NUMBERS = {symbol: number for number, symbol in enumerate(_ELEMENT_SYMBOLS, start=1)}
_TRANSITION_METAL_Z = set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81)) | set(range(104, 113))


def _formula_stats(value: Any) -> list[float]:
    """Return composition features that remain defined for unseen formula strings.

    These features deliberately avoid a fitted vocabulary: strict target/family/
    element OOD tests must have a numerical representation for chemistry that
    was not observed in the training text n-grams.
    """
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value or "")
    tokens = re.findall(r"([A-Z][a-z]?)(?:([0-9]+(?:\.[0-9]+)?))?", text)
    weighted_z: list[float] = []
    oxygen = 0.0
    transition = 0.0
    for symbol, count_text in tokens:
        z = float(_ATOMIC_NUMBERS.get(symbol, 0))
        count = float(count_text) if count_text else 1.0
        if z <= 0:
            continue
        weighted_z.extend([z] * max(1, min(int(round(count)), 100)))
        oxygen += count * (1.0 if symbol == "O" else 0.0)
        transition += count * (1.0 if int(z) in _TRANSITION_METAL_Z else 0.0)
    if not weighted_z:
        return [0.0] * 7
    values = np.asarray(weighted_z, dtype=float)
    total_atoms = float(values.size)
    return [
        total_atoms / 100.0,
        float(len(set(values))) / 20.0,
        float(values.mean()) / 118.0,
        float(values.std()) / 118.0,
        float(np.mean(values > 20.0)),
        float(oxygen / max(total_atoms, 1.0)),
        float(transition / max(total_atoms, 1.0)),
    ]


class StructureEquivariantEncoder:
    """Analytic E(3)-equivariant local vector-message encoder.

    The per-node vector messages rotate with the input Cartesian frame, while
    the pooled summaries used by the classifier are rotation/translation
    invariant.  This is a real geometric encoder, but not a learned neural
    equivariant architecture.
    """

    version = "PATH-Mat-E3-analytic-vector-message-v1"

    def __init__(self, *, cutoff_A: float = 6.0, radial_bins: int = 8) -> None:
        self.cutoff_A = float(cutoff_A)
        self.radial_centers = np.linspace(0.5, self.cutoff_A, int(radial_bins), dtype=float)
        self.radial_width = max(self.cutoff_A / max(2 * len(self.radial_centers), 1), 1e-3)

    @staticmethod
    def _atomic_number(value: Any) -> float:
        if isinstance(value, (int, float)):
            return _finite(value)
        token = str(value or "").strip()
        return float(_ATOMIC_NUMBERS.get(token, 0.0))

    def encode(self, structure: Mapping[str, Any] | None) -> dict[str, Any]:
        structure = dict(structure or {})
        positions = structure.get("positions", structure.get("cartesian_positions"))
        if positions is None:
            return {"missing": True, "vector_features": [], "invariant_features": [0.0] * (len(self.radial_centers) + 4), "encoder_version": self.version}
        coordinates = np.asarray(positions, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0 or not np.isfinite(coordinates).all():
            raise ValueError("structure positions must be a finite [N,3] array")
        raw_species = structure.get("atomic_numbers", structure.get("elements", [0.0] * len(coordinates)))
        atomic_numbers = np.asarray([self._atomic_number(item) for item in raw_species], dtype=float)
        if len(atomic_numbers) != len(coordinates):
            raise ValueError("structure species and positions have different lengths")
        vectors = np.zeros_like(coordinates)
        density = np.zeros(len(self.radial_centers), dtype=float)
        pair_count = 0
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                delta = coordinates[j] - coordinates[i]
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-12 or distance > self.cutoff_A:
                    continue
                direction = delta / distance
                radial = np.exp(-0.5 * ((distance - self.radial_centers) / self.radial_width) ** 2)
                weight = float(radial.sum()) * (0.5 + atomic_numbers[j] / 100.0)
                reverse_weight = float(radial.sum()) * (0.5 + atomic_numbers[i] / 100.0)
                vectors[i] += weight * direction
                vectors[j] -= reverse_weight * direction
                density += radial
                pair_count += 1
        norms = np.linalg.norm(vectors, axis=1) if len(vectors) else np.zeros(0, dtype=float)
        invariant_features = np.concatenate([
            density / max(len(coordinates), 1),
            np.asarray([
                float(np.mean(atomic_numbers)) / 100.0,
                float(np.std(atomic_numbers)) / 100.0,
                float(np.mean(norms)) if len(norms) else 0.0,
                float(pair_count) / max(len(coordinates), 1),
            ], dtype=float),
        ])
        return {
            "missing": False,
            "vector_features": vectors.tolist(),
            "invariant_features": invariant_features.tolist(),
            "encoder_version": self.version,
            "node_count": int(len(coordinates)),
            "pair_count": int(pair_count),
        }


class LearnedE3MessageEncoder:
    """Small dependency-free trainable E(3) message-passing backend.

    Scalar radial gates are learned jointly with a logistic structure head.
    Pair messages are scalar(direction-invariant) weights times relative
    vectors, so vector messages rotate covariantly and pooled features remain
    invariant.  The optimizer is intentionally simple and deterministic so
    the backend can be replayed on CPU without torch/e3nn.
    """

    version = "PATH-Mat-E3-learned-radial-message-v1"

    def __init__(self, *, cutoff_A: float = 6.0, radial_bins: int = 8) -> None:
        self.cutoff_A = float(cutoff_A)
        self.radial_centers = np.linspace(0.5, self.cutoff_A, int(radial_bins), dtype=float)
        self.radial_width = max(self.cutoff_A / max(2 * len(self.radial_centers), 1), 1e-3)
        self.theta = np.zeros(len(self.radial_centers), dtype=float)
        self.head_weight = np.zeros(len(self.radial_centers) + 5, dtype=float)
        self.head_bias = 0.0
        self.fitted = False
        self.training_manifest: dict[str, Any] = {}
        self.calibration_scale = 1.0
        self.calibration_bias = 0.0
        self.calibration_manifest: dict[str, Any] = {"status": "uncalibrated_raw_sigmoid"}

    @staticmethod
    def _atomic_number(value: Any) -> float:
        if isinstance(value, (int, float)):
            return _finite(value)
        return float(_ATOMIC_NUMBERS.get(str(value or "").strip(), 0.0))

    def _encode_one(self, structure: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
        structure = dict(structure or {})
        coordinates = np.asarray(structure.get("positions", structure.get("cartesian_positions", [])), dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0 or not np.isfinite(coordinates).all():
            return np.zeros(len(self.radial_centers) + 5, dtype=float), np.zeros((0, 3), dtype=float)
        raw_species = structure.get("atomic_numbers", structure.get("elements", [0.0] * len(coordinates)))
        atomic_numbers = np.asarray([self._atomic_number(item) for item in raw_species], dtype=float)
        if len(atomic_numbers) != len(coordinates):
            raise ValueError("structure species and positions have different lengths")
        if len(coordinates) > 128:
            coordinates = coordinates[:128]
            atomic_numbers = atomic_numbers[:128]
        vectors = np.zeros_like(coordinates)
        density = np.zeros(len(self.radial_centers), dtype=float)
        gates = np.exp(np.clip(self.theta, -6.0, 6.0))
        pair_count = 0
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                delta = coordinates[j] - coordinates[i]
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-12 or distance > self.cutoff_A:
                    continue
                direction = delta / distance
                radial = np.exp(-0.5 * ((distance - self.radial_centers) / self.radial_width) ** 2)
                scalar = float(np.dot(radial, gates)) * (0.5 + (atomic_numbers[i] + atomic_numbers[j]) / 200.0)
                vectors[i] += scalar * direction
                vectors[j] -= scalar * direction
                density += radial * gates
                pair_count += 1
        norms = np.linalg.norm(vectors, axis=1)
        invariant = np.concatenate([
            density / max(len(coordinates), 1),
            np.asarray([
                float(np.mean(norms)) if len(norms) else 0.0,
                float(np.mean(norms ** 2)) if len(norms) else 0.0,
                float(np.mean(atomic_numbers)) / 100.0,
                float(np.std(atomic_numbers)) / 100.0,
                float(pair_count) / max(len(coordinates), 1),
            ], dtype=float),
        ])
        return invariant, vectors

    def encode(self, structure: Mapping[str, Any] | None) -> dict[str, Any]:
        invariant, vectors = self._encode_one(structure)
        return {"invariant_features": invariant.tolist(), "vector_features": vectors.tolist(), "encoder_version": self.version}

    def _matrix(self, structures: list[Mapping[str, Any] | None]) -> np.ndarray:
        return np.asarray([self._encode_one(structure)[0] for structure in structures], dtype=float)

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))

    def _loss(self, features: np.ndarray, labels: np.ndarray) -> float:
        probabilities = self._sigmoid(features @ self.head_weight + self.head_bias)
        return float(-np.mean(labels * np.log(probabilities + 1e-8) + (1.0 - labels) * np.log(1.0 - probabilities + 1e-8)))

    def fit(self, structures: list[Mapping[str, Any] | None], labels: np.ndarray, *, epochs: int = 24, learning_rate: float = 0.04, max_samples: int = 512) -> "LearnedE3MessageEncoder":
        if len(structures) != len(labels) or len(structures) < 20 or len(np.unique(labels)) < 2:
            raise ValueError("at least twenty E3 training pairs with two classes are required")
        order = np.argsort(np.arange(len(structures)) * 1103515245 % 2147483647)
        order = order[:min(max_samples, len(order))]
        chosen_structures = [structures[int(index)] for index in order]
        chosen_labels = np.asarray(labels, dtype=float)[order]
        self.theta[:] = 0.0
        self.head_weight[:] = 0.0
        self.head_bias = 0.0
        self.calibration_scale = 1.0
        self.calibration_bias = 0.0
        self.calibration_manifest = {"status": "uncalibrated_raw_sigmoid"}
        epsilon = 1e-3
        for _ in range(int(epochs)):
            features = self._matrix(chosen_structures)
            probabilities = self._sigmoid(features @ self.head_weight + self.head_bias)
            error = probabilities - chosen_labels
            self.head_weight -= learning_rate * (features.T @ error / len(features) + 1e-4 * self.head_weight)
            self.head_bias -= learning_rate * float(error.mean())
            theta_gradient = np.zeros_like(self.theta)
            for index in range(len(self.theta)):
                self.theta[index] += epsilon
                plus = self._loss(self._matrix(chosen_structures), chosen_labels)
                self.theta[index] -= 2.0 * epsilon
                minus = self._loss(self._matrix(chosen_structures), chosen_labels)
                self.theta[index] += epsilon
                theta_gradient[index] = (plus - minus) / (2.0 * epsilon)
            self.theta -= learning_rate * np.clip(theta_gradient, -2.0, 2.0)
        self.fitted = True
        self.training_manifest = {"training_records": int(len(chosen_structures)), "epochs": int(epochs), "learning_rate": learning_rate, "loss": self._loss(self._matrix(chosen_structures), chosen_labels), "backend": self.version}
        return self

    def _raw_predict_proba(self, structures: list[Mapping[str, Any] | None]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("learned E3 backend has not been fitted")
        features = self._matrix(structures)
        return self._sigmoid(features @ self.head_weight + self.head_bias)

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped))

    def fit_calibration(self, structures: list[Mapping[str, Any] | None], labels: np.ndarray) -> "LearnedE3MessageEncoder":
        """Fit a Platt calibrator on a preblind calibration split only."""
        if not self.fitted or len(structures) != len(labels) or len(structures) < 20 or len(np.unique(labels)) < 2:
            raise ValueError("at least twenty fitted E3 calibration pairs with two classes are required")
        from sklearn.linear_model import LogisticRegression
        raw = self._raw_predict_proba(structures)
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=0)
        calibrator.fit(self._logit(raw).reshape(-1, 1), np.asarray(labels, dtype=int))
        self.calibration_scale = float(calibrator.coef_[0, 0])
        self.calibration_bias = float(calibrator.intercept_[0])
        self.calibration_manifest = {
            "status": "platt_calibrated_preblind_only",
            "records": int(len(structures)),
            "positive_records": int(np.asarray(labels, dtype=int).sum()),
            "scale": self.calibration_scale,
            "bias": self.calibration_bias,
            "labels_used_for_calibration": "preblind_calibration_split_only",
        }
        return self

    def predict_proba(self, structures: list[Mapping[str, Any] | None]) -> np.ndarray:
        raw = self._raw_predict_proba(structures)
        scale = float(getattr(self, "calibration_scale", 1.0))
        bias = float(getattr(self, "calibration_bias", 0.0))
        return self._sigmoid(scale * self._logit(raw) + bias)

    def equivariance_error(self, structure: Mapping[str, Any], rotation: np.ndarray, translation: np.ndarray | None = None) -> dict[str, float]:
        translation = np.zeros(3, dtype=float) if translation is None else np.asarray(translation, dtype=float)
        base = dict(structure)
        coordinates = np.asarray(base["positions"], dtype=float)
        transformed = dict(base)
        transformed["positions"] = (coordinates @ np.asarray(rotation, dtype=float).T) + translation
        invariant_a, vectors_a = self._encode_one(base)
        invariant_b, vectors_b = self._encode_one(transformed)
        rotated_vectors = vectors_a @ np.asarray(rotation, dtype=float).T
        return {"invariant_max_abs_error": float(np.max(np.abs(invariant_a - invariant_b))), "vector_max_abs_error": float(np.max(np.abs(rotated_vectors - vectors_b))) if len(vectors_a) else 0.0}


class ProcessEventEncoder:
    """Sequence-aware, deterministic process-event representation."""

    version = "PATH-Mat-process-event-sequence-v1"

    @staticmethod
    def _action_hash(action: str) -> float:
        import hashlib
        digest = hashlib.sha256(action.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def encode(self, record: Mapping[str, Any] | None) -> dict[str, Any]:
        record = dict(record or {})
        raw_events = record.get("process_events") or record.get("events") or record.get("process_actions") or []
        events: list[dict[str, Any]] = []
        for raw in raw_events if isinstance(raw_events, list) else [raw_events]:
            if isinstance(raw, Mapping):
                action = str(raw.get("action", raw.get("name", "unknown")))
                temperature = _finite(raw.get("temperature_C", record.get("temperature_C")))
                duration = _finite(raw.get("duration_h", raw.get("duration_s", 0.0)))
            else:
                action = str(raw)
                temperature = _finite(record.get("temperature_C"))
                duration = 0.0
            events.append({"action": action, "action_hash": self._action_hash(action), "temperature_C": temperature, "duration_h": duration})
        if not events:
            return {"event_matrix": [], "pooled_features": [0.0] * 5, "event_count": 0, "encoder_version": self.version}
        matrix = np.asarray([[item["action_hash"], (index + 1) / len(events), item["temperature_C"] / 1000.0, item["duration_h"], float(index > 0)] for index, item in enumerate(events)], dtype=float)
        pooled = np.asarray([
            float(len(events)),
            float(sum(item["duration_h"] for item in events)),
            float(np.mean([item["temperature_C"] for item in events])),
            float(np.mean([item["action_hash"] for item in events])),
            float(np.mean(matrix[:, 1])),
        ], dtype=float)
        return {"event_matrix": matrix.tolist(), "pooled_features": pooled.tolist(), "event_count": len(events), "encoder_version": self.version}


@dataclass(frozen=True)
class BeliefState:
    phase_fractions: dict[str, float]
    impurity_probability: float
    uncertainty: float
    ood_score: float
    observed_modalities: tuple[str, ...] = ()
    history: tuple[str, ...] = ()

    def validate(self) -> None:
        values = np.asarray(list(self.phase_fractions.values()) + [self.impurity_probability, self.uncertainty, self.ood_score], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("belief state contains non-finite values")
        if any(value < 0 for value in self.phase_fractions.values()) or abs(sum(self.phase_fractions.values()) - 1.0) > 1e-8:
            raise ValueError("phase fractions must be a non-negative simplex")
        if not 0 <= self.impurity_probability <= 1 or not 0 <= self.uncertainty <= 1 or not 0 <= self.ood_score <= 1:
            raise ValueError("belief probabilities must lie in [0, 1]")


@dataclass(frozen=True)
class TransitionPrediction:
    belief: BeliefState
    predicted_outcome: str
    source_ids: tuple[str, ...]
    model_version: str
    claim_status: str = "proxy"
    falsification_tests: tuple[str, ...] = ()
    predicted_competitor_phases: dict[str, float] = field(default_factory=dict)
    failure_probability: float = 0.0

    def validate(self) -> None:
        self.belief.validate()
        if self.claim_status not in {"proxy", "hypothesis", "protocol", "verified"}:
            raise ValueError("invalid claim status")
        if not self.model_version or not self.source_ids:
            raise ValueError("model_version and source_ids are required")
        if not 0.0 <= self.failure_probability <= 1.0 or any(not 0.0 <= value <= 1.0 for value in self.predicted_competitor_phases.values()):
            raise ValueError("phase and failure probabilities must lie in [0, 1]")


class PathTransitionWorldModel:
    """Hybrid observational world model with a stable v3 interface."""

    version = "PATH-Mat-v3-transition-proxy-v2"

    def __init__(self, feature_mode: str = "standard") -> None:
        allowed_modes = {"standard", "formula_numeric", "element_numeric", "numeric_only"}
        if feature_mode not in allowed_modes:
            raise ValueError(f"feature_mode must be one of {sorted(allowed_modes)}")
        self.feature_mode = feature_mode
        if feature_mode == "element_numeric":
            self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[A-Z][a-z]?\b", lowercase=False, ngram_range=(1, 2), min_df=1, dtype=np.float32)
        else:
            self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3, max_features=8000, dtype=np.float32)
        base_classifier = LogisticRegression(max_iter=300, class_weight=None, solver="liblinear", random_state=0)
        self.classifier = CalibratedClassifierCV(estimator=base_classifier, method="sigmoid", cv=3, n_jobs=1)
        self.structure_encoder = StructureEquivariantEncoder()
        self.learned_e3_backend: LearnedE3MessageEncoder | None = None
        # The learned structure branch is an auxiliary predictor until a
        # preblind calibration split demonstrates that it improves the fused
        # risk score.  Never default to an arbitrary non-zero influence.
        self.e3_fusion_weight = 0.0
        self.e3_fusion_manifest: dict[str, Any] = {
            "status": "not_selected_no_preblind_fusion_calibration",
            "selected_weight": 0.0,
        }
        self.process_encoder = ProcessEventEncoder()
        self.phase_classifier: OneVsRestClassifier | None = None
        self.phase_classes: list[str] = []
        self.numeric_medians: dict[str, float] = {}
        self.numeric_scales: dict[str, float] = {}
        self._feature_numeric_columns = list(NUMERIC_COLUMNS)
        self.numeric_feature_weight = 3.0
        self.known_targets: set[str] = set()
        self.fitted = False

    @staticmethod
    def _text(frame: pd.DataFrame) -> pd.Series:
        # Do not include observed outcome/phase labels in predictive features.
        # The phase field is reserved for supervision of the optional phase
        # head; including it here would leak the target into the transition
        # classifier and make OOD metrics invalid.
        columns = ["target_formula", "precursor_formulas", "process_actions", "atmosphere"]
        text = pd.Series("", index=frame.index, dtype="object")
        for column in columns:
            if column in frame:
                text = text + " " + frame[column].fillna("").astype(str)
        return text

    @staticmethod
    def _formula_text(frame: pd.DataFrame) -> pd.Series:
        """Formula-only text view for domain-shift robustness candidates."""
        text = pd.Series("", index=frame.index, dtype="object")
        for column in ("target_formula", "precursor_formulas"):
            if column in frame:
                text = text + " " + frame[column].fillna("").astype(str)
        return text

    def _numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=frame.index)
        empty = pd.Series(index=frame.index, dtype=object)
        output["temperature_C"] = pd.to_numeric(frame.get("temperature_C", empty), errors="coerce")
        output["duration_h"] = pd.to_numeric(frame.get("duration_h", empty), errors="coerce")
        action_series = frame.get("process_actions", empty)
        output["process_action_count"] = action_series.apply(lambda value: len(value) if isinstance(value, list) else 0)
        output["precursor_count"] = frame.get("precursor_formulas", empty).apply(lambda value: len(value) if isinstance(value, list) else 0)
        output["has_oxygen"] = frame.get("target_formula", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str).str.contains("O").astype(float)
        output["has_phosphorus"] = frame.get("target_formula", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str).str.contains("P").astype(float)
        process_rows = [self.process_encoder.encode(row).get("pooled_features", [0.0] * 5) for row in frame.to_dict(orient="records")]
        process_matrix = np.asarray(process_rows, dtype=float) if process_rows else np.zeros((0, 5), dtype=float)
        output["process_event_count"] = process_matrix[:, 0] if len(process_matrix) else 0.0
        output["process_duration_h"] = process_matrix[:, 1] if len(process_matrix) else 0.0
        output["process_temperature_mean"] = process_matrix[:, 2] if len(process_matrix) else 0.0
        output["process_action_hash_mean"] = process_matrix[:, 3] if len(process_matrix) else 0.0
        output["process_order_mean"] = process_matrix[:, 4] if len(process_matrix) else 0.0
        structure_rows = [self.structure_encoder.encode(row.get("structure") if isinstance(row.get("structure"), Mapping) else {}).get("invariant_features", [0.0] * (len(self.structure_encoder.radial_centers) + 4)) for row in frame.to_dict(orient="records")]
        structure_matrix = np.asarray(structure_rows, dtype=float) if structure_rows else np.zeros((0, len(self.structure_encoder.radial_centers) + 4), dtype=float)
        output["structure_mean_atomic_number"] = structure_matrix[:, -4] if len(structure_matrix) else 0.0
        output["structure_atomic_number_std"] = structure_matrix[:, -3] if len(structure_matrix) else 0.0
        output["structure_mean_vector_norm"] = structure_matrix[:, -2] if len(structure_matrix) else 0.0
        output["structure_density_mean"] = structure_matrix[:, -1] if len(structure_matrix) else 0.0
        target_stats = np.asarray(
            [_formula_stats(value) for value in frame.get("target_formula", empty)],
            dtype=float,
        )
        precursor_stats = np.asarray(
            [_formula_stats(value) for value in frame.get("precursor_formulas", empty)],
            dtype=float,
        )
        for prefix, matrix in (("target_formula", target_stats), ("precursor_formula", precursor_stats)):
            output[f"{prefix}_atom_count"] = matrix[:, 0] if len(matrix) else 0.0
            output[f"{prefix}_unique_elements"] = matrix[:, 1] if len(matrix) else 0.0
            output[f"{prefix}_mean_atomic_number"] = matrix[:, 2] if len(matrix) else 0.0
            output[f"{prefix}_atomic_number_std"] = matrix[:, 3] if len(matrix) else 0.0
            output[f"{prefix}_heavy_atom_fraction"] = matrix[:, 4] if len(matrix) else 0.0
            output[f"{prefix}_oxygen_fraction"] = matrix[:, 5] if len(matrix) else 0.0
            output[f"{prefix}_transition_metal_fraction"] = matrix[:, 6] if len(matrix) else 0.0
        return output

    def _features(self, frame: pd.DataFrame, *, fit_numeric: bool = False):
        text_source = self._formula_text(frame) if self.feature_mode in {"formula_numeric", "element_numeric"} else self._text(frame)
        numeric = self._numeric(frame)
        feature_numeric_columns = list(getattr(self, "_feature_numeric_columns", NUMERIC_COLUMNS))
        if fit_numeric:
            self.numeric_medians = {column: float(numeric[column].median()) if column in numeric and numeric[column].notna().any() else 0.0 for column in feature_numeric_columns}
            filled = numeric.reindex(columns=feature_numeric_columns).fillna(self.numeric_medians).fillna(0.0)
            self.numeric_scales = {
                column: max(float(filled[column].std(ddof=0)), 0.05)
                for column in feature_numeric_columns
            }
        numeric = numeric.reindex(columns=feature_numeric_columns).fillna(self.numeric_medians).fillna(0.0)
        for column in feature_numeric_columns:
            median = self.numeric_medians.get(column, 0.0)
            scale = self.numeric_scales.get(column, 1.0)
            numeric[column] = (numeric[column] - median) / scale
        numeric = numeric.clip(-8.0, 8.0) * self.numeric_feature_weight
        numeric_matrix = csr_matrix(numeric.to_numpy(dtype=float))
        if self.feature_mode == "numeric_only":
            return numeric_matrix
        text_matrix = self.vectorizer.transform(text_source)
        return hstack([text_matrix, numeric_matrix], format="csr")

    @staticmethod
    def _labels(frame: pd.DataFrame) -> np.ndarray:
        outcome = frame.get("reported_outcome", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)
        impurity = outcome.str.contains("impurity|failure|no_product", case=False, regex=True)
        explicit_impurity = frame.get("reported_impurity_phases", pd.Series(index=frame.index, dtype=object)).apply(lambda value: bool(value) if isinstance(value, list) else False)
        return (impurity | explicit_impurity).astype(int).to_numpy()

    def fit(self, frame: pd.DataFrame) -> "PathTransitionWorldModel":
        if len(frame) < 10:
            raise ValueError("at least ten historical records are required")
        labels = self._labels(frame)
        if len(np.unique(labels)) < 2:
            raise ValueError("training data must contain both reported outcome classes")
        if self.feature_mode != "numeric_only":
            text_source = self._formula_text(frame) if self.feature_mode in {"formula_numeric", "element_numeric"} else self._text(frame)
            self.vectorizer.fit(text_source)
        self._features(frame, fit_numeric=True)
        self.classifier.fit(self._features(frame), labels)
        phase_counter: dict[str, int] = {}
        phase_lists = frame.get("reported_impurity_phases", pd.Series(index=frame.index, dtype=object)).apply(lambda value: [str(item) for item in value] if isinstance(value, list) else [])
        for phase_list in phase_lists:
            for phase in set(phase_list):
                phase_counter[phase] = phase_counter.get(phase, 0) + 1
        self.phase_classes = [phase for phase, count in sorted(phase_counter.items(), key=lambda item: (-item[1], item[0]))[:32] if count >= 2]
        if self.phase_classes:
            phase_labels = np.asarray([[1 if phase in phase_list else 0 for phase in self.phase_classes] for phase_list in phase_lists], dtype=int)
            self.phase_classifier = OneVsRestClassifier(LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced", random_state=0))
            self.phase_classifier.fit(self._features(frame), phase_labels)
        self.known_targets = set(frame.get("target_formula", pd.Series(dtype=str)).dropna().astype(str))
        self.fitted = True
        return self

    def capability_manifest(self) -> dict[str, Any]:
        learned_e3 = self.learned_e3_backend is not None and self.learned_e3_backend.fitted
        return {
            "feature_mode": self.feature_mode,
            "structure_encoder": self.structure_encoder.version,
            "process_event_encoder": self.process_encoder.version,
            "phase_prediction": "one_vs_rest_top32" if self.phase_classifier is not None else "unavailable_no_phase_labels",
            "learned_backend": "jointly_trained_e3_radial_message_structure_head_plus_text_proxy" if learned_e3 else "calibrated_logistic_on_text_and_fixed_geometry_event_summaries",
            "end_to_end_equivariant_training": learned_e3,
            "learned_e3_backend": {**self.learned_e3_backend.training_manifest, "calibration": getattr(self.learned_e3_backend, "calibration_manifest", {"status": "legacy_uncalibrated"})} if learned_e3 else None,
            "learned_e3_fusion": {
                "weight": float(getattr(self, "e3_fusion_weight", 0.0)),
                "selection": dict(getattr(self, "e3_fusion_manifest", {})),
            },
            "claim_status": "proxy",
        }

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("world model has not been fitted")

    def predict_impurity_probability(self, frame: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        base = self.classifier.predict_proba(self._features(frame))[:, 1]
        if self.learned_e3_backend is None or not self.learned_e3_backend.fitted or "structure" not in frame:
            return base
        structures = [value if isinstance(value, Mapping) else None for value in frame["structure"].tolist()]
        present = np.asarray([bool(value and value.get("positions", value.get("cartesian_positions"))) for value in structures], dtype=bool)
        if not present.any():
            return base
        e3 = self.learned_e3_backend.predict_proba(structures)
        fusion_weight = float(np.clip(_finite(getattr(self, "e3_fusion_weight", 0.0)), 0.0, 1.0))
        fused = (1.0 - fusion_weight) * base + fusion_weight * e3
        return np.where(present, fused, base)

    def predict_transition(self, row: Mapping[str, Any], *, source_ids: tuple[str, ...] = ("pathmat_v3_training_records",)) -> TransitionPrediction:
        self._require_fitted()
        frame = pd.DataFrame([dict(row)])
        probability = float(self.predict_impurity_probability(frame)[0])
        target = str(row.get("target_formula") or "unknown_target")
        ood = 0.85 if target not in self.known_targets else 0.15
        uncertainty = float(min(1.0, 2.0 * probability * (1.0 - probability) + 0.25 * ood))
        belief = BeliefState(
            phase_fractions={"target_proxy": 1.0 - probability, "reported_impurity_proxy": probability},
            impurity_probability=probability,
            uncertainty=uncertainty,
            ood_score=ood,
            observed_modalities=tuple(sorted(str(item) for item in row.get("observed_modalities", ()) or ())),
            history=("observational_transition_proxy",),
        )
        belief.validate()
        phase_probabilities: dict[str, float] = {}
        if self.phase_classifier is not None and self.phase_classes:
            raw_phase_probabilities = np.asarray(self.phase_classifier.predict_proba(self._features(frame)))
            if raw_phase_probabilities.ndim == 1:
                raw_phase_probabilities = raw_phase_probabilities[:, None]
            phase_probabilities = {phase: float(value) for phase, value in zip(self.phase_classes, raw_phase_probabilities[0]) if float(value) >= 0.05}
        prediction = TransitionPrediction(
            belief=belief,
            predicted_outcome="reported_impurity" if probability >= 0.5 else "reported_synthesis",
            source_ids=source_ids,
            model_version=self.version,
            falsification_tests=("evaluate_on_time_split", "evaluate_on_family_split", "do_not_claim_causal_intervention"),
            predicted_competitor_phases=phase_probabilities,
            failure_probability=probability,
        )
        prediction.validate()
        return prediction

    def counterfactual_proxy(self, row: Mapping[str, Any], *, temperature_shift_C: float = 0.0, duration_shift_h: float = 0.0) -> dict[str, Any]:
        base = dict(row)
        changed = dict(row)
        if changed.get("temperature_C") is not None:
            changed["temperature_C"] = _finite(changed.get("temperature_C")) + temperature_shift_C
        if changed.get("duration_h") is not None:
            changed["duration_h"] = _finite(changed.get("duration_h")) + duration_shift_h
        before = self.predict_transition(base)
        after = self.predict_transition(changed)
        return {
            "base_impurity_probability": before.belief.impurity_probability,
            "counterfactual_impurity_probability": after.belief.impurity_probability,
            "delta": after.belief.impurity_probability - before.belief.impurity_probability,
            "intervention_status": "associative_proxy_not_causal",
            "changed_fields": {"temperature_C": temperature_shift_C, "duration_h": duration_shift_h},
            "model_version": self.version,
        }

    def information_gain_proxy(self, row: Mapping[str, Any]) -> float:
        prediction = self.predict_transition(row)
        entropy = -(prediction.belief.impurity_probability * np.log2(max(prediction.belief.impurity_probability, 1e-12)) + (1 - prediction.belief.impurity_probability) * np.log2(max(1 - prediction.belief.impurity_probability, 1e-12)))
        return float(min(1.0, entropy * (0.5 + 0.5 * prediction.belief.ood_score)))

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> "PathTransitionWorldModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("serialized object is not a PathTransitionWorldModel")
        # Older transition artifacts were serialized with a scikit-learn
        # LogisticRegression whose fitted estimator no longer carries the
        # ``multi_class`` attribute in the runtime version used by Ibex.
        # Restore the effective default at load time so inference remains
        # portable without mutating the artifact on disk.
        phase_classifier = getattr(model, "phase_classifier", None)
        for estimator in getattr(phase_classifier, "estimators_", ()) or ():
            if not hasattr(estimator, "multi_class"):
                estimator.multi_class = "auto"
        # Artifacts created before feature-mode metadata was introduced used
        # the original mixed text + numeric feature view.
        if not hasattr(model, "feature_mode"):
            model.feature_mode = "standard"
        if not hasattr(model, "_feature_numeric_columns"):
            # The first transition artifacts stored the fitted numeric schema
            # only in the median dictionary. Reusing that insertion order
            # preserves the original sparse feature positions, while missing
            # legacy fields are deterministically imputed by the stored median.
            model._feature_numeric_columns = list(getattr(model, "numeric_medians", {}) or NUMERIC_COLUMNS)
        # Artifacts predating the preblind fusion gate used an arbitrary 0.5
        # E(3) mixture.  On load they fail closed to the calibrated base score
        # until a reproducible selection manifest is attached by retraining.
        if not hasattr(model, "e3_fusion_weight"):
            model.e3_fusion_weight = 0.0
        if not hasattr(model, "e3_fusion_manifest"):
            model.e3_fusion_manifest = {
                "status": "legacy_artifact_fails_closed_until_preblind_fusion_calibration",
                "selected_weight": float(model.e3_fusion_weight),
            }
        model._require_fitted()
        return model
