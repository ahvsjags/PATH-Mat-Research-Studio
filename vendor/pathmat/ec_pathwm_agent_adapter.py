"""Read-only binding from EC-PathWM to CentralBrainAgentV2.

The adapter deliberately exposes route scoring and abstention only. It does
not write world state, execute tools, request physical actuation, or promote
public-record proxy outputs to experimental claims.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Protocol

import joblib


class ECPathWMPredictor(Protocol):
    def predict(self, record: Mapping[str, Any]) -> dict[str, Any]:
        ...


class ECPathWMAdapter:
    """Stable, claim-bounded interface consumed by the central brain."""

    version = "PATH-Mat-EC-PathWM-agent-binding-v1"

    def __init__(self, model: ECPathWMPredictor):
        if not callable(getattr(model, "predict", None)):
            raise TypeError("EC-PathWM model must expose predict(record)")
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "ECPathWMAdapter":
        return cls(joblib.load(path))

    @staticmethod
    def _finite(value: Any, name: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"EC-PathWM output is not finite: {name}")
        return result

    def score(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise TypeError("EC-PathWM record must be a mapping")
        raw = self.model.predict(record)
        if not isinstance(raw, Mapping):
            raise TypeError("EC-PathWM predict must return a mapping")
        required = ("target_probability_proxy", "competitor_probabilities", "path_score", "evidence_risk", "abstain")
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"EC-PathWM output missing fields: {missing}")
        competitor = raw["competitor_probabilities"]
        if not isinstance(competitor, Mapping):
            raise TypeError("competitor_probabilities must be a mapping")
        output = dict(raw)
        output["target_probability_proxy"] = self._finite(raw["target_probability_proxy"], "target_probability_proxy")
        output["path_score"] = self._finite(raw["path_score"], "path_score")
        output["evidence_risk"] = self._finite(raw["evidence_risk"], "evidence_risk")
        output["competitor_probabilities"] = {
            str(key): self._finite(value, f"competitor_probabilities[{key}]")
            for key, value in competitor.items()
        }
        output["abstain"] = bool(raw["abstain"])
        output["adapter_version"] = self.version
        output["execution_authority"] = "read_only_evidence"
        output["physical_actuation"] = False
        output["production_simulation"] = False
        output["claim_status"] = "proxy"
        output["claim_boundary"] = str(raw.get("claim_boundary") or "retrospective public-record proxy; no next-material-state or causal claim")
        return output

    def score_condition(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Expose an explicitly non-probabilistic candidate-ranking signal."""
        if not isinstance(record, Mapping):
            raise TypeError("EC-PathWM condition record must be a mapping")
        scorer = getattr(self.model, "predict_condition_score", None)
        if not callable(scorer):
            raise AttributeError("EC-PathWM model does not expose predict_condition_score")
        raw = scorer(record)
        if not isinstance(raw, Mapping):
            raise TypeError("EC-PathWM condition score must be a mapping")
        output = dict(raw)
        for key in ("condition_target_score", "condition_competitor_score", "condition_margin", "ensemble_spread"):
            output[key] = self._finite(raw[key], key)
        output["adapter_version"] = self.version
        output["execution_authority"] = "read_only_evidence"
        output["physical_actuation"] = False
        output["claim_status"] = "ranking_signal"
        return output

    predict = score
