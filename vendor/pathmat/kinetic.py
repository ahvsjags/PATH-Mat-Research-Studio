from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from pathmat.core import PhaseState


NUMERIC_COLUMNS = ["temperature_C_min", "temperature_C_max", "duration_value_min", "duration_value_max", "process_step_count"]


class MinimalKineticWorldModel:
    """Observational binary outcome model with an explicit phase-simplex proxy.

    The label is reported-impurity vs no-reported-impurity, not quantitative phase purity.
    Intervention predictions are counterfactual proxies until randomized data are available.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=12000)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.numeric_medians: dict[str, float] = {}

    @staticmethod
    def _text(frame: pd.DataFrame) -> pd.Series:
        return frame["target_formula"].fillna("").astype(str) + " " + frame["precursor_formulas"].fillna("").astype(str) + " " + frame["chemistry_family"].fillna("").astype(str) + " " + frame.get("process_action_types", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)

    def _features(self, frame: pd.DataFrame, fit_numeric: bool = False):
        text_matrix = self.vectorizer.transform(self._text(frame))
        numeric = frame.reindex(columns=NUMERIC_COLUMNS).apply(pd.to_numeric, errors="coerce")
        if fit_numeric:
            self.numeric_medians = {column: float(numeric[column].median()) if numeric[column].notna().any() else 0.0 for column in NUMERIC_COLUMNS}
        numeric = numeric.fillna(self.numeric_medians).fillna(0.0)
        return hstack([text_matrix, csr_matrix(numeric.to_numpy(dtype=float))], format="csr")

    def fit(self, frame: pd.DataFrame, labels: np.ndarray) -> "MinimalKineticWorldModel":
        self.vectorizer.fit(self._text(frame))
        features = self._features(frame, fit_numeric=True)
        self.model.fit(features, labels)
        return self

    def predict_reported_impurity(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._features(frame))[:, 1]

    def predict_state(self, row: dict[str, Any]) -> PhaseState:
        frame = pd.DataFrame([row])
        probability = float(self.predict_reported_impurity(frame)[0])
        state = PhaseState(conserved_elements={"record": 1.0}, phase_fractions={"target_proxy": 1.0 - probability, "reported_impurity_proxy": probability}, time_step=0, history=[])
        state.validate_simplex()
        return state

    @staticmethod
    def intervention_row(row: dict[str, Any], temperature_shift_C: float = 0.0, duration_shift_h: float = 0.0) -> dict[str, Any]:
        changed = dict(row)
        for column in ("temperature_C_min", "temperature_C_max"):
            if changed.get(column) is not None:
                changed[column] = float(changed[column]) + temperature_shift_C
        for column in ("duration_value_min", "duration_value_max"):
            if changed.get(column) is not None:
                changed[column] = float(changed[column]) + duration_shift_h
        return changed

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
