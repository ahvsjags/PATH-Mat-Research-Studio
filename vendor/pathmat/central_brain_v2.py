"""CentralBrainAgent v2 bindings for the PATH-Mat v3 transition world model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from pathmat.central_brain import BrainAction, BrainTask, CentralBrainAgent, DeterministicPathPlanner
from pathmat.ec_pathwm_agent_adapter import ECPathWMAdapter
from pathmat.memory import MemoryStore
from pathmat.v3_world_model import PathTransitionWorldModel


class TransitionPlanner(DeterministicPathPlanner):
    """Deterministic planner that requests a state transition prediction."""

    def propose(self, task: BrainTask, world_state, memory: MemoryStore, cycle: int) -> list[BrainAction]:
        if cycle > 0:
            return []
        ec_record = task.metadata.get("ec_pathwm_record")
        if isinstance(ec_record, Mapping):
            return [BrainAction(
                "ec_pathwm_score",
                "local_reversible_compute",
                {"record": dict(ec_record)},
                task.source_ids,
                "Score target/competitive-path risk with the evidence-conditioned EC-PathWM and preserve abstention boundaries.",
                True,
            )]
        record = task.metadata.get("transition_record")
        if not isinstance(record, Mapping):
            return [BrainAction("abstain", "local_reversible_compute", {"reason": "transition_record_missing"}, task.source_ids, "A source-linked transition record is required.", False)]
        return [BrainAction(
            "transition_predict",
            "local_reversible_compute",
            {
                "record": dict(record),
                "temperature_shift_C": float(task.metadata.get("temperature_shift_C", 0.0)),
                "duration_shift_h": float(task.metadata.get("duration_shift_h", 0.0)),
            },
            task.source_ids,
            "Predict the next reported phase/outcome and quantify information gain with the PATH-Mat v3 transition model.",
            True,
        )]


class CentralBrainAgentV2(CentralBrainAgent):
    """CentralBrain v2 with a learnable observational transition backend.

    The inherited controller still owns all state, policy gates, event hashes,
    checkpointing and replay.  The transition backend can only produce proxy
    predictions; it cannot enable physical actuation or upgrade claims.
    """

    def __init__(self, *, transition_model: PathTransitionWorldModel | None = None, ec_pathwm_adapter: ECPathWMAdapter | None = None, ec_pathwm_model: Any | None = None, simulation_enabled: bool = False, **kwargs: Any) -> None:
        kwargs.setdefault("planner", TransitionPlanner())
        super().__init__(**kwargs)
        self.transition_model = transition_model
        if ec_pathwm_adapter is not None and ec_pathwm_model is not None:
            raise ValueError("provide either ec_pathwm_adapter or ec_pathwm_model, not both")
        self.ec_pathwm_adapter = ec_pathwm_adapter or (ECPathWMAdapter(ec_pathwm_model) if ec_pathwm_model is not None else None)
        self.simulation_enabled = bool(simulation_enabled)

    @classmethod
    def from_model_path(cls, path: str | Path, **kwargs: Any) -> "CentralBrainAgentV2":
        return cls(transition_model=PathTransitionWorldModel.load(path), **kwargs)

    @classmethod
    def from_ec_pathwm_model_path(cls, path: str | Path, **kwargs: Any) -> "CentralBrainAgentV2":
        return cls(ec_pathwm_adapter=ECPathWMAdapter.load(path), **kwargs)

    @staticmethod
    def _ec_record(action: BrainAction) -> Mapping[str, Any]:
        record = action.payload.get("record")
        if not isinstance(record, Mapping):
            raise TypeError("ec_pathwm_record must be a mapping")
        return record

    def _ec_prediction(self, action: BrainAction) -> dict[str, Any]:
        if self.ec_pathwm_adapter is None:
            raise RuntimeError("ec_pathwm_adapter_not_loaded")
        return self.ec_pathwm_adapter.score(self._ec_record(action))

    def _score_action(self, task: BrainTask, action: BrainAction) -> dict[str, Any]:
        payload = super()._score_action(task, action)
        if action.name == "ec_pathwm_score":
            try:
                prediction = self._ec_prediction(action)
                payload["ec_pathwm"] = prediction
            except (TypeError, ValueError, RuntimeError) as exc:
                payload["ec_pathwm_error"] = f"{type(exc).__name__}:{exc}"
        return payload

    def _gate(self, task: BrainTask, action: BrainAction) -> tuple[bool, list[str]]:
        allowed, reasons = super()._gate(task, action)
        if self.use_safety and task.metadata.get("adversarial_input"):
            reasons.append("adversarial_input_blocked_by_safety")
        if action.name == "transition_predict" and self.transition_model is None:
            reasons.append("transition_model_not_loaded")
        if action.name == "ec_pathwm_score":
            if not self.use_world_model:
                reasons.append("ec_pathwm_disabled_by_no_world_model_ablation")
            elif self.ec_pathwm_adapter is None:
                reasons.append("ec_pathwm_adapter_not_loaded")
            else:
                try:
                    prediction = self._ec_prediction(action)
                    if prediction.get("abstain"):
                        reasons.append("ec_pathwm_abstention_required")
                        reasons.extend(f"ec_pathwm:{reason}" for reason in prediction.get("abstention_reasons", []))
                except (TypeError, ValueError, RuntimeError) as exc:
                    reasons.append(f"ec_pathwm_score_invalid:{type(exc).__name__}")
        if action.name == "simulation_request" and not self.simulation_enabled:
            reasons.append("simulation_disabled_by_v2_policy")
        return not reasons, reasons

    @staticmethod
    def _decision_policy(record: Mapping[str, Any], prediction: Any, information_gain: float) -> dict[str, Any]:
        """Rank reversible next steps from the current belief state.

        This is a policy output, not an execution request.  The selector makes
        uncertainty and information gain first-class decision variables while
        keeping physical actuation and production simulation outside the action
        space.  Missing XRD/Raman/SEM is treated as missing evidence, never as
        a negative observation.
        """
        belief = prediction.belief
        observed = {str(item).lower() for item in belief.observed_modalities}
        beta = 0.65
        gamma = 0.80
        eta = 0.10
        risk = min(1.0, 0.65 * float(belief.impurity_probability) + 0.35 * float(belief.ood_score))
        candidates: list[dict[str, Any]] = []

        def candidate(name: str, utility: float, cost: float, rationale: str, authority: str = "read_only_evidence") -> None:
            score = float(utility + beta * information_gain - gamma * risk - eta * cost)
            candidates.append({
                "action": name,
                "authority_level": authority,
                "score": score,
                "expected_utility": float(utility),
                "information_gain": float(information_gain),
                "risk": float(risk),
                "cost": float(cost),
                "rationale": rationale,
                "execution_status": "plan_only",
            })

        if "xrd" not in observed:
            candidate("request_external_xrd_evidence", 0.55, 0.20, "XRD is an unobserved modality and can distinguish target/competitor phase hypotheses.")
        if "raman" not in observed or "sem" not in observed:
            candidate("retrieve_missing_multimodal_evidence", 0.42, 0.15, "A missing modality limits belief-state reconstruction; retrieve only source-linked public evidence.")
        candidate("compare_competing_paths", 0.30, 0.05, "Compare precursor/process alternatives under the same proxy model and preserve counterfactual status.")
        candidate("continue_reversible_computation", 0.24, 0.10, "Run bounded local computation when the belief state is inside the model support.", "local_reversible_compute")
        abstain_utility = 0.25 + 0.75 * max(float(belief.uncertainty), float(belief.ood_score))
        candidate("abstain", abstain_utility, 0.0, "Uncertainty or OOD risk is high; do not force a material conclusion.", "local_reversible_compute")
        candidates.sort(key=lambda item: (-item["score"], item["action"]))
        return {
            "policy_version": "PATH-Mat-CentralBrain-belief-information-gain-v1",
            "objective": "expected_utility + beta*information_gain - gamma*risk - eta*cost",
            "weights": {"beta": beta, "gamma": gamma, "eta": eta},
            "belief_state": asdict(belief),
            "candidate_actions": candidates,
            "recommended_action": candidates[0]["action"],
            "physical_actuation": "disabled",
            "production_simulation": "disabled_by_default",
            "claim_status": "proxy",
        }

    def _execute(self, action: BrainAction) -> dict[str, Any]:
        if action.name == "ec_pathwm_score":
            try:
                prediction = self._ec_prediction(action)
            except (TypeError, ValueError, RuntimeError) as exc:
                return {"status": "external_dependency", "reason": f"ec_pathwm:{type(exc).__name__}"}
            return {
                "status": "completed",
                "ec_pathwm_prediction": prediction,
                "model": prediction.get("adapter_version", "EC-PathWM"),
                "claim_status": "proxy",
                "physical_actuation": False,
                "production_simulation": False,
            }
        if action.name == "transition_predict":
            if not self.use_world_model:
                return {
                    "status": "completed",
                    "prediction": None,
                    "information_gain_proxy": None,
                    "model": "ablation_no_world_model",
                    "claim_status": "proxy",
                    "simulation_enabled": self.simulation_enabled,
                }
            if self.transition_model is None:
                return {"status": "external_dependency", "reason": "transition_model_not_loaded"}
            record = dict(action.payload.get("record", {}))
            source_ids = tuple(action.source_ids) or ("pathmat_v3_training_records",)
            prediction = self.transition_model.predict_transition(record, source_ids=source_ids)
            information_gain = self.transition_model.information_gain_proxy(record)
            output: dict[str, Any] = {
                "status": "completed",
                "prediction": asdict(prediction),
                "information_gain_proxy": information_gain,
                "decision_policy": self._decision_policy(record, prediction, information_gain),
                "model": self.transition_model.version,
                "claim_status": "proxy",
                "simulation_enabled": self.simulation_enabled,
            }
            temperature_shift = float(action.payload.get("temperature_shift_C", 0.0))
            duration_shift = float(action.payload.get("duration_shift_h", 0.0))
            if temperature_shift or duration_shift:
                output["counterfactual_proxy"] = self.transition_model.counterfactual_proxy(record, temperature_shift_C=temperature_shift, duration_shift_h=duration_shift)
            return output
        return super()._execute(action)

    def _result(self, task, status, answer, state, events, next_cycle):
        result = super()._result(task, status, answer, state, events, next_cycle)
        result.metrics["world_model_backend"] = self.transition_model.version if self.transition_model else "unloaded"
        if self.transition_model is not None:
            fusion = (self.transition_model.capability_manifest().get("learned_e3_fusion") or {})
            selection = fusion.get("selection") or {}
            result.metrics["world_model_e3_fusion_weight"] = float(fusion.get("weight", 0.0))
            result.metrics["world_model_e3_fusion_selection_status"] = selection.get("status", "not_available")
            result.metrics["world_model_e3_operational_role"] = selection.get("operational_role", selection.get("e3_role", "not_available"))
        result.metrics["simulation_enabled"] = self.simulation_enabled
        result.metrics["claim_status"] = "proxy" if status == "completed" else "none"
        return result
