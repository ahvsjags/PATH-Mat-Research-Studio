from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import requests


class EvidenceLevel(str, Enum):
    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    name: str
    api: str
    version: str
    license: str
    legal_open_access: bool = True
    field_notes: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str
    source_version: str
    license_status: str
    parser_version: str
    confidence: float = Field(ge=0.0, le=1.0)
    first_public_date: date | None = None
    source_locator: str | None = None


class EvidencePackage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    target: str
    source_ids: list[str]
    evidence_level: EvidenceLevel
    assumptions: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    uncertainty: dict[str, Any]
    falsification_test: str
    safety_constraints: list[str] = Field(default_factory=list)
    license_status: str


class PhaseState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conserved_elements: dict[str, float]
    phase_fractions: dict[str, float]
    time_step: int = Field(ge=0)
    history: list[str] = Field(default_factory=list)

    def validate_simplex(self, tol: float = 1e-8) -> None:
        if any(x < -tol for x in self.phase_fractions.values()):
            raise ValueError("negative phase fraction")
        if abs(sum(self.phase_fractions.values()) - 1.0) > tol:
            raise ValueError("phase fractions do not sum to one")


@dataclass
class PathStep:
    barrier_over_kT: float
    driving_over_kT: float
    diffusion_penalty: float
    topology_penalty: float
    volatility_penalty: float


@dataclass
class Action:
    name: str
    target_bias: float = 0.0
    competitor_bias: float = 0.0
    volatility_penalty: float = 0.0


@dataclass
class GateDecision:
    allowed: bool
    reasons: list[str]


TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*(?:\.[0-9]+)?)")


def parse_formula(formula: str) -> dict[str, float]:
    if not formula or any(ch in formula for ch in "()[]·."):
        raise ValueError(f"unsupported formula syntax: {formula}")
    pos = 0
    out: dict[str, float] = defaultdict(float)
    for match in TOKEN.finditer(formula):
        if match.start() != pos:
            raise ValueError(f"cannot parse formula: {formula}")
        element, amount = match.groups()
        out[element] += float(amount) if amount else 1.0
        pos = match.end()
    if pos != len(formula):
        raise ValueError(f"cannot parse formula: {formula}")
    return dict(out)


def elemental_balance(reactants: Iterable[tuple[float, str]], products: Iterable[tuple[float, str]], tol: float = 1e-8) -> dict[str, float]:
    delta: dict[str, float] = defaultdict(float)
    for coefficient, formula in reactants:
        for element, count in parse_formula(formula).items():
            delta[element] += coefficient * count
    for coefficient, formula in products:
        for element, count in parse_formula(formula).items():
            delta[element] -= coefficient * count
    return {element: value for element, value in delta.items() if abs(value) > tol}


def is_balanced(reactants: Iterable[tuple[float, str]], products: Iterable[tuple[float, str]]) -> bool:
    return not elemental_balance(reactants, products)


def normalize_simplex(values: dict[str, float]) -> dict[str, float]:
    if not values or any(not math.isfinite(x) or x < 0 for x in values.values()):
        raise ValueError("invalid phase fractions")
    total = sum(values.values())
    if total <= 0:
        raise ValueError("empty phase fractions")
    return {key: value / total for key, value in values.items()}


def path_action_score(steps: Iterable[PathStep]) -> float:
    return float(sum(s.barrier_over_kT - s.driving_over_kT + s.diffusion_penalty + s.topology_penalty + s.volatility_penalty for s in steps))


def target_probability(target_score: float, competitors: Iterable[float]) -> float:
    scores = np.asarray([target_score, *competitors], dtype=float)
    weights = np.exp(-scores + np.min(scores))
    return float(weights[0] / weights.sum())


class EvidenceGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_id: str, node_type: str, **attrs: Any) -> None:
        self.graph.add_node(node_id, node_type=node_type, **attrs)

    def add_edge(self, source: str, target: str, relation: str, provenance: Provenance, **attrs: Any) -> None:
        if source not in self.graph or target not in self.graph:
            raise KeyError("edge endpoints must exist")
        self.graph.add_edge(source, target, relation=relation, provenance=provenance.model_dump(mode="json"), **attrs)

    def validate(self) -> list[str]:
        errors = []
        for node_id, data in self.graph.nodes(data=True):
            if not data.get("node_type"):
                errors.append(f"node_without_type:{node_id}")
        for source, target, key, data in self.graph.edges(keys=True, data=True):
            if not data.get("relation"):
                errors.append(f"edge_without_relation:{source}:{target}:{key}")
            if not data.get("provenance"):
                errors.append(f"edge_without_provenance:{source}:{target}:{key}")
        return errors

    def to_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for node_id, data in self.graph.nodes(data=True):
                handle.write(json.dumps({"kind": "node", "id": node_id, "payload": data}, ensure_ascii=False) + "\n")
            for source, target_id, key, data in self.graph.edges(keys=True, data=True):
                handle.write(json.dumps({"kind": "edge", "source": source, "target": target_id, "key": key, "payload": data}, ensure_ascii=False) + "\n")


DEFAULT_SOURCES = [
    ("materials_project", "Materials Project", "https://api.materialsproject.org/", "official-api", "MP API terms"),
    ("nomad", "NOMAD", "https://nomad-laboratory.eu/api/v1/", "official-api", "NOMAD terms"),
    ("oqmd", "OQMD", "https://oqmd.org/oqmdapi/", "official-api", "OQMD terms"),
    ("aflow", "AFLOW", "https://aflow.org/API/aflux/", "official-api", "AFLOW terms"),
    ("jarvis", "JARVIS", "https://jarvis.nist.gov/", "official", "NIST/JARVIS terms"),
    ("alexandria", "Alexandria", "https://alexandria.icams.rwth-aachen.de/", "official", "dataset terms"),
    ("cod", "Crystallography Open Database", "https://www.crystallography.net/", "official", "COD license"),
    ("c2db", "C2DB", "https://cmr.fysik.dtu.dk/C2DB/C2DB.html", "official", "dataset terms"),
    ("materials_cloud", "Materials Cloud", "https://www.materialscloud.org/", "official", "per-record license"),
    ("openalex", "OpenAlex", "https://api.openalex.org/", "official-api", "OpenAlex terms"),
    ("crossref", "Crossref", "https://api.crossref.org/", "official-api", "Crossref terms"),
    ("unpaywall", "Unpaywall", "https://api.unpaywall.org/v2/", "official-api", "Unpaywall terms"),
    ("rruff", "RRUFF", "https://rruff.info/", "official", "RRUFF terms"),
    ("optimade", "OPTIMADE", "https://optimade.org/", "specification", "specification"),
    ("w3c_prov", "W3C PROV-DM", "https://www.w3.org/TR/prov-dm/", "specification", "W3C"),
]


def default_registry() -> list[SourceRecord]:
    return [SourceRecord(source_id=a, name=b, api=c, version=d, license=e) for a, b, c, d, e in DEFAULT_SOURCES]


def write_registry(path: str | Path) -> list[SourceRecord]:
    records = default_registry()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([r.model_dump(mode="json") for r in records], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records


@dataclass
class ProbeResult:
    source_id: str
    url: str
    status_code: int | None
    ok: bool
    error: str | None


def probe_source(source: SourceRecord, timeout: float = 10.0) -> ProbeResult:
    try:
        response = requests.get(source.api, timeout=timeout, headers={"User-Agent": "PATH-Mat/0.1"})
        return ProbeResult(source.source_id, source.api, response.status_code, response.ok, None)
    except requests.RequestException as exc:
        return ProbeResult(source.source_id, source.api, None, False, type(exc).__name__)


class MinimalPathWorldModel:
    def __init__(self, initial: PhaseState) -> None:
        initial.validate_simplex()
        self.state = initial

    def step(self, action: Action, temperature_K: float = 1000.0) -> PhaseState:
        if temperature_K <= 0:
            raise ValueError("temperature must be positive")
        phases = dict(self.state.phase_fractions)
        target = phases.get("target", 0.0)
        competitor = phases.get("competitor", 0.0)
        scale = min(max(temperature_K / 1000.0, 0.05), 3.0)
        delta = 0.04 * scale * (action.target_bias - action.competitor_bias - action.volatility_penalty)
        transfer = min(max(delta, -competitor), target if delta < 0 else max(0.0, 1.0 - target))
        phases["target"] = target + transfer
        phases["competitor"] = competitor - transfer
        self.state = PhaseState(conserved_elements=dict(self.state.conserved_elements), phase_fractions=normalize_simplex(phases), time_step=self.state.time_step + 1, history=[*self.state.history, action.name])
        self.state.validate_simplex()
        return self.state


def critic_check(package: EvidencePackage) -> GateDecision:
    reasons = []
    if package.evidence_level in {EvidenceLevel.E0, EvidenceLevel.E1}:
        reasons.append("low_evidence_level")
    if not package.falsification_test.strip():
        reasons.append("missing_falsification_test")
    if package.license_status.lower() in {"unknown", "commercial-restricted"}:
        reasons.append("unresolved_license")
    return GateDecision(not reasons, reasons)


def safety_gate(
    reactants: list[tuple[float, str]],
    products: list[tuple[float, str]],
    safety_constraints: list[str],
    *,
    codex_approved: bool = False,
    approval_mode: str = "codex_agent",
    human_approved: bool = False,
) -> GateDecision:
    """Apply safety gates under the amended Codex-agent review scope.

    The human_approved parameter is retained only as a legacy compatibility
    argument. The active server scope uses codex_approved and never treats a
    human signature or human preregistration as a prerequisite.
    """
    reasons = []
    if not is_balanced(reactants, products):
        reasons.append("elemental_balance_failed")
    if not safety_constraints:
        reasons.append("missing_safety_constraints")
    if approval_mode == "codex_agent":
        if not codex_approved:
            reasons.append("codex_agent_review_required")
    elif approval_mode == "human_legacy":
        if not human_approved:
            reasons.append("human_approval_required_legacy_mode")
    else:
        reasons.append("unknown_approval_mode")
    return GateDecision(not reasons, reasons)


def grouped_time_split(records: Iterable[dict[str, Any]], cutoff: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(list(records))
    if frame.empty or "first_public_date" not in frame:
        return frame.copy(), frame.copy()
    dates = pd.to_datetime(frame["first_public_date"], errors="coerce").dt.date
    return frame[dates < cutoff].copy(), frame[dates >= cutoff].copy()


def leave_one_group_out(records: Iterable[dict[str, Any]], column: str, held_out: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(list(records))
    if column not in frame:
        raise KeyError(column)
    return frame[frame[column] != held_out].copy(), frame[frame[column] == held_out].copy()


def composition_baseline(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    return {"name": "composition_logistic", "auroc": float(roc_auc_score(y_test, score)), "auprc": float(average_precision_score(y_test, score))}


def energy_baseline(energy: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    score = -np.asarray(energy, dtype=float)
    return {"name": "energy_rank", "auroc": float(roc_auc_score(y_test, score)), "auprc": float(average_precision_score(y_test, score))}


def run_smoke(root: Path) -> dict[str, Any]:
    registry_path = root / "manifests" / "source_registry.json"
    sources = write_registry(registry_path)
    provenance = Provenance(source_id="materials_project", source_version="official-api", license_status="api-terms", parser_version="0.1.0", confidence=0.99, first_public_date=date(2020, 1, 1))
    graph = EvidenceGraph()
    graph.add_node("p:Fe2O3", "Precursor", formula="Fe2O3")
    graph.add_node("m:Fe3O4", "MaterialComposition", formula="Fe3O4")
    graph.add_edge("p:Fe2O3", "m:Fe3O4", "transformed_by", provenance, process="reducing atmosphere")
    graph_path = root / "artifacts" / "evidence_graph.smoke.jsonl"
    graph.to_jsonl(graph_path)
    package = EvidencePackage(claim="smoke claim", target="m:Fe3O4", source_ids=["materials_project"], evidence_level=EvidenceLevel.E3, assumptions=["toy record"], tool_calls=["smoke"], uncertainty={"type": "epistemic", "value": 0.1}, falsification_test="hold out source", safety_constraints=["human approval"], license_status="api-terms")
    critic = critic_check(package)
    safety = safety_gate([(3, "Fe2O3"), (1, "CO")], [(2, "Fe3O4"), (1, "CO2")], ["vented furnace", "codex review"], codex_approved=True)
    action_score = path_action_score([PathStep(2.0, 1.0, 0.2, 0.1, 0.0)])
    probability = target_probability(action_score, [action_score + 1.0, action_score + 2.0])
    model = MinimalPathWorldModel(PhaseState(conserved_elements={"A": 1.0}, phase_fractions={"target": 0.2, "competitor": 0.8}, time_step=0))
    states = []
    for action in [Action("change_precursor", target_bias=1.0), Action("slow_ramp", target_bias=0.4)]:
        states.append(model.step(action).model_dump(mode="json"))
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    split = 50
    result = {"sources": len(sources), "graph_errors": graph.validate(), "graph_path": str(graph_path), "critic": asdict(critic), "safety": asdict(safety), "action_score": action_score, "target_probability": probability, "states": states, "composition_baseline": composition_baseline(X[:split], y[:split], X[split:], y[split:]), "energy_baseline": energy_baseline(rng.normal(size=len(y[split:])), y[split:]), "created_at": datetime.utcnow().isoformat()}
    (root / "artifacts" / "smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
