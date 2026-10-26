#!/usr/bin/env python3
"""Local PATH-Mat material-generation and design agent.

The native backend proposes charge-balanced cubic ABO3 perovskite hypotheses
using Shannon-radius geometry. It does not predict band gaps, thermodynamic
stability, or experimental synthesizability.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
from io import StringIO
import json
import math
import platform
import subprocess
import sys
import zipfile
from math import gcd, lcm
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Species, Structure
from pymatgen.io.cif import CifParser


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(child.relative_to(path).as_posix().encode("utf-8"))
        h.update(bytes.fromhex(sha256_file(child)))
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ValueError("design config must have schema_version: 1")
    task = cfg.get("design_task", {})
    if task.get("prototype") != "cubic_ABO3_perovskite":
        raise ValueError("native backend currently supports cubic_ABO3_perovskite only")
    if int(task.get("max_proposals", 0)) < 1:
        raise ValueError("design_task.max_proposals must be positive")
    low, high = map(float, task["tolerance_factor_window"])
    if not 0 < low < high < 2:
        raise ValueError("invalid tolerance_factor_window")
    if not task.get("a_site") or not task.get("b_site"):
        raise ValueError("a_site and b_site oxidation-state dictionaries are required")
    return cfg


@dataclass
class AgentTrace:
    path: Path

    def emit(self, agent: str, action: str, status: str, **details: Any) -> None:
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "agent": agent, "action": action, "status": status, **details,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, default=str) + "\n")


class PlannerAgent:
    """Selects a local deterministic backend and records the execution plan."""

    @staticmethod
    def plan() -> list[dict[str, Any]]:
        return [
            {"step": 1, "agent": "CompositionAgent", "tool": "enumerate_charge_balanced_ABO3"},
            {"step": 2, "agent": "StructureAgent", "tool": "build_cubic_Pm-3m_CIF"},
            {"step": 3, "agent": "GeometryAgent", "tool": "Shannon_tolerance_factor_screen"},
            {"step": 4, "agent": "NoveltyAgent", "tool": "compare_hash_verified_reference_snapshots"},
            {"step": 5, "agent": "RouteAgent", "tool": "balanced_route_generation_and_frozen_PATH-Mat_EC-PathWM_scoring"},
            {"step": 6, "agent": "EvidenceAgent", "tool": "write_candidate_ledger_and_trace"},
        ]


class CompositionAgent:
    """Enumerates formally charge-balanced A+B+3O compositions."""

    @staticmethod
    def propose(cfg: dict[str, Any]) -> list[dict[str, Any]]:
        task = cfg["design_task"]
        result = []
        for a, qa in task["a_site"].items():
            for b, qb in task["b_site"].items():
                charge = int(qa) + int(qb) - 6
                result.append({
                    "a": a, "a_charge": int(qa), "b": b, "b_charge": int(qb),
                    "formula": Composition({a: 1, b: 1, "O": 3}).reduced_formula,
                    "formal_charge_sum": charge,
                    "proposal_status": "charge_balanced" if charge == 0 else "rejected_charge_imbalance",
                })
        return result


class StructureAgent:
    """Builds idealized, unrelaxed cubic Pm-3m perovskite structure hypotheses."""

    @staticmethod
    def radius(element: str, oxidation: int, cn: str, spin: str = "") -> float:
        value = Species(element, oxidation).get_shannon_radius(cn, spin=spin)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Shannon radius unavailable for {element}{oxidation:+d}, CN={cn}")
        return float(value)

    def build(self, proposal: dict[str, Any], task: dict[str, Any]) -> tuple[Structure, dict[str, float]]:
        a, qa, b, qb = proposal["a"], proposal["a_charge"], proposal["b"], proposal["b_charge"]
        spin = str(task.get("b_site_spin", {}).get(b, ""))
        ra = self.radius(a, qa, str(task.get("a_site_coordination", "XII")))
        rb = self.radius(b, qb, str(task.get("b_site_coordination", "VI")), spin)
        ro = self.radius("O", -2, str(task.get("oxygen_coordination", "VI")))
        t = (ra + ro) / (math.sqrt(2.0) * (rb + ro))
        lattice_a = 2.0 * (rb + ro)
        structure = Structure(
            Lattice.cubic(lattice_a),
            [Species(a, qa), Species(b, qb), Species("O", -2), Species("O", -2), Species("O", -2)],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
            coords_are_cartesian=False, to_unit_cell=True,
        )
        if abs(float(structure.charge or 0)) > 1e-8:
            raise ValueError("generated structure is not formally charge neutral")
        return structure, {
            "r_a_shannon_angstrom": ra, "r_b_shannon_angstrom": rb,
            "r_o_shannon_angstrom": ro, "tolerance_factor": t,
            "initial_lattice_a_angstrom": lattice_a,
        }


class GeometryAgent:
    """Ranks by a declared geometric prior; this is not a property model."""

    @staticmethod
    def evaluate(t: float, window: tuple[float, float]) -> tuple[bool, float]:
        low, high = window
        midpoint = (low + high) / 2
        score = max(0.0, 1.0 - abs(t - midpoint) / ((high - low) / 2))
        return low <= t <= high, score


class PathMatRouteAgent:
    """Rank balanced, hypothetical route schedules with the local PATH-Mat model.

    Scores are pre-calibration within-batch ranking signals. The model's
    source/evidence abstention gates are recorded rather than bypassed.
    """

    DEFAULT_SCHEDULES = (
        {"schedule_id": "lower_thermal_budget", "calcination_C": 850.0, "sintering_C": 1100.0, "calcination_h": 6.0, "sintering_h": 8.0},
        {"schedule_id": "mid_thermal_budget", "calcination_C": 950.0, "sintering_C": 1200.0, "calcination_h": 6.0, "sintering_h": 8.0},
        {"schedule_id": "higher_thermal_budget", "calcination_C": 1050.0, "sintering_C": 1300.0, "calcination_h": 6.0, "sintering_h": 8.0},
    )
    FIELDS = [
        "candidate_id", "target_formula", "route_id", "precursor_formulas", "balanced_reaction",
        "schedule_id", "actions_json", "calcination_C", "sintering_C", "calcination_h", "sintering_h",
        "condition_target_score", "condition_competitor_score", "condition_margin", "ensemble_spread",
        "feature_support_json", "model_abstain", "abstention_reasons", "evidence_risk", "path_score_proxy",
        "score_semantics", "within_candidate_rank",
    ]

    def __init__(self, model_path: Path, vendor_path: Path, trace: AgentTrace):
        self.model_path, self.vendor_path = model_path.resolve(), vendor_path.resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"PATH-Mat model artifact is missing: {self.model_path}")
        if not self.vendor_path.is_dir():
            raise FileNotFoundError(f"PATH-Mat vendor package is missing: {self.vendor_path}")
        self.trace = trace
        self.model_hash = sha256_file(self.model_path)
        app_root = self.vendor_path.parent
        windows_python = app_root / ".venv" / "Scripts" / "python.exe"
        posix_python = app_root / ".venv" / "bin" / "python"
        app_python = windows_python if windows_python.is_file() else posix_python
        self.python_executable = app_python if app_python.is_file() else Path(sys.executable)
        self.worker_path = Path(__file__).resolve().parents[1] / "analysis" / "score_pathmat_route_hypotheses.py"
        if not self.worker_path.is_file():
            raise FileNotFoundError(f"PATH-Mat route-scoring worker is missing: {self.worker_path}")

    @staticmethod
    def precursor_recipe(row: dict[str, Any]) -> tuple[list[str], str]:
        """Balanced recipes for registered oxide-perovskite prototypes."""
        a, qa = str(row["a_site"]), int(row["a_oxidation_state"])
        b, qb = str(row["b_site"]), int(row["b_oxidation_state"])
        if row.get("prototype") == "rocksalt_ordered_double_perovskite_A2BBpO6":
            bp, qbp = str(row["b_prime_site"]), int(row["b_prime_oxidation_state"])
            if 2 * qa + qb + qbp != 12:
                raise ValueError("A2BB'O6 precursor template requires a charge-neutral target composition")

            def oxide_formula(element: str, oxidation: int) -> Composition:
                if oxidation <= 0:
                    raise ValueError(f"invalid cation oxidation state for oxide precursor: {element}{oxidation:+d}")
                divisor = gcd(oxidation, 2)
                return Composition({element: 2 // divisor, "O": oxidation // divisor})

            target = Composition({a: 2, b: 1, bp: 1, "O": 6})
            cation_counts = [(a, 2, qa), (b, 1, qb), (bp, 1, qbp)]
            oxide_specs = [(element, count, oxide_formula(element, oxidation))
                           for element, count, oxidation in cation_counts]
            product_scale = lcm(*(int(oxide[element]) // gcd(count, int(oxide[element]))
                                  for element, count, oxide in oxide_specs))
            coefficients = [product_scale * count // int(oxide[element])
                            for element, count, oxide in oxide_specs]
            precursors = [oxide.reduced_formula for _, _, oxide in oxide_specs]
            lhs = sum((coefficient * oxide for coefficient, (_, _, oxide)
                       in zip(coefficients, oxide_specs)), Composition({}))
            rhs = product_scale * target
            if lhs != rhs:
                raise ValueError(f"unbalanced oxide reaction generated for {target.reduced_formula}")
            left = " + ".join(f"{coefficient if coefficient != 1 else ''}{formula}"
                              for coefficient, formula in zip(coefficients, precursors))
            right = f"{product_scale if product_scale != 1 else ''}{target.reduced_formula}"
            return precursors, f"{left} -> {right}"
        if (qa, qb) == (2, 4):
            precursors, reaction = [f"{a}CO3", f"{b}O2"], f"{a}CO3 + {b}O2 -> {row['formula']} + CO2"
            lhs, rhs = Composition(precursors[0]) + Composition(precursors[1]), Composition(row["formula"]) + Composition("CO2")
        elif (qa, qb) == (3, 3):
            precursors, reaction = [f"{a}2O3", f"{b}2O3"], f"{a}2O3 + {b}2O3 -> 2 {row['formula']}"
            lhs, rhs = Composition(precursors[0]) + Composition(precursors[1]), 2 * Composition(row["formula"])
        else:
            raise ValueError(f"No registered balanced precursor template for A/B valences {qa}+/{qb}+")
        if lhs != rhs:
            raise ValueError(f"Unbalanced precursor reaction generated for {row['formula']}: {reaction}")
        return precursors, reaction

    @staticmethod
    def _record(row: dict[str, Any], schedule: dict[str, Any], precursors: list[str]) -> dict[str, Any]:
        actions = [
            {"name": "Mixing"}, {"name": "Grinding"},
            {"name": "HeatingOperation", "temperature_C": schedule["calcination_C"], "duration_h": schedule["calcination_h"], "atmosphere": "air"},
            {"name": "Grinding"},
            {"name": "HeatingOperation", "temperature_C": schedule["sintering_C"], "duration_h": schedule["sintering_h"], "atmosphere": "air"},
        ]
        return {
            "target_formula": row["formula"], "precursor_formulas": precursors,
            "route_available": True, "actions": actions, "chemistry_family": "oxide",
            "source": {"source_id": "pathmat_design_agent_unobserved_hypothesis", "evidence_level": "E0",
                       "measurement_status": "not_measured", "publication_year": None,
                       "observed": False, "is_simulated": True},
            "evidence_conditioned_competitor": {"status": "unmatched"}, "observed_modalities": [],
        }

    def score_candidates(self, candidates: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
        route_rows: list[dict[str, Any]] = []
        per_candidate: dict[str, list[dict[str, Any]]] = {}
        scoring_inputs: list[dict[str, Any]] = []
        self.trace.emit("RouteAgent", "load_local_EC-PathWM_model", "completed",
                        model_path=str(self.model_path), model_sha256=self.model_hash,
                        worker_python=str(self.python_executable),
                        worker_script=str(self.worker_path))
        for candidate in candidates:
            precursors, reaction = self.precursor_recipe(candidate)
            scored = []
            for schedule in self.DEFAULT_SCHEDULES:
                record = self._record(candidate, schedule, precursors)
                route_id = hashlib.sha256(
                    f"{candidate['candidate_id']}|{schedule['schedule_id']}|{self.model_hash}".encode("utf-8")
                ).hexdigest()[:20]
                item = {
                    "candidate_id": candidate["candidate_id"], "target_formula": candidate["formula"],
                    "route_id": route_id, "precursor_formulas": ";".join(precursors),
                    "balanced_reaction": reaction, "schedule_id": schedule["schedule_id"],
                    "actions_json": json.dumps(record["actions"], separators=(",", ":")),
                    "calcination_C": schedule["calcination_C"], "sintering_C": schedule["sintering_C"],
                    "calcination_h": schedule["calcination_h"], "sintering_h": schedule["sintering_h"],
                    "condition_target_score": "", "condition_competitor_score": "", "condition_margin": "",
                    "ensemble_spread": "", "feature_support_json": "", "model_abstain": "",
                    "abstention_reasons": "", "evidence_risk": "", "path_score_proxy": "",
                    "score_semantics": "within-batch pre-calibration PATH-Mat source-label ranking signal; not probability or synthesis recommendation",
                    "_rank": 0.0,
                }
                scored.append(item)
                scoring_inputs.append({"route_id": route_id, "record": record})
            per_candidate[candidate["candidate_id"]] = scored

        input_path = output_dir / "pathmat_route_model_inputs.json"
        worker_output_path = output_dir / "pathmat_route_model_outputs.json"
        input_path.write_text(json.dumps({"model_path": str(self.model_path), "vendor_path": str(self.vendor_path),
                                          "records": scoring_inputs}, indent=2), encoding="utf-8")
        command = [str(self.python_executable), str(self.worker_path), "--input", str(input_path),
                   "--output", str(worker_output_path)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        self.trace.emit("RouteAgent", "score_route_hypotheses_with_frozen_model", "completed" if completed.returncode == 0 else "failed",
                        command=command, stdout=completed.stdout[-2000:], stderr=completed.stderr[-3000:],
                        model_sha256=self.model_hash)
        if completed.returncode != 0 or not worker_output_path.is_file():
            raise RuntimeError(f"PATH-Mat route scorer failed: {completed.stderr[-3000:]}")
        worker_result = json.loads(worker_output_path.read_text(encoding="utf-8"))
        score_by_route = {item["route_id"]: item for item in worker_result["results"]}
        for candidate in candidates:
            scored = per_candidate[candidate["candidate_id"]]
            for item in scored:
                result = score_by_route[item["route_id"]]
                item.update({
                    "condition_target_score": result["condition_target_score"],
                    "condition_competitor_score": result["condition_competitor_score"],
                    "condition_margin": result["condition_margin"], "ensemble_spread": result["ensemble_spread"],
                    "feature_support_json": json.dumps(result.get("feature_support", {}), sort_keys=True),
                    "model_abstain": result["model_abstain"],
                    "abstention_reasons": ";".join(result.get("abstention_reasons") or []),
                    "evidence_risk": result["evidence_risk"], "path_score_proxy": result["path_score_proxy"],
                    "_rank": float(result["condition_target_score"]),
                })
            scored.sort(key=lambda item: (-item["_rank"], item["schedule_id"]))
            for rank, item in enumerate(scored, 1):
                item["within_candidate_rank"] = rank
                route_rows.append(item)
            best = scored[0]
            candidate.update({
                "pathmat_route_candidate_count": len(scored), "pathmat_best_route_id": best["route_id"],
                "pathmat_best_schedule": best["schedule_id"],
                "pathmat_best_condition_target_score": best["condition_target_score"],
                "pathmat_route_model_abstain": best["model_abstain"],
            })

        out_path = output_dir / "pathmat_route_hypotheses.csv"
        with out_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.FIELDS)
            writer.writeheader()
            for item in route_rows:
                writer.writerow({key: item.get(key, "") for key in self.FIELDS})
        support = {
            key: any(json.loads(item["feature_support_json"]).get(key, False) for item in route_rows)
            for key in ("temperature", "duration", "atmosphere_category")
        }
        metadata = worker_result.get("model_metadata", {}) or {}
        result = {
            "status": "direct_model_scored_hypothetical_balanced_routes",
            "model_path": str(self.model_path), "model_sha256": self.model_hash,
            "model_adapter_version": worker_result.get("adapter_version"), "model_name": metadata.get("model_name"),
            "fit_record_count": metadata.get("fit_record_count"),
            "training_source_coverage": metadata.get("source_coverage", {}),
            "training_family_coverage": metadata.get("family_coverage", {}),
            "candidate_count": len(candidates), "route_hypothesis_count": len(route_rows),
            "schedule_templates": list(self.DEFAULT_SCHEDULES), "observed_route_feature_support": support,
            "scored_route_model_abstain_count": sum(bool(item["model_abstain"]) for item in route_rows),
            "distinct_condition_scores": len({round(float(item["condition_target_score"]), 12) for item in route_rows}),
            "scoring_runtime_python": worker_result.get("python_version"),
            "scoring_runtime_sklearn": worker_result.get("sklearn_version"),
            "model_input_sha256": sha256_file(input_path), "model_output_sha256": sha256_file(worker_output_path),
            "score_semantics": "pre-calibration within-batch ranking signal trained against retrospective public-record labels; not calibrated success probability, route recommendation, or causal estimate",
            "claim_boundary": "Generated precursor-balanced route/schedule hypotheses are computationally ranked. Source/family/evidence gates are preserved; unobserved candidates may be abstained. No synthesis feasibility or outcome is established.",
            "output_csv": str(out_path.resolve()), "output_sha256": sha256_file(out_path),
            "candidate_shortlist": [
                {"candidate_id": cid, "formula": next(c["formula"] for c in candidates if c["candidate_id"] == cid),
                 "best_schedule": routes[0]["schedule_id"], "condition_target_score": routes[0]["condition_target_score"],
                 "model_abstain": routes[0]["model_abstain"]}
                for cid, routes in per_candidate.items()
            ],
        }
        summary_path = output_dir / "pathmat_route_agent_summary.json"
        summary_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        self.trace.emit("RouteAgent", "enumerate_and_score_balanced_route_hypotheses", "completed",
                        candidate_count=len(candidates), route_hypothesis_count=len(route_rows),
                        model_sha256=self.model_hash, output_sha256=sha256_file(out_path),
                        model_input_sha256=sha256_file(input_path), model_output_sha256=sha256_file(worker_output_path))
        return result


class NoveltyAgent:
    """Checks against complete, hash-verified local CIF reference snapshots."""

    def __init__(self, manifest: Path | None, matcher_stol: float = 0.5):
        self.manifest = manifest
        self.matcher = StructureMatcher(stol=matcher_stol, primitive_cell=True, scale=True, attempt_supercell=True)
        self.formulas: set[str] = set()
        self.structures: list[tuple[str, Structure]] = []
        self.snapshots: list[dict[str, Any]] = []
        if manifest:
            self.load()

    def load(self) -> None:
        assert self.manifest is not None
        with self.manifest.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if not row.get("local_path") or not row.get("sha256"):
                raise ValueError("reference rows require local_path and sha256")
            path = Path(row["local_path"])
            if not path.is_absolute():
                path = (self.manifest.parent / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f"reference snapshot missing: {path}")
            observed = snapshot_sha256(path)
            if observed.lower() != row["sha256"].lower():
                raise ValueError(f"reference snapshot digest mismatch: {path}")
            cifs: list[tuple[str, str]] = []
            if path.is_dir():
                cifs = [(p.name, p.read_text(encoding="utf-8", errors="replace"))
                        for p in sorted(path.rglob("*.cif"))]
            elif path.suffix.lower() == ".cif":
                cifs = [(path.name, path.read_text(encoding="utf-8", errors="replace"))]
            elif zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as zf:
                    cifs = [(n, zf.read(n).decode("utf-8", errors="replace"))
                            for n in sorted(zf.namelist())
                            if n.lower().endswith(".cif") and ".." not in Path(n).parts]
            else:
                raise ValueError(f"reference must be a CIF, ZIP or directory: {path}")
            parsed = 0
            for name, text in cifs:
                try:
                    try:
                        structure = Structure.from_str(text, fmt="cif")
                    except Exception:
                        # Some otherwise usable experimental CIFs contain tiny
                        # occupancy round-off (e.g. 1.01); allow only a narrow
                        # tolerance and record the exact raw source snapshot.
                        structure = CifParser(StringIO(text), occupancy_tolerance=1.02).parse_structures()[0]
                    self.formulas.add(structure.composition.reduced_formula)
                    # Oxidation states are not consistently encoded in reference CIFs.
                    # Compare structural identity by element and geometry, not by
                    # whether one CIF carries an explicit valence annotation.
                    structure.remove_oxidation_states()
                    self.structures.append((f"{row.get('dataset_id', 'snapshot')}:{name}", structure))
                    parsed += 1
                except Exception:
                    continue
            self.snapshots.append({
                "dataset_id": row.get("dataset_id"), "version": row.get("version"),
                "access_date": row.get("access_date"), "license": row.get("license"),
                "path": str(path), "sha256": observed,
                "declared_structure_count": row.get("structure_count"), "parsed_cif_count": parsed,
            })

    def check(self, formula: str, structure: Structure) -> dict[str, Any]:
        if not self.snapshots:
            return {"formula_novelty": "not_evaluated", "structure_novelty": "not_evaluated", "matched_reference": ""}
        candidate = structure.copy()
        candidate.remove_oxidation_states()
        match = next((name for name, ref in self.structures if self.matcher.fit(candidate, ref)), "")
        return {
            "formula_novelty": formula not in self.formulas,
            "structure_novelty": not bool(match),
            "matched_reference": match,
        }


class EvidenceAgent:
    """Persists every candidate and rejected composition with artifact hashes."""

    fields = [
        "candidate_id", "formula", "a_site", "a_oxidation_state", "b_site", "b_oxidation_state",
        "formal_charge_sum", "proposal_status", "geometry_pass", "geometry_score", "tolerance_factor",
        "r_a_shannon_angstrom", "r_b_shannon_angstrom", "r_o_shannon_angstrom",
        "initial_lattice_a_angstrom", "cif_path", "cif_sha256", "formula_novelty",
        "structure_novelty", "matched_reference", "property_prediction_status",
        "thermodynamic_stability_status", "experimental_status", "pathmat_route_candidate_count",
        "pathmat_best_route_id", "pathmat_best_schedule", "pathmat_best_condition_target_score",
        "pathmat_route_model_abstain", "design_status",
    ]

    def write(self, out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
        with (out_dir / "candidate_ledger.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)
        (out_dir / "design_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report = f"""# PATH-Mat local material-design agent run

- Design: {summary['design_id']}
- Generated CIF hypotheses: {summary['counts']['cif_written']}
- Geometry-window passes: {summary['counts']['geometry_pass']}
- Database novelty: {summary['novelty_status']}
- Property prediction: {summary['property_prediction_status']}
- Independent DFT/phonons: {summary['independent_calculation_status']}
- PATH-Mat route integration: {summary['pathmat_route_integration']['status']}
- Direct route hypotheses scored: {summary['pathmat_route_integration'].get('route_hypothesis_count', 0)}
- PATH-Mat abstentions among direct route hypotheses: {summary['pathmat_route_integration'].get('scored_route_model_abstain_count', 0)}

The local backend enumerates formally charge-balanced ABO3 compositions, builds idealized cubic Pm-3m structures, and applies a Shannon-radius tolerance-factor prior. When configured, the RouteAgent constructs balanced precursor-route and fixed thermal-schedule hypotheses and scores them with the local frozen PATH-Mat EC-PathWM condition-ranking interface. These are within-batch signals trained against retrospective public-record labels; model coverage and abstention are recorded. They are not calibrated probabilities, synthesis recommendations, or causal estimates. CIFs remain unrelaxed structure hypotheses.

Database novelty is evaluated only when a complete, hash-verified, use-permitted reference manifest is supplied. Missing reference coverage is reported as not evaluated. No candidate is promoted to a confirmed new material without independent property and stability calculations.

## Run artifacts

- candidate_ledger.csv: full proposal ledger, including rejected pairs
- structures/: generated CIF hypotheses
- pathmat_route_hypotheses.csv: balanced route/schedule hypotheses scored by the local PATH-Mat model, when enabled
- pathmat_route_agent_summary.json: frozen-model digest, coverage, score semantics and shortlist, when enabled
- pathmat_route_model_inputs.json and pathmat_route_model_outputs.json: exact records passed to and returned by the PATH-Mat scoring worker, when enabled
- agent_trace.jsonl: timestamped agent/tool actions
- design_summary.json: summary and per-CIF SHA-256 digests
- pathmat_route_ranking.csv: optional source-label route ranking, if configured
"""
        (out_dir / "DESIGN_REPORT.md").write_text(report, encoding="utf-8")


class PathMatDesignAgent:
    """Orchestrates local composition, structure, screening and evidence agents."""

    def run(self, config_path: Path, out_dir: Path, overwrite: bool = False) -> Path:
        config_path, out_dir = config_path.resolve(), out_dir.resolve()
        cfg = load_config(config_path)
        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"refusing to overwrite non-empty output: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        structures_dir = out_dir / "structures"
        structures_dir.mkdir(exist_ok=True)
        trace = AgentTrace(out_dir / "agent_trace.jsonl")
        plan = PlannerAgent.plan()
        trace.emit("PlannerAgent", "create_local_plan", "completed", plan=plan,
                   backend="local_charge_balanced_prototype_enumerator")

        proposals = CompositionAgent.propose(cfg)
        trace.emit("CompositionAgent", "enumerate_charge_balanced_ABO3", "completed",
                   total=len(proposals), charge_balanced=sum(p["proposal_status"] == "charge_balanced" for p in proposals))
        ref_cfg = cfg.get("reference", {})
        manifest_value = ref_cfg.get("manifest", "")
        manifest = (config_path.parent / manifest_value).resolve() if manifest_value else None
        if manifest and not manifest.exists():
            raise FileNotFoundError(f"reference manifest does not exist: {manifest}")
        novelty = NoveltyAgent(manifest, float(ref_cfg.get("matcher_stol", 0.5)))
        trace.emit("NoveltyAgent", "verify_reference_snapshots",
                   "completed" if novelty.snapshots else "not_configured",
                   count=len(novelty.snapshots), manifest=str(manifest) if manifest else None)

        task = cfg["design_task"]
        window = tuple(map(float, task["tolerance_factor_window"]))
        structure_agent, geometry_agent = StructureAgent(), GeometryAgent()
        rows: list[dict[str, Any]] = []
        generated = 0
        for p in proposals:
            row: dict[str, Any] = {
                "candidate_id": "", "formula": p["formula"], "a_site": p["a"],
                "a_oxidation_state": p["a_charge"], "b_site": p["b"],
                "b_oxidation_state": p["b_charge"], "formal_charge_sum": p["formal_charge_sum"],
                "proposal_status": p["proposal_status"], "geometry_pass": False, "geometry_score": "",
                "tolerance_factor": "", "r_a_shannon_angstrom": "", "r_b_shannon_angstrom": "",
                "r_o_shannon_angstrom": "", "initial_lattice_a_angstrom": "", "cif_path": "",
                "cif_sha256": "", "formula_novelty": "not_evaluated", "structure_novelty": "not_evaluated",
                "matched_reference": "", "property_prediction_status": "not_run",
                "thermodynamic_stability_status": "not_run", "experimental_status": "not_performed",
                "design_status": "rejected_charge_imbalance" if p["proposal_status"] != "charge_balanced" else "",
            }
            if p["proposal_status"] != "charge_balanced":
                rows.append(row)
                continue
            try:
                structure, params = structure_agent.build(p, task)
            except Exception as exc:
                row["proposal_status"] = "rejected_missing_crystallographic_data"
                row["design_status"] = f"rejected_structure_build:{type(exc).__name__}"
                rows.append(row)
                continue
            passed, score = geometry_agent.evaluate(params["tolerance_factor"], window)
            row.update(params)
            row["geometry_pass"] = passed
            row["geometry_score"] = round(score, 6)
            row["tolerance_factor"] = round(params["tolerance_factor"], 6)
            if not passed:
                row["design_status"] = "rejected_tolerance_factor_window"
                rows.append(row)
                continue
            if generated >= int(task["max_proposals"]):
                row["design_status"] = "not_generated_candidate_budget"
                rows.append(row)
                continue
            candidate_id = hashlib.sha256(
                f"{cfg['design_id']}\\0{p['formula']}\\0Pm-3m\\0{params['initial_lattice_a_angstrom']:.8f}".encode()
            ).hexdigest()[:20]
            cif = structures_dir / f"{candidate_id}_{p['formula']}.cif"
            structure.to(filename=cif, fmt="cif")
            cif_hash = sha256_file(cif)
            novelty_result = novelty.check(p["formula"], structure)
            row.update({
                "candidate_id": candidate_id, "cif_path": cif.relative_to(out_dir).as_posix(),
                "cif_sha256": cif_hash, **novelty_result,
                "design_status": "unrelaxed_structure_hypothesis",
            })
            if novelty_result["formula_novelty"] is False and novelty_result["structure_novelty"] is False:
                row["design_status"] = "known_formula_and_structure_in_reference"
            generated += 1
            rows.append(row)

        rows.sort(key=lambda r: (
            r["design_status"] not in {"unrelaxed_structure_hypothesis", "known_formula_and_structure_in_reference"},
            -float(r["geometry_score"] or 0), r["formula"], r["a_site"], r["b_site"],
        ))
        valid = [r for r in rows if r["cif_path"]]
        route_cfg = cfg.get("pathmat_routes", {})
        direct_route_result: dict[str, Any] = {"status": "not_configured"}
        direct_model_value = route_cfg.get("model_path", "")
        direct_vendor_value = route_cfg.get("vendor_path", "")
        if bool(direct_model_value) != bool(direct_vendor_value):
            raise ValueError("pathmat_routes direct scoring requires both model_path and vendor_path")
        if direct_model_value and direct_vendor_value:
            model_path = Path(direct_model_value)
            vendor_path = Path(direct_vendor_value)
            model_path = model_path.resolve() if model_path.is_absolute() else (config_path.parent / model_path).resolve()
            vendor_path = vendor_path.resolve() if vendor_path.is_absolute() else (config_path.parent / vendor_path).resolve()
            direct_route_result = PathMatRouteAgent(model_path, vendor_path, trace).score_candidates(valid, out_dir)

        route_scores_value = route_cfg.get("route_scores_csv", "")
        model_manifest_value = route_cfg.get("model_manifest", "")
        route_result: dict[str, Any] = direct_route_result
        if bool(route_scores_value) != bool(model_manifest_value):
            raise ValueError("pathmat_routes requires both route_scores_csv and model_manifest")
        if route_scores_value and model_manifest_value:
            route_scores = Path(route_scores_value)
            model_manifest = Path(model_manifest_value)
            route_scores = route_scores.resolve() if route_scores.is_absolute() else (config_path.parent / route_scores).resolve()
            model_manifest = model_manifest.resolve() if model_manifest.is_absolute() else (config_path.parent / model_manifest).resolve()
            if not route_scores.is_file() or not model_manifest.is_file():
                raise FileNotFoundError("configured PATH-Mat route-score CSV or model manifest is missing")
            route_input = out_dir / "route_join_candidates.csv"
            with route_input.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["candidate_id", "reduced_formula", "computational_candidate", "confirmation_status"])
                writer.writeheader()
                for row in valid:
                    writer.writerow({"candidate_id": row["candidate_id"], "reduced_formula": row["formula"],
                                     "computational_candidate": False, "confirmation_status": row["design_status"]})
            joined_routes = out_dir / "pathmat_route_ranking.csv"
            join_script = Path(__file__).resolve().parents[1] / "analysis" / "join_routes.py"
            command = [sys.executable, str(join_script), "--candidate-audit", str(route_input),
                       "--route-scores", str(route_scores), "--out", str(joined_routes),
                       "--model-manifest", str(model_manifest), "--include-unconfirmed"]
            completed = subprocess.run(command, cwd=join_script.parent, capture_output=True, text=True, check=False)
            trace.emit("RouteAgent", "join_candidate_formulas_to_PATH-Mat_route_scores",
                       "completed" if completed.returncode == 0 else "failed",
                       command=command, stdout=completed.stdout[-4000:], stderr=completed.stderr[-4000:])
            if completed.returncode != 0 or not joined_routes.is_file():
                raise RuntimeError(f"PATH-Mat route join failed: {completed.stderr[-2000:]}")
            with joined_routes.open(newline="", encoding="utf-8-sig") as stream:
                route_count = sum(1 for _ in csv.DictReader(stream))
            joined_route_result = {
                "status": "joined_source_label_scores_for_review_only",
                "route_scores_csv": str(route_scores), "route_scores_sha256": sha256_file(route_scores),
                "model_manifest": str(model_manifest), "model_manifest_sha256": sha256_file(model_manifest),
                "joined_route_rows": route_count, "output": str(joined_routes),
                "output_sha256": sha256_file(joined_routes),
                "score_semantics": "source-label score only; not synthesis-success probability",
            }
            if direct_route_result.get("status") != "not_configured":
                route_result = {
                    "status": "direct_model_scoring_plus_pre_scored_route_join",
                    "direct_model_scoring": direct_route_result,
                    "pre_scored_route_join": joined_route_result,
                }
            else:
                route_result = joined_route_result
        summary = {
            "schema_version": 1, "design_id": cfg["design_id"],
            "objective": cfg.get("description", ""), "created_utc": datetime.now(timezone.utc).isoformat(),
            "backend": "local_charge_balanced_prototype_enumerator",
            "agent_type": "deterministic multi-agent scientific workflow",
            "config_path": str(config_path), "config_sha256": sha256_file(config_path),
            "python": sys.version, "platform": platform.platform(),
            "pymatgen_version": importlib.metadata.version("pymatgen"),
            "prototype": task["prototype"],
            "geometry_rule": "Shannon t=(r_A+r_O)/(sqrt(2)*(r_B+r_O)); cubic initial a=2*(r_B+r_O)",
            "tolerance_factor_window": list(window),
            "novelty_status": "evaluated_against_hash_verified_snapshots" if novelty.snapshots else "not_evaluated_no_reference_manifest",
            "reference_snapshots": novelty.snapshots,
            "property_prediction_status": "not_run_no_property_model_configured",
            "independent_calculation_status": "not_run_DFT_and_phonon_not_configured",
            "experimental_status": "not_performed",
            "pathmat_route_integration": route_result,
            "claim_boundary": "Generated CIFs are unrelaxed structure hypotheses. No band gap, thermodynamic or phonon stability, database novelty, synthesizability, or experimental performance is established unless separate evidence is joined.",
            "counts": {
                "composition_pairs_considered": len(proposals),
                "charge_balanced": sum(p["proposal_status"] == "charge_balanced" for p in proposals),
                "geometry_pass": sum(bool(r["geometry_pass"]) for r in rows),
                "cif_written": len(valid),
                "reference_matched_formula_and_structure": sum(r["design_status"] == "known_formula_and_structure_in_reference" for r in valid),
            },
            "candidate_ids": [r["candidate_id"] for r in valid],
            "artifact_sha256": {r["cif_path"]: r["cif_sha256"] for r in valid},
        }
        EvidenceAgent().write(out_dir, rows, summary)
        trace.emit("EvidenceAgent", "write_ledger_and_hashes", "completed",
                   candidates=len(valid), ledger=str(out_dir / "candidate_ledger.csv"))
        return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local PATH-Mat materials generation/design agent.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = PathMatDesignAgent().run(args.config, args.out_dir, overwrite=args.overwrite)
    except Exception as exc:
        parser.exit(2, f"design agent stopped: {type(exc).__name__}: {exc}\\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
