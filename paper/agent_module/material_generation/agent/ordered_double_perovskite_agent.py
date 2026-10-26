#!/usr/bin/env python3
"""Generate auditable rock-salt ordered A2BB'O6 structure hypotheses.

This local design arm expands PATH-Mat's end-member ABO3 proposal space. It
uses charge balance, Shannon-radius geometry and a fixed B-site ordering rule;
it does not predict properties or thermodynamic/phonon stability.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pymatgen.core import Composition, Lattice, Species, Structure

HERE = Path(__file__).resolve().parent
MODULE_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from pathmat_design_agent import AgentTrace, NoveltyAgent, PathMatRouteAgent, sha256_file  # noqa: E402


def _radius(element: str, oxidation: int, coordination: str, spin: str = "") -> float:
    value = Species(element, oxidation).get_shannon_radius(coordination, spin=spin)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Shannon radius unavailable for {element}{oxidation:+d} CN={coordination}")
    return float(value)


def _candidate_id(design_id: str, formula: str, structure: Structure) -> str:
    payload = f"{design_id}\0{formula}\0Fm-3m-rocksalt-B-order\0{structure.as_dict()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def propose(config: dict[str, Any]) -> list[dict[str, Any]]:
    task = config["design_task"]
    proposals = []
    for a, qa in task["a_site"].items():
        for b, qb in task["b_site"].items():
            for bp, qbp in task["b_prime_site"].items():
                charge = 2 * int(qa) + int(qb) + int(qbp) - 12
                if b == bp:
                    continue
                proposals.append({
                    "prototype": "rocksalt_ordered_double_perovskite_A2BBpO6",
                    "a_site": a, "a_oxidation_state": int(qa),
                    "b_site": b, "b_oxidation_state": int(qb),
                    "b_prime_site": bp, "b_prime_oxidation_state": int(qbp),
                    "formula": Composition({a: 2, b: 1, bp: 1, "O": 6}).reduced_formula,
                    "formal_charge_sum": charge,
                    "proposal_status": "charge_balanced" if charge == 0 else "rejected_charge_imbalance",
                })
    return proposals


def build_structure(proposal: dict[str, Any], task: dict[str, Any]) -> tuple[Structure, dict[str, float]]:
    a, qa = proposal["a_site"], proposal["a_oxidation_state"]
    b, qb = proposal["b_site"], proposal["b_oxidation_state"]
    bp, qbp = proposal["b_prime_site"], proposal["b_prime_oxidation_state"]
    ra = _radius(a, qa, str(task.get("a_site_coordination", "XII")))
    rb = _radius(b, qb, str(task.get("b_site_coordination", "VI")), str(task.get("b_site_spin", {}).get(b, "")))
    rbp = _radius(bp, qbp, str(task.get("b_site_coordination", "VI")), str(task.get("b_prime_site_spin", {}).get(bp, "")))
    ro = _radius("O", -2, str(task.get("oxygen_coordination", "VI")))
    rb_mean = (rb + rbp) / 2
    tolerance = (ra + ro) / (math.sqrt(2.0) * (rb_mean + ro))
    lattice_a = 2.0 * (rb_mean + ro)

    primitive = Structure(
        Lattice.cubic(lattice_a),
        [Species(a, qa), Species(b, qb), Species("O", -2), Species("O", -2), Species("O", -2)],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        coords_are_cartesian=False, to_unit_cell=True,
    )
    structure = primitive * [2, 2, 2]
    b_indices = [i for i, site in enumerate(structure) if site.specie.symbol == b]
    if len(b_indices) != 8:
        raise AssertionError(f"expected 8 B sites in 2x2x2 cell; observed {len(b_indices)}")
    ordered = {b: 0, bp: 0}
    for i in b_indices:
        # B sites are at (n + 1/2)/2 in the doubled conventional cell.
        cell_index = [int(round(float(x) * 2 - 0.5)) % 2 for x in structure[i].frac_coords]
        element = b if sum(cell_index) % 2 == 0 else bp
        structure.replace(i, Species(element, qb if element == b else qbp))
        ordered[element] += 1
    if ordered != {b: 4, bp: 4}:
        raise AssertionError(f"rock-salt B-site order must contain four of each cation: {ordered}")
    if abs(float(structure.charge or 0.0)) > 1e-8:
        raise AssertionError("generated double-perovskite supercell is not charge neutral")
    if len(structure) != 40:
        raise AssertionError("A2BB'O6 2x2x2 ordered supercell should contain 40 atoms")
    oct_b, oct_bp = rb / ro, rbp / ro
    return structure, {
        "r_a_shannon_angstrom": ra, "r_b_shannon_angstrom": rb,
        "r_b_prime_shannon_angstrom": rbp, "r_o_shannon_angstrom": ro,
        "mean_b_site_radius_angstrom": rb_mean,
        "tolerance_factor": tolerance,
        "b_oct_factor": oct_b, "b_prime_oct_factor": oct_bp,
        "b_site_radius_mismatch_angstrom": abs(rb - rbp),
        "initial_conventional_lattice_a_angstrom": lattice_a,
        "initial_supercell_formula": structure.composition.reduced_formula,
    }


def run(config_path: Path, output_dir: Path) -> Path:
    config_path, output_dir = config_path.resolve(), output_dir.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 1:
        raise ValueError("config must declare schema_version: 1")
    task = cfg["design_task"]
    if task.get("prototype") != "rocksalt_ordered_double_perovskite_A2BBpO6":
        raise ValueError("this design arm only generates rock-salt ordered A2BB'O6 hypotheses")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    structures_dir = output_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    trace = AgentTrace(output_dir / "agent_trace.jsonl")
    proposals = propose(cfg)
    trace.emit("CompositionAgent", "enumerate_charge_balanced_A2BBpO6", "completed",
               proposals=len(proposals), charge_balanced=sum(p["proposal_status"] == "charge_balanced" for p in proposals))

    t_low, t_high = map(float, task["tolerance_factor_window"])
    mu_low, mu_high = map(float, task["octahedral_factor_window"])
    candidates: list[dict[str, Any]] = []
    rejected = []
    seen_formulas: set[str] = set()
    reference_cfg = cfg.get("reference", {})
    manifest_value = str(reference_cfg.get("manifest", "")).strip()
    manifest_path = (config_path.parent / manifest_value).resolve() if manifest_value else None
    if manifest_path and not manifest_path.is_file():
        raise FileNotFoundError(f"reference manifest not found: {manifest_path}")
    novelty_agent = NoveltyAgent(manifest_path, float(reference_cfg.get("matcher_stol", 0.5)))
    trace.emit("NoveltyAgent", "load_hash_verified_reference_snapshots",
               "completed" if novelty_agent.snapshots else "not_configured",
               snapshot_count=len(novelty_agent.snapshots), manifest=str(manifest_path) if manifest_path else None)
    for proposal in proposals:
        if proposal["proposal_status"] != "charge_balanced":
            rejected.append({**proposal, "design_status": "rejected_charge_imbalance"})
            continue
        try:
            structure, geom = build_structure(proposal, task)
        except (ValueError, KeyError) as exc:
            rejected.append({**proposal, "design_status": f"rejected_missing_radius_data:{type(exc).__name__}"})
            continue
        reasons = []
        if not t_low <= geom["tolerance_factor"] <= t_high:
            reasons.append("tolerance_factor_outside_window")
        if not mu_low <= geom["b_oct_factor"] <= mu_high:
            reasons.append("B_octahedral_factor_outside_window")
        if not mu_low <= geom["b_prime_oct_factor"] <= mu_high:
            reasons.append("B_prime_octahedral_factor_outside_window")
        if reasons:
            rejected.append({**proposal, **geom, "geometry_pass": False,
                             "geometry_rejection_reasons": ";".join(reasons),
                             "design_status": "rejected_geometric_prior"})
            continue
        if proposal["formula"] in seen_formulas:
            rejected.append({**proposal, **geom, "geometry_pass": True,
                             "geometry_rejection_reasons": "duplicate_reduced_formula_within_search",
                             "design_status": "rejected_duplicate_formula_within_search"})
            continue
        seen_formulas.add(proposal["formula"])
        cid = _candidate_id(cfg["design_id"], proposal["formula"], structure)
        cif = structures_dir / f"{cid}_{proposal['formula']}.cif"
        structure.to(filename=cif, fmt="cif")
        novelty = novelty_agent.check(proposal["formula"], structure)
        candidates.append({
            **proposal, **geom, "candidate_id": cid, "cif_path": cif.relative_to(output_dir).as_posix(),
            "cif_sha256": sha256_file(cif), "geometry_pass": True,
            **novelty,
            "target_property_status": "not_evaluated_no_target_property_model",
            "thermodynamic_stability_status": "not_evaluated_no_energy_model",
            "phonon_stability_status": "not_evaluated_no_phonon_calculation",
            "experimental_status": "not_performed",
            "design_status": "unrelaxed_ordered_structure_hypothesis",
        })
    trace.emit("StructureAgent", "build_ordered_2x2x2_supercells", "completed",
               cif_count=len(candidates), rejected=len(rejected), supercell_atoms=40)

    route_cfg = cfg["pathmat_routes"]
    def resolve_config_path(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else config_path.parent / path).resolve()
    route_agent = PathMatRouteAgent(resolve_config_path(route_cfg["model_path"]),
                                    resolve_config_path(route_cfg["vendor_path"]), trace)
    route_summary = route_agent.score_candidates(candidates, output_dir) if candidates else {"status": "no_geometry_screened_candidates"}

    fields = [
        "candidate_id", "formula", "a_site", "a_oxidation_state", "b_site", "b_oxidation_state",
        "b_prime_site", "b_prime_oxidation_state", "formal_charge_sum", "geometry_pass", "tolerance_factor",
        "b_oct_factor", "b_prime_oct_factor", "b_site_radius_mismatch_angstrom", "initial_conventional_lattice_a_angstrom",
        "geometry_rejection_reasons", "cif_path", "cif_sha256", "formula_novelty", "structure_novelty",
        "matched_reference", "target_property_status",
        "thermodynamic_stability_status", "phonon_stability_status", "experimental_status", "design_status",
    ]
    with (output_dir / "candidate_ledger.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates + rejected)

    summary = {
        "schema_version": 1, "design_id": cfg["design_id"], "objective": cfg.get("description", ""),
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "prototype": task["prototype"],
        "structure_model": "40-atom 2x2x2 cubic perovskite supercell with rock-salt B-site order; unrelaxed",
        "geometric_filters": {"tolerance_factor_window": [t_low, t_high], "octahedral_factor_window": [mu_low, mu_high],
                              "interpretation": "coarse ionic-size compatibility priors, not stability predictions"},
        "counts": {"composition_pairs_enumerated": len(proposals),
                   "charge_balanced": sum(p["proposal_status"] == "charge_balanced" for p in proposals),
                   "geometry_screened_candidates": len(candidates), "geometry_rejected_or_unbuildable": len(rejected),
                   "unique_generated_formulas": len({row["formula"] for row in candidates}),
                   "cif_written": len(candidates), "atoms_per_cif": 40,
                   "route_hypotheses": route_summary.get("route_hypothesis_count", 0),
                   "pathmat_abstentions": route_summary.get("scored_route_model_abstain_count", 0)},
        "novelty_status": "evaluated_against_hash_verified_snapshots" if novelty_agent.snapshots else "not_evaluated_no_hash_verified_reference_manifest",
        "reference_snapshots": novelty_agent.snapshots,
        "target_property_status": "not_evaluated_no_target_property_model",
        "thermodynamic_stability_status": "not_evaluated_no_independent_energy_model_or_DFT",
        "phonon_stability_status": "not_evaluated_no_phonon_calculation",
        "experimental_status": "not_performed",
        "pathmat_route_integration": route_summary,
            "claim_boundary": "This run computationally designed charge-balanced, geometrically screened, unrelaxed ordered double-perovskite candidate structures. Database novelty, target-property hits, thermodynamic/phonon stability, synthesis feasibility, and experimental realization require separate validation.",
        "candidates": [{k: row[k] for k in ("candidate_id", "formula", "cif_path", "cif_sha256", "tolerance_factor",
                                              "b_oct_factor", "b_prime_oct_factor", "b_site_radius_mismatch_angstrom")}
                       for row in candidates],
        "artifact_sha256": {row["cif_path"]: row["cif_sha256"] for row in candidates},
    }
    (output_dir / "design_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "DESIGN_REPORT.md").write_text(
        "# PATH-Mat ordered double-perovskite design-agent run\n\n"
        f"- Composition pairs enumerated: {summary['counts']['composition_pairs_enumerated']}\n"
        f"- Charge-balanced: {summary['counts']['charge_balanced']}\n"
        f"- Geometric-screen passes / CIFs: {summary['counts']['geometry_screened_candidates']}\n"
        f"- Balanced route hypotheses scored by PATH-Mat: {summary['counts']['route_hypotheses']}\n"
        f"- PATH-Mat abstentions: {summary['counts']['pathmat_abstentions']}\n\n"
        + summary["claim_boundary"] + "\n", encoding="utf-8")
    trace.emit("EvidenceAgent", "write_candidate_ledger_hashes_and_route_join", "completed",
               candidate_count=len(candidates), route_summary=route_summary)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local PATH-Mat rock-salt ordered double-perovskite design agent.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config, args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
