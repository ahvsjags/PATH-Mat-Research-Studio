from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Species


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "material_generation" / "config" / "v280_repair_fresh_b4b4_v1.yaml"
IDENTITY_SCAN = PROJECT / "source_data" / "prior_identity_scan_v280.json"
BUILDER_PATH = PROJECT / "material_generation" / "agent" / "ordered_double_perovskite_agent.py"
CATALOG_PATH = PROJECT / "material_generation" / "agent" / "ordering_catalog.py"
OUT = PROJECT / "runs" / "holdout" / "v280_failure_repair_fresh_b4b4_v1" / "geometry"
MATCH = StructureMatcher(ltol=0.2, stol=0.2, angle_tol=2.0, primitive_cell=True,
                         scale=True, attempt_supercell=True, allow_subset=False)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def formula_for(a: str, b: str, bp: str) -> str:
    return Composition({a: 2, b: 1, bp: 1, "O": 6}).reduced_formula


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite geometry preflight: {OUT}")
    scan = json.loads(IDENTITY_SCAN.read_text(encoding="utf-8"))
    if (scan.get("status") != "V280_PRIOR_IDENTITY_SCAN_COMPLETE"
            or scan.get("identity_only") is not True
            or scan.get("scores_read") is not False
            or scan.get("outcomes_read") is not False
            or scan.get("required_versions_missing")
            or scan.get("scan_errors")):
        raise RuntimeError("identity-only scan is incomplete or not outcome-isolated")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    task = cfg["design_task"]
    a_sites = task["a_site"]
    b_species = task["b_site"]
    bp_species = task["b_prime_site"]
    selection = task["pair_selection"]
    pair_cap = int(selection["cluster_cap"])
    seed = str(selection["deterministic_seed"])
    old_pairs = set(scan["prior_bsite_pair_identities"])
    old_formulas = set(scan["prior_reduced_formula_identities"])
    builder = load_module(BUILDER_PATH, "v280_ordered_double_perovskite_builder")
    catalog = load_module(CATALOG_PATH, "v280_ordering_catalog")

    # Build and enumerate one stoichiometric template. The enumeration is an
    # unscored symmetry operation and fixes a shared six-class ordering catalog.
    template_proposal = {
        "a_site": "Ca", "a_oxidation_state": 2,
        "b_site": "Fe", "b_oxidation_state": 4,
        "b_prime_site": "Ti", "b_prime_oxidation_state": 4,
        "formula": formula_for("Ca", "Fe", "Ti"),
    }
    template, _ = builder.build_structure(template_proposal, task)
    orders = catalog.enumerate_bsite_orderings(
        template, b_site="Fe", b_oxidation_state=4,
        b_prime_site="Ti", b_prime_oxidation_state=4,
    )
    if len(orders) != 6 or [x[1]["ordering_class"] for x in orders] != list(range(6)):
        raise RuntimeError("expected the canonical six-class 2x2x2 B-site catalog")
    bitstrings = [x[1]["site_assignment_bitstring"] for x in orders]

    # Generate the complete unscored geometry-eligible pool and apply the
    # predeclared, identity-only exclusions and SHA-256 cap.
    pool: list[dict] = []
    excluded: list[dict] = []
    t_low, t_high = map(float, task["tolerance_factor_window"])
    mu_low, mu_high = map(float, task["octahedral_factor_window"])
    for b, bp in itertools.combinations(sorted(b_species), 2):
        pair_id = "-".join(sorted((b, bp)))
        cluster_id = f"A2_B4_B4::{pair_id}::ox4-4"
        if pair_id in old_pairs:
            excluded.append({"pair_cluster": cluster_id, "reason": "prior_executed_pair_identity"})
            continue
        per_a = []
        failed = None
        for a, qa in a_sites.items():
            formula = formula_for(a, b, bp)
            if formula in old_formulas:
                failed = {"reason": "prior_exact_reduced_formula", "formula": formula}
                break
            proposal = {
                "a_site": a, "a_oxidation_state": int(qa),
                "b_site": b, "b_oxidation_state": int(b_species[b]),
                "b_prime_site": bp, "b_prime_oxidation_state": int(bp_species[bp]),
                "formula": formula,
                "formal_charge_sum": 2 * int(qa) + int(b_species[b]) + int(bp_species[bp]) - 12,
            }
            if proposal["formal_charge_sum"] != 0:
                failed = {"reason": "charge_imbalance", "formula": formula}
                break
            try:
                parent, geometry = builder.build_structure(proposal, task)
            except Exception as exc:
                failed = {"reason": f"radius_or_structure:{type(exc).__name__}:{str(exc)[:160]}", "formula": formula}
                break
            checks = {
                "reduced_formula_matches": parent.composition.reduced_formula == formula,
                "neutral_parent": abs(float(parent.charge or 0.0)) <= 1e-8,
                "tolerance_in_range": t_low <= geometry["tolerance_factor"] <= t_high,
                "b_octa_in_range": mu_low <= geometry["b_oct_factor"] <= mu_high,
                "b_prime_octa_in_range": mu_low <= geometry["b_prime_oct_factor"] <= mu_high,
            }
            if not all(checks.values()):
                failed = {"reason": "geometry_or_parent_check", "formula": formula, "checks": checks}
                break
            per_a.append({"proposal": proposal, "geometry": geometry, "checks": checks})
        if failed:
            excluded.append({"pair_cluster": cluster_id, **failed})
        elif len(per_a) == len(a_sites):
            pool.append({
                "pair_id": pair_id, "pair_cluster": cluster_id,
                "selection_sha256": hashlib.sha256(f"{seed}|{cluster_id}".encode()).hexdigest(),
                "per_a": per_a,
            })
    pool.sort(key=lambda x: (x["selection_sha256"], x["pair_cluster"]))
    if len(pool) < pair_cap:
        raise RuntimeError(f"only {len(pool)} fresh complete pair clusters; required cap is {pair_cap}")
    selected = pool[:pair_cap]

    OUT.mkdir(parents=True)
    (OUT / "parents").mkdir()
    (OUT / "catalog").mkdir()
    template_dir = OUT / "template"
    template_dir.mkdir()
    template_record = {
        "template_formula": template_proposal["formula"],
        "enumeration_method": "exhaustive_choose_8_4_pymatgen_StructureMatcher",
        "ordering_catalog_sha256": sha(CATALOG_PATH),
        "bitstrings_by_class": bitstrings,
        "oracle_scoring_performed": False,
    }
    (template_dir / "ordering_templates.json").write_text(json.dumps(template_record, indent=2) + "\n", encoding="utf-8")
    template_b_indices = [i for i, site in enumerate(template) if site.specie.symbol in {"Fe", "Ti"}]
    if len(template_b_indices) != 8:
        raise RuntimeError("template B sublattice must contain eight sites")
    coord_key = lambda structure, idx: tuple(round(float(x), 8) for x in structure[idx].frac_coords)
    template_coords = [coord_key(template, i) for i in template_b_indices]

    records: list[dict] = []
    selected_pool_summary = []
    for entry in selected:
        b, bp = entry["pair_id"].split("-")
        selected_pool_summary.append({"pair_id": entry["pair_id"], "pair_cluster": entry["pair_cluster"], "selection_sha256": entry["selection_sha256"]})
        for item in entry["per_a"]:
            proposal = item["proposal"]
            geometry = item["geometry"]
            checks = item["checks"]
            a = proposal["a_site"]
            formula = proposal["formula"]
            parent, recomputed_geometry = builder.build_structure(proposal, task)
            if recomputed_geometry != geometry:
                raise RuntimeError(f"non-deterministic geometry rebuild for {formula}")
            parent_path = OUT / "parents" / f"{formula}_parent.cif"
            parent.to(filename=parent_path, fmt="cif")
            parent_by_coord = {coord_key(parent, i): i for i in range(len(parent))}
            if len(parent) != 40 or not set(template_coords).issubset(parent_by_coord):
                raise RuntimeError(f"B-sublattice coordinate mismatch for {formula}")
            aligned_b_indices = [parent_by_coord[c] for c in template_coords]
            class_dir = OUT / "catalog" / formula
            class_dir.mkdir()
            class_records = []
            for cls, bitstring in enumerate(bitstrings):
                structure = parent.copy()
                for j, idx in enumerate(aligned_b_indices):
                    if bitstring[j] == "1":
                        structure.replace(idx, Species(b, int(b_species[b])))
                    else:
                        structure.replace(idx, Species(bp, int(bp_species[bp])))
                if structure.composition.reduced_formula != formula or abs(float(structure.charge or 0.0)) > 1e-8:
                    raise RuntimeError(f"class composition/charge mismatch for {formula}/class{cls}")
                class_path = class_dir / f"class{cls}.cif"
                structure.to(filename=class_path, fmt="cif")
                class_records.append({
                    "ordering_class": cls,
                    "class_cif": class_path.relative_to(OUT).as_posix(),
                    "class_cif_sha256": sha(class_path),
                    "site_assignment_bitstring": bitstring,
                })
            records.append({
                "family_id": "A2_B4_B4",
                **proposal,
                **geometry,
                "b_site_pair": f"{b}-{bp}",
                "pair_cluster": entry["pair_cluster"],
                "pair_selection_sha256": entry["selection_sha256"],
                "parent_cif": parent_path.relative_to(OUT).as_posix(),
                "parent_sha256": sha(parent_path),
                "preflight_checks": checks,
                "class_records": class_records,
            })
    records.sort(key=lambda x: (x["pair_cluster"], x["a_site"], x["formula"]))
    formulas = [x["formula"] for x in records]
    pair_ids = {x["pair_cluster"] for x in records}
    if len(records) != pair_cap * len(a_sites) or len(set(formulas)) != len(formulas) or len(pair_ids) != pair_cap:
        raise RuntimeError("selected grid size, formula uniqueness, or pair cap mismatch")

    preflight = {
        "status": "V280_GEOMETRY_ONLY_UNSCORED_PREFLIGHT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha(CONFIG),
        "identity_scan_sha256": sha(IDENTITY_SCAN),
        "builder_sha256": sha(BUILDER_PATH),
        "ordering_catalog_sha256": sha(CATALOG_PATH),
        "template_sha256": sha(template_dir / "ordering_templates.json"),
        "proposal_pair_count_before_geometry": len(list(itertools.combinations(sorted(b_species), 2))),
        "fresh_complete_pair_count_before_cap": len(pool),
        "pair_cap": pair_cap,
        "pair_selection_seed": seed,
        "selected_pair_clusters": selected_pool_summary,
        "excluded_pair_clusters": excluded,
        "formula_count": len(records),
        "pair_cluster_count": len(pair_ids),
        "a_site_count": len(a_sites),
        "a_sites": sorted(a_sites),
        "formula_counts_by_a_site": {a: sum(x["a_site"] == a for x in records) for a in sorted(a_sites)},
        "records": records,
        "oracle_scoring_performed": False,
        "selection_outcomes_used": False,
        "freshness_definition": "B/B-prime element-pair identity and exact reduced-formula identity absent from actual executed PATH-Mat records through V279; identity-only scan of raw JSONL fields; no prior scores or outcomes read or used.",
        "geometry_filter_interpretation": "Shannon-radius tolerance/octahedral screens create a matched computational test grid; they do not establish stability or synthesizability.",
    }
    preflight_path = OUT / "preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "status": preflight["status"],
        "config_sha256": sha(CONFIG),
        "identity_scan_sha256": sha(IDENTITY_SCAN),
        "builder_sha256": sha(BUILDER_PATH),
        "ordering_catalog_sha256": sha(CATALOG_PATH),
        "template_sha256": sha(template_dir / "ordering_templates.json"),
        "preflight_sha256": sha(preflight_path),
        "formula_count": len(records), "pair_cluster_count": len(pair_ids),
        "a_sites": sorted(a_sites), "oracle_scoring_performed": False,
    }
    manifest_path = OUT / "preflight_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest | {"fresh_complete_pair_count_before_cap": len(pool), "selected_pair_ids": sorted({x["pair_id"] for x in selected})}, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Specified spin=.*not consistent with database spin.*")
        main()
