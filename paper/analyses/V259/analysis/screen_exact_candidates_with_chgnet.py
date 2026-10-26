#!/usr/bin/env python3
"""Relax PATH-Mat candidate CIFs and same-formula published structures with pinned CHGNet.

The calculation is an ML interatomic-potential prescreen, not a first-principles DFT
calculation. Relative energies are only compared within each identical composition.
"""
from __future__ import annotations

import csv
import hashlib
from importlib.metadata import version as package_version
import json
import os
import sys
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
PROJECT = MODULE.parent.parent
RUN = MODULE / "runs" / "local_pathmat_double_perovskite_20261017_v2"
DFT_SCREEN = MODULE / "runs" / "published_dft_stability_20261017_v1"
SOURCE_DIR = MODULE / "data" / "references" / "acs_chem_mater_2024_double_perovskite"
SOURCE_ZIP = SOURCE_DIR / "cm4c00576_si_002_structures.zip"
COD_REFERENCE_DIR = MODULE / "runs" / "cod_targeted_snapshot_20261017_v2" / "reference_cifs"
INPUT_CSV = DFT_SCREEN / "pathmat_candidate_published_dft_stability.csv"
ROUTE_CSV = DFT_SCREEN / "pathmat_dft_target_filtered_route_hypotheses.csv"
RUNTIME = MODULE / "runtime" / "chgnet_0_4_2"
OUT = MODULE / "runs" / "chgnet_exact_structure_screen_20261017_v1"
CHECKPOINT = OUT / "completed_candidate_pairs.jsonl"
MODEL_REL = Path("chgnet") / "pretrained" / "0.3.0" / "chgnet_0.3.0_e29f68s314m37.pth.tar"
MODEL_PATH = RUNTIME / MODEL_REL
EXPECTED_SOURCE_SHA256 = "a6319ab66b28c7a8d008b7a2f0143a0fec944cf7805470e01de530904e337c97"
FORCE_TOL = 0.05  # eV/angstrom
MAX_STEPS = 500
RELATIVE_ENERGY_LIMIT_EV_ATOM = 0.050
MATCHER = {"ltol": 0.2, "stol": 0.5, "angle_tol": 5, "primitive_cell": True, "scale": True, "attempt_supercell": True}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def checkpoint_rows() -> dict[str, dict]:
    if not CHECKPOINT.exists():
        return {}
    found: dict[str, dict] = {}
    with CHECKPOINT.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                found[row["candidate_id"]] = row
    return found


def source_poscar_member(names: list[str], formula: str) -> str:
    hits = [name for name in names if name.startswith("POSCAR_FILE/") and name.rsplit("/", 1)[-1].endswith("-" + formula)]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one published POSCAR for {formula}; found {hits}")
    return hits[0]


def relax_one(structure, relaxer, model, *, label: str, output_cif: Path | None = None) -> dict:
    import numpy as np

    result = relaxer.relax(
        structure,
        fmax=FORCE_TOL,
        steps=MAX_STEPS,
        relax_cell=True,
        verbose=False,
        assign_magmoms=True,
    )
    final = result["final_structure"]
    prediction = model.predict_structure(final)
    forces = np.asarray(prediction["f"], dtype=float)
    final_fmax = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
    energy = float(np.asarray(prediction["e"]).reshape(-1)[0])
    steps = len(result["trajectory"].energies)
    converged = final_fmax <= FORCE_TOL + 1e-6
    result_data = {
        "energy_eV_atom": energy,
        "final_max_force_eV_A": final_fmax,
        "force_converged": converged,
        "optimizer_steps_recorded": steps,
        "atom_count": len(final),
    }
    if output_cif is not None:
        output_cif.parent.mkdir(parents=True, exist_ok=True)
        final.to(filename=str(output_cif), fmt="cif")
        result_data["relaxed_cif"] = str(output_cif.relative_to(OUT)).replace("\\", "/")
        result_data["relaxed_cif_sha256"] = sha256(output_cif)
    return result_data, final


def build_row(candidate: dict[str, str], zip_file: zipfile.ZipFile, zip_names: list[str], model, relaxer) -> dict:
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    cid, formula = candidate["candidate_id"], candidate["formula"]
    cif_path = RUN / candidate["candidate_cif"]
    if sha256(cif_path) != candidate["candidate_cif_sha256"]:
        raise RuntimeError(f"Generated CIF digest mismatch for {formula}")
    candidate_structure = Structure.from_file(cif_path)
    member = source_poscar_member(zip_names, formula)
    raw_poscar = zip_file.read(member)
    source_structure = Structure.from_str(raw_poscar.decode("utf-8"), fmt="poscar")
    if source_structure.composition.reduced_composition != candidate_structure.composition.reduced_composition:
        raise RuntimeError(f"Formula mismatch between candidate and published POSCAR: {formula} / {member}")

    relaxed_path = OUT / "relaxed_candidate_cifs" / f"{cid[:12]}_{formula}_CHGNet_relaxed.cif"
    generated, relaxed_generated = relax_one(candidate_structure, relaxer, model, label="generated", output_cif=relaxed_path)
    published, relaxed_published = relax_one(source_structure, relaxer, model, label="source")
    delta = generated["energy_eV_atom"] - published["energy_eV_atom"]
    matcher = StructureMatcher(**MATCHER)
    collapsed_to_published = bool(matcher.fit(relaxed_generated, relaxed_published))
    source_hull = float(candidate["source_decomposition_energy_above_hull_eV_atom"])
    source_perovskite = candidate["source_relaxed_structure_class"] == "perovskite"
    pass_screen = (
        source_perovskite
        and source_hull <= 0.050
        and generated["force_converged"]
        and published["force_converged"]
        and delta <= RELATIVE_ENERGY_LIMIT_EV_ATOM
        and not collapsed_to_published
    )
    reason = []
    if not source_perovskite:
        reason.append("source_DFT_class_not_perovskite")
    if source_hull > 0.050:
        reason.append("source_composition_hull_distance_over_50_meV_atom")
    if not generated["force_converged"]:
        reason.append("candidate_CHGNet_relaxation_force_not_converged")
    if not published["force_converged"]:
        reason.append("reference_CHGNet_relaxation_force_not_converged")
    if delta > RELATIVE_ENERGY_LIMIT_EV_ATOM:
        reason.append("candidate_over_50_meV_atom_above_same_formula_reference")
    if collapsed_to_published:
        reason.append("relaxed_candidate_matches_same_formula_reference")
    if pass_screen:
        reason = ["pass"]
    return {
        "candidate_id": cid,
        "formula": formula,
        "input_cif": candidate["candidate_cif"],
        "input_cif_sha256": candidate["candidate_cif_sha256"],
        "source_poscar_member": member,
        "source_poscar_sha256": sha256_bytes(raw_poscar),
        "source_published_hull_eV_atom": source_hull,
        "source_published_relaxed_class": candidate["source_relaxed_structure_class"],
        "generated_relaxed": generated,
        "source_relaxed_same_model": published,
        "same_formula_relative_energy_eV_atom": delta,
        "same_formula_relative_energy_meV_atom": delta * 1000.0,
        "relaxed_candidate_matches_same_formula_published_structure": collapsed_to_published,
        "screen_pass": pass_screen,
        "screen_reason": reason,
    }


def main() -> None:
    if not (INPUT_CSV.is_file() and ROUTE_CSV.is_file() and SOURCE_ZIP.is_file() and MODEL_PATH.is_file() and COD_REFERENCE_DIR.is_dir()):
        raise FileNotFoundError("A required frozen candidate/source/model input is missing")
    source_digest = sha256(SOURCE_ZIP)
    if source_digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"Published POSCAR archive digest mismatch: {source_digest}")
    if not RUNTIME.is_dir():
        raise FileNotFoundError(RUNTIME)
    sys.path.insert(0, str(RUNTIME))
    import numpy as np
    import pymatgen
    import torch
    from chgnet import __version__ as chgnet_package_version
    from chgnet.model import CHGNet, StructOptimizer
    from ase import __version__ as ase_version
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20261017)
    if device == "cuda":
        torch.cuda.manual_seed_all(20261017)
    # Suppress known benign SciPy FrechetCellFilter conditioning warnings; all
    # convergence decisions are instead made from the final predicted forces.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad=True to a scalar")
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = read_csv(INPUT_CSV)
    candidate_formulas = {r["formula"] for r in candidates}
    cod_refs: dict[str, list[tuple[str, object]]] = {}
    cod_parse_errors = []
    for cod_path in sorted(COD_REFERENCE_DIR.glob("*.cif")):
        try:
            cod_structure = Structure.from_file(str(cod_path), occupancy_tolerance=1.02)
            cod_formula = cod_structure.composition.reduced_formula
            if cod_formula in candidate_formulas:
                cod_refs.setdefault(cod_formula, []).append((cod_path.stem, cod_structure))
        except Exception as exc:
            cod_parse_errors.append({"file": cod_path.name, "error": f"{type(exc).__name__}: {exc}"})
    cod_matcher = StructureMatcher(**MATCHER)
    existing = checkpoint_rows()
    model = CHGNet.load(model_name="0.3.0", use_device=device, verbose=False)
    relaxer = StructOptimizer(model=model, use_device=device)
    with zipfile.ZipFile(SOURCE_ZIP) as zf:
        names = zf.namelist()
        for candidate in candidates:
            cid = candidate["candidate_id"]
            if cid in existing:
                # Resume only if the exact parent CIF and source archive remain the same.
                if existing[cid]["input_cif_sha256"] != candidate["candidate_cif_sha256"] or existing[cid]["source_archive_sha256"] != source_digest:
                    raise RuntimeError(f"Resume input digest changed for {candidate['formula']}")
                continue
            row = build_row(candidate, zf, names, model, relaxer)
            row["source_archive_sha256"] = source_digest
            row["model_checkpoint_sha256"] = sha256(MODEL_PATH)
            row["model_name"] = "CHGNet 0.3.0 MPtrj-pretrained, loaded by package 0.4.2"
            row["created_utc"] = datetime.now(timezone.utc).isoformat()
            with CHECKPOINT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing[cid] = row
            print(json.dumps({"formula": row["formula"], "delta_e_meV_atom": round(row["same_formula_relative_energy_meV_atom"], 3), "converged": row["generated_relaxed"]["force_converged"], "screen_pass": row["screen_pass"]}, ensure_ascii=False), flush=True)

    result_rows = [existing[r["candidate_id"]] for r in candidates]
    # Recheck novelty after MLIP relaxation against every retrieved exact-formula
    # COD structure, including CIFs with slight occupancy excess (tolerance 1.02).
    for r in result_rows:
        hits = [cod_id for cod_id, cod_structure in cod_refs.get(r["formula"], [])
                if cod_matcher.fit(Structure.from_file(str(OUT / r["generated_relaxed"]["relaxed_cif"])), cod_structure)]
        r["relaxed_candidate_matches_cod_structure"] = bool(hits)
        r["matching_cod_ids_after_relaxation"] = hits
        reasons = [] if r["screen_pass"] else [x for x in r["screen_reason"] if x != "pass"]
        if hits:
            reasons.append("relaxed_candidate_matches_exact_formula_COD_structure")
        r["screen_pass"] = bool(r["screen_pass"] and not hits)
        r["screen_reason"] = ["pass"] if r["screen_pass"] else sorted(set(reasons))
    # Persist the complete evidence lineage, including the post-relax COD check.
    temp_checkpoint = CHECKPOINT.with_suffix(".jsonl.tmp")
    temp_checkpoint.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in result_rows) + "\n", encoding="utf-8")
    temp_checkpoint.replace(CHECKPOINT)
    summary = {
        "analysis_date": "2026-10-17",
        "n_candidates": len(result_rows),
        "n_candidate_force_converged": sum(r["generated_relaxed"]["force_converged"] for r in result_rows),
        "n_reference_force_converged": sum(r["source_relaxed_same_model"]["force_converged"] for r in result_rows),
        "n_candidate_relaxations_collapsed_to_source_structure": sum(r["relaxed_candidate_matches_same_formula_published_structure"] for r in result_rows),
        "n_relaxed_candidates_matching_exact_formula_cod_structure": sum(r["relaxed_candidate_matches_cod_structure"] for r in result_rows),
        "n_exact_formula_cod_references_compared": sum(len(v) for v in cod_refs.values()),
        "cod_reference_parse_errors": cod_parse_errors,
        "same_formula_relative_energy_meV_atom": {
            "min": min(r["same_formula_relative_energy_meV_atom"] for r in result_rows),
            "median": float(np.median([r["same_formula_relative_energy_meV_atom"] for r in result_rows])),
            "max": max(r["same_formula_relative_energy_meV_atom"] for r in result_rows),
            "n_candidate_lower_energy_than_source_reference": sum(r["same_formula_relative_energy_meV_atom"] < 0 for r in result_rows),
            "n_within_25_meV_atom": sum(r["same_formula_relative_energy_meV_atom"] <= 25 for r in result_rows),
            "n_within_50_meV_atom": sum(r["same_formula_relative_energy_meV_atom"] <= 50 for r in result_rows),
        },
        "target_gate": {
            "description": "source DFT class=perovskite; source hull <=50 meV/atom; both same-composition CHGNet relaxations force-converged at 0.05 eV/A; generated candidate energy <=50 meV/atom above the same-formula source structure under the same CHGNet model; relaxed candidate remains structurally distinct from the published source structure and all retrieved exact-formula COD references",
            "n_pass": sum(r["screen_pass"] for r in result_rows),
            "passed_formulas": [r["formula"] for r in result_rows if r["screen_pass"]],
        },
        "model": {
            "package": "chgnet",
            "package_version": chgnet_package_version,
            "checkpoint_name": "0.3.0",
            "checkpoint_sha256": sha256(MODEL_PATH),
            "pretraining": "Materials Project trajectory (MPtrj); CHGNet is an ML interatomic potential, not DFT",
            "device": device,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "ase": ase_version,
            "pymatgen": package_version("pymatgen"),
            "force_tolerance_eV_A": FORCE_TOL,
            "max_steps": MAX_STEPS,
            "relax_cell": True,
            "structure_matcher": MATCHER,
        },
        "inputs": {
            "candidate_screen_csv_sha256": sha256(INPUT_CSV),
            "source_poscar_archive_sha256": source_digest,
            "cod_reference_cif_sha256": {p.name: sha256(p) for p in sorted(COD_REFERENCE_DIR.glob("*.cif"))},
            "source_stability_table_sha256": sha256(SOURCE_DIR / "cm4c00576_si_003.xlsx"),
            "candidate_count": len(candidates),
        },
        "interpretation": "Relative energies are model-predicted same-composition comparisons between a PATH-Mat candidate and a published relaxed structure, after both are relaxed using the same CHGNet model. They are not DFT energies, hull distances, or experimental stability measurements. The 50 meV/atom threshold is a computational prescreen gate, not a validated synthesis outcome.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = ["candidate_id", "formula", "same_formula_relative_energy_meV_atom", "candidate_energy_eV_atom", "source_structure_energy_eV_atom", "candidate_force_max_eV_A", "candidate_force_converged", "candidate_optimizer_steps", "source_force_max_eV_A", "source_force_converged", "source_poscar_member", "source_poscar_sha256", "collapsed_to_reference", "relaxed_candidate_matches_cod_structure", "matching_cod_ids_after_relaxation", "source_hull_meV_atom", "source_relaxed_class", "screen_pass", "screen_reason", "relaxed_candidate_cif", "relaxed_candidate_cif_sha256"]
    with (OUT / "candidate_chgnet_screen.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in result_rows:
            w.writerow({
                "candidate_id": r["candidate_id"], "formula": r["formula"],
                "same_formula_relative_energy_meV_atom": f"{r['same_formula_relative_energy_meV_atom']:.4f}",
                "candidate_energy_eV_atom": f"{r['generated_relaxed']['energy_eV_atom']:.8f}",
                "source_structure_energy_eV_atom": f"{r['source_relaxed_same_model']['energy_eV_atom']:.8f}",
                "candidate_force_max_eV_A": f"{r['generated_relaxed']['final_max_force_eV_A']:.6f}",
                "candidate_force_converged": r["generated_relaxed"]["force_converged"],
                "candidate_optimizer_steps": r["generated_relaxed"]["optimizer_steps_recorded"],
                "source_force_max_eV_A": f"{r['source_relaxed_same_model']['final_max_force_eV_A']:.6f}",
                "source_force_converged": r["source_relaxed_same_model"]["force_converged"],
                "source_poscar_member": r["source_poscar_member"], "source_poscar_sha256": r["source_poscar_sha256"],
                "collapsed_to_reference": r["relaxed_candidate_matches_same_formula_published_structure"],
                "relaxed_candidate_matches_cod_structure": r["relaxed_candidate_matches_cod_structure"],
                "matching_cod_ids_after_relaxation": ";".join(r["matching_cod_ids_after_relaxation"]),
                "source_hull_meV_atom": f"{r['source_published_hull_eV_atom']*1000:.3f}",
                "source_relaxed_class": r["source_published_relaxed_class"],
                "screen_pass": r["screen_pass"], "screen_reason": ";".join(r["screen_reason"]),
                "relaxed_candidate_cif": r["generated_relaxed"]["relaxed_cif"],
                "relaxed_candidate_cif_sha256": r["generated_relaxed"]["relaxed_cif_sha256"],
            })
    passing = {r["candidate_id"] for r in result_rows if r["screen_pass"]}
    route_rows = [r for r in read_csv(ROUTE_CSV) if r["candidate_id"] in passing]
    if route_rows:
        with (OUT / "pathmat_routes_for_chgnet_pass_candidates.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(route_rows[0]))
            w.writeheader(); w.writerows(route_rows)
    report = [
        "# Exact-structure CHGNet prescreen of PATH-Mat candidate materials", "",
        f"- Candidate structures evaluated: **{summary['n_candidates']}**.",
        f"- Candidate CIF relaxations converged at max force ≤{FORCE_TOL} eV/Å: **{summary['n_candidate_force_converged']}/{summary['n_candidates']}**.",
        f"- Same-formula published-structure CHGNet relaxations converged: **{summary['n_reference_force_converged']}/{summary['n_candidates']}**.",
        f"- Candidate relaxations structurally matching the source structure after relaxation: **{summary['n_candidate_relaxations_collapsed_to_source_structure']}/{summary['n_candidates']}**.",
        f"- Relaxed candidates matching a retrieved exact-formula COD structure: **{summary['n_relaxed_candidates_matching_exact_formula_cod_structure']}/{summary['n_candidates']}** ({summary['n_exact_formula_cod_references_compared']} structures compared).",
        f"- Candidate/reference CHGNet relative energies (same formula): **{summary['same_formula_relative_energy_meV_atom']['min']:.1f} to {summary['same_formula_relative_energy_meV_atom']['max']:.1f} meV/atom** (median {summary['same_formula_relative_energy_meV_atom']['median']:.1f}).",
        f"- Relative energy ≤25 meV/atom: **{summary['same_formula_relative_energy_meV_atom']['n_within_25_meV_atom']}/{summary['n_candidates']}**; ≤50 meV/atom: **{summary['same_formula_relative_energy_meV_atom']['n_within_50_meV_atom']}/{summary['n_candidates']}**.",
        f"- Pass the combined composition/relaxation/relative-energy/structure-distinctness gate: **{summary['target_gate']['n_pass']}/{summary['n_candidates']}**; linked route rows **{len(route_rows)}** (model abstention flags retained).", "",
        "**Interpretation.** CHGNet is used as a pretrained machine-learning interatomic potential to relax the generated CIF and a same-formula published structure under identical settings. The within-composition energy difference provides an exact-structure prescreen and can reveal collapse to a known reference. It is not DFT or an energy-above-hull calculation, and its ≤50 meV/atom gate does not establish experimental synthesizability.", "",
        f"Checkpoint SHA-256: `{sha256(MODEL_PATH)}`. Source archive SHA-256: `{source_digest}`.",
    ]
    (OUT / "CHGNET_EXACT_STRUCTURE_SCREEN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "summary": summary, "route_rows": len(route_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
