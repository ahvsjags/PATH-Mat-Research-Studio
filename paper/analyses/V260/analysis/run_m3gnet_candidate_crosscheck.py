#!/usr/bin/env python3
"""Independent static M3GNet cross-check of generated double-perovskite CIFs.

This is intentionally a single-point energy/force comparison on the frozen
input geometries. It is not DFT, not a geometry-relaxation result, and not a
claim of synthesizability. Candidate and same-formula published POSCAR are
scored with the same potential and per-atom normalization.
"""
from __future__ import annotations

import csv
import hashlib
import json
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matgl
import numpy as np
import torch
from matgl.ext.ase import PESCalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parents[1]
V259 = ROOT.parent / "PATH-Mat_General_Order_Effect_V259"
MODULE = V259 / "release_candidate" / "material_generation"
RUN = MODULE / "runs" / "local_pathmat_double_perovskite_20261017_v2"
DFT = MODULE / "runs" / "published_dft_stability_20261017_v1"
INPUT = DFT / "pathmat_candidate_published_dft_stability.csv"
POSCAR_ZIP = MODULE / "data" / "references" / "acs_chem_mater_2024_double_perovskite" / "cm4c00576_si_002_structures.zip"
CHGNET = MODULE / "runs" / "chgnet_exact_structure_screen_20261017_v1" / "candidate_chgnet_screen.csv"
MODEL_CACHE = ROOT / ".cache_home" / ".cache" / "matgl" / "models--materialyze--M3GNet-PES-MatPES-PBE-2025.2"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def formula_poscar(names: list[str], formula: str) -> str:
    hits = [x for x in names if x.startswith("POSCAR_FILE/") and x.rsplit("/", 1)[-1].endswith("-" + formula)]
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one source POSCAR for {formula}; found {len(hits)}")
    return hits[0]


def score(structure: Structure, calculator: PESCalculator) -> tuple[float, float, int]:
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy()) / len(atoms)
    forces = np.asarray(atoms.get_forces(), dtype=float)
    fmax = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
    return energy, fmax, len(atoms)


def main() -> None:
    for p in (INPUT, POSCAR_ZIP, CHGNET):
        if not p.is_file():
            raise FileNotFoundError(p)
    warnings.filterwarnings("ignore", message=".*set_calculator.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*logm result may be inaccurate.*", category=RuntimeWarning)
    torch.manual_seed(20261017)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    potential = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
    potential.to(device)
    calculator = PESCalculator(potential)
    source_rows = read_csv(INPUT)
    chgnet_by_formula = {r["formula"]: r for r in read_csv(CHGNET)}
    results: list[dict] = []
    with zipfile.ZipFile(POSCAR_ZIP) as zf:
        members = zf.namelist()
        for i, row in enumerate(source_rows, 1):
            formula = row["formula"]
            generated_cif = RUN / row["candidate_cif"]
            if sha256(generated_cif) != row["candidate_cif_sha256"]:
                raise RuntimeError(f"Candidate CIF hash mismatch: {formula}")
            candidate = Structure.from_file(str(generated_cif))
            member = formula_poscar(members, formula)
            reference = Structure.from_str(zf.read(member).decode("utf-8"), fmt="poscar")
            if candidate.composition.element_composition.reduced_composition != reference.composition.element_composition.reduced_composition:
                raise RuntimeError(f"Formula mismatch: {formula}")
            try:
                e_c, f_c, n_c = score(candidate, calculator)
                e_r, f_r, n_r = score(reference, calculator)
                delta = 1000.0 * (e_c - e_r)
                chg = chgnet_by_formula.get(formula, {})
                row_out = {
                    "candidate_id": row["candidate_id"], "formula": formula,
                    "candidate_cif_sha256": sha256(generated_cif), "source_poscar_member": member,
                    "candidate_atom_count": n_c, "reference_atom_count": n_r,
                    "m3gnet_candidate_singlepoint_eV_atom": e_c,
                    "m3gnet_reference_singlepoint_eV_atom": e_r,
                    "m3gnet_candidate_minus_reference_meV_atom": delta,
                    "m3gnet_candidate_max_force_eV_A": f_c,
                    "m3gnet_reference_max_force_eV_A": f_r,
                    "m3gnet_static_gate_le_50_meV": delta <= 50.0,
                    "chgnet_candidate_minus_reference_meV_atom": float(chg["same_formula_relative_energy_meV_atom"]) if chg else None,
                    "chgnet_gate_le_50_meV": str(chg.get("screen_pass", "")).lower() == "true" if chg else None,
                    "model_relaxation_performed": False,
                    "error": "",
                }
            except Exception as exc:
                row_out = {"candidate_id": row["candidate_id"], "formula": formula,
                           "candidate_cif_sha256": sha256(generated_cif), "source_poscar_member": member,
                           "error": f"{type(exc).__name__}: {exc}"}
            results.append(row_out)
            print(f"{i}/{len(source_rows)} {formula} " + (f"delta={row_out['m3gnet_candidate_minus_reference_meV_atom']:.2f} meV/atom" if not row_out.get("error") else row_out["error"]), flush=True)

    out_csv = OUT / "M3GNET_STATIC_CANDIDATE_CROSSCHECK_V260.csv"
    fields = list(dict.fromkeys(k for r in results for k in r))
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    valid = [r for r in results if not r.get("error")]
    delta_m = np.array([r["m3gnet_candidate_minus_reference_meV_atom"] for r in valid], float)
    chg_delta = np.array([r["chgnet_candidate_minus_reference_meV_atom"] for r in valid], float)
    chg_gate = np.array([r["chgnet_gate_le_50_meV"] for r in valid], bool)
    m3_gate = np.array([r["m3gnet_static_gate_le_50_meV"] for r in valid], bool)
    spearman = spearmanr(delta_m, chg_delta) if len(valid) > 2 else None
    kt = kendalltau(delta_m, chg_delta) if len(valid) > 2 else None
    model_ref = MODEL_CACHE / "refs" / "main"
    snapshot = model_ref.read_text(encoding="utf-8").strip() if model_ref.is_file() else None
    summary = {
        "analysis_id": "V260_M3GNET_STATIC_CROSSCHECK",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_candidates": len(results), "n_successful_singlepoint_pairs": len(valid),
        "n_failed": len(results) - len(valid),
        "m3gnet_candidate_minus_reference_meV_atom": {
            "median": float(np.median(delta_m)) if len(valid) else None,
            "min": float(delta_m.min()) if len(valid) else None,
            "max": float(delta_m.max()) if len(valid) else None,
            "n_le_25_meV_atom": int(np.sum(delta_m <= 25.0)),
            "n_le_50_meV_atom": int(np.sum(m3_gate)),
        },
        "chgnet_relaxed_screen_pass_n": int(np.sum(chg_gate)),
        "m3gnet_static_le_50_meV_n": int(np.sum(m3_gate)),
        "overlap_of_chgnet_pass_and_m3gnet_static_le_50_n": int(np.sum(chg_gate & m3_gate)),
        "m3gnet_static_threshold_is_sensitivity_gate_only": True,
        "rank_concordance_vs_chgnet_energy_differences": {
            "spearman_rho": float(spearman.statistic) if spearman else None,
            "spearman_p_descriptive": float(spearman.pvalue) if spearman else None,
            "kendall_tau": float(kt.statistic) if kt else None,
            "kendall_p_descriptive": float(kt.pvalue) if kt else None,
        },
        "model": {
            "name": "M3GNet-PES-MatPES-PBE-2025.2", "matgl_version": matgl.__version__,
            "model_repository": "materialyze/M3GNet-PES-MatPES-PBE-2025.2",
            "snapshot_revision": snapshot,
            "state_pt_sha256": sha256(MODEL_CACHE / "snapshots" / snapshot / "state.pt") if snapshot and (MODEL_CACHE / "snapshots" / snapshot / "state.pt").is_file() else None,
            "device": device, "torch_version": torch.__version__,
            "data_basis": "MatPES-PBE pretrained machine-learning interatomic potential; not first-principles DFT",
        },
        "settings": {"geometry": "original generated CIF and source POSCAR; no relaxation", "energy_normalization": "eV/atom", "force_tolerance_eV_A": 0.05, "threshold_meV_atom": 50.0, "comparison": "same-formula candidate minus published structure, same model"},
        "inputs": {"candidate_table_sha256": sha256(INPUT), "published_poscar_archive_sha256": sha256(POSCAR_ZIP), "chgnet_screen_sha256": sha256(CHGNET), "n_candidates_with_source_reference": len(source_rows)},
        "interpretation": "Independent single-point MLIP cross-check on the frozen starting geometries. The M3GNet values are not relaxed energies, DFT energies, hull distances, phonon stability, synthesis feasibility, or experimental validation. The matched 50 meV/atom count is a sensitivity comparison only; it does not replace the CHGNet relaxation gate.",
    }
    (OUT / "M3GNET_STATIC_CANDIDATE_CROSSCHECK_V260.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_candidates", "n_successful_singlepoint_pairs", "n_failed", "m3gnet_candidate_minus_reference_meV_atom", "chgnet_relaxed_screen_pass_n", "m3gnet_static_le_50_meV_n", "overlap_of_chgnet_pass_and_m3gnet_static_le_50_n", "rank_concordance_vs_chgnet_energy_differences")}, indent=2))


if __name__ == "__main__":
    main()
