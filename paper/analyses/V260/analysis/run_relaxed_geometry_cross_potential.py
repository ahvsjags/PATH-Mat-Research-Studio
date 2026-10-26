#!/usr/bin/env python3
"""Cross-potential energy audit on a common, CHGNet-relaxed geometry set.

The newly generated and published same-formula structures are first relaxed
with the same pinned CHGNet setup. CHGNet and M3GNet then independently score
the exact same final geometries. This separates energy-model agreement from
the optimizer used to create the geometries. It is still an MLIP analysis, not
DFT, phonon stability, or a synthesis-feasibility test.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
V259 = ROOT.parent / "PATH-Mat_General_Order_Effect_V259"
MODULE = V259 / "release_candidate" / "material_generation"
RUN = MODULE / "runs" / "local_pathmat_double_perovskite_20261017_v2"
DFT = MODULE / "runs" / "published_dft_stability_20261017_v1"
CHG_RUN = MODULE / "runs" / "chgnet_exact_structure_screen_20261017_v1"
INPUT = DFT / "pathmat_candidate_published_dft_stability.csv"
CHG_CSV = CHG_RUN / "candidate_chgnet_screen.csv"
CHG_CHECKPOINT = MODULE / "runtime" / "chgnet_0_4_2" / "chgnet" / "pretrained" / "0.3.0" / "chgnet_0.3.0_e29f68s314m37.pth.tar"
POSCAR_ZIP = MODULE / "data" / "references" / "acs_chem_mater_2024_double_perovskite" / "cm4c00576_si_002_structures.zip"
M3_MODEL_CACHE = ROOT / ".cache_home" / ".cache" / "matgl" / "models--materialyze--M3GNet-PES-MatPES-PBE-2025.2"
OUT = ROOT / "results" / "RELAXED_GEOMETRY_CROSS_POTENTIAL_V260"
REF_DIR = OUT / "reference_relaxed_cifs"
CHECKPOINT = OUT / "reference_relaxation_checkpoint.jsonl"
OUT.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)
FORCE_TOL = 0.05
MAX_STEPS = 500
ENERGY_GATE = 50.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def poscar_for_formula(names: list[str], formula: str) -> str:
    hits = [n for n in names if n.startswith("POSCAR_FILE/") and n.rsplit("/", 1)[-1].endswith("-" + formula)]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one same-formula source POSCAR for {formula}, got {len(hits)}")
    return hits[0]


def score(structure: Structure, calculator) -> tuple[float, float]:
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy()) / len(atoms)
    forces = np.asarray(atoms.get_forces(), dtype=float)
    fmax = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
    return energy, fmax


def main() -> None:
    for p in (INPUT, CHG_CSV, CHG_CHECKPOINT, POSCAR_ZIP):
        if not p.is_file():
            raise FileNotFoundError(p)
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Converting a tensor with requires_grad=True.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*logm result may be inaccurate.*")
    candidates = read_csv(INPUT)
    chg_screen = read_csv(CHG_CSV)
    chg_by_id = {r["candidate_id"]: r for r in chg_screen}
    existing = {}
    if CHECKPOINT.is_file():
        for line in CHECKPOINT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["candidate_id"]] = r

    # Reuse precisely the original CHGNet 0.3.0 checkpoint and its bundled runtime.
    runtime = MODULE / "runtime" / "chgnet_0_4_2"
    sys.path.insert(0, str(runtime))
    from chgnet.model import CHGNet, StructOptimizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20261017)
    if device == "cuda":
        torch.cuda.manual_seed_all(20261017)
    chg_model = CHGNet.load(model_name="0.3.0", use_device=device, verbose=False)
    optimizer = StructOptimizer(model=chg_model, use_device=device)
    with zipfile.ZipFile(POSCAR_ZIP) as zf:
        names = zf.namelist()
        for i, row in enumerate(candidates, 1):
            cid, formula = row["candidate_id"], row["formula"]
            if cid in existing:
                continue
            member = poscar_for_formula(names, formula)
            reference = Structure.from_str(zf.read(member).decode("utf-8"), fmt="poscar")
            result = optimizer.relax(reference, fmax=FORCE_TOL, steps=MAX_STEPS, relax_cell=True, verbose=False, assign_magmoms=True)
            final = result["final_structure"]
            pred = chg_model.predict_structure(final)
            forces = np.asarray(pred["f"], dtype=float)
            fmax = float(np.linalg.norm(forces, axis=1).max())
            energy = float(np.asarray(pred["e"]).reshape(-1)[0]) / len(final)
            cif = REF_DIR / f"{cid[:12]}_{formula}_source_CHGNet_relaxed.cif"
            final.to(filename=str(cif), fmt="cif")
            saved = {"candidate_id": cid, "formula": formula, "source_poscar_member": member,
                     "source_poscar_sha256": hashlib.sha256(zf.read(member)).hexdigest(),
                     "source_relaxed_cif": str(cif.relative_to(ROOT)).replace("\\", "/"),
                     "source_relaxed_cif_sha256": sha256(cif), "chgnet_source_energy_eV_atom": energy,
                     "chgnet_source_max_force_eV_A": fmax, "force_converged": fmax <= FORCE_TOL + 1e-6,
                     "steps_recorded": len(result["trajectory"].energies)}
            existing[cid] = saved
            with CHECKPOINT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(saved, ensure_ascii=False) + "\n")
            print(f"reference {i}/{len(candidates)} {formula} converged={saved['force_converged']}", flush=True)

    del optimizer, chg_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Both MLIPs now score exactly the same candidate/reference geometries.
    import matgl
    from matgl.ext.ase import PESCalculator

    potential = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
    potential.to(device)
    m3_calculator = PESCalculator(potential)
    # Load CHGNet again solely for single-point scoring on the frozen common geometries.
    chg_model = CHGNet.load(model_name="0.3.0", use_device=device, verbose=False)
    chg_rows = []
    for i, row in enumerate(candidates, 1):
        cid, formula = row["candidate_id"], row["formula"]
        chg = chg_by_id[cid]
        cand_cif = CHG_RUN / chg["relaxed_candidate_cif"]
        if sha256(cand_cif) != chg["relaxed_candidate_cif_sha256"]:
            raise RuntimeError(f"CHGNet candidate CIF hash mismatch: {formula}")
        candidate = Structure.from_file(str(cand_cif))
        ref_meta = existing[cid]
        ref_cif = ROOT / ref_meta["source_relaxed_cif"]
        if sha256(ref_cif) != ref_meta["source_relaxed_cif_sha256"]:
            raise RuntimeError(f"source relaxed CIF hash mismatch: {formula}")
        reference = Structure.from_file(str(ref_cif))
        if candidate.composition.element_composition.reduced_composition != reference.composition.element_composition.reduced_composition:
            raise RuntimeError(f"relaxed pair formula mismatch: {formula}")
        e_mc, f_mc = score(candidate, m3_calculator)
        e_mr, f_mr = score(reference, m3_calculator)
        chg_cand = float(chg["candidate_energy_eV_atom"])
        chg_ref = float(chg["source_structure_energy_eV_atom"])
        delta_chg = (chg_cand - chg_ref) * 1000
        delta_m3 = (e_mc - e_mr) * 1000
        chg_single = chg_model.predict_structure(candidate)
        ref_single = chg_model.predict_structure(reference)
        # Use stored CHGNet relaxed energies for the gate; these are the model's
        # matched-relaxation values. Single-point scores above are for model concordance.
        chg_rows.append({
            "candidate_id": cid, "formula": formula,
            "candidate_relaxed_cif_sha256": sha256(cand_cif), "reference_relaxed_cif_sha256": sha256(ref_cif),
            "chgnet_relaxed_candidate_minus_reference_meV_atom": delta_chg,
            # CHGNet.predict_structure returns e in eV/atom (not total eV).
            "chgnet_singlepoint_on_common_geometry_candidate_eV_atom": float(np.asarray(chg_single["e"]).reshape(-1)[0]),
            "chgnet_singlepoint_on_common_geometry_reference_eV_atom": float(np.asarray(ref_single["e"]).reshape(-1)[0]),
            "chgnet_singlepoint_delta_meV_atom": (float(np.asarray(chg_single["e"]).reshape(-1)[0]) - float(np.asarray(ref_single["e"]).reshape(-1)[0])) * 1000,
            "m3gnet_singlepoint_delta_meV_atom": delta_m3,
            "m3gnet_candidate_max_force_eV_A_on_chgnet_relaxed_geometry": f_mc,
            "m3gnet_reference_max_force_eV_A_on_chgnet_relaxed_geometry": f_mr,
            "m3gnet_singlepoint_gate_le_50_meV": delta_m3 <= ENERGY_GATE,
            "chgnet_relaxed_gate_le_50_meV": delta_chg <= ENERGY_GATE,
            "both_singlepoint_energy_delta_le_50_meV": (float(np.asarray(chg_single["e"]).reshape(-1)[0]) - float(np.asarray(ref_single["e"]).reshape(-1)[0])) * 1000 <= ENERGY_GATE and delta_m3 <= ENERGY_GATE,
        })
        print(f"score {i}/{len(candidates)} {formula} CHGNet={delta_chg:.2f} M3GNet={delta_m3:.2f} meV/atom", flush=True)

    out_csv = OUT / "CROSS_POTENTIAL_COMMON_GEOMETRY_V260.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(chg_rows[0]))
        w.writeheader(); w.writerows(chg_rows)
    d_chg = np.array([r["chgnet_singlepoint_delta_meV_atom"] for r in chg_rows], float)
    d_m3 = np.array([r["m3gnet_singlepoint_delta_meV_atom"] for r in chg_rows], float)
    sp, kt = spearmanr(d_chg, d_m3), kendalltau(d_chg, d_m3)
    n_both = sum(r["both_singlepoint_energy_delta_le_50_meV"] for r in chg_rows)
    n_chg_relax = sum(r["chgnet_relaxed_gate_le_50_meV"] for r in chg_rows)
    n_m3 = sum(r["m3gnet_singlepoint_gate_le_50_meV"] for r in chg_rows)
    summary = {
        "analysis_id": "V260_CROSS_POTENTIAL_COMMON_GEOMETRY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_candidate_pairs": len(chg_rows), "n_chgnet_reference_relaxation_converged": sum(x["force_converged"] for x in existing.values()),
        "chgnet_relax_gate_le_50_n": int(n_chg_relax), "m3gnet_common_geometry_singlepoint_le_50_n": int(n_m3),
        "common_geometry_chgnet_singlepoint_le_50_n": int(sum(r["chgnet_singlepoint_delta_meV_atom"] <= ENERGY_GATE for r in chg_rows)),
        "both_mlip_singlepoint_delta_le_50_n": int(n_both),
        "chgnet_relaxed_gate_and_m3gnet_singlepoint_gate_overlap_n": int(sum(r["chgnet_relaxed_gate_le_50_meV"] and r["m3gnet_singlepoint_gate_le_50_meV"] for r in chg_rows)),
        "cross_potential_singlepoint_delta_rank_concordance": {"spearman_rho": float(sp.statistic), "spearman_p_descriptive": float(sp.pvalue), "kendall_tau": float(kt.statistic), "kendall_p_descriptive": float(kt.pvalue)},
        "m3gnet_singlepoint_delta_meV_atom": {"median": float(np.median(d_m3)), "min": float(d_m3.min()), "max": float(d_m3.max()), "n_negative": int(np.sum(d_m3 < 0))},
        "chgnet_singlepoint_delta_meV_atom": {"median": float(np.median(d_chg)), "min": float(d_chg.min()), "max": float(d_chg.max()), "n_negative": int(np.sum(d_chg < 0))},
        "models": {"CHGNet": {"checkpoint_sha256": sha256(CHG_CHECKPOINT), "checkpoint": "0.3.0, MPtrj-pretrained"},
                   "M3GNet": {"name": "M3GNet-PES-MatPES-PBE-2025.2", "matgl_version": matgl.__version__, "repository": "materialyze/M3GNet-PES-MatPES-PBE-2025.2"}},
        "settings": {"geometry_source": "candidate and reference structures relaxed using the same CHGNet 0.3.0 workflow; both MLIPs evaluate identical frozen geometries", "comparison": "same-formula candidate minus published relaxed reference, eV/atom", "gate_meV_atom": ENERGY_GATE, "CHGNet_relax_fmax_eV_A": FORCE_TOL, "CHGNet_relax_max_steps": MAX_STEPS},
        "input_digests": {"candidate_table_sha256": sha256(INPUT), "source_poscar_archive_sha256": sha256(POSCAR_ZIP), "chgnet_screen_csv_sha256": sha256(CHG_CSV), "m3gnet_state_sha256": sha256(M3_MODEL_CACHE / "snapshots" / (M3_MODEL_CACHE / "refs" / "main").read_text().strip() / "state.pt")},
        "interpretation": "This cross-potential agreement is an additional computational robustness check. M3GNet single-point energy on CHGNet-relaxed geometries is not an independent relaxation or DFT calculation; neither MLIP result establishes hull stability, phonon stability, chemical synthesizability, or experimental realization. The 50 meV/atom gate remains a model-based prescreen.",
    }
    (OUT / "CROSS_POTENTIAL_COMMON_GEOMETRY_V260.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_candidate_pairs", "n_chgnet_reference_relaxation_converged", "chgnet_relax_gate_le_50_n", "m3gnet_common_geometry_singlepoint_le_50_n", "both_mlip_singlepoint_delta_le_50_n", "chgnet_relaxed_gate_and_m3gnet_singlepoint_gate_overlap_n", "cross_potential_singlepoint_delta_rank_concordance", "m3gnet_singlepoint_delta_meV_atom")}, indent=2))


if __name__ == "__main__":
    main()
