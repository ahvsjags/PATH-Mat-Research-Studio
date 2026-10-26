#!/usr/bin/env python3
"""TensorNet single-point audit on the frozen common geometry set."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matgl
import numpy as np
import torch
from matgl.ext.ase import PESCalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
V259 = ROOT.parent / "PATH-Mat_General_Order_Effect_V259" / "release_candidate" / "material_generation"
CHG_RUN = V259 / "runs" / "chgnet_exact_structure_screen_20261017_v1"
OUT = ROOT / "results" / "RELAXED_GEOMETRY_CROSS_POTENTIAL_V260"
REFERENCE_CIFS = OUT / "reference_relaxed_cifs"
CHG_CSV = CHG_RUN / "candidate_chgnet_screen.csv"
M3_CSV = OUT / "CROSS_POTENTIAL_COMMON_GEOMETRY_V260.csv"
M3_CACHE = ROOT / ".cache_home" / ".cache" / "matgl" / "models--materialyze--TensorNet-PES-MatPES-PBE-2025.2"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    chg_rows = list(csv.DictReader(CHG_CSV.open(newline="", encoding="utf-8-sig")))
    m3_rows = {r["candidate_id"]: r for r in csv.DictReader(M3_CSV.open(newline="", encoding="utf-8-sig"))}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pot = matgl.load_model("TensorNet-PES-MatPES-PBE-2025.2")
    pot.to(device)
    calc = PESCalculator(pot)
    rows = []
    for i, chg in enumerate(chg_rows, 1):
        cid, formula = chg["candidate_id"], chg["formula"]
        candidate_cif = CHG_RUN / chg["relaxed_candidate_cif"]
        reference_cif = REFERENCE_CIFS / f"{cid[:12]}_{formula}_source_CHGNet_relaxed.cif"
        if sha256(candidate_cif) != chg["relaxed_candidate_cif_sha256"]:
            raise RuntimeError(f"Candidate relaxed CIF hash mismatch: {formula}")
        if not reference_cif.is_file():
            raise FileNotFoundError(reference_cif)
        candidate = Structure.from_file(str(candidate_cif))
        reference = Structure.from_file(str(reference_cif))
        scores = []
        for s in (candidate, reference):
            atoms = AseAtomsAdaptor.get_atoms(s)
            atoms.calc = calc
            e = float(atoms.get_potential_energy()) / len(atoms)
            fmax = float(np.linalg.norm(np.asarray(atoms.get_forces()), axis=1).max())
            scores.append((e, fmax))
        delta = (scores[0][0] - scores[1][0]) * 1000.0
        m3 = m3_rows[cid]
        rows.append({
            "candidate_id": cid, "formula": formula,
            "candidate_relaxed_cif_sha256": sha256(candidate_cif),
            "reference_relaxed_cif_sha256": sha256(reference_cif),
            "tensornet_singlepoint_candidate_eV_atom": scores[0][0],
            "tensornet_singlepoint_reference_eV_atom": scores[1][0],
            "tensornet_singlepoint_delta_meV_atom": delta,
            "tensornet_candidate_force_on_chgnet_relaxed_geometry_eV_A": scores[0][1],
            "tensornet_reference_force_on_chgnet_relaxed_geometry_eV_A": scores[1][1],
            "tensornet_delta_le_50_meV": delta <= 50.0,
            "chgnet_singlepoint_delta_meV_atom": float(m3["chgnet_singlepoint_delta_meV_atom"]),
            "m3gnet_singlepoint_delta_meV_atom": float(m3["m3gnet_singlepoint_delta_meV_atom"]),
            "chgnet_relaxed_gate_le_50_meV": str(chg["screen_pass"]).lower() == "true",
        })
        print(f"{i}/{len(chg_rows)} {formula} Δ={delta:.2f} meV/atom", flush=True)
    out_csv = OUT / "TENSORNET_COMMON_GEOMETRY_V260.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    t = np.array([r["tensornet_singlepoint_delta_meV_atom"] for r in rows], float)
    c = np.array([r["chgnet_singlepoint_delta_meV_atom"] for r in rows], float)
    m = np.array([r["m3gnet_singlepoint_delta_meV_atom"] for r in rows], float)
    sp_c, kt_c = spearmanr(t, c), kendalltau(t, c)
    sp_m, kt_m = spearmanr(t, m), kendalltau(t, m)
    t_gate = np.array([r["tensornet_delta_le_50_meV"] for r in rows], bool)
    c_gate = c <= 50.0
    m_gate = m <= 50.0
    summary = {
        "analysis_id": "V260_TENSORNET_COMMON_GEOMETRY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_candidate_pairs": len(rows),
        "tensornet_delta_meV_atom": {"median": float(np.median(t)), "min": float(t.min()), "max": float(t.max()), "n_le_50": int(t_gate.sum()), "n_negative": int(np.sum(t < 0))},
        "n_chgnet_relaxed_gate_pass": int(sum(str(r["screen_pass"]).lower() == "true" for r in chg_rows)),
        "n_tensornet_static_and_chgnet_relaxed_gate_overlap": int(sum(t_gate[i] and str(chg_rows[i]["screen_pass"]).lower() == "true" for i in range(len(rows)))),
        "n_all_three_mlips_static_deltas_le_50": int(np.sum(t_gate & c_gate & m_gate)),
        "rank_concordance": {
            "TensorNet_vs_CHGNet_singlepoint_same_geometry": {"spearman_rho": float(sp_c.statistic), "spearman_p_descriptive": float(sp_c.pvalue), "kendall_tau": float(kt_c.statistic), "kendall_p_descriptive": float(kt_c.pvalue)},
            "TensorNet_vs_M3GNet_singlepoint_same_geometry": {"spearman_rho": float(sp_m.statistic), "spearman_p_descriptive": float(sp_m.pvalue), "kendall_tau": float(kt_m.statistic), "kendall_p_descriptive": float(kt_m.pvalue)},
        },
        "model": {"name": "TensorNet-PES-MatPES-PBE-2025.2", "matgl_version": matgl.__version__, "repository": "materialyze/TensorNet-PES-MatPES-PBE-2025.2", "snapshot_revision": (M3_CACHE / "refs" / "main").read_text(encoding="utf-8").strip(), "state_pt_sha256": sha256(M3_CACHE / "snapshots" / (M3_CACHE / "refs" / "main").read_text(encoding="utf-8").strip() / "state.pt"), "device": device, "torch": torch.__version__, "pretraining": "MatPES-PBE; ML interatomic potential"},
        "geometry": "Exact CHGNet-relaxed candidate and same-formula reference CIFs; unchanged geometries used for CHGNet, M3GNet and TensorNet single-point predictions.",
        "interpretation": "Cross-architecture MLIP energy agreement is computational robustness evidence only. It is not DFT, hull stability, phonon stability, or experimental synthesis feasibility. Forces are reported because the common geometries were not relaxed under TensorNet; single-point energy thresholds are sensitivity filters only.",
        "input_sha256": {"chgnet_screen_csv": sha256(CHG_CSV), "m3gnet_crosscheck_csv": sha256(M3_CSV)},
    }
    (OUT / "TENSORNET_COMMON_GEOMETRY_V260.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
