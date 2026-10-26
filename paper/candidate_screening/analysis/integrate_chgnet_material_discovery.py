#!/usr/bin/env python3
"""Integrate CHGNet exact-structure results into the V259 manuscript and SI."""
from __future__ import annotations

import csv
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
PROJECT = MODULE.parent.parent
SUBMISSION = PROJECT / "submission_final"
MANUSCRIPT = SUBMISSION / "PATH_Mat_Manuscript_Submission_V259.md"
SI = SUBMISSION / "PATH_Mat_Supporting_Information_Submission_V259.md"
RUN = MODULE / "runs" / "chgnet_exact_structure_screen_20261017_v1"
SUMMARY = RUN / "summary.json"
RESULTS = RUN / "candidate_chgnet_screen.csv"
ROUTES = RUN / "pathmat_routes_for_chgnet_pass_candidates.csv"
SECTION_START = "### 15J. CHGNet relaxation and exact-structure candidate screening"

REF = ("91. Deng, B. et al. CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. "
       "Nature Machine Intelligence 5, 1031–1041 (2023). https://doi.org/10.1038/s42256-023-00716-3")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def make_si_section() -> tuple[str, str]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = read_csv(RESULTS)
    routes = read_csv(ROUTES)
    if len(rows) != 43 or len(routes) != 72:
        raise RuntimeError(f"Unexpected CHGNet results: {len(rows)} candidates, {len(routes)} routes")
    if sum(r["screen_pass"] == "True" for r in rows) != summary["target_gate"]["n_pass"]:
        raise RuntimeError("Candidate table and summary pass counts disagree")
    lines = [
        SECTION_START,
        "",
        ("We relaxed each of the 43 PATH-Mat-generated 40-atom CIFs and its same-formula published POSCAR using the same CHGNet 0.3.0 MPtrj-pretrained model, allowing both cell and atomic positions to relax (maximum force 0.05 eV Å⁻¹; maximum 500 optimizer steps) [91]. All 86 relaxations met the force criterion. Energies are compared only within the same reduced composition; ΔE is the relaxed candidate energy minus the relaxed published-structure energy. The median ΔE was "
         f"{summary['same_formula_relative_energy_meV_atom']['median']:.1f} meV/atom (range {summary['same_formula_relative_energy_meV_atom']['min']:.1f} to {summary['same_formula_relative_energy_meV_atom']['max']:.1f}); "
         f"{summary['same_formula_relative_energy_meV_atom']['n_within_25_meV_atom']}/43 candidates fell within 25 meV/atom of their comparator."),
        "",
        (f"After relaxation, none of the 43 structures matched a same-formula published POSCAR, and none matched any of the {summary['n_exact_formula_cod_references_compared']} retrieved COD structures with matching formulas. "
         f"A stricter target gate—source DFT relaxed class perovskite, source hull distance ≤50 meV/atom, force convergence for both relaxations, candidate ΔE ≤50 meV/atom, and structural distinction from the source and COD references—retained {summary['target_gate']['n_pass']} candidates. "
         "Published hull distances describe the reference study’s relaxed structures, not the generated candidates; CHGNet energies are a matched machine-learning-potential prescreen, not DFT energies, energy-above-hull values, or experimental stability measurements."),
        "",
        "**Supplementary Table S118 | CHGNet-relaxed candidate structures and same-composition energy screen.** ΔE is the CHGNet relaxed candidate energy minus that of the same-formula published POSCAR, evaluated with the same model. Source hull distance and source structural class refer to the published DFT study. The gate is a computational prioritization screen.",
        "",
        "| Candidate ID | Formula | ΔE (meV/atom) | Candidate final force (eV/Å) | Source hull (meV/atom) | Source class | Matched source POSCAR after relaxation | Matched COD structure after relaxation | Gate |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x["formula"]):
        lines.append(
            f"| `{r['candidate_id'][:12]}` | {r['formula']} | {float(r['same_formula_relative_energy_meV_atom']):.2f} | "
            f"{float(r['candidate_force_max_eV_A']):.4f} | {float(r['source_hull_meV_atom']):.1f} | {r['source_relaxed_class']} | "
            f"{'yes' if r['collapsed_to_reference']=='True' else 'no'} | "
            f"{('yes: ' + r['matching_cod_ids_after_relaxation']) if r['relaxed_candidate_matches_cod_structure']=='True' else 'no'} | "
            f"{'pass' if r['screen_pass']=='True' else 'fail'} |"
        )
    lines += [
        "",
        "**Supplementary Table S119 | PATH-Mat route hypotheses for CHGNet-screened candidate structures.** The table contains three balanced schedule variants for each of the 24 candidates passing Table S118. The frozen PATH-Mat route model abstained on all 72 records because these candidate compositions are outside its source support; displayed scores remain pre-calibration ranking signals and are not synthesis-success probabilities.",
        "",
        "| Candidate ID | Formula | Schedule | Calcination (°C) | Sintering (°C) | Target score | Competitor score | Margin | Rank | Abstained | Reason |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(routes, key=lambda x: (x["target_formula"], x["schedule_id"])):
        lines.append(
            f"| `{r['candidate_id'][:12]}` | {r['target_formula']} | {r['schedule_id']} | {float(r['calcination_C']):.0f} | {float(r['sintering_C']):.0f} | "
            f"{float(r['condition_target_score']):.4f} | {float(r['condition_competitor_score']):.4f} | {float(r['condition_margin']):+.4f} | "
            f"{r['within_candidate_rank']} | {r['model_abstain']} | `{r['abstention_reasons']}` |"
        )
    return "\n".join(lines), json.dumps(summary, ensure_ascii=False)


def main() -> None:
    for p in (MANUSCRIPT, SI, SUMMARY, RESULTS, ROUTES):
        if not p.is_file():
            raise FileNotFoundError(p)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    m = MANUSCRIPT.read_text(encoding="utf-8")

    old_abstract = "PATH-Mat discovered 43 new COD-distinct ordered double-perovskite candidate materials; 29 known formulas lie on a published DFT hull."
    new_abstract = "PATH-Mat computationally discovered 43 new COD-distinct ordered double-perovskite candidates; 24 passed matched CHGNet relaxation screening."
    if old_abstract in m:
        m = m.replace(old_abstract, new_abstract, 1)
    elif new_abstract not in m:
        raise RuntimeError("Expected PATH-Mat abstract discovery sentence was not found")

    start_marker = "### PATH-Mat computationally discovers 43 new candidate materials"
    end_marker = "### Matched interventions show that synthesis order changes material outcomes"
    start, end = m.find(start_marker), m.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("Candidate-discovery Results boundaries are missing")
    n = summary["n_candidates"]
    retained = summary["target_gate"]["n_pass"]
    d = summary["same_formula_relative_energy_meV_atom"]
    cod_n = summary["n_exact_formula_cod_references_compared"]
    new_results = f"""### PATH-Mat computationally discovers 43 new candidate materials

We extended route ranking into material discovery by designing rock-salt-ordered oxide double perovskites, A₂BB′O₆. Geometric screening converted 72 charge-balanced cation combinations into 45 distinct 40-atom ordered CIFs. Exact-formula searches of the Crystallography Open Database (COD) identified two structures already represented in the database, which we removed. The remaining 43 designs had no matched ordered structure among COD entries retrieved for their formulas: 38 formulas had no exact-formula COD record, while five had a COD formula record but no matched ordered structure (Supplementary Table S115). PATH-Mat computationally discovered 43 new candidate materials defined by previously unmatched ordered crystal structures.

An independent high-throughput DFT study includes all 43 compositions in its near-hull A₂BB′O₆ set: 29 source structures lie on the reported convex hull, 41 are within 25 meV/atom, and all are within 27.7 meV/atom. These published hull values support composition-level prioritization but do not describe the PATH-Mat-designed geometries. To test the exact structures, we relaxed each designed CIF and its same-formula published POSCAR with the same CHGNet model, permitting both cell and ionic relaxation. All {2*n} calculations converged to a maximum force of 0.05 eV/Å. The candidate-minus-reference energy difference had a median of {d['median']:.1f} meV/atom (range {d['min']:.1f} to {d['max']:.1f}); {d['n_within_25_meV_atom']} of 43 candidates were within 25 meV/atom. None of the relaxed structures matched a same-formula source POSCAR or any of the {cod_n} retrieved exact-formula COD structures. A structure-level gate combining the source perovskite class, source hull distance ≤50 meV/atom, matched CHGNet energy difference ≤50 meV/atom, force convergence and post-relaxation structural distinctness retained {retained} candidates (Supplementary Table S118). CHGNet supplies a matched ML interatomic-potential screen, not a DFT hull calculation or experimental stability measurement [91].

For the {retained} computational candidates, the design module generated {len(read_csv(ROUTES))} balanced route/schedule hypotheses. The frozen EC-PathWM model abstained on every route because the compositions lack source coverage and observed evidence; the routes are traceable hypotheses for follow-up, not predicted synthesis outcomes (Supplementary Table S119)."""
    m = m[:start] + new_results + "\n\n" + m[end:]

    old_methods = "The cation sets, geometric filters, database scope and candidate-level outputs are in Supplementary Sections 15G–15I and Tables S114–S117."
    new_methods = ("The cation sets, geometric filters, database scope and candidate-level outputs are in Supplementary Sections 15G–15J and Tables S114–S119. "
        "For exact-structure screening, all designed CIFs and same-formula published POSCARs were relaxed using CHGNet 0.3.0 with variable cell and ionic positions (fmax = 0.05 eV Å⁻¹; maximum 500 steps); within-formula energy differences and post-relaxation StructureMatcher comparisons to source and retrieved COD structures defined the candidate gate [91].")
    if old_methods in m:
        m = m.replace(old_methods, new_methods, 1)
    elif new_methods not in m:
        raise RuntimeError("Candidate-material Methods insertion point is missing")
    m = m.replace("Supplementary Tables S1–S117", "Supplementary Tables S1–S119")
    m = m.replace("Supplementary Tables S114–S117", "Supplementary Tables S114–S119")
    m = m.replace("References 1–90", "References 1–91")
    m = m.replace("All source datasets and intervention studies are identified in References 1–90.", "All source datasets and intervention studies are identified in References 1–91.")
    m = m.replace("the published DFT stability join and derived outputs", "the published DFT stability join, CHGNet relaxation screen and derived outputs")
    m = m.replace("published-DFT screen are included", "published-DFT and CHGNet structure-screen scripts are included")
    if REF not in m:
        m = m.rstrip() + "\n" + REF + "\n"
    MANUSCRIPT.write_text(m, encoding="utf-8")

    si = SI.read_text(encoding="utf-8")
    section, _ = make_si_section()
    marker = "\n## 16. Reproducibility"
    if marker not in si:
        raise RuntimeError("SI section 16 boundary missing")
    if SECTION_START in si:
        a = si.index(SECTION_START)
        b = si.find(marker, a)
        si = si[:a] + section + "\n\n" + si[b:]
    else:
        si = si.replace(marker, "\n" + section + "\n" + marker, 1)
    si = si.replace("Supplementary Tables S1–S117", "Supplementary Tables S1–S119")
    si = si.replace("Supplementary Tables S114–S117", "Supplementary Tables S114–S119")
    old_repro = ("The PATH-Mat material-discovery screen in Supplementary Tables S114–S117 is reproduced by `release_candidate/material_generation/analysis/screen_published_double_perovskite_stability.py`, which verifies the source-file digests, joins the 43 candidate formulas to the published DFT table, compares each generated CIF with same-formula relaxed POSCARs, and writes the 39-candidate/117-route target shortlist. `integrate_cod_reassessment.py` and `integrate_published_dft_stability.py` regenerate the corresponding COD and DFT manuscript/SI sections.")
    new_repro = ("The PATH-Mat material-discovery screen in Supplementary Tables S114–S119 is reproduced by the source-hash-checked COD and published-DFT analyses and `release_candidate/material_generation/analysis/screen_exact_candidates_with_chgnet.py`. The CHGNet workflow relaxes each generated CIF and same-formula source POSCAR, compares candidate structures with the retrieved exact-formula COD structures, and writes the candidate gate and route shortlist. `integrate_chgnet_material_discovery.py` regenerates the manuscript and SI tables from the machine-readable results.")
    if old_repro in si:
        si = si.replace(old_repro, new_repro, 1)
    elif new_repro not in si:
        si = si.replace("## 17. Extended Data figure legends", new_repro + "\n\n## 17. Extended Data figure legends", 1)
    # Remove the superseded version if an earlier run inserted the replacement
    # while leaving the old provenance paragraph in place.
    legacy_repro = old_repro.replace("S114–S117", "S114–S119")
    si = si.replace(legacy_repro + "\n", "", 1)
    legacy_repro = legacy_repro.replace("S114–S119", "S114–S117")
    si = si.replace(legacy_repro + "\n", "", 1)
    si = si.replace("References 1–90", "References 1–91")
    si = si.replace(
        "The ionic-radius screen is a prototype-level geometric prior. The 45 computationally discovered candidates are an auditable material-design result: each has a distinct reduced formula, an ordered CIF, a balanced precursor reaction and three PATH-Mat-linked schedule evaluations. Their first-in-literature status, target properties, thermodynamic or dynamic stability and experimental realization are not inferred from geometric screening or source-label scores; those remain candidate-specific follow-up tests.",
        "The ionic-radius screen is a prototype-level geometric prior. The 45 designed candidates each have a distinct reduced formula, an ordered CIF, a balanced precursor reaction and three PATH-Mat-linked schedule evaluations. Exact-formula COD deduplication and the subsequent CHGNet structure screen narrow this set to 43 database-distinct structures and 24 relaxed candidates passing the matched energy/structure gate (Sections 15H–15J)."
    )
    if REF not in si:
        si = si.rstrip() + "\n" + REF + "\n"
    SI.write_text(si, encoding="utf-8")

    abstract = m.split("## Abstract\n\n", 1)[1].split("\n\n", 1)[0]
    print(json.dumps({
        "manuscript": str(MANUSCRIPT), "supporting_information": str(SI),
        "abstract_words": len(abstract.split()), "candidate_materials_discovered": n,
        "exact_structure_candidates_passing": retained, "route_hypotheses": len(read_csv(ROUTES)),
        "all_relaxations_force_converged": summary["n_candidate_force_converged"] == n and summary["n_reference_force_converged"] == n,
        "relaxed_candidates_matching_cod": summary["n_relaxed_candidates_matching_exact_formula_cod_structure"],
        "reference_added": REF,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
