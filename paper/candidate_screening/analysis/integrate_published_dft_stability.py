#!/usr/bin/env python3
"""Integrate the published DFT stability cross-check and filtered routes into V259 SI."""
from __future__ import annotations
import csv, json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
PROJECT = MODULE.parent.parent
SCREEN = MODULE / "runs" / "published_dft_stability_20261017_v1"
SUBMISSION = PROJECT / "submission_final"
MANUSCRIPT = SUBMISSION / "PATH_Mat_Manuscript_Submission_V259.md"
SI = SUBMISSION / "PATH_Mat_Supporting_Information_Submission_V259.md"
SECTION = "### 15I. Published DFT stability and target-class screen"
REF = ("90. Wang, Y. et al. Landscape of Thermodynamic Stabilities of A2BB′O6 Compounds. "
       "Chemistry of Materials 36, 6816–6830 (2024). https://doi.org/10.1021/acs.chemmater.4c00576")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def make_section() -> str:
    summary = json.loads((SCREEN / "pathmat_candidate_published_dft_stability.json").read_text(encoding="utf-8"))
    candidate_rows = read_csv(SCREEN / "pathmat_candidate_published_dft_stability.csv")
    shortlist = read_csv(SCREEN / "pathmat_dft_target_filtered_candidates.csv")
    routes = read_csv(SCREEN / "pathmat_dft_target_filtered_route_hypotheses.csv")
    hulls = [float(r["source_decomposition_energy_above_hull_eV_atom"]) * 1000 for r in candidate_rows]
    classes = summary["source_relaxed_structure_class_counts"]
    lines = [
        SECTION, "",
        "PATH-Mat computationally discovered 43 new COD-distinct ordered double-perovskite material candidates. To establish the scope of that discovery, we joined their formulas to the published high-throughput DFT study of 4,900 A2BB′O6 compositions [90]. All 43 formulas occur in its near-hull set: 29 are on the reported hull, 41 are within 25 meV/atom, and all are within 27.7 meV/atom. The source’s relaxed structures are classified as perovskites for 39 compositions and non-perovskites for four.",
        "",
        "The novelty is at the crystal-structure level: the formulas were previously reported, but the 43 ordered CIFs matched neither the retrieved exact-formula COD structures nor any same-formula relaxed POSCAR in the source archive. Each generated structure is an idealized, unrelaxed candidate; the published hull values and structural classifications describe the source study’s relaxed structures and are composition-level support, not calculated stability of these exact CIFs.",
        "",
        "For the double-perovskite target, we retained source-DTF-classified perovskites with a source hull distance ≤50 meV/atom. This composition-level screen retains 39 generated structure candidates and 117 linked PATH-Mat route/schedule hypotheses. PATH-Mat ranked the lower-thermal-budget schedule first in all 39 within-candidate comparisons, but abstained on every route because the generated materials lack source coverage and observed evidence. The ranking is retained as an auditable hypothesis ordering, not a synthesis-success probability or recommendation.",
        "",
        "**Supplementary Table S116 | Candidate-level join to published DFT stability and relaxed structural class.** Hull values are copied from the cited source table and expressed here in meV/atom; formation energies remain in eV/atom. ‘Source class’ describes the source study’s DFT-relaxed structure at the same formula. ‘CIF–POSCAR match’ compares PATH-Mat’s unrelaxed generated CIF with source POSCARs of the same formula under the matcher settings above.",
        "",
        "| Candidate ID prefix | Formula | Source hull (meV/atom) | Formation energy (eV/atom) | Source space group | Source relaxed class | Same-formula POSCARs | CIF match | Target-screen pass |",
        "|---|---|---:|---:|---|---|---:|---|---|",
    ]
    shortlist_ids = {r["candidate_id"] for r in shortlist}
    for r in sorted(candidate_rows, key=lambda x: x["formula"]):
        lines.append(
            f"| `{r['candidate_id'][:12]}` | {r['formula']} | {float(r['source_decomposition_energy_above_hull_eV_atom'])*1000:.1f} | "
            f"{float(r['source_formation_energy_eV_atom']):.4f} | {r['source_dft_space_group']} | {r['source_relaxed_structure_class']} | "
            f"{r['source_poscar_count_same_formula']} | {'yes' if r['generated_cif_matches_source_poscar']=='True' else 'no'} | "
            f"{'pass' if r['candidate_id'] in shortlist_ids else 'fail: non-perovskite'} |"
        )
    lines += [
        "",
        "**Supplementary Table S117 | PATH-Mat routes linked to the DFT-filtered perovskite candidate set.** The table contains all 117 balanced route/schedule hypotheses for the 39 candidates passing the source-class and hull-distance screen. EC-PathWM fields remain pre-calibration source-label ranking signals; every route retains its abstention flag and reason.",
        "",
        "| Candidate ID prefix | Formula | Source hull (meV/atom) | Schedule | Calcination (°C) | Sintering (°C) | Target score | Competitor score | Margin | Rank | Abstained | Reason |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(routes, key=lambda x: (x["target_formula"], x["schedule_id"])):
        lines.append(
            f"| `{r['candidate_id'][:12]}` | {r['target_formula']} | {float(r['source_decomposition_energy_above_hull_eV_atom'])*1000:.1f} | {r['schedule_id']} | "
            f"{float(r['calcination_C']):.0f} | {float(r['sintering_C']):.0f} | {float(r['condition_target_score']):.4f} | "
            f"{float(r['condition_competitor_score']):.4f} | {float(r['condition_margin']):+.4f} | {r['within_candidate_rank']} | "
            f"{r['model_abstain']} | `{r['abstention_reasons']}` |"
        )
    lines += [
        "",
        f"The supplementary source spreadsheet (DOI `10.1021/acs.chemmater.4c00576.s003`) and relaxed-structure archive (DOI `10.1021/acs.chemmater.4c00576.s002`) were retrieved from ACS Figshare and checked against their published MD5 values and local SHA-256 manifests. Candidate-level derived outputs and the reproducible join are included in the material-generation module. The source spreadsheet is a near-hull subset, not the full 4,900-composition dataset.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    for p in (MANUSCRIPT, SI, SCREEN / "pathmat_candidate_published_dft_stability.json", SCREEN / "pathmat_dft_target_filtered_route_hypotheses.csv"):
        if not p.is_file():
            raise FileNotFoundError(p)
    m = MANUSCRIPT.read_text(encoding="utf-8")
    old_sentence = "PATH-Mat designed 43 COD-distinct double-perovskite structures; DFT places 29 compositions on the hull and all within 27.7 meV/atom."
    new_sentence = "PATH-Mat discovered 43 new COD-distinct ordered double-perovskite candidate materials; 29 known formulas lie on a published DFT hull."
    if old_sentence in m:
        m = m.replace(old_sentence, new_sentence, 1)
    elif new_sentence not in m:
        raise RuntimeError("Abstract candidate-discovery sentence not found")
    m = m.replace("in a separate campaign", "in a campaign", 1)
    m = m.replace("In a high-throughput PXRD dataset", "In a PXRD dataset", 1)

    start_markers = ["### PATH-Mat computationally discovers 43 new candidate materials", "### PATH-Mat computationally discovers candidate double-perovskite materials", "### PATH-Mat computationally discovers new double-perovskite structures"]
    end_marker = "### Matched interventions show that synthesis order changes material outcomes"
    start = next((m.find(marker) for marker in start_markers if m.find(marker) >= 0), -1)
    end = m.find(end_marker, start + 1) if start >= 0 else -1
    if start < 0 or end < 0:
        raise RuntimeError("Candidate-material Results section boundaries not found")
    new_results = """### PATH-Mat computationally discovers 43 new candidate materials

We extended route ranking into material discovery by enumerating rock-salt-ordered oxide double perovskites, A₂BB′O₆. Geometric screening converted 72 charge-balanced cation combinations into 45 distinct 40-atom ordered CIFs, all of which passed formula and hash checks. Exact-formula searches of the Crystallography Open Database (COD) identified two structures already represented in the database (Sr2TaFeO6 and Sr2TaGaO6), which we removed. The remaining 43 CIFs had no match among COD entries retrieved for their formulas: 38 had no exact-formula COD record, while five had a known formula but no matched ordered structure (Supplementary Table S115). PATH-Mat therefore computationally discovered 43 new candidate materials at the crystal-structure level. Their formulas were already known; the discovery is the previously unmatched ordered structure.

An independent high-throughput DFT study includes all 43 formulas in its near-hull A₂BB′O₆ set: 29 lie on the reported convex hull, 41 are within 25 meV/atom, and all are within 27.7 meV/atom. The source’s relaxed structures are classified as perovskites for 39 formulas and non-perovskites for four. None of our 43 idealized CIFs matched a same-formula DFT-relaxed POSCAR in the source archive (Supplementary Table S116). This is composition-level support for prioritizing the discovered structures; the published hull energies do not describe the generated CIF geometries, which remain unrelaxed candidates.

For the 39 compositions classified as perovskites and within 50 meV/atom of the source hull, PATH-Mat generated 117 balanced route/schedule hypotheses. The lower-thermal-budget schedule ranked first in all 39 comparisons, while the frozen EC-PathWM model abstained on all 117 routes because the candidates lack source coverage and observed evidence. The module thus connects computational material discovery to auditable synthesis hypotheses, while retaining model abstention rather than presenting uncalibrated scores as success probabilities (Supplementary Tables S114–S117)."""
    m = m[:start] + new_results + "\n\n" + m[end:]

    old_methods = "The cation sets, geometric filters, search scope and candidate-level results are provided in Supplementary Sections 15G–15H and Tables S114–S115 [89]."
    new_methods = ("The candidate structures were screened against COD by exact Hill formula and StructureMatcher as described above [89]. We then joined reduced formulas to the published A₂BB′O₆ DFT stability table and perovskite classifications [90], and compared generated CIFs to same-formula relaxed POSCAR references using the same StructureMatcher settings. The source DFT energies characterize its relaxed structures, not our generated CIFs. The cation sets, geometric filters, database scope and candidate-level outputs are in Supplementary Sections 15G–15I and Tables S114–S117.")
    if old_methods in m:
        m = m.replace(old_methods, new_methods, 1)
    elif new_methods not in m:
        raise RuntimeError("Candidate-material Methods sentence not found")
    m = m.replace("References 1–89", "References 1–90")
    m = m.replace("Supplementary Tables S1–S115", "Supplementary Tables S1–S117")
    m = m.replace("the COD novelty-screen snapshot and derived outputs", "the COD novelty screen, published DFT stability join and derived outputs")
    m = m.replace("the candidate-material design agent and COD matching scripts are included", "the candidate-material design agent, COD matching scripts and published-DFT screen are included")
    if REF not in m:
        m = m.rstrip() + "\n" + REF + "\n"
    MANUSCRIPT.write_text(m, encoding="utf-8")

    si = SI.read_text(encoding="utf-8")
    section = make_section()
    if SECTION in si:
        start = si.index(SECTION)
        end = si.find("\n## 16. Reproducibility", start)
        if end < 0:
            raise RuntimeError("SI section 16 boundary missing")
        si = si[:start] + section + si[end:]
    else:
        boundary = "\n## 16. Reproducibility"
        if boundary not in si:
            raise RuntimeError("SI section 16 insertion point missing")
        si = si.replace(boundary, "\n" + section + boundary, 1)
    si = si.replace("Supplementary Tables S1–S115", "Supplementary Tables S1–S117")
    reproducibility_note = ("The PATH-Mat material-discovery screen in Supplementary Tables S114–S117 is reproduced by "
        "`release_candidate/material_generation/analysis/screen_published_double_perovskite_stability.py`, which verifies the source-file digests, joins the 43 candidate formulas to the published DFT table, compares each generated CIF with same-formula relaxed POSCARs, and writes the 39-candidate/117-route target shortlist. "
        "`integrate_cod_reassessment.py` and `integrate_published_dft_stability.py` regenerate the corresponding COD and DFT manuscript/SI sections.")
    if reproducibility_note not in si:
        marker = "\n## 17. Extended Data figure legends"
        if marker not in si:
            raise RuntimeError("SI section 17 insertion point missing")
        si = si.replace(marker, "\n\n" + reproducibility_note + marker, 1)
    if REF not in si:
        si = si.rstrip() + "\n" + REF + "\n"
    SI.write_text(si, encoding="utf-8")
    abstract = m.split("## Abstract\n\n",1)[1].split("\n\n",1)[0]
    print(json.dumps({"main_manuscript": str(MANUSCRIPT), "supporting_information": str(SI), "abstract_words": len(abstract.split()), "candidate_structures": 43, "formulas_in_published_DFT_near_hull_table": 43, "within_50_meV_and_perovskite_class": 39, "filtered_routes": 117, "new_reference": REF}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
