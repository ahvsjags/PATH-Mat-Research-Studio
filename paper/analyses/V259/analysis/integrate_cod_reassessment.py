#!/usr/bin/env python3
"""Integrate the verified COD structure-deduplication audit into V259 manuscript/SI."""
from __future__ import annotations
import csv
import json
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
PROJECT = MODULE.parent.parent
AUDIT = MODULE / "runs" / "cod_targeted_snapshot_20261017_v2" / "reassessment_occupancy_tolerance_1.02"
SUBMISSION = PROJECT / "submission_final"
MANUSCRIPT = SUBMISSION / "PATH_Mat_Manuscript_Submission_V259.md"
SI = SUBMISSION / "PATH_Mat_Supporting_Information_Submission_V259.md"
SECTION = "### 15H. Candidate-focused COD structure screen and deduplication"
REF = ("89. Gražulis, S. et al. Crystallography Open Database (COD): an open-access collection of crystal structures "
       "and platform for world-wide collaboration. Nucleic Acids Research 40, D420–D427 (2012). "
       "https://doi.org/10.1093/nar/gkr900")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def candidate_status(row: dict[str, str]) -> str:
    if row["structure_status"] == "matching_COD_structure_found":
        return "Exact COD structure match; excluded"
    if row["formula_status"] == "formula_recorded_in_COD":
        return "Formula in COD; ordered structure not matched"
    return "No exact-formula COD record"


def make_section() -> str:
    summary = json.loads((AUDIT / "cod_novelty_reassessment.json").read_text(encoding="utf-8"))
    cand = rows(AUDIT / "cod_candidate_novelty_screen.csv")
    retained = {r["candidate_id"] for r in rows(AUDIT / "cod_distinct_candidate_shortlist.csv")}
    route_rows = rows(AUDIT / "cod_filtered_pathmat_route_hypotheses.csv")
    body = [
        SECTION, "",
        "We queried the Crystallography Open Database (COD) by the exact Hill formula of each of the 45 geometry-screened candidates [89]. A positive-control query for SrTiO3 returned eight records. The candidate queries returned entries for seven formulas; we retrieved and hash-checked nine corresponding CIFs. One reference CIF with a rounded site occupancy of 1.01 required the narrow parser fallback `occupancy_tolerance = 1.02`. Candidate and reference oxidation-state annotations were removed before matching with `StructureMatcher(stol = 0.5, primitive_cell = True, scale = True, attempt_supercell = True)`.",
        "",
        "Two generated structures exactly matched COD entries and were removed from the new-structure shortlist: Sr2TaFeO6 (COD 1528814) and Sr2TaGaO6 (COD 2100516). The other 43 candidates had no structural match among COD entries retrieved for their exact formulas. Of these, 38 had no exact-formula COD record; five had COD records for the formula but a different ordered structure. We therefore report the computational discovery of 43 new candidate materials with COD-distinct ordered structures. This is a candidate-level structure comparison against the dated exact-formula COD search, not a claim of worldwide novelty, calculated stability, or experimental realization.",
        "",
        "| COD structure-screen outcome | Count |", "|---|---:|",
        f"| Generated structures entering the screen | {summary['candidate_count']} |",
        f"| Formulas with an exact COD formula record | {summary['formula_records_found']} |",
        f"| Formulas without an exact COD formula record | {summary['formulas_without_exact_COD_record']} |",
        f"| Exact structure matches removed from shortlist | {summary['exact_COD_structure_matches']} |",
        f"| Formula-known structures without a matched ordered structure | {summary['formula_known_but_structure_not_matched']} |",
        f"| COD-distinct candidate structures retained | {summary['COD_distinct_candidate_structures_retained']} |",
        f"| Balanced PATH-Mat route/schedule hypotheses retained | {summary['route_hypotheses_after_structure_dedup']} |",
        f"| Retained routes on which frozen EC-PathWM abstained | {summary['retained_route_model_abstentions']} |",
        "",
        "**Supplementary Table S115 | Candidate-level formula and structure comparison against the dated COD search.** Candidate-ID prefixes map to full SHA-256 identifiers and generated CIF paths in the candidate ledger. A ‘no exact-formula COD record’ status is limited to the exact-formula search. ‘Formula in COD; ordered structure not matched’ means the candidate structure did not match any retrieved CIF for that formula under the stated matcher settings.",
        "",
        "| Candidate ID prefix | Formula | COD formula records | COD IDs | Structure comparison | Retained route hypotheses |",
        "|---|---|---:|---|---|---:|",
    ]
    for r in sorted(cand, key=lambda x: x["formula"]):
        body.append(f"| `{r['candidate_id'][:12]}` | {r['formula']} | {r['cod_formula_result_count']} | {r['cod_ids'] or '—'} | {candidate_status(r)} | {3 if r['candidate_id'] in retained else 0} |")
    body += [
        "",
        f"After removing the two known structure matches, the retained 43 candidates carry {len(route_rows)} balanced precursor/schedule hypotheses. The frozen EC-PathWM model abstained on all retained routes because they lack source coverage and observed evidence; model scores are not interpreted as synthesis-success probabilities. Raw query responses, reference CIFs, manifests, hashes and reassessment files are preserved in `release_candidate/material_generation/runs/cod_targeted_snapshot_20261017_v2/`.",
        ""
    ]
    return "\n".join(body)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    if new in text:
        return text
    raise RuntimeError(f"Could not locate {label} for idempotent update")


def main() -> None:
    for p in (MANUSCRIPT, SI, AUDIT / "cod_novelty_reassessment.json", AUDIT / "cod_candidate_novelty_screen.csv"):
        if not p.is_file():
            raise FileNotFoundError(p)
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    manuscript = replace_once(
        manuscript,
        "PATH-Mat also computationally discovered 45 new candidate double-perovskites linked to 135 balanced precursor schedules.",
        "PATH-Mat computationally discovered 43 new COD-distinct ordered double-perovskite candidate materials linked to 129 balanced route hypotheses.",
        "abstract discovery claim")
    old_results = ("To extend route ranking into material design, we applied the integrated PATH-Mat agent to rock-salt-ordered oxide double perovskites, A₂BB′O₆. From 72 charge-balanced cation combinations, radius-based geometric screening selected 45 compositions, each written as a 40-atom ordered CIF; the set contains 45 distinct reduced formulas, and all CIFs passed formula and hash checks. The agent then generated three balanced oxide-precursor schedules for each candidate, yielding 135 route/schedule evaluations with the frozen EC-PathWM model. The lower-thermal-budget schedule ranked first in all 45 within-candidate comparisons. Because these compositions had no source records, EC-PathWM abstained on all 135 routes; the scores therefore remain candidate-ranking signals rather than synthesis-success probabilities. This module delivers 45 newly designed candidate materials as reproducible structures linked to explicit synthesis hypotheses (Supplementary Table S114).")
    new_results = ("To extend route ranking into material design, we applied the integrated PATH-Mat agent to rock-salt-ordered oxide double perovskites, A₂BB′O₆. From 72 charge-balanced cation combinations, geometric screening generated 45 distinct compositions as 40-atom ordered CIFs; all passed formula and hash checks. Exact-formula searches of the Crystallography Open Database (COD) identified two structures already represented in the database (Sr2TaFeO6 and Sr2TaGaO6), which we removed. The remaining 43 structures had no match among the COD entries retrieved for their formulas: 38 had no exact-formula record, while five had a known formula but an unmatched ordered structure (Supplementary Table S115). This yields a computational discovery set of 43 new candidate materials with COD-distinct ordered structures. PATH-Mat linked these structures to 129 balanced oxide-precursor schedules and ranked the lower-thermal-budget schedule first in all 43 within-candidate comparisons. EC-PathWM abstained on all 129 routes because these designs lack source coverage and observed evidence; the scores are not synthesis-success probabilities (Supplementary Tables S114–S115).")
    manuscript = replace_once(manuscript, old_results, new_results, "candidate-design Results paragraph")
    old_methods = "The exact cation sets, geometric cutoffs and candidate-level results are provided in Supplementary Section 15G and Table S114."
    new_methods = ("For candidate novelty screening, we queried COD by exact Hill formula, retrieved formula-matched CIFs and compared oxidation-state-stripped structures using `StructureMatcher` (`stol = 0.5`, `primitive_cell = True`, `scale = True`, `attempt_supercell = True`); exact structural matches were removed from the discovery shortlist. The cation sets, geometric filters, search scope and candidate-level results are provided in Supplementary Sections 15G–15H and Tables S114–S115 [89].")
    manuscript = replace_once(manuscript, old_methods, new_methods, "candidate-design Methods statement")
    manuscript = manuscript.replace("References 1–88", "References 1–89")
    manuscript = manuscript.replace("Supplementary Tables S1–S114", "Supplementary Tables S1–S115")
    manuscript = manuscript.replace("The analysis package contains code and derived outputs;", "The analysis package contains code, the COD novelty-screen snapshot and derived outputs;")
    manuscript = manuscript.replace("source-specific analyses and the candidate-material design agent are included", "source-specific analyses, the candidate-material design agent and COD matching scripts are included")
    if REF not in manuscript:
        manuscript = manuscript.rstrip() + "\n" + REF + "\n"
    MANUSCRIPT.write_text(manuscript, encoding="utf-8")

    si = SI.read_text(encoding="utf-8")
    si = si.replace("The geometric filters retained 45 of 72 charge-balanced proposals as computationally discovered candidate materials; 27 failed the configured filters.", "The geometric filters retained 45 of 72 charge-balanced proposals as the initial computational design set; 27 failed the configured filters.")
    si = si.replace("The discovery set contains 45 distinct reduced formulas, each represented by a newly generated 40-atom CIF. Every CIF was parsed back and matched to its ledger formula, and all per-file SHA-256 digests agreed with the run manifest. Here, ‘discovered’ means computationally generated and selected in this study; first-in-literature novelty has not been established against a comprehensive structure database. No target-property, formation-energy/energy-above-hull, DFT-relaxation or phonon model was run.", "The initial design set contains 45 distinct reduced formulas, each represented by a newly generated 40-atom CIF. Every CIF was parsed back and matched to its ledger formula, and all per-file SHA-256 digests agreed with the run manifest. The candidate-focused COD screen in Supplementary Section 15H removed two exact structure matches and retained 43 COD-distinct ordered structures as computational discoveries. No target-property, formation-energy/energy-above-hull, DFT-relaxation or phonon model was run.")
    si = si.replace("**Supplementary Table S114 | Computational discovery set of rock-salt-ordered double-perovskite candidates generated by PATH-Mat.**", "**Supplementary Table S114 | Initial generated set of rock-salt-ordered double-perovskite candidates from PATH-Mat.**")
    si = si.replace("‘New candidate material’ denotes a distinct composition/structure designed and generated in this study; global database novelty and physical realization were not tested.", "‘Generated candidate material’ denotes a composition/structure designed in this study. Table S115 reports the subsequent COD structure screen: 43 candidates were retained as COD-distinct and two exact structure matches were removed. COD-distinct status is limited to the dated exact-formula search and does not establish physical realization or worldwide novelty.")
    section = make_section()
    if SECTION in si:
        start = si.index(SECTION)
        end = si.find("\n## 16. Reproducibility", start)
        if end < 0:
            raise RuntimeError("Could not locate SI section 16 after existing COD section")
        si = si[:start] + section + si[end:]
    else:
        marker = "\n## 16. Reproducibility"
        if marker not in si:
            raise RuntimeError("Could not locate SI section 16 insertion point")
        si = si.replace(marker, "\n" + section + marker, 1)
    if REF not in si:
        si = si.rstrip() + "\n" + REF + "\n"
    SI.write_text(si, encoding="utf-8")
    print(json.dumps({"manuscript": str(MANUSCRIPT), "supporting_information": str(SI), "candidate_structures_generated": 45, "COD_distinct_discovery_candidates": 43, "exact_COD_matches_removed": 2, "route_hypotheses_retained": 129}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
