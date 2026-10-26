#!/usr/bin/env python3
"""Join PATH-Mat COD-distinct candidates to a published A2BB'O6 DFT stability dataset."""
from __future__ import annotations
import csv, hashlib, json, re, zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure

MODULE = Path(__file__).resolve().parents[1]
DATA = MODULE / "data" / "references" / "acs_chem_mater_2024_double_perovskite"
COD = MODULE / "runs" / "cod_targeted_snapshot_20261017_v2" / "reassessment_occupancy_tolerance_1.02"
RUN = MODULE / "runs" / "local_pathmat_double_perovskite_20261017_v2"
OUT = MODULE / "runs" / "published_dft_stability_20261017_v1"
XLSX = DATA / "cm4c00576_si_003.xlsx"
POSCAR_ZIP = DATA / "cm4c00576_si_002_structures.zip"
EXPECTED_SHA = {
    XLSX.name: "859997d89b79bf5d8bd8c7cd7d7b77182106ce9456bb7f0385ba9a2e22c36e2b",
    POSCAR_ZIP.name: "a6319ab66b28c7a8d008b7a2f0143a0fec944cf7805470e01de530904e337c97",
}
SOURCE = {
    "article_doi": "10.1021/acs.chemmater.4c00576",
    "stability_table_doi": "10.1021/acs.chemmater.4c00576.s003",
    "relaxed_structures_doi": "10.1021/acs.chemmater.4c00576.s002",
    "article": "Wang et al., Landscape of Thermodynamic Stabilities of A2BB′O6 Compounds, Chemistry of Materials 36 (2024), 6816–6830",
    "scope": "The spreadsheet lists structures in the source study within 0.1 eV/atom of its calculated convex hull; it is not the complete 4,900-composition table.",
    "license": "CC BY-NC 4.0; retain attribution and source terms; raw inputs are not copied into the submission SI.",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reduced_formula(value: object) -> str:
    return Composition(str(value)).reduced_formula


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {p.name: sha256(p) for p in (XLSX, POSCAR_ZIP)}
    if hashes != EXPECTED_SHA:
        raise RuntimeError(f"Source input digest mismatch: {hashes}")

    candidates = pd.read_csv(COD / "cod_distinct_candidate_shortlist.csv")
    if len(candidates) != 43:
        raise RuntimeError(f"Expected frozen 43-candidate COD-distinct shortlist, got {len(candidates)}")
    candidates["formula_key"] = candidates["formula"].map(reduced_formula)
    dataset = pd.read_excel(XLSX, sheet_name="stable_metastable")
    required = {"Comp.", "space group", "convex hull (eV/atom)", "formation energy (eV/atom)"}
    if not required.issubset(dataset.columns):
        raise RuntimeError(f"Unexpected source spreadsheet schema: {dataset.columns.tolist()}")
    class_col = "Unnamed: 7"
    if class_col not in dataset.columns:
        class_col = dataset.columns[-1]
    dataset["formula_key"] = dataset["Comp."].map(reduced_formula)
    duplicate_formulas = dataset[dataset["formula_key"].duplicated(keep=False)]["formula_key"].unique().tolist()
    if duplicate_formulas:
        raise RuntimeError(f"Source has duplicate formula rows requiring adjudication: {duplicate_formulas}")

    with zipfile.ZipFile(POSCAR_ZIP) as archive:
        by_formula: dict[str, list[str]] = {}
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            file_name = name.rsplit("/", 1)[-1]
            match = re.search(r"-(.+)$", file_name)
            if match:
                try:
                    formula = reduced_formula(match.group(1))
                except Exception:
                    continue
                by_formula.setdefault(formula, []).append(name)

        matcher = StructureMatcher(stol=0.5, primitive_cell=True, scale=True, attempt_supercell=True)
        out_rows: list[dict[str, object]] = []
        missing: list[str] = []
        for row in candidates.to_dict("records"):
            formula = row["formula_key"]
            source_rows = dataset[dataset["formula_key"] == formula]
            if len(source_rows) != 1:
                missing.append(formula)
                continue
            source = source_rows.iloc[0]
            candidate_structure = Structure.from_file(RUN / row["candidate_cif"])
            ref_paths = by_formula.get(formula, [])
            matches: list[str] = []
            parse_errors: list[str] = []
            for ref_path in ref_paths:
                try:
                    reference_structure = Structure.from_str(archive.read(ref_path).decode("utf-8"), fmt="poscar")
                    if matcher.fit(candidate_structure, reference_structure):
                        matches.append(ref_path)
                except Exception as exc:
                    parse_errors.append(f"{ref_path}: {type(exc).__name__}: {exc}")
            hull = float(source["convex hull (eV/atom)"])
            formation = float(source["formation energy (eV/atom)"])
            class_value = str(source[class_col]).strip().lower()
            out_rows.append({
                "candidate_id": row["candidate_id"],
                "formula": row["formula"],
                "candidate_cif": row["candidate_cif"],
                "candidate_cif_sha256": row["candidate_cif_sha256"],
                "source_dft_formula": source["Comp."],
                "source_dft_space_group": source["space group"],
                "source_decomposition_energy_above_hull_eV_atom": hull,
                "source_formation_energy_eV_atom": formation,
                "source_relaxed_structure_class": class_value,
                "within_25_meV_atom_source_hull": hull <= 0.025,
                "within_50_meV_atom_source_hull": hull <= 0.050,
                "source_poscar_count_same_formula": len(ref_paths),
                "generated_cif_matches_source_poscar": bool(matches),
                "matched_poscar_paths": ";".join(matches),
                "source_poscar_parse_errors": " | ".join(parse_errors),
                "interpretation": "Published DFT stability/classification is for the source study's relaxed structure at the same composition, not the newly generated unrelaxed CIF.",
            })
        if missing:
            raise RuntimeError(f"Candidates missing from positive/near-hull source table: {missing}")

    out_csv = OUT / "pathmat_candidate_published_dft_stability.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    hulls = [r["source_decomposition_energy_above_hull_eV_atom"] for r in out_rows]
    classes = Counter(r["source_relaxed_structure_class"] for r in out_rows)
    stable_rows = pd.DataFrame(out_rows)
    target_rows = stable_rows[(stable_rows["source_relaxed_structure_class"] == "perovskite") & (stable_rows["source_decomposition_energy_above_hull_eV_atom"] <= 0.050)].copy()
    route_rows = pd.read_csv(RUN / "pathmat_route_hypotheses.csv")
    filtered_routes = route_rows.merge(target_rows[["candidate_id", "source_decomposition_energy_above_hull_eV_atom", "source_relaxed_structure_class"]], on="candidate_id", how="inner", validate="many_to_one")
    if len(target_rows) != 39 or len(filtered_routes) != 117:
        raise RuntimeError(f"Unexpected DFT target-filtered set: {len(target_rows)} candidates, {len(filtered_routes)} routes")
    shortlist_path = OUT / "pathmat_dft_target_filtered_candidates.csv"
    target_rows.to_csv(shortlist_path, index=False)
    routes_path = OUT / "pathmat_dft_target_filtered_route_hypotheses.csv"
    filtered_routes.to_csv(routes_path, index=False)
    summary = {
        "analysis_date": "2026-10-17",
        "n_cod_distinct_generated_candidates": len(out_rows),
        "n_found_in_source_DFT_near_hull_table": len(out_rows),
        "n_with_source_hull_distance_le_25_meV_atom": sum(x <= 0.025 for x in hulls),
        "n_with_source_hull_distance_le_50_meV_atom": sum(x <= 0.050 for x in hulls),
        "n_exactly_on_source_hull": sum(x == 0 for x in hulls),
        "source_hull_distance_eV_atom": {"min": min(hulls), "median": sorted(hulls)[len(hulls)//2], "max": max(hulls), "mean": sum(hulls)/len(hulls)},
        "source_relaxed_structure_class_counts": dict(classes),
        "n_generated_cif_matches_to_any_same_formula_source_POSCAR": sum(bool(r["generated_cif_matches_source_poscar"]) for r in out_rows),
        "n_candidate_formulas_with_same_formula_source_POSCAR": sum(int(r["source_poscar_count_same_formula"]) > 0 for r in out_rows),
        "target_filter": "source DFT-relaxed structure class = perovskite and decomposition energy above hull <= 0.050 eV/atom",
        "n_dft_target_screen_candidates": len(target_rows),
        "n_dft_target_screen_route_hypotheses": len(filtered_routes),
        "n_dft_target_screen_route_model_abstentions": int(filtered_routes["model_abstain"].astype(str).str.lower().isin(["true", "1"]).sum()),
        "structure_matcher": "pymatgen StructureMatcher(stol=0.5, primitive_cell=True, scale=True, attempt_supercell=True)",
        "structure_comparison_scope": "Each generated CIF was compared only with same-formula relaxed POSCAR structures in the source supplementary archive.",
        "source": SOURCE,
        "source_files": {name: {"sha256": hashes[name], "bytes": (DATA/name).stat().st_size} for name in hashes},
        "output_csv_sha256": sha256(out_csv),
        "critical_scope": "Hull energies and relaxed perovskite classifications belong to published DFT-relaxed structures at the same compositions; they are composition-level external evidence and are not recalculated energies or stability proof for PATH-Mat CIFs.",
    }
    (OUT / "pathmat_candidate_published_dft_stability.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    report = [
        "# Published DFT stability and perovskite-class screen",
        "",
        f"The 43 COD-distinct PATH-Mat-generated double-perovskite candidate CIFs were joined by reduced formula to the public Wang et al. high-throughput A2BB′O6 dataset (article DOI {SOURCE['article_doi']}; stability table DOI {SOURCE['stability_table_doi']}; relaxed POSCAR DOI {SOURCE['relaxed_structures_doi']}).",
        "",
        f"- Formula matches in the source near-hull table: **{len(out_rows)}/43**.",
        f"- Source DFT decomposition energy above hull: **{min(hulls)*1000:.1f}–{max(hulls)*1000:.1f} meV/atom**, median **{sorted(hulls)[len(hulls)//2]*1000:.1f} meV/atom**; **{sum(x == 0 for x in hulls)}/43** lie on the source hull and **{sum(x <= 0.050 for x in hulls)}/43** are within 50 meV/atom.",
        f"- Source DFT-relaxed structure classification: **{classes.get('perovskite',0)} perovskite; {classes.get('non-perovskite',0)} non-perovskite**.",
        f"- Generated CIFs matching a same-formula source POSCAR under the fixed StructureMatcher: **{sum(bool(r['generated_cif_matches_source_poscar']) for r in out_rows)}/43**.",
        f"- Target-filtered shortlist (source relaxed class = perovskite; source hull distance ≤50 meV/atom): **{len(target_rows)} candidate structures / {len(filtered_routes)} linked route hypotheses**; PATH-Mat abstained on all **{summary['n_dft_target_screen_route_model_abstentions']}** routes.",
        "",
        "**Interpretation boundary.** This lookup supports the thermodynamic promise of the *compositions* and provides a published DFT-relaxed structural-class signal. It does not assign those hull energies to PATH-Mat’s idealized generated CIF geometries. The source spreadsheet is a filtered list of entries within 0.1 eV/atom of its calculated hull, not the complete 4,900-composition dataset; all 43 formula matches are positive evidence in that included region. The four source-classified non-perovskite relaxations are retained and flagged rather than hidden.",
        "",
        f"Source input SHA-256: XLSX `{hashes[XLSX.name]}`; POSCAR ZIP `{hashes[POSCAR_ZIP.name]}`. The source archive is licensed CC BY-NC 4.0; this package keeps the public-source metadata and derived candidate rows with attribution.",
        "",
        "Candidate-level values and same-formula POSCAR match results are in `pathmat_candidate_published_dft_stability.csv`; the 39-row composition-level target shortlist and 117-row joined route file are `pathmat_dft_target_filtered_candidates.csv` and `pathmat_dft_target_filtered_route_hypotheses.csv`; machine-readable summary and input digests are in `pathmat_candidate_published_dft_stability.json`.",
        "",
    ]
    (OUT / "PUBLISHED_DFT_STABILITY_SCREEN.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
