#!/usr/bin/env python3
"""Audit generated CIFs against hash-frozen references and staged calculations.

This program never infers stability or a target property from the generative
model's conditioning value. Final candidate status requires a separate DFT row
and a positive phonon-stability record for the exact generated structure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install PyYAML and pymatgen in the analysis environment.") from exc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(path: Path) -> str:
    """Stable digest of a file or extracted snapshot directory."""
    if path.is_file():
        return sha256(path)
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(child.relative_to(path).as_posix().encode("utf-8"))
        h.update(bytes.fromhex(sha256(child)))
    return h.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("schema_version") != 1:
        raise SystemExit("Unsupported config schema_version.")
    return cfg


def read_snapshot_structures(snapshot: Path) -> list[tuple[str, Structure]]:
    """Load CIFs from a hash-frozen directory, .cif file, or ZIP snapshot."""
    records: list[tuple[str, Structure]] = []
    if snapshot.is_file() and snapshot.suffix.lower() == ".cif":
        return [(snapshot.name, Structure.from_file(snapshot))]
    if snapshot.is_file() and zipfile.is_zipfile(snapshot):
        with zipfile.ZipFile(snapshot) as zf:
            for name in sorted(n for n in zf.namelist() if n.lower().endswith(".cif")):
                records.append((f"{snapshot.name}!/{name}", Structure.from_str(zf.read(name).decode("utf-8"), fmt="cif")))
        return records
    if snapshot.is_dir():
        for path in sorted(snapshot.rglob("*.cif")):
            records.append((str(path), Structure.from_file(path)))
        return records
    raise SystemExit(f"Reference snapshot must be a CIF, ZIP, or directory: {snapshot}")


def load_manifest(path: Path) -> tuple[list[dict[str, str]], list[tuple[str, Structure]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset_id", "version", "access_date", "license", "local_path", "sha256", "structure_count"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"Reference manifest must contain rows and columns: {sorted(required)}")
    all_structures: list[tuple[str, Structure]] = []
    for row in rows:
        for col in required:
            if not row.get(col, "").strip():
                raise SystemExit(f"Reference manifest field {col!r} is blank.")
        snapshot = Path(row["local_path"])
        if not snapshot.is_absolute():
            snapshot = path.parent / snapshot
        if not snapshot.exists():
            raise SystemExit(f"Reference snapshot not found: {snapshot}")
        digest = tree_sha256(snapshot)
        if digest.lower() != row["sha256"].lower():
            raise SystemExit(f"Reference snapshot digest mismatch: {snapshot}")
        structures = read_snapshot_structures(snapshot)
        expected_count = int(row["structure_count"])
        if len(structures) != expected_count:
            raise SystemExit(
                f"Snapshot structure count mismatch for {row['dataset_id']}: "
                f"manifest={expected_count}, observed={len(structures)}"
            )
        row["_resolved_path"] = str(snapshot.resolve())
        row["_observed_sha256"] = digest
        row["_observed_structure_count"] = str(len(structures))
        all_structures.extend((f"{row['dataset_id']}:{row['version']}:{label}", s) for label, s in structures)
    return rows, all_structures


def _is_hash(value: str) -> bool:
    value = value.strip().lower()
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def load_screen(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    """Key calculation records by path-derived candidate ID and input digest."""
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "candidate_id", "cif_sha256", "screen_level", "calculator", "calculator_version",
            "band_gap_eV", "energy_above_hull_meV_per_atom",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit(f"Screen CSV must have columns: {sorted(required)}")
        rows = list(reader)
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        level = row["screen_level"].strip().lower()
        key = (row["candidate_id"].strip(), level)
        if not key[0] or level not in {"mlff", "dft"}:
            raise SystemExit(f"Invalid screen key: {key}")
        if key in out:
            raise SystemExit(f"Duplicate screen row: {key}")
        if not _is_hash(row["cif_sha256"]):
            raise SystemExit(f"Invalid cif_sha256 for {key}")
        for col in ["calculator", "calculator_version", "band_gap_eV", "energy_above_hull_meV_per_atom"]:
            if not row.get(col, "").strip():
                raise SystemExit(f"{col} missing for screen row {key}")
        for col in ["band_gap_eV", "energy_above_hull_meV_per_atom"]:
            if not math.isfinite(float(row[col])):
                raise SystemExit(f"Non-finite {col} for {key}")
        if level == "dft":
            required_dft = [
                "method", "relaxed_cif_path", "relaxed_cif_sha256", "hull_reference",
                "hull_reference_version", "calculation_id", "input_path", "input_digest",
                "calculation_bundle_path", "calculation_bundle_sha256", "phonon_stable",
                "phonon_output_path", "phonon_output_sha256", "phonon_calculator",
                "phonon_calculator_version",
            ]
            for col in required_dft:
                if not row.get(col, "").strip():
                    raise SystemExit(f"DFT {col} missing for {key}")
            for col in ["relaxed_cif_sha256", "input_digest", "calculation_bundle_sha256", "phonon_output_sha256"]:
                if not _is_hash(row[col]):
                    raise SystemExit(f"Invalid SHA-256 in DFT {col} for {key}")
            for path_key, hash_key in [
                ("relaxed_cif_path", "relaxed_cif_sha256"),
                ("input_path", "input_digest"),
                ("calculation_bundle_path", "calculation_bundle_sha256"),
                ("phonon_output_path", "phonon_output_sha256"),
            ]:
                artifact = Path(row[path_key])
                if not artifact.is_absolute():
                    artifact = path.parent / artifact
                if not artifact.exists():
                    raise SystemExit(f"DFT artifact {path_key} does not exist for {key}: {artifact}")
                observed = tree_sha256(artifact)
                if observed.lower() != row[hash_key].lower():
                    raise SystemExit(f"DFT artifact digest mismatch for {key}: {path_key}")
                row[f"_{path_key}_resolved"] = str(artifact.resolve())
            if row["phonon_stable"].strip().lower() not in {"true", "false", "1", "0", "yes", "no"}:
                raise SystemExit(f"Invalid phonon_stable value for {key}")
        out[key] = row
    return out


def yes(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def match_reference(structure: Structure, references: dict[str, list[tuple[str, Structure]]], matcher: StructureMatcher) -> str | None:
    formula = structure.composition.reduced_formula
    for ref_label, ref_structure in references.get(formula, []):
        if matcher.fit(structure, ref_structure):
            return ref_label
    return None


def iter_cifs(run_dir: Path) -> Iterable[Path]:
    return sorted(run_dir.rglob("*.cif"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--reference-manifest", type=Path, required=True)
    ap.add_argument("--screen-csv", type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    target = cfg["target"]
    novelty_cfg = cfg.get("novelty", {})
    screen_cfg = cfg.get("screening", {})
    low, high = map(float, target["band_gap_window_eV"])
    required_elements = set(target.get("required_elements", []))
    prohibited = set(target.get("prohibited_elements", []))
    prescreen_hull = float(screen_cfg.get("prescreen_e_hull_max_meV_per_atom", 100.0))
    dft_hull = float(screen_cfg.get("final_dft_e_hull_max_meV_per_atom", target.get("max_energy_above_hull_meV_per_atom", 25.0)))
    shortlist_max = int(screen_cfg.get("shortlist_max_candidates_per_seed", screen_cfg.get("shortlist_max_candidates", 10)))
    if shortlist_max < 1:
        raise SystemExit("shortlist_max_candidates_per_seed must be positive")
    target_gap = float(cfg["generator"]["target_value_eV"])
    matcher = StructureMatcher(stol=float(novelty_cfg.get("matcher_stol", 0.5)), primitive_cell=True, scale=True, attempt_supercell=True)
    manifest_rows, reference_rows = load_manifest(args.reference_manifest)
    refs_by_formula: dict[str, list[tuple[str, Structure]]] = defaultdict(list)
    ref_formulas: set[str] = set()
    for label, structure in reference_rows:
        formula = structure.composition.reduced_formula
        refs_by_formula[formula].append((label, structure))
        ref_formulas.add(formula)
    screens = load_screen(args.screen_csv)
    cifs = list(iter_cifs(args.run_dir))
    if not cifs:
        raise SystemExit(f"No CIF files found below {args.run_dir}")

    seen_structures: dict[str, list[tuple[str, Structure]]] = defaultdict(list)
    seen_formulas: set[str] = set()
    rows: list[dict[str, Any]] = []
    for cif in cifs:
        content_hash = sha256(cif)
        relative = cif.relative_to(args.run_dir).as_posix()
        # Include design arm and seed in the identity namespace. The same CIF
        # produced in two arms/seeds must remain traceable as two observations.
        run_namespace = f"{args.run_dir.parent.name}/{args.run_dir.name}"
        candidate_id = hashlib.sha256(f"{run_namespace}\0{relative}\0{content_hash}".encode("utf-8")).hexdigest()
        key = (candidate_id, "mlff")
        dft_key = (candidate_id, "dft")
        reasons: list[str] = []
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "run_namespace": run_namespace,
            "relative_path": relative,
            "cif_sha256": content_hash,
            "parse_valid": False,
            "composition": "",
            "reduced_formula": "",
            "n_sites": 0,
            "space_group_number": "",
            "oxide_chemistry_pass": False,
            "composition_novel_vs_snapshot": False,
            "structure_novel_vs_snapshot": False,
            "composition_duplicate_within_run": False,
            "structure_duplicate_within_run": False,
            "matched_reference": "",
            "matched_generated": "",
            "mlff_band_gap_eV": "",
            "mlff_e_hull_meV_atom": "",
            "mlff_calculator": "",
            "mlff_calculator_version": "",
            "prescreen_pass": False,
            "shortlist_selected": False,
            "shortlist_rank": "",
            "dft_band_gap_eV": "",
            "dft_e_hull_meV_atom": "",
            "dft_method": "",
            "dft_calculator": "",
            "dft_calculator_version": "",
            "dft_calculation_id": "",
            "dft_relaxed_cif_sha256": "",
            "dft_calculation_bundle_sha256": "",
            "dft_hull_reference": "",
            "dft_hull_reference_version": "",
            "phonon_stable": "",
            "target_hit": False,
            "stable_by_threshold": False,
            "confirmation_status": "awaiting_screen",
            "computational_candidate": False,
            "filter_reasons": "",
        }
        structure = None
        try:
            structure = Structure.from_file(cif)
            composition = structure.composition
            formula = composition.reduced_formula
            elements = {str(el) for el in composition.elements}
            oxide_pass = (
                "O" in elements
                and any(el != "O" for el in elements)
                and required_elements.issubset(elements)
                and not (elements & prohibited)
            )
            reference_match = match_reference(structure, refs_by_formula, matcher)
            same_formula_generated = seen_formulas.__contains__(formula)
            internal_match = next(
                (name for name, previous in seen_structures[formula] if matcher.fit(structure, previous)), None
            )
            try:
                space_group = SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
            except Exception:
                space_group = ""
                reasons.append("space_group_unresolved")
            row.update({
                "parse_valid": True,
                "composition": composition.formula,
                "reduced_formula": formula,
                "n_sites": len(structure),
                "space_group_number": space_group,
                "oxide_chemistry_pass": oxide_pass,
                "composition_novel_vs_snapshot": formula not in ref_formulas,
                "structure_novel_vs_snapshot": reference_match is None,
                "composition_duplicate_within_run": same_formula_generated,
                "structure_duplicate_within_run": internal_match is not None,
                "matched_reference": reference_match or "",
                "matched_generated": internal_match or "",
            })
            # Retain only a representative structure per exact structure type for deduplication.
            if internal_match is None:
                seen_structures[formula].append((candidate_id, structure))
            seen_formulas.add(formula)
            if not oxide_pass:
                reasons.append("not_target_oxide_chemistry")
            if formula in ref_formulas:
                reasons.append("reduced_formula_present_in_frozen_reference")
            if reference_match:
                reasons.append("structure_match_in_frozen_reference")
            if internal_match:
                reasons.append("duplicate_structure_within_run")
        except Exception as exc:
            reasons.append(f"cif_parse_error:{type(exc).__name__}")

        mlff = screens.get(key)
        dft = screens.get(dft_key)
        if mlff and mlff["cif_sha256"].lower() != content_hash.lower():
            raise SystemExit(f"MLFF screen input CIF digest mismatch for {candidate_id}")
        if dft and dft["cif_sha256"].lower() != content_hash.lower():
            raise SystemExit(f"DFT screen input CIF digest mismatch for {candidate_id}")
        if dft and dft["input_digest"].lower() != content_hash.lower():
            raise SystemExit(f"DFT calculation input is not the exact candidate CIF for {candidate_id}")
        if mlff:
            bg = float(mlff["band_gap_eV"])
            hull = float(mlff["energy_above_hull_meV_per_atom"])
            row["mlff_band_gap_eV"] = bg
            row["mlff_e_hull_meV_atom"] = hull
            row["mlff_calculator"] = mlff["calculator"]
            row["mlff_calculator_version"] = mlff["calculator_version"]
            row["prescreen_pass"] = low <= bg <= high and hull <= prescreen_hull
            if not low <= bg <= high:
                reasons.append("mlff_property_outside_target")
            if hull > prescreen_hull:
                reasons.append("mlff_e_hull_above_prescreen_threshold")
        else:
            reasons.append("missing_mlff_prescreen_row")

        if dft:
            relaxed_path = Path(dft["_relaxed_cif_path_resolved"])
            relaxed_structure = Structure.from_file(relaxed_path)
            if structure is None or relaxed_structure.composition.reduced_formula != structure.composition.reduced_formula:
                raise SystemExit(f"Relaxed DFT structure composition does not match its input CIF for {candidate_id}")
            bg = float(dft["band_gap_eV"])
            hull = float(dft["energy_above_hull_meV_per_atom"])
            phonon = yes(dft["phonon_stable"])
            target_hit = low <= bg <= high
            stable = hull <= dft_hull
            row.update({
                "dft_band_gap_eV": bg,
                "dft_e_hull_meV_atom": hull,
                "dft_hull_reference": dft["hull_reference"],
                "dft_hull_reference_version": dft["hull_reference_version"],
                "dft_method": dft["method"],
                "dft_calculator": dft["calculator"],
                "dft_calculator_version": dft["calculator_version"],
                "dft_calculation_id": dft["calculation_id"],
                "dft_relaxed_cif_sha256": dft["relaxed_cif_sha256"],
                "dft_calculation_bundle_sha256": dft["calculation_bundle_sha256"],
                "phonon_output_sha256": dft["phonon_output_sha256"],
                "phonon_calculator": dft["phonon_calculator"],
                "phonon_calculator_version": dft["phonon_calculator_version"],
                "phonon_stable": phonon,
                "target_hit": target_hit,
                "stable_by_threshold": stable,
                "confirmation_status": "DFT_confirmed" if target_hit and stable and phonon else "DFT_rejected",
            })
            if not target_hit:
                reasons.append("dft_property_outside_target")
            if not stable:
                reasons.append("dft_e_hull_above_threshold")
            if not phonon:
                reasons.append("phonon_unstable")
        else:
            row["confirmation_status"] = "not_shortlisted_by_MLFF"
        row["filter_reasons"] = ";".join(reasons)
        rows.append(row)

    # A fixed, diversity-aware shortlist is selected before any DFT/phonon outcomes
    # are joined: one structure per reduced formula, ordered by MLFF hull, distance
    # from the target gap, then stable candidate ID. Each run directory is one seed.
    shortlist_pool = [
        r for r in rows
        if r["parse_valid"] and r["oxide_chemistry_pass"]
        and r["composition_novel_vs_snapshot"] and r["structure_novel_vs_snapshot"]
        and not r["structure_duplicate_within_run"] and r["prescreen_pass"]
    ]
    shortlist_pool.sort(key=lambda r: (
        float(r["mlff_e_hull_meV_atom"]),
        abs(float(r["mlff_band_gap_eV"]) - target_gap),
        r["candidate_id"],
    ))
    shortlist: list[dict[str, Any]] = []
    selected_formulas: set[str] = set()
    for candidate in shortlist_pool:
        if candidate["reduced_formula"] in selected_formulas:
            continue
        shortlist.append(candidate)
        selected_formulas.add(candidate["reduced_formula"])
        if len(shortlist) >= shortlist_max:
            break
    shortlist_ids = {r["candidate_id"] for r in shortlist}
    for rank, candidate in enumerate(shortlist, 1):
        candidate["shortlist_selected"] = True
        candidate["shortlist_rank"] = rank
    for row in rows:
        is_selected = row["candidate_id"] in shortlist_ids
        has_dft = (row["candidate_id"], "dft") in screens
        if has_dft and not is_selected:
            raise SystemExit(f"DFT was run for a candidate outside the frozen shortlist: {row['candidate_id']}")
        if is_selected and not has_dft:
            row["confirmation_status"] = "shortlist_pending_DFT_and_phonons"
            if "awaiting_independent_DFT_and_phonon_confirmation" not in row["filter_reasons"]:
                row["filter_reasons"] = ";".join(filter(None, [row["filter_reasons"], "awaiting_independent_DFT_and_phonon_confirmation"]))
        elif row["prescreen_pass"] and not is_selected:
            row["confirmation_status"] = "not_selected_by_shortlist_budget"
            row["filter_reasons"] = ";".join(filter(None, [row["filter_reasons"], "not_selected_by_diversity_aware_shortlist"]))
        row["computational_candidate"] = bool(
            is_selected and row["parse_valid"] and row["oxide_chemistry_pass"]
            and row["composition_novel_vs_snapshot"] and row["structure_novel_vs_snapshot"]
            and not row["structure_duplicate_within_run"]
            and row["confirmation_status"] == "DFT_confirmed"
        )

    # Catch screen rows for unknown candidates; silently unused score records are dangerous.
    candidate_ids = {row["candidate_id"] for row in rows}
    orphan_screen_rows = sorted(k for k in screens if k[0] not in candidate_ids)
    if orphan_screen_rows:
        raise SystemExit(f"Screen CSV contains {len(orphan_screen_rows)} rows for unknown candidate IDs.")

    out_csv = args.run_dir / "candidate_audit.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter({"generated_cifs": len(cifs)})
    for field in ["parse_valid", "oxide_chemistry_pass", "composition_novel_vs_snapshot", "structure_novel_vs_snapshot", "prescreen_pass", "shortlist_selected", "target_hit", "stable_by_threshold", "computational_candidate"]:
        counts[field] = sum(bool(row[field]) for row in rows)
    counts["awaiting_independent_DFT_and_phonons"] = sum(row["confirmation_status"] == "shortlist_pending_DFT_and_phonons" for row in rows)
    counts["not_selected_by_shortlist_budget"] = sum(row["confirmation_status"] == "not_selected_by_shortlist_budget" for row in rows)
    counts["reference_structures"] = len(reference_rows)
    counts["reference_reduced_formulas"] = len(ref_formulas)
    counts["matched_mlff_rows"] = sum((row["candidate_id"], "mlff") in screens for row in rows)
    counts["matched_dft_rows"] = sum((row["candidate_id"], "dft") in screens for row in rows)
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "reference_manifest": str(args.reference_manifest.resolve()),
        "reference_manifest_sha256": sha256(args.reference_manifest),
        "reference_snapshots": [{k: row[k] for k in ["dataset_id", "version", "access_date", "license", "_resolved_path", "_observed_sha256", "_observed_structure_count"]} for row in manifest_rows],
        "screen_csv": str(args.screen_csv.resolve()) if args.screen_csv else None,
        "matcher": {"algorithm": "pymatgen.StructureMatcher", "stol": float(novelty_cfg.get("matcher_stol", 0.5)), "primitive_cell": True, "scale": True, "attempt_supercell": True},
        "acceptance": {
            "chemistry": "oxide: oxygen plus at least one non-oxygen element; prohibited elements excluded",
            "band_gap_eV": [low, high],
            "mlff_prescreen_e_hull_meV_per_atom_max": prescreen_hull,
            "shortlist_max_candidates_per_seed": shortlist_max,
            "shortlist_rule": "unique reduced formulas; ascending MLFF hull, then absolute distance to target gap, then candidate_id",
            "final_dft_e_hull_meV_per_atom_max": dft_hull,
            "final_phonon_stability_required": True,
            "novelty_scope": "relative only to the frozen, explicitly listed database snapshots",
            "all_are_computational_only": True,
        },
        "counts": dict(counts),
        "candidate_audit_csv": str(out_csv.resolve()),
    }
    out_json = args.run_dir / "candidate_audit_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
