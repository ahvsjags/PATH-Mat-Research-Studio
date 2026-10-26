#!/usr/bin/env python3
"""Reassess an existing COD candidate snapshot with narrow occupancy fallback.

This reuses saved API responses/CIFs; it performs no network access and never
modifies the frozen generation run. It emits a corrected per-candidate screen
and a route shortlist excluding exact COD structure duplicates.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymatgen.core import Structure

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE / "agent"))
from pathmat_design_agent import NoveltyAgent, sha256_file  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reassess(run_dir: Path, snapshot_dir: Path) -> dict[str, Any]:
    run_dir, snapshot_dir = run_dir.resolve(), snapshot_dir.resolve()
    output = snapshot_dir / "reassessment_occupancy_tolerance_1.02"
    output.mkdir(exist_ok=False)

    formula_map = json.loads((snapshot_dir / "cod_formula_query_map.json").read_text(encoding="utf-8"))
    retrieval = json.loads((snapshot_dir / "cod_retrieval_manifest.json").read_text(encoding="utf-8"))["retrieved_entries"]
    manifest_path = snapshot_dir / "reference_manifest.csv"
    novelty = NoveltyAgent(manifest_path, matcher_stol=0.5)
    if not novelty.snapshots:
        raise RuntimeError("COD reference snapshot did not load")
    parsed_count = novelty.snapshots[0]["parsed_cif_count"]
    fetched_count = sum(bool(r.get("path")) for r in retrieval.values())
    if parsed_count != fetched_count:
        raise RuntimeError(f"reference parse is incomplete: parsed={parsed_count}, fetched={fetched_count}")

    candidates = [r for r in _read_csv(run_dir / "candidate_ledger.csv")
                  if r.get("candidate_id") and r.get("geometry_pass", "").lower() == "true"]
    formula_results = formula_map["candidates"]
    if len(candidates) != len(formula_results):
        raise RuntimeError(f"candidate/query count mismatch: {len(candidates)} vs {len(formula_results)}")

    screen_rows = []
    for candidate in candidates:
        formula = candidate["formula"]
        q = formula_results[formula]
        ids = q["cod_ids"]
        # Preserve each downloaded reference digest; also prove candidate CIF identity.
        for cod_id in ids:
            rec = retrieval.get(str(cod_id))
            if not rec or rec.get("path") is None:
                continue
            ref_path = snapshot_dir / rec["path"]
            if sha256_file(ref_path) != rec["sha256"]:
                raise ValueError(f"COD CIF hash mismatch: {ref_path}")
        candidate_path = run_dir / candidate["cif_path"]
        if sha256_file(candidate_path) != candidate["cif_sha256"]:
            raise ValueError(f"candidate CIF hash mismatch: {candidate_path}")
        matched = novelty.check(formula, Structure.from_file(candidate_path))
        if matched["matched_reference"]:
            structure_status = "matching_COD_structure_found"
        elif not ids:
            structure_status = "no_exact_formula_entry_in_COD_query"
        else:
            structure_status = "formula_recorded_but_no_structure_match_in_COD_query"
        screen_rows.append({
            "candidate_id": candidate["candidate_id"], "formula": formula,
            "candidate_cif": candidate["cif_path"], "candidate_cif_sha256": candidate["cif_sha256"],
            "cod_hill_formula_query": q["query_formula_hill"],
            "cod_formula_result_count": q["result_count"], "cod_ids": ";".join(ids),
            "formula_status": "formula_recorded_in_COD" if ids else "no_exact_formula_record_in_COD_query",
            "structure_status": structure_status, "matched_reference": matched["matched_reference"],
            "cod_cif_parse_method": ";".join(sorted({retrieval[str(i)].get("parse_method", "") for i in ids
                                                         if str(i) in retrieval})),
            "snapshot_scope": "dated exact-formula COD query; not a complete database snapshot",
        })
    _write_csv(output / "cod_candidate_novelty_screen.csv", screen_rows)

    duplicate_ids = {r["candidate_id"] for r in screen_rows if r["structure_status"] == "matching_COD_structure_found"}
    retained_ids = {r["candidate_id"] for r in screen_rows if r["candidate_id"] not in duplicate_ids}
    route_rows = _read_csv(run_dir / "pathmat_route_hypotheses.csv")
    retained_routes = [r for r in route_rows if r.get("candidate_id") in retained_ids]
    removed_routes = [r for r in route_rows if r.get("candidate_id") in duplicate_ids]
    _write_csv(output / "cod_distinct_candidate_shortlist.csv",
               [r for r in screen_rows if r["candidate_id"] in retained_ids])
    _write_csv(output / "cod_filtered_pathmat_route_hypotheses.csv", retained_routes)

    counts = Counter(r["structure_status"] for r in screen_rows)
    formula_hits = sum(r["formula_status"] == "formula_recorded_in_COD" for r in screen_rows)
    summary = {
        "reassessed_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": str(snapshot_dir),
        "reference_snapshot_sha256": novelty.snapshots[0]["sha256"],
        "COD_reference_CIFs_fetched": fetched_count,
        "COD_reference_CIFs_parsed": parsed_count,
        "CIF_parser_fallback": "CifParser occupancy_tolerance=1.02 after default parse failure",
        "structure_matcher": "StructureMatcher(stol=0.5, primitive_cell=True, scale=True, attempt_supercell=True); oxidation states ignored",
        "candidate_count": len(screen_rows),
        "formula_records_found": formula_hits,
        "formulas_without_exact_COD_record": len(screen_rows) - formula_hits,
        "exact_COD_structure_matches": counts["matching_COD_structure_found"],
        "formula_known_but_structure_not_matched": counts["formula_recorded_but_no_structure_match_in_COD_query"],
        "no_exact_formula_record": counts["no_exact_formula_entry_in_COD_query"],
        "COD_distinct_candidate_structures_retained": len(retained_ids),
        "candidate_specific_route_hypotheses_before_COD_dedup": len(route_rows),
        "candidate_specific_route_hypotheses_excluded_as_structure_duplicates": len(removed_routes),
        "route_hypotheses_after_structure_dedup": len(retained_routes),
        "retained_route_model_abstentions": sum(r.get("model_abstain", "").lower() == "true" for r in retained_routes),
        "positive_control_formula_search_records": formula_map["control"]["returned_entries"],
        "interpretation": "The COD screen identifies known formulas and exact structural duplicates within a dated candidate-focused formula query. COD-distinct means no StructureMatcher match among retrieved COD CIFs for the exact queried formula; formula no-hit is limited to the COD search and is not a claim of worldwide novelty.",
    }
    (output / "cod_novelty_reassessment.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(output, summary, screen_rows)
    return summary


def _write_report(out: Path, s: dict[str, Any], rows: list[dict[str, str]]) -> None:
    matches = [r for r in rows if r["structure_status"] == "matching_COD_structure_found"]
    known_nonmatches = [r for r in rows if r["structure_status"] == "formula_recorded_but_no_structure_match_in_COD_query"]
    lines = [
        "# COD candidate novelty reassessment",
        "",
        f"- Candidates: {s['candidate_count']}",
        f"- Exact formula records in COD: {s['formula_records_found']}",
        f"- No exact formula record in COD: {s['formulas_without_exact_COD_record']}",
        f"- Exact generated-structure matches: {s['exact_COD_structure_matches']}",
        f"- Formula-known, structure-not-matched: {s['formula_known_but_structure_not_matched']}",
        f"- Candidate structures retained after duplicate removal: {s['COD_distinct_candidate_structures_retained']}",
        f"- Route hypotheses after removing matching structures: {s['route_hypotheses_after_structure_dedup']} ({s['retained_route_model_abstentions']} abstentions)",
        "",
        "## Exact structure duplicates removed",
        "",
    ]
    lines.extend(f"- {r['formula']} — {r['matched_reference']}" for r in matches)
    if not matches:
        lines.append("- None")
    lines += ["", "## Formula present, generated structure not matched", ""]
    lines.extend(f"- {r['formula']} — COD IDs {r['cod_ids']}" for r in known_nonmatches)
    if not known_nonmatches:
        lines.append("- None")
    lines += ["", "## Scope", "", s["interpretation"], ""]
    (out / "COD_NOVELTY_REASSESSMENT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = reassess(args.run_dir, args.snapshot_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
