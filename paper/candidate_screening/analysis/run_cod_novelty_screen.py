#!/usr/bin/env python3
"""Build a candidate-focused COD reference snapshot and compare generated CIFs.

The query is exact empirical formula for every retained PATH-Mat candidate.
Positive controls verify the COD formula convention before any no-hit is used.
The output is a dated, hash-manifested targeted snapshot, not a full COD mirror.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifParser

HERE = Path(__file__).resolve()
MODULE = HERE.parents[1]
AGENT_DIR = MODULE / "agent"
sys.path.insert(0, str(AGENT_DIR))
from pathmat_design_agent import snapshot_sha256, sha256_file  # noqa: E402

COD_RESULT = "https://www.crystallography.net/cod/result"
COD_ENTRY = "https://www.crystallography.net/cod/{cod_id}.cif"
USER_AGENT = "PATH-Mat candidate structure audit/1.0 (read-only research query)"
COD_LICENSE = "CC0-1.0"


def cod_hill_formula(formula: str) -> str:
    """Return formula tokens in COD's documented Hill order."""
    comp = Composition(formula)
    amounts = comp.get_el_amt_dict()
    if "C" in amounts:
        order = ["C"]
        if "H" in amounts:
            order.append("H")
        order.extend(sorted(el for el in amounts if el not in {"C", "H"}))
    else:
        order = sorted(amounts)
    tokens = []
    for el in order:
        amount = amounts[el]
        n = int(round(amount)) if abs(amount - round(amount)) < 1e-8 else amount
        tokens.append(f"{el}{'' if n == 1 else n}")
    return " ".join(tokens)


def _get(url: str, timeout: int = 30, retries: int = 3) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                return response.read()
        except Exception as exc:  # network retries are recorded by the caller on final failure
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def query_formula(formula: str) -> tuple[str, bytes, list[dict[str, Any]]]:
    query_formula_text = cod_hill_formula(formula)
    params = {
        "formula": query_formula_text,
        "format": "json",
        "include_duplicates": "1",
        "include_errors": "1",
        "include_theoretical": "1",
    }
    url = f"{COD_RESULT}?{urllib.parse.urlencode(params)}"
    raw = _get(url)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"COD returned non-list JSON for {formula}: {type(value).__name__}")
    return url, raw, value


def _load_candidates(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "candidate_ledger.csv"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    candidates = [r for r in rows if r.get("candidate_id") and r.get("geometry_pass", "").lower() == "true"]
    if not candidates:
        raise ValueError(f"no geometry-screened candidate rows in {path}")
    formulas = [r["formula"] for r in candidates]
    if len(set(formulas)) != len(formulas):
        raise ValueError("candidate ledger contains duplicate reduced formulas")
    return candidates


def run(run_dir: Path, output_dir: Path, pause_s: float = 0.2) -> Path:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_cod_formula_queries"
    cif_dir = output_dir / "reference_cifs"
    raw_dir.mkdir()
    cif_dir.mkdir()
    candidates = _load_candidates(run_dir)

    # A positive control catches formula-order or endpoint errors before no-hits.
    control_url, control_raw, control = query_formula("SrTiO3")
    if not control:
        raise RuntimeError("COD positive control SrTiO3 returned no records; aborting novelty calls")
    (raw_dir / "CONTROL_SrTiO3.json").write_bytes(control_raw)
    control_record = {"formula": "SrTiO3", "query_formula": cod_hill_formula("SrTiO3"),
                      "url": control_url, "returned_entries": len(control),
                      "raw_sha256": hashlib.sha256(control_raw).hexdigest(), "expected": "nonempty"}

    formula_results: dict[str, dict[str, Any]] = {}
    all_records: dict[str, dict[str, Any]] = {}
    for i, candidate in enumerate(candidates, 1):
        formula = candidate["formula"]
        url, raw, records = query_formula(formula)
        qhash = hashlib.sha256(raw).hexdigest()
        qfile = raw_dir / f"{qhash[:16]}.json"
        qfile.write_bytes(raw)
        ids = sorted({str(rec["file"]) for rec in records if rec.get("file")})
        formula_results[formula] = {
            "candidate_formula": formula,
            "query_formula_hill": cod_hill_formula(formula),
            "query_url": url,
            "query_json_path": qfile.relative_to(output_dir).as_posix(),
            "query_json_sha256": qhash,
            "result_count": len(records),
            "cod_ids": ids,
        }
        for record in records:
            if record.get("file"):
                all_records[str(record["file"])] = record
        if i < len(candidates):
            time.sleep(max(0.0, pause_s))

    retrieval: dict[str, dict[str, Any]] = {}
    for i, cod_id in enumerate(sorted(all_records), 1):
        url = COD_ENTRY.format(cod_id=urllib.parse.quote(cod_id, safe=""))
        try:
            raw = _get(url)
            text = raw.decode("utf-8", errors="replace")
            if "data_" not in text.lower() or "_cell_length_a" not in text.lower():
                raise ValueError("response did not resemble a CIF with unit-cell data")
            path = cif_dir / f"{cod_id}.cif"
            path.write_bytes(raw)
            try:
                try:
                    parsed = Structure.from_file(path)
                    parse_method = "pymatgen_default"
                except Exception:
                    # COD entries can contain small occupancy-rounding excesses;
                    # accept only a narrow tolerance and preserve the raw CIF.
                    parsed = CifParser(str(path), occupancy_tolerance=1.02).parse_structures()[0]
                    parse_method = "pymatgen_cifparser_occupancy_tolerance_1.02"
                parsed_formula = parsed.composition.reduced_formula
                parse_status = "parsed"
            except Exception as exc:
                parsed_formula = ""
                parse_method = "unparsed"
                parse_status = f"parse_error:{type(exc).__name__}"
            retrieval[cod_id] = {
                "cod_id": cod_id, "url": url, "path": path.relative_to(output_dir).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "parse_status": parse_status, "parse_method": parse_method,
                "parsed_reduced_formula": parsed_formula,
                "metadata": all_records[cod_id],
            }
        except Exception as exc:
            retrieval[cod_id] = {"cod_id": cod_id, "url": url, "parse_status": "retrieval_error",
                                 "error": f"{type(exc).__name__}: {exc}", "metadata": all_records[cod_id]}
        if i < len(all_records):
            time.sleep(max(0.0, pause_s))

    # Save the exact raw query map before structure matching.
    (output_dir / "cod_formula_query_map.json").write_text(
        json.dumps({"control": control_record, "candidates": formula_results}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    (output_dir / "cod_retrieval_manifest.json").write_text(
        json.dumps({"retrieved_entries": retrieval}, indent=2, ensure_ascii=False), encoding="utf-8")

    cifs_by_id = {k: output_dir / v["path"] for k, v in retrieval.items()
                  if v.get("parse_status") == "parsed" and v.get("path")}
    cif_parse_success_count = len(cifs_by_id)
    cif_download_count = sum(bool(v.get("path")) for v in retrieval.values())
    dir_digest = snapshot_sha256(cif_dir)
    manifest_path = output_dir / "reference_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset_id", "version", "access_date", "license",
                                                   "local_path", "sha256", "structure_count", "coverage"])
        writer.writeheader()
        writer.writerow({
            "dataset_id": "Crystallography Open Database (COD)",
            "version": datetime.now(timezone.utc).strftime("targeted-query-%Y%m%d"),
            "access_date": datetime.now(timezone.utc).date().isoformat(), "license": COD_LICENSE,
            "local_path": "reference_cifs", "sha256": dir_digest, "structure_count": cif_download_count,
            "coverage": f"Exact empirical-formula queries for {len(candidates)} retained candidate formulas; not a full COD mirror",
        })

    # Load the locally hash-verified snapshot through the same novelty agent used
    # by generation, with oxidation-state annotations ignored during structural match.
    from pathmat_design_agent import NoveltyAgent  # noqa: E402
    novelty = NoveltyAgent(manifest_path, matcher_stol=0.5)
    if not novelty.snapshots or novelty.snapshots[0]["sha256"] != dir_digest:
        raise RuntimeError("local COD snapshot failed the NoveltyAgent digest check")

    rows_out: list[dict[str, Any]] = []
    structure_match_counts = 0
    formula_hit_count = 0
    retrieval_failures = sum(v.get("parse_status") == "retrieval_error" for v in retrieval.values())
    parse_failures = sum(str(v.get("parse_status", "")).startswith("parse_error") for v in retrieval.values())
    for candidate in candidates:
        formula = candidate["formula"]
        result = formula_results[formula]
        ids = result["cod_ids"]
        if ids:
            formula_hit_count += 1
        structure = Structure.from_file(run_dir / candidate["cif_path"])
        match = novelty.check(formula, structure)
        parsed_ids = [cid for cid in ids if cid in cifs_by_id]
        failed_ids = [cid for cid in ids if cid not in cifs_by_id]
        if match["matched_reference"]:
            structure_status = "matching_COD_structure_found"
            structure_match_counts += 1
        elif not ids:
            structure_status = "no_exact_formula_entry_in_COD_query"
        elif failed_ids or not parsed_ids:
            structure_status = "unresolved_some_COD_CIFs_not_parsed_or_retrieved"
        else:
            structure_status = "formula_recorded_but_no_structure_match_in_COD_query"
        formula_status = "formula_recorded_in_COD" if ids else "no_exact_formula_record_in_COD_query"
        rows_out.append({
            "candidate_id": candidate["candidate_id"], "formula": formula,
            "candidate_cif": candidate["cif_path"], "candidate_cif_sha256": candidate["cif_sha256"],
            "cod_hill_formula_query": result["query_formula_hill"], "cod_formula_result_count": result["result_count"],
            "cod_ids": ";".join(ids), "formula_status": formula_status,
            "structure_status": structure_status, "matched_reference": match["matched_reference"],
            "cod_cif_parse_failures_or_missing": ";".join(failed_ids),
            "snapshot_scope": "COD exact-formula queries for candidate formula set; retrieval date in manifest",
        })
    screen_path = output_dir / "cod_candidate_novelty_screen.csv"
    with screen_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    query_incomplete = len(formula_results) != len(candidates)
    summary = {
        "schema_version": 1,
        "dataset": "Crystallography Open Database (COD)",
        "query_endpoint": COD_RESULT,
        "entry_endpoint_template": COD_ENTRY,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "license": COD_LICENSE,
        "scope": "Exact Hill-formula searches for each geometry-retained PATH-Mat candidate; not a full-database snapshot.",
        "candidate_count": len(candidates), "formula_queries_completed": len(formula_results),
        "positive_control": control_record,
        "candidate_formulas_with_COD_records": formula_hit_count,
        "candidate_formulas_without_exact_COD_records": len(candidates) - formula_hit_count,
        "candidate_structures_matching_COD_CIF": structure_match_counts,
        "retrieval_failures": retrieval_failures, "parse_failures": parse_failures,
        "raw_query_hash_count": len(formula_results) + 1,
        "reference_cif_count": cif_download_count,
        "reference_cif_parse_success_count": cif_parse_success_count,
        "reference_cif_directory_sha256": dir_digest,
        "reference_manifest": str(manifest_path), "candidate_screen": str(screen_path),
        "query_complete": not query_incomplete and retrieval_failures == 0,
        "interpretation": "A no-hit means absent from this dated exact-formula COD query, not globally novel. A formula hit with no structure match denotes an ordered candidate not matched to retrieved COD CIFs under StructureMatcher(stol=0.5, primitive_cell=True, scale=True, attempt_supercell=True).",
    }
    (output_dir / "cod_novelty_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(output_dir, summary, rows_out, retrieval)
    return output_dir


def _write_report(out: Path, summary: dict[str, Any], rows: list[dict[str, Any]], retrieval: dict[str, dict[str, Any]]) -> None:
    formula_hits = [r for r in rows if r["formula_status"] == "formula_recorded_in_COD"]
    struct_matches = [r for r in rows if r["structure_status"] == "matching_COD_structure_found"]
    lines = [
        "# PATH-Mat candidate-focused COD novelty screen",
        "",
        f"- Retrieval time (UTC): {summary['retrieved_utc']}",
        f"- Candidate structures screened: {summary['candidate_count']}",
        f"- Positive-control query (SrTiO3): {summary['positive_control']['returned_entries']} COD records",
        f"- Candidate formulas with COD exact-formula records: {summary['candidate_formulas_with_COD_records']}",
        f"- Candidate formulas without COD exact-formula records: {summary['candidate_formulas_without_exact_COD_records']}",
        f"- Candidate structures matched to retrieved COD CIFs: {summary['candidate_structures_matching_COD_CIF']}",
        f"- Retrieved reference CIFs / retrieval failures / parse failures: {summary['reference_cif_count']} / {summary['retrieval_failures']} / {summary['parse_failures']}",
        f"- Candidate-formula reference-directory SHA-256: `{summary['reference_cif_directory_sha256']}`",
        "",
        "## Formula hits",
        "",
        "| Candidate formula | COD IDs | structure status |",
        "|---|---|---|",
    ]
    if formula_hits:
        lines.extend(f"| {r['formula']} | {r['cod_ids']} | {r['structure_status']} |" for r in formula_hits)
    else:
        lines.append("| None | — | — |")
    lines += [
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "A formula-level hit means the composition has a COD record; it does not establish that the generated ordered structure is already reported. A no-hit is limited to the COD exact-formula endpoint and retrieval date. The query map, raw responses, retrieved CIFs, hashes and per-candidate decisions are retained alongside this report.",
        "",
        "COD API query syntax and supported formula search are documented at https://wiki.crystallography.net/RESTful_API/. COD structures and database content are provided under CC0; original structure authors should be acknowledged.",
        "",
    ]
    (out / "COD_NOVELTY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="existing PATH-Mat run with candidate_ledger.csv and CIFs")
    parser.add_argument("--out-dir", type=Path, required=True, help="new empty output directory for COD responses and audit")
    parser.add_argument("--pause-s", type=float, default=0.2, help="pause between read-only COD requests")
    args = parser.parse_args()
    path = run(args.run_dir, args.out_dir, args.pause_s)
    print(path)
    print(json.dumps(json.loads((path / "cod_novelty_summary.json").read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
