#!/usr/bin/env python3
"""Aggregate-only exact reduced-formula overlap audit; never exports route rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import pandas as pd
from pymatgen.core import Composition


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def reduced(formula: object) -> str | None:
    try:
        return Composition(str(formula)).reduced_formula
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--astral-table", required=True, type=Path)
    ap.add_argument("--lee-archive", required=True, type=Path,
                    help="authorized local archive; only aggregate counts are written")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    astral = pd.read_csv(a.astral_table)
    astral_norm = {x for x in (reduced(v) for v in astral["target_formula"].dropna()) if x}
    lee_norm: set[str] = set()
    lee_rows = 0
    invalid = 0
    with a.lee_archive.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source = row.get("source")
            source_id = source.get("source_id") if isinstance(source, dict) else source
            if source_id != "lee2025_impurities":
                continue
            lee_rows += 1
            formula = reduced(row.get("target_formula"))
            if formula:
                lee_norm.add(formula)
            else:
                invalid += 1
    shared = astral_norm & lee_norm
    result = {
        "schema_version": "pathmat_astral_lee_formula_overlap_v173",
        "status": "aggregate_only_complete",
        "normalization": "pymatgen Composition(...).reduced_formula; exact formula only, not structure/family similarity",
        "astral_table_sha256": sha256(a.astral_table),
        "lee_archive_sha256": sha256(a.lee_archive),
        "lee_archive_rows_scanned_for_source": int(lee_rows),
        "lee_rows_with_unparseable_target_formula": int(invalid),
        "astral_unique_reduced_target_formulas": int(len(astral_norm)),
        "astral_target_formulas_exactly_present_in_lee_source_archive": int(len(shared)),
        "astral_target_formulas_not_exactly_present_in_lee_source_archive": int(len(astral_norm - lee_norm)),
        "astral_rows_with_target_formula_seen_in_lee_archive": int(sum(reduced(v) in shared for v in astral["target_formula"])),
        "astral_rows_with_target_formula_not_seen_in_lee_archive": int(sum(reduced(v) not in lee_norm for v in astral["target_formula"])),
        "privacy_boundary": "No route record, DOI, target formula, precursor list or row-level join key is emitted.",
        "interpretation": "Exact formula overlap is a chemical novelty check only. A target not seen as the exact formula may still be compositionally or structurally related to training materials.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k.endswith("formulas") or k.endswith("archive") or k.endswith("target_formula_seen_in_lee_archive")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        raise SystemExit(main())
