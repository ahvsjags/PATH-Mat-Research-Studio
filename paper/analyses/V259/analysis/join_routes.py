#!/usr/bin/env python3
"""Join independently scored PATH-Mat route candidates to generated materials.

This is deliberately a join/ranking adapter, not a synthesis planner. The input
route scores must come from a separately versioned PATH-Mat scorer. The score is
interpreted only as the model's source-label score; lower values are sorted first
for review, not asserted to be experimental success probabilities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from pymatgen.core import Composition


def normalize_formula(value: str) -> str:
    return Composition(value).reduced_formula


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-audit", type=Path, required=True)
    ap.add_argument("--route-scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-manifest", type=Path, required=True)
    ap.add_argument("--include-unconfirmed", action="store_true")
    args = ap.parse_args()
    c_fields, candidates = read_csv(args.candidate_audit)
    r_fields, route_rows = read_csv(args.route_scores)
    required_c = {"candidate_id", "reduced_formula", "computational_candidate", "confirmation_status"}
    required_r = {"target_formula", "route_id", "balance_status", "b2_source_label_score", "route_source_id"}
    if not required_c.issubset(c_fields):
        raise SystemExit(f"Candidate audit needs columns: {sorted(required_c)}")
    if not required_r.issubset(r_fields):
        raise SystemExit(f"Route-score table needs columns: {sorted(required_r)}")
    if not args.model_manifest.exists():
        raise SystemExit(f"PATH-Mat model/provenance manifest is required: {args.model_manifest}")
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    for key in ["model_id", "model_version", "training_data_id", "score_semantics"]:
        if not model_manifest.get(key):
            raise SystemExit(f"Route model manifest is missing {key!r}")
    if "source-label" not in model_manifest["score_semantics"].lower():
        raise SystemExit("Model manifest must state that B2 is a source-label score; do not imply experimental success probability.")

    eligible = []
    for c in candidates:
        if args.include_unconfirmed or c["computational_candidate"].strip().lower() in {"true", "1"}:
            eligible.append(c)
    routes_by_formula: dict[str, list[dict[str, str]]] = {}
    for r in route_rows:
        if r["balance_status"].strip().lower() not in {"balanced", "true", "yes"}:
            continue
        try:
            formula = normalize_formula(r["target_formula"])
            score = float(r["b2_source_label_score"])
        except Exception as exc:
            raise SystemExit(f"Invalid route target/score in row {r.get('route_id')}: {exc}") from exc
        if not 0.0 <= score <= 1.0:
            raise SystemExit(f"B2 source-label score must be in [0,1]: {r['route_id']}")
        r["_normalized_formula"] = formula
        r["_numeric_score"] = score
        routes_by_formula.setdefault(formula, []).append(r)

    out_rows = []
    for c in eligible:
        formula = normalize_formula(c["reduced_formula"])
        for rank, r in enumerate(sorted(routes_by_formula.get(formula, []), key=lambda x: (x["_numeric_score"], x["route_id"])), 1):
            out_rows.append({
                "candidate_id": c["candidate_id"],
                "reduced_formula": formula,
                "candidate_confirmation_status": c["confirmation_status"],
                "route_id": r["route_id"],
                "route_source_id": r["route_source_id"],
                "rank_by_b2_source_label_score": rank,
                "b2_source_label_score": r["_numeric_score"],
                "route_balance_status": r["balance_status"],
                "route_actions": r.get("route_actions", ""),
                "precursor_set": r.get("precursor_set", ""),
                "score_interpretation": "PATH-Mat source-label score; lower values prioritized for review; not a success probability",
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["candidate_id", "reduced_formula", "candidate_confirmation_status", "route_id", "route_source_id", "rank_by_b2_source_label_score", "b2_source_label_score", "route_balance_status", "route_actions", "precursor_set", "score_interpretation"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    report = {
        "candidate_audit": str(args.candidate_audit.resolve()),
        "route_scores": str(args.route_scores.resolve()),
        "model_manifest": str(args.model_manifest.resolve()),
        "model_manifest_sha256": hashlib.sha256(args.model_manifest.read_bytes()).hexdigest(),
        "eligible_material_candidates": len(eligible),
        "matched_candidate_route_pairs": len(out_rows),
        "candidate_formulas_without_route": sorted({normalize_formula(c["reduced_formula"]) for c in eligible} - set(routes_by_formula)),
        "interpretation": "Computational route-prioritization output only; it does not establish reaction feasibility, synthesis success, or causal improvement.",
        "output": str(args.out.resolve()),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
