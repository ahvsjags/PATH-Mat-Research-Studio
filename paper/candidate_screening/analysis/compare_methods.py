#!/usr/bin/env python3
"""Compare same-budget candidate-generation runs without cherry-picking seeds."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def bootstrap_mean_ci(values: list[float], n_boot: int, seed: int) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_boot):
        estimates.append(mean(rng.choices(values, k=len(values))))
    estimates.sort()
    return [estimates[int(0.025 * (n_boot - 1))], estimates[int(0.975 * (n_boot - 1))]]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Required provenance artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def generator_signature(generator: dict[str, Any]) -> dict[str, Any]:
    # Only the guidance strength should differ in the preregistered paired contrast.
    allowed_difference = {"guidance_factor"}
    return {key: value for key, value in generator.items() if key not in allowed_difference}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap-replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()

    by_method_seed: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    provenance: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for audit in sorted(args.runs_root.rglob("candidate_audit.csv")):
        method = audit.parent.parent.name
        seed = audit.parent.name
        run_manifest = load_json(audit.parent.parent / "run_manifest.json")
        audit_summary = load_json(audit.parent / "candidate_audit_summary.json")
        try:
            seed_value = int(seed.split("_")[-1])
        except ValueError as exc:
            raise SystemExit(f"Seed directory must end with an integer: {seed}") from exc
        run_entry = next((r for r in run_manifest.get("runs", []) if int(r.get("seed", -1)) == seed_value), None)
        if run_entry is None or run_entry.get("status") != "completed":
            raise SystemExit(f"No completed generation entry for {method}/{seed}")
        if not run_manifest.get("checkpoint_sha256"):
            raise SystemExit(f"Run lacks immutable checkpoint digest: {method}/{seed}")
        if audit_summary.get("config_sha256") != run_manifest.get("config_sha256"):
            raise SystemExit(f"Config digest mismatch between run and audit for {method}/{seed}")
        if not audit_summary.get("reference_manifest_sha256"):
            raise SystemExit(f"Audit lacks frozen reference-manifest digest: {method}/{seed}")
        with audit.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if audit_summary.get("counts", {}).get("generated_cifs") != len(rows):
            raise SystemExit(f"Candidate audit count mismatch for {method}/{seed}")
        counts = audit_summary.get("counts", {})
        if counts.get("matched_mlff_rows") != len(rows):
            raise SystemExit(f"Primary comparison requires one matched MLFF prescreen per CIF: {method}/{seed}")
        if counts.get("shortlist_selected") != counts.get("matched_dft_rows"):
            raise SystemExit(f"Every frozen-shortlist structure must have a matched DFT/phonon row before comparison: {method}/{seed}")
        if counts.get("awaiting_independent_DFT_and_phonons", 0) != 0:
            raise SystemExit(f"Primary comparison is incomplete: pending DFT/phonon shortlist rows in {method}/{seed}")
        generator = run_manifest.get("generator", {})
        expected = int(generator.get("batch_size", 0)) * int(generator.get("num_batches", 0))
        if generator.get("max_candidates_per_seed") is not None:
            expected = min(expected, int(generator["max_candidates_per_seed"]))
        if len(rows) != expected:
            raise SystemExit(f"Generated sample count differs from registered budget for {method}/{seed}: {len(rows)} != {expected}")
        by_method_seed[method][seed] = rows
        provenance[method][seed] = {
            "generator": generator,
            "generator_signature": generator_signature(generator),
            "target": run_manifest.get("target"),
            "checkpoint_sha256": run_manifest["checkpoint_sha256"],
            "model_config_sha256": run_manifest.get("model_config_sha256"),
            "acceptance": audit_summary.get("acceptance"),
            "reference_manifest_sha256": audit_summary["reference_manifest_sha256"],
            "reference_snapshots": audit_summary.get("reference_snapshots"),
        }

    if not by_method_seed:
        raise SystemExit(f"No candidate_audit.csv files found below {args.runs_root}")

    report: dict[str, Any] = {
        "primary_metric": "computational_candidates_per_1000_generated",
        "acceptance_definition": "valid oxide, database-novel composition and structure, independent target hit, independent E_hull threshold pass and phonon-stable",
        "registered_budget_per_seed_by_method": {
            method: {
                seed: int(generator.get("batch_size", 0)) * int(generator.get("num_batches", 0))
                if generator.get("max_candidates_per_seed") is None
                else min(
                    int(generator.get("batch_size", 0)) * int(generator.get("num_batches", 0)),
                    int(generator["max_candidates_per_seed"]),
                )
                for seed, item in method_seeds.items()
                for generator in [provenance[method][seed]["generator"]]
            }
            for method, method_seeds in by_method_seed.items()
        },
        "methods": {},
        "pairwise_differences": {},
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.seed, "unit": "independent generation seed; intervals reported only with >=10 seeds"},
    }
    seed_rates: dict[str, list[float]] = {}
    for method, seeds in sorted(by_method_seed.items()):
        per_seed = {}
        rates = []
        for seed, rows in sorted(seeds.items()):
            n = len(rows)
            hits = sum(row.get("computational_candidate", "").lower() in {"true", "1"} for row in rows)
            rate = 1000.0 * hits / n if n else 0.0
            rates.append(rate)
            per_seed[seed] = {"n_generated": n, "accepted_candidates": hits, "per_1000": rate}
        seed_rates[method] = rates
        report["methods"][method] = {
            "n_seeds": len(rates),
            "mean_candidates_per_1000": mean(rates) if rates else 0.0,
            "seed_bootstrap_95_ci": bootstrap_mean_ci(rates, args.bootstrap_replicates, args.seed) if len(rates) >= 10 else None,
            "inferential_status": "seed-level interval available" if len(rates) >= 10 else "descriptive_only_fewer_than_10_independent_seeds",
            "per_seed": per_seed,
        }

    methods = sorted(seed_rates)
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            seeds_a = provenance[a]
            seeds_b = provenance[b]
            common = sorted(set(seeds_a) & set(seeds_b))
            checks = {
                "same_seed_set": set(seeds_a) == set(seeds_b),
                "same_checkpoint": all(seeds_a[s]["checkpoint_sha256"] == seeds_b[s]["checkpoint_sha256"] for s in common),
                "same_model_config": all(seeds_a[s]["model_config_sha256"] == seeds_b[s]["model_config_sha256"] for s in common),
                "same_target": all(seeds_a[s]["target"] == seeds_b[s]["target"] for s in common),
                "same_generation_except_guidance": all(seeds_a[s]["generator_signature"] == seeds_b[s]["generator_signature"] for s in common),
                "different_guidance_factors": all(seeds_a[s]["generator"].get("guidance_factor") != seeds_b[s]["generator"].get("guidance_factor") for s in common),
                "same_acceptance_gates": all(seeds_a[s]["acceptance"] == seeds_b[s]["acceptance"] for s in common),
                "same_reference_snapshots": all(seeds_a[s]["reference_manifest_sha256"] == seeds_b[s]["reference_manifest_sha256"] for s in common),
                "matched_per_seed_budget": all(
                    report["methods"][a]["per_seed"].get(s, {}).get("n_generated") == report["methods"][b]["per_seed"].get(s, {}).get("n_generated")
                    for s in common
                ),
            }
            if not all(checks.values()):
                report["pairwise_differences"][f"{a} - {b}"] = {
                    "status": "not_comparable",
                    "provenance_checks": checks,
                    "shared_seed_ids": common,
                }
                continue
            shared_seeds = sorted(set(by_method_seed[a]) & set(by_method_seed[b]))
            if set(by_method_seed[a]) != set(by_method_seed[b]):
                report["pairwise_differences"][f"{a} - {b}"] = {
                    "status": "not_computed",
                    "reason": "seed identifiers differ; paired bootstrap requires exact seed matching",
                    "shared_seed_ids": shared_seeds,
                }
                continue
            per_a = report["methods"][a]["per_seed"]
            per_b = report["methods"][b]["per_seed"]
            diffs = [per_a[s]["per_1000"] - per_b[s]["per_1000"] for s in shared_seeds]
            report["pairwise_differences"][f"{a} - {b}"] = {
                "mean_difference_per_1000": mean(diffs),
                "paired_seed_bootstrap_95_ci": bootstrap_mean_ci(diffs, args.bootstrap_replicates, args.seed + i + 1) if len(diffs) >= 10 else None,
                "inferential_status": "paired_seed_interval_available" if len(diffs) >= 10 else "descriptive_only_fewer_than_10_independent_seeds",
                "seed_ids": shared_seeds,
                "seed_differences": diffs,
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
