#!/usr/bin/env python3
"""Retrospective budget-aware hybrid replay for the AIRES held-out linker set.

The first 10 attempts follow the fixed combined-feature ranking; after that,
the policy switches to random exploration among untried records. This is an
exploratory policy analysis of already inspected public labels, not prospective
PATH-Mat validation or an independent confirmatory test.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT.parent / "PATH-Mat_Evidence_Strengthening_V211" / "results" / "AIRES_UNSEEN_LINKER_PREDICTIONS_V211.csv"
OUT = ROOT / "results"
BUDGETS = [10, 25, 50, 100, 200, 400]
EXPLORATION_SWITCH = 10
N_SIM = 50_000
SEED = 20261017


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discover_from_order(order: np.ndarray, y: np.ndarray, groups: np.ndarray, budgets: list[int]) -> dict[int, int]:
    """One attempt is one reaction row; positive discovery removes that linker."""
    seen_positive: set[str] = set()
    attempts = hits = 0
    out: dict[int, int] = {}
    for idx in order:
        linker = str(groups[idx])
        if linker in seen_positive:
            continue
        attempts += 1
        if y[idx] == 1:
            hits += 1
            seen_positive.add(linker)
        if attempts in budgets:
            out[attempts] = hits
        if attempts >= max(budgets):
            break
    for b in budgets:
        out.setdefault(b, hits)
    return out


def ranked_order(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.lexsort((rng.random(len(scores)), -scores))


def hybrid_curve(scores: np.ndarray, y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> dict[int, int]:
    # Execute a ten-attempt exploitation tranche using the static ranker.
    rank = ranked_order(scores, rng)
    positive_linkers: set[str] = set()
    tested: set[int] = set()
    attempts = hits = 0
    curve: dict[int, int] = {}
    for idx in rank:
        g = str(groups[idx])
        if g in positive_linkers:
            continue
        tested.add(int(idx))
        attempts += 1
        if y[idx] == 1:
            hits += 1
            positive_linkers.add(g)
        if attempts in BUDGETS:
            curve[attempts] = hits
        if attempts >= EXPLORATION_SWITCH:
            break
    # Continue by uniform random exploration among untried records; a linker
    # found positive is retired, matching the source's discovery accounting.
    remaining = [i for i in range(len(y)) if i not in tested and str(groups[i]) not in positive_linkers]
    rng.shuffle(remaining)
    for idx in remaining:
        g = str(groups[idx])
        if g in positive_linkers:
            continue
        attempts += 1
        if y[idx] == 1:
            hits += 1
            positive_linkers.add(g)
        if attempts in BUDGETS:
            curve[attempts] = hits
        if attempts >= max(BUDGETS):
            break
    for b in BUDGETS:
        curve.setdefault(b, hits)
    return curve


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    with INPUT.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    y = np.asarray([int(r["Crystal_clf"]) for r in rows], int)
    groups = np.asarray([r["LinkerCanonicalSmiles"] for r in rows], str)
    score = np.asarray([float(r["score_chem_plus_conditions"]) for r in rows], float)
    if len(rows) != 1702 or len(set(groups)) != 48:
        raise RuntimeError(f"Unexpected held-out cohort: rows={len(rows)}, linkers={len(set(groups))}")
    rng = np.random.default_rng(SEED)
    arrays = {key: [] for key in ("ranked_static", "hybrid_rank10_then_random", "random")}
    for _ in range(N_SIM):
        ranked = ranked_order(score, rng)
        arrays["ranked_static"].append(discover_from_order(ranked, y, groups, BUDGETS))
        arrays["hybrid_rank10_then_random"].append(hybrid_curve(score, y, groups, rng))
        arrays["random"].append(discover_from_order(rng.permutation(len(y)), y, groups, BUDGETS))
    rows_out = []
    for b in BUDGETS:
        summary = {k: np.asarray([rep[b] for rep in v], float) for k, v in arrays.items()}
        for name, values in summary.items():
            rows_out.append({
                "budget_reaction_attempts": b,
                "policy": name,
                "mean_new_successful_linkers": float(values.mean()),
                "q025": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "q975": float(np.quantile(values, 0.975)),
                "probability_at_least_static_ranked_mean": float(np.mean(values >= summary["ranked_static"].mean())) if name != "ranked_static" else 1.0,
                "probability_at_least_random_mean": float(np.mean(values >= summary["random"].mean())) if name != "random" else 1.0,
            })
    out_csv = OUT / "AIRES_HYBRID_BUDGET_POLICY_V260.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    payload = {
        "analysis_id": "V260_AIRES_RANK10_THEN_RANDOM_REPLAY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(rows), "n_linkers": len(set(groups)), "n_positive_rows": int(y.sum()),
        "budget_switch": EXPLORATION_SWITCH, "simulation_replicates": N_SIM, "seed": SEED,
        "input_sha256": sha256(INPUT),
        "policies": "Static combined-feature rank; hybrid uses first 10 attempts by that rank and randomizes the remaining untried rows; random ordering baseline.",
        "status": "post_hoc_exploratory_static_replay_on_public_previously_inspected_labels; not prospective; not PATH-Mat frozen-weight test; no new experiments",
        "accounting": "One attempt is one labeled reaction row. A linker is counted once after its first positive outcome, and later rows for that successful linker are skipped.",
        "results": rows_out,
        "interpretation": "The hybrid policy is intended as a practical response to budget sensitivity, not evidence that the underlying static model is universally superior. The cutoff of 10 is chosen for this exploratory replay and must be locked before any new external or prospective validation.",
    }
    (OUT / "AIRES_HYBRID_BUDGET_POLICY_V260.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in rows_out:
        if row["policy"] == "hybrid_rank10_then_random":
            print(row["budget_reaction_attempts"], round(row["mean_new_successful_linkers"], 3), round(row["q025"], 3), round(row["q975"], 3))


if __name__ == "__main__":
    main()
