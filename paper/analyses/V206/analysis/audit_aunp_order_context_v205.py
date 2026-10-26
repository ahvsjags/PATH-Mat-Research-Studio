"""Recompute descriptive order-by-context contrasts from published summary values.

This script does not infer batch-level uncertainty. For the 2018 AgNP contrasts,
the final concentrations differ between feed directions; those contrasts are
reported as contextual patterns and are not treated as isolated order effects.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "AUNP_ORDER_CONTEXT_SOURCE_VALUES_V205.csv"
OUTPUT = HERE / "AUNP_ORDER_CONTEXT_AUDIT_V205.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


rows = list(csv.DictReader(INPUT.open(encoding="utf-8-sig", newline="")))
groups = defaultdict(dict)
for row in rows:
    key = (row["study"], row["system"], row["context_value"])
    assert row["order_a"] not in groups[key], f"duplicate order A: {key}"
    groups[key][row["order_a"]] = row
    assert row["order_b"] not in groups[key], f"duplicate order B: {key}"
    groups[key][row["order_b"]] = row

contrasts = []
for (study, system, context), cells in sorted(groups.items()):
    a = next(iter(cells.values()))["order_a"]
    first = cells[a]
    second = cells[first["order_b"]]
    effect = float(first["mean_a_nm"]) - float(first["mean_b_nm"])
    contrasts.append({
        "study": study,
        "system": system,
        "context": context,
        "contrast": f"{first['order_a']} - {first['order_b']}",
        "effect_nm": round(effect, 4),
        "final_composition_matched": first["final_composition_matched"].lower() == "true",
        "particle_count_per_condition": first["particle_count_per_condition"],
        "independent_batch_count_for_size_summary": first["batch_count_for_size_summary"],
        "source_doi": first["source_doi"],
    })

by_context = defaultdict(dict)
for item in contrasts:
    by_context[(item["study"], item["system"])][item["context"]] = item["effect_nm"]

interactions = []
for (study, system), effects in sorted(by_context.items()):
    if study == "Salloum2025":
        low, high = effects["5:1"], effects["17:1"]
        label = "17:1 minus 5:1 order contrast"
    else:
        low, high = effects["slow"], effects["fast"]
        label = "fast minus slow order contrast"
    interactions.append({
        "study": study,
        "system": system,
        "interaction_definition": label,
        "low_context_order_effect_nm": round(low, 4),
        "high_context_order_effect_nm": round(high, 4),
        "difference_in_differences_nm": round(high - low, 4),
        "matched_final_composition": all(
            c["final_composition_matched"]
            for c in contrasts if c["study"] == study and c["system"] == system
        ),
        "inference": "descriptive contrast of published TEM means; no batch-level interval calculated",
    })

# Guardrails against accidental sign or transcription changes.
expected = {
    ("Salloum2025", "AuNP", "5:1"): 1.0,
    ("Salloum2025", "AuNP", "17:1"): -8.0,
    ("Besenhard2018", "AuNP", "slow"): 1.7,
    ("Besenhard2018", "AuNP", "fast"): -3.1,
    ("Besenhard2018", "AgNP", "slow"): 1.6,
    ("Besenhard2018", "AgNP", "fast"): -4.8,
}
for item in contrasts:
    assert item["effect_nm"] == expected[(item["study"], item["system"], item["context"])], item
assert {x["system"]: x["difference_in_differences_nm"] for x in interactions if x["study"] == "Besenhard2018"} == {
    "AgNP": -6.4,
    "AuNP": -4.8,
}

result = {
    "version": "V205",
    "purpose": "recompute order-by-context point contrasts from published nanoparticle summary means",
    "input": INPUT.name,
    "input_sha256": sha256(INPUT),
    "sources": [
        {"doi": "10.1039/D5NR02727F", "type": "peer-reviewed article", "scope": "AuNP order by citrate:gold ratio"},
        {"doi": "10.1039/C8CE01014E", "type": "peer-reviewed article", "scope": "AgNP and AuNP order by feed/mixing regime"},
    ],
    "contrasts": contrasts,
    "interactions": interactions,
    "interpretation": {
        "independent_gold_studies_show_sign_reversal": True,
        "gold_2018_final_composition_matched": True,
        "silver_2018_final_composition_matched": False,
        "silver_2018_isolated_order_effect_claim": False,
        "batch_level_uncertainty_available_for_tem_means": False,
        "particle_level_sd_or_particle_count_is_not_batch_replication": True,
        "does_not_evaluate_path_mat_or_establish_path_mat_policy": True,
    },
}
OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), "contrasts": contrasts, "interactions": interactions}, indent=2))
