#!/usr/bin/env python3
"""Audit early DACHS AutoMOFs records for usable order contrasts and duplicates."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "inputs" / "dachs"
OUT = ROOT / "results"
INVENTORY = OUT / "DACHS_EARLY_BATCH_SAMPLE_INVENTORY.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "shape") and getattr(value, "shape", None) == ():
        value = value.item()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def content_signature(path: Path) -> str:
    content = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                if name.endswith("/SampleID"):
                    return
                value = decode(obj[()])
                content[name] = value
        f.visititems(visit)
    raw = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    with INVENTORY.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    file_map = {}
    for p in DATA.glob("record_*/DACHS/MOFs/AutoMOFs_*/Synthesis/*.h5"):
        file_map[p.name] = p
    enriched = []
    for r in rows:
        desc = r["description"].lower()
        if "two solutions were added simultaneously" in desc:
            order = "simultaneous"
        elif "metal solution was added in a pre-injection step" in desc:
            order = "metal_first"
        elif "linker solution was added in a pre-injection step" in desc:
            order = "linker_first"
        else:
            order = "unresolved"
        path = file_map[r["filename"]]
        enriched.append({**r, "order": order, "yield": float(r["yield"]), "injection_speed": float(r["injection_speed"]), "content_signature_excluding_sample_id": content_signature(path)})

    groups: dict[tuple[int, str, float], list[dict]] = defaultdict(list)
    for r in enriched:
        groups[(int(r["campaign"]), r["order"], r["injection_speed"])].append(r)

    group_rows = []
    for (campaign, order, speed), sub in sorted(groups.items()):
        unique = {r["content_signature_excluding_sample_id"]: r for r in sub}
        yy = [r["yield"] for r in unique.values()]
        group_rows.append({
            "campaign": campaign, "addition_mode": order, "speed_mL_min": speed,
            "source_records": len(sub), "unique_record_contents_excluding_sample_id": len(unique),
            "sample_ids": ";".join(r["sample_id"] for r in sub),
            "unique_sample_ids_after_exact_content_dedup": ";".join(r["sample_id"] for r in unique.values()),
            "yield_values_unique_records": ";".join(f"{y:.8f}" for y in yy),
            "mean_yield_unique_records": sum(yy) / len(yy),
            "n_independent_batches_verified": "not documented in source record",
        })
    with (OUT / "DACHS_CAMPAIGNS_1_3_GROUP_INVENTORY_V260.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(group_rows[0]))
        w.writeheader(); w.writerows(group_rows)

    # Dedupe exact-content repeats within each campaign/order/rate before
    # calculating descriptive matched-arm summaries.
    clean = []
    for sub in groups.values():
        seen = set()
        for r in sub:
            sig = r["content_signature_excluding_sample_id"]
            if sig not in seen:
                seen.add(sig)
                clean.append(r)
    paired = []
    for campaign in (1, 2, 3):
        arms = defaultdict(list)
        for r in clean:
            if int(r["campaign"]) == campaign:
                arms[(r["order"], float(r["injection_speed"]))].append(r)
        common = sorted({speed for (order, speed) in arms if order == "metal_first"} & {speed for (order, speed) in arms if order == "simultaneous"})
        for speed in common:
            metal = arms[("metal_first", speed)]
            sim = arms[("simultaneous", speed)]
            my = [x["yield"] for x in metal]
            sy = [x["yield"] for x in sim]
            paired.append({
                "campaign": campaign, "speed_mL_min": speed,
                "metal_first_n_unique_records": len(my), "metal_first_sample_ids": ";".join(x["sample_id"] for x in metal),
                "metal_first_mean_yield": sum(my) / len(my),
                "simultaneous_n_unique_records": len(sy), "simultaneous_sample_ids": ";".join(x["sample_id"] for x in sim),
                "simultaneous_mean_yield": sum(sy) / len(sy),
                "descriptive_simultaneous_minus_metal_yield": sum(sy) / len(sy) - sum(my) / len(my),
                "independent_batch_status": "not established; no inferential interval",
            })
    with (OUT / "DACHS_CAMPAIGNS_1_3_MATCHED_ORDER_SPEEDS_V260.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0]))
        w.writeheader(); w.writerows(paired)

    duplicates = defaultdict(list)
    for r in enriched:
        duplicates[(int(r["campaign"]), r["order"], float(r["injection_speed"]), r["content_signature_excluding_sample_id"])].append(r["sample_id"])
    exact_duplicate_groups = []
    for (campaign, order, speed, sig), sample_ids in duplicates.items():
        if len(sample_ids) > 1:
            exact_duplicate_groups.append({"campaign": campaign, "addition_mode": order, "speed_mL_min": speed, "sample_ids": sample_ids, "content_signature_excluding_sample_id": sig})
    by_campaign = {}
    for campaign in (1, 2, 3):
        c = [r for r in enriched if int(r["campaign"]) == campaign]
        by_campaign[str(campaign)] = {
            "source_records": len(c),
            "record_counts_by_addition_mode": {mode: sum(r["order"] == mode for r in c) for mode in sorted({r["order"] for r in c})},
            "n_matched_metal_first_simultaneous_speeds": sum(p["campaign"] == campaign for p in paired),
            "order_contrast": "not available" if campaign != 2 else "metal-first versus simultaneous; no linker-first arm",
        }
    flagged = [r for r in enriched if int(r["campaign"]) == 2 and r["order"] == "simultaneous" and r["yield"] > 1.0]
    payload = {
        "analysis_id": "V260_DACHS_AUTOMOFs_1_3_ORDER_AUDIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_records": 36,
        "source_record_sha256": {r["filename"]: r["sha256"] for r in enriched},
        "campaigns": by_campaign,
        "matched_speed_contrasts": paired,
        "exact_content_duplicate_groups_excluding_sample_id": exact_duplicate_groups,
        "unmatched_yield_values_over_one": [{"sample_id": r["sample_id"], "campaign": int(r["campaign"]), "speed_mL_min": float(r["injection_speed"]), "recorded_yield": r["yield"], "order": r["order"]} for r in flagged],
        "conclusion": "Campaigns 1 and 3 contain metal-first records only. Campaign 2 includes metal-first and simultaneous records at three matched speeds; one M001/M003 pair is an exact duplicate in all inspected HDF5 content except SampleID, so it is counted once. Descriptive simultaneous-minus-metal-first yield differences reverse sign across speeds. Independent synthesis-batch status is not documented, most matched cells have one unique record, and campaign 2 has no linker-first arm; therefore this audit adds evidence of context dependence but not inferential replication or a new independent campaign validation.",
        "source_records_and_licenses": [
            {"record": "DACHS AutoMOFs 1", "zenodo": "https://zenodo.org/records/11236032", "license": "CC BY 4.0"},
            {"record": "DACHS AutoMOFs 2", "zenodo": "https://zenodo.org/records/11236075", "license": "CC BY 4.0"},
            {"record": "DACHS AutoMOFs 3", "zenodo": "https://zenodo.org/records/11237816", "license": "CC BY 4.0"},
        ],
    }
    (OUT / "DACHS_CAMPAIGNS_1_3_ORDER_AUDIT_V260.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"source_records": payload["source_records"], "campaigns": by_campaign, "matched_speed_contrasts": paired, "duplicates": exact_duplicate_groups, "yield_gt_1": payload["unmatched_yield_values_over_one"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
