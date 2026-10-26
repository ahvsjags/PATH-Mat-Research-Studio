#!/usr/bin/env python3
"""Inventory early public DACHS AutoMOFs synthesis files for usable order contrasts.

This script is an eligibility audit only. It does not promote unmatched samples
to replicates and does not infer characterization outcomes absent from source.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "inputs" / "dachs"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


def decode(x):
    if isinstance(x, h5py.Dataset):
        x = x[()]
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    try:
        if getattr(x, "shape", None) == ():
            x = x.item()
    except Exception:
        pass
    if isinstance(x, (int, float)):
        return x
    return str(x)


def scalar(f, path):
    if path not in f:
        return None
    try:
        return decode(f[path])
    except Exception:
        return None


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []
for p in sorted(DATA.glob("record_*/DACHS/MOFs/AutoMOFs_*/Synthesis/*.h5")):
    m = re.search(r"AutoMOFs_(\d+)_([A-Z]\d+)\.h5$", p.name)
    if not m:
        continue
    campaign, sid = int(m.group(1)), m.group(2)
    with h5py.File(p, "r") as f:
        desc = scalar(f, "DACHS/Synthesis/Description") or ""
        params = f.get("DACHS/Synthesis/DerivedParameters")
        derived = {}
        if isinstance(params, h5py.Group):
            for name, grp in params.items():
                value = scalar(f, f"DACHS/Synthesis/DerivedParameters/{name}/Value")
                if value is not None:
                    derived[name] = value
        log = []
        raw = f.get("DACHS/Synthesis/RawLog")
        if isinstance(raw, h5py.Group):
            for _, grp in raw.items():
                if not isinstance(grp, h5py.Group):
                    continue
                msg = scalar(grp, "Message")
                if msg is None:
                    continue
                log.append({
                    "message": str(msg),
                    "timestamp": scalar(grp, "TimeStamp"),
                    "value": scalar(grp, "Value"),
                    "unit": scalar(grp, "Unit"),
                })
        rows.append({
            "campaign": campaign,
            "sample_id": sid,
            "filename": p.name,
            "description": desc,
            "group_prefix": sid[0],
            "injection_speed": derived.get("InjectionSpeed"),
            "metal_to_linker_ratio": derived.get("MetalToLinkerRatio"),
            "stirrer_rpm": derived.get("StirrerSpeed"),
            "reaction_time_s": derived.get("ReactionTime"),
            "temperature_c": derived.get("LabTemperature"),
            "yield": derived.get("SynthesisYield"),
            "final_mass_g": derived.get("FinalMass"),
            "derived_parameter_names": ";".join(sorted(derived)),
            "injection_log_json": json.dumps(log, ensure_ascii=False),
            "sha256": sha(p),
        })

with (OUT / "DACHS_EARLY_BATCH_SAMPLE_INVENTORY.csv").open("w", newline="", encoding="utf-8-sig") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary = {}
for campaign in sorted({r["campaign"] for r in rows}):
    sub = [r for r in rows if r["campaign"] == campaign]
    summary[str(campaign)] = {
        "n_samples": len(sub),
        "prefix_counts": {k: sum(r["group_prefix"] == k for r in sub) for k in sorted({r["group_prefix"] for r in sub})},
        "n_with_yield": sum(r["yield"] is not None for r in sub),
        "descriptions": {r["sample_id"]: r["description"] for r in sub},
        "has_matched_speed_levels": any(r["injection_speed"] is not None for r in sub),
        "samples_with_order_log": [r["sample_id"] for r in sub if "injection" in r["injection_log_json"].lower()],
    }
(OUT / "DACHS_EARLY_BATCH_INVENTORY_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "descriptions"} for k, v in summary.items()}, indent=2, ensure_ascii=False))
