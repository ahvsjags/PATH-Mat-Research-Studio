#!/usr/bin/env python3
"""Run- and formulation-matched acidity audit for the public RE-MOF PXRD grid.

This is an exploratory, source-adapted association analysis. It is explicitly
not a randomized treatment-effect estimate, a frozen PATH-Mat transfer test,
or a validation of impurity/purity labels.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs" / "rare_earth_mof_campaign.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 23416
DRAWS = 100_000
NO_PRODUCT = {"no solid product", "no crystalline product"}
UNKNOWN = {"", "unknown phase"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load() -> pd.DataFrame:
    d = pd.read_csv(INPUT)
    d["run_id"] = d["SampleID"].astype(str).str.extract(r"MG2-(\d+)-", expand=False)
    d["well_order"] = d["SampleID"].astype(str).str.extract(r"-(\d+)$", expand=False).astype(int)
    d["primary_phase_normalized"] = d["Primary Phase"].fillna("").astype(str).str.strip().str.lower()
    # Keep only single-metal experiments. The endpoint is source-assigned
    # presence of a named crystalline primary product, not phase purity.
    d = d[~d["Metal"].astype(str).str.contains(r"\+", regex=True)].copy()
    p = d["primary_phase_normalized"]
    d["y_named_crystalline_product"] = np.where(
        p.isin(NO_PRODUCT), 0.0, np.where(p.isin(UNKNOWN), np.nan, 1.0)
    )
    return d.reset_index(drop=True)


def run_cluster_bootstrap(run_values: pd.Series, *, draws: int = DRAWS, seed: int = SEED) -> dict:
    vals = run_values.dropna().astype(float)
    arr = vals.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(draws, len(arr)))
    boot = arr[idx].mean(axis=1)
    return {
        "n_run_clusters": int(len(arr)),
        "mean_equal_run": float(arr.mean()),
        "bootstrap_percentile_95ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def main() -> None:
    d = load()
    known = d[d["y_named_crystalline_product"].notna()].copy()
    known["y"] = known["y_named_crystalline_product"].astype(int)

    # Structural audit: temperature is one-to-one with run blocks here. It is
    # therefore not treated as an independently identified temperature effect.
    run_temp_n = d.groupby("run_id")["Temp"].nunique()
    temp_runs = d.groupby("Temp")["run_id"].nunique()
    if not (run_temp_n == 1).all():
        raise AssertionError("At least one source run spans multiple temperatures")
    run_temp = d.groupby("run_id")["Temp"].first()

    # The source records enumerate acetic-acid levels in blocks within each
    # run × metal sequence. Capture this because execution order can confound
    # the observational acidity association.
    acid_sequence_audit = {}
    for (run, metal), g in d.groupby(["run_id", "Metal"]):
        ordered = g.sort_values("well_order")["AA Conc"].astype(float).to_numpy()
        acid_sequence_audit[f"{run}|{metal}"] = {
            "n_rows": int(len(g)),
            "temperature_C": float(g["Temp"].iloc[0]),
            "acid_levels_in_sample_order": list(dict.fromkeys(ordered.tolist())),
            "nondecreasing_acid_in_sample_order": bool(np.all(np.diff(ordered) >= 0)),
        }

    keys = ["run_id", "Metal", "Linker"]
    p = known.pivot_table(index=keys, columns="AA Conc", values="y", aggfunc="first")
    if not {0, 2, 5, 10}.issubset(set(p.columns)):
        raise AssertionError("Expected acetic-acid levels 0, 2, 5, and 10 M")

    # Pairwise within-run, within-metal, within-linker contrasts.
    pair_tables = {}
    run_contrasts = {}
    for dose in [2, 5, 10]:
        q = p[[0, dose]].dropna().copy()
        q["delta_high_minus_zero"] = q[dose] - q[0]
        q["n_pair_outcome"] = np.select(
            [q["delta_high_minus_zero"] < 0, q["delta_high_minus_zero"] > 0],
            ["zero_acid_only_product", "high_acid_only_product"], default="tie"
        )
        flat = q.reset_index()
        pair_tables[dose] = flat
        by_run = flat.groupby("run_id")["delta_high_minus_zero"].mean()
        run_contrasts[dose] = by_run

    # Full-dose slope per exact run × metal × linker cell, using all four
    # measured acid levels and only complete four-level strata.
    complete = p[[0, 2, 5, 10]].dropna().copy()
    doses = np.array([0.0, 2.0, 5.0, 10.0])
    dose_center = doses - doses.mean()
    slope = ((complete.to_numpy(dtype=float) - complete.to_numpy(dtype=float).mean(axis=1, keepdims=True)) * dose_center).sum(axis=1) / (dose_center**2).sum()
    complete["linear_slope_per_M"] = slope
    slope_by_run = complete.reset_index().groupby("run_id")["linear_slope_per_M"].mean()

    contrast_summary = {}
    pair_frames = []
    run_rows = pd.DataFrame({"run_id": sorted(d["run_id"].dropna().unique())})
    run_rows["temperature_C"] = run_rows["run_id"].map(run_temp)
    for dose, q in pair_tables.items():
        q["acid_high_M"] = dose
        pair_frames.append(q)
        delta = q["delta_high_minus_zero"]
        per_run = q.groupby("run_id")["delta_high_minus_zero"].mean()
        nonzero = per_run[per_run != 0]
        neg = int((nonzero < 0).sum())
        pos = int((nonzero > 0).sum())
        sign_p = float(binomtest(min(neg, pos), n=neg + pos, p=0.5, alternative="two-sided").pvalue) if neg + pos else None
        contrast_summary[f"{dose}_M_minus_0_M"] = {
            "n_matched_run_metal_linker_pairs": int(len(q)),
            "pairwise_direction_counts": {str(k): int(v) for k, v in q["n_pair_outcome"].value_counts().to_dict().items()},
            "pooled_pair_mean_risk_difference": float(delta.mean()),
            "equal_run_mean_risk_difference": float(per_run.mean()),
            "equal_run_cluster_bootstrap": run_cluster_bootstrap(per_run),
            "run_direction_counts": {"negative": neg, "positive": pos, "zero": int((per_run == 0).sum())},
            "exact_two_sided_sign_test_nonzero_runs": sign_p,
        }
        run_rows[f"n_pairs_{dose}M_vs_0M"] = q.groupby("run_id").size()
        run_rows[f"mean_delta_{dose}M_minus_0M"] = per_run

    slope_boot = run_cluster_bootstrap(slope_by_run)
    sign_slope = slope_by_run[slope_by_run != 0]
    slope_neg = int((sign_slope < 0).sum())
    slope_pos = int((sign_slope > 0).sum())
    slope_sign_p = float(binomtest(min(slope_neg, slope_pos), n=slope_neg + slope_pos, p=0.5, alternative="two-sided").pvalue) if slope_neg + slope_pos else None
    dose_means = known.groupby("AA Conc")["y"].agg(["mean", "count"]).reset_index()
    matched_dose_means = complete[[0, 2, 5, 10]].mean(axis=0)

    result = {
        "schema": "pathmat_v234_re_mof_within_run_acid_matched_audit",
        "status": "exploratory post hoc matched association; not randomized, not a causal estimate, not frozen PATH-Mat transfer",
        "source": "Gaidimas et al., Chemical Science 2026, DOI 10.1039/D5SC09992G; Zenodo DOI 10.5281/zenodo.17902549",
        "input_file": INPUT.name,
        "input_sha256": sha256(INPUT),
        "cohort": {
            "single_metal_rows": int(len(d)),
            "known_endpoint_rows": int(len(known)),
            "positive_named_crystalline_primary_product": int(known["y"].sum()),
            "negative_explicit_no_solid_or_no_crystalline_product": int((known["y"] == 0).sum()),
            "unknown_or_missing_primary_phase_excluded": int(d["y_named_crystalline_product"].isna().sum()),
            "metals": int(known["Metal"].nunique()),
            "linkers": int(known["Linker"].nunique()),
            "run_blocks": int(known["run_id"].nunique()),
        },
        "endpoint": "named source-assigned crystalline primary product=1; explicit no-solid/no-crystalline-product=0; unknown/missing excluded. This is not phase purity or impurity absence.",
        "primary_matched_estimand": "Difference in observed crystalline-primary-product fraction between 10 M and 0 M acetic acid within the same run × metal × linker cell; run means equally weighted.",
        "matched_contrasts": contrast_summary,
        "four_level_dose_trend": {
            "complete_run_metal_linker_strata": int(len(complete)),
            "run_balanced_linear_slope_per_M": slope_boot,
            "run_slope_direction_counts": {"negative": slope_neg, "positive": slope_pos, "zero": int((slope_by_run == 0).sum())},
            "exact_two_sided_sign_test_nonzero_runs": slope_sign_p,
            "matched_equal_stratum_rates_by_acid_M": {str(k): float(v) for k, v in matched_dose_means.to_dict().items()},
            "unadjusted_observed_rates_by_acid_M": dose_means.to_dict(orient="records"),
        },
        "design_audit": {
            "runs_with_single_temperature": int((run_temp_n == 1).sum()),
            "temperature_to_run_counts": {str(k): int(v) for k, v in temp_runs.to_dict().items()},
            "temperature_is_run_block_aliased": True,
            "acid_levels_are_non_decreasing_in_sample_order_within_all_run_metal_blocks": bool(all(x["nondecreasing_acid_in_sample_order"] for x in acid_sequence_audit.values())),
            "run_details": acid_sequence_audit,
            "interpretation": "Matching fixes run, temperature, metal and linker; however, within each run × metal block acid levels appear in ascending order in the source SampleID sequence, and randomized treatment order is not documented. Any execution-order effect therefore remains a plausible confound. Temperature effects are not separately identified from run blocks.",
        },
        "claim_boundary": [
            "Supports a source-specific, within-run association between acid condition and a PXRD-assigned primary-product endpoint, independent of publication-text length.",
            "Does not estimate causal acid effects because randomization is undocumented and acid level is correlated with sample sequence.",
            "Does not establish phase purity, absence of secondary phases, a zero-shot transfer, a generic route-order effect, or PATH-Mat-guided optimization.",
        ],
        "bootstrap": {"unit": "run_id cluster; each run contributes one run-level mean", "draws": DRAWS, "seed": SEED, "interval": "percentile 95%"},
    }
    if result["cohort"]["single_metal_rows"] != 1344 or result["cohort"]["known_endpoint_rows"] != 1276:
        raise AssertionError(f"Unexpected cohort reconciliation: {result['cohort']}")
    if len(pair_tables[10]) != 304 or int((pair_tables[10]["delta_high_minus_zero"] < 0).sum()) != 22 or int((pair_tables[10]["delta_high_minus_zero"] > 0).sum()) != 2:
        raise AssertionError("Unexpected exact 10 M vs 0 M matched contrast")
    if len(complete) != 303:
        raise AssertionError("Unexpected four-level complete-stratum count")

    (OUT / "V234_RE_MOF_WITHIN_RUN_ACID_MATCHED_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.concat(pair_frames, ignore_index=True).to_csv(OUT / "V234_RE_MOF_WITHIN_RUN_ACID_MATCHED_PAIRS.csv", index=False)
    run_rows.to_csv(OUT / "V234_RE_MOF_WITHIN_RUN_ACID_MATCHED_BY_RUN.csv", index=False)
    print(json.dumps({"cohort": result["cohort"], "10M_vs_0M": contrast_summary["10_M_minus_0_M"], "four_level_slope": result["four_level_dose_trend"], "design_audit": {k:v for k,v in result["design_audit"].items() if k != "run_details"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
