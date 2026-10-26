"""Reproduce a source-grounded raw-data analysis of Au–peptide timed intervention.

This is an external published-materials case study, not a PATH-Mat test set.
It reads only public Zenodo files already selectively extracted and hashed in
the V222 intake manifest. Experimental condition is the intentional delay
before adding Z2 to an initially Z2M246I-containing Au synthesis.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "zenodo_8303993" / "Data_Share _V2" / "Data"
UV = DATA / "UV-Vis" / "Intervention_Kinetics_UVVIS"
DLS = DATA / "DLS" / "2023_01_27_Z2_Z2M246I"
ANALYSIS = ROOT / "analysis"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_from_manifest(member: str, manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [x for x in manifest["selected_members"] if x["archive_member"] == member]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one manifest entry for {member}, found {len(matches)}")
    return matches[0]["sha256"]


def verify_hash(path: Path, member: str, manifest_path: Path) -> str:
    actual = file_sha256(path)
    expected = sha256_from_manifest(member, manifest_path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {member}: {actual} != {expected}")
    return actual


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def main() -> None:
    intake_manifest = ANALYSIS / "ZENODO_8303993_UVVIS_INTAKE_MANIFEST_V222.json"
    dls_manifest = ANALYSIS / "ZENODO_8303993_DLS_INTAKE_MANIFEST_V222.json"
    source_files = {
        "uvvis_si": UV / "SI.csv",
        "water_xlsx": UV / "Water.xlsx",
        "dls_si": DLS / "SI.csv",
        "dls_export": DLS / "exported.txt",
    }
    member_names = {
        "uvvis_si": "Data_Share _V2/Data/UV-Vis/Intervention_Kinetics_UVVIS/SI.csv",
        "water_xlsx": "Data_Share _V2/Data/UV-Vis/Intervention_Kinetics_UVVIS/Water.xlsx",
        "dls_si": "Data_Share _V2/Data/DLS/2023_01_27_Z2_Z2M246I/SI.csv",
        "dls_export": "Data_Share _V2/Data/DLS/2023_01_27_Z2_Z2M246I/exported.txt",
    }
    hash_manifest = {}
    for k, p in source_files.items():
        mf = dls_manifest if k.startswith("dls_") else intake_manifest
        hash_manifest[k] = verify_hash(p, member_names[k], mf)

    # The public author notebook corrects pre-intervention spectra to the
    # post-addition sample volume: Water_600 minus Water_520. Terminal spectra
    # are after addition and are not volume-adjusted.
    water = pd.read_excel(source_files["water_xlsx"], sheet_name=None)
    corr = water["Water_600"].iloc[:, 0].to_numpy(dtype=float) - water["Water_520"].iloc[:, 0].to_numpy(dtype=float)
    uv_si = pd.read_csv(source_files["uvvis_si"])
    uv_rows = []
    uv_timeseries_rows = []
    for r in uv_si.itertuples(index=False):
        fname, delay = str(r.Filename), r.Label
        if not str(fname).startswith("2023_01_19_"):
            continue
        delay = float(delay)
        path = UV / fname
        member = f"Data_Share _V2/Data/UV-Vis/Intervention_Kinetics_UVVIS/{fname}"
        source_hash = verify_hash(path, member, intake_manifest)
        df = pd.read_csv(path)
        wl = df["Wavelength"].to_numpy(dtype=float)
        times = np.array([float(c) for c in df.columns[1:]], dtype=float)
        if len(wl) != len(corr):
            raise RuntimeError(f"Water correction length {len(corr)} != {len(wl)} for {fname}")
        response_by_time = []
        for j, t in enumerate(times):
            y = df.iloc[:, j + 1].to_numpy(dtype=float).copy()
            if t < delay:
                y += corr
            short_mask = (wl >= 450) & (wl <= 600)
            long_mask = (wl >= 700) & (wl <= 995)
            a530 = float(y[np.argmin(np.abs(wl - 530))])
            a995 = float(y[np.argmin(np.abs(wl - 995))])
            short_area = trapz(y[short_mask], wl[short_mask])
            long_area = trapz(y[long_mask], wl[long_mask])
            ratio = long_area / short_area if short_area != 0 else math.nan
            uv_timeseries_rows.append({
                "source_file": fname,
                "source_sha256": source_hash,
                "z2_addition_delay_min": int(delay),
                "elapsed_time_min": int(t),
                "A530": a530,
                "A995": a995,
                "A995_over_A530": a995 / a530 if a530 else math.nan,
                "A450_600_integral": short_area,
                "A700_995_integral": long_area,
                "long_to_short_integral_ratio": ratio,
                "is_terminal_spectrum": bool(t == times.max()),
            })
            response_by_time.append((t, a530, a995, short_area, long_area, ratio))
        terminal = response_by_time[-1]
        uv_rows.append({
            "source_file": fname,
            "source_sha256": source_hash,
            "experiment_date": "2023-01-19",
            "z2_addition_delay_min": int(delay),
            "n_spectra": int(len(times)),
            "time_min_first": float(times.min()),
            "time_min_last": float(times.max()),
            "wavelength_nm_first": float(wl.min()),
            "wavelength_nm_last": float(wl.max()),
            "terminal_elapsed_time_min": int(terminal[0]),
            "terminal_A530": terminal[1],
            "terminal_A995": terminal[2],
            "terminal_A995_over_A530": terminal[2] / terminal[1] if terminal[1] else math.nan,
            "terminal_A450_600_integral": terminal[3],
            "terminal_A700_995_integral": terminal[4],
            "terminal_long_to_short_integral_ratio": terminal[5],
            "independent_synthesis_replicates_reported_in_archive": 1,
            "pre_intervention_adjustment": "author notebook: add Water_600 minus Water_520 to spectra with elapsed time < Z2 delay",
        })

    # Join DLS sample metadata to instrument rows. Well Index i maps to sample
    # A(i+1); controls are explicitly kept separate from intervention levels.
    dls_meta = pd.read_csv(source_files["dls_si"])
    dls = pd.read_csv(source_files["dls_export"], sep="\t", encoding="cp1252")
    dls["sample_well"] = dls["Sample Name"].astype(str).str.extract(r"\b(A\d+)_")
    dls_rows = []
    for _, m in dls_meta.iterrows():
        idx = int(m["Well Index"])
        sample_well = f"A{idx + 1}"
        z2 = float(m["Z2 concentration molarity"])
        delay4 = int(m["Delay4"])
        candidates = dls[dls["sample_well"] == sample_well]
        if z2 == 0:
            candidates = dls[dls["Sample Name"].astype(str).str.startswith(sample_well + "_Control")]
            if len(candidates) == 0:
                raise RuntimeError("Could not join no-Z2 control DLS sample")
            # Two records are repeat readings of one control sample; retain both
            # with no inference of independent synthesis replication.
        elif len(candidates) != 1:
            raise RuntimeError(f"Expected one DLS row for {sample_well}, found {len(candidates)}")
        for _, obs in candidates.iterrows():
            if z2 > 0:
                # Figure 10 in the source paper reports delays 1,3,...,13 min;
                # released plate metadata stores the inter-step Delay4=0,2,...,12.
                delay_label = delay4 + 1
                group = "timed_Z2_intervention"
            else:
                delay_label = None
                group = "no_Z2_control"
            dls_rows.append({
                "source_sample_name": obs["Sample Name"],
                "well_index": idx,
                "well": m["Well"],
                "group": group,
                "metadata_Delay4_min": delay4,
                "source_paper_delay_label_min": delay_label,
                "Z2_concentration_M": z2,
                "Z2M246I_concentration_M": float(m["Z2M246I concentration molarity"]),
                "HAuCl4_concentration_M": float(m["HAuCl4 concentration molarity"]),
                "HEPES_concentration_M": float(m["HEPES concentration molarity"]),
                "total_volume_uL": float(m["Total Volume"]),
                "record_number": int(obs["Record Number"]),
                "measurement_datetime": str(obs["Measurement Date and Time"]),
                "Z_average_nm": float(obs["Z-Average (d.nm)"]),
                "intensity_mean_nm": float(obs["Intensity Mean (d.nm)"]),
                "PDI": float(obs["PdI"]),
                "instrument_size_peak_nm": float(obs["Size Peak (d.nm)"]),
                "instrument_size_peak_percent": float(obs["Size Peak (Percent)"]),
                "independent_synthesis_replicates_reported_in_archive": 1,
                "interpretation": "single instrument record per intervention level; Z-average is an apparent DLS size descriptor for mixed anisotropic shapes",
            })

    uv_df = pd.DataFrame(uv_rows).sort_values("z2_addition_delay_min")
    uv_ts_df = pd.DataFrame(uv_timeseries_rows).sort_values(["z2_addition_delay_min", "elapsed_time_min"])
    dls_df = pd.DataFrame(dls_rows).sort_values(["group", "source_paper_delay_label_min", "measurement_datetime"], na_position="first")
    uv_df.to_csv(ANALYSIS / "AU_INTERVENTION_UVVIS_TERMINAL_V222.csv", index=False, float_format="%.10g")
    uv_ts_df.to_csv(ANALYSIS / "AU_INTERVENTION_UVVIS_TIMESERIES_V222.csv", index=False, float_format="%.10g")
    dls_df.to_csv(ANALYSIS / "AU_INTERVENTION_DLS_RAW_SUMMARY_V222.csv", index=False, float_format="%.10g")

    uv_ratio = uv_df["terminal_long_to_short_integral_ratio"].to_numpy(dtype=float)
    uv_a995 = uv_df["terminal_A995"].to_numpy(dtype=float)
    dls_intervention = dls_df[dls_df["group"] == "timed_Z2_intervention"].sort_values("source_paper_delay_label_min")
    x = dls_intervention["source_paper_delay_label_min"].to_numpy(dtype=float)
    y = dls_intervention["Z_average_nm"].to_numpy(dtype=float)
    # Descriptive rank association only; no p-values/CIs because one synthesis
    # unit is reported per delay and the timing conditions are not replicate rows.
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    n_inc = int(np.sum(np.diff(y) > 0))
    summary = {
        "source": {
            "article_doi": "10.1021/acs.langmuir.3c01577",
            "data_doi": "10.5281/zenodo.8303993",
            "data_url": "https://zenodo.org/records/8303993",
            "intake": "byte-range selective; full 1.17-GB archive not downloaded",
            "all_input_sha256_verified_against_intake_manifest": True,
        },
        "design": {
            "system": "Au nanoparticle/nanoplatelet growth with Z2M246I initially present and timed Z2 addition",
            "UVVis_delay_conditions_min": [int(v) for v in uv_df["z2_addition_delay_min"]],
            "UVVis_independent_synthesis_unit_per_delay": 1,
            "DLS_delay_conditions_min": [int(v) for v in x],
            "DLS_independent_synthesis_unit_per_delay": 1,
            "matched_final_composition_in_DLS_metadata": bool(
                dls_intervention["Z2_concentration_M"].nunique() == 1
                and dls_intervention["Z2M246I_concentration_M"].nunique() == 1
                and dls_intervention["HAuCl4_concentration_M"].nunique() == 1
                and dls_intervention["HEPES_concentration_M"].nunique() == 1
                and dls_intervention["total_volume_uL"].nunique() == 1
            ),
            "controls": "no-Z2 control present in DLS; UV-Vis no-Z2 and no-Z2M246I controls are from 2023-01-30, distinct experimental date",
        },
        "results": {
            "UVVis_terminal_elapsed_time_min": int(uv_df["terminal_elapsed_time_min"].iloc[0]),
            "UVVis_long_to_short_area_ratio_by_delay": [round(float(v), 6) for v in uv_ratio],
            "UVVis_ratio_strictly_increasing_across_four_delays": bool(np.all(np.diff(uv_ratio) > 0)),
            "UVVis_A995_by_delay": [round(float(v), 6) for v in uv_a995],
            "UVVis_A995_delay13_over_delay1_fold": float(uv_a995[-1] / uv_a995[0]),
            "DLS_Z_average_nm_by_delay": [round(float(v), 6) for v in y],
            "DLS_Z_average_delay13_minus_delay1_nm": float(y[-1] - y[0]),
            "DLS_Z_average_delay13_over_delay1_fold": float(y[-1] / y[0]),
            "DLS_Z_average_spearman_rho_descriptive_only": rho,
            "DLS_adjacent_increases_of_six": n_inc,
            "DLS_control_records_same_control_sample_technical_repeat": int(
                (dls_df["group"] == "no_Z2_control").sum()
            ),
        },
        "scope": {
            "supports": "strong, source-specific, cross-modal association between later Z2 intervention and altered Au particle optical/DLS signatures under matched recorded final composition",
            "does_not_support": [
                "PATH-Mat zero-shot prediction or model transfer",
                "universal materials generalization",
                "a prospectively evaluated PATH-Mat recommendation policy",
                "replicate-level inferential uncertainty or a universal causal effect",
                "phase purity or crystallographic ground truth from UV-Vis/DLS alone",
            ],
            "causal_scope": "controlled intervention is consistent with a system-specific timing effect; published raw archive has one independent synthesis unit per timing level, so this reanalysis adds no replicate-level causal CI",
        },
        "inputs": hash_manifest,
    }
    (ANALYSIS / "AU_TIMED_INTERVENTION_EVIDENCE_SUMMARY_V222.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
