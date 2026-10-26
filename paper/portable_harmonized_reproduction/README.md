# Source-only harmonized comparison

This runner is independent of the earlier PATH-Mat workspace directories. It reproduces only the new three-model comparison, not the historical B0–B3 fits. Inputs are not bundled: the internal conflict-filtered normalized cohort and raw sol-gel records require authorized access and applicable redistribution rights. Public source provenance is cited by the analysis package. A source DOI is not the accession for this derived analysis.

Set these environment variables to lawful, checksum-matched files:

- `PATHMAT_PRIMARY_JSONL`: exact conflict-filtered normalized JSONL (SHA256 4cc3210fd0d782e26cfc75ecfb84dd1afb12c4d34c545174704b21f90a3b30e8).
- `PATHMAT_LEE_RAW_GZ`: source raw JSON gzip (SHA256 59fb0608e594721a4055fb6a623f180651f2d7aef737a5d4e13db63595b039c0).
- `PATHMAT_SOLGEL_ZIP`: source sol-gel ZIP (SHA256 65da667fcecedf73942a4b8b898e14a1e6479c7371be3fa50162ac9e3f0d5d03).
- `PATHMAT_ANALYSIS_OUT`: writable analysis output directory.

Run `python run_harmonized_zero_shot.py` after installing the pinned dependencies. The source-only coefficients/calibration/thresholds are locked before external outcomes are loaded. All three models and six contrasts are saved. Outputs include internal row-level predictions; do not publish them automatically. Inputs and full output are not part of the proposed aggregate deposit.

The archive was inspected before this test was designed. Results are exploratory, conditional comparisons. Stored action-list order is not verified physical chronology. The primary success is a modest external ranking increment, not reliable transferred probabilities or universally confirmed order utility.
