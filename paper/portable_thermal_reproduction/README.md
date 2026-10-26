# Portable thermal-endpoint reproduction

This local preparation package contains analysis code, fixed plans and aggregate checks. It does not include the internal primary cohort or unlicensed sol-gel data. No public deposition or accession is implied.

## Source

Precursor Genome v1.0 is published under CC BY 4.0 at [Zenodo](https://zenodo.org/records/21285546). The JSON ledger is supplied by its [source GitHub repository](https://github.com/lauren-walters/precursor-genome). The ledger used here has SHA256 `f8b8aabb5b8718ad912075b9aa10373a5e81659b496348f14c926e432eb42dac`. Retrieve it lawfully from the source; the runner refuses a different snapshot. Raw diffraction/refinement archives are unnecessary for this ledger-derived reanalysis, which does not perform new refinement.

## Environment and inputs

Python 3.11.9. `requirements.txt` pins the exact computational dependency versions. In PowerShell, from this directory:

```powershell
$env:PATHMAT_PUBLIC_LEDGER = 'ABSOLUTE_PATH_TO_LEDGER_JSON'
$env:PATHMAT_ANALYSIS_OUT = 'ABSOLUTE_PATH_TO_OUTPUT_DIRECTORY'
python run_thermal_controls.py
```

Input and output locations are explicit. No PATH-Mat version folder or IBEX connection is required. The script writes all five split/cohort analyses, endpoint mapping and change audit, fold definitions, prediction arrays and the aggregate comparison table. It uses one worker, fixed estimators and seeds, with no tuning.

## Assertions and interpretation

The corrected control is `sort(interpolated32)`, with identical values to the time-ordered trace. Catalogue-number suffixes are removed without deleting chemical hydrate parentheses. Shared profile/formula components are disjoint in the strict profile-domain comparison. Score vectors and split groups allow recomputation of every new thermal contrast.

These are retrospective, post-inspection, target-source-supervised comparisons. Conditional bootstrap intervals exclude model-refit and source-sampling uncertainty. Reused temperature arrays are recorded metadata groups, not proven independent furnace batches. Reversal after refitting is an algorithmic reindexing check, not a physical time-direction test.

The original primary estimator artifacts were created under a different scikit-learn version. They are not loaded in this package. [scikit-learn's model persistence documentation](https://scikit-learn.org/stable/model_persistence.html) describes the same-environment requirement; this package separates new-analysis reproduction from legacy-estimator recovery.

## Saved primary-score evaluation

The separately supplied, authorized frozen test-score snapshot can be evaluated without loading estimator objects:

```powershell
python recompute_frozen_primary_metrics.py --predictions 'ABSOLUTE_PATH_TO_AUTHORIZED_FROZEN_CSV_GZ' --output 'ABSOLUTE_PATH_TO_METRICS_JSON'
```

The expected SHA256 and test-population assertions are embedded in that runner. The snapshot is not included in this archive; obtain it only under the applicable source/derived-data permissions. This check reproduces score-derived metrics, not the historical fitting environment or a blinded validation.
