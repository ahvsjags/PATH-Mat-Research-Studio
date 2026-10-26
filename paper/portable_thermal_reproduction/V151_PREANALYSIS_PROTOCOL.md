# Corrected thermal-profile comparison: fixed analysis plan

Date: 2026-10-01. This is an exploratory methodological audit of an already inspected public archive, not preregistration of an independent experiment.

## Question and repair

The earlier comparator used quantiles of the raw measured temperatures; the chronological vector used interpolation at 32 uniformly spaced normalized times. Those are different discretizations and need not have the same multiset. They therefore do not isolate chronology. The corrected control sorts the **actual 32 interpolated values**, without changing any value, per record.

Reversing all chronological columns in both training and testing and refitting is an invertible feature reindexing. It is a numerical sanity check, not a test of a physical arrow of time. No directional interpretation will be drawn from it.

## Locked analysis before corrected model fits

1. Preserve the previously defined 1,005 accepted active refinements, exact residual-precursor endpoint, formula normalization, source snapshot and two fixed model families.
2. Audit the discrepancy between legacy raw-sample quantiles and exact sorted interpolated temperatures for all 952 available profiles. Assert rowwise exact temperature-multiset identity for the corrected contrast.
3. Dataset A: all 1,005 records and the original shared feature base, including its 12 profile summaries. Compare exact-sorted32 versus chronological32. The shared base already contains chronology and this analysis measures only the incremental feature encoding.
4. Dataset B: the 952 complete-profile records and an order-blind shared base: composition, input chemistry, heating setpoint/time and profile total duration only. Exclude start/end temperatures, rates, quarter averages, peak time, and all other chronology summaries. Append either exact-sorted32 or chronological32. This is the clean arrangement contrast.
5. For each dataset run both (i) five-fold exact-target-formula-grouped cross-validation and (ii) leave-one-precursor-formula-out fitting, excluding training target formulas that appear in the held-precursor test fold. Average the two precursor-fold predictions of each record. Report both schemes; do not select the favorable one.
6. Fixed models: Ridge alpha=10, train-only median imputation and scaling; ExtraTrees 500 trees, minimum leaf size=8, max_features=0.8, seed=14829, one worker. No model/threshold tuning. MAE is primary; include a training-fold mean baseline, RMSE, and the error-change distribution for context.
7. Save all row-level prediction arrays, identifiers, group memberships, fold definitions and dependency versions locally; publish aggregate tables and the lawful source retrieval instructions without redistributing internal or unlicensed sources.
8. Eight chronology comparisons (two datasets × two split schemes × two models) form one explicitly exploratory family. Report 95% paired target-formula bootstrap intervals and Bonferroni 99.375% intervals (10,000 draws), conditional on fixed predictions. These intervals do not represent complete model-refit/source uncertainty. No hypothesis-testing p-values or confirmatory claims will be derived from them.
9. Audit outcome weight sums, negative/nonfinite weights, unknown phase formulas and normalization; show endpoint range and source quality eligibility. Verify train/test target and held-precursor isolation in every saved fold.

## Interpretation

An interval excluding zero in this selected sensitivity is not independent external confirmation. Any effect must be reported on its absolute [0,1] phase-fraction scale, as conditional supervised adaptation in this same source. The generic action-order evidence and zero-shot transfer conclusions remain unchanged. Report a null or weaker result without substituting a more favorable control.
