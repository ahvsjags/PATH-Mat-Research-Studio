# Fixed computational responses to the frozen independent reviews

This protocol follows all three frozen V152 reviews and precedes the added fits. It is exploratory after archive and test inspection. No figures, historical fits, labels or partition roles are changed. Results of every listed comparison are retained.

## Count semantics and model symmetry

The exact 70,636-record cohort and original 55,082/5,901/3,773/5,880 partition are required. Canonical source hashes must match. Four mutually exclusive source entry categories are audited: Starting (regardless of reference flag); non-Starting reference; direct non-Starting/non-reference; total = their sum. Count distributions, correlations and reference-bearing record counts are reported.

Seven representations share richer formula-atomic-fraction composition: composition alone; total count; direct count; excluded (Starting + non-Starting reference) count; direct count + common class proportions; total count + source word proxy + condition presence; and the preceding word/presence representation + native action proportions + numeric conditions. Scalar counts and word proxy are untransformed in every branch, preserving the earlier richer count comparison. The last two branches share their reporting-proxy inputs.

Two fixed classifiers are symmetric across all seven representations: balanced liblinear logistic regression C=1, max_iter=600, seed=31415; and the previously used ExtraTrees classifier, 200 trees, min_samples_leaf=10, max_features=0.8, balanced, seed=31415. No search or test-based choice occurs. Development fits vocabulary, coefficients and condition-imputation medians. Isotonic calibration uses only calibration records. No thresholds are needed.

Six contrasts per classifier and score type are total−composition, direct−composition, excluded−composition, direct−total, direct-taxonomy−direct, and native-conditions−word-presence. Raw and calibrated AP/AUROC/distinct-score counts are listed separately. The 24 AP contrasts receive 10,000 paired component draws, seed 1532031, nominal 95% and Bonferroni-24 intervals (99.7917%). These conditional intervals do not cover all accumulated adaptive analyses or refitting/model-selection uncertainty. Failure to reject zero is not equivalence.

Original B0/B1 raw scores are extracted from saved estimators only if available and identity-checked. Loading historical objects in the current environment is explicitly not a same-version refit. Otherwise the frozen calibrated estimand remains unchanged and the new raw-score controls carry representation-level interpretation.

## PXRD-linked exact population dependence

Only the exact 956 eligible IDs in the existing revealed score file are used; labels and saved calibrated scores are unchanged. Link rows transitively through shared target string or identical full stored time/temperature arrays. The profile fingerprint is of the full recorded arrays, not the interpolated 32-point summary. Missing fields create no inferred links. Record furnace/batch field availability without claiming that shared furnace/profile proves a common batch. Recompute B2−B0 and B2−B1 with 10,000 component draws, seed 1532032, 95% and Bonferroni-two 97.5% intervals. Original sample-ID intervals remain descriptive provenance.

## Source human/LLM identity diagnostic

Check source line-order pairs across the original 279-row prefix, reporting DOI, normalized target formula and available record-key matches; retain mismatches and all extra rows in an internal index. No rematching based on labels or favorable agreement. If shared identity is absent, the agreement result is demoted to a source-convention diagnostic. This is not primary blinded human adjudication.

## Reproducibility

Build a no-old-workspace input chain only from file-level authorized inputs or an explicit source index; do not redistribute private input. Compute source-index / semantic cohort hashes and exact partition checks. A source-linked row selection manifest is not itself independent reconstruction of conflict decisions; disclose the distinction. No public project accession is claimed until a repository actually issues a usable identifier.
