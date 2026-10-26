# Source harmonization and order recoverability

Date: 2026-10-01. This protocol precedes V152 fitting/scoring, but follows extensive inspection of the primary and external archives. All new results are therefore exploratory, not a new preregistered or untouched external validation.

## Order
Audit the exact 70,636-row primary cohort against its canonical-hash-matched Lee raw records. Report direct non-Starting action counts, reference-action exclusion, repeated/local action IDs, list/ID discordance and availability of a global sentence/time index. Do not sort sentence-local IDs into a fictitious full-route chronology. Existing B2–B1 and blocked thermal-order results are retained irrespective of this audit's outcome.

## Zero-shot transfer
Use the existing Lee development/calibration/selection partitions only. Fit three fixed balanced logistic models (liblinear, C=1, max_iter=600, seed=31415): richer composition; composition plus a common unordered action taxonomy; the latter plus strictly unit-converted maximum/mean temperature and duration and field-presence flags. Starting and reference acts are excluded. Heating maps to heat; Mixing and grinding to mix; Shaping, pressing and deposition to shape; remaining classes to other. Add log(1+direct action count) and class proportions. Numeric medians for imputation are fitted on development data only; temperature is divided by 1500 and duration by 24.

Isotonic calibration uses Lee calibration data; the threshold uses Lee selection data. Save model parameters, dictionaries, imputation, calibration, threshold and hashes before reading external labels. No external fitting, recalibration, threshold adjustment, model selection, feature selection or early stopping.

External evaluation uses the previously inspected sol-gel archive, restricted to binary source labels with DOI/formula, excluding all target formulas and DOIs in the exact primary Lee cohort. Formula normalization is whitespace removal and lowercase, not chemistry-equivalence detection. Report this limitation. No record is dropped for feature absence or low confidence. Group external uncertainty by transitive DOI/formula components. The primary endpoint remains source-reported impurity; the external endpoint remains source-classified impure versus pure. These are related reporting endpoints, not adjudicated phase-purity outcomes.

Primary ranking is assessed on raw scores, avoiding isotonic tie artifacts; calibrated metrics are reported separately with prediction distributions, coverage, threshold outcomes and Brier/ECE. Three AP contrasts per cohort (actions–composition, full–composition, full–actions), six total, receive paired component bootstrap 95% and Bonferroni-six 99.1667% intervals, 3,000 draws, seed=1522026. Success in exploratory ranking requires external action/full superiority to composition and AP above prevalence; this alone does not establish useful calibration, physical synthesis generalization or confirmatory replication.

No additional model grid or outcome-dependent subgroup is introduced if these results fail. The new model will not replace frozen primary metrics. Figures are unchanged.

## Accession
Prepare a minimal code/aggregate-results deposit, excluding internal row-level data and source archives with unresolved redistribution rights. A public record requires actual creator metadata, confirmed licenses, authenticated repository access and a verified public DOI. A local ZIP, reserved DOI or an existing source DOI is not the project's public accession.
