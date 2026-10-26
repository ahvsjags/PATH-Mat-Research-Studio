# Endpoint and duplicated-profile correction addendum

Fixed on 2026-10-01 after the three independent reviews were frozen and before corrected-endpoint model fitting. The incomplete previous calculation used the old endpoint and is explicitly superseded, not used as evidence.

## Phase identifiers

Parse the formula prefix before the source's underscore-separated crystallographic metadata. For prefixes ending in a **numeric catalogue identifier of at least three digits in parentheses**, remove only that terminal identifier; retain all chemical parentheses and hydrate terms. Preserve the original identifier in the mapping table. Validate all six previously unparsed phase-name forms against the ledger's matching CIF filenames and composition semantics. Include a test for FeC2O4(H2O)2(161344), LiOH(H2O)(9138), phosphate identifiers and ordinary bracketed formulae. Never modify source phase weights, choose another refinement or convert unknown phase weight to confirmed non-precursor. Any still-unparsed positive phase weight makes a conservative unmatched-weight diagnostic; report counts, tails and exclusions if a fully defined endpoint is unavailable.

Keep the same accepted active-case eligibility, normalization, model families, original folds and all corrected value-preserving profile comparisons. Recompute every continuous endpoint result used in the new manuscript. The old phase-fraction outputs and shuffled-profile result are superseded and will not be used to support a positive claim. Also recompute composition, input-chemistry and condition-summary baselines in formula-grouped validation.

## Repeated profiles

Report exact fingerprints of each source's full time/temperature arrays and their multiplicity. Shared arrays are not assumed to identify independent physical furnace batches. Audit test-to-train fingerprint overlap under each existing split.

Add a third strict complete-profile evaluation: five-fold grouping by connected components of **shared profile fingerprint OR shared reduced target formula**. Use the clean order-blind base plus exact-sorted32 versus chronological32; no shared fingerprint or target formula may cross folds. Use the same fixed two regressors. This asks about unseen recorded profiles, not necessarily new physical furnaces or laboratories. Bootstrap these connected profile/formula components rather than individual records. Report both successful and unsuccessful contrasts. For a pooled family of the eight prior corrected profile contrasts and these two strict contrasts, additionally report conditional Bonferroni 99.5% intervals (10,000 draws).

No old per-record random shuffle is treated as independent profile-domain evidence: it breaks consistent representation of duplicated curves. Global reversal remains only a column-reindexing sanity check. Neither control will support a physical time-direction claim.

## Source-category semantics

Use the source name `completely_reacted` without interpreting it as depletion of both precursors or target-phase formation. The ledger notes for PG_2543 classify a mixed phase using Si cation partitioning at a 0.98 threshold while allowing MgO residual fraction about 0.911. Report this as an explicit source-category/continuous-endpoint counterexample. Keep labels unmodified. Do not infer universal category rules from this single example without source-code verification.
