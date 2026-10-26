# Exact primary-cohort reproduction

Run `python prepare_selector.py` to assemble and verify the source selector before the steps below.

Obtain the Lee raw export from the genuine source DOI https://doi.org/10.6084/m9.figshare.30423274 (CC BY 4.0), credit Lee et al. (2025), and use the checksum-matched JSON gzip listed in expected_checks.json. The selector is derived from PATH-Mat's exact conflict-filtered cohort; it is a scientific curation input, not an independently recovered conflict-filter algorithm. Only source indices, content hashes and identifiers/partition roles are distributed here, not complete private records.

1. In a separate working directory run `python reconstruct_primary.py --lee-raw AUTHORIZED_PUBLIC_RAW.gz --out reconstructed_projection.jsonl`.
2. Set PATHMAT_PRIMARY_JSONL to the reconstructed projection and PATHMAT_LEE_RAW_GZ to that authorized public raw file.
3. Install the listed environment dependencies and run `python run_count_semantics.py`.

The runner re-generates the frozen DOI/formula component graph, asserts the exact 70,636 and 55,082/5,901/3,773/5,880 counts and partition digest, matches native source records and executes all symmetric fixed fits. All branches and signs are retained. PATHMAT_REPRO_LOGISTIC_ONLY=1 is a bounded reproduction check of the seven logistic branches, not the full two-classifier study. No tree results are inferred from that bounded execution. The full runner executes both models.

The model-relevant projection was compared field-by-field with the exact internal cohort, including target, precursors, source DOI, labels and every stored action name/condition. Its digest is a different object from the private 442 MB file; equality of model inputs is proved without claiming byte identity of unneeded internal metadata. Historical frozen-estimator scores use scikit-learn 1.6.1, whereas new refits use 1.8.0; current re-execution does not claim the old environment was recreated. Legacy score recomputation and the new raw-score decomposition are separate.

The full original conflict decisions involved the mixed archive and are not independently reconstructed by selecting source rows. This limitation is explicit. The selector makes the exact submitted cohort reproducible from public source contents while preserving the prohibition on distributing the complete internal dataset.
