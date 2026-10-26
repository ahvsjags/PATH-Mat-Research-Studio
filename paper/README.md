# PATH-Mat research code and source data

Research materials for **Synthesis sequence as a context-dependent variable in materials design**.

## Contents

| Directory | Contents |
|---|---|
| `source_index_reproduction/` | Exact primary-cohort source selector, split assignments, checksum checks and count/semantics reproduction |
| `portable_thermal_reproduction/` | Thermal-profile controls and frozen primary-metric recomputation |
| `portable_harmonized_reproduction/` | Cross-source feature mapping and frozen evaluation |
| `analyses/` | Source-specific condition, order, PXRD, candidate-ranking and robustness analyses |
| `agent_module/` | Candidate generation, critique, independent verification, repair, controller and memory interfaces, with tests |
| `candidate_screening/` | COD structure matching, published-DFT joins and CHGNet screening code |
| `failure_aware_relaxation/` | Frozen fixed-budget relaxation protocol, execution and analysis scripts |
| `figures/` | Figure 7 plotting code and public aggregate inputs |
| `data/` | Candidate-screening summaries, six rendered structure CIFs and paired 432-composition search data |
| `aggregate_results/` and `aggregate_checks/` | Primary-control and cross-source summary outputs |

The web application in the repository root and the research analyses in `paper/` have separate entry points.

## Primary-cohort reproduction

Obtain the Lee source from [the original Figshare dataset](https://doi.org/10.6084/m9.figshare.30423274). Use the JSON gzip whose SHA-256 is recorded in `source_index_reproduction/expected_checks.json`. Credit Lee et al. and follow the source's CC BY 4.0 terms.

```sh
cd paper/source_index_reproduction
python prepare_selector.py
python reconstruct_primary.py --lee-raw /path/to/authorized_lee_raw.json.gz --out /path/to/reconstructed_projection.jsonl
pip install -r requirements.txt
# Set PATHMAT_PRIMARY_JSONL and PATHMAT_LEE_RAW_GZ to those two files.
python run_count_semantics.py
```

The selector reconstructs the 70,636-record model-input projection and original partitions from public source rows. The complete internal trajectory dataset is not distributed. The historical conflict-filter decisions are supplied through the explicit selector, rather than recomputed from an unrelated source export.

## Candidate-design workflow

```sh
cd paper/agent_module
pip install -r requirements-core.txt
cd material_generation
python workflow.py --config config/integration_demo.yaml --output ../new_design_run
python -m pytest -q tests
```

Optional EC-PathWM scoring takes the separately configured model and vendor paths documented in `agent_module/README.md`. Predictor files in the repository root belong to the existing demonstration deployment; they are not the primary B0–B3 estimators.

## Computational source data

`data/figure7/Figure7_summary.json` reports candidate-screening counts and aggregate search statistics. Individual candidate energy, structure-hash and pass/fail records are not distributed. `data/figure7/structure_cifs/` contains the six structures rendered in Figure 7. `data/fixed_budget/S124_formula_level_source_data.csv` contains the complete paired 432-formula basin counts. The frozen protocol and independent audit are distributed as gzip-compressed JSON where needed.

```sh
cd paper
python verify_public_data.py
python figures/make_public_summary.py
```

The full Figure 7 script (`figures/make_figure7.py`) is supplied as code; its candidate-level panel requires the separately held, non-public energy tables. The public-summary figure uses only the released aggregates and paired 432-formula data. Install matplotlib to run that summary; the numerical search additionally requires the versions of CHGNet, ASE and PyTorch stated in the protocol and manuscript. Separate source-analysis folders retain their original study-specific inputs and path settings. Obtain external datasets from the sources in `DATA_SOURCES.md` and configure those paths before running them.

## Data and rights

`FILE_MANIFEST.json` records file digests and relative source lineage. Public source identifiers, selection indices and authored derived results are included; complete internal records, publisher article files, microscopy images and secret credentials are excluded. Third-party datasets and software retain their original licenses. Copyright in the original research code remains with its authors; this release does not assign a new blanket license to third-party material.
