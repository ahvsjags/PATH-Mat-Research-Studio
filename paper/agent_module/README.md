# PATH-Mat integrated candidate-design agent module

This module is part of the PATH-Mat work, alongside the existing route-prediction benchmarks, structured-condition analyses and candidate-material screens. It packages previously implemented agent components behind one executable workflow. It does not replace B0–B3 or reinterpret their results.

## Components and interfaces

| Component | Input | Implemented output | Source interface |
|---|---|---|---|
| Designer | Cation sets, formal oxidation states and geometry windows | Charge-balanced A2BB′O6 CIFs, balanced oxide reactions and three thermal-schedule hypotheses | `material_generation/agent/ordered_double_perovskite_agent.py`; `PathMatRouteAgent.precursor_recipe` |
| Skeptic | Candidate-specific failed checks or near-threshold numerical results | A critique bound to the candidate and saved artifact | `SkepticAgent`; `RestartSkeptic` |
| Verifier | CIF, expected composition, hashes, checkpoint and force records | Confirmed, rejected or unresolved checks | `Verifier`; `RestartVerifier` |
| Repairer | A verified recoverable issue and its registered parent | New restored artifact, or a bounded continuation through an injected relaxer | `RepairAgent`; `ContinuationRepairer` |
| Controller | Verified issues and remaining budget | Next action, class-3 fallback or one LBFGS continuation | `closed_loop_controller.run_candidate`; `FailureAwareController` |
| Verified memory | Verified development events | Append-only experience records and retrieval, with evaluation writes blocked | `VerifiedMemory` |

These agents are deterministic software roles. Running this package does not require an external LLM. Memory records experience; it does not update predictor weights. The numerical policy adapter provides an explicit tool-dispatch interface; the original frozen confirmation scripts, not this new adapter, generated the S124 scientific result.

## Run the functional workflow

From `material_generation/`:

```powershell
python workflow.py --config config/integration_demo.yaml --output ../validation/new_design_run
python workflow.py --config config/integration_demo.yaml --output ../validation/new_recovery_run --inject-byte-fault
python -m pytest -q tests
```

`--inject-byte-fault` appends a labelled byte-level test fault to one active copy. The registered parent is preserved, the defect is checked, and a new restored CIF is produced. This option is an engineering test, not a scientific material-design experiment.

To score generated route hypotheses with an existing frozen PATH-Mat deployment, supply both paths explicitly:

```powershell
python workflow.py --config config/integration_demo.yaml --output ../validation/new_scored_run --route-model MODEL.joblib --route-vendor VENDOR_DIRECTORY
```

The model and vendor tree are external configured inputs and are not silently substituted. The recorded demonstration used the existing EC-PathWM model with SHA-256 `8c5645e4d59883c08284d815729d3c765720495ded7d3f001f54694772d02b09`. This deployment is separate from the B0–B3 primary benchmark. It returned scores for all six route hypotheses while preserving six abstention decisions.

Evaluation mode can read a frozen development-memory snapshot but must not change it:

```powershell
python workflow.py --config config/integration_demo.yaml --output ../validation/new_evaluation_run --split confirmatory --memory-seed DEVELOPMENT_MEMORY.jsonl --inject-byte-fault
```

Here `confirmatory` is an execution mode for testing read-only memory. The included two-candidate demonstration is not a new confirmatory materials dataset.

## Outputs

- `candidate_records.json`: proposed composition, parent/final CIF hashes, balanced reaction and final checks.
- `parents/`, `active/`, `repaired/`: original, checked and restored artifacts, without overwriting parents.
- `routes/route_hypotheses.json`: three schedule hypotheses per retained composition.
- `routes/pathmat_route_hypotheses.csv`: scores and abstentions when a model is configured.
- `traces/` and `workflow_trace.jsonl`: per-candidate and orchestration events.
- `verified_memory.jsonl`: verified development experience or an unchanged evaluation snapshot.
- `workflow_summary.json`: counts, configured model metadata and runtime versions.

## Recorded verification

The included suite passed 25 tests. The development and evaluation functional runs each generated two CIFs and six routes, restored one deliberately corrupted active copy, and passed final artifact checks for both candidates. The evaluation memory digest was unchanged. These checks verify module operation, not material stability or new search performance.

The existing 432-formula failure-aware local-search result is retained separately in Supplementary Table S124 and `../reproducibility/failure_aware_relaxation/`. Its 17 additional accepted basins are attributed to that frozen policy, not separately counted as a new gain from the skeptic or memory.

The legacy `validation/chgnet_oracle.py` helper identifies its pretrained checkpoint explicitly. For the published 432-formula result, use the frozen CHGNet 0.4.2 runner and serialized-output force checks in the separate confirmation package.
