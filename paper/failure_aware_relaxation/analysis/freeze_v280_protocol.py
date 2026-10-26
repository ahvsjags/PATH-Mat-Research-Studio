from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]
HOLD = PROJECT / "runs" / "holdout" / "v280_failure_repair_fresh_b4b4_v1"
GEOMETRY = HOLD / "geometry"
CONFIG = PROJECT / "material_generation" / "config" / "v280_repair_fresh_b4b4_v1.yaml"
SCAN = PROJECT / "source_data" / "prior_identity_scan_v280.json"
PREFLIGHT = GEOMETRY / "preflight.json"
PREFLIGHT_MANIFEST = GEOMETRY / "preflight_manifest.json"
DEV_PROBE = Path("/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V279/source_data/V268/development/v268_pair_policy_development/parents/Ba2MnMoO6_parent.cif")
DEV_SMOKE = Path("/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V276/runs/development/v276_runtime_smoke/smoke.json")
DEV_V0_SMOKE = Path("/ibex/user/xup0a/PATH-Mat_VCR_Closed_Loop_V276/runs/development/v276_v0_development_smoke.json")
PROTOCOL = PROJECT / "protocol" / "v280_failure_repair_confirmation_v1.json"
LOCK = PROJECT / "protocol" / "v280_failure_repair_confirmation_v1.lock.json"

CODE_FILES = [
    "analysis/scan_v280_prior_identity.py",
    "material_generation/analysis/prepare_v280_geometry.py",
    "material_generation/analysis/freeze_v280_protocol.py",
    "material_generation/analysis/run_v280_confirmation.py",
    "material_generation/analysis/analyze_v280_confirmation.py",
    "material_generation/analysis/audit_v280_confirmation.py",
    "material_generation/analysis/v280_locked_confirmation.sbatch",
    "material_generation/analysis/v280_analyze_audit.sbatch",
    "material_generation/agent/ordered_double_perovskite_agent.py",
    "material_generation/agent/pathmat_design_agent.py",
    "material_generation/agent/ordering_catalog.py",
]
DEV_FILES = {
    "V268_development_verified": "source_data/V268_development_verified.json",
    "V268_nested_portfolio_cv": "runs/development/v280_v268_nested_portfolio_cv.json",
    "V280_failure_repair_development": "runs/development/v280_failure_repair_policy.json",
    "V269_restart_development_records": "source_data/v269_development_restart_records.jsonl",
    "V269_restart_development_analysis": "source_data/v269_development_restart_analysis.json",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if PROTOCOL.exists() or LOCK.exists():
        raise FileExistsError("protocol already frozen; refusing to replace it")
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    manifest = json.loads(PREFLIGHT_MANIFEST.read_text(encoding="utf-8"))
    runtime_smoke = json.loads(DEV_SMOKE.read_text(encoding="utf-8"))
    v0_smoke = json.loads(DEV_V0_SMOKE.read_text(encoding="utf-8"))
    if (scan.get("status") != "V280_PRIOR_IDENTITY_SCAN_COMPLETE"
            or scan.get("identity_only") is not True
            or scan.get("scores_read") is not False
            or scan.get("outcomes_read") is not False
            or scan.get("required_versions_missing")
            or scan.get("scan_errors")):
        raise RuntimeError("identity-only history scan failed its completeness guard")
    if scan.get("versions_requested") != list(range(268, 280)):
        raise RuntimeError("identity scan does not cover every required version through V279")
    if preflight.get("status") != "V280_GEOMETRY_ONLY_UNSCORED_PREFLIGHT" or preflight.get("oracle_scoring_performed") is not False:
        raise RuntimeError("geometry preflight is not explicitly unscored")
    if preflight.get("selection_outcomes_used") is not False:
        raise RuntimeError("candidate grid selection indicates outcome use")
    if (len(preflight.get("records", [])) != 432 or preflight.get("pair_cluster_count") != 72
            or preflight.get("formula_count") != 432 or preflight.get("a_site_count") != 6):
        raise RuntimeError("frozen preflight must contain exactly 432 formulas / 72 pair clusters")
    if runtime_smoke.get("status") != "CPU_RUNTIME_SMOKE_COMPLETED" or runtime_smoke.get("source_dataset") != "V268 development only":
        raise RuntimeError("CHGNet runtime smoke test must be explicitly development-only")
    if runtime_smoke.get("checkpoint_sha256") != "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1" or runtime_smoke.get("package_version") != "0.4.2" or runtime_smoke.get("device") != "cpu":
        raise RuntimeError("runtime smoke does not match frozen CHGNet identity")
    if v0_smoke.get("status") != "V0_DEVELOPMENT_SMOKE_PASS" or v0_smoke.get("overall") != "pass" or v0_smoke.get("source_dataset") != "V268 development only":
        raise RuntimeError("V0 development gate smoke did not pass")
    if sha(DEV_PROBE) != runtime_smoke.get("input_sha256") or sha(DEV_PROBE) != v0_smoke.get("input_sha256", sha(DEV_PROBE)):
        raise RuntimeError("development-only smoke input does not match probe bytes")
    if manifest.get("preflight_sha256") != sha(PREFLIGHT) or manifest.get("status") != preflight["status"]:
        raise RuntimeError("preflight manifest does not bind the geometry grid")
    old_pairs = set(scan["prior_bsite_pair_identities"])
    old_formulas = set(scan["prior_reduced_formula_identities"])
    formula_ids = set()
    pair_ids = set()
    candidate_records = preflight["records"]
    for record in candidate_records:
        if record["formula"] in old_formulas:
            raise RuntimeError(f"exact formula collision in holdout grid: {record['formula']}")
        pair = "-".join(sorted(record["b_site_pair"].split("-")))
        if pair in old_pairs:
            raise RuntimeError(f"previously scored pair identity in holdout grid: {pair}")
        formula_ids.add(record["formula"])
        pair_ids.add(record["pair_cluster"])
        class_records = record.get("class_records", [])
        if len(class_records) != 6 or {int(x["ordering_class"]) for x in class_records} != set(range(6)):
            raise RuntimeError(f"missing six-class catalog: {record['formula']}")
        if int(record["formal_charge_sum"]) != 0 or not record["preflight_checks"].get("neutral_parent"):
            raise RuntimeError(f"neutrality check failed: {record['formula']}")
        parent = (GEOMETRY / record["parent_cif"]).resolve(strict=True)
        parent.relative_to(GEOMETRY.resolve(strict=True))
        if sha(parent) != record["parent_sha256"]:
            raise RuntimeError(f"parent CIF digest mismatch: {record['formula']}")
        for class_record in class_records:
            cif = (GEOMETRY / class_record["class_cif"]).resolve(strict=True)
            cif.relative_to(GEOMETRY.resolve(strict=True))
            if sha(cif) != class_record["class_cif_sha256"]:
                raise RuntimeError(f"catalog CIF digest mismatch: {record['formula']}/class{class_record['ordering_class']}")
    if len(formula_ids) != 432 or len(pair_ids) != 72:
        raise RuntimeError("frozen formula/pair uniqueness mismatch")

    for rel in CODE_FILES:
        if not (PROJECT / rel).is_file():
            raise FileNotFoundError(PROJECT / rel)
    dev_hashes = {}
    for name, rel in DEV_FILES.items():
        path = PROJECT / rel
        if not path.is_file():
            raise FileNotFoundError(f"missing development lineage: {path}")
        dev_hashes[name] = {"path": rel, "sha256": sha(path)}

    protocol = {
        "schema_version": 1,
        "protocol_id": "PATH-Mat-VCR-V280-failure-triggered-force-continuation-confirmation",
        "status": "LOCKED_BEFORE_HOLDOUT_SCORING",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Test whether a hash-verified, V0-passing force-only failure can be recovered by an LBFGS continuation within the same four-call candidate budget, increasing unique accepted relaxed basins over static ordering classes 0/1/2/3.",
        "development_boundary": {
            "allowed_strategy_sources": ["V268 development only", "V269 paired optimizer-restart development only"],
            "V269_holdout_used": False,
            "V279_confirmation_outcomes_used_for_strategy": False,
            "V280_holdout_used_for_strategy_or_selection": False,
            "selection_rule": "The repair trigger, lowest-index tie-break, fallback class 3, four-call budget, endpoints, bootstrap unit/seeds, and dual-CI decision were fixed before any V280 CHGNet calls.",
            "evidence_sha256": dev_hashes,
            "runtime_probe_sha256": {
                "input_sha256": sha(DEV_PROBE),
                "runtime_smoke_sha256": sha(DEV_SMOKE),
                "v0_smoke_sha256": sha(DEV_V0_SMOKE),
            },
        },
        "development_runtime_probe": {
            "probe_path": str(DEV_PROBE),
            "input_sha256": sha(DEV_PROBE),
            "smoke_path": str(DEV_SMOKE),
            "smoke_result_sha256": sha(DEV_SMOKE),
            "smoke_status": runtime_smoke["status"],
            "v0_smoke_path": str(DEV_V0_SMOKE),
            "v0_smoke_result_sha256": sha(DEV_V0_SMOKE),
            "v0_smoke_status": v0_smoke["status"],
        },
        "holdout": {
            "path": "runs/holdout/v280_failure_repair_fresh_b4b4_v1",
            "chemistry": "A2+ B4+ B4+ O6 rock-salt ordered double perovskites; six A-site cations; exact Shannon-radius geometry windows frozen in the YAML grid.",
            "formula_count": 432,
            "pair_cluster_count": 72,
            "a_site_count": 6,
            "a_sites": ["Ba", "Ca", "Eu", "Pb", "Sm", "Sr"],
            "identity_scan_versions": scan["versions_requested"],
            "identity_scan_sha256": sha(SCAN),
            "identity_only_scan_complete": True,
            "scores_read": False,
            "outcomes_read": False,
            "config_sha256": sha(CONFIG),
            "preflight_sha256": sha(PREFLIGHT),
            "preflight_manifest_sha256": sha(PREFLIGHT_MANIFEST),
            "geometry_only_scoring_performed": False,
            "exact_formula_collisions": 0,
            "prior_pair_identity_collisions": 0,
            "pair_selection_seed": preflight["pair_selection_seed"],
            "selected_pair_clusters": preflight["selected_pair_clusters"],
        },
        "arms": {
            "baseline_static_0123": {
                "classes": [0, 1, 2, 3],
                "optimizer": "FIRE",
                "assignments_per_formula": 4,
            },
            "candidate_failure_triggered_lbfgs": {
                "base_classes": [0, 1, 2],
                "fourth_slot_policy": "If at least one of classes 0/1/2 has a hash-verified saved terminal CIF that independently passes V0, independently recomputed serialized CHGNet max force is greater than 0.050001 eV/angstrom, and the recorded runtime call succeeded, use one LBFGS continuation from the lowest-index qualifying terminal CIF. Otherwise run class 3 with FIRE.",
                "optimizer_base": "FIRE",
                "repair_optimizer": "LBFGS",
                "assignments_per_formula": 4,
            },
            "call_budget": "Four relaxation assignments per formula per arm. Classes 0/1/2 are shared identical FIRE results across arms. Candidate fourth slot is one FIRE class-3 run without trigger or one LBFGS continuation with trigger; baseline always runs class 3.",
        },
        "evaluation": {
            "prototype": "40-atom 2x2x2 rocksalt-ordered A2BBpO6 perovskite supercell",
            "device": "cpu",
            "chgnet_package_version": "0.4.2",
            "chgnet_checkpoint_sha256": "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1",
            "optimizer_steps": 500,
            "cell_relaxation": True,
            "fmax_eV_A": 0.05,
            "serialized_force_threshold_eV_A": 0.05,
            "force_tolerance_eV_A": 1e-6,
            "displacement_sigma_angstrom": 0.03,
            "class_seed_map": {"0": 101, "1": 211, "2": 307, "3": 401},
            "v0_minimum_interatomic_distance_angstrom": 0.75,
            "acceptance": "successful call AND V0 pass AND serialized-output CHGNet maximum force <= 0.050001 eV/angstrom",
            "unique_basin_matcher": {"ltol": 0.2, "stol": 0.3, "angle_tol_degrees": 5.0, "primitive_cell": True, "scale": True, "attempt_supercell": True, "allow_subset": False},
            "assignments_per_formula_per_arm": 4,
            "primary_endpoint": "candidate unique accepted basins minus baseline unique accepted basins per formula",
            "primary_uncertainty": {
                "pair_cluster_bootstrap_draws": 50000,
                "pair_cluster_bootstrap_seed": 28032,
                "two_way_a_site_x_pair_cluster_bootstrap_draws": 50000,
                "two_way_a_site_x_pair_cluster_bootstrap_seed": 28033,
                "formula_iid_bootstrap_secondary_draws": 50000,
                "formula_iid_bootstrap_secondary_seed": 28031,
                "confidence_level": 0.95,
            },
            "success_criterion": "Both the pair-cluster bootstrap 95% CI and two-way A-site x B/B-prime-pair bootstrap 95% CI for mean paired yield difference have lower bounds strictly greater than zero.",
            "no_rescoring_rule": "V280 is one locked confirmation. If the dual-CI criterion fails, its outcomes may not be used to tune another strategy or to claim confirmation; any new strategy needs development-only evidence and a new unscored identity-fresh set.",
        },
        "claim_scope": "A positive result supports a cost-matched failure-aware CHGNet local-relaxation search policy for this frozen computational double-perovskite grid. It does not by itself establish DFT stability, synthesis feasibility, experimental yield, or universal performance.",
        "candidate_records": candidate_records,
        "integrity": {"code_sha256": {rel: sha(PROJECT / rel) for rel in CODE_FILES}},
    }
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    lock = {
        "status": "LOCKED_BEFORE_HOLDOUT_SCORING",
        "protocol_path": PROTOCOL.relative_to(PROJECT).as_posix(),
        "protocol_sha256": sha(PROTOCOL),
        "freeze_script_sha256": sha(HERE),
        "code_sha256": protocol["integrity"]["code_sha256"],
        "identity_scan_sha256": sha(SCAN),
        "config_sha256": sha(CONFIG),
        "preflight_sha256": sha(PREFLIGHT),
        "preflight_manifest_sha256": sha(PREFLIGHT_MANIFEST),
        "formula_count": 432,
        "pair_cluster_count": 72,
        "holdout_oracle_scoring_performed_at_lock": False,
    }
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": protocol["status"], "protocol_sha256": lock["protocol_sha256"],
        "lock_sha256": sha(LOCK), "formula_count": 432,
        "pair_cluster_count": 72, "prior_pair_identity_collisions": 0,
        "exact_formula_collisions": 0,
        "holdout_scoring_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
