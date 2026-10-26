from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import yaml
from pymatgen.core import Lattice, Structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import run_generation  # noqa: E402


MODULE = Path(__file__).resolve().parents[1]
ANALYSIS = MODULE / "analysis"
JOBS = MODULE / "jobs"


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, str(ANALYSIS / name), *map(str, args)],
        cwd=MODULE,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{name} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def tree_digest(path: Path) -> str:
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(child.relative_to(path).as_posix().encode("utf-8"))
        h.update(bytes.fromhex(hashlib.sha256(child.read_bytes()).hexdigest()))
    return h.hexdigest()


class PipelineSmokeTests(unittest.TestCase):
    def test_hf_job_requires_verified_private_storage_before_compute(self) -> None:
        source = (JOBS / "hf_mattergen_smoke_job.py").read_text(encoding="utf-8")
        create_repo = source.index('hub_api.create_repo(repo_id=repo_id, repo_type="dataset", private=True')
        verify_private = source.index('if not bool(getattr(repo_info, "private", False))', create_repo)
        cuda_probe = source.index('env = run(["python", "-c", "import torch;')
        install = source.index('run(["python", "-m", "pip", "install", "-e", "."')
        self.assertLess(create_repo, verify_private)
        self.assertLess(verify_private, cuda_probe)
        self.assertLess(verify_private, install)
        self.assertNotIn("private=False", source)
        self.assertIn("private artifact persistence is mandatory", source)

    def test_generation_dry_run_has_fire_compatible_condition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dry_run"
            config = MODULE / "config" / "design.yaml"
            run_script("run_generation.py", "--config", str(config), "--out-root", str(out), "--dry-run")
            manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
            command = manifest["runs"][0]["command"]
            self.assertIn("--properties_to_condition_on={dft_band_gap:1.5}", command)
            self.assertIn("--record_trajectories=False", command)
            self.assertNotIn("--properties_to_condition_on={dft_band_gap: 1.5}", command)

    def test_cif_archive_extracts_only_safe_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "good.zip"
            with zipfile.ZipFile(good, "w") as zf:
                zf.writestr("structures/a.cif", "data_test")
                zf.writestr("noise.txt", "ignored")
            extracted = run_generation.extract_cif_archive(good, root / "out")
            self.assertEqual([p.name for p in extracted], ["a.cif"])
            bad = root / "bad.zip"
            with zipfile.ZipFile(bad, "w") as zf:
                zf.writestr("../escape.cif", "data_test")
            with self.assertRaises(ValueError):
                run_generation.extract_cif_archive(bad, root / "bad_out")

    def test_reference_freeze_accepts_cif_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cif = Structure(Lattice.cubic(4.2), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(fmt="cif")
            archive = root / "reference.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("dataset/mgo.cif", cif)
            manifest = root / "reference_manifest.csv"
            run_script(
                "freeze_reference.py", "--dataset-id", "fixture-zip", "--version", "v1",
                "--access-date", "2026-10-17", "--license", "test-fixture-only",
                "--snapshot", str(archive), "--manifest", str(manifest),
            )
            with manifest.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["structure_count"], "1")

    def test_screen_uses_frozen_reference_and_path_scoped_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = root / "references"
            refs.mkdir()
            Structure(Lattice.cubic(4.2), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]).to(filename=refs / "mgo.cif")
            manifest = root / "reference_manifest.csv"
            run_script(
                "freeze_reference.py", "--dataset-id", "fixture", "--version", "v1",
                "--access-date", "2026-10-17", "--license", "test-fixture-only",
                "--snapshot", str(refs), "--manifest", str(manifest),
            )

            run_dir = root / "runs" / "pathmat_mattergen_oxide_bandgap_1p5eV_v1" / "seed_20261017"
            (run_dir / "seed_a").mkdir(parents=True)
            (run_dir / "seed_b").mkdir(parents=True)
            # Same CIF filename in different subfolders must not collide in the screen join.
            structures = [
                Structure(Lattice.cubic(3.8), ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
                Structure(Lattice.cubic(4.1), ["Na", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
            ]
            screen_rows = []
            for subdir, structure in zip(["seed_a", "seed_b"], structures):
                cif = run_dir / subdir / "0.cif"
                structure.to(filename=cif)
                rel = cif.relative_to(run_dir).as_posix()
                cif_hash = hashlib.sha256(cif.read_bytes()).hexdigest()
                namespace = f"{run_dir.parent.name}/{run_dir.name}"
                candidate_id = hashlib.sha256(f"{namespace}\0{rel}\0{cif_hash}".encode()).hexdigest()
                common = {
                    "candidate_id": candidate_id,
                    "cif_sha256": cif_hash,
                    "calculator": "fixture-calculator",
                    "calculator_version": "0.1",
                    "method": "test-only",
                    "band_gap_eV": "1.5",
                    "energy_above_hull_meV_per_atom": "10",
                }
                screen_rows.append({**common, "screen_level": "mlff"})
                bundle = root / f"bundle_{subdir}"
                bundle.mkdir()
                (bundle / "result.txt").write_text("independent fixture evidence", encoding="utf-8")
                phonon = root / f"phonons_{subdir}.yaml"
                phonon.write_text("fixture phonon frequencies; stable=true\n", encoding="utf-8")
                screen_rows.append({
                    **common,
                    "screen_level": "dft",
                    "method": "PBE+U fixture",
                    "relaxed_cif_path": str(cif),
                    "relaxed_cif_sha256": cif_hash,
                    "hull_reference": "fixture-hull",
                    "hull_reference_version": "v1",
                    "calculation_id": f"calc-{subdir}",
                    "input_path": str(cif),
                    "input_digest": cif_hash,
                    "calculation_bundle_path": str(bundle),
                    "calculation_bundle_sha256": tree_digest(bundle),
                    "phonon_output_path": str(phonon),
                    "phonon_output_sha256": hashlib.sha256(phonon.read_bytes()).hexdigest(),
                    "phonon_calculator": "fixture phonon code",
                    "phonon_calculator_version": "0.1",
                    "phonon_stable": "true",
                })
            screen_csv = root / "screen.csv"
            with screen_csv.open("w", newline="", encoding="utf-8") as f:
                fieldnames = list(dict.fromkeys(key for row in screen_rows for key in row))
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(screen_rows)

            run_script(
                "screen_candidates.py", "--run-dir", str(run_dir),
                "--config", str(MODULE / "config" / "design.yaml"),
                "--reference-manifest", str(manifest), "--screen-csv", str(screen_csv),
            )
            with (run_dir / "candidate_audit.csv").open(newline="", encoding="utf-8") as f:
                audited = list(csv.DictReader(f))
            self.assertEqual(len(audited), 2)
            self.assertEqual(len({row["candidate_id"] for row in audited}), 2)
            self.assertEqual(len({row["run_namespace"] for row in audited}), 1)
            self.assertTrue(all(row["computational_candidate"] == "True" for row in audited))

            # A DFT record must identify the exact original candidate file, not
            # merely a different structure with the same reduced formula.
            bad_rows = [dict(row) for row in screen_rows]
            bad_rows[1]["input_digest"] = "0" * 64
            bad_screen_csv = root / "bad_screen.csv"
            with bad_screen_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in bad_rows for k in row)))
                writer.writeheader()
                writer.writerows(bad_rows)
            failed = subprocess.run(
                [sys.executable, str(ANALYSIS / "screen_candidates.py"), "--run-dir", str(run_dir),
                 "--config", str(MODULE / "config" / "design.yaml"), "--reference-manifest", str(manifest),
                 "--screen-csv", str(bad_screen_csv)],
                cwd=MODULE, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("artifact digest mismatch", failed.stderr)

            route_scores = root / "route_scores.csv"
            with route_scores.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["target_formula", "route_id", "balance_status", "route_source_id", "b2_source_label_score", "route_actions", "precursor_set"])
                writer.writeheader()
                writer.writerow({"target_formula": "LiO", "route_id": "r1", "balance_status": "balanced", "route_source_id": "doi:test", "b2_source_label_score": "0.2", "route_actions": "mix;heat", "precursor_set": "Li2O;O2"})
            model_manifest = root / "model_manifest.json"
            model_manifest.write_text(json.dumps({
                "model_id": "PATH-Mat-B2", "model_version": "v-test",
                "training_data_id": "fixture", "score_semantics": "B2 source-label score",
            }), encoding="utf-8")
            joined = root / "routes_joined.csv"
            run_script("join_routes.py", "--candidate-audit", str(run_dir / "candidate_audit.csv"),
                       "--route-scores", str(route_scores), "--out", str(joined),
                       "--model-manifest", str(model_manifest))
            with joined.open(newline="", encoding="utf-8") as f:
                joined_rows = list(csv.DictReader(f))
            self.assertEqual(len(joined_rows), 1)
            self.assertIn("not a success probability", joined_rows[0]["score_interpretation"])

    def test_paired_seed_intervals_require_ten_matching_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for method, hits in [("guided", 2), ("guidance_zero", 1)]:
                config_hash = hashlib.sha256(method.encode()).hexdigest()
                generation_manifest = {
                    "config_sha256": config_hash,
                    "checkpoint_sha256": "a" * 64,
                    "model_config_sha256": "c" * 64,
                    "target": {"chemistry_class": "oxide", "band_gap_window_eV": [1.0, 2.0]},
                    "generator": {
                        "implementation": "mattergen", "pretrained_name": "dft_band_gap",
                        "target_property": "dft_band_gap", "target_value_eV": 1.5,
                        "guidance_factor": 2.0 if method == "guided" else 0.0,
                        "conditioning_enabled": True, "batch_size": 1, "num_batches": 10,
                        "seeds": list(range(10)), "max_candidates_per_seed": 10,
                    },
                    "runs": [{"seed": seed, "status": "completed"} for seed in range(10)],
                }
                (root / method).mkdir(parents=True)
                (root / method / "run_manifest.json").write_text(json.dumps(generation_manifest), encoding="utf-8")
                for seed in range(10):
                    folder = root / method / f"seed_{seed:02d}"
                    folder.mkdir(parents=True)
                    with (folder / "candidate_audit.csv").open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["computational_candidate"])
                        writer.writeheader()
                        writer.writerows({"computational_candidate": str(i < hits)} for i in range(10))
                    audit_summary = {
                        "config_sha256": config_hash,
                        "reference_manifest_sha256": "b" * 64,
                        "acceptance": {"band_gap_eV": [1.0, 2.0], "e_hull_max": 25},
                        "reference_snapshots": [],
                        "counts": {
                            "generated_cifs": 10,
                            "matched_mlff_rows": 10,
                            "shortlist_selected": 10,
                            "matched_dft_rows": 10,
                            "awaiting_independent_DFT_and_phonons": 0,
                        },
                    }
                    (folder / "candidate_audit_summary.json").write_text(json.dumps(audit_summary), encoding="utf-8")
            output = root / "comparison.json"
            run_script("compare_methods.py", "--runs-root", str(root), "--out", str(output), "--bootstrap-replicates", "200")
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["methods"]["guided"]["n_seeds"], 10)
            self.assertIsNotNone(report["methods"]["guided"]["seed_bootstrap_95_ci"])
            diff_key = next(key for key in report["pairwise_differences"] if {"guided", "guidance_zero"} == set(key.split(" - ")))
            diff = report["pairwise_differences"][diff_key]
            self.assertEqual(diff["inferential_status"], "paired_seed_interval_available")
            self.assertEqual(diff["seed_ids"], [f"seed_{i:02d}" for i in range(10)])


if __name__ == "__main__":
    unittest.main()
