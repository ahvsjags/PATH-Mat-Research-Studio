from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Structure

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from agent.pathmat_design_agent import PathMatDesignAgent, PathMatRouteAgent  # noqa: E402


class LocalDesignAgentTests(unittest.TestCase):
    def test_local_agent_generates_auditable_unrelaxed_cifs(self) -> None:
        config = MODULE / "config" / "design_agent_perovskite.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            out = PathMatDesignAgent().run(config, Path(tmp) / "run")
            summary = json.loads((out / "design_summary.json").read_text(encoding="utf-8"))
            with (out / "candidate_ledger.csv").open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            generated = [r for r in rows if r["cif_path"]]
            self.assertGreater(summary["counts"]["cif_written"], 0)
            self.assertEqual(summary["novelty_status"], "not_evaluated_no_reference_manifest")
            self.assertEqual(summary["property_prediction_status"], "not_run_no_property_model_configured")
            self.assertEqual(summary["experimental_status"], "not_performed")
            route_integration = summary["pathmat_route_integration"]
            self.assertEqual(route_integration["status"], "direct_model_scored_hypothetical_balanced_routes")
            self.assertEqual(route_integration["route_hypothesis_count"], 3 * summary["counts"]["cif_written"])
            self.assertEqual(route_integration["scored_route_model_abstain_count"], route_integration["route_hypothesis_count"])
            self.assertEqual(route_integration["observed_route_feature_support"],
                             {"temperature": True, "duration": True, "atmosphere_category": False})
            with (out / "pathmat_route_hypotheses.csv").open(newline="", encoding="utf-8-sig") as f:
                routes = list(csv.DictReader(f))
            self.assertEqual(len(routes), route_integration["route_hypothesis_count"])
            self.assertTrue(all(r["model_abstain"] == "True" for r in routes))
            self.assertTrue((out / "pathmat_route_model_inputs.json").is_file())
            self.assertTrue((out / "pathmat_route_model_outputs.json").is_file())
            self.assertTrue((out / "agent_trace.jsonl").is_file())
            self.assertTrue((out / "DESIGN_REPORT.md").is_file())
            for row in generated:
                cif = out / row["cif_path"]
                self.assertTrue(cif.is_file())
                self.assertEqual(hashlib.sha256(cif.read_bytes()).hexdigest(), row["cif_sha256"])
                structure = Structure.from_file(cif)
                self.assertEqual(structure.composition.reduced_formula, row["formula"])
                self.assertAlmostEqual(float(structure.charge or 0), 0, places=6)
                self.assertEqual(row["design_status"], "unrelaxed_structure_hypothesis")

    def test_agent_refuses_to_overwrite_existing_run(self) -> None:
        import yaml

        config = MODULE / "config" / "design_agent_perovskite.yaml"
        with tempfile.TemporaryDirectory() as tmp:
            cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
            cfg["pathmat_routes"] = {"route_scores_csv": "", "model_manifest": ""}
            local_config = Path(tmp) / "design.yaml"
            local_config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            out = Path(tmp) / "run"
            PathMatDesignAgent().run(local_config, out)
            with self.assertRaises(FileExistsError):
                PathMatDesignAgent().run(local_config, out)

    def test_precursor_templates_are_stoichiometrically_balanced(self) -> None:
        from pymatgen.core import Composition

        carbonate_row = {"formula": "CaTiO3", "a_site": "Ca", "a_oxidation_state": 2,
                         "b_site": "Ti", "b_oxidation_state": 4}
        carbonate, _ = PathMatRouteAgent.precursor_recipe(carbonate_row)
        self.assertEqual(Composition(carbonate[0]) + Composition(carbonate[1]),
                         Composition("CaTiO3") + Composition("CO2"))
        oxide_row = {"formula": "LaFeO3", "a_site": "La", "a_oxidation_state": 3,
                     "b_site": "Fe", "b_oxidation_state": 3}
        oxides, _ = PathMatRouteAgent.precursor_recipe(oxide_row)
        self.assertEqual(Composition(oxides[0]) + Composition(oxides[1]), 2 * Composition("LaFeO3"))

    def test_optional_pathmat_route_agent_is_auditable_and_bounded(self) -> None:
        import yaml

        base = yaml.safe_load((MODULE / "config" / "design_agent_perovskite.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            routes = root / "route_scores.csv"
            with routes.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["target_formula", "route_id", "balance_status", "b2_source_label_score", "route_source_id"])
                writer.writeheader()
                writer.writerow({"target_formula": "SrTiO3", "route_id": "fixture-route", "balance_status": "balanced",
                                 "b2_source_label_score": "0.2", "route_source_id": "fixture-source"})
            model = root / "pathmat_model_manifest.json"
            model.write_text(json.dumps({"model_id": "fixture-B2", "model_version": "v1",
                                         "training_data_id": "fixture", "score_semantics": "B2 source-label score"}), encoding="utf-8")
            base["pathmat_routes"] = {"route_scores_csv": str(routes), "model_manifest": str(model)}
            config = root / "design.yaml"
            config.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
            out = PathMatDesignAgent().run(config, root / "run")
            summary = json.loads((out / "design_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["pathmat_route_integration"]["status"], "joined_source_label_scores_for_review_only")
            self.assertGreaterEqual(summary["pathmat_route_integration"]["joined_route_rows"], 1)
            with (out / "pathmat_route_ranking.csv").open(newline="", encoding="utf-8") as f:
                joined = list(csv.DictReader(f))
            self.assertTrue(all("not a success probability" in row["score_interpretation"] for row in joined))


if __name__ == "__main__":
    unittest.main()
