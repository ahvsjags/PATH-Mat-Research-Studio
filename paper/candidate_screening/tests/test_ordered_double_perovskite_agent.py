from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Composition, Structure

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE / "agent"))
from ordered_double_perovskite_agent import build_structure, propose  # noqa: E402
from pathmat_design_agent import PathMatRouteAgent  # noqa: E402
import yaml  # noqa: E402


class OrderedDoublePerovskiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load((MODULE / "config" / "design_agent_double_perovskite.yaml").read_text(encoding="utf-8"))

    def test_charge_neutral_40_atom_rocksalt_supercell(self) -> None:
        proposal = next(p for p in propose(self.config) if p["formula"] == "Sr2NbFeO6")
        structure, metrics = build_structure(proposal, self.config["design_task"])
        self.assertEqual(len(structure), 40)
        self.assertEqual(structure.composition.reduced_formula, "Sr2NbFeO6")
        self.assertAlmostEqual(float(structure.charge or 0), 0.0, places=8)
        self.assertEqual(sum(site.specie.symbol == "Fe" for site in structure), 4)
        self.assertEqual(sum(site.specie.symbol == "Nb" for site in structure), 4)
        self.assertTrue(0.80 <= metrics["tolerance_factor"] <= 1.08)

    def test_charge_enumeration_and_oxide_reaction_balance(self) -> None:
        proposals = propose(self.config)
        self.assertEqual(len(proposals), 72)
        self.assertTrue(all(p["formal_charge_sum"] == 0 for p in proposals))
        p = next(p for p in proposals if p["formula"] == "Sr2NbFeO6")
        precursors, equation = PathMatRouteAgent.precursor_recipe(p)
        self.assertIn("SrO", precursors)
        self.assertIn("Fe2O3", precursors)
        self.assertIn("Nb2O5", precursors)
        self.assertIn("-> 2Sr2NbFeO6", equation)
        lhs = 4 * Composition("SrO") + Composition("Fe2O3") + Composition("Nb2O5")
        rhs = 2 * Composition("Sr2NbFeO6")
        self.assertEqual(lhs, rhs)

    def test_generated_cif_round_trip_and_formula(self) -> None:
        p = next(p for p in propose(self.config) if p["formula"] == "Ca2TaCrO6")
        structure, _ = build_structure(p, self.config["design_task"])
        with tempfile.TemporaryDirectory() as tmp:
            cif = Path(tmp) / "candidate.cif"
            structure.to(filename=cif, fmt="cif")
            loaded = Structure.from_file(cif)
            self.assertEqual(loaded.composition.reduced_formula, "Ca2TaCrO6")
            self.assertEqual(len(loaded), 40)


if __name__ == "__main__":
    unittest.main()
