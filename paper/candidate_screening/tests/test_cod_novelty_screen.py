from __future__ import annotations

import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE / "analysis"))
from run_cod_novelty_screen import cod_hill_formula  # noqa: E402


def test_cod_hill_order_for_ternary_perovskite() -> None:
    assert cod_hill_formula("SrTiO3") == "O3 Sr Ti"


def test_cod_hill_order_for_double_perovskite() -> None:
    assert cod_hill_formula("Ba2NbFeO6") == "Ba2 Fe Nb O6"


def test_cod_hill_order_keeps_carbon_and_hydrogen_first() -> None:
    assert cod_hill_formula("C8H10N4O2") == "C8 H10 N4 O2"
