from __future__ import annotations

import json
from pathlib import Path

from pymatgen.core import Structure

from agent.ordering_search import generate_ordering_variant, generate_tilt_variant


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "p1_smoke"


def test_ordering_children_preserve_formula_counts_and_parent():
    row = json.loads((FIXTURES / "fixture_manifest.json").read_text(encoding="utf-8"))["candidates"][0]
    parent_path = FIXTURES / row["cif_path"]
    parent = Structure.from_file(parent_path, primitive=False)
    parent_before = parent.copy()
    composition = parent.composition.get_el_amt_dict()
    cations = [el for el in composition if el not in {"O", "Ca", "Sr", "Ba"}]
    pentavalent = {"Nb", "Ta", "Sb", "V"}
    b_prime = next(el for el in cations if el in pentavalent)
    b_site = next(el for el in cations if el != b_prime)
    proposal = {"formula": row["formula"], "b_site": b_site, "b_oxidation_state": 3,
                "b_prime_site": b_prime, "b_prime_oxidation_state": 5}
    meta_ids = set()
    for ordering in ("layered_001", "columnar_110"):
        child, metadata = generate_ordering_variant(parent, proposal,
                                                     parent_candidate_id=row["candidate_id"],
                                                     ordering=ordering)
        assert child.composition.reduced_formula == parent.composition.reduced_formula
        assert metadata["composition_unchanged"] is True
        assert sorted(metadata["b_site_counts"].values()) == [4, 4]
        assert metadata["candidate_id"] != row["candidate_id"]
        meta_ids.add(metadata["candidate_id"])
    assert len(meta_ids) == 2
    assert parent == parent_before


def test_unknown_ordering_fails_closed():
    row = json.loads((FIXTURES / "fixture_manifest.json").read_text(encoding="utf-8"))["candidates"][0]
    parent = Structure.from_file(FIXTURES / row["cif_path"], primitive=False)
    composition = parent.composition.get_el_amt_dict()
    cations = [el for el in composition if el not in {"O", "Ca", "Sr", "Ba"}]
    pentavalent = {"Nb", "Ta", "Sb", "V"}
    b_prime = next(el for el in cations if el in pentavalent)
    b_site = next(el for el in cations if el != b_prime)
    proposal = {"formula": row["formula"], "b_site": b_site, "b_oxidation_state": 3,
                "b_prime_site": b_prime, "b_prime_oxidation_state": 5}
    import pytest
    with pytest.raises(ValueError, match="unsupported B-site ordering"):
        generate_ordering_variant(parent, proposal, parent_candidate_id=row["candidate_id"], ordering="invented")


def test_tilt_children_preserve_composition_and_parent():
    row = json.loads((FIXTURES / "fixture_manifest.json").read_text(encoding="utf-8"))["candidates"][0]
    parent = Structure.from_file(FIXTURES / row["cif_path"], primitive=False)
    parent_copy = parent.copy()
    elements = {el for el in parent.composition.get_el_amt_dict() if el not in {"O", "Ca", "Sr", "Ba"}}
    for variant in ("a0a0c_minus_approx", "a_minus_a_minus_a_minus_approx"):
        child, meta = generate_tilt_variant(parent, variant=variant, theta_degrees=5.0,
                                            b_site_elements=elements)
        assert child.composition.reduced_formula == parent.composition.reduced_formula
        assert meta["composition_unchanged"] is True
        assert meta["oxygen_sites_displaced"] > 0
        assert 0 < meta["maximum_oxygen_displacement_angstrom"] < 0.3
        assert child != parent
    assert parent == parent_copy
