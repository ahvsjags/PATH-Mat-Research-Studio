"""Enumerate symmetry-distinct 4:4 B-site orderings for A2BB'O6 prototypes.

The catalog is an operator library, not a stability predictor. It preserves the
composition and parent cell and binds every child to the parent digest.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Species, Structure


def _canonical_matcher() -> StructureMatcher:
    # Tight enough to identify exact parent-cell symmetry equivalence before
    # relaxation; the looser frozen endpoint matcher is used only after relax.
    return StructureMatcher(ltol=0.2, stol=0.2, angle_tol=2.0,
                           primitive_cell=True, scale=True,
                           attempt_supercell=True, allow_subset=False)


def enumerate_bsite_orderings(
    parent: Structure,
    *,
    b_site: str,
    b_oxidation_state: int,
    b_prime_site: str,
    b_prime_oxidation_state: int,
    expected_sites_per_species: int = 4,
) -> list[tuple[Structure, dict[str, Any]]]:
    """Return one deterministic representative per symmetry-distinct 4:4 order.

    Enumeration is over all ``choose(8, 4)`` assignments on the 2x2x2
    perovskite B sublattice. Equivalence is established by a fixed
    ``StructureMatcher`` on the unrelaxed parent cell, not by agent judgement.
    """
    b_indices = [i for i, site in enumerate(parent)
                 if site.specie.symbol in {b_site, b_prime_site}]
    if len(b_indices) != 2 * expected_sites_per_species:
        raise ValueError(f"expected {2 * expected_sites_per_species} B sites; got {len(b_indices)}")
    if parent.composition[b_site] != expected_sites_per_species or parent.composition[b_prime_site] != expected_sites_per_species:
        raise ValueError("parent must contain the declared 4:4 B-site composition")

    matcher = _canonical_matcher()
    representatives: list[tuple[Structure, dict[str, Any]]] = []
    classes: list[list[Structure]] = []
    for selected in combinations(range(len(b_indices)), expected_sites_per_species):
        selected_set = set(selected)
        child = parent.copy()
        for local_idx, site_idx in enumerate(b_indices):
            symbol, oxidation = ((b_site, b_oxidation_state) if local_idx in selected_set
                                 else (b_prime_site, b_prime_oxidation_state))
            child.replace(site_idx, Species(symbol, oxidation))
        if abs(float(child.charge or 0.0)) > 1e-8:
            raise ValueError("enumerated ordering does not preserve formal charge neutrality")
        matched = None
        for class_idx, group in enumerate(classes):
            if matcher.fit(child, group[0]):
                matched = class_idx
                group.append(child)
                break
        if matched is not None:
            continue
        classes.append([child])
        bitstring = "".join("1" if idx in selected_set else "0" for idx in range(len(b_indices)))
        payload = json.dumps(child.as_dict(), sort_keys=True, separators=(",", ":"))
        structure_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        representatives.append((child, {
            "operator": "bsite_ordering_catalog",
            "ordering_class": len(representatives),
            "site_assignment_bitstring": bitstring,
            "selected_b_indices": list(selected),
            "equivalence_class_size_so_far": 1,
            "parent_formula": parent.composition.reduced_formula,
            "formula": child.composition.reduced_formula,
            "b_site": b_site,
            "b_prime_site": b_prime_site,
            "composition_preserved": child.composition.reduced_formula == parent.composition.reduced_formula,
            "charge_sum": float(child.charge or 0.0),
            "structure_payload_sha256": structure_hash,
            "matches_parent_ordering": bool(matcher.fit(child, parent)),
        }))

    # Attach complete orbit sizes after enumeration; all metadata remains
    # deterministic in lexicographic bitstring order.
    for meta, group in zip((m for _, m in representatives), classes):
        meta["equivalence_class_size"] = len(group)
        meta.pop("equivalence_class_size_so_far", None)
    return representatives

