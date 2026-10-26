"""Auditable B-site-ordering alternatives for ordered A2BB'O6 design hypotheses."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Species, Structure


ORDERING_RULES = {
    "rocksalt_111": lambda index: (sum(index) % 2) == 0,
    "layered_001": lambda index: (index[2] % 2) == 0,
    "columnar_110": lambda index: ((index[0] + index[1]) % 2) == 0,
}


def generate_ordering_variant(parent: Structure, proposal: dict[str, Any], *, parent_candidate_id: str,
                              ordering: str) -> tuple[Structure, dict[str, Any]]:
    """Return a same-composition child with a declared B/B' ordering.

    Reassignments are limited to the eight B-sublattice sites in the 2x2x2
    supercell. A child gets a new identity; the parent is never overwritten.
    """
    if ordering not in ORDERING_RULES:
        raise ValueError(f"unsupported B-site ordering: {ordering}")
    structure = deepcopy(parent)
    b, qb = str(proposal["b_site"]), int(proposal["b_oxidation_state"])
    bp, qbp = str(proposal["b_prime_site"]), int(proposal["b_prime_oxidation_state"])
    b_indices = [i for i, site in enumerate(structure) if site.specie.symbol in {b, bp}]
    if len(b_indices) != 8:
        raise ValueError(f"expected 8 B-sublattice sites, observed {len(b_indices)}")
    counts = {b: 0, bp: 0}
    rule = ORDERING_RULES[ordering]
    for i in b_indices:
        index = [int(round(float(x) * 2 - 0.5)) % 2 for x in structure[i].frac_coords]
        use_b = bool(rule(index))
        symbol, oxidation = (b, qb) if use_b else (bp, qbp)
        structure.replace(i, Species(symbol, oxidation))
        counts[symbol] += 1
    if counts != {b: 4, bp: 4}:
        raise ValueError(f"ordering does not preserve the 4:4 B/B' count: {counts}")
    payload = json.dumps(structure.as_dict(), sort_keys=True, separators=(",", ":"))
    child_id = hashlib.sha256(f"{parent_candidate_id}\0{ordering}\0{payload}".encode()).hexdigest()
    return structure, {
        "candidate_id": child_id,
        "parent_candidate_id": parent_candidate_id,
        "formula": str(proposal["formula"]),
        "b_site": b,
        "b_prime_site": bp,
        "ordering": ordering,
        "b_site_counts": counts,
        "composition_unchanged": True,
        "candidate_kind": "same_composition_ordering_child",
    }


def write_ordering_variant(parent: Structure, proposal: dict[str, Any], *, parent_candidate_id: str,
                           ordering: str, output_dir: Path) -> tuple[Structure, dict[str, Any]]:
    structure, metadata = generate_ordering_variant(parent, proposal,
                                                     parent_candidate_id=parent_candidate_id,
                                                     ordering=ordering)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{metadata['candidate_id']}_{metadata['formula']}_{ordering}.cif"
    structure.to(filename=path, fmt="cif")
    metadata["cif_path"] = str(path)
    metadata["cif_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return structure, metadata


TILT_RULES = {"a0a0c_minus_approx", "a_minus_a_minus_a_minus_approx"}


def generate_tilt_variant(parent: Structure, *, variant: str, theta_degrees: float = 5.0,
                          b_site_elements: set[str] | None = None) -> tuple[Structure, dict[str, Any]]:
    """Seed a small, coherent oxygen-rotation distortion before MLIP relaxation.

    Shared oxygen displacements are averaged across their neighboring BO6
    octahedra. The `approx` suffix is intentional: this is an initialization
    heuristic, not a fully symmetrized Glazer-mode construction.
    """
    if variant not in TILT_RULES:
        raise ValueError(f"unsupported tilt initialization: {variant}")
    if not 0.0 < theta_degrees <= 12.0:
        raise ValueError("theta_degrees must be in (0, 12] for the pilot")
    structure = deepcopy(parent)
    elements = b_site_elements or {site.specie.symbol for site in structure if site.specie.symbol not in {"O", "Ca", "Sr", "Ba"}}
    b_indices = [i for i, site in enumerate(structure) if site.specie.symbol in elements]
    contributions: dict[int, list[np.ndarray]] = defaultdict(list)
    theta = np.deg2rad(float(theta_degrees))
    axes = {
        "a0a0c_minus_approx": np.array([0.0, 0.0, 1.0]),
        "a_minus_a_minus_a_minus_approx": np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    }
    for i in b_indices:
        b_center = structure[i]
        if variant == "a0a0c_minus_approx":
            cell = [int(round(float(x) * 2 - 0.5)) % 2 for x in b_center.frac_coords]
            sign = -1.0 if (cell[0] + cell[1]) % 2 else 1.0
        else:
            cell = [int(round(float(x) * 2 - 0.5)) % 2 for x in b_center.frac_coords]
            sign = -1.0 if sum(cell) % 2 else 1.0
        omega = axes[variant] * theta * sign
        oxygen_neighbors = [n for n in structure.get_neighbors(b_center, r=2.65) if n.specie.symbol == "O"]
        oxygen_neighbors.sort(key=lambda n: n.nn_distance)
        if len(oxygen_neighbors) < 6:
            raise ValueError(f"B site {i} has only {len(oxygen_neighbors)} oxygen neighbors within 2.65 Å")
        for neighbor in oxygen_neighbors[:6]:
            displacement = np.cross(omega, np.asarray(neighbor.coords) - np.asarray(b_center.coords))
            contributions[int(neighbor.index)].append(displacement)
    if not contributions:
        raise ValueError("no B-site octahedra found for tilt construction")
    displacements = {i: np.mean(vectors, axis=0) for i, vectors in contributions.items()}
    for i, vector in displacements.items():
        structure.translate_sites([i], vector, frac_coords=False, to_unit_cell=True)
    max_disp = max(float(np.linalg.norm(v)) for v in displacements.values())
    metadata = {
        "variant": variant,
        "theta_degrees": float(theta_degrees),
        "b_octahedra_count": len(b_indices),
        "oxygen_sites_displaced": len(displacements),
        "maximum_oxygen_displacement_angstrom": max_disp,
        "composition_unchanged": structure.composition.reduced_formula == parent.composition.reduced_formula,
        "construction": "average of neighboring-octahedron small-rotation displacement proposals",
        "interpretation": "initialization heuristic only; final structure must be relaxed and independently checked",
    }
    return structure, metadata
