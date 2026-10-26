"""Pinned local CHGNet relaxer for equal-budget design-arm evaluation."""
from __future__ import annotations

import hashlib
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure


EXPECTED_CHECKPOINT_SHA256 = "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"
EXPECTED_CHECKPOINT_REL = Path("chgnet") / "pretrained" / "0.3.0" / "chgnet_0.3.0_e29f68s314m37.pth.tar"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CHGNetOracle:
    """The output is a model screening energy, never a DFT/experiment label."""

    def __init__(self, runtime_root: Path, *, fmax: float = 0.05, max_steps: int = 500):
        self.runtime_root = runtime_root.resolve(strict=True)
        self.checkpoint = self.runtime_root / EXPECTED_CHECKPOINT_REL
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        if self.checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("CHGNet checkpoint SHA-256 differs from the frozen experiment")
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        if self.fmax <= 0 or self.max_steps <= 0:
            raise ValueError("fmax and max_steps must be positive")
        sys.path.insert(0, str(self.runtime_root))
        from chgnet import __version__ as package_version
        from chgnet.model import CHGNet, StructOptimizer
        import torch

        self.package_version = package_version
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CHGNet.load(model_name="0.3.0", use_device=self.device, verbose=False)
        self.relaxer = StructOptimizer(model=self.model, use_device=self.device)

    def relax_file(self, cif_path: Path, output_cif: Path) -> dict[str, Any]:
        input_hash = sha256_file(cif_path)
        structure = Structure.from_file(cif_path, primitive=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.relaxer.relax(structure, fmax=self.fmax, steps=self.max_steps,
                                        relax_cell=True, verbose=False, assign_magmoms=True)
            final = result["final_structure"]
            prediction = self.model.predict_structure(final)
        energy = float(np.asarray(prediction["e"]).reshape(-1)[0])
        forces = np.asarray(prediction["f"], dtype=float)
        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else float("nan")
        steps = len(result["trajectory"].energies)
        converged = bool(np.isfinite(max_force) and max_force <= self.fmax + 1e-6)
        warning_counts = Counter((item.category.__name__, str(item.message)) for item in caught)
        output_cif.parent.mkdir(parents=True, exist_ok=True)
        final.to(filename=output_cif, fmt="cif")
        return {
            "input_path": str(cif_path), "input_sha256": input_hash,
            "output_path": str(output_cif), "output_sha256": sha256_file(output_cif),
            "formula": final.composition.reduced_formula,
            "energy_eV_atom": energy,
            "max_force_eV_A": max_force,
            "force_converged": converged,
            "steps": steps,
            "atom_count": len(final),
            "model": "CHGNet 0.3.0 MPtrj-pretrained",
            "package_version": self.package_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "fmax_eV_A": self.fmax,
            "max_steps": self.max_steps,
            "runtime_warning_count": len(caught),
            "runtime_warning_summary": [{"category": category, "message": message, "count": count}
                                        for (category, message), count in warning_counts.items()],
            "interpretation": "ML interatomic-potential relaxation screen; not DFT, convex-hull stability, or experimental validation",
        }
