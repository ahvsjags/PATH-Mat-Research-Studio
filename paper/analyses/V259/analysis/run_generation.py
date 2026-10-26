#!/usr/bin/env python3
"""Reproducible launcher for MatterGen target-conditioned crystal generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_cif_archive(archive: Path, destination: Path) -> list[Path]:
    """Safely unpack only CIFs; reject traversal, symlinks, and duplicate paths."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    seen: set[str] = set()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts or ":" in pure.parts[0] or "\x00" in name:
                raise ValueError(f"Unsafe path in MatterGen CIF archive: {info.filename}")
            # Unix ZIP symlink entries are identified by file type in upper mode bits.
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symlink not allowed in MatterGen CIF archive: {info.filename}")
            if not name.lower().endswith(".cif"):
                continue
            rel = Path(*pure.parts)
            key = rel.as_posix().lower()
            if key in seen:
                raise ValueError(f"Duplicate CIF path in MatterGen archive: {info.filename}")
            seen.add(key)
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            extracted.append(target)
    return sorted(extracted)


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read the design config.") from exc
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("schema_version") != 1:
        raise SystemExit("Unsupported design config schema_version.")
    gen = cfg["generator"]
    if int(gen["batch_size"]) < 1 or int(gen["num_batches"]) < 1:
        raise SystemExit("batch_size and num_batches must be positive.")
    if len(gen["seeds"]) != len(set(gen["seeds"])):
        raise SystemExit("Seeds must be unique.")
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--mattergen-generate", default="mattergen-generate")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Pinned local MatterGen checkpoint directory; preferred for reproducibility.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    gen = cfg["generator"]
    out_root = (args.out_root or config_path.parent.parent / "runs" / cfg["design_id"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    executable = shutil.which(args.mattergen_generate)
    if not executable and not args.dry_run:
        raise SystemExit(
            "mattergen-generate was not found. Install the official MatterGen package "
            "in the active environment and verify the checkpoint weights before running."
        )

    model_dir = args.model_dir.resolve() if args.model_dir else None
    if not args.dry_run and model_dir is None:
        raise SystemExit("A real generation run requires --model-dir pointing to an immutable, digest-verified checkpoint snapshot.")
    if model_dir is not None:
        checkpoint = model_dir / "checkpoints" / "last.ckpt"
        model_config = model_dir / "config.yaml"
        if not checkpoint.is_file():
            raise SystemExit(f"Pinned model directory must contain checkpoints/last.ckpt: {model_dir}")
        if not model_config.is_file():
            raise SystemExit(f"Pinned model directory must contain config.yaml: {model_dir}")
        pointer_prefix = checkpoint.open("rb").read(128)
        if b"git-lfs.github.com/spec/v1" in pointer_prefix:
            raise SystemExit(f"Checkpoint is still a Git-LFS pointer, not weights: {checkpoint}")
    run_manifest: dict[str, Any] = {
        "design_id": cfg["design_id"],
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "mattergen_generate": executable or args.mattergen_generate,
        "target": cfg["target"],
        "generator": gen,
        "model_dir": str(model_dir) if model_dir else None,
        "checkpoint_sha256": sha256(model_dir / "checkpoints" / "last.ckpt") if model_dir else None,
        "model_config_sha256": sha256(model_dir / "config.yaml") if model_dir else None,
        "model_manifest_path": str(model_dir.parent.parent / "model_manifest.json") if model_dir and (model_dir.parent.parent / "model_manifest.json").exists() else None,
        "runs": [],
    }

    n_requested = int(gen["batch_size"]) * int(gen["num_batches"])
    if gen.get("max_candidates_per_seed") is not None:
        n_requested = min(n_requested, int(gen["max_candidates_per_seed"]))
    batches = (n_requested + int(gen["batch_size"]) - 1) // int(gen["batch_size"])

    for seed in gen["seeds"]:
        run_dir = out_root / f"seed_{int(seed)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.dry_run and any(run_dir.iterdir()):
            raise SystemExit(f"Refusing to overwrite a non-empty run directory; choose a new --out-root: {run_dir}")
        command = [executable or args.mattergen_generate, str(run_dir)]
        if model_dir is not None:
            command.append(f"--model_path={model_dir}")
        else:
            command.append(f"--pretrained-name={gen['pretrained_name']}")
        command.extend(
            [
                f"--batch_size={int(gen['batch_size'])}",
                f"--num_batches={batches}",
                f"--seed={int(seed)}",
                f"--record_trajectories={str(bool(gen.get('record_trajectories', False)))}",
            ]
        )
        if gen.get("conditioning_enabled", True):
            command.extend(
                [
                    # Fire's dict parser expects no whitespace inside the mapping.
                    f"--properties_to_condition_on={{{gen['target_property']}:{float(gen['target_value_eV'])}}}",
                    f"--diffusion_guidance_factor={float(gen['guidance_factor'])}",
                ]
            )
        entry = {
            "seed": int(seed),
            "run_dir": str(run_dir),
            "command": command,
            "requested_candidates_upper_bound": n_requested,
            "status": "dry_run" if args.dry_run else "started",
            "checkpoint_sha256": run_manifest["checkpoint_sha256"],
            "model_config_sha256": run_manifest["model_config_sha256"],
        }
        if not args.dry_run:
            completed = subprocess.run(command, check=False, text=True, capture_output=True)
            stdout_path = run_dir / "mattergen.stdout.log"
            stderr_path = run_dir / "mattergen.stderr.log"
            stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
            stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
            entry.update(
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-8000:],
                    "stderr": completed.stderr[-8000:],
                    "stdout_path": str(stdout_path),
                    "stdout_sha256": sha256(stdout_path),
                    "stderr_path": str(stderr_path),
                    "stderr_sha256": sha256(stderr_path),
                    "status": "completed" if completed.returncode == 0 else "failed",
                }
            )
            if completed.returncode != 0:
                run_manifest["runs"].append(entry)
                manifest_path = out_root / "run_manifest.json"
                manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
                raise SystemExit(f"MatterGen failed for seed {seed}; see {manifest_path}")

            expected = run_dir / "generated_crystals_cif.zip"
            if expected.exists():
                entry["cif_zip"] = str(expected)
                entry["cif_zip_sha256"] = sha256(expected)
                try:
                    extracted = extract_cif_archive(expected, run_dir / "generated_structures")
                except Exception as exc:
                    entry["status"] = "invalid_cif_archive"
                    entry["archive_error"] = f"{type(exc).__name__}: {exc}"
                    run_manifest["runs"].append(entry)
                    (out_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
                    raise SystemExit(f"Unsafe/invalid CIF archive; failure recorded in {out_root / 'run_manifest.json'}") from exc
                entry["generated_cif_count"] = len(extracted)
                entry["generated_cif_sha256"] = {p.relative_to(run_dir).as_posix(): sha256(p) for p in extracted}
                if len(extracted) != n_requested:
                    entry["status"] = "generated_count_mismatch"
                    run_manifest["runs"].append(entry)
                    (out_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
                    raise SystemExit(f"Expected {n_requested} CIFs but extracted {len(extracted)}; failure recorded in {out_root / 'run_manifest.json'}")
            else:
                entry["status"] = "completed_output_missing"
                run_manifest["runs"].append(entry)
                (out_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
                raise SystemExit(f"Expected generated CIF archive not found; failure recorded in {out_root / 'run_manifest.json'}: {expected}")
        else:
            entry["display_command"] = subprocess.list2cmdline(command)
        run_manifest["runs"].append(entry)

    manifest_path = out_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
