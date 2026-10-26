#!/usr/bin/env python3
"""Create a reference-snapshot manifest row after hashing the local snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import zipfile
from pathlib import Path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_hash(path: Path) -> str:
    if path.is_file():
        return file_hash(path)
    h = hashlib.sha256()
    for p in sorted(x for x in path.rglob("*") if x.is_file()):
        h.update(p.relative_to(path).as_posix().encode())
        h.update(bytes.fromhex(file_hash(p)))
    return h.hexdigest()


def cif_count(path: Path) -> int:
    if path.is_file() and path.suffix.lower() == ".cif":
        return 1
    if path.is_dir():
        return sum(1 for _ in path.rglob("*.cif"))
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            return sum(1 for name in zf.namelist() if name.lower().endswith(".cif"))
    raise SystemExit("Reference snapshot must be a .cif, ZIP, or directory containing CIF files.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--access-date", required=True, help="YYYY-MM-DD snapshot/access date")
    ap.add_argument("--license", required=True, help="Exact license/terms identifier; do not infer permission")
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    snapshot = args.snapshot.resolve()
    if not snapshot.exists():
        raise SystemExit(f"Snapshot does not exist: {snapshot}")
    row = {
        "dataset_id": args.dataset_id,
        "version": args.version,
        "access_date": args.access_date,
        "license": args.license,
        "local_path": str(snapshot),
        "sha256": snapshot_hash(snapshot),
        "structure_count": cif_count(snapshot),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.manifest.exists() and args.manifest.stat().st_size:
        with args.manifest.open(newline="", encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))
    if any(r.get("dataset_id") == args.dataset_id and r.get("version") == args.version for r in existing):
        raise SystemExit(f"Manifest already contains {args.dataset_id} version {args.version}; refusing overwrite.")
    fields = ["dataset_id", "version", "access_date", "license", "local_path", "sha256", "structure_count"]
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing + [row])
    print(args.manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
