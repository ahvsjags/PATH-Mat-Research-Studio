#!/usr/bin/env python3
"""Resolve and snapshot an official MatterGen checkpoint by immutable HF revision."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="microsoft/mattergen")
    ap.add_argument("--model-name", default="dft_band_gap")
    ap.add_argument("--revision", default="main", help="Branch/tag or immutable commit hash")
    ap.add_argument("--destination", type=Path, required=True)
    args = ap.parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub in the MatterGen environment first.") from exc

    info = HfApi().model_info(args.repo_id, revision=args.revision)
    commit = info.sha
    if not commit or len(commit) < 40:
        raise SystemExit("Hugging Face did not return an immutable commit SHA; refusing unpinned download.")
    destination = args.destination.resolve()
    local = Path(snapshot_download(
        repo_id=args.repo_id,
        revision=commit,
        local_dir=destination,
        allow_patterns=[f"checkpoints/{args.model_name}/**"],
    ))
    model_dir = local / "checkpoints" / args.model_name
    checkpoint = model_dir / "checkpoints" / "last.ckpt"
    config = model_dir / "config.yaml"
    for p in [checkpoint, config]:
        if not p.is_file():
            raise SystemExit(f"Required checkpoint asset missing: {p}")
    prefix = checkpoint.open("rb").read(128)
    if b"git-lfs.github.com/spec/v1" in prefix:
        raise SystemExit(f"Downloaded checkpoint is still an LFS pointer: {checkpoint}")
    manifest = {
        "repo_id": args.repo_id,
        "model_name": args.model_name,
        "requested_revision": args.revision,
        "resolved_commit_sha": commit,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": digest(checkpoint),
        "config_path": str(config.resolve()),
        "config_sha256": digest(config),
        "source": f"https://huggingface.co/{args.repo_id}/tree/{commit}",
    }
    manifest_path = destination / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
