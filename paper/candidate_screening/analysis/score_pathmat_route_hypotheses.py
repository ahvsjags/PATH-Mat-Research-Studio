#!/usr/bin/env python3
"""Run the frozen local PATH-Mat model in its native app environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.input.read_text(encoding="utf-8"))
    model_path = Path(request["model_path"]).resolve()
    vendor_path = Path(request["vendor_path"]).resolve()
    if not model_path.is_file() or not vendor_path.is_dir():
        raise FileNotFoundError("The configured PATH-Mat model or vendor tree is unavailable")
    sys.path.insert(0, str(vendor_path))
    from pathmat.ec_pathwm_agent_adapter import ECPathWMAdapter
    import sklearn

    adapter = ECPathWMAdapter.load(model_path)
    results = []
    for item in request["records"]:
        record = item["record"]
        signal = adapter.score_condition(record)
        gate = adapter.score(record)
        results.append({
            "route_id": item["route_id"],
            "condition_target_score": signal["condition_target_score"],
            "condition_competitor_score": signal["condition_competitor_score"],
            "condition_margin": signal["condition_margin"],
            "ensemble_spread": signal["ensemble_spread"],
            "feature_support": signal.get("feature_support", {}),
            "model_abstain": gate["abstain"],
            "abstention_reasons": gate.get("abstention_reasons", []),
            "evidence_risk": gate["evidence_risk"],
            "path_score_proxy": gate["path_score"],
        })
    metadata = dict(getattr(adapter.model, "metadata", {}) or {})
    output = {
        "adapter_version": adapter.version,
        "model_sha256": sha256_file(model_path),
        "model_metadata": metadata,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "input_route_count": len(request["records"]),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"scored": len(results), "model_sha256": output["model_sha256"],
                      "sklearn_version": output["sklearn_version"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
