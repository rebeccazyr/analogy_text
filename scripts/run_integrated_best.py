#!/usr/bin/env python3
"""Build and prove the zero-loss integrated champion submission.

This entry point intentionally uses only the three frozen, leaderboard-proven
component vectors recorded in manifest.json.  It verifies every component
hash, combines the vectors by ID, and requires the generated submission to be
byte-identical to the frozen champion before reporting success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.combine_recomputed_metrics import (
        combine_recomputed_metrics,
        read_metric,
    )
except ModuleNotFoundError:  # Direct execution from scripts/.
    from combine_recomputed_metrics import (  # type: ignore[no-redef]
        combine_recomputed_metrics,
        read_metric,
    )


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("TCC", "MS", "M")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_integrated_best(
    root: Path = ROOT,
    output_dir: Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Build the champion from frozen components and prove exact parity."""
    if output_dir is None:
        output_dir = root / "runs/integrated-best"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component_paths: dict[str, Path] = {}
    component_audit: dict[str, Any] = {}
    expected_ids = set(range(62))

    for metric in METRICS:
        metadata = manifest["components"][metric]
        path = _resolve(root, metadata["path"])
        actual_hash = sha256(path)
        expected_hash = metadata["sha256"]
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen {metric} hash mismatch: {actual_hash} != "
                f"{expected_hash}"
            )
        predictions = read_metric(path, metric)
        if set(predictions) != expected_ids:
            raise ValueError(
                f"Frozen {metric} IDs must be exactly 0..61; got "
                f"{len(predictions)} rows"
            )
        component_paths[metric] = path
        component_audit[metric] = {
            "method": metadata["method"],
            "reasoning_effort": metadata["reasoning_effort"],
            "path": str(path.relative_to(root)),
            "sha256": actual_hash,
            "rows": len(predictions),
            "distribution": {
                str(score): list(predictions.values()).count(score)
                for score in (0, 1, 2)
            },
        }

    generated_path = output_dir / "submission.csv"
    combine_recomputed_metrics(
        component_paths["TCC"],
        component_paths["MS"],
        component_paths["M"],
        generated_path,
    )

    frozen_metadata = manifest["submission"]
    frozen_path = _resolve(root, frozen_metadata["path"])
    frozen_hash = sha256(frozen_path)
    if frozen_hash != frozen_metadata["sha256"]:
        raise ValueError(
            "Frozen champion submission no longer matches manifest.json"
        )

    generated_hash = sha256(generated_path)
    byte_identical = generated_path.read_bytes() == frozen_path.read_bytes()
    audit: dict[str, Any] = {
        "status": "byte_identical" if byte_identical else "mismatch",
        "proof": (
            "The generated submission was rebuilt from the three frozen "
            "component vectors and compared byte-for-byte with the current "
            "leaderboard champion submission."
        ),
        "manifest": str(manifest_path.relative_to(root)),
        "model": manifest["model"],
        "reasoning": manifest["reasoning"],
        "components": component_audit,
        "generated_submission": {
            "path": _display_path(generated_path, root),
            "sha256": generated_hash,
            "rows": 62,
        },
        "frozen_champion_submission": {
            "path": str(frozen_path.relative_to(root)),
            "sha256": frozen_hash,
        },
        "leaderboard": manifest["leaderboard"],
    }
    audit_path = output_dir / "parity_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not byte_identical:
        raise ValueError(
            "Integrated-best mismatch: generated submission hash "
            f"{generated_hash}, frozen champion hash {frozen_hash}. "
            f"See {audit_path}."
        )
    return generated_path, audit_path, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the current champion from frozen TCC/MS/M components "
            "and prove byte-identical parity."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs/integrated-best",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission, audit_path, audit = build_integrated_best(
        ROOT,
        args.output_dir,
    )
    print(f"Integrated-best submission: {submission}")
    print(f"Parity audit: {audit_path}")
    print(f"Status: {audit['status']}")
    print(f"SHA256: {audit['generated_submission']['sha256']}")


if __name__ == "__main__":
    main()
