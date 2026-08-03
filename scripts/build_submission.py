#!/usr/bin/env python3
"""Build the best known submission from its three frozen text components.

This module deliberately uses only Python's standard library.  It does not run
LLM agents; it combines their audited predictions by ID and verifies the result
against the known-good leaderboard file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
EXPECTED_IDS = list(range(62))
SUBMISSION_COLUMNS = ["id", "TCC", "MS", "M", "VC", "VA", "VE"]

DEFAULT_TCC = (
    WORKSPACE
    / "artifacts/frozen/target_concept_coverage_predictions.csv"
)
DEFAULT_MS_BASE = (
    WORKSPACE
    / "artifacts/frozen/mapping_strength_predictions.csv"
)
DEFAULT_M = (
    WORKSPACE
    / "artifacts/frozen/metaphoricity_predictions.csv"
)
DEFAULT_OUTPUT = WORKSPACE / "output/submission.csv"
DEFAULT_AUDIT = WORKSPACE / "output/build_audit.json"
DEFAULT_EXPECTED = (
    WORKSPACE
    / "artifacts/frozen/known_good_submission.csv"
)


class SubmissionBuildError(ValueError):
    """Raised when a component cannot produce a valid competition file."""


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else WORKSPACE / path


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise SubmissionBuildError(f"Missing input file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns - fieldnames
        if missing:
            raise SubmissionBuildError(
                f"{path} is missing columns: {sorted(missing)}"
            )
        return list(reader)


def _integer(value: str, *, label: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise SubmissionBuildError(f"{label} must be an integer; got {value!r}") from error
    if str(parsed) != str(value).strip():
        raise SubmissionBuildError(f"{label} must be an integer; got {value!r}")
    if not minimum <= parsed <= maximum:
        raise SubmissionBuildError(
            f"{label} must be in {minimum}..{maximum}; got {parsed}"
        )
    return parsed


def _index_rows(
    path: Path,
    *,
    required_columns: set[str],
) -> dict[int, dict[str, str]]:
    rows = _read_rows(path, required_columns | {"id"})
    indexed: dict[int, dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        example_id = _integer(
            row["id"],
            label=f"id at {path}:{row_number}",
            minimum=0,
            maximum=61,
        )
        if example_id in indexed:
            raise SubmissionBuildError(f"Duplicate id={example_id} in {path}")
        indexed[example_id] = row
    if sorted(indexed) != EXPECTED_IDS:
        missing = sorted(set(EXPECTED_IDS) - set(indexed))
        extra = sorted(set(indexed) - set(EXPECTED_IDS))
        raise SubmissionBuildError(
            f"{path} must contain ids 0..61; missing={missing}, extra={extra}"
        )
    return indexed


def _submission_text(rows: list[list[int | str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def _read_submission_matrix(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))


def build_submission(
    *,
    tcc_path: Path = DEFAULT_TCC,
    ms_base_path: Path = DEFAULT_MS_BASE,
    m_path: Path = DEFAULT_M,
    output_path: Path = DEFAULT_OUTPUT,
    audit_path: Path = DEFAULT_AUDIT,
    expected_path: Path | None = DEFAULT_EXPECTED,
) -> dict[str, Any]:
    """Merge TCC, MS, and M predictions and return a machine-readable audit."""

    tcc_path = _resolve(tcc_path)
    ms_base_path = _resolve(ms_base_path)
    m_path = _resolve(m_path)
    output_path = _resolve(output_path)
    audit_path = _resolve(audit_path)
    expected_path = _resolve(expected_path) if expected_path is not None else None

    tcc_rows = _index_rows(tcc_path, required_columns={"TCC"})
    base_rows = _index_rows(ms_base_path, required_columns={"MS"})
    m_rows = _index_rows(m_path, required_columns={"M"})

    data: list[list[int]] = []
    for example_id in EXPECTED_IDS:
        tcc = _integer(
            tcc_rows[example_id]["TCC"],
            label=f"TCC for id={example_id}",
            minimum=0,
            maximum=2,
        )
        ms = _integer(
            base_rows[example_id]["MS"],
            label=f"MS for id={example_id}",
            minimum=0,
            maximum=2,
        )
        m_score = _integer(
            m_rows[example_id]["M"],
            label=f"M for id={example_id}",
            minimum=0,
            maximum=2,
        )
        video = [
            _integer(
                base_rows[example_id][column],
                label=f"{column} for id={example_id}",
                minimum=0,
                maximum=3,
            )
            if column in base_rows[example_id]
            and base_rows[example_id][column] not in {None, ""}
            else 0
            for column in ("VC", "VA", "VE")
        ]
        if video != [0, 0, 0]:
            raise SubmissionBuildError(
                f"Best text-only submission requires zero video scores; "
                f"id={example_id} has {video}"
            )
        data.append([example_id, tcc, ms, m_score, *video])

    matrix: list[list[int | str]] = [SUBMISSION_COLUMNS, *data]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_submission_text(matrix), encoding="utf-8", newline="")

    expected_match: bool | None = None
    expected_sha256: str | None = None
    if expected_path is not None:
        if not expected_path.is_file():
            raise SubmissionBuildError(f"Missing expected submission: {expected_path}")
        expected_matrix = _read_submission_matrix(expected_path)
        actual_matrix = _read_submission_matrix(output_path)
        expected_match = actual_matrix == expected_matrix
        expected_sha256 = _sha256(expected_path)

    audit: dict[str, Any] = {
        "method": {
            "TCC": "tcc_v1_facet_conservative_v1",
            "MS": "original_v1_mapping_extractor_ms_judge",
            "M": "v7_1_role_audit_loo",
            "video": "all_zero",
        },
        "sources": {
            "TCC": {"path": _display_path(tcc_path), "sha256": _sha256(tcc_path)},
            "MS_base": {
                "path": _display_path(ms_base_path),
                "sha256": _sha256(ms_base_path),
            },
            "M": {"path": _display_path(m_path), "sha256": _sha256(m_path)},
        },
        "output": {
            "path": _display_path(output_path),
            "sha256": _sha256(output_path),
            "rows": len(data),
            "columns": SUBMISSION_COLUMNS,
        },
        "distributions": {
            column: dict(sorted(Counter(row[index] for row in data).items()))
            for index, column in ((1, "TCC"), (2, "MS"), (3, "M"))
        },
        "verification": {
            "expected_path": (
                _display_path(expected_path) if expected_path is not None else None
            ),
            "expected_sha256": expected_sha256,
            "value_for_value_match": expected_match,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if expected_match is False:
        raise SubmissionBuildError(
            f"Generated submission differs from known-good file: {expected_path}"
        )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and verify the best known analogy submission."
    )
    parser.add_argument("--tcc", type=Path, default=DEFAULT_TCC)
    parser.add_argument("--ms-base", type=Path, default=DEFAULT_MS_BASE)
    parser.add_argument("--m", type=Path, default=DEFAULT_M)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED,
        help="Known-good file used for value-for-value verification.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Build without comparing to the known-good leaderboard file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit = build_submission(
        tcc_path=args.tcc,
        ms_base_path=args.ms_base,
        m_path=args.m,
        output_path=args.output,
        audit_path=args.audit,
        expected_path=None if args.no_verify else args.expected,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
