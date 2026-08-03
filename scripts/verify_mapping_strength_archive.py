#!/usr/bin/env python3
"""Verify the archived original-v1 MS chain from prompts to final labels."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analogy_agents.pipeline import load_split
from analogy_agents.original_mapping_strength_prompts import (
    mapping_extractor_prompt,
    ms_judge_prompt,
)
from analogy_agents.original_mapping_strength_schemas import (
    MappingAnalysis,
    MSJudgment,
)


CACHE_ROOT = (
    REPO
    / "artifacts/mapping_strength_evidence/cache/original_prompt"
    / "openai_gpt_oss_120b"
)
FROZEN_TEST = REPO / "artifacts/frozen/mapping_strength_predictions.csv"
VALIDATION_AUDIT = (
    REPO / "artifacts/frozen/original_pipeline_validation_scores.json"
)
EXPECTED_MODEL = "openai/gpt-oss-120b"
EXPECTED_PROMPT_VERSION = "v1"


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n{user}".encode("utf-8")).hexdigest()[:16]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"Missing MS archive file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_cache_metadata(
    payload: dict[str, Any],
    *,
    agent: str,
    expected_hash: str,
) -> None:
    if payload.get("agent") != agent:
        raise AssertionError(f"Unexpected agent: {payload.get('agent')!r}")
    if payload.get("model") != EXPECTED_MODEL:
        raise AssertionError(f"Unexpected model: {payload.get('model')!r}")
    if payload.get("prompt_version") != EXPECTED_PROMPT_VERSION:
        raise AssertionError(
            f"Unexpected prompt version: {payload.get('prompt_version')!r}"
        )
    if payload.get("prompt_hash") != expected_hash:
        raise AssertionError(
            f"Prompt hash mismatch for {agent}: "
            f"{payload.get('prompt_hash')!r} != {expected_hash!r}"
        )


def frozen_test_scores() -> dict[int, int]:
    with FROZEN_TEST.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["id"]): int(row["MS"])
            for row in csv.DictReader(handle)
        }


def validation_scores() -> list[int]:
    payload = read_json(VALIDATION_AUDIT)
    return [int(value) for value in payload["MS"]["predicted"]]


def verify_mapping_strength_archive() -> dict[str, Any]:
    test_expected = frozen_test_scores()
    validation_expected = validation_scores()
    mapping_matches = 0
    judge_matches = 0
    schema_validated = 0
    predictions: dict[str, list[int]] = {}

    for split, expected_rows in (("validation", 12), ("test", 62)):
        rows = load_split(REPO / "challenge-dataset", split)
        if len(rows) != expected_rows:
            raise AssertionError(
                f"{split} row count changed: {len(rows)} != {expected_rows}"
            )
        split_predictions: list[int] = []
        for row in rows:
            example_id = int(row["id"])
            cache_dir = CACHE_ROOT / split / f"{example_id:03d}"
            mapping_payload = read_json(cache_dir / "mapping_extractor.json")
            mapping_messages = mapping_extractor_prompt(
                row["target"],
                row["description"],
                row["analogy"],
            )
            mapping_expected_hash = prompt_hash(*mapping_messages)
            verify_cache_metadata(
                mapping_payload,
                agent="mapping_extractor",
                expected_hash=mapping_expected_hash,
            )
            mapping = MappingAnalysis.model_validate(mapping_payload["result"])
            mapping_matches += 1
            schema_validated += 1

            judge_payload = read_json(cache_dir / "ms_judge.json")
            judge_messages = ms_judge_prompt(
                row["target"],
                row["description"],
                row["analogy"],
                mapping.model_dump(),
            )
            judge_expected_hash = prompt_hash(*judge_messages)
            verify_cache_metadata(
                judge_payload,
                agent="ms_judge",
                expected_hash=judge_expected_hash,
            )
            judgment = MSJudgment.model_validate(judge_payload["result"])
            judge_matches += 1
            schema_validated += 1
            split_predictions.append(int(judgment.recommended_score))

        predictions[split] = split_predictions

    if predictions["validation"] != validation_expected:
        raise AssertionError("Archived validation MS predictions changed")
    test_from_cache = {
        example_id: score
        for example_id, score in enumerate(predictions["test"])
    }
    if test_from_cache != test_expected:
        raise AssertionError("Archived test MS predictions differ from frozen CSV")

    return {
        "examples": 74,
        "cache_files": 148,
        "model": EXPECTED_MODEL,
        "prompt_version": EXPECTED_PROMPT_VERSION,
        "mapping_prompt_hash_matches": mapping_matches,
        "ms_prompt_hash_matches": judge_matches,
        "schema_validations": schema_validated,
        "validation_predictions_match": True,
        "test_predictions_match_frozen_submission": True,
    }


def main() -> None:
    print(
        json.dumps(
            verify_mapping_strength_archive(),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
