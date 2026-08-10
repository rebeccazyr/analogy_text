#!/usr/bin/env python3
"""Generate label-safe text augmentation data without inventing new gold labels.

The script creates two datasets:

1. Gold-preserving rewrites of labeled validation analogies. A rewrite inherits
   the original labels only when every blind reviewer says that TCC, MS, and M
   are all semantically invariant.
2. Counterfactual pairs for new concepts. These receive only a relative
   relation (higher/lower/same) for one metric. They never receive synthetic
   absolute 0/1/2 labels.

Raw generations, every review, rejected rows, and quality summaries are kept
for auditability and safe resumption.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar

import pyarrow.parquet as pq
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from together import AsyncTogether


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONCEPTS = REPO_ROOT / "data_augmentation/concepts.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data_augmentation/pilot"
PROMPT_VERSION = "augmentation_relative_constraints_v1"

Metric = Literal["TCC", "MS", "M"]
Relation = Literal["a_higher", "b_higher", "same", "unclear"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InvarianceVariant(StrictModel):
    variant_id: Literal["P1", "P2", "P3"]
    transformation: Literal[
        "prose_restructure",
        "lexical_paraphrase",
        "compression_without_information_loss",
    ]
    analogy: str = Field(min_length=120)
    preservation_notes: list[str] = Field(min_length=2, max_length=8)
    content_added: Literal[False]
    content_removed: Literal[False]


class InvarianceGeneration(StrictModel):
    variants: list[InvarianceVariant] = Field(min_length=3, max_length=3)


class InvarianceAssessment(StrictModel):
    sample_id: str
    tcc_equivalent: bool
    ms_equivalent: bool
    m_equivalent: bool
    semantic_drift: list[str] = Field(max_length=8)
    artifact_flags: list[str] = Field(max_length=8)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=20)


class InvarianceReview(StrictModel):
    assessments: list[InvarianceAssessment] = Field(min_length=3, max_length=3)


class RelativePair(StrictModel):
    pair_id: Literal[
        "TCC_CORE_DELETION",
        "TCC_SURFACE_INVARIANCE",
        "MS_ROLE_MISMATCH",
        "MS_SURFACE_INVARIANCE",
        "M_LITERAL_TO_ADJACENT",
        "M_ADJACENT_TO_CROSS_DOMAIN",
    ]
    metric: Metric
    expected_relation: Relation
    analogy_a: str = Field(min_length=120)
    analogy_b: str = Field(min_length=120)
    controlled_operation: str = Field(min_length=20)
    non_target_invariance: list[str] = Field(min_length=2, max_length=6)


class RelativeGeneration(StrictModel):
    core_facets: list[str] = Field(min_length=2, max_length=6)
    pairs: list[RelativePair] = Field(min_length=6, max_length=6)


class RelativeAssessment(StrictModel):
    sample_id: str
    tcc_relation: Relation
    ms_relation: Relation
    m_relation: Relation
    artifact_flags: list[str] = Field(max_length=8)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=20)


class RelativeReview(StrictModel):
    assessments: list[RelativeAssessment] = Field(min_length=6, max_length=6)


T = TypeVar("T", bound=BaseModel)

PAIR_SPECS: dict[str, tuple[str, str]] = {
    "TCC_CORE_DELETION": ("TCC", "a_higher"),
    "TCC_SURFACE_INVARIANCE": ("TCC", "same"),
    "MS_ROLE_MISMATCH": ("MS", "a_higher"),
    "MS_SURFACE_INVARIANCE": ("MS", "same"),
    "M_LITERAL_TO_ADJACENT": ("M", "b_higher"),
    "M_ADJACENT_TO_CROSS_DOMAIN": ("M", "b_higher"),
}

REVIEW_STYLES = (
    "Evidence auditor: require explicit textual support and flag every semantic change.",
    "Adversarial falsifier: actively search for subtle label-changing drift or target leakage.",
    "Ordinal boundary specialist: focus on the exact 0/1/2 metric boundaries and metric separation.",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n{user}".encode("utf-8")).hexdigest()[:20]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def load_validation_rows() -> list[dict[str, Any]]:
    path = REPO_ROOT / "challenge-dataset/data/validation-00000-of-00001.parquet"
    return [
        {"id": index, **row}
        for index, row in enumerate(pq.read_table(path).to_pylist())
    ]


def parse_selection(value: str | None, available: list[str]) -> list[str]:
    if not value:
        return available
    requested = [item.strip() for item in value.split(",") if item.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Unknown IDs: {missing}")
    return [item for item in available if item in requested]


def invariance_generator_prompt(row: dict[str, Any]) -> tuple[str, str]:
    system = """ROLE: Conservative gold-preserving analogy rewriter.

Create exactly three surface variants of the supplied analogy. Preserve every
claim, mapping, causal direction, degree of coverage, literal source concept,
and source-to-target conceptual distance. You may change wording, formatting,
or remove verbal redundancy, but you must not add or remove any metric-relevant
information. Do not improve weaknesses. Do not repair errors. Do not make an
implicit relation more explicit. Do not mention scores or this task.

P1 must restructure prose, P2 must be a lexical paraphrase, and P3 must compress
only genuine redundancy. If compression would remove information, retain it.
Keep each result in the benchmark's natural 50-220 word range."""
    user = json.dumps(
        {
            "TARGET": row["target"],
            "DESCRIPTION": row["description"],
            "ORIGINAL_ANALOGY": row["analogy"],
        },
        ensure_ascii=False,
        indent=2,
    )
    return system, user


def invariance_review_prompt(
    row: dict[str, Any], variants: list[dict[str, str]], style: str
) -> tuple[str, str]:
    system = f"""ROLE: Blind semantic-invariance reviewer.

{style}

You are not told any gold labels or intended transformations. Compare each
candidate with the original and independently decide whether all evidence
relevant to each metric is unchanged:

- TCC: which important DESCRIPTION content is semantically covered.
- MS: source/target roles, relations, operations, and causal direction.
- M: literal source, literal applicability, native relation, and semantic type
  shifts of central roles.

Equivalent wording and formatting are allowed. Any addition, omission,
clarification, repair, weakened claim, changed source mechanism, or altered
mapping makes the affected metric non-equivalent. Be conservative."""
    user = json.dumps(
        {
            "TARGET": row["target"],
            "DESCRIPTION": row["description"],
            "ORIGINAL_ANALOGY": row["analogy"],
            "CANDIDATES": variants,
        },
        ensure_ascii=False,
        indent=2,
    )
    return system, user


def relative_generator_prompt(concept: dict[str, Any]) -> tuple[str, str]:
    system = """ROLE: Controlled counterfactual-pair author for analogy metrics.

Create exactly the six requested A/B pairs for one target. Do not assign
absolute 0/1/2 labels. Each pair must implement only its stated relative
relation while keeping the other two metrics invariant.

Metric definitions:
- TCC is important target-concept coverage, independent of mapping correctness.
- MS is structural mapping soundness, independent of coverage and domain distance.
- M is literal-source-to-target conceptual distance and role-type translation,
  independent of coverage and mapping quality.

Required pairs:
1. TCC_CORE_DELETION: A covers all core facets; B is the same strong mapping but
   omits exactly one core facet. TCC(A)>TCC(B); MS and M stay equal.
2. TCC_SURFACE_INVARIANCE: two surface forms with identical content and mapping.
3. MS_ROLE_MISMATCH: both cover the target equally and use the same source
   domain, but B introduces exactly one central role/function mismatch.
4. MS_SURFACE_INVARIANCE: two surface forms with identical mapping strength.
5. M_LITERAL_TO_ADJACENT: A is a real target instance/application; B is a
   non-literal but adjacent native relation. Both have full coverage and sound mapping.
6. M_ADJACENT_TO_CROSS_DOMAIN: A preserves a native relation with at most one
   central carrier/type shift; B uses a genuinely cross-domain projection with
   multiple central type changes. Both have full coverage and sound mapping.

Analogies must be standalone, natural, 50-220 words, and must not contain score
labels, editing commentary, or suspicious phrases copied across unrelated pairs."""
    user = json.dumps(
        {
            "TARGET": concept["target"],
            "DESCRIPTION": concept["description"],
            "DOMAIN": concept["domain"],
            "PAIR_CONTRACT": {
                pair_id: {"metric": metric, "expected_relation": relation}
                for pair_id, (metric, relation) in PAIR_SPECS.items()
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return system, user


def relative_review_prompt(
    concept: dict[str, Any], pairs: list[dict[str, str]], style: str
) -> tuple[str, str]:
    system = f"""ROLE: Blind pairwise analogy-metric reviewer.

{style}

You are not told the intended operation or relation. For every opaque A/B pair,
independently determine the relation for all three metrics: a_higher, b_higher,
same, or unclear.

- Higher TCC means more important DESCRIPTION content is covered.
- Higher MS means stronger, more internally consistent structural mappings.
- Higher M means greater conceptual distance and more semantic type translation.

Judge semantics, not length, fluency, creativity, or usefulness. Flag target
terminology unnaturally injected into a source, direct score hints, template
artifacts, or changes that confound multiple metrics."""
    user = json.dumps(
        {
            "TARGET": concept["target"],
            "DESCRIPTION": concept["description"],
            "PAIRS": pairs,
        },
        ensure_ascii=False,
        indent=2,
    )
    return system, user


class StructuredCaller:
    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path,
        max_concurrency: int,
        refresh: bool,
        reasoning_effort: str,
    ) -> None:
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise RuntimeError("TOGETHER_API_KEY is not available")
        self.client = AsyncTogether(api_key=api_key)
        self.model = model
        self.cache_dir = cache_dir
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.refresh = refresh
        self.reasoning_effort = reasoning_effort

    async def call(
        self,
        *,
        cache_key: str,
        agent_name: str,
        output_model: type[T],
        system: str,
        user: str,
        seed: int,
        max_tokens: int,
    ) -> T:
        cache_path = self.cache_dir / f"{cache_key}.json"
        current_hash = prompt_hash(system, user)
        if cache_path.exists() and not self.refresh:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("prompt_hash") == current_hash:
                return output_model.model_validate(payload["result"])

        schema = output_model.model_json_schema()
        schema_text = "\n\nReturn only JSON matching this schema:\n" + json.dumps(
            schema, ensure_ascii=False
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system + schema_text},
                            {"role": "user", "content": user},
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": agent_name, "schema": schema},
                        },
                        temperature=0.2,
                        top_p=1.0,
                        reasoning_effort=self.reasoning_effort,
                        max_tokens=max_tokens,
                        seed=seed + attempt,
                    )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("empty structured response")
                result = output_model.model_validate_json(content)
                usage = getattr(response, "usage", None)
                atomic_json(
                    cache_path,
                    {
                        "model": self.model,
                        "agent": agent_name,
                        "prompt_version": PROMPT_VERSION,
                        "prompt_hash": current_hash,
                        "created_at": utc_now(),
                        "usage": usage.model_dump() if hasattr(usage, "model_dump") else {},
                        "result": result.model_dump(),
                    },
                )
                return result
            except Exception as error:  # retries are deliberately bounded
                last_error = error
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"{agent_name} failed: {last_error}") from last_error


def validate_invariance_generation(result: InvarianceGeneration) -> None:
    expected = {"P1", "P2", "P3"}
    actual = {variant.variant_id for variant in result.variants}
    if actual != expected:
        raise ValueError(f"Expected variants {expected}, got {actual}")
    if len({variant.transformation for variant in result.variants}) != 3:
        raise ValueError("Invariance transformations must be unique")


def validate_relative_generation(result: RelativeGeneration) -> None:
    actual = {pair.pair_id for pair in result.pairs}
    if actual != set(PAIR_SPECS):
        raise ValueError(f"Relative pair IDs do not match contract: {actual}")
    for pair in result.pairs:
        metric, relation = PAIR_SPECS[pair.pair_id]
        if pair.metric != metric or pair.expected_relation != relation:
            raise ValueError(f"Contract mismatch for {pair.pair_id}")


def review_map(review: BaseModel, expected_ids: set[str]) -> dict[str, Any]:
    assessments = getattr(review, "assessments")
    mapped = {assessment.sample_id: assessment for assessment in assessments}
    if set(mapped) != expected_ids or len(mapped) != len(assessments):
        raise ValueError(
            f"Review IDs mismatch: expected={expected_ids}, actual={set(mapped)}"
        )
    return mapped


def invariance_status(assessments: list[InvarianceAssessment]) -> str:
    passes = [
        item.tcc_equivalent
        and item.ms_equivalent
        and item.m_equivalent
        and not item.artifact_flags
        for item in assessments
    ]
    if all(passes) and sum(item.confidence for item in assessments) / len(assessments) >= 0.8:
        return "accepted"
    if sum(passes) >= 2:
        return "review"
    return "rejected"


def relative_status(
    pair: RelativePair, assessments: list[RelativeAssessment]
) -> str:
    target_metric, expected_relation = PAIR_SPECS[pair.pair_id]
    target_field = f"{target_metric.lower()}_relation"
    other_fields = [
        f"{metric.lower()}_relation"
        for metric in ("TCC", "MS", "M")
        if metric != target_metric
    ]
    passes = []
    for item in assessments:
        target_ok = getattr(item, target_field) == expected_relation
        invariants_ok = all(getattr(item, field) == "same" for field in other_fields)
        passes.append(target_ok and invariants_ok and not item.artifact_flags)
    if all(passes) and sum(item.confidence for item in assessments) / len(assessments) >= 0.75:
        return "accepted"
    if sum(passes) >= 2:
        return "review"
    return "rejected"


async def process_invariance(
    caller: StructuredCaller,
    row: dict[str, Any],
    output_dir: Path,
    reviewer_count: int,
) -> list[dict[str, Any]]:
    entity = f"validation_{row['id']:03d}"
    system, user = invariance_generator_prompt(row)
    generated = await caller.call(
        cache_key=f"invariance/{entity}/generator",
        agent_name="invariance_generator",
        output_model=InvarianceGeneration,
        system=system,
        user=user,
        seed=42 + row["id"],
        max_tokens=6500,
    )
    validate_invariance_generation(generated)
    variants = sorted(generated.variants, key=lambda item: item.variant_id)
    opaque = [
        {"sample_id": f"I{row['id']:02d}{index}", "analogy": variant.analogy}
        for index, variant in enumerate(variants, start=1)
    ]

    review_tasks = []
    for reviewer_index, style in enumerate(REVIEW_STYLES[:reviewer_count]):
        shuffled = list(opaque)
        random.Random(stable_int(f"{entity}:{reviewer_index}")).shuffle(shuffled)
        review_system, review_user = invariance_review_prompt(row, shuffled, style)
        review_tasks.append(
            caller.call(
                cache_key=f"invariance/{entity}/reviewer_{reviewer_index + 1}",
                agent_name="invariance_reviewer",
                output_model=InvarianceReview,
                system=review_system,
                user=review_user,
                seed=101 + reviewer_index * 997 + row["id"],
                max_tokens=4000,
            )
        )
    reviews = await asyncio.gather(*review_tasks)
    review_maps = [review_map(review, {item["sample_id"] for item in opaque}) for review in reviews]

    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants, start=1):
        sample_id = f"I{row['id']:02d}{index}"
        assessments = [mapped[sample_id] for mapped in review_maps]
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_kind": "gold_preserving_invariance",
                "source_split": "validation",
                "source_id": row["id"],
                "target": row["target"],
                "description": row["description"],
                "original_analogy": row["analogy"],
                "analogy": variant.analogy,
                "transformation": variant.transformation,
                "labels": {"TCC": row["TCC"], "MS": row["MS"], "M": row["M"]},
                "label_source": "inherited_original_gold_after_unanimous_invariance",
                "status": invariance_status(assessments),
                "reviews": [item.model_dump() for item in assessments],
            }
        )
    atomic_json(
        output_dir / f"raw/invariance/{entity}.json",
        {"input": row, "generation": generated.model_dump(), "reviews": [r.model_dump() for r in reviews]},
    )
    print(f"invariance {entity}: {Counter(item['status'] for item in rows)}", flush=True)
    return rows


async def process_relative(
    caller: StructuredCaller,
    concept: dict[str, Any],
    output_dir: Path,
    reviewer_count: int,
) -> list[dict[str, Any]]:
    concept_id = concept["concept_id"]
    system, user = relative_generator_prompt(concept)
    generated = await caller.call(
        cache_key=f"relative/{concept_id}/generator",
        agent_name="relative_pair_generator",
        output_model=RelativeGeneration,
        system=system,
        user=user,
        seed=4242 + stable_int(concept_id) % 10000,
        max_tokens=8000,
    )
    validate_relative_generation(generated)
    pairs = sorted(generated.pairs, key=lambda item: item.pair_id)
    opaque = [
        {
            "sample_id": f"R{concept_id[1:]}{index}",
            "analogy_a": pair.analogy_a,
            "analogy_b": pair.analogy_b,
        }
        for index, pair in enumerate(pairs, start=1)
    ]

    review_tasks = []
    for reviewer_index, style in enumerate(REVIEW_STYLES[:reviewer_count]):
        shuffled = list(opaque)
        random.Random(stable_int(f"{concept_id}:{reviewer_index}")).shuffle(shuffled)
        review_system, review_user = relative_review_prompt(concept, shuffled, style)
        review_tasks.append(
            caller.call(
                cache_key=f"relative/{concept_id}/reviewer_{reviewer_index + 1}",
                agent_name="relative_pair_reviewer",
                output_model=RelativeReview,
                system=review_system,
                user=review_user,
                seed=202 + reviewer_index * 991 + stable_int(concept_id) % 1000,
                max_tokens=5000,
            )
        )
    reviews = await asyncio.gather(*review_tasks)
    review_maps = [review_map(review, {item["sample_id"] for item in opaque}) for review in reviews]

    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        sample_id = f"R{concept_id[1:]}{index}"
        assessments = [mapped[sample_id] for mapped in review_maps]
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_kind": "relative_counterfactual_pair",
                "concept_id": concept_id,
                "split": concept["split"],
                "domain": concept["domain"],
                "target": concept["target"],
                "description": concept["description"],
                "metric": pair.metric,
                "relation": pair.expected_relation,
                "analogy_a": pair.analogy_a,
                "analogy_b": pair.analogy_b,
                "controlled_operation": pair.controlled_operation,
                "absolute_labels": None,
                "label_source": "operation_defined_relative_constraint",
                "status": relative_status(pair, assessments),
                "reviews": [item.model_dump() for item in assessments],
            }
        )
    atomic_json(
        output_dir / f"raw/relative/{concept_id}.json",
        {"input": concept, "generation": generated.model_dump(), "reviews": [r.model_dump() for r in reviews]},
    )
    print(f"relative {concept_id}: {Counter(item['status'] for item in rows)}", flush=True)
    return rows


def build_report(invariance: list[dict[str, Any]], relative: list[dict[str, Any]]) -> dict[str, Any]:
    def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(row["status"] for row in rows).items()))

    relative_by_metric = {
        metric: status_counts([row for row in relative if row["metric"] == metric])
        for metric in ("TCC", "MS", "M")
    }
    return {
        "prompt_version": PROMPT_VERSION,
        "created_at": utc_now(),
        "policy": {
            "new_absolute_pseudo_labels": False,
            "invariance_acceptance": "all reviewers preserve TCC/MS/M, no artifacts, mean confidence >= 0.80",
            "relative_acceptance": "all reviewers confirm target direction and both non-target invariants, no artifacts, mean confidence >= 0.75",
        },
        "invariance": {"total": len(invariance), "status": status_counts(invariance)},
        "relative": {"total": len(relative), "status": status_counts(relative), "by_metric": relative_by_metric},
    }


def export_outputs(
    output_dir: Path,
    invariance: list[dict[str, Any]],
    relative: list[dict[str, Any]],
) -> None:
    invariance = sorted(invariance, key=lambda row: row["sample_id"])
    relative = sorted(relative, key=lambda row: row["sample_id"])
    write_jsonl(output_dir / "reviewed/invariance_all.jsonl", invariance)
    write_jsonl(output_dir / "reviewed/relative_all.jsonl", relative)
    write_jsonl(
        output_dir / "accepted/gold_invariance.jsonl",
        [row for row in invariance if row["status"] == "accepted"],
    )
    accepted_relative = [row for row in relative if row["status"] == "accepted"]
    write_jsonl(output_dir / "accepted/relative_pairs.jsonl", accepted_relative)
    for split in ("train", "validation", "test"):
        write_jsonl(
            output_dir / f"accepted/relative_pairs_{split}.jsonl",
            [row for row in accepted_relative if row["split"] == split],
        )
    atomic_json(output_dir / "quality_report.json", build_report(invariance, relative))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "invariance", "relative"], default="all")
    parser.add_argument("--concepts", type=Path, default=DEFAULT_CONCEPTS)
    parser.add_argument("--concept-ids", help="Comma-separated concept IDs")
    parser.add_argument("--validation-ids", help="Comma-separated validation IDs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--reviewers", type=int, choices=[2, 3], default=3)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv(REPO_ROOT.parent / ".env")
    load_dotenv(REPO_ROOT / ".env")
    concepts = load_jsonl(args.concepts)
    validation = load_validation_rows()

    concept_ids = parse_selection(
        args.concept_ids, [str(row["concept_id"]) for row in concepts]
    )
    validation_ids = parse_selection(
        args.validation_ids, [str(row["id"]) for row in validation]
    )
    concepts = [row for row in concepts if row["concept_id"] in concept_ids]
    validation = [row for row in validation if str(row["id"]) in validation_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    caller = StructuredCaller(
        model=args.model,
        cache_dir=args.output_dir / "cache",
        max_concurrency=args.max_concurrency,
        refresh=args.refresh,
        reasoning_effort=args.reasoning_effort,
    )

    invariance_rows: list[dict[str, Any]] = []
    relative_rows: list[dict[str, Any]] = []
    if args.mode in {"all", "invariance"}:
        groups = await asyncio.gather(
            *(process_invariance(caller, row, args.output_dir, args.reviewers) for row in validation)
        )
        invariance_rows = [item for group in groups for item in group]
    elif (args.output_dir / "reviewed/invariance_all.jsonl").exists():
        invariance_rows = load_jsonl(args.output_dir / "reviewed/invariance_all.jsonl")

    if args.mode in {"all", "relative"}:
        groups = await asyncio.gather(
            *(process_relative(caller, concept, args.output_dir, args.reviewers) for concept in concepts)
        )
        relative_rows = [item for group in groups for item in group]
    elif (args.output_dir / "reviewed/relative_all.jsonl").exists():
        relative_rows = load_jsonl(args.output_dir / "reviewed/relative_all.jsonl")

    export_outputs(args.output_dir, invariance_rows, relative_rows)
    print(json.dumps(build_report(invariance_rows, relative_rows), ensure_ascii=False, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
