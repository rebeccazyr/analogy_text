#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from analogy_agents.metaphoricity_taxonomy import load_m_taxonomy
from analogy_agents.pipeline import (
    PipelineConfig,
    SixAgentPipeline,
    load_archived_v1_tcc,
    load_split,
    score_m_validation,
    score_ms_validation,
    score_validation,
    score_tcc_validation,
    write_m_outputs,
    write_ms_outputs,
    write_run_outputs,
    write_tcc_outputs,
)
from analogy_agents.prompts import (
    concept_decomposer_prompt,
    domain_classifier_prompt,
    literal_instance_prompt,
    m_taxonomy_literal_prompt,
    m_taxonomy_source_analysis_prompt,
    m_operation_extractor_prompt,
    m_native_source_frame_prompt,
    m_relation_source_frame_prompt,
    m_relation_target_frame_prompt,
    m_two_gate_source_frame_prompt,
    m_two_gate_target_frame_prompt,
    mapping_extractor_prompt,
    topic_importance_prompt,
)
from analogy_agents.original_target_coverage_prompts import (
    concept_decomposer_prompt as v1_concept_decomposer_prompt,
)
from analogy_agents.original_mapping_strength_prompts import (
    mapping_extractor_prompt as original_mapping_extractor_prompt,
)


def parse_ids(value: str | None, row_count: int) -> list[int]:
    if not value:
        return list(range(row_count))
    selected: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            selected.update(range(int(start_text), int(end_text) + 1))
        else:
            selected.add(int(chunk))
    invalid = sorted(index for index in selected if index < 0 or index >= row_count)
    if invalid:
        raise ValueError(f"IDs outside split range: {invalid}")
    return sorted(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the six-agent text analogy evaluation pipeline."
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "target-coverage",
            "mapping-strength",
            "metaphoricity",
            "tcc",
            "tcc-importance",
            "tcc-v1-conservative",
            "tcc-v1-prompt-conservative",
            "tcc-v1-facet-conservative",
            "m",
            "m-taxonomy",
            "m-taxonomy-agent",
            "m-conceptual-distance",
            "m-conceptual-distance-critic",
            "m-operation-audit",
            "m-native-scope-audit",
            "m-two-gate",
            "m-relation-gate",
        ],
        default="all",
        help=(
            "Run all agents, the legacy TCC path, the TCC path with a "
            "topic-importance judge, the archived-v1 conservative correction, "
            "the correction executed on the exact original v1 prompts, the "
            "facet-level coverage-audit experiment, the hash-verified original "
            "v1 Mapping Strength path, the v7 metaphoricity path, "
            "the fixed-rule taxonomy path, or the taxonomy path with an "
            "independent final M agent, or the overall conceptual-distance "
            "agent path, or that path with a native-neighborhood critic and "
            "independent adjudicator, or the operation-first balanced audit."
            " The native-scope audit retains the complete source concept and "
            "uses a contrastive final agent."
            " The two-gate path independently frames target and source before "
            "applying literal and native-neighborhood boundaries."
            " The relation-gate path replaces role counting with native "
            "relation identity and carrier constraints."
        ),
    )
    parser.add_argument(
        "--v1-archive-dir",
        type=Path,
        default=Path("archives/first_version_v1_20260729"),
        help="Immutable v1 archive used by --mode tcc-v1-conservative.",
    )
    parser.add_argument(
        "--ids",
        help="Comma-separated IDs or ranges, e.g. 0,2,4-7. Defaults to all rows.",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("challenge-dataset")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/gpt_oss_120b_v1")
    )
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument(
        "--sample-concurrency",
        type=int,
        default=1,
        help="Number of examples evaluated concurrently. API calls remain capped by --max-concurrency.",
    )
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".agent_cache"),
        help=(
            "Cache root for structured agent calls. Use a distinct directory "
            "for controlled model-parameter ablations."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and show the first-stage plan without calling Together.",
    )
    return parser


async def run_examples(
    pipeline: SixAgentPipeline,
    rows_by_id: dict[int, dict],
    selected_ids: list[int],
    split: str,
    output_dir: Path,
    sample_concurrency: int,
) -> list[dict]:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_semaphore = asyncio.Semaphore(sample_concurrency)

    async def run_one(position: int, example_id: int) -> dict:
        async with sample_semaphore:
            row = rows_by_id[example_id]
            print(
                f"[{position}/{len(selected_ids)}] id={example_id} target={row['target']}",
                flush=True,
            )
            result = await pipeline.evaluate(row, split)
            sample_path = sample_dir / f"{example_id:03d}.json"
            sample_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            prediction = result["prediction"]
            print(
                f"  id={example_id} -> "
                f"TCC={prediction['TCC']} MS={prediction['MS']} M={prediction['M']}",
                flush=True,
            )
            return result

    tasks = [
        asyncio.create_task(run_one(position, example_id))
        for position, example_id in enumerate(selected_ids, start=1)
    ]
    return list(await asyncio.gather(*tasks))


def concept_group_key(row: dict) -> tuple[str, str]:
    """Identical targets and descriptions must share one decomposition."""
    return row["target"], row["description"]


def archived_decomposition_group_key(
    row: dict,
    decomposition: dict,
) -> tuple[str, str, str]:
    """Share one importance judgment only for identical archived v1 topics."""
    return (
        row["target"],
        row["description"],
        json.dumps(decomposition, sort_keys=True),
    )


async def run_tcc_examples(
    pipeline: SixAgentPipeline,
    rows_by_id: dict[int, dict],
    selected_ids: list[int],
    split: str,
    output_dir: Path,
    sample_concurrency: int,
    apply_topic_importance: bool,
) -> list[dict]:
    sample_dir = output_dir / "target_coverage_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[int]] = {}
    for example_id in selected_ids:
        key = concept_group_key(rows_by_id[example_id])
        groups.setdefault(key, []).append(example_id)

    print(
        f"TCC-only: {len(selected_ids)} examples, "
        f"{len(groups)} unique target+description groups",
        flush=True,
    )
    sample_semaphore = asyncio.Semaphore(sample_concurrency)

    async def prepare_group(key: tuple[str, str], ids: list[int]):
        async with sample_semaphore:
            representative = rows_by_id[ids[0]]
            print(
                f"[decompose] ids={ids} target={representative['target']}",
                flush=True,
            )
            decomposition = await pipeline.decompose(representative, split)
            topic_importance = None
            if apply_topic_importance:
                print(
                    f"[topic-importance] ids={ids} "
                    f"target={representative['target']}",
                    flush=True,
                )
                topic_importance = await pipeline.judge_topic_importance(
                    representative,
                    split,
                    decomposition,
                )
            return key, (decomposition, topic_importance)

    prepared_groups = dict(
        await asyncio.gather(
            *[
                asyncio.create_task(prepare_group(key, ids))
                for key, ids in groups.items()
            ]
        )
    )

    async def judge_one(position: int, example_id: int) -> dict:
        async with sample_semaphore:
            row = rows_by_id[example_id]
            decomposition, topic_importance = prepared_groups[
                concept_group_key(row)
            ]
            result = await pipeline.evaluate_tcc(
                row,
                split,
                decomposition,
                apply_topic_importance=apply_topic_importance,
                topic_importance=topic_importance,
            )
            sample_path = sample_dir / f"{example_id:03d}.json"
            sample_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[tcc {position}/{len(selected_ids)}] id={example_id} "
                f"ratio={result['coverage_ratio']:.3f} "
                f"TCC={result['prediction']['TCC']}"
                + (
                    " retained_topics="
                    f"{len(result['topic_importance_policy']['retained_topic_ids'])}"
                    if apply_topic_importance
                    else ""
                ),
                flush=True,
            )
            return result

    return list(
        await asyncio.gather(
            *[
                asyncio.create_task(judge_one(position, example_id))
                for position, example_id in enumerate(
                    selected_ids, start=1
                )
            ]
        )
    )


async def run_v1_conservative_tcc_examples(
    pipeline: SixAgentPipeline,
    rows_by_id: dict[int, dict],
    selected_ids: list[int],
    split: str,
    output_dir: Path,
    sample_concurrency: int,
    archive_dir: Path,
) -> list[dict]:
    """Run a description-only automatic correction over immutable v1."""
    sample_dir = output_dir / "target_coverage_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_semaphore = asyncio.Semaphore(sample_concurrency)

    archived_by_id = {
        example_id: load_archived_v1_tcc(archive_dir, split, example_id)
        for example_id in selected_ids
    }
    groups: dict[tuple[str, str, str], list[int]] = {}
    for example_id in selected_ids:
        row = rows_by_id[example_id]
        decomposition = archived_by_id[example_id][0]
        key = archived_decomposition_group_key(
            row,
            decomposition.model_dump(),
        )
        groups.setdefault(key, []).append(example_id)

    print(
        f"Archived-v1 conservative TCC: {len(selected_ids)} examples, "
        f"{len(groups)} unique v1 decompositions",
        flush=True,
    )

    async def prepare_group(
        key: tuple[str, str, str],
        ids: list[int],
    ):
        async with sample_semaphore:
            representative_id = ids[0]
            row = rows_by_id[representative_id]
            decomposition = archived_by_id[representative_id][0]
            print(
                f"[v1 topic-importance] ids={ids} target={row['target']}",
                flush=True,
            )
            judgment = await pipeline.judge_topic_importance(
                row,
                split,
                decomposition,
            )
            return key, judgment

    importance_by_group = dict(
        await asyncio.gather(
            *[
                asyncio.create_task(prepare_group(key, ids))
                for key, ids in groups.items()
            ]
        )
    )

    async def correct_one(position: int, example_id: int) -> dict:
        async with sample_semaphore:
            row = rows_by_id[example_id]
            (
                decomposition,
                archived_tcc,
                original_score,
                archived_tcc_result,
            ) = archived_by_id[example_id]
            key = archived_decomposition_group_key(
                row,
                decomposition.model_dump(),
            )
            result = await pipeline.evaluate_archived_v1_tcc_with_importance(
                row,
                split,
                decomposition,
                archived_tcc,
                original_score,
                archived_tcc_result,
                importance_by_group[key],
            )
            sample_path = sample_dir / f"{example_id:03d}.json"
            sample_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            policy = result["v1_conservative_policy"]
            print(
                f"[v1 correction {position}/{len(selected_ids)}] "
                f"id={example_id} {original_score}"
                f"->{result['prediction']['TCC']} "
                f"filtered={policy['filtered_topic_ids']} "
                f"blocking={policy['blocking_topic_ids']}",
                flush=True,
            )
            return result

    return list(
        await asyncio.gather(
            *[
                asyncio.create_task(correct_one(position, example_id))
                for position, example_id in enumerate(
                    selected_ids, start=1
                )
            ]
        )
    )


async def run_exact_v1_conservative_tcc_examples(
    pipeline: SixAgentPipeline,
    rows_by_id: dict[int, dict],
    selected_ids: list[int],
    split: str,
    output_dir: Path,
    sample_concurrency: int,
    facet_audit: bool = False,
) -> list[dict]:
    """Execute exact v1 prompts before applying the general correction."""
    sample_dir = output_dir / "target_coverage_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_semaphore = asyncio.Semaphore(sample_concurrency)

    print(
        "Exact-v1-prompt "
        + ("facet-audit" if facet_audit else "conservative")
        + f" TCC: {len(selected_ids)} examples",
        flush=True,
    )

    async def run_one(position: int, example_id: int) -> dict:
        async with sample_semaphore:
            row = rows_by_id[example_id]
            print(
                f"[exact v1 {position}/{len(selected_ids)}] "
                f"id={example_id} target={row['target']}",
                flush=True,
            )
            if facet_audit:
                result = (
                    await pipeline.evaluate_exact_v1_tcc_with_facet_audit(
                        row,
                        split,
                    )
                )
                policy_key = "v1_facet_conservative_policy"
            else:
                result = await pipeline.evaluate_exact_v1_tcc_with_importance(
                    row,
                    split,
                )
                policy_key = "v1_conservative_policy"
            sample_path = sample_dir / f"{example_id:03d}.json"
            sample_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            policy = result[policy_key]
            print(
                f"  id={example_id} "
                f"{policy['original_v1_score']}"
                f"->{result['prediction']['TCC']} "
                f"filtered={policy['filtered_topic_ids']} "
                f"audited={policy['audited_upgraded_topic_ids']} "
                f"blocking={policy['blocking_topic_ids']}",
                flush=True,
            )
            return result

    return list(
        await asyncio.gather(
            *[
                asyncio.create_task(run_one(position, example_id))
                for position, example_id in enumerate(
                    selected_ids,
                    start=1,
                )
            ]
        )
    )


async def run_original_mapping_strength_examples(
    pipeline: SixAgentPipeline,
    rows_by_id: dict[int, dict],
    selected_ids: list[int],
    split: str,
    output_dir: Path,
    sample_concurrency: int,
) -> list[dict]:
    """Execute the recovered original-v1 MappingExtractor and MSJudge."""
    sample_dir = output_dir / "mapping_strength_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_semaphore = asyncio.Semaphore(sample_concurrency)

    print(
        f"Original Mapping Strength: {len(selected_ids)} examples",
        flush=True,
    )

    async def run_one(position: int, example_id: int) -> dict:
        async with sample_semaphore:
            row = rows_by_id[example_id]
            print(
                f"[mapping strength {position}/{len(selected_ids)}] "
                f"id={example_id} target={row['target']}",
                flush=True,
            )
            result = await pipeline.evaluate_original_mapping_strength(
                row,
                split,
            )
            sample_path = sample_dir / f"{example_id:03d}.json"
            sample_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"  id={example_id} -> MS={result['prediction']['MS']} "
                f"latent={result['latent_scores']['MS']:.3f}",
                flush=True,
            )
            return result

    return list(
        await asyncio.gather(
            *[
                asyncio.create_task(run_one(position, example_id))
                for position, example_id in enumerate(
                    selected_ids,
                    start=1,
                )
            ]
        )
    )


async def run_m_examples(
    pipeline: SixAgentPipeline,
    rows_by_id: dict[int, dict],
    selected_ids: list[int],
    split: str,
    output_dir: Path,
    sample_concurrency: int,
    use_taxonomy: bool = False,
    use_taxonomy_agent: bool = False,
    use_conceptual_distance: bool = False,
    use_conceptual_distance_critic: bool = False,
    use_operation_audit: bool = False,
    use_native_scope_audit: bool = False,
    use_two_gate: bool = False,
    use_relation_gate: bool = False,
) -> list[dict]:
    sample_dir = output_dir / "metaphoricity_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_semaphore = asyncio.Semaphore(sample_concurrency)

    async def judge_one(position: int, example_id: int) -> dict:
        async with sample_semaphore:
            row = rows_by_id[example_id]
            print(
                f"[m {position}/{len(selected_ids)}] "
                f"id={example_id} target={row['target']}",
                flush=True,
            )
            if use_relation_gate:
                evaluator = pipeline.evaluate_m_relation_gate
            elif use_two_gate:
                evaluator = pipeline.evaluate_m_two_gate
            elif use_native_scope_audit:
                evaluator = pipeline.evaluate_m_native_scope_audit
            elif use_operation_audit:
                evaluator = pipeline.evaluate_m_operation_audit
            elif use_conceptual_distance_critic:
                evaluator = pipeline.evaluate_m_conceptual_distance_critic
            elif use_conceptual_distance:
                evaluator = pipeline.evaluate_m_conceptual_distance
            elif use_taxonomy_agent:
                evaluator = pipeline.evaluate_m_taxonomy_agent
            elif use_taxonomy:
                evaluator = pipeline.evaluate_m_taxonomy
            else:
                evaluator = pipeline.evaluate_m
            result = await evaluator(row, split)
            sample_path = sample_dir / f"{example_id:03d}.json"
            sample_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"  id={example_id} -> M={result['prediction']['M']} "
                f"latent={result['latent_scores']['M']:.3f}",
                flush=True,
            )
            return result

    return list(
        await asyncio.gather(
            *[
                asyncio.create_task(judge_one(position, example_id))
                for position, example_id in enumerate(
                    selected_ids, start=1
                )
            ]
        )
    )


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    rows = load_split(args.dataset_dir, args.split)
    selected_ids = parse_ids(args.ids, len(rows))
    rows_by_id = {row["id"]: row for row in rows}

    if args.dry_run:
        first = rows_by_id[selected_ids[0]]
        if args.mode in {
            "tcc",
            "tcc-importance",
            "tcc-v1-conservative",
            "tcc-v1-prompt-conservative",
            "tcc-v1-facet-conservative",
            "target-coverage",
        }:
            if args.mode == "tcc-v1-conservative":
                archived_decomposition = load_archived_v1_tcc(
                    args.v1_archive_dir,
                    args.split,
                    selected_ids[0],
                )[0]
                decomposition_payload = archived_decomposition.model_dump()
            elif args.mode in {
                "tcc-v1-prompt-conservative",
                "tcc-v1-facet-conservative",
                "target-coverage",
            }:
                decomposition_payload = {
                    "target_summary": "<exact v1 ConceptDecomposer output>",
                    "topics": [
                        {
                            "topic_id": "T1",
                            "topic": "<candidate topic>",
                            "importance": "core",
                        },
                        {
                            "topic_id": "T2",
                            "topic": "<candidate topic>",
                            "importance": "supporting",
                        },
                    ],
                }
            else:
                decomposition_payload = {
                    "target_summary": "<ConceptDecomposer output>",
                    "topics": [
                        {
                            "topic_id": "T1",
                            "topic": "<candidate topic>",
                            "importance": "core",
                        }
                    ],
                }
            prompts = {
                "concept_decomposer": (
                    ("<loaded from archived v1>", "")
                    if args.mode == "tcc-v1-conservative"
                    else (
                        v1_concept_decomposer_prompt(
                            first["target"],
                            first["description"],
                        )
                        if args.mode in {
                            "tcc-v1-prompt-conservative",
                            "tcc-v1-facet-conservative",
                            "target-coverage",
                        }
                        else concept_decomposer_prompt(
                            first["target"],
                            first["description"],
                        )
                    )
                )
            }
            if args.mode in {
                "tcc-importance",
                "tcc-v1-conservative",
                "tcc-v1-prompt-conservative",
                "tcc-v1-facet-conservative",
                "target-coverage",
            }:
                prompts["topic_importance_judge"] = topic_importance_prompt(
                    first["target"],
                    first["description"],
                    decomposition_payload,
                )
        elif args.mode == "mapping-strength":
            prompts = {
                "original_mapping_extractor": original_mapping_extractor_prompt(
                    first["target"],
                    first["description"],
                    first["analogy"],
                ),
                "original_mapping_strength_judge": (
                    "MSJudge runs after the exact-v1 mapping result.",
                    "All 74 archived prompt hashes are verified.",
                ),
            }
        elif args.mode == "m-relation-gate":
            prompts = {
                "m_relation_target_signature": m_relation_target_frame_prompt(
                    first["target"],
                    first["description"],
                ),
                "m_relation_blind_source_signature": (
                    m_relation_source_frame_prompt(first["analogy"])
                ),
                "m_relation_literal_boundary": (
                    "Strict M=0 denotation gate runs after relation framing.",
                    "Validation anchors are physically leave-one-out.",
                ),
                "m_relation_identity_boundary": (
                    "M=1/M=2 relation identity and carrier gate runs last.",
                    "Python applies a frozen terminology and gloss-removal rule.",
                ),
            }
        elif args.mode == "m-two-gate":
            prompts = {
                "m_two_gate_target_frame": m_two_gate_target_frame_prompt(
                    first["target"],
                    first["description"],
                ),
                "m_two_gate_blind_source_frame": (
                    m_two_gate_source_frame_prompt(first["analogy"])
                ),
                "m_two_gate_literal_boundary": (
                    "Strict M=0 denotation gate runs after independent framing.",
                    "It receives physically leave-one-out anchors on validation.",
                ),
                "m_two_gate_native_boundary": (
                    "M=1/M=2 minimum-transform gate runs after framing.",
                    "Python applies the fixed v7 ordinal rule to both gates.",
                ),
            }
        elif args.mode == "m-native-scope-audit":
            prompts = {
                "m_native_source_frame": m_native_source_frame_prompt(
                    first["target"],
                    first["description"],
                    first["analogy"],
                ),
                "m_literal_scope_auditor": (
                    "Strict denotation auditor runs after source framing.",
                    "It provides evidence but no M score.",
                ),
                "m_native_neighborhood_auditor": (
                    "Native-neighborhood auditor runs after source framing.",
                    "It provides evidence but no M score.",
                ),
                "m_native_scope_adjudicator": (
                    "Contrastive native-scope adjudicator runs last.",
                    "It is the only component that assigns M.",
                ),
            }
        elif args.mode == "m-operation-audit":
            prompts = {
                "m_operation_extractor": m_operation_extractor_prompt(
                    first["target"],
                    first["description"],
                    first["analogy"],
                ),
                "m_literal_applicability_advocate": (
                    "Literal advocate runs after operation extraction.",
                    "It provides evidence but no M score.",
                ),
                "m_native_relation_role_critic": (
                    "Relation and role critic runs after operation extraction.",
                    "It provides evidence but no M score.",
                ),
                "m_operation_adjudicator": (
                    "Balanced operation-first adjudicator runs last.",
                    "It is the only component that assigns M.",
                ),
            }
        elif args.mode in {
            "m",
            "metaphoricity",
            "m-taxonomy",
            "m-taxonomy-agent",
            "m-conceptual-distance",
            "m-conceptual-distance-critic",
        }:
            taxonomy_mode = args.mode in {
                "m-taxonomy",
                "m-taxonomy-agent",
                "m-conceptual-distance",
                "m-conceptual-distance-critic",
            }
            prompts = {
                "source_domain_classifier": (
                    m_taxonomy_source_analysis_prompt
                    if taxonomy_mode
                    else domain_classifier_prompt
                )(first["target"], first["description"], first["analogy"]),
                "literal_instance_judge": (
                    m_taxonomy_literal_prompt
                    if taxonomy_mode
                    else literal_instance_prompt
                )(first["target"], first["description"], first["analogy"]),
            }
            if args.mode in {
                "m-taxonomy",
                "m-taxonomy-agent",
                "m-conceptual-distance",
                "m-conceptual-distance-critic",
            }:
                taxonomy = load_m_taxonomy()
                prompts["m_taxonomy_mapper"] = (
                    "Taxonomy mapper runs after first-stage evidence.",
                    json.dumps(
                        {
                            "taxonomy_version": taxonomy.version,
                            "target_profile": taxonomy.target_profile(
                                first["target"]
                            ),
                            "domain_nodes": len(taxonomy.domains.nodes),
                            "entity_nodes": len(taxonomy.entities.nodes),
                            "relation_nodes": len(taxonomy.relations.nodes),
                        },
                        ensure_ascii=False,
                    ),
                )
                if args.mode == "m-taxonomy-agent":
                    prompts["m_taxonomy_final_judge"] = (
                        "Final judge runs after the three-axis taxonomy mapping.",
                        "Uses domain, relation, and entity-role evidence without "
                        "a deterministic LCA cutoff.",
                    )
                if args.mode == "m-conceptual-distance":
                    prompts["m_conceptual_distance_judge"] = (
                        "Overall conceptual-distance judge runs last.",
                        "Taxonomy axes are supporting evidence, not separate "
                        "criteria or votes.",
                    )
                if args.mode == "m-conceptual-distance-critic":
                    prompts["m_native_neighborhood_critic"] = (
                        "Critic audits whether proximity is native or analogy-constructed.",
                        "It does not assign the final score.",
                    )
                    prompts["m_conceptual_distance_adjudicator"] = (
                        "Independent adjudicator runs after the critique.",
                        "It returns the final overall conceptual-distance score.",
                    )
        else:
            prompts = {
                "concept_decomposer": concept_decomposer_prompt(
                    first["target"], first["description"]
                ),
                "mapping_extractor": mapping_extractor_prompt(
                    first["target"], first["description"], first["analogy"]
                ),
                "source_domain_classifier": domain_classifier_prompt(
                    first["target"], first["description"], first["analogy"]
                ),
                "literal_instance_judge": literal_instance_prompt(
                    first["target"], first["description"], first["analogy"]
                ),
            }
        print(
            json.dumps(
                {
                    "split": args.split,
                    "row_count": len(rows),
                    "selected_ids": selected_ids,
                    "first_target": first["target"],
                    "first_stage_agents": list(prompts),
                    "prompt_characters": {
                        name: len(system) + len(user)
                        for name, (system, user) in prompts.items()
                    },
                    "model": args.model,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    config = PipelineConfig(
        model=args.model,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
    )
    pipeline = SixAgentPipeline(config)
    if args.mode in {
        "tcc-v1-prompt-conservative",
        "tcc-v1-facet-conservative",
        "target-coverage",
    }:
        results = asyncio.run(
            run_exact_v1_conservative_tcc_examples(
                pipeline,
                rows_by_id,
                selected_ids,
                args.split,
                args.output_dir,
                args.sample_concurrency,
                args.mode in {
                    "tcc-v1-facet-conservative",
                    "target-coverage",
                },
            )
        )
        details_path, predictions_path = write_tcc_outputs(
            results,
            args.output_dir,
            args.split,
        )
    elif args.mode == "tcc-v1-conservative":
        results = asyncio.run(
            run_v1_conservative_tcc_examples(
                pipeline,
                rows_by_id,
                selected_ids,
                args.split,
                args.output_dir,
                args.sample_concurrency,
                args.v1_archive_dir,
            )
        )
        details_path, predictions_path = write_tcc_outputs(
            results, args.output_dir, args.split
        )
    elif args.mode in {"tcc", "tcc-importance"}:
        results = asyncio.run(
            run_tcc_examples(
                pipeline,
                rows_by_id,
                selected_ids,
                args.split,
                args.output_dir,
                args.sample_concurrency,
                args.mode == "tcc-importance",
            )
        )
        details_path, predictions_path = write_tcc_outputs(
            results, args.output_dir, args.split
        )
    elif args.mode == "mapping-strength":
        results = asyncio.run(
            run_original_mapping_strength_examples(
                pipeline,
                rows_by_id,
                selected_ids,
                args.split,
                args.output_dir,
                args.sample_concurrency,
            )
        )
        details_path, predictions_path = write_ms_outputs(
            results, args.output_dir, args.split
        )
    elif args.mode in {
        "m",
        "metaphoricity",
        "m-taxonomy",
        "m-taxonomy-agent",
        "m-conceptual-distance",
        "m-conceptual-distance-critic",
        "m-operation-audit",
        "m-native-scope-audit",
        "m-two-gate",
        "m-relation-gate",
    }:
        results = asyncio.run(
            run_m_examples(
                pipeline,
                rows_by_id,
                selected_ids,
                args.split,
                args.output_dir,
                args.sample_concurrency,
                args.mode in {
                    "m-taxonomy",
                    "m-taxonomy-agent",
                    "m-conceptual-distance",
                    "m-conceptual-distance-critic",
                },
                args.mode == "m-taxonomy-agent",
                args.mode in {
                    "m-conceptual-distance",
                    "m-conceptual-distance-critic",
                },
                args.mode == "m-conceptual-distance-critic",
                args.mode == "m-operation-audit",
                args.mode == "m-native-scope-audit",
                args.mode == "m-two-gate",
                args.mode == "m-relation-gate",
            )
        )
        details_path, predictions_path = write_m_outputs(
            results, args.output_dir, args.split
        )
    else:
        results = asyncio.run(
            run_examples(
                pipeline,
                rows_by_id,
                selected_ids,
                args.split,
                args.output_dir,
                args.sample_concurrency,
            )
        )
        details_path, predictions_path = write_run_outputs(
            results, args.output_dir, args.split
        )
    print(f"Details: {details_path}")
    print(f"Predictions: {predictions_path}")
    submission_path = args.output_dir / "submission.csv"
    if submission_path.exists():
        print(f"Submission: {submission_path}")

    if args.split == "validation":
        if args.mode in {
            "tcc",
            "tcc-importance",
            "tcc-v1-conservative",
            "tcc-v1-prompt-conservative",
            "tcc-v1-facet-conservative",
            "target-coverage",
        }:
            scores = score_tcc_validation(results, rows)
            score_path = args.output_dir / "validation_tcc_scores.json"
        elif args.mode == "mapping-strength":
            scores = score_ms_validation(results, rows)
            score_path = (
                args.output_dir / "validation_mapping_strength_scores.json"
            )
        elif args.mode in {
            "m",
            "metaphoricity",
            "m-taxonomy",
            "m-taxonomy-agent",
            "m-conceptual-distance",
            "m-conceptual-distance-critic",
            "m-operation-audit",
            "m-native-scope-audit",
            "m-two-gate",
            "m-relation-gate",
        }:
            scores = score_m_validation(results, rows)
            score_path = args.output_dir / "validation_m_scores.json"
        else:
            scores = score_validation(results, rows)
            score_path = args.output_dir / "validation_scores.json"
        score_path.write_text(
            json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
