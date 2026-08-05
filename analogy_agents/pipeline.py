import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypeVar

import pyarrow.parquet as pq
from pydantic import BaseModel
from scipy.stats import kendalltau, spearmanr
from together import Together

from .metaphoricity_cosine import (
    DEFAULT_M_CONCEPT_WEIGHT,
    DEFAULT_M_COSINE_THRESHOLD,
    DEFAULT_M_EMBEDDING_DEVICE,
    DEFAULT_M_EMBEDDING_MODEL,
    M_COSINE_POLICY_VERSION,
    M_EMBEDDING_BACKEND,
    cosine_distance,
    m_cosine_embedding_texts,
    m_cosine_latent_score,
    m_score_from_cosine,
)
from .metaphoricity_taxonomy import (
    M_CONCEPTUAL_DISTANCE_CACHE_NAMESPACE,
    M_CONCEPTUAL_DISTANCE_CRITIC_CACHE_NAMESPACE,
    M_CONCEPTUAL_DISTANCE_CRITIC_VERSION,
    M_CONCEPTUAL_DISTANCE_VERSION,
    M_TAXONOMY_AGENT_CACHE_NAMESPACE,
    M_TAXONOMY_AGENT_VERSION,
    M_TAXONOMY_CACHE_NAMESPACE,
    M_TAXONOMY_POLICY_VERSION,
    load_m_taxonomy,
)
from .prompts import (
    M_NATIVE_SCOPE_AUDIT_VERSION,
    M_ORDINAL_POLICY_VERSION,
    M_OPERATION_AUDIT_VERSION,
    M_RELATION_GATE_VERSION,
    M_TWO_GATE_VERSION,
    PROMPT_VERSION,
    TCC_COVERAGE_AUDIT_POLICY_VERSION,
    TCC_FACET_COVERAGE_AUDIT_POLICY_VERSION,
    TCC_TOPIC_IMPORTANCE_POLICY_VERSION,
    concept_decomposer_prompt,
    coverage_audit_prompt,
    domain_classifier_prompt,
    facet_coverage_audit_prompt,
    literal_instance_prompt,
    m_calibration_anchors,
    m_conceptual_distance_adjudicator_prompt,
    m_conceptual_distance_critic_prompt,
    m_conceptual_distance_final_prompt,
    m_judge_prompt,
    m_ordinal_prompt,
    m_literal_advocate_prompt,
    m_literal_scope_auditor_prompt,
    m_native_neighborhood_auditor_prompt,
    m_native_relation_critic_prompt,
    m_native_scope_adjudicator_prompt,
    m_native_source_frame_prompt,
    m_operation_adjudicator_prompt,
    m_operation_extractor_prompt,
    m_relation_identity_prompt,
    m_relation_literal_prompt,
    m_relation_source_frame_prompt,
    m_relation_target_frame_prompt,
    m_taxonomy_literal_prompt,
    m_taxonomy_final_judge_prompt,
    m_taxonomy_prompt,
    m_taxonomy_source_analysis_prompt,
    m_two_gate_literal_prompt,
    m_two_gate_native_prompt,
    m_two_gate_source_frame_prompt,
    m_two_gate_target_frame_prompt,
    mapping_extractor_prompt,
    ms_judge_prompt,
    tcc_judge_prompt,
    topic_importance_prompt,
)
from .schemas import (
    ConceptDecomposition,
    CoverageAssessment,
    CoverageAuditAssessment,
    CoverageAuditJudgment,
    DomainAnalysis,
    FacetCoverageAuditJudgment,
    LiteralInstanceJudgment,
    MConceptualDistanceCritique,
    MConceptualDistanceJudgment,
    MJudgment,
    MLiteralApplicabilityAdvocacy,
    MLiteralScopeAudit,
    MNativeNeighborhoodAudit,
    MNativeRelationCritique,
    MNativeScopeJudgment,
    MNativeSourceFrame,
    MOperationAnalysis,
    MOrdinalJudgment,
    MRelationIdentityAudit,
    MRelationSourceFrame,
    MRelationTargetFrame,
    MTaxonomyFinalJudgment,
    MTaxonomyJudgment,
    MTwoGateLiteralAudit,
    MTwoGateNativeAudit,
    MTwoGateSourceFrame,
    MTwoGateTargetFrame,
    MappingAnalysis,
    MSJudgment,
    ScoreProbabilities,
    TCCJudgment,
    TopicImportanceJudgment,
)
from .original_target_coverage_prompts import (
    PROMPT_VERSION as V1_PROMPT_VERSION,
    concept_decomposer_prompt as v1_concept_decomposer_prompt,
    tcc_judge_prompt as v1_tcc_judge_prompt,
)
from .original_mapping_strength_prompts import (
    PROMPT_VERSION as ORIGINAL_MS_PROMPT_VERSION,
    mapping_extractor_prompt as original_mapping_extractor_prompt,
    ms_judge_prompt as original_mapping_strength_judge_prompt,
)
from .original_mapping_strength_schemas import (
    MappingAnalysis as OriginalMappingAnalysis,
    MSJudgment as OriginalMappingStrengthJudgment,
)
from .ms_native_prompts import (
    MS_NATIVE_INTEGRITY_VERSION,
    ms_blind_source_frame_prompt,
    ms_calibration_anchors,
    ms_native_integrity_audit_prompt,
    ms_target_frame_prompt,
)
from .ms_native_schemas import (
    MSBlindSourceFrame,
    MSNativeIntegrityAudit,
    MSTargetMechanismFrame,
)
from .ms_corrective_prompts import (
    MS_CONSERVATIVE_CORRECTION_VERSION,
    ms_conservative_correction_prompt,
    ms_counterfactual_zero_gate_prompt,
    ms_corrective_blind_source_prompt,
)
from .ms_corrective_schemas import (
    MSConservativeCorrectionAudit,
    MSZeroGateAudit,
)
from .original_target_coverage_schemas import (
    ConceptDecomposition as V1ConceptDecomposition,
    TCCJudgment as V1TCCJudgment,
)


T = TypeVar("T", bound=BaseModel)

TCC_STATUS_VALUE = {
    "absent": 0.0,
    "partial": 0.5,
    "covered": 1.0,
}
TCC_FULL_COVERAGE_THRESHOLD = 0.80
TCC_V1_AUTO_CONSERVATIVE_POLICY_VERSION = "tcc_v1_auto_conservative_v3"
TCC_V1_FACET_CONSERVATIVE_POLICY_VERSION = "tcc_v1_facet_conservative_v1"
TCC_V1_EXACT_CACHE_NAMESPACE = "tcc_v1_exact_prompt"
ORIGINAL_MS_CACHE_NAMESPACE = "original_prompt"


@dataclass(frozen=True)
class PipelineConfig:
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.2
    reasoning_effort: str = "medium"
    max_tokens: int = 2200
    seed: int = 42
    max_concurrency: int = 3
    max_retries: int = 3
    cache_dir: Path = Path(".agent_cache")
    refresh_cache: bool = False
    embedding_model: str = DEFAULT_M_EMBEDDING_MODEL
    embedding_device: str = DEFAULT_M_EMBEDDING_DEVICE
    m_concept_weight: float = DEFAULT_M_CONCEPT_WEIGHT
    m_cosine_threshold: float = DEFAULT_M_COSINE_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 <= self.m_concept_weight <= 1.0:
            raise ValueError("m_concept_weight must be in [0, 1]")
        if not 0.0 <= self.m_cosine_threshold <= 2.0:
            raise ValueError("m_cosine_threshold must be in [0, 2]")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must be non-empty")
        if not self.embedding_device.strip():
            raise ValueError("embedding_device must be non-empty")


def load_split(dataset_dir: Path, split: str) -> list[dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    parquet_path = dataset_dir / "data" / f"{split}-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    return [{"id": index, **row} for index, row in enumerate(rows)]


@lru_cache(maxsize=2)
def load_frozen_original_ms(split: str) -> dict[int, dict[str, Any]]:
    """Load the tracked, hash-verified medium-reasoning v1 MS evidence."""
    base_dir = Path(__file__).resolve().parents[1]
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported split for frozen MS: {split!r}")

    evidence_dir = (
        base_dir
        / "artifacts/mapping_strength_evidence/cache/original_prompt"
        / "openai_gpt_oss_120b"
        / split
    )
    rows = load_split(base_dir / "challenge-dataset", split)
    records: dict[int, dict[str, Any]] = {}
    for row in rows:
        example_id = int(row["id"])
        cache_dir = evidence_dir / f"{example_id:03d}"
        mapping_path = cache_dir / "mapping_extractor.json"
        judgment_path = cache_dir / "ms_judge.json"
        if not mapping_path.exists():
            raise FileNotFoundError(mapping_path)
        if not judgment_path.exists():
            raise FileNotFoundError(judgment_path)
        mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        judgment_payload = json.loads(
            judgment_path.read_text(encoding="utf-8")
        )
        records[example_id] = {
            "id": example_id,
            "target": row["target"],
            "agents": {
                "mapping_extractor_v1_exact": mapping_payload["result"],
                "ms_judge_v1_exact": judgment_payload["result"],
            },
        }
    return records


def load_archived_v1_tcc(
    archive_dir: Path,
    split: str,
    example_id: int,
) -> tuple[ConceptDecomposition, TCCJudgment, int, dict[str, Any]]:
    """Load the immutable v1 decomposition and coverage evidence for one row."""
    cache_dir = (
        archive_dir
        / "cache"
        / "openai_gpt_oss_120b"
        / split
        / f"{example_id:03d}"
    )
    decomposition_path = cache_dir / "concept_decomposer.json"
    tcc_path = cache_dir / "tcc_judge.json"
    if not decomposition_path.exists():
        raise FileNotFoundError(decomposition_path)
    if not tcc_path.exists():
        raise FileNotFoundError(tcc_path)

    decomposition_payload = json.loads(
        decomposition_path.read_text(encoding="utf-8")
    )
    tcc_payload = json.loads(tcc_path.read_text(encoding="utf-8"))
    if decomposition_payload.get("prompt_version") != "v1":
        raise ValueError(
            f"Expected archived v1 decomposition at {decomposition_path}"
        )
    if tcc_payload.get("prompt_version") != "v1":
        raise ValueError(f"Expected archived v1 TCC judgment at {tcc_path}")

    decomposition = ConceptDecomposition.model_validate(
        decomposition_payload["result"]
    )
    archived_tcc_result = tcc_payload["result"]
    judgment = TCCJudgment(
        assessments=[
            CoverageAssessment.model_validate(assessment)
            for assessment in archived_tcc_result["assessments"]
        ]
    )
    expected_topic_ids = [
        topic.topic_id for topic in decomposition.topics
    ]
    validate_tcc_topic_ids(judgment, expected_topic_ids)

    original_score = int(archived_tcc_result["recommended_score"])
    if original_score not in {0, 1, 2}:
        raise ValueError(
            f"Archived v1 TCC score must be in 0..2, got {original_score}"
        )
    return decomposition, judgment, original_score, archived_tcc_result


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n{user}".encode("utf-8")).hexdigest()[:16]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


class SixAgentPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        api_key = os.environ.get("TOGETHER_API_KEY")
        self.client = Together(api_key=api_key) if api_key else None
        self.semaphore = asyncio.Semaphore(config.max_concurrency)
        self.embedding_semaphore = asyncio.Semaphore(1)
        self._embedding_encoder: Any | None = None

    def _cache_path(
        self,
        split: str,
        example_id: int,
        agent_name: str,
        cache_namespace: str | None = None,
    ) -> Path:
        return (
            self.config.cache_dir
            / (cache_namespace or PROMPT_VERSION)
            / _slug(self.config.model)
            / split
            / f"{example_id:03d}"
            / f"{agent_name}.json"
        )

    def _embedding_cache_path(
        self,
        split: str,
        example_id: int,
    ) -> Path:
        return (
            self.config.cache_dir
            / M_COSINE_POLICY_VERSION
            / _slug(self.config.embedding_model)
            / split
            / f"{example_id:03d}"
            / "concept_domain_embeddings.json"
        )

    async def _embed_m_cosine_texts(
        self,
        *,
        split: str,
        example_id: int,
        texts: dict[str, str],
    ) -> dict[str, list[float]]:
        ordered_names = [
            "source_concept",
            "target_concept",
            "source_domain",
            "target_domain",
        ]
        if list(texts) != ordered_names:
            raise ValueError(
                f"Unexpected M cosine text order: {list(texts)!r}"
            )
        ordered_texts = [texts[name].replace("\n", " ") for name in ordered_names]
        input_hash = hashlib.sha256(
            json.dumps(ordered_texts, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        cache_path = self._embedding_cache_path(split, example_id)
        if cache_path.exists() and not self.config.refresh_cache:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("input_hash") == input_hash
                and cached.get("embedding_backend") == M_EMBEDDING_BACKEND
                and cached.get("embedding_model") == self.config.embedding_model
            ):
                print(
                    f"[m embedding cache-hit] id={example_id} "
                    f"model={self.config.embedding_model}",
                    flush=True,
                )
                return {
                    name: [float(value) for value in cached["embeddings"][name]]
                    for name in ordered_names
                }

        print(
            f"[m embedding waiting] id={example_id} "
            f"model={self.config.embedding_model} "
            f"device={self.config.embedding_device}",
            flush=True,
        )
        async with self.embedding_semaphore:
            started = time.monotonic()
            print(f"[m embedding start] id={example_id}", flush=True)
            vectors = await asyncio.to_thread(
                self._encode_m_cosine_texts_locally,
                ordered_texts,
            )
            print(
                f"[m embedding response] id={example_id} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
        if len(vectors) != len(ordered_names):
            raise ValueError(
                "Local embedding model returned an unexpected number of vectors: "
                f"{len(vectors)} != {len(ordered_names)}"
            )
        embeddings = {
            name: [float(value) for value in vector]
            for name, vector in zip(ordered_names, vectors)
        }
        dimensions = {len(vector) for vector in embeddings.values()}
        if len(dimensions) != 1 or not next(iter(dimensions)):
            raise ValueError(
                f"Embedding dimensions must match and be non-zero: {dimensions}"
            )
        _atomic_write_json(
            cache_path,
            {
                "policy_version": M_COSINE_POLICY_VERSION,
                "embedding_backend": M_EMBEDDING_BACKEND,
                "embedding_model": self.config.embedding_model,
                "embedding_device": self.config.embedding_device,
                "input_hash": input_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "texts": texts,
                "embeddings": embeddings,
            },
        )
        return embeddings

    def _encode_m_cosine_texts_locally(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Encode one M example locally, lazily loading one shared model."""
        if self._embedding_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Local M embeddings require sentence-transformers. "
                    "Install the project requirements before using --mode m-cosine."
                ) from error

            device = (
                None
                if self.config.embedding_device == DEFAULT_M_EMBEDDING_DEVICE
                else self.config.embedding_device
            )
            started = time.monotonic()
            print(
                f"[m embedding model-load] model={self.config.embedding_model} "
                f"device={self.config.embedding_device}",
                flush=True,
            )
            self._embedding_encoder = SentenceTransformer(
                self.config.embedding_model,
                device=device,
            )
            print(
                f"[m embedding model-ready] model={self.config.embedding_model} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

        encoded = self._embedding_encoder.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return encoded.tolist()

    async def _call_structured(
        self,
        *,
        split: str,
        example_id: int,
        agent_name: str,
        output_model: type[T],
        system: str,
        user: str,
        validate_result: Callable[[T], None] | None = None,
        cache_namespace: str | None = None,
        prompt_version: str | None = None,
    ) -> T:
        cache_path = self._cache_path(
            split,
            example_id,
            agent_name,
            cache_namespace,
        )
        current_hash = _prompt_hash(system, user)
        if cache_path.exists() and not self.config.refresh_cache:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("prompt_hash") == current_hash:
                cached_result = output_model.model_validate(cached["result"])
                if validate_result is None:
                    print(
                        f"[agent cache-hit] id={example_id} agent={agent_name}",
                        flush=True,
                    )
                    return cached_result
                try:
                    validate_result(cached_result)
                except ValueError:
                    pass
                else:
                    print(
                        f"[agent cache-hit] id={example_id} agent={agent_name}",
                        flush=True,
                    )
                    return cached_result

        if self.client is None:
            raise RuntimeError(
                "TOGETHER_API_KEY is not set and no matching cached response "
                "was found. Put the key in .env or export it in the shell."
            )

        schema = output_model.model_json_schema()
        schema_instruction = (
            "\n\nJSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
        )

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            request_started: float | None = None
            try:
                print(
                    f"[agent waiting] id={example_id} agent={agent_name} "
                    f"attempt={attempt}/{self.config.max_retries}",
                    flush=True,
                )
                async with self.semaphore:
                    request_started = time.monotonic()
                    print(
                        f"[agent request] id={example_id} agent={agent_name} "
                        f"attempt={attempt}/{self.config.max_retries} "
                        f"model={self.config.model} "
                        f"reasoning={self.config.reasoning_effort} "
                        "client=sync-threaded",
                        flush=True,
                    )
                    response = await asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.config.model,
                        messages=[
                            {
                                "role": "system",
                                "content": system + schema_instruction,
                            },
                            {"role": "user", "content": user},
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": agent_name,
                                "schema": schema,
                            },
                        },
                        temperature=self.config.temperature,
                        top_p=1.0,
                        reasoning_effort=self.config.reasoning_effort,
                        max_tokens=self.config.max_tokens,
                        seed=self.config.seed + attempt - 1,
                    )
                print(
                    f"[agent response] id={example_id} agent={agent_name} "
                    f"attempt={attempt}/{self.config.max_retries} "
                    f"elapsed={time.monotonic() - request_started:.1f}s",
                    flush=True,
                )

                choice = response.choices[0]
                message = choice.message
                content = message.content
                reasoning = getattr(message, "reasoning", None) or ""
                usage = _usage_dict(response)
                print(
                    f"[agent output] id={example_id} agent={agent_name} "
                    f"finish_reason={getattr(choice, 'finish_reason', None)} "
                    f"content_chars={len(content or '')} "
                    f"reasoning_chars={len(reasoning)} "
                    f"prompt_tokens={usage.get('prompt_tokens')} "
                    f"completion_tokens={usage.get('completion_tokens')}",
                    flush=True,
                )
                if not content:
                    raise ValueError(
                        f"{agent_name} returned empty content "
                        f"(finish_reason={getattr(choice, 'finish_reason', None)}, "
                        f"reasoning_chars={len(reasoning)}, "
                        f"completion_tokens={usage.get('completion_tokens')}, "
                        f"max_tokens={self.config.max_tokens})"
                    )
                result = output_model.model_validate_json(content)
                if validate_result is not None:
                    validate_result(result)
                _atomic_write_json(
                    cache_path,
                    {
                        "agent": agent_name,
                        "model": self.config.model,
                        "prompt_version": prompt_version or PROMPT_VERSION,
                        "prompt_hash": current_hash,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "usage": _usage_dict(response),
                        "result": result.model_dump(),
                    },
                )
                print(
                    f"[agent cached] id={example_id} agent={agent_name}",
                    flush=True,
                )
                return result
            except Exception as error:
                last_error = error
                elapsed = (
                    f"{time.monotonic() - request_started:.1f}s"
                    if request_started is not None
                    else "not-started"
                )
                print(
                    f"[agent error] id={example_id} agent={agent_name} "
                    f"attempt={attempt}/{self.config.max_retries} "
                    f"elapsed={elapsed} error={type(error).__name__}: {error}",
                    flush=True,
                )
                if attempt == self.config.max_retries:
                    break
                await asyncio.sleep(2 ** (attempt - 1))

        raise RuntimeError(
            f"{agent_name} failed after {self.config.max_retries} attempts: {last_error}"
        ) from last_error

    async def evaluate(
        self, example: dict[str, Any], split: str
    ) -> dict[str, Any]:
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]

        decomposition_messages = concept_decomposer_prompt(target, description)
        mapping_messages = mapping_extractor_prompt(target, description, analogy)
        domain_messages = domain_classifier_prompt(target, description, analogy)
        literal_messages = literal_instance_prompt(target, description, analogy)

        decomposition, mapping, domain, literal = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="concept_decomposer",
                output_model=ConceptDecomposition,
                system=decomposition_messages[0],
                user=decomposition_messages[1],
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="mapping_extractor",
                output_model=MappingAnalysis,
                system=mapping_messages[0],
                user=mapping_messages[1],
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="source_domain_classifier",
                output_model=DomainAnalysis,
                system=domain_messages[0],
                user=domain_messages[1],
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="literal_instance_judge",
                output_model=LiteralInstanceJudgment,
                system=literal_messages[0],
                user=literal_messages[1],
            ),
        )

        tcc_messages = tcc_judge_prompt(
            target, description, analogy, decomposition.model_dump()
        )
        expected_tcc_topic_ids = [
            topic.topic_id for topic in decomposition.topics
        ]
        ms_messages = ms_judge_prompt(
            target, description, analogy, mapping.model_dump()
        )
        ordinal_messages = m_ordinal_prompt(
            target,
            description,
            analogy,
            domain.model_dump(),
            literal.model_dump(),
            m_calibration_anchors(split, example_id),
        )

        tcc, ms, ordinal = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="tcc_judge",
                output_model=TCCJudgment,
                system=tcc_messages[0],
                user=tcc_messages[1],
                validate_result=lambda result: validate_tcc_topic_ids(
                    result, expected_tcc_topic_ids
                ),
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="ms_judge",
                output_model=MSJudgment,
                system=ms_messages[0],
                user=ms_messages[1],
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_ordinal_judge_v7",
                output_model=MOrdinalJudgment,
                system=ordinal_messages[0],
                user=ordinal_messages[1],
            ),
        )

        tcc_ratio = tcc_coverage_ratio(tcc)
        tcc_score = tcc_score_from_ratio(tcc_ratio)
        m_score = m_score_from_ordinal(ordinal)
        m_confidence = ordinal.confidence
        m_probabilities = m_probabilities_from_score(
            m_score, m_confidence
        )
        return {
            "id": example_id,
            "target": target,
            "prediction": {
                "TCC": tcc_score,
                "MS": ms.recommended_score,
                "M": m_score,
            },
            "latent_scores": {
                "TCC": 2 * tcc_ratio,
                "MS": ms.score_probabilities.expected_score(),
                "M": m_probabilities.expected_score(),
            },
            "confidence": {
                "MS": ms.confidence,
                "M": m_confidence,
            },
            "coverage_ratio": tcc_ratio,
            "topic_scores": tcc_topic_scores(tcc),
            "m_policy": m_ordinal_policy_trace(
                ordinal, m_score, split
            ),
            "agents": {
                "concept_decomposer": decomposition.model_dump(),
                "mapping_extractor": mapping.model_dump(),
                "source_domain_classifier": domain.model_dump(),
                "literal_instance_judge": literal.model_dump(),
                "tcc_judge": tcc.model_dump(),
                "ms_judge": ms.model_dump(),
                "m_ordinal_judge": ordinal.model_dump(),
            },
        }

    async def decompose(
        self,
        example: dict[str, Any],
        split: str,
    ) -> ConceptDecomposition:
        """Run only ConceptDecomposer for a representative concept group."""
        decomposition_messages = concept_decomposer_prompt(
            example["target"], example["description"]
        )
        return await self._call_structured(
            split=split,
            example_id=int(example["id"]),
            agent_name="concept_decomposer",
            output_model=ConceptDecomposition,
            system=decomposition_messages[0],
            user=decomposition_messages[1],
        )

    async def decompose_v1_exact(
        self,
        example: dict[str, Any],
        split: str,
    ) -> V1ConceptDecomposition:
        """Execute the hash-verified original v1 ConceptDecomposer."""
        messages = v1_concept_decomposer_prompt(
            example["target"],
            example["description"],
        )
        return await self._call_structured(
            split=split,
            example_id=int(example["id"]),
            agent_name="concept_decomposer",
            output_model=V1ConceptDecomposition,
            system=messages[0],
            user=messages[1],
            cache_namespace=TCC_V1_EXACT_CACHE_NAMESPACE,
            prompt_version=V1_PROMPT_VERSION,
        )

    async def judge_tcc_v1_exact(
        self,
        example: dict[str, Any],
        split: str,
        decomposition: V1ConceptDecomposition,
    ) -> V1TCCJudgment:
        """Execute the hash-verified original v1 TCCJudge."""
        messages = v1_tcc_judge_prompt(
            example["target"],
            example["description"],
            example["analogy"],
            decomposition.model_dump(),
        )
        expected_topic_ids = [
            topic.topic_id for topic in decomposition.topics
        ]
        return await self._call_structured(
            split=split,
            example_id=int(example["id"]),
            agent_name="tcc_judge",
            output_model=V1TCCJudgment,
            system=messages[0],
            user=messages[1],
            validate_result=lambda result: validate_tcc_topic_ids(
                result,
                expected_topic_ids,
            ),
            cache_namespace=TCC_V1_EXACT_CACHE_NAMESPACE,
            prompt_version=V1_PROMPT_VERSION,
        )

    async def evaluate_exact_v1_tcc_with_importance(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Run exact v1 prompts, then apply the sample-agnostic correction."""
        v1_decomposition = await self.decompose_v1_exact(example, split)
        decomposition = ConceptDecomposition.model_validate(
            v1_decomposition.model_dump()
        )

        v1_tcc, topic_importance = await asyncio.gather(
            self.judge_tcc_v1_exact(
                example,
                split,
                v1_decomposition,
            ),
            self.judge_topic_importance(
                example,
                split,
                decomposition,
            ),
        )
        tcc = TCCJudgment(
            assessments=[
                CoverageAssessment.model_validate(
                    assessment.model_dump()
                )
                for assessment in v1_tcc.assessments
            ]
        )
        original_score = int(v1_tcc.recommended_score)
        preliminary = v1_conservative_tcc_correction(
            original_score,
            decomposition,
            tcc,
            topic_importance,
        )

        coverage_audit = None
        if (
            original_score == 1
            and preliminary["blocking_topic_ids"]
        ):
            coverage_audit = await self.audit_retained_tcc_blockers(
                example,
                split,
                decomposition,
                tcc,
                preliminary["blocking_topic_ids"],
            )
        correction = v1_conservative_tcc_correction(
            original_score,
            decomposition,
            tcc,
            topic_importance,
            coverage_audit,
        )

        agents: dict[str, Any] = {
            "concept_decomposer_v1_exact": v1_decomposition.model_dump(),
            "tcc_judge_v1_exact": v1_tcc.model_dump(),
            "topic_relation_judge": topic_importance.model_dump(),
        }
        if coverage_audit is not None:
            agents["retained_topic_coverage_auditor"] = (
                coverage_audit.model_dump()
            )

        decomposition_messages = v1_concept_decomposer_prompt(
            example["target"],
            example["description"],
        )
        tcc_messages = v1_tcc_judge_prompt(
            example["target"],
            example["description"],
            example["analogy"],
            v1_decomposition.model_dump(),
        )
        return {
            "id": int(example["id"]),
            "target": example["target"],
            "prediction": {"TCC": correction["final_score"]},
            "latent_scores": {"TCC": float(correction["final_score"])},
            "v1_prompt_execution": {
                "prompt_version": V1_PROMPT_VERSION,
                "cache_namespace": TCC_V1_EXACT_CACHE_NAMESPACE,
                "source": "recovered_original_v1_source",
                "concept_decomposer_prompt_hash": _prompt_hash(
                    *decomposition_messages
                ),
                "tcc_judge_prompt_hash": _prompt_hash(*tcc_messages),
            },
            "v1_conservative_policy": correction,
            "agents": agents,
        }

    async def evaluate_exact_v1_tcc_with_facet_audit(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Run exact v1 and derive blocker upgrades from atomic facets."""
        v1_decomposition = await self.decompose_v1_exact(example, split)
        decomposition = ConceptDecomposition.model_validate(
            v1_decomposition.model_dump()
        )

        v1_tcc, topic_importance = await asyncio.gather(
            self.judge_tcc_v1_exact(
                example,
                split,
                v1_decomposition,
            ),
            self.judge_topic_importance(
                example,
                split,
                decomposition,
            ),
        )
        tcc = TCCJudgment(
            assessments=[
                CoverageAssessment.model_validate(
                    assessment.model_dump()
                )
                for assessment in v1_tcc.assessments
            ]
        )
        original_score = int(v1_tcc.recommended_score)
        preliminary = v1_conservative_tcc_correction(
            original_score,
            decomposition,
            tcc,
            topic_importance,
        )

        facet_audit = None
        derived_coverage_audit = None
        facet_policy_trace = None
        if (
            original_score == 1
            and preliminary["blocking_topic_ids"]
        ):
            facet_audit = await self.audit_retained_tcc_blockers_by_facet(
                example,
                split,
                decomposition,
                tcc,
                preliminary["retained_topic_ids"],
                preliminary["blocking_topic_ids"],
            )
            (
                derived_coverage_audit,
                facet_policy_trace,
            ) = facet_audit_to_coverage_audit(facet_audit)

        correction = v1_conservative_tcc_correction(
            original_score,
            decomposition,
            tcc,
            topic_importance,
            derived_coverage_audit,
        )
        correction["version"] = TCC_V1_FACET_CONSERVATIVE_POLICY_VERSION
        correction["coverage_audit_version"] = (
            TCC_FACET_COVERAGE_AUDIT_POLICY_VERSION
            if facet_audit is not None
            else None
        )
        correction["facet_audit_policy"] = facet_policy_trace

        agents: dict[str, Any] = {
            "concept_decomposer_v1_exact": v1_decomposition.model_dump(),
            "tcc_judge_v1_exact": v1_tcc.model_dump(),
            "topic_relation_judge": topic_importance.model_dump(),
        }
        if facet_audit is not None:
            agents["retained_topic_facet_auditor_v2"] = (
                facet_audit.model_dump()
            )
            assert derived_coverage_audit is not None
            agents["deterministic_facet_decisions"] = (
                derived_coverage_audit.model_dump()
            )

        decomposition_messages = v1_concept_decomposer_prompt(
            example["target"],
            example["description"],
        )
        tcc_messages = v1_tcc_judge_prompt(
            example["target"],
            example["description"],
            example["analogy"],
            v1_decomposition.model_dump(),
        )
        return {
            "id": int(example["id"]),
            "target": example["target"],
            "prediction": {"TCC": correction["final_score"]},
            "latent_scores": {"TCC": float(correction["final_score"])},
            "v1_prompt_execution": {
                "prompt_version": V1_PROMPT_VERSION,
                "cache_namespace": TCC_V1_EXACT_CACHE_NAMESPACE,
                "source": "recovered_original_v1_source",
                "concept_decomposer_prompt_hash": _prompt_hash(
                    *decomposition_messages
                ),
                "tcc_judge_prompt_hash": _prompt_hash(*tcc_messages),
            },
            "v1_facet_conservative_policy": correction,
            "agents": agents,
        }

    async def evaluate_original_mapping_strength(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Run the hash-verified original MappingExtractor and MSJudge."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]

        mapping_messages = original_mapping_extractor_prompt(
            target,
            description,
            analogy,
        )
        mapping = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="mapping_extractor",
            output_model=OriginalMappingAnalysis,
            system=mapping_messages[0],
            user=mapping_messages[1],
            cache_namespace=ORIGINAL_MS_CACHE_NAMESPACE,
            prompt_version=ORIGINAL_MS_PROMPT_VERSION,
        )

        ms_messages = original_mapping_strength_judge_prompt(
            target,
            description,
            analogy,
            mapping.model_dump(),
        )
        judgment = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="ms_judge",
            output_model=OriginalMappingStrengthJudgment,
            system=ms_messages[0],
            user=ms_messages[1],
            cache_namespace=ORIGINAL_MS_CACHE_NAMESPACE,
            prompt_version=ORIGINAL_MS_PROMPT_VERSION,
        )
        return {
            "id": example_id,
            "target": target,
            "prediction": {"MS": int(judgment.recommended_score)},
            "latent_scores": {
                "MS": judgment.score_probabilities.expected_score()
            },
            "confidence": {"MS": judgment.confidence},
            "v1_prompt_execution": {
                "prompt_version": ORIGINAL_MS_PROMPT_VERSION,
                "cache_namespace": ORIGINAL_MS_CACHE_NAMESPACE,
                "source": "74_of_74_archived_hashes_verified",
                "mapping_extractor_prompt_hash": _prompt_hash(
                    *mapping_messages
                ),
                "ms_judge_prompt_hash": _prompt_hash(*ms_messages),
            },
            "agents": {
                "original_mapping_extractor": mapping.model_dump(),
                "original_mapping_strength_judge": judgment.model_dump(),
            },
        }

    async def evaluate_ms_native_integrity(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Conservatively correct v1 MS using blind native-source evidence."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]

        frozen_record = load_frozen_original_ms(split).get(example_id)
        if frozen_record is None:
            raise KeyError(
                f"Frozen original MS evidence missing {split} id={example_id}"
            )
        if frozen_record["target"] != target:
            raise ValueError(
                "Frozen original MS target mismatch for "
                f"{split} id={example_id}: "
                f"{frozen_record['target']!r} != {target!r}"
            )
        frozen_agents = frozen_record["agents"]
        mapping = OriginalMappingAnalysis.model_validate(
            frozen_agents["mapping_extractor_v1_exact"]
        )
        baseline = OriginalMappingStrengthJudgment.model_validate(
            frozen_agents["ms_judge_v1_exact"]
        )

        source_messages = ms_corrective_blind_source_prompt(analogy)
        source_frames = await asyncio.gather(
            *[
                self._call_structured(
                    split=split,
                    example_id=example_id,
                    agent_name=f"ms_zero_gate_blind_source_v1_vote_{vote}",
                    output_model=MSBlindSourceFrame,
                    system=source_messages[0],
                    user=source_messages[1],
                    cache_namespace=MS_CONSERVATIVE_CORRECTION_VERSION,
                    prompt_version=MS_CONSERVATIVE_CORRECTION_VERSION,
                    validate_result=validate_ms_blind_source_frame,
                )
                for vote in range(3)
            ]
        )
        audit_message_sets = [
            ms_counterfactual_zero_gate_prompt(
                target,
                description,
                analogy,
                mapping.model_dump(),
                source_frame.model_dump(),
                split,
                example_id,
            )
            for source_frame in source_frames
        ]
        audits = await asyncio.gather(
            *[
                self._call_structured(
                    split=split,
                    example_id=example_id,
                    agent_name=f"ms_counterfactual_zero_gate_v1_vote_{vote}",
                    output_model=MSZeroGateAudit,
                    system=messages[0],
                    user=messages[1],
                    cache_namespace=MS_CONSERVATIVE_CORRECTION_VERSION,
                    prompt_version=MS_CONSERVATIVE_CORRECTION_VERSION,
                    validate_result=validate_ms_zero_gate_audit,
                )
                for vote, messages in enumerate(audit_message_sets)
            ]
        )
        baseline_score = int(baseline.recommended_score)
        correction_votes = [
            ms_score_from_zero_gate(
                baseline_score,
                audit,
                source_frame,
                analogy,
            )
            for audit, source_frame in zip(audits, source_frames, strict=True)
        ]
        score = 0 if correction_votes.count(0) >= 2 else baseline_score
        representative_index = correction_votes.index(score)
        representative_audit = audits[representative_index]
        return {
            "id": example_id,
            "target": target,
            "prediction": {"MS": score},
            "latent_scores": {
                "MS": (
                    baseline.score_probabilities.expected_score()
                    if score == baseline_score
                    else float(score)
                )
            },
            "confidence": {
                "MS": sum(audit.confidence for audit in audits) / len(audits)
            },
            "ms_native_integrity_policy": {
                "version": MS_CONSERVATIVE_CORRECTION_VERSION,
                "decision_source": "v1_baseline_then_frozen_conservative_correction",
                "baseline_score": baseline_score,
                "baseline_source": "frozen_verified_v1_medium",
                "correction_votes": correction_votes,
                "ensemble_size": len(audits),
                "native_structural_support": (
                    representative_audit.native_structural_support
                ),
                "target_import_dependency": (
                    representative_audit.target_import_dependency
                ),
                "core_relation": representative_audit.core_relation,
                "counterfactual_result": (
                    representative_audit.counterfactual_result
                ),
                "decisive_failure": representative_audit.failure_type,
                "leave_one_out_calibration": split == "validation",
            },
            "agents": {
                "ms_v1_mapping_extractor": mapping.model_dump(),
                "ms_v1_baseline_judge": baseline.model_dump(),
                "ms_zero_gate_blind_source_frames": [
                    source_frame.model_dump() for source_frame in source_frames
                ],
                "ms_counterfactual_zero_gate_audits": [
                    audit.model_dump() for audit in audits
                ],
            },
        }

    async def evaluate_tcc(
        self,
        example: dict[str, Any],
        split: str,
        decomposition: ConceptDecomposition | None = None,
        *,
        apply_topic_importance: bool = False,
        topic_importance: TopicImportanceJudgment | None = None,
    ) -> dict[str, Any]:
        """Run only the concept-decomposition and TCC path."""
        if decomposition is None:
            decomposition = await self.decompose(example, split)

        candidate_decomposition = decomposition
        if apply_topic_importance:
            if topic_importance is None:
                topic_importance = await self.judge_topic_importance(
                    example,
                    split,
                    candidate_decomposition,
                )
            decomposition = refine_tcc_topics(
                candidate_decomposition,
                topic_importance,
            )

        tcc_messages = tcc_judge_prompt(
            example["target"],
            example["description"],
            example["analogy"],
            decomposition.model_dump(),
        )
        expected_topic_ids = [
            topic.topic_id for topic in decomposition.topics
        ]
        tcc = await self._call_structured(
            split=split,
            example_id=int(example["id"]),
            agent_name="tcc_judge",
            output_model=TCCJudgment,
            system=tcc_messages[0],
            user=tcc_messages[1],
            validate_result=lambda result: validate_tcc_topic_ids(
                result, expected_topic_ids
            ),
        )

        ratio = tcc_coverage_ratio(tcc)
        threshold_score = tcc_score_from_ratio(ratio)
        result = {
            "id": int(example["id"]),
            "target": example["target"],
            "prediction": {"TCC": threshold_score},
            "latent_scores": {"TCC": 2 * ratio},
            "coverage_ratio": ratio,
            "topic_scores": tcc_topic_scores(tcc),
            "threshold_policy": {
                "absent": TCC_STATUS_VALUE["absent"],
                "partial": TCC_STATUS_VALUE["partial"],
                "covered": TCC_STATUS_VALUE["covered"],
                "score_2_threshold": TCC_FULL_COVERAGE_THRESHOLD,
            },
            "agents": {
                "concept_decomposer": decomposition.model_dump(),
                "tcc_judge": tcc.model_dump(),
            },
        }
        if apply_topic_importance:
            assert topic_importance is not None
            retained_topic_ids = [
                topic.topic_id for topic in decomposition.topics
            ]
            candidate_topic_ids = [
                topic.topic_id for topic in candidate_decomposition.topics
            ]
            result.update(
                {
                    "topic_importance_policy": {
                        "version": TCC_TOPIC_IMPORTANCE_POLICY_VERSION,
                        "candidate_topic_ids": candidate_topic_ids,
                        "retained_topic_ids": retained_topic_ids,
                        "filtered_topic_ids": [
                            topic_id
                            for topic_id in candidate_topic_ids
                            if topic_id not in retained_topic_ids
                        ],
                    },
                }
            )
            result["agents"] = {
                "concept_decomposer": candidate_decomposition.model_dump(),
                "topic_importance_judge": topic_importance.model_dump(),
                "refined_decomposition": decomposition.model_dump(),
                "tcc_judge": tcc.model_dump(),
            }
        return result

    async def judge_topic_importance(
        self,
        example: dict[str, Any],
        split: str,
        decomposition: ConceptDecomposition,
    ) -> TopicImportanceJudgment:
        """Classify candidate topics without inspecting the analogy."""
        messages = topic_importance_prompt(
            example["target"],
            example["description"],
            decomposition.model_dump(),
        )
        expected_topic_ids = [
            topic.topic_id for topic in decomposition.topics
        ]
        return await self._call_structured(
            split=split,
            example_id=int(example["id"]),
            agent_name="topic_importance_judge",
            output_model=TopicImportanceJudgment,
            system=messages[0],
            user=messages[1],
            validate_result=lambda result: validate_topic_importance(
                result,
                expected_topic_ids,
                example["description"],
            ),
        )

    async def evaluate_archived_v1_tcc_with_importance(
        self,
        example: dict[str, Any],
        split: str,
        decomposition: ConceptDecomposition,
        archived_tcc: TCCJudgment,
        original_score: int,
        archived_tcc_result: dict[str, Any],
        topic_importance: TopicImportanceJudgment | None = None,
    ) -> dict[str, Any]:
        """Conservatively correct v1 with relation filtering and coverage audit."""
        if topic_importance is None:
            topic_importance = await self.judge_topic_importance(
                example,
                split,
                decomposition,
            )
        preliminary = v1_conservative_tcc_correction(
            original_score,
            decomposition,
            archived_tcc,
            topic_importance,
        )
        coverage_audit = None
        if (
            original_score == 1
            and preliminary["blocking_topic_ids"]
        ):
            coverage_audit = await self.audit_retained_tcc_blockers(
                example,
                split,
                decomposition,
                archived_tcc,
                preliminary["blocking_topic_ids"],
            )
        correction = v1_conservative_tcc_correction(
            original_score,
            decomposition,
            archived_tcc,
            topic_importance,
            coverage_audit,
        )
        agents = {
            "concept_decomposer_v1": decomposition.model_dump(),
            "topic_relation_judge": topic_importance.model_dump(),
            "tcc_judge_v1": archived_tcc_result,
        }
        if coverage_audit is not None:
            agents["retained_topic_coverage_auditor"] = (
                coverage_audit.model_dump()
            )
        return {
            "id": int(example["id"]),
            "target": example["target"],
            "prediction": {"TCC": correction["final_score"]},
            "latent_scores": {"TCC": float(correction["final_score"])},
            "v1_conservative_policy": correction,
            "agents": agents,
        }

    async def audit_retained_tcc_blockers(
        self,
        example: dict[str, Any],
        split: str,
        decomposition: ConceptDecomposition,
        archived_tcc: TCCJudgment,
        blocker_topic_ids: list[str],
    ) -> CoverageAuditJudgment:
        """Recheck retained v1 blockers without changing topic granularity."""
        topic_by_id = {
            topic.topic_id: topic.topic
            for topic in decomposition.topics
        }
        assessment_by_id = {
            assessment.topic_id: assessment
            for assessment in archived_tcc.assessments
        }
        retained_blockers = [
            {
                "topic_id": topic_id,
                "topic": topic_by_id[topic_id],
                "original_status": assessment_by_id[topic_id].status,
                "previous_evidence": assessment_by_id[topic_id].evidence,
            }
            for topic_id in blocker_topic_ids
        ]
        messages = coverage_audit_prompt(
            example["target"],
            example["description"],
            example["analogy"],
            retained_blockers,
        )
        expected_statuses = [
            (item["topic_id"], item["original_status"])
            for item in retained_blockers
        ]
        return await self._call_structured(
            split=split,
            example_id=int(example["id"]),
            agent_name="retained_topic_coverage_auditor",
            output_model=CoverageAuditJudgment,
            system=messages[0],
            user=messages[1],
            validate_result=lambda result: validate_coverage_audit(
                result,
                expected_statuses,
            ),
        )

    async def audit_retained_tcc_blockers_by_facet(
        self,
        example: dict[str, Any],
        split: str,
        decomposition: ConceptDecomposition,
        v1_tcc: TCCJudgment,
        retained_topic_ids: list[str],
        blocker_topic_ids: list[str],
    ) -> FacetCoverageAuditJudgment:
        """Audit blockers as atomic facets without letting the model score."""
        topic_by_id = {
            topic.topic_id: topic.topic
            for topic in decomposition.topics
        }
        assessment_by_id = {
            assessment.topic_id: assessment
            for assessment in v1_tcc.assessments
        }
        retained_topic_context = [
            {
                "topic_id": topic_id,
                "topic": topic_by_id[topic_id],
                "v1_status": assessment_by_id[topic_id].status,
                "v1_evidence": assessment_by_id[topic_id].evidence,
            }
            for topic_id in retained_topic_ids
        ]
        blockers_to_audit = [
            {
                "topic_id": topic_id,
                "topic": topic_by_id[topic_id],
                "original_status": assessment_by_id[topic_id].status,
                "previous_evidence": assessment_by_id[topic_id].evidence,
            }
            for topic_id in blocker_topic_ids
        ]
        messages = facet_coverage_audit_prompt(
            example["target"],
            example["description"],
            example["analogy"],
            retained_topic_context,
            blockers_to_audit,
        )
        expected_statuses = [
            (item["topic_id"], item["original_status"])
            for item in blockers_to_audit
        ]
        return await self._call_structured(
            split=split,
            example_id=int(example["id"]),
            agent_name="retained_topic_facet_auditor_v2",
            output_model=FacetCoverageAuditJudgment,
            system=messages[0],
            user=messages[1],
            validate_result=lambda result: validate_facet_coverage_audit(
                result,
                expected_statuses,
                example["description"],
            ),
        )

    async def evaluate_m(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Run only the source-domain analysis and metaphoricity path."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]

        domain_messages = domain_classifier_prompt(target, description, analogy)
        literal_messages = literal_instance_prompt(target, description, analogy)
        domain, literal = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="source_domain_classifier",
                output_model=DomainAnalysis,
                system=domain_messages[0],
                user=domain_messages[1],
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="literal_instance_judge",
                output_model=LiteralInstanceJudgment,
                system=literal_messages[0],
                user=literal_messages[1],
            ),
        )
        ordinal_messages = m_ordinal_prompt(
            target,
            description,
            analogy,
            domain.model_dump(),
            literal.model_dump(),
            m_calibration_anchors(split, example_id),
        )
        ordinal = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_ordinal_judge_v7",
            output_model=MOrdinalJudgment,
            system=ordinal_messages[0],
            user=ordinal_messages[1],
        )
        m_score = m_score_from_ordinal(ordinal)
        m_confidence = ordinal.confidence
        m_probabilities = m_probabilities_from_score(m_score, m_confidence)
        return {
            "id": example_id,
            "target": target,
            "prediction": {"M": m_score},
            "latent_scores": {"M": m_probabilities.expected_score()},
            "confidence": {"M": m_confidence},
            "m_policy": m_ordinal_policy_trace(
                ordinal, m_score, split
            ),
            "agents": {
                "source_domain_classifier": domain.model_dump(),
                "literal_instance_judge": literal.model_dump(),
                "m_ordinal_judge": ordinal.model_dump(),
            },
        }

    async def evaluate_m_cosine(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Run the current literal gate plus deterministic concept/domain cosine."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]

        print(
            f"[m llm-evidence start] id={example_id} "
            "agents=source_domain_classifier,literal_instance_judge",
            flush=True,
        )
        domain_messages = domain_classifier_prompt(target, description, analogy)
        literal_messages = literal_instance_prompt(target, description, analogy)
        domain, literal = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="source_domain_classifier",
                output_model=DomainAnalysis,
                system=domain_messages[0],
                user=domain_messages[1],
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="literal_instance_judge",
                output_model=LiteralInstanceJudgment,
                system=literal_messages[0],
                user=literal_messages[1],
            ),
        )
        print(f"[m llm-evidence ready] id={example_id}", flush=True)

        embedding_texts = m_cosine_embedding_texts(domain)
        embeddings = await self._embed_m_cosine_texts(
            split=split,
            example_id=example_id,
            texts=embedding_texts,
        )
        concept_distance = cosine_distance(
            embeddings["source_concept"],
            embeddings["target_concept"],
        )
        domain_distance = cosine_distance(
            embeddings["source_domain"],
            embeddings["target_domain"],
        )
        m_score, combined_distance = m_score_from_cosine(
            literal_instance=literal.literal_instance,
            concept_distance=concept_distance,
            domain_distance=domain_distance,
            concept_weight=self.config.m_concept_weight,
            nonliteral_threshold=self.config.m_cosine_threshold,
        )
        latent_score = m_cosine_latent_score(
            literal.literal_instance,
            combined_distance,
        )
        decisive_rule = (
            "literal_instance"
            if literal.literal_instance == "yes"
            else (
                "nonliteral_distance_at_or_below_threshold"
                if m_score == 1
                else "nonliteral_distance_above_threshold"
            )
        )
        return {
            "id": example_id,
            "target": target,
            "prediction": {"M": m_score},
            "latent_scores": {"M": latent_score},
            "confidence": {"M": literal.confidence},
            "m_cosine_policy": {
                "version": M_COSINE_POLICY_VERSION,
                "decisive_rule": decisive_rule,
                "literal_instance": literal.literal_instance,
                "embedding_backend": M_EMBEDDING_BACKEND,
                "embedding_model": self.config.embedding_model,
                "embedding_device": self.config.embedding_device,
                "concept_weight": self.config.m_concept_weight,
                "domain_weight": 1.0 - self.config.m_concept_weight,
                "nonliteral_threshold": self.config.m_cosine_threshold,
                "concept_distance": concept_distance,
                "domain_distance": domain_distance,
                "combined_distance": combined_distance,
                "threshold_margin": combined_distance
                - self.config.m_cosine_threshold,
                "embedding_texts": embedding_texts,
            },
            "agents": {
                "source_domain_classifier": domain.model_dump(),
                "literal_instance_judge": literal.model_dump(),
            },
        }

    async def evaluate_m_two_gate(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Run blind source/target framing and two independent M boundaries."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]

        target_messages = m_two_gate_target_frame_prompt(target, description)
        source_messages = m_two_gate_source_frame_prompt(analogy)
        target_frame, source_frame = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_two_gate_target_frame_v1",
                output_model=MTwoGateTargetFrame,
                system=target_messages[0],
                user=target_messages[1],
                cache_namespace=M_TWO_GATE_VERSION,
                prompt_version=M_TWO_GATE_VERSION,
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_two_gate_blind_source_frame_v1",
                output_model=MTwoGateSourceFrame,
                system=source_messages[0],
                user=source_messages[1],
                cache_namespace=M_TWO_GATE_VERSION,
                prompt_version=M_TWO_GATE_VERSION,
            ),
        )

        anchors = m_calibration_anchors(split, example_id)
        literal_messages = m_two_gate_literal_prompt(
            target,
            description,
            target_frame.model_dump(),
            source_frame.model_dump(),
            anchors,
        )
        native_messages = m_two_gate_native_prompt(
            target,
            description,
            target_frame.model_dump(),
            source_frame.model_dump(),
            anchors,
        )
        literal_audit, native_audit = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_two_gate_literal_boundary_v1",
                output_model=MTwoGateLiteralAudit,
                system=literal_messages[0],
                user=literal_messages[1],
                cache_namespace=M_TWO_GATE_VERSION,
                prompt_version=M_TWO_GATE_VERSION,
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_two_gate_native_boundary_v1",
                output_model=MTwoGateNativeAudit,
                system=native_messages[0],
                user=native_messages[1],
                cache_namespace=M_TWO_GATE_VERSION,
                prompt_version=M_TWO_GATE_VERSION,
                validate_result=lambda result: validate_m_two_gate_native_audit(
                    result,
                    target_frame,
                ),
            ),
        )

        m_score = m_score_from_two_gate(literal_audit, native_audit)
        if m_score == 0:
            m_confidence = literal_audit.confidence
        else:
            m_confidence = min(
                literal_audit.confidence,
                native_audit.confidence,
            )
        m_probabilities = m_probabilities_from_score(m_score, m_confidence)
        return {
            "id": example_id,
            "target": target,
            "prediction": {"M": m_score},
            "latent_scores": {"M": m_probabilities.expected_score()},
            "confidence": {"M": m_confidence},
            "m_decision": {
                "version": M_TWO_GATE_VERSION,
                "decision_source": "blind_frames_then_two_fixed_gates",
                "literal_instance": literal_audit.literal_instance,
                "native_relation_match": native_audit.native_relation_match,
                "role_change_degree": native_audit.role_change_degree,
                "leave_one_out_calibration": split == "validation",
                "available_anchor_ids": [
                    anchor["anchor_id"] for anchor in anchors
                ],
            },
            "agents": {
                "m_two_gate_target_frame": target_frame.model_dump(),
                "m_two_gate_blind_source_frame": source_frame.model_dump(),
                "m_two_gate_literal_boundary": literal_audit.model_dump(),
                "m_two_gate_native_boundary": native_audit.model_dump(),
            },
        }

    async def evaluate_m_relation_gate(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Run blind relation signatures and literal/native identity gates."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]

        target_messages = m_relation_target_frame_prompt(target, description)
        source_messages = m_relation_source_frame_prompt(analogy)
        target_frame, source_frame = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_relation_target_signature_v1",
                output_model=MRelationTargetFrame,
                system=target_messages[0],
                user=target_messages[1],
                cache_namespace=M_RELATION_GATE_VERSION,
                prompt_version=M_RELATION_GATE_VERSION,
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_relation_blind_source_signature_v1",
                output_model=MRelationSourceFrame,
                system=source_messages[0],
                user=source_messages[1],
                cache_namespace=M_RELATION_GATE_VERSION,
                prompt_version=M_RELATION_GATE_VERSION,
            ),
        )

        anchors = m_calibration_anchors(split, example_id)
        literal_messages = m_relation_literal_prompt(
            target,
            description,
            target_frame.model_dump(),
            source_frame.model_dump(),
            anchors,
        )
        identity_messages = m_relation_identity_prompt(
            target,
            description,
            target_frame.model_dump(),
            source_frame.model_dump(),
            anchors,
        )
        literal_audit, identity_audit = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_relation_literal_boundary_v1",
                output_model=MTwoGateLiteralAudit,
                system=literal_messages[0],
                user=literal_messages[1],
                cache_namespace=M_RELATION_GATE_VERSION,
                prompt_version=M_RELATION_GATE_VERSION,
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_relation_identity_boundary_v1",
                output_model=MRelationIdentityAudit,
                system=identity_messages[0],
                user=identity_messages[1],
                cache_namespace=M_RELATION_GATE_VERSION,
                prompt_version=M_RELATION_GATE_VERSION,
            ),
        )

        m_score = m_score_from_relation_gate(literal_audit, identity_audit)
        if m_score == 0:
            m_confidence = literal_audit.confidence
        else:
            m_confidence = min(
                literal_audit.confidence,
                identity_audit.confidence,
            )
        m_probabilities = m_probabilities_from_score(m_score, m_confidence)
        return {
            "id": example_id,
            "target": target,
            "prediction": {"M": m_score},
            "latent_scores": {"M": m_probabilities.expected_score()},
            "confidence": {"M": m_confidence},
            "m_decision": {
                "version": M_RELATION_GATE_VERSION,
                "decision_source": "blind_relation_frames_then_fixed_identity_gate",
                "literal_instance": literal_audit.literal_instance,
                "relation_status": identity_audit.relation_status,
                "carrier_compatibility": identity_audit.carrier_compatibility,
                "terminology_test": identity_audit.terminology_test,
                "gloss_removal_test": identity_audit.gloss_removal_test,
                "analogy_specific_reinterpretation": (
                    identity_audit.analogy_specific_reinterpretation
                ),
                "leave_one_out_calibration": split == "validation",
                "available_anchor_ids": [
                    anchor["anchor_id"] for anchor in anchors
                ],
            },
            "agents": {
                "m_relation_target_signature": target_frame.model_dump(),
                "m_relation_blind_source_signature": source_frame.model_dump(),
                "m_relation_literal_boundary": literal_audit.model_dump(),
                "m_relation_identity_boundary": identity_audit.model_dump(),
            },
        }

    async def evaluate_m_taxonomy(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Run the frozen fair-taxonomy mapper and deterministic LCA policy."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]
        taxonomy = load_m_taxonomy()

        domain_messages = m_taxonomy_source_analysis_prompt(
            target, description, analogy
        )
        literal_messages = m_taxonomy_literal_prompt(
            target, description, analogy
        )
        domain, literal = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_taxonomy_source_analyzer_v2",
                output_model=DomainAnalysis,
                system=domain_messages[0],
                user=domain_messages[1],
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_taxonomy_literal_judge_v2",
                output_model=LiteralInstanceJudgment,
                system=literal_messages[0],
                user=literal_messages[1],
            ),
        )
        taxonomy_messages = m_taxonomy_prompt(
            target,
            description,
            analogy,
            domain.model_dump(),
            literal.model_dump(),
            taxonomy.prompt_payload(target),
        )
        judgment = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_taxonomy_mapper_v9_v2",
            output_model=MTaxonomyJudgment,
            system=taxonomy_messages[0],
            user=taxonomy_messages[1],
            validate_result=lambda result: taxonomy.validate_judgment(
                target,
                result,
            ),
            cache_namespace=M_TAXONOMY_CACHE_NAMESPACE,
            prompt_version=M_TAXONOMY_POLICY_VERSION,
        )
        policy = taxonomy.score_trace(
            target,
            judgment,
            literal.literal_instance,
        )
        m_score = int(policy["score"])
        m_confidence = judgment.confidence
        m_probabilities = m_probabilities_from_score(m_score, m_confidence)
        return {
            "id": example_id,
            "target": target,
            "prediction": {"M": m_score},
            "latent_scores": {"M": m_probabilities.expected_score()},
            "confidence": {"M": m_confidence},
            "m_policy": policy,
            "agents": {
                "source_domain_classifier": domain.model_dump(),
                "literal_instance_judge": literal.model_dump(),
                "m_taxonomy_mapper": judgment.model_dump(),
            },
        }

    async def evaluate_m_taxonomy_agent(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Map three taxonomy axes, then let an independent agent score M."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]
        taxonomy = load_m_taxonomy()

        domain_messages = m_taxonomy_source_analysis_prompt(
            target, description, analogy
        )
        literal_messages = m_taxonomy_literal_prompt(
            target, description, analogy
        )
        domain, literal = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_taxonomy_source_analyzer_v2",
                output_model=DomainAnalysis,
                system=domain_messages[0],
                user=domain_messages[1],
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_taxonomy_literal_judge_v2",
                output_model=LiteralInstanceJudgment,
                system=literal_messages[0],
                user=literal_messages[1],
            ),
        )
        taxonomy_messages = m_taxonomy_prompt(
            target,
            description,
            analogy,
            domain.model_dump(),
            literal.model_dump(),
            taxonomy.prompt_payload(target),
        )
        mapping = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_taxonomy_mapper_v9_v2",
            output_model=MTaxonomyJudgment,
            system=taxonomy_messages[0],
            user=taxonomy_messages[1],
            validate_result=lambda result: taxonomy.validate_judgment(
                target,
                result,
            ),
            cache_namespace=M_TAXONOMY_CACHE_NAMESPACE,
            prompt_version=M_TAXONOMY_POLICY_VERSION,
        )
        taxonomy_evidence = taxonomy.comparison_trace(target, mapping)
        final_messages = m_taxonomy_final_judge_prompt(
            target,
            description,
            analogy,
            domain.model_dump(),
            literal.model_dump(),
            taxonomy_evidence,
        )
        final_judgment = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_taxonomy_final_judge_v1",
            output_model=MTaxonomyFinalJudgment,
            system=final_messages[0],
            user=final_messages[1],
            cache_namespace=M_TAXONOMY_AGENT_CACHE_NAMESPACE,
            prompt_version=M_TAXONOMY_AGENT_VERSION,
        )
        return {
            "id": example_id,
            "target": target,
            "prediction": {"M": final_judgment.recommended_score},
            "latent_scores": {
                "M": final_judgment.score_probabilities.expected_score()
            },
            "confidence": {"M": final_judgment.confidence},
            "m_decision": {
                "version": M_TAXONOMY_AGENT_VERSION,
                "decision_source": "independent_final_agent",
                "taxonomy_evidence": taxonomy_evidence,
            },
            "agents": {
                "source_domain_classifier": domain.model_dump(),
                "literal_instance_judge": literal.model_dump(),
                "m_taxonomy_mapper": mapping.model_dump(),
                "m_taxonomy_final_judge": final_judgment.model_dump(),
            },
        }

    async def evaluate_m_conceptual_distance(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Judge overall source-target conceptual distance from taxonomy evidence."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]
        taxonomy = load_m_taxonomy()

        domain_messages = m_taxonomy_source_analysis_prompt(
            target, description, analogy
        )
        literal_messages = m_taxonomy_literal_prompt(
            target, description, analogy
        )
        domain, literal = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_taxonomy_source_analyzer_v2",
                output_model=DomainAnalysis,
                system=domain_messages[0],
                user=domain_messages[1],
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_taxonomy_literal_judge_v2",
                output_model=LiteralInstanceJudgment,
                system=literal_messages[0],
                user=literal_messages[1],
            ),
        )
        taxonomy_messages = m_taxonomy_prompt(
            target,
            description,
            analogy,
            domain.model_dump(),
            literal.model_dump(),
            taxonomy.prompt_payload(target),
        )
        mapping = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_taxonomy_mapper_v9_v2",
            output_model=MTaxonomyJudgment,
            system=taxonomy_messages[0],
            user=taxonomy_messages[1],
            validate_result=lambda result: taxonomy.validate_judgment(
                target,
                result,
            ),
            cache_namespace=M_TAXONOMY_CACHE_NAMESPACE,
            prompt_version=M_TAXONOMY_POLICY_VERSION,
        )
        taxonomy_evidence = taxonomy.comparison_trace(target, mapping)
        final_messages = m_conceptual_distance_final_prompt(
            target,
            description,
            analogy,
            domain.model_dump(),
            literal.model_dump(),
            taxonomy_evidence,
        )
        final_judgment = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_overall_conceptual_distance_judge_v1",
            output_model=MConceptualDistanceJudgment,
            system=final_messages[0],
            user=final_messages[1],
            cache_namespace=M_CONCEPTUAL_DISTANCE_CACHE_NAMESPACE,
            prompt_version=M_CONCEPTUAL_DISTANCE_VERSION,
        )
        return {
            "id": example_id,
            "target": target,
            "prediction": {"M": final_judgment.recommended_score},
            "latent_scores": {
                "M": final_judgment.score_probabilities.expected_score()
            },
            "confidence": {"M": final_judgment.confidence},
            "m_decision": {
                "version": M_CONCEPTUAL_DISTANCE_VERSION,
                "decision_source": "overall_conceptual_distance_agent",
                "taxonomy_evidence": taxonomy_evidence,
            },
            "agents": {
                "source_domain_classifier": domain.model_dump(),
                "literal_instance_judge": literal.model_dump(),
                "m_taxonomy_mapper": mapping.model_dump(),
                "m_conceptual_distance_judge": final_judgment.model_dump(),
            },
        }

    async def evaluate_m_conceptual_distance_critic(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Critique native proximity, then adjudicate overall conceptual distance."""
        provisional_result = await self.evaluate_m_conceptual_distance(
            example,
            split,
        )
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]
        example_id = int(example["id"])
        provisional = MConceptualDistanceJudgment.model_validate(
            provisional_result["agents"]["m_conceptual_distance_judge"]
        )
        taxonomy_evidence = provisional_result["m_decision"][
            "taxonomy_evidence"
        ]
        source_analysis = provisional_result["agents"][
            "source_domain_classifier"
        ]
        literal_analysis = provisional_result["agents"][
            "literal_instance_judge"
        ]

        critic_messages = m_conceptual_distance_critic_prompt(
            target,
            description,
            analogy,
            provisional.model_dump(),
            taxonomy_evidence,
        )
        critique = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_native_neighborhood_critic_v1",
            output_model=MConceptualDistanceCritique,
            system=critic_messages[0],
            user=critic_messages[1],
            cache_namespace=M_CONCEPTUAL_DISTANCE_CRITIC_CACHE_NAMESPACE,
            prompt_version=M_CONCEPTUAL_DISTANCE_CRITIC_VERSION,
        )
        adjudicator_messages = m_conceptual_distance_adjudicator_prompt(
            target,
            description,
            analogy,
            source_analysis,
            literal_analysis,
            taxonomy_evidence,
            provisional.model_dump(),
            critique.model_dump(),
        )
        final_judgment = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_conceptual_distance_adjudicator_v1",
            output_model=MConceptualDistanceJudgment,
            system=adjudicator_messages[0],
            user=adjudicator_messages[1],
            cache_namespace=M_CONCEPTUAL_DISTANCE_CRITIC_CACHE_NAMESPACE,
            prompt_version=M_CONCEPTUAL_DISTANCE_CRITIC_VERSION,
        )
        agents = dict(provisional_result["agents"])
        agents["m_native_neighborhood_critic"] = critique.model_dump()
        agents["m_conceptual_distance_adjudicator"] = (
            final_judgment.model_dump()
        )
        return {
            "id": example_id,
            "target": target,
            "prediction": {"M": final_judgment.recommended_score},
            "latent_scores": {
                "M": final_judgment.score_probabilities.expected_score()
            },
            "confidence": {"M": final_judgment.confidence},
            "m_decision": {
                "version": M_CONCEPTUAL_DISTANCE_CRITIC_VERSION,
                "decision_source": "critic_then_independent_adjudicator",
                "taxonomy_evidence": taxonomy_evidence,
                "provisional_score": provisional.recommended_score,
            },
            "agents": agents,
        }

    async def evaluate_m_operation_audit(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Run a label-free operation-first balanced M adjudication."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]

        operation_messages = m_operation_extractor_prompt(
            target,
            description,
            analogy,
        )
        operation = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_operation_extractor_v1",
            output_model=MOperationAnalysis,
            system=operation_messages[0],
            user=operation_messages[1],
            cache_namespace=M_OPERATION_AUDIT_VERSION,
            prompt_version=M_OPERATION_AUDIT_VERSION,
        )
        literal_messages = m_literal_advocate_prompt(
            target,
            description,
            analogy,
            operation.model_dump(),
        )
        relation_messages = m_native_relation_critic_prompt(
            target,
            description,
            analogy,
            operation.model_dump(),
        )
        literal, relation = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_literal_applicability_advocate_v1",
                output_model=MLiteralApplicabilityAdvocacy,
                system=literal_messages[0],
                user=literal_messages[1],
                cache_namespace=M_OPERATION_AUDIT_VERSION,
                prompt_version=M_OPERATION_AUDIT_VERSION,
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_native_relation_role_critic_v1",
                output_model=MNativeRelationCritique,
                system=relation_messages[0],
                user=relation_messages[1],
                cache_namespace=M_OPERATION_AUDIT_VERSION,
                prompt_version=M_OPERATION_AUDIT_VERSION,
            ),
        )
        final_messages = m_operation_adjudicator_prompt(
            target,
            description,
            analogy,
            operation.model_dump(),
            literal.model_dump(),
            relation.model_dump(),
        )
        final_judgment = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_operation_adjudicator_v1",
            output_model=MConceptualDistanceJudgment,
            system=final_messages[0],
            user=final_messages[1],
            cache_namespace=M_OPERATION_AUDIT_VERSION,
            prompt_version=M_OPERATION_AUDIT_VERSION,
        )
        return {
            "id": example_id,
            "target": target,
            "prediction": {"M": final_judgment.recommended_score},
            "latent_scores": {
                "M": final_judgment.score_probabilities.expected_score()
            },
            "confidence": {"M": final_judgment.confidence},
            "m_decision": {
                "version": M_OPERATION_AUDIT_VERSION,
                "decision_source": "balanced_operation_first_adjudicator",
            },
            "agents": {
                "m_operation_extractor": operation.model_dump(),
                "m_literal_applicability_advocate": literal.model_dump(),
                "m_native_relation_role_critic": relation.model_dump(),
                "m_operation_adjudicator": final_judgment.model_dump(),
            },
        }

    async def evaluate_m_native_scope_audit(
        self,
        example: dict[str, Any],
        split: str,
    ) -> dict[str, Any]:
        """Run a label-free native-source-scope contrastive M audit."""
        example_id = int(example["id"])
        target = example["target"]
        description = example["description"]
        analogy = example["analogy"]

        frame_messages = m_native_source_frame_prompt(
            target,
            description,
            analogy,
        )
        source_frame = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_native_source_frame_extractor_v1",
            output_model=MNativeSourceFrame,
            system=frame_messages[0],
            user=frame_messages[1],
            cache_namespace=M_NATIVE_SCOPE_AUDIT_VERSION,
            prompt_version=M_NATIVE_SCOPE_AUDIT_VERSION,
        )

        literal_messages = m_literal_scope_auditor_prompt(
            target,
            description,
            analogy,
            source_frame.model_dump(),
        )
        neighbor_messages = m_native_neighborhood_auditor_prompt(
            target,
            description,
            analogy,
            source_frame.model_dump(),
        )
        literal_scope, native_neighborhood = await asyncio.gather(
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_literal_scope_auditor_v1",
                output_model=MLiteralScopeAudit,
                system=literal_messages[0],
                user=literal_messages[1],
                cache_namespace=M_NATIVE_SCOPE_AUDIT_VERSION,
                prompt_version=M_NATIVE_SCOPE_AUDIT_VERSION,
            ),
            self._call_structured(
                split=split,
                example_id=example_id,
                agent_name="m_native_neighborhood_auditor_v1",
                output_model=MNativeNeighborhoodAudit,
                system=neighbor_messages[0],
                user=neighbor_messages[1],
                cache_namespace=M_NATIVE_SCOPE_AUDIT_VERSION,
                prompt_version=M_NATIVE_SCOPE_AUDIT_VERSION,
            ),
        )

        final_messages = m_native_scope_adjudicator_prompt(
            target,
            description,
            analogy,
            source_frame.model_dump(),
            literal_scope.model_dump(),
            native_neighborhood.model_dump(),
        )
        final_judgment = await self._call_structured(
            split=split,
            example_id=example_id,
            agent_name="m_native_scope_adjudicator_v1",
            output_model=MNativeScopeJudgment,
            system=final_messages[0],
            user=final_messages[1],
            cache_namespace=M_NATIVE_SCOPE_AUDIT_VERSION,
            prompt_version=M_NATIVE_SCOPE_AUDIT_VERSION,
        )
        return {
            "id": example_id,
            "target": target,
            "prediction": {"M": final_judgment.recommended_score},
            "latent_scores": {
                "M": final_judgment.score_probabilities.expected_score()
            },
            "confidence": {"M": final_judgment.confidence},
            "m_decision": {
                "version": M_NATIVE_SCOPE_AUDIT_VERSION,
                "decision_source": "native_scope_contrastive_adjudicator",
            },
            "agents": {
                "m_native_source_frame": source_frame.model_dump(),
                "m_literal_scope_auditor": literal_scope.model_dump(),
                "m_native_neighborhood_auditor": (
                    native_neighborhood.model_dump()
                ),
                "m_native_scope_adjudicator": final_judgment.model_dump(),
            },
        }


def m_score_from_ordinal(judgment: MOrdinalJudgment) -> int:
    """Apply the v7.1 ordinal boundary to one independently judged analogy."""
    if judgment.literal_instance == "yes":
        return 0
    if (
        judgment.native_relation_match == "yes"
        and judgment.role_change_degree == "none_or_one"
    ):
        return 1
    return 2


def validate_ms_native_integrity_audit(
    audit: MSNativeIntegrityAudit,
    target_frame: MSTargetMechanismFrame,
) -> None:
    """Require the auditor to assess every target requirement in order."""
    expected_requirements = [
        requirement.requirement
        for requirement in target_frame.core_requirements
    ]
    returned_requirements = [
        alignment.target_requirement
        for alignment in audit.requirement_alignments
    ]
    if returned_requirements != expected_requirements:
        raise ValueError(
            "MS native-integrity audit must return target requirements "
            "verbatim and in order: "
            f"{returned_requirements!r} != {expected_requirements!r}"
        )


def validate_ms_correction_audit(
    audit: MSConservativeCorrectionAudit,
    mapping_count: int,
) -> None:
    """Require exactly one ordered audit record per extracted mapping."""
    returned_indices = [claim.mapping_index for claim in audit.claim_audits]
    expected_indices = list(range(mapping_count))
    if returned_indices != expected_indices:
        raise ValueError(
            "MS correction audit must return every mapping index in order: "
            f"{returned_indices!r} != {expected_indices!r}"
        )
    primary_indices = audit.primary_mapping_indices
    if primary_indices != sorted(set(primary_indices)):
        raise ValueError(
            "MS primary mapping indices must be unique and sorted: "
            f"{primary_indices!r}"
        )
    if any(index < 0 or index >= mapping_count for index in primary_indices):
        raise ValueError(
            "MS primary mapping index is outside extracted mappings: "
            f"{primary_indices!r}"
        )
    audited_primary = [
        claim.mapping_index
        for claim in audit.claim_audits
        if claim.importance == "primary"
    ]
    if audited_primary != primary_indices:
        raise ValueError(
            "MS primary index list must match per-claim importance: "
            f"{primary_indices!r} != {audited_primary!r}"
        )


def validate_ms_zero_gate_audit(audit: MSZeroGateAudit) -> None:
    """Keep the binary gate's failure label consistent with its counterfactual."""
    if (
        audit.counterfactual_result == "collapses"
        and audit.failure_type == "none"
    ):
        raise ValueError(
            "Collapsed MS counterfactual must name a decisive failure type"
        )
    if (
        audit.counterfactual_result != "collapses"
        and audit.failure_type != "none"
    ):
        raise ValueError(
            "Non-collapsed MS counterfactual cannot name a decisive failure"
        )


def validate_ms_blind_source_frame(frame: MSBlindSourceFrame) -> None:
    """Keep fictional-source ontology and coherence fields consistent."""
    if frame.source_ontology == "explicitly_fictional_rule_system":
        if frame.fictional_mechanism_coherence == "not_applicable":
            raise ValueError(
                "Fictional source must assess fictional mechanism coherence"
            )
    elif frame.fictional_mechanism_coherence != "not_applicable":
        raise ValueError(
            "Non-fictional source must use not_applicable coherence"
        )


def ms_score_from_zero_gate(
    baseline_score: int,
    audit: MSZeroGateAudit,
    source_frame: MSBlindSourceFrame,
    analogy: str,
) -> int:
    """Return zero only for a structurally corroborated collapse verdict."""
    formal_mechanism_pattern = re.compile(
        r"algebra|equation|factoriz|symbolic expression|gradient|derivative|"
        r"backprop|error propagation|x[²^]|√|π",
        re.IGNORECASE,
    )
    analogy_has_formal_mechanism = bool(
        formal_mechanism_pattern.search(analogy)
    )
    decisive_mechanism_import = analogy_has_formal_mechanism and any(
        detail.dependency == "essential"
        and (
            detail.import_kind == "formal_calculation"
            or formal_mechanism_pattern.search(
                f"{detail.detail} {detail.why_not_native}"
            )
        )
        for detail in source_frame.imported_target_details
    )
    coherent_fictional_mechanism = (
        source_frame.source_ontology == "explicitly_fictional_rule_system"
        and source_frame.fictional_mechanism_coherence == "yes"
    )
    injected_operation_dominates = (
        decisive_mechanism_import
        and not coherent_fictional_mechanism
        and audit.counterfactual_result != "intact"
        and audit.native_structural_support != "substantial"
    )
    impossible_or_reversed = (
        audit.counterfactual_result == "collapses"
        and audit.failure_type
        in {"reversed_core_relation", "impossible_source_operation"}
    )
    missing_recursive_identity = (
        audit.self_reference_target == "yes"
        and audit.same_process_on_smaller_instance == "no"
        and audit.native_nesting_or_self_reference == "no"
        and audit.linear_handoff_only == "yes"
        and audit.counterfactual_result == "collapses"
    )
    if (
        injected_operation_dominates
        or missing_recursive_identity
        or impossible_or_reversed
    ):
        return 0
    return baseline_score


def ms_score_from_conservative_correction(
    baseline: OriginalMappingStrengthJudgment,
    audit: MSConservativeCorrectionAudit,
) -> int:
    """Apply one frozen correction rule without replacing the v1 baseline."""
    baseline_score = int(baseline.recommended_score)
    if audit.decisive_failure != "none":
        return 0
    if baseline_score == 1 and audit.promotion_safe == "yes":
        non_sound_indices = [
            index
            for index, assessment in enumerate(baseline.assessments)
            if assessment.judgment != "sound"
        ]
        safe_issue_types = {
            "terminology_precision",
            "auxiliary_sequence_statement",
        }
        if non_sound_indices and all(
            audit.claim_audits[index].issue_type in safe_issue_types
            for index in non_sound_indices
        ):
            return 2
    return baseline_score


def ms_score_from_native_integrity(audit: MSNativeIntegrityAudit) -> int:
    """Apply the frozen native-source integrity boundary to MS evidence."""
    if (
        audit.decisive_failure != "none"
        or audit.native_core_alignment == "none"
        or audit.source_integrity == "target_constructed"
        or audit.causal_consistency == "contradictory"
    ):
        return 0
    if (
        audit.native_core_alignment == "partial"
        or audit.source_integrity == "partly_forced"
        or audit.causal_consistency == "limited_mismatch"
    ):
        return 1
    return 2


def m_score_from_two_gate(
    literal_audit: MTwoGateLiteralAudit,
    native_audit: MTwoGateNativeAudit,
) -> int:
    """Apply the v7 ordinal rule to independently produced v18 gate evidence."""
    if literal_audit.literal_instance == "yes":
        return 0
    if (
        native_audit.native_relation_match == "yes"
        and native_audit.role_change_degree == "none_or_one"
    ):
        return 1
    return 2


def m_score_from_relation_gate(
    literal_audit: MTwoGateLiteralAudit,
    identity_audit: MRelationIdentityAudit,
) -> int:
    """Apply the v18.2 denotation and native-relation identity boundaries."""
    if literal_audit.literal_instance == "yes":
        return 0
    accepted_relations = {
        "same_native_relation",
        "established_cross_domain_extension",
        "adjacent_technical_relation",
    }
    if (
        identity_audit.native_neighborhood == "yes"
        and identity_audit.relation_status in accepted_relations
        and identity_audit.carrier_compatibility != "incompatible"
        and identity_audit.terminology_test == "independently_supported"
        and identity_audit.gloss_removal_test == "relation_survives"
        and identity_audit.analogy_specific_reinterpretation in {"none", "limited"}
    ):
        return 1
    return 2


def validate_m_two_gate_native_audit(
    audit: MTwoGateNativeAudit,
    target_frame: MTwoGateTargetFrame,
) -> None:
    """Require complete defining-role coverage and a consistent shift total."""
    expected_roles = [
        role.role
        for role in target_frame.essential_roles
        if role.necessity == "defining"
    ]
    returned_roles = [alignment.target_role for alignment in audit.role_alignments]
    if returned_roles != expected_roles:
        raise ValueError(
            "Two-gate native audit must return defining target roles in order: "
            f"{returned_roles!r} != {expected_roles!r}"
        )

    shifts = int(audit.mechanism_independent_shift == "yes") + sum(
        alignment.independent_shift == "yes"
        for alignment in audit.role_alignments
    )
    expected_degree = "none_or_one" if shifts <= 1 else "multiple"
    if audit.role_change_degree != expected_degree:
        raise ValueError(
            "Two-gate shift degree is inconsistent with structured alignments: "
            f"{audit.role_change_degree!r} != {expected_degree!r}"
        )


def m_ordinal_policy_trace(
    judgment: MOrdinalJudgment,
    score: int,
    split: str,
) -> dict[str, Any]:
    """Expose the auditable v7.1 decision without per-item overrides."""
    return {
        "version": M_ORDINAL_POLICY_VERSION,
        "score": score,
        "literal_instance": judgment.literal_instance,
        "native_relation_match": judgment.native_relation_match,
        "role_change_degree": judgment.role_change_degree,
        "central_role_changes": judgment.central_role_changes,
        "judge_recommended_score": judgment.recommended_score,
        "nearest_score_0_anchor_id": judgment.nearest_score_0_anchor_id,
        "nearest_score_1_anchor_id": judgment.nearest_score_1_anchor_id,
        "nearest_score_2_anchor_id": judgment.nearest_score_2_anchor_id,
        "leave_one_out_calibration": split == "validation",
    }


def tcc_coverage_ratio(judgment: TCCJudgment) -> float:
    if not judgment.assessments:
        return 0.0
    return sum(
        TCC_STATUS_VALUE[assessment.status]
        for assessment in judgment.assessments
    ) / len(judgment.assessments)


def validate_tcc_topic_ids(
    judgment: TCCJudgment, expected_topic_ids: list[str]
) -> None:
    returned_topic_ids = [
        assessment.topic_id for assessment in judgment.assessments
    ]
    if returned_topic_ids != expected_topic_ids:
        raise ValueError(
            "TCC assessment topic IDs must exactly match the supplied topics "
            f"in order: expected={expected_topic_ids}, "
            f"returned={returned_topic_ids}"
        )


def validate_topic_importance(
    judgment: TopicImportanceJudgment,
    expected_topic_ids: list[str],
    description: str | None = None,
) -> None:
    returned_topic_ids = [
        assessment.topic_id for assessment in judgment.assessments
    ]
    if returned_topic_ids != expected_topic_ids:
        raise ValueError(
            "Topic-importance assessment IDs must exactly match the supplied "
            f"topics in order: expected={expected_topic_ids}, "
            f"returned={returned_topic_ids}"
        )

    decisions = {
        assessment.topic_id: assessment.decision
        for assessment in judgment.assessments
    }
    kept_topic_ids = {
        topic_id for topic_id, decision in decisions.items()
        if decision == "keep"
    }
    if not kept_topic_ids:
        raise ValueError("TopicImportanceJudge must keep at least one topic")

    normalized_description = (
        normalize_evidence_span(description)
        if description is not None
        else None
    )
    relation_decision = {
        "independent_requirement": "keep",
        "entailed_restatement": "merge",
        "illustrative_example": "contextual_detail",
        "measurement_convention": "contextual_detail",
        "implementation_alternative": "contextual_detail",
    }
    for assessment in judgment.assessments:
        if normalized_description is not None:
            normalized_evidence = normalize_evidence_span(
                assessment.description_evidence
            )
            if not evidence_is_grounded(
                normalized_evidence,
                normalized_description,
            ):
                raise ValueError(
                    f"Topic {assessment.topic_id} description_evidence must "
                    "be an exact DESCRIPTION span; received="
                    f"{assessment.description_evidence!r}"
                )
        required_decision = relation_decision[assessment.relation_to_parent]
        if assessment.decision != required_decision:
            raise ValueError(
                f"Topic {assessment.topic_id} relation "
                f"{assessment.relation_to_parent} requires decision "
                f"{required_decision}; got {assessment.decision}"
            )
        parent_topic_id = assessment.parent_topic_id
        if assessment.decision == "keep":
            if parent_topic_id is not None:
                raise ValueError(
                    f"Kept topic {assessment.topic_id} cannot have a parent"
                )
        else:
            if parent_topic_id is None:
                raise ValueError(
                    f"Filtered topic {assessment.topic_id} requires "
                    "parent_topic_id"
                )
            if parent_topic_id == assessment.topic_id:
                raise ValueError(
                    f"Filtered topic {assessment.topic_id} cannot target itself"
                )
            if parent_topic_id not in kept_topic_ids:
                raise ValueError(
                    f"Filtered topic {assessment.topic_id} must target a kept topic; "
                    f"got {parent_topic_id}"
                )


def normalize_evidence_span(value: str) -> str:
    """Normalize whitespace and typographic variants for exact-span checks."""
    punctuation_translation = str.maketrans(
        {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
        }
    )
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(punctuation_translation)
    normalized = normalized.replace("**", "").replace("__", "")
    normalized = normalized.replace("`", "")
    return " ".join(normalized.split()).casefold()


def evidence_is_grounded(
    normalized_evidence: str,
    normalized_description: str,
) -> bool:
    """Accept one exact span or ordered exact spans joined by an ellipsis."""
    if normalized_evidence in normalized_description:
        return True
    contains_ellipsis = bool(re.search(r"(?:\.\.\.|…)", normalized_evidence))
    parts = [
        part.strip()
        for part in re.split(r"(?:\.\.\.|…)", normalized_evidence)
        if part.strip()
    ]
    if contains_ellipsis and parts:
        search_start = 0
        all_parts_found = True
        for part in parts:
            position = normalized_description.find(part, search_start)
            if position < 0:
                all_parts_found = False
                break
            search_start = position + len(part)
        if all_parts_found:
            return True

    fuzzy_evidence = (
        normalized_evidence.rsplit(":", 1)[-1].strip()
        if ":" in normalized_evidence
        else normalized_evidence
    )
    raw_evidence_words = re.findall(r"[a-z0-9]+", fuzzy_evidence)
    ignored_words = {"a", "an", "the", "of", "s"}
    evidence_words = [
        word for word in raw_evidence_words if word not in ignored_words
    ]
    description_words = [
        word
        for word in re.findall(r"[a-z0-9]+", normalized_description)
        if word not in ignored_words
    ]
    if len(raw_evidence_words) < 4 or len(evidence_words) < 3:
        return False
    previous = [0] * (len(description_words) + 1)
    for evidence_word in evidence_words:
        current = [0]
        for index, description_word in enumerate(description_words, start=1):
            tokens_match = (
                evidence_word == description_word
                or (
                    len(evidence_word) >= 5
                    and len(description_word) >= 5
                    and evidence_word[:5] == description_word[:5]
                )
            )
            if tokens_match:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    ordered_match_ratio = previous[-1] / len(evidence_words)
    return ordered_match_ratio >= 0.85


def validate_coverage_audit(
    judgment: CoverageAuditJudgment,
    expected_statuses: list[tuple[str, str]],
) -> None:
    returned = [
        (assessment.topic_id, assessment.original_status)
        for assessment in judgment.assessments
    ]
    if returned != expected_statuses:
        raise ValueError(
            "Coverage-audit topics and original statuses must exactly match "
            f"the supplied blockers in order: expected={expected_statuses}, "
            f"returned={returned}"
        )


def validate_facet_coverage_audit(
    judgment: FacetCoverageAuditJudgment,
    expected_statuses: list[tuple[str, str]],
    description: str,
) -> None:
    returned = [
        (assessment.topic_id, assessment.original_status)
        for assessment in judgment.assessments
    ]
    if returned != expected_statuses:
        raise ValueError(
            "Facet-audit topics and original statuses must exactly match "
            f"the supplied blockers in order: expected={expected_statuses}, "
            f"returned={returned}"
        )

    normalized_description = normalize_evidence_span(description)
    for assessment in judgment.assessments:
        seen_facets: set[tuple[str, str]] = set()
        for facet in assessment.facets:
            normalized_evidence = normalize_evidence_span(
                facet.description_evidence
            )
            if not evidence_is_grounded(
                normalized_evidence,
                normalized_description,
            ):
                raise ValueError(
                    f"Topic {assessment.topic_id} facet description_evidence "
                    "must be grounded in DESCRIPTION; received="
                    f"{facet.description_evidence!r}"
                )
            facet_key = (
                normalize_evidence_span(facet.facet),
                facet.facet_kind,
            )
            if facet_key in seen_facets:
                raise ValueError(
                    f"Topic {assessment.topic_id} returned duplicate facet "
                    f"{facet.facet!r}"
                )
            seen_facets.add(facet_key)


def facet_audit_to_coverage_audit(
    judgment: FacetCoverageAuditJudgment,
) -> tuple[CoverageAuditJudgment, dict[str, Any]]:
    """Derive conservative upgrade decisions from atomic facet evidence."""
    blocking_kinds = {
        "substantive_function_or_relation",
        "scope_or_constraint",
    }
    allowed_residual_kinds = {
        "category_or_medium",
        "terminology",
        "illustrative_detail",
        "measurement_convention",
    }
    derived_assessments: list[CoverageAuditAssessment] = []
    traces: list[dict[str, Any]] = []
    for assessment in judgment.assessments:
        missing_kinds = {
            facet.facet_kind
            for facet in assessment.facets
            if facet.status == "missing"
        }
        missing_blocking_kinds = sorted(missing_kinds & blocking_kinds)
        missing_allowed_kinds = sorted(
            missing_kinds & allowed_residual_kinds
        )
        unexpected_missing_kinds = sorted(
            missing_kinds - blocking_kinds - allowed_residual_kinds
        )
        upgrade = not missing_blocking_kinds and not unexpected_missing_kinds
        decision = "upgrade_to_covered" if upgrade else "uphold"
        rationale = (
            "All substantive and scope facets are functionally realized; "
            f"residual gaps={missing_allowed_kinds or ['none']}."
            if upgrade
            else "A substantive function, relation, scope, or constraint "
            "remains missing; "
            f"blocking gaps={missing_blocking_kinds}."
        )
        derived_assessments.append(
            CoverageAuditAssessment(
                topic_id=assessment.topic_id,
                original_status=assessment.original_status,
                decision=decision,
                analogy_evidence="; ".join(
                    facet.analogy_evidence
                    for facet in assessment.facets
                ),
                rationale=rationale,
                confidence=assessment.confidence,
            )
        )
        traces.append(
            {
                "topic_id": assessment.topic_id,
                "missing_facet_kinds": sorted(missing_kinds),
                "missing_blocking_kinds": missing_blocking_kinds,
                "allowed_residual_kinds": missing_allowed_kinds,
                "derived_decision": decision,
            }
        )
    return (
        CoverageAuditJudgment(
            assessments=derived_assessments,
            summary="Deterministic decisions derived from facet evidence.",
        ),
        {
            "version": TCC_FACET_COVERAGE_AUDIT_POLICY_VERSION,
            "allowed_residual_kinds": sorted(allowed_residual_kinds),
            "blocking_kinds": sorted(blocking_kinds),
            "assessments": traces,
        },
    )


def refine_tcc_topics(
    decomposition: ConceptDecomposition,
    judgment: TopicImportanceJudgment,
) -> ConceptDecomposition:
    expected_topic_ids = [topic.topic_id for topic in decomposition.topics]
    validate_topic_importance(judgment, expected_topic_ids)
    kept_topic_ids = {
        assessment.topic_id
        for assessment in judgment.assessments
        if assessment.decision == "keep"
    }
    return ConceptDecomposition(
        target_summary=decomposition.target_summary,
        topics=[
            topic
            for topic in decomposition.topics
            if topic.topic_id in kept_topic_ids
        ],
    )


def v1_conservative_tcc_correction(
    original_score: int,
    decomposition: ConceptDecomposition,
    archived_tcc: TCCJudgment,
    topic_importance: TopicImportanceJudgment,
    coverage_audit: CoverageAuditJudgment | None = None,
) -> dict[str, Any]:
    """Replace v1.1's ID overrides with one auditable, sample-agnostic rule.

    The v1 prediction is preserved unless it is 1 and every non-covered v1
    topic is independently classified as overlap or contextual detail. This
    makes the only allowed change a conservative 1 -> 2 promotion.
    """
    if original_score not in {0, 1, 2}:
        raise ValueError(f"original_score must be in 0..2, got {original_score}")

    topic_ids = [topic.topic_id for topic in decomposition.topics]
    validate_tcc_topic_ids(archived_tcc, topic_ids)
    validate_topic_importance(topic_importance, topic_ids)

    decision_by_topic = {
        assessment.topic_id: assessment.decision
        for assessment in topic_importance.assessments
    }
    archived_status_by_topic = {
        assessment.topic_id: assessment.status
        for assessment in archived_tcc.assessments
    }
    retained_topic_ids = [
        topic_id
        for topic_id in topic_ids
        if decision_by_topic[topic_id] == "keep"
    ]
    filtered_topic_ids = [
        topic_id
        for topic_id in topic_ids
        if decision_by_topic[topic_id] != "keep"
    ]
    original_blocking_topic_ids = [
        topic_id
        for topic_id in retained_topic_ids
        if archived_status_by_topic[topic_id] != "covered"
    ]
    audited_status_by_topic = dict(archived_status_by_topic)
    audited_upgraded_topic_ids: list[str] = []
    if coverage_audit is not None:
        expected_statuses = [
            (topic_id, archived_status_by_topic[topic_id])
            for topic_id in original_blocking_topic_ids
        ]
        validate_coverage_audit(coverage_audit, expected_statuses)
        for assessment in coverage_audit.assessments:
            if assessment.decision == "upgrade_to_covered":
                audited_status_by_topic[assessment.topic_id] = "covered"
                audited_upgraded_topic_ids.append(assessment.topic_id)

    blocking_topic_ids = [
        topic_id
        for topic_id in retained_topic_ids
        if audited_status_by_topic[topic_id] != "covered"
    ]

    promote = (
        original_score == 1
        and bool(filtered_topic_ids or audited_upgraded_topic_ids)
        and not blocking_topic_ids
    )
    final_score = 2 if promote else original_score
    return {
        "version": TCC_V1_AUTO_CONSERVATIVE_POLICY_VERSION,
        "original_v1_score": original_score,
        "final_score": final_score,
        "changed": final_score != original_score,
        "decisive_rule": (
            "promote_all_retained_topics_covered"
            if promote
            else "preserve_v1_score"
        ),
        "candidate_topic_ids": topic_ids,
        "retained_topic_ids": retained_topic_ids,
        "filtered_topic_ids": filtered_topic_ids,
        "original_blocking_topic_ids": original_blocking_topic_ids,
        "audited_upgraded_topic_ids": audited_upgraded_topic_ids,
        "blocking_topic_ids": blocking_topic_ids,
        "archived_status_by_topic": archived_status_by_topic,
        "audited_status_by_topic": audited_status_by_topic,
        "coverage_audit_version": (
            TCC_COVERAGE_AUDIT_POLICY_VERSION
            if coverage_audit is not None
            else None
        ),
    }


def tcc_topic_scores(judgment: TCCJudgment) -> dict[str, float]:
    return {
        assessment.topic_id: TCC_STATUS_VALUE[assessment.status]
        for assessment in judgment.assessments
    }


def tcc_score_from_ratio(ratio: float) -> int:
    if ratio <= 0:
        return 0
    if ratio >= TCC_FULL_COVERAGE_THRESHOLD:
        return 2
    return 1


def m_score_from_evidence(
    literal: LiteralInstanceJudgment,
    domain: DomainAnalysis,
    judgment: MJudgment,
) -> int:
    """Apply the v6 fixed M policy to decoupled literal and distance evidence."""
    if (
        literal.behavior_match == "yes"
        and literal.target_scope_match == "yes"
        and literal.literal_instance == "yes"
        and (
            literal.target_scope_type == "general_formal_or_practice"
            or domain.domain_distance in {"same", "related"}
        )
    ):
        return 0

    replaced_role_count = sum(
        alignment.semantic_relation == "replaced"
        for alignment in domain.role_alignments
    )
    if replaced_role_count >= 3:
        return 2
    if (
        literal.native_relation_match == "yes"
        and literal.role_type_preservation == "none_or_one_shift"
    ):
        return 1
    if replaced_role_count >= 2:
        return 2
    if literal.role_type_preservation == "multiple_type_changes":
        return 2
    if judgment.role_translation == "multiple_replacements":
        return 2
    if judgment.perceived_distance == "very_different":
        return 2
    if (
        domain.domain_distance == "different"
        and literal.native_relation_match != "yes"
    ):
        return 2
    if (
        judgment.perceived_distance == "unclear"
        and domain.domain_distance == "different"
        and replaced_role_count >= 1
    ):
        return 2
    return 1


def m_probabilities_from_score(
    score: int, confidence: float
) -> ScoreProbabilities:
    """Create probabilities aligned with the deterministic M decision."""
    if score not in {0, 1, 2}:
        raise ValueError(f"M score must be in 0..2, got {score}")
    primary = min(1.0, max(1 / 3, confidence))
    secondary = (1.0 - primary) / 2
    values = [secondary, secondary, secondary]
    values[score] = primary
    return ScoreProbabilities(
        score_0=values[0],
        score_1=values[1],
        score_2=values[2],
    )


def m_policy_trace(
    literal: LiteralInstanceJudgment,
    domain: DomainAnalysis,
    judgment: MJudgment,
    score: int,
) -> dict[str, Any]:
    """Expose the exact evidence and rule responsible for an M prediction."""
    replaced_role_count = sum(
        alignment.semantic_relation == "replaced"
        for alignment in domain.role_alignments
    )
    if (
        literal.behavior_match == "yes"
        and literal.target_scope_match == "yes"
        and literal.literal_instance == "yes"
        and (
            literal.target_scope_type == "general_formal_or_practice"
            or domain.domain_distance in {"same", "related"}
        )
    ):
        decisive_rule = "literal_instance"
    elif replaced_role_count >= 3:
        decisive_rule = "three_or_more_replaced_roles"
    elif (
        literal.native_relation_match == "yes"
        and literal.role_type_preservation == "none_or_one_shift"
    ):
        decisive_rule = "native_relation_with_preserved_roles"
    elif replaced_role_count >= 2:
        decisive_rule = "two_or_more_replaced_roles"
    elif literal.role_type_preservation == "multiple_type_changes":
        decisive_rule = "literal_judge_multiple_type_changes"
    elif judgment.role_translation == "multiple_replacements":
        decisive_rule = "judge_multiple_replacements"
    elif judgment.perceived_distance == "very_different":
        decisive_rule = "judge_very_different"
    elif (
        domain.domain_distance == "different"
        and literal.native_relation_match != "yes"
    ):
        decisive_rule = "different_domain_without_native_relation"
    elif (
        judgment.perceived_distance == "unclear"
        and domain.domain_distance == "different"
        and replaced_role_count >= 1
    ):
        decisive_rule = "different_domain_with_replaced_role_fallback"
    else:
        decisive_rule = "single_step_or_moderate_distance"
    return {
        "version": PROMPT_VERSION,
        "score": score,
        "decisive_rule": decisive_rule,
        "behavior_match": literal.behavior_match,
        "target_scope_match": literal.target_scope_match,
        "literal_instance": literal.literal_instance,
        "target_scope_type": literal.target_scope_type,
        "native_relation_match": literal.native_relation_match,
        "role_type_preservation": literal.role_type_preservation,
        "replaced_role_count": replaced_role_count,
        "role_translation": judgment.role_translation,
        "domain_distance": domain.domain_distance,
        "perceived_distance": judgment.perceived_distance,
    }


def write_run_outputs(
    results: list[dict[str, Any]], output_dir: Path, split: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{split}_details.jsonl"
    csv_path = output_dir / f"{split}_predictions.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: item["id"]):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "TCC", "MS", "M", "VC", "VA", "VE"]
        )
        writer.writeheader()
        for result in sorted(results, key=lambda item: item["id"]):
            prediction = result["prediction"]
            writer.writerow(
                {
                    "id": result["id"],
                    "TCC": prediction["TCC"],
                    "MS": prediction["MS"],
                    "M": prediction["M"],
                    "VC": 0,
                    "VA": 0,
                    "VE": 0,
                }
            )

    result_ids = sorted(result["id"] for result in results)
    if split == "test" and result_ids == list(range(62)):
        shutil.copyfile(csv_path, output_dir / "submission.csv")
    return jsonl_path, csv_path


def write_tcc_outputs(
    results: list[dict[str, Any]], output_dir: Path, split: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{split}_target_coverage_details.jsonl"
    csv_path = output_dir / f"{split}_target_coverage_predictions.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: item["id"]):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "TCC"])
        writer.writeheader()
        for result in sorted(results, key=lambda item: item["id"]):
            writer.writerow(
                {"id": result["id"], "TCC": result["prediction"]["TCC"]}
            )
    return jsonl_path, csv_path


def write_ms_outputs(
    results: list[dict[str, Any]], output_dir: Path, split: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{split}_mapping_strength_details.jsonl"
    csv_path = output_dir / f"{split}_mapping_strength_predictions.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: item["id"]):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "MS"])
        writer.writeheader()
        for result in sorted(results, key=lambda item: item["id"]):
            writer.writerow(
                {"id": result["id"], "MS": result["prediction"]["MS"]}
            )
    return jsonl_path, csv_path


def write_m_outputs(
    results: list[dict[str, Any]], output_dir: Path, split: str
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{split}_metaphoricity_details.jsonl"
    csv_path = output_dir / f"{split}_metaphoricity_predictions.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in sorted(results, key=lambda item: item["id"]):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "M"])
        writer.writeheader()
        for result in sorted(results, key=lambda item: item["id"]):
            writer.writerow(
                {"id": result["id"], "M": result["prediction"]["M"]}
            )
    return jsonl_path, csv_path


def score_validation(
    results: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gold_by_id = {row["id"]: row for row in validation_rows}
    metrics: dict[str, Any] = {}
    for metric in ("TCC", "MS", "M"):
        y_true = [gold_by_id[result["id"]][metric] for result in results]
        y_pred = [result["prediction"][metric] for result in results]
        if len(y_true) < 2:
            tau = rho = float("nan")
        else:
            tau = float(kendalltau(y_true, y_pred).statistic)
            rho = float(spearmanr(y_true, y_pred).statistic)
        metrics[metric] = {
            "kendall": 0.0 if tau != tau else tau,
            "spearman": 0.0 if rho != rho else rho,
            "gold": y_true,
            "predicted": y_pred,
        }

    metrics["TEXT_AVG"] = {
        "kendall": sum(metrics[name]["kendall"] for name in ("TCC", "MS", "M"))
        / 3,
        "spearman": sum(
            metrics[name]["spearman"] for name in ("TCC", "MS", "M")
        )
        / 3,
    }
    return metrics


def score_tcc_validation(
    results: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gold_by_id = {row["id"]: row for row in validation_rows}
    y_true = [gold_by_id[result["id"]]["TCC"] for result in results]
    y_pred = [result["prediction"]["TCC"] for result in results]
    if len(y_true) < 2:
        tau = rho = float("nan")
    else:
        tau = float(kendalltau(y_true, y_pred).statistic)
        rho = float(spearmanr(y_true, y_pred).statistic)
    return {
        "TCC": {
            "kendall": 0.0 if tau != tau else tau,
            "spearman": 0.0 if rho != rho else rho,
            "accuracy": sum(
                prediction == gold
                for prediction, gold in zip(y_pred, y_true)
            )
            / len(y_true),
            "gold": y_true,
            "predicted": y_pred,
        }
    }


def score_ms_validation(
    results: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gold_by_id = {row["id"]: row for row in validation_rows}
    y_true = [gold_by_id[result["id"]]["MS"] for result in results]
    y_pred = [result["prediction"]["MS"] for result in results]
    if len(y_true) < 2:
        tau = rho = float("nan")
    else:
        tau = float(kendalltau(y_true, y_pred).statistic)
        rho = float(spearmanr(y_true, y_pred).statistic)
    return {
        "MS": {
            "kendall": 0.0 if tau != tau else tau,
            "spearman": 0.0 if rho != rho else rho,
            "accuracy": sum(
                prediction == gold
                for prediction, gold in zip(y_pred, y_true)
            )
            / len(y_true),
            "gold": y_true,
            "predicted": y_pred,
        }
    }


def score_m_validation(
    results: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    gold_by_id = {row["id"]: row for row in validation_rows}
    y_true = [gold_by_id[result["id"]]["M"] for result in results]
    y_pred = [result["prediction"]["M"] for result in results]
    if len(y_true) < 2:
        tau = rho = float("nan")
    else:
        tau = float(kendalltau(y_true, y_pred).statistic)
        rho = float(spearmanr(y_true, y_pred).statistic)
    return {
        "M": {
            "kendall": 0.0 if tau != tau else tau,
            "spearman": 0.0 if rho != rho else rho,
            "accuracy": sum(
                prediction == gold
                for prediction, gold in zip(y_pred, y_true)
            )
            / len(y_true),
            "gold": y_true,
            "predicted": y_pred,
        }
    }


def timed_run(coroutine: Any) -> tuple[Any, float]:
    started = time.monotonic()
    return asyncio.run(coroutine), time.monotonic() - started
