"""Structured contracts for the shared semantic analysis front end.

The front end deliberately keeps target-only and source-only evidence in
separate models.  This prevents analogy content from changing TCC topics and
prevents target terminology from being mistaken for a native source
operation.  A third model records cross-domain alignments after both blind
frames have been created.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .ms_native_schemas import MSImportedDetail


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SharedSemanticRole(StrictModel):
    role_id: str = Field(description="Stable short identifier such as TR1 or SR1.")
    role: str = Field(min_length=1)
    semantic_type: str = Field(min_length=1)
    necessity: Literal["defining", "supporting"]
    evidence: str = Field(min_length=1)


class SharedSemanticRelation(StrictModel):
    relation_id: str = Field(description="Stable short identifier such as R1.")
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    direction_or_order: str = Field(min_length=1)
    necessity: Literal["defining", "supporting"]
    evidence: str = Field(min_length=1)


class SharedTargetTopic(StrictModel):
    topic_id: str = Field(description="Stable short identifier such as T1.")
    topic: str = Field(min_length=1)
    importance: Literal["core", "supporting"]
    decision: Literal["keep", "merge", "contextual_detail"]
    relation_to_parent: Literal[
        "independent_requirement",
        "entailed_restatement",
        "illustrative_example",
        "measurement_convention",
        "implementation_alternative",
    ]
    parent_topic_id: str | None = None
    description_evidence: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class SharedTargetFrame(StrictModel):
    target_concept: str = Field(min_length=1)
    target_domain: str = Field(min_length=1)
    target_summary: str = Field(min_length=1)
    defining_mechanism: str = Field(min_length=1)
    success_condition: str = Field(min_length=1)
    topics: list[SharedTargetTopic] = Field(min_length=2, max_length=6)
    roles: list[SharedSemanticRole] = Field(min_length=1, max_length=8)
    relations: list[SharedSemanticRelation] = Field(min_length=1, max_length=8)
    constraints: list[str] = Field(max_length=6)


class SharedSourceFrame(StrictModel):
    source_concept: str = Field(min_length=1)
    literal_source_domain: str = Field(min_length=1)
    source_ontology: Literal[
        "ordinary_real_world",
        "explicitly_fictional_rule_system",
        "target_relabeling_only",
    ]
    fictional_mechanism_coherence: Literal["yes", "no", "not_applicable"]
    literal_source_summary: str = Field(min_length=1)
    ordinary_source_goal: str = Field(min_length=1)
    native_mechanism: str = Field(min_length=1)
    roles: list[SharedSemanticRole] = Field(min_length=1, max_length=8)
    relations: list[SharedSemanticRelation] = Field(min_length=1, max_length=8)
    removed_mapping_language: list[str] = Field(max_length=8)
    imported_target_details: list[MSImportedDetail] = Field(max_length=8)
    source_story_coherence: Literal["coherent", "partly_forced", "incoherent"]


class SharedAlignment(StrictModel):
    source_item: str = Field(min_length=1)
    target_item: str = Field(min_length=1)
    alignment_kind: Literal["concept", "role", "relation", "operation", "constraint"]
    relation: str = Field(min_length=1)
    preservation: Literal["preserved", "partial", "replaced", "contradicted"]
    central: bool
    evidence: str = Field(min_length=1)


class SharedMappingFrame(StrictModel):
    source_concept: str = Field(min_length=1)
    target_concept: str = Field(min_length=1)
    shared_process: str = Field(min_length=1)
    domain_distance: Literal["same", "related", "different", "unclear"]
    alignments: list[SharedAlignment] = Field(min_length=1, max_length=12)
    potential_breaks: list[str] = Field(max_length=8)


class SharedSemanticAnalysis(StrictModel):
    target_frame: SharedTargetFrame
    source_frame: SharedSourceFrame
    mapping_frame: SharedMappingFrame
