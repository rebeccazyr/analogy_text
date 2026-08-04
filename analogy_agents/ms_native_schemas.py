"""Structured contracts for the native-source-integrity MS experiment."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MSCoreRequirement(StrictModel):
    requirement: str = Field(min_length=1)
    role_or_operation: Literal["role", "relation", "operation", "causal_order"]
    necessity: Literal["defining", "supporting"]


class MSTargetMechanismFrame(StrictModel):
    target_summary: str = Field(min_length=1)
    defining_mechanism: str = Field(min_length=1)
    success_condition: str = Field(min_length=1)
    core_requirements: list[MSCoreRequirement] = Field(min_length=1, max_length=5)
    non_defining_details: list[str] = Field(max_length=5)


class MSImportedDetail(StrictModel):
    detail: str = Field(min_length=1)
    why_not_native: str = Field(min_length=1)
    import_kind: Literal[
        "label_or_gloss",
        "property_claim",
        "generic_operation",
        "formal_calculation",
        "self_referential_operation",
        "causal_relation",
    ]
    dependency: Literal["decorative", "supporting", "essential"]


class MSBlindSourceFrame(StrictModel):
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
    native_roles_and_operations: list[str] = Field(min_length=1, max_length=8)
    removed_mapping_language: list[str] = Field(max_length=8)
    imported_target_details: list[MSImportedDetail] = Field(max_length=8)
    source_story_coherence: Literal["coherent", "partly_forced", "incoherent"]


class MSRequirementAlignment(StrictModel):
    target_requirement: str = Field(min_length=1)
    source_support: str = Field(min_length=1)
    status: Literal[
        "native_full",
        "native_partial",
        "imported_only",
        "absent",
        "contradicted",
    ]
    rationale: str = Field(min_length=1)


class MSNativeIntegrityAudit(StrictModel):
    requirement_alignments: list[MSRequirementAlignment] = Field(
        min_length=1,
        max_length=5,
    )
    native_core_alignment: Literal["full", "partial", "none"]
    source_integrity: Literal["independent", "partly_forced", "target_constructed"]
    causal_consistency: Literal["consistent", "limited_mismatch", "contradictory"]
    decisive_failure: Literal[
        "none",
        "target_injection",
        "missing_core_mechanism",
        "reversed_core_relation",
        "impossible_source_operation",
    ]
    strongest_positive_evidence: str = Field(min_length=1)
    strongest_negative_evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
