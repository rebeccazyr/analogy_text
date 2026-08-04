"""Structured contracts for conservative MS correction over the v1 judge."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MSClaimCorrectionAudit(StrictModel):
    mapping_index: int = Field(ge=0)
    importance: Literal["primary", "supporting"]
    native_status: Literal[
        "sound",
        "limited_stretch",
        "imported_only",
        "absent",
        "contradicted",
        "impossible",
    ]
    causal_role_preserved: Literal["yes", "partly", "no"]
    issue_type: Literal[
        "none",
        "terminology_precision",
        "auxiliary_sequence_statement",
        "cross_mechanism_stretch",
        "physical_feasibility",
        "core_relation_mismatch",
        "target_injection",
    ]
    evidence: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class MSConservativeCorrectionAudit(StrictModel):
    claim_audits: list[MSClaimCorrectionAudit] = Field(min_length=1)
    primary_mapping_indices: list[int] = Field(min_length=1)
    core_mapping_spine: Literal["sound", "mixed", "broken"]
    target_construction_dependency: Literal["none", "supporting", "essential"]
    baseline_issue_scope: Literal["none", "auxiliary", "core"]
    decisive_failure: Literal[
        "none",
        "target_injection",
        "missing_core_mechanism",
        "reversed_core_relation",
        "impossible_source_operation",
    ]
    promotion_safe: Literal["yes", "no"]
    strongest_native_evidence: str = Field(min_length=1)
    strongest_failure_evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class MSZeroGateAudit(StrictModel):
    defining_target_operation: str = Field(min_length=1)
    literal_source_remainder: str = Field(min_length=1)
    native_structural_support: Literal[
        "substantial",
        "limited",
        "thematic_only",
        "none",
    ]
    target_import_dependency: Literal["none", "supporting", "decisive"]
    core_relation: Literal["consistent", "limited", "broken"]
    counterfactual_result: Literal["intact", "weakened", "collapses"]
    self_reference_target: Literal["yes", "no"]
    same_process_on_smaller_instance: Literal["yes", "no", "not_applicable"]
    native_nesting_or_self_reference: Literal["yes", "no", "not_applicable"]
    linear_handoff_only: Literal["yes", "no", "not_applicable"]
    failure_type: Literal[
        "none",
        "target_injection",
        "missing_core_mechanism",
        "reversed_core_relation",
        "impossible_source_operation",
    ]
    strongest_surviving_evidence: str = Field(min_length=1)
    strongest_failure_evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
