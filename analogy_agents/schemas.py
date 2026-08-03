from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Topic(StrictModel):
    topic_id: str = Field(description="Stable short identifier such as T1.")
    topic: str = Field(
        description="One independently scorable topic from the provided description."
    )
    importance: Literal["core", "supporting"]


class ConceptDecomposition(StrictModel):
    target_summary: str
    topics: list[Topic] = Field(min_length=1, max_length=6)


class TopicImportanceAssessment(StrictModel):
    topic_id: str
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


class TopicImportanceJudgment(StrictModel):
    assessments: list[TopicImportanceAssessment] = Field(min_length=1)
    summary: str


class CoverageAuditAssessment(StrictModel):
    topic_id: str
    original_status: Literal["partial", "absent"]
    decision: Literal["uphold", "upgrade_to_covered"]
    analogy_evidence: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class CoverageAuditJudgment(StrictModel):
    assessments: list[CoverageAuditAssessment] = Field(min_length=1)
    summary: str


class CoverageFacetAssessment(StrictModel):
    facet: str = Field(min_length=1)
    facet_kind: Literal[
        "substantive_function_or_relation",
        "scope_or_constraint",
        "category_or_medium",
        "terminology",
        "illustrative_detail",
        "measurement_convention",
    ]
    status: Literal["covered", "missing"]
    description_evidence: str = Field(min_length=1)
    analogy_evidence: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class FacetCoverageAuditAssessment(StrictModel):
    topic_id: str
    original_status: Literal["partial", "absent"]
    facets: list[CoverageFacetAssessment] = Field(min_length=1, max_length=8)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class FacetCoverageAuditJudgment(StrictModel):
    assessments: list[FacetCoverageAuditAssessment] = Field(min_length=1)
    summary: str


class MappingRecord(StrictModel):
    source_element: str
    target_element: str
    relation: str
    evidence: str


class MappingAnalysis(StrictModel):
    source_concept: str
    target_concept: str
    mappings: list[MappingRecord]
    process_summary: str
    potential_breaks: list[str]


class RoleAlignment(StrictModel):
    source_role: str
    target_role: str
    semantic_relation: Literal["preserved", "replaced"]
    rationale: str


class DomainAnalysis(StrictModel):
    literal_source_summary: str
    source_concept: str
    source_mechanism: str
    target_concept: str
    target_signature: str
    source_domain: str
    target_domain: str
    domain_distance: Literal["same", "related", "different", "unclear"]
    role_alignments: list[RoleAlignment] = Field(max_length=8)


class LiteralInstanceJudgment(StrictModel):
    literal_source_mechanism: str
    target_defining_mechanism: str
    target_scope_type: Literal[
        "general_formal_or_practice",
        "domain_specific",
        "unclear",
    ]
    behavior_match: Literal["yes", "no", "unclear"]
    target_scope_match: Literal["yes", "no", "unclear"]
    literal_instance: Literal["yes", "no", "unclear"]
    native_relation_match: Literal["yes", "no", "unclear"]
    role_type_preservation: Literal[
        "none_or_one_shift",
        "multiple_type_changes",
        "unclear",
    ]
    evidence: str
    confidence: float = Field(ge=0, le=1)


class ScoreProbabilities(StrictModel):
    score_0: float = Field(ge=0, le=1)
    score_1: float = Field(ge=0, le=1)
    score_2: float = Field(ge=0, le=1)

    def normalized(self) -> dict[int, float]:
        values = [self.score_0, self.score_1, self.score_2]
        total = sum(values)
        if total <= 0:
            return {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
        return {index: value / total for index, value in enumerate(values)}

    def expected_score(self) -> float:
        probabilities = self.normalized()
        return probabilities[1] + 2 * probabilities[2]


class CoverageAssessment(StrictModel):
    topic_id: str
    evidence: str
    status: Literal["absent", "partial", "covered"]


class TCCJudgment(StrictModel):
    assessments: list[CoverageAssessment] = Field(min_length=1)


class MappingAssessment(StrictModel):
    source_element: str
    target_element: str
    judgment: Literal["sound", "stretch", "inconsistent"]
    rationale: str


class MSJudgment(StrictModel):
    assessments: list[MappingAssessment]
    structural_issues: list[str]
    score_probabilities: ScoreProbabilities
    recommended_score: Literal[0, 1, 2]
    confidence: float = Field(ge=0, le=1)
    rationale: str


class MJudgment(StrictModel):
    role_translation: Literal[
        "none",
        "single_shift",
        "multiple_replacements",
        "unclear",
    ]
    role_translation_evidence: str
    perceived_distance: Literal[
        "very_similar",
        "moderately_similar",
        "very_different",
        "unclear",
    ]
    perceived_distance_evidence: str
    confidence: float = Field(ge=0, le=1)


class MOrdinalJudgment(StrictModel):
    literal_source_summary: str
    literal_instance: Literal["yes", "no", "unclear"]
    native_relation_match: Literal["yes", "no", "unclear"]
    central_role_changes: list[str] = Field(max_length=6)
    role_change_degree: Literal[
        "none_or_one",
        "multiple",
        "unclear",
    ]
    nearest_score_0_anchor_id: str
    nearest_score_1_anchor_id: str
    nearest_score_2_anchor_id: str
    score_probabilities: ScoreProbabilities
    recommended_score: Literal[0, 1, 2]
    confidence: float = Field(ge=0, le=1)
    rationale: str


class TaxonomyRoleMapping(StrictModel):
    source_role: str = Field(min_length=1)
    target_role: str = Field(min_length=1)
    source_entity_path: str = Field(min_length=1)
    target_entity_path: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class MTaxonomyJudgment(StrictModel):
    literal_source_summary: str = Field(min_length=1)
    applicability: Literal[
        "instance_of",
        "implementation_of",
        "application_of",
        "measurement_of",
        "specialization_of",
        "adjacent_native_relation",
        "metaphorical_projection",
        "unclear",
    ]
    applicability_evidence: str = Field(min_length=1)
    source_domain_path: str = Field(min_length=1)
    source_relation_path: str = Field(min_length=1)
    target_relation_path: str = Field(min_length=1)
    role_mappings: list[TaxonomyRoleMapping] = Field(
        min_length=2,
        max_length=6,
    )
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class MTaxonomyFinalJudgment(StrictModel):
    domain_assessment: Literal[
        "literal_or_same_scope",
        "related_scope",
        "different_scope",
        "unclear",
    ]
    relation_assessment: Literal[
        "same_mechanism",
        "same_relation_family",
        "different_mechanism",
        "unclear",
    ]
    role_assessment: Literal[
        "roles_preserved",
        "limited_type_shift",
        "multiple_cross_kind_shifts",
        "unclear",
    ]
    score_probabilities: ScoreProbabilities
    recommended_score: Literal[0, 1, 2]
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class MConceptualDistanceJudgment(StrictModel):
    source_concept_summary: str = Field(min_length=1)
    target_concept_summary: str = Field(min_length=1)
    shared_conceptual_core: str = Field(min_length=1)
    required_reinterpretation: Literal[
        "minimal",
        "moderate",
        "substantial",
        "unclear",
    ]
    score_probabilities: ScoreProbabilities
    recommended_score: Literal[0, 1, 2]
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class MConceptualDistanceCritique(StrictModel):
    native_relationship: Literal[
        "same_concept",
        "recognized_conceptual_neighbors",
        "analogy_constructed_similarity",
        "unclear",
    ]
    shared_core_specificity: Literal[
        "concept_defining",
        "family_level",
        "generic_structural_pattern",
        "unclear",
    ]
    abstraction_dependency: Literal[
        "low",
        "moderate",
        "high",
        "unclear",
    ]
    provisional_score_concern: Literal[
        "none",
        "possibly_too_low",
        "possibly_too_high",
        "unclear",
    ]
    critique: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class MOperationRoleMapping(StrictModel):
    source_role: str = Field(min_length=1)
    target_role: str = Field(min_length=1)
    source_role_type: str = Field(min_length=1)
    target_role_type: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class MOperationAnalysis(StrictModel):
    literal_source_summary: str = Field(min_length=1)
    smallest_source_operation: str = Field(min_length=1)
    source_operation_purpose: str = Field(min_length=1)
    target_defining_operation: str = Field(min_length=1)
    target_operation_purpose: str = Field(min_length=1)
    central_role_mappings: list[MOperationRoleMapping] = Field(
        min_length=1,
        max_length=6,
    )
    evidence: str = Field(min_length=1)


class MLiteralApplicabilityAdvocacy(StrictModel):
    applicability_relation: Literal[
        "instance",
        "implementation",
        "application",
        "measurement",
        "specialization",
        "nonliteral_resemblance",
        "unclear",
    ]
    defining_behavior_match: Literal["yes", "no", "unclear"]
    scope_compatibility: Literal[
        "inside_target_scope",
        "valid_realization_of_general_target",
        "outside_target_scope",
        "unclear",
    ]
    embodiment_or_medium_only_difference: Literal[
        "yes",
        "no",
        "unclear",
    ]
    literal_case_strength: Literal["strong", "plausible", "none", "unclear"]
    strongest_literal_case: str = Field(min_length=1)
    contrary_evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class MNativeRelationCritique(StrictModel):
    boundary_relevance: Literal[
        "literal_case_may_control",
        "nonliteral_distance_boundary",
        "unclear",
    ]
    native_relation: Literal[
        "same_native_relation",
        "neighboring_native_relation",
        "analogy_imposed_relation",
        "unclear",
    ]
    role_alignment: Literal[
        "roles_preserved",
        "limited_embodiment_shift",
        "multiple_type_substitutions",
        "unclear",
    ]
    shared_core_specificity: Literal[
        "concept_defining",
        "family_level",
        "generic_structure",
        "unclear",
    ]
    reinterpretation_burden: Literal[
        "minimal",
        "moderate",
        "substantial",
        "unclear",
    ]
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class MNativeSourceFrame(StrictModel):
    literal_source_account: str = Field(min_length=1)
    native_source_concept: str = Field(min_length=1)
    native_source_category: str = Field(min_length=1)
    ordinary_source_purpose: str = Field(min_length=1)
    constitutive_source_operation: str = Field(min_length=1)
    operation_status_in_source: Literal[
        "native_and_constitutive",
        "native_but_incidental",
        "analogy_staged_or_reframed",
        "unclear",
    ]
    target_semantic_kind: Literal[
        "algorithm_or_formal_procedure",
        "system_or_artifact_class",
        "property_or_phenomenon",
        "relation_or_practice",
        "unclear",
    ]
    target_carrier_requirement: Literal[
        "medium_independent",
        "requires_specific_bearer_or_domain",
        "source_bearer_entails_target_class",
        "unclear",
    ]
    literal_source_realization: str = Field(min_length=1)
    source_realization_kind: Literal[
        "enacted_procedure",
        "system_or_artifact",
        "property_or_state",
        "relation_or_practice",
        "unclear",
    ]
    target_concept_summary: str = Field(min_length=1)
    target_ordinary_scope: str = Field(min_length=1)
    target_defining_commitments: list[str] = Field(min_length=1, max_length=6)
    realization_commitment_evidence: list[str] = Field(
        min_length=1,
        max_length=6,
    )
    central_role_mappings: list[MOperationRoleMapping] = Field(
        min_length=1,
        max_length=6,
    )
    removed_author_glosses: list[str] = Field(max_length=6)
    evidence: str = Field(min_length=1)


class MLiteralScopeAudit(StrictModel):
    ordinary_denotation_test: Literal[
        "source_entails_target",
        "compatible_but_not_entailing",
        "outside_target_denotation",
        "unclear",
    ]
    target_term_substitution: Literal[
        "natural_literal_description",
        "specialized_or_debatable_description",
        "metaphorical_or_false_description",
        "unclear",
    ]
    defining_commitments: Literal[
        "all_preserved",
        "partly_preserved",
        "not_preserved",
        "unclear",
    ]
    native_scope_compatibility: Literal[
        "inside_ordinary_target_scope",
        "valid_cross_medium_realization",
        "analogy_constructed_resemblance",
        "outside_target_scope",
        "unclear",
    ]
    strongest_literal_evidence: str = Field(min_length=1)
    literal_falsification: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class MNativeNeighborhoodAudit(StrictModel):
    pre_analogy_association: Literal[
        "direct_conceptual_neighbors",
        "distant_but_related",
        "not_native_neighbors",
        "unclear",
    ]
    shared_parent_specificity: Literal[
        "concept_specific",
        "established_relation_family",
        "broad_functional_abstraction",
        "analogy_constructed",
        "unclear",
    ]
    role_frame_preservation: Literal[
        "preserved",
        "limited_natural_shift",
        "substantial_ontological_substitution",
        "unclear",
    ]
    target_specific_commitments_survive: Literal[
        "yes",
        "partly",
        "no",
        "unclear",
    ]
    strongest_neighbor_evidence: str = Field(min_length=1)
    neighbor_falsification: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class MNativeScopeJudgment(StrictModel):
    literal_target_entailment: Literal["supported", "refuted", "unclear"]
    native_conceptual_neighborhood: Literal[
        "supported",
        "refuted",
        "unclear",
    ]
    abstraction_requirement: Literal[
        "none",
        "limited",
        "substantial",
        "unclear",
    ]
    score_0_counterfactual: str = Field(min_length=1)
    score_1_counterfactual: str = Field(min_length=1)
    score_2_counterfactual: str = Field(min_length=1)
    score_probabilities: ScoreProbabilities
    recommended_score: Literal[0, 1, 2]
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class MPairwiseDistanceComparison(StrictModel):
    a_literal_source_summary: str = Field(min_length=1)
    b_literal_source_summary: str = Field(min_length=1)
    decisive_contrast: str = Field(min_length=1)
    more_conceptually_distant: Literal["a", "b", "tie"]
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class MGlobalLiteralBoundary(StrictModel):
    rank_count: int = Field(ge=1)
    m0_end_rank: int
    boundary_item_ids: list[int] = Field(max_length=8)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class MGlobalSubstantialBoundary(StrictModel):
    rank_count: int = Field(ge=1)
    m2_start_rank: int
    boundary_item_ids: list[int] = Field(max_length=8)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class MGlobalTierJudgment(StrictModel):
    rank_count: int = Field(ge=1)
    m0_end_rank: int
    m1_end_rank: int
    m0_boundary_rationale: str = Field(min_length=1)
    m1_m2_boundary_rationale: str = Field(min_length=1)
    ambiguous_item_ids: list[int] = Field(max_length=12)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class MCodexOrdinalBatchItem(MOrdinalJudgment):
    id: int = Field(ge=0)


class MCodexOrdinalBatch(StrictModel):
    results: list[MCodexOrdinalBatchItem] = Field(min_length=1, max_length=16)


class MTwoGateSemanticRole(StrictModel):
    role: str = Field(min_length=1)
    semantic_kind: str = Field(min_length=1)
    necessity: Literal["defining", "supporting"]


class MTwoGateTargetFrame(StrictModel):
    target_summary: str = Field(min_length=1)
    semantic_kind: Literal[
        "algorithm_or_formal_procedure",
        "system_or_artifact_class",
        "property_or_phenomenon",
        "relation_or_practice",
        "unclear",
    ]
    ordinary_scope: str = Field(min_length=1)
    defining_mechanism: str = Field(min_length=1)
    carrier_requirement: Literal[
        "medium_independent",
        "domain_or_bearer_specific",
        "mixed_or_context_dependent",
        "unclear",
    ]
    essential_roles: list[MTwoGateSemanticRole] = Field(
        min_length=1,
        max_length=4,
    )
class MTwoGateSourceFrame(StrictModel):
    literal_source_account: str = Field(min_length=1)
    source_concept: str = Field(min_length=1)
    semantic_kind: str = Field(min_length=1)
    ordinary_source_purpose: str = Field(min_length=1)
    native_mechanism: str = Field(min_length=1)
    central_roles: list[MTwoGateSemanticRole] = Field(
        min_length=1,
        max_length=4,
    )
    removed_mapping_language: list[str] = Field(max_length=6)


class MTwoGateLiteralAudit(StrictModel):
    ordinary_denotation_test: Literal[
        "source_entails_target",
        "compatible_but_not_entailing",
        "outside_target_denotation",
        "unclear",
    ]
    target_term_substitution: Literal[
        "natural_literal_description",
        "specialized_or_debatable_description",
        "metaphorical_or_false_description",
        "unclear",
    ]
    defining_mechanism_match: Literal[
        "same_mechanism",
        "partial_or_abstract_similarity",
        "different_mechanism",
        "unclear",
    ]
    literal_instance: Literal["yes", "no", "unclear"]
    strongest_literal_evidence: str = Field(min_length=1)
    literal_falsification: str = Field(min_length=1)
    nearest_score_0_anchor_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class MTwoGateRoleAlignment(StrictModel):
    target_role: str = Field(min_length=1)
    source_role: str = Field(min_length=1)
    alignment: Literal[
        "ordinary_meaning_preserved",
        "limited_embodiment_or_context_shift",
        "different_semantic_kind",
        "missing_target_role",
        "unclear",
    ]
    independent_shift: Literal["yes", "no", "unclear"]
    evidence: str = Field(min_length=1)


class MTwoGateNativeAudit(StrictModel):
    pre_analogy_relation: Literal[
        "direct_native_neighbors",
        "recognized_relation_family",
        "generic_shared_pattern",
        "analogy_constructed",
        "unclear",
    ]
    shared_native_relation: str = Field(min_length=1)
    relation_specificity: Literal[
        "concept_specific",
        "specific_relation_family",
        "broad_function_or_purpose",
        "unclear",
    ]
    mechanism_alignment: Literal[
        "ordinary_meaning_preserved",
        "limited_embodiment_or_context_shift",
        "different_semantic_kind",
        "unclear",
    ]
    mechanism_independent_shift: Literal["yes", "no", "unclear"]
    mechanism_evidence: str = Field(min_length=1)
    role_alignments: list[MTwoGateRoleAlignment] = Field(
        min_length=1,
        max_length=4,
    )
    role_change_degree: Literal[
        "none_or_one",
        "multiple",
        "unclear",
    ]
    native_relation_match: Literal["yes", "no", "unclear"]
    strongest_m1_evidence: str = Field(min_length=1)
    strongest_m2_evidence: str = Field(min_length=1)
    nearest_score_1_anchor_id: str = Field(min_length=1)
    nearest_score_2_anchor_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class MRelationTargetFrame(StrictModel):
    target_summary: str = Field(min_length=1)
    broad_target_scope: str = Field(min_length=1)
    description_specific_scope: str = Field(min_length=1)
    semantic_kind: Literal[
        "algorithm_or_formal_procedure",
        "system_or_artifact_class",
        "property_or_phenomenon",
        "relation_or_practice",
        "unclear",
    ]
    defining_relation: str = Field(min_length=1)
    relation_invariants: list[str] = Field(min_length=1, max_length=5)
    carrier_constraint: Literal[
        "medium_independent",
        "domain_or_bearer_specific",
        "established_cross_disciplinary_concept",
        "mixed_or_context_dependent",
        "unclear",
    ]
    carrier_evidence: str = Field(min_length=1)
    ordinary_relation_terms: list[str] = Field(min_length=1, max_length=5)


class MRelationSourceFrame(StrictModel):
    literal_source_account: str = Field(min_length=1)
    native_source_concept: str = Field(min_length=1)
    source_carrier_or_domain: str = Field(min_length=1)
    native_relation: str = Field(min_length=1)
    native_relation_invariants: list[str] = Field(min_length=1, max_length=5)
    ordinary_source_terms: list[str] = Field(min_length=1, max_length=5)
    removed_mapping_language: list[str] = Field(max_length=6)


class MRelationIdentityAudit(StrictModel):
    relation_status: Literal[
        "same_native_relation",
        "established_cross_domain_extension",
        "adjacent_technical_relation",
        "generic_functional_similarity",
        "analogy_constructed_relation",
        "unclear",
    ]
    carrier_compatibility: Literal[
        "compatible",
        "recognized_extension",
        "incompatible",
        "unclear",
    ]
    terminology_test: Literal[
        "independently_supported",
        "only_broad_paraphrase_supported",
        "not_supported",
        "unclear",
    ]
    terminology_evidence: str = Field(min_length=1)
    gloss_removal_test: Literal[
        "relation_survives",
        "relation_weakens_to_generic_pattern",
        "relation_collapses",
        "unclear",
    ]
    invariant_matches: list[str] = Field(max_length=5)
    invariant_failures: list[str] = Field(max_length=5)
    analogy_specific_reinterpretation: Literal[
        "none",
        "limited",
        "substantial",
        "unclear",
    ]
    native_neighborhood: Literal["yes", "no", "unclear"]
    strongest_m1_evidence: str = Field(min_length=1)
    strongest_m2_evidence: str = Field(min_length=1)
    nearest_score_1_anchor_id: str = Field(min_length=1)
    nearest_score_2_anchor_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
