from __future__ import annotations

import json
import unittest
from pathlib import Path

from analogy_agents.pipeline import (
    facet_audit_to_coverage_audit,
    load_frozen_original_ms,
    load_split,
    m_score_from_ordinal,
    ms_score_from_zero_gate,
    shared_analysis_to_domain,
    shared_mapping_to_original,
    shared_source_to_ms_blind_source,
    shared_target_to_topic_importance,
    shared_target_to_v1_decomposition,
    validate_shared_target_frame,
    v1_conservative_tcc_correction,
)
from analogy_agents.ms_corrective_schemas import MSZeroGateAudit
from analogy_agents.ms_native_schemas import MSBlindSourceFrame, MSImportedDetail
from analogy_agents.prompts import M_CALIBRATION_ANCHORS, m_calibration_anchors
from analogy_agents.schemas import (
    ConceptDecomposition,
    CoverageAssessment,
    CoverageFacetAssessment,
    FacetCoverageAuditAssessment,
    FacetCoverageAuditJudgment,
    MOrdinalJudgment,
    ScoreProbabilities,
    TCCJudgment,
    Topic,
    TopicImportanceAssessment,
    TopicImportanceJudgment,
)
from analogy_agents.shared_schemas import (
    SharedAlignment,
    SharedMappingFrame,
    SharedSemanticAnalysis,
    SharedSemanticRelation,
    SharedSemanticRole,
    SharedSourceFrame,
    SharedTargetFrame,
    SharedTargetTopic,
)
from analogy_agents.shared_prompts import (
    shared_source_frame_prompt,
    shared_target_frame_prompt,
)


class ActivePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validation = load_split(Path("challenge-dataset"), "validation")
        cls.test = load_split(Path("challenge-dataset"), "test")

    def test_text_splits_are_complete(self) -> None:
        self.assertEqual(len(self.validation), 12)
        self.assertEqual(len(self.test), 62)

    def test_zero_gate_loads_tracked_v1_evidence(self) -> None:
        validation = load_frozen_original_ms("validation")
        test = load_frozen_original_ms("test")
        self.assertEqual(len(validation), 12)
        self.assertEqual(len(test), 62)
        self.assertEqual(test[15]["target"].lower(), "recursion")
        self.assertEqual(
            test[15]["agents"]["ms_judge_v1_exact"]["recommended_score"],
            1,
        )

    def test_ms_zero_gate_catches_missing_recursive_identity(self) -> None:
        audit = self._zero_gate_audit(
            counterfactual_result="collapses",
            self_reference_target="yes",
            same_process_on_smaller_instance="no",
            native_nesting_or_self_reference="no",
            linear_handoff_only="yes",
            failure_type="missing_core_mechanism",
        )
        self.assertEqual(
            ms_score_from_zero_gate(
                1,
                audit,
                self._blind_source_frame(),
                "People pass a request up and a result back down.",
            ),
            0,
        )

    def test_ms_zero_gate_catches_decisive_formal_import(self) -> None:
        audit = self._zero_gate_audit(
            native_structural_support="limited",
            target_import_dependency="decisive",
            counterfactual_result="weakened",
        )
        source_frame = self._blind_source_frame(
            imported_target_details=[
                MSImportedDetail(
                    detail="gradient calculation",
                    why_not_native="Workers do not calculate derivatives.",
                    import_kind="formal_calculation",
                    dependency="essential",
                )
            ]
        )
        self.assertEqual(
            ms_score_from_zero_gate(
                1,
                audit,
                source_frame,
                "The feedback is described as a gradient and derivative.",
            ),
            0,
        )

    def test_ms_zero_gate_preserves_a_usable_native_mechanism(self) -> None:
        audit = self._zero_gate_audit(
            native_structural_support="limited",
            counterfactual_result="weakened",
        )
        self.assertEqual(
            ms_score_from_zero_gate(
                1,
                audit,
                self._blind_source_frame(),
                "Items can be added to and removed from a physical cart.",
            ),
            1,
        )

    def test_tcc_correction_promotes_only_a_clear_score_one(self) -> None:
        decomposition = ConceptDecomposition(
            target_summary="A mechanism with one function and one example.",
            topics=[
                Topic(topic_id="T1", topic="Core function", importance="core"),
                Topic(
                    topic_id="T2",
                    topic="Illustrative example",
                    importance="supporting",
                ),
            ],
        )
        original = TCCJudgment(
            assessments=[
                CoverageAssessment(
                    topic_id="T1", evidence="Function present", status="covered"
                ),
                CoverageAssessment(
                    topic_id="T2", evidence="Example omitted", status="absent"
                ),
            ]
        )
        importance = TopicImportanceJudgment(
            assessments=[
                TopicImportanceAssessment(
                    topic_id="T1",
                    decision="keep",
                    relation_to_parent="independent_requirement",
                    description_evidence="Core function",
                    rationale="Defining behavior.",
                    confidence=0.95,
                ),
                TopicImportanceAssessment(
                    topic_id="T2",
                    decision="contextual_detail",
                    relation_to_parent="illustrative_example",
                    parent_topic_id="T1",
                    description_evidence="Illustrative example",
                    rationale="Non-exhaustive example.",
                    confidence=0.95,
                ),
            ],
            summary="Keep the function and filter the example.",
        )

        promoted = v1_conservative_tcc_correction(
            1, decomposition, original, importance
        )
        preserved = v1_conservative_tcc_correction(
            2, decomposition, original, importance
        )

        self.assertEqual(promoted["final_score"], 2)
        self.assertEqual(promoted["decisive_rule"], "promote_all_retained_topics_covered")
        self.assertEqual(preserved["final_score"], 2)
        self.assertFalse(preserved["changed"])

    def test_tcc_facet_policy_ignores_category_only_gap(self) -> None:
        judgment = FacetCoverageAuditJudgment(
            assessments=[
                FacetCoverageAuditAssessment(
                    topic_id="T1",
                    original_status="partial",
                    facets=[
                        CoverageFacetAssessment(
                            facet="Performs the defining function",
                            facet_kind="substantive_function_or_relation",
                            status="covered",
                            description_evidence="defining function",
                            analogy_evidence="The source performs the function.",
                            rationale="The behavior is present.",
                        ),
                        CoverageFacetAssessment(
                            facet="Uses the technical category name",
                            facet_kind="category_or_medium",
                            status="missing",
                            description_evidence="technical category",
                            analogy_evidence="Only the name is absent.",
                            rationale="No substantive content is missing.",
                        ),
                    ],
                    rationale="Only a category label remains.",
                    confidence=0.9,
                )
            ],
            summary="Category-only residual gap.",
        )
        derived, trace = facet_audit_to_coverage_audit(judgment)
        self.assertEqual(derived.assessments[0].decision, "upgrade_to_covered")
        self.assertEqual(
            trace["assessments"][0]["missing_facet_kinds"],
            ["category_or_medium"],
        )

    def test_tcc_facet_policy_preserves_scope_gap(self) -> None:
        judgment = FacetCoverageAuditJudgment(
            assessments=[
                FacetCoverageAuditAssessment(
                    topic_id="T1",
                    original_status="partial",
                    facets=[
                        CoverageFacetAssessment(
                            facet="Restricted operating scope",
                            facet_kind="scope_or_constraint",
                            status="missing",
                            description_evidence="only in the restricted scope",
                            analogy_evidence="No restriction is represented.",
                            rationale="The condition changes when the concept applies.",
                        )
                    ],
                    rationale="A scope condition is absent.",
                    confidence=0.9,
                )
            ],
            summary="Scope gap.",
        )
        derived, _ = facet_audit_to_coverage_audit(judgment)
        self.assertEqual(derived.assessments[0].decision, "uphold")

    def test_m_boundary_is_literal_first(self) -> None:
        judgment = self._m_judgment(
            literal_instance="yes",
            native_relation_match="no",
            role_change_degree="multiple",
        )
        self.assertEqual(m_score_from_ordinal(judgment), 0)

    def test_m_boundary_reserves_one_for_native_relation(self) -> None:
        judgment = self._m_judgment(
            literal_instance="no",
            native_relation_match="yes",
            role_change_degree="none_or_one",
        )
        self.assertEqual(m_score_from_ordinal(judgment), 1)

    def test_m_boundary_uses_two_for_cross_domain_projection(self) -> None:
        judgment = self._m_judgment(
            literal_instance="no",
            native_relation_match="yes",
            role_change_degree="multiple",
        )
        self.assertEqual(m_score_from_ordinal(judgment), 2)

    def test_validation_anchor_is_physically_left_out(self) -> None:
        validation_anchors = m_calibration_anchors("validation", 5)
        test_anchors = m_calibration_anchors("test", 5)
        self.assertEqual(len(validation_anchors), len(M_CALIBRATION_ANCHORS) - 1)
        self.assertNotIn(
            "validation_5",
            {anchor["anchor_id"] for anchor in validation_anchors},
        )
        self.assertEqual(len(test_anchors), len(M_CALIBRATION_ANCHORS))

    def test_shared_frontend_adapts_to_all_three_metric_contracts(self) -> None:
        shared = self._shared_analysis()
        decomposition = shared_target_to_v1_decomposition(
            shared.target_frame
        )
        importance = shared_target_to_topic_importance(shared.target_frame)
        source = shared_source_to_ms_blind_source(shared.source_frame)
        mapping = shared_mapping_to_original(shared.mapping_frame)
        domain = shared_analysis_to_domain(shared)

        self.assertEqual([topic.topic_id for topic in decomposition.topics], ["T1", "T2"])
        self.assertEqual(
            [item.decision for item in importance.assessments],
            ["keep", "contextual_detail"],
        )
        self.assertIn("worker", source.native_roles_and_operations[0])
        self.assertEqual(mapping.mappings[0].source_element, "worker")
        self.assertEqual(domain.role_alignments[0].semantic_relation, "preserved")

    def test_shared_target_rejects_a_dependent_topic_without_kept_parent(self) -> None:
        frame = self._shared_analysis().target_frame.model_copy(deep=True)
        frame.topics[1].parent_topic_id = "missing"
        with self.assertRaisesRegex(ValueError, "must name a kept parent"):
            validate_shared_target_frame(frame)

    def test_shared_frontend_visibility_firewalls(self) -> None:
        _, target_user = shared_target_frame_prompt("Target", "Description")
        _, source_user = shared_source_frame_prompt("Analogy")
        self.assertEqual(
            set(json.loads(target_user)),
            {"target", "description"},
        )
        self.assertEqual(
            set(json.loads(source_user)),
            {"analogy"},
        )

    @staticmethod
    def _m_judgment(
        *,
        literal_instance: str,
        native_relation_match: str,
        role_change_degree: str,
    ) -> MOrdinalJudgment:
        return MOrdinalJudgment(
            literal_source_summary="A literal source mechanism.",
            literal_instance=literal_instance,
            native_relation_match=native_relation_match,
            central_role_changes=["source role -> target role"],
            role_change_degree=role_change_degree,
            nearest_score_0_anchor_id="validation_8",
            nearest_score_1_anchor_id="validation_1",
            nearest_score_2_anchor_id="validation_0",
            score_probabilities=ScoreProbabilities(
                score_0=0.1, score_1=0.2, score_2=0.7
            ),
            recommended_score=2,
            confidence=0.9,
            rationale="Boundary test.",
        )

    @staticmethod
    def _blind_source_frame(
        *,
        imported_target_details: list[MSImportedDetail] | None = None,
    ) -> MSBlindSourceFrame:
        return MSBlindSourceFrame(
            literal_source_domain="ordinary physical activity",
            source_ontology="ordinary_real_world",
            fictional_mechanism_coherence="not_applicable",
            literal_source_summary="People perform an ordinary source action.",
            ordinary_source_goal="Complete the source-domain task.",
            native_mechanism="A native source operation remains.",
            native_roles_and_operations=["actor performs source action"],
            removed_mapping_language=[],
            imported_target_details=imported_target_details or [],
            source_story_coherence="coherent",
        )

    @staticmethod
    def _zero_gate_audit(**overrides: str) -> MSZeroGateAudit:
        values = {
            "defining_target_operation": "A defining operation.",
            "literal_source_remainder": "A native source operation remains.",
            "native_structural_support": "substantial",
            "target_import_dependency": "none",
            "core_relation": "consistent",
            "counterfactual_result": "intact",
            "self_reference_target": "no",
            "same_process_on_smaller_instance": "not_applicable",
            "native_nesting_or_self_reference": "not_applicable",
            "linear_handoff_only": "not_applicable",
            "failure_type": "none",
            "strongest_surviving_evidence": "The source operation survives.",
            "strongest_failure_evidence": "No decisive failure.",
            "confidence": 0.9,
            "rationale": "Policy boundary test.",
        }
        values.update(overrides)
        return MSZeroGateAudit(**values)

    @staticmethod
    def _shared_analysis() -> SharedSemanticAnalysis:
        target_role = SharedSemanticRole(
            role_id="TR1",
            role="processor",
            semantic_type="computational actor",
            necessity="defining",
            evidence="processor transforms input",
        )
        source_role = SharedSemanticRole(
            role_id="SR1",
            role="worker",
            semantic_type="person",
            necessity="defining",
            evidence="worker transforms material",
        )
        target_relation = SharedSemanticRelation(
            relation_id="TRel1",
            subject="processor",
            predicate="transforms",
            object="input",
            direction_or_order="input before output",
            necessity="defining",
            evidence="transforms input into output",
        )
        source_relation = SharedSemanticRelation(
            relation_id="SRel1",
            subject="worker",
            predicate="transforms",
            object="material",
            direction_or_order="material before product",
            necessity="defining",
            evidence="transforms material into a product",
        )
        target = SharedTargetFrame(
            target_concept="transformation",
            target_domain="computation",
            target_summary="A processor transforms an input.",
            defining_mechanism="input is transformed into output",
            success_condition="an output is produced",
            topics=[
                SharedTargetTopic(
                    topic_id="T1",
                    topic="Transform input into output",
                    importance="core",
                    decision="keep",
                    relation_to_parent="independent_requirement",
                    description_evidence="transforms input into output",
                    rationale="Defining mechanism.",
                    confidence=0.95,
                ),
                SharedTargetTopic(
                    topic_id="T2",
                    topic="For example, text input",
                    importance="supporting",
                    decision="contextual_detail",
                    relation_to_parent="illustrative_example",
                    parent_topic_id="T1",
                    description_evidence="for example, text input",
                    rationale="Non-exhaustive example.",
                    confidence=0.9,
                ),
            ],
            roles=[target_role],
            relations=[target_relation],
            constraints=[],
        )
        source = SharedSourceFrame(
            source_concept="craft work",
            literal_source_domain="workshop",
            source_ontology="ordinary_real_world",
            fictional_mechanism_coherence="not_applicable",
            literal_source_summary="A worker transforms material.",
            ordinary_source_goal="Produce a finished object.",
            native_mechanism="manual transformation",
            roles=[source_role],
            relations=[source_relation],
            removed_mapping_language=[],
            imported_target_details=[],
            source_story_coherence="coherent",
        )
        mapping = SharedMappingFrame(
            source_concept="craft work",
            target_concept="transformation",
            shared_process="an actor transforms an input into an output",
            domain_distance="different",
            alignments=[
                SharedAlignment(
                    source_item="worker",
                    target_item="processor",
                    alignment_kind="role",
                    relation="both perform the transformation",
                    preservation="preserved",
                    central=True,
                    evidence="the worker acts like the processor",
                )
            ],
            potential_breaks=[],
        )
        return SharedSemanticAnalysis(
            target_frame=target,
            source_frame=source,
            mapping_frame=mapping,
        )


if __name__ == "__main__":
    unittest.main()
