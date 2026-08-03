from __future__ import annotations

import unittest
from pathlib import Path

from analogy_agents.pipeline import (
    facet_audit_to_coverage_audit,
    load_split,
    m_score_from_ordinal,
    v1_conservative_tcc_correction,
)
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


class ActivePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.validation = load_split(Path("challenge-dataset"), "validation")
        cls.test = load_split(Path("challenge-dataset"), "test")

    def test_text_splits_are_complete(self) -> None:
        self.assertEqual(len(self.validation), 12)
        self.assertEqual(len(self.test), 62)

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


if __name__ == "__main__":
    unittest.main()
