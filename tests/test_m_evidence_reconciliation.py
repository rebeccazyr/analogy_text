from __future__ import annotations

import unittest

from analogy_agents.pipeline import (
    load_frozen_m_baseline,
    m_score_from_reconciled_evidence,
    reconcile_m_duplicate_source_groups,
)
from analogy_agents.schemas import (
    DomainAnalysis,
    LiteralInstanceJudgment,
    MOrdinalJudgment,
    ScoreProbabilities,
)


class MEvidenceReconciliationTest(unittest.TestCase):
    def test_related_native_evidence_restores_score_one(self) -> None:
        score, rule = m_score_from_reconciled_evidence(
            self._domain("related"),
            self._literal(
                behavior_match="no",
                literal_instance="no",
                native_relation_match="yes",
                role_type_preservation="none_or_one_shift",
            ),
            self._ordinal(
                literal_instance="no",
                native_relation_match="yes",
                role_change_degree="multiple",
            ),
            baseline_score=2,
        )
        self.assertEqual(
            (score, rule),
            (1, "related_domain_native_evidence_restored"),
        )

    def test_two_judge_literal_consensus_restores_score_zero(self) -> None:
        score, rule = m_score_from_reconciled_evidence(
            self._domain("different"),
            self._literal(
                behavior_match="yes",
                target_scope_match="yes",
                literal_instance="yes",
            ),
            self._ordinal(
                literal_instance="yes",
                native_relation_match="no",
                role_change_degree="none_or_one",
            ),
            baseline_score=1,
        )
        self.assertEqual(
            (score, rule),
            (0, "two_judge_literal_consensus_restored"),
        )

    def test_duplicate_source_group_uses_conservative_consistency(self) -> None:
        results = [
            self._group_result(2, 1, "none_or_one"),
            self._group_result(20, 2, "multiple"),
        ]
        reconcile_m_duplicate_source_groups(results)
        self.assertEqual([item["prediction"]["M"] for item in results], [2, 2])
        self.assertEqual(
            results[0]["m_reconciliation"]["decisive_rule"],
            "duplicate_source_frozen_consistency",
        )

    def test_frozen_baselines_are_complete(self) -> None:
        self.assertEqual(len(load_frozen_m_baseline("validation")), 12)
        self.assertEqual(len(load_frozen_m_baseline("test")), 62)

    @staticmethod
    def _domain(distance: str) -> DomainAnalysis:
        return DomainAnalysis(
            literal_source_summary="A literal source.",
            source_concept="source concept",
            source_mechanism="source mechanism",
            target_concept="target concept",
            target_signature="target signature",
            source_domain="source domain",
            target_domain="target domain",
            domain_distance=distance,
            role_alignments=[],
        )

    @staticmethod
    def _literal(**overrides: str) -> LiteralInstanceJudgment:
        values = {
            "literal_source_mechanism": "A source mechanism.",
            "target_defining_mechanism": "A target mechanism.",
            "target_scope_type": "domain_specific",
            "behavior_match": "no",
            "target_scope_match": "no",
            "literal_instance": "no",
            "native_relation_match": "yes",
            "role_type_preservation": "none_or_one_shift",
            "evidence": "Structured boundary evidence.",
            "confidence": 0.9,
        }
        values.update(overrides)
        return LiteralInstanceJudgment(**values)

    @staticmethod
    def _ordinal(
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
                score_0=0.1,
                score_1=0.2,
                score_2=0.7,
            ),
            recommended_score=2,
            confidence=0.9,
            rationale="Boundary test.",
        )

    @staticmethod
    def _group_result(example_id: int, score: int, role_change_degree: str) -> dict:
        return {
            "id": example_id,
            "target": "Hallucination",
            "prediction": {"M": score},
            "latent_scores": {"M": float(score)},
            "agents": {
                "source_domain_classifier": {"source_concept": "tour guide"},
                "literal_instance_judge": {
                    "literal_instance": "no",
                    "native_relation_match": "yes",
                },
                "m_ordinal_judge": {
                    "literal_instance": "no",
                    "native_relation_match": "yes",
                    "role_change_degree": role_change_degree,
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
