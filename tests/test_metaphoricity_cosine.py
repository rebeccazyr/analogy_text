from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from analogy_agents.metaphoricity_cosine import (
    combined_m_cosine_distance,
    cosine_distance,
    m_cosine_embedding_texts,
    m_cosine_latent_score,
    m_score_from_cosine,
)


class MetaphoricityCosineTest(unittest.TestCase):
    def test_cosine_distance_has_expected_boundaries(self) -> None:
        self.assertAlmostEqual(cosine_distance([1.0, 0.0], [1.0, 0.0]), 0.0)
        self.assertAlmostEqual(cosine_distance([1.0, 0.0], [0.0, 1.0]), 1.0)
        self.assertAlmostEqual(cosine_distance([1.0, 0.0], [-1.0, 0.0]), 2.0)

    def test_cosine_distance_rejects_invalid_vectors(self) -> None:
        with self.assertRaises(ValueError):
            cosine_distance([], [])
        with self.assertRaises(ValueError):
            cosine_distance([0.0, 0.0], [1.0, 0.0])
        with self.assertRaises(ValueError):
            cosine_distance([math.nan], [1.0])

    def test_embedding_texts_use_symmetric_structured_fields(self) -> None:
        texts = m_cosine_embedding_texts(self._domain_analysis())
        self.assertEqual(
            list(texts),
            [
                "source_concept",
                "target_concept",
                "source_domain",
                "target_domain",
            ],
        )
        self.assertIn("piano skill reuse", texts["source_concept"])
        self.assertIn("knowledge transfer", texts["target_concept"])
        self.assertEqual(texts["source_domain"], "Domain: human learning.")
        self.assertEqual(texts["target_domain"], "Domain: machine learning.")

    def test_combined_distance_uses_explicit_concept_weight(self) -> None:
        self.assertAlmostEqual(
            combined_m_cosine_distance(0.2, 0.8, concept_weight=0.75),
            0.35,
        )

    def test_literal_gate_overrides_large_cosine_distance(self) -> None:
        score, distance = m_score_from_cosine(
            literal_instance="yes",
            concept_distance=1.2,
            domain_distance=1.0,
            concept_weight=0.5,
            nonliteral_threshold=0.35,
        )
        self.assertEqual(score, 0)
        self.assertAlmostEqual(distance, 1.1)
        self.assertEqual(m_cosine_latent_score("yes", distance), 0.0)

    def test_nonliteral_threshold_separates_one_and_two(self) -> None:
        close_score, close_distance = m_score_from_cosine(
            literal_instance="no",
            concept_distance=0.2,
            domain_distance=0.4,
            concept_weight=0.5,
            nonliteral_threshold=0.35,
        )
        far_score, far_distance = m_score_from_cosine(
            literal_instance="unclear",
            concept_distance=0.6,
            domain_distance=0.8,
            concept_weight=0.5,
            nonliteral_threshold=0.35,
        )
        self.assertEqual((close_score, far_score), (1, 2))
        self.assertAlmostEqual(close_distance, 0.3)
        self.assertAlmostEqual(far_distance, 0.7)
        self.assertAlmostEqual(m_cosine_latent_score("no", close_distance), 1.15)

    @staticmethod
    def _domain_analysis() -> SimpleNamespace:
        return SimpleNamespace(
            source_concept="piano skill reuse",
            source_mechanism="reuse prior skill on a related new piece",
            target_concept="knowledge transfer",
            target_signature="reuse prior knowledge on a related new task",
            source_domain="human learning",
            target_domain="machine learning",
        )


if __name__ == "__main__":
    unittest.main()
