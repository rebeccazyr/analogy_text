import unittest

from scripts.generate_augmentation_data import (
    InvarianceAssessment,
    RelativeAssessment,
    RelativePair,
    invariance_status,
    relative_status,
)


class AugmentationAcceptanceTests(unittest.TestCase):
    def test_invariance_requires_unanimous_equivalence(self):
        assessments = [
            InvarianceAssessment(
                sample_id="S1",
                tcc_equivalent=True,
                ms_equivalent=True,
                m_equivalent=True,
                semantic_drift=[],
                artifact_flags=[],
                confidence=0.9,
                rationale="All metric-relevant content is unchanged.",
            )
            for _ in range(3)
        ]
        self.assertEqual(invariance_status(assessments), "accepted")
        assessments[0].tcc_equivalent = False
        self.assertEqual(invariance_status(assessments), "review")

    def test_relative_requires_target_change_and_non_target_invariance(self):
        pair = RelativePair(
            pair_id="MS_ROLE_MISMATCH",
            metric="MS",
            expected_relation="a_higher",
            analogy_a="A" * 120,
            analogy_b="B" * 120,
            controlled_operation="Introduce one central role mismatch in B.",
            non_target_invariance=["TCC unchanged", "M unchanged"],
        )
        assessments = [
            RelativeAssessment(
                sample_id="S1",
                tcc_relation="same",
                ms_relation="a_higher",
                m_relation="same",
                artifact_flags=[],
                confidence=0.9,
                rationale="Only mapping strength changes between the pair.",
            )
            for _ in range(3)
        ]
        self.assertEqual(relative_status(pair, assessments), "accepted")
        assessments[1].m_relation = "b_higher"
        self.assertEqual(relative_status(pair, assessments), "review")


if __name__ == "__main__":
    unittest.main()
