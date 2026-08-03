from __future__ import annotations

import unittest

from scripts.verify_mapping_strength_archive import (
    verify_mapping_strength_archive,
)


class OriginalMappingStrengthArchiveTest(unittest.TestCase):
    def test_prompt_cache_and_archived_scores_form_one_chain(self) -> None:
        audit = verify_mapping_strength_archive()
        self.assertEqual(audit["mapping_prompt_hash_matches"], 74)
        self.assertEqual(audit["ms_prompt_hash_matches"], 74)
        self.assertTrue(audit["validation_predictions_match"])
        self.assertTrue(audit["test_predictions_match_archived_scores"])


if __name__ == "__main__":
    unittest.main()
