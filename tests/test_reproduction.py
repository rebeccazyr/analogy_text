from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_submission import (
    DEFAULT_EXPECTED,
    DEFAULT_M,
    DEFAULT_MS_BASE,
    DEFAULT_TCC,
    build_submission,
)


EXPECTED_SHA256 = (
    "eaaed257e856be97f59601dd17ae41f3bccba9356cb140bdd862efe7a38293ee"
)


class FrozenSubmissionTest(unittest.TestCase):
    def test_frozen_components_reproduce_known_good_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "submission.csv"
            audit_path = Path(temp_dir) / "audit.json"
            audit = build_submission(
                tcc_path=DEFAULT_TCC,
                ms_base_path=DEFAULT_MS_BASE,
                m_path=DEFAULT_M,
                output_path=output,
                audit_path=audit_path,
                expected_path=DEFAULT_EXPECTED,
            )

            self.assertTrue(audit["verification"]["value_for_value_match"])
            self.assertEqual(audit["output"]["rows"], 62)
            self.assertEqual(audit["output"]["sha256"], EXPECTED_SHA256)


if __name__ == "__main__":
    unittest.main()
