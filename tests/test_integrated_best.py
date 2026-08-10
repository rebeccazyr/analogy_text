from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_integrated_best import ROOT, build_integrated_best


class IntegratedBestParityTest(unittest.TestCase):
    def test_rebuild_is_byte_identical_to_frozen_champion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            submission, audit_path, audit = build_integrated_best(
                ROOT,
                Path(directory),
            )
            frozen = ROOT / "artifacts/frozen/submission.csv"
            self.assertEqual(submission.read_bytes(), frozen.read_bytes())
            self.assertEqual(audit["status"], "byte_identical")
            self.assertEqual(
                audit["generated_submission"]["sha256"],
                audit["frozen_champion_submission"]["sha256"],
            )
            self.assertTrue(audit_path.is_file())
            self.assertEqual(
                {
                    metric: audit["components"][metric]["rows"]
                    for metric in ("TCC", "MS", "M")
                },
                {"TCC": 62, "MS": 62, "M": 62},
            )


if __name__ == "__main__":
    unittest.main()
