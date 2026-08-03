from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.combine_recomputed_metrics import combine_recomputed_metrics


class CombineRecomputedMetricsTest(unittest.TestCase):
    def test_combines_new_metric_outputs_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            for metric, rows in {
                "TCC": [(1, 2), (0, 1)],
                "MS": [(0, 2), (1, 1)],
                "M": [(1, 0), (0, 2)],
            }.items():
                path = root / f"{metric.lower()}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["id", metric])
                    writer.writerows(rows)
                inputs[metric] = path

            output = root / "submission.csv"
            combine_recomputed_metrics(
                inputs["TCC"], inputs["MS"], inputs["M"], output
            )

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["id"] for row in rows], ["0", "1"])
            self.assertEqual(rows[0], {
                "id": "0", "TCC": "1", "MS": "2", "M": "2",
                "VC": "0", "VA": "0", "VE": "0",
            })


if __name__ == "__main__":
    unittest.main()
