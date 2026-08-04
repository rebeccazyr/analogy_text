from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_metric(path: Path, metric: str) -> dict[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["id"]): int(row[metric])
            for row in csv.DictReader(handle)
        }


class LatestSubmissionSnapshotTest(unittest.TestCase):
    def test_manifest_hashes_and_submission_components_match(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text())
        components: dict[str, dict[int, int]] = {}
        for metric in ("TCC", "MS", "M"):
            metadata = manifest["components"][metric]
            path = ROOT / metadata["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                metadata["sha256"],
            )
            components[metric] = read_metric(path, metric)

        submission_path = ROOT / manifest["submission"]["path"]
        self.assertEqual(
            hashlib.sha256(submission_path.read_bytes()).hexdigest(),
            manifest["submission"]["sha256"],
        )
        with submission_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["id"]) for row in rows], list(range(62)))
        for metric in ("TCC", "MS", "M"):
            self.assertEqual(
                [int(row[metric]) for row in rows],
                [components[metric][example_id] for example_id in range(62)],
            )
        for metric in ("VC", "VA", "VE"):
            self.assertEqual({int(row[metric]) for row in rows}, {0})

    def test_only_m_uses_high_reasoning(self) -> None:
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual(
            manifest["reasoning"],
            {"TCC": "medium", "MS": "medium", "M": "high"},
        )

    def test_zero_gate_replaces_only_the_two_v1_false_positives(self) -> None:
        baseline = read_metric(
            ROOT
            / "artifacts/frozen/mapping_strength_v1_baseline_predictions.csv",
            "MS",
        )
        active = read_metric(
            ROOT / "artifacts/frozen/mapping_strength_predictions.csv",
            "MS",
        )
        changes = {
            example_id: (baseline[example_id], active[example_id])
            for example_id in baseline
            if baseline[example_id] != active[example_id]
        }
        self.assertEqual(changes, {15: (1, 0), 22: (1, 0)})

    def test_reported_text_averages_match_component_metrics(self) -> None:
        metrics = json.loads(
            (ROOT / "artifacts/frozen/leaderboard_metrics.json").read_text()
        )
        self.assertAlmostEqual(
            metrics["text_kendall_AVG"],
            sum(
                metrics[name]
                for name in ("kendall_TCC", "kendall_MS", "kendall_M")
            )
            / 3,
        )
        self.assertAlmostEqual(
            metrics["text_spearman_AVG"],
            sum(
                metrics[name]
                for name in ("spearman_TCC", "spearman_MS", "spearman_M")
            )
            / 3,
        )


if __name__ == "__main__":
    unittest.main()
