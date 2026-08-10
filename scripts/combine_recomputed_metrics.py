#!/usr/bin/env python3
"""Combine freshly recomputed TCC, MS, and M outputs for submission."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_COLUMNS = {"TCC", "MS", "M"}
SUBMISSION_COLUMNS = ["id", "TCC", "MS", "M", "VC", "VA", "VE"]


def read_metric(path: Path, metric: str) -> dict[int, int]:
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"Unsupported metric: {metric}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {metric} predictions: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"id", metric}.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: id,{metric}")
        scores: dict[int, int] = {}
        for row in reader:
            example_id = int(row["id"])
            score = int(row[metric])
            if example_id in scores:
                raise ValueError(f"Duplicate id {example_id} in {path}")
            if score not in {0, 1, 2}:
                raise ValueError(
                    f"Invalid {metric} score {score} for id {example_id}; expected 0, 1, or 2"
                )
            scores[example_id] = score
    return scores


def combine_recomputed_metrics(
    tcc_path: Path,
    ms_path: Path,
    metaphoricity_path: Path,
    output_path: Path,
) -> Path:
    metric_scores = {
        "TCC": read_metric(tcc_path, "TCC"),
        "MS": read_metric(ms_path, "MS"),
        "M": read_metric(metaphoricity_path, "M"),
    }
    id_sets = {metric: set(scores) for metric, scores in metric_scores.items()}
    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        raise ValueError(
            "Recomputed metric files contain different ID sets: "
            + ", ".join(
                f"{metric}={len(ids)} rows" for metric, ids in id_sets.items()
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUBMISSION_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for example_id in sorted(id_sets["TCC"]):
            writer.writerow(
                {
                    "id": example_id,
                    "TCC": metric_scores["TCC"][example_id],
                    "MS": metric_scores["MS"][example_id],
                    "M": metric_scores["M"][example_id],
                    "VC": 0,
                    "VA": 0,
                    "VE": 0,
                }
            )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine newly generated metric predictions into one submission CSV."
    )
    parser.add_argument("--tcc", type=Path, required=True)
    parser.add_argument("--ms", type=Path, required=True)
    parser.add_argument("--metaphoricity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = combine_recomputed_metrics(
        args.tcc,
        args.ms,
        args.metaphoricity,
        args.output,
    )
    print(f"Combined recomputed metrics: {output}")


if __name__ == "__main__":
    main()
