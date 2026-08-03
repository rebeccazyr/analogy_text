import hashlib
import random
from collections import Counter
from typing import Iterable

import numpy as np
from scipy.optimize import minimize


PAIRWISE_SEED = 20260802
PAIRWISE_OFFSETS = (1, 2, 4, 8, 16)
BRADLEY_TERRY_L2 = 0.1


def build_pair_schedule(
    example_ids: Iterable[int],
    seed: int = PAIRWISE_SEED,
) -> list[tuple[int, int]]:
    ids = sorted(example_ids)
    if len(ids) < 2:
        return []
    if len(ids) <= 16:
        canonical_pairs = [
            (ids[left], ids[right])
            for left in range(len(ids))
            for right in range(left + 1, len(ids))
        ]
    else:
        ring = list(ids)
        random.Random(seed).shuffle(ring)
        offsets = [
            offset
            for offset in (*PAIRWISE_OFFSETS, len(ring) // 2)
            if 0 < offset <= len(ring) // 2
        ]
        pair_set: set[tuple[int, int]] = set()
        for offset in offsets:
            for position, first_id in enumerate(ring):
                second_id = ring[(position + offset) % len(ring)]
                pair_set.add(tuple(sorted((first_id, second_id))))
        canonical_pairs = sorted(pair_set)

    oriented_pairs: list[tuple[int, int]] = []
    for first_id, second_id in canonical_pairs:
        digest = hashlib.sha256(
            f"{seed}:{first_id}:{second_id}".encode("utf-8")
        ).digest()
        if digest[0] % 2:
            oriented_pairs.append((second_id, first_id))
        else:
            oriented_pairs.append((first_id, second_id))
    return oriented_pairs


def schedule_degrees(
    schedule: Iterable[tuple[int, int]],
) -> Counter[int]:
    degrees: Counter[int] = Counter()
    for first_id, second_id in schedule:
        degrees[first_id] += 1
        degrees[second_id] += 1
    return degrees


def fit_bradley_terry(
    example_ids: Iterable[int],
    comparisons: Iterable[dict[str, float | int | str]],
    l2: float = BRADLEY_TERRY_L2,
) -> dict[int, float]:
    ids = sorted(example_ids)
    if not ids:
        return {}
    index = {example_id: position for position, example_id in enumerate(ids)}
    observations: list[tuple[int, int, float]] = []
    for comparison in comparisons:
        first_id = int(comparison["a_id"])
        second_id = int(comparison["b_id"])
        outcome = str(comparison["more_conceptually_distant"])
        raw_target = {"a": 1.0, "b": 0.0, "tie": 0.5}[outcome]
        confidence = float(comparison.get("confidence", 1.0))
        target = 0.5 + (raw_target - 0.5) * confidence
        observations.append((index[first_id], index[second_id], target))

    def objective(scores: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.5 * l2 * float(np.dot(scores, scores))
        gradient = l2 * scores.copy()
        for first_index, second_index, target in observations:
            difference = scores[first_index] - scores[second_index]
            loss += float(np.logaddexp(0.0, difference) - target * difference)
            probability = 1.0 / (1.0 + np.exp(-difference))
            residual = probability - target
            gradient[first_index] += residual
            gradient[second_index] -= residual
        return loss, gradient

    result = minimize(
        fun=lambda scores: objective(scores)[0],
        x0=np.zeros(len(ids), dtype=float),
        jac=lambda scores: objective(scores)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Bradley-Terry fit failed: {result.message}")
    centered = result.x - float(np.mean(result.x))
    return {
        example_id: float(centered[position])
        for position, example_id in enumerate(ids)
    }


def assign_global_tiers(
    ranked_ids: list[int],
    m0_end_rank: int,
    m1_end_rank: int,
) -> dict[int, int]:
    if not ranked_ids:
        return {}
    if not -1 <= m0_end_rank <= m1_end_rank < len(ranked_ids):
        raise ValueError(
            "Expected -1 <= m0_end_rank <= m1_end_rank < rank_count"
        )
    predictions: dict[int, int] = {}
    for rank, example_id in enumerate(ranked_ids):
        if rank <= m0_end_rank:
            score = 0
        elif rank <= m1_end_rank:
            score = 1
        else:
            score = 2
        predictions[example_id] = score
    return predictions
