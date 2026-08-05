"""Deterministic cosine scorer for the literal-gate M experiment."""

from __future__ import annotations

import math
from typing import Protocol, Sequence


class DomainAnalysisLike(Protocol):
    source_concept: str
    source_mechanism: str
    target_concept: str
    target_signature: str
    source_domain: str
    target_domain: str


M_COSINE_POLICY_VERSION = "m_v19_literal_gate_domain_cosine_v2_local"
M_EMBEDDING_BACKEND = "sentence-transformers"
DEFAULT_M_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_M_EMBEDDING_DEVICE = "auto"
DEFAULT_M_CONCEPT_WEIGHT = 0.5
DEFAULT_M_COSINE_THRESHOLD = 0.35


def m_cosine_embedding_texts(domain: DomainAnalysisLike) -> dict[str, str]:
    """Build symmetric concept and domain texts from structured M evidence."""
    return {
        "source_concept": (
            f"Concept: {domain.source_concept}. "
            f"Defining mechanism: {domain.source_mechanism}."
        ),
        "target_concept": (
            f"Concept: {domain.target_concept}. "
            f"Defining mechanism: {domain.target_signature}."
        ),
        "source_domain": f"Domain: {domain.source_domain}.",
        "target_domain": f"Domain: {domain.target_domain}.",
    }


def cosine_distance(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Return 1-cosine after validating finite, non-zero vectors."""
    if not left or not right:
        raise ValueError("Cosine vectors must be non-empty")
    if len(left) != len(right):
        raise ValueError(
            f"Cosine vector dimensions differ: {len(left)} != {len(right)}"
        )
    if not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("Cosine vectors must contain only finite values")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine vectors must have non-zero norm")

    similarity = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    ) / (left_norm * right_norm)
    similarity = min(1.0, max(-1.0, similarity))
    return 1.0 - similarity


def combined_m_cosine_distance(
    concept_distance: float,
    domain_distance: float,
    concept_weight: float,
) -> float:
    """Combine concept and domain distances with one explicit weight."""
    if not 0.0 <= concept_distance <= 2.0:
        raise ValueError("concept_distance must be in [0, 2]")
    if not 0.0 <= domain_distance <= 2.0:
        raise ValueError("domain_distance must be in [0, 2]")
    if not 0.0 <= concept_weight <= 1.0:
        raise ValueError("concept_weight must be in [0, 1]")
    return (
        concept_weight * concept_distance
        + (1.0 - concept_weight) * domain_distance
    )


def m_score_from_cosine(
    *,
    literal_instance: str,
    concept_distance: float,
    domain_distance: float,
    concept_weight: float = DEFAULT_M_CONCEPT_WEIGHT,
    nonliteral_threshold: float = DEFAULT_M_COSINE_THRESHOLD,
) -> tuple[int, float]:
    """Apply the literal-first M=0 gate and cosine M=1/2 boundary."""
    if literal_instance not in {"yes", "no", "unclear"}:
        raise ValueError(f"Unsupported literal_instance: {literal_instance!r}")
    if not 0.0 <= nonliteral_threshold <= 2.0:
        raise ValueError("nonliteral_threshold must be in [0, 2]")

    combined_distance = combined_m_cosine_distance(
        concept_distance,
        domain_distance,
        concept_weight,
    )
    if literal_instance == "yes":
        return 0, combined_distance
    if combined_distance <= nonliteral_threshold:
        return 1, combined_distance
    return 2, combined_distance


def m_cosine_latent_score(
    literal_instance: str,
    combined_distance: float,
) -> float:
    """Map raw nonliteral distance to [1,2], while literal instances stay 0."""
    if not 0.0 <= combined_distance <= 2.0:
        raise ValueError("combined_distance must be in [0, 2]")
    if literal_instance == "yes":
        return 0.0
    return 1.0 + combined_distance / 2.0
