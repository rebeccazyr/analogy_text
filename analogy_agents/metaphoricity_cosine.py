"""Deterministic cosine scorer for the literal-gate M experiment."""

from __future__ import annotations

import math
from typing import Mapping, Protocol, Sequence


class DomainAnalysisLike(Protocol):
    source_concept: str
    source_mechanism: str
    target_concept: str
    target_signature: str
    source_domain: str
    target_domain: str


M_COSINE_POLICY_VERSION = "m_v19_literal_gate_domain_cosine_v2_local"
M_FEATURE_POLICY_VERSION = "m_v20_literal_gate_feature_ablation_v1_local"
M_EMBEDDING_BACKEND = "sentence-transformers"
DEFAULT_M_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_M_EMBEDDING_DEVICE = "auto"
DEFAULT_M_CONCEPT_WEIGHT = 0.5
DEFAULT_M_COSINE_THRESHOLD = 0.35

# Pre-registered feature combinations for the small-data ablation.  Every
# feature points in the same direction: 0 is closer to M=1 and 1 is closer to
# M=2.  Keeping the weights fixed leaves only the cutoff to calibrate later.
M_FEATURE_WEIGHTS: dict[str, dict[str, float]] = {
    "e1": {"mechanism_distance": 1.0},
    "e2": {"native_relation_mismatch": 1.0},
    "e3": {
        "mechanism_distance": 0.60,
        "native_relation_mismatch": 0.40,
    },
    "e4": {
        "mechanism_distance": 0.50,
        "native_relation_mismatch": 0.30,
        "role_type_shift": 0.20,
    },
    "e5": {
        "mechanism_distance": 0.45,
        "native_relation_mismatch": 0.25,
        "role_type_shift": 0.15,
        "concept_distance": 0.075,
        "domain_distance": 0.075,
    },
}

DEFAULT_M_FEATURE_THRESHOLDS: dict[str, float] = {
    "e1": 0.35,
    "e2": 0.50,
    "e3": 0.35,
    "e4": 0.35,
    "e5": 0.35,
}


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


def m_feature_embedding_texts(domain: DomainAnalysisLike) -> dict[str, str]:
    """Build disentangled mechanism, concept, and domain embedding texts."""
    return {
        "source_mechanism": f"Mechanism: {domain.source_mechanism}.",
        "target_mechanism": f"Mechanism: {domain.target_signature}.",
        "source_concept": f"Concept: {domain.source_concept}.",
        "target_concept": f"Concept: {domain.target_concept}.",
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


def bounded_cosine_feature(distance: float) -> float:
    """Put cosine distance on the shared [0, 1] feature scale."""
    if not 0.0 <= distance <= 2.0:
        raise ValueError("Cosine distance must be in [0, 2]")
    return min(distance, 1.0)


def native_relation_mismatch(value: str) -> float:
    """Map native-relation evidence to an M=2-directed numeric feature."""
    try:
        return {"yes": 0.0, "unclear": 0.5, "no": 1.0}[value]
    except KeyError as error:
        raise ValueError(f"Unsupported native_relation_match: {value!r}") from error


def role_type_shift(value: str) -> float:
    """Map role-preservation evidence to an M=2-directed numeric feature."""
    try:
        return {
            "none_or_one_shift": 0.0,
            "unclear": 0.5,
            "multiple_type_changes": 1.0,
        }[value]
    except KeyError as error:
        raise ValueError(f"Unsupported role_type_preservation: {value!r}") from error


def combine_m_features(
    feature_values: Mapping[str, float],
    weights: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    """Combine normalized features and return score plus contributions."""
    if not weights:
        raise ValueError("Feature weights must be non-empty")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Feature weights must sum to 1")

    contributions: dict[str, float] = {}
    for name, weight in weights.items():
        if name not in feature_values:
            raise ValueError(f"Missing feature value: {name}")
        value = feature_values[name]
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Feature {name} must be in [0, 1]")
        if weight < 0.0:
            raise ValueError(f"Feature weight {name} must be non-negative")
        contributions[name] = weight * value
    return sum(contributions.values()), contributions


def score_m_feature_experiment(
    *,
    experiment: str,
    literal_instance: str,
    feature_values: Mapping[str, float],
    threshold: float | None = None,
) -> tuple[int, float, dict[str, float], float]:
    """Score one E1--E5 configuration behind the shared literal M=0 gate."""
    if experiment not in M_FEATURE_WEIGHTS:
        raise ValueError(f"Unsupported M feature experiment: {experiment!r}")
    if literal_instance not in {"yes", "no", "unclear"}:
        raise ValueError(f"Unsupported literal_instance: {literal_instance!r}")

    cutoff = (
        DEFAULT_M_FEATURE_THRESHOLDS[experiment]
        if threshold is None
        else threshold
    )
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("M feature threshold must be in [0, 1]")

    combined, contributions = combine_m_features(
        feature_values,
        M_FEATURE_WEIGHTS[experiment],
    )
    if literal_instance == "yes":
        return 0, combined, contributions, cutoff
    return (1 if combined <= cutoff else 2), combined, contributions, cutoff


def m_feature_latent_score(literal_instance: str, combined_score: float) -> float:
    """Map the shared [0,1] nonliteral score to the ordinal [1,2] interval."""
    if not 0.0 <= combined_score <= 1.0:
        raise ValueError("combined_score must be in [0, 1]")
    if literal_instance == "yes":
        return 0.0
    return 1.0 + combined_score
