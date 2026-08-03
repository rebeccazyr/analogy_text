"""Exact output schemas used by the original v1 TCC path.

Class names, field order, constraints, and descriptions are kept identical
because their generated JSON Schema is part of the model's system message.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Topic(StrictModel):
    topic_id: str = Field(description="Stable short identifier such as T1.")
    topic: str = Field(description="One atomic topic from the provided description.")
    importance: Literal["core", "supporting"]


class ConceptDecomposition(StrictModel):
    target_summary: str
    topics: list[Topic] = Field(min_length=2, max_length=6)


class ScoreProbabilities(StrictModel):
    score_0: float = Field(ge=0, le=1)
    score_1: float = Field(ge=0, le=1)
    score_2: float = Field(ge=0, le=1)

    def normalized(self) -> dict[int, float]:
        values = [self.score_0, self.score_1, self.score_2]
        total = sum(values)
        if total <= 0:
            return {0: 1 / 3, 1: 1 / 3, 2: 1 / 3}
        return {index: value / total for index, value in enumerate(values)}

    def expected_score(self) -> float:
        probabilities = self.normalized()
        return probabilities[1] + 2 * probabilities[2]


class CoverageAssessment(StrictModel):
    topic_id: str
    status: Literal["absent", "partial", "covered"]
    evidence: str


class TCCJudgment(StrictModel):
    assessments: list[CoverageAssessment]
    missing_topics: list[str]
    score_probabilities: ScoreProbabilities
    recommended_score: Literal[0, 1, 2]
    confidence: float = Field(ge=0, le=1)
    rationale: str
