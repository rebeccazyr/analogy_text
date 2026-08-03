"""Structured-output contracts for the recovered original-v1 MS path."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MappingRecord(StrictModel):
    source_element: str
    target_element: str
    relation: str
    evidence: str


class MappingAnalysis(StrictModel):
    source_concept: str
    target_concept: str
    mappings: list[MappingRecord]
    process_summary: str
    potential_breaks: list[str]


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


class MappingAssessment(StrictModel):
    source_element: str
    target_element: str
    judgment: Literal["sound", "stretch", "inconsistent"]
    rationale: str


class MSJudgment(StrictModel):
    assessments: list[MappingAssessment]
    structural_issues: list[str]
    score_probabilities: ScoreProbabilities
    recommended_score: Literal[0, 1, 2]
    confidence: float = Field(ge=0, le=1)
    rationale: str
