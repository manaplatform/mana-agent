"""Mode-specific contracts for the supervised native tool loop."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mana_agent.agent.plan_service import Question
from mana_agent.coding.models import CODING_RESULT_TOOL
from mana_agent.repository_evidence.models import RuntimeWiringEvidence


class AdaptiveCodingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disposition: Literal["complete", "clarify", "blocked"]
    summary: str = Field(pattern=r"\S", max_length=4000)
    questions: tuple[Question, ...] = Field(default=(), max_length=8)
    wiring: RuntimeWiringEvidence | None = None

    @model_validator(mode="after")
    def validate_questions(self):
        if bool(self.questions) != (self.disposition == "clarify"):
            raise ValueError("Only the clarify disposition requires questions")
        if len({question.field_id for question in self.questions}) != len(self.questions):
            raise ValueError("Clarification field IDs must be unique")
        return self


def result_tool():
    return {"type": "function", "function": {
        "name": CODING_RESULT_TOOL,
        "description": "Return a completed coding result, material clarification questions, or a concrete blocker.",
        "parameters": AdaptiveCodingResult.model_json_schema(),
    }}
