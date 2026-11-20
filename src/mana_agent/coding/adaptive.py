"""Mode-specific contracts for the supervised native tool loop."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mana_agent.agent.plan_service import Question
from mana_agent.coding.evidence_tools import INTEGRATE, INVESTIGATE
from mana_agent.coding.models import CODING_RESULT_TOOL, ISOLATED_CHECK_TOOL
from mana_agent.coding.scheduler import WRITE_TOOL_NAMES
from mana_agent.repository_evidence.models import RuntimeWiringEvidence


EVIDENCE_REFERENCE_TOOL = "repository_evidence_get"
REPOSITORY_EVIDENCE_TOOLS = frozenset({
    "ls", "list_files", "batch_list_files", "read_file", "repo_batch_read", "batch_read_files",
    "repo_search", "batch_search_files", "find_symbols", "call_graph",
    "repository_evidence_query", EVIDENCE_REFERENCE_TOOL, INVESTIGATE, INTEGRATE,
})
SCOPED_SOURCE_TOOLS = frozenset({"read_file", "repo_search"})
EVIDENCE_NEED = "evidence_need"
EVIDENCE_NEED_MAX_LENGTH = 500
EVIDENCE_NEED_DESCRIPTION = "Name the missing fact and the specific edit or check the answer could change."
_EDIT_AND_CHECK_TOOLS = WRITE_TOOL_NAMES | {ISOLATED_CHECK_TOOL}
_BROAD_EVIDENCE_TOOLS = (REPOSITORY_EVIDENCE_TOOLS | {"repo_batch_search"}) - SCOPED_SOURCE_TOOLS - {EVIDENCE_REFERENCE_TOOL}


@dataclass(frozen=True)
class AdaptiveWriteBudget:
    remaining_tools: int
    remaining_turns: int
    tool_reserve: int

    @property
    def optional_calls(self) -> int:
        if self.remaining_turns <= self.tool_reserve + 1:
            return 0
        return max(0, self.remaining_tools - self.tool_reserve)

    @property
    def result_only(self) -> bool:
        return self.remaining_tools < max(1, self.tool_reserve) or self.remaining_turns == 1

    def allows(self, name: str) -> bool:
        if name == CODING_RESULT_TOOL:
            return True
        if self.result_only:
            return False
        if name in WRITE_TOOL_NAMES:
            return self.remaining_tools >= 2
        if name == ISOLATED_CHECK_TOOL:
            return True
        return self.optional_calls > 0 and name not in _BROAD_EVIDENCE_TOOLS

    def instruction(self) -> str:
        allowance = (f"This assignment has {self.remaining_tools} tool calls and {self.remaining_turns} model turns left, "
                     f"including this turn. Up to {self.optional_calls} tool calls are available for optional work.")
        if self.result_only:
            return f"{allowance} Call {CODING_RESULT_TOOL} alone. Report a concrete blocker if the work is incomplete."
        if not self.optional_calls:
            if self.remaining_turns == 2 and self.remaining_tools >= 2:
                return (f"{allowance} This is the last tool turn. If editing, put the focused check after the edit "
                        "in the same tool group.")
            return f"{allowance} Use the remaining capacity for an edit and focused verification, or report a blocker."
        return (f"{allowance} Include source reads or searches only when their answer could change an edit or check. "
                f"Supply {EVIDENCE_NEED} with each read or search. Use existing observations and prefer an edit "
                "or check whenever the work supports it.")

    def reject_group(self, names: list[str]) -> tuple[str, str] | None:
        if self.result_only:
            return "RESULT_REQUIRED", f"Call {CODING_RESULT_TOOL} alone with the current outcome or blocker."
        last_edit = max((index for index, name in enumerate(names) if name in WRITE_TOOL_NAMES), default=-1)
        optional = sum(name not in _EDIT_AND_CHECK_TOOLS for name in names)
        if last_edit == -1 and self.tool_reserve == 2:
            # A diagnostic check can run before a repair when there is capacity for both.
            optional += names.count(ISOLATED_CHECK_TOOL)
        if len(names) > self.remaining_tools or optional > self.optional_calls:
            return "BUDGET_EXCEEDED", "Select a group within the current allowance or act on the retained observations."
        if (last_edit >= 0 and ISOLATED_CHECK_TOOL not in names[last_edit + 1:]
                and (len(names) == self.remaining_tools or self.remaining_turns == 2)):
            return ("VERIFICATION_RESERVE_REQUIRED",
                    "Leave room for a focused check after the edit or schedule it after the edit in this group.")
        return None


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
