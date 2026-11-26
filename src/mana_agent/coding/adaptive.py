"""Mode-specific contracts for the supervised native tool loop."""

from dataclasses import dataclass
from itertools import takewhile
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from mana_agent.agent.mode import VerificationCommand
from mana_agent.agent.plan_service import Question
from mana_agent.coding.evidence_tools import INTEGRATE, INVESTIGATE
from mana_agent.coding.models import CODING_RESULT_TOOL, ISOLATED_CHECK_TOOL
from mana_agent.coding.scheduler import WRITE_TOOL_NAMES
from mana_agent.repository_evidence.models import RuntimeWiringEvidence


EVIDENCE_REFERENCE_TOOL = "repository_evidence_get"
REPOSITORY_EVIDENCE_TOOLS = frozenset({
    "ls", "list_files", "batch_list_files", "read_file", "repo_batch_read", "batch_read_files",
    "repo_search", "repo_batch_search", "batch_search_files", "find_symbols", "call_graph",
    "repository_evidence_query", EVIDENCE_REFERENCE_TOOL, INVESTIGATE, INTEGRATE,
})
SCOPED_SOURCE_TOOLS = frozenset({"ls", "read_file", "repo_search"})
EVIDENCE_NEED = "evidence_need"
EVIDENCE_NEED_MAX_LENGTH = 500
EVIDENCE_NEED_DESCRIPTION = "Name the missing fact and the specific edit or check the answer could change."
BUDGET_RENEWAL_TOOL = "renew_coding_budget"
SHELL_TOOL_NAMES = frozenset({"run_command", "run_script_once", "shell"})
USE_REPOSITORY_TOOL = "USE_REPOSITORY_TOOL"
REPOSITORY_INSPECTION_DESCRIPTION = (
    "Use read_file with a path or repo_search with literal paths or a targeted glob for source inspection. "
    "If the search requires an unsupported option, select run_command with a literal argv array. "
    "For the top-level layout, call ls with only evidence_need. Use git_status or git_diff for checkout state. "
    f"Select {ISOLATED_CHECK_TOOL} separately with direct argv for a build or check."
)
VERIFICATION_DESCRIPTION = (
    "Run a direct build or check in the assigned checkout. Set cwd to '.' or a repository-relative directory. "
    "Use setup_argv for prerequisites in the same checkout. Use file editing tools for source changes."
)
_PATCH_TOOLS = frozenset({"apply_patch", "apply_patch_batch", "batch_apply_patches"})
_SOURCE_INSPECTION_PROGRAMS = frozenset({"cat", "find", "grep", "egrep", "fgrep", "head", "ls", "rg", "sed", "tail", "wc"})
_POSIX_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish"})
_WINDOWS_SHELLS = frozenset({"cmd", "powershell", "pwsh"})
_EDIT_AND_CHECK_TOOLS = WRITE_TOOL_NAMES | {ISOLATED_CHECK_TOOL}
_BROAD_EVIDENCE_TOOLS = REPOSITORY_EVIDENCE_TOOLS - SCOPED_SOURCE_TOOLS - {EVIDENCE_REFERENCE_TOOL}


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
                f"Supply {EVIDENCE_NEED} with each listing, read, or search. Use existing observations and prefer an edit "
                "or check whenever the work supports it. If a search location is unknown, call ls with only evidence_need.")

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
    wiring: RuntimeWiringEvidence | None = Field(
        default=None, description="Required for complete when the assignment changes production runtime behavior.",
    )

    @model_validator(mode="after")
    def validate_questions(self):
        if bool(self.questions) != (self.disposition == "clarify"):
            raise PydanticCustomError("coding_result_questions_required", "Only the clarify disposition requires questions")
        if len({question.field_id for question in self.questions}) != len(self.questions):
            raise PydanticCustomError("coding_result_question_ids", "Clarification field IDs must be unique")
        return self


class AdaptiveBudgetRenewal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(pattern=r"\S", max_length=500, description="What advanced and which edit or check still needs budget.")


class AdaptiveVerificationCommand(VerificationCommand):
    setup_argv: tuple[tuple[str, ...], ...] = Field(default=(), max_length=4)
    timeout: int = Field(default=120, ge=1, le=3600)

    @model_validator(mode="after")
    def validate_direct_commands(self):
        if self.image:
            raise ValueError("Checkout verification runs locally without an image.")
        for argv in (*self.setup_argv, self.argv):
            if not argv or not argv[0].strip() or any("\x00" in argument for argument in argv):
                raise ValueError("Each setup or verification command needs a literal argv array.")
            executable = PurePosixPath(argv[0].replace("\\", "/")).name.lower().removesuffix(".exe")
            flags = tuple(takewhile(lambda argument: argument.startswith("-") and argument != "--", argv[1:]))
            if (executable in _POSIX_SHELLS and any("c" in flag[1:] for flag in flags if not flag.startswith("--"))
                    or executable in _WINDOWS_SHELLS and any(
                        flag.lower() in {"/c", "/k", "-c", "-command", "-encodedcommand", "-enc"} for flag in argv[1:]
                    )):
                raise ValueError("Select direct argv commands and use setup_argv for prerequisites.")
        return self


def validate_adaptive_tool(name: str, arguments: dict[str, Any]) -> tuple[str, str] | None:
    if name in _PATCH_TOOLS:
        from mana_agent.tools.apply_patch import PatchFormatError, extract_patch_operations

        patches = [arguments] if name == "apply_patch" else arguments.get("patches")
        limit = 50 if name == "batch_apply_patches" else 20
        if not isinstance(patches, list) or not patches or len(patches) > limit:
            return "INVALID_PATCH_FORMAT", f"Select one to {limit} Codex patches, each in a patch field."
        for index, row in enumerate(patches, start=1):
            try:
                extract_patch_operations(row.get("patch") if isinstance(row, dict) else None)
            except PatchFormatError as exc:
                return ("INVALID_PATCH_FORMAT", f"Patch {index}: {exc}. Use *** Begin Patch, a Codex file operation, "
                        "and *** End Patch. Use multi_edit_file for exact replacements in one file.")
    elif name == ISOLATED_CHECK_TOOL:
        try:
            command = AdaptiveVerificationCommand.model_validate(arguments)
        except ValueError:
            return "INVALID_VERIFICATION_COMMAND", VERIFICATION_DESCRIPTION
        if command.uses_git_checkout or any(VerificationCommand(argv=argv).uses_git_checkout for argv in command.setup_argv):
            return "INVALID_VERIFICATION_COMMAND", "Use a project build or check for verification."
    elif name in SHELL_TOOL_NAMES:
        from mana_agent.tools.shell_exec import split_shell_command

        action = arguments.get("action")
        commands = action.get("commands") if isinstance(action, dict) else arguments.get("commands")
        command = (commands[0] if isinstance(commands, list) and len(commands) == 1 else
                   arguments.get("command") or arguments.get("script") or arguments.get("cmd"))
        if isinstance(command, str):
            try:
                argv = split_shell_command(command)
            except ValueError:
                return None  # The ordinary shell tool owns command-format errors.
            if argv and (PurePosixPath(argv[0]).name in _SOURCE_INSPECTION_PROGRAMS
                         or (argv[0] == "git" and argv[1:2] in (["status"], ["diff"], ["ls-files"]))):
                return USE_REPOSITORY_TOOL, REPOSITORY_INSPECTION_DESCRIPTION
    return None


def renewal_tool():
    return {"type": "function", "function": {
        "name": BUDGET_RENEWAL_TOOL,
        "description": "Request an accounting-backed extension to finish this assignment. Call this tool alone.",
        "parameters": AdaptiveBudgetRenewal.model_json_schema(),
    }}


def result_tool():
    return {"type": "function", "function": {
        "name": CODING_RESULT_TOOL,
        "description": "Return a completed coding result, material clarification questions, or a concrete blocker.",
        "parameters": AdaptiveCodingResult.model_json_schema(),
    }}
