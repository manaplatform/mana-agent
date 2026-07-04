# Agent Behavior

This document describes the expected behavior of the coding agent used by
`mana-agent`.

## Behavior Principles

The agent should:

- understand the user request and current context,
- gather repository evidence before concluding,
- prefer direct citations from repository files,
- make concrete changes only when evidence supports them,
- run checks after edits when possible,
- report what changed and what was not verified.

## Typical Workflow

1. Clarify the task.
2. Search the repository for relevant code or docs.
3. Read the source files that support the answer or change.
4. Edit only the necessary files.
5. Verify the change with tests or smoke checks.
6. Summarize the result with file citations.

## Normal Auto Chat

`mana-agent chat` supports natural-language requests without requiring slash
commands. Normal messages are classified into a bounded mode before the agent
uses tools:

- answer-only: questions such as `Where is this handled?`, `What calls this?`,
  or `Explain this flow`.
- plan-only: requests such as `Give me a plan for this feature` or `Suggest an
  approach`.
- edit: implementation requests such as `Fix this bug`, `Add this command`, or
  `Rename this module`.
- review: requests such as `Review my diff` or `Check what is wrong`.
- verify: requests such as `Run tests` or `Verify this`.
- analyze: bounded project or module analysis requests.

The normal auto router is intentionally small and fast. It uses targeted search,
caps candidate files and file reads, limits discovery rounds, and stops once it
has enough evidence to answer or act. It does not perform a full repository
analysis unless the user explicitly asks for analysis.

Only edit mode can expose mutation tools such as `apply_patch`, `write_file`,
`create_file`, or `delete_file`. Answer, plan, review, verify, and analyze modes
are read-only with respect to source files. Short follow-ups such as `continue`,
`do it`, or `verify` reuse compact state from the previous normal chat turn
instead of rediscovering from scratch.

## Tool Execution Hierarchy

The coding agent owns planning and steering, but it does not execute repository
tools directly. For tool-capable work, it builds the checklist, carries
structured edit intent such as `requires_edit` and `target_files`, and observes
the live queue results.

Tool execution must pass through `agent_work_queue.QueueManager`, which seeds
and runs `AgentWorkQueue` from the same queue module. The queue runner owns
readiness, dependency handling, retries, deduplication, and event publishing.
`CodingAgentSniffer` remains in `agent_work_queue_adapters` and reacts to
completed jobs by emitting follow-up `WorkItem`s for reads, edits, and
verification.

The worker process is the only layer that invokes the tool-capable `AskAgent`
runtime. If the coding-agent session has no queue manager attached, the request
is blocked as unavailable instead of falling back to direct `ask_agent.run*` or
bare worker calls.

## Prompt Cache Boundary

The coding agent prompt is split into stable and ephemeral layers. Stable prompt
state contains core identity, tool rules, agent behavior rules, a compact skill
index, repository rules from `AGENTS.md`, and safety/verification rules. This
state is cached per `CodingAgent` session and rebuilt only when stable inputs
change: prompt template version, mana-agent version, enabled tools, skill index,
repository rules, identity/rules, or model/provider profile.

The compact skill index contains only `name`, `description`, and `trigger` for
each discovered skill. Full `SKILL.md` bodies are not stable prompt content.
When the current task, detected files, mode, or prior results match a trigger,
the agent loads the full body through `read_skill(skill_name)` and keeps it in
ephemeral context for that turn.

Per-turn context is appended separately as an ephemeral developer/context
message. It contains the current task, detected mode, retrieved snippets,
summarized tool results, recent local summary, and temporary constraints. Current
user messages, retrieved files, tool output, turn numbers, patches, command
output, and temporary plans must not enter the stable cache key. Oversized
ephemeral context is trimmed or summarized before it is resent.

## Reporting Expectations

When finishing a task, the agent should report:

- changed files,
- key checks run,
- any skipped checks,
- remaining risks or unknowns.

## In-chat Slash Commands

Some chat inputs are intercepted before the model runs. These are deterministic,
read-only operations that never invoke the LLM or coding agent:

- `/flow` — inspect or reset the active coding flow.
- `/analyze` — analyze the current project and write report artifacts under
  `.mana/` (`json`, `markdown`/`md`, `html`, `dot`, `graphml`, `mermaid`, or
  `all`). With no arguments it opens a numbered format menu. The only side effect
  is writing the selected `.mana/` artifacts; source files are never modified.

Anything that is not a recognized slash command is treated as a normal request
and routed through normal auto chat. Slash commands always override the auto
router.

## Related Docs

- [Architecture](./08-architecture.md)
- [Tool System](./13-tool-system.md)
- [README](../README.md)
