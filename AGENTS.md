# AGENTS.md

# Mana-Agent Repository Agent Instructions

These instructions apply to the entire repository.

Mana-Agent is a repository-aware AI coding and analysis tool. It provides CLI workflows for chat, codebase analysis, planning, multi-agent execution, memory-aware decisions, tool routing, and automated code modification.

Agents working in this repository must prioritize correctness, safety, maintainability, minimal changes, and verifiable results.

---

## 1. Core Working Rules

* Inspect the current repository state before editing files.
* Preserve user changes and avoid reverting unrelated work.
* Keep changes focused on the requested task.
* Follow existing project patterns before introducing new structure.
* Read relevant files before making code changes.
* Do not guess implementation details when the repository can be inspected.
* Do not modify unrelated files.
* Do not remove existing behavior unless explicitly requested.
* Do not commit, push, tag, release, or publish unless explicitly requested.
* Run the most relevant checks or tests when code changes are made.
* If checks cannot be run, clearly state why.

---

## 2. Required Workflow

Use this workflow for most tasks:

```text
Inspect → Understand → Plan → Edit → Verify → Update CHANGELOG → Summarize
```

### 2.1 Inspect

Before editing, inspect the repository state.

Recommended command:

```bash
git status --short
```

Also inspect relevant files, tests, CLI entrypoints, prompt builders, tool managers, or runtime modules based on the task.

### 2.2 Understand

Before making changes:

* Identify the exact requested behavior.
* Locate the current implementation.
* Check existing tests.
* Check related documentation.
* Understand current project conventions.

### 2.3 Plan

For non-trivial changes, create a concise implementation plan.

The plan should include:

* Files likely to change.
* Main implementation steps.
* Verification strategy.
* Any risks or compatibility concerns.

### 2.4 Edit

When editing:

* Make minimal, focused changes.
* Prefer small functions and clear names.
* Preserve public APIs unless a breaking change is explicitly requested.
* Keep backward compatibility where possible.
* Avoid broad rewrites.
* Avoid hardcoded task-specific hacks.
* Avoid duplicate logic.
* Avoid hidden global state.

### 2.5 Verify

Run the most relevant checks for the changed area.

Examples:

```bash
python -m pytest
python -m pytest tests/<relevant_test_file>.py
python -m mana-agent --help
python -m mana-agent chat --help
python -m mana-agent analyze --help
python -m mana-agent plan --help
```

Use targeted tests first, then broader tests when core behavior changes.

### 2.6 Update CHANGELOG

Every repository change must update `CHANGELOG.md`.

### 2.7 Summarize

Final responses should clearly include:

* What changed.
* Files changed.
* Verification performed.
* Any remaining risks or notes.

---

## 3. Change Log Rule

Update `CHANGELOG.md` with every repository change.

Each entry must include:

* The date of the change.
* A short summary of what changed.
* Any verification performed.
* A note when verification was not run.

Recommended format:

```markdown
## YYYY-MM-DD

- Updated <area> to <summary of change>.
  - Verification: `<command>` passed.
```

If verification was not run:

```markdown
## YYYY-MM-DD

- Updated <area> to <summary of change>.
  - Verification: Not run. Reason: <reason>.
```

Do not skip the changelog unless the user explicitly says not to update it.

---

## 4. Repository Goals

Mana-Agent should provide:

* A professional CLI experience.
* Chat-first repository interaction.
* Codebase analysis and planning.
* Multi-agent orchestration.
* Tool-aware execution.
* Memory-aware decision making.
* Stable prompt construction.
* Progressive skill loading.
* Efficient token usage.
* Reliable patching and verification.
* Extensible architecture for future agents, tools, and workflows.

---

## 5. CLI Guidelines

The CLI should remain user-friendly, professional, and predictable.

Expected commands include:

```bash
mana-agent chat
mana-agent analyze
mana-agent plan
```

When no command is provided, the CLI should guide the user interactively.

CLI behavior should include:

* Clear banners.
* Readable terminal output.
* Helpful errors.
* Clean exit behavior.
* Quiet logs by default.
* Verbose logs only when requested.
* No noisy internal traces unless debug or verbose mode is enabled.

Do not remove old commands unless explicitly requested.

---

## 6. Chat Mode Guidelines

Chat mode is the primary user-facing workflow.

Chat mode should support:

* Repository questions.
* Code edits.
* Planning.
* Analysis.
* Tool execution.
* Multi-agent reasoning.
* Memory-aware decisions.

Chat mode must not spam logs by default.

Verbose mode may show:

* Tool calls.
* Sub-agent activity.
* Memory usage.
* Routing decisions.
* Token usage.
* Verification steps.

---

## 7. Analyze Mode Guidelines

Analyze mode should inspect repositories and produce useful findings.

It should:

* Avoid endless loops.
* Avoid repeated file reads when unchanged.
* Avoid over-discovery.
* Read enough context before reaching conclusions.
* Produce actionable output.
* Support future report generation.

Analyze mode must not continue discovering files after enough relevant context has been selected.

---

## 8. Plan Mode Guidelines

Plan mode should produce implementation plans before code changes.

A good plan includes:

* Scope.
* Affected files.
* Architecture impact.
* Step-by-step implementation.
* Verification strategy.
* Rollback or safety notes when relevant.

Plan mode should ask clarifying questions only when the task cannot be safely understood from repository context.

---

## 9. Multi-Agent Runtime Guidelines

Mana-Agent is designed around a hierarchical multi-agent runtime.

Expected structure:

```text
Main Agent
  → Taskboard
    → Sub-agents
      → Queue Manager
        → Tool Manager
          → Tools
```

Agents must have:

* Stable IDs.
* Clear roles.
* Bounded responsibilities.
* Permission-aware memory access.
* Communication paths for decisions.
* Verification steps before completion.

The main agent is responsible for:

* Understanding the task.
* Creating or updating the taskboard.
* Decomposing work.
* Delegating focused subtasks.
* Verifying final results.
* Producing the final response.

Sub-agents are responsible for:

* Focused execution.
* Reporting results.
* Avoiding duplicate work.
* Respecting permission level.
* Returning clear completion status.

Agents should communicate when decisions affect shared state, architecture, tool usage, or final output.

---

## 10. Taskboard Guidelines

The taskboard should track:

* Task ID.
* Parent task ID when applicable.
* Agent or sub-agent owner.
* Current status.
* Required files or tools.
* Dependencies.
* Verification requirements.
* Completion result.

Avoid creating duplicate tasks.

Before creating a new task, check whether an equivalent task already exists or has already been completed.

---

## 11. Memory Guidelines

Memory must improve decisions and reduce duplicate work.

Use memory to:

* Avoid duplicate tasks.
* Avoid reading unchanged files repeatedly.
* Cache file observations.
* Track completed subtasks.
* Track tool results.
* Preserve useful project context.
* Improve routing decisions.

Memory must not:

* Override fresh repository state.
* Hide errors.
* Cause stale decisions.
* Bypass verification.
* Leak private or unrelated context.
* Replace reading files when fresh code state is required.

File-read memory should be invalidated when:

* File content changes.
* Git state changes.
* Branch changes.
* Task scope changes.
* The user requests a refresh.
* Verification output contradicts cached assumptions.

---

## 12. Prompt System Guidelines

Mana-Agent should use stable prompt layers.

Recommended structure:

```text
Stable Prompt
  - identity
  - tool rules
  - mode rules
  - compact skill index
  - project memory snapshot

Ephemeral Prompt
  - current task
  - selected files
  - retrieved context
  - latest tool results
  - verification output
```

Do not rebuild the full system prompt unnecessarily every turn.

Prioritize:

* Prompt cache efficiency.
* Low token usage.
* Stable behavior.
* Clear separation between permanent rules and current-task context.

---

## 13. Skills System Guidelines

Skills should live at the repository root:

```text
/skills/<skill_name>/SKILL.md
```

Use progressive skill loading.

The default prompt should include only:

* Skill name.
* Description.
* Trigger condition.

Load the full `SKILL.md` only when the skill is relevant.

Agents must not inject every skill file into every prompt.

---

## 14. Tool Execution Rules

Use tools carefully and efficiently.

Prefer batch operations when possible:

* Batch file reads.
* Batch searches.
* Batch patches.
* Single verification scripts.

Avoid:

* Many tiny repeated tool calls.
* Reading the same file multiple times without changes.
* Running expensive commands unnecessarily.
* Using tools without a clear purpose.
* Continuing tool loops without new information.

Every tool result should be interpreted before the next action.

---

## 15. File Reading Rules

Before editing a file, read the relevant current content.

Do not rely only on memory or assumptions.

Avoid reading:

* `.pyc` files.
* Virtual environments.
* Build outputs.
* Cache directories.
* Generated artifacts.
* Unrelated documentation.

Common ignored paths include:

```text
.venv/
venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
build/
*.egg-info/
node_modules/
```

---

## 16. Code Style Guidelines

Use professional Python practices.

Preferred style:

* Clear names.
* Type hints where useful.
* Small functions.
* Explicit exceptions.
* Dataclasses or typed models when helpful.
* Deterministic logic.
* Readable tests.
* Clear separation of responsibilities.

Avoid:

* Broad `except Exception` without handling.
* Silent failures.
* Magic strings spread across the codebase.
* Circular imports.
* Hidden side effects.
* Hardcoded absolute paths.
* Large functions with multiple responsibilities.

---

## 17. Testing Guidelines

When changing behavior, add or update tests.

Tests should cover:

* Normal path.
* Edge cases.
* Failure path.
* Backward compatibility.
* CLI behavior when relevant.
* Memory behavior when relevant.
* Multi-agent routing when relevant.
* Tool execution behavior when relevant.

Do not remove tests unless they are obsolete and replaced by better coverage.

Prefer targeted tests for fast verification, then run broader tests when core systems change.

---

## 18. Git Guidelines

Before making changes, inspect repository state:

```bash
git status --short
```

Protect user work:

* Do not overwrite uncommitted changes.
* Do not revert unrelated edits.
* Do not delete branches or tags unless explicitly requested.
* Do not force-push unless explicitly requested.
* Do not amend commits unless explicitly requested.
* Do not create commits unless explicitly requested.

For conflict resolution:

* Inspect both sides of the conflict.
* Preserve intended behavior from both sides when possible.
* Keep the resolution focused.
* Run relevant tests after resolving.
* Continue rebase only after conflicts are fully resolved.

---

## 19. Documentation Guidelines

Update documentation when user-facing behavior changes.

Relevant files may include:

* `README.md`
* `CHANGELOG.md`
* CLI help text
* Docs under `docs/`
* Examples
* Workflow documentation
* `AGENTS.md`

Documentation should be clear, practical, and current.

---

## 20. Error Handling Guidelines

Errors should be actionable.

Good errors explain:

* What failed.
* Why it likely failed.
* How to fix it.
* Whether the task can continue.

Avoid exposing noisy internal stack traces to end users unless verbose or debug mode is enabled.

---

## 21. Logging Guidelines

Default output should be clean.

Logs should not pollute the chat UI unless verbose mode is enabled.

Use log levels correctly:

* `DEBUG` for internal details.
* `INFO` for important lifecycle events.
* `WARNING` for recoverable issues.
* `ERROR` for failures.

Avoid printing tool-worker internals, memory traces, or routing logs in normal mode.

---

## 22. Performance Guidelines

Mana-Agent should be efficient with:

* Tokens.
* File reads.
* Tool calls.
* Subprocesses.
* Prompt construction.
* Memory lookups.
* Test execution.

Avoid repeated expensive work.

Cache only when invalidation is clear and safe.

---

## 23. Security Guidelines

Do not expose secrets.

Never print or include:

* API keys.
* Tokens.
* Passwords.
* Private SSH keys.
* Environment secrets.
* Full `.env` contents.

When reading configuration files, redact sensitive values in summaries.

Do not execute untrusted scripts without clear reason.

Do not send secrets to tools, logs, prompts, or generated reports.

---

## 24. Dependency Guidelines

Do not add new dependencies unless necessary.

Before adding a dependency:

* Check whether the standard library is enough.
* Check existing dependencies.
* Consider maintenance cost.
* Consider package size and compatibility.
* Update packaging files correctly.

If dependencies change, update:

* `pyproject.toml`.
* Lock files if used.
* Documentation if install behavior changes.
* `CHANGELOG.md`.

---

## 25. Release Workflow Guidelines

Release automation should be reliable and reproducible.

A professional release workflow should:

* Run tests.
* Build packages.
* Create artifacts.
* Publish only from trusted branches or tags.
* Support intended platforms.
* Generate changelog or release notes.
* Avoid publishing broken builds.

Version values should follow standard Python package versioning.

Examples:

```toml
version = "0.0.10"
version = "0.1.0"
version = "1.0.0"
```

---

## 26. Backward Compatibility Rules

Preserve existing behavior unless the task explicitly requires a breaking change.

Before changing behavior, check:

* Existing tests.
* CLI help output.
* Public function names.
* Configuration keys.
* Prompt contracts.
* Tool schemas.
* Memory schema.
* Documentation.

When a breaking change is required, document it clearly in `CHANGELOG.md`.

---

## 27. Definition of Done

A task is complete only when:

* The requested behavior is implemented.
* Relevant files were inspected.
* Changes are minimal and focused.
* User changes were preserved.
* Relevant tests or checks were run when possible.
* `CHANGELOG.md` was updated.
* Documentation was updated when needed.
* No unrelated files were modified.
* Final summary clearly explains the result.

---

## 28. Final Response Format

When finishing a coding task, respond with:

```text
Implemented:
- ...

Changed:
- ...

Verified:
- ...

Notes:
- ...
```

If something could not be completed, explain exactly what remains and why.

---

## 29. Important Instruction

Do not skip any part of the task.

If the task is large, complete it in safe, logical steps.

If there is uncertainty, inspect the repository first. Ask only when the ambiguity blocks safe progress.

Always prefer a correct, verified, maintainable solution over a fast but fragile one.
