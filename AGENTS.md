# AGENTS.md

# Mana-Agent Repository Agent Instructions

These instructions apply to the entire repository.

Mana-Agent is a repository-aware AI coding and analysis tool. It provides CLI workflows for chat, codebase analysis, planning, multi-agent execution, memory-aware decisions, tool routing, and automated code modification.

Agents working in this repository must prioritize correctness, safety, maintainability, minimal changes, model-driven decisions, and verifiable results.

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
* Do not use fallback logic for routing, tool choice, workflow choice, planning, search, editing, verification, or final response generation.
* All meaningful behavior must depend on explicit model decisions represented in code.

---

## 2. Required Workflow

Use this workflow for most tasks:

```text
Inspect → Understand → Plan → Model Decision → Validate Decision → Edit → Verify → Update CHANGELOG → Summarize
```

No workflow may skip the model decision step when a meaningful decision is required.

### 2.1 Inspect

Before editing, inspect the repository state.

Recommended command:

```bash
git status --short
```

Also inspect relevant files, tests, CLI entrypoints, prompt builders, tool managers, runtime modules, model decision schemas, or routing modules based on the task.

### 2.2 Understand

Before making changes:

* Identify the exact requested behavior.
* Locate the current implementation.
* Check existing tests.
* Check related documentation.
* Understand current project conventions.
* Identify where decisions are currently made.
* Identify and remove fallback logic related to the requested area.

### 2.3 Plan

For non-trivial changes, create a concise implementation plan.

The plan should include:

* Files likely to change.
* Main implementation steps.
* Required model decision objects or schemas.
* Verification strategy.
* Any risks or compatibility concerns.

### 2.4 Model Decision

Before selecting a workflow, tool, agent, search method, edit action, or verification path, Mana-Agent must obtain a structured model decision.

The decision must determine:

* Task intent.
* Required workflow.
* Required agent or sub-agent.
* Required tools.
* Required context.
* Required files.
* Required verification.
* Whether the task is safe to continue.
* Whether the task is complete.

Model decisions must be explicit, structured, and validated before execution.

### 2.5 Validate Decision

Before executing a model-selected action:

* Validate the decision schema.
* Validate tool names.
* Validate tool arguments.
* Validate permissions.
* Validate file paths.
* Validate safety constraints.
* Validate that required context exists.

If validation fails, stop safely.

Do not continue with fallback behavior.

### 2.6 Edit

When editing:

* Make minimal, focused changes.
* Prefer small functions and clear names.
* Preserve public APIs unless a breaking change is explicitly requested.
* Keep backward compatibility where possible, except when removing fallback behavior explicitly requested by the user.
* Avoid broad rewrites.
* Avoid hardcoded task-specific hacks.
* Avoid keyword routing.
* Avoid heuristic fallback classifiers.
* Avoid default fallback tools.
* Avoid duplicate logic.
* Avoid hidden global state.
* Avoid any behavior that bypasses the model decision layer.

### 2.7 Verify

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

When changing model-decision behavior, tests must prove that invalid or missing model decisions stop safely and do not trigger fallback behavior.

### 2.8 Update CHANGELOG

Every repository change must update `CHANGELOG.md`.

### 2.9 Summarize

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
* Model-driven routing.
* Model-driven tool selection.
* Model-driven planning.
* Stable prompt construction.
* Progressive skill loading.
* Efficient token usage.
* Reliable patching and verification.
* Extensible architecture for future agents, tools, and workflows.

Mana-Agent must not depend on hardcoded keyword behavior or fallback functionality.

---

## 5. Model-Decision-Only Execution Policy

Mana-Agent must not use fallback logic for routing, planning, searching, tool selection, editing, verification, or final response generation.

All meaningful behavior must depend on explicit model decisions represented in code through typed decision objects, schemas, taskboard entries, agent messages, or structured planner outputs.

### 5.1 Strict Rules

* Do not use keyword-based routing as a fallback.
* Do not route tasks with checks like `if "search" in user_input`, `if "edit" in prompt`, or similar hardcoded text matching.
* Do not use heuristic fallback classifiers when the model decision is missing, invalid, empty, or uncertain.
* Do not silently continue with default behavior when a model decision fails.
* Do not replace a failed model decision with a fixed built-in action.
* Do not create hidden backup flows that bypass the model decision layer.
* Do not hardcode tool selection, agent selection, file selection, search behavior, or workflow selection based on user text.
* Do not infer intent from keywords when the repository, memory, taskboard, or model decision layer should decide.
* Do not use “best effort fallback” behavior that performs a task without a valid model decision.
* Do not downgrade from model-driven execution to simple static code logic.
* Do not add emergency fallback handlers that choose actions automatically.
* Do not add default routing decisions for unknown, invalid, or failed model outputs.

### 5.2 Required Behavior

When a decision is needed, Mana-Agent must ask the appropriate model layer to produce a structured decision.

The model decision should determine:

* Whether the task needs repository inspection.
* Whether the task needs web search.
* Whether the task needs repo search.
* Whether the task needs file reading.
* Whether the task needs code modification.
* Whether the task needs verification.
* Whether the task needs summarization.
* Which agent or sub-agent should handle the task.
* Which tools are allowed for the next step.
* Which files are relevant.
* Whether more context is required.
* Whether the task is safe to continue.
* Whether the task is complete.

### 5.3 Failure Behavior

If the model decision is unavailable, invalid, incomplete, unsafe, or fails schema validation:

* Stop the current workflow safely.
* Return a clear actionable error.
* Explain which decision failed.
* Do not execute a fallback action.
* Do not guess the next step.
* Do not continue with keyword, heuristic, or default behavior.
* Do not call a default tool.
* Do not create a default task.
* Do not choose a default agent.
* Do not continue with stale memory as a replacement for the failed decision.

### 5.4 Allowed Deterministic Logic

Deterministic code is allowed only for mechanical execution after a model decision has been made.

Allowed deterministic logic includes:

* Schema validation.
* Permission checks.
* Safety checks.
* Tool argument validation.
* File existence checks.
* Cache invalidation checks.
* Retry for transport or API errors.
* Formatting model decisions for display.
* Executing the exact tool call selected by the model decision.
* Verifying that model-selected files, tools, or actions are valid.
* Checking whether selected files still exist.
* Checking whether selected tools are registered.
* Checking whether selected commands are allowed.

Deterministic code must not choose the task intent, workflow, tool, agent, search behavior, file set, edit path, verification path, or final action unless that choice was already made by the model decision layer.

### 5.5 Decision Contract

Every major workflow should follow this contract:

```text
User Input
  → Context Collection
  → Model Decision
  → Decision Validation
  → Tool / Agent Execution
  → Verification
  → Model Summary / Final Response
```

No workflow may skip the `Model Decision` step and replace it with fallback logic.

### 5.6 Forbidden Behavior Examples

Forbidden:

```python
if "search" in user_input.lower():
    return run_web_search(user_input)
```

Forbidden:

```python
if "edit" in user_input.lower() or "fix" in user_input.lower():
    return run_coding_agent(user_input)
```

Forbidden:

```python
decision = model_decide(task)
if not decision:
    decision = Decision(tool="repo_search")
```

Forbidden:

```python
if planner_failed:
    return simple_keyword_router(user_input)
```

Forbidden:

```python
try:
    decision = model_decide(task)
except Exception:
    return default_chat_response(task)
```

Forbidden:

```python
if not selected_tool:
    selected_tool = "repo_search"
```

Forbidden:

```python
if not selected_agent:
    selected_agent = "main_agent"
```

Allowed:

```python
decision = model_decide(task)
validated = validate_decision(decision)
execute_model_selected_action(validated)
```

Allowed failure behavior:

```python
decision = model_decide(task)

if not decision or not decision.is_valid:
    raise DecisionRequiredError(
        "The model did not return a valid routing decision. No fallback action was executed."
    )
```

### 5.7 Implementation Requirement

When removing existing fallback behavior, replace it with:

* A typed decision model.
* A model prompt that requests the decision.
* Strict schema validation.
* Explicit failure handling.
* Tests proving fallback behavior is not used.
* Clear changelog documentation.

Tests should verify that when the model decision is missing or invalid, Mana-Agent stops safely instead of using keyword matching, default tools, default agents, or heuristic routing.

---

## 6. CLI Guidelines

The CLI should remain user-friendly, professional, and predictable.

Expected commands include:

```bash
mana-agent chat
mana-agent analyze
mana-agent plan
```

When no command is provided, the CLI should guide the user interactively.

First-run interactive setup must use the dedicated TUI/config modules and persist user-level settings under `~/.mana`. Preserve the precedence order: CLI flags, environment variables and `.env`, `~/.mana` config/secrets, then safe defaults. Non-interactive runs must not block on prompts.

CLI behavior should include:

* Clear banners.
* Readable terminal output.
* Helpful errors.
* Clean exit behavior.
* Quiet logs by default.
* Verbose logs only when requested.
* No noisy internal traces unless debug or verbose mode is enabled.

Do not remove old commands unless explicitly requested.

CLI command selection, chat routing, analysis routing, and planning routing must not depend on keyword fallback behavior.

---

## 7. Chat Mode Guidelines

Chat mode is the primary user-facing workflow.

Chat mode should support:

* Repository questions.
* Code edits.
* Planning.
* Analysis.
* Tool execution.
* Multi-agent reasoning.
* Memory-aware decisions.
* Model-driven routing.

Chat mode must not spam logs by default.

Verbose mode may show:

* Tool calls.
* Sub-agent activity.
* Memory usage.
* Routing decisions.
* Token usage.
* Verification steps.

Chat mode must not route based on hardcoded keywords.

For example:

* Do not detect search tasks using `"search"` keyword checks.
* Do not detect edit tasks using `"fix"`, `"update"`, or `"change"` keyword checks.
* Do not detect analysis tasks using `"analyze"` keyword checks.
* Do not select tools from user text with hardcoded string matching.

Instead, chat mode must request a structured model decision and execute only the validated model-selected workflow.

---

## 8. Analyze Mode Guidelines

Analyze mode should inspect repositories and produce useful findings.

It should:

* Avoid endless loops.
* Avoid repeated file reads when unchanged.
* Avoid over-discovery.
* Read enough context before reaching conclusions.
* Produce actionable output.
* Support future report generation.
* Depend on model decisions for scope, file selection, and next actions.

Analyze mode must not continue discovering files after enough relevant context has been selected.

Analyze mode must not use fallback file discovery or keyword-driven search when the model decision layer fails.

If the analysis decision is missing or invalid, stop safely and report the failed decision.

---

## 9. Plan Mode Guidelines

Plan mode should produce implementation plans before code changes.

A good plan includes:

* Scope.
* Affected files.
* Architecture impact.
* Step-by-step implementation.
* Verification strategy.
* Rollback or safety notes when relevant.

Plan mode should ask clarifying questions only when the task cannot be safely understood from repository context.

Plan mode must not use fallback planning templates as a substitute for a model decision.

Static plan formatting is allowed only after the model has produced the actual plan content and decisions.

---

## 10. Multi-Agent Runtime Guidelines

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
* Typed decision inputs and outputs.

The main agent is responsible for:

* Understanding the task.
* Creating or updating the taskboard.
* Decomposing work.
* Delegating focused subtasks.
* Verifying final results.
* Producing the final response.
* Ensuring every meaningful action is backed by a validated model decision.

Sub-agents are responsible for:

* Focused execution.
* Reporting results.
* Avoiding duplicate work.
* Respecting permission level.
* Returning clear completion status.
* Executing only validated model-selected actions.

Agents should communicate when decisions affect shared state, architecture, tool usage, memory, or final output.

No agent or sub-agent may use fallback routing, fallback tool selection, fallback task creation, or fallback verification behavior.

---

## 11. Taskboard Guidelines

The taskboard should track:

* Task ID.
* Parent task ID when applicable.
* Agent or sub-agent owner.
* Current status.
* Required files or tools.
* Dependencies.
* Verification requirements.
* Completion result.
* Source model decision ID or decision object reference.

Avoid creating duplicate tasks.

Before creating a new task, check whether an equivalent task already exists or has already been completed.

Task creation must be driven by a model decision.

Do not create fallback tasks when the model decision is missing, invalid, or uncertain.

---

## 12. Memory Guidelines

Memory must improve decisions and reduce duplicate work.

Use memory to:

* Avoid duplicate tasks.
* Avoid reading unchanged files repeatedly.
* Cache file observations.
* Track completed subtasks.
* Track tool results.
* Preserve useful project context.
* Improve routing decisions.
* Provide context to the model decision layer.

Memory must not:

* Override fresh repository state.
* Hide errors.
* Cause stale decisions.
* Bypass verification.
* Leak private or unrelated context.
* Replace reading files when fresh code state is required.
* Replace the model decision layer.
* Act as fallback routing.
* Choose tools, agents, or workflows without a validated model decision.

File-read memory should be invalidated when:

* File content changes.
* Git state changes.
* Branch changes.
* Task scope changes.
* The user requests a refresh.
* Verification output contradicts cached assumptions.
* The model decision requires fresh context.

---

## 13. Prompt System Guidelines

Mana-Agent should use stable prompt layers.

Recommended structure:

```text
Stable Prompt
  - identity
  - tool rules
  - mode rules
  - model-decision-only policy
  - compact skill index
  - project memory snapshot

Ephemeral Prompt
  - current task
  - selected files
  - retrieved context
  - latest tool results
  - verification output
  - current decision request
```

Do not rebuild the full system prompt unnecessarily every turn.

Prioritize:

* Prompt cache efficiency.
* Low token usage.
* Stable behavior.
* Clear separation between permanent rules and current-task context.
* Explicit model decision outputs.

Prompts must not instruct the model to rely on keyword fallback behavior.

Prompts should request structured decisions when actions are needed.

---

## 14. Skills System Guidelines

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

Skill selection must be model-driven.

Do not select skills through keyword fallback logic.

A deterministic skill registry may validate skill names and load selected skill files only after the model decision identifies the relevant skill.

---

## 15. Tool Execution Rules

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
* Selecting tools through keyword matching.
* Selecting tools through fallback defaults.

Every tool result should be interpreted before the next action.

Tool execution must follow this order:

```text
Model Decision → Decision Validation → Tool Execution → Result Interpretation
```

The tool manager may validate and execute a selected tool, but it must not independently choose the tool through fallback behavior.

---

## 16. File Reading Rules

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

File selection should be model-driven when the relevant files are not already known.

Deterministic ignore rules are allowed for safety and performance.

Do not use fallback file search when model-selected file context is missing or invalid.

---

## 17. Code Style Guidelines

Use professional Python practices.

Preferred style:

* Clear names.
* Type hints where useful.
* Small functions.
* Explicit exceptions.
* Dataclasses or typed models when helpful.
* Deterministic logic for validation and execution.
* Typed decision models.
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
* Keyword-routing helpers.
* Heuristic fallback classifiers.
* Default action fallbacks.
* Silent downgrade from model-driven execution to static behavior.

---

## 18. Testing Guidelines

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
* Model decision behavior when relevant.
* Invalid model decision behavior.
* Missing model decision behavior.
* No fallback execution behavior.

Do not remove tests unless they are obsolete and replaced by better coverage.

Prefer targeted tests for fast verification, then run broader tests when core systems change.

When removing fallback behavior, tests must prove:

* Missing decisions fail safely.
* Invalid decisions fail safely.
* No keyword router is called.
* No default tool is selected.
* No default agent is selected.
* No hidden fallback path executes.

---

## 19. Git Guidelines

Before making changes, inspect repository state:

```bash
git status --short
```

Mana-Agent Git operations must use the model-driven Git tool namespace when running inside the application. The model must select Git actions from task context, repository state, current changes, and safety policy; do not add keyword routing for words such as commit, push, or branch.

Supported canonical Git tools include:

```text
git.status
git.diff
git.log
git.show
git.branch
git.switch
git.checkout
git.create_branch
git.add
git.restore
git.stash
git.commit
git.push
git.pull
git.fetch
git.remote
git.tag
git.merge
git.rebase
git.revert
git.reset
git.clean
git.config
git.generic
git.help
```

`git.help(all=true)` must discover local Git commands from `git help -a`; do not maintain a permanent hardcoded inventory of every Git command. `git.generic` must pass argv lists to `subprocess.run(["git", *args], shell=False)` in the resolved repository root and return structured, redacted output.

Before committing, inspect `git status --short`, `git diff`, and `git diff --staged`; stage only files relevant to the current task; generate the commit message from the staged diff, request, changed files, and verification result. Do not use `git add .` unless the model has verified all changed files are relevant.

Before pushing, inspect status, current branch, remotes, and upstream. Push with `-u origin <branch>` only when no upstream exists. Never force-push by default.

Block destructive or history-rewrite actions such as `git reset --hard`, `git clean -fd`, `git branch -D`, `git push --force`, `git push --force-with-lease`, `git push --delete`, `git rebase --onto`, `git filter-branch`, `git update-ref`, `git reflog expire`, and `git gc --prune=now` unless explicit user intent and risk handling are present.

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

Do not use fallback conflict resolution strategies such as blindly accepting one side unless the user explicitly requested that exact behavior.

---

## 20. Documentation Guidelines

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

When model-decision behavior changes, documentation must explain:

* What decision object is used.
* What happens when the decision is invalid.
* That no fallback behavior is executed.
* How the behavior is verified.

---

## 21. Error Handling Guidelines

Errors should be actionable.

Good errors explain:

* What failed.
* Why it likely failed.
* How to fix it.
* Whether the task can continue.

Avoid exposing noisy internal stack traces to end users unless verbose or debug mode is enabled.

Decision errors must be explicit.

When a required model decision fails:

* Raise or return a clear decision error.
* Do not continue.
* Do not call fallback tools.
* Do not use default behavior.
* Do not hide the failure.

Recommended error style:

```text
Model decision failed: <decision_name>.
No fallback action was executed.
Reason: <validation_error>.
```

---

## 22. Logging Guidelines

Default output should be clean.

Logs should not pollute the chat UI unless verbose mode is enabled.

Use log levels correctly:

* `DEBUG` for internal details.
* `INFO` for important lifecycle events.
* `WARNING` for recoverable issues.
* `ERROR` for failures.

Avoid printing tool-worker internals, memory traces, or routing logs in normal mode.

When decision validation fails, log the failure clearly in verbose or debug mode without exposing secrets.

Do not log noisy fallback attempts because fallback attempts must not exist.

---

## 23. Performance Guidelines

Mana-Agent should be efficient with:

* Tokens.
* File reads.
* Tool calls.
* Subprocesses.
* Prompt construction.
* Memory lookups.
* Test execution.
* Model decision calls.

Avoid repeated expensive work.

Cache only when invalidation is clear and safe.

Caching must not replace required model decisions.

Performance optimizations must not introduce fallback routing, fallback search, fallback planning, or fallback tool selection.

---

## 24. Security Guidelines

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

Security checks are deterministic validation and may block execution.

Security checks must not choose alternate fallback actions.

---

## 25. Dependency Guidelines

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

Do not add dependencies to support fallback behavior.

---

## 26. Release Workflow Guidelines

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

Release workflows may use deterministic checks, but release decisions should remain explicit and safe.

Do not publish through fallback behavior when required release metadata or validation is missing.

---

## 27. Backward Compatibility Rules

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

Removing fallback behavior is an explicitly required architecture rule.

Backward compatibility must not preserve:

* Keyword routing.
* Heuristic fallback classifiers.
* Default fallback tools.
* Default fallback agents.
* Silent fallback behavior.
* Hidden backup workflows that bypass model decisions.

When a breaking change is required, document it clearly in `CHANGELOG.md`.

---

## 28. Definition of Done

A task is complete only when:

* The requested behavior is implemented.
* Relevant files were inspected.
* Changes are minimal and focused.
* User changes were preserved.
* Relevant tests or checks were run when possible.
* `CHANGELOG.md` was updated.
* Documentation was updated when needed.
* No unrelated files were modified.
* No fallback, keyword-routing, or heuristic decision path was introduced.
* Existing fallback behavior related to the task was removed or replaced.
* All meaningful workflow choices are made through validated model decision objects.
* Invalid or missing model decisions fail safely instead of continuing with default behavior.
* Final summary clearly explains the result.

---

## 29. Final Response Format

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

## 30. Important Instruction

Do not skip any part of the task.

If the task is large, complete it in safe, logical steps.

If there is uncertainty, inspect the repository first. Ask only when the ambiguity blocks safe progress.

Always prefer a correct, verified, maintainable solution over a fast but fragile one.

Do not use fallback behavior to avoid uncertainty.

When a model decision is required but unavailable, invalid, or unsafe, stop and report the decision failure clearly.

The correct behavior is:

```text
No valid model decision → no action executed.
```

The incorrect behavior is:

```text
No valid model decision → fallback tool / fallback route / fallback response.
```

Mana-Agent must be model-decision-first, validation-driven, and fallback-free.
