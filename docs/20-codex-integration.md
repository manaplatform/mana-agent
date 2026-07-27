# Coding runtimes: Codex and internal

Mana-Agent selects one coding backend before each coding turn. Codex remains
the default authoritative runtime when enabled; the internal backend uses
Mana-Agent's model-driven planner, repository tools, queue, reviewer, and
verifier when Codex is explicitly disabled or the internal backend is selected.
Once the shared model route selects a Codex coding turn, one Codex turn owns repository inspection, coding
decisions, planning, edits, review, and proportional verification. Mana-Agent
retains the outer chat route, worktree allocation, permissions, event streaming,
result normalization, and explicit merge control.

The official Codex SDK is currently TypeScript (`@openai/codex-sdk`) and wraps
the Codex CLI. There is no official `openai-codex` Python package or
`AsyncCodex` client. Mana-Agent therefore integrates from Python through the
official `codex app-server` JSON-RPC protocol, which exposes thread and turn
lifecycle methods, streaming notifications, and `turn/interrupt` cancellation.
See the [Codex app-server documentation](https://learn.chatgpt.com/docs/app-server).

## Responsibility boundary

```text
Shared model route selects a coding turn
  → Mana-managed isolated worktree for writes
    → one authoritative Codex thread and turn
      → inspect, decide, plan, edit, review, verify
        → normalized events and result
          → explicit merge candidate
```

The frontend compatibility class is `CodexCodingAgentShim`. It preserves the
existing `generate`, `generate_dir_mode`, and `generate_auto_execute` call
surface, but does not invoke the legacy Mana planner, tool worker, or queue
executor. Its checklist preview deliberately returns no Mana checklist because
the plan belongs to the Codex turn.

Every Codex turn obtains a fresh task-aware decision from the gateway routing
authority before `thread/start` or `turn/start`. The routed model replaces the
configured Codex model hint for that invocation, and the decision ID/mode are
recorded with Codex lifecycle events. A retry or resumed task must obtain a new
decision; Codex never chooses an arbitrary backup model.

There is no runtime fallback. A turn selected for Codex stays on Codex: if the
app-server is unavailable, authentication fails, the protocol fails, or the
result is invalid, the turn stops with an explicit Codex error. The internal
backend is selected only before execution begins. An underspecified edit request must be
clarified by Codex; the shim instructs it not to invent a repository change.

Writing tasks require a separate clean Git worktree. The Codex prompt prohibits
commits, pushes, publishing, credential access, and permission elevation.
Approval requests become failed task results for Mana-Agent to surface; the
backend never self-approves them.

## Installation and authentication

Install the official Codex CLI using an OpenAI-supported installation method.
Mana-Agent runs do not require `codex login`; they use the provider credential
already selected in Mana-Agent configuration:

```bash
mana-agent codex status --repo .
mana-agent codex doctor --repo .
```

For each app-server process, Mana-Agent creates an owner-only runtime directory
under `~/.mana/runtime/codex/`, writes a generated `config.toml` selecting the
custom `mana_runtime` Responses provider, and passes its API key only through
the child environment. It removes the directory after shutdown. It does not
read or modify `~/.codex/config.toml` or `~/.codex/auth.json`, and it never
falls back to a global Codex login when the Mana provider configuration is
missing, invalid, or not Responses-compatible.

## Configuration

Add these values to `~/.mana/config.toml`:

```toml
MANA_CODING_BACKEND = "codex"
MANA_CODEX_ENABLED = true
MANA_CODEX_BIN = "codex"
MANA_CODEX_MAX_WORKERS = 2
MANA_CODEX_STREAM_EVENTS = true
MANA_CODEX_WORKTREE_ISOLATION = false
MANA_CODEX_TASK_TIMEOUT_SECONDS = 1800
MANA_CODEX_ALLOW_NETWORK = false
MANA_CODEX_MODEL = ""
```

New configurations explicitly store `MANA_CODING_BACKEND` as `codex` or
`internal`. For backward compatibility, configurations without this key select
Codex when `MANA_CODEX_ENABLED = true` and the internal backend when it is
false. Explicitly selecting `codex` while disabling Codex is a configuration
error; Mana-Agent never silently rewrites the choice. Internal mode neither
starts the Codex process nor requires Codex login or credentials. Network access remains disabled by policy unless
a future validated execution decision and sandbox implementation explicitly
support it.

Codex write turns run in the selected repository root by default, so edits are
made directly in the checkout opened by Mana-Agent. Set
`MANA_CODEX_WORKTREE_ISOLATION = true` only when a workflow requires an
isolated managed worktree under Mana's state directory.

## Runtime contracts

- `mana_agent.coding` contains provider-neutral task, workspace, result, event,
  registry, and orchestrator contracts.
- `mana_agent.integrations.codex` owns the app-server process, protocol,
  prompts, event mapping, result parsing, health checks, per-run provider
  configuration/environment isolation, and backend.
- The gateway stack is the shared CLI, TUI, API, and dashboard selection surface.
- `CodexCodingAgentShim` and `InternalCodingAgentShim` preserve the same frontend
  coding-agent surface and publish the same normalized live event contract.
- `CodexWorkerPool` bounds concurrency and serializes tasks whose declared file
  scopes overlap. Empty scopes are treated conservatively as overlapping.
- Each logical coding task starts one Codex thread. Repair turns may reuse that
  thread when the caller retains its thread ID; unrelated tasks must not.

Codex owns task-specific verification and reports its commands and results in
`CodingTaskResult`. Mana-Agent preserves that evidence and leaves the completed
write worktree as a merge candidate; it does not run a second coding planner or
silently merge the branch.

## Live execution events

Both backends publish ordered, task- and turn-associated events for backend and
turn lifecycle, approved reasoning summaries, plans, tools, commands, files,
patches, tests, warnings, failures, timing, and provider-reported usage. The
Codex adapter normalizes protocol notifications immediately and does not expose
raw payloads to the TUI. Missing usage values remain absent. Event IDs are
deduplicated, output previews are bounded, and credentials are redacted before
rendering or persistence.

The Textual chat renders these events as one live execution panel per frontend
turn. Updates are scheduled through the existing thread-safe chat history,
preserve scroll position when the user is inspecting older output, and collapse
to a compact completed status without copying diagnostics into the assistant's
final answer.
