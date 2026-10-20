# Batch completion and assistant-reference regression

Date: 2026-10-20. Status: locally implemented; verification not run by the agent.

## Symptoms and reproduction

- An explicitly selected Batch coding turn returned a submission acknowledgment
  with completed status and tests passed before any provider output or tool work.
- Batch recovery polled once at another task's startup and had no ongoing wake
  path to continue dependent file-editing or command steps.
- Auto mode could submit to OpenAI merely because a model supported Batch. This
  violated the required policy: official OpenAI Batch only when explicitly asked;
  otherwise native internal/parallel batching.
- A follow-up such as “yes, I want it” could lose the assistant offer that made it
  meaningful, especially across frontend/background appends or long responses.
- A continuation payload missing an action was converted to FINALIZE. Unsafe
  independent follow-up classifications could be changed to safe in code.

To reproduce the completion gap, execute an internal coding task with explicit
Batch mode while the provider reports validating. The expected state is waiting,
with no changes or test claims; the old code returned completed. To exercise the
reference gap, append an assistant offer beyond character 300, then send a terse
confirmation through another frontend or an already-cached gateway session.

## Verified source mechanisms

The internal coding Batch workflow only submitted analysis and returned text;
its caller initialized completed status and derived tests_passed from it. The
analysis retrieval helper returned ready text rather than executing tools.
Submission deduplication excluded terminal jobs and intent was persisted only
once the create response arrived. Finalization could mark empty or early results
imported. Revision checks accepted missing files or unavailable revisions.

The ordinary console CLI also bypassed the shared gateway, maintaining a separate
routing and execution path. Console turns now use the shared gateway with the
existing frontend rendering and explicit command handlers retained.
The pre-gateway single-prompt direct-edit shortcut is removed, and the shared
turn engine's direct editor is unavailable to required Batch tasks. Administrative
status replies remain immediate; an executable command route without a deferred
contract fails explicitly rather than bypassing Batch.
Mixed coding source contracts are rejected before synchronous source aggregation.
Immediate status/clarification labels cannot authorize an executable Batch route.
Explicit console planning questions are persisted with both visible roles before
plan execution enters that gateway. Missing question-model responses now stop
instead of selecting a static fallback. Native coding also rejects a missing
model instead of reporting a read-only placeholder as a completed coding task.
The native batch inference helper had a separate catch-all that manufactured a
successful assistant message after provider errors. Failed native items now carry
an explicit error; missing models or unsupported options never select a model or
fabricate output. Native batches retain regular inference prices.

Routing had a 300-character assistant prefix. Session state could remain stale
when another frontend appended to durable history. The context compactor rebuilt
EntryRouteContext without the newly added role-preserving context fields; this
boundary now preserves and budgets them. Execution context and the routing
context also had different lifetimes. The precise reported live follow-up incident
has not been independently reproduced; the source defects and new mocked
regressions are the implementation evidence, not a passing verification claim.

## Corrected invariants and components

- Batch policy records source message/decision IDs on tasks, Supervisor records,
  checkpoints, and related follow-ups. Independent requests retain native routing.
  A task-scoped execution context rejects an attempt by a model-selected tool to
  upgrade a native task into a paid OpenAI Batch submission.
- `batch/operations.py`, transport validation, and transactional result imports
  preserve submission intent, provider receipts, request correlation, CAS model
  checkpoints, and pending final delivery. Schema additions are additive.
- The Autonomous Loop and native engine share deferred model/tool turns.
  Saved task limits survive process restarts rather than being replaced by new
  process defaults.
  Supervisor Batch waits release leases and restore only the matching branch with
  fresh credentials, bounded waiting, revision checks, and action receipts.
  Context/session restoration after acquiring a fresh lease is inside the same
  failure boundary as tool execution, so a failed restore cannot strand a live
  lane with its background heartbeat.
  The shared AskAgent shell binding includes the trusted deferred call ID in
  action identity. Distinct commands in later waves cannot reuse an old command
  receipt; an approval retry explicitly reuses its approved action after checking
  task identity. Approval waiting returns to the durable inbox immediately.
  Sibling tool calls paused by approval receive explicit correlated not-executed
  results before the next model wave; they are never silently dropped or run
  through the human wait.
  An escrowed final result interrupted before verification re-enters the existing
  Supervisor verifier before delivery, without starting another model/tool wave.
  A consumed wake remains durable while queued, allowing restart between wake
  consumption and worker acquisition. Polling does not fail an active local phase
  merely because it currently has no remote request pending.
  Provider completion windows are validated against the provider completion
  receipt, not the time a restarted app happens to import that receipt. Explicit
  user and active-parent deadlines still apply at resume.
  Watcher startup and idle retirement share a workspace lifecycle lock. The
  registered worker stops its heartbeat, marks itself stopping, and releases its
  singleton only after rechecking durable demand. Concurrent submission cannot
  attach to an already retiring watcher. A 30-second process startup grace keeps
  another launcher from declaring a worker stale before PID identity is published.
- Command dispatch uses the existing transactional action gateway. Read-only
  workspaces cannot run write-capable shell tools. Interrupted actions are not
  blindly replayed. No tests are inferred from successful model generation.
  Distinct command call IDs receive distinct action keys, while approval retries
  remain bound to their original approved action. Observed verification receipts
  survive restart and are invalidated by later workspace mutations.
- Shared conversation context refreshes durable history, keeps complete paired
  turns, names observed message IDs, and marks truncation. Legacy conversation
  services receive the same bounded context when they lack a history parameter.
  Entry routing can request bounded older conversation retrieval before choosing
  a final route, including when there are no prior task candidates.
- Missing or contradictory model decisions fail validation instead of selecting
  defaults. Clarification and approval responses remain separate durable inputs.

The multi-task child scheduler has no deferred Batch child wake contract yet;
required Batch compound lifecycles therefore fail explicitly instead of starting
synchronous children. Supported coding workflows may contain multiple edits and
commands. Routes must expose actual registered tools to the deferred runner.

No existing configuration keys or Batch tool aliases are removed. Legacy auto
remote selection is intentionally changed: `mode=openai` or a validated required
Batch task policy is necessary. A registry entry declaring support is a capability,
not authorization. Explicit deadline and parent constraints remain enforced.

## Do not reintroduce

Do not equate job submission, provider completion, result import, tool execution,
model FINALIZE, and independently verified user-goal completion. Do not enable
paid OpenAI Batch from model names, request count, or a default auto-enable flag.
Do not silently switch a required Batch task to a different backend or transport.
Do not drop the assistant half of a conversation during compaction, restart,
frontend switching, or prompt construction. Never turn an unsafe or missing model
decision into permission to proceed.
Do not drop opaque Responses reasoning receipts between function-call waves or
render their internal protocol data as an assistant conversation message.
Do not replace saved task limits on restart, reuse command receipts for distinct
deferred call IDs, or send only part of a tool-call wave's required outputs.

## User-owned verification

```bash
./venv/bin/pytest tests/test_openai_batch_runtime.py tests/test_openai_batch_coding_engine.py tests/test_conversation_followup_context.py tests/gateway/test_followup_classifier.py -q
./venv/bin/pytest tests/test_openai_batch_*.py tests/gateway tests/execution tests/execution_supervisor --ignore=tests/test_openai_batch_live_integration.py -q
./venv/bin/pytest tests/test_coding_engine_modes.py tests/test_internal_coding_engine_workflow.py tests/test_chat_planning_mode.py tests/test_chat_direct_commands.py tests/test_chat_console_logging.py tests/test_ask_agent.py -q
./venv/bin/pytest tests/test_unified_sessions_commands_background.py -q
```

Coverage includes deferred waves, restarts, stale workspaces, invalid responses,
unsafe follow-up flags, exact message retrieval, caller fencing, and shared
frontend history. Live platform behavior and app UI delivery remain unverified
until the user runs the relevant tests and authorizes a live end-to-end run.
