# Resilient Execution Supervisor

The execution supervisor is the durable authority for long-running root tasks,
delegated children, attempts, recovery, and successful completion. It extends
the existing taskboard and gateway lanes: taskboard IDs remain the public task
identity, while each retry or reassignment receives a new attempt ID.

## Lifecycle

Non-terminal states are `created`, `queued`, `leased`, `running`,
`checkpointing`, `waiting`, `retry_scheduled`, `replanning`, `cancelling`, and
`completed_pending_verification`. Terminal states are `cancelled`, `failed`,
`budget_exhausted`, and `completed`. Every transition is validated and compare-and-set persisted.
An invalid transition emits an immutable `invalid_transition` event and raises
a typed error without executing another action.

```text
created → queued → leased → running → completed_pending_verification
                         ↘ checkpointing / waiting ↗          ↓
                         ↘ retry_scheduled → queued         completed
                         ↘ replanning → queued
                         ↘ cancelling → cancelled
                         ↘ failed
```

User-visible success comes only from `completed`; child text and lane success
claims cannot bypass `completed_pending_verification`.

## Storage, leases, and attempts

Local mode requires no external service. Atomic JSON records and immutable
JSONL events are stored under:

```text
~/.mana/execution/
  tasks/ attempts/ actions/ checkpoints/ results/ artefacts/ events/ logs/
```

Temporary files are flushed and atomically replaced. A cross-process lock
guards compare-and-set updates and lease acquisition on Windows, macOS, and
Linux. Execution logs rotate to one bounded previous segment. Secret-shaped
fields and payload values are redacted before persistence.

A worker must present the current task ID, attempt ID, and unguessable lease
token to start, heartbeat, checkpoint, or publish a result. Heartbeats extend a
bounded lease. Once the lease expires or is replaced, constant-time token and
attempt checks reject the stale worker's result. Only one queued task claimant
can create the active attempt. Raw lease credentials remain memory-only; task,
attempt, and escrow records persist only SHA-256 token hashes.

## Checkpoints and result escrow

Checkpoints are schema-versioned and record resumable payload, completed and
pending steps, tool results, workspace/Git references, generated files, and
verification state. Recovery uses a checkpoint only when a typed
`RecoveryDecision` explicitly selects `resume_checkpoint` and the checkpoint
still validates.

When a new gateway request has stopped task candidates, a separate strict model
decision compares the complete new request with candidate intent and progress.
It selects `resume_checkpoint`, `retry_task`, `return_verified`, `reverify`,
`start_fresh`, or `stop` and
records same-work, freshness, checkpoint-validity, repeat-safety, and
continuation-safety judgments. `retry_task` reuses the exact durable task ID
when the work is stable and equivalent but no valid checkpoint exists; the new
attempt restarts the unfinished request under that existing identity.
Current or account-backed information such as prices, mailbox state, calendar
state, news, weather, availability, and remote state must be fetched through a
fresh execution rather than restored from stale checkpoint evidence. The
runtime validates the exact selected task or task/checkpoint pair and rejects
missing, invalid, inconsistent, or unsafe decisions without choosing a fallback
action.
An approved resume supplies the executor with redacted completed steps, pending
steps, resume payload, workspace/Git references, and generated-file state; it
does not expose stored tool results or treat checkpoint data as instructions.

Child results are atomically written to `results/` before the task advertises
the result ID. The parent reads unacknowledged escrow entries and durably
acknowledges consumption. A restart after child completion but before parent
notification therefore re-delivers the same result without re-executing the
child.

Escrow records persist the `produced → stored → available → delivery_pending →
delivered → acknowledged/rejected` lifecycle. The attempt generation is stored
with every result and action. A result or action update from an older generation
is rejected even if a delayed worker still has its old process-local state.

## Retry and recovery safety

Every task and consequential action records one side-effect classification:
`read_only`, `idempotent`, `conditionally_idempotent`, `deduplicated`,
`compensatable`, `non_idempotent`, or `unknown`. Read-only work
is safely recoverable. Idempotent/deduplicated retries require a stable
idempotency key. Compensation must have been selected explicitly. Unknown work
may resume from the exact model-selected checkpoint, or restart under the same
task ID only when the strict recovery decision explicitly confirms equivalent,
non-live work whose actions are safe to repeat and no irreversible side effect
was recorded. Unapproved unknown retries and non-idempotent ambiguous work fail
for intervention. The error warns when the external action may already have
occurred.

Infrastructure, model, tool, verification, lease-loss, and replan budgets are
tracked separately. Backoff is exponential with deterministic jitter and a
configured cap. Task identity remains stable while attempts change. A typed
`RecoveryDecision` carries the source decision ID, action, category, reason,
selected agent/worker/model, checkpoint, and `safe_to_continue`; invalid or
missing decisions stop without fallback routing.

Startup recovery is idempotent. It reconnects parent-child links, re-verifies
escrowed results, marks expired attempts lost, schedules only policy-safe work,
preserves ambiguous side effects for review, and repairs the LaneCoordinator and
TaskBoard projections from supervisor state. Queue and retry state remain visible
throughout recovery.

## Authoritative ownership and projections

- `ExecutionSupervisor` owns logical task/attempt state, retry and resume
  eligibility, checkpoint validity, escrow acknowledgement, verification, and
  final completion.
- `LaneCoordinator` owns lane admission, capacity, queueing, workers, handoffs,
  file/repository locks, and resource reservations. `finish(COMPLETED)` submits
  evidence; it does not declare logical success.
- `TaskBoard` is a compatibility/UI projection. `DONE` is accepted only through
  a supervisor completion projection containing the durable result ID and passed
  verification manifest. Duplicate memory matches remain advisory and are never
  created as `SKIPPED` work.
- `ExecutionManager` owns provider-neutral sandbox/process lifecycle only. Its
  spec, handle, request, result, and events carry execution, root task, attempt,
  checkpoint, session, workspace, and repository identity; sandbox readiness or
  exit code does not establish task completion.
- `ContextCostGovernor` is the model/tool-call admission and accounting
  authority. It atomically reserves estimated concurrent spend, preserves a
  verification reserve, releases unused reservations, records actual usage once,
  and persists a content-addressed context manifest for every admitted or blocked
  model call.

TaskBoard state writes now use file flush, atomic replacement, and directory
flush. Corrupt state fails closed instead of silently becoming an empty board.
Schema-version 2 supervisor, checkpoint, and TaskBoard records load older fields
through defaulted typed models; new records add attempt generations, execution
scope/linkage, expanded checkpoint references, escrow lifecycle, and action
receipts. TaskBoard context compaction stores the complete redacted source in the
existing context artifact store and returns a hash-checked retrieval envelope.

## Completion contracts

Supported contracts are `file_exists`, `directory_exists`, `git_diff_present`,
`git_commit_exists`, `command_succeeded`, `structured_result_valid`,
`remote_resource_confirmed`, and registered `custom_verifier`. File checks
enforce the allowed workspace, type, minimum size, optional checksum, and
optional modification by the active attempt. The resulting manifest records
existence, size, SHA-256, attempt provenance, and verification time.

If any check fails, the claimed result remains in escrow and the task stays
`completed_pending_verification`; it never emits final success.

## Execution-path integration

Gateway lane reservations create the supervisor record with the same taskboard
ID before Codex, an internal/model-routed agent, a local tool, Docker, SSH, or a
remote runtime starts. Lane handoffs reuse a live attempt rather than claiming a
second lease, while process restarts deliberately discard the raw in-memory
lease token and wait for safe recovery. Fleet runs are durable parents whose
platform jobs are supervised children. A2A delegation and each claimed cron or
automation run hold their own leases; Teach-generated automation jobs use that
same path. Automation retry definitions must declare side-effect safety before
more than one attempt is permitted.

The supervisor executes the agent/model/worker selected by the existing typed
routing or recovery decision. It validates those identifiers and never selects
a substitute route, default tool, fallback agent, or recovery workflow.

## Parent waits and cancellation

Parent progress supports `fail_fast`, `wait_all`, `best_effort`,
`minimum_success_count`, and `dependency_graph` policies plus an absolute
deadline (defaulting to the existing routing task timeout). Status always reports active blockers and whether the deadline has
elapsed, preventing an invisible indefinite wait.

Cancellation propagates child-first and uses a cooperative cancellation state.
Logs, checkpoints, partial artifacts, and escrow are retained. A task that has
entered an explicitly marked irreversible side-effect phase remains active with
`blocked_by_side_effect`; the supervisor does not pretend that cancellation
reversed the external action.

## Operator workflows

Successful verified file generation:

```bash
mana-agent tasks status task_123
mana-agent tasks artefacts task_123
# state: completed; verification.status: passed
```

Worker loss and safe read-only recovery:

```bash
mana-agent tasks recover
mana-agent tasks status task_123
# old attempt: lost; state: retry_scheduled; checkpoint retained
```

Blocked non-idempotent retry:

```bash
mana-agent tasks retry task_123
# refuses: action may already have occurred; no retry scheduled
```

The retry command automatically creates a typed operator decision bound to the
task ID and uses the `model` retry budget by default. Use `--category` to select
a different budget, or `--decision-json` for a standalone advanced recovery
decision. Do not pass the taskboard's `decisions.json` registry: those entries
describe routing and do not authorize recovery. Automatic attachment does not
bypass side-effect, idempotency, checkpoint, or retry-budget validation.

Cancel a parent with children:

```bash
mana-agent tasks cancel task_parent --reason "request withdrawn"
mana-agent tasks tree task_parent
# cancellable descendants are cancelled child-first; irreversible work is flagged
```

Completed child waiting in escrow after restart:

```bash
mana-agent tasks status task_parent
# parent progress and unconsumed child result remain durable
mana-agent tasks recover
```

The authenticated API mirrors these operations under `/api/v1/tasks` and
publishes live events at `/api/v1/tasks/ws/events`. The dashboard Taskboard view
shows the same task/attempt IDs, leases, heartbeats, budgets, checkpoints,
children, artifacts, failures, and recovery reasons. Frequent heartbeat events
remain durable in execution logs and are aggregated to at most one live UI
update per task per minute.

## Troubleshooting

Use `tasks logs` for the immutable transition history, `tasks tree` for blocked
descendants, and `tasks artefacts` for verification evidence. Run `tasks
recover` after an operator-confirmed service interruption. Do not manually edit
task JSON. For a corrupted checkpoint, select another valid checkpoint through
the model decision layer or restart only if the side-effect policy permits it.
