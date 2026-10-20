# OpenAI Batch and native parallel execution

## Selection and scope

The official OpenAI Batch API is used **only after an explicit user request for
OpenAI Batch**. The entry model records a typed task execution policy with the
source message and decision IDs. Related follow-ups inherit that task policy;
independent requests do not. Coding uses the internal engine for that task without
changing the configured global coding backend.

Ordinary batching uses Mana's internal parallel execution. Model support, workload
size, `mode="auto"`, or the legacy `always_enable_if_supported` configuration field
do not authorize an OpenAI Batch submission. Those configuration keys remain
readable for compatibility. Explicit API/tool callers select `mode="openai"`;
router adapters select `mode_override="batch"`. A selected but unavailable Batch
transport fails explicitly rather than switching to ordinary inference or another
provider. Status, routing, and clarification replies remain immediate.
Tools invoked inside a native task cannot opt that task into paid Batch by
changing a tool argument; the host-bound task policy must require it.

The Batch API executes model requests; file edits and shell commands execute in
Mana through registered local tools after their decisions are returned. Tool
permissions, transactional approvals, workspace confinement, revision checks,
resource locks, and execution budgets still apply. Dependent tool interactions are
successive Batch jobs, not dependent requests in one JSONL file.

## Durable lifecycle

1. Persist an operation ID and payload fingerprint before upload or create.
2. Upload with `purpose=batch`, then submit the selected endpoint and model.
3. Save the provider receipt and a resumable model/tool-loop checkpoint.
4. Return `waiting`, project TaskBoard as waiting for tools, and suspend the
   Supervisor with `waiting_for_batch`. Release the old execution lease and locks.
5. A registered `batch.watch` background process polls every 30 seconds, with
   transient-error backoff capped at five minutes. One process serves a workspace.
6. Import results transactionally, correlating by `custom_id`. Resume the saved
   branch under a fresh lease. Completed tool receipts prevent blind replay.
7. Continue model/tool waves, including structured continuation evaluation, until
   the model returns a valid terminal decision. Supervisor verification remains
   required before success is delivered to the original conversation.

An uncertain create outcome is reconciled using the existing operation metadata;
it is not permission to create another paid job. Missing, duplicate, unknown,
malformed, incomplete, refused, or failed model results cannot become success.
Provider failures, cancellation, and expiry retain their terminal status even if
partial results can be imported. An interrupted tool with an uncertain outcome
requires reconciliation before it can run again.

For tasks with a default active-execution deadline, bounded provider waiting does
not spend the active deadline. Explicit deadlines and active parent deadlines
remain enforced. Idle polling consumes no model turns or no-progress cycles.
An app restarted after a wave's window reconciles its provider completion receipt;
an output completed within that window can still be imported and resumed later.
Batch completion windows are currently 24 hours per wave; dependent workflows can
therefore take substantially longer than synchronous execution. See the
[official Batch guide](https://developers.openai.com/api/docs/guides/batch).

Required Batch tasks using the separate `multi_task` child scheduler currently
return `batch_route_unsupported` before starting children. Repository workflows
with multiple edits or commands use the supported coding route. Routes without
registered, permission-aware tool bindings also fail explicitly. A frontend exit
detaches a pending Batch task; an explicit cancellation, session deletion, or
session replacement still cancels it.
The direct file-edit shortcut cannot execute required Batch work. Batch coding
currently binds repository sources; additional source contracts are rejected
before execution. Immediate status/clarification replies use the conversation
route, so a response label cannot bypass a deferred executable route.

## Configuration and tools

Existing `OPENAI_BATCH_*` settings and `[openai.batch]` configuration keys remain
supported. `enabled=false` rejects an explicit remote request. Platform credentials
must resolve to an actual OpenAI API key; missing keys no longer use a dummy key.
Only registered, explicitly supported model/endpoint capabilities may submit.

```python
# Caller has explicitly selected the official OpenAI Batch API.
job = manager.submit_batch(
    requests,
    model=selected_model,
    mode="openai",
    task_id=task_id,
    idempotency_key=operation_id,
)
```

`batch_create_job` / `batch_submit_requests` default to native internal execution;
pass `mode="openai"` for an explicitly requested official job. `batch_list_jobs`
reads local state unless `sync_platform=true` is explicitly requested. Existing
status, cancellation, download tools and aliases remain available. Downloading
results does not execute returned commands or apply analysis text as a patch.
`retrieve_and_apply_batch_results` remains a compatibility analysis helper; its
`verified_and_ready` status describes revision-checked output, not an applied edit.
Actual edits use the durable native tool loop.
The standalone native batch helper supports text and conversation inputs. It
rejects unsupported generation options and tool-bearing raw requests explicitly;
registered file/command interactions use the native coding tool loop. Failed
native inference returns failed item receipts, never placeholder success text.

Results use native Responses output items or Chat Completions messages. Tool call
IDs and outputs remain correlated across waves. Cost estimates aggregate item
usage and use the selected model's prices with the Batch discount. Unknown prices
remain explicitly unknown and cannot satisfy a spending-limit check.
Responses waves replay opaque encrypted reasoning receipts with function calls
and observed tool outputs. Raw reasoning text is excluded from visible history.
This follows OpenAI's [reasoning continuation contract](https://developers.openai.com/api/docs/guides/reasoning).

## Conversation continuity and recovery

CLI, TUI, dashboard, API, ACP and A2A share durable gateway history. Each incoming
turn refreshes that history, preserving both user and visible assistant messages.
Bounded recent exchanges, source IDs and explicit truncation indicators are shared
with routing and execution. Exact message-ID retrieval reads complete turns in
the current session. Older context remains available through bounded retrieval;
private model reasoning is not conversation history.
When recent history is insufficient, the entry model can request up to two
bounded history reads before choosing an execution route. Missing or ambiguous
references require clarification. Explicit console planning questions and answers
are also recorded in the shared history; unavailable question models return an
error without selecting substitute questions.

Approvals and clarifications use the durable Human Inbox. A provider wake cannot
resume a human-waiting branch or grant permission. Resumed final output is delivered
with a stable message ID after Supervisor verification. A persisted delivery marker
supports retrying a lost final notification without repeating tools.

## Regression and verification

See [the permanent regression note](regressions/batch-and-followup-context.md).
These changes are locally authored and unverified until user-owned checks run:

```bash
./venv/bin/pytest tests/test_openai_batch_runtime.py tests/test_openai_batch_coding_engine.py tests/test_conversation_followup_context.py tests/gateway/test_followup_classifier.py -q
./venv/bin/pytest tests/test_openai_batch_*.py tests/gateway tests/execution tests/execution_supervisor --ignore=tests/test_openai_batch_live_integration.py -q
```

Live tests require separate authorization because they create paid platform jobs.
