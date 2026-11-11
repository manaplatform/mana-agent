# Hierarchical tool execution and retained activity

Status: local implementation, uncommitted; automated checks, live providers,
interactive UI, cross-platform behavior, and performance acceptance are unverified.
No tests, linters, type checks, builds, publication, or deployment were run by
the agent. The following describes source changes, not a verified release.

## Symptoms, triggers, and verified causes

Repository inspection found independent execution boundaries. The legacy queue
checked worker roles, but the internal coding dispatcher directly called local
implementations. MainAgent assembled technical searches/Git jobs under a
CodingAgent identity; a gateway shortcut edited README before backend dispatch.
The subprocess worker also ran an AskAgent model loop. These mechanisms mixed
decision ownership with execution and left the queue policy insufficient to
enforce a repository-wide hierarchy.

Tool completion callbacks omitted actual results. Commands buffered output until
exit; file results commonly contained hashes/ranges without applied diffs. Public
projection removed cwd, exit status, and diff details. Reducers replaced metadata
on updates, and the dashboard rebuilt the timeline. Those verified mechanisms
explain why adding descriptive labels alone could not restore execution evidence.
Older tests concentrated on the original transport and final events rather than
the common authority/evidence boundary.

The adapter review also demonstrated that the internal document dispatcher
imported a nonexistent `ToolManager`, passed unsupported `workspace_root`
constructor arguments, and used the wrong document-helper signature. Native
document calls now use `ToolsManager(root)._execute_document_tool(payload, root=...)`
inside the already-bound operation. A dedicated document-detection regression
covers that path. Legacy failure-memory records now identify job/tool without
persisting raw arguments or using the nonexistent `job.id` field.

## Corrected boundaries

MainAgent delegates repository work to CodingAgent; deterministic ToolRunner
executes selected operations. CodingAgent owns technical inspection, Git, and
wiring decisions. VerifierAgent keeps independent judgment and delegates checks
through CodingAgent. Non-coding specialists retain their lanes and use the same
gate. Bootstrap instructions/configuration/workspace loading remains deterministic
infrastructure. TaskBoard remains task authority, the Autonomous Loop decides
continuation, and Execution Supervisor retains leases, receipts, checkpoints,
result escrow, and completion authority. UI replay never executes work.

Versioned `ToolOperationRequest`/`ToolOperationResult` augment legacy contracts
with task/root/parent, actor/level, parent execution, supervisor attempt/generation,
operation attempt, decision/call, capability/scope, workspace, deadline,
cancellation, output, and outcome identities. An envelope is not authority.
Host bindings validate registered specialist lineage; MainAgent, prefix-only
identities, unknown effects, foreign scopes, and stale attempts cannot execute.
Custom callbacks require explicit `ToolContract` effect metadata. Gateway lane
admission binds all route scopes; native shell/repository tools and domain
server/media/remote submissions share ToolRunner without changing their specialist
lane or replacing their authoritative approval/result paths.

Trusted direct embeddings must explicitly bind authority or use
`ToolExecutionDispatcher.for_coding_task(..., task_id=...)`. An unbound constructor
remains loadable but refuses execution. Existing IDs and the `tool_worker` role
remain compatible; ToolRunner does not require a model call. Legacy subprocess
`run_tools` planning moves into the parent CodingAgent runtime. The subprocess
accepts one `execute_operation` and rejects the former model-planning endpoint.
Unknown policy names cannot widen an allowlist. A selected Redis transport failure
is explicit; the compatibility factory does not start a local replacement. Lost replies are not retried
automatically and are reported as uncertain, requiring receipt reconciliation.
Legacy Redis JSON jobs have no host-bound authority and now fail before model/tool
execution until explicitly rebound in an admitted host scope. They are never
silently moved to the local transport. This is a compatibility restriction, not
an automatic migration of persisted jobs.

## Evidence, streaming, and recovery

The worker validates arguments, scope, cancellation, lane authorization, and
supervisor ownership. Consequential actions use existing durable supervisor
receipts. Duplicate delivery within a bound runtime returns its recorded result;
matching durable actions require reconciliation before replay after interruption.

Mutation snapshots are taken inside the workspace mutation lock against the
immediate dirty baseline. Evidence distinguishes actual create/edit/delete,
exact-content rename, binary, partial, rollback, and no-op outcomes. Snapshot
coverage is bounded; missing coverage is not an applied text diff. Unknown
command write scopes serialize within their assigned workspace. Legacy registered
mutation commands now enter the same scope. Generated markdown fallback edits,
automatic deliverable relocation/deletion, and controller-side patch rebuilding
are disabled; a fresh CodingAgent decision must select any further mutation.
The supervisor store fence makes receipt preparation/deduplication atomic,
including the prepared-before-start interval. Native receipt updates also bind the
current attempt and lease; expired/replaced workers cannot publish a successful
receipt. An interrupted receipt remains for supervisor reconciliation.

The shared subprocess adapter drains stdout/stderr independently, preserves
argv separately from its redacted display string, records actual subprocesses
under distinct operation identities, and preserves partial output on timeout or
cancellation. Local process, Docker, and SSH advertise command streaming;
providers without it preserve buffered results. Backend selection never changes
because presentation capabilities differ.

External Codex keeps its inner loop. Its records say `codex_backend_owned`;
combined output stays combined and unavailable fields stay unavailable. Provider
file details have provider provenance. A separately observed workspace diff is
scoped to the entire turn, not one Codex operation. No Mana Decisions call is
inserted into Codex's inner tool loop.

## Shared trace, API, and UI

Events continue through `ExecutionEventHub` and its existing journal/cursors.
Operation keys include supervisor and operation attempts. Metadata merges across
lifecycle revisions, stream offsets preserve repeated chunks, and terminal
operations cannot be reopened by stale running updates.

Redacted fragments use the scoped `ContextArtifactStore`. Defaults are 10 MiB per
operation, 100 MiB per execution (`MANA_EXECUTION_TRACE_MAX_BYTES`), and existing
30-day artifact retention. Execution accounting is shared and restored from
journaled fragment sizes. Pipes continue to drain past capture limits; omitted
bytes and incomplete traces are explicit. Raw arguments/environment, lease
credentials, prompts, and private reasoning remain excluded. Streaming withholds
unfinished lines/tokens and suppresses multiline private-key blocks. Unknown
sensitive prose still requires careful producer contracts.

The existing conversation API adds:

```text
GET /api/v1/conversations/{conversation}/executions/{execution}/operations/{operation}
GET /api/v1/conversations/{conversation}/executions/{execution}/operations/{operation}/content
```

Both use the existing root/repository selector, `attempt_id`, and optional
`operation_attempt_id`. Content accepts `channel=stdout|stderr|combined|diff|result`,
`offset`, and `limit` (1–64,000 characters). Responses include next offset,
available length, `has_more`, and trace completeness. References come from owned
operation events, never caller-supplied artifact paths. The existing bearer
guard and root/conversation scope apply; expired output is unavailable.

Main Activity and Coding Activity remain the only activity containers. Coding
rows expand to exact commands/cwd, streams, outcomes, applied diffs, and paged
retained output. Clipboard actions copy redacted visible text. TUI rows and
dashboard keyed nodes preserve expansion and selection; active operations remain
beside bounded recent history. The 250 ms presentation target and full live
steering/scroll/reconnection behavior require interactive acceptance.

## Do not reintroduce

- Never treat ID prefixes, `approved=True`, or old persisted records as authority.
- Never put model planning, semantic retries, policy widening, or backend fallback
  inside ToolRunner or its subprocess.
- Never let MainAgent impersonate CodingAgent to select repository operations.
- Never bypass the gate for cached results or direct edits.
- Never restore the obsolete document adapter name/signature or log raw queue arguments.
- Never display intended patches as applied diffs, infer success from invocation,
  or attribute a turn diff to an individual provider operation.
- Never expose raw arguments/environment, split credentials before redaction,
  mix parallel streams, drop repeated chunks, or restore separate archive UI.
- Never execute from replay or refresh stale worker authorization from a newer lease.

## User-owned verification

These commands were **not run by the agent**. Run from the repository root:

```bash
venv/bin/python -m pytest tests/coding/test_tool_runner_execution.py tests/gateway/test_execution_hierarchy.py tests/dashboard/test_operation_trace_api.py tests/test_operation_rows.py -q
venv/bin/python -m pytest tests/test_programmatic_tool_calling.py tests/test_parallel_tool_scheduler.py tests/test_coding_engine_modes.py tests/test_tool_worker_process.py tests/test_small_direct_edit.py tests/test_agent_work_queue.py tests/test_tools_manager.py tests/test_agent_orchestrator.py tests/test_multi_agent_core.py tests/test_shell_executor.py tests/test_activity_stream.py tests/test_execution_trace.py tests/test_codex_coding_visibility.py -q
venv/bin/python -m pytest tests/gateway tests/execution tests/execution_supervisor tests/coding tests/test_full_access_mode.py tests/test_execution_modes.py tests/test_tui_live_tools_scroll.py tests/test_tui_approval_realtime.py tests/test_dashboard_live_chat.py tests/test_api_conversations.py -q
node --test tests/dashboard/*.test.mjs
venv/bin/python -m pytest -q
```

Manually exercise a long command, concurrent streams, actual and partial edits,
approval wait, live steering, cancellation, and reconnect spanning multiple replay
pages in terminal and dashboard. Confirm stable expansion/selection, redacted
copying, retained details after refresh, both coding backends, and Plan Mode.
Acceptance remains pending until supplied evidence demonstrates these invariants.
