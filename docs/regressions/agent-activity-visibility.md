# Agent activity visibility and recovery

The [hierarchical execution extension](hierarchical-tool-execution.md) adds
deterministic authority, retained output, applied mutation evidence, and stable
operation rows. It preserves these consolidated containers and privacy/replay
invariants; runtime acceptance of the extension remains pending.

Status: implemented locally, uncommitted, runtime verification pending. Tests and
performance measurements have not been run by the agent. This note describes
source changes, not a production deployment or a verified acceptance result.

## Symptoms and causes

The affected surfaces are the Textual chat, Rich terminal chat, and dashboard.
The following mechanisms were found by source inspection:

| Trigger | Previous behavior and failed invariant | Corrected boundary |
| --- | --- | --- |
| General routing, planning, or loop progress | A coding-only filter discarded useful general activity; projection lost ownership and the `type` alias. | `chat/activity.py` normalizes aliases and projects typed identities and allowed public fields for every activity domain. |
| A legacy producer reuses an event ID | Frontend ID deduplication discarded a later outcome. | Exact delivery fingerprints are separate from operation keys; lifecycle revisions update their operation. Provider sequence is separate from conversation sequence. |
| Parallel agents start different phases | One active-step pointer could finish a sibling without an observed outcome. | Reducers maintain independent operations by task, attempt, agent, and operation. Root completion leaves missing outcomes unverified. |
| Internal coding executes several steps | `stream()` collected the entire execution before yielding. | A bounded asynchronous producer/consumer path yields while execution runs. |
| A compatibility provider sends `reasoning_content` | Responses-shaped summary fields could be confused with an authorized public summary. | Bridged, unknown, encrypted, and raw reasoning remain private. Only explicit agent explanations or completed, supported native summaries pass projection. |
| Steering admission or acknowledgement is slow/fails | The user saw no message until admission returned; applied could appear before the fenced update. Session fallback could choose another session. | Clients show a pending user row immediately. Session/execution selection is explicit. Applied status follows the committed acknowledgement. |
| High-volume tool activity or a user reading older text | Collections grew; forced scrolling and competing compatibility-log writes disrupted reading. Read paths could be listed as modifications. | Bounded operation projections, 10 Hz repaint coalescing, selection/scroll preservation, exclusive Main/Coding box rendering, transport-level history pagination, and successful mutation evidence. |
| Concurrent publication, disconnect, or queue overflow | Sequence allocation, append, and callbacks could disagree; limited-tail replay and a maximum-sequence cursor skipped gaps. | Portable journal locks serialize append/sequence allocation. Ordered callbacks drain outside locks. Forward pages recover the complete gap; clients keep a contiguous cursor. |

Existing tests primarily exercised sequential, coding-specific delivery and final
rendering. They did not establish concurrent ownership, privacy provenance,
publication/acknowledgement ordering, or complete multipage replay. New regression
fixtures cover these boundaries; their execution remains user-owned.

## Implemented contracts

`cli.events.ChatEvent` and `coding.models.AgentEvent` retain existing names and
aliases and add task/root/parent, actor, attempt/generation, operation,
producer-sequence, and execution-mode metadata. `ActivityEvent` is the general
chat-history event; `CodingActivityEvent` remains a compatible alias.

`services/activity_context.py` carries immutable turn ownership through gateway,
controller, specialist, coding, and parent-side worker IPC callbacks. Publications
from supervised workers check the original attempt, generation, lease, running
state, and cancellation under the existing store fence. Raw lease credentials
never enter public events or IPC identity. Recorded supervisor transitions and
persisted TaskBoard progress publish observations; neither the hub nor the UI
changes execution state or selects an action.

The hub is the public publication path. Projection precedes append/delivery and
also applies to direct chat history and historic replay. Allowed public fields
are redacted, protected credentials are removed where scoped context is
available, control/terminal escapes are removed, and UI text is rendered
literally. Raw arguments, prompts, continuation state, raw provider payloads, and
private reasoning are excluded. Visibility presets never relax these controls.

Native OpenAI Responses summaries are requested with `reasoning.summary=auto`
only for an explicitly supported provider/model/transport and native API
endpoint during an admitted activity turn. Codex summary sections are assembled
privately until complete, including fragmented credentials. Raw reasoning deltas
remain private. Unsupported sources simply provide observable operations and
elapsed time. No additional summarizer calls or backend changes are introduced.
See [Responses reasoning summaries](https://developers.openai.com/api/docs/guides/reasoning#reasoning-summaries)
and [Codex item deltas](https://learn.chatgpt.com/docs/app-server#item-deltas).

Presentation retains active operations plus 200 recent operations per turn and
initially renders up to 40. The hub cache remains 4,000 events. Frontend delivery
queues, ordered hub dispatch backlog, and the internal stream queue are bounded at
1,024. Hub dispatch overflow holds back later deliveries in that conversation
until a replay control record follows the delivered prefix; internal presentation
overflow coalesces by operation, with an explicit degradation event if distinct
updates exceed its additional bounded backlog. Public lifecycle events are
journaled independently of that presentation backlog. Legacy synchronous
subscribers remain compatible; frontends use buffered callbacks or a bounded
WebSocket queue. Slow consumers recover through the durable journal.

`GET /api/v1/conversations/{id}/events` adds `after_sequence` (forward pages) and
`before_sequence` (older pages). They are mutually exclusive. WebSocket recovery
subscribes before replaying all pages through a fixed high-water mark.
Nonpublic historic rows become content-free cursor records. Persistence failure
produces an explicit warning and unsequenced live observations rather than
reserving unrecoverable public sequence numbers. Subscribed processes poll
appended journal records. A complete record missing only its final newline is
recovered before allocating the next sequence.

`MANA_ACTIVITY_VISIBILITY=minimal|normal|verbose` is persisted through the existing
user configuration, with `normal` as the default. `/activity` shows recent
activity in terminal chat; `/activity minimal`, `/activity normal`, and
`/activity verbose` change presentation. TUI **Ctrl+End** returns to latest
activity. Textual chat shows general and coding operations in one inline stream
per turn with earlier rows paged inside that stream. Dashboard activity uses its
existing views. Durable cursor pagination remains available to recovery and API
clients.

Planning events carry `execution_mode=plan`. Saving a plan artifact is separate
from implementation, and inspection does not enter the changed-file list. A
completed backend call or model explanation does not prove a verified user goal.
Tests are displayed only when their actual authorized execution reports an event.

## Steering and approvals

The text-only `POST /api/v1/conversations/{id}/steering` endpoint accepts
`execution_id`, `request_id`, `text`, `kind`, `scope`, and optional `target_id`, plus
the existing root/repository selector. It uses the existing bearer-token guard
and the gateway already owning the session. It never starts another turn.
Missing ownership or an inactive/ambiguous/wrong-session target returns a
conflict. Attachments are unsupported.

The user row is keyed by request ID and anchored to its original turn. Queued,
delivered, applied, rejected, and superseded states come from recorded admission
and acknowledgement. Delivered means guidance reached a safe execution boundary;
applied follows successful fenced acknowledgement. A failed acknowledgement
cannot publish applied or consume guidance. Textual execution approvals use an inline prompt with the existing expiry,
matching-request validation, and guarded approval endpoints.

## Limits and compatibility

No schema migration, dependency addition, release version change, commit, or
deployment is part of this change. Existing JSONL records and aliases remain
readable; new fields are additive. Historical private data already written to an
old journal is suppressed on replay, not rewritten on disk. Previously missing
records cannot be reconstructed from a corrupted/unavailable journal; degraded
stream warnings make that limitation explicit.

Cross-process observation works through shared journals. Dashboard steering
requires the API/dashboard runtime to hold the cached gateway that owns the live
session; it does not create a remote execution-control transport. External Codex
retains its own inner coding loop and supported steering boundaries. Public
summaries may be absent. Arbitrary unknown sensitive prose cannot be reliably
identified by a regex; producers must honor explicit public-summary contracts,
allowlisted fields, protected-context exclusion, and trusted provenance.

The 250 ms local visibility target, queue stress behavior, Windows file locking,
provider interoperability, and full UI acceptance remain unmeasured until the
following checks and manual scenarios are performed.

## 2026-11-09: consolidate activity presentation

The first activity implementation added an archive button and expander beside
TUI activity boxes and a separate archive section inside each dashboard box.
Those extra surfaces duplicated activity presentation. At the user's request,
the TUI archive widgets/handler, dashboard archive renderer/cache/fetch path,
and CLI backward-page shortcut are removed. Main Activity and Coding Activity
continue to render their existing live projections and retain their bounded
state, including internal coding events. The dashboard's Show all control
expands retained operations within those same boxes. No stored activity is
deleted and the hub's durable replay and REST cursor contracts remain unchanged.

Regression coverage checks both containers still receive the correct events,
that no archive UI is mounted, and that the CLI no longer offers backward-page
navigation. The dashboard DOM fixture supplies the selection-listener APIs used
by the current renderer. Runtime verification is pending; not run by the agent.
User-owned checks for this change:

```bash
venv/bin/python -m pytest tests/test_execution_trace.py tests/test_chat_command_dispatcher.py -q
node --test tests/dashboard/live_chat_approval_ui.test.mjs
```

## Do not reintroduce

- Keep activity in its turn-scoped stream. Keep the Textual earlier-operation
  control inside that stream and preserve the dashboard’s existing containers.
- Do not expose bridge `reasoning_content`, raw reasoning deltas, encrypted state,
  prompts, or arbitrary invocation arguments under any summary or verbosity label.
- Do not drop ownership when normalizing `type`/`event_type`, deduplicate lifecycle
  revisions by occurrence ID alone, or use provider sequence as a public cursor.
- Do not complete sibling operations when a phase starts, mark reads as changed
  files, invent preparation successes, or equate backend completion with verification.
- Do not refresh a worker's credentials from current state, accept stale attempt
  publications, or fall back to another session for steering.
- Do not publish applied before acknowledgement commits, or let activity failures
  authorize, retry, interrupt, cancel, or finalize execution.
- Do not run subscribers under publication/storage locks, recover only a log tail,
  advance recovery past a gap, force reader scrolling, or let two renderers own a box.

## User-owned verification

These commands were **not run by the agent**. Run from the repository root.
The first pair covers projection, privacy, correlation, replay, and reducer parity:

```bash
venv/bin/python -m pytest tests/test_activity_stream.py tests/test_execution_trace.py tests/test_codex_coding_visibility.py -q
node --test tests/dashboard/live_chat_reducer.test.mjs
```

Runtime boundaries, UI lifecycle, endpoint authentication, and terminal behavior:

```bash
venv/bin/python -m pytest tests/test_activity_stream_integration.py tests/test_codex_tui_lifecycle.py tests/test_tui_live_tools_scroll.py tests/test_cli_ux_helpers.py tests/test_chat_ui_events_tokens.py tests/test_dashboard_live_chat.py tests/test_api_conversations.py -q
```

Preserved execution, cancellation, approvals, steering, and supervision gates:

```bash
venv/bin/python -m pytest tests/execution/test_live_steering_integration.py tests/execution/test_unified_autonomous_loop.py tests/execution/test_autonomous_loop_taskboard.py tests/execution_supervisor/test_supervisor_core.py tests/execution_supervisor/test_checkpoint_lifecycle_races.py tests/test_execution_modes.py tests/test_tui_approval_realtime.py -q
node --test tests/dashboard/live_chat_reducer.test.mjs tests/dashboard/live_chat_approval_ui.test.mjs
```

Manually run an authorized long read-only task in each interface: confirm progress
appears before the answer, concurrent actors remain separate, reading/selecting
older content is stable, the latest action works, and no archive section or
load-earlier control appears. Activity must stay in Main Activity/Coding
Activity. Steer the active turn
while routing and while executing; verify the same panel follows its user row and
that another session remains unchanged. Disconnect/reconnect after more than one
replay page. Exercise cancellation/recovery, a failed steering acknowledgement,
and unavailable journal storage. Inspect native vs bridged provider fixtures for
private content before accepting the feature. Plan Mode must show planning and
artifact saving without implementation or fabricated test outcomes.

## 2026-11-21 Textual activity and inline approvals

The Textual renderer previously dispatched tool events to separate activity boxes
and could also mount a tool card for the same call. During a fast stream the UI
read only the latest history revision, which discarded earlier output chunks.
Session switching could also clear activity replayed before transcript replacement.
The approval callback removed every request in the queued matching snapshot after
a successful decision, even though the authoritative resolver grants only matching
requests.

The transcript now consumes a single redacted operation projection keyed by
task, attempt, actor, operation, and explicit legacy call correlation. It merges
channel offsets before coalescing display updates. Recent operation rows are
bounded and older rows use the durable event cursor. Expanded rows fetch retained
stdout, stderr, combined output, diffs, and results through the existing scoped
content reader. Preview length and completeness depend on the producing runtime
and its retention policy. Live events are held during session replay so older
durable revisions can supply command metadata and initial output before new
revisions are applied. Inline approval controls advance only after the
correlated decision resolves. Matching approval reuse remains subject to the
authority’s equivalence check.

**Do not reintroduce.** Keep one rendering path per operation. Preserve output by
channel and offset before collapsing updates. Restore messages before attaching
activity. Never dismiss a queued approval solely because it was in the snapshot
of candidates for matching approval.

Run these from the checkout root:

```sh
venv/bin/python -m pytest tests/test_tui_*.py tests/test_codex_tui_lifecycle.py \
  tests/chat/test_tui_attachments.py tests/test_operation_rows.py tests/test_activity_stream.py \
  tests/test_activity_stream_integration.py tests/test_execution_trace.py tests/test_chat_command_dispatcher.py \
  tests/execution/test_live_steering_integration.py tests/execution/test_unified_autonomous_loop.py \
  tests/test_computer_control.py tests/human_inbox/test_approval_equivalence.py tests/test_api_conversations.py -q
node --test tests/dashboard/live_chat_reducer.test.mjs tests/dashboard/live_chat_approval_ui.test.mjs
venv/bin/mana-agent chat --tui
```

### Windows CI transcript replay timing

Six Windows CI tests failed while looking up preloaded user messages. The
layout tests hit an empty query before measuring text, and the `/new` test
found zero widgets before submitting the command.

`ChatLog` defers the initial history replay with `call_after_refresh`. The
replay mounts a `UserMessage` row, which then composes its nested
`SelectableText`. Textual's `Pilot.pause()` drains messages for the widgets it
can see when the pause starts. The replay callback and new descendants can
finish in later cycles, so a single pause can return with an empty query on
Windows. A fixed number of pauses makes the same assumption about scheduling.

The shared `wait_for_tui_widgets` fixture polls for the expected widget count,
completed mounting, and a nonempty layout region. It also waits for the DOM to
empty after `/new`. The tests still assert the exact wrap dimensions and check
the old session deletion, new session activation, and history clearing.

**Do not reintroduce.** Wait for the expected transcript widgets and their
layout before inspecting dynamically replayed or appended messages. Keep wrap
expectations out of the readiness condition so incorrect layout still fails
the regression assertion.

Run from the checkout root on Windows with Python 3.12. The agent did not run
these tests or rerun CI.

```powershell
python -m pytest -q tests/test_tui_message_layout.py tests/test_tui_tool_card_layout.py tests/test_tui_auto_chat_tool_events.py
```
