# Live text steering: integration status

Live Steer Mode is **not yet available** in the chat TUI or dashboard. The
storage primitive in `execution/steering.py` is not connected to an execution
and must not be presented as receiving or applying steering. It stores user
text; callers must use a private session directory.

## Observed gap

The Textual chat app rejects input while `_turn_in_progress` is true. Dashboard
messages go through `ConversationService.send_message`, which creates a new
execution ID. Neither is currently a safe intervention path. In addition,
`process_chat_turn` accepts an `autonomous_controller` argument, but does not
call its `execute_goal` loop; active chat and coding loops therefore cannot be
steered merely by inserting a poll into that controller. The configuration
screen already uses Ctrl+S to save. Voice Live mode is a separate feature.

The original standalone ledger provides typed, ordered and idempotent input
records but cannot fence against supervisor lifecycle changes. The separate
`SteeringCoordinator` is admission-only groundwork: inside the supervisor's
atomic task update it checks a caller-supplied authenticated session identity,
running state, current attempt and generation, lease expiry and cancellation;
it records ordered, deduplicated pending entries in the task record. It does
not authenticate the caller itself and is not wired to a frontend or agent.
Task and worker scopes and interrupts are rejected until routing and supervisor
interruption are integrated. Task-store redaction applies to persisted text.
Pending records survive task-store reload, but no checkpoint restoration or
cross-generation replay policy exists. Nothing marks entries applied,
superseded or rejected. Admission is not delivery or execution acknowledgment.

The controller now exposes a **read-only** pending snapshot for the current
worker, fenced by the task, attempt, generation, raw lease token, expiry and
cancellation. It cannot claim or apply the records. A checkpoint also contains
an audit snapshot taken under the task-store lock; the task record remains
authoritative and may contain newer entries. Recovery must never overwrite it
with an older checkpoint snapshot. Neither change enables live steering.

## Chat/controller integration hazard

The current chat path invokes `process_chat_turn` directly, and the internal
coding engine owns an additional model/tool loop inside that call. Wrapping a
whole chat turn in `execute_goal` could repeat a turn or its side effects.
Furthermore, the existing autonomous loop reads the current task's attempt and
stored lease token at checkpoint time instead of retaining a caller's original
attempt credentials. A replaced worker therefore cannot safely use that loop
as a steering consumer without first propagating and enforcing the original
lease identity throughout execution. The new read-only API requires the raw
worker token; the task's stored hash is not accepted. It is only a snapshot:
an eventual decision/acknowledgement must recheck the fence atomically.

## Remaining implementation plan

1. Connect both frontends to an authenticated gateway intervention endpoint
   bound to the actual session, supervisor task, attempt and generation. Do not
   use the ordinary dashboard message endpoint to launch another turn. Ensure
   the Ctrl+S binding is limited to chat and cannot steal approval/config input.
2. Coordinate admission consumption and acknowledgment with supervisor fencing
   and durable checkpoints; a standalone read of supervisor state followed by
   an unrelated SQLite write is not an atomic lease or cancellation check.
   Restore pending and applied state without replay after recovery.
3. Route input to safe boundaries in the actual chat, internal coding and
   delegated worker loops, and explicitly handle unsupported external backends.
   Preserve in-flight tools and scoped isolation; interrupt only through the
   supervisor's explicit cancellation/fencing semantics.
4. Use a validated model decision for semantic classification, reconciliation
   and plan changes. Reuse TaskBoard and existing Plan Mode/approval gates;
   changing a previously approved action requires fresh authorization. Compact
   active guidance without allowing superseded instructions to leak forward.
5. Add end-to-end frontend, multithreading, worker isolation, checkpoint,
   cancellation-race, plan and approval tests before enabling the feature.

**Do not reintroduce:** reporting a queued message as applied or treating a
steering message as permission to execute an action. Only the current fenced
execution may apply guidance, and approval and policy boundaries remain in force.
Do not treat a caller-supplied session ID as authentication or use the standalone
SQLite ledger as a lifecycle fence. Do not expose pending entries to models until
approval, safe-boundary delivery, recovery, and scope semantics are implemented.
Never let a retired chat worker adopt the currently stored attempt or replace
the authoritative steering log with an older checkpoint snapshot.
