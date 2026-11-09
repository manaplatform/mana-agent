# Live text steering: integration status

Live Steer Mode is **integrated and active** across the chat TUI and the agent gateway.
Users can guide active executions and in-flight turns directly by typing into the chat box
and pressing **Enter** without requiring separate shortcut combinations or modal dialogs.

## Resolved Gaps & Architecture

1. **Direct Input Steering (No Shortcut Required)**:
   The Textual chat app no longer rejects input while `_turn_in_progress` is true.
   When a turn is running, entering text and pressing **Enter** automatically forwards
   the instruction as live steering via `gateway.submit_steering`. The former `Ctrl+S`
   binding in chat has been removed, eliminating conflicts with the configuration screen save.
   The `/steer <instruction>` command remains supported as an alias.
   When a steered message is queued, the persistent Activity Box (`ExecutionPanel`) is dynamically
   positioned directly after the queued steered message (on the last message) rather than remaining
   stuck above it, and records the `steering.queued` event directly into the live activity stream.

2. **Unified Turn and Routing Admission**:
   `submit_steering` in `AgentChatGateway` no longer fails with `"No active execution found to steer"`
   when a turn is still in its routing/planning phase or executing conversational routes.
   Guidance received before lane task creation is admitted into the session steering queue
   and drained into the supervisor's `SteeringCoordinator` upon lane task reservation.

3. **Injection into Routing and Execution**:
   - `EntryRouteContext` accepts `live_steering_guidance` which is rendered into the router prompt
     payload, ensuring `EntryRouter` accounts for user guidance during route selection.
   - `_execute_entry_route_bound` injects active guidance into execution text across routes.
   - `process_chat_turn` checks for pending steering before model calls and full-auto resume
     cycles, appends guidance to `request_for_generation`, and atomically acknowledges applied
     entries via `SteeringCoordinator.apply_steering(...)`.
   - `AutonomousExecutionController` continues to enforce boundary checks, supervisor generation
     fencing, and human approval transitions for replanning.

4. **Fencing and Safety Invariants**:
   - Only the active fenced execution attempt may apply or consume guidance.
   - Replaced workers or retired attempts are fenced out by `SteeringCoordinator`.
   - Approval policy gates remain strictly enforced: live steering guidance is context and
     intent, not an automatic grant of elevated tool permissions.

## Public delivery and acknowledgement

TUI and dashboard render a pending user steering message immediately, keyed by
`request_id` and anchored to the active turn. Admission is session-scoped; an
explicit execution target must match that session's active task or admitted turn.
The gateway does not fall back to tasks belonging to another session. Queued,
delivered, applied, rejected, and superseded events use the shared public hub.
Delivered describes acceptance at a safe boundary; applied is emitted only after
the fenced acknowledgement persists successfully. Approval gates run before
guidance consumption. A failed acknowledgement cannot report applied.

The authenticated `POST /api/v1/conversations/{conversation_id}/steering` endpoint
accepts the existing text/kind/scope/target/request fields plus a required
`execution_id`. It uses the cached owning gateway and never submits a new chat
turn. Attachments are unsupported. A separate API process without that owning
gateway returns a conflict; shared journals support cross-process observation,
not remote execution control.

## Preserved execution boundaries

The chat path invokes `process_chat_turn` directly, and internal coding retains
its model/tool loop. This activity change does not wrap whole turns in another
controller or repeat their side effects. Original attempt credentials are carried
through controller and parent-side worker callbacks; a stored token hash or a
newer attempt cannot authorize an old publisher. External Codex retains its
supported inner execution/steering boundaries; no engine switch is introduced.

Runtime verification is pending. See [activity causes, corrected invariants,
limitations, and exact user-owned checks](regressions/agent-activity-visibility.md).
