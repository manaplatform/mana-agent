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
