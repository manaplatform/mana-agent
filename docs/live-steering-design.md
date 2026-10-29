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

The new ledger provides typed, execution-scoped, ordered and idempotent input
records across concurrent writers. It does not validate sender ownership,
supervisor state, lease, generation, expiry, plan approval, or worker scope;
nor does it apply, supersede, reject, checkpoint, or deliver an instruction.
An append is not an acceptance or an acknowledgment by an execution.

## Remaining implementation plan

1. Connect both frontends to an authenticated gateway intervention endpoint
   bound to the actual session, supervisor task, attempt and generation. Do not
   use the ordinary dashboard message endpoint to launch another turn. Ensure
   the Ctrl+S binding is limited to chat and cannot steal approval/config input.
2. Coordinate ledger consumption and acknowledgment with supervisor fencing
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
