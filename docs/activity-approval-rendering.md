# Approval request activity rendering

Approval-required lifecycle events are rendered in the existing consolidated
execution panel: general approvals belong in **Activity**, while approvals
associated with a coding backend belong in **Coding Activity**.

When an approval is required for transactional actions, computer control, server
actions, or API requests, the approval request is presented directly as an
interactive **modal popup dialog** (`ComputerPermissionScreen` in the TUI, and
`.approval-modal` with `.approval-modal-backdrop` overlay in the Dashboard)
over the entire chat UI.

In the TUI, the modal screen is pushed directly via `push_screen` over the entire
chat interface with a dimmed backdrop and centered dialog containing the action
title, scrollable command preview, scope, and action buttons (`Deny`, `Allow once`,
`Allow all matching queued` for transactional requests, `This session`, `Always` for
computer requests). If multiple approval requests arrive while a modal is active,
subsequent requests are queued cleanly and displayed sequentially as each modal
is decided or dismissed. An active modal can also be dismissed via the `Esc` key.

In the Dashboard, the approval modal is rendered as an overlay dialog centered
over the conversation interface with a darkened backdrop (`.approval-modal-backdrop`),
bringing immediate focus to the pending approval without cluttering the chat log.
Each modal stays visible until its decision is submitted or its expiration deadline
passes.

The modal architecture keeps approval decisions prominent and actionable across the
entire chat UI without taking away permanent layout space from conversation history.
The modal uses the same durable inbox IDs, endpoints, and computer permission choices
as before; rendering never authorizes execution. Failed requests remain available
for retry. When a computer permission is recorded but its action still requires a
separate exact-action confirmation, the dashboard shows that confirmation
requirement and identifier in a status notice; it does not grant the confirmation.
A successful explicit approval scrolls the conversation to its end; a denied or
failed request does not force the scroll. In the activity stream, pending approvals
clearly display their waiting status (`⚠️ [WAITING FOR APPROVAL]`). Switching sessions
clears any queued modals.

Runtime event producers may provide `backend` and `tool_name` at the event's
top level or inside `metadata`, `details`, or `payload`. Classification must
preserve this context when projecting events into the TUI. In particular, an
`action.approval.required` event tagged with `backend: internal`, `codex`, or
`coding` must not be silently routed to general Activity merely because the
approval tool name itself is not a coding tool.

**Do not reintroduce:** do not discard top-level backend classification context
before `ExecutionPanel` selects its Activity box. Keep general events in
Activity and coding-backend events in Coding Activity. Do not restore separate
dashboard request cards, grant approval during render, resurrect resolved
requests on replay, or hide failures that need retry. Modal controls must
submit an explicit decision to the guarded endpoint. A decision for one request
must not mark other pending requests resolved.

## Approving matching requests

The transactional approval controls offer a match-all option — labelled
**Allow all matching queued** in the permission modal screen. This
approves the displayed request and considers only requests already queued when
the user makes that choice; it is not a standing authorization for future
requests or other inbox entries. Non-matching or unverified requests remain
queued for individual review. The comparison is deterministic and fails closed:
task, execution, reviewer, risk and request material must match. Transactional
actions additionally require matching current action digests, policy and full
invocation fingerprints; only adapters with sufficient fingerprints (shell and
MCP) are automatically matched. Missing or protected data that cannot be
compared prevents a match. Each request still goes through the usual reviewer,
expiration, action and grant checks, with a version check against changes after
comparison. No policy-denied action is authorized by this option.

`/approve` approves a pending implementation plan; it does not silently approve
inbox requests. Review those through the approval modal or inbox instead.

## Expiration and terminal item handling

Approval modals represent actionable, pending decisions. Inbox items or actions that have reached a terminal status (`EXPIRED`, `CANCELLED`, `APPROVED`, `DENIED`, etc.) or whose `expires_at` timestamp is in the past must never be displayed in the approval modal.

1. **Suppression at entry**: Both the dashboard (`live_chat.js`) and TUI (`app.py`) inspect the item's `status` and `expires_at` before presenting a modal. If an item is already terminal or expired, it is suppressed.
2. **Real-time dismissal**: If an item expires or is invalidated while awaiting review, events such as `inbox.item.expired`, `action.approval.expired`, and `action.approval.invalidated` dismiss the active modal immediately. Additionally, both surfaces schedule timers based on `expires_at` to auto-dismiss when the deadline arrives.
3. **Graceful terminal resolution**: If a user attempts to decide a modal whose underlying request became terminal or expired concurrently, the surfaces treat errors like `"inbox item is already terminal: expired"` as terminal outcomes, dismissing the modal and notifying the user instead of leaving the modal stuck in an error state.

User-owned focused verification command:

```bash
PYTHONPATH=src pytest tests/test_tui_approval_realtime.py -q
PYTHONPATH=src pytest tests/test_computer_control.py -q
node --test tests/dashboard/live_chat_approval_ui.test.mjs tests/dashboard/live_chat_reducer.test.mjs
```

## Shared activity and steering

The public activity projection preserves approval request identity, scope,
expiration, and safe display fields. Visibility settings affect presentation
only; existing guarded decisions and exact-request matching still grant any
required authorization. Steering pending/queued/delivered/applied states are
observations, never approval grants. Applied is published after the fenced
acknowledgement commits. See [activity regression contracts](regressions/agent-activity-visibility.md).
