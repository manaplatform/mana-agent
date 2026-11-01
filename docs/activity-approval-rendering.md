# Approval request activity rendering

Approval-required lifecycle events are rendered in the existing consolidated
execution panel: general approvals belong in **Activity**, while approvals
associated with a coding backend belong in **Coding Activity**. Dashboard
transactional, server, API and computer decisions are made in a floating
**toast stack** above the conversation. Each toast stays visible until its own
decision is recorded and shows the request preview and scope. The TUI positions
the toast layer directly above the input bar within the layout flow for
transactional, server, API, computer and remote SSH requests, avoiding obscuring
or blocking conversation cells. It disappears when empty; requests disappear
only after a successful decision or expiration. Computer and remote SSH requests
retain their once, session and always choices. Approval tools do not create
separate TUI cards.

Previously the dashboard rendered transactional controls inside activity rows
and computer permission controls on separate timeline cards, and both surfaces
used a collapsible **Pending approvals** drawer. Resolved requests could remain
visible as if they still required attention, and the same pending request was
shown twice — once as a drawer entry and once as an activity row. The toast
stack replaces both control locations and the drawer, trading inline controls
for one consistently visible queue. The toast stack uses the same durable inbox
IDs, endpoints and computer permission choices as before; rendering never
authorizes execution. Failed requests remain available for retry. When a
computer permission is recorded but its action still requires a separate
exact-action confirmation, the dashboard shows that confirmation requirement
and identifier in a status notice; it does not grant the confirmation. A
successful explicit approval scrolls the conversation to its end; a denied or
failed request does not force the scroll. The queue does not depend on the
truncated activity history. In the TUI, the toast layer is laid out cleanly
without covering or hiding the activity block or messages underneath it, and
pending approvals in the activity stream clearly display their waiting status
(`⚠️ [WAITING FOR APPROVAL]`). Failed decisions remain actionable. Switching
sessions clears the old toasts.

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
requests on replay, or hide failures that need retry. Toast controls must
submit an explicit decision to the guarded endpoint. The TUI toast buttons use
the same guarded decision handler and leave failed decisions available for
retry. A decision for one request must not mark other pending requests
resolved. Do not render a pending approval as both a toast and an activity row:
the pending request belongs to the toast stack alone, and the activity stream
must hide it until its decision is recorded.

## Approving matching requests

The transactional approval controls offer a match-all option — labelled
**Approve all matching queued** in the TUI toast and **Allow all matching
queued** in the permission screen. This
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
inbox requests. Review those through the approval toast or inbox instead.

## Expiration and terminal item handling

Approval toasts represent actionable, pending decisions. Inbox items or actions that have reached a terminal status (`EXPIRED`, `CANCELLED`, `APPROVED`, `DENIED`, etc.) or whose `expires_at` timestamp is in the past must never be displayed in the approval toast stack or overlay.

1. **Suppression at entry**: Both the dashboard (`live_chat.js`) and TUI (`app.py`) inspect the item's `status` and `expires_at` before mounting a toast. If an item is already terminal or expired, it is suppressed.
2. **Real-time dismissal**: If an item expires or is invalidated while awaiting review, events such as `inbox.item.expired`, `action.approval.expired`, and `action.approval.invalidated` dismiss the toast immediately. Additionally, both surfaces schedule timers based on `expires_at` to auto-dismiss when the deadline arrives.
3. **Graceful terminal resolution**: If a user attempts to decide a toast whose underlying request became terminal or expired concurrently, the surfaces treat errors like `"inbox item is already terminal: expired"` as terminal outcomes, dismissing the toast and notifying the user instead of leaving the toast stuck in an error state.

User-owned focused verification command:

```bash
PYTHONPATH=src pytest tests/test_tui_approval_realtime.py -q
PYTHONPATH=src pytest tests/test_computer_control.py -q
node --test tests/dashboard/live_chat_approval_ui.test.mjs tests/dashboard/live_chat_reducer.test.mjs
```
