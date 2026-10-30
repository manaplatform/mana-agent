# Approval request activity rendering

Approval-required lifecycle events are rendered in the existing consolidated
execution panel: general approvals belong in **Activity**, while approvals
associated with a coding backend belong in **Coding Activity**. Dashboard
transactional, server and API approval decisions are made inside that activity
row rather than on a separate timeline card. The TUI places the request preview,
scope and decision buttons inside the matching Activity or Coding Activity box.
Computer and remote SSH permission requests retain their existing separate
dialog because their permission scopes differ. Approval tools do not create
separate TUI cards.

Previously the dashboard independently rendered the same transactional event
as both an activity row and a timeline card. The card contained the only
approval controls, so removing it without moving those controls would strand the
request. The controls now use the same durable inbox ID and endpoint as before;
rendering an event does not authorize execution. A successful explicit approval
scrolls the conversation to its end; a denied or failed request does not force
the scroll. Pending requests remain in the truncated activity history.

Runtime event producers may provide `backend` and `tool_name` at the event's
top level or inside `metadata`, `details`, or `payload`. Classification must
preserve this context when projecting events into the TUI. In particular, an
`action.approval.required` event tagged with `backend: internal`, `codex`, or
`coding` must not be silently routed to general Activity merely because the
approval tool name itself is not a coding tool.

**Do not reintroduce:** do not discard top-level backend classification context
before `ExecutionPanel` selects its Activity box. Keep general approvals in
Activity and coding-backend approvals in Coding Activity. Do not restore a
separate dashboard transactional approval card or grant approval during render;
the inline controls must still submit an explicit decision to the guarded
endpoint. The TUI inline buttons use the same guarded decision handler and
leave failed decisions available for retry. A decision for one request must not
mark other pending requests resolved.

## Approving matching requests

The transactional approval controls offer **Allow all matching queued**. This
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
inbox requests. Review those through the activity controls or inbox instead.

User-owned focused verification command:

```bash
PYTHONPATH=src pytest tests/test_tui_approval_realtime.py -q
node --test tests/dashboard/live_chat_approval_ui.test.mjs tests/dashboard/live_chat_reducer.test.mjs
```
