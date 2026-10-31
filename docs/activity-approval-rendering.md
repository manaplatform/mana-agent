# Approval request activity rendering

Approval-required lifecycle events are rendered in the existing consolidated
execution panel: general approvals belong in **Activity**, while approvals
associated with a coding backend belong in **Coding Activity**. Dashboard
transactional, server, API and computer decisions are made in a single
collapsible **Pending approvals** drawer above the conversation. It opens for
new requests, retains each preview and scope, and hides a request after its
decision is recorded. The TUI continues to place transactional, server and API
controls inside the matching Activity or Coding Activity box. TUI computer and
remote SSH permission requests retain their scoped dialog. Approval tools do
not create separate TUI cards.

Previously the dashboard rendered transactional controls inside activity rows
and computer permission controls on separate timeline cards. Resolved requests
could remain visible as if they still required attention. The drawer replaces
both control locations, trading inline controls for one consistently visible
queue. Activity retains lifecycle events, but resolved request rows are hidden;
the reducer retains decision state to prevent replay from reopening them. The
drawer uses the same durable inbox IDs, endpoints and computer permission
choices as before; rendering never authorizes execution. Failed requests remain
available for retry. A successful explicit approval scrolls the conversation to
its end; a denied or failed request does not force the scroll. The queue does
not depend on the truncated activity history.

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
requests on replay, or hide failures that need retry. Drawer controls must submit
an explicit decision to the guarded endpoint. The TUI inline buttons use the
same guarded decision handler and leave failed decisions available for retry.
A decision for one request must not mark other pending requests resolved.

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
