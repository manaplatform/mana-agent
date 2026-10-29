# Approval request activity rendering

Approval-required lifecycle events are rendered in the existing consolidated
execution panel: general approvals belong in **Activity**, while approvals
associated with a coding backend belong in **Coding Activity**. The approval
dialog remains the interaction surface for the user to approve or deny; it is
not a replacement for the activity record.

Runtime event producers may provide `backend` and `tool_name` at the event's
top level or inside `metadata`, `details`, or `payload`. Classification must
preserve this context when projecting events into the TUI. In particular, an
`action.approval.required` event tagged with `backend: internal`, `codex`, or
`coding` must not be silently routed to general Activity merely because the
approval tool name itself is not a coding tool.

**Do not reintroduce:** do not discard top-level backend classification context
before `ExecutionPanel` selects its Activity box. Keep general approvals in
Activity, coding-backend approvals in Coding Activity, and retain the separate
approval dialog for the actual authorization decision.

User-owned focused verification command:

```bash
PYTHONPATH=src pytest tests/test_tui_approval_realtime.py -q
```
