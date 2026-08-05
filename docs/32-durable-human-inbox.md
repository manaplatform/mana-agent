# Durable Human-in-the-Loop Inbox

Mana-Agent persists approval and clarification requests independently from the
gateway, worker, dashboard, or terminal process that created them. The inbox is
the authority for the human request and response; transient events only notify
clients that authoritative state should be reloaded.

## Architecture and storage

The `human_inbox` package separates typed request/response models, the repository,
identity resolution, signed tokens, notification adapters, lifecycle/recovery,
and user surfaces. Local records follow Mana's existing atomic state-store pattern:

```text
~/.mana/inbox/
  items/          mutable optimistic-versioned item projections
  protected/      owner-access-controlled full context and sensitive answers
  audit/          immutable one-file-per-event audit evidence
  deliveries/     immutable notification attempt records
  logs/            reserved redacted diagnostics
  identities.json explicit reviewer, role, and group directory
  response-signing.key
```

Writes use a cross-process lock, file and directory flush, atomic replacement,
version checks, request/response idempotency keys, deduplication keys, and durable
resume claims. WebSockets, futures, and process-local queues are never the source
of truth. The protected directory and signing key use owner-only permissions where
the platform supports them. Cards, events, audit details, and third-party
notifications receive only explicitly disclosed, redacted fields.

Mana-Agent creates this directory only when a validated workflow actually needs a
durable approval or structured human response. A route that stops because a
capability is unavailable, including `COMPUTER_NOT_AVAILABLE`, creates none of
these inbox paths or files.

## Reviewer directory

Specific-person assignment is explicit and authorizes only that identity. Role and
group assignment is resolved through `~/.mana/inbox/identities.json`:

```json
{
  "identities": [
    {"identity_id": "alex", "display_name": "Alex", "roles": ["security"], "groups": ["platform"], "active": true},
    {"identity_id": "sam", "roles": ["finance"], "groups": [], "active": true}
  ]
}
```

If a role or group has no eligible member, the item remains pending with a
configuration error. Mana-Agent does not route it to an arbitrary administrator.
Delegation creates a new linked record and retains historical ownership. Current
membership is rechecked when an item is read or answered.

## Branch suspension and recovery

An input-producing agent supplies a typed `InboxRequest` selected by its validated
workflow decision. For a supervised branch, the request must reference that
branch's active durable checkpoint. Creation then:

1. persists the inbox item;
2. moves only the referenced task branch to `waiting` with
   `waiting_for_approval` or `waiting_for_clarification` metadata;
3. releases its lease and worker resources;
4. leaves sibling branches runnable;
5. records the branch-suspension audit event.

An approved or answered item atomically obtains one durable resume claim. The
supervisor validates the task, inbox item, and checkpoint tuple; persists the
structured response in `TaskRecord.human_inputs`; and queues exactly that branch.
The claim ID is also persisted on the task, preventing duplicate continuation
after retries, concurrent responses, or a crash between response persistence and
resume completion. Denial is a human decision, not an execution failure.

Startup reconciliation restores missing waiting projections, completes valid
responses whose process crashed before resume, identifies orphaned requests, and
reports branches waiting without an item. Run:

```bash
mana-agent inbox maintain
mana-agent doctor --only persistence/human-inbox
```

`inbox maintain` is an idempotent cron/automation hook for expiration, reminders,
and reconciliation. It does not choose a replan or escalation fallback. Policies
requesting those actions remain blocked and emit evidence for a new validated
model decision.

Legacy server and remote-SSH adapters also claim a durable one-attempt execution
fence before external side effects. A crash with an uncompleted claim is reported
by `doctor` for outcome reconciliation and is never retried automatically; this
prefers an explicit uncertain outcome over executing the approved action twice.

## Binary action approval

The transactional pipeline is:

```text
propose → preview → policy decision → durable inbox approval
        → exact approval grant → execute → verify → commit or compensate
```

The inbox item binds the policy decision ID, permission request ID, action intent
ID, and canonical digest of the exact action, targets, redacted parameters,
declared effects, risk, disclosure, reversibility, and preview. Material changes
supersede the old item and require a new decision. Approval authorizes one attempt
of the exact action; it is not execution evidence. Existing dashboard and TUI
approval entry points delegate to this durable record and the existing
single-use `ApprovalGrant` rather than creating another approval authority.
The chat gateway also maps legacy server and remote-SSH permission request IDs
to inbox items before emitting their existing events. Their exact protected
execution intents can be reconstructed after a gateway restart, while the
original request IDs remain valid compatibility handles.

Every action approval also stores explicit `reversible`, `compensatable`,
`irreversible`, `externally_visible`, `data_disclosing`, and
`potentially_billable` labels. Labels whose value was not declared by the
adapter remain `null` (unknown); the inbox does not guess a safer classification.

Example minimal card:

```json
{
  "title": "Approve file delete",
  "risk_level": "medium",
  "branch_id": "task_child_7",
  "minimal_context": {"action_type": "file", "operation": "delete", "action_count": 1, "resource_count": 1, "effect_labels": {"reversible": true, "compensatable": false, "irreversible": false, "externally_visible": false, "data_disclosing": false, "potentially_billable": false}},
  "other_work_continues": true,
  "disclosed_fields": ["action_type", "operation", "action_count", "resource_count"]
}
```

## Clarification requests

Clarifications declare typed fields, choices, free-form policy, validation
constraints, and sensitivity. Answers are persisted as structured human task
inputs. If a field is sensitive, the item and checkpoint contain only a protected
response reference.

The shared gateway exposes typed `propose_human_input`, `observe_human_input`,
and `consume_human_input` boundaries. Observation and consumption require the
exact originating agent and task IDs; no agent-facing method can submit or forge
a reviewer response. Protected answers are dereferenced only through that
task-scoped consume boundary and remain references in durable branch state.

```python
InboxRequest(
    request_type="clarification",
    requested_fields=[{
        "field_id": "region",
        "prompt": "Which deployment region?",
        "expected_type": "choice",
        "choices": ["eu", "us"],
        "allow_free_form": False,
    }],
    allowed_responses=["answer"],
    ...,
)
```

## CLI and dashboard

```bash
mana-agent inbox list --status pending --status delivered
mana-agent inbox show <inbox-item-id>
mana-agent inbox approve <inbox-item-id> --comment "Reviewed exact preview"
mana-agent inbox deny <inbox-item-id> --comment "Target scope is too broad"
mana-agent inbox answer <inbox-item-id> --answer '{"region":"eu"}'
```

Inside chat and the TUI, `/inbox list`, `/inbox show <id>`, `/inbox approve
<id>`, `/inbox deny <id>`, and `/inbox answer <id> '<json>'` use the same
repository. New requests appear as minimal, non-blocking chat-history notices;
the notice is never the source of request state.

Filters include reviewer, role, group, task, branch, request type, and status. The
Dashboard Human Inbox page reloads persisted cards and audit history on each
render. Shared live events carry only minimal notifications and instruct clients
to reload. The API is under `/api/v1/inbox`; a web response first requests a
short-lived operation-specific token, then submits it with the matching
`X-Mana-CSRF` value. Token signatures, current reviewer authorization, nonce,
operation, item state, expiry, and action digest are all revalidated.

## Expiry, notifications, security, and observability

Expiration uses a compare-and-set terminal transition, revokes every issued token,
and applies only the explicitly configured behavior: remain blocked, cancel the
branch, deny by default, request replanning, or escalate. High-risk requests are
never auto-approved. Notification delivery is an adapter boundary; failures are
recorded and never remove or fail the item. Reminder counts and delivery attempts
are durable and rate-limited by the request policy.

Every read and response checks authorization. Raw response tokens, protected
context, secrets, and sensitive answers are excluded from cards, URLs, socket
events, notification payloads, metrics, and audit details. Audit events record the
fields disclosed at decision time. `/api/v1/inbox/metrics` reports aggregate
pending age, response latency, outcome/risk counts, expirations, delivery failures,
and rejected-response counts without approval content.

The initial release supports complete binary approval and structured
clarification. The schemas reserve clean extension points for edited-parameter
approval, quorum, conditional, and multi-step review; none is simulated through a
single reviewer field.

## Operational examples

A role-routed request uses an explicit directory-backed assignment:

```python
reviewer=ReviewerAssignment(reviewer_type="role", reviewer_id="security")
```

An expiring request declares what happens instead of relying on a default action:

```python
escalation_policy=EscalationPolicy(expiry_behavior="cancel_branch")
```

For a dashboard response, request an operation-specific token from
`POST /api/v1/inbox/<id>/response-token`, then submit it to
`POST /api/v1/inbox/<id>/respond` with its matching `X-Mana-CSRF` header. The
dashboard reloads the item afterward rather than treating the response event as
history.

For branch-specific suspension, the item carries the child `task_id`,
`branch_id`, and active `checkpoint_id`. A valid terminal response adds one
structured entry to that child's `human_inputs` and queues only that child; its
parent and runnable siblings are not resumed or stopped. A denial is injected as
a distinct `deny` input so the branch can stop, compensate, or request a new
model-driven plan without representing the human decision as execution failure.
