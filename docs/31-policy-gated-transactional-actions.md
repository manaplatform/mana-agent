# Policy-Gated Transactional Actions

Approval-required decisions now create a persisted human inbox item before any
UI prompt is surfaced. The record binds the policy decision ID, exact action and
preview digest, reviewer assignment, disclosed snapshot, expiry, and existing
permission request ID. Dashboard/TUI compatibility endpoints delegate into that
record and the existing single-use `ApprovalGrant`; they are not a second source
of approval truth. Denial and expiration remain distinct human/action states.
See [Durable Human-in-the-Loop Inbox](32-durable-human-inbox.md#binary-action-approval).

Mana-Agent treats a generated tool call as an action proposal, never as authorization. Consequential actions use the shared `transactional_actions` gateway:

```text
proposed → previewing → awaiting_policy → awaiting_approval (when required)
→ approved → executing → verifying → committed
```

Failure, cancellation, expiration, compensation, and manual-recovery states are persisted separately. Invalid state transitions and duplicate execution attempts fail closed. A tool return value alone cannot produce `committed`; complete adapter verification evidence is required.

## Intent, policy, and approval binding

`ActionIntent` records the stable action/task/transaction IDs, actor and origin agent, normalized redacted arguments, exact resources, capabilities, effects, disclosure and blast-radius classifications, reversibility, idempotency key, verification plan, and compensation strategy. Persisted arguments and previews must not contain raw credentials or sensitive bodies.

The deterministic policy evaluates normalized adapter data. Its outcomes are `allow`, `require_approval`, and `deny`. Unknown tools, unknown disclosure for HTTP, workspace escapes, malformed argv/URLs, secret disclosure, and unsafe destinations fail closed. The initial rules allow bounded reversible file writes inside configured workspace roots, require exact approval for destructive file actions and shell/remote mutations, and deny destructive shell executables, unclassified tools, insecure HTTP by default, and disallowed hosts.

Narrow infrastructure exceptions are typed policy rules rather than bypasses. The workspace coordinator may create a verified managed worktree, and may merge or remove it only with its existing validated explicit-intent checks. A persisted verification queue job may execute a bounded command and commit from an exit-status/output-hash receipt. Validated task Git workflows may perform local staging, branch, and commit operations; remote writes still require exact approval. These contexts are injected by internal coordinators and are not exposed as model tool arguments.

An approval is single-use and bound to the action ID, transaction ID, normalized intent, preview digest, policy fingerprint, scope, and expiration. A material action, preview, transaction, or policy change invalidates it. A narrow transaction approval issues one separately consumable grant per awaiting action, with every grant additionally bound to the exact durable transaction plan; it is not ambient approval for later actions.

## Previews

File previews contain affected paths, existence and hashes, operation, unified text diff where practical, collision/data-loss risks, and the exact normalized invocation. Shell previews show argv (never a softened shell-string summary), cwd, redacted environment, declared outputs, and process-effect risk. HTTP previews show method, redacted URL and host, redacted headers, body hash/size, disclosed-data summary, remote effect, and native idempotency support.

Example file preview:

```json
{"summary":"edit local file","resources":[{"path":"/workspace/app.py","change":"edit","before_sha256":"…"}],"diff":"--- /workspace/app.py\n+++ /workspace/app.py\n…"}
```

Example shell preview:

```json
{"exact_invocation":{"argv":["python","-m","tool"],"cwd":"/workspace","environment":{"token":"***REDACTED***"}},"risks":["process effects may exceed declared outputs"]}
```

Example HTTP preview:

```json
{"summary":"HTTP PATCH remote mutation","resources":[{"url":"https://api.example.test/items/1","host":"api.example.test"}],"disclosed_data":["request body sha256=… bytes=42"]}
```

## Verification, idempotency, and recovery

File adapters verify existence/absence, hashes, and move source state. Shell adapters require a zero exit status and every declared output; exit status without observable outputs is incomplete. HTTP adapters verify response semantics and record whether an independent remote-state query was available; provider-specific adapters should add that query whenever the service exposes one. Message adapters should record provider acceptance and immutable IDs without implying recall.

The durable idempotency registry rejects conflicting key reuse, blocks in-progress duplicates, and returns a prior result only when its action is committed with complete verification. This prevents supervisor or worker restart retries from repeating verified effects.

Local file snapshots enable verified rollback when preconditions still hold. A corrective email or remote mutation is compensation, not rollback. Compensation is registered by tool/operation, checks eligibility and pre-execution evidence, and must be proposed as a new policy-gated action. Irreversible or unknown actions are never compensated automatically.

`TransactionIntent` coordinates ordered/dependent actions using `stop_on_failure`, `continue_safe_actions`, `compensate_completed_actions`, or `manual_recovery_required`. Cross-adapter transactions are explicitly labeled coordinated, not atomic. Final summaries report per-action verification and manual-recovery requirements.

## Adapter development requirements

Every consequential adapter must implement intent normalization, redacted preview, execution, observable verification, native-idempotency declaration, and conservative reversibility. Optional compensation must declare eligibility, required snapshots, the compensating action, verification, and unsafe cases. Register the adapter behind the shared gateway; direct invocation is not an accepted compatibility path.

The initial adapter types are file create/edit/move/delete, repository patches, argv-based shell, Git mutations, and HTTP `POST`/`PUT`/`PATCH`/`DELETE`. API binary response artifacts also use the file adapter. Provider-specific adapters for email/messaging, calendar, GitHub, databases, cloud resources, computer control, and physical devices must supply their own truthful verification and reversibility contracts.

The model-tool dispatch boundary default-denies every tool that is neither explicitly read-only nor registered behind this gateway. Browser/computer, MCP mutation, automation, media-generation, canvas mutation, server mutation, and generic project-verification execution remain unavailable through that boundary until truthful adapters are registered; they do not fall back to their legacy executor.

## Events and audit

Transactional events use the shared execution-event envelope, so CLI, TUI, dashboard, gateway clients, and logs can render proposal, preview, policy, approval, execution, verification, commit, compensation, and manual-recovery phases. Redacted durable audit JSONL is stored under:

```text
~/.mana/transactional_actions/audit/actions.jsonl
```

Action, transaction, approval, idempotency, and snapshot records live beside that audit directory. Secrets, authorization headers, raw sensitive bodies, and secret environment values are not written to previews or audit records.
