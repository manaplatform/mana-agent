# Mana-Agent — Resilient Execution Supervisor Test Scenarios

## Test objective

Verify that the Resilient Execution Supervisor can safely supervise long-running and delegated work across crashes, worker loss, provider failure, retries, checkpoints, verification, human waits, cancellation, and process restarts without:

- losing progress;
- executing work twice;
- accepting stale worker results;
- falsely marking work completed;
- retrying unsafe side effects;
- silently falling back to another execution provider;
- exceeding retry/recovery budgets;
- losing child results;
- leaving tasks permanently stuck.

---

# Scenario 1 — Normal verified execution

### Purpose

Validate the basic successful lifecycle.

### User prompt

```text
Create a file named resilient-test.txt in the current workspace containing:

Resilient execution supervisor test passed.
```

### Expected lifecycle

```text
created
→ queued
→ leased
→ running
→ completed_pending_verification
→ completed
```

### Expected behavior

The supervisor must:

1. create a durable task before execution starts;
2. create an attempt with a unique attempt ID;
3. acquire a worker lease;
4. execute through the selected execution provider;
5. write the result into result escrow;
6. verify `resilient-test.txt`;
7. generate an artifact verification manifest;
8. transition to `completed` only after verification succeeds.

### Verify

```bash
mana-agent tasks status <task_id>
mana-agent tasks artefacts <task_id>
mana-agent tasks logs <task_id>
```

Expected:

```text
state: completed
verification.status: passed
attempt_count: 1
```

The artifact should contain:

```text
path
size
sha256
attempt_id
verified_at
```

---

# Scenario 2 — Crash after checkpoint

### Purpose

Verify checkpoint-based recovery after the Mana-Agent process disappears.

### User prompt

```text
Perform this task in five steps.

1. Create resilient-recovery/
2. Create resilient-recovery/step1.txt
3. Create resilient-recovery/step2.txt
4. Create resilient-recovery/step3.txt
5. Create resilient-recovery/final.txt

Checkpoint after each completed step.
```

### Test procedure

Allow the task to complete steps 1–3.

Terminate Mana-Agent abruptly:

```bash
kill -9 <mana_agent_pid>
```

Restart Mana-Agent.

Then run:

```bash
mana-agent tasks recover
```

### Expected behavior

The original task ID must remain unchanged.

The old attempt should become:

```text
lost
```

A recovery decision must explicitly select:

```text
resume_checkpoint
```

A new attempt ID must be created.

The executor must receive only the validated recovery information:

```text
completed_steps
pending_steps
resume_payload
workspace references
generated-file state
```

It must not blindly replay previous tool output.

### Critical assertion

Steps 1–3 must **not execute again**.

Expected:

```text
task_id: unchanged
old_attempt: lost
new_attempt: created
recovery: resume_checkpoint
```

Final state:

```text
completed
verification.status: passed
```

---

# Scenario 3 — Worker lease expires

### Purpose

Ensure an expired worker cannot later publish a result.

### Test

Start a deliberately long-running task:

```text
Run a long read-only analysis that takes long enough for worker recovery testing.
Checkpoint progress periodically.
```

Stop worker heartbeats without killing its process.

Wait until its lease expires.

Allow recovery to create a new attempt.

Then allow the original worker to attempt to publish its result.

### Expected result

The original worker must be rejected because its:

```text
attempt_id
lease_token
attempt_generation
```

are stale.

Expected event:

```text
stale_worker_result_rejected
```

The stale worker must not:

```text
change task state
replace result escrow
complete the task
```

Only the currently leased attempt can publish authoritative state.

---

# Scenario 4 — Process crash after child result storage

### Purpose

Test result escrow durability.

### Structure

```text
parent task
 └── child task
```

### Prompt

```text
Create a parent task that delegates generation of child-result.txt to one child task,
then verifies the returned file.
```

### Procedure

Allow the child to finish.

The child result should reach:

```text
produced
→ stored
→ available
```

Kill Mana-Agent before the parent acknowledges the child result.

Restart.

Run:

```bash
mana-agent tasks recover
```

### Expected behavior

The child must **not run again**.

The existing escrow result must be delivered again to the parent.

Expected lifecycle:

```text
available
→ delivery_pending
→ delivered
→ acknowledged
```

### Critical assertion

```text
child attempt count = 1
```

No duplicate child execution is permitted.

---

# Scenario 5 — Verification prevents false completion

### Purpose

Prove that agent text cannot establish successful completion.

### User prompt

```text
Create verification-test.txt containing "verified".
```

### Fault injection

Modify the executor/test provider so that it returns:

```text
Successfully created verification-test.txt
```

without actually creating the file.

### Expected behavior

The lane may report execution success, but the supervisor must transition only to:

```text
completed_pending_verification
```

Verification must fail:

```text
file_exists: false
```

Task must **not** become:

```text
completed
```

### Expected state

```text
state: completed_pending_verification
verification.status: failed
```

This is one of the most important supervisor tests.

---

# Scenario 6 — Artifact modified by wrong attempt

### Purpose

Validate attempt provenance checks.

### Procedure

Attempt 1 starts creating:

```text
attempt-artifact.txt
```

Force Attempt 1 to fail.

Start Attempt 2.

Before Attempt 2 finishes, externally create or modify:

```text
attempt-artifact.txt
```

using Attempt 1 or another process.

### Expected behavior

If the completion contract requires:

```text
modified_by_active_attempt
```

verification must reject the artifact.

The supervisor must not accept filesystem existence alone as proof.

---

# Scenario 7 — Safe read-only retry

### Purpose

Verify automatic recovery of safe operations.

### Prompt

```text
Inspect the repository and count all Python files.
Do not modify anything.
```

Side-effect classification:

```text
read_only
```

### Fault injection

Kill the executor halfway through.

### Expected behavior

Recovery may safely schedule:

```text
retry_task
```

or:

```text
resume_checkpoint
```

depending on checkpoint availability.

Task ID remains stable.

Attempt ID changes.

Expected:

```text
attempt_1: lost
attempt_2: running
task_id: unchanged
```

---

# Scenario 8 — Block unsafe non-idempotent retry

### Purpose

Verify that ambiguous external side effects are not replayed.

### Example task

Using a mock external API:

```text
Send a notification to customer 123.
```

Classify action as:

```text
non_idempotent
```

### Fault injection

Make the external server accept the request, then terminate the executor before Mana-Agent receives the response.

The supervisor therefore cannot know whether the action happened.

### Recovery attempt

```bash
mana-agent tasks retry <task_id>
```

### Expected behavior

Retry must be refused.

Expected:

```text
action may already have occurred
retry not scheduled
manual intervention required
```

The action must **not be executed again**.

---

# Scenario 9 — Deduplicated side-effect retry

### Purpose

Test allowed side-effect recovery.

### Action

Use a mock API supporting an idempotency key:

```text
POST /orders
Idempotency-Key: mana-task-123-action-1
```

Classification:

```text
deduplicated
```

### Fault injection

Allow the remote order creation to succeed.

Drop the response.

Retry.

### Expected behavior

Mana-Agent must reuse the stable idempotency key.

The external system should return the original result.

Expected:

```text
external orders created: 1
execution attempts: 2
```

Never:

```text
external orders created: 2
```

---

# Scenario 10 — No execution-provider fallback

### Purpose

Verify execution fabric and supervisor coordination.

Configure:

```toml
[MANA_EXECUTION_ROUTING]
deny_silent_fallback = true
```

### Prompt

```text
Run this task explicitly in Docker:
python -c "print('docker supervisor test')"
```

Set:

```text
explicit_provider = "local-docker"
```

Stop Docker before execution.

### Expected behavior

The task must fail with a provider/capability error.

It must **not** execute through:

```text
local-process
```

Expected:

```text
selected_provider: local-docker
execution: rejected/unavailable
fallback_provider: none
```

---

# Scenario 11 — Explicit SSH execution survives supervisor lifecycle

### Purpose

Verify SSH identity and execution metadata survive supervision.

### Prompt

```text
SSH to the configured test host and run:

echo resilient-ssh-test > resilient-ssh-test.txt

Then verify the remote file exists.
```

### Requirements

Connection information must come from the validated request/configuration.

Mana-Agent must preserve:

```text
hostname
user
port
identity_file reference
remote task
```

It must not invent missing SSH connection information.

### Fault injection

Disconnect the Mana-Agent process while the remote operation is running.

### Expected behavior

Recovery must determine whether the operation is safe to resume/retry.

A non-idempotent unknown command must not automatically replay.

A known read-only or explicitly idempotent SSH command can recover according to policy.

---

# Scenario 12 — Docker sandbox disappears

### Purpose

Validate provider loss and supervisor recovery.

### Prompt

```text
Run a multi-step repository inspection inside local-docker.
Checkpoint after every major step.
```

### Procedure

During execution:

```bash
docker kill <container>
docker rm <container>
```

### Expected behavior

ExecutionManager reports runtime loss.

ExecutionSupervisor determines recovery.

If checkpoint is valid:

```text
resume_checkpoint
```

A new container may be created while preserving the logical task ID.

Expected:

```text
logical task ID: same
sandbox/container ID: may change
attempt ID: new
```

---

# Scenario 13 — Invalid state transition

### Purpose

Verify state-machine enforcement.

### Injection

Attempt:

```text
created → completed
```

or:

```text
completed → running
```

### Expected behavior

The supervisor must:

1. reject the transition;
2. emit immutable:

```text
invalid_transition
```

3. raise a typed error;
4. execute no additional action.

The task state must remain unchanged.

---

# Scenario 14 — Concurrent claim race

### Purpose

Ensure two workers cannot lease the same queued task.

### Procedure

Start two worker processes simultaneously.

Both try to claim the same queued task.

### Expected behavior

Exactly one succeeds.

Expected:

```text
worker A: lease acquired
worker B: CAS/lease acquisition rejected
```

or the reverse.

Assertions:

```text
active attempts = 1
valid lease tokens = 1
```

---

# Scenario 15 — Restart during checkpoint write

### Purpose

Test atomic persistence.

### Procedure

Crash the process while checkpoint state is being written.

### Expected behavior

After restart, Mana-Agent must find either:

```text
previous complete checkpoint
```

or:

```text
new complete checkpoint
```

It must never load partially written JSON as valid state.

Temporary files must not become authoritative records.

---

# Scenario 16 — Corrupted checkpoint

### Purpose

Validate fail-closed recovery.

### Procedure

Manually corrupt a test checkpoint fixture:

```json
{
  "schema_version": 999,
  "payload": "broken"
```

or alter its checksum.

### Expected behavior

The supervisor must reject:

```text
resume_checkpoint
```

using that checkpoint.

It must not silently ignore validation failure and restore arbitrary state.

Recovery may proceed only if another valid recovery decision is produced.

---

# Scenario 17 — Recovery budget exhaustion

### Purpose

Ensure infinitely failing work terminates.

### Prompt

Use a test executor that always raises:

```text
TransientExecutionError
```

Configure a small retry budget such as:

```text
retry budget: 3
```

### Expected sequence

```text
attempt 1 failed
→ retry_scheduled

attempt 2 failed
→ retry_scheduled

attempt 3 failed
→ budget_exhausted
```

Expected final state:

```text
budget_exhausted
```

No fourth execution is allowed.

---

# Scenario 18 — Separate failure budgets

### Purpose

Verify retry categories do not incorrectly consume each other.

Inject failures separately for:

```text
infrastructure
model
tool
verification
lease_loss
replan
```

### Expected behavior

Each category must update its own counter.

For example:

```text
model failures: 2
verification failures: 1
infrastructure failures: 0
```

A verification failure must not accidentally consume the infrastructure retry budget.

---

# Scenario 19 — Parent waits for multiple children

### Structure

```text
parent
├── child A
├── child B
└── child C
```

Run variants using:

```text
fail_fast
wait_all
best_effort
minimum_success_count
dependency_graph
```

### Example

Set:

```text
minimum_success_count = 2
```

Results:

```text
child A: completed
child B: failed
child C: completed
```

### Expected behavior

Parent can continue because:

```text
success_count = 2
```

The parent status must show blockers while children remain unresolved.

---

# Scenario 20 — Parent restart with running children

### Purpose

Ensure parent-child linkage remains durable.

### Procedure

Start:

```text
parent
├── child A running
└── child B running
```

Kill Mana-Agent.

Restart.

### Expected behavior

Startup reconciliation must restore:

```text
parent-child relationships
child task states
checkpoints
attempt generations
result escrow
```

No child should become detached or duplicated.

---

# Scenario 21 — Child completion while parent offline

### Purpose

Test delayed delivery.

### Procedure

Allow a remote child to complete while the parent process is unavailable.

Restart the parent.

### Expected behavior

The parent discovers the unacknowledged result from escrow.

No child re-execution occurs.

---

# Scenario 22 — Cancellation propagation

### Structure

```text
parent
├── child A
├── child B
└── child C
```

Execute:

```bash
mana-agent tasks cancel <parent_id> --reason "supervisor test"
```

### Expected behavior

Cancellation propagates:

```text
child-first
```

Cancellable children become:

```text
cancelled
```

Then the parent becomes:

```text
cancelled
```

Logs, checkpoints, artifacts, and escrow remain available.

---

# Scenario 23 — Cancellation during irreversible action

### Purpose

Ensure Mana-Agent does not pretend an external action was undone.

### Example

Use a mocked action:

```text
submit payment
```

Mark the action phase as irreversible.

Cancel the parent after submission starts.

### Expected behavior

The task should report:

```text
blocked_by_side_effect
```

or equivalent explicit irreversible-side-effect state/reason.

It must not report successful rollback unless compensation actually exists and succeeds.

---

# Scenario 24 — Wall-clock deadline expires

### Purpose

Verify dead tasks cannot be resurrected.

### Configure

Give the task a short absolute deadline.

Stop execution until the deadline expires.

Run:

```bash
mana-agent tasks recover
```

### Expected behavior

The supervisor must not:

```text
retry
resume checkpoint
replan
```

the expired task identity.

Expected:

```text
deadline_elapsed: true
requeue: false
```

A new chat request for the same work should create a **new task ID** with lineage to the previous task.

---

# Scenario 25 — Human approval wait

### Purpose

Test integration with Durable Human-in-the-Loop Inbox.

### Task

Run an operation requiring approval.

Expected task state:

```text
running
→ waiting
```

The branch stores:

```text
waiting_inbox_item_id
wait_reason
human_inputs
consumed_resume_claim_ids
```

### Important assertions

While waiting:

- its durable checkpoint exists;
- its worker lease is cleared;
- sibling branches remain runnable;
- the entire parent task does not need to be frozen.

Approve through the inbox.

Expected:

```text
waiting
→ queued
→ leased
→ running
```

Only the referenced branch resumes.

---

# Scenario 26 — Approval persisted immediately before crash

### Purpose

Test exactly-once HITL continuation.

### Procedure

1. task waits for approval;
2. approval response is persisted;
3. crash Mana-Agent before resume state is written;
4. restart;
5. run recovery.

### Expected behavior

Startup reconciliation discovers the persisted terminal response.

The affected branch resumes exactly once.

Assertion:

```text
continuation executions = 1
```

---

# Scenario 27 — Duplicate approval response

### Purpose

Protect against replayed human responses.

### Procedure

Approve one inbox item.

Submit the same signed response/claim again.

### Expected behavior

The stored:

```text
consumed_resume_claim_ids
```

must cause the second resume attempt to be rejected or ignored.

No duplicate task attempt should start.

---

# Scenario 28 — Scoped memory preserved across retry

### Purpose

Test supervisor integration with scoped memory capsules.

### Setup

Child receives capsule revisions:

```json
{
  "project": 18,
  "task": 4,
  "team": 7
}
```

Kill the child.

Recover it.

### Expected behavior

The replacement attempt must receive exactly:

```json
{
  "project": 18,
  "task": 4,
  "team": 7
}
```

It must not rebuild context from unrestricted parent history.

---

# Scenario 29 — Result escrow does not automatically merge memory

### Purpose

Validate separation between task result completion and memory approval.

### Procedure

A child produces:

```text
result
+
new capsule revision
```

### Expected behavior

Result escrow records the returned capsule revision.

But successful child completion must **not** automatically authorize shared-memory merge.

---

# Scenario 30 — Context/cost reservation survives execution failure

### Purpose

Test integration with ContextCostGovernor.

### Procedure

Start a model-backed supervised task.

Reserve estimated tokens.

Crash the executor before completion.

### Expected behavior

The system must correctly reconcile:

```text
reserved tokens
actual tokens consumed
released unused reservation
verification reserve
```

Usage must not be counted twice after recovery.

---

# Scenario 31 — Recovery does not double-charge model usage

### Procedure

Attempt 1 consumes model tokens and crashes.

Attempt 2 resumes.

### Expected behavior

Accounting should resemble:

```text
attempt_1 actual usage = recorded once
attempt_2 actual usage = recorded once
```

Never:

```text
attempt_1 usage duplicated during recovery
```

---

# Scenario 32 — Supervisor vs LaneCoordinator ownership

### Purpose

Confirm lane completion is not logical task completion.

### Injection

Make:

```text
LaneCoordinator.finish(COMPLETED)
```

execute successfully while the completion contract intentionally fails.

### Expected behavior

Lane status may indicate execution finished.

Supervisor remains:

```text
completed_pending_verification
```

TaskBoard must not project:

```text
DONE
```

until the supervisor provides:

```text
durable result ID
+
passed verification manifest
```

---

# Scenario 33 — ExecutionManager process success is insufficient

### Purpose

Verify provider process exit is not treated as task completion.

### Executor

Run:

```bash
exit 0
```

without producing the required output artifact.

### Expected behavior

ExecutionManager:

```text
exit_code = 0
```

Supervisor:

```text
verification failed
state != completed
```

---

# Scenario 34 — Stale TaskBoard projection

### Purpose

Ensure TaskBoard cannot override durable supervisor state.

### Setup

Supervisor contains:

```text
task_123 = running
```

TaskBoard projection is deleted or changed to:

```text
DONE
```

Restart.

### Expected behavior

Startup reconciliation restores the TaskBoard projection from supervisor state.

The stale projection must never become authoritative.

---

# Scenario 35 — Supervisor store survives TaskBoard loss

Delete only the test TaskBoard projection.

Keep:

```text
~/.mana/execution/
```

Restart Mana-Agent.

### Expected behavior

Supervisor tasks remain intact.

TaskBoard is reconstructed/projected from authoritative supervisor state.

No new task identity is created for the existing work.

---

# Scenario 36 — Corrupted TaskBoard fails closed

Inject corrupted TaskBoard JSON.

### Expected behavior

Mana-Agent must not interpret corruption as:

```text
empty board
```

It should surface a storage/state error and preserve authoritative execution records.

---

# Scenario 37 — Stale action generation rejected

### Purpose

Ensure delayed actions from previous attempts cannot mutate state.

### Sequence

```text
attempt generation 1 starts action
attempt generation 1 is lost
attempt generation 2 begins
generation 1 action callback arrives
```

### Expected behavior

Generation 1 update must be rejected.

Only generation 2 may update the current task/action state.

---

# Scenario 38 — Recovery decision missing

### Purpose

Verify no implicit recovery fallback exists.

### Setup

A stopped task is recoverable, but the recovery decision service returns no valid decision.

### Expected behavior

The supervisor must stop.

It must not automatically choose:

```text
retry
resume
local execution
another model
another worker
```

Expected error should explain the missing/invalid recovery decision.

---

# Scenario 39 — Recovery decision references wrong checkpoint

### Setup

Task:

```text
task_A
```

Decision incorrectly references:

```text
checkpoint belonging to task_B
```

### Expected behavior

Runtime validation rejects the decision.

No fallback checkpoint should be selected.

---

# Scenario 40 — Live data must not resume stale evidence

### Purpose

Test freshness policy.

### Prompt

```text
Get the current cryptocurrency price.
```

Allow a checkpoint to contain a previous price.

Stop the task.

Later send a continuation requesting the current price again.

### Expected behavior

Mana-Agent must execute a fresh retrieval.

It must not return the old checkpoint price as current information.

The same principle should be tested for:

```text
email
calendar
weather
news
availability
account state
remote resource state
```

---

# Recommended chaos test

This should be the main end-to-end acceptance test.

## Task

```text
Analyze this repository.

Delegate three independent subtasks:

1. Count Python files.
2. Produce architecture-summary.md.
3. Produce test-summary.json.

Checkpoint after every completed child result.

After all children complete, produce resilient-supervisor-report.md containing
their combined results.

Verify every generated artifact before completing.
```

## Inject failures

During the task:

1. kill child A worker;
2. kill the Docker container running child B;
3. let child C complete but kill Mana-Agent before result acknowledgement;
4. restart Mana-Agent;
5. run recovery;
6. expire one recovered worker lease;
7. allow its old worker to send a stale result;
8. recover again;
9. allow all current attempts to complete.

## Expected final execution tree

```text
root
├── child_A
│   ├── attempt_1 LOST
│   └── attempt_2 COMPLETED
│
├── child_B
│   ├── attempt_1 LOST
│   └── attempt_2 COMPLETED
│
└── child_C
    └── attempt_1 COMPLETED
        result recovered from escrow
```

Root:

```text
completed
verification.status = passed
```

### Mandatory assertions

```text
root task IDs duplicated                    = 0
child logical task IDs duplicated           = 0
child C re-executions                       = 0
accepted stale results                      = 0
unverified completed tasks                  = 0
lost escrow results                         = 0
duplicate external side effects             = 0
silent execution-provider fallbacks         = 0
tasks left indefinitely waiting             = 0
```

---

# Persistence verification

After the tests inspect:

```bash
find ~/.mana/execution -maxdepth 2 -type f
```

Expected storage areas:

```text
~/.mana/execution/
├── tasks/
├── attempts/
├── actions/
├── checkpoints/
├── results/
├── artefacts/
├── events/
└── logs/
```

Verify that persisted records do not contain:

```text
raw lease token
SSH private key
API key
credential value
secret payload
```

Only appropriate hashes/references should be persisted.

---

# Final acceptance criteria

The Resilient Execution Supervisor passes the test suite only if all of these invariants hold:

1. **Task identity is durable.**
   Retry and recovery change attempt IDs, not logical task IDs.

2. **Exactly one worker owns the active attempt.**

3. **Stale workers cannot publish authoritative results.**

4. **Checkpoints resume verified progress instead of blindly replaying work.**

5. **Child results survive parent/process crashes.**

6. **External side effects are never ambiguously replayed.**

7. **Provider failure never triggers an unauthorized local fallback.**

8. **Execution success is different from verified task success.**

9. **TaskBoard and LaneCoordinator cannot independently mark logical work complete.**

10. **Deadlines, cancellation, budgets, leases, and verification are enforced after restart.**

11. **Human approval resumes only the affected branch and only once.**

12. **Scoped memory revision maps survive restart/retry unchanged.**

13. **Context/token accounting remains correct across failed attempts and recovery.**

14. **A process crash at any transition boundary cannot produce duplicate successful effects.**

15. **After recovery every task ends in an explainable durable state:**

```text
completed
cancelled
failed
budget_exhausted
waiting for explicit intervention
```

No task should disappear, silently restart, or remain permanently ambiguous.