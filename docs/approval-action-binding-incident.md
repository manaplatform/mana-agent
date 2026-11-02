# Approval judgment and action binding incident

## Symptom and evidence

A reported model-judgment denial followed an approval request for a Git commit. The quoted commit and the original denial log were not available in this checkout; their precise cause and whether a human actually approved cannot be established from this investigation. An approval request is not evidence of an approval. No quoted commit was executed during this work.

## Verified gaps

`approval_request` used model judgment to approve or deny an inbox item and could submit under the assigned human reviewer's identity, or issue a grant itself. The internal coding dispatcher also converted foreign approvals into new grants by model judgment, and automatically retried failed commands with a model-suggested replacement. Such judgment is not a human decision. Standalone command requests and execution adapters use different task IDs, argv representations and execution metadata, while Git has its own adapter and exact action identity; a shell-command grant cannot authorize an independent Git action. An invalid grant was sometimes retried without the grant. Git commit previews previously did not fingerprint staged contents, so changes could retain the same porcelain status after approval.

## Changes and safe use

Request approval for the **action_id/inbox_item_id returned by the pending execution**, not a free-form command intended for another wrapper. A human must resolve that exact durable inbox request; resume the same action with its grant, task, arguments and execution context. Human grants remain subject to expiration, current policy, actor authorization, preview, and single-use checks. Generic Git tools resume through the same persisted action and inbox decision on retry; a shell grant is not interchangeable. For commits, changed repository state or staged diff invalidates the approval. Policy-allowed actions and explicitly configured always-approve mode are separate from human grants.

The compatibility `enable_model_judgment` option no longer resolves human requests. Previously model-resolved inbox items with the `model_judgment` channel are rejected by the transactional gateway. Explicit unknown inbox IDs fail rather than falling back to a different request. The internal coding Git dispatcher supplies task provenance and accepts an exact `action_approval_id` on Git tools; a changed task, argv, policy context, timeout or protected-action setting cannot reuse the grant. These bindings change Git idempotency identities, so older pending Git actions require a fresh request. Commits that stage worktree content during execution (such as `-a` or pathspec commits) are not supported by the index-only preview; stage first. Read-only coding sandboxes reject Git mutations and generic Git calls. Other Git entry points still require an independent provenance audit; do not infer blanket cross-wrapper authorization. The unreachable legacy shell judgment/retry path has been removed. Existing legacy tests that expected automatic model authorization or cross-task grant translation require updates.

AskAgent's generic, branch, staging, commit and push tools now expose `action_approval_id` and bind the Git action to their run and agent. Queued Git tools likewise pass their task and job provenance and exact grant to the Git adapter; a supplied repository path cannot redirect these calls outside the assigned checkout. Queue retries must retain the original job identity. Previously a generic queue call received a `validated_queue_git` policy exception solely because of its wrapper name, allowing local Git mutations including commits without approval. That exception has been removed: generic and named queue actions now follow the same Git approval policy. These changes do not make grants transferable between the internal coding, AskAgent, and queue wrappers.

AskAgent's approval tool now uses the assigned project root when resolving a pending action and its policy fingerprint, instead of the process working directory. This allows the same Git action to resume after a human decision when the agent's checkout differs from the server's working directory. Model risk assessments are explicitly advisory; neither an assessment nor a suggested replacement authorizes an action.

AskAgent also now supplies its trusted run ID and agent identity to the approval lookup, so a different run cannot retrieve the request or its grant using an action or inbox ID. Legacy callers without a run ID remain unscoped by task; the lookup is not a general authenticated identity boundary. Exact grant validation remains necessary at execution.

Approval lookup follow-up: internal coding previously accepted a model-supplied workspace for `approval_request`, while the lookup ignored task, agent and decision provenance supplied by its caller. A caller with an identifier could observe a different request or retrieve its grant; exact action checks at execution remained a separate gate. Internal coding now rejects workspace overrides and uses its assigned root. When provenance is supplied, lookup checks it against the inbox item and action, and checks file, Git and shell actions against the workspace before returning status or a grant. Unscoped legacy callers remain supported, so this is not a substitute for authenticated caller identity throughout the application. Tests for both identifier forms, foreign provenance, workspace and exact lookup are added but not run in this follow-up.

**Do not reintroduce:** model-controlled approval workspace selection or ignoring supplied task/agent/decision provenance during grant lookup. Continue to validate the exact grant at execution.

Task-scoped computer grants can cover compatible later actions only if their originating action has a current, exact, authenticated human approval; an orphaned or legacy model-issued task grant alone is not sufficient. An approved inbox item referencing a missing action fails closed instead of reporting a usable approval. Git show rejects option-like revisions so a read-only sandbox cannot pass a write option through the revision argument.

The general inbox wait path previously checked only task and agent, not the supplied source decision, so an approval for a different decision in the same task could be reported as applicable. General approval waits and `approval_request` now also reject legacy model-judgment responses, expired approvals, and responses whose reviewer is no longer authorized. These checks do not make a general approval a transactional action grant or make cross-wrapper grants transferable.

The originating policy decision for a reusable task grant must also remain unexpired and match the current policy fingerprint. A resumed action already in `APPROVED` state must recheck its human inbox expiry before claiming execution. These checks close separate expiry gaps; they do not provide an atomic cancellation fence.

### Commit index fingerprint and isolation follow-up

The earlier commit preview hashed `git diff --cached`. That is presentation output, not a complete index identity: Git configuration such as `diff.ignoreSubmodules` can omit staged gitlink changes. The commit adapter now hashes all NUL-delimited `git ls-files --stage --full-name` entries, including object IDs, modes, stages and paths, and rechecks them immediately before execution. Failure to read the index invalidates the action rather than running the commit. Pre-existing commit approvals with only the older staged-diff field fail closed and require a new request.

To fence against concurrent index modifications or hooks changing staged content during the execution window, the commit adapter now creates an isolated index snapshot (`_fenced_commit_index`), re-verifies its staged digest against the approved preview digest, and invokes the Git commit with `GIT_INDEX_FILE` bound exclusively to the snapshot. Concurrent index changes in `.git/index` cannot pollute the commit and remain intact in the working index. Upon successful commit, `.git/index` is synchronized with the committed index if no concurrent changes were introduced.

**Do not reintroduce:** use of configurable diff presentation as the sole binding for approved commit contents, proceeding when the index cannot be re-read, or running commits directly against a shared mutable index without snapshot fencing.

### Internal tool contract and shell follow-up

The internal programmatic schemas omitted `action_approval_id` for `run_command` and `run_script_once`, even though the dispatcher accepts it. Their published result schemas, and the Git result schema, also omitted the pending action and inbox identifiers; the approval result schema omitted the action ID. This made an exact resume difficult or impossible for schema-bound clients. Both direct and programmatic definitions now expose these fields; they still require a current human decision for the original action.

The internal `shell` dispatcher previously looked only for a command string and ignored the documented `action.commands` and top-level `commands` arrays. It could therefore propose an empty command rather than the command the caller requested. It now accepts one nonempty command through either array, rejects ambiguity and multiple commands, and uses the supplied nested timeout. Submit multiple operations as separate calls; an approval for one does not authorize another. The existing shell output-size and multi-command protocol limitations are not resolved here.

Shell intents now bind the specific internal wrapper (`shell`, `run_command`, or `run_script_once`), not a shared `shell` label. Older pending command actions with the shared label require a fresh approval; a `run_command` that policy previously treated as a read-only `shell` operation may now require approval. Do not transfer grants across wrappers. The new schema and dispatcher regression tests stub the final subprocess, and have not been run for this follow-up.

The shared shell policy also treated an entire shell script as read-only based on its first command. Shell composition, substitution, or Git output flags could therefore follow a read-only prefix without an approval. Nested shell invocations now use the more conservative plan-mode read-only validator before receiving a policy allow. Other commands continue to the existing deterministic policy and approval gates. The policy fingerprint includes an implementation revision, invalidating pre-change pending decisions and grants across action types, even with a custom policy version. This is not a general shell sandbox or proof that all allowlisted commands are side-effect-free.

**Do not reintroduce:** schemas that hide the pending action or exact-grant input, silently dropping a submitted command, treating similar commands in different execution wrappers as the same authorization, or classifying an interpreted shell script from its first command alone.

An approval-required action already in `APPROVED` state also fails closed if its durable human decision is missing. A valid task-scoped grant can authorize a compatible action during the same execution call after its source decision is revalidated; if that call is interrupted before claiming execution, recovery fails closed rather than treating the persisted state as proof of a human decision for the new action.

### Supervisor cancellation fencing and atomic execution claim follow-up

`ActionGateway._assert_branch_runnable` and `_claim_branch_execution` now enforce supervisor cancellation fencing for all supervised actions. Execution checks verify that:
- The task is not cancelled or in cancellation (`ExecutionState.CANCELLED`, `ExecutionState.CANCELLING`, or cancellation status in `{REQUESTED, ACKNOWLEDGED, COMPLETED, FORCED, PARTIALLY_COMPLETED}`).
- No ancestor task in the supervision hierarchy has been cancelled.
- The task is not terminal and has resumed from human waiting.
- The branch checkpoint ID and execution attempt ID match the current branch attempt.

During `execute`, `_claim_branch_execution` atomically validates these invariants under the supervisor store lock and sets `task.irreversible_side_effect_started = True`. Any subsequent supervisor `cancel()` invocation is safely blocked (`BLOCKED_BY_SIDE_EFFECT`) rather than cancelling an active side effect. In addition, `HumanInboxService.claim_action_execution` fences concurrent executions of the durable human approval.

**Do not reintroduce:** allowing approved actions to execute under cancelled tasks or ancestors, bypassing checkpoint or attempt validation, or claiming execution without atomic supervisor side-effect marking.

### Standalone shell executor follow-up

`ShellExecutor.execute_action` previously accepted any nonempty approval ID or a callback's `approve` boolean and invoked `run` directly; an ID was neither bound to the command nor consumed. Its automatic wait could also return an orphan grant without a human decision. The approval-enabled path now proposes and executes the same shell adapter through the transactional gateway, requires a matching task, command, timeout and output limit, and validates the human inbox decision before consuming the grant. A supplied grant cannot cover a multi-command call; automatic approval requests are per command. The conservative plan-mode classifier is used before treating a command as read-only. Callback booleans without a grant no longer authorize execution. Previously issued approvals from the standalone shell wrapper need fresh authorization because their action bindings differ.

This change does not make `ShellExecutor.run` an authorization boundary: it remains a low-level direct subprocess primitive, as does explicitly configuring `needs_approval=False`. Callers must not expose either to untrusted model input without another execution gate. Timed-out automatic waits emit the inbox/action IDs in the approval event, but the shell output itself does not carry these IDs. Supervisor cancellation fencing and commit-index isolation races have now been addressed with atomic execution claims and isolated index snapshot fencing.

**Do not reintroduce:** interpreting an arbitrary grant ID, callback boolean, or registry entry as permission to invoke the subprocess outside the exact transactional action.

**Do not reintroduce:** model decisions, task similarity, free-form command text, or an approval request must never impersonate a reviewer, transfer a grant to another action, auto-execute a suggested command, or bypass deterministic policy and exact-preview validation.

## Verification

The focused approval, Git propagation, human inbox and transactional gateway suites passed 34 tests. Git execution was stubbed for the approval propagation test; no commit was executed. A broader run including the legacy model-judgment suite reported 31 passed and 6 failed: those six assert automatic model authorization, foreign-grant translation, or automatic suggested-command retries, which conflict with the corrected safety contract and need revision. Broader routing and other Git entry-point coverage remains necessary.

An attempted broader test run in the current host returned output for a different,
model-suggested inspection command after the original test command failed. Its
reported successful exit therefore does not establish that the tests passed.
This host behavior is separate from the modified workspace and requires its own
fix or restart before it can be independently verified against this checkout.

A later focused run wrapped pytest in a subprocess and printed the exact argv and
exit code: `./venv/bin/pytest tests/coding/test_approval_binding_regressions.py tests/coding/test_git_approval_propagation.py tests/human_inbox/test_approval_tools.py tests/gateway/test_transactional_approval.py -q` reported exit 0 and 29 passed. The host still attached a model-judgment result to that command, so the command-execution concern above remains unresolved; the run does not independently validate the host's approval boundary.
The same subprocess-based method reported exit 0 and 194 passed for `./venv/bin/pytest tests/transactional_actions/test_policy_gated_actions.py -q`.

A subsequent focused run of `./venv/bin/python -m pytest tests/coding/test_approval_binding_regressions.py tests/coding/test_git_approval_propagation.py tests/human_inbox/test_approval_tools.py tests/gateway/test_transactional_approval.py tests/transactional_actions/test_policy_gated_actions.py -q` reported 225 passed. The host again attached model-judgment metadata, so this does not independently validate the host approval boundary. The command for removing unreachable legacy shell code was itself held for human approval and was not executed; that cleanup remains pending.

For the AskAgent and queue propagation follow-up, the host reported 7 passed for `./venv/bin/python -m pytest tests/coding/test_git_approval_propagation.py tests/test_ask_agent.py::test_ask_git_commit_resumes_only_exact_task_action tests/test_git_tools.py::test_queue_tools_manager_executes_git_namespace -q`. Commit execution in these propagation tests is stubbed. The host still returned a model-judgment field with the test command; this verifies neither the host's approval boundary nor unrelated legacy suites.

After removal of the generic queue exception, the host reported 227 passed across the focused coding approval binding, Git propagation, human inbox, transactional approval, and policy-gated action suites. The queue regression covers both named and generic commit wrappers with execution stubbed. Six additional AskAgent and advisory-judgment tests passed. The host again attached model-judgment metadata to these test commands; these results do not validate the host's separate authorization boundary. The six obsolete legacy tests mentioned above were not rerun or updated in this follow-up.

After cleanup of the unreachable shell path and the missing-human-decision check, the host reported 228 passed for the focused coding approval binding, Git propagation, human inbox, transactional approval, and policy-gated action suites. The host continued to attach model-judgment metadata; this does not validate the host's own authorization boundary.

The host's automatic suggested-command execution was subsequently observed directly: after a pytest failure it inserted a missing test import with a model-suggested command and reran the tests, although only the original pytest invocation was requested. A subsequent invocation reported 250 passing tests across the focused coding, Git, inbox, AskAgent, gateway and policy suites. The host still attached model-judgment metadata. The host is executing separate runtime code, so changes in this checkout do not establish that its unsafe retry path has been replaced; restart or update that host and independently verify its safety boundary.
