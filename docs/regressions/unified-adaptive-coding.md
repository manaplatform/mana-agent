# Unified adaptive coding

## Symptom and cause

An ordinary Internal AUTO or DEBUG request could identify the failing file and
still enter the gateway's draft and scoped patch-proposal workflow. The proposal
ran read-only, so new reads or a failed check could push the task back into
evidence collection or plan revision before Coding could make the next edit.
Both the focused inspection service and the compound evidence tool capped reads
at 120 lines. Model projections could further clip source content. Read
duplicates also carried different execution metadata, which could conceal
repeated work from no-progress detection.

## Corrected contract

- The gateway binds a single realtime native Coding assignment to the validated
  AUTO or DEBUG decision. ToolRunner checks the lane capabilities, checkout,
  current lease, and action policy. The model chooses its first tool and may
  move between activities throughout the attempt.
- PLAN retains the existing scoped draft, approval digest, read-only proposal,
  and checked application. Saved staged workflows keep their plan and receipts.
- Native reads and batch reads share a budget-capped source reader. The result
  reports full and partial coverage, the total line count, the content hash,
  and whether the output stopped early. Evidence publication is opportunistic.
  Repeated reads return an observation reference without echoing the content.
- Realtime write verification runs in the assigned checkout. A failed check
  returns to Coding. Completion validates observed passing checks against the
  current source versions associated with tool receipts and observed drift.
- Recovery reconciles tool operation identities with supervisor action receipts
  before continuing. A revoked lease or an uncertain operation stops execution.
  Previously completed writes retain their original receipt hashes.

**Do not reintroduce** a required repository investigation, read quota, plan
draft, or patch-proposal handoff on the realtime Internal AUTO or DEBUG path.
Preserve ToolRunner authorization and verified completion at their existing
boundaries.

## Missing tool call in a native coding response

In realtime Internal AUTO or DEBUG, a model could reply in prose after a tool
result or on its first turn. The prompt told the model how to submit a final
result, but the loop immediately raised "The coding model must select an action
or call submit_coding_result" when the reply contained no tool call. Prior tests
exercised rejected terminal results and provider timeouts. Their fixtures
always returned a tool call for successful inference in an adaptive turn.

The [Internal Coding loop](../../src/mana_agent/coding/internal_engine.py)
now states the per-turn requirement and offers one same-format correction on
the existing assignment. The correction consumes a model turn and uses the same
offered tools. It retains previous tool results and includes the provider's
assistant message when needed to keep its continuation fingerprint valid.
Coding Activity records the missing call in a fixed diagnostic. A second omission
or an exhausted turn budget yields an explicit error. Completion still requires
the typed `submit_coding_result` call and its applicable evidence checks.

**Do not reintroduce** prose as implicit completion or select a replacement
action in the host. Keep corrections bounded by the existing model turn budget.

User verification covers correction with and without a provider continuation,
repeated prose, the turn budget, and preservation of the earlier read. The agent
did not run this check.

```sh
venv/bin/python -m pytest -q tests/test_internal_coding_engine_workflow.py -k adaptive_coding_corrects_a_missing_tool_call
```

## Disconnected runtime wiring in a completion report

At 00:21 Tehran time on November 21, Internal AUTO task
`task_20261121_000006` edited `src/cli/cli_scan.c` and recorded two passing
checkout checks. Its completion report was rejected after a correction because
the runtime wiring edges formed a disconnected path. The retained logs do not
identify the first rejection or the actual mismatched edge pairs.

The validator required consecutive edges to have matching identifiers. The
tool schema omitted that ordering rule and the correction only gave a generic
connectivity error. The terminal traceback also included an excerpt of the
rejected model payload. Existing rejection coverage exercised unsupported
completion on a read-only task.

The [wiring contract](../../src/mana_agent/repository_evidence/models.py)
now describes the order in the schema and reports all disconnected pairs by
their one-based edge numbers. [Internal Coding](../../src/mana_agent/coding/internal_engine.py)
gives an example in its instructions and sends the same field-specific error
to Coding Activity and the existing single correction. Rejected schema inputs
stay out of diagnostics. A corrected path still needs the current passing
checkout receipt and the existing runtime integration check.

**Do not reintroduce** raw rejected inputs in correction messages or accept a
completion report with disconnected wiring. Keep the model responsible for
supplying source-backed links and keep correction within its allotted turns.

The regression exercises a corrected chain and a second disconnected report
after a passing check. Both cases keep the original edit and check receipts.
The agent did not run the tests or replay the blocked task.

```sh
venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py -k 'corrects_runtime_wiring or rejects_unsupported_completion'
```

## Evidence exhausted the edit budget

On November 20 at 12:00:55 Tehran time, realtime Internal AUTO task
`task_20261120_000021` exhausted its 50 tool calls before its first edit. The
write-enabled assignment ran 49 file reads, searches, and listings plus one
Git status check. The checkpoint recorded zero changed files. Four more
model responses exceeded the remaining budget.

The [native loop](../../src/mana_agent/coding/internal_engine.py) counted a novel
observation as progress even when the model had made no progress toward an edit.
The allowance appeared only after a group was rejected. The earlier staged
patch-handoff reserve did not apply to realtime write assignments.

Realtime AUTO and DEBUG now admit targeted reads, edits, and focused checks in
one loop. The model supplies an evidence need on the read or search itself.
The runtime bounds each whole group, shows the remaining allowance before the
next decision, and adjusts the reserve based on observed changes and focused
checks. The same assignment retains current observations when a selection is
rejected. The staged tool contracts and saved checkpoint format are unchanged.

**Do not reintroduce** a mandatory discovery phase or allow optional tool work
to consume the capacity needed to edit, check, and submit the result. Preserve
the model's ability to select a scoped read after an edit or check.

User-owned verification covers a reduced-budget replay, bounded rejection,
interleaving reads and repairs, and resuming with source evidence. The agent
did not run the checks or replay the live task.

```sh
venv/bin/python -m pytest -q tests/test_internal_coding_engine_workflow.py tests/gateway/test_debug_plan_mode_contract.py -k adaptive_coding
venv/bin/python -m pytest -q tests/test_internal_coding_engine_workflow.py tests/coding/test_scoped_evidence_tools.py tests/gateway/test_debug_plan_mode_contract.py
```

## Accounting renewal after partial edits

On November 20 at 12:29 Tehran time, realtime Internal AUTO assignment
`task_20261120_000023` exhausted its 50-call allowance after two successful
edits. It spent 46 calls on inspection, including 21 shell commands. An edit
group could not fit with verification. A unified diff sent to the Codex batch
patch tool consumed another call. Verification received an absolute working
directory and a shell script that mixed edits and checks. No check ran.

The adaptive loop had no route to the accounting renewal policy. It also
charged before checking patch syntax or the verification command contract.
Direct shell inspection bypassed the source tools and their evidence needs.

The model can now request renewal near the limit, including at zero remaining
tool calls. The gateway checks live task authority, observed progress,
remaining accounting capacity, and the configured renewal count. The grant
persists with the assignment. It extends tool and model-turn counts within
existing token and monetary enforcement. Replaying a grant cannot add it
twice. Patch syntax, shell source inspection, and invalid verification inputs
are checked before dispatch. Direct setup commands and the final check can
share one isolated snapshot. A failed setup leaves the final command unrun.

**Do not reintroduce** attempt-local renewal allowances, charges for rejected
tool selections, or mixed edit and verification scripts in the isolated
checker. Keep approved mutations and check receipts on the same assignment.

The agent did not run these checks or replay the live assignment.

```sh
venv/bin/python -m pytest -q tests/test_internal_coding_engine_workflow.py tests/gateway/test_debug_plan_mode_contract.py -k adaptive_coding
venv/bin/python -m pytest -q tests/execution/test_autonomous_accounting_renewal.py tests/coding/test_coding_budget_renewal.py tests/test_repository_tools.py
```

## Distinct tool errors shared one correction

On November 20 at 12:48 Tehran time, Internal AUTO assignment
`task_20261120_000027` selected a scoped search and shell inspection in one
group. The shell inspection caused the whole group to be rejected. The next
search failed scope validation, and the assignment stopped after seven tool
calls with zero changed files.

The native loop used one correction flag for every kind of rejected selection.
It also returned the same diagnostic for every call in a rejected group and
replaced search validation details with a generic scope message.

Distinct validation errors now get separate, bounded corrections. The group
result identifies the rejected call so the model can resubmit any still-needed
sibling calls. Search errors describe the scope problem. A model that lacks a
search location can select a top-level listing and use an observed directory.

**Do not reintroduce** a shared one-shot correction flag across distinct
validation errors, automatically choose a repository-wide search, or execute
part of a rejected group.

The regression covers a shell rejection followed by a missing or malformed
search scope, a model-selected listing, and a repaired edit with a passing
check. The repeated oversized-group test originally passed the same directory
as the repository and worktree, so `WorkspaceContext` rejected it before the
budget guard ran. The fixture now uses separate temporary directories. With
two optional calls and capacity for one, it checks that every call receives a
budget rejection, the repeated group fails after one correction, and the source
is untouched. **Do not reintroduce** in-place workspace setup for this test.

The agent did not run these tests or retry the live assignment.

```sh
venv/bin/python -m pytest -q tests/test_internal_coding_engine_workflow.py tests/gateway/test_debug_plan_mode_contract.py -k 'corrects_shell_and_search_scope_separately or rejects_repeated_oversized_groups'
```

## Shell inspection repeated during correction

At 17:28 Tehran time on November 20, Internal DEBUG assignment
`task_20261120_000046` stopped with `USE_REPOSITORY_TOOL` after two successive
shell selections. Coding Activity recorded seven completed repository tool
calls and the saved assignment had zero changed files. The saved history
shows a rejected compound shell command starting with `ls` and including a
CMake invocation. Earlier in the same attempt, the provider rejected extra
`ls` arguments, and the correction presented a shell example.

The shell inspection guard correctly rejected the group, but the correction
offered the same generic shell tools and described only the inspection half
of the rejected work. The provider argument error added shell syntax guidance
regardless of which tool failed. Earlier tests corrected distinct reason
codes without checking which tools the model received during correction.

The [adaptive validator](../../src/mana_agent/coding/adaptive.py) now describes
the scoped repository arguments and directs builds to a separate direct
verification call. The [native loop](../../src/mana_agent/coding/internal_engine.py)
suppresses the generic shell tools during that correction and checkpoints its
pending reason codes. An accepted tool group or processed budget renewal
ends the correction sequence. Repository scope, group atomicity, approvals,
and the existing turn and correction limits still apply. Older checkpoints
can resume without the new optional field.

**Do not reintroduce** generic shell offers during correction of shell source
inspection or shell syntax advice for an unrelated provider schema error.
Keep the replacement action model-selected and reject the entire invalid group.

The regression fixtures cover a malformed `ls` response, a later mixed
inspection and build request, completion through a direct check, and rejection
when a model repeatedly ignores the narrowed offers after resume. The agent
did not run the fixtures or replay the original assignment.

```sh
venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/test_internal_coding_engine_workflow.py -k 'recovers_from_shell_inspection or rejects_shell_during_saved_inspection or corrects_shell_and_search_scope or preserves_patch_budget_after_evidence_rejection or rejects_repeated_oversized_groups'
```

## Windows CI paths, source bytes, and Batch wait

The Windows Python 3.12 CI run reported eight failures.

- The ToolRunner snapshot serialized repository-relative paths using the host
  separator. After a write to a nested file, AUTO rejected its own mutation
  receipt at the checkpoint path validator. Top-level files avoided the
  problem because their paths contain no separator. Snapshots now emit
  slash-separated paths. The gateway converts native separators in older local
  receipts before reconciling edits and moves. The checkout, path, and
  file-digest checks still apply.
  [ToolRunner snapshot](../../src/mana_agent/multi_agent/agents/tool_runner_agent.py)
  [Gateway reconciliation](../../src/mana_agent/gateway/mode_workflow.py)
- Python text-mode writes produced CRLF fixture bytes on Windows. Direct source
  excerpts preserved those line endings, and source stamps hashed the bytes on
  disk. The tests now write explicit bytes and cover both LF and CRLF for
  multiline citations and native read stamps.
- The Batch approval test listed durable task records while its approval worker
  was still checkpointing. The log showed a Windows `PermissionError` opening a
  task JSON file, consistent with a concurrent atomic replacement. The supervisor
  store now applies the existing bounded replacement retry schedule to task reads.
  Recovery still stops on persistent access errors or malformed task records.
  The Batch test waits for the approval worker to publish the wait before
  looking up and resuming the bound child step.
  [Supervisor store](../../src/mana_agent/execution_supervisor/store.py)

**Do not reintroduce** host-native separators into mutation receipts, infer a
source stamp from displayed text, or skip unreadable task records during recovery.
Keep Batch wake assertions synchronized with the completion of the approval
worker's transition.

User verification on Windows. The agent did not run these commands.

```powershell
python -m pytest -q tests/coding/test_investigation_engines.py tests/coding/test_tool_runner_execution.py tests/test_execution_modes.py tests/test_internal_coding_engine_workflow.py -k 'multiline_code_citation or rename_evidence or reads_only_selected_lines or whole_files'
python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/execution_supervisor/test_supervisor_core.py -k 'edits_and_repairs_on_one_assignment or corrects_shell_and_search_scope_separately or legacy_mutation_paths or character_truncated_line or batch_proposal_wakes or retries_read_denial or retries_transient_replace_denial'
python -m pytest -q
```

## Checkout builds and repeated renewal

On November 20, two Internal AUTO assignments stopped at the checkpoint after
running builds in the assigned checkout. In the second attempt, the worker
recorded both source edits and build artifacts. When a generated file grew,
the bounded content snapshot no longer retained several other files. The
worker treated gaps in diff capture as unknown file versions, so the gateway
could neither checkpoint the work nor evaluate another build or renewal.

ToolRunner compares content digests independently of the bytes held for
diffs. If a file genuinely cannot be observed, the checkpoint identifies the
uncertain path. Existing uncertain receipts still require review.
[ToolRunner](../../src/mana_agent/multi_agent/agents/tool_runner_agent.py)

For realtime Internal write assignments, `verify_coding_change` runs the
model-selected build or check and optional prerequisites in the assigned
checkout through the transactional shell gate. Git-ignored build output stays
available for incremental builds. The source receipt includes tracked paths
and non-ignored candidates. Checks bind to the observed source versions.
[Gateway workflow](../../src/mana_agent/gateway/mode_workflow.py)

The model can repair and rerun a failed build on the same assignment. The
default realtime renewal policy uses accounting capacity and observed
progress instead of a fixed renewal count. An explicit configuration cap
still applies. The same failed build against the same source and setup
registers as repeated work even if timestamps or other output vary. Coding
Activity reports the condition before the no-progress stop.

**Do not reintroduce** dependence on diff capture for mutation certainty, a
fixed default renewal count for progressing realtime repair, or treating
identical failed builds as progress because their output text changed.

The agent did not run these checks or replay either cancelled assignment.

```sh
python -m pytest -q tests/coding/test_tool_runner_execution.py -k 'digests_survive or unobserved_file_versions'
python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py -k 'adaptive_checkout_build or adaptive_coding_renews or model_wait_failure'
```

### Ignored build output at command checkpoints

At 00:11 Tehran time on November 21, the Internal AUTO coding assignment
`task_20261121_000004` passed its build and all four CTest tests. The shell then
created the requested branch and committed `sections.c` as `1612a2a`. Its
receipt contained an unknown edit for `os_signatures.c` with both digests
empty. The gateway blocked while saving the checkpoint. Git showed the
commit changed only `sections.c`, and the coding task was later cancelled.

The checkout verification path used Git's source list. The shell path walked
the whole checkout and spent most of its 100 MiB aggregate hashing allowance
on ignored build output. Applying that selection to the reviewed checkout
left roughly 2.4 MiB when it reached the unchanged 4.4 MiB source file. This
produced an unknown receipt even though the shell had exited successfully.
The earlier regression covered exhausted diff retention and genuinely
unavailable digests. It did not cover a Git shell command after a build.

ToolRunner now uses the same tracked and non-ignored source selection for
mutating Git checkout commands. Previously observed files remain candidates
for the post-command snapshot when the command changes Git's index or ignore
rules. Native snapshots stream digests under their cancellation and deadline
checks. The capture limit still bounds retained diff data. An unreadable file
or interrupted observation still yields uncertain evidence. The gateway
continues to require reconciliation of preexisting uncertain receipts.
[ToolRunner](../../src/mana_agent/multi_agent/agents/tool_runner_agent.py)

**Do not reintroduce** an aggregate byte allowance that makes a readable
file appear unobserved in a running native command. Keep ignored build
output out of implicit Git source snapshots and require evidence for uncertain
side effects.

User verification covers the Git command, digest retention, unreadable files,
interrupted observations, discovery, and the surrounding adaptive workflow.
The agent did not run these tests or replay the cancelled task.

```sh
venv/bin/python -m pytest -q tests/coding/test_tool_runner_execution.py tests/test_intent_driven_discovery.py -k 'git_command_hashes_source or digests_survive or unobserved_file_versions or cancelled_post_command_snapshot or plain_shell_operation'
venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py -k 'adaptive_checkout_build or adaptive_coding_edits_and_repairs'
```

### Failed checkout checks during successful repairs

After checkout verification became a supervised command, AUTO and DEBUG could
repair a failed check and later pass it, yet still stop at the supervisor's
completion gate. The runner recorded the earlier check as a failed action even
when its verification receipt contained a known exit and the source snapshot
showed no mutation. The build renewal fixture also intercepted activity at the
wrong module, so it failed before exercising the workflow.

The runner marks a failed checkout check as reconciled after it records a
normal exit for the command or its prerequisite and observes no changes across
complete source snapshots. The check still reports failure to the model and
the gateway still requires passing verification against the current files.
Interrupted commands, incomplete source observations, and failed checks that
change source still block completion. Historical action receipts retain their
recorded state. The fixture listens at the activity function used by Coding.

**Do not reintroduce** treating an observed negative check as an unresolved
side effect when its source is unchanged. Keep unresolved actions blocking
completion when their outcome or source effects are uncertain.

```sh
venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py
```

## Provider rejected shell arguments

The provider returned a shell tool call that failed the closed argument schema
before any shell action ran. Internal Coding stopped after completing two source
reads. The rejected inputs were redacted. The registered schema omitted
`cwd`, which the execution handler already supported. The loop offered a
correction for an unavailable tool name and had no correction path for schema
rejection.

The shell schema now declares its repository-relative working directory.
The provider validator gives Coding a typed error for rejected arguments.
On the same assignment, Coding gives the model one opportunity to select
valid calls, using remaining model turns and retaining completed observations.
**Do not reintroduce** dispatch of an invalid group or sanitize rejected input
into an inferred command.

The agent did not run these checks.

```sh
python -m pytest -q tests/llm/test_engine.py -k 'native_shell_schema or unregistered_and_invalid_tool_calls'
python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py -k preserves_patch_budget_after_evidence_rejection
```

## Workspace changes after a tool receipt

On November 25, 2026, a realtime Internal coding task in `mana-agent-test`
stopped at its checkpoint with `A changed file no longer matches its tool
receipt`. The saved shell receipts from task `task_20261125_000002` contain
6,634 mutations from the initial dependency setup before `.gitignore` was
created. After it was added, the later install commands completed and Git
excluded dependency directories from their source snapshots. The only saved
digest still differing from the checkout during investigation was
`frontend/node_modules/.package-lock.json`. The top-level frontend lockfile
was a separate, non-ignored source candidate.

Checkpointing treated an earlier command's artifact receipts as permanent
source inputs. It also converted a mismatch on a genuine source path into an
exception before the model could read the new version. The recovery and
verification gates used the same assumed post-tool hashes. Existing tests
covered repeated checks and provider interruption with stable files.

The gateway now reconciles command receipts with the checkout's current Git
ignore rules and keeps tracked and explicitly edited paths. It records the
current versions for verification and reports only edits still supported by
their receipts. Internal Coding keeps completed operations and continues on
the same assignment with refreshed observations and bounded model turns. If a
file changes during inference, the runtime drops the superseded selection
before dispatch. A checkpoint resume makes the same source comparison using
the saved operations. Older receipts remain intact. Verification requires a
pass on current inputs, and ambiguous operations still require reconciliation
under the supervisor's existing action rules.

**Do not reintroduce** a terminal error solely because a current workspace
file differs from a settled tool receipt. Preserve the receipt, refresh the
affected observations, and require current verification before completion.

The agent did not run the regression tests.

```sh
venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py -k 'filters_ignored_command_artifacts or reconciles_workspace_drift or resumes_after_a_model_wait_failure'
```

### Repeating a completed checkout check after source changes

The user ran the workspace-change regressions on November 25, 2026. The
artifact and checkpoint-resume cases passed. The same-attempt source-change
case failed when Coding selected the same `verify_coding_change` command
against the new file contents. The supervisor rejected the selection as a
matching consequential action before the second check could execute.

The supervisor already permits a fresh call for a completed local command.
Its effect-scope mapping omitted `verify_coding_change`, so the first check
had an `UNKNOWN` action scope even after a successful exit. The action model's
assignment validator also wrote a string into the enum field when updating
an unknown action, which produced the serializer warning in the same run.

The action model now infers checkout verification as a local repository
command, including when loading a legacy receipt with unknown scope. It
preserves enum values during assignment. A fresh selection receives its own
action receipt after a settled check, and current-source verification remains
required. The supervisor still fences a matching action with an unsettled
outcome. The existing gateway regression exercises two runs of the same
command on one assignment. The supervisor regressions cover completed,
legacy, and unsettled action records.

**Do not reintroduce** unknown effect scope for checkout verification or reuse
a prior pass when the current source hashes differ. Preserve duplicate-action
fencing until the previous command's outcome has been reconciled.

The agent did not run these checks.

```sh
venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py -k 'filters_ignored_command_artifacts or reconciles_workspace_drift or resumes_after_a_model_wait_failure'
venv/bin/python -m pytest -q tests/execution_supervisor/test_supervisor_core.py -k 'run_command_action_can_reexecute or existing_unknown_effect_scope or checkout_verification_repeat_fences or unknown_action_scope_fails_closed'
```

## Verification

Run the focused regressions first.

```sh
venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/test_internal_coding_engine_workflow.py -k 'adaptive_coding or whole_files or draft_waits_for_exact_approval'
```

The broader regression suite adds Evidence Board ingestion, recovery, the
coding toolbox, activity rendering, entry routing, and Batch compatibility.

```sh
venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/test_internal_coding_engine_workflow.py tests/coding/test_investigation_engines.py tests/test_coding_tool_system.py tests/execution/test_coding_research_recovery.py tests/test_operation_rows.py tests/gateway/test_entry_routing.py tests/test_openai_batch_coding_engine.py tests/gateway/test_coding_lifecycle_observability.py
node --check src/mana_agent/dashboard/components/live_chat.js
```

The gateway fixtures provide deterministic model actions. Verification
requires Docker and an appropriate local image in a live session.
