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
- Verification runs against an isolated source snapshot. A failed check
  returns to Coding. Completion validates observed passing checks against the
  latest changed-file receipts. A model assertion of success is insufficient.
- Recovery reconciles tool operation identities with supervisor action receipts
  before continuing. A revoked lease or an uncertain operation stops execution.
  Previously completed writes retain their verified source hashes.

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
