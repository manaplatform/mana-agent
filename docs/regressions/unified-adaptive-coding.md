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
