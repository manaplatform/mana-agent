# Debug and Plan execution contracts

## Automatic coding workflows

The entry router could activate an approval workflow from ordinary prose, and
the CLI and TUI could interpret the same text independently and send it to
different executors. A regular coding turn could run the backend before saving
an accepted plan. The workflow parent and its coding child also cannot both
acquire conflicting repository locks. A public gateway test exposed the
parent waiting indefinitely for its own child.

The gateway now validates one AUTO or DEBUG decision for a coding turn. The
controller advances scoped decisions and the TaskBoard retains the plan,
evidence, mode transitions, and completed steps. The supervised parent
coordinates the workflow. Scoped coding children acquire the patch leases
and record their receipts. Separate verification records let the parent
repair a failed check and rerun it without repeating a completed patch.
Only explicit PLAN commands open approval of a revision.
On resume, an old checkpoint that proves core implementation completed can
continue to feature integration without applying the core changes again.
Other legacy coding attempts require reconciliation before they can enter the
staged workflow.

Codex app-server patch proposals also use a private result field. The ordinary
assistant summary cleanup collapses repeated spaces, which can change
indentation inside a patch encoded in JSON. The proposal parser validates the
envelope and forwards its original patch text to the scoped review and
mutation gate. Invalid proposals block the step. The proposal field is excluded
from ordinary result serialization.

**Do not reintroduce** text-based mode selection, an inferred PLAN entry,
parent write locks that conflict with child leases, or prose cleanup on exact
patch content. Passing verification must come from executed checks, with
coordinator authority for runtime integration.

Run `venv/bin/python -m pytest -q tests/gateway/test_entry_routing.py tests/gateway/test_debug_plan_mode_contract.py tests/gateway/test_chat_gateway.py tests/gateway/test_codex_session_lifecycle.py tests/test_codex_integration.py` for routing, supervised execution, legacy checkpoint recovery and Codex protocol fixtures. Run `node --test tests/dashboard/live_chat_reducer.test.mjs` for the dashboard activity reducer.

The full CLI smoke suite still contains fixtures for the removed console
auto-execute pipeline and for its response rendering and flow state. Review
those failures with `venv/bin/python -m pytest -q tests/test_cli_smoke.py`
before using the full suite as a release check. Keep the gateway as the
entry point when migrating those fixtures.

## Symptom and cause

A bug report in ordinary chat had no dedicated evidence-first coding route. The Plan service added prompt keywords to repeated repository-wide globs and searches. Missing or invalid model output could produce a generic implementation template. The console, TUI, and standalone plan command also had different clarification and approval paths. Approval was stored on a session object, and the coding handoff carried only the original goal. The execution controller could therefore replan the work without the steps or boundaries that the user reviewed.

To reproduce the old behavior, use `/plan` for an ambiguous repository change, then approve it and compare the coding request with the reviewed steps. A separate console planning flag asked a fixed number of questions even for a clear request. The previous Plan workflow could also start through the root CLI's generic repository preparation path, which initialized Git before the proposal if the directory had no repository.

## Invariants

- A main model decision selects a scoped repository read, clarification, plan draft, or debug repair. Explicit PLAN can use a native read-only session. Mana asks for one correction on an invalid main decision using the same model and response format. Debug starts with reported evidence and must cite a narrower observation before widening the scope.
- A reviewable plan has ordered steps, exact path operations, explicit verification, and a revision digest. Only the human inbox can authorize that digest. Actor, task, revision, workspace, and supervisor lease must match before mutation.
- A coding engine proposes a patch in read-only mode. Mana reviews the patch for scope, applies it through the existing action policy, and records the observed result. The device's general approval mode cannot supply plan approval.
- Cancellation or a material revision invalidates the authority to start another patch. An incomplete mutation receipt prevents replay. A retry of a failed check may reuse a previously verified patch without repeating it.
- Existing session-only plans or approval flags have no execution authority. Recreate them through the current Plan entry points.

**Do not reintroduce** keyword-based plan execution, unconditional repository scans before debugging, or a goal-only handoff from approval to a writable coding engine.

## Decision validation regression

An action such as `draft` without a `draft` payload satisfied the provider's JSON Schema but failed the Pydantic mode contract. PlanService performed that semantic validation after the native engine's correction loop, then reduced the result to `Invalid or unavailable ModeDecision model decision (ValidationError)`. Tool-based decisions also passed the synthetic tool as a raw request option, which the native request boundary rejects. Strict Python validation could reject JSON arrays representing typed tuples.

Pass the typed contract through inference so semantic validation participates in the one correction attempt. Use strict JSON validation for all response formats and register the synthetic tool through the native binding interface. Report only schema fields and safe checks. Recovery reopens the blocked TaskBoard item before the new attempt, so it can wait for inbox approval again. A blocked investigation resumes with its existing evidence and still requires approval to execute.

**Do not reintroduce** a second, unrecoverable validation pass outside the engine, raw rejected model payloads in errors or correction prompts, or fallback to a different decision format after rejection.

## Provider decoding regression

The OpenAI-compatible adapter represents open-key objects in strict schemas as JSON-encoded strings. `ModeDecision` used that representation for file operations, clarification choices, and debugging report details. A draft with an encoded `{"app.py":"update"}` passes the provider schema because it is a string. After decoding, the application requires a list of operations and raises `The provider response did not match the application schema after decoding.` The correction received the same message without the shape of the rejected field. Earlier fixtures returned application objects directly and skipped the adapter that exposed the mismatch.

Mode decision schemas now advertise lists of typed records. A file record requires a repository-relative path and a nonempty list of permitted operations. A verification-only step uses an empty list. The application rejects duplicate record keys and converts valid records to the existing dictionary format before saving state. Older saved plans still load with the same approval digest. The separate coding backend debugging envelope retains its dictionary fields.

**Do not reintroduce** JSON strings for typed Mode decision fields or silently collapse repeated file paths and choice values into one entry.

## File-read evidence regression

Plan and Debug could stop with `Invalid ModeDecision model decision` after both responses selected `read_file` with multiple paths, no paths, or level 4. The error grouped path count and level, so the reported failure alone does not identify which field was invalid. The schema exposed the same path list and levels for every tool. A decision could therefore satisfy the provider schema and still fail the application's file-read check. Earlier workflow fixtures supplied valid narrow reads and skipped this mismatch.

The native evidence tools have separate schemas for file reads and discovery. A file read requires exactly one repository-relative path and a level from 1 to 3. Level 4 remains available to discovery tools under the expansion and audit rules. The model can select several independent operations in one response. Each operation has its own evidence ID and fingerprint. The main agent grants the initial audit scope when the user requests it.

**Do not reintroduce** a shared evidence schema that allows level 4 or zero or multiple paths for a file read. Keep wider discovery and exact reads as separate model selections.

## Decision activity and provider stops

A Debug turn appeared to stall at “Task started” and eventually displayed `Invalid or unavailable ModeDecision model decision (InvalidResponseError)`. The conversation journal recorded two model requests and zero tool starts. The first model response arrived after about 33 seconds. The second request failed after about 108 seconds and used 26,214 output tokens, matching the configured output limit inferred from the recorded model and budget. The journal retained the error class and usage, and the provider's stop reason could not be recovered.

The mode workflow used the frontend turn ID for the supervisor's lifecycle events, then switched the model and tool activity to the task ID when it established the attempt scope. The frontend consequently missed progress on the visible turn. Also, PlanService replaced invalid-response diagnostics with the exception type. Existing workflow fixtures advanced scripted decisions and did not assert that lifecycle, model, and tool events appeared together in the turn journal.

The mode workflow now keeps the frontend turn ID and redaction context for activity within the same session. Background work can recover the originating turn ID from the supervisor record. Task ID, attempt generation, and lease checks continue to fence publication. Chat Completions token-limit and filter stops, plus explicit refusals and empty responses, can carry application-authored detail into the failed model event and the mode result. Unknown failures get a generic diagnostic. Provider stops still end the decision without selecting another action. There is no migration for existing tasks.

**Do not reintroduce** substituting the task ID for an available frontend turn ID, publishing provider response text as failure detail, or automatically retrying a provider stop by increasing its budget or changing models.

## OpenRouter Claude decision rejection

On 2026-11-16, a DEBUG turn using `anthropic/claude-haiku-5.5` through OpenRouter started and failed its first mode model request before returning a decision. The turn displayed `Invalid or unavailable ModeDecision model decision (ProviderRequestError)`. The journal recorded the selected model and failed operation but kept no HTTP status. The exact rejection from that turn is unknown.

The OpenRouter Chat Completions adapter sent Claude constraints such as `maxLength` on investigation objectives and `maxItems` in research reports. Claude's structured-output API documents those keywords as unsupported. The adapter now expresses these bounds in provider-facing field descriptions for OpenRouter's Anthropic model namespace. It retains the action discriminator values and required fields. Mana validates the response against the original bounds and typed decision before it can dispatch an action. The provider fixtures previously tested decoding and a simple schema without Claude's unsupported bounds. [Claude structured-output schema limitations](https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations)

Native inference had also emitted the failure activity before classifying the HTTP status. PlanService then collapsed the typed failure into its exception name. Buffered and streaming requests now report the provider and HTTP status in the same model activity and mode result, with the failure code in activity metadata. The classification uses the status alone. HTTP rejections stop the decision without a correction or automatic replay. Existing saved tasks and plans require no migration.

**Do not reintroduce** Claude-incompatible bounds in its provider-facing strict schema, validation against the provider schema alone, or raw provider messages in the public activity.

Run `venv/bin/python -m pytest -q tests/llm/test_providers.py::test_openrouter_claude_mode_schema_preserves_local_constraints tests/llm/test_engine.py::test_mode_decision_reports_provider_http_failure_without_replaying`. The fixtures cover the OpenRouter wire schema, rejection of an overlong decision, buffered and streaming HTTP failures, redaction, and request count. These checks were not run by the agent. To verify live provider compatibility, restart Mana-Agent and use the same model for a read-only DEBUG request such as explaining the return value of a local function.

## Direct Anthropic decisions

A DEBUG turn still received HTTP 400 with `openrouter/anthropic/claude-haiku-5.5` after the OpenRouter schema change. The local model settings selected OpenRouter explicitly. To send decisions to Anthropic's own Messages API, select **Anthropic** and the upstream model `claude-haiku-5-5` in Mana's provider configuration. That model is stored as `anthropic/claude-haiku-5-5` in the primary model setting. A new turn can use that assignment after restarting Mana-Agent. Existing workflow tasks retain their saved provider and model on resume.

The native Messages adapter previously forwarded the original application schema verbatim. It included string and array bounds Claude documents as unsupported. The adapter now uses the same compatible schema encoding as OpenRouter's Claude request path. It sends the format in `output_config.format`, retains the action-specific required fields and discriminators, and decodes and validates the response against the original schema before dispatch. The stored plan and approval formats require no migration. [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations)

**Do not reintroduce** a raw application schema in the native Messages request or a provider switch after a failed decision.

Run `venv/bin/python -m pytest -q tests/llm/test_engine.py::test_anthropic_mode_decision_uses_messages_and_preserves_validation tests/llm/test_providers.py`. The direct transport fixture checks the upstream model ID, the native Claude request format, and one same-provider correction when a response exceeds the original length bound. Tests and live provider acceptance were not run by the agent.

## OpenRouter Messages for Claude

With `MANA_LLM_API_MODE = "auto"` and the active assignment `openrouter/anthropic/claude-haiku-5.5`, a model request still failed with HTTP 400. The native Anthropic schema change applied to the direct Anthropic adapter. The OpenRouter engine still used Chat Completions because automatic protocol selection considered only the provider. The original 400 did not retain the provider's detailed rejection reason.

Automatic protocol selection now sends OpenRouter's `anthropic/…` model IDs to its Messages endpoint. It keeps the OpenRouter credential, catalog base, exact model ID, and configured provider preferences. Capability lookup uses the same protocol as execution. The Messages adapter sends the schema in `output_config.format` and Mana validates the result against the original contract. An explicit API mode still controls OpenRouter's transport. Restart Mana-Agent to load the new code. Existing tasks keep their assigned provider and model, and continuation receipts remain bound to their original protocol. HTTP rejections end the request without switching API protocols. [OpenRouter Claude integration](https://openrouter.ai/docs/guides/coding-agents/claude-code-integration#how-it-works)

**Do not reintroduce** provider-only protocol selection for OpenRouter Claude models in automatic mode or use direct Anthropic credentials on an OpenRouter request.

Run `venv/bin/python -m pytest -q tests/llm/test_engine.py -k 'openrouter_claude_uses_messages or openrouter_model_protocol'`. The fixtures use the SDK with an in-memory HTTP transport to check the exact URL, bearer authentication, decision tool, and one request on HTTP 400. They also check capability lookup and explicit mode selection. These checks and a live provider request were not run by the agent.

## OpenRouter Claude decision tools

On 2026-11-16 the Sonnet 5.5 journal showed two HTTP 400 responses on `anthropic_messages`. The entry router's request with its structured schema failed, the subsequent request without a schema succeeded, and the first `ModeDecision` request failed before any repository tool ran. Both failed calls requested composite schemas. The trace lacked the HTTP response bodies needed to identify which upstream validation rule rejected them.

OpenRouter's Messages path now uses a single ordinary `record_decision` client tool for `EntryRoutingOutput` and `ModeDecision`. Claude receives each original action schema as the tool input schema without compiling it as a constrained response grammar. Mana requires exactly one call, validates it against the original schema and Pydantic contract, then applies the usual routing, evidence, and authorization checks. The tool only returns data. A malformed response can get one same-format correction. Native routing propagates HTTP failures and skips the legacy plain-text replay. The selected provider, model, validated actions, and saved approvals retain their existing identities. [Claude strict tool use and grammar validation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)

**Do not reintroduce** the native response grammar for these two OpenRouter Claude contracts, or silently resend a failed native entry decision as plain text.

Run `venv/bin/python -m pytest -q tests/llm/test_engine.py tests/gateway/test_entry_routing.py`. The SDK fixture covers both decision contracts with a valid tool response and an HTTP rejection. The fixture and live provider acceptance were not run by the agent.

## Entry routing tool correction

At 23:21 on 2026-11-18 the first entry-routing response omitted
`route`, `confidence`, and `required_sources` from `record_decision`.
The existing correction produced a valid coding route before TaskBoard
started the task. The journal logged the rejected response as a failed
model request, even though the provider returned successfully. It does
not retain the rejected response body.

The entry prompt called for JSON-only output, while the selected tool
transport asked the model to put the decision in a function call. The
prompt now follows the configured transport and states the required
fields. The tool submission instruction names the required top-level
arguments from its schema. The full typed decision still gates routing.

When the existing correction is available, activity marks a provider
response that fails schema validation as `Decision needs correction`.
A second invalid response or a provider request failure still appears as
a failure. The correction budget, transport, and provider remain the
same.

Do not reintroduce conflicting format instructions or synthesize missing
routing fields.

```sh
venv/bin/python -m pytest -q tests/llm/test_engine.py tests/test_tui_approval_realtime.py -k 'entry_route_reports_schema_rejection or rejected_decision_and_successful_correction'
```

The agent did not run the tests or replay the provider request.

## Action payload regression

Plan or Debug could block after the model's correction with a generic list of action requirements. For example, a response that selected `draft` with a null draft satisfied the provider schema and then failed the application validator. The original schema described the action and every optional payload independently. The earlier correction fix gave the model another attempt, while the schema still allowed invalid combinations on both calls. The reported diagnostic alone gives no evidence about which combination the live model selected.

The response object now contains one `decision` with a schema branch selected by its `action`. The `inspect` branch requires evidence, `clarify` requires one or more questions, `draft` and `repair` require a plan draft, and `research` requires a debugging report. Fields from other branches are rejected. The wire schema uses `anyOf` inside a root object for the supported structured-output transports. Local validation selects the branch by its action tag and still enforces semantic checks within the existing one-attempt correction. The gateway continues to validate the active mode, observed evidence references, and approval before execution.

Only transient model decisions use the new response shape. Saved revisions, evidence, and approval digests use their existing formats.

**Do not reintroduce** independent nullable action payloads in the model-facing schema or infer a missing action or payload from the rest of the response.

## Inspection path correction

On 2026-11-17, AUTO task `task_20261117_000001` saved three successful observations, then blocked after two OpenRouter Claude decision responses. The error named the second request in the new inspection batch and mixed `literal error`, `too short`, and repository-path failures. TaskBoard retained `evidence-1`, `evidence-2`, and `evidence-3`. The journal did not retain the rejected path or identify the selected tool.

The local inspection contract tried every read-tool schema for an invalid request. The diagnostic reported the first four failures across those schemas, so errors from unselected tools could hide the relevant path error from both the model and the user. Each request now selects its local validator by its `tool` value. The provider still receives an `anyOf` with the same tool alternatives. The decision prompt explains which path syntax to use and how to proceed when only protected or out-of-checkout evidence would answer the question. Correction details show the selected tool and field, give a safe next step, and omit rejected values. The same model gets one correction. An invalid batch dispatches no reads, while observations from earlier decisions stay available to the same task on resume. No data migration is needed.

**Do not reintroduce** validation against unselected tool branches or echo rejected path values. Keep protected-path and checkout-confinement checks in place even during correction.

Run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/llm/test_providers.py -k 'inspection_path_correction or investigation_batch_rejects or mode_decision_wire_schema'`. The fixtures cover same-transport correction, rejection of protected paths before dispatch, redacted diagnostics, and resumption with saved evidence. The agent did not run the fixtures or retry the live task.

## Repeated invalid search paths

On 2026-11-17, DEBUG task `task_20261117_000004` using OpenRouter DeepSeek returned an invalid initial decision and one invalid correction. The final diagnostic identified four `repo_search` requests with invalid `paths[0]` entries. The task had no repository observations and was later cancelled. The journal did not retain the rejected values, so it cannot establish whether the paths contained wildcards, absolute paths, or another invalid form.

The discovery wire schema previously accepted arbitrary strings as path entries, and the correction asked the model to fix the response without showing how to build a valid replacement batch. The discovery schema now rejects empty strings, glob metacharacters, drive separators, and control characters in `paths` before tool dispatch. The existing path and scope validation still checks relative paths, traversal, protected paths, and level limits. The same-model correction now explains that all reads in the rejected batch were skipped and shows how to request a shallow layout listing or send a targeted `glob` with empty `paths`. The model can select fewer reads or use saved evidence to make another decision.

**Do not reintroduce** accepting patterns as literal paths, moving rejected path data into `glob` automatically, or dispatching valid requests from an invalid batch.

Run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/llm/test_providers.py -k 'inspection_correction_rebuilds or inspection_path_correction or investigation_batch_rejects or native_investigation_tool_schema or mode_decision_wire_schema'`. The regression uses a scripted batch with four invalid path entries followed by a model-selected layout query and a scoped search. The fixtures and live provider behavior were not verified by the agent.

## Directory listing decision schema

On 2026-11-17, AUTO task `task_20261117_000005` using OpenRouter DeepSeek blocked after the correction returned a level-1 `ls` request that failed its directory listing contract. The journal shows zero reads and does not retain the rejected arguments. Setting a search field such as `glob` to an alternate value reproduces the validation failure.

The `ls` branch inherited path, query, glob, and line-range fields from the generic evidence request. The ModeDecision schema advertised those fields even though directory listing could only use their defaults. Native PLAN already stripped the extra fields from its registered tool schema, which let the two schemas drift. The shared model-facing listing schema now exposes only `tool`, `level`, and `reason`. PLAN derives its existing arguments from the same schema. The correction prompt includes an exact listing request and points to `list_files` or `repo_search` for scoped reads. Local request defaults and saved evidence fingerprints are preserved.

**Do not reintroduce** search or line-range fields in the model-facing `ls` schema. Keep its traversal limited to immediate repository directories.

Run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/llm/test_providers.py -k 'inspection_correction_rebuilds or debug_can_find_an_error_with_no_reported_location or mode_decision_wire_schema or native_investigation_tool_schema'`. The fixtures cover strict schema decoding, rejection of extra arguments, fingerprint compatibility, and same-model correction followed by scoped discovery. The agent did not run the fixtures or resume the live task.

## Duplicate inspection correction

On 2026-11-17, DEBUG task `task_20261117_000006` using OpenRouter DeepSeek blocked on its first inspection decision with “Choose distinct inspection operations that have not already run.” TaskBoard recorded zero repository reads and zero observations, so the duplicate was within the proposed batch. The journal did not retain the rejected operations.

The response passed typed decision validation. The gateway checked for duplicate requests only while preparing dispatch, after the model's correction window had closed. AUTO and DEBUG now validate inspection scope and read fingerprints against a snapshot of current TaskBoard evidence during decision validation. A duplicate within a batch identifies both request positions. A repeat of a saved read identifies its evidence ID. The same model gets the existing bounded correction and can select a different batch or another allowed action. The gateway still revalidates scope and fingerprints immediately before dispatch. Saved evidence retains its existing fingerprint format.

**Do not reintroduce** dispatch-only duplicate validation for inspection decisions. Preserve all-or-nothing validation and require a corrected model decision before any read from the rejected batch can be selected again.

Run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py -k 'duplicate_inspection or deduplicates_aliases or investigation_batch_rejects or over_budget_read_batch or unjustified_expansion'`. The fixtures cover correction of same-batch and saved-read duplicates, normalized aliases, and blocking after a repeated invalid selection. The agent did not run the fixtures or resume the live task.

## Saved evidence in duplicate corrections

On 2026-11-17, DEBUG task `task_20261117_000007` saved 11 observations and then blocked after its model requested the same `ls` operation twice, including during correction. TaskBoard had the layout under `evidence-1`. The model prompt included evidence IDs for the full history but only showed the latest eight observations. The correction identified `evidence-1` without supplying its result.

The workflow keeps the shallow layout alongside recent observations. The same model's single correction attempt receives the matching saved observation as data. The public validation message contains the diagnostic and evidence ID. The correction uses the existing observation compaction, and the layout preview includes at most 50 directory names. TaskBoard retains the complete result with its original read receipt.

**Do not reintroduce** telling the model to use an evicted observation without supplying it. Keep rejecting duplicates within a batch and keep saved observation contents out of public errors.

Run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/llm/test_engine.py -k 'duplicate_inspection or compact_observation_bounds_saved_layout or mode_decision_repairs_semantic_failure or inspection_correction_rebuilds'`. The fixtures exercise retention of an older layout, recovery of an older failed read for correction, and blocking if the model repeats that failed read. The agent did not run these fixtures or resume the live task.

## Reuse of saved inspection results

On 2026-11-17, DEBUG task `task_20261117_000008` received the duplicate error after four successful repository reads. `evidence-1` held the shallow directory listing and was within the recent evidence window. The corrected decision repeated that listing. The journal did not retain the first rejected decision or the model's prompt.

AUTO and DEBUG now resolve a model-selected exact match to the original successful result from the same task. The gateway validates the complete inspection batch, including the selected tool, arguments, and scope. It dispatches only the new reads and returns the saved evidence ID and observation to the next decision. Reuse preserves the original receipt, level, and evidence ID. A reuse-only batch counts toward the existing no-progress limit, after which the next decision uses the evidence already collected. Duplicates within a batch, failed records, and direct PLAN investigator batches retain their rejection checks.

**Do not reintroduce** treating an exact successful match to saved inspection evidence as a fatal AUTO or DEBUG error. Do not execute the matched read again, create a second observation for it, or count reuse as new evidence.

Run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py -k 'reuses_saved_layout or repeated_repository_results or duplicate_inspection or over_budget_read_batch'`. The fixtures cover a saved listing plus a new read under a one-call remaining budget, reuse-only loops, and rejection and correction of invalid duplicates. The agent did not run the fixtures or resume the live task.

## Text search query contract

On 2026-11-17, DEBUG task `task_20261117_000009` using OpenRouter DeepSeek saved 96 observations before decision validation rejected the second request in a proposed batch. The corrected `repo_search` request lacked a nonblank query. The journal retained the validation failure and the saved observations. It did not retain the rejected argument.

`repo_search` inherited the optional query used by other discovery operations. The model could return a missing, empty, or whitespace-only query that satisfied the provider schema and failed only during local semantic validation. Text search now has its own decision branch, shared with the native PLAN tool, with a required string containing a non-whitespace character. Query text retains its original whitespace and existing read fingerprints. File listings and symbol queries retain their existing behavior. The correction prompt explains the difference between finding literal text and listing files by path or glob.

**Do not reintroduce** an optional or blank query in the model-facing `repo_search` schema. Validate the complete batch before dispatching any read.

Run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/llm/test_providers.py -k 'repo_search_requires_nonblank_query or inspection_correction_rebuilds or native_investigation_tool_schema or mode_decision_wire_schema'`. The fixtures cover schema and runtime validation, query preservation, query-free discovery, and same-model correction of a batch containing a valid read and a blank search. The agent did not run the fixtures or resume the live task.

## Verification result replies

On 2026-11-17, AUTO task `task_20261117_000002` applied its version change to `pyproject.toml` and saved a patch receipt. Its first verification command could not start because the local Docker check could not find `python:3.12`. The blocked reply printed the raw command representation, eight evidence IDs, and the pre-edit research findings. The reply omitted the file already changed. Approval and clarification waits also appended the full plan when reading back a blocked or completed result.

The Docker preflight used the generic exception path, and the chat reply appended the repository investigation report to every blocker. The preflight now reports a pending verification prerequisite and closes the matching Coding Activity entry with the command marked as unrun. The chat reply shows any files already changed, the verification status, and a resume command for the same task. Failed checks retain their exit status and output in Coding Activity and result data. Completed changes report verified work and the worktree when applicable. Status reads return the same outcome and structured evidence. Resume still checks the saved checkout and patch receipts before continuing.

**Do not reintroduce** pre-edit findings, evidence ID dumps, raw command arguments, or the full approved plan into a terminal status reply. Keep an unrun command out of the check results and preserve the verified completion gate.

Run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py -k 'missing_verification_image or resuming_failed_verification or auto_records_internal_plan or auto_uses_the_required_worktree'`. The missing-image fixture stops before Docker starts a container, then resumes using the original patch receipt. These tests and the live task were not run by the agent.

## Investigation latency

On 2026-11-16, the Debug trace for `task_20261116_000021` recorded 18 completed model requests taking 794 seconds in total. The 18 repository operations took under one second total. Three consecutive reads of known files required three provider round trips. Batching independent reads helped, but each dependent read still required the main model to select another typed action. Earlier prompts also repeated execution receipts alongside the observations.

The native investigator then introduced its own eight-turn and 24-operation allowance. When the model asked for a read on its final allowed turn, the gateway rejected the batch and blocked the workflow with “The native investigation reached its model turn limit before reporting findings.” The configured workflow budget could still have room for coding. Earlier tests asserted the block instead of checking that Coding could use the evidence already collected.

AUTO and DEBUG now select scoped read batches as ordinary Coding decisions. Each batch is validated against prior evidence before dispatch. Each tool charge and observation is saved on the active task. Repeated or empty results do not reset the no-progress guard. The next decision can draft or repair as soon as the current evidence supports the scope. The selected coding backend can read additional files within that scope while proposing the patch. A task blocked by a real limit returns its saved evidence and any known gaps.

The supervisor persists that the Coding task coordinates the workflow. Its implementation and verification steps still hold their own locks. A synchronous step reuses the parent's worker slot, so two running Coding tasks can both start their steps at lane capacity. Explicit PLAN retains native tools and revision approvals. Its native calls also use the shared budget and return a partial report at a shared boundary. Existing tasks retain their evidence and counters, including allowance already charged for an interrupted legacy attempt.

**Do not reintroduce** a separate investigation quota, a report as a prerequisite to coding, or a coordinator lock or worker reservation that blocks its own step. Keep batching independent requests and exclude execution receipts from prompts.

User verification is `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/gateway/test_lane_coordinator.py tests/gateway/test_entry_routing.py tests/gateway/test_turn_budget_accounting.py tests/gateway/test_multi_task_orchestration.py tests/llm/test_engine.py`. These checks were not run by the agent.

## Clarification wait and resume

On 2026-11-16, `task_20261116_000023` displayed a clarification with six questions and an inbox command. The supervisor recorded `waiting_for_clarification` with a checkpoint and inbox item; the inbox item was delivered and had no response or resume claim. The task had paused. A later `tui_quit` canceled the task and the unanswered request. The item was assigned to the local Mana user ID, while the terminal's answer handlers used the OS username. The trace does not show a rejected submission for this item, but that identity mismatch would reject a reply.

The TUI also selected the latest Plan or Debug task to find the form, so an AUTO prompt or another task's prompt could be discarded. The Plan command reported waits as preparation failures. After a background resume, another clarification or plan review was published through the worker activity path even though waiting had already released the worker lease. Plain console chat printed the question list without collecting the answers. Finally, the inbox rejected custom answers for choice fields marked to allow free text.

Terminal chat now opens the request linked to the exact session, task, and inbox item. The console collects explicit answers before waiting for the result, and a subsequent prompt is projected from the supervisor's persisted wait. Local answers use the local Mana identity when it matches the reviewer, preserving legacy OS reviewer assignments and checks on explicit actors. Inbox validation accepts custom replies only for choice fields that allow them. An accepted response uses the existing inbox claim and resumes the same task with its accumulated evidence and answers. No persisted data migration is needed.

**Do not reintroduce** looking up a clarification by the newest task, using the OS username for a Mana-assigned reviewer, publishing a saved wait as worker activity after its lease expired, or treating question recommendations as answers.

User verification: `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py::test_repository_clarification_resumes_exact_wait_and_preserves_evidence tests/test_cli_smoke.py::test_chat_planning_mode_requires_revision_approval_after_clarification tests/test_chat_planning_mode.py tests/human_inbox/test_durable_inbox.py`. For live acceptance, trigger several related questions in AUTO or DEBUG, submit them in the TUI or console, and confirm the same task resumes or presents the next question. Decline to answer once and verify that the saved inbox item remains pending. These checks were not run by the agent.

## Clarifications caused by a shortened request

On 2026-11-16, DEBUG tasks `task_20261116_000024` through `task_20261116_000026` opened clarification waits after the router returned one-character objectives for a 1,636-character request. Two of the waits asked for the task description or expected result. TaskBoard retained the user's request, but the coding dispatch passed the router's objective as the request to the mode model. The decision schema required a nonblank objective, so it accepted those summaries. Gateway fixtures generally used a complete objective and did not expose the loss of context.

Coding dispatch now takes the original request from the same TaskBoard task and supplies the resolved reference and bounded conversation context separately for follow-ups. An older routed task recovers its original request on its next workflow decision. The workflow model uses that information alongside observations and saved answers. A validated `clarify` action opens the inbox only for a material gap that requires user input. Research may suggest a question for the workflow model to evaluate. Validation rejects blank questions and field IDs that already have answers. Outstanding inbox requests and approved plan digests retain their existing format.

**Do not reintroduce** replacing the original request with a routing summary or opening a repository clarification from an entry routing field or research continuation. Ask only for the information missing from the request, relevant conversation, and evidence.

User-owned verification: `venv/bin/python -m pytest -q tests/gateway/test_entry_routing.py tests/gateway/test_debug_plan_mode_contract.py tests/llm/test_engine.py`. Replay the original request with the same model and confirm DEBUG proceeds without asking for details already supplied. Automated checks and live replay were not run by the agent.

## Verification

Run `venv/bin/python -m pytest tests/test_execution_modes.py tests/test_chat_planning_mode.py tests/test_chat_command_dispatcher.py tests/test_cli_modes_skills.py tests/gateway/test_debug_plan_mode_contract.py tests/test_codex_integration.py -q`. The gateway tests use deterministic model responses and isolated inboxes. The Codex protocol test uses an app-server fixture. Live acceptance requires an authorized provider and a local Docker image with the target project's test dependencies.

To verify the validation repair, run `venv/bin/python -m pytest tests/llm/test_engine.py tests/gateway/test_debug_plan_mode_contract.py -q`. The inference tests exercise native, tool, and text responses with local fixtures and assert that no rejected values are surfaced.

Run `venv/bin/python -m pytest tests/llm/test_providers.py -k 'native_mode_decision or native_debug_report' -q` to exercise the real Responses and Chat Completions decoder against local fixtures. The tests verify that the schema rejects the old wire shape, accepts typed records, rejects duplicate paths, and preserves the approval digest. Live provider acceptance remains unverified.

For file-read evidence, run `venv/bin/python -m pytest -q tests/llm/test_providers.py tests/gateway/test_debug_plan_mode_contract.py -k 'investigation_tool_schema or investigation_batch or repository_audit'`. These fixtures cover rejection of invalid tool shapes before dispatch and a narrow read after an explicit repository audit. The agent has not run these checks or exercised a live provider. Restart Mana-Agent from the updated checkout and use `/task resume <task-id>` to exercise a blocked investigation.

For the activity and provider-stop regression, run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py::test_debug_mode_activity_keeps_the_frontend_turn tests/llm/test_engine.py::test_mode_decision_reports_why_the_provider_stopped_correction`. The first fixture checks that task, model, and tool events share the originating turn and retain redaction. The second checks the provider stop diagnostic and the two-request limit. The agent did not run these checks. Live terminal acceptance can use a local fixture that asks Debug to inspect a reported file.

For action payloads, run `venv/bin/python -m pytest -q tests/llm/test_providers.py tests/llm/test_engine.py tests/gateway/test_debug_plan_mode_contract.py tests/test_execution_modes.py`. The fixtures check the provider schema for each action, reject missing or unrelated payloads, exercise semantic correction using all three response strategies, and confirm a resumed draft still waits for approval. These checks and live provider acceptance remain pending.

For investigation latency, run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/llm/test_engine.py tests/llm/test_providers.py`. The fixtures cover dependent reads in one native session, invalid batches, bounded operation counts, cancellation, resume, and PLAN approval. Replay the same Debug request with the same model and checkout. Compare main decision count, provider calls, time to first evidence, and time to result. The agent did not run the tests or measure a live turn after the change.


## Repository Evidence Board task architecture

The repository-evidence implementation supersedes the embedded native investigator
and understanding/investigating phase descriptions above. AUTO, DEBUG, and PLAN
now materialize investigation and understanding as supervised TaskBoard tasks,
using the selected coding engine in read-only mode. The existing controller
advances decisions and the existing approval, scope, verification, and integration
gates remain authoritative. See [current contracts, migration, and user-owned
verification](repository-evidence-orchestration.md). Legacy observations without
source provenance remain diagnostic context rather than verified board entries.
