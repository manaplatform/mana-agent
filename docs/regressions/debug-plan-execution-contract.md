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

- A main model decision selects a repository investigation, clarification, plan draft, or debug repair. The investigation model selects observations through native read tools. Mana asks for one correction on an invalid main decision using the same model and response format. Debug starts with reported evidence and must cite a narrower observation before widening the investigation.
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

## Action payload regression

Plan or Debug could block after the model's correction with a generic list of action requirements. For example, a response that selected `draft` with a null draft satisfied the provider schema and then failed the application validator. The original schema described the action and every optional payload independently. The earlier correction fix gave the model another attempt, while the schema still allowed invalid combinations on both calls. The reported diagnostic alone gives no evidence about which combination the live model selected.

The response object now contains one `decision` with a schema branch selected by its `action`. The `inspect` branch requires evidence, `clarify` requires one or more questions, `draft` and `repair` require a plan draft, and `research` requires a debugging report. Fields from other branches are rejected. The wire schema uses `anyOf` inside a root object for the supported structured-output transports. Local validation selects the branch by its action tag and still enforces semantic checks within the existing one-attempt correction. The gateway continues to validate the active mode, observed evidence references, and approval before execution.

Only transient model decisions use the new response shape. Saved revisions, evidence, and approval digests use their existing formats.

**Do not reintroduce** independent nullable action payloads in the model-facing schema or infer a missing action or payload from the rest of the response.

## Investigation latency

On 2026-11-16, the Debug trace for `task_20261116_000021` recorded 18 completed model requests taking 794 seconds in total. The 18 repository operations took under one second total. Three consecutive reads of known files required three provider round trips. Batching independent reads helped, but each dependent read still required the main model to select another typed action. Earlier prompts also repeated execution receipts alongside the observations.

AUTO, DEBUG, and PLAN use one `investigate` decision when the main agent needs more evidence. The selected chat model takes a read-only coding role and uses native tools. The same session can choose a later read from a search result. The ToolRunner validates read authority and records the operation on the originating turn. The main agent saves completed batches on TaskBoard and uses the report and observed evidence for its next decision. Malformed and out of scope batches stop before dispatch.

The main agent reserves up to eight native model turns and 24 repository calls within the task's remaining limits. The investigator counts usage locally and the main agent settles the allowance once. A crashed attempt retains its reserved allowance and its saved observations. Resume keeps the same TaskBoard task and fences late responses from the previous attempt. Saved evidence and approved plans require no migration.

**Do not reintroduce** main model decisions for individual repository reads, investigation budget renewal, or execution receipts in the model prompt. Validate all selections in a native response against the evidence available before that response.

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
