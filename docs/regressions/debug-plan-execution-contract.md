# Debug and Plan execution contracts

## Symptom and cause

A bug report in ordinary chat had no dedicated evidence-first coding route. The Plan service added prompt keywords to repeated repository-wide globs and searches. Missing or invalid model output could produce a generic implementation template. The console, TUI, and standalone plan command also had different clarification and approval paths. Approval was stored on a session object, and the coding handoff carried only the original goal. The execution controller could therefore replan the work without the steps or boundaries that the user reviewed.

To reproduce the old behavior, use `/plan` for an ambiguous repository change, then approve it and compare the coding request with the reviewed steps. A separate console planning flag asked a fixed number of questions even for a clear request. The previous Plan workflow could also start through the root CLI's generic repository preparation path, which initialized Git before the proposal if the directory had no repository.

## Invariants

- A model decision selects the next observation, clarification, plan draft, or debug repair. Mana asks for one correction on the same model and response format and stops if the correction fails. Debug starts with reported evidence and must cite a narrower observation before widening the investigation.
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

The model-facing evidence schema now has separate file-read and discovery branches. A file read requires exactly one repository-relative path, a level from 1 to 3, and an empty initial repository scope. Level 4 remains available to discovery tools under the existing expansion and audit rules. The prompt tells the model to select a level-1 read for a known file, including after wider discovery. Each read remains a separate operation and a model decision can include several independent operations. Path checks, bounded line reads, and the limit of one correction on the selected model and response format still apply. Saved evidence fingerprints and plan approval records use their existing formats.

**Do not reintroduce** a shared evidence schema that allows level 4 or zero or multiple paths for a file read. Keep wider discovery and exact reads as separate model selections.

## Decision activity and provider stops

A Debug turn appeared to stall at “Task started” and eventually displayed `Invalid or unavailable ModeDecision model decision (InvalidResponseError)`. The conversation journal recorded two model requests and zero tool starts. The first model response arrived after about 33 seconds. The second request failed after about 108 seconds and used 26,214 output tokens, matching the configured output limit inferred from the recorded model and budget. The journal retained the error class and usage, and the provider's stop reason could not be recovered.

The mode workflow used the frontend turn ID for the supervisor's lifecycle events, then switched the model and tool activity to the task ID when it established the attempt scope. The frontend consequently missed progress on the visible turn. Also, PlanService replaced invalid-response diagnostics with the exception type. Existing workflow fixtures advanced scripted decisions and did not assert that lifecycle, model, and tool events appeared together in the turn journal.

The mode workflow now keeps the frontend turn ID and redaction context for activity within the same session. Background work can recover the originating turn ID from the supervisor record. Task ID, attempt generation, and lease checks continue to fence publication. Chat Completions token-limit and filter stops, plus explicit refusals and empty responses, can carry application-authored detail into the failed model event and the mode result. Unknown failures get a generic diagnostic. Provider stops still end the decision without selecting another action. There is no migration for existing tasks.

**Do not reintroduce** substituting the task ID for an available frontend turn ID, publishing provider response text as failure detail, or automatically retrying a provider stop by increasing its budget or changing models.

## Action payload regression

Plan or Debug could block after the model's correction with a generic list of action requirements. For example, a response that selected `draft` with a null draft satisfied the provider schema and then failed the application validator. The original schema described the action and every optional payload independently. The earlier correction fix gave the model another attempt, while the schema still allowed invalid combinations on both calls. The reported diagnostic alone gives no evidence about which combination the live model selected.

The response object now contains one `decision` with a schema branch selected by its `action`. The `inspect` branch requires evidence, `clarify` requires one or more questions, `draft` and `repair` require a plan draft, and `research` requires a debugging report. Fields from other branches are rejected. The wire schema uses `anyOf` inside a root object for the supported structured-output transports. Local validation selects the branch by its action tag and still enforces semantic checks within the existing one-attempt correction. The gateway continues to validate the active mode, observed evidence references, and approval before execution.

Only transient model decisions use the new response shape. Saved revisions, evidence, and approval digests use their existing formats.

**Do not reintroduce** independent nullable action payloads in the model-facing schema or infer a missing action or payload from the rest of the response.

## Investigation latency

On 2026-11-16, the Debug trace for `task_20261116_000021` recorded seven decisions of 11–73 seconds each before individual tool calls. The calls themselves finished in 20–52 milliseconds. Three consecutive reads of known files required three provider round trips. The modeled input grew from roughly 8,800 to 15,300 estimated tokens across the observed decision requests. Each request included recent tool results with their execution receipts, including identifiers, timings, and artifact references. One discovery tool also failed because the model put glob patterns in its literal paths field.

`PlanService` now gives Plan and Debug a bounded `inspect_batch` action for two to four independent repository observations. Every selection is checked against the evidence and tool budget available before the batch, so a discovery result cannot authorize another operation in the same batch. Repeated operations and invalid discovery arguments go through the existing decision correction. The workflow checks the active lease and uses the same read-only dispatch for each operation. It saves each result under its own evidence ID, so retry can use observations saved before an interruption. The next decision receives evidence IDs, levels, and observed results. Tool execution records remain in task state and activity. Existing single-operation decisions and saved evidence remain compatible.

**Do not reintroduce** one provider call per independent read or full tool execution receipts in the investigation prompt. Validate a whole batch before dispatch and charge each operation to the tool budget.

## Verification

Run `venv/bin/python -m pytest tests/test_execution_modes.py tests/test_chat_planning_mode.py tests/test_chat_command_dispatcher.py tests/test_cli_modes_skills.py tests/gateway/test_debug_plan_mode_contract.py tests/test_codex_integration.py -q`. The gateway tests use deterministic model responses and isolated inboxes. The Codex protocol test uses an app-server fixture. Live acceptance requires an authorized provider and a local Docker image with the target project's test dependencies.

To verify the validation repair, run `venv/bin/python -m pytest tests/llm/test_engine.py tests/gateway/test_debug_plan_mode_contract.py -q`. The inference tests exercise native, tool, and text responses with local fixtures and assert that no rejected values are surfaced.

Run `venv/bin/python -m pytest tests/llm/test_providers.py -k 'native_mode_decision or native_debug_report' -q` to exercise the real Responses and Chat Completions decoder against local fixtures. The tests verify that the schema rejects the old wire shape, accepts typed records, rejects duplicate paths, and preserves the approval digest. Live provider acceptance remains unverified.

For file-read evidence, run `venv/bin/python -m pytest -q tests/llm/test_providers.py tests/llm/test_engine.py tests/gateway/test_debug_plan_mode_contract.py -k 'mode_evidence or mode_decision or repository_audit'`. These fixtures cover rejection of invalid file-read shapes, the same-format correction, and a narrow read after an explicit repository audit. The agent has not run these checks or exercised a live provider. Restart Mana-Agent from the updated checkout and use `/task resume <task-id>` to exercise the original blocked investigation.

For the activity and provider-stop regression, run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py::test_debug_mode_activity_keeps_the_frontend_turn tests/llm/test_engine.py::test_mode_decision_reports_why_the_provider_stopped_correction`. The first fixture checks that task, model, and tool events share the originating turn and retain redaction. The second checks the provider stop diagnostic and the two-request limit. The agent did not run these checks. Live terminal acceptance can use a local fixture that asks Debug to inspect a reported file.

For action payloads, run `venv/bin/python -m pytest -q tests/llm/test_providers.py tests/llm/test_engine.py tests/gateway/test_debug_plan_mode_contract.py tests/test_execution_modes.py`. The fixtures check the provider schema for each action, reject missing or unrelated payloads, exercise semantic correction using all three response strategies, and confirm a resumed draft still waits for approval. These checks and live provider acceptance remain pending.

For investigation latency, run `venv/bin/python -m pytest -q tests/gateway/test_debug_plan_mode_contract.py tests/llm/test_engine.py tests/llm/test_providers.py -k 'investigation_batch or mode_decision_wire_schema'`. The fixtures cover two reads from one decision, budget and scope rejection before dispatch, bounded provider schemas, discovery argument correction, and resuming with saved evidence. Reproduce interactively in the updated Mana-Agent process with a Plan or Debug request naming two or more independent files. Compare model and tool events on that turn and check that one model decision can produce multiple tool starts. The agent did not run the tests or measure a live turn after the change.
