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

## Verification

Run `venv/bin/python -m pytest tests/test_execution_modes.py tests/test_chat_planning_mode.py tests/test_chat_command_dispatcher.py tests/test_cli_modes_skills.py tests/gateway/test_debug_plan_mode_contract.py tests/test_codex_integration.py -q`. The gateway tests use deterministic model responses and isolated inboxes. The Codex protocol test uses an app-server fixture. Live acceptance requires an authorized provider and a local Docker image with the target project's test dependencies.

To verify the validation repair, run `venv/bin/python -m pytest tests/llm/test_engine.py tests/gateway/test_debug_plan_mode_contract.py -q`. The inference tests exercise native, tool, and text responses with local fixtures and assert that no rejected values are surfaced.

Run `venv/bin/python -m pytest tests/llm/test_providers.py -k 'native_mode_decision or native_debug_report' -q` to exercise the real Responses and Chat Completions decoder against local fixtures. The tests verify that the schema rejects the old wire shape, accepts typed records, rejects duplicate paths, and preserves the approval digest. Live provider acceptance remains unverified.
