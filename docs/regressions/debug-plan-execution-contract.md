# Debug and Plan execution contracts

## Symptom and cause

A bug report in ordinary chat had no dedicated evidence-first coding route. The Plan service added prompt keywords to repeated repository-wide globs and searches. Missing or invalid model output could produce a generic implementation template. The console, TUI, and standalone plan command also had different clarification and approval paths. Approval was stored on a session object, and the coding handoff carried only the original goal. The execution controller could therefore replan the work without the steps or boundaries that the user reviewed.

To reproduce the old behavior, use `/plan` for an ambiguous repository change, then approve it and compare the coding request with the reviewed steps. A separate console planning flag asked a fixed number of questions even for a clear request. The previous Plan workflow could also start through the root CLI's generic repository preparation path, which initialized Git before the proposal if the directory had no repository.

## Invariants

- A model decision selects the next observation, clarification, plan draft, or debug repair. A malformed decision stops that workflow. Debug starts with reported evidence and must cite a narrower observation before widening the investigation.
- A reviewable plan has ordered steps, exact path operations, explicit verification, and a revision digest. Only the human inbox can authorize that digest. Actor, task, revision, workspace, and supervisor lease must match before mutation.
- A coding engine proposes a patch in read-only mode. Mana reviews the patch for scope, applies it through the existing action policy, and records the observed result. The device's general approval mode cannot supply plan approval.
- Cancellation or a material revision invalidates the authority to start another patch. An incomplete mutation receipt prevents replay. A retry of a failed check may reuse a previously verified patch without repeating it.
- Existing session-only plans or approval flags have no execution authority. Recreate them through the current Plan entry points.

**Do not reintroduce** keyword-based plan execution, unconditional repository scans before debugging, or a goal-only handoff from approval to a writable coding engine.

## Verification

Run `venv/bin/python -m pytest tests/test_execution_modes.py tests/test_chat_planning_mode.py tests/test_chat_command_dispatcher.py tests/test_cli_modes_skills.py tests/gateway/test_debug_plan_mode_contract.py tests/test_codex_integration.py -q`. The gateway tests use deterministic model responses and isolated inboxes. The Codex protocol test uses an app-server fixture. Live acceptance requires an authorized provider and a local Docker image with the target project's test dependencies.
