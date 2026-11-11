# Unresolved coding bugs: Main Model research handoff

Date: 2026-11-10. Status: implemented locally; agent execution of verification is prohibited by repository Rule 40. The commands below have **not been run by the agent**. Live provider, Codex, Batch and UI acceptance remain unverified.

## Symptom and demonstrated architectural gaps

An unresolved dependency bug could end a coding turn or repeat local fixes without a typed handoff to the Main Model. The gateway's continuation check only covered pass-limit/needs-resume boundaries. Native results had no debugging report; the Codex parser treated a successful no-diff write turn as a mutation failure. Consequently an intentional request for external evidence could not be distinguished from unfinished mutation work.

Source inspection also established that general search could fall back to keyword routing, repository-wide search memory could satisfy a different task based on confidence alone, and GitHub issue normalization read code-search fields instead of issue title/body. These are source-grounded mechanisms, not claims that a historical production incident was reproduced.

## Corrected invariants

The existing autonomous controller owns the research step. Both engines can return a validated `CodingDebuggingReport` in a final JSON envelope. Such a turn is `waiting`; it is neither successful completion nor a reason for a forced mutation retry. Ordinary final responses remain compatible. Invalid report envelopes fail explicitly.

The Main Model receives the report, actual evidence, previous fixes, environment, search history and remaining allowance. Its strict coding continuation decision may attach a research plan only to `CONTINUE` or `REPLAN`. Intent-only routing payloads cannot manufacture coding completion. A familiar locally resolved bug requires no search; unfamiliar external behavior can justify research before a failed fix.

Research uses existing web/GitHub providers with an explicit decision override and repository search memory disabled for this path. Every outbound query and qualifier is sanitized before provider dispatch or research telemetry. Source authority and version applicability are evaluated by the Main Model; a GitHub domain alone does not establish upstream ownership. Excerpts remain labeled as excerpts. The existing read-only browser tools can inspect observed public sources. Retrieved content is untrusted data and cannot authorize tools, commands, patches or permissions.

Provider failure diagnostics on this path retain exception types rather than raw exception chains, which can contain credential-bearing URLs or response bodies. Invalid model/report diagnostics likewise omit rejected input values. The evidence sent back to the Main Model is a compact window; full task-scoped outcomes remain durable without being replayed into every prompt.

New coding admissions receive search/browser eligibility when the feature is enabled; lane permissions, per-task restrictions and provider availability still gate each operation. Old task permissions are not enlarged on recovery. Mana's research permission is separate from Codex's sandbox networking and does not grant Codex network access.

## State, recovery and compatibility

Research state is versioned in the existing supervisor provider metadata/checkpoint and exposed on the existing TaskBoard item through an optional field. Older records default to empty research state. No new task, worker, research agent, database or dependency is introduced. Task IDs, owners, completed children, assigned workspace and backend selection remain intact.

Request reservations precede dispatch; outcomes precede delivery. Failed network requests consume the search allowance. Cached outcomes do not. An interrupted request remains consumed and uncertain rather than being automatically retried. Dependency-manifest, dependency-version and runtime changes invalidate cache applicability without resetting counters. Restoring a task cannot increase its saved research limits.

Coding continuation tickets distinguish pending, started, Batch-suspended and delivered work. Completed receipts are reused. A started continuation with an unknown outcome raises `CODING_RESEARCH_RECONCILIATION_REQUIRED` and cannot be automatically replayed. Internal resumes retain completed operation fingerprints and reject identical side-effect replays; external resumes retain the original Codex thread and do not start a fresh thread if it is missing. Batch keeps its original model/tool run and completed waves. Pending Batch Main Model decisions retain the exact request across wakes.

When result escrow already contains the exact task, attempt/generation and continuation-ticket receipt, recovery consumes that recorded result rather than rerunning coding. An unrelated or rejected receipt cannot authorize replay or completion.

Main Model clarification requests use the existing durable human inbox, with a branch-specific deduplication key and checkpoint. An authenticated answer resumes only that queued coding branch and is supplied to the next Main Model decision. Cancellation or a superseded/expired lease prevents late findings or delivery.

Research events use `investigation.coding_research.*` and appear in the existing Coding Activity box on CLI/TUI/dashboard surfaces. Public events contain concise phase/status information; private reasoning, raw model payloads and queries are not streamed as chat cards.

## Configuration and limits

Persist these settings in `~/.mana/config.toml` through the established configuration precedence:

```toml
MANA_CODING_RESEARCH_ENABLED = true
MANA_CODING_RESEARCH_MAX_SEARCHES = 12
MANA_CODING_RESEARCH_MAX_ROUNDS = 4
```

Each model-selected round permits up to three searches and three source inspections. Search and model/tool/time/spending allowances remain subordinate to the parent execution budget; research does not renew them. Searches stop on low value, repeated no progress, denial or exhaustion. A fresh Main Model decision must choose useful local work or explain the blocker when a provider is unavailable. An invalid decision stops the workflow rather than selecting an alternative action.

## Do not reintroduce

- Never use a keyword match, failure counter or missing model output to choose research.
- Never allow a debugging report or a research recommendation to satisfy mutation, integration, review or verification requirements.
- Never invoke the generic search decision fallback or confidence-only cross-task cache for coding research.
- Never transmit raw repository code, credentials, private paths or internal host information as a search query.
- Never reset research counters on recovery, silently enlarge task permissions, switch coding backends, or replay an uncertain coding continuation.
- Never replace the consolidated activity boxes with per-search panels or expose private reasoning.

## User-owned verification

Targeted regression tests (fake models/providers; isolated leased tasks):

```bash
venv/bin/python -m pytest tests/coding/test_bug_research.py tests/gateway/test_coding_bug_research.py tests/execution/test_coding_research_recovery.py -q
```

Compatibility and dashboard classification:

```bash
venv/bin/python -m pytest tests/test_search_router.py tests/test_coding_engine_modes.py tests/test_activity_stream_integration.py tests/gateway/test_checkpoint_resume_invariants.py -q
node --test tests/dashboard/live_chat_reducer.test.mjs
```

For live acceptance, configure an allowed search provider and use an unresolved version-specific dependency error in each coding backend. Confirm that the Main Model selects a narrow query, reports sources and uncertainty, and returns to the same task/workspace. Exercise unavailable search, live steering, cancellation, and interruption before/after delivery. Exercise Batch submission/wake and durable clarification separately. A researched fix is not validated until its authorized goal-specific verification actually succeeds.
