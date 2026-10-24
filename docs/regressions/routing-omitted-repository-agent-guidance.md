# Regression: routing omitted repository agent guidance

## Symptom and trigger

A job entering through either the gateway entry router or the MainAgent router could begin its routing decision without reading the repository's root `AGENTS.md`. A repository may use this file to specify local development conventions and scoped workflow requirements, so the routing model could make a decision without that applicable context. The issue affects fresh entry decisions and atomic multi-task child entry decisions.

## Root cause

`src/mana_agent/prompting/repo_rules.py` already provided a repository-rule renderer, but neither routing boundary used it. Gateway `EntryRouteContext` also did not preserve the active repository root through context compaction, so the entry router had no reliable repository path from which to load guidance. MainAgent's `Router` passed only the semantic request and command hint to the decision engine.

## Permanent fix

- Entry routing now receives the repository root in its route context, preserves it through compaction, reads root-level `AGENTS.md` and/or `agents.md`, and supplies the content to both the normal route-decision prompt and the Decisions-choice evidence.
- MainAgent loads the same guidance before repository-scope selection and includes it in both the scope decision and the structured route, review, and bounded repair payloads. Autonomous MainAgent execution carries the repository root and loaded guidance through its strategy-routing context.
- The shared loader reports when no root guidance exists and refuses to read a guidance symlink whose resolved target is outside the repository.
- Prompts explicitly treat repository text as subordinate data; it cannot override system/safety policy, permissions, or the user's explicit request.

## Regression coverage

Added tests verify entry-prompt inclusion, MainAgent scope and route decision inclusion, lowercase filename support, absent-file reporting, and rejection of outside-repository symlink content.

## Do not reintroduce

Do not start an entry, MainAgent, or MainAgent-autonomous routing job without performing the repository-root guidance lookup and passing its result to the relevant routing model. Preserve `repository_root` through gateway context compaction and keep repository guidance separate from user request text and subordinate to system and authorization policy.

## User-owned verification

The repository instructions prohibit the agent from running tests by default; these commands are provided for the user:

- `./venv/bin/pytest tests/gateway/test_entry_routing.py -k test_entry_router_includes_repository_agents_guidance -q`
- `./venv/bin/pytest tests/multi_agent/test_router_repository_guidance.py -q`
- `./venv/bin/pytest tests/gateway/test_entry_routing.py -q`
