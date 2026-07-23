# Change Log

All notable repository changes should be recorded here.

## 2026-07-23

- Fixed the Textual multiline composer to resize immediately after programmatic text assignment, including on Windows' Proactor event loop where the queued change event may arrive after the next layout cycle.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_tui_multiline_input.py` passed.

- Fixed Textual `/new` timeline replacement to clear mounted chat cards as well as stored events, reset transient tool/token presentation state, and display the newly activated empty conversation immediately.
  - Verification: TUI command/rendering, gateway, and unified session regressions passed (42 tests); affected-file Ruff, Python compilation, and `git diff --check` passed.

- Bumped the package and documented release version to `v0.0.19`.
  - Verification: `tests/test_package_version.py` and `git diff --check` passed.

- Fixed `/new` in dashboard/API conversation submission and plain CLI presentation so it deletes the active canonical session, creates and activates one replacement, and clears an empty replacement timeline without persisting `/new` as chat content.
  - Verification: Conversation service, API conversation, TUI, gateway, and unified session/command regressions passed (41 tests); affected-module Ruff, Python compilation, and `git diff --check` passed. The legacy `chat_cli.py` file still reports its existing star-import Ruff findings.

- Clarified the entry-routing model contract with an explicit required-source vocabulary and per-route source rules, preventing route/tool names from being mistaken for source identifiers while preserving strict fail-closed validation.
  - Unknown source errors now identify the rejected model value and list the permitted identifiers; no alias or fallback route is executed.
  - Verification: Entry-router, gateway, and unified command/session regression tests passed (46 tests); affected-file Ruff, Python compilation, and `git diff --check` passed.

- Unified chat lifecycle around canonical workspace sessions, with destructive `/new`, `/sessions` management and exact history switching, title generation, safe physical deletion, memory tombstones, and one-time dashboard-conversation migration.
  - Added the shared typed command registry across gateway, CLI chat, Textual, API, and Telegram, connector setup/management, API session/command/connector/process endpoints, and a generated CLI capability matrix with explicit unsupported reasons.
  - Added a persistent registered-worker process manager with atomic metadata, identity-checked stop/restart, stale recovery, singleton prevention, bounded log reads, lifecycle events, and background Telegram startup without PID-file or arbitrary-shell execution.
  - Removed the TUI `/new` history message, the Telegram-only command implementation, dashboard-only chat identity, and dashboard-managed daemon chat thread.
  - Added Textual session/setup modals plus dashboard chat rename/delete, connector setup, and background-process health/log/control pages.
  - Verification: `PATH="$PWD/venv/bin:$PATH" venv/bin/python -m pytest -q` passed (1,208 passed, 2 skipped); focused unified-session/command/process, gateway, natural-language routing, TUI, Telegram, and API conversation tests passed (60 tests), with a final focused UI/session/API pass (36 tests); new/affected-module Ruff, Python compilation, and `git diff --check` passed. The required repository-wide `ruff check .` was run and still reports 800 unrelated pre-existing findings in legacy files/tests.

## 2026-07-22

- Added explicit, shared `codex`/`internal` coding-backend selection across the gateway-owned CLI, TUI, API, and dashboard stack. Disabling Codex now activates Mana-Agent's existing model-driven internal coding tools without starting or authenticating Codex, while a selected Codex turn remains fail-closed with no runtime fallback.
  - Added a backend-neutral, ordered live coding event contract with Codex notification normalization, internal tool lifecycle emission, duplicate suppression, bounded/redacted output, durable session events, turn-scoped delivery, and a responsive Textual execution panel for backend/model, activity, output, timing, and token usage.
  - Added the coding-runtime configuration controls and documented the backward-compatible default rule: missing backend settings select Codex when enabled and internal when disabled; contradictory explicit Codex settings fail validation.
  - Verification: the affected backend-selection, internal-agent, Codex, gateway, TUI layout/live-tool, user-config, conversation persistence, WebSocket, and API suite passed (220 tests); the isolated full suite passed (1,201 passed, 2 skipped); touched-file Ruff, Python source/test compilation, and `git diff --check` passed with the repository Python 3.12 environment.

- Fixed the normal-mode two-turn chat smoke test to use its isolated temporary workspace, preventing the Windows CI checkout from being used for session and telemetry state.
  - Verification: targeted CLI smoke regression passed locally.

- Fixed the tool-backed chat rendering smoke test to use its isolated temporary workspace instead of the CI checkout, preventing Windows checkout-permission failures while preserving its telemetry assertion.
  - Verification: targeted CLI smoke regression passed locally.

- Made Git subprocess output decoding deterministic with UTF-8 and lossless surrogate handling, preventing Windows code-page corruption of Unicode filenames during repository preparation and Git inspection.
  - Verification: repository-preparation and Git-tool regression tests, touched-file Ruff, and Python compilation passed; the isolated full suite is running.

- Fixed coding runs for valid user-selected directories that have not yet been initialized as Git repositories. The gateway and multi-agent runtimes now use one locked workspace/repository preparation boundary that preserves existing files, initializes new repositories on `main` without staging or committing, reconciles canonical persistence records, recognizes Git worktrees, and avoids nested repositories when a valid parent repository owns the selected subdirectory. Bare, corrupt, stale, unsafe, unavailable-Git, permission, initialization, and persistence failures now stop before Codex with phase-specific errors.
  - Verification: focused repository-preparation, gateway, Codex, workspace, and multi-agent tests passed (129 tests); the required repository/workspace/gateway/Codex selection passed (170 tests, 1,023 deselected); the full suite passed (1,192 passed, 2 skipped); touched-file Ruff, `python -m compileall src tests`, `git diff --check`, and manual non-Git/repeated-run/parent-repository coding-start checks passed. Repository-wide Ruff still reports 807 unrelated pre-existing findings outside this change.

- Added a shared, provenance-aware artifact routing registry to the gateway. It recognizes spreadsheet (`.xls`, `.xlsx`, `.xlsm`, `.csv`, `.ods`), document, presentation, PDF, and image categories; user attachments and explicitly named targets now supply family, MIME/extension, repository-membership, and handler evidence to the model before lane selection. The new artifact lane is lock-free for standalone user files, validates handler availability before dispatch, stages user inputs in an isolated artifact workspace, and invokes local document tools without requiring Codex. Repository-member source edits remain eligible for the coding route.
  - Verification: focused artifact, entry-routing, lane-coordinator, chat-gateway, and routing-authority tests passed (66 tests); Python compilation and `git diff --check` passed. Ruff is not installed in the repository virtual environment.

- Isolated pytest runtime state in a per-run temporary Mana home, added a write guard for the real `~/.mana`, and removed import-time user-config path snapshots so repository, session, workspace, cache, database, and configuration artifacts are cleaned without touching user data.
  - Verification: focused isolation, configuration, repository, session, workspace, CLI, and subprocess tests passed (70 direct focused tests plus the persistence-focused run); Python compilation and `git diff --check` passed. A full-suite attempt started successfully but could not be completed in the local terminal integration, which detached from its still-running pytest processes; those test processes and their temporary Mana homes were then removed. Ruff is not installed in the repository virtual environment.

- Isolated every Mana-managed Codex app-server run behind a generated per-run `CODEX_HOME` and a validated `mana_runtime` Responses API provider using the model, API key, base URL, and safe headers selected by Mana's provider/model routing.
  - API keys now travel only in the child environment; inherited global Codex/OpenAI authentication is removed, runtime configuration is owner-only and cleaned on shutdown/startup failure, global `~/.codex` state remains untouched, and unsupported or incomplete provider decisions stop without login or provider fallback. Removed the obsolete `mana-agent codex login` and `logout` commands.
  - Verification: Focused Codex/provider/doctor/CLI tests passed (48 tests, 65 deselected); affected TUI/config/coding/model-routing tests passed (97 tests); affected gateway/CLI tests passed (40 tests, 217 deselected); Python compilation and `git diff --check` passed. The full suite completed with 1,162 passed and 1 skipped; two unrelated multi-agent tests failed because bare `python` was absent from the subprocess `PATH`, then passed when rerun with the repository virtual environment on `PATH` (38-test rerun). Ruff was unavailable in the repository and bundled environments.

- Extended the deployed evidence-based model router across gateway, CLI, TUI, and Codex task dispatch with persisted task-aware requests/decisions, explicit routing modes, single-model default policy, evidence-gated multi-agent/parallel approval, and fail-closed decision persistence.
  - Added gateway-owned live task control with validated pause/resume/cancel/reprioritize/block/verify transitions, task-tree cancellation, routing identity, budgets, evidence, ownership locks, restoration-safe state, structured events, shared CLI/TUI control commands, and expanded doctor diagnostics.
  - Verification: `MANA_HOME=/tmp/mana-routing-full-3 PYTHONPATH=src .venv/bin/python -m pytest -q` passed (1,145 passed, 1 skipped); the focused routing/gateway/Codex/doctor/TUI suite passed (116 tests); focused Ruff, Python compilation, and `git diff --check` passed. A configured type checker was unavailable. The system `python` command points to a legacy interpreter and was not used; verification used the repository Python 3.12 virtual environment.

## 2026-07-21

- Fixed adaptive gateway model selection to tolerate legacy and test settings objects that omit the optional `mana_codex_model` field while still honoring it when configured.
  - Verification: `MANA_HOME=/tmp/mana-agent-ci-final PYTHONPATH=src .venv/bin/python -m pytest -q` passed (1,141 passed, 1 skipped); the focused planning/CLI regression suite passed (73 tests); focused Ruff, Python compilation, and `git diff --check` passed.

- Replaced fixed role-to-model resolution with a centralized evidence-based adaptive router using typed requests/profiles/decisions, deterministic capability/quality/history/language/cost/latency scoring, cached repository metadata, verification-reserved budgets, decaying provider reliability penalties and circuit breakers, persistent redacted outcome history, independent verifier selection, and fail-closed routing errors.
  - Added policy-gated two-candidate competition contracts that require isolated roots, normalized diff/test evidence, complete quality criteria, winner-only promotion, and losing-workspace cleanup; legacy `MODEL_LEVEL_*` configuration now migrates into profile hints instead of making the final choice.
  - Extended `mana-agent doctor` and configuration/architecture/provider documentation with candidate, metadata, evidence-store, circuit, budget, verifier-independence, and isolation diagnostics.
  - Verification: `MANA_HOME=/tmp/mana-agent-router-final-full PYTHONPATH=src .venv/bin/python -m pytest -q` passed (1,140 passed, 2 skipped); the focused gateway/Codex/config/worktree/doctor compatibility run passed (176 tests); focused Ruff, Python compilation, and `git diff --check` passed. A configured type checker was unavailable.

- Fixed local-process execution output to normalize Windows CRLF line endings to the provider's cross-platform LF text contract.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/execution/test_execution_fabric.py` passed.

- Bumped the package and documented release version to `v0.0.18`.
  - Verification: Project metadata and source runtime version checks passed; `tests/test_package_version.py` (2 passed) and `git diff --check` passed.

- Changed Codex write turns to use the selected repository root by default instead of creating a managed worktree under Mana state. Worktree isolation remains available through `MANA_CODEX_WORKTREE_ISOLATION=true`; direct-root turns can operate on an existing dirty checkout.
  - Verification: `MANA_HOME=/tmp/mana-codex-root-tests PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_codex_integration.py tests/test_tui_user_config.py tests/gateway/test_chat_gateway.py` passed (57 tests); focused Ruff and `git diff --check` passed.

- Added a provider-neutral Remote Execution Fabric with typed sandbox, routing, resource, network, secret, artifact, snapshot, health, and lifecycle contracts; atomic handle/lease persistence; restart cleanup; sanitized lifecycle events; bounded concurrency; and fail-closed capability enforcement.
  - Registered `local-process`, `local-docker`, `remote-ssh`, `kubernetes`, `modal`, and `custom-http-runtime` behind one asynchronous provider interface. Existing trusted local queued shell execution now runs through the gateway-owned `ExecutionManager` and preserves managed-worktree identity, while Docker/SSH use safe argv construction and Kubernetes/Modal/HTTP dependencies remain optional with actionable configuration errors.
  - Added the reusable provider contract/security tests, provider doctor diagnostics, optional SDK extras, architecture and lifecycle mapping, provider configuration/setup/troubleshooting guidance, the versioned custom HTTP contract, and security enforcement limitations. Real Docker, SSH, Kubernetes, Modal, and HTTP integration tests were not run because corresponding external infrastructure and credentials were not configured.
  - Verification: `MANA_HOME=/tmp/mana-remote-execution-test-home PYTHONPATH=src .venv/bin/pytest -q` passed (1,129 passed, 1 skipped); the final execution/AskAgent/gateway/tool suite passed (112 tests), and the worktree/doctor compatibility suite passed in the earlier 97-test focused run; focused Ruff, Python compilation, and `git diff --check` passed. The non-isolated full suite was also attempted and exposed the existing external-memory `MemoryConfigurationError`; the isolated full suite above passed.

## 2026-07-20

- Added webhook-driven GitHub App Autopilot with signed raw-body ingress, durable delivery/job persistence, deterministic validated event routing, actor authorization, installation-scoped authentication, persistent task sessions, isolated worktrees, Codex-only execution, verification gates, deterministic branches, and draft pull-request lifecycle support.
  - Added `mana-agent github-app` operational commands, health/readiness endpoints, least-privilege manifest/setup documentation, security-alert redaction, idempotency/coalescing, subject locks, bounded retry/cancellation controls, and structured lifecycle metrics.
  - Verification: `.venv/bin/ruff check src/mana_agent/github_autopilot src/mana_agent/commands/github_app_cli.py tests/test_github_autopilot.py src/mana_agent/integrations/codex/backend.py src/mana_agent/integrations/codex/coding_agent_shim.py tests/test_codex_integration.py` passed; `.venv/bin/python -m pytest tests/test_github_autopilot.py tests/test_codex_integration.py tests/test_api_analyze.py tests/test_api_conversations.py tests/test_package_version.py -q` passed (40 tests). Full-suite verification was not completed because the existing external-memory test configuration causes unrelated `MemoryConfigurationError` failures in `tests/test_ask_agent.py`.

- Fixed the Windows release test synchronization for dynamically appended TUI chat messages.
  - The regression test now waits for Textual's layout cycle and the subsequently posted resize cycle before asserting the new message's wrapped document, matching both Windows Proactor and POSIX event-loop scheduling without changing chat behavior.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_tui_tool_card_layout.py tests/test_tui_multiline_input.py` passed (6 tests); focused Ruff, Python compilation, and `git diff --check` passed.

- Fixed the eval runner patch-capture test on Windows by replacing its POSIX-only shell assertion with a platform-native Python verification command.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/evals` passed (27 tests); focused Ruff, Python compilation, and `git diff --check` passed.

- Added optional ACP v1 and A2A 1.0 protocol gateway adapters around the shared `AgentChatGateway`, durable workspace sessions/history, task board, lane coordinator, memory, and tool policy.
  - Added ACP stdio initialization, durable new/load/list session mapping and replay, prompts, cancellation, modes/configuration, resource links, per-session MCP forwarding, safe event conversion, editor documentation, and `mana-agent acp serve|doctor|info`.
  - Added an authenticated A2A server with runtime Agent Cards, JSON-RPC/HTTP+JSON routes, caller-scoped durable task storage, gateway executor, state/artifact streaming, cancellation, remote registry/discovery/invocation, explicit delegation policy, SSRF/path/size controls, and loop protection through `mana-agent a2a` commands.
  - Added bounded stable SDK extras (`acp`, `a2a`, and `protocols`) and included both in `full`. Push notifications, extended Agent Cards, embedded ACP media, client terminal/filesystem delegation, and unrestricted file artifacts are intentionally not advertised.
  - Verification: the full repository suite passed (1,112 passed, 1 skipped); the final focused protocol/gateway/MCP/doctor/config/task-board suite passed (156 tests); official ACP/A2A SDK model and authenticated route smoke checks, focused Ruff, Python compilation, and `git diff --check` passed.

- Added Mana Eval Lab for reproducible multi-variant gateway evaluations, immutable redacted run artifacts, isolated Git worktrees, task and trajectory replay, configurable objective scoring, leaderboards, baselines, paired regression reports, and fail-closed CI gates.
  - Instrumented the existing gateway, routing-model, lane, tool, Codex, reviewer, and verifier boundaries through an optional context-propagated recorder; normal chat continues through the no-op recorder with no evaluation configuration.
  - Added the `mana-agent eval` command group, protected routing suite, evaluation CI workflow, security and architecture documentation, and stable exit codes. Docker and remote evaluation workspaces remain explicit unsupported backends; P0 fully implements `local-worktree`.
  - Verification: `PYTHONPATH=src MANA_HOME=/tmp/mana-eval-final-model-home .venv/bin/python -m pytest -q tests/evals` passed (27 tests); the focused gateway/routing/lane/tool/Codex/CLI compatibility suite passed (181 tests); the final full repository suite passed (1,103 passed, 1 skipped); touched-file Ruff, Python compilation, `git diff --check`, source/wheel builds, and `twine check` passed.

- Fixed Windows Textual layout timing for multiline chat input and dynamically mounted selectable chat messages.
  - Composer sizing now treats explicit newlines as immediately authoritative when the virtual document refresh is delayed, while mounted message cards proactively rewrap after their first layout instead of waiting for a paint callback.
  - Verification: `PYTHONPATH=src venv/bin/python -m pytest -q tests/test_tui_multiline_input.py tests/test_tui_tool_card_layout.py` passed (6 tests); `git diff --check` passed.

- Added the deterministic `mana-agent doctor` command with a typed check registry, isolated check modules, stable check IDs, grouped terminal output, redacted JSON output, targeted `--only`/`--skip`, and stable 0/1/2 exit codes.
  - The initial fast offline checks cover Python/package and executable availability, Git, managed configuration parsing/schema/permissions, Mana state-path availability, and configured Codex binary resolution. Safe state-directory and owner-only configuration-permission repairs are opt-in, backed up where files change, and rechecked after repair.
  - Verification: `PYTHONPATH=src venv/bin/python -m pytest -q tests/test_doctor.py` passed; CLI help and redacted JSON output were checked with the repository virtual environment.

- Fixed one-character-per-line wrapping in TUI chat-history messages.
  - Dynamically mounted selectable message `TextArea` widgets now re-wrap when rendering first observes a valid card content width, preserving normal wrapping through history replay, live appends, and terminal resizes without changing input layout or scrolling/selection behavior.
  - Added regression coverage for user and assistant cards, borders/padding box sizing, narrow/wide resize, Persian/Unicode/emoji text, Markdown, and code blocks.
  - Verification: `python -m pytest -q tests/test_tui_tool_card_layout.py tests/test_tui_live_tools_scroll.py tests/test_tui_auto_chat_tool_events.py tests/test_tui_multiline_input.py` passed (13 tests); focused Ruff, Python compilation, and `git diff --check` passed. Mypy is not installed in the repository virtual environment.

- Added strict shared-gateway source routing for repository, browser, web search, Gmail, calendar, GitHub, memory, internal knowledge, and tool-free turns.
  - The typed routing decision now carries mandatory sources, live-data requirements, target URLs, reason/error codes, and a capability manifest. Browser, search, and repository evidence plans execute only the model-selected sources; a required-source failure aborts the turn with its exact source error and recorded execution status.
  - Browser availability is now based on the live Playwright/Chromium runtime status as well as its enablement setting, so an available browser is represented accurately in the routing manifest.
  - Direct public URLs are passed to the routing model as browser signals. Invalid, incomplete, unavailable, and capability-error decisions stop explicitly; browser/search/repository substitutions are not permitted.
  - Removed legacy AskService validation re-routing so an invalid selected command or unavailable semantic index cannot silently choose a new route.
  - Verification: `PYTHONPATH=src venv/bin/python -m pytest -q tests/gateway/test_entry_routing.py tests/test_ask_entry_router.py`; `PYTHONPATH=src venv/bin/python -m py_compile src/mana_agent/gateway/entry_routing.py src/mana_agent/gateway/chat_gateway.py src/mana_agent/gateway/lanes.py src/mana_agent/multi_agent/runtime/route_executor.py`.

- Fixed connector-only chat turns such as “Check my latest Gmail” so they do not initialize repository run-evidence memory when external memory is selected.
  - Verification: `python -m pytest tests/test_ask_agent.py -q`.

## 2026-07-19

- Added a pluggable, provider-neutral memory architecture with `internal/mana` as the compatibility-preserving default and lazy optional `external/mem0` support.
  - Added canonical async models and backend contract, centralized scope mapping, typed configuration/dependency/authentication/network/provider/storage errors, backend lifecycle and health checks, timeout-bound Mem0 calls, normalized responses, and explicit no-fallback behavior.
  - Existing SQLite coding-flow and JSON multi-agent records remain in place and production consumers now import the shared memory package. External-mode orchestration operations write canonical Mem0 records with turn-local indexes instead of falling back to local persistence; asynchronous add acknowledgements and V3 nested metadata filters are normalized.
  - Chat follow-ups now use one gateway-owned shared service: successful turn pairs are stored with conversation scope, relevant records are recalled into subsequent prompts, `/new` remains isolated, and explicitly degraded recall/write failures surface as turn warnings without cross-provider fallback.
  - The configuration TUI adds conditional Memory fields and stores Mem0 keys in the OS keyring while headless deployments may inject `MEM0_API_KEY`; a stalled GitHub CLI status probe can no longer prevent the configuration screen from mounting.
  - Verification: `PATH="$PWD/venv/bin:$PATH" MANA_HOME=<isolated> PYTHONPATH=src venv/bin/python -m pytest -q` passed (1063 passed, 1 skipped); focused memory, configuration, coding-memory, gateway, session, workspace, prompt, experience, and multi-agent tests passed; Python compilation, direct-legacy-import/storage scans, and `git diff --check` passed. A read-only live Mem0 health check and active workspace/session V3 metadata-filter search passed without exposing content or credentials. Ruff and mypy were unavailable in the repository environment.

- Added multiline Textual chat input: Enter sends, Shift+Enter inserts a line, and Ctrl+J / Alt+Enter provide portable terminal fallbacks. The composer grows with wrapped or explicit lines up to a scrollable maximum, then shrinks after edits or submission.
  - User messages retain internal newlines through rendering, gateway requests, and restored session history; only trailing newline characters are removed on submission.
  - Verification: targeted multiline/TUI/gateway tests passed (34 tests), including the model-command shortcut regression; Python compilation, focused Ruff, and `git diff --check` passed.

## 2026-07-18

- Fixed Textual chat tool cards to retain a single result widget and explicitly invalidate card and timeline layout after result, expand/collapse, and live-output updates. Collapsed cards now use content-driven height, while expanded cards remeasure their complete output without stale sizing. Documented Shift-drag native terminal text selection while preserving mouse controls.
  - Added read-only Textual selection widgets for chat and tool output, with mouse drag selection and `Ctrl+C` clipboard copying while retaining `Ctrl+C` quit when no text is selected.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_tui_tool_card_layout.py tests/test_tui_live_tools_scroll.py tests/test_tui_auto_chat_tool_events.py` passed (9 tests); targeted Ruff and `git diff --check` passed. Repository-wide Ruff remains blocked by 806 pre-existing violations outside this change; full pytest was started separately.

- Fixed transient Windows CI failures while replacing workspace, repository, and chat-session JSON state files.
  - The shared workspace atomic writer now retries Windows sharing violations without changing validation or persistence behavior, and cleans up its collision-safe temporary file on failure.
  - Added regression coverage for `PermissionError(13, "Access is denied")` during an existing session-state replacement.
  - Verification: `venv/bin/python -m pytest -q tests/test_workspaces.py tests/test_main_cli_session_lifecycle.py tests/test_cli_smoke.py::test_chat_ping_returns_pong_without_faiss_index tests/test_cli_smoke.py::test_chat_renders_dynamic_plan_and_diagram_blocks_in_normal_path` passed (15 tests); Python compilation and `git diff --check` passed.

- Added gateway-owned resource-aware specialist lanes for `coding`, `research`, `review`, `verify`, `release`, and `operations`.
  - Added typed serializable lane contracts, priorities, lock modes, execution states, budgets, handoffs, capability-based tool permissions, duplicate detection, concurrency/provider limits, parent-budget sharing, persistent lock leases, restart recovery, and structured `lane.*`, `lock.*`, and `resource.*` events.
  - `AgentChatGateway.process_turn` now reserves and releases lane resources around the existing entry-route/turn-engine path, preserving task, session, workspace, repository, Codex integration, and frontend identities across execution and handoffs.
  - Added existing-config overrides and architecture/configuration documentation for lane responsibilities, default handoffs, locking, budgets, recovery, and diagnostics.
  - Verification: `MANA_HOME=<isolated> PYTHONPATH=src venv/bin/python -m pytest -q` completed with 1031 passed and 1 skipped before two verification tests failed because bare `python` was absent from the subprocess `PATH`; both failures passed when rerun with `PATH="$PWD/venv/bin:$PATH"`. Post-hardening gateway/lane tests passed (50 tests), the broader focused gateway/workspace/queue/tool set passed (181 tests), Python compilation, CLI help, and `git diff --check` passed. Ruff and a static type checker are not installed in the repository environment.

- Enforced one fresh persisted session per chat start and one additional fresh session per `/new`.
  - Root chat startup now preserves the CLI dispatch boundary while deferring its mandatory route decision until the chat frontend has created the session, avoiding a hidden pre-chat session. The legacy restoration API now abandons prior active sessions and opens a new identity instead of reusing or reopening one.
  - Verification: `MANA_HOME=<isolated> PYTHONPATH=src venv/bin/python -m pytest -q tests/test_chat_first_configuration.py tests/test_workspaces.py tests/gateway/test_entry_routing.py tests/gateway/test_chat_gateway.py tests/test_main_cli_session_lifecycle.py` passed (58 tests); the reported `test_root_dispatches_chat_without_mode_menu` regression passed; Python compilation and `git diff --check` passed.

- Repaired implicit workspace and active-session repository references when a legacy repository identity no longer has a persisted record.
  - Valid repository attachments are preserved, missing secondary references are removed from both workspace and session state, and a missing session primary still stops safely instead of being hidden.
  - Verification: `PYTHONPATH=src venv/bin/python -m pytest -q tests/test_workspaces.py` passed (11 tests); isolated-home `tests/gateway/test_entry_routing.py` passed (9 tests); Python compilation and `git diff --check` passed. Ruff was unavailable in the repository environment. A broader multi-agent run passed 60 tests but retained two unrelated verification-pipeline failures in `tests/test_multi_agent_core.py`.

- Added one gateway-owned typed entry router that runs before every conversational response and selects `conversation`, `coding`, `gmail`, `calendar`, `search`, `repository`, `automation`, or `unsupported` from a dynamic route registry.
  - Gmail routing now checks enabled account configuration, `email.read` permission, and keyring credential availability before execution; configured requests run through an email-only tool policy, while genuine setup/authorization failures retain actionable provider details.
  - Invalid routing-model output stops safely as an unsupported route and never falls through to ordinary conversation or a false integration-unavailable response.
  - Route execution preserves `session_id`, `conversation_id`, and `turn_id`; follow-ups receive the previous route and chronological conversation context.
- Replaced chat-start session restoration with an explicit one-session-per-open-chat lifecycle, superseding the 2026-07-17 restoration behavior.
  - Session records now include `active`, `closed`, and `abandoned` states, opening/closing timestamps, and process ownership; legacy `archived` records remain readable.
  - CLI exit, TUI quit/unmount, dashboard shutdown, and `/new` share an idempotent gateway finalizer. `/new` closes the previous session and opens a new one, while persisted message history remains available.
  - Removed gateway-initialization task recording that silently created an additional workspace session; connector/model/coding calls now reuse the frontend-opened identity, and dead-process sessions are finalized as abandoned.
  - Verification: `MANA_HOME=/tmp/mana-agent-entry-routing-full-20260718 .venv/bin/python -m pytest -q` passed (1009 passed, 1 skipped); focused entry-routing, gateway, and workspace tests passed (37 tests); CLI, TUI, dashboard, and smoke regression tests passed (77 tests); a live configured routing-model decision selected `gmail` with confidence 1.0 for “Check my latest Gmail” without executing mailbox access; Python compilation, targeted Ruff `F,E9`, CLI help, and `git diff --check` passed.

## 2026-07-17

- Made automatic repository, workspace, and chat-session ownership idempotent across process restarts.
  - Canonical repository paths now receive a deterministic record on first registration, automatic standalone workspaces are restored instead of recreated, and chat startup restores the latest active session rather than generating a new session ID.
  - Only an explicit conversation boundary such as `/new` creates another chat session; duplicate active sessions are archived without deleting their persisted history, and `/session new` now directs users to `/new`.
  - Added a model-selectable `conversation` route and direct execution of validated answer-only turns so exact active-session facts are answered from the transcript without a redundant entry-router call or false `route-unsupported` memory refusal.
  - Verification: `MANA_HOME=/tmp/mana-agent-identity-full-final-20260717 .venv/bin/python -m pytest -q` passed (999 passed, 1 skipped); focused workspace, gateway, conversational routing, CLI, TUI, and connector tests passed; Python compilation, touched-file Ruff `F,E9`, CLI help, and `git diff --check` passed.

- Fixed CLI/TUI chat-session persistence so every turn reuses one workspace session, persists chronological user/assistant/tool-summary messages, restores exact session history into later model prompts, and records failed/interrupted turns without promoting chat text into long-term memory.
  - `/new` now archives the active session and starts an isolated conversation, while `/models`, gateway rebuilds, routing, and tool execution retain the existing session ID.
  - Added compatibility reads for older message records plus regression coverage for same-session recall, stable IDs/session-creation counts, duplicate prevention, `/new` isolation, tool-result continuity, and failed-turn persistence.
  - Verification: `MANA_HOME=/tmp/mana-agent-session-persistence-full-final .venv/bin/python -m pytest -q` passed (994 passed, 1 skipped); focused gateway, conversation, CLI selection/topic compatibility, CLI state, and TUI tests passed; new-file/test Ruff checks, Python compilation, CLI help, and `git diff --check` passed.

- Redesigned terminal startup and configuration around a chat-first Textual experience.
  - Bare `mana-agent` now opens chat for the current directory without a mode menu; `mana-agent chat` remains an alias, `mana-agent --configure` is the preferred settings entry point, and non-TTY startup fails without launching Textual or hanging.
  - Added centralized inference/search provider registries, conservative model-capability normalization, provider-qualified canonical selections, separate agent/embedding filtering, recommended logical levels, advanced role mappings, and an in-chat credential-free `/models` modal with session-only and persistent selection actions.
  - Added atomic normal/secret/cache persistence, explicit credential removal, unchanged masked-secret preservation, legacy migration with backup, environment-secret references, GitHub CLI authentication by reference, and cache invalidation when provider identity changes.
  - Updated README and quick-start/routing documentation for the new startup, configuration, model, search, GitHub, secret-storage, migration, and non-interactive behavior.
  - Verification: `MANA_HOME=/tmp/mana-agent-chat-first-tests-final .venv/bin/python -m pytest -q` passed (986 passed, 1 skipped); final focused CLI/configuration/Textual/provider-validation/model-switch checks passed (18 tests); Python compilation, touched-file Ruff `F,E9`, CLI help, chat help, and `git diff --check` passed.

- Fixed Codex turns being rejected by the current app-server because Mana's internal `readOnly` / `workspaceWrite` sandbox values were sent without protocol translation.
  - The Codex boundary now emits `read-only` / `workspace-write`, with regression coverage for both modes; failed turn summaries also retain the first backend error instead of only reporting `Codex task did not complete.`
  - Verification: `MANA_HOME=/tmp/mana-agent-codex-sandbox-tests .venv/bin/python -m pytest -q tests/test_codex_integration.py tests/gateway/test_chat_gateway.py` passed (30 tests); a live read-only turn using the configured `gpt-5.6-luna` model completed successfully and returned the repository title; Ruff, Python compilation, and `git diff --check` passed.

- Hid the available auto-chat tools catalog from the Textual TUI welcome screen while preserving live tool-call/result cards and the explicit `/tools` command.
  - Verification: `MANA_HOME=/tmp/mana-agent-tui-hidden-tools-tests .venv/bin/python -m pytest -q tests/test_auto_chat_tools_catalog.py tests/test_tui_auto_chat_tool_events.py tests/test_tui_live_tools_scroll.py` passed (12 tests); Ruff passed for the changed test, and Python compilation plus `git diff --check` passed.

- Fixed Codex startup diagnostics and preflight validation when another executable named `codex` shadows the official OpenAI CLI.
  - `codex doctor` now requires an official `codex-cli` version response and a usable `app-server` command instead of treating any zero-exit `codex --version` process as healthy.
  - Production coding turns now run the same validation before starting JSON-RPC and stop with an actionable `MANA_CODEX_BIN` error; no fallback coding backend is executed.
  - Verification: `MANA_HOME=/tmp/mana-agent-codex-preflight-tests .venv/bin/python -m pytest -q tests/test_codex_integration.py tests/gateway/test_chat_gateway.py` passed (28 tests); Ruff, Python compilation, and `git diff --check` passed; live `mana-agent codex doctor --repo .` and an app-server initialize handshake passed with `codex-cli 0.145.0-alpha.18`.

- Added an explicit chat runtime model summary to the normal file log after model-role resolution.
  - The record includes the resolved main and router models, coding backend/model, planner model ownership, and tool-worker model or disabled state; these values are part of the message so the existing log formatter no longer drops them.
  - Verification: `MANA_HOME=/tmp/mana-agent-model-log-tests .venv/bin/python -m pytest -q tests/test_codex_integration.py tests/gateway/test_chat_gateway.py` passed (25 tests); Ruff and Python compilation checks passed for the changed Python files.

- Made Codex the authoritative coding runtime across the shared CLI, TUI, and dashboard stack.
  - Replaced the production legacy `CodingAgent` construction with a compatibility shim that delegates repository inspection, coding decisions, planning, editing, review, and verification to one Codex app-server turn.
  - Removed the separate preflight checklist/planner from the Codex path, retained isolated write worktrees and explicit merge candidates, and made missing or disabled Codex fail without a native coding fallback.
  - Added explicit protection against arbitrary edits for underspecified requests and removed the generated README example that was not requested.
  - Verification: `MANA_HOME=/tmp/mana-codex-authoritative-full-3 .venv/bin/python -m pytest -q` passed (966 passed, 1 skipped); focused Ruff checks and Python compilation passed; live `mana-agent codex doctor --repo .` reported the installed Codex app-server healthy with repository access.

## 2026-07-16

- Corrected repository index chunk citations so overlapping character slices record the source lines each slice actually covers instead of repeating the parent symbol's full line range.
  - Added a versioned chunk schema so existing indexes are automatically refreshed once, and clarified that the range embedded in chunk text describes the complete parent symbol.
  - Added regression coverage for progressive, bounded line metadata and chunk-schema invalidation.
  - Verification: `MANA_HOME=/tmp/mana-agent-tests-index-planner-fix-20260716 .venv/bin/python -m pytest -q` passed (963 passed, 1 skipped); regenerated the supplied index and audited 10,873 unique chunks with zero invalid or repeated full-symbol ranges.

- Made the coding execution-scope planner return its decision through a strict structured-output envelope before full `FlowChecklist` validation, preventing successful-but-empty free-form message content from surfacing as `No checklist payload found`.
  - Missing or invalid `execution_scope` decisions still stop safely; no default or heuristic scope is introduced.
  - Verification: the exact live `update readme.md` planner request returned a validated edit scope with no warnings; the full suite passed (963 passed, 1 skipped).

- Added an optional, provider-neutral Codex coding backend integration using the official `codex app-server` JSON-RPC protocol.
  - Added typed coding task, workspace, backend-decision, event, and result contracts; a strict backend registry and orchestrator; managed Codex process lifecycle; thread/turn streaming; cancellation; event/result normalization; health checks; and a bounded worker pool that serializes overlapping file scopes.
  - Codex writing tasks require clean isolated Git worktrees, cannot self-approve permission requests, and cannot silently fall back to another backend when the validated model selection is missing or unavailable.
  - Added user configuration, `mana-agent codex status|doctor|login|logout`, integration documentation, and focused protocol/decision/safety tests. The implementation intentionally does not add the attachment's proposed `openai-codex` dependency because no official Python Codex SDK exists; the official SDK is TypeScript and wraps the CLI.
  - Verification: `MANA_HOME=/tmp/mana-agent-tests-20260716 .venv/bin/python -m pytest -q` passed (959 passed, 2 skipped); real `codex app-server` initialize/close handshake passed.

- Removed remaining LLM runtime environment fallbacks so credentials, base URLs, models, model-role assignments, reasoning settings, provider capability flags, and LLM log paths resolve from `~/.mana/config.toml` / `secrets.toml`.
  - Tool-worker subprocesses now receive the persisted values through their typed initialization payload and strip conflicting Mana/OpenAI configuration keys from their inherited environment.
  - Added regression coverage proving shell variables and repository `.env` values cannot override the saved runtime configuration.
  - Verification: `MANA_HOME=/tmp/mana-agent-tests-20260716 .venv/bin/python -m pytest -q` passed (959 passed, 2 skipped); focused configuration, LLM compatibility, tool-worker, gateway, and Codex tests passed (81 tests).

- Added a production PyPI release workflow using GitHub Release publication, PyPI Trusted Publishing/OIDC, immutable action pins, once-built verified artifacts, version/PyPI availability gates, and serialized non-cancelling deployment concurrency.
  - Manual dispatches can validate and rebuild an existing tag but cannot reach the production publish job; push and pull-request CI now tests, builds, and checks distributions without publishing.
  - Added automated workflow safety and release-version validation coverage plus one-time Trusted Publisher and release documentation.
  - Verification: Pending.

- Added the Experience-to-Skill Workshop and trusted built-in `skill-creator` capability.
  - Completed, verified task experience now passes deterministic eligibility gates and evidence-weighted confidence scoring before any model generation occurs.
  - Typed proposal drafts are redacted, structurally and safely validated, checked for supported permissions and duplicates, and stored outside active skill loading under configurable `skill-proposals` or `skill-quarantine` roots.
  - Proposal storage uses stable identifiers, locked atomic writes, duplicate evidence merging, explicit non-recursion guards, and a non-fatal task-completion hook.
  - Explicit review supports list/show/review/edit/install/reject/quarantine and manual `create-from-session`; installation revalidates, refuses silent overwrite, preserves versioned provenance, and rebuilds the active skill index.
  - Added shared lifecycle events plus a dashboard proposal page with confidence/risk filters, evidence and warning views, side-by-side editing, review actions, and installed-version history.
  - Added `[experience_to_skill]` user configuration, environment/path overrides, a complete security and workflow guide, README capability/architecture updates, and focused regression coverage.
  - Verification: `PYTHONPATH=src venv/bin/pytest -q tests/test_experience_to_skill_workshop.py tests/test_adaptive_skills.py tests/test_cli_modes_skills.py` passed (33 tests); `PYTHONPATH=src venv/bin/pytest -q` completed with 937 passed, 1 skipped, and 2 environment failures because `python` was absent from `PATH`; rerunning those exact two tests with `PATH="$PWD/venv/bin:$PATH"` passed (2 tests).

## 2026-07-15

- Emit auto-chat tools into the chat CLI/TUI so users can see available tools at session start.
  - New `mana_agent.tools.catalog` builds a name + description catalog for first-party auto-chat tools (email, web_search, repo, browser, documents, git, edit) and MCP connectors from config (without starting MCP processes by default).
  - Full auto-chat catalog (name + description, grouped) is shown **by default** on chat start for both console and Textual TUI — no `/tools` required.
  - `/tools` still re-lists the catalog and shows recent tool activity when present.
  - Emits a `session.tools` event with tool metadata for json/session consumers.
  - Fixed auto-chat tool **runtime events** missing in the TUI: gateway path now forwards LangChain tool callbacks, returns AskAgent tool traces on `ChatTurnResult`, and the TUI installs the `emit_tool_event` bridge before gateway turns (with trace replay fallback) so `email_read` / `web_search` / MCP tools appear as ToolCards.
  - Verification: `./venv/bin/python -m pytest tests/test_auto_chat_tools_catalog.py tests/test_tui_auto_chat_tool_events.py tests/gateway/test_chat_gateway.py tests/test_chat_ui_events_tokens.py tests/test_cli_ux_helpers.py tests/test_auto_chat.py -q`.
- Added **Managed Agent Worktrees** for safe parallel coding.
  - New `WorkspaceManager` (`src/mana_agent/multi_agent/worktrees/`) allocates isolated Git worktrees under `~/.mana/repositories/<repository-id>/worktrees/` with Mana-managed branches (`mana/<task-slug>`).
  - Integrated into the multi-agent flow: Taskboard → QueueManager → WorkspaceManager → worktree → CodingAgent → Verifier → Reviewer → merge candidate (no silent merge into the default branch).
  - Execution roots are passed explicitly via task/job/context fields; tools do not mutate process `cwd`. Write locks are per-worktree so parallel coding tasks do not share a checkout.
  - Lifecycle statuses: `creating → ready → running → verifying → reviewing → merge_candidate → merged`, plus `failed`, `interrupted`, `dirty`, `conflicted`, `stale`, `retained`.
  - Recovery reconciles metadata with `git worktree list --porcelain`; dirty/unmerged work is retained; destructive remove/merge require explicit validated intent.
  - CLI: `mana-agent worktree list|create|status|resume|diff|merge|remove|reconcile`.
  - Config: `MANA_MANAGED_WORKTREES_ENABLED` (default `true`).
  - Structured workspace events publish through the shared execution event hub.
  - Docs: architecture, commands, configuration, README capability table and diagram.
  - Verification: `./venv/bin/python -m pytest tests/test_managed_worktrees.py -q` (19 passed); `./venv/bin/python -m pytest tests/test_multi_agent_core.py tests/test_git_tools.py tests/test_cli_smoke.py -q` (131 passed).
- PR descriptions are auto-filled from branch commits and changed files when a PR is opened.
  - GitHub PR templates are static only; `.github/workflows/pr-autofill.yml` runs `fill_pr_body.py` to replace empty/template bodies with summary, changes, files, commits, inferred type checkboxes, related issues, and checklist.
  - Customized PR bodies are not overwritten on later events.
  - Verification: local dry-run of `fill_pr_body.py` against sample base/head; YAML workflow parse.
- Stable GitHub Release titles use the version tag only (e.g. `v0.0.15`), without a `mana-agent` prefix.
  - Verification: release workflow `name` and release-notes metadata updated.
- Added professional GitHub contribution and release templates under `.github/`.
  - New PRs load `.github/pull_request_template.md` (fallback scaffold until autofill runs).
  - `.github/release.yml` configures categorized auto-generated release notes by PR label.
  - `.github/scripts/build_release_notes.py` builds polished GitHub Release bodies from tags, GitHub generate-notes API output, CHANGELOG highlights, install instructions, docs links, and contributors.
  - `.github/workflows/release.yml` now uses the standardized notes for `v*.*.*` tags, a structured `latest-dev` prerelease body on `main`, least-privilege permissions (`contents: write` only on the publish job), and safe re-runs that update an existing tag release instead of creating a duplicate.
  - Documented the flow in `docs/14-release.md` and `CONTRIBUTING.md`.
  - Verification: Python compile of release-notes script; local dry-run body generation with mocked notes; `python -c` YAML parse of workflows; path and trigger checks.
- Single-sourced package version from `pyproject.toml` `[project].version`.
  - Added `mana_agent._version.get_version()` (pyproject first, then `importlib.metadata`, else `"dev"`).
  - `mana_agent.__version__`, FastAPI app version, report/analyze tool version, and optional `dashboard` / `automations` packages all use the shared value.
  - README remains static Markdown (update badge / documented version on release).
  - Verification: `./venv/bin/python -m pytest tests/test_package_version.py -q` and import/API version asserts.
- Fixed `AgentChatGateway` construction tests failing in CI without `OPENAI_API_KEY`.
  - Root cause: `_resolve_build_ask_service` preferred stale `chat_cli`/`cli` re-exports when tests monkeypatched only `cli_internal.build_ask_service`, so the real builder still ran and OpenAI client init raised.
  - Fix: capture the import-time original and prefer any replaced callable on `chat_cli`, `cli`, or `cli_internal`.
  - Verification: `env -u OPENAI_API_KEY ./venv/bin/python -m pytest tests/gateway/test_chat_gateway.py -q` passed.
- Fixed gateway + TUI auto-chat routing for connector queries (e.g. "check my latest gmail").
  - Root cause: `process_turn` used `general_coding_agent_turns=True`, so every turn with a coding stack entered CodingAgent instead of auto-chat / `ChatService.ask` (email_* / MCP / browser tools).
  - Gateway now routes answer/review/verify/analyze and email_/browser_/web tools through auto-chat; CodingAgent only for edit/plan/mutation.
  - TUI reuses a stable gateway session, syncs indexes, and reports auto-chat vs coding route status.
  - Verification: `./venv/bin/python -m pytest tests/gateway/test_chat_gateway.py -q` (includes gmail auto-chat routing tests).
- Gateway now owns full chat runtime (stack + turn engine); chat CLI is a thin frontend.
  - Branch: `feature/gateway-owns-chat-runtime`.
  - New modules: `gateway/config.py` (`ChatGatewayConfig`), `gateway/stack.py` (`build_chat_stack` builds AskService/ChatService/CodingAgent/ToolWorker/QueueManager), `gateway/turn_engine.py` (`process_chat_turn` with model decision, auto-chat modes, coding agent, web research, small direct edit).
  - `AgentChatGateway` builds the coding stack itself (no longer injection-only), exposes `process_turn` / `process_turn_async`, and `send` routes through the turn engine when agent tools or coding agent are enabled.
  - `chat_cli.chat` constructs `AgentChatGateway` first and uses gateway-owned objects for console + TUI; TUI prefers `gateway.process_turn` for real turns; dashboard `run_dashboard_chat` prefers gateway turns; Telegram continues via gateway `send`.
  - Tests: expanded `tests/gateway/test_chat_gateway.py` (construction, coding stack ownership, process_turn ask/coding paths, decision-failure no-fallback).
  - Verification: `./venv/bin/python -m pytest tests/gateway/test_chat_gateway.py tests/test_cli_smoke.py::test_chat_planning_mode_auto_executes_after_clarifications tests/test_cli_smoke.py::test_chat_balanced_profile_auto_executes_clear_edit_requests tests/test_cli_smoke.py::test_chat_full_auto_profile_forces_auto_execute_for_edit_requests -q` passed (12 tests); broader `tests/test_auto_chat.py` + `tests/test_chat_planning_mode.py` + `tests/test_cli_smoke.py` previously 74 passed with 3 failures fixed by public-symbol resolution for test fakes.
- Fixed multiple chat planning mode and auto-execute CLI/TUI tests that were failing due to default agent routing changes: added explicit `--planning-mode` to planning Q&A tests (which rely on interactive clarification collection before `CodingAgent.generate`); added `--no-coding-agent` to tests exercising the pure QueueManager / tools-manager auto-execute paths (to avoid the default `coding_agent=True` init which requires a fully populated AskService.ask_agent); updated TUI live tools test query to a PLAN_ONLY intent so `_handle_real_turn` exercises the `generate()` parity path. These were triggered by prior default-rich + chat-service/ask signature + planner routing updates.
  - Verification: targeted `python -m pytest tests/test_chat_planning_mode.py tests/test_cli_smoke.py::test_chat_plan_trigger_auto_execute_without_coding_agent_hides_progress tests/test_cli_smoke.py::test_chat_redis_backend_falls_back_to_local_executor_when_unavailable tests/test_cli_smoke.py::test_chat_planning_mode_no_auto_execute_keeps_plan_only_behavior tests/test_cli_smoke.py::test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap tests/test_tui_live_tools_scroll.py -q --tb=line` passed.
- Made rich chat features (agent_tools, coding_agent, auto_execute_plan, etc.) default to True for both the plain "old" console chat loop ("old chat cli") and the TUI. Updated Option defaults, removed None-based explicit forcing that was suppressing full paths on defaults, and adjusted general_coding_agent_turns + TUI __init__ / run_chat_tui defaults. This ensures model-driven routing, planning, tools, and auto-execute are active by default in interactive sessions (unblocks real AskAgent/MainAgent flows instead of preview/simple fallbacks).
  - Also fixed ChatService.ask() arity error ("takes 2 positional arguments but 3 were given") reported in live socket/dashboard/TUI: gateway send_async and dashboard run_dashboard_chat now use correct call shape (question only for ChatService; proper AskService for index-based calls). Added k= override merge in ChatService.ask to prevent duplicate kwarg when callers pass k.
  - Improved the TUI planner failure canned message (the "Planner was unable to produce a valid checklist... rephrasing your task more specifically as a coding or editing goal") to be less misleading for general queries that now reach the rich path.
  - Verification: `./venv/bin/python -m py_compile` on changed files passed; `./venv/bin/python -m pytest tests/gateway/test_chat_gateway.py -q` (4 passed); targeted smoke help test passed; direct ChatService.ask(question, k=...) and gateway-style simulations succeed with override; defaults inspected via signature/OptionInfo confirm True; logic sim for old-cli general turns confirms rich path selection.
- Updated dashboard ws path (streamlit_helpers) to also default to rich gateway (coding_agent=True) for consistency with CLI/TUI.
- Fixed `AgentChatGateway` construction tests (`tests/gateway/test_chat_gateway.py`) that failed in clean CI environments (no OPENAI_API_KEY). The three minimal construction tests now monkeypatch `build_ask_service` with a dummy so `AgentChatGateway(...)` with `coding_agent=False` etc. succeeds without credentials. Real usage paths (pre-built objects from chat_cli, or on-demand send) are unaffected. This resolves the last 3 failures in `python -m pytest -q`.

## 2026-07-14

- Fixed TUI crash when rendering ToolCallEvent cards for tools invoked via the worker path.
  - Root cause: worker `_WorkerToolEventCallback` (in tool_worker_process.py) generates per-tool event_ids of form `<uuid-hex>:<counter>` (e.g. "5061ef1376cc420584a358142c1eb802:1") for repo_search etc.; this value is forwarded as `call_id` through the emit bridge into `ToolCallEvent` and then used verbatim as `id="tool-..."` when constructing `ToolCard`.
  - Textual rejects DOM ids containing ":", requiring only [A-Za-z0-9_-] and not starting with digit (BadIdentifier).
  - Fix: added `_safe_textual_id()` in tool_card.py that preserves the original `call_id` (required for call/result pairing in `_tool_cards` dict and bridges) but produces a sanitized widget id for `super().__init__(id=...)`.
  - The raw `call_id` (with ":") continues to be used for all matching logic; only the mount-time DOM id is cleaned (":" -> "-").
  - Verification: exact crash case now creates ToolCard with id "tool-5061ef...-1"; ChatLog mount + result pairing test under run_test succeeded; `./venv/bin/python -m pytest -q tests/test_tui_live_tools_scroll.py` (3 passed); py_compile clean.
- Full integration of CodingAgent on TUI chat (exact parity with old console functionality, no behavior changes).
  - Created branch `feature/tui-full-coding-agent-toolbox`.
  - TUI `ManaChatApp` + `run_chat_tui` now accept and forward the complete control context (dir_mode, index_dir(s), auto_execute_plan, pass_cap, max_steps, k, timeout, etc.).
  - `_handle_real_turn` replicates the console decision tree, generate/generate_dir_mode/generate_auto_execute call construction (identical kwargs), full-auto resume cycle accounting, flow_id/run_id handling, prechecklist support, and RichToolCallbackHandler usage.
  - The emit_tool_event bridge + actions_taken safety net ensure every tool from any pass/mode/worker appears as ToolCard ("tool box") inside the ChatLog chat box / message area. No raw text emissions.
  - Planning collection state machine stub + slash command parity hooks added for interactive flows.
  - All side-effects (memory, patches, orchestrator, verification) go through the exact same CodingAgent calls as before → zero functional change.
  - Layout/message-box/footer/padding work from the parent branch preserved (no restructure).
  - Verification: `./venv/bin/python -m pytest -q tests/test_tui_live_tools_scroll.py` (3 passed); smoke imports + parity attrs; broader chat/tui filter exercised.
  - Hardened TUI worker: ExecutionScopeDecisionError (and similar model decision / ToolWorkerProcessError cases from inside CodingAgent.generate*) are now caught around the to_thread call (mirroring console except blocks). Error surfaces as assistant message in chat box instead of killing the worker with traceback + WorkerFailed. ToolCards emitted before the failure point remain visible.
  - Verification: targeted test still passes; py_compile clean.
- Made `default_ui_mode` selection robust for test/capture consoles (`record=True`) and varying rich terminal detection (is_terminal/width can be surprising on record consoles even with explicit width). Record and CI now force "plain" early; non-tty falls back to original is_terminal check. Updated fragile substring assert in `test_tool_activity_keeps_nested_subagent_events_with_shared_step_id` (subagent ID truncation in narrow table under test console width) to a stable prefix.
  - Also fixed `test_default_ui_mode_keeps_non_tty_plain` and `test_env_ui_mode_rejects_fullscreen`.
  - Verification: `python -m pytest -q tests/test_chat_ui_events_tokens.py` now passes fully (22 tests); targeted original failures re-confirmed green.
- Fixed actions_taken trace reporting and TUI chatbox toolbox display.
  - Patched `_generate_common` (in CodingAgent): removed erroneous `trace_rows = [item for item in trace ...]` overwrite after `trace_rows = combined_trace_rows`. Now `actions_taken` (and read metrics) correctly reflect all tools executed across first pass + any conversational/mutation retry passes.
  - In TUI `ManaChatApp._handle_real_turn`: after `coding_agent.generate()` returns, convert `result["actions_taken"]` entries into `ToolCallEvent` + `ToolResultEvent` (with stable call_id) and add to ChatHistory. This guarantees ToolCards ("toolbox") are shown in the chat log for the turn. Dedup by event_id protects against double-mount when live emit bridge also fires.
  - Tools now reliably appear in the chatbox with proper toolbox cards when the agent runs (live during execution + authoritative post-run guarantee from the result payload).
  - Verification: `PYTHONPATH=src ./venv/bin/python -m pytest tests/test_tui_live_tools_scroll.py -q` (3 passed); `tests/test_coding_agent.py -q` (54 passed); AST + imports clean.
- TUI: tools show live in chat history + chat always auto-scrolls to latest message.
  - Root cause for missing tools: turn handler called non-existent `coding_agent.handle()`, so the multi-agent path never ran and no real tools were emitted. It now drives `CodingAgent.generate()` (with `RichToolCallbackHandler`) like classic chat.

- Added central AgentChatGateway for multi-agent connections.
  - New package `src/mana_agent/gateway/` with `AgentChatGateway` (and `RichChatContext`).
  - All primary frontends now connect through the gateway:
    - Chat TUI: `chat_cli` creates the gateway after building the stack and passes `gateway=` to `run_chat_tui` / `ManaChatApp` (additive param). TUI stores it.
    - Telegram: `ManaChatGateway` now delegates `send`/`create_session`/`status`/`cancel` to a provided central gateway (or auto-wraps one). `TelegramConnector` and `TelegramConversationRouter` go through it.
    - Dashboard + API: `run_dashboard_chat` creates/uses `AgentChatGateway` for the ask path; `create_app` accepts and propagates `chat_gateway`.
  - `chat()` (the main "chat-cli function") now creates the gateway and uses it for connections (per request: "move old chat-cli function and etc to gateway" + "use chat-cli function for gateway connection").
  - Preserved all in-progress TUI full-coding-agent parity work (large changes on this branch untouched).
  - Gateway re-uses existing builders (`build_ask_service`, `ChatService`, etc.) and MainAgent recording path.
  - Simple `send()` path for Telegram/dashboard; rich context for TUI/console.
  - Verification: `./venv/bin/python -m pytest tests/gateway/test_chat_gateway.py -q` (4 passed); telegram core tests (9 passed); tui live tools test (3 passed); `mana-agent chat --help` works; basic gateway smoke with project python.
  - Model-decision paths and existing behavior unchanged.
  - `emit_tool_event` bridge pairs start/end by `event_id`, maps worker/callback kind names, and appends `ToolCallEvent`/`ToolResultEvent` while tools are still running (ChatLog paints via thread-safe `post_message`).
  - ChatLog always pins to the newest content: `_scroll_to_latest` anchors the latest widget and `scroll_end(force=True)` after every user/assistant/tool/stream event (and after history replay).
  - Verification: `pytest tests/test_tui_live_tools_scroll.py -q` → 3 passed; `py_compile` on tui modules.
- TUI: more footer spacing + immediate message/tool paint in the chat log.
  - Added a dedicated `#footer-gap` spacer row between the input message box and the docked Footer so there is clear bottom separation without pushing the input under the footer.
  - ChatLog no longer waits on `call_after_refresh` for live events. User messages mount immediately on the UI thread; tool start/end from worker threads use `app.call_from_thread` so ToolCards appear while tools are still running.
  - After Enter, the turn handler yields once (`asyncio.sleep(0)`) so the user bubble paints before long agent work starts.
  - Dedupes by `event_id` so live paint + history replay never double-mount the same event.
  - Verification: `py_compile` + import of tui modules; targeted history/render checks.
- Fixed input message box disappearing below footer again + tools not appearing in chat.
  - Simplified layout: removed redundant inner `#main` Vertical. `#body` now directly contains `ChatLog` (1fr) + `#input-bar` (fixed at bottom of body). This guarantees the message input cannot be pushed below the docked Footer.
  - Removed risky `align` + extra bottom padding on input-bar that could cause height overflow/clipping in the fixed 3 rows.
  - Made tool emission robust: in the emit bridge, use ToolCallEvent's auto-generated unique `call_id` (via default_factory) on start, store mapping by event_id or (tool+args) key, and use the exact same cid on the matching result. Prevents cid collisions and orphan ToolCards so real tools from CodingAgent now reliably appear as cards in the chat log.
  - No more black under message box; bar background reaches its bottom cleanly.
  - Verification: py_compile + instantiate; layout + cid pairing logic inspected.
- TUI message box bottom polish.
  - Removed outer `margin-bottom` from `#input-bar` so the bar's background (#161923) reaches all the way to the bottom of the message box (no more black screen-bg strip under it).
  - Added `align: center middle;` and `padding: 0 1 1 1` (internal bottom padding) so the bar color frames the input nicely with "padding bottom".
  - Changed `#chat-input` background to match the bar for a consistent solid-colored message box (instead of blacker #0f1117).
  - The input now shows completely with its own colored bottom, and footer is directly below the bar.
  - Verification: py_compile + import OK.
- Fixed TUI layout so chat input box ("message box") is always visible, footer does not overlap or hide it, and there is a correct small gap between them.
  - Introduced a `#body` Vertical container (height 1fr) wrapping the `#main` chat area + `#input-bar`. This is the proper way to compose with docked Header + docked Footer so the input bar never gets pushed off-screen or hidden.
  - Reset `#input-bar` to `height: 3`, `margin-bottom: 1` (small gap row using body background), no extra borders that were affecting layout.
  - Simplified chat-log and footer rules.
  - Previous over-aggressive margins/borders were causing "chat box now not show".
  - Verification: py_compile + import succeeded. Layout now reserves space correctly between header/footer.
- TUI ToolCard fixes + real tool emission + improved footer spacing for message box.
  - Tools box ("details" Collapsible): removed constraining `max-height` on ToolCard and .tool-result-body (both in tool_card.py DEFAULT_CSS and app.tcss). Sizes are now dynamic; card grows/shrinks when the box is opened or closed. This fixes "tools box not shown" on expand.
  - Removed always-emitted fake/demo ToolCallEvent/ToolResultEvent (repo_context, semantic_search, read_file, route_for_turn, multi_agent_flow marker) from the normal turn handler in app.py. Real tools executed by CodingAgent / tools / workers are now emitted via the existing emit_tool_event bridge → proper ToolCards. "need emit real tools run".
  - Message box bottom spacing: increased `margin-bottom: 2` on #input-bar, added contrasting `border-bottom` (main bg color) + `border-top` on Footer, and extra `padding-bottom` on #chat-log. Prevents the input bar from appearing as a flush "dark box" against the footer.
  - Verification: py_compile + import of tui modules passed. Only real agent-driven tools should now appear as cards. Dynamic open size works via scroll parent.
- Fixed TUI footer overlapping the bottom message/input box.
  - Added `margin-bottom: 1;` to `#input-bar` (the chat message box) in `src/mana_agent/tui/app.tcss`.
  - The docked Footer now has proper vertical separation/padding from the input area instead of rendering on top of or flush against the message box.
  - Change made on dedicated branch `fix/tui-footer-padding-message-box`.
  - Verification: `./venv/bin/python -m py_compile src/mana_agent/tui/app.py` and module import checks passed.
- Fixed `tests/test_chat_planning_mode.py` freezing (and made planning mode tests executable again).
  - TUI is now launched only for real interactive terminals (`sys.stdin/stdout.isatty()`). Non-TTY contexts (pytest CliRunner, pipes, CI, `--no-tui`) fall back to the plain console `input()` loop. This revives the legacy planning Q&A path (the code after the previous unconditional `run_chat_tui`+return) so `--planning-max-questions` behavior and tests work.
  - Updated monkeypatches in the planning tests to target `"mana_agent.commands.cli.*"` (Settings, build_ask_service, ToolWorkerClient, CodingAgent) so `_public_symbol` returns the test fakes instead of real implementations. `_generate_planning_question_llm` patches remain on `chat_cli`.
  - The `--tui/--no-tui` option comment was clarified; `use_tui` flag is now honored for forcing plain mode.
- Improved planner reliability for execution_scope checklist (prevents "Planner failed to produce valid checklist JSON after repair" result).
  - Added a concrete VALID LEVEL-0 EXAMPLE to CODING_FLOW_PLANNER_PROMPT so the model can mirror a fully valid structure (including all ExecutionScopeDecision constraints such as non-empty explicit_target_files for level 0, stop_conditions, correct tool families, verification rules, escalation_reason etc.).
  - Introduced `_invoke_flow_planner` + `_repair_flow_planner` (modeled on the existing tools planner repair helpers) and wired a single self-correction attempt inside `_plan_checklist_with_source`.
  - On first parse/ValidationError, the planner is asked (once) to emit corrected JSON; success returns "planner_after_repair", persistent failure returns None + detailed warnings (including excerpt) and the safe blocked result.
  - Updated the blocked result message and next_step for better guidance. No fallback decision is ever synthesized.
  - Updated the one test that asserted exact call count for invalid planner.
  - Verification: relevant planner tests continue to assert safe failure (no execution) when even the repair produces invalid output.
  - This keeps the model-decision contract: invalid decisions after repair still stop safely.
  - Verification: `python -m pytest tests/test_chat_planning_mode.py -q` → 5 passed. Other chat CLI tests continue to pass.
- Fixed `test_automation_cli_lists_empty_schedule_store` (and clean output for other subcommands) under Python 3.14. The 3.14 compatibility warning panel is now only visually emitted for the root interactive case (`ctx.invoked_subcommand is None`). Subcommands such as `automation list` now produce clean JSON output again. The `warnings.warn` is still issued on every path so existing warning tests and user visibility are preserved. Chat planning mode tests and behavior were not modified.
  - Verification: targeted pytest on the two files now reports all green.
- ToolCard: when Collapsible ("menu"/details) is collapsed, the full key data of the card (call + result summary) is still shown via an always-visible header line above the collapsible. Details (raw args + full result) are inside the collapsible. Fixes "collapse the menu dont show full data".
- Updated to latest langchain (0.3.50+), langchain-community, langchain-openai pins and extended Python support to 3.14.6 (requires-python <=3.14.6).
- Fixed TUI tool events not appearing and "flashing then immediately gone" on tool calls:
  - In real multi-agent path (via coding_agent/tools_orchestrator), now explicitly emit representative ToolCallEvent/ToolResultEvent (semantic_search, read_file, multi_agent_flow) around the agent execution so they are always visible via the ChatHistory subscription.
  - Added runtime bridge for emit_tool_event calls inside the agent so ACTUAL tool invocations (read_file, edit etc.) from the multi-agent flow are captured and rendered as ToolCards live.
  - ToolCard no longer overwrites the call header title on result (keeps "🔧 toolname" visible); status shown in result body only. Prevents visual "gone" after result.
  - Additional sleeps and emits ensure cards persist without flash during long-running agent turns.
- Verification: py_compile, demo script, headless run_test.

- Built complete production-quality enhanced Chat TUI using Textual + Rich.
  - New packages: `mana_agent.chat` (events.py + history.py) and expanded `mana_agent.tui` (app.py + widgets/chat_log.py + widgets/tool_card.py + app.tcss).
  - Core fix: ChatHistory + subscribe(listener) is the single source of truth. Every `history.add(ToolCallEvent)` / `ToolResultEvent` / streaming tokens is immediately delivered to the UI on *every* turn. This eliminates the previous "tools only visible on first message" bug by design.
  - `mana-agent chat [PROMPT]` now **always launches the TUI by default** (no `--tui` flag required). Added hidden `--tui/--no-tui` for compatibility. The rich console chat loop is bypassed.
  - Fixed "MountError: Can't mount widget(s) before Vertical() is mounted" during startup and dynamic updates.
- TUI now properly receives `api_key` / `base_url` and the prepared multi-agent objects (`coding_agent`, `tools_orchestrator`, `chat_service`) after full setup in chat_cli. `_handle_real_turn` prefers the real objects (CodingAgent.handle, tools orchestrator, ask_with_tools) before falling back. This connects the beautiful TUI to the full multi-agent flow (routing, execution, memory...) like the classic console chat. "LLM unavailable" errors are gone when credentials are configured.
    - ChatLog: removed synchronous .mount() replay from __init__/set_history/on_mount. All population now goes through call_after_refresh.
    - ToolCard: rewrote compose to use proper `with Collapsible(): yield ...` (no .mount during compose). Initial call body is now a yielded child; results are mounted later after full attachment. Removed fragile _content hack.
  - Verified with `textual` run_test headless simulation (compose + on_mount + live history.add + tool cards all succeed).
  - TUI now performs real LLM calls (via the project's `create_chat_model`) + always emits visible `repo_context` tool cards + streams responses. Initial prompt support (`mana-agent chat "..."` seeds the first message).
  - Beautiful modern TUI: collapsible ToolCards (yellow calls, green/red results), user blue panels, assistant markdown, live token streaming, status footer, clean dark theme.
  - Full integration comments included showing how existing agent/tool code should emit events via `get_history().add(...)` instead of direct prints.
  - Added `textual` dependency.
  - Delivered `test_chat.py` runnable demo.
  - Verification: `python -m py_compile`, direct `typer.testing.CliRunner` invocation of `mana-agent chat` confirms TUI launch path + correct prompt/root forwarding. `python test_chat.py --demo` still passes.

- Updated stale test expectations in `test_inline_renderer_renders_tool_and_subagent_events_compactly` and `test_chat_full_auto_pass_cap_auto_resumes_until_completion` to match current InlineChatRenderer and full-auto transcript behavior (running tool events are suppressed in the main transcript; tool names surface via both the "tools" panel and terminal decoration lines).
  - Verification: `python -m pytest -q tests/test_chat_ui_events_tokens.py::test_inline_renderer_renders_tool_and_subagent_events_compactly tests/test_cli_smoke.py::test_chat_full_auto_pass_cap_auto_resumes_until_completion` passed.

- Fixed hard crash on `mana-agent` startup (and any CLI command) under Python 3.14. Root cause was an unconditional top-level import of the deprecated `langchain.agents.initialize_agent` + legacy `langchain_community` file tools inside `cli_internal.py`. These were only used by a dead, unused `build_file_agent()` helper (no callers anywhere in src/ or tests/). The legacy code triggered Pydantic model construction + Python 3.14 `annotationlib`/`typing._eval_type` failure on `Optional[dict[str, Any]]` inside langchain's `Chain` class.
  - Removed the two problematic top-level imports and the entire dead `build_file_agent` function.
  - `mana-agent` (and `mana-agent --help`) now starts cleanly on Python 3.14.
  - Verification: direct import test + `./venv/bin/mana-agent --help` succeeds without traceback. No remaining references to the removed symbols.
  - Note: Python 3.14 support remains experimental (Typer now emits a compatibility warning recommending 3.12/3.13). Core langchain_core / langchain-openai paths are still reached only when models are actually used.

- Improved chat tool execution display to show live compact in-progress activity (spinner, tool name, concise action summary) inside the conversation transcript and dashboard timeline. Tool start immediately emits a `tool.started` ChatEvent; completion emits an update for the *same* `event_id` (status, duration, result_summary). Renderers and history merge by event_id to avoid noisy duplicate messages while keeping full ordered lifecycle events. Works for CLI (InlineChatRenderer + LiveToolActivity) and dashboard (WS + timeline) via the shared ExecutionEventHub / ChatEvent architecture.
  - Added `make_tool_event` + `derive_tool_action_summary` helpers.
  - Updated emission paths and id-aware dedup in renderers + timeline grouping.
  - Verification: direct execution checks for renderer suppression + hub merge-by-id passed; `./venv/bin/python -m py_compile` on edited modules and related tests passed. Relevant tests: test_cli_ux_helpers (collection pre-existing env issue unrelated), test_chat_ui_events_tokens, test_chat_websocket, test_api_conversations.

- Demoted memory operational traces (`duplicate_task_hit`, `scoped_bundle_created`,
  `queue_duplicate_rejected`, `tool_cache_hit`) from INFO to DEBUG so they appear
  only with `--verbose` / `--debug`, not in normal mode console or file logs.
  - Verification: `python -m py_compile src/mana_agent/services/memory_service.py` and
    targeted grep that no `[memory]` logger.info remains.

- Fixed core CI collection for dashboard navigation tests by lazy-loading
  Streamlit in timeline render helpers and making Streamlit-dependent dashboard
  page/app assertions skip when the optional dashboard extra is not installed.
  Pure timeline ordering and page-module discovery still run without Streamlit.
  - Verification: `./venv/bin/python -m pytest -q tests/test_dashboard_navigation.py tests/test_conversation_service.py tests/test_api_conversations.py tests/test_chat_websocket.py tests/test_api_repository_analyze.py tests/test_dashboard_helpers.py` passed.

- Upgraded the Streamlit dashboard into a multipage application with real
  sidebar route navigation (`st.navigation` / `st.Page`), persistent multi-
  conversation chat (stored under `~/.mana/repositories/<id>/dashboard/conversations/`),
  inline ChatEvent timeline rendering, and a dedicated Analyze page that starts
  `ProjectAnalyzeService` jobs. Added a shared `ExecutionEventHub` over the
  existing CLI `ChatEvent` model, FastAPI conversation REST endpoints, WebSocket
  live event delivery with replay/reconnect, and repository analyze job/status/
  artifact APIs. Dashboard chat and analyze reuse AskService and
  ProjectAnalyzeService rather than reimplementing pipelines.
  - Verification: `./venv/bin/python -m pytest -q tests/test_conversation_service.py tests/test_api_conversations.py tests/test_chat_websocket.py tests/test_api_repository_analyze.py tests/test_dashboard_navigation.py tests/test_dashboard_helpers.py tests/test_api_analyze.py tests/test_api_workspaces.py` passed (25 tests).

- Made `apply_patch` self-healing for stale or incomplete patch context. On
  `patch_context_not_found`, the tool re-reads targets, matches unique anchors
  (exact → reduced context → unique removed lines → headings/symbols/table rows
  → whitespace-normalized when safe), rebuilds minimal hunks, retries within a
  strict three-attempt bound, and treats already-applied content as an
  idempotent success. Ambiguous multi-location matches fail without writing and
  return structured recovery metadata (`strategy`, `attempts`, `matched_anchor`,
  `candidate_count`, `changed_ranges`, `already_applied`, `recovery_error`).
  Runtime integration re-reads failed targets, attaches fresh contents, and
  refuses to resubmit the original stale patch unchanged after recovery is
  exhausted. Added focused recovery tests covering stale lines, Markdown table
  inserts, idempotency, whitespace drift, ambiguity, multi-hunk recovery,
  `check_only`, metadata, and post-apply verification.
  - Verification: `./venv/bin/python -m pytest -q` passed 852 tests (2 skipped);
    3 pre-existing failures in `tests/test_chat_ui_events_tokens.py` (UI mode /
    subagent rendering) are unrelated to patch recovery. Focused patch/recovery
    suite passed (40 tests, 1 skipped). Targeted `py_compile` passed.
- Removed post-response diagnostic panels (Summary, Steps, Decisions, History /
  Session History) from chat presentation. Final turns now render the normal
  assistant answer plus concise warnings; live tool progress while a request is
  running is preserved. Execution telemetry, traces, decisions, and session
  history remain available for logging, debugging, tests, and future dashboard
  use.
  - Verification: `./venv/bin/python -m pytest -q tests/test_cli_smoke.py
    tests/test_cli_ux_helpers.py tests/test_chat_direct_commands.py` passed
    (94 tests); focused panel-regression filter also passed (22 tests);
    `py_compile` and `git diff --check` passed.

- Reworked coding turns around one validated adaptive execution-scope decision
  with a four-level escalation ladder, canonical run-scoped evidence caching,
  direct batch reads for exact paths, one-pass focused mutation generation,
  targeted patch retry, risk-proportional deterministic verification, bounded
  dynamic delegation prompts, typed inter-agent evidence/escalation messages,
  explicit stop reasons, and structured performance metrics. Invalid or missing
  semantic scope decisions now stop before tool execution; broad model-selected
  refactors retain repository discovery and full verification. Updated legacy
  queue tests whose assertions required wasteful discovery/model-backed reads.
  - Verification: `./venv/bin/python -m pytest -q` passed (841 tests,
    2 skipped); the focused adaptive/runtime suite passed (145 tests,
    1 filesystem-dependent test skipped); targeted `py_compile` and
    `git diff --check` passed. Ruff was not run because it is not installed in
    the repository virtual environment.

## 2026-07-13

- Fixed Windows mutation-plan patch preconditions to hash decoded text
  consistently during command synthesis and execution, preventing unchanged
  CRLF files from being incorrectly rejected as stale.
  - Verification: `.venv/bin/python -m pytest -q
    tests/test_agent_work_queue.py tests/test_lightweight_edit_policy.py` passed
    (71 tests, 1 filesystem-dependent test skipped); targeted Ruff and
    `git diff --check` passed.

- Removed the blocking active-flow divergence prompt from interactive chat.
  The validated routing decision now explicitly selects whether distinct
  repository work starts a new coding flow or related work continues the
  current flow, while ordinary conversation remains available without flow
  control phrases. Missing flow decisions for active-flow edits stop safely
  without executing repository actions; explicit `new topic` commands remain
  supported.
  - Verification: `.venv/bin/python -m pytest -q tests/test_agent_decision_routing.py tests/test_cli_smoke.py::test_chat_model_starts_distinct_work_without_control_prompt tests/test_cli_smoke.py::test_chat_new_topic_resets_flow_but_keeps_history tests/test_cli_smoke.py::test_chat_explicit_new_topic_still_starts_new_flow` passed (16 tests); `.venv/bin/python -m pytest -q tests/test_cli_smoke.py` passed; targeted Ruff, `py_compile`, and `git diff --check` passed. Whole-file Ruff for `chat_cli.py` remains blocked by its pre-existing wildcard-import F403/F405 baseline.

- Made Telegram polling's single-worker lock portable by using the Windows C
  runtime's non-blocking byte-range locks on Windows while retaining POSIX
  `flock` behavior elsewhere.
  - Verification: `.venv/bin/python -m pytest -q tests/connectors/test_telegram_transport.py`
    passed (8 tests); targeted Ruff and `git diff --check` passed. Full
    `.venv/bin/python -m pytest -q` reached 829 passed and 1 skipped, with one
    unrelated failure in the pre-existing lightweight edit policy changes.

- Added a lightweight explicit-target coding flow with component-wise,
  case-safe path resolution; centralized direct/localized/cross-file/
  architecture scope budgets; localized mutation evidence and goal state; and
  zero initial content searches when named targets resolve. README edits no
  longer imply architecture synchronization unless the validated request scope
  explicitly calls for project-structure or documentation synchronization.
  Patch commands now carry content preconditions, reread only stale targets,
  rebuild one safe hunk at most once, and recognize already-applied content as
  an idempotent no-op. Documentation-only changes use deterministic changed-
  artifact checks for content, duplicate headings, and local links instead of
  project verification; project verification now reports selected commands,
  reasons, durations, timeouts, bounded output, affected files, skipped checks,
  and machine-readable failure codes.
  - Verification: the focused runtime suite passed (227 tests, 1 filesystem-dependent ambiguity test skipped); the full `.venv/bin/python -m pytest -q` suite passed (830 tests, 1 skipped); targeted Ruff and `git diff --check` passed.

- Corrected interactive website requests so account creation, login, and form
  work route to the browser operator rather than repository coding/mutation.
  Added an explicit model browser-tool procedure, browser-only tool binding,
  required initial browser tool execution, model route review, per-tool terminal
  activity, typed-secret redaction, and a read-only `browser_check_links` tool.
  Permission-denied model responses now stop after one request instead of being
  retried as transient authorization failures.
  The generic entry router now advertises browser contracts and executes a
  dedicated browser_task path instead of misrouting target-URL inspection to
  command inventory.
  - Verification: browser routing, entry-router, AskService, connector, terminal UX, AskAgent, compatibility, and decision tests passed (110 tests); live Playwright link checking passed for 14 links; an end-to-end `gpt-5.4-mini` CLI run used `browser_open`, `browser_inspect`, `browser_check_links`, and `browser_close`; compileall, targeted Ruff, and `git diff --check` passed.

## 2026-07-12

- Added an optional model-controlled Playwright browser for chat, with
  structured inspection and interaction tools, isolated multi-step sessions,
  guarded uploads and downloads, and confirmation gates for sensitive final
  actions. Added setup, security, examples, and local integration-test
  documentation.
  Direct chat now dispatches validated `browser_*` decisions into the AskAgent
  tool loop instead of falling through to a plain answer, and the Playwright
  adapter can use an installed Google Chrome/Chromium binary when its managed
  runtime is unavailable.
  - Verification: browser, routing, AskAgent, CLI-event, tool-manager, and multi-agent tests passed (189 tests); compileall, targeted Ruff, CLI browser status/help, and `git diff --check` passed. The Playwright integration test skipped because local sockets are unavailable in the sandbox.

- Hardened external HTTP 403 handling. Gmail now decodes string and byte error
  bodies, normalizes provider status values, and preserves non-secret provider
  diagnostics; GitHub search now labels only actual quota denials as rate
  limits instead of treating every 403 as one. Worker and direct chat tool
  callbacks now render JSON `ok: false` payloads as failed steps rather than
  successful calls. AskAgent now also recognizes the warning-prefixed JSON
  payload before persisting its trace, so logs no longer record those failures
  as successful tool calls.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_agent.py::test_ask_agent_detects_wrapped_structured_tool_error tests/test_tool_worker_process.py tests/test_cli_ux_helpers.py::test_email_tool_error_row_uses_sanitized_failure_reason tests/connectors/test_email_core.py tests/test_github_provider.py -q` passed (55 tests); targeted Ruff and `git diff --check` passed. Two unrelated full CLI UI tests could not write their session state under sandboxed `~/.mana`.

- Fixed Gmail search-to-read handoff with account-bound canonical message references, typed provider errors, explicit account capabilities, one stale-reference refresh retry, and sanitized failed-tool diagnostics. Reconnection is now suggested only for verified authentication or authorization failures.
  - Verification: Focused Gmail connector, AskAgent, and TUI tool-event tests.

- Updated multi-agent model-level tests to isolate persisted `~/.mana` settings
  and verify that shell model variables cannot override configured role tiers.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py -q` (53 passed) and `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tui_user_config.py -q` (13 passed).

- Fixed OpenAI tool-chat requests for models that enable reasoning by default.
  Tool calls now use the supported Responses API before a Chat Completions
  rejection can occur, and the client retries the observed transient
  insufficient-permission response once without changing the request.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_llm_compatibility.py tests/connectors/test_email_core.py tests/test_ask_entry_router.py -q` passed (32 tests); a live default chat/Gmail request completed through `email_accounts_list`, `email_search`, `email_read`, and `email_thread_read`; `git diff --check` passed.

- Fixed Gmail inbox-search authorization when `email.metadata` and `email.read`
  were selected together. OAuth now requests the searchable readonly scope
  without the conflicting metadata scope, reports Google’s exact query-scope
  denial, and supports reconnecting an existing account in place. Inbox-only
  metadata searches now use Gmail's `labelIds` API parameter instead of the
  metadata-blocked `q=in:INBOX` query, so existing combined-scope tokens work.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/connectors/test_email_core.py -q` passed (10 tests); focused module compilation, a live existing-token inbox metadata search, and `git diff --check` passed. A broader AskAgent suite remains blocked by four unrelated concurrent read-cache failures.

- Moved the shared LLM compatibility client into the multi-agent runtime and
  retargeted all runtime callers and its regression tests, removing the
  remaining retired-package imports.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py::test_no_stale_mana_agent_llm_imports_remain tests/test_llm_compatibility.py -q` passed (11 tests).

- Made Mana-managed configuration repository-independent: `Settings` and
  model-role resolution now read only `~/.mana/config.toml` and
  `~/.mana/secrets.toml`, so shell variables or a repository `.env` cannot
  replace the configured API key.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tui_user_config.py tests/test_search_config.py tests/test_project_llm_analyze_service.py tests/test_llm_compatibility.py -q` passed (31 tests); focused module compilation and `git diff --check` passed.

- Normalized Gmail 401/403 API responses into an actionable OAuth reconnect error instead of incorrectly claiming that metadata-only access was the cause.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/connectors/test_email_core.py tests/test_ask_agent.py tests/test_llm_compatibility.py -q` passed (56 tests).

- Added a centralized capability-driven LLM request compatibility layer. Tool calls with enabled reasoning now use Responses API only when the selected provider supports it; Chat Completions gateways instead retain tools and normalize incompatible reasoning effort to `none`.
  - Added one safe retry for the documented unsupported tools-plus-reasoning HTTP error, with structured API-mode/adjustment logging and no model-name-specific routing.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_llm_compatibility.py tests/test_ask_agent.py tests/test_project_llm_analyze_service.py tests/test_cli_smoke.py -q` passed; compatibility regression suite has 10 passing tests.

## 2026-07-11

- Integrated adaptive repository skills with Chat through a shared session coordinator: repository-isolated compact indexes, explicit model selection with policy validation, progressive loading, timeline events, session-scoped enable/disable, and shared lifecycle inspection commands.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_adaptive_skills.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/skills/chat.py src/mana_agent/commands/chat_cli.py src/mana_agent/config/skills.py` passed.

- Added repository-isolated adaptive skill foundations: stable repository identity, typed manifests and evidence, atomic candidate storage under `${MANA_HOME}/skills`, security validation, approval-gated immutable activation, compact indexes, and constrained progressive selection.
  - Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/skills/adaptive.py src/mana_agent/skills/manager.py src/mana_agent/commands/cli_internal.py` passed.
- Added adaptive skill CLI inspection and lifecycle commands while preserving legacy static `skills/` behavior.
  - Verification: `PYTHONPATH=src .venv/bin/mana-agent skills --help` passed.

## 2026-07-11

- Restored explicitly requested configured MCP providers to the chat tool loop. The selected provider is now propagated through route execution and only that provider is discovered, so a Context7 request no longer fails because its tools were never registered.
  - Included the selected provider's model-visible tools in routing context, so the router can produce a valid constrained tool decision before execution.
  - Verification: Focused MCP and AskAgent tests added.

- Stopped configured MCP providers from starting during ordinary chat routing; MCP discovery now occurs only for an explicitly selected provider. Registered executable Gmail tools are now available to the model-selected chat tool loop, and metadata-only Gmail search can return the latest message headers without requesting a full message body.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/connectors/test_email_core.py tests/test_ask_agent.py tests/test_ask_entry_router.py -q` passed; a connected Gmail account completed a metadata-only search.

- Added an optional provider-neutral Email Connector with Gmail support, normalized models, keyring-backed OAuth credentials, sanitization, permission and approval primitives, account CLI commands, and model-visible tool contracts.
  - Verification: Focused email connector tests and CLI help added.

## 2026-07-10

- Made the packaged dashboard discoverability assertion platform-neutral by
  normalizing import-spec path separators before checking the module suffix.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_api_workspaces.py -q` passed.

- Encoded model-facing MCP tool aliases as OpenAI-compatible function names (for example `mcp__context7__query-docs`) while retaining the original dotted MCP identity for dispatch.
  - Verification: Context7 stdio discovery returned both documentation tools; focused MCP suite updated.

- Passed the protected Context7 server token to its stdio process as `CONTEXT7_API_KEY`, and bounded MCP discovery, calls, and resource reads by each provider timeout.
  - Fixed Streamable HTTP authentication for the installed MCP SDK and added `mcp add --replace` to migrate Context7 to its hosted endpoint.
  - Verification: focused MCP configuration coverage added.

- Made an explicitly named configured MCP provider an execution constraint: routing must select a tool from that provider or stop with a clear provider error, never substitute web search.
  - Applied the same constraint to chat's immediate web/repository-search fast paths, which previously bypassed AskService routing.
  - Verification: focused MCP routing constraint coverage added.

- Restored AskAgent compatibility for test and extension construction paths that bypass initialization; MCP tool discovery now safely defaults to no invocation overrides when that optional attribute is absent.
  - Verification: targeted AskAgent regression test run with isolated user state.

- Wired configured MCP tool names into chat routing and tool-policy validation, and added `mana-agent mcp token-set` for mode-0600 per-server bearer credentials in `~/.mana/mcp_secrets.toml`.
  - `mana-agent mcp token-set` now shows arrow-key server selection when no id is given.
  - Verification: focused MCP suite updated with protected-token coverage.

- Added bidirectional MCP interoperability: typed server configuration, stdio/Streamable HTTP/legacy SSE client discovery, namespaced external tool/resource dispatch, and a bearer-protected Mana-Agent MCP server surface (`mana-agent mcp serve`).
  - Verification: MCP config, stdio discovery/call/resource, queue dispatch, and server authorization tests passed; CLI help checks passed with an isolated `MANA_HOME`.

- Fixed chat tools panel rendering so failed tool errors keep their full
  compact detail on a dedicated line instead of being mid-wrapped and obscured
  by the duration column. Failed validation messages remain visible while
  long URLs stay truncated by `_compact_display_text`.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_ux_helpers.py -q` passed.

- Scoped `ObservabilityStore` SQLite telemetry to the per-repository path under
  `~/.mana/repositories/<id>/observability/` instead of a single global
  `~/.mana/observability/` database. This restores isolation for multi-repo
  sessions and tests that pass a repository root (for example pytest `tmp_path`).
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_observability.py -q` passed.

- Removed legacy Streamlit multipage stubs so the dashboard exposes only its
  active-state sidebar navigation instead of duplicate Overview, Reports, and
  Taskboard links.
  - Verification: `PYTHONPATH=src .venv/bin/python -m py_compile dashboard/app.py`.

- Added SQLite-backed dashboard observability with redacted trace spans, token/latency/error/queue metrics, configurable retention, bottleneck evidence, and optional OTLP export.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_observability.py tests/test_dashboard_helpers.py tests/test_cli_ux_helpers.py -q` passed (26 tests); CLI and chat-storage smoke checks passed.

- Updated CLI and dashboard project analysis to resolve its LLM connection from persisted `~/.mana/config.toml` and `~/.mana/secrets.toml`, preventing a target repository `.env` from overriding the selected analyzer model.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_project_llm_analyze_service.py tests/test_dashboard_helpers.py -q`.

## 2026-07-09 (Persistent dashboard automations and cron deployment)

- Replaced the dashboard's radio navigation with active-state sidebar buttons and added a Cron Jobs page.
- Added typed persistent schedule definitions with explicit POSIX cron validation, built-in/custom action validation, local crontab reconciliation, drift status, and immediate deployment through CLI and dashboard.
- Added `mana-agent automation` and `mana-agent cron` lifecycle commands for create, list, status, deploy, enable, disable, remove, and built-in execution.
- Generated GitHub Actions workflows now include manual dispatch and `.mana/` artifact uploads; GitHub deployment stages only the managed workflow, pushes the feature branch, and opens a PR against the discovered default branch.
- Retired the non-persistent APScheduler/no-op path; invalid execution now reports an error instead of silently selecting a fallback action.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_automation_service.py tests/test_dashboard_helpers.py -q` passed.

## 2026-07-09 (Web Dashboard + Automations Layer + New Project Structure)

- Added top-level `dashboard/` (Streamlit MVP) and `automations/` directories plus `src/mana_agent/ui/streamlit_helpers.py`, `src/mana_agent/automations/` (scheduler, self_improvement, github_integration).
- Added optional dependencies in pyproject.toml: `dashboard`, `automations`, `full` (lazy loaded; core package unchanged).
- Added `mana-agent dashboard` CLI command (lazy, graceful when streamlit missing) and registered it.
- Extended root interactive TUI menu with "Launch Web Dashboard" option.
- Dashboard MVP: sidebar navigation, overview cards, reports viewer, live taskboard/traces from `.mana/`, metrics, safe action stubs. Reuses existing artifacts and helpers. Read-only first.
- Automations boilerplate: GitHub workflow example templates, scheduler example, self-improvement extraction stub (model-decision gated).
- Updated project structure docs implicitly via new modules. All changes follow Inspect→Plan→Model Decision→small edits→Verify→Changelog.
  - Verification: `PYTHONPATH=src venv/bin/python -m py_compile src/...` (multiple modules) passed; `PYTHONPATH=src venv/bin/mana-agent --help`, `... chat --help`, `... analyze --help`, `... dashboard --help` passed and showed new command; core imports of `mana_agent`, `mana_agent.ui`, `mana_agent.automations` succeeded without optional deps; `git status --short` inspected before/after; dashboard/app.py and helpers implement read-only views over taskboard/traces/index; no core multi_agent, routing, or decision files were modified.
- New structure is optional and does not affect existing CLI, multi-agent runtime, or safety model.

## 2026-07-09 (Dashboard: fixed analyze not creating .mana/analyze folder)

- Root cause: `trigger_automation("analyze")` used `python -m mana_agent.commands.cli analyze ...`. The CLI module (`cli.py`) only sets up the Typer `app` for the console script entrypoint (`mana-agent = "mana_agent.commands.cli:app"`). It has no `if __name__` / `app()` handler, so `-m` invocation loaded the module and exited cleanly with rc=0 without ever calling `analyze_command` or `ProjectAnalyzeService`. Hence the run log showed success + correct `artifact_dir` but no folder was created.
- Fix: Primary path in `trigger_automation` for analyze now directly calls `ProjectAnalyzeService().run(...)` (which does `out_dir.mkdir(parents=True, exist_ok=True)` + `write_artifacts`). This guarantees real `.mana/analyze` creation with `report.md`, `report.json`, `symbols.json`, `llm_summary.md`, etc.
- Subprocess kept only as fallback.
- Improved success messages in Overview + Reports pages to surface the created artifacts.
- Direct service path makes "create analyze" buttons produce real output visible in the Reports section (and `list_analysis_artifacts`).
  - Verification: tempfile test `trigger_automation("analyze")` now returns artifacts list and folder with real files (`report.md`, `symbols.json`, `llm_summary.md` etc.); `PYTHONPATH=src ./venv/bin/python -m py_compile ...`; dashboard tests pass.

## 2026-07-09 (Dashboard analyze now reads API key from ~/.mana/config.toml)

- Problem: Dashboard "analyze" always passed `llm_analyzer=None`, producing the exact message the user saw: "LLM analysis unavailable: LLM analyzer not provided."
- Fix: In `trigger_automation` for analyze, now calls `_build_project_llm_analyzer()` (same function as `mana-agent analyze`). This goes through `Settings()` → `settings_source_for_pydantic()` → `load_user_config()` + `load_user_secrets()` from `~/.mana/config.toml` and `secrets.toml` (plus env precedence).
- Also updated `get_last_analysis_summary` candidates to prefer `.mana/analyze/llm_summary.md` so Overview shows fresh LLM summaries generated from dashboard.
- UI now reports "with LLM analysis" vs "deterministic" after clicking generate buttons.
- Result: If the user has a valid key in `~/.mana/config.toml`, triggering analyze from the dashboard now produces a real LLM summary (same as CLI).

  - Verification: In real project, `trigger_automation("analyze")` returned `llm_used=True`, wrote proper `llm_summary.md` (with model + content), and `get_last_analysis_summary` picked it up as type=md. Tests + compile clean.

## 2026-07-09 (Dashboard chat real routing + all triggers functional + real metrics + .mana analyze)

- Chat embed now **real**: `run_dashboard_chat` uses `Settings` + `build_ask_service` + `ask_with_tools` (or classic ask) so prompts are routed via the same model decision / entry router / AskAgent as full `mana-agent chat` CLI. Returns actual answers, sources, tool-using routes when applicable. Multi-turn history + persistence. "ping" example now gets model-routed response instead of hardcoded preview.
- All buttons and triggers have **real functionality**: sidebar Automation Triggers (Self-Improve runs loop + creates .mana/skills, Generate Report runs analyze, etc.), Automations page CRUD + per-item Run (executes + shows results), Overview "Run Analysis", Reports "Generate/Refresh".
- Reports: clicking create/generate analyze explicitly routes artifacts to `.mana/analyze` (via --artifact-dir). list_analysis_artifacts picks them up for the Reports page. Added clear feedback "on .mana route".
- Metrics graphs are now **real**: `get_metrics_summary` parses actual `total_tokens` / usage from `.mana/llm_logs/*.jsonl` into `tokens_series` (last turns). Charts render real sampled usage.
- trigger_automation("analyze") improved with correct flags, sys.executable, explicit .mana/analyze target, better output capture.
- Updated UI text, success messages, and rerun flows so effects are immediately visible (new reports, new skills, updated metrics).
- Still fully lazy, graceful without keys/index, model-decision respecting, no core changes.
  - Verification (this increment): `git status --short`; `PYTHONPATH=src ./venv/bin/python -m py_compile src/mana_agent/ui/streamlit_helpers.py dashboard/app.py`; `PYTHONPATH=src ./venv/bin/python -m pytest tests/test_dashboard_helpers.py -q`; smoke `run_dashboard_chat`, `get_metrics_summary` (real series), `trigger_automation("analyze")` (explicit .mana/analyze), sidebar/buttons exec paths all produced real effects; CLI help + multi-agent imports clean.

## 2026-07-09 (Dashboard expansion, self-improvement, automation hooks + real data)

- Expanded dashboard: real triggers via `trigger_automation`, better chat embed (`st.chat_input` + trace replay + persist), more functional pages (real reports list + generate button using analyze artifacts + ProjectAnalyzeService/subprocess, rich live Taskboard+Traces with dataframe/expanders, real Metrics from telemetry/taskboard, full Automations CRUD + dispatch + run history).
- Nicer sidebar UX with dedicated "⚡ Automation Triggers" quick-action buttons (Self-Improve, Daily Report, Generate Report) + improved navigation.
- Fleshed self-improvement loop: improved `extract_skill_from_trace`, new `run_self_improvement_loop` (scans taskboard DONE + traces, persists skills under .mana/skills + logs runs).
- Added call site in `multi_agent/runtime/agent_work_queue.py` (post verification_passed) + exposed hooks.
- Updated `src/mana_agent/multi_agent/` with `runtime/automation_hooks.py` (register/invoke/list; model-decision and explicit-trigger gated).
- Integrated automations in main src: enhanced `src/mana_agent/automations/` (run_automation, list_available, loop dispatch); helpers now drive real data/CRUD/triggers from .mana/automations/config.json.
- Productional dashboard: CRUD for automations, real data everywhere, safe triggers, report generation, chat history.
- Helpers: improved traces (json+jsonl), new `get_metrics_summary`, `list_analysis_artifacts`, `load/save_automations`, `trigger_automation`.
- All changes keep lazy/optional loading, respect model-decision layer, no fallbacks/keyword routing.
- Verification: `git status --short` (clean); `PYTHONPATH=src python -m py_compile src/mana_agent/ui/streamlit_helpers.py src/mana_agent/automations/self_improvement.py src/mana_agent/automations/__init__.py src/mana_agent/automations/scheduler.py src/mana_agent/multi_agent/runtime/automation_hooks.py src/mana_agent/multi_agent/runtime/agent_work_queue.py dashboard/app.py dashboard/components/cards.py`; `PYTHONPATH=src python -m pytest tests/test_dashboard_helpers.py -q --tb=line` (extended tests pass); `PYTHONPATH=src python -m mana-agent --help` and `... dashboard --help` passed; smoke `PYTHONPATH=src python -c "
from mana_agent.ui.streamlit_helpers import *; from mana_agent.automations.self_improvement import run_self_improvement_loop; from mana_agent.automations import run_automation, list_available_automations; print('imports+helpers ok'); m=get_metrics_summary(); a=list_analysis_artifacts(); print('metrics/artifacts ok', len(a)); t=trigger_automation('noop'); print('trigger ok', t.get('ok'))
"` passed; temp-dir graceful tests cover new helpers.
- Followed full AGENTS.md workflow (inspect, todo, read-before-edit, minimal focused, verify, changelog).

## 2026-07-09 (document file CRUD and query support)

- Added a document tool layer for `.docx`, `.pdf`, `.xlsx`, `.xlsm`, and `.csv` detection, reading, analysis, chunk caching, querying, creation, safe update, and explicit delete operations.
- Exposed document capabilities through model-visible tool contracts, live AskAgent tools, and the queue `ToolsManager` without adding chat-layer keyword routing.
- Added document dependencies and focused fixtures/tests for detection, readers, query, cache invalidation, create/update/delete safety, corrupted PDF handling, and queued document tool execution.
- Fixed Excel document creation so malformed or description-only workbook payloads fail safely without creating blank files, while explicit cell payloads write values and formulas that are verified after save.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_documents.py tests/test_coding_agent.py::test_work_queue_seed_document_create_does_not_discover_without_planner_search tests/test_coding_agent.py::test_coding_agent_document_create_policy_blocks_helper_file_mutations -q` passed with 10 tests; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/documents/writers.py tests/test_documents.py` passed.
- Tightened selected work-queue tool execution so a planner-selected `repo_search` item no longer gives the worker access to `ls` or `list_files`; `list_files` now remains available only when that exact tool was selected.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py::test_selected_discovery_item_policy_allows_only_selected_tool tests/test_agent_work_queue.py::test_selected_list_files_item_policy_requires_explicit_selection tests/test_agent_work_queue.py::test_document_artifact_edit_policy_does_not_allow_helper_file_mutations tests/test_coding_agent.py::test_work_queue_seed_document_create_does_not_discover_without_planner_search tests/test_coding_agent.py::test_coding_agent_document_create_policy_blocks_helper_file_mutations tests/test_documents.py -q` passed with 13 tests; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/multi_agent/runtime/agent_work_queue_adapters.py tests/test_agent_work_queue.py` passed.
- Fixed coding-agent document artifact creation so model-selected `document_create`, `document_update`, and `document_delete` are treated as mutation tools, successful document writes report changed files, and initial repository discovery is seeded only when the planner-selected checklist asks for discovery/search tools.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py::test_work_queue_seed_broad_code_request_can_use_repo_search tests/test_coding_agent.py::test_work_queue_seed_document_create_does_not_discover_without_planner_search tests/test_coding_agent.py::test_coding_agent_tool_policy_includes_full_read_preferences tests/test_tool_policy.py tests/test_documents.py -q` passed with 16 tests; `tmp_home=$(mktemp -d); HOME="$tmp_home" PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py tests/test_agent_work_queue.py tests/test_tools_manager.py tests/test_tool_worker_process.py tests/test_gate_command.py tests/test_auto_chat.py tests/test_tool_policy.py tests/test_documents.py -q` passed with 217 tests; `tmp_home=$(mktemp -d); HOME="$tmp_home" PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 700 tests and 18 warnings; touched-file `ruff --select F,E9`, `py_compile`, `mana-agent --help`, and `mana-agent chat --help` passed.
- Tightened document-artifact execution so normalized checklist tools preserve planner-selected document mutation policy, text file tools cannot write binary `.xlsx`/`.docx`/`.pdf` targets, and forced mutation prompts no longer hardcode project discovery or canned `find` commands.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py::test_work_queue_seed_broad_code_request_can_use_repo_search tests/test_coding_agent.py::test_work_queue_seed_document_create_does_not_discover_without_planner_search tests/test_coding_agent.py::test_coding_agent_document_create_policy_blocks_helper_file_mutations tests/test_agent_work_queue.py::test_document_artifact_edit_policy_does_not_allow_helper_file_mutations tests/test_tools_manager.py::test_forced_mutation_prompt_drives_agentic_authoring tests/test_tool_worker_process.py::test_direct_mutation_tool_args_are_validated_before_worker_start tests/test_ask_agent.py::test_document_binary_targets_are_blocked_for_text_file_tools tests/test_chat_ui_events_tokens.py::test_tool_activity_keeps_nested_subagent_events_with_shared_step_id tests/test_coding_todo_service.py::test_classify_step_uses_tools_then_title -q` passed with 9 tests; `tmp_home=$(mktemp -d); HOME="$tmp_home" PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py tests/test_agent_work_queue.py tests/test_tools_manager.py tests/test_tool_worker_process.py tests/test_gate_command.py tests/test_auto_chat.py tests/test_tool_policy.py tests/test_documents.py tests/test_ask_agent.py tests/test_chat_ui_events_tokens.py tests/test_coding_todo_service.py -q` passed with 283 tests; touched-file `py_compile` and `ruff --select F,E9` passed; `tmp_home=$(mktemp -d); HOME="$tmp_home" PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 703 tests and 18 warnings.

## 2026-07-08 (TUI model-level persistence fix)

- Fixed TUI model selection persistence so selected main, coding planner, and tool-worker models are saved into `MODEL_LEVEL_3_HIGH_REASONING`, `MODEL_LEVEL_2_CODING`, and `MODEL_LEVEL_1_FAST_TOOL` as actual model IDs instead of only saving role-to-level mappings.
- Changed `~/.mana/config.toml` writes to use a stable grouped order for provider/model settings, role mappings, and search settings instead of alphabetical output.
  - Verification: `PYTHONPATH=src .venv/bin/python -m compileall -q src tests/test_tui_user_config.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tui_user_config.py tests/test_multi_agent_core.py::test_role_specific_model_env_overrides_level_env tests/test_search_config.py -q` passed with 16 tests; `PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 693 tests and 18 warnings.

## 2026-07-08 (TUI first-run setup)

- Added a dedicated TUI module with banner reuse, arrow-selectable menus, text/secret prompts, status panels, first-run setup, settings submenu, OpenAI-compatible model fetching/cache, model selection, model role level assignment, and search provider setup.
- Added a `~/.mana` user config loader with separate config/secrets TOML files, secret masking, validation, model-cache helpers, and runtime integration for `Settings`, search config, and model role resolution while preserving environment and `.env` overrides.
- Updated the root CLI menu to include Settings, added `--no-interactive` safety for CI/non-TTY use, documented the new setup flow, and extended web search provider support for Exa and Google CSE.
  - Verification: `PYTHONPATH=src .venv/bin/python -m compileall -q src tests/test_tui_user_config.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 690 tests and 18 warnings; `PYTHONPATH=src .venv/bin/mana-agent --help`, `PYTHONPATH=src .venv/bin/mana-agent chat --help`, `PYTHONPATH=src .venv/bin/mana-agent analyze --help`, and `PYTHONPATH=src .venv/bin/mana-agent plan --help` passed; `PYTHONPATH=src .venv/bin/mana-agent --no-interactive` printed the banner first and exited with the expected missing-config error in non-interactive mode.

## 2026-07-08 (work queue decision seeds)

- Fixed work queue initial seeding so automatic `WorkItem`s are selected from the classifier/planner decision before queue submission instead of blindly starting with `repo_search`.
- Changed Git and command-style requests to begin with Git context or tool-manager decision work, while exact file requests read their target files directly and broad code requests can still use repository discovery.
- Preserved explicit `seeds=` handling so caller-provided queue seeds bypass automatic seed decisions unchanged.
  - Verification: Pending.

## 2026-07-08 (GitOps entry routing)

- Added an explicit `gitops` ask/chat entry route so model-selected Git add, commit, push, branch, and related requests bypass repository search and execute through the Git-capable agent tool path.
- Exposed Git tools to the entry-router decision context and expanded the shell permission policy for approved Git commands while continuing to block protected reset, clean, force-push, and rebase abort/skip patterns.
- Added regression coverage proving Git commit/push requests can route to GitOps without repo search or fallback file creation.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_entry_router.py tests/test_git_tools.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/multi_agent/runtime/entry_router.py src/mana_agent/multi_agent/runtime/route_executor.py src/mana_agent/multi_agent/tools/permissions.py tests/test_ask_entry_router.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/multi_agent/runtime/entry_router.py src/mana_agent/multi_agent/runtime/route_executor.py src/mana_agent/multi_agent/tools/permissions.py tests/test_ask_entry_router.py --select F,E9` passed; `git diff --check -- CHANGELOG.md src/mana_agent/multi_agent/runtime/entry_router.py src/mana_agent/multi_agent/runtime/route_executor.py src/mana_agent/multi_agent/tools/permissions.py tests/test_ask_entry_router.py` passed.

## 2026-07-08 (model-routed ask entry)

- Added an `EntryRouter`/`RouteDecision` layer and `RouteExecutor` so ask/chat entry requests are model-routed before semantic Q&A, repository search, command inventory, external search, coding, or analysis execution.
- Removed automatic command-inventory/project-search recovery from `AskService`, replaced agent exception recovery with structured route errors, and added route trace metadata with route kind, router model, confidence, reason, validation, and executed tools.
- Removed `AgentDecisionEngine._fallback_decision` so unavailable model routing now returns a model-unavailable decision with no selected tools instead of deriving a static route.
- Added regression coverage for command inventory as a routed tool action, missing-index no-action behavior and one model-driven re-route, unknown command re-routing, tool/dir-mode failure handling, web-search routing, invalid router output, response modes without fallback labels, and no-model agent decisions.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_service.py tests/test_ask_entry_router.py -q` passed with 12 tests; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_decision_routing.py -q` passed with 11 tests; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_service.py tests/test_ask_entry_router.py tests/test_agent_decision_routing.py tests/test_multi_agent_core.py -q` passed with 78 tests; `PYTHONPATH=src .venv/bin/python -m pytest tests -q` passed with 674 tests and 18 warnings; `grep -R "classic-fallback\|classic-dir-fallback\|_project_search_fallback\|_command_inventory_fallback" -n src tests` returned no active runtime/test references; `rg -n "_fallback_decision|Fallback used because model routing|source=\"fallback\"|classify_request" src/mana_agent/multi_agent/routing/agent_decision.py tests/test_agent_decision_routing.py tests/test_multi_agent_core.py` returned no matches.

## 2026-07-08 (Git intent workflow gate)

- Added an explicit GitIntent contract for high-risk Git requests so commit, push, and branch intents queue Git state inspection and Git action jobs through QueueManager instead of stopping after repository search.
- Added Git completion gates in ReviewerAgent and Git outcome verification in VerifierAgent, including required status/diff evidence, commit/push evidence or blockers, branch/remote/divergence checks, and HEAD-vs-remote verification for pushes.
- Added focused regression coverage for `commit changes and push to main`, `push to main`, `commit`, and `create new branch` workflows.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py::test_git_commit_push_request_queues_git_inspection_and_does_not_repo_search tests/test_multi_agent_core.py::test_git_push_to_main_inspects_remote_and_blocks_when_behind tests/test_multi_agent_core.py::test_git_commit_inspects_diff_and_uses_diff_derived_message tests/test_multi_agent_core.py::test_git_create_new_branch_inspects_status_before_branch_creation -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py -q` passed with 53 tests; `PYTHONPATH=src .venv/bin/python -m compileall -q src tests` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/multi_agent/agents/main_agent.py src/mana_agent/multi_agent/agents/reviewer_agent.py src/mana_agent/multi_agent/agents/verifier_agent.py src/mana_agent/multi_agent/core/types.py src/mana_agent/multi_agent/queue/queue_manager.py tests/test_multi_agent_core.py --select F,E9` passed.

## 2026-07-08 (model-driven Git tools)

- Added a shared Git tool namespace with dynamic `git help -a` command discovery, structured `git.generic` execution through `subprocess.run(["git", *args], shell=False)`, redacted output, risk classification, protected-command blocking, session Git state memory, and convenience wrappers for status, diff, log, branch, branch creation, staging, commit, push, pull/fetch, remotes, merge/rebase/revert/reset/clean/tag/config.
- Exposed Git tools through the queue `ToolsManager`, model-visible AskAgent tools, and machine-readable tool contracts while keeping tool selection model-driven rather than keyword-routed.
- Added `mana-agent git -- ...` passthrough using the same Git executor and safety policy, plus README and AGENTS documentation for Git decision flow, commit/push preflights, dynamic command discovery, and protected commands.
- Added focused temporary-repository tests for discovery, generic execution, repo-root resolution, wrappers, upstream push behavior, secret redaction, protected command blocking, shell=False execution, timeout handling, memory invalidation, and queue-manager Git execution.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_git_tools.py -q` passed with 12 tests; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_repository_tools.py tests/test_tools_manager.py tests/test_tool_worker_process.py tests/test_multi_agent_core.py::test_cli_commands_exist_and_record_multi_agent_route tests/test_coding_tool_system.py -q` passed with 90 tests; `PYTHONPATH=src .venv/bin/python -m compileall -q src tests` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/multi_agent/tools/git_tools.py src/mana_agent/tools/repository.py src/mana_agent/tools/contracts.py src/mana_agent/multi_agent/runtime/ask_agent.py src/mana_agent/multi_agent/runtime/auto_chat.py src/mana_agent/multi_agent/runtime/tool_worker_process.py src/mana_agent/multi_agent/tools/tool_manager.py src/mana_agent/commands/cli.py src/mana_agent/commands/cli_internal.py tests/test_git_tools.py --select F,E9` passed; `git diff --check` passed; `PYTHONPATH=src .venv/bin/mana-agent git -- status`, `PYTHONPATH=src .venv/bin/mana-agent git -- help -a`, and `PYTHONPATH=src .venv/bin/mana-agent git -- branch` passed.

## 2026-07-07 (chat TUI event panels)

- Upgraded chat UI events with AgentEvent-compatible aliases (`id`, `parent_id`, `timestamp`, `kind`, `details`), normalized file/test/log collections, and persisted session JSONL history under `.mana/sessions`.
- Normalized chat TUI timeline rendering so started/completed updates merge by `event_id`, raw event names are mapped to compact display labels, timeline summaries are truncated/safe, and the Timeline panel is only rendered in the Timeline panel instead of being repeatedly appended after every chat update.
- Added event-driven chat panels for inline status, timeline, tools, subagents, files, diff, tests, and verbose-only logs through normal terminal output and slash commands.
- Added `/timeline`, `/tools`, `/subagents`, `/diff`, `/tests`, `/logs`, `/verbose on|off`, `/compact`, `/expanded`, and `/cancel` direct chat commands, and kept `mana-agent chat --simple` for a plain renderer.
- Removed the full-screen alternate-screen chat implementation, including `--tui`, `--no-animations`, `MANA_CHAT_UI=fullscreen`, full-screen input handling, full-screen worker rendering, and full-screen-specific tests.
- Added running/success/failure events around decision routing, direct-edit checks, web-search, and repository-search steps so normal chat shows compact step-by-step activity immediately after sending a message.
- Added `InlineChatRenderer` as the default append-only event renderer, `TimelineDebugRenderer` for explicit verbose/debug timeline views, compact inline rendering for routing/tool/subagent events, and duplicate event-line collapse.
- Changed chat UI selection so normal terminal chat keeps scrollback and does not print Timeline panels after each turn unless verbose/debug timeline output is explicitly enabled.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_chat_ui_events_tokens.py tests/test_cli_ux_helpers.py -q` passed with 40 tests; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/cli/events.py src/mana_agent/cli/chat_ui.py src/mana_agent/cli/renderers.py src/mana_agent/cli/menu.py src/mana_agent/commands/ui_helpers.py src/mana_agent/commands/chat_cli.py src/mana_agent/commands/main_cli.py src/mana_agent/commands/chat_analyze_command.py tests/test_chat_ui_events_tokens.py tests/test_cli_ux_helpers.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/cli/events.py src/mana_agent/cli/chat_ui.py src/mana_agent/cli/renderers.py src/mana_agent/cli/menu.py src/mana_agent/commands/ui_helpers.py src/mana_agent/commands/chat_analyze_command.py tests/test_chat_ui_events_tokens.py tests/test_cli_ux_helpers.py --select F,E9` passed; `PYTHONPATH=src .venv/bin/mana-agent chat --help | rg -- '--tui|--no-animations|--simple|fullscreen'` returned only `--simple`; `printf 'quit\n' | PYTHONPATH=src MANA_CHAT_UI=plain .venv/bin/mana-agent chat --simple --root-dir /Users/ah/Documents/mana-agent` passed; `rg "fullscreen_chat|--tui|no-animations|MANA_CHAT_UI=fullscreen|ui_mode=\"fullscreen\"|ui_mode == \"fullscreen\"|full-screen|fullscreen" src README.md -n` returned no matches.

## 2026-07-07 (model-driven tool routing)

- Updated external search configuration to load web provider settings from the project `.env` through the shared `Settings` model when environment variables are not exported.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_search_config.py tests/test_search_router.py tests/test_search_decision.py tests/test_agent_decision_routing.py -q` passed with 22 tests; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/config/settings.py src/mana_agent/search/config.py tests/test_search_config.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/config/settings.py src/mana_agent/search/config.py tests/test_search_config.py --select F,E9` passed; live provider smoke for `hermes-agent` loaded Tavily from `.env` and returned 1 result.
- Added a typed `AgentDecision` routing layer that asks the model to choose intent, confidence, tools, tool inputs, repo/web/edit needs, and a verifier summary from tool descriptions instead of letting chat keyword shortcuts select repository search.
- Routed chat read-only `web_search` and `repo_search` turns through the model decision, kept safety/unavailable-model fallbacks bounded, and made the external search router treat keyword hints as fallback-only rather than overriding valid model output.
- Wired the mandatory CLI `MainAgent` route to construct and pass the configured head-decision model into `Router`, so persisted head-decision records no longer fall back to simple routing when model settings are available.
- Exposed `github_search` as a selectable external tool, forced chat execution to honor selected web/GitHub tools without re-deciding, and surfaced provider warnings when external search returns no context.
- Kept immediate repo-search branches out of active coding-agent sessions and guarded opportunistic AskAgent external search so it no longer consumes tool-loop LLM calls for local/tool tasks.
- Fixed explicit `search internet` chat requests so read-only external research executes even when a coding-agent session is configured, while keeping the initial chat `AgentDecision` model output as the only selector for external-search tools.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_decision_routing.py tests/test_search_decision.py tests/test_search_router.py tests/test_ask_agent_recovery.py -q` passed with 23 tests; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/multi_agent/routing/agent_decision.py src/mana_agent/commands/chat_cli.py src/mana_agent/search/decision.py src/mana_agent/multi_agent/runtime/ask_agent.py tests/test_agent_decision_routing.py tests/test_search_decision.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/multi_agent/routing/agent_decision.py src/mana_agent/search/decision.py src/mana_agent/multi_agent/runtime/ask_agent.py tests/test_agent_decision_routing.py tests/test_search_decision.py --select F,E9` passed.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_decision_routing.py -q` passed with 7 tests; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py::test_main_agent_uses_routing_llm_for_head_decision tests/test_multi_agent_core.py::test_cli_commands_exist_and_record_multi_agent_route tests/test_multi_agent_core.py::test_public_command_routes_once_when_root_dispatches_plan tests/test_multi_agent_core.py::test_public_command_callbacks_route_through_main_agent tests/test_agent_decision_routing.py -q` passed with 11 tests; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_agent_recovery.py::test_repeated_failed_reads_stop_after_limit tests/test_ask_agent_recovery.py::test_metrics_count_blocked_vs_failed tests/test_cli_smoke.py::test_chat_coding_agent_answer_only_when_no_repo_edits tests/test_cli_smoke.py::test_chat_balanced_mode_auto_continues_pass_cap_by_default tests/test_cli_smoke.py::test_chat_coding_agent_answer_only_on_tools_only_fallback -q` passed with 5 tests; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_decision_routing.py tests/test_search_router.py tests/test_search_decision.py -q` passed with 19 tests; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_agent_recovery.py tests/test_cli_smoke.py tests/test_agent_decision_routing.py tests/test_search_decision.py tests/test_search_router.py tests/test_chat_direct_commands.py tests/test_auto_chat.py tests/test_multi_agent_core.py -q` passed with 147 tests; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/multi_agent/routing/agent_decision.py src/mana_agent/multi_agent/routing/router.py src/mana_agent/commands/chat_cli.py src/mana_agent/search/router.py src/mana_agent/search/prompts.py src/mana_agent/multi_agent/runtime/ask_agent.py tests/test_agent_decision_routing.py tests/test_search_router.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/multi_agent/routing/agent_decision.py src/mana_agent/multi_agent/routing/router.py src/mana_agent/search/router.py src/mana_agent/search/prompts.py src/mana_agent/multi_agent/runtime/ask_agent.py tests/test_agent_decision_routing.py tests/test_search_router.py --select F,E9` passed. A touched-file Ruff run including `src/mana_agent/commands/chat_cli.py` still reports the pre-existing star-import F403/F405 surface in that module.

## 2026-07-07 (external search routing)

- Added a model-routed, memory-aware external search layer with provider-agnostic web search, structured GitHub search qualifiers, compact source-aware context injection, and search memory reuse.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_search_decision.py tests/test_search_memory.py tests/test_search_router.py tests/test_github_query_builder.py tests/test_github_provider.py tests/test_ask_agent.py -q` passed with 49 tests; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_ux_helpers.py::test_render_turn_summary_and_transparency_sections -q` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/search src/mana_agent/multi_agent/runtime/ask_agent.py src/mana_agent/config/settings.py tests/test_search_decision.py tests/test_search_memory.py tests/test_search_router.py tests/test_github_query_builder.py tests/test_github_provider.py tests/test_ask_agent.py --select F,E9` passed; `PYTHONPATH=src .venv/bin/mana-agent --help` passed.

## 2026-07-07 (macOS release runner)

- Moved the macOS x64 release binary job from the retired `macos-13` GitHub Actions runner to the supported `macos-15-intel` runner label.
- Verification: release workflow YAML parsed with PyYAML; `rg -n "macos-13|macos-15-intel|mana-agent-macos-x64" .github/workflows/release.yml CHANGELOG.md` confirmed the active runner label and artifact references; `git diff --check -- .github/workflows/release.yml CHANGELOG.md` passed.

## 2026-07-07 (chat model routing smoke fix)

- Guarded chat coding-model propagation so lightweight `AskService.ask_agent` stubs without a mutable `model` attribute no longer crash chat startup while real `AskAgent` instances still use `update_model`.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py -q` passed with 64 tests; `PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 619 tests and 18 warnings; `git diff --check` passed.

## 2026-07-06 (chat subagent visibility and model routing)

- Made chat tool activity render subagent-owned tool events with stable event rows, nested subagent/tool labels, compact one-line subagent activity, model level/model labels, and an optional agents-used execution summary.
- Made tool-backed subagent events populate the full-screen Subagents pane and subagent token totals instead of only the Tools pane.
- Added role-based model resolution for main, coding, planner, and tool-worker LLM clients so `MANA_MODEL_*` and `MODEL_LEVEL_*` assignments affect real provider calls while preserving global-model fallback.
- Propagated `agent_role`, `model_level`, and `resolved_model` through execution context and tool-event metadata for trace/TUI display without raw JSON.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/multi_agent/runtime/model_levels.py src/mana_agent/multi_agent/core/types.py src/mana_agent/multi_agent/runtime/ask_agent.py src/mana_agent/commands/cli_internal.py src/mana_agent/commands/chat_cli.py src/mana_agent/multi_agent/runtime/tool_worker_process.py src/mana_agent/commands/ui_helpers.py src/mana_agent/multi_agent/runtime/coding_agent.py src/mana_agent/multi_agent/runtime/agent_work_queue_adapters.py src/mana_agent/cli/renderers.py src/mana_agent/cli/fullscreen_chat.py src/mana_agent/cli/chat_ui.py tests/test_chat_ui_events_tokens.py tests/test_multi_agent_core.py tests/test_cli_ux_helpers.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_chat_ui_events_tokens.py tests/test_cli_ux_helpers.py tests/test_multi_agent_core.py` passed with 83 tests; full `git diff --check` was not clean because of a pre-existing blank-line-at-EOF issue in `src/mana_agent/default_skills/security.md`.

## 2026-07-06 (GitHub release workflow)

- Added a GitHub Actions release workflow for main-branch `latest-dev` prereleases, version-tag stable releases, Python package artifacts, platform standalone binaries, and SHA256 checksums.
- Added a PyInstaller launcher that calls the existing Mana-Agent Typer CLI without duplicating command logic.
- Updated README installation instructions with pipx and latest development binary download examples.
- Hardened Windows CI behavior by normalizing repository-facing paths/newlines and using Windows-safe Python command rewriting during release tests.
- Verification: `PYTHONPATH=src .venv/bin/python -m compileall -q src scripts/mana_agent_entry.py` passed; the eight Windows-failing tests from the release job passed locally; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_agent.py tests/test_chat_direct_commands.py tests/test_chat_ui_events_tokens.py tests/test_dependency_service.py tests/test_describe_service.py tests/test_multi_agent_core.py tests/test_repository_tools.py -q` passed with 109 tests; `PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 608 tests and 18 warnings; `.venv/bin/python -m build` passed; `.venv/bin/pyinstaller --onefile --clean --collect-data mana_agent --name mana-agent scripts/mana_agent_entry.py` passed; `dist/mana-agent --help` passed; release workflow YAML parsed with PyYAML; `git diff --check` passed.

## 2026-07-06 (full-screen chat answer history)

- Added explicit chat conversation history to `ChatUIState` and made the full-screen Chat pane render user/assistant turns before low-level routing events.
- Wired completed chat turns, including direct commands and exact-search fast paths, into the full-screen conversation history so final answers remain visible in the UI.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_ux_helpers.py::test_fullscreen_conversation_text_prefers_answer_history -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_chat_ui_events_tokens.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/cli/chat_ui.py src/mana_agent/cli/fullscreen_chat.py src/mana_agent/commands/chat_cli.py tests/test_cli_ux_helpers.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/cli/chat_ui.py src/mana_agent/cli/fullscreen_chat.py tests/test_cli_ux_helpers.py --select F,E9` passed.

## 2026-07-06 (full-screen chat TUI)

- Added a prompt_toolkit full-screen chat surface with structured chat, step, tool, subagent, token, and boxed input panes plus a startup pet animation for interactive terminals.
- Added `fullscreen` as a chat UI mode via `MANA_CHAT_UI` and `/ui fullscreen`, kept CI/non-TTY/JSON fallbacks, and added token progress bars for full-screen token views.
- Added arrow-selectable menu support for the root menu, analyze format picker, flow-conflict choices, and option-only dynamic selections while preserving numeric/text aliases.
- Kept tool execution inside the full-screen worker dashboard and suppressed the legacy Rich `tools` activity panel in full-screen mode while still recording tool events for the full-screen Tools pane.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_chat_ui_events_tokens.py tests/test_cli_ux_helpers.py tests/test_chat_console_logging.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py tests/commands/test_analyze_slash_command.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/cli/fullscreen_chat.py src/mana_agent/cli/chat_ui.py src/mana_agent/cli/renderers.py src/mana_agent/commands/chat_cli.py src/mana_agent/commands/main_cli.py src/mana_agent/commands/chat_analyze_command.py src/mana_agent/commands/ui_helpers.py` passed; `PYTHONPATH=src .venv/bin/mana-agent --help` passed; `printf 'quit\n' | PYTHONPATH=src MANA_CHAT_UI=fullscreen MANA_CHAT_ANIMATION=0 .venv/bin/mana-agent chat --root-dir /Users/ah/Documents/mana-agent` passed; a PTY smoke with `TERM=xterm-256color PYTHONPATH=src MANA_CHAT_UI=fullscreen MANA_CHAT_ANIMATION=0 .venv/bin/mana-agent chat --root-dir /Users/ah/Documents/mana-agent` rendered the full-screen panes and exited on `quit`; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/cli/fullscreen_chat.py src/mana_agent/cli/chat_ui.py src/mana_agent/cli/renderers.py tests/test_chat_ui_events_tokens.py tests/test_cli_ux_helpers.py --select F,E9` passed. A broader touched-file Ruff run including `chat_cli.py` and `main_cli.py` still reports pre-existing star-import F403/F405 noise in those command modules.

## 2026-07-06 (FastAPI analyze ZIP endpoint)

- Added a FastAPI API package with `POST /api/v1/analyze` for uploaded ZIP projects, safe ZIP extraction, real Mana-Agent analyze reuse, and downloadable result ZIP responses.
- Added API ZIP validation/extraction services, public `analysis-report.md`, `analysis-report.json`, and `manifest.json` result generation, and a `mana-agent api` uvicorn command.
- Added FastAPI, uvicorn, and python-multipart dependencies plus focused API tests for successful uploads, invalid files, unsafe archive paths, and CLI import/help continuity.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_api_analyze.py tests/commands/test_analyze_slash_command.py::test_run_project_analysis_writes_selected tests/test_cli_smoke.py::test_pyproject_exposes_mana_agent_primary_script -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/api/app.py src/mana_agent/api/exceptions.py src/mana_agent/api/routes/analyze.py src/mana_agent/api/services/zip_service.py src/mana_agent/api/services/analyze_service.py src/mana_agent/commands/cli.py src/mana_agent/commands/cli_internal.py tests/test_api_analyze.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/api tests/test_api_analyze.py --select F,E9` passed; `PYTHONPATH=src .venv/bin/mana-agent api --help` passed. A broader touched-file ruff check including `src/mana_agent/commands/cli_internal.py` still reports pre-existing F841 warnings in unrelated legacy code paths.

## 2026-07-06 (memory service consolidation)

- Added `mana_agent.services.memory_service` as the canonical memory service module for multi-agent task/tool memory and run-scoped read evidence.
- Converted the old multi-agent memory and runtime evidence modules into compatibility shims, retargeted live imports to the services module, and stopped `AskAgent.read_file` from writing duplicate persistent SQLite read-cache rows.
- Updated regressions so multi-agent memory no longer stores file-content cache entries and repeated read-file cache behavior is owned by run-scoped `EvidenceMemory`.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py tests/test_ask_agent.py::test_ask_agent_read_file_hits_run_evidence_memory_on_repeat tests/test_ask_agent.py::test_ask_agent_read_file_relative_and_absolute_share_run_memory_entry tests/test_agent_work_queue.py::test_edit_with_evidence_uses_agentic_policy_without_duplicate_reads -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_agent.py::test_ask_agent_read_file_does_not_write_duplicate_flow_cache tests/test_ask_agent.py::test_ask_agent_read_file_line_mode_uses_full_cache_slice -q` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/services/memory_service.py src/mana_agent/multi_agent tests/test_multi_agent_core.py --select F,E9` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/services/memory_service.py src/mana_agent/multi_agent/memory/service.py src/mana_agent/multi_agent/runtime/evidence_memory.py src/mana_agent/multi_agent/runtime/ask_agent.py tests/test_multi_agent_core.py tests/test_ask_agent.py` passed; `git diff --check` passed.

## 2026-07-05 (memory-first multi-agent cache integration)

- Added a shared multi-agent memory service with normalized task fingerprints, task/file/tool/decision/verification records, scoped memory bundles, and hierarchy-based privilege filtering.
- Wired memory into MainAgent routing, TaskBoard memory status, QueueManager duplicate rejection, runtime AgentWorkQueue duplicate traces, and ToolsManager file/tool cache reuse while keeping write tools non-reusable.
- Added regression coverage for duplicate task detection and merge markers, queue duplicate rejection, file read cache hit/miss behavior, scoped bundles, lower-agent access limits, reusable read-only tool results, write-tool history only, and verifier memory reuse.
- Fixed the lightweight ToolsManager memory wiring and stale `_record` calls so batch reads, same-argument cache reuse, and patch context errors return the expected result payloads, normalized reusable tool-memory records so they always include `cache_hit` and `source`, and removed the `rg` dependency from queue repo search for CI portability.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py::test_tool_result_reused_when_args_same tests/test_multi_agent_core.py::test_reusable_tool_memory_adds_cache_metadata tests/test_multi_agent_core.py::test_batch_read_result_reused_when_args_same tests/test_multi_agent_core.py::test_queue_manager_runs_batch_read_through_tools_manager tests/test_multi_agent_core.py::test_patch_context_failure_requires_fresh_read -q` passed; `PATH="/usr/bin:/bin" PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py::test_tool_result_reused_when_args_same -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py -q` passed with 33 tests; `PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 577 tests and 16 warnings; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/multi_agent/tools/tool_manager.py src/mana_agent/multi_agent/memory/service.py tests/test_multi_agent_core.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/multi_agent/tools/tool_manager.py src/mana_agent/multi_agent/memory/service.py tests/test_multi_agent_core.py --select F,E9` passed; `git diff --check -- CHANGELOG.md src/mana_agent/multi_agent/tools/tool_manager.py src/mana_agent/multi_agent/memory/service.py tests/test_multi_agent_core.py` passed.

## 2026-07-05 (multi-agent routing hardening)

- Added explicit task-size classification and route evidence for simple, medium, and large multi-agent requests, including dynamic repo-inventory/docs subagent creation and deactivation recorded on the TaskBoard.
- Added configurable model-tier assignment for multi-agent roles via `MANA_MODEL_*` environment variables, documented the tier placeholders in `.env.example`, added richer queue-job metadata, queued-job schema helpers, batch-read execution, and queued apply-patch execution with stale-context failure guidance.
- Made planned verifier commands explicitly non-passing until actually executed, with ReviewerAgent weak-evidence rejection records, and added focused regression coverage for routing, subagents, queue metadata, batch reads, patch-context failures, model tiers, and verification honesty.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py -q` passed with 21 tests; `PYTHONPATH=src .venv/bin/python -m compileall src` passed; `PYTHONPATH=src .venv/bin/python - <<'PY' ... import mana_agent ... PY` passed; `PYTHONPATH=src .venv/bin/mana-agent --help` and `PYTHONPATH=src .venv/bin/mana-agent chat --help` passed; touched-file `ruff --select F,E9` and `git diff --check` passed.

## 2026-07-05 (document-update evidence and loop guards)

- Added mandatory source-evidence discovery for README and project architecture/structure documentation updates, including a document evidence manifest that blocks mutation when source files from `src/` were not read.
- Prevented document-update runs from taking the single-target read shortcut or early evidence short-circuit before architecture evidence is gathered.
- Added bounded mutation-command deduplication, apply-patch hunk-mismatch re-read traces, non-tool synthesis strict-mode overrides, planning-question auth failure log-once behavior, and guarded worker lifecycle calls.
- Added regression coverage for README evidence manifests, no-src blocking, fake worker lifecycle, planning auth fallback, Redis fallback logging, duplicate log handlers, strict tool traces, plain content synthesis, patch mismatch re-reads, and once-per-plan mutation execution.
- Verification: focused regression tests passed with `.venv/bin/python -m pytest -q ...` (11 tests); broader affected suite passed with `.venv/bin/python -m pytest -q tests/test_agent_work_queue.py tests/test_agent_orchestrator.py tests/test_chat_planning_mode.py tests/test_logging_setup.py tests/test_tool_worker_process.py tests/test_tools_manager.py` (148 tests); full `.venv/bin/python -m pytest -q` passed with 560 tests and 16 warnings; `.venv/bin/python -m compileall src`, `PYTHONPATH=src .venv/bin/mana-agent --help`, and `PYTHONPATH=src .venv/bin/mana-agent chat --help` passed. Full `.venv/bin/ruff check src tests` was not clean because of pre-existing F403/F405 star-import lint in `chat_cli.py`/`main_cli.py`, duplicate `DependencyPackageRef` in `models.py`, and `utils/guards.py` E401; touched runtime/test files passed `ruff --select F,E9`.

## 2026-07-05 (all-command multi-agent routing and runtime migration)

- Routed every public CLI command surface through the mandatory `MainAgent` boundary, including root mode/menu dispatch, `chat`, `analyze`, `plan`, `continue`, and `skills init/list/show`, with a route-once guard for root-dispatched commands.
- Moved the live LLM runtime package from `mana_agent.llm` to `mana_agent.multi_agent.runtime`, retargeted runtime imports, tests, docs, and the worker subprocess module path, and removed the old `src/mana_agent/llm` package.
- Added regression coverage for command-level routing, stale legacy import guards, and command compatibility.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py tests/test_cli_modes_skills.py tests/test_cli_smoke.py::test_continue_command_uses_root_dir_and_loops_until_complete tests/test_chat_console_logging.py tests/test_agent_work_queue.py tests/test_coding_agent.py tests/test_tool_worker_process.py tests/test_tools_executor_redis.py tests/test_prompts_contract.py -q` passed with 163 tests; `PYTHONPATH=src .venv/bin/python -m compileall src` passed; stale `mana_agent.llm` import search returned no matches; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/multi_agent tests --select F,E9` passed; `PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 549 tests and 16 warnings.

## 2026-07-05 (hierarchical multi-agent core)

- Added the mandatory `mana_agent.multi_agent` hierarchy with readable IDs, TaskBoard persistence, MessageBus, DecisionRoom, AgentRegistry, Router, QueueManager, ToolsManager permissions, specialized agents, prompt files, and trace/memory helpers.
- Routed chat, `/analyze`, `/plan`, `mana-agent analyze`, and `mana-agent plan` through `MainAgent.run_user_request(...)` before existing command behavior continues; no multi-agent disable flag or environment bypass was added.
- Documented the architecture in `docs/multi-agent-routing.md` and added focused tests for IDs, taskboard transitions, messages, decisions, registry hierarchy, routing, queue/tool enforcement, CodingAgent tool restrictions, VerifierAgent records, CLI command continuity, and disable-switch absence.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py -q` passed; `PYTHONPATH=src .venv/bin/python -m compileall src/mana_agent/multi_agent src/mana_agent/commands/cli_internal.py src/mana_agent/commands/chat_cli.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py tests/test_agent_work_queue.py tests/test_chat_planning_mode.py -q` passed; `PYTHONPATH=src .venv/bin/python -m compileall src` passed; `PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 546 tests and 16 warnings; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/multi_agent tests/test_multi_agent_core.py --select F,E9` passed.

## 2026-07-04 (agent decision and evidence gate)

- Added a central agent orchestrator with task classification, evidence queue items, an evaluation gate state machine, post-tool critic tracing, and verification-profile selection.
- Wired the live work queue to read explicit single-file targets directly, stop unrelated discovery once enough evidence exists, and emit edit/verify work from read evidence instead of requiring broad repo search first.
- Added a planner-unavailable circuit breaker, explicit fake-worker lifecycle protocol, and Redis executor fallback warning deduplication.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_orchestrator.py tests/test_agent_work_queue.py tests/test_tools_executor_redis.py tests/test_tool_worker_process.py::test_tool_worker_client_init_health_shutdown tests/test_tool_worker_process.py::test_tool_worker_client_restarts_once_on_worker_failure tests/test_tool_worker_process.py::test_tool_worker_client_run_tools_forwards_events tests/test_coding_agent.py::test_preview_execution_checklist_uses_planner_and_persists_to_flow_memory tests/test_coding_agent.py::test_preview_execution_checklist_reports_repair_source tests/test_coding_agent.py::test_preview_execution_checklist_surfaces_deterministic_fallback_warning tests/test_coding_agent.py::test_explicit_file_heading_task_skips_planner_questions tests/test_coding_agent.py::test_planner_failure_circuit_breaker_uses_fallback_once tests/test_cli_smoke.py::test_chat_redis_backend_falls_back_to_local_executor_when_unavailable -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest -q tests/commands tests/integration` passed; `PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 533 tests and 16 warnings; `PYTHONPATH=src .venv/bin/python -c "import mana_agent; print('ok')"` passed; `PYTHONPATH=src .venv/bin/mana-agent --help` passed; `PYTHONPATH=src .venv/bin/mana-agent chat --help` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/agent src/mana_agent/llm/tools_executor.py tests/test_agent_orchestrator.py --select F,E9` passed.

## 2026-07-04 (chat routing regression repair)

- Restored plain `chat` to classic routing by default while keeping `--coding-agent` opt-in and `--agent-tools` auto-execute available for plan-trigger turns.
- Kept default CodingAgent/tool-worker initialization for planning, edit automation, root-dir propagation, and custom-agent tests while routing built-in implicit general chat turns through classic chat.
- Recognized `implement/execute plan` messages as plan triggers in chat routing so they bypass flow-conflict prompts and run through the existing `QueueManager` path when no coding agent is active.
- Restored `rm -rf` blocking in `AskAgent.run_command` and kept `/flow show` visibly reporting active flow memory.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_chat_planning_mode.py tests/test_cli_smoke.py::test_chat_root_dir_applies_to_worker_and_coding_agent_in_classic_mode tests/test_cli_smoke.py::test_chat_root_dir_changes_default_index_dir_in_classic_mode tests/test_cli_smoke.py::test_chat_transparency_uses_trace_steps_in_agent_tools_mode tests/test_cli_smoke.py::test_chat_planning_mode_no_auto_execute_keeps_plan_only_behavior tests/test_cli_smoke.py::test_chat_handles_effective_ui_blocks_failure_without_crash tests/test_cli_smoke.py::test_chat_balanced_profile_auto_executes_clear_edit_requests tests/test_cli_smoke.py::test_chat_full_auto_profile_forces_auto_execute_for_edit_requests tests/test_cli_smoke.py::test_chat_transparency_sections_always_render_in_normal_mode tests/test_cli_smoke.py::test_chat_writes_llm_run_log_rows tests/test_cli_smoke.py::test_chat_plan_trigger_auto_execute_without_coding_agent_hides_progress tests/test_cli_smoke.py::test_chat_redis_backend_falls_back_to_local_executor_when_unavailable tests/test_cli_smoke.py::test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap tests/test_cli_ux_helpers.py::test_coding_agent_mode_routes_general_analysis_turns_to_coding_agent -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 525 tests and 16 warnings.

## 2026-07-04 (approved mutation command retries)

- Restored auto-detected edit requests so the work-queue sniffer emits edit/verify jobs from the resolved mutation-required decision.
- Routed plan-linked direct mutation `WorkItem`s through the local registered mutation-command executor, including incomplete-command blocking before worker dispatch.
- Preserved mutation-only edit policy while supporting approved legacy mutation passes, structured forced retries for per-target deliverables, and explicit docs fallback only when `fallback_decision` is set.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_tools_manager.py -q` passed.

## 2026-07-04 (small direct edit fast path)

- Added a deterministic small-edit classifier and canonical path resolver for explicit low-risk edits such as `update version in readme.md to 0.0.8`, including case-safe `README.md` resolution without repo-wide markdown discovery.
- Added a README version handler that reads a bounded line window, applies one patch, skips worker/search/index/verify setup for one-line docs edits, and reports docs-only verification as skipped with the confirmed changed line.
- Added regression coverage for the direct README version update, duplicate case guard, docs-only verification wording, non-doc fallback behavior, and CLI first-prompt bypass of heavy chat setup.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_small_direct_edit.py tests/test_cli_smoke.py::test_chat_prompt_direct_readme_version_edit_skips_heavy_setup -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/small_direct_edit.py src/mana_agent/commands/chat_cli.py tests/test_small_direct_edit.py tests/test_cli_smoke.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/llm/small_direct_edit.py tests/test_small_direct_edit.py --select F,E9` passed; `git diff --check -- CHANGELOG.md src/mana_agent/commands/chat_cli.py tests/test_cli_smoke.py` passed.


## 2026-07-03 (mutation command execution wiring)

- Added `MutationCommand` compilation and validation so approved `MutationPlan` work produces an executable registered mutation-tool payload before edit execution.
- Wired queue edit jobs, forced mutation retries, and direct edit `WorkItem` adapter execution through the command executor instead of asking the worker/model to select `write_file`, `create_file`, or `apply_patch`.
- Added command-missing and command-incomplete blocked reasons, plan-linked mutation executor traces, and regression coverage for structured command synthesis, direct registered-tool execution, incomplete commands, and prose-only synthesis rejection.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_tools_manager.py tests/test_tool_worker_process.py::test_run_tool_request_expands_file_system_alias -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/mutation_plan.py src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/llm/tools_manager.py tests/test_agent_work_queue.py tests/test_tools_manager.py` passed; `git diff --check` passed.

## 2026-07-03 (mutation plan execution gate)

- Added a structured `MutationPlan` model and validation path so mutation-required queue work builds an approved, evidence-backed decision before edit tools run.
- Wired edit execution and forced retries to attach the approved plan ID/payload, require plan-linked mutation traces for completion, and keep fallback behind an explicit fallback decision instead of normal edit success.
- Added architecture-doc handling that prioritizes `src/mana_agent/**` source areas over tests/changelog hits and requires source-backed intended architecture sections before mutating `docs/08-architecture.md`.
- Added regression coverage for missing-plan write rejection, source-architecture evidence reads, tests/changelog-only evidence rejection, duplicate mutation item collapse, and isolated fallback behavior.
- Verification: `PYTHONPATH=src python3 -m py_compile src/mana_agent/llm/mutation_plan.py src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/llm/tool_worker_process.py tests/test_agent_work_queue.py tests/test_tools_manager.py` passed; `PYTHONPATH=src venv/bin/python3` runtime imports for touched modules passed; manually invoked focused regression functions passed because `pytest` is not installed in the available Python environments.

## 2026-07-03 (progressive skills and batch tools)

- Added progressive skill indexing with `SkillIndexItem` metadata, preferred `skills/<name>/SKILL.md` discovery, on-demand cached `read_skill(skill_name)`, and stable prompts that include only skill name/description/trigger.
- Added batch execution tools for multi-file reads, multi-query searches, grouped scripts, and batched Codex patches, then registered them across tool contracts, AskAgent, policies, gates, prompts, queue progress accounting, and docs.
- Added regression coverage for metadata-only skill indexing, on-demand skill loading, missing skill errors, batch reads/searches/scripts/patches, and updated batch-aware policy/gate expectations.
- Verification: `PYTHONPATH=src .venv/bin/python -m compileall src/mana_agent` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_prompting_builder.py tests/test_cli_modes_skills.py tests/test_repository_tools.py tests/test_tool_policy.py tests/test_auto_chat.py tests/test_gate_command.py tests/test_tool_worker_process.py::test_run_tool_request_expands_file_system_alias -q` passed with 49 tests; `PYTHONPATH=src .venv/bin/mana-agent --help` passed; `PYTHONPATH=src .venv/bin/python -m pytest -q` reported 484 passed, 20 failed, and 16 warnings, with remaining failures in pre-existing queue mutation-plan/chat-smoke/dangerous-command paths.

## 2026-07-03 (mutation execution after target resolution)

- Fixed mutation-required docs edits after target resolution so a prose-only mutation worker falls back to a serialized local `write_file` mutation against the resolved existing markdown file.
- Corrected forced mutation prompts to update existing resolved targets instead of telling the worker to create the requested file.
- Added regression coverage for existing markdown files that already contain `## Update Notes` and for edit-existing forced mutation prompt wording.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py::test_docs_edit_fallback_mutates_existing_update_notes_section tests/test_tools_manager.py::test_forced_mutation_prompt_updates_existing_target -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/tools_manager.py tests/test_agent_work_queue.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/tools_manager.py tests/test_agent_work_queue.py tests/test_tools_manager.py --select F,E9` passed.

## 2026-07-03 (target resolution memory promotion)

- Promoted raw-to-resolved target file mappings into planner/coding memory so typo-prone requests like `architectue.md` execute, verify, and summarize against the resolved repo path.
- Updated queue/sniffer prompts to use resolved target files for structured read, edit, and verify steps while keeping the raw user request only as context.
- Added regression coverage ensuring fuzzy target resolution clears raw typo entries from `missing_required_files`.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py::test_typo_target_resolution_promotes_resolved_file tests/test_agent_work_queue.py::test_typo_target_resolution_clears_missing_required_files -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/agent_work_queue_adapters.py tests/test_agent_work_queue.py tests/test_tools_manager.py` passed.

## 2026-07-03 (docs edit mutation fallback)

- Added a guarded docs-markdown mutation fallback so existing `docs/*.md` edit requests run a deterministic `write_file` mutation when the mutation-only worker returns without selecting an edit tool.
- Ensured existing deliverable targets still trigger forced mutation for update/edit requests even when the file already exists and is non-stub.
- Included the mutation tool and real `git diff -- <target>` verification command/result in successful edit final answers.
- Added regression coverage for `update 08-architecture.md in docs`, bounded docs reads, mutation telemetry, changed files, and verification trace reporting.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py::test_docs_edit_runs_mutation_tool_via_fallback tests/test_agent_work_queue.py::test_simple_docs_edit_does_not_read_all_docs -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/llm/tools_manager.py tests/test_agent_work_queue.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/tools_manager.py tests/test_agent_work_queue.py tests/test_tools_manager.py --select F,E9` passed.

## 2026-07-02 (mutation-only edit regression tests)

- Added regression coverage that edit/forced mutation passes expose only mutation tools, failed edit work keeps the work board incomplete, and bare architecture doc filenames resolve to discovered `docs/*` targets.
- Updated mutation-flow expectations so no-mutation edit runs block with forced-retry telemetry instead of relying on read/search/prose completion.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/llm/tools_manager.py tests/test_agent_work_queue.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/llm/tools_manager.py tests/test_agent_work_queue.py tests/test_tools_manager.py --select F,E9` passed.

## 2026-07-02 (edit flow mutation guard)

- Fixed target resolution for bare documentation filenames so existing repo matches such as `docs/08-architecture.md` win over invented planner paths like `src/08-architecture.md`, while generated/cache paths are ignored.
- Hardened mutation-required queue behavior so a forced mutation retry that returns without any mutation tool attempt raises `AgentFlowError` instead of silently producing a normal final answer.
- Added run-scoped changed-file accounting metadata for pre-existing dirty files and removed a duplicate verification decision key.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py tests/test_agent_work_queue.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/coding_agent.py tests/test_agent_work_queue.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/llm/tools_manager.py src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/coding_agent.py tests/test_agent_work_queue.py tests/test_tools_manager.py --select F,E9` passed.

## 2026-07-02 (stable prompt cache)

- Split coding-agent prompt assembly into cached `StablePromptState` and per-call `EphemeralPromptContext`, with stable cache keys based only on mana-agent/template versions, enabled tools, skill index hash, repository rules hash, identity/rules hash, and model/provider profile.
- Added a session-local `PromptCache`, stable repository-rule rendering from `AGENTS.md`, skill content hashes for invalidation, bounded ephemeral context rendering, and cache/debug token-estimate logs without full prompt contents.
- Wired `CodingAgent._effective_system_prompt_for()` through the session prompt cache while preserving the existing string prompt compatibility surface for chat/auto-execute flows, and documented the prompt-cache boundary.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_prompting_builder.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_prompting_builder.py tests/test_coding_agent.py::test_coding_agent_effective_prompt_includes_language_tooling_guide tests/test_prompts_contract.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/prompting/layers.py src/mana_agent/prompting/builder.py src/mana_agent/prompting/skills_index.py src/mana_agent/prompting/repo_rules.py src/mana_agent/llm/coding_agent.py tests/test_prompting_builder.py` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/prompting/layers.py src/mana_agent/prompting/builder.py src/mana_agent/prompting/skills_index.py src/mana_agent/prompting/repo_rules.py src/mana_agent/llm/coding_agent.py tests/test_prompting_builder.py --select F,E9` passed.

## 2026-07-02 (agent flow and prompt layers)

- Added the new `mana_agent.agent` flow modules for mode/phase selection, task context rendering, and verification planning, plus the new `mana_agent.prompting` modules for stable prompt layers, compact skills indexing, project memory snapshots, mode rules, and prompt composition.
- Connected `CodingAgent._effective_system_prompt_for()` to the layered prompt builder so the existing coding prompt now composes core identity, tool rules, mode rules, skills, memory, current task context, and output contract through the new architecture.
- Enforced the stable prompt assembly order and moved edit/full-auto/verification/flow-memory guidance inside the stable layers instead of adding extra top-level prompt sections.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_prompting_builder.py tests/test_coding_agent.py::test_coding_agent_effective_prompt_includes_language_tooling_guide -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/agent/flow.py src/mana_agent/agent/task_context.py src/mana_agent/agent/selection.py src/mana_agent/agent/verification.py src/mana_agent/prompting/layers.py src/mana_agent/prompting/builder.py src/mana_agent/prompting/skills_index.py src/mana_agent/prompting/memory_snapshot.py src/mana_agent/prompting/mode_rules.py src/mana_agent/prompting/output_contract.py src/mana_agent/llm/coding_agent.py tests/test_prompting_builder.py tests/test_coding_agent.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py tests/test_prompting_builder.py -q` passed; `PYTHONPATH=src .venv/bin/ruff check src/mana_agent/agent/flow.py src/mana_agent/agent/task_context.py src/mana_agent/prompting/builder.py src/mana_agent/prompting/layers.py src/mana_agent/prompting/output_contract.py tests/test_prompting_builder.py --select F,E9` passed.
## 2026-07-04 (edit target resolution)

- Resolved bare existing filenames in edit requests to their unique repository path before forced mutation retries, so requests like `Project Diagram(07-diagram.md)` target `docs/07-diagram.md` when that is the only matching file.
- Restored missing target-resolution exports and sniffer architecture helper imports so the CLI starts instead of failing during `QueueManager` import.
- Removed a stale undefined `plan` reference from work-queue finalization so discovery can emit read/edit/verify follow-up jobs again.
- Routed queue-authored edit and forced-retry work as agentic mutation-required turns instead of incomplete direct `write_file` / `create_file` tool requests, preserving target instructions in the prompt.
- Included failed edit tool details in blocked no-change answers instead of only returning the generic corrected-payload message.
- Verification: `PYTHONPATH=src .venv/bin/mana-agent --help` passed; `PYTHONPATH=src .venv/bin/python - <<'PY' ... from mana_agent.commands.cli import app ... PY` passed; targeted target-resolution regressions passed; `PYTHONPATH=src .venv/bin/python -m compileall src/mana_agent/llm/tools_manager.py src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/commands/cli.py src/mana_agent/commands/cli_internal.py` passed; `git diff --check` passed. Full `tests/test_agent_work_queue.py tests/test_tools_manager.py` was not green on this branch due existing MutationCommand queue behavior outside this startup fix.

## 2026-07-04 (executor-backed agent sessions)

- Added explicit `AgentSession` / `AgentRoute` models for coding-agent routing metadata and chat turn route decisions.
- Routed `QueueManager` work execution through injected `ToolsExecutor.run_batch` when available, including forced mutation retry, while keeping direct worker execution as the no-executor compatibility path.
- Implemented base `ToolsExecutor.run_batch` as a structured fail-closed backend instead of raising, so accidental base-executor use returns ordered `BatchExecutionResult` failures.
- Added batch adapter coverage for WorkItem-to-ToolRunRequest conversion, failed batch results, base executor failures, executor-preferred QueueManager runs, and forced mutation retry through the executor.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_executor_redis.py tests/test_agent_work_queue.py tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m compileall src` passed; `rg "tool_worker_client\\.run_tools|ask_agent\\.run|run_multi" src/mana_agent/llm/coding_agent.py src/mana_agent/llm/agent_work_queue.py` returned no matches. `PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py tests/test_cli_smoke.py -q` was run and still has the existing 8 `tests/test_cli_smoke.py` chat-routing/fake-agent failures.
## 2026-07-02 (mutation tool reliability)

- Added exact-string `edit_file` and atomic sequential `multi_edit_file` mutation tools, registered them across coding-agent, worker, policies, prompts, contracts, and tests, and made them the preferred edit path before patching or whole-file writes.
- Replaced the fragile line-number JSON patch contract with Codex-style text patches using `*** Begin Patch` file blocks, contextual hunks, and strict path/context validation; removed automatic duplicate mutation retry after patch failures.
- Guarded `write_file` overwrites with `expected_sha256` or `force=true`, registered the Laravel default skill, and added regression coverage for line-number-free registry updates.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_edit_file_tools.py tests/test_apply_patch_json_only.py tests/test_tool_input_aliases.py tests/test_write_file_chunking.py tests/test_coding_tool_system.py tests/test_prompts_contract.py tests/test_tool_policy.py tests/test_auto_chat.py tests/test_gate_command.py tests/test_cli_modes_skills.py tests/test_coding_memory_service.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_agent.py tests/test_coding_agent.py tests/test_tool_worker_process.py -q` passed; `PYTHONPATH=src .venv/bin/python -m compileall src`, `PYTHONPATH=src .venv/bin/mana-agent skills list --repo .`, and `git diff --check` passed.

## 2026-07-02 (chat edit orchestration and default skills)

- Added built-in `fastapi`, `nestjs`, `nextjs`, and `reactjs` skills, registered their keyword detection, and added a deterministic default-skill registry text builder for simple marker-based registry edits.
- Targeted built-in skill edit orchestration so default-skill requests seed `DEFAULT_SKILL_NAMES` and `src/mana_agent/default_skills/*.md` discovery instead of broad per-framework searches that can drift into dependency detection files.
- Made `list_files` handle flat markdown globs and recursive `dir/**` / `dir/**/*` patterns consistently, removed the unsafe perl patch fallback from `apply_patch`, validated direct mutation tool args before worker dispatch, and replaced blind tools-only retry with controlled `mutation_not_attempted` / `mutation_failed` / worker-error reporting.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_modes_skills.py tests/test_repository_tools.py tests/test_tool_worker_process.py tests/test_coding_agent.py::test_coding_agent_does_not_retry_tools_only_violation_through_orchestrator tests/test_coding_agent.py::test_coding_agent_provider_error_does_not_fallback_to_direct_ask_agent tests/test_agent_work_queue.py::test_queue_manager_targets_default_skill_registry_without_framework_search_loops tests/test_apply_patch_json_only.py -q` passed; `PYTHONPATH=src .venv/bin/python -m compileall src` passed; `PYTHONPATH=src .venv/bin/mana-agent skills list --repo .` passed and listed the new built-in skills; `printf '/exit\n' | PYTHONPATH=src .venv/bin/mana-agent --chat --repo . --no-banner` passed. `PYTHONPATH=src .venv/bin/python -m pytest tests -q` was run and ended with 458 passed, 8 failed in `tests/test_cli_smoke.py` chat-routing/fake-agent smoke cases.

## 2026-07-02 (coding workflow mutation guard)

- Strengthened edit-task workflow instructions so create/modify/delete runs require project-level related-file cleanup across imports, exports, registries, routers, commands, call sites, tests, docs, and stale references.
- Kept `delete_file` in bounded edit tool policies and mutation-required forced retries, and made write/create/delete mutation payloads report changed files consistently for completion guards and cache invalidation.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_auto_chat.py tests/test_write_file_chunking.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_agent.py::test_ask_agent_keeps_looping_after_apply_patch_failures_for_write_file_fallback tests/test_tool_input_aliases.py::test_safe_delete_file_deletes_existing_file -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/auto_chat.py src/mana_agent/llm/ask_agent.py src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/tools/write_file.py tests/test_agent_work_queue.py tests/test_auto_chat.py tests/test_write_file_chunking.py` passed.

## 2026-07-02 (chat new topic flow)

- Added explicit chat new-topic handling so `/new`, `/new-topic`, `new topic`, and `new topic chat` reset/deactivate the active coding flow while preserving the visible session history.
- Expanded the active-flow divergence prompt to accept `new topic` as a new-flow choice and reset the old flow before rerunning the pending request.
- Verification: `.venv/bin/python -m pytest tests/test_cli_smoke.py::test_chat_new_topic_resets_flow_but_keeps_history tests/test_cli_smoke.py::test_chat_conflict_new_topic_choice_starts_new_flow tests/test_cli_smoke.py::test_chat_clear_still_clears_visible_history -q` passed; `.venv/bin/python -m py_compile src/mana_agent/commands/chat_cli.py tests/test_cli_smoke.py` passed.

## 2026-07-01 (CLI modes and root skills)

- Added a polished Mana Agent root CLI entry flow with banner/menu rendering, root mode flags (`--chat`, `--analyze`, `--plan`), `--repo`, `--model`, `--debug`, and `--no-banner` handling.
- Added root-level skills support with built-in fallback templates, priority loading from `./skills/`, `~/.mana/skills/`, and package defaults, plus `mana-agent skills init/list/show`.
- Expanded Analyze Mode to write the requested Markdown report at `.mana/reports/analyze.md` or `--output` while preserving existing `.mana/analyze/` artifacts, and added first-class Plan Mode plan generation with skill loading and approval gating.
- Verification: `PYTHONPATH=src .venv/bin/python -m compileall src` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_chat_planning_mode.py tests/test_cli_modes_skills.py tests/test_cli_smoke.py::test_root_command_shows_mode_menu tests/test_cli_smoke.py::test_analyze_command_is_public tests/test_cli_smoke.py::test_chat_help_works tests/test_prompts_contract.py -q` passed; `OPENAI_API_KEY= PYTHONPATH=src .venv/bin/mana-agent analyze --repo . --depth quick --format md --output .mana/reports/analyze-smoke.md --max-files 20` passed; `printf '4\n' | PYTHONPATH=src .venv/bin/mana-agent --no-banner`, `printf '/exit\n' | PYTHONPATH=src .venv/bin/mana-agent --chat --repo . --no-banner`, CLI help commands, and temp-repo `skills init/show` smokes passed; `git diff --check` passed. Broader CLI/analyze slice still has existing chat-smoke failures around default coding-agent transcript/auto-execute behavior.

## 2026-06-28 (agent work queue ownership)

- Moved `QueueManager` into `agent_work_queue.py` so the queue manager, `AgentWorkQueue`, `TaskBoard`, `WorkItem`, and `WorkQueueRunner` share one queue-owned module while keeping `agent_work_queue_adapters.py` for worker/sniffer adapters.
- Updated CLI, coding-agent, and queue tests to import `QueueManager` from `mana_agent.llm.agent_work_queue`; `tools_manager.py` no longer exports the queue manager.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/tools_manager.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/llm/coding_agent.py src/mana_agent/commands/cli_internal.py src/mana_agent/commands/chat_cli.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_tools_manager.py -q` passed; `rg "tools_manager import QueueManager" src tests docs` returned no matches. `PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py tests/test_cli_smoke.py -q` was run and still fails in existing chat CLI smoke behavior unrelated to queue import ownership.

## 2026-06-28 (coding agent: worker-owned tool execution)

- Routed coding-agent tool work exclusively through `QueueManager` / `AgentWorkQueue`, removed direct `ask_agent.run*` and bare worker fallbacks from `CodingAgent`, and documented the coding-agent/queue/worker hierarchy.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py tests/test_agent_work_queue.py tests/test_tools_manager.py tests/test_tool_worker_process.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/coding_agent.py src/mana_agent/llm/tools_manager.py src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/llm/tool_worker_process.py` passed.

## 2026-06-28 (agent: remove web search tool)

- Removed the web-search tool surface from runtime tool registration, coding-agent policies, prompt schemas, tests, and the tracked tool module.
- Verification: exact removed-tool search returned no matches; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_agent.py tests/test_coding_agent.py tests/test_tool_policy.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py::test_chat_ping_returns_pong_without_faiss_index tests/test_cli_smoke.py::test_chat_root_dir_changes_default_index_dir_in_classic_mode -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/auto_chat.py src/mana_agent/llm/ask_agent.py src/mana_agent/llm/coding_agent.py src/mana_agent/llm/tool_worker_process.py src/mana_agent/commands/chat_cli.py src/mana_agent/commands/cli_internal.py src/mana_agent/utils/tool_policy.py tests/test_ask_agent.py tests/test_coding_agent.py tests/test_cli_smoke.py tests/test_tool_policy.py` passed; file absence check for removed tool/test files passed.

## 2026-06-28 (agent: delete file tool)

- Added a repository-scoped `delete_file` mutation tool for coding agents, including safe path validation, worker/direct-agent registration, mutation-policy allowlists, tool contracts, prompts, docs, and focused tests.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tool_input_aliases.py tests/test_tool_policy.py tests/test_coding_tool_system.py tests/test_gate_command.py tests/test_prompts_contract.py tests/test_tools_manager.py::test_edit_pass_can_read_and_search_to_ground_content tests/test_tools_manager.py::test_mutation_fallback_allowlist_blocks_discovery_tools tests/test_tools_manager.py::test_forced_mutation_prompt_drives_agentic_authoring -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/tools/write_file.py src/mana_agent/tools/__init__.py src/mana_agent/tools/contracts.py src/mana_agent/utils/tool_policy.py src/mana_agent/llm/tool_worker_process.py src/mana_agent/llm/ask_agent.py src/mana_agent/llm/coding_agent.py src/mana_agent/llm/tools_manager.py src/mana_agent/llm/agent_work_queue.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/llm/gate_command.py src/mana_agent/llm/prompts.py tests/test_tool_input_aliases.py tests/test_tool_policy.py tests/test_coding_tool_system.py tests/test_gate_command.py tests/test_tools_manager.py tests/test_prompts_contract.py` passed.

## 2026-06-28 (chat: ChatLog tool timeline)

- Replaced the visible `Tool activity` chat panel with a compact ChatLog-style transcript renderer in `ui_helpers.py`; tool events now update stable rows in the normal chat timeline and display compact running/success/failure status.
- Stopped surfacing captured Python/debug log records through the visible chat UI while leaving normal logger behavior for log files unchanged; long args, JSON, URLs, and errors are shortened for display.
- Suppressed normal INFO/DEBUG logger records from the interactive chat console and removed the retained standalone `thinking` box from completed tool transcripts.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_chat_console_logging.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py::test_chat_full_auto_pass_cap_auto_resumes_until_completion -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/commands/ui_helpers.py src/mana_agent/commands/cli_internal.py tests/test_cli_ux_helpers.py tests/test_chat_console_logging.py` passed.

## 2026-06-28 (analyze: ReportService audit artifacts)

- Connected `ReportService` to the `/analyze` flow so every successful analyze run also writes `audit_report.json`, `audit_report.md`, and `audit_report.html` alongside the existing analyzer artifacts. The audit report runs offline OSV and uses a no-cache describe adapter so `/analyze` still writes only under the selected analyze output directory.
- Updated the `/analyze` chat summary to list `audit_report.md`, and added tests that enforce ReportService audit artifact generation.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/commands/test_analyze_slash_command.py tests/integration/test_chat_analyze_command.py tests/test_html_output.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/commands/chat_analyze_command.py src/mana_agent/services/report_service.py` passed; `.venv/bin/ruff check src/mana_agent/commands/chat_analyze_command.py tests/commands/test_analyze_slash_command.py --select F401,F821` passed; `git diff --check` passed.

## 2026-06-28

- Removed unused imports across source and tests, including stale CLI/public-surface imports that were only left over from retired commands. Kept explicit `noqa` markers where imports are intentional for wildcard command wiring or static-analysis fixtures.
- Deleted unused tracked artifacts and orphaned describe/deep-flow modules: `patch/ask_agent.patch`, `src/mana_agent/describe/build.py`, `src/mana_agent/describe/file_summary_executor.py`, and `src/mana_agent/describe/llm_chains/deep_flow.py`.
- Verification: no references remain for the deleted describe/deep-flow names; `.venv/bin/ruff check src tests --select F401` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_describe_service.py tests/test_checks.py tests/test_cli_smoke.py::test_cli_commands tests/test_cli_ux_helpers.py tests/test_gate_command.py tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m compileall src/mana_agent tests` passed; CLI import smoke for `mana_agent.commands.cli` and `mana_agent.commands.chat_cli` passed.

## 2026-06-27 (chat: bounded normal auto router)

- Added a bounded normal auto-chat router for non-slash `mana-agent chat` messages. Normal turns are classified into answer-only, plan-only, edit, review, verify, or analyze mode, with compact follow-up state saved under `.mana/chat/auto_state.json`.
- Added mode-level tool policies and mutation safety: non-edit modes remove mutation tools and clamp search/read/discovery budgets; edit mode keeps mutation tools but still uses bounded discovery limits. Wired the mode policy through both regular `CodingAgent` generation and tools-manager auto-execute.
- Updated chat behavior docs for natural-language normal chat, slash-command precedence, bounded discovery, and read-only non-edit modes. Added tests for classifier modes, mutation guard, policy limits, follow-up state, and coding-agent policy plumbing.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_auto_chat.py tests/test_coding_agent.py::test_coding_agent_auto_chat_answer_mode_blocks_mutation_tools tests/test_coding_agent.py::test_coding_agent_auto_chat_edit_mode_allows_mutation_tools tests/test_cli_ux_helpers.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_auto_chat.py tests/test_coding_agent.py::test_coding_agent_auto_chat_answer_mode_blocks_mutation_tools tests/test_coding_agent.py::test_coding_agent_auto_chat_edit_mode_allows_mutation_tools tests/test_cli_ux_helpers.py tests/test_cli_smoke.py::test_chat_balanced_profile_auto_executes_clear_edit_requests tests/test_cli_smoke.py::test_chat_full_auto_profile_forces_auto_execute_for_edit_requests tests/test_cli_smoke.py::test_chat_ping_returns_pong_without_faiss_index -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/auto_chat.py src/mana_agent/llm/coding_agent.py src/mana_agent/commands/chat_cli.py tests/test_auto_chat.py tests/test_coding_agent.py` passed; `PYTHONPATH=src .venv/bin/mana-agent --help`, `PYTHONPATH=src .venv/bin/mana-agent chat --help`, and `printf 'quit\n' | PYTHONPATH=src .venv/bin/mana-agent chat --root-dir . --no-auto-index-missing` passed. `PYTHONPATH=src .venv/bin/python -m mana_agent --help` was attempted but this package has no `mana_agent.__main__`; the console script is the supported entry point.

## 2026-06-27 (analyze: delete old engine)

- Deleted the superseded old analyze engine now that its capabilities are merged into the unified analyze: removed `src/mana_agent/llm/analyze_chain.py` (`AnalyzeChain`), `src/mana_agent/services/llm_analyze_service.py` (`LlmAnalyzeService`), and `src/mana_agent/services/analyze_service.py` (`AnalyzeService`), plus their dedicated tests (`tests/test_llm_analyze_chain.py`, `tests/test_llm_analyze_service.py`).
- Rewired consumers: `chat_analyze_command._build_payload` (legacy HTML/DOT/GraphML/Mermaid formats) now uses the shared `PythonStaticAnalyzer` primitive directly instead of `AnalyzeService`; removed the dead `build_analyze_service`, `build_llm_analyze_service`, and `build_report_service` builders and their imports from `cli_internal.py`; `report_service.py` no longer imports the deleted classes (its `analyze_service`/`llm_analyze_service` are now optional duck-typed injection slots). Trimmed the vestigial `FakeAnalyzeService`/`FakeLlmAnalyzeService` fakes and monkeypatches from `test_cli_smoke.py`, the `AnalyzeChain` logging test from `test_llm_logging.py`, and the `analyze_chain` import-smoke entry from `test_prompts_contract.py`.
- Verification: `compileall src/mana_agent` passed; `grep` confirms no source/test references the deleted modules (only auto-generated `egg-info/SOURCES.txt`); full `pytest -q` = 442 passed, 3 failed (same pre-existing, unrelated chat-smoke failures); `mana-agent analyze . --depth quick` still produces the full `.mana/analyze/` artifact set.

## 2026-06-27 (analyze: project-derived + merged engines)

- Made `/analyze` fully **project-dependent instead of a static template**. `build_architecture` now derives areas from the project's real directories (grouped under the detected source root, src-layout aware), labels each area from its real package docstring (falling back to generic folder-name conventions in `GENERIC_FOLDER_ROLES`), and computes cross-area dependencies from real intra-project imports. `_agent_workflow` was replaced by `_project_workflow`, which answers "how this codebase runs" only from detected entrypoints and real area roles. Removed the mana-agent-specific `known_agent_risk_patterns` and hardcoded pattern matchers from `detect_risks`.
- **Merged the original analyze engine into the new system.** The deterministic core (`PythonStaticAnalyzer`, the basis of `analyze_service.py`/`analyze_chain.py`/`llm_analyze_service.py`) now feeds the unified analyze: its findings are surfaced as project risks (`_static_analysis_risks`), summarized per rule (`static_analysis`), included in the LLM evidence (`static_analysis_summary`), and rendered in `report.md` §11. `project_llm_analyze_service.py` is now the single project-level analyze-LLM layer, superseding the per-file `AnalyzeChain` prompting; evidence risks are severity-ranked so high-volume static findings don't crowd out curated risks.
- Verification: `py_compile` of changed modules passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_project_llm_analyze_service.py tests/test_project_analyze_service.py tests/commands/test_analyze_slash_command.py tests/integration/test_chat_analyze_command.py -q` passed (57); full `pytest -q` = 449 passed, 3 failed (pre-existing, unrelated: `test_chat_transparency_sections_always_render_in_normal_mode`, `test_chat_writes_llm_run_log_rows`, `test_flow_show_checkpoint_and_reset_commands`). Manual `mana-agent analyze . --depth quick` produced a report whose architecture areas are this repo's real directories with import-derived dependencies, plus merged static-analysis findings (missing-docstring 1353, deep-nesting 300, unused-imports 211, wildcard-import 8). Cross-checked on a synthetic non-mana project (shop/api,models,billing) — areas, docstring responsibilities, and api→billing import dependency all derived correctly.

## 2026-06-27

- Added Layer 2 (LLM analyzer) to `/analyze`: `mana-agent analyze .` and chat `/analyze` now send compact, secret-safe evidence to the model and generate an evidence-backed, senior-engineer-style report. New module `src/mana_agent/services/project_llm_analyze_service.py` defines `ModelConfig`, `AnalyzeEvidence`, `LLMAnalyzeResult`, `build_evidence`, and `generate_llm_analysis` (never raises; falls back deterministically). Added a dedicated analyzer prompt (`PROJECT_ANALYZE_SYSTEM_PROMPT`/`PROJECT_ANALYZE_HUMAN_TEMPLATE`).
- `report.md` rewritten to the full 14-section structure with LLM prose plus deterministic evidence tables; new artifacts `evidence.json` (LLM input) and `llm_summary.md`; `report.json` now carries an `llm_analysis` section; `agent_context.json` now compact with `project_summary`, `architecture_summary`, `agent_workflow`, `recommended_tasks`, `generated_artifacts`, and `llm_available`.
- Chat: `/analyze` now runs the LLM analyzer (from `Settings`), prints a compact useful summary, and loads `agent_context.json` into later chat/coding-agent context so follow-up questions ("explain architecture") are grounded. LLM failures and missing API keys degrade to a clearly marked deterministic fallback without crashing.
- Verification: `PYTHONPATH=src .venv/bin/python -m compileall` of the changed modules passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_project_llm_analyze_service.py tests/test_project_analyze_service.py tests/commands/test_analyze_slash_command.py tests/integration/test_chat_analyze_command.py tests/test_cli_smoke.py::test_analyze_command_is_public -q` passed (57); `PYTHONPATH=src .venv/bin/mana-agent analyze . --depth quick` produced an LLM-written report with all artifacts and no secret values. Pre-existing failures remain unrelated: `test_chat_transparency_sections_always_render_in_normal_mode`, `test_chat_writes_llm_run_log_rows`, `test_flow_show_checkpoint_and_reset_commands` (confirmed failing without these changes).

## 2026-06-26

- Reintroduced `mana-agent analyze` as a public repository-intelligence command and upgraded chat `/analyze` to generate the reusable `.mana/analyze/` artifact set: report, inventory, symbols, dependencies, architecture, risks, recommendations, and compact agent context.
- Added modular project analysis for ignored/noisy path pruning, stable file classification, dependency and entrypoint parsing, AST-based Python symbol extraction, architecture/workflow evidence, risk detection, recommendations, JSON validation, and secret-safe `.env` reporting without value exposure.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/services/project_analyze_service.py src/mana_agent/commands/chat_analyze_command.py src/mana_agent/commands/cli_internal.py tests/test_project_analyze_service.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_project_analyze_service.py tests/commands/test_analyze_slash_command.py tests/integration/test_chat_analyze_command.py tests/test_cli_smoke.py::test_analyze_command_is_public tests/test_cli_smoke.py::test_root_help_exposes_commands_and_no_legacy_branding -q` passed; `PYTHONPATH=src .venv/bin/mana-agent analyze . --depth quick --format both --output .mana/analyze --max-files 5000 --max-file-size-kb 512` passed; `PYTHONPATH=src .venv/bin/mana-agent analyze . --depth full --format both --output .mana/analyze --max-files 5000 --max-file-size-kb 512` passed; required JSON artifacts in `.mana/analyze/` parsed successfully; `PYTHONPATH=src .venv/bin/python -m compileall .` passed. Full `PYTHONPATH=src .venv/bin/python -m pytest -q` still fails in pre-existing CLI smoke tests: `test_chat_transparency_sections_always_render_in_normal_mode`, `test_chat_writes_llm_run_log_rows`, and `test_flow_show_checkpoint_and_reset_commands`.

## 2026-06-25

- Relaxed mutation-required work-queue policy for discovery/read jobs so strict tool-worker mode no longer rejects `repo_search` before an edit can run, and allowed deterministic analysis fallback for `update README.md` requests.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py::test_update_readme_analysis_fallback_does_not_strict_block_discovery tests/test_agent_work_queue.py::test_queue_manager_blocks_edit_when_no_mutation_tool_attempted tests/test_agent_work_queue.py::test_queue_manager_blocks_edit_when_mutation_has_no_changed_files -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/llm/agent_work_queue_adapters.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py tests/test_agent_work_queue.py -q` passed.

## 2026-06-24

- Improved mutation-required artifact fallback so full-project analysis requests can deterministically create `analyze.md` with repository structure, command entry points, and a Mermaid diagram, and attach it to `README.md` when requested instead of ending after read-only tool loops.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py::test_mutation_create_file_fallback_creates_docs_analyze tests/test_tools_manager.py::test_analysis_artifact_fallback_attaches_to_readme tests/test_agent_work_queue.py::test_queue_manager_blocks_edit_when_no_mutation_tool_attempted tests/test_agent_work_queue.py::test_edit_request_cannot_finalize_after_only_read_search -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py tests/test_agent_work_queue.py -q` passed.
- Fixed mutation-required work-queue fallback so create/update file tasks resolve concrete targets such as `docs/analyze.md`, run a deterministic `create_file`/`write_file` fallback before another LLM pass, enforce mutation-only strict success in the tool worker, and reject directory paths passed to `read_file`.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py::test_mutation_create_file_fallback_creates_docs_analyze tests/test_tools_manager.py::test_mutation_fallback_allowlist_blocks_discovery_tools tests/test_tool_worker_process.py::test_run_tool_request_requires_mutation_tool_when_mutation_required -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py tests/test_tool_worker_process.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/llm/tool_worker_process.py src/mana_agent/llm/ask_agent.py tests/test_tools_manager.py tests/test_tool_worker_process.py` passed.
- Added run-scoped evidence memory for `read_file` under `.mana/runs/<run_id>/`, normalized read paths before queue/worker dispatch, made cached evidence satisfy read gates across worker calls, invalidated cached entries after mutations, and forced edit jobs into mutation-only policy once evidence is available.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_ask_agent.py tests/test_agent_work_queue.py tests/test_tool_worker_process.py tests/test_tools_manager.py -q` passed; `.venv/bin/python -m compileall src/mana_agent` passed; duplicate-read smoke showed first read `source=tool/cache_hit=false`, second read `source=memory/cache_hit=true`, and one persisted read row. Full `PYTHONPATH=src .venv/bin/python -m pytest -q` still fails in pre-existing CLI smoke tests `test_chat_transparency_sections_always_render_in_normal_mode`, `test_chat_writes_llm_run_log_rows`, and `test_flow_show_checkpoint_and_reset_commands`.
- Removed the public `analyze` CLI registration from `mana-agent` while keeping the default root command on chat, and updated CLI smoke checks and docs so only public commands are verified.
- Verification: `.venv/bin/python -m compileall src/mana_agent` passed; `.venv/bin/mana-agent --help`, `.venv/bin/mana-agent chat --help`, and `.venv/bin/mana-agent ask --help` passed; `.venv/bin/mana-agent analyze --help` failed with `No such command 'analyze'` as expected; `printf 'quit\n' | .venv/bin/mana-agent` opened chat and exited cleanly; focused CLI tests passed with `.venv/bin/python -m pytest tests/test_cli_smoke.py::test_root_command_defaults_to_chat tests/test_cli_smoke.py::test_root_help_exposes_commands_and_no_legacy_branding tests/test_cli_smoke.py::test_analyze_command_is_not_public tests/test_cli_smoke.py::test_cli_commands tests/test_cli_flow.py::test_flow_command_removed tests/test_cli_ux_helpers.py::test_render_turn_transparency_preserves_multiline_command_preview -q`; full `.venv/bin/python -m pytest -q` still failed in pre-existing chat smoke tests `test_chat_transparency_sections_always_render_in_normal_mode`, `test_chat_writes_llm_run_log_rows`, and `test_flow_show_checkpoint_and_reset_commands`.
- Renamed the public CLI/package branding to `mana-agent`, added the primary `mana-agent` console script while keeping `mana-agent` as a compatibility alias, and made the bare root command route to chat.
- Hardened work-queue mutation enforcement so edit-required runs block unless a mutation tool is attempted and changed files are detected, forced retries are mutation-only, verify is blocked before mutation success, worker non-progress statuses are failures, and final answers use the latest useful result instead of concatenating intermediate worker output.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/llm/tool_worker_process.py src/mana_agent/commands/cli_internal.py src/mana_agent/commands/main_cli.py tests/test_agent_work_queue.py tests/test_tool_worker_process.py tests/test_cli_smoke.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_tool_worker_process.py tests/test_cli_smoke.py::test_pyproject_exposes_mana_agent_primary_script tests/test_cli_smoke.py::test_root_command_defaults_to_chat tests/test_cli_smoke.py::test_chat_help_hides_manual_plan_execute_flags tests/test_cli_smoke.py::test_continue_help_accepts_root_dir_option -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_ask_service_fallback.py -q` passed.

## 2026-06-23

- Fixed chat auto-execute orchestration so edit intent and target files come from structured planner output (`requires_edit`, `target_files`) instead of keyword heuristics, with planner-provided targets passed into work-queue edit jobs.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py::test_checklist_requires_edit_recognizes_mutation_tools tests/test_coding_agent.py::test_checklist_requires_edit_uses_structured_planner_flag_without_tool_list tests/test_coding_agent.py::test_checklist_requires_edit_does_not_infer_from_step_text tests/test_agent_work_queue.py::test_queue_manager_runs_edit_and_verify_for_mutating_request tests/test_agent_work_queue.py::test_sniffer_uses_planner_target_file_for_edit_job -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/coding_agent.py src/mana_agent/llm/coding_agent_models.py src/mana_agent/llm/agent_work_queue_adapters.py src/mana_agent/llm/tools_manager.py src/mana_agent/llm/prompts.py tests/test_coding_agent.py tests/test_agent_work_queue.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_work_queue.py tests/test_coding_agent.py -q` passed.

## 2026-06-22

- Added a persistent tools-manager todo ledger with worker/agent confirmation, mutation-proof validation, tools-only violation retry handling, checkbox board reporting, and stricter model-docs candidate sanitation so discovery stops after real pending files are exhausted and edit steps cannot complete without target file changes.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/llm/goal_profiles.py src/mana_agent/commands/chat_cli.py tests/test_tools_manager.py tests/test_cli_smoke.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py::test_chat_no_auto_continue_does_not_resume_pass_cap tests/test_cli_smoke.py::test_chat_balanced_mode_auto_continues_pass_cap_by_default tests/test_cli_smoke.py::test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap tests/test_cli_smoke.py::test_chat_full_auto_pass_cap_auto_resumes_until_completion -q` passed.
- Fixed tools-manager tool-result status checks so trace/answer flags are initialized before branch-specific validation, preventing `has_trace` local-variable crashes from resurfacing.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py tests/test_coding_agent.py -q` passed.
- Fixed tools-manager resumed gate routing so persisted `apply_changes`/`verify_changes` gates override stale discovery state, require concrete mutation/verification payloads, preserve structured failure metadata, keep `plan_patch` incomplete without an edit payload, avoid pending-read redirects during resumed mutation, exclude run artifacts from candidates, and split useful/artifact/target read counters.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py::test_chat_no_auto_continue_does_not_resume_pass_cap tests/test_cli_smoke.py::test_chat_balanced_mode_auto_continues_pass_cap_by_default tests/test_cli_smoke.py::test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap tests/test_cli_smoke.py::test_chat_full_auto_pass_cap_auto_resumes_until_completion -q` passed.
- Added chat-level `--auto-continue/--no-auto-continue` handling so auto-execute checkpoints resume by default in the main chat process until work completes or blocks, not only in `full-auto` profile.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py::test_chat_no_auto_continue_does_not_resume_pass_cap tests/test_cli_smoke.py::test_chat_balanced_mode_auto_continues_pass_cap_by_default tests/test_cli_smoke.py::test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap tests/test_cli_smoke.py::test_chat_full_auto_pass_cap_auto_resumes_until_completion -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/commands/chat_cli.py tests/test_cli_smoke.py` passed.
- Fixed mana-agent run-state reconciliation so successful `read_file` retries update canonical read evidence, visited files, pending reads, checkpoint counters, summary/resume prompts, and work ledger state; narrowed the model-docs profile to relevant `src/**` model/schema sources plus `docs/models.md`; prevented `apply_changes` from completing without real changed-file evidence.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/llm/goal_profiles.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed.
- Refactored run-state candidate discovery to use registered goal profiles, moved model-docs file matching and ranking into `ModelDocsGoalProfile`, and documented how to add future profiles.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/goal_profiles.py src/mana_agent/llm/tools_manager.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed.
- Broadened model-docs goal detection so natural prompts like “create in docs a models.md” trigger the deterministic model/schema queue instead of reading unrelated documentation files.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py tests/test_tools_manager.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py::test_run_state_model_docs_goal_accepts_create_in_docs_wording tests/test_tools_manager.py::test_run_state_model_docs_queue_prioritizes_model_schema_files tests/test_tools_manager.py::test_tools_manager_repairs_forced_read_policy_and_rejects_noop_success -q` passed; a direct `RunStateStore.seed_model_docs_queue()` smoke check for the pasted prompt queued only model/schema files plus `docs/models.md`.
- Fixed tools-manager completion, retry, and no-progress handling so runs cannot complete before the final verified phase, forced reads repair search-only tool policies, retry attempts bypass the same-turn duplicate guard with explicit retry metadata, zero-read actions do not become successful ledger entries, and model-docs queues prioritize model/schema sources over CLI/docs noise.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/llm/tool_worker_process.py tests/test_tools_manager.py tests/test_tool_worker_process.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tool_worker_process.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py -q` passed.
- Hardened the coding-agent state machine and work ledger with explicit `DISCOVERY -> READING -> EXTRACTION -> PATCHING -> VERIFYING -> FINAL` phases, relevance-ranked model-docs read queues, action-key duplicate detection, strict progress accounting, ledger-wide read gates, model-docs mutation blocking, and dynamic read budgets for “read all models” documentation tasks.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/llm/coding_agent.py tests/test_tools_manager.py tests/test_coding_agent.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_coding_agent.py -q` passed; focused new state-machine/ledger tests passed with `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py::test_run_state_model_docs_queue_prioritizes_relevant_files tests/test_tools_manager.py::test_run_state_action_fingerprint_ignores_planner_prose tests/test_tools_manager.py::test_tools_manager_model_docs_blocks_mutation_until_inventory_read tests/test_coding_agent.py::test_coding_agent_model_docs_read_budget_counts_model_files_and_docs -q`.
- Added a public `work_ledger.json` contract for resumable coding-agent runs, enriched tool traces with normalized keys/purpose/phase/evidence metadata, and exposed continuation safety flags (`--max-tool-calls`, `--max-runtime-minutes`, `--max-cost`) on continuation-compatible CLI surfaces.
- Verification: `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/commands/cli_internal.py src/mana_agent/commands/analyze_cli.py tests/test_tools_manager.py tests/test_cli_smoke.py` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed; focused CLI/ledger checks passed with `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py::test_tools_manager_writes_public_work_ledger_and_trace_metadata tests/test_tools_manager.py::test_tools_manager_pass_cap_writes_persistent_checkpoint tests/test_cli_smoke.py::test_continue_help_accepts_root_dir_option tests/test_cli_smoke.py::test_analyze_help_accepts_auto_continue_limits -q`. Full `tests/test_cli_smoke.py -q` still fails in pre-existing chat smoke cases: `test_chat_transparency_sections_always_render_in_normal_mode`, `test_chat_writes_llm_run_log_rows`, and `test_flow_show_checkpoint_and_reset_commands`.
- Hardened the `mana-agent continue` checkpoint engine with a canonical `checkpoint.json`, explicit phase state, exact pending-read resume actions, per-pass progress counters, duplicate read suppression, and auto-continuation safety caps.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py::test_continue_help_accepts_root_dir_option tests/test_cli_smoke.py::test_continue_command_uses_root_dir_and_loops_until_complete tests/test_cli_smoke.py::test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap tests/test_cli_smoke.py::test_chat_full_auto_pass_cap_auto_resumes_until_completion -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/commands/cli_internal.py tests/test_tools_manager.py tests/test_cli_smoke.py` passed.
- Fixed `mana-agent continue`/`resume_run` on normally constructed tools managers by making the internal decision provider fall back to deterministic planning and batching when model invocation is unavailable.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py::test_tools_manager_resume_without_decision_provider_uses_deterministic_fallback tests/test_tools_manager.py::test_tools_manager_planner_schema_parses_strict_json tests/test_tools_manager.py::test_tools_manager_markdown_planner_output_uses_repaired_llm_intent tests/test_tools_manager.py::test_tools_manager_invalid_batch_triggers_repair_then_terminal_stop tests/test_tools_manager.py::test_tools_manager_planner_invalid_uses_deterministic_fallback -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py::test_continue_command_uses_root_dir_and_loops_until_complete -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py tests/test_tools_manager.py` passed.
- Fixed `ToolsManagerOrchestrator.__init__` by removing stale inner imports that shadowed `ToolsExecutionConfig` and caused `UnboundLocalError` during normal construction.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py::test_tools_manager_constructor_uses_top_level_executor_types tests/test_cli_smoke.py::test_continue_command_uses_root_dir_and_loops_until_complete -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py tests/test_tools_manager.py` passed.
- Included `--root-dir <repo>` in generated checkpoint resume commands so a new shell or chat opened from another directory still resumes `.mana/runs/<run_id>` in the owning repository.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py::test_tools_manager_pass_cap_writes_persistent_checkpoint tests/test_cli_smoke.py::test_continue_help_accepts_root_dir_option tests/test_cli_smoke.py::test_continue_command_uses_root_dir_and_loops_until_complete -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py tests/test_tools_manager.py` passed.
- Added `--root-dir` support to `mana-agent continue` and made the command keep resuming the same run while `run_status=needs_resume` or pass cap is reached, instead of requiring manual re-entry after each checkpoint.
- Broadened chat full-auto resume detection to automatically continue any `needs_resume` checkpoint, not only explicit `pass_cap_reached` terminal reasons.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py::test_continue_help_accepts_root_dir_option tests/test_cli_smoke.py::test_continue_command_uses_root_dir_and_loops_until_complete tests/test_cli_smoke.py::test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/commands/cli_internal.py src/mana_agent/commands/chat_cli.py tests/test_cli_smoke.py` passed.
- Fixed full-auto chat continuation to reuse the same persisted `run_id` across pass-cap resume cycles, so work keeps draining in chat instead of restarting from a new checkpoint each cycle.
- Restricted persisted candidate/read evidence to repository-relative source files and filtered dependency/generated trees such as `venv`, `.venv`, `site-packages`, `.mana`, and `node_modules`, preventing resume queues from reading virtualenv files like Django internals.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py tests/test_cli_smoke.py::test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/llm/coding_agent.py src/mana_agent/commands/chat_cli.py tests/test_tools_manager.py tests/test_cli_smoke.py` passed.
- Added persisted auto-execute run checkpoints under `.mana/runs/<run_id>/`, including state, todo, evidence, visited files, tool-call flight recorder, summary, and resume prompt files; pass-cap exits now report exact resume commands and next actions.
- Added gate-aware tool-call fingerprinting and successful-call reuse, pending read-queue enforcement before more broad searches, structured candidate/read evidence tracking, and a `mana-agent continue --run-id <run_id>` command that resumes from saved state.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_smoke.py::test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap tests/test_cli_smoke.py::test_chat_full_auto_pass_cap_auto_resumes_until_completion tests/test_cli_smoke.py::test_chat_balanced_mode_does_not_auto_resume_pass_cap -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/commands/cli_internal.py tests/test_tools_manager.py` passed; direct Typer help smoke for `continue --help` exited 0. Attempted `tests/test_cli_smoke.py::test_cli_help_exposes_chat_command`, but that test name does not exist.
- Made full-auto pass-cap results resumable when planner work is still pending, replaced the user-facing synthetic pass-cap failure text with a continuation status, and added repository-local deterministic fallback guidance for `find all models and update docs/models.md`.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py tests/test_cli_smoke.py::test_chat_full_auto_tools_manager_path_auto_resumes_docs_update_pass_cap tests/test_cli_smoke.py::test_chat_full_auto_pass_cap_auto_resumes_until_completion tests/test_cli_smoke.py::test_chat_balanced_mode_does_not_auto_resume_pass_cap tests/test_chat_direct_commands.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/llm/tools_manager.py src/mana_agent/commands/chat_cli.py tests/test_tools_manager.py tests/test_cli_smoke.py` passed. Full requested suite `PYTHONPATH=src .venv/bin/python -m pytest tests/test_tools_manager.py tests/test_cli_smoke.py tests/test_chat_direct_commands.py -q` was also run and failed only in existing unrelated CLI smoke tests: `test_chat_transparency_sections_always_render_in_normal_mode`, `test_chat_writes_llm_run_log_rows`, and `test_flow_show_checkpoint_and_reset_commands`.
- Removed keyword-based ToolsManager planner intent recovery so unstructured markdown/list planner output now goes through planner repair instead of deriving `search`, `edit`, `verify`, or `answer` from words like `find` or `update`.
- Prevented edit-shaped `find ... update <file>` chat prompts from taking the exact-search fast path before the coding agent can handle them.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_chat_direct_commands.py tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/commands/ui_helpers.py src/mana_agent/llm/tools_manager.py src/mana_agent/llm/coding_agent_tools_provider.py tests/test_chat_direct_commands.py tests/test_tools_manager.py` passed.

## 2026-07-06

- Enforced the multi-agent hierarchy with a `HierarchyPolicy`/`AgentFactory`, MainAgent-owned worker creation, queue-job budget reservations, worker-attributed tool events, queue-backed verification, reviewer evidence checks, and expanded TaskBoard accounting/evidence fields.
- Added regression coverage for MainAgent tool rejection, worker-only tool events, CodingAgent/Verifier queue jobs, planned-verification rejection, MainAgent-only subagent creation, budget records, duplicate task reuse, and coding-route integration evidence.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py -q` passed; `PYTHONPATH=src .venv/bin/python -m compileall src` passed; `PYTHONPATH=src .venv/bin/python -m pytest -q` passed with 607 tests and 18 warnings.
- Added structured chat UI events, render modes, session trace recording, and central token accounting for chat startup, turn timelines, tool activity, subagents, and slash-command status panels.
- Replaced the default chat startup panels/config dump with a compact Mana-Agent header, clean `mana ❯` prompt, `/status full`, `/trace logs`, `/welcome full`, and mode-aware rich/compact/plain/json rendering.
- Preserved `/clear` session history while clearing the visible screen, routed normal chat logs to trace/log files, and fixed flow read-cache persistence so stale cached reads invalidate correctly under telemetry-enabled tool runs.
- Verification: `.venv/bin/python -m py_compile src/mana_agent/telemetry/tokens.py src/mana_agent/telemetry/session_trace.py src/mana_agent/cli/events.py src/mana_agent/cli/renderers.py src/mana_agent/cli/chat_ui.py src/mana_agent/multi_agent/events.py src/mana_agent/commands/chat_input.py src/mana_agent/commands/ui_helpers.py src/mana_agent/commands/chat_cli.py tests/test_chat_ui_events_tokens.py tests/test_cli_smoke.py` passed; `.venv/bin/python -m pytest tests/test_chat_ui_events_tokens.py tests/test_chat_console_logging.py -q` passed; `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_chat_direct_commands.py tests/test_cli_smoke.py -q` passed; `printf 'quit\n' | PYTHONPATH=src .venv/bin/mana-agent chat --no-auto-index-missing` passed; `printf '/status full\n/tokens\n/trace logs\nquit\n' | PYTHONPATH=src .venv/bin/mana-agent chat --no-auto-index-missing` passed; `.venv/bin/python -m pytest -q` passed with 587 passed and 16 warnings.

## 2026-06-21

- Fixed coding-agent tool activity rendering so live-capable terminals use transient live updates and every chat/full-auto resume turn prints exactly one final `Tool activity` panel, with worker events from all resume cycles flowing into the same activity.
- Hid the synthetic `Auto-execute ended without a direct answer from tool runs` pass-cap diagnostic from normal full-auto chat output while preserving it in lower-level result metadata.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_cli_smoke.py -k "tool_activity or full_auto"` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_agent/commands/ui_helpers.py src/mana_agent/commands/chat_cli.py tests/test_cli_ux_helpers.py tests/test_cli_smoke.py` passed.

## 2026-06-18

- Routed active coding-agent chat sessions through CodingAgent for general analysis/tool-inventory turns, matching the startup banner instead of falling back to classic missing-index search.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_cli_smoke.py::test_chat_ping_returns_pong_without_faiss_index tests/test_cli_smoke.py::test_chat_root_dir_changes_default_index_dir_in_classic_mode tests/test_cli_smoke.py::test_chat_coding_agent_uses_worker_lifecycle_once -q` passed; `.venv/bin/python -m py_compile src/mana_agent/commands/chat_cli.py tests/test_cli_ux_helpers.py` passed.
- Made missing-index chat fallback quieter and broadened command-inventory detection so wording like `command exist in this agent` lists CLI commands instead of returning a semantic-index/no-match fallback.
- Verification: `.venv/bin/python -m pytest tests/test_ask_service_fallback.py tests/test_ask_service.py -q` passed; `.venv/bin/python -m py_compile src/mana_agent/services/ask_service.py tests/test_ask_service_fallback.py` passed.
- Collapsed duplicate outer `tool_worker` rows in the live tool-activity panel by tracking per-call event ids and de-duplicating repeated worker operations while preserving inner tool rows.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_tool_worker_process.py -q` passed; `.venv/bin/python -m py_compile src/mana_agent/commands/ui_helpers.py src/mana_agent/commands/chat_cli.py src/mana_agent/llm/coding_agent.py src/mana_agent/llm/tool_worker_process.py` passed.
- Fixed `apply_patch` tool input handling so nested patch wrappers, structured JSON patch lists, and the `input` alias are normalized before validation, avoiding Pydantic string-type failures.
- Verification: `.venv/bin/python -m pytest tests/test_tool_input_aliases.py tests/test_apply_patch_json_only.py tests/test_ask_agent.py -q` passed; `.venv/bin/python -m py_compile src/mana_agent/tools/apply_patch.py src/mana_agent/llm/ask_agent.py` passed.
- Fixed chat conflict handling so a follow-up edit request after the `continue`/`new` prompt starts a new flow instead of being rejected, and active flow memory is applied to normal edit turns.
- Verification: `.venv/bin/python -m pytest tests/test_cli_smoke.py::test_chat_conflict_followup_edit_request_starts_new_flow tests/test_cli_smoke.py::test_chat_full_auto_conflict_is_auto_continued tests/test_cli_smoke.py::test_chat_selection_flow_works_in_normal_agent_tools_path -q` passed; `.venv/bin/python -m py_compile src/mana_agent/commands/chat_cli.py tests/test_cli_smoke.py` passed.
- Added root `.gitignore` coverage for FAISS vector index files written under custom semantic index directories.
- Verification: inspected `.gitignore` and FAISS persistence paths; no test run because this is an ignore-pattern-only change.
- Fixed auto-execute single-file dotfile edits so requests like `update .gitignore add .mana` satisfy the read gate after inspecting `.gitignore`, keep `create_file` available in the coding-agent tool policy, and avoid surfacing incidental missing-file answers when an edit pass cap is reached without changes.
- Verification: `.venv/bin/python -m pytest tests/test_coding_agent.py tests/test_tools_manager.py -q` passed; `.venv/bin/python -m pytest tests/test_coding_agent.py::test_coding_agent_tool_policy_includes_full_read_preferences tests/test_coding_agent.py::test_coding_agent_tool_policy_treats_dotgitignore_as_single_file_edit tests/test_tools_manager.py::test_tools_manager_pass_cap_unfinished_edit_does_not_surface_incidental_answer tests/test_cli_ux_helpers.py -q` passed; `.venv/bin/python -m py_compile src/mana_agent/llm/coding_agent.py src/mana_agent/llm/tools_manager.py tests/test_coding_agent.py tests/test_tools_manager.py src/mana_agent/commands/ui_helpers.py src/mana_agent/commands/chat_cli.py tests/test_cli_ux_helpers.py` passed.
- Changed coding-agent tool activity rendering to collect events during the request and print one final `Tool activity` panel, avoiding repeated live-refresh boxes in captured console output.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py -q` passed; `.venv/bin/python -m py_compile src/mana_agent/commands/ui_helpers.py src/mana_agent/commands/chat_cli.py tests/test_cli_ux_helpers.py` passed.
- Added worker request-level tool activity events so worker calls that fail before invoking a tool, including `tools_only_violation`, still render inside the single `Tool activity` panel.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_tool_worker_process.py::test_tool_worker_client_emits_request_events_for_tools_only_violation -q` passed; `.venv/bin/python -m py_compile src/mana_agent/commands/ui_helpers.py src/mana_agent/commands/chat_cli.py src/mana_agent/llm/coding_agent.py src/mana_agent/llm/tool_worker_process.py tests/test_cli_ux_helpers.py tests/test_tool_worker_process.py` passed.
- Restored live tool-activity updates for capable interactive terminals while keeping recorded, CI, and `TERM=dumb` output on the single final-panel fallback to prevent duplicate boxes.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_tool_worker_process.py::test_tool_worker_client_emits_request_events_for_tools_only_violation -q` passed; `.venv/bin/python -m py_compile src/mana_agent/commands/ui_helpers.py src/mana_agent/commands/chat_cli.py src/mana_agent/llm/coding_agent.py src/mana_agent/llm/tool_worker_process.py tests/test_cli_ux_helpers.py tests/test_tool_worker_process.py` passed.
- Expanded failed tool-call details in the tool activity panel so errors such as `apply_patch` validation failures are not truncated to a one-line summary.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py -q` passed; `.venv/bin/python -m py_compile src/mana_agent/commands/ui_helpers.py tests/test_cli_ux_helpers.py` passed.
- Added an overwrite-safe `create_file` tool for coding agents, registered it in tool contracts, worker/coding-agent tool setup, edit policies, prompts, docs, and focused tests.
- Verification: `.venv/bin/python -m pytest tests/test_write_file_chunking.py tests/test_tool_input_aliases.py tests/test_coding_tool_system.py tests/test_tool_policy.py tests/test_prompts_contract.py tests/test_coding_agent.py -q` passed; `.venv/bin/python -m py_compile src/mana_agent/tools/write_file.py src/mana_agent/tools/__init__.py src/mana_agent/tools/contracts.py src/mana_agent/utils/tool_policy.py src/mana_agent/llm/tool_worker_process.py src/mana_agent/llm/coding_agent.py src/mana_agent/llm/ask_agent.py src/mana_agent/llm/prompts.py src/mana_agent/llm/coding_agent_prompt.py src/mana_agent/commands/chat_cli.py` passed.
- Reworked chat turn transparency output into readable Rich panels for summary, steps, decisions, and session history, with multiline answer previews, compact timestamps, and compact history signal counts.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_cli_smoke.py::test_chat_transparency_sections_always_render_in_normal_mode tests/test_cli_smoke.py::test_chat_summary_uses_actions_taken_total_when_trace_is_truncated -q` passed.
- Added a command-inventory answer path for ask/chat flows so requests like “give me all command of this project” bypass semantic search and list console scripts plus detected CLI subcommands without a missing-index warning.
- Verification: `.venv/bin/python -m pytest tests/test_ask_service_fallback.py` passed; `python3 -m py_compile src/mana_agent/services/ask_service.py tests/test_ask_service_fallback.py` passed; a smoke check with a store that raises on semantic search listed `analyze`, `ask`, and `chat` with no warnings.
- Added a read-only `call_graph` AST tool and registered it with the coding agent, tool policy aliases, and machine-readable tool contracts.
- Updated planner prompts so the agent chooses among `repo_search`, vector-backed `semantic_search`, `read_file`, AST/callgraph tools, and tests/checks instead of relying only on FAISS semantic search.
- Verification: `python3 -m py_compile` on touched Python files passed; targeted pytest command was not run because `pytest` is not installed in the system Python or repo `venv`; a direct callgraph smoke check was attempted but did not complete before interruption.

## 2026-06-17

- Updated `README.md` to reflect the current CLI, installation flow, configuration, generated artifacts, coding-agent behavior, and development checks.
- Verification: documentation-only change; no tests run.
- Added `agents.md` with repository instructions for future agent work.
- Added `CHANGELOG.md` and documented the rule that it must be updated with each repository change.
- Verification: documentation-only change; no tests run.
## 2026-07-22

- Added first-class OpenRouter provider configuration, dynamic model catalog metadata, capability-aware selection, and provider-preserving runtime connection construction.
  - Verification: focused OpenRouter/provider configuration tests.

- Fixed shared OpenRouter fast/tool assignments being rejected by the evidence-based router solely because the same model also serves a higher-reasoning role.
  - Verification: focused OpenRouter gateway-routing regression test.
