# Change Log

All notable repository changes should be recorded here.

## 2026-06-22

- Removed keyword-based ToolsManager planner intent recovery so unstructured markdown/list planner output now goes through planner repair instead of deriving `search`, `edit`, `verify`, or `answer` from words like `find` or `update`.
- Prevented edit-shaped `find ... update <file>` chat prompts from taking the exact-search fast path before the coding agent can handle them.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_chat_direct_commands.py tests/test_tools_manager.py -q` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_analyzer/commands/ui_helpers.py src/mana_analyzer/llm/tools_manager.py src/mana_analyzer/llm/coding_agent_tools_provider.py tests/test_chat_direct_commands.py tests/test_tools_manager.py` passed.

## 2026-06-21

- Fixed coding-agent tool activity rendering so live-capable terminals use transient live updates and every chat/full-auto resume turn prints exactly one final `Tool activity` panel, with worker events from all resume cycles flowing into the same activity.
- Hid the synthetic `Auto-execute ended without a direct answer from tool runs` pass-cap diagnostic from normal full-auto chat output while preserving it in lower-level result metadata.
- Verification: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_cli_smoke.py -k "tool_activity or full_auto"` passed; `PYTHONPATH=src .venv/bin/python -m py_compile src/mana_analyzer/commands/ui_helpers.py src/mana_analyzer/commands/chat_cli.py tests/test_cli_ux_helpers.py tests/test_cli_smoke.py` passed.

## 2026-06-18

- Routed active coding-agent chat sessions through CodingAgent for general analysis/tool-inventory turns, matching the startup banner instead of falling back to classic missing-index search.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_cli_smoke.py::test_chat_ping_returns_pong_without_faiss_index tests/test_cli_smoke.py::test_chat_root_dir_changes_default_index_dir_in_classic_mode tests/test_cli_smoke.py::test_chat_coding_agent_uses_worker_lifecycle_once -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/commands/chat_cli.py tests/test_cli_ux_helpers.py` passed.
- Made missing-index chat fallback quieter and broadened command-inventory detection so wording like `command exist in this analyzor` lists CLI commands instead of returning a semantic-index/no-match fallback.
- Verification: `.venv/bin/python -m pytest tests/test_ask_service_fallback.py tests/test_ask_service.py -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/services/ask_service.py tests/test_ask_service_fallback.py` passed.
- Collapsed duplicate outer `tool_worker` rows in the live tool-activity panel by tracking per-call event ids and de-duplicating repeated worker operations while preserving inner tool rows.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_tool_worker_process.py -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/commands/ui_helpers.py src/mana_analyzer/commands/chat_cli.py src/mana_analyzer/llm/coding_agent.py src/mana_analyzer/llm/tool_worker_process.py` passed.
- Fixed `apply_patch` tool input handling so nested patch wrappers, structured JSON patch lists, and the `input` alias are normalized before validation, avoiding Pydantic string-type failures.
- Verification: `.venv/bin/python -m pytest tests/test_tool_input_aliases.py tests/test_apply_patch_json_only.py tests/test_ask_agent.py -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/tools/apply_patch.py src/mana_analyzer/llm/ask_agent.py` passed.
- Fixed chat conflict handling so a follow-up edit request after the `continue`/`new` prompt starts a new flow instead of being rejected, and active flow memory is applied to normal edit turns.
- Verification: `.venv/bin/python -m pytest tests/test_cli_smoke.py::test_chat_conflict_followup_edit_request_starts_new_flow tests/test_cli_smoke.py::test_chat_full_auto_conflict_is_auto_continued tests/test_cli_smoke.py::test_chat_selection_flow_works_in_normal_agent_tools_path -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/commands/chat_cli.py tests/test_cli_smoke.py` passed.
- Added root `.gitignore` coverage for FAISS vector index files written under custom semantic index directories.
- Verification: inspected `.gitignore` and FAISS persistence paths; no test run because this is an ignore-pattern-only change.
- Fixed auto-execute single-file dotfile edits so requests like `update .gitignore add .mana` satisfy the read gate after inspecting `.gitignore`, keep `create_file` available in the coding-agent tool policy, and avoid surfacing incidental missing-file answers when an edit pass cap is reached without changes.
- Verification: `.venv/bin/python -m pytest tests/test_coding_agent.py tests/test_tools_manager.py -q` passed; `.venv/bin/python -m pytest tests/test_coding_agent.py::test_coding_agent_tool_policy_includes_full_read_preferences tests/test_coding_agent.py::test_coding_agent_tool_policy_treats_dotgitignore_as_single_file_edit tests/test_tools_manager.py::test_tools_manager_pass_cap_unfinished_edit_does_not_surface_incidental_answer tests/test_cli_ux_helpers.py -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/llm/coding_agent.py src/mana_analyzer/llm/tools_manager.py tests/test_coding_agent.py tests/test_tools_manager.py src/mana_analyzer/commands/ui_helpers.py src/mana_analyzer/commands/chat_cli.py tests/test_cli_ux_helpers.py` passed.
- Changed coding-agent tool activity rendering to collect events during the request and print one final `Tool activity` panel, avoiding repeated live-refresh boxes in captured console output.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/commands/ui_helpers.py src/mana_analyzer/commands/chat_cli.py tests/test_cli_ux_helpers.py` passed.
- Added worker request-level tool activity events so worker calls that fail before invoking a tool, including `tools_only_violation`, still render inside the single `Tool activity` panel.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_tool_worker_process.py::test_tool_worker_client_emits_request_events_for_tools_only_violation -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/commands/ui_helpers.py src/mana_analyzer/commands/chat_cli.py src/mana_analyzer/llm/coding_agent.py src/mana_analyzer/llm/tool_worker_process.py tests/test_cli_ux_helpers.py tests/test_tool_worker_process.py` passed.
- Restored live tool-activity updates for capable interactive terminals while keeping recorded, CI, and `TERM=dumb` output on the single final-panel fallback to prevent duplicate boxes.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_tool_worker_process.py::test_tool_worker_client_emits_request_events_for_tools_only_violation -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/commands/ui_helpers.py src/mana_analyzer/commands/chat_cli.py src/mana_analyzer/llm/coding_agent.py src/mana_analyzer/llm/tool_worker_process.py tests/test_cli_ux_helpers.py tests/test_tool_worker_process.py` passed.
- Expanded failed tool-call details in the tool activity panel so errors such as `apply_patch` validation failures are not truncated to a one-line summary.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/commands/ui_helpers.py tests/test_cli_ux_helpers.py` passed.
- Added an overwrite-safe `create_file` tool for coding agents, registered it in tool contracts, worker/coding-agent tool setup, edit policies, prompts, docs, and focused tests.
- Verification: `.venv/bin/python -m pytest tests/test_write_file_chunking.py tests/test_tool_input_aliases.py tests/test_coding_tool_system.py tests/test_tool_policy.py tests/test_prompts_contract.py tests/test_coding_agent.py -q` passed; `.venv/bin/python -m py_compile src/mana_analyzer/tools/write_file.py src/mana_analyzer/tools/__init__.py src/mana_analyzer/tools/contracts.py src/mana_analyzer/utils/tool_policy.py src/mana_analyzer/llm/tool_worker_process.py src/mana_analyzer/llm/coding_agent.py src/mana_analyzer/llm/ask_agent.py src/mana_analyzer/llm/prompts.py src/mana_analyzer/llm/coding_agent_prompt.py src/mana_analyzer/commands/chat_cli.py` passed.
- Reworked chat turn transparency output into readable Rich panels for summary, steps, decisions, and session history, with multiline answer previews, compact timestamps, and compact history signal counts.
- Verification: `.venv/bin/python -m pytest tests/test_cli_ux_helpers.py tests/test_cli_smoke.py::test_chat_transparency_sections_always_render_in_normal_mode tests/test_cli_smoke.py::test_chat_summary_uses_actions_taken_total_when_trace_is_truncated -q` passed.
- Added a command-inventory answer path for ask/chat flows so requests like “give me all command of this project” bypass semantic search and list console scripts plus detected CLI subcommands without a missing-index warning.
- Verification: `.venv/bin/python -m pytest tests/test_ask_service_fallback.py` passed; `python3 -m py_compile src/mana_analyzer/services/ask_service.py tests/test_ask_service_fallback.py` passed; a smoke check with a store that raises on semantic search listed `analyze`, `ask`, and `chat` with no warnings.
- Added a read-only `call_graph` AST tool and registered it with the coding agent, tool policy aliases, and machine-readable tool contracts.
- Updated planner prompts so the agent chooses among `repo_search`, vector-backed `semantic_search`, `read_file`, AST/callgraph tools, and tests/checks instead of relying only on FAISS semantic search.
- Verification: `python3 -m py_compile` on touched Python files passed; targeted pytest command was not run because `pytest` is not installed in the system Python or repo `venv`; a direct callgraph smoke check was attempted but did not complete before interruption.

## 2026-06-17

- Updated `README.md` to reflect the current CLI, installation flow, configuration, generated artifacts, coding-agent behavior, and development checks.
- Verification: documentation-only change; no tests run.
- Added `agents.md` with repository instructions for future agent work.
- Added `CHANGELOG.md` and documented the rule that it must be updated with each repository change.
- Verification: documentation-only change; no tests run.
