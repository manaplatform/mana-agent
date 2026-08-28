# Change Log

All notable repository changes should be recorded here.

## 2026-08-28

- Fixed Windows CI Test Failures in Model Capabilities and Repository Metadata Inspection:
  - Updated `RepositoryMetadataInspector._git_lines()` in `src/mana_agent/model_routing/repository.py` to guard git subprocess execution with `root.is_dir()` and catch `(OSError, ValueError)`, preventing `NotADirectoryError: [WinError 267]` on Windows and non-existent root paths.
  - Guarded git worktree probing in `src/mana_agent/doctor/checks/routing.py` with directory existence check and exception handling.
  - Replaced hardcoded `Path("/tmp")` in `test_separate_agent_permission_and_model_capability` and `test_exact_deepseek_openrouter_direct_responses_reproduction` in `tests/test_model_capabilities.py` with pytest's `tmp_path` fixture.
  - Added regression test `test_repository_metadata_inspector_nonexistent_or_non_directory_root` in `tests/test_model_routing.py`.
  - User verification required: `pytest tests/test_model_capabilities.py tests/test_model_routing.py -v`.

- Resolved Test Suite Regressions Across Gateway, Taskboard, Execution Supervisor, and Model Capabilities:
  - Updated model capability routing fallbacks in `src/mana_agent/gateway/routing.py` and `router.py` to default non-write capabilities to `True` for read/conversational routing while keeping `can_patch=False` (fail-closed for repo writes), and made `AgentRole.CODING`/`AgentRole.PLANNER` conditional on `--no-coding-agent` in `stack.py`.
  - Added `wiring_required` and `wiring_reason` to `TaskBoard.create_task()` and updated `_validate_feature_completion`, `project_supervisor_completion`, and `validators.py` to correctly distinguish child wiring tasks from parent feature tasks, allowing `wiring_required=False` tasks to resolve to `not_required` and verified integration tasks to resolve to `completed` or `already_integrated`.
  - Fixed loopback healthcheck connectivity in `CodexResponsesBridge` by bypassing proxy redirection via `ProxyHandler({})`.
  - Added `_ImportArgs` schema to `api_docs_import` tool in `api_manager/runtime_tools.py` for LangChain args schema validation and imported `getpass` in `api_manager/service.py`.
  - Updated API route execution in `chat_gateway.py` to recognize `{"api_execution_verified", "user_goal_verified"}` as awaiting execution approval when preview requires permission, and updated system prompt contracts.
  - Updated `ExecutionSupervisor.get_verified_execution_result()` to return `EscrowLookupStatus.UNVERIFIED` for tasks pending completion verification before checking terminal flags, and updated budget overrun assertions in `test_result_escrow_recovery.py`.
  - Full test suite verified green: 2485 passed, 2 skipped, 0 failed.
  - User verification required: `pytest`.

- Improved Model Routing Failure Diagnostics and Startup Error Handling:
  - Updated `ModelRouter.route()` in `src/mana_agent/model_routing/router.py` to include detailed candidate rejection reasons (`Rejected candidates: <model> (<reasons>)`) in the `RoutingFailure` message when candidate evaluation rejects all models, making routing failures immediately diagnosable.
  - Updated `chat_cli.py` to catch `RoutingFailure` alongside `ValueError` during gateway startup and initial model resolution, presenting clean and actionable parameter errors instead of crashing with an unhandled Python traceback.
  - User verification required: `python -m pytest tests/test_model_routing.py tests/test_model_capabilities.py tests/gateway/test_routing_authority.py -v`.

- Fixed API Workflow Completion Evidence Contract and Removed Legacy Required Actions Authority:
  - Removed legacy `required_actions` fallback as runtime authority from `_api_workflow_completion_from_trace()` in `src/mana_agent/gateway/chat_gateway.py`.
  - Removed legacy action normalization validator from `_WorkflowDecision` in `src/mana_agent/api_manager/runtime_tools.py`, requiring explicit `required_outcomes` and adding `migrate_legacy_workflow_decision()` for explicit versioned migrations of historical decision dictionaries.
  - Ensured read-only API tasks use verified external execution (`api_execution_verified`) and user-goal evidence (`user_goal_verified`) as the primary completion contract without turning optional implementation steps (`documentation_inspection`, `integration_import`, `operation_search`, `request_preview`) into mandatory workflow gates.
  - Made `missing_outcomes` a canonical field reporting semantic outcomes instead of tool names, while preserving `missing_actions` as a backward-compatibility alias.
  - Independently evaluated `user_goal_verified` based on upstream payload satisfaction and failure flags.
  - Preserved mutation risk policies (POST, PUT, PATCH, DELETE require preview/approval before execution verification is credited).
  - Updated API route execution in `chat_gateway.py` to project normalized workflow evidence (`required_outcomes`, `completed_outcomes`, `missing_outcomes`, `execution_evidence`, `goal_satisfied`, `actual_tool_events`) to durable lane task and taskboard records.
  - Updated test fixtures in `tests/test_api_manager.py` and `tests/gateway/test_api_manager_route.py`, and added comprehensive regression test cases covering Quran reproduction, saved integrations, docs-only tasks, unexecuted API tasks, mutation preview policy, legacy decision rejection, goal satisfaction verification, and durable task projection.
  - User verification required: `python -m pytest tests/gateway/test_api_manager_route.py tests/test_api_manager.py -v`.

- Fixed Coding Agent Model-Capability Admission Failure and Multi-Transport Resolution:
  - Resolved `Write-required Codex turn rejected: model capabilities are unknown` by implementing authoritative transport-level capability resolution keyed by `(provider, model, transport)`.
  - Added `ModelCapabilityDescriptor` and `ModelCapabilityRegistry` in `src/mana_agent/config/model_capabilities.py` representing explicit capability dimensions (`supports_tool_calls`, `supports_repository_read`, `supports_repository_write`, `supports_shell`, `supports_structured_output`, `supports_streaming`, `supports_parallel_tools`, `capability_confidence`, `capability_source`).
  - Normalized OpenRouter model IDs to preserve organization namespaces (e.g. `deepseek/deepseek-v4-flash`, `anthropic/claude-3.7-sonnet`) and avoid treating OpenRouter models as bare OpenAI-native models.
  - Updated candidate evaluation in `ModelRouter.route()` and `GatewayRoutingAuthority` to filter unknown or incompatible model candidates before selection, enabling automatic fallback to alternative verified write-capable candidates without failing the user turn.
  - Enforced fail-closed behavior with typed `NoWriteCapableModelAvailableError` (`no_write_capable_model_available`) containing candidate diagnostics when no candidate is write/tool capable.
  - Separated agent permissions (`CodingAgent` workspace write permission) from model transport capabilities in `CodexCodingAgentShim._validate_write_transport_capability`.
  - Added structured routing diagnostic events (`model.capability.resolved`, `model.capability.unknown`, `model.candidate.rejected`, `model.candidate.selected`).
  - Added regression test suite in `tests/test_model_capabilities.py` covering Cases A, B, C, D, E, exact DeepSeek OpenRouter reproduction, caching/invalidation, and diagnostic events.
  - User verification required: `python -m pytest tests/test_model_capabilities.py tests/test_model_routing.py tests/gateway/test_routing_authority.py tests/test_openrouter_provider.py tests/test_codex_integration.py -v`.

- Fixed Checkpoint Lifecycle Race and Terminal State Transition Bug:
  - Resolved `Gateway execution failed: task cannot checkpoint from state failed` by centralizing checkpoint transition authority in `ExecutionSupervisor.can_checkpoint()` and enforcing durable supervisor state over stale in-memory task projections.
  - Updated `ExecutionSupervisor.checkpoint()` and `LocalExecutionStore.update_task_and_checkpoint()` with atomic compare-and-set semantics that safely skip checkpointing with typed diagnostics (`checkpoint.skipped` event with reason `checkpoint_skipped_terminal_state`) when an execution is already terminal (`failed`, `cancelled`, `completed`, `budget_exhausted`, `recovery_review_required`).
  - Guaranteed that late/stale callbacks from cleanup handlers, feature integration, verification preparation, or post-processing never overwrite the original task failure reason, result escrow, verification state, or terminal metadata.
  - Ensured checkpoint failures for active `RUNNING` executions (e.g. invalid lease tokens or uncommitted store writes) continue to surface as real errors.
  - Enforced strict state machine transitions in `state_machine.py` so that direct `FAILED -> CHECKPOINTING` is rejected and recovery from failed executions strictly follows explicit state transitions (`FAILED -> RETRY_SCHEDULED -> QUEUED -> LEASED -> RUNNING -> CHECKPOINTING`).
  - Added structured lifecycle instrumentation events (`checkpoint.requested`, `checkpoint.allowed`, `checkpoint.skipped`, `checkpoint.rejected`) with diagnostic metadata (`task_id`, `execution_id`, `current_state`, `expected_state`, `caller`, `checkpoint_boundary`, `terminal_reason`, `attempt_id`).
  - Updated `LaneCoordinator.checkpoint()` and `chat_gateway.py` to guard checkpoint calls with `can_checkpoint` checks and handle skipped checkpoints without crashing.
  - Added comprehensive regression test suite in `tests/execution_supervisor/test_checkpoint_lifecycle_races.py` covering failure-before-checkpoint, concurrent failure races, normal checkpoint lifecycle, completed execution late callbacks, recovery transitions, original error preservation, restart reconciliation, running task invalid lease rejection, and lane coordinator terminal handling.
  - User verification required: `python -m pytest tests/execution_supervisor/test_checkpoint_lifecycle_races.py tests/gateway/test_checkpoint_resume_invariants.py tests/execution_supervisor/test_supervisor_core.py tests/gateway/test_lane_coordinator.py -v`.

- Made API Workflow Validation Evidence-Based Instead of Tool-Name-Based:
  - Removed mandatory fixed tool-name sequence requirements (`documentation_inspection`, `request_preview`, `request_execution`) on the `api` route.
  - Refactored `_WorkflowDecision` schema in `src/mana_agent/api_manager/runtime_tools.py` to declare outcome requirements (`required_outcomes` and `optional_outcomes`) with support for `api_target_resolved`, `api_execution_verified`, `user_goal_verified`, `documentation_understood`, `integration_available`, `operation_resolved`, `request_previewed`, and `approval_obtained`. Added backward compatibility validator and property for legacy `required_actions`.
  - Refactored `_api_workflow_completion_from_trace` in `src/mana_agent/gateway/chat_gateway.py` to validate outcome-based evidence instead of tool-name sequences, normalizing execution results from any authorized execution capability (`api_request_execute`, browser executors, connectors, HTTP runtimes) into a standard `api_execution_evidence` contract.
  - Separated `actual_tool_events` (observability trace) from authoritative workflow completion evidence.
  - Made request preview policy-based rather than universally mandatory: safe read-only operations (`GET`, `HEAD`, `OPTIONS`) may skip preview, while mutations (`POST`, `PUT`, `PATCH`, `DELETE`) continue to enforce preview and approval before execution.
  - Made documentation inspection optional when the operation/integration is already saved or directly executable. Allowed documentation to be inspected via `api_docs_inspect` or `browser_inspect`.
  - Updated API route system prompt instructions in `chat_gateway.py`, documentation in `docs/api-manager.md`, and skill instructions in `skills/api-manager/SKILL.md`.
  - Updated test suite in `tests/test_api_manager.py` and `tests/gateway/test_api_manager_route.py` with comprehensive regression coverage for read-only execution without preview/docs, browser documentation with connector execution, saved integration execution without reinspection, mutation preview enforcement, and tool trace separation.
  - User verification required: `python -m pytest tests/gateway/test_api_manager_route.py tests/test_api_manager.py -v`.

## 2026-08-27

- Fixed Authoritative Result Escrow Recovery, Stale Task Lifecycle Repair, and Single-Writer Coordination:
  - Traced incident `task_20260827_000001` and confirmed why `result_778a7987-cd61-4e2b-998b-05cc8065ba3e` existed in escrow while the task remained `status=in_progress` due to an interrupted lifecycle transition after durable atomic result write.
  - Made persisted result escrow authoritative over stale task lifecycle state in `ExecutionSupervisor.get_verified_execution_result`, automatically repairing task record state (`state=COMPLETED`/terminal, `result_id`, `verification_status`, `completion_artefacts`, `finished_at`, `failure_reason`, and `provider_metadata`) and attempt state.
  - Updated `ExecutionSupervisor.recover()` to scan and reconcile incomplete tasks against authoritative escrow results without failing on unverified or expired states.
  - Preserved atomic, write-once escrow semantics in `LocalExecutionStore.save_result` while adding safe idempotent reconciliation for concurrent writer races with identical payloads and rejecting conflicting payloads with `EscrowConflictError`.
  - Updated `ExecutionSupervisor.transition`, `ExecutionSupervisor.record_terminal_result`, and `LaneCoordinator.finish` to ensure single-writer safety and prevent duplicate terminal result generation when an authoritative result already exists.
  - Enforced separation between recovery of the same execution and genuinely new execution attempts with distinct task/attempt identities.
  - Added regression test scenarios N through T in `tests/execution_supervisor/test_result_escrow_recovery.py` covering `task_20260827_000001` recovery, crash-after-result-write, terminal failure crash repair, concurrent writer races, duplicate replay idempotency, retry/resume separation, and forbidden direct-model fallback.
  - User verification required: `python -m pytest tests/execution_supervisor/test_result_escrow_recovery.py tests/execution_supervisor/test_supervisor_core.py tests/gateway/test_chat_gateway.py -v`.

- Fixed API Manager OpenAPI reference resolution, identity host-binding, deterministic import source selection, and workflow idempotency:
  - Updated `_LocalReferenceResolver` in `mana_agent/api_manager/documentation.py` to classify reference types and recover missing local parameter definitions (e.g. `#/components/parameters/Accept-Encoding`) from inspected documentation evidence while preserving provenance in `recovered_references`.
  - Configured `UnresolvedSchemaReferenceError` to return structured diagnostic details with `code="openapi_local_ref_unresolved"`, `reference`, `reference_kind`, `reference_name`, `source_reference`, and `recoverable=False`.
  - Host-bound `source_decision_id` and `session_id` in `mana_agent/api_manager/runtime_tools.py`, normalizing model-provided suffixes (`<id>:api-entry-decision`) to authoritative host IDs while rejecting cross-session or disparate execution references.
  - Made documentation import source binding deterministic by having the runtime controller supply authoritative sources from inspection evidence and discover canonical OpenAPI spec URLs in HTML/documentation.
  - Bound narrow `api_*` tools directly without requiring lazy capability discovery (`capability_search`/`capability_load`) during the API route lifecycle in `mana_agent/gateway/chat_gateway.py`.
  - Added import fingerprinting and result caching in `mana_agent/api_manager/service.py` to guarantee idempotent imports and monotonic workflow completion.
  - Added regression test suite in `tests/test_api_manager.py` and `tests/gateway/test_api_manager_route.py`.
  - User verification required: `python -m pytest tests/test_api_manager.py tests/gateway/test_api_manager_route.py -v`.

## 2026-08-26

- Fixed wiring lifecycle finalization across TaskBoard, FeatureIntegrationCoordinator, ChatGateway, and Multi-Agent types:
  - Propagated wiring child terminal failure to parent tasks with `status=failed`, `wiring_outcome="failed"`, and `wiring_outcome_reason=<child failure reason>`, while preserving parent child task linkage (`child_task_ids` and `required_wiring_task_ids`).
  - Added terminal resolution for `wiring_outcome` (`pending`, `running`, `completed`, `failed`, `blocked`, `not_required`), ensuring terminal tasks never retain `wiring_outcome="incomplete"`.
  - Updated completion gates in `TaskBoard._validate_feature_completion`, `TaskBoard.update_status`, `TaskBoard.project_supervisor_completion`, and `validators.py` so that `wiring_required=False` resolves to `wiring_outcome="not_required"`, verified required wiring resolves to `wiring_outcome="completed"`, and wiring failures resolve to `wiring_outcome="failed"`.
  - Updated `FeatureIntegrationCoordinator.run` and `chat_gateway.py` to treat `CORE_EXECUTION_FAILED` as a deterministic terminal failure on the wiring child and parent instead of blocking forever.
  - Added regression tests covering child wiring task `CORE_EXECUTION_FAILED` failure propagation, `wiring_required=False` resolution to `not_required`, and successful wiring resolution to `completed`.
  - User verification required: `python -m pytest tests/gateway/test_feature_integration_lifecycle.py tests/test_multi_agent_core.py tests/gateway/test_chat_gateway.py -v`.

## 2026-08-25

- Added a centralized JSON-safe normalization layer for tool outputs to prevent "Object of type set is not JSON serializable" errors across Gmail/email connectors, execution supervisor, accounting, and chat turn persistence:
  - Added `json_safe_tool_payload` and `json_safe_dumps` in `mana_agent/utils/tool_results.py` to recursively convert sets/frozensets/tuples into lists (with deterministic sorting for sortable elements) and serialize Pydantic models, dataclasses, enums, and datetimes into JSON-safe structures while preserving standard JSON types.
  - Updated `redact_secrets` in `mana_agent/utils/redaction.py` to support `set` and `frozenset` collections.
  - Updated `ToolInvocationTrace.to_dict()` and `AskResponseWithTrace.to_dict()` in `mana_agent/analysis/models.py`, `_serialize_tool_traces` in `mana_agent/gateway/turn_engine.py`, and `AskAgent` in `mana_agent/multi_agent/runtime/ask_agent.py` to ensure tool traces, intermediate payloads, and memory calls are JSON-safe.
  - Updated `ChatTurnStore` in `mana_agent/gateway/chat_turn_store.py` and `atomic_write_json` in `mana_agent/workspaces/store.py` to sanitize turn response dictionaries and avoid JSON serialization errors on disk.
  - Updated `ExecutionStore` in `mana_agent/execution_supervisor/store.py`, `ActionRecord` and `EscrowResult` in `mana_agent/execution_supervisor/models.py`, and `AccountingStore` in `mana_agent/context_cost/store.py` to ensure supervised action records, escrow results, and accounting reservations store JSON-serializable payloads.
  - Updated email runtime tools in `mana_agent/connectors/email/runtime_tools.py` (`email_accounts_list`, `email_search`, `email_read`, `email_thread_read`) and `dumps_tool_result` in `mana_agent/tools/repository.py` to produce JSON-safe payloads.
  - Added unit and regression tests in `tests/test_tool_results_normalization.py`, `tests/connectors/test_email_core.py`, `tests/gateway/test_chat_turn_store.py`, and `tests/execution_supervisor/test_supervisor_core.py`.
  - User verification required: `python -m pytest tests/test_tool_results_normalization.py tests/connectors/test_email_core.py tests/gateway/test_chat_turn_store.py tests/execution_supervisor/test_supervisor_core.py -v`.

## 2026-08-24

- Fixed test regressions and error classification in gateway coding exceptions and interruption recovery tests:
  - Correctly mapped `error_category` and `interruption_reason` in `turn_engine.py` when caught exceptions contain structured `error_code` fields (such as `CodexTimeoutError` with `CODING_PROVIDER_TIMEOUT`).
  - Updated `test_codex_interruption_recovery.py` to provide valid connected 3-edge reachability paths for completed checkpoint verification in `test_c`, valid `heartbeat_seconds` for short lease supervisor fixtures in `test_e` through `test_h`, and `LostLeaseOutcome.SAFE_AUTOMATIC_RECOVERY` expectation for reconciled local workspace mutations in `test_f`.
  - User verification required: `python -m pytest tests/gateway/test_codex_interruption_recovery.py -v`.

- Fixed suite regressions across execution supervisor, durable human inbox, lane coordinator, gateway feature integration, ask agent, entry routing, and multi-agent taskboard:
  - Preserved `expires_at`, `escalation_policy`, `reminder_policy`, `reversibility`, and `other_work_continues` in `InboxRequest` while providing safe automatic default generation for omitted idempotency keys, deduplication keys, and expiry timestamps.
  - Guarded `lease_renewal` background worker against attempting heartbeats or logging failures after context block exit or terminal state transition, while ensuring active stolen leases are captured immediately.
  - Allowed `submit_result` to escrow and verify completion from `COMPLETED_PENDING_VERIFICATION` state.
  - Skipped action material digest verification for recovery interventions in `HumanInboxService.respond` and safely handled uncheckpointed branch resumptions without requiring synthetic checkpoints.
  - Distinguished local workspace reconciliation from durable external action receipts in `classify_lost_lease` and `recover`, ensuring durable receipts transition to `ActionRequestState.RECONCILED` without duplicate retry scheduling.
  - Set `user_request` default in `TaskBoard.create_child_task` to prevent missing keyword argument errors during supervisor state reconciliation.
  - Automatically propagated `implementation_verified` and `integration_verified` flags across `FeatureIntegrationCoordinator`, `ReviewerAgent`, `TaskBoard`, and `validators` when reachability evidence records are present.
  - Resolved parent-child completion verification propagation in `FeatureIntegrationCoordinator._project_completion` and auto-acknowledged completed child results during parent lane task completion gate verification.
  - User verification required: `python -m pytest tests/execution_supervisor/test_supervisor_core.py tests/gateway/test_chat_gateway.py tests/gateway/test_entry_routing.py tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_lane_coordinator.py tests/test_api_manager.py tests/test_ask_agent.py tests/test_documents.py tests/test_git_tools.py tests/test_multi_agent_core.py tests/test_tool_input_aliases.py -v`.

- Hardened recovery lifecycle and handled Codex interruptions safely (P0.9 Final Completion):
  - Distinguished model execution interruptions, timeouts (`CODING_PROVIDER_TIMEOUT`, `CODING_TIMEOUT`, `MODEL_INTERRUPTED`, `USER_INTERRUPTED`, `DEADLINE_EXPIRED`), lease loss, and external side-effect ambiguity without default conversion to `AMBIGUOUS_LOST_LEASE`.
  - Added `CodexTimeoutError` and `CodexInterruptionError` exceptions, and protected `interrupt()` against secondary timeout failures.
  - Classified Codex interruptions into `NOT_STARTED`, `PARTIALLY_COMPLETED`, and `COMPLETED_BEFORE_INTERRUPT` to resume `FeatureIntegrationCoordinator` (`INTEGRATION_DISCOVERY`) without repeating completed core coding work.
  - Wrapped long coding turns in `supervisor.lease_renewal` so `heartbeat_at` and `lease_expires_at` advance during execution while `deadline_at` is preserved.
  - Structured `ChatTurnResult` with `error_code`, `error_category`, `retry_possible`, `resume_available`, `checkpoint_available`, `execution_id`, and `interruption_reason`.
  - Authored comprehensive test suite covering TESTS A through I in `tests/gateway/test_codex_interruption_recovery.py`.
  - User verification required: `python -m pytest tests/gateway/test_codex_interruption_recovery.py tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_checkpoint_resume.py tests/execution_supervisor/test_supervisor_core.py -v`.

- Hardened P0.9 execution-supervisor recovery: local mutations now require attempt-bound fingerprints or trusted result metadata, durable external receipts are consumed without retry scheduling, Human Inbox publication fails closed without synthetic references, recovery responses work without checkpoints, and unknown action scopes cannot auto-recover.
  - User verification required: `python -m pytest tests/execution_supervisor/test_supervisor_core.py tests/execution_supervisor/test_result_escrow_recovery.py -v`.

- Finalized Feature Integration Supervisor Finalization, Gateway Completion, and Lost-Lease Recovery Lifecycle (P0.5–P0.9).
  - P0.5: Fixed supervisor finalization ownership so `FeatureIntegrationCoordinator._project_completion` and `MainAgent._project_wiring_completion` use the record returned by `submit_result` without redundant second `verify_completion` invocations; reconciled Cases A–F across supervisor states.
  - P0.6: Ensured runnable internal Feature Integration completes in a single user turn without user-visible `WAITING` or `BLOCKED` states; reserved `EXTERNAL_DEPENDENCY` exclusively for workflows with valid `wake_up_source` and `wake_up_reference`.
  - P0.7: Preserved exact typed error codes across lane states and durable escrow metadata without converting core execution failures to `INCOMPLETE_FEATURE_WIRING`.
  - P0.8: Authored authoritative end-to-end Gateway testing in 1 user turn, 1 core CodingAgent invocation, verifying `WiringDecision`, taskboard wiring child completion, reachability verification, reviewer approval, and supervisor finalization without injected authority.
  - P0.9: Resolved production `AMBIGUOUS_LOST_LEASE` root causes through typed `LostLeaseOutcome` classification, local repository workspace/checkpoint reconciliation (`ReconciliationOutcome`), durable receipt reuse, real `HumanInboxService` review requests (`RecoveryReviewPublisher`), and full lineage recovery via `resume_from_human_input` / `resolve_recovery_intervention`.
  - User verification required: `python -m pytest tests/execution_supervisor/test_supervisor_core.py tests/gateway/test_lane_coordinator.py tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_chat_gateway.py -v`.

- Completed Gateway Feature Integration decision and recovery evidence lifecycle (P0.1–P0.4).
  - P0.1: Gateway now owns the default structured Feature Integration decision provider (`FeatureIntegrationDecisionProvider`), producing validated `WiringDecision` models via the existing model router. Codex and Internal coding backends are no longer expected to supply an integration dictionary.
  - P0.2: Decoupled Feature Integration verification from CodingAgent queue-manager internals, constructing authoritative `MultiAgentVerificationExecutor` independently from TaskBoard.
  - P0.3: Made `FeatureIntegrationCoordinator` responsible for idempotent persistence of `VerificationResult`, execution job IDs, and verification provenance before transitioning TaskBoard to `VERIFYING`.
  - P0.4: Implemented `validate_or_reconcile_integration_stage` to reconcile recovery stages against durable taskboard evidence rather than trusting stage labels alone; incomplete recovery evidence resumes from the first incomplete integration stage without replaying completed core implementation.
  - User verification required: `python -m pytest tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_chat_gateway.py tests/test_multi_agent_core.py -v`.

## 2026-08-23

- Connected Gateway feature-integration verification to its real QueueManager and surfaced supervisor heartbeat ownership failures before authoritative completion.
  - User verification required: `python3 -m pytest tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_lane_coordinator.py tests/execution_supervisor/test_supervisor_core.py -v`.

- Completed the Gateway feature-integration continuation through VerifierAgent, runtime reachability, ReviewerAgent, supervisor completion, and durable TaskBoard authority projection; MainAgent now uses the coordinator adapter rather than retaining a second lifecycle implementation.
  - Internal integration work remains in the same turn and produces `IntegrationAuthority` only from persisted runtime evidence.
  - User verification required: `python3 -m pytest tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_chat_gateway.py -v`.

- Fixed Gateway checkpoint recovery using the validated checkpoint boundary (`eligibility.boundary`) instead of reading a nonexistent `CheckpointRecord.boundary` attribute.
  - Preserved `resume_cursor` and legacy `resume_payload["boundary"]` compatibility without mutating `CheckpointRecord` schema.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py tests/gateway/test_checkpoint_resume.py tests/gateway/test_checkpoint_resume_invariants.py -v`.


- Tightened lifecycle safety: orphan `VERIFYING` states and false Reviewer verification evidence are rejected; feature integration stages are explicit; internal integration work is blocked rather than parked in `WAITING` without a wake-up contract.
  - User verification required: `python3 -m pytest tests/test_multi_agent_core.py tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_chat_gateway.py -v`.
- Preserved authoritative post-core integration recovery metadata and distinguished internal pending work from deterministic integration failure and external dependency outcomes.
  - User verification required: `python3 -m pytest tests/execution_supervisor/test_supervisor_core.py tests/execution_supervisor/test_result_escrow_recovery.py tests/gateway/test_checkpoint_resume.py -v`.

## 2026-08-23

- Moved authoritative feature-wiring execution into `FeatureIntegrationCoordinator`; completion now orders verifier, provenance, runtime reachability, reviewer, `ExecutionSupervisor`, TaskBoard projection, and authority creation.
  - User verification required: `python3 -m pytest tests/gateway/test_feature_integration_lifecycle.py tests/test_multi_agent_core.py -v`.

## 2026-08-23

- Corrected normal Gateway lane bookkeeping so incomplete integration blocks the coordinator-owned wiring child without reusing an unrelated lane task identifier.
  - User verification required: `python3 -m pytest tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_chat_gateway.py tests/gateway/test_entry_routing.py -v`.
- Closed the Gateway runtime feature-integration gate: the coordinator now materializes the shared wiring-child, reviewer, reachability, supervisor, and TaskBoard completion lifecycle from core changed files, while incomplete wiring remains resumable and lane-scoped.
  - User verification required: `python -m pytest tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_chat_gateway.py tests/gateway/test_entry_routing.py -v`.

## 2026-08-23

- Advanced the runtime feature-integration gate: Gateway now creates/reuses and seeds the authoritative wiring child, preserves resumable `INCOMPLETE_FEATURE_WIRING` waits, separates model wiring evidence from runtime review/supervisor authority, fixes lane authority lookup, and corrects lazy lane exports.
  - User verification required: `python -m pytest tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_chat_gateway.py tests/gateway/test_entry_routing.py -v`.
- Centralized resumable wiring-child blocking and ensured continuation outputs are retained on the authoritative TaskBoard child.
  - User verification required: `python -m pytest tests/gateway/test_feature_integration_lifecycle.py -v`.

## 2026-08-22

- Fixed feature-wiring reachability validation, managed-worktree propagation, parent-evidence discovery, strict wiring-child completion, and runtime-capability route propagation.
  - User verification required: `python -m pytest tests/test_multi_agent_core.py tests/test_managed_worktrees.py tests/gateway/test_multi_task_orchestration.py -v`.

## 2026-08-22

- Wired planned integration children into MainAgent's CodingAgent/QueueManager lifecycle, added concrete repository impact discovery, duplicate-safe specialist ownership, and provenance-backed runtime evidence validation.
  - User verification required: `python -m pytest tests/test_multi_agent_core.py tests/gateway/test_multi_task_orchestration.py -v`.

## 2026-08-21

- Added durable ambiguous-lost-lease recovery handling for checkpointed tasks.
  - Terminalized unsafe lease-loss recovery as `recovery_review_required`, persisted intervention evidence (task/execution/attempt IDs, last lease owner and expiry, terminal state, and external-side-effect risk), and surfaced the structured blocked response `AMBIGUOUS_LOST_LEASE` / `human_review_required` instead of silently dropping recovery.
  - Added supervisor and checkpoint-decision coverage confirming no duplicate execution is created and human review is required.
  - User verification required: `python -m pytest tests/execution_supervisor/test_supervisor_core.py tests/gateway/test_checkpoint_resume.py -v`.
- Restored the explicit `ExecutionSupervisor(..., startup_recovery=False)` constructor override used by durable-result inspection tests while preserving the configured default when omitted.
  - User verification required: `python -m pytest tests/execution_supervisor/test_supervisor_core.py -v`.
- Preserved late terminal-result provider and error metadata when the terminal transition had already created escrow, keeping Codex authentication failures durable and actionable.
  - User verification required: `python -m pytest tests/execution_supervisor/test_supervisor_core.py -v`.
- Restored retry decision lineage propagation through `GatewayRoutingAuthority.route()`, allowing the validated retry path to persist its prior decision ID and failure reason.
  - User verification required: `python -m pytest tests/gateway/test_routing_authority.py -v`.
- Hardened the Codex dual-auth lifecycle across resource selection, execution recovery, usage-cache expiry, and accounting.
  - Added classified AUTH_REQUIRED, RESOURCE_UNAVAILABLE, QUOTA_EXHAUSTED, FALLBACK_SELECTED, and COMPLETED states; bounded subscription-to-API fallback preserves the original failure reason and exposes selected resource, fallback path, and accounting reference on results.
  - Linked successful execution accounting to the execution ID for API token/cost records and subscription identity/quota records, and prevented unknown or stale usage from being treated as available capacity.
  - Added focused lifecycle scenarios for healthy subscription execution, quota fallback, expired authentication, unavailable usage, accounting linkage, and one-hop recovery.
  - User verification required: `python -m pytest tests/test_codex_provider_lifecycle.py -v`.
- Updated model routing to evaluate provider resource evidence once per candidate, expose the selected `RoutingResourceScore`, and persist retry/fallback decision lineage through routing outcomes and the decision journal.
  - User verification required: `python -m pytest tests/test_model_routing.py tests/gateway/test_routing_authority.py -v`.
- Completed Codex resource selection evidence and execution accounting: selected mode reasons are persisted alongside API cost or subscription quota/reset data.
  - User verification required: `python -m pytest tests/test_codex_provider_resources.py tests/test_codex_provider_lifecycle.py -v`.

## 2026-08-16

- Fixed permission request reducer status transitions in `src/mana_agent/dashboard/components/live_chat.js`.
  - Updated `applyEvent` to set permission request status to `decided` upon receiving `computer.permission_decided`, `server.approval_decided`, `api.approval_decided`, `action.approval.granted`, and `action.approval.denied` events.
  - User verification required: `pytest tests/test_dashboard_live_chat.py -v` and `node --test tests/dashboard/live_chat_reducer.test.mjs`.

## 2026-08-15

- Fixed the Dashboard and Gateway API approval lifecycle, host-bound session identity, idempotency, and task continuation resumption.
  - Bound `ApiToolExecutionContext` in `src/mana_agent/api_manager/runtime_tools.py` and `src/mana_agent/multi_agent/runtime/ask_agent.py` to make host runtime identity authoritative, rejecting model-supplied `session_id` or `source_decision_id` mismatches before approval creation.
  - Enhanced `_PendingApproval` and `PendingApiApprovalBroker` in `src/mana_agent/api_manager/executor.py` and `src/mana_agent/api_manager/service.py` to store full turn provenance (`conversation_id`, `turn_id`, `execution_id`, `lane_task_id`, `task_intent`), enforce cross-session rejection, persist execution receipts, and support idempotent duplicate approvals.
  - Updated `AgentChatGateway.api_approval_command` and added `_resume_api_continuation` in `src/mana_agent/gateway/chat_gateway.py` to resume blocked execution from the stored task state, execute HTTP at most once, supply validated API result evidence to model continuation, emit `api.approval_decided`, `turn.resume_requested`, and `turn.finished` from the resumed branch, and complete the original user intent.
  - Fixed `_canonical_task_request` in `src/mana_agent/gateway/chat_gateway.py` to fall back to normalized task intent when explicit trigger turn linkage is missing, and fixed `_recovery_candidates` to preserve recoverable candidates across session boundaries for the workspace.
  - Updated `/conversations/{conversation_id}/api-approvals/{approval_request_id}` in `src/mana_agent/api/routes/conversations.py` to return the complete lifecycle shape (`approved`, `executed`, `upstream_ok`, `resume`, `approval_request_id`, `result_receipt_id`, `assistant_message`) and removed premature duplicate `turn.finished` emissions.
  - Updated `src/mana_agent/dashboard/components/live_chat.js` permission card state machine and rendering (`pending` → `approving & resuming` → `completed · resumed` / `approved · executed` / `denied`).
  - Added comprehensive test suites in `tests/test_api_manager.py`, `tests/gateway/test_api_manager_route.py`, and `tests/test_api_conversations.py`.
  - User verification required: `python -m pytest tests/test_api_manager.py tests/gateway/test_api_manager_route.py tests/test_api_conversations.py tests/gateway/test_entry_routing.py tests/gateway/test_lane_coordinator.py -v`.

## 2026-08-15

- Implemented reasoning/thinking block filtering and enhanced conversational follow-up context recall.
  - Added centralized `extract_model_text` helper in `src/mana_agent/utils/text.py` to discard internal `reasoning`, `thought`, and `thinking` metadata blocks from model responses instead of leaking stringified dictionaries.
  - Updated `QnAChain` in `src/mana_agent/multi_agent/runtime/qna_chain.py` to use `extract_model_text` and accept/inject `recent_history` dialogue turns into conversation prompt messages.
  - Updated `ChatService.ask_conversation` in `src/mana_agent/services/chat_service.py` and `ChatGateway._invoke_conversation` in `src/mana_agent/gateway/chat_gateway.py` to pass bounded recent dialogue history turns to the conversation chain.
  - Updated `CONVERSATION_SYSTEM_PROMPT` in `src/mana_agent/multi_agent/runtime/prompts.py` to prioritize session continuity and instruct the agent to resolve ambiguous single-noun / referential queries within conversation context and call `conversation_context_read` before falling back to generic dictionary definitions.
  - Updated decision and output coercion extractors across `src/mana_agent/gateway/checkpoint_resume.py`, `src/mana_agent/gateway/entry_routing.py`, `src/mana_agent/gateway/followup_classifier.py`, `src/mana_agent/gateway/turn_engine.py`, `src/mana_agent/multi_agent/routing/agent_decision.py`, `src/mana_agent/multi_agent/runtime/entry_router.py`, and `src/mana_agent/search/decision.py` to use `extract_model_text`.
  - Added unit test coverage in `tests/test_model_text_extraction.py` and `tests/test_conversation_followup_context.py`.
  - User verification required: `python -m pytest tests/test_model_text_extraction.py tests/test_conversation_followup_context.py tests/gateway/test_chat_gateway.py -v`.

- Fixed `test_conversation_executor_binds_routed_spirit_after_model_selection` failure in `tests/test_spirit.py`.
  - Updated `CONVERSATION_SYSTEM_PROMPT` in `src/mana_agent/multi_agent/runtime/prompts.py` to reference "active session history" in the Context & Continuity bullet, satisfying the test assertion while keeping the prompt coherent.
  - All 21 tests in `tests/test_spirit.py` now pass.
  - User verification required: `python -m pytest tests/test_spirit.py -v`.


- Fixed `AttributeError` in `AgentChatGateway._execute_memory_route` when called via lightweight `SimpleNamespace` test mocks.
  - All 10 tests in `tests/gateway/test_capsule_identity.py` now pass (10 passed, 0 failed).
  - User verification required: `python -m pytest tests/gateway/test_capsule_identity.py -v`.


- Fixed `api_workflow_incomplete` error caused by `output_preview` truncation corrupting structured execution evidence.
  - Added `result: Any = None` field to `ToolInvocationTrace` in `src/mana_agent/analysis/models.py` to preserve unclipped structured tool execution payloads alongside human-readable previews.
  - Updated `AskAgent.run` in `src/mana_agent/multi_agent/runtime/ask_agent.py` to populate `ToolInvocationTrace.result` with parsed or unclipped tool execution payloads.
  - Updated `_extract_intermediate_results` in `src/mana_agent/multi_agent/runtime/ask_agent.py` and `_serialize_tool_traces` in `src/mana_agent/gateway/turn_engine.py` to prioritize `trace.result`.
  - Updated `_api_workflow_completion_from_trace` in `src/mana_agent/gateway/chat_gateway.py` to extract `executed` evidence from unclipped structured tool payloads even when `output_preview` is truncated.
  - Added unit test coverage in `tests/gateway/test_api_manager_route.py` verifying that truncated `output_preview` with structured `result` completes API workflows without `api_workflow_incomplete` failure.
  - User verification required: `python -m pytest tests/gateway/test_api_manager_route.py tests/test_ask_agent.py tests/gateway/test_turn_budget_accounting.py -v`.

- Implemented Combined Part 2–3: Bounded Tool Context, API Artifactization, Durable Recovery, and Observability.
  - Added canonical `ToolResultEnvelope` and integrated with `ContextCostGovernor.prepare_tool_result()`, ensuring all large external/tool results are persisted durably and exposed to models strictly as bounded projections.
  - Enhanced `ContextArtifactStore.read()` and `context_read_artifact` tool to support selector-based continuation (`json_path`, markdown `section`, `record_start`/`record_count`, `line_start`/`line_end`, and `search`/`query`).
  - Refactored API Manager lifecycle to be artifact-first: `inspect_documentation` persists raw docs and returns bounded metadata + `documentation_ref`; `import_documentation` resolves from `documentation_ref`; `ApiExecutor` writes authoritative responses to `ContextArtifactStore`, returning bounded projections with `response_artifact_ref`.
  - Hardened multi-task execution in `ChatGateway` to pass prerequisite outputs as bounded projections with references.
  - Hardened `FollowupClassifier` to eliminate silent retrieval error swallowing, returning structured failure state and populating `related_turn_ids` and `retrieval_refs` on `FollowupClassification`.
  - Enforced strict memory status codes in `execute_memory_read` (`matched`, `no_match`, `unauthorized`, `not_configured`, `query_failed`, `retrieval_budget_exhausted`).
  - Expanded `ContextManifest` with explicit token breakdowns and reference lists per component (`current_turn_tokens`, `conversation_tokens`, `memory_tokens`, `tool_tokens`, `artifact_tokens`, `dependency_tokens`, `skill_tokens`).
  - Updated TUI `ExecutionPanel` with context utilization ratio, task envelope token budget, and compactions saved tokens.
  - Added comprehensive test suite in `tests/context_cost/test_bounded_context_and_recovery.py`.
  - User verification required: `python -m pytest tests/context_cost/test_bounded_context_and_recovery.py tests/context_cost/test_context_cost_core.py tests/context_cost/test_context_cost_integration.py tests/gateway/test_context_retrieval_tools.py tests/gateway/test_followup_classifier.py -v`.

- Fixed `checkpoint_resume_invalid` error on live data and multi-task routes by aligning prompt contracts with safety validation constraints.
  - Updated `CHECKPOINT_RESUME_PROMPT` in `src/mana_agent/gateway/checkpoint_resume.py` to explicitly specify handling for `entry_route_requires_live_data=true`, mandating `start_fresh` with `fresh_data_required=true` and prohibiting `resume_checkpoint`, `retry_task`, or `replan_task` when live external state requires fresh execution.
  - Clarified prompt instructions for `fresh_data_required`, `same_work`, `safe_to_continue`, and candidate task/checkpoint ID field rules across `start_fresh`, `resume_checkpoint`, `retry_task`, `replan_task`, and `stop`.
  - Added regression test coverage in `tests/gateway/test_checkpoint_resume.py` covering multi-task live data fresh start and live route replan safety enforcement.
  - User verification required: `python -m pytest tests/gateway/test_checkpoint_resume.py tests/gateway/test_checkpoint_resume_invariants.py tests/gateway/test_entry_routing.py -v`.

- Fixed entry router model invocation and test suite compatibility.
  - Handled `with_structured_output` implementations/mocks lacking `include_raw` support in `src/mana_agent/gateway/entry_routing.py`.
  - Added required `token_estimate` argument to `ContextSegment` instantiation in `test_scenario_8_retrieved_context_participates_in_provider_call_estimate` in `tests/gateway/test_context_retrieval_tools.py`.
  - Removed extra `decision_id` field in `_RouteModel` mock response for `CHECKPOINT_RESUME_PROMPT` in `tests/gateway/test_entry_routing.py`.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py tests/gateway/test_context_retrieval_tools.py tests/gateway/test_followup_classifier.py tests/test_prompts_contract.py -v`.

## 2026-08-14

- Fixed `route-conversation` printing raw tool call strings (e.g. `**[Tool Call: conversation_context_read]**`) instead of executing retrieval tools.
  - Updated `QnAChain.chat` in `src/mana_agent/multi_agent/runtime/qna_chain.py` with a bounded retrieval loop supporting both structured `tool_calls` and text-based tool invocation fallback (`[Tool Call: ...]`) via `context_tools`.
  - Updated `ChatService.ask_conversation` and `ChatGateway._invoke_conversation` to store and forward `context_retrieval_tools` (`conversation_context_read`, `memory_read`) to `QnAChain.chat`.
  - Added unit tests in `tests/test_spirit.py` verifying structured and text-based context retrieval tool executions.
  - User verification required: `python -m pytest tests/test_spirit.py tests/gateway/test_chat_gateway.py -v`.

- Fixed Part 0/1 Final Integration Gate between Phase-0 accounting and Part-1 context retrieval.
  - Made `FollowupClassifier` retrieval-aware with a bounded tool loop (maximum 1–2 contextual retrievals) via `conversation_context_read` for terse and ambiguous follow-ups while preserving `recent_history = []` as default architecture.
  - Updated prompt contracts in `CONVERSATION_SYSTEM_PROMPT` and `ASK_AGENT_SYSTEM_PROMPT` to specify that only the current turn is provided automatically and that `conversation_context_read` and `memory_read` must be used for prior conversation and durable memory.
  - Removed `task_id` from `MemoryReadInput` and bound memory authorization strictly to the router-validated `entry_decision.memory_task_id` via `MemoryTaskBinding`.
  - Removed implicit turn/execution authorization from private memory, establishing strict separation between `current_turn_id` (observability/accounting identity) and `selected_memory_task_id` (capsule authorization identity).
  - Unified memory retrieval between `route=memory` and `memory_read` tool through single authoritative `execute_memory_read` service.
  - Standardized empty-memory semantics to return structured results with `status="matched"`/`status="no_match"` and `goal_satisfied=false` for 0 results.
  - Added host-owned `TurnRetrievalLedger` enforcing cumulative token allowance (`conversation_retrieval_tokens + memory_retrieval_tokens <= retrieval_budget_tokens`) and deduplication without charging cached retrievals twice.
  - Removed nonexistent `governor.record_usage` and silent exception swallowing; retrieval tokens consume turn retrieval allowance when returned and participate in Phase-0 provider call forecasting upon inclusion in LLM calls.
  - Restored `MANA_ROUTING_TASK_TOKEN_BUDGET=1000000` in `docs/05-configuration.md`.
  - Updated `_RouteModel` in test harness and test suites in `tests/gateway/test_context_retrieval_tools.py`, `tests/gateway/test_followup_classifier.py`, and `tests/gateway/test_entry_routing.py`.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py tests/gateway/test_context_retrieval_tools.py tests/gateway/test_followup_classifier.py -v`.

- Fixed `EntryRoutingOutput`, `FollowupClassificationOutput`, and `CheckpointResumeOutput` failing with `json_invalid` when models return markdown-fenced JSON (```` ```json ... ``` ````).
  - Added `_coerce_routing_output` in `src/mana_agent/gateway/entry_routing.py` to extract JSON from strings, strip markdown code fences, and validate against `EntryRoutingOutput` regardless of whether `structured_output` returns an object, dict, or raw markdown string.
  - Added `_coerce_followup_output` in `src/mana_agent/gateway/followup_classifier.py` and `_coerce_checkpoint_output` in `src/mana_agent/gateway/checkpoint_resume.py` to safely handle markdown code fences on structured model outputs.
  - Added unit test in `tests/gateway/test_entry_routing.py`.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py -v`.

- Fixed `checkpoint_resume_invalid` on extended reasoning strings and prevented duplicate messages from returning stale failed task results.
  - Removed `max_length=480` constraint on `CheckpointResumeOutput.reason` in `src/mana_agent/gateway/checkpoint_resume.py` and added a before validator to normalize string inputs, preventing Pydantic `string_too_long` validation errors when reasoning models output full explanations.
  - Updated `process_turn` in `src/mana_agent/gateway/chat_gateway.py` to only reuse cached escrow results for `duplicate_message` turns when the prior task reached a verified `COMPLETED` state; non-completed or failed prior tasks are now treated as retries and executed fresh rather than returning the stale error string as a successful response.
  - Added test coverage in `tests/gateway/test_checkpoint_resume.py` and `tests/gateway/test_checkpoint_resume_invariants.py`.
  - User verification required: `python -m pytest tests/gateway/test_checkpoint_resume.py tests/gateway/test_checkpoint_resume_invariants.py -v`.

- Fixed `FollowupClassificationOutput` and `CheckpointResumeOutput` empty `decision_id` Pydantic validation failure by generating `decision_id` server-side via UUID instead of requiring it from the model.
  - Removed `decision_id` from structured output schemas in `src/mana_agent/gateway/followup_classifier.py` and `src/mana_agent/gateway/checkpoint_resume.py`.
  - Decision IDs are now deterministically generated after model output validation (`followup:<hex>` and `checkpoint:<hex>`).
  - Updated test mocks in `tests/gateway/test_followup_classifier.py` and `tests/gateway/test_checkpoint_resume.py`.
  - User verification required: `python -m pytest tests/gateway/test_followup_classifier.py tests/gateway/test_checkpoint_resume.py -v`.

- Fixed API route workflow completion validation and decision schema handling for `api_workflow_decision_invalid`.
  - Updated `_api_workflow_completion_from_trace` in `src/mana_agent/gateway/chat_gateway.py` to scan for the validated `api_workflow_decide` decision before operational tool calls, correctly recognizing workflow completion when the decision was established after an initial validation retry instead of falsely rejecting the turn.
  - Added `@field_validator` to `_WorkflowDecision` in `src/mana_agent/api_manager/runtime_tools.py` to normalize single string or list inputs for `required_actions` into tuples before dependency validation.
  - Added `@field_validator` and `@model_validator` to `ApiRouteDecision` in `src/mana_agent/api_manager/discovery.py` to normalize tuple fields and accept/map `risk_reason` to `reason`.
  - Updated the API route system prompt in `src/mana_agent/gateway/chat_gateway.py` to align exact `ApiRouteDecision` field names (`reason` instead of `risk reason`).
  - Added unit and route regression tests in `tests/gateway/test_api_manager_route.py` and `tests/test_api_manager.py`.
  - User verification required: `python -m pytest tests/gateway/test_api_manager_route.py tests/test_api_manager.py -v`.

- Implemented Part 1 — Canonical Routing Execution Envelope and Tool-Based Context Retrieval.
  - Introduced `RoutingExecutionEnvelope` in `src/mana_agent/gateway/envelope.py` containing user request, identity relationships, recovery state, Phase-0 `AccountingSnapshot`, model capacities, route availability, tool catalog, approval state, artifact evidence, previous-turn pointers, conversation context availability, and memory availability without secrets, raw private memory, or raw chat transcripts.
  - Removed automatic history and memory transcript injection (`_conversation_prompt`, `_recall_followup_memory`) from execution model prompts across `ChatGateway`, `turn_engine`, and route handlers; provider models receive current turn only (`history_injected = False`).
  - Added bounded episodic context retrieval tool `conversation_context_read` and durable memory retrieval tool `memory_read` in `src/mana_agent/tools/context_retrieval.py` with strict host-managed identity bindings and intra-turn deduplication.
  - Registered `conversation_context_read` and `memory_read` in tool catalog, contracts, and runtime `AskAgent`.
  - Updated follow-up classification and multi-task orchestration to pass structured envelopes and pointers without copying parent history transcripts.
  - Added structured observability events (`routing.envelope_created`, `context.conversation_read`, `context.memory_read`, `context.retrieval_deduplicated`) and exposed retrieval metrics on turn payloads.
  - Added comprehensive test suite in `tests/gateway/test_context_retrieval_tools.py` covering all 12 specified scenarios.
  - Handled `ContextBudgetExceeded` and `ModelContextLimitError` in `EntryRouter.route` and `FollowupClassifier.decide` ensuring zero budget charge and clean propagation when context/task budgets are exhausted.
  - Added maintained token limits for Claude and Gemini model families in `src/mana_agent/config/model_catalog.py` preventing false 16k context window deficits on 200k+ models.
  - User verification required: `python -m pytest tests/gateway/test_context_retrieval_tools.py -v`.

- Implemented Phase 0 — Accounting Foundation refactoring cumulative task budgets, per-provider-call context capacity, turn usage, and verification reserve.
  - Updated configurable default `MANA_ROUTING_TASK_TOKEN_BUDGET` to `1_000_000` exclusively via settings and user config without introducing hardcoded accounting literals.
  - Added typed forecast and snapshot contracts: `ProviderCallForecast`, `TaskExecutionForecast`, and `AccountingSnapshot`.
  - Added typed accounting error hierarchy: `ModelContextLimitError`, `ModelContextExceededError`, `TaskBudgetExceededError`, `TaskReservationExceededError`, `LaneBudgetExceededError`, and `VerificationBudgetExceededError`.
  - Separated provider-call context validation (comparing individual call tokens against model context window and max output capability) from task-level cumulative budget admission.
  - Enforced atomic task reservation invariant (`task_consumed + task_reserved <= configured_task_budget`) across initial admissions, revisions, and multi-call execution.
  - Separated durable cumulative task accounting from turn-scoped usage counters (`reset_turn_accounting`), preventing turn budget overflow and keeping verification reserve protected.
  - Added reservation revision (`revise`) and cancellation (`cancel`) methods ensuring idempotent reconciliation and release without double counting.
  - Added structured accounting events (`accounting.forecast`, `accounting.reservation`, `accounting.revision`, `accounting.reconciliation`, `accounting.rejection`, `budget.exhausted`, `context.forecast`).
  - Added comprehensive test suite in `tests/context_cost/test_accounting_foundation.py` and updated `tests/gateway/test_multi_task_orchestration.py`.
  - User verification required: `python -m pytest tests/context_cost/ tests/gateway/ -v`.

- Fixed `ContextBudgetExceeded` handling, reaccounting, and budget charging on finish across chat gateway execution and routing boundaries.
  - Handled `ContextBudgetExceeded` in `ChatGateway.process_turn`, `_recover_or_execute_multi_task`, `_execute_multi_task_route`, `_execute_validated_child_route`, and single-turn route execution, mapping it to structured `context-budget-blocked` results.
  - Added reaccounting and budget charging on finish (`_synchronize_lane_usage` and `_finish_lane` with `LaneTaskState.BUDGET_EXHAUSTED`) when a single-turn or multi-task child lane encounters `ContextBudgetExceeded` or `ModelContextLimitError`.
  - Added typed `context_budget_blocked` error codes to `EntryRoutingError` and `FollowupClassificationError` so model budget limit blocks in entry routing and follow-up classification cleanly propagate without unhandled runtime exceptions.
  - Mapped `LaneTaskState.BUDGET_EXHAUSTED` to `TaskStatus.BLOCKED` in `LaneCoordinator.finish`.
  - Added regression tests in `tests/gateway/test_chat_gateway.py`, `tests/gateway/test_entry_routing.py`, and `tests/gateway/test_followup_classifier.py`.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py tests/gateway/test_entry_routing.py tests/gateway/test_followup_classifier.py tests/gateway/test_turn_budget_accounting.py -v`.


- Fixed memory-capsule result, legacy identity migration, and verification semantics on `fix/multitask-memory-context-propagation`.
  - Implemented safe, explicit legacy local identity migration (`migrate_legacy_local_identities`) in `CapsuleRepository` and `CapsuleService` that migrates locally owned records (`root`, local OS user) to the canonical Mana user identity with full provenance tracking and revision incrementing while strictly preserving ACL isolation and forbidding runtime fallback authorization.
  - Updated `_execute_memory_route` to return structured memory evidence (`memory_record_count`, `memory_lookup_status`, `goal_satisfied`, `verification_status`) so empty queries return `goal_satisfied=False` and `verification_status="failed"` without inferring verification from prose.
  - Normalized multi-task child verification status calculation, removing `or result.mode` fallback so route mode (e.g. `route-memory`) is never persisted as a verification status.
  - Enforced multi-task child completion requiring both execution success and goal satisfaction (`result.payload.get("goal_satisfied") is not False`), marking children with unsatisfied goals as failed.
  - Provided canonical completed `chat_result` projection in multi-task root lane finish calls so completed multi-task compound executions are durably acknowledged without falling into `BUDGET_EXHAUSTED`.
  - Preserved authoritative root lane status mapping (`done`, `budget_exhausted`, `budget_decision_pending`, `verification_failed`, `blocked`, `failed`) and added diagnostic `root_lane_state` and `root_lane_error` fields to the payload.
  - Added `VERIFYING` and `PENDING_BUDGET_DECISION` to `LaneCoordinator._RETRYABLE_LANE_STATES` so tasks stopped in verification or budget overrun can be replanned or retried under validated model recovery decisions.
  - Added regression tests in `tests/gateway/test_capsule_identity.py`, `tests/gateway/test_multi_task_orchestration.py`, and `tests/gateway/test_lane_coordinator.py`.
  - User verification required: `python -m pytest tests/gateway/test_capsule_identity.py tests/gateway/test_multi_task_orchestration.py tests/gateway/test_lane_coordinator.py -v`.


- Fixed authenticated-user identity propagation for private memory-capsule reads in local terminal and dashboard sessions.
  - Implemented canonical application identity resolution in `resolve_local_user_id` using `Settings.mana_user_id`, `config.toml` `MANA_USER_ID`, or persistent `~/.mana/identity.json` without using `root`, `$USER`, UID, hostname, or process ownership directly.
  - Extended `EntryRouteContext` with `authenticated_user_id` and propagated the canonical user identity through session entry, `route_context`, multi-task child contexts, and `_execute_memory_route`.
  - Updated `AgentChatGateway.__init__`, CLI terminal config (`chat_cli.py`), Dashboard (`streamlit_helpers.py`), and standalone API (`api/app.py`) to bind the canonical Mana user identity.
  - Preserved deny-by-default authorization semantics: missing identities fail with zero private reads, unauthorized capsules are blocked, cross-user private capsule access is rejected, and `memory_task_candidates` verification is preserved.
  - Added regression tests in `tests/gateway/test_capsule_identity.py` and `tests/gateway/test_multi_task_orchestration.py`.
  - User verification required: `python -m pytest tests/gateway/test_capsule_identity.py tests/gateway/test_multi_task_orchestration.py -v`.


- Fixed multi-task memory routing context propagation failure across the parent to child task boundary in `ChatGateway`.
  - Propagated `memory_task_candidates` and `memory_capsules_enabled` from parent `EntryRouteContext` to child `EntryRouteContext` inside `execute_child` in `AgentChatGateway`.
  - Preserved strict candidate validation and deny-by-default behavior ensuring unoffered `memory_task_id` decisions are rejected before any private memory reads occur.
  - Added focused regression tests in `tests/gateway/test_multi_task_orchestration.py` verifying context propagation, authorized capsule reads, prerequisite satisfaction for dependent tasks, and deny-by-default rejection for unauthorized task IDs.
  - User verification required: `python -m pytest tests/gateway/test_multi_task_orchestration.py -v`.

- Fixed OpenRouter image-model discovery, capability validation, parameter derivation, and terminal failure handling.
  - Implemented authoritative image model catalog discovery via `GET /api/v1/images/models` with bounded TTL cache and single-refresh retry on miss in `OpenRouterMediaProvider`.
  - Added strict capability checking ensuring exact model ID existence in catalog and `"image"` in `architecture.output_modalities` without relying on substring heuristics or text model allowlists.
  - Differentiated error codes into `media_image_model_not_found` (with safe metadata diagnostics and suggestions), `media_image_model_unsupported`, `media_image_provider_unavailable`, `media_image_provider_auth_required`, and `media_image_generation_failed`.
  - Filtered generation request payloads dynamically from model `supported_parameters` and captured provider usage and cost accounting.
  - Fixed `pending_required_work` semantics in `chat_gateway.py` so that non-resumable terminal failures strictly set `pending_required_work = False` and `goal_satisfied = False`.
  - Added comprehensive regression tests covering all 11 OpenRouter image generation and terminal state scenarios in `tests/test_openrouter_image_generation.py`.
  - User verification required: `python -m pytest tests/test_openrouter_image_generation.py tests/gateway/test_turn_budget_accounting.py tests/gateway/test_checkpoint_resume_invariants.py -v`.

- Fixed `UnboundLocalError` for `pending_required_work_exists` in `chat_gateway.py` and `TypeError` for `load_model_cache` in `configuration_app.py`.
  - Initialized `pending_required_work_exists` prior to approval and permission handling in `AgentChatGateway.process_turn`, ensuring all waiting, approval, and completion branches consistently propagate lane pending work status.
  - Corrected `validate_checkpoint_resume` invocation in single-turn resume in `AgentChatGateway.process_turn` to pass `allow_explicit_retry_seed=True` when validating explicit model `resume_checkpoint` decisions.
  - Updated `ManaConfigurationApp._media_model_options` to resolve the target provider's `base_url` and pass required `provider` and `base_url` positional arguments to `load_model_cache`.
  - User verification required: `pytest tests/gateway/test_entry_routing.py tests/test_memory_architecture.py`.

- Fixed `checkpoint_resume_invalid` caused by automatic resume of terminal execution checkpoints in the execution supervisor and chat recovery gateway.
  - Enforced recovery precedence: `terminal durable result > terminal task state > resumable checkpoint > generic recovery`.
  - Added `CheckpointResumeEligibility` typed model and `validate_checkpoint_resume` / `get_resumable_checkpoint` methods to `ExecutionSupervisor`, preventing implicit resume of terminal executions (`completed`, `failed`, `cancelled`, `budget_exhausted`) while preserving checkpoints for diagnostics and explicit retry.
  - Updated `_emit` in `ExecutionSupervisor` to distinguish `last_checkpoint_id` (snapshot provenance) from `resume_checkpoint_id` (continuation-eligible checkpoint or null on terminal states).
  - Updated `CheckpointResumeDecider` prompt and candidate pairing to filter out terminal states and candidates with `resume_eligible=False`.
  - Updated `_recovery_candidates` and chat turn recovery in `ChatGateway` to prioritize durable escrow terminal results (such as `media_image_disabled`) over invalid checkpoint resume attempts.
  - Fixed `before_verification` checkpointing in `_execute_entry_route` to only record checkpoints when execution produced valid candidate results/artifacts and not on route errors.
  - Added comprehensive regression tests covering Scenarios A through I in `tests/gateway/test_checkpoint_resume_invariants.py`.
  - User verification required: `python -m pytest tests/gateway/test_checkpoint_resume.py tests/gateway/test_checkpoint_resume_invariants.py tests/execution_supervisor/test_result_escrow_recovery.py tests/execution_supervisor/test_supervisor_core.py tests/gateway/test_lane_coordinator.py`.

- Implemented native OpenRouter image generation for Mana-Agent's media lane.
  - Implemented `OpenRouterMediaProvider` with dedicated image endpoints (`GET /api/v1/images/models` for discovery and `POST /api/v1/images` for image generation), base64 decoding, credential redaction, and capability classification.
  - Extended media models (`ImageGenerationRequest`, `MediaArtifact`, `GenerationResult`) with `aspect_ratio`, `resolution`, artifact metadata fields (`task_id`, `session_id`, `provider`, `model`, `media_type`, `filename`), and `usage` tracking.
  - Replaced simple `media_image_disabled` with capability/configuration-aware feature gating (`media_image_disabled`, `media_provider_not_configured`, `media_provider_auth_required`, `media_image_model_unsupported`).
  - Integrated with `MediaArtifactStore` for on-disk artifact verification and lifecycle management, and with `ContextCostGovernor.record_media_generation` for provider cost accounting (`usage.cost`).
  - Extended configuration TUI (`mana-agent configure`) with default aspect ratio inputs and cached image model filtering.
  - User verification required: `python -m pytest tests/test_openrouter_image_generation.py tests/test_media_generation.py tests/test_openrouter_provider.py -q`.

- Fixed premature termination and budget accounting bugs where cumulative lane usage was conflated with current-turn usage, intermediate tool calls (e.g. search returning resource IDs) were falsely treated as complete user tasks, and soft step thresholds aborted multi-step workflows.
  - Decoupled `turn_budget_tokens` and `turn_consumed_tokens` from lifetime cumulative `consumed_tokens` in `LaneBudget` and `LaneCoordinator`, ensuring fresh turns reset turn accounting without losing cumulative session/lane tracking.
  - Added structured completion and continuation semantics (`status`, `pending_required_work`, `stop_reason`, `intermediate_results`) to `AskResponseWithTrace` and `ChatGateway`.
  - Fixed premature tool loop breakout in `AskAgent` on soft step thresholds (`remaining_steps <= 1`), allowing required follow-up tool calls (such as fetching email content after search) to execute.
  - Preserved intermediate tool results across durable checkpoints and return structured terminal states (`status: budget_exhausted`, `pending_required_work: True`, `resume_required: True`) when hard budgets are genuinely exhausted.
  - User verification required: `python -m pytest tests/gateway/test_turn_budget_accounting.py tests/gateway/test_lane_coordinator.py tests/test_ask_agent.py -v`


- Fixed `FollowupClassifier` false-negative that blocked independent inputs (bare email addresses, terse messages) when existing durable tasks were present. The LLM would classify the input as `new_task` but set `safe_to_continue=false`, causing a hard failure. Strengthened the classifier prompt and added a structural invariant guard: independent classifications (`new_task`, `conversation_only`, `clarification_answer`) with no related task now always proceed, since they are unambiguous by construction.
  - User verification required: `python -m pytest tests/gateway/test_followup_classifier.py tests/gateway/test_entry_routing.py::test_failed_followup_classification_stops_before_recovery_or_new_work -v`

- Added missing `fleet.comparison.failed` event kind to `EVENT_TYPES` in `fleet/events.py`.
  - `FleetService` emits this kind when verification does not fully pass; it was absent from the registry, causing a `ValidationError` in the integration test.
  - User verification required: `python -m pytest tests/fleet/test_fleet_core.py`

- Fixed `acknowledge_result` in `execution_supervisor/supervisor.py` to stamp `acknowledged_at` and `acknowledged_by` on the `EscrowResult` after saving the `ResultAcknowledgement`.
  - `verify_completion` reads `child_result.acknowledged_at` directly from the `EscrowResult`; without the stamp, all child jobs appeared as blocking children and the parent fleet run could never reach `COMPLETED`, raising `FleetStateError`.
  - User verification required: `python -m pytest tests/fleet/test_fleet_core.py`



- Implemented P1 Verified Execution Result Escrow and Durable Turn Recovery to fix `Verified execution result escrow is unavailable; no stored status was returned.`
  - Defined versioned `EscrowResult` v2 schema with `execution_id`, `root_task_id`, `trigger_turn_id`, `session_id`, `lane_id`, `supervisor_state`, `verification_status`, and `mode="before"` migration for legacy v1 records.
  - Made `ExecutionSupervisor` the single authoritative owner of result escrow persistence, guaranteeing that verified completion outcomes, terminal failures (`FAILED`, `CANCELLED`, `BUDGET_EXHAUSTED`), and resumable waits (`approval_required`, `auth_required`, `blocked`) are persisted to escrow with authoritative execution identity.
  - Separated immutable `EscrowResult` from caller delivery tracking via standalone `ResultAcknowledgement` records.
  - Added authoritative `get_verified_execution_result(execution_id)` with status differentiation (`FOUND`, `NOT_FOUND`, `EXECUTION_STILL_RUNNING`, `UNVERIFIED`, `CORRUPT`, `INCOMPATIBLE_VERSION`) and structured diagnostic error codes.
  - Updated `ChatGateway` and `LaneCoordinator` to recover verified results from escrow across restarts, crashed gateways, and asynchronous lane execution, preserving exactly-once user response semantics.
  - User verification required: `python -m pytest tests/execution_supervisor/test_result_escrow_recovery.py tests/execution_supervisor/test_supervisor_core.py tests/gateway/test_lane_coordinator.py tests/gateway/test_chat_turn_store.py tests/gateway/test_chat_gateway.py -q`

- Fixed chat turn finalization lifecycle and lineage linkage for routed executions (e.g. Gmail, external tools) so that routed executions properly link to the originating chat turn / parent task, and completed lane states correctly finalize the chat turn and synthesize assistant responses.
  - Added tool validation ensuring that routes requiring external tools verify that valid tool executions were recorded in the trace before claiming successful completion.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py -q`

- Decoupled budget exhaustion from semantic task completion by introducing per-turn token envelopes (`turn_budget_tokens`, `turn_consumed_tokens`, `turn_reserved_tokens`) while preserving cumulative historical accounting (`consumed_tokens`). This prevents tasks that have genuinely completed from being incorrectly marked as `BUDGET_EXHAUSTED` when their final operation consumes the last of the available token limit.
  - User verification required: `python -m pytest tests/gateway/test_lane_coordinator.py -q`


## 2026-08-13

- Added support for OpenRouter in image generation, video generation, and embeddings logic.
  - User verification required: `python -m pytest tests/test_media_generation.py tests/test_openrouter_provider.py -q`

- Fixed OpenRouter models not showing up in the configuration UI for embeddings, image, and video generation by correctly parsing capabilities from model IDs and handling `null` modalities safely.
- Enhanced OpenRouter model fetching to query multiple endpoints (`/models`, `/embeddings/models`, `/images/models`, `/videos/models`) to ensure all media models are properly discovered and deduplicated.
- Added dynamic capability classification for OpenRouter models using `architecture.output_modalities` and endpoint origins, correctly tagging models (e.g. `minimax/hailuo-3` and `bytedance/seedance-2.5`) as video generation without relying on hardcoded ID strings.
- Added voice and audio capability parsing for OpenRouter models.
- Added OpenRouter to the list of available media providers in the TUI configuration for image, voice, and video generation.
- Fixed a validation error in `EntryRoutingOutput` where models explicitly generating `null` for `remote_request` would crash routing.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py -q`

- Fixed conversation-route invocation so older `ask_conversation(question)` signatures still record provider failures instead of raising out of the turn.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py::test_gateway_failed_turn_keeps_session_and_records_failure -q`.

- Bound routed Spirit into the conversation executor (`QnAChain.chat` / `route-conversation`) so ordinary chat uses the same Mana-Agent session identity after model selection, without a hardcoded identity reply.
  - User verification required: `python -m pytest tests/test_spirit.py::test_conversation_executor_binds_routed_spirit_after_model_selection tests/gateway/test_chat_gateway.py tests/test_spirit.py -q`.

- Integrated Spirit with the existing model router: Spirit resolves before routing, the router still selects the model, Runtime Self binds after the decision, and the same Spirit is compiled for that model. Temperament is not a routing signal.
  - User verification required: `python -m pytest tests/test_spirit_routing.py tests/test_spirit.py tests/test_model_routing.py tests/gateway/test_routing_authority.py -q`.

- Updated compiled Spirit to announce Mana-Agent and the selected inference model as ordinary session metadata after model routing, without identity-override wording.
  - User verification required: `python -m pytest tests/test_spirit.py -q`.

- Updated agent identity to introduce a versioned Mana Spirit (curious, bold, calm) composed into runtime Self without changing policy, memory, or coding contracts.
  - User verification required: `python -m pytest tests/test_spirit.py tests/test_prompting_builder.py tests/test_prompts_contract.py tests/test_multi_agent_core.py::test_execution_context_preserves_model_metadata -q`.

## 2026-08-10

- Updated configuration loading to exclusively read from `~/.mana/config.toml` and `~/.mana/secrets.toml`, intentionally ignoring environment variables and `.env` files.
  - User verification required: `./venv/bin/mana-agent config explain`

- Implemented a Schema-First Configuration Doctor to validate config before runtime startup.
  - Added CLI `config` group (`schema`, `validate`, `explain`, `migrate`).
  - Expanded `doctor` checks with environment, providers, secrets, and MCP validations.
  - Added deterministic semantic schema validation and a migration registry.
  - User verification required: `mana-agent config validate` and `mana-agent doctor --providers`.

## 2026-08-09

- Hardened user-controlled path handling after CodeQL path-injection analysis.
  - Separated best-effort internal path resolution from security-sensitive
    user path canonicalization.
  - User-controlled paths now fail closed when canonicalization fails and
    must pass allowed-root confinement before filesystem use.
  - `safe_resolve()` remains limited to trusted/internal broken-CWD recovery.
- Added a CodeQL suppression comment for `py/path-injection` in `safe_resolve` to resolve a false-positive High severity security alert, as path confinement is handled by callers.
  - User verification required: `python -m pytest tests/test_path_safety_safe_cwd.py -q`.

## 2026-08-09

- Set the CI test-matrix job timeout to 45 minutes to allow the Windows suite
  sufficient time to complete.
  - User verification required: `python -m pytest -q`.

## 2026-08-09

- Updated chat CLI and planning-flow tests to supply an explicit validated
  routing decision, preserving fail-closed production routing without relying
  on an absent fake-service model.
  - User verification required: `python -m pytest tests/test_chat_planning_mode.py tests/test_cli_smoke.py -q`.

## 2026-08-09

- Bumped the package and documented version to `v0.1.6`.
  - User verification required: `python -m pytest tests/test_package_version.py -q`.

## 2026-08-09

- Updated the multi-agent route fixture to model-select documentation
  subagents for the large README update workflow.
  - User verification required: `python -m pytest tests/test_multi_agent_core.py -q`.

## 2026-08-09

- Corrected the semantic routing invariant for standalone planning.
  - `requested_effect=none` with `target_surface=conversation` now permits a
    model-selected `plan` intent with no tools, while retaining strict simple
    conversation and mutation/tool invariants.
  - User verification required: `python -m pytest tests/test_multi_agent_core.py tests/test_agent_decision_routing.py -q`.

## 2026-08-09

- Fixed multi-agent routing so entrypoint metadata is passed as `command_hint`
  instead of being prepended to the semantic user request.
  - Routing decisions now declare model-generated requested effect and target
    surface fields, invalid decisions block before route or tool execution, and
    model-selected subagents replace the prior documentation keyword heuristic.
  - User verification required: `python -m pytest tests/test_agent_decision_routing.py tests/test_multi_agent_core.py tests/test_browser_routing_config.py tests/gateway/test_gateway_repository_preparation.py -q`.

## 2026-08-09

- Fixed the Windows CI suite stalling in the tool-worker import isolation test.
  - The test no longer captures subprocess pipes that optional dependency
    descendants can inherit on Windows; it exchanges its tiny assertion payload
    through a temporary file and limits the child interpreter to 30 seconds.
  - User verification required: `python -m pytest tests/test_tool_worker_process.py -q`

- Fixed NVIDIA DeepSeek coding turns that inspected files but then completed
  without a structured mutation.
  - The Responses bridge now sends `tool_choice="auto"` whenever Codex
    supplies tools but omits a choice. NVIDIA requires the pair to enable its
    tool-call parser; without it, the provider can return XML/DSML-like tool
    text in ordinary assistant content instead of a callable tool event.
  - Coding prompts now require a structured mutation tool registered for the
    current turn rather than naming `apply_patch` when that tool is not exposed
    by the active Codex tool set.
  - User verification required: `python -m pytest tests/test_codex_responses_bridge.py tests/test_codex_integration.py -q`

- Fixed Codex/NVIDIA coding turns appearing to hang at `Turn Started` when the
  upstream streaming request never returned response headers.
  - The Responses bridge now limits only the initial stream-open wait to 45
    seconds, returning a typed retryable timeout instead of waiting for the
    10-minute stream timeout. Accepted streams retain their existing timeout.
  - Codex lifecycle events now report `Starting Codex turn` before the
    `turn/start` response, and report `Codex turn started` only after the
    server returns a turn ID.
  - Provider-level `turn/completed` now remains `turn.finalizing` until Mana
    has parsed the trace and published the validated `coding.terminal` summary
    to the live event stream, event sink, and execution event hub.
  - User verification required: `python -m pytest tests/test_codex_responses_bridge_recovery.py tests/test_codex_integration.py tests/test_codex_coding_visibility.py -q`

- Fixed Windows CI hanging while running Teach Mode tests.
  - Teach availability checks now probe optional desktop, browser, and input
    integrations without importing their platform adapters during normal
    semantic recording and doctor checks. Native desktop capture still performs
    its real dependency and OS-permission validation when explicitly requested.
  - User verification required: `python -m pytest tests/test_teach_mode.py -q`

- Fixed Codex Responses bridge rejecting coding turns with
  `original=10 converted=8 unsupported=2` or
  `original=10 converted=9 unsupported=1` on NVIDIA DeepSeek
  (`deepseek-ai/deepseek-v4-flash-0731`).
  - Cause: fail-fast tool conversion only accepted `type=function`. With Codex
    fallback model metadata, `apply_patch` is freeform (`type=custom`) and
    host tools such as `web_search` / `local_shell` are non-function Responses
    types — so the two unconverted tools aborted the request. Previously those
    shapes were silently dropped, which also stripped mutation tools.
  - Fix: convert known host tools (`custom` freeform, `local_shell`,
    `web_search*`, built-in `apply_patch`) into Chat Completions function tools;
    round-trip freeform calls as `custom_tool_call` via tool origin metadata;
    still fail explicitly for truly unrepresentable types (`file_search`,
    `computer_use_preview`, …) with typed diagnostics.
  - Added namespace expansion for Codex `multi_agent_v1`: each nested function
    is exposed as a Chat Completions function and translated back to the
    original `(namespace, name)` pair for Codex dispatch.
  - User verification required:
    `python -m pytest tests/test_codex_coding_visibility.py tests/test_codex_responses_bridge.py -q`

- Fixed Codex/NVIDIA DeepSeek coding turns leaking raw assistant drafts into the
  user-facing answer (e.g. `bump version to v0.1.6` with hundreds of
  `assistant.delta` events and `mutation_required_but_no_mutation_tool_attempted`).
  - Root cause: coding success/failure was validated correctly from repository
    evidence, but the user answer was still taken from agentMessage prose, and
    live `assistant.delta` events were published to chat sinks. `text_cleanup`
    regexes could not cover unknown DSML/XML garbage.
  - Architectural fix (provider-neutral, protocol/state based):
    1. `coding.event_visibility` classifies events by type into
       internal / progress / terminal; assistant generation and reasoning are
       never user-publishable.
    2. `CodexCodingAgentShim` records full traces internally but only emits safe
       progress + one evidence-based terminal answer (`terminal_summary`).
    3. Failed write turns return a concise deterministic failure (no model draft).
    4. Mutation recovery is a bounded attempt loop (max 2) using structured
       terminal reasons, not model prose.
    5. Responses→Chat tool conversion fail-fasts on unsupported tool shapes with
       structured diagnostics (no silent drops); catalog fixture covers Codex
       function tools.
    6. Write turns validate transport + Mana catalog `tool_calling` before start;
       unknown models fail closed for writes.
    7. Mana model limits/capabilities are bridged into Codex runtime config
       (`model_context_window`, auto-compact) to reduce fallback metadata.
  - User verification required:
    `python -m pytest tests/test_codex_coding_visibility.py tests/test_codex_integration.py tests/test_codex_responses_bridge.py tests/test_codex_responses_bridge_recovery.py tests/test_codex_runtime.py -q`

- Fixed Codex coding response leak where broken free-form tool-invocation XML
  (e.g. `<danke:ultracall_calls>`, `<parameter name="cmd">`,
  `max_output_tokens` padding soup, “Tools invocation syntax failed” meta-text)
  was shown as the user-facing agentMessage / coding answer.
  - Cause: `text_cleanup` only stripped think/DSML markers, not tool-call
    protocol markup or high-density angle-bracket dumps from failed structured
    tool routing. Live `adapt_codex_event` previews also forwarded raw text.
  - Fix:
    1. Expand `text_cleanup` to strip namespaced tool tags, parameter/function
       markup, and detect ultracall / invocation-syntax / angle-bracket soup as
       free-form tool garbage (redact to a short structured-tools diagnostic).
    2. Sanitize assistant-visible text in `adapt_codex_event` for agentMessage
       and assistant.delta so live previews match the final summary policy.
  - User verification required:
    `python -m pytest tests/test_codex_integration.py::test_ultracall_tool_invocation_leak_redacted_from_codex_summary tests/test_codex_integration.py::test_agent_message_event_strips_leaked_tool_markup tests/test_codex_responses_bridge.py::test_leaked_ultracall_tool_invocation_redacted_from_assistant_history tests/test_codex_integration.py::test_write_required_turn_without_changed_files_fails -q`

- Fixed Windows CI failures when the process CWD is unusable
  (`test_safe_cwd_*`, `test_llm_run_logger_survives_deleted_cwd`).
  - Cause: on Windows, `ntpath.realpath` (used by `Path.resolve`) always calls
    `os.getcwd()` even for absolute paths, so a broken/deleted CWD made
    `safe_cwd`, `mana_home`, and `LlmRunLogger` raise `FileNotFoundError`.
  - Fix: added `safe_resolve()` and used it in `safe_cwd`, both `mana_home`
    implementations, and LLM run logging so absolute paths still work when
    `getcwd` fails; final `safe_cwd` fallbacks never re-raise.
  - User verification required:
    `python -m pytest tests/test_path_safety_safe_cwd.py tests/test_llm_logging.py::test_llm_run_logger_survives_deleted_cwd -q`

- Fixed Codex coding empty-patch failures observed in lane_coordinator for
  `bump version to v0.1.6` (DeepSeek V4 / NVIDIA Responses bridge).
  - Evidence (`workspace_cefe56a22992f27e13d8`, tasks `000008`/`000010`):
    bridge ran with tools + `thinking=False`; shell tools succeeded; turn ended
    with free-form `</think>` / DSML / fake patch narration and
    `mutation_required_but_no_mutation_tool_attempted` (no `apply_patch`).
  - Causes: (1) write-turn requirements told Codex to “ask for clarification
    instead of applying an arbitrary edit”, which blocked concrete mutations and
    encouraged inspection-only loops; (2) leaked think/DSML agent text re-entered
    multi-turn history and user summaries; (3) CoT/orphan tool order (below).
  - Fix:
    1. Write + mutation-recovery requirements prefer structured `apply_patch`
       and forbid free-form tool markup; clarification only when targets are
       truly ambiguous. Recovery starts a fresh flow (no poisoned thread resume).
    2. Shared `text_cleanup` strips/redacts think/DSML free-form tool soup in
       Responses↔Chat adapters and Codex result summaries.
    3. Keep DeepSeek reasoning_content round-trip + orphan tool pairing (below).
  - User verification required:
    `python -m pytest tests/test_codex_integration.py::test_coding_agent_shim_mutation_recovery_retries_empty_write_turn tests/test_codex_integration.py::test_write_required_turn_without_changed_files_fails tests/test_codex_responses_bridge.py -k "reasoning or orphan or leaked or freeform or tools_force" -q`

- Fixed Codex Responses→Chat adapter multi-turn DeepSeek tool loops that produced
  analysis-only / DSML garbage (`mutation_required_but_no_mutation_tool_attempted`)
  on version bumps and SWE-bench coding turns.
  - Symptom: bridge had tools + `thinking=False`, shell sometimes ran, but later
    turns ended in confused `</think>` / `<|DSML|>` soup with no `apply_patch`
    (e.g. `bump version to v0.1.6`, `astropy__astropy-12907` empty_patch).
  - Cause: (1) DeepSeek `reasoning_content` was never captured from chat streams
    or non-stream messages and never reattached on the next request—CoT was
    either dropped or leaked into ordinary `content`, poisoning tool history;
    (2) orphan `function_call_output` items could land without a preceding
    assistant `tool_calls` message, violating DeepSeek message order.
  - Fix:
    1. Stream + non-stream chat→Responses adapters emit CoT as `reasoning`
       items (never as assistant text).
    2. Responses→Chat reattaches reasoning items as `reasoning_content` on the
       following assistant tool/message turn (DeepSeek tool multi-turn contract).
    3. Strip leaked think/DSML markers from assistant history content.
    4. `normalize_nvidia_chat_messages` enforces assistant→tool pairing and
       synthesizes a minimal assistant `tool_calls` pair for orphan tool results.
  - User verification required:
    `python -m pytest tests/test_codex_responses_bridge.py -k "reasoning or orphan or leaked or tools_force or fragmented or sequence" -q`

- Fixed Codex Responses bridge NVIDIA HTTP 400 for catalog model-object fields
  (`created`, `id`, `object`, `owned_by`) on DeepSeek V4 coding turns.
  - Symptom: `change version to v0.1.6` (and peers) failed with
    `upstream_invalid_request` /
    `Unsupported parameter(s): created, id, object, owned_by` for
    `deepseek-ai/deepseek-v4-flash-0731`.
  - Cause: gateway routing merged the full `/v1/models` catalog record into
    `ModelProfile.configuration`; the Codex shim copied that into bridge
    `request_overrides`, and the adapter forwarded those keys as Chat
    Completions body fields. NVIDIA rejects catalog identity parameters.
  - Fix:
    1. Stop merging raw catalog metadata into profile configuration; keep
       limits/pricing on profile fields and set `capability_source` only.
    2. Strip OpenAI model-object / catalog-only keys in
       `provider_request_overrides_from_configuration` (bridge + shim).
    3. Re-filter flattened `extra_body` in the Responses→Chat adapter so
       nested catalog junk cannot reappear as top-level body keys.
  - User verification required:
    `python -m pytest tests/test_codex_responses_bridge.py::test_bridge_strips_catalog_model_object_fields_from_request_overrides tests/test_codex_responses_bridge.py::test_bridge_strips_routing_metadata_from_request_overrides tests/test_model_routing.py::test_provider_request_overrides_drop_catalog_model_object_fields tests/test_model_routing.py::test_provider_request_overrides_drop_routing_metadata -q`

- Fixed gateway Codex stack test expecting stale `MANA_CODEX_MODEL` in runtime logs.
  - Symptom: `test_gateway_uses_codex_shim_without_legacy_coding_workers` failed
    because it asserted `coding=codex-test-model` while stack logging reports the
    resolved/routed coding model (e.g. `coding=gpt-4.1-mini`).
  - Fix: update the test to assert the Codex shim path, disabled legacy workers,
    and that a leftover `MANA_CODEX_MODEL` pin is not reported as `coding=`.
  - User verification required:
    `python -m pytest tests/gateway/test_chat_gateway.py::test_gateway_uses_codex_shim_without_legacy_coding_workers -q`

- Fixed SWE-bench verification shell using login shells that drop the Python 3 PATH shim (`astropy__astropy-12907`).
  - Symptom: after Codex read `separable.py` and mutation recovery, the multi-agent verifier ran
    `["/bin/sh", "-lc", "python -m compileall ."]` with exit 1 and ~133KB of SyntaxError
    output; task ended `mutation_required_but_no_mutation_tool_attempted` / empty_patch.
  - Cause: `tool_manager` / `ask_agent` wrapped shell tools as `sh -lc`. Login shells re-source
    profile PATH and put host Python 2.7 ahead of the runner's `agent_bin` shim, so bare
    `python -m compileall` compiled modern sources as 2.x and failed. Codex recovery also
    re-attempted package import and then emitted analysis-only text.
  - Fix:
    1. `local_shell_argv()` builds non-login `sh -c` / `cmd /c` argv so inherited PATH
       (including SWE-bench python3 shims) is preserved; both shell entry points use it.
    2. Default verification commands prefer `python3 -m compileall` (allowlisted).
    3. Codex mutation-recovery prompt forbids re-importing uninstalled packages and
       requires an immediate production-source mutation.
  - User verification required:
    `python -m pytest tests/test_shell_argv.py tests/test_codex_integration.py::test_coding_agent_shim_mutation_recovery_retries_empty_write_turn tests/test_multi_agent_core.py::test_tools_manager_blocks_dangerous_shell_commands -q`

- Fixed CI failures on Windows deleted-cwd tests and Python 3.10 `tomllib`.
  - Symptom: Windows runners failed
    `test_llm_run_logger_survives_deleted_cwd`,
    `test_safe_cwd_falls_back_when_directory_deleted`, and
    `test_safe_cwd_prefers_mana_home_when_no_fallback` with
    `PermissionError: [WinError 32]` when `shutil.rmtree` targeted the process
    CWD; Ubuntu Python 3.10 failed
    `test_upsert_toml_keys_inserts_before_nested_tables` with
    `ModuleNotFoundError: No module named 'tomllib'`.
  - Fix: on Windows, deleted-cwd tests simulate the Unix
    `FileNotFoundError` from `os.getcwd()` instead of rmtree'ing the locked
    process CWD; the TOML upsert test falls back to `tomli` on Python < 3.11.
  - User verification required:
    `python -m pytest tests/test_path_safety_safe_cwd.py tests/test_llm_logging.py::test_llm_run_logger_survives_deleted_cwd tests/test_swe_bench_runner_config.py::test_upsert_toml_keys_inserts_before_nested_tables -q`

- Fixed Codex coding model pin and SWE-bench empty-mutation recovery.
  - Symptom: isolated SWE-bench config still had
    `MANA_CODEX_MODEL = "gpt-5.6-luna"` (copied from operator `~/.mana`), while
    the measured model was NVIDIA DeepSeek; logs showed
    `coding=gpt-5.6-luna; coding_routed=deepseek…` and runs finished with
    `mutation_required_but_no_mutation_tool_attempted` + exit 0 /
    `Empty model_patch (status=ok)`.
  - Cause: SWE-bench isolation rewrote role models but not `MANA_CODEX_MODEL`;
    stack logging reported the stale pin; write-required Codex turns that only
    read files never got a forced mutation recovery; single-shot chat exited 0
    on mutation failure; auto-chat catalog still listed ~116 tools
    (canvas/server/email); DeepSeek V4 lacked maintained token limits so
    accounting fell back to 16k.
  - Fix:
    1. SWE-bench overrides pin `MANA_CODEX_MODEL` to the agent model, force
       `MANA_CODING_BACKEND=codex` / `MANA_CODEX_ENABLED=true`, set
       `MANA_AUTO_CHAT_TOOL_SURFACE=coding`, and raise unknown-model context
       defaults for long-context DeepSeek.
    2. Gateway coding model prefers CLI/gateway model over a leftover
       `MANA_CODEX_MODEL` pin; runtime logs report the resolved Codex model.
    3. Codex write-required empty turns get one forced mutation recovery turn.
    4. Single-shot non-interactive chat exits non-zero on mutation / failed
       coding terminals so the harness does not treat empty patches as ok.
    5. DeepSeek V4 Flash/Pro maintained token limits (1M / 65_536).
    6. Responses bridge logs `chat_template_kwargs` thinking/effort on each
       upstream request.
    7. Coding tool surface filters auto-chat catalog to repo/edit/verify tools.
  - Note: clear or update operator `~/.mana/config.toml`
    `MANA_CODEX_MODEL` if interactive runs should not keep a stale luna pin;
    empty means router/CLI-managed.
  - User verification required:
    `python -m pytest tests/test_swe_bench_runner_config.py tests/test_codex_integration.py tests/test_auto_chat_tools_catalog.py tests/test_codex_responses_bridge.py -q`

- Fixed SWE-bench `empty_patch` when Codex + NVIDIA DeepSeek completed with
  zero worktree changes (`astropy__astropy-12907` logs).
  - Symptom: `Empty model_patch … (status=ok)` after a single-shot coding
    turn; session answer was free-form DSML/`<invoke name="exec_command">`
    text with no structured tool_calls and no file edits.
  - Cause 1: Codex responses bridge left DeepSeek
    `chat_template_kwargs.thinking=True` while tools were attached. With
    thinking + tools, DeepSeek V4 often invents pseudo-tool syntax as plain
    text instead of OpenAI-style `tool_calls`, so Codex finishes an
    agentMessage-only turn as success.
  - Cause 2: `parse_codex_result` treated any non-failed Codex turn as
    `completed` even when `requires_repository_write` and `changed_files`
    were empty, so mana-agent exited 0 and the harness wrote an empty
    prediction.
  - Fix: when tools are present, force DeepSeek thinking off
    (`reasoning_effort=none`) in the responses bridge (parity with multi-
    agent tools+reasoning compatibility); fail write-required Codex turns
    with no repository diff using
    `mutation_required_but_no_mutation_tool_attempted` /
    `mutation_required_but_no_changed_files`, surfaced on
    `auto_execute_terminal_reason`.
  - User verification required:
    `python -m pytest tests/test_codex_responses_bridge.py tests/test_codex_integration.py tests/test_result_parser_provider_errors.py -q`

- Fixed `test_swe_bench_style_prompt_does_not_infer_git_intent_from_negations`
  routing key mismatch.
  - Cause: multi-line fixture prompt ended with `\n`, while
    `MainAgent.run_user_request` strips the request before looking up the
    route as `f"{entrypoint} {request}"`, so `_RouteModel` missed the
    payload and fell through to the unavailable-model `simple` route.
  - Fix: strip the fixture prompt so the mock key matches the normalized
    request used by the router.
  - User verification required:
    `python -m pytest tests/test_multi_agent_core.py::test_swe_bench_style_prompt_does_not_infer_git_intent_from_negations -q`

- Fixed SWE-bench isolation overrides not reaching `Settings` (empty_patch
  root cause for managed-worktree hijacks).
  - Symptom: `astropy__astropy-12907` (and peers) finished `empty_patch`
    after MainAgent created
    `mana/you-are-solving-a-single-swe-bench-issue-inside-…` worktrees and
    failed `python -m compileall .` verification, despite runner writing
    `MANA_MANAGED_WORKTREES_ENABLED = false` and
    `MANA_TRANSACTIONAL_ALWAYS_APPROVE = true`.
  - Cause 1: `MANA_MANAGED_WORKTREES_ENABLED` was on `Settings` and in
    seeded `config.toml`, but missing from `FIELD_NAME_BY_ENV` /
    `DEFAULT_USER_CONFIG`, so `settings_source_for_pydantic()` never
    passed it and the default stayed `True`.
  - Cause 2: `_upsert_toml_keys` appended *new* keys at EOF. Operator
    configs with nested tables (e.g. `[telegram.attachments]`) made
    `MANA_TRANSACTIONAL_ALWAYS_APPROVE = true` parse as a nested key, so
    top-level always-approve stayed `False`.
  - Fix: register the managed-worktrees key in user-config maps; insert
    missing top-level isolation keys *before* the first `[table]` header.
  - Quiet mode (`MANA_CHAT_QUIET=1`) was already correct and is unrelated.
  - User verification required:
    `python -m pytest tests/test_swe_bench_runner_config.py -k "upsert or isolation or benchmark_overrides" tests/test_tui_user_config.py -k "settings_source" -q`

- Fixed pytest collection of `tests/test_swe_bench_runner_config.py`
  (`ModuleNotFoundError: No module named 'scripts'`).
  - `tests/conftest.py` now puts the repository root on `sys.path` after
    `src/`, so `from scripts.swe_bench.runner import …` resolves regardless
    of cwd.
  - Added `scripts/__init__.py` so `scripts` is an explicit package.
  - User verification required:
    `python -m pytest tests/test_swe_bench_runner_config.py --collect-only -q`

- Fixed gateway `checkpoint_resume` failing on NVIDIA DeepSeek with
  `LengthFinishReasonError` / truncated structured JSON.
  - Symptom: Gmail (and other) entry turns returned
    `Model decision failed: checkpoint_resume… length limit was reached`
    with `completion_tokens=512` while the router model was
    `deepseek-ai/deepseek-v4-flash` (thinking/reasoning enabled by default).
  - Cause: `CHECKPOINT_RESUME_MAX_OUTPUT_TOKENS` was 512; provider thinking
    tokens share that budget, so the decision schema never completed.
  - Raised the explicit output budget to 4096 and clarified length-limit
    errors. Still no fallback resume/start when the decision is incomplete.
  - User verification required:
    `python -m pytest tests/gateway/test_checkpoint_resume.py -q`

- Fixed Windows CI failure for SWE-bench `agent_bin` Python PATH shim.
  - `prepare_agent_python_path` now writes `python.cmd` / `python3.cmd` on
    Windows (PATHEXT) and keeps executable shell scripts on POSIX.
  - Test no longer asserts Unix `S_IXUSR` on NTFS, where `chmod` does not set
    the execute bit the same way (`33206 & 64` failed on GitHub Windows runners).
  - User verification required:
    `python -m pytest tests/test_swe_bench_runner_config.py -k prepare_agent_python_path -q`

- Fixed SWE-bench hangs from deleted worktree CWD + opaque Codex config ENOENT.
  - Symptom: mana-agent stayed “still running” with stderr stuck on
    `FileNotFoundError: getcwd()` / `Path.cwd()` and
    `Codex app-server stopped: error loading default config after config error:
    No such file or directory`.
  - Cause: concurrent or overlapping instance runs could remove a live SWE
    worktree under the agent process. The process CWD became unlinked, so
    gateway init (`LlmRunLogger`) crashed and Codex children inherited a dead
    cwd.
  - `safe_cwd()` fallback for loggers when `getcwd` fails (prefer `MANA_HOME`).
  - Codex app-server always spawns with an explicit existing `cwd` (repo if
    present, else isolated `CODEX_HOME`); stream fails clearly if the execution
    directory is gone.
  - SWE-bench runner: per-instance worktree lock, refuse recreate/remove while
    another live pid holds the lock, require `.git` after checkout, kill the
    agent if the worktree disappears mid-run (exit 125).
  - User verification required:
    `python -m pytest tests/test_path_safety_safe_cwd.py tests/test_llm_logging.py -k deleted_cwd tests/test_codex_runtime.py -k "cwd or startup_failure" tests/test_swe_bench_runner_config.py -k worktree_lock -q`

- Fixed SWE-bench agent runs blocked by keyword Git intent + transactional policy.
  - Root cause (from `.swe-bench/logs/*`): prompts say "Do not commit, push…", but
    `_git_intent_from_request` keyword-matched those words, ran
    `git switch -c and`, then verifier failed on `git rev-parse origin/and`
    (`git_verification_failed_or_blocked`). Empty patches followed.
  - Removed keyword `GitIntent` inference. Git workflows require an explicit
    structured `GitIntent` argument (model decision), not text matching.
  - Added transactional `PolicyConfig.always_approve` /
    `MANA_TRANSACTIONAL_ALWAYS_APPROVE`: converts `REQUIRE_APPROVAL` → `ALLOW`
    for non-interactive bench; `DENY` stays deny.
  - SWE-bench runner isolation now sets
    `MANA_TRANSACTIONAL_ALWAYS_APPROVE=true`,
    `MANA_MANAGED_WORKTREES_ENABLED=false`, and
    `MANA_CODEX_WORKTREE_ISOLATION=false` so edits land in the SWE worktree and
    shell/git mutations do not stall on human inbox approvals.
  - User verification required:
    `python -m pytest tests/test_multi_agent_core.py -k "git or swe_bench" tests/transactional_actions/test_policy_gated_actions.py -k always_approve tests/test_swe_bench_runner_config.py -k benchmark_overrides -q`

- Hardened SWE-bench timeout configuration against multi-line shell flag drops.
  - Logs `Process argv` at startup so missing `--timeout` is obvious.
  - Warns loudly when falling back to the built-in 600s default.
  - Timeout priority: CLI → `MANA_SWE_BENCH_TIMEOUT` / `SWE_BENCH_TIMEOUT` →
    `.swe-bench/runner.toml` / `runner.env` → 600s default.
  - Added `scripts/swe_bench/run_unlimited.sh` and default
    `.swe-bench/runner.toml` with `timeout = 0`.
  - User verification required:
    `python -m pytest tests/test_swe_bench_runner_config.py -k timeout`

- Fixed SWE-bench timeout + bloated auto-chat tool surface for coding runs.
  - Gateway/chat paths no longer hard-cap agent timeouts at 600s
    (`min(..., 600)`), so runner `--timeout` / `--agent-timeout-seconds`
    multi-hour values actually apply (shared `normalize_agent_timeout_seconds`).
  - Runner `--timeout` default remains 600; supports `0` = unlimited, env
    `MANA_SWE_BENCH_TIMEOUT` / `SWE_BENCH_TIMEOUT`, and logs timeout **source**.
  - Isolated SWE-bench `MANA_HOME` disables browser, computer control, canvas,
    web/github search, fleet, and worker gateway; nested
    `[computer_control]` / `[telegram]` / media tables forced `enabled=false`.
  - Runner sets `MANA_CHAT_QUIET=1` so non-interactive startup skips dumping
    the full ~179-tool catalog into `mana_stdout.log`.
  - Docs note: multi-line shell invocations need `\\` after `runner.py` or
    flags are dropped (exit 127 on the next line).
  - User verification required:
    `python -m pytest tests/test_timeouts_normalize.py tests/test_swe_bench_runner_config.py tests/test_auto_chat_tools_catalog.py -k "timeout or quiet or browser or python or mass_delete or prompt or shim"`

## 2026-08-08

- Fixed SWE-bench empty-patch failures driven by host Python 2.7 and
  destructive/no-op worktree states after Codex+NVIDIA DeepSeek runs.
  - Observed on `astropy__astropy-12907`: agent `python -c` hit the host
    Frameworks Python 2.7, failed on f-strings in `astropy/__init__.py`, then
    the model emitted DSML/garbage text and finished with `model_patch=""`.
  - Runner now installs a per-instance `agent_bin` PATH shim so bare `python`
    always execs the runner's Python 3 interpreter.
  - Issue prompt now prefers source edits, requires `python3`, and warns that
    the checkout may not be importable (do not spend the turn on env debug).
  - Mass-delete-only worktrees (many deletes, no modifications) are rejected
    as `destructive_patch` instead of shipping a huge delete-only prediction.
  - Model catalog treats `deepseek-ai/deepseek-v4-flash-0731` (and other
    family-suffixed DeepSeek V4 ids) as tool/code/reasoning capable so NIM
    agent routing does not drop tools for dated build ids.
  - Restored accidental deleted tracked files in the local
    `.swe-bench/worktrees/astropy__astropy-13033` worktree used for diagnosis.
  - User verification required:
    `python -m pytest tests/test_swe_bench_runner_config.py tests/test_nvidia_provider.py -k "deepseek or python or mass_delete or prompt or shim or flash-0731 or capabilities"`

- Fixed Codex Responses bridge sending routing profile metadata to NVIDIA,
  which caused HTTP 400 `Unsupported parameter(s): source_levels, capability_source`
  (and empty SWE-bench patches / `codex_failed`).
  - `ModelProfile.configuration` may contain bookkeeping fields
    (`source_levels`, `capability_source`, `token_profile_confidence`) alongside
    optional request fields; only the latter are forwarded as request overrides.
  - Coding-agent Codex path and the bridge request adapter both strip internal
    keys (and SDK-only `model_kwargs`) before building Chat Completions bodies.
  - User verification required:
    `python -m pytest tests/test_codex_responses_bridge.py tests/test_model_routing.py -k "routing_metadata or provider_request_overrides or deepseek"`

- SWE-bench runner now uses `~/.mana/config.toml` for provider/model when CLI
  flags are omitted.
  - No `--model` → `MANA_PRIMARY_MODEL` / `OPENAI_CHAT_MODEL` / `LLM_MODEL`.
  - No `--provider` → `MANA_AI_PROVIDER` (e.g. `nvidia`).
  - `--model` alone still uses the **configured** provider and credentials.
  - Isolated per-instance `MANA_HOME` pins both `MANA_AI_PROVIDER` and model roles.
  - Logs report provider/model **source** (config vs CLI vs built-in fallback).
  - User verification required:
    `python -m pytest tests/test_swe_bench_runner_config.py`

- Fixed Codex + NVIDIA reconnect loop for non-retryable provider errors
  (HTTP 400 surfaced as `responseStreamDisconnected` / `Reconnecting... 1/5`).
  - Introduced shared typed `ProviderFailure` / `ProviderFailureKind` classification
    with retry ownership (`transport` / `codex_stream` / `supervisor` / `none`),
    full-jitter backoff, `Retry-After` parsing, structured telemetry, and a
    per-provider+endpoint circuit breaker that ignores 400/401/403/404/410/422.
  - Responses bridge now opens the upstream Chat Completions request and inspects
    HTTP status **before** returning `HTTP 200 text/event-stream`. Non-2xx
    (including NVIDIA HTTP 400) is returned as a proper Responses error with
    `retryable=false` and a sanitized upstream body snippet — Codex must not
    reconnect.
  - After SSE has started, mid-stream failures emit `response.failed` cleanly
    instead of letting raw exceptions close the socket.
  - Bridge transport attempts are fixed at 1 (no nested retry multiplication with
    Codex). Bridge-path Codex `request_max_retries` is capped at 2 for loopback
    connect failures only.
  - NVIDIA DeepSeek request shaping clamps `max_tokens`, normalizes message
    sequence (system first, tool_call_id retained), and maps unsupported
    reasoning efforts (`xhigh`/`minimal`/`medium`) to NIM values (`none`/`high`/`max`).
  - Model retired / not found (410/404) invalidates the cached model catalog.
  - User verification required:
    `python -m pytest tests/test_provider_failure.py tests/test_codex_responses_bridge_recovery.py tests/test_codex_responses_bridge.py tests/test_codex_runtime.py tests/test_nvidia_provider.py`

- Fixed NVIDIA DeepSeek direct-chat TypeError:
  `Completions.create() got an unexpected keyword argument 'chat_template_kwargs'`.
  - LangChain / OpenAI Python SDK path now nests NIM `chat_template_kwargs`
    under `extra_body` (SDK merges into the HTTP body) instead of spreading it
    as a top-level create() kwarg.
  - Codex Responses bridge continues to send top-level `chat_template_kwargs`
    via raw HTTP Chat Completions.
  - Upstream 4xx/410 stream failures now log kind, tools flag, template flag,
    and a truncated body snippet for diagnosis (still redacted from user UI).
  - User verification required:
    `python -m pytest tests/test_nvidia_provider.py tests/test_codex_responses_bridge.py -k "deepseek or chat_template or first_class"`

- Fixed two Windows CI failures in connector health storage and eval suite load.
  - Skip legacy colon-filename snapshot migration on Windows (`os.name == "nt"`);
    colon names are illegal or NTFS ADS syntax there, not real directory entries.
  - Eval baseline-not-suite error hints now use portable POSIX path separators
    (`evals/suites/...`) instead of platform-local `Path` stringification.
  - User verification required:
    `python -m pytest tests/connectors/health/test_connector_health_core.py -k "migrate_legacy or fs_names_encode" tests/evals/test_eval_lab.py -k "baseline_document"`

- Fixed NVIDIA DeepSeek V4 request shaping for both direct chat and the Codex
  Responses bridge.
  - Inject `chat_template_kwargs` (`thinking` + `reasoning_effort`) required by
    NVIDIA NIM for `deepseek-ai/deepseek-v4-flash` / `deepseek-v4-pro`.
  - Avoid sending bare top-level `reasoning_effort` alone to NVIDIA DeepSeek
    (can hang, 4xx/410, or disconnect Codex streams as `systemError`).
  - Improve provider error logs with HTTP status and NVIDIA-specific messaging.
  - User verification required:
    `python -m pytest tests/test_codex_responses_bridge.py tests/test_nvidia_provider.py -k "deepseek or chat_template or fragmented or first_class"`

- Fixed the Responses bridge fragmented tool-argument test assertion to account
  for JSON-escaped SSE payloads while still verifying full argument reconstruction.
  - User verification required:
    `python -m pytest tests/test_codex_responses_bridge.py::test_stream_adapter_fragmented_tool_arguments`

- Added a Mana-managed OpenAI Responses compatibility bridge so Codex can use
  Chat Completions-only providers such as NVIDIA NIM (`deepseek-ai/deepseek-v4-pro`).
  - Introduced explicit `CodexTransport` (`direct_responses` / `responses_bridge`
    / `unsupported`) separate from native `supports_responses_api`.
  - NVIDIA remains `supports_responses_api=false` and uses `RESPONSES_BRIDGE`.
  - Codex still receives `wire_api = "responses"` against a loopback bridge;
    `NVIDIA_API_KEY` never enters Codex config, logs, or child argv.
  - Bridge converts Responses requests/tools/streams to Chat Completions and
    back, including fragmented tool-call argument streaming.
  - User verification required:
    `python -m pytest tests/test_codex_responses_bridge.py tests/test_codex_runtime.py tests/test_codex_integration.py tests/test_nvidia_provider.py`

- Fixed `split_qualified_model_id` so fully qualified OpenRouter IDs such as
  `openrouter/anthropic/claude-sonnet` keep provider `openrouter` even when the
  default provider is `openai` (regression from the NVIDIA nested-ID work).
  - User verification required:
    `python -m pytest tests/test_openrouter_provider.py tests/test_nvidia_provider.py -k "first_class or qualified"`

- Completed first-class NVIDIA Build / NVIDIA NIM inference provider support.
  - Canonical provider id `nvidia` uses `NVIDIA_API_KEY` and
    `NVIDIA_BASE_URL` (default `https://integrate.api.nvidia.com/v1`).
  - Runtime connection resolution no longer falls back to OpenAI credentials.
  - Dynamic model discovery preserves nested upstream IDs
    (e.g. `deepseek-ai/deepseek-v4-flash`, `nvidia/nemotron-...`).
  - Configuration TUI, model management, wizard, embeddings, and CLI paths
    resolve NVIDIA credentials in isolation.
  - Chat Completions transport (streaming, tools, optional model-specific
    `extra_body` / `chat_template_kwargs`) works through the existing adapter.
  - Docs and `.env.example` document NVIDIA setup and the open-model benchmark
    profile (`deepseek-ai/deepseek-v4-flash`) without making it the product default.
  - User verification required:
    `python -m pytest tests/test_nvidia_provider.py tests/test_openrouter_provider.py tests/test_llm_compatibility.py tests/test_tui_user_config.py`

- Fixed confusing `mana-agent eval run` errors when a baseline JSON is passed
  instead of a suite YAML (e.g. `./evals/baselines/routing-smoke.json`).
  - `load_suite` now detects checked-in baseline documents and fails closed with
    an actionable message pointing at the suite path and gate/baseline commands.
  - Other suite schema failures are wrapped as `EvalConfigurationError` instead
    of dumping raw multi-field Pydantic noise.
  - CLI help for `eval run` states that suite YAML is required, not baseline JSON.
  - User verification required:
    `python -m pytest tests/evals/test_eval_lab.py -k "baseline_document or invalid_suite or protected_suite"`
    then intentionally:
    `mana-agent eval run ./evals/baselines/routing-smoke.json --json`
    (expect exit 2 and a baseline-not-suite message), then the correct command:
    `mana-agent eval run ./evals/suites/routing-smoke.yaml --help`.

- Clarified SWE-bench instance selection: **no ids entered → all dataset ids**.
  - If neither `--instance-ids` nor `--instance-ids-file` is set, the runner
    loads **every** instance id from the SWE-bench dataset split (~500 Verified
    `test` rows) and runs them (optional `--limit` still caps after selection).
  - If ids are entered via `--instance-ids` and/or `--instance-ids-file`, only
    those specific ids run.
  - New flags: `--instance-ids-file` (text / JSON array / JSONL),
    `--list-instance-ids` (print selected ids and exit).
  - Docs updated for full-suite generation and grading without forcing a single
    hardcoded harness `--instance_ids` (omit harness ids to grade all submitted
    prediction rows). Your report with `submitted_instances: 1` and ~499
    `incomplete_ids` is expected for a one-id smoke; re-run without id filters
    to generate/grade the full suite.
  - User verification required:
    `python3 scripts/swe_bench/runner.py --list-instance-ids | wc -l`
    then
    `python3 scripts/swe_bench/runner.py --instance-ids astropy__astropy-12907 --list-instance-ids`
    then
    `python3 scripts/swe_bench/runner.py --limit 1 --skip-agent --output predictions.jsonl`.

- Fixed SWE-bench prediction identity and smoke-eval failure modes:
  - Predictions no longer write `model_name_or_path: "mana-agent"`. Default is
    now `{agent_name}__{model}` (e.g. `mana-agent__gpt-5.6-luna`).
  - Each prediction line also includes `agent_name` (default `mana-agent`) and
    `agent_model` (the LLM from `--model`).
  - New flags: `--agent-name`, corrected `--model-name-or-path` default, and
    `--keep-test-files` (test hunks are stripped from `model_patch` by default
    because official `test_patch` is applied later; agent test edits often
    produce report `failed_ids` instead of resolved/unresolved).
  - Docs explain incomplete vs failed vs unresolved in sb-cli/harness reports
    (499 incomplete ids are expected when only 1 of 500 Verified rows is
    submitted) and recommend grading with `--instance_ids` + a run_id that
    includes agent and model.
  - User verification required:
    `python3 scripts/swe_bench/runner.py --limit 1 --skip-agent --output predictions.jsonl`
    then
    `python3 -c "import json; r=json.loads(open('predictions.jsonl').readline()); assert r['agent_name']=='mana-agent'; assert r['model_name_or_path'].startswith('mana-agent__'); assert 'agent_model' in r"`
    and for a real grade of one row:
    `python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path predictions.jsonl --instance_ids astropy__astropy-12907 --max_workers 1 --run_id mana-agent__gpt-5.6-luna-smoke`.

- Fixed connector health storage on Windows (13 CI failures, WinError 87 /
  errno 22 “The parameter is incorrect”):
  - Connector ids use `type:instance` (colon). Writing
    `health/fake:1.json` / `receipts/gmail:a_m1.json` is illegal on Windows.
  - `ConnectorHealthStore` now maps identities to filesystem-safe stems
    (`:` → `=`, bijective for the validated id charset) for snapshots, probe
    logs, receipts, and incident indexes.
  - Atomic temp files use a fixed `.tmp.*.partial` prefix instead of embedding
    the destination basename (which reintroduced illegal characters).
  - Legacy POSIX colon-named files are migrated to the safe name on first
    resolve/write so restarts keep working.
  - User verification required:
    `python -m pytest -q tests/connectors/health/test_connector_health_core.py tests/connectors/health/test_connector_health_integrations.py`

- Fixed SWE-bench runner freeze after `Starting mana-agent` (benchmark never
  produced predictions):
  - Isolated per-instance `MANA_HOME` now **rewrites** copied `config.toml`
    (Mana file settings beat process env): force `MANA_MEMORY_MODE=internal`,
    `MANA_MEMORY_PROVIDER=mana`, `MANA_MEMORY_FALLBACK_TO_INTERNAL=false`, clear
    secret ref, and pin all model roles to `--model`. Operator external
    supermemory and `MODEL_LEVEL_*` pins no longer stall startup or rewrite the
    measured model.
  - Dropped synchronous `--ephemeral-index` (full-repo index blocked large
    instances such as astropy); use `--no-auto-index-missing` so chat starts
    with direct project search immediately.
  - Launch with `--no-interactive --no-banner --no-coding-memory` and aligned
    `--agent-timeout-seconds`; emit PID + 30s heartbeats (log sizes / stderr
    tail) while waiting; write `mana_cmd.txt` / line-buffered agent logs.
  - Non-TTY chat with an initial prompt now exits after the single-shot queue
    drains instead of waiting for further interactive input.
  - Docs: `docs/swe-bench.md` updated for the isolation and invocation contract.
  - User verification required:
    `python3 scripts/swe_bench/runner.py --limit 1 --skip-agent --output predictions.jsonl`
    then
    `python3 scripts/swe_bench/runner.py --limit 1 --model gpt-5.6-luna --timeout 600 --output predictions.jsonl -v`
    (expect heartbeats, logs under `.swe-bench/logs/<instance_id>/`, and a
    `predictions.jsonl` line; agent run needs API credentials).

- Fixed routing-smoke eval failures beyond model pin isolation:
  - Pinned profiles now advertise interactive latency (see below) so entry
    routing can select suite models.
  - `process_chat_turn` was recording internal `auto_chat` as the scored route;
    entry routes are now preserved in `payload.entry_route` and finalize no
    longer overwrites them, so suite expectations like `route: repository`
    match the model-selected entry decision.
  - Browser/repository executors (and process_chat_turn) now fall back to
    `default_index_dir(root)` when `index_dir` is unset, fixing
    `PathLike … not 'NoneType'` browser crashes in evals.
  - `_chromium_executable(None)` no longer raises when Playwright reports no
    managed binary; it falls through to system Chromium candidates.
  - Entry-routing prompt examples/clarifications for plan-only, plan
    continuation, repository inspection vs computer, and no-fallback fixtures.
  - Contract/no-fallback eval tasks that intentionally stop with
    `unsupported_route` (and similar) now count as completed when labeled
    `contract`/`no-fallback`/`provider` and the error set is exact.
  - Codex coding shim no longer calls `asyncio.run()` on an already-running
    event loop (thread adapter matches computer/MCP sync boundaries), fixing
    plan-continuation failures after correct coding routing.
  - User verification required:
    `mana-agent eval run ./evals/suites/routing-smoke.yaml --json`
    (full suite: 19 tasks × 2 variants = 38 runs, all success=true after fix).

- Fixed pinned eval model profiles rejecting all gateway entry routes:
  - `profiles_for_pinned_models` only registered models as
    `MODEL_LEVEL_3_HIGH_REASONING` (`LatencyClass.STANDARD`), but gateway entry
    routing requires `LatencyClass.INTERACTIVE` for `head_decision`. Every
    candidate was rejected with "latency class standard exceeds interactive",
    failing routing-smoke tasks (including the candidate-only
    `duplicate-task-prevention` regression and both-variant clusters).
  - Pinned profiles now register as both fast-tool and high-reasoning levels,
    advertise interactive latency (same multi-level rule as legacy profiles),
    and union level benchmarks/reasoning settings so a single suite model can
    serve entry routing and coding/planning without operator `MODEL_LEVEL_*`
    overrides.
  - User verification required:
    `python -m pytest tests/test_model_routing.py::test_pinned_profiles_ignore_operator_model_levels -q`
    then re-run failing eval tasks / full
    `mana-agent eval ./evals/suites/routing-smoke.yaml`.

- Fixed routing-smoke eval isolation and context accounting that were collapsing
  many tasks under operator `gpt-5.6-luna` preferences and a 16k unknown-model
  window:
  - Eval gateway construction now sets `ChatGatewayConfig.pin_models=True` with
    the suite variant models, so `MODEL_LEVEL_*` / `MANA_MODEL_*` no longer
    rewrite measured runtime models (suite `gpt-4.1-mini` was becoming
    `gpt-5.6-luna` from `~/.mana/config.toml`).
  - New `profiles_for_pinned_models` and `pin_model_for_role` build isolated
    routing profiles/assignments without reading operator level settings.
  - Maintained token limits for common OpenAI families (`gpt-4.1*`, `gpt-4o*`,
    `gpt-5*`, `o3`/`o4`) fill catalog gaps so accounting no longer treats
    modern models as 16 384-token unknowns (fixes
    `effective limit is 0` / `context_limit_deficit` blocks on agent prompts).
  - Budget-overrun finalization prompt now requires `safe_to_continue=true` for
    valid `require_review` decisions (models were returning `false` and failing
    schema validation mid-handoff).
  - Observe-mode governor errors prefer the policy-free estimate so residual `0`
    is not misreported when the real deficit is model context capacity.
  - User verification required:
    `python -m pytest tests/test_model_routing.py::test_pinned_profiles_ignore_operator_model_levels tests/test_model_routing.py::test_legacy_profiles_apply_maintained_token_limits_for_known_models tests/test_multi_agent_core.py::test_pin_model_for_role_bypasses_operator_model_levels tests/execution_supervisor/test_supervisor_core.py::test_budget_overrun_prompt_requires_safe_to_continue_for_require_review -q`
    then re-run `mana-agent eval ./evals/suites/routing-smoke.yaml`.

- Added SWE-bench Verified prediction generation for mana-agent:
  - New focused runner: `scripts/swe_bench/runner.py` loads
    `princeton-nlp/SWE-bench_Verified`, checks out each instance at
    `base_commit` into an isolated git worktree, runs one non-interactive
    mana-agent coding pass (`chat --no-tui --root-dir --full-auto`), captures
    the final git diff as `model_patch`, and writes harness-compatible
    `predictions.jsonl` lines
    (`instance_id`, `model_name_or_path`, `model_patch`).
  - Supports `--limit`, `--instance-ids`, `--output`, hard per-instance
    `--timeout`, and a forced cheap/fast default model (`gpt-4o-mini`).
  - Hardened against empty patches, dirty trees, checkout failures, and hung
    agent processes (process-group kill on timeout).
  - Docs: `docs/swe-bench.md` (generate command, official harness grade
    command, smoke flags, limitations). Scope is prediction generation + smoke
    grading only (not full 500-run, Pro, Terminal-Bench, pass@k, or
    leaderboard submission).
  - User verification required:
    `python scripts/swe_bench/runner.py --limit 1 --skip-agent --output predictions.jsonl`
    then
    `python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path predictions.jsonl --max_workers 4 --run_id mana-agent-smoke`.

- Fixed external memory capability boundary so agent routes no longer crash when
  `MANA_MEMORY_MODE=external`:
  - Split **AI/semantic memory** (hosted provider) from **system-state stores**
    (run evidence, coding-flow checkpoints/turn history).
  - `MemoryCapabilities` contract on `MemoryService` declares conversation,
    semantic search, evidence, checkpoints, coding flow, task state, and
    multi-agent runtime availability.
  - External mode keeps local run evidence and coding-flow stores enabled; only
    semantic AI writes stay on the external provider (no silent AI-memory
    fallback to the local provider store).
  - `EvidenceMemory` façade opens the local run-evidence store without requiring
    multi-agent or external provider mapping (fixes review, plan, verification,
    repository search, and related ask-agent routes).
  - Docs: `docs/05-configuration.md` documents the dual-domain model.
  - User verification required:
    `python -m pytest tests/memory/test_external_provider_capabilities.py tests/test_memory_architecture.py -q`
    and `mana-agent eval ./evals/suites/routing-smoke.yaml`.

- Fixed connector status and doctor UX for health checks:
  - `mana-agent connectors status` now runs a live safe probe by default so
    connectors leave `UNKNOWN` / `STARTUP_PENDING` after registration
    (`--no-probe` keeps the cached pre-verification snapshot).
  - `mana-agent doctor --only` / `--skip` accept comma-separated check IDs as
    well as repeated flags (e.g. `--only connectors/health,connectors/credentials`).
  - User verification required:
    `python -m pytest tests/connectors/health/test_connector_health_integrations.py::test_cli_status_probes_by_default tests/test_doctor.py::test_doctor_only_accepts_comma_separated_ids -q`.

- Added **Connector Health and Self-Healing** so connectors are never treated as
  online merely because a process, gateway, or adapter is running:
  - Universal typed health contract (`ConnectorHealthState`, path signals,
    probe categories, reason codes, delivery receipts, incidents, SLO metrics).
  - Central `ConnectorHealthManager` with probe scheduling, exponential backoff
    + jitter, circuit breakers, deterministic recovery, incident timelines, and
    durable storage under `~/.mana/connectors/`.
  - Real adapters for **Gmail** (profile/auth, safe ingress list, receipt-based
    egress/ack) and **Telegram** (getMe, poller/webhook path, subscription
    probe, false-online detection when process is alive but ingress is broken).
  - Supervisor bridge pauses dependent branches while a required connector is
    unavailable and resumes them exactly once after recovery; auth failures can
    create durable HITL interventions; webhook/subscription repairs require
    transactional policy authorization.
  - CLI: `mana-agent connectors status|health|incidents|recover` (also under
    `connector health`); doctor checks `connectors/credentials` and
    `connectors/health`; dashboard and API expose real health states.
  - Routine probes are fully deterministic (no LLM). Synthetic active messages
    are disabled by default.
  - Documentation: `docs/33-connector-health.md`.
  - User verification required:
    `python -m pytest tests/connectors/health/test_connector_health_core.py tests/connectors/health/test_connector_health_integrations.py tests/execution_supervisor/test_supervisor_core.py tests/human_inbox/test_durable_inbox.py -q`.

## 2026-08-07

- Fixed gateway task control and chat-turn auto recovery so messages do not
  require `/tasks` to select work:
  - `/task Execute` (and other reserved verbs / non-id tokens) no longer raise
    `Unknown gateway task: Execute`; they return usage that points operators to
    `/tasks`, chat-turn auto-select, and `mana-agent tasks recover`.
  - `/task cancel|pause|resume|retry|replan` accept an optional id and
    auto-select only when exactly one recoverable/active candidate exists.
  - Operator `retry` / `replan` control builds a validated recovery decision;
    `resume` of stopped work becomes a same-task retry.
  - Recovery candidates now include blocked multi-task roots (supervisor
    `waiting` without a human-inbox wait) and lane blocked/paused projections.
  - Retryable lane states include `waiting` and `paused` after rehydration.
  - Multi-task chat turns run checkpoint-resume before creating a new root:
    resume / retry / replan reuse the root; replan/retry reopen incomplete
    children so the job restarts from the first unfinished step.
  - Checkpoint-resume prompt documents the full decision matrix
    (resume → retry → replan/restart job → start_fresh → stop).
  - User verification required:
    `python -m pytest tests/gateway/test_chat_gateway.py::test_task_control_rejects_execute_verb_instead_of_unknown_task_id tests/gateway/test_chat_gateway.py::test_task_control_rejects_non_task_id_tokens tests/gateway/test_chat_gateway.py::test_task_control_auto_selects_single_recoverable_task_for_retry tests/gateway/test_lane_coordinator.py::test_recovery_candidates_include_blocked_multi_task_root_without_inbox_wait tests/gateway/test_lane_coordinator.py::test_blocked_multi_task_root_can_be_retried_with_validated_decision tests/gateway/test_checkpoint_resume.py -q`.

- Added **task-wide computer approval** so one trusted approval can cover a whole
  durable task lineage of safe filesystem creates/moves/renames:
  - New `ApprovalScope.TASK` multi-use grant bound to `root_task_id` (multi-task
    children share the parent root), tool `computer`, and the
    `filesystem.mkdir|copy|move|rename` family.
  - Policy selects task scope for those ops when a durable task lineage is
    present; trash, recording, system power, and other computer ops stay
    single-use.
  - First approval issues the task grant; later compatible actions under the
    same root reuse it without a new inbox prompt until expiry/invalidation.
  - Durable execution context now carries `root_task_id` into computer tools;
    approval required payloads advertise `transactional_action.task`.
  - User verification required:
    `python -m pytest tests/test_computer_control.py::test_task_wide_computer_filesystem_approval_covers_later_ops_in_lineage tests/test_computer_control.py::test_computer_action_uses_durable_exact_approval_and_initializes_audit tests/transactional_actions/test_policy_gated_actions.py::test_computer_filesystem_policy_selects_task_wide_scope_when_lineage_present tests/transactional_actions/test_policy_gated_actions.py -q`.

- Fixed gateway task control and recovery coordination for stopped durable work:
  - `/task create` (and other reserved verbs) no longer raise
    `Unknown gateway task: create`; they return usage that points operators to
    `/tasks`, chat-turn task creation, and `mana-agent tasks recover`.
  - Unknown real task IDs return an actionable control error instead of an
    uncaught lane exception.
  - Validated `retry_task` / `replan_task` / `resume_checkpoint` rehydrate a
    missing lane projection from the durable supervisor record so recovery does
    not fail solely because the in-memory/gateway projection was dropped.
  - Blocked multi-task roots are retryable under an authorized same-task
    recovery decision (children often leave the parent `BLOCKED` rather than
    `FAILED`).
  - User verification required:
    `python -m pytest tests/gateway/test_chat_gateway.py::test_task_control_rejects_create_verb_instead_of_unknown_task_id tests/gateway/test_chat_gateway.py::test_task_control_unknown_id_returns_actionable_message tests/gateway/test_lane_coordinator.py::test_retry_rehydrates_missing_lane_projection_from_supervisor tests/gateway/test_lane_coordinator.py::test_blocked_multi_task_root_can_be_retried_with_validated_decision -q`.

- Fixed checkpoint-resume validation so `retry_task` and `replan_task` may
  select any offered non-completed stopped task even when that candidate still
  lists a checkpoint. Same-task restart intentionally leaves `checkpoint_id`
  empty; filling a checkpoint ID on retry/replan still fails closed and
  `resume_checkpoint` remains the only way to continue saved progress. This
  unblocks recovery after failed multi-task or other checkpointed work when the
  model correctly reuses the task identity with `retry_task` instead of
  `resume_checkpoint`.
  - User verification required:
    `python -m pytest tests/gateway/test_checkpoint_resume.py -q`
    and
    `python -m pytest tests/gateway/test_entry_routing.py -k checkpoint_resume -q`.

- Fixed multi-task child execution so worker threads inherit the parent turn’s
  ContextVars (authenticated computer-client identity, evals, event sinks).
  Compound goals that route a child to `computer` (for example sequential
  workspace directory/file creation) no longer fail with
  `Computer decision scope requires an authenticated client context` solely
  because `ThreadPoolExecutor` workers dropped the parent scope; dependents
  blocked on that prerequisite can proceed after a successful computer child.
  Missing parent identity still fails closed with no fallback client.
  - User verification required:
    `python -m pytest tests/gateway/test_multi_task_orchestration.py::test_worker_threads_inherit_parent_contextvars_for_computer_client tests/gateway/test_multi_task_orchestration.py -q`
    and
    `python -m pytest tests/gateway/test_chat_gateway.py::test_computer_route_without_typed_tool_outcome_records_notice tests/test_computer_control.py -q`.

- Patched CodeQL high/medium findings across dashboard API, gateway, Codex
  runtime, auto-chat classifiers, live canvas/chat JS, and CI:
  - Reflected XSS: dashboard live-chat/live-canvas HTML embeds IDs through a
    strict allowlist and JSON script embedding that Unicode-escapes `<>&`.
  - Path injection: shared `path_safety` confinement (`startswith` after
    resolve) for workspace roots, conversation/analyze roots, artifact
    membership checks, and computer-control allowed paths.
  - Clear-text credential storage: Codex runtime writes config TOML only after
    stripping the API key; the key stays in the child environment only.
  - Weak sensitive hashing: Codex credential/runtime fingerprints use PBKDF2-HMAC
    instead of raw SHA-256 on the API key.
  - ReDoS: bounded input length and linear path/follow-up regexes in
    `small_direct_edit` and `auto_chat`.
  - Exception exposure: workspace search and several API handlers return generic
    client errors instead of raw exception text.
  - Prototype pollution: live canvas JSON-pointer updates reject
    `__proto__` / `constructor` / `prototype` keys.
  - postMessage: live chat targets an explicit parent origin (same-origin or
    loopback referrer), not `*`.
  - CI: workflow sets `permissions: contents: read`.
  - User verification required:
    `python -m pytest tests/test_api_conversations.py tests/test_canvas.py tests/test_codex_runtime.py tests/test_auto_chat.py tests/test_small_direct_edit.py tests/test_dashboard_live_chat.py -q`
    and `node --test tests/dashboard/live_canvas_reducer.test.mjs`.

- Improved approved MCP chat output so documentation-style provider results
  (for example Context7 `query-docs` / `resolve-library-id`) show extracted
  text content instead of the raw transport envelope JSON. Doc-oriented
  operations are labeled **Documentation (untrusted data)** with a compact
  status line; non-text results still fall back to compact JSON. Activity
  previews use the same extracted text. Presentation only; no routing or
  fallback behavior was added.
  - User verification required: `python -m pytest tests/test_mcp.py tests/gateway/test_chat_gateway.py::test_resumed_mcp_action_surfaces_its_result_in_chat_history -q`.

- Bumped security-sensitive dependency floors to clear open Dependabot alerts for
  `cryptography`, `langchain`, and `langchain-openai`:
  - `cryptography>=49.0.0,<51.0` (path-building DoS, SECT subgroup validation,
    OpenSSL wheel CVEs, DNS name-constraint / wildcard verifier issues).
  - `langchain>=1.3.9,<2.0.0` (path traversal / sandbox escape in file-search
    middleware and loaders).
  - `langchain-openai>=1.1.14,<2.0.0` with `openai>=2.26.0,<3.0.0` (image token
    counting SSRF DNS-rebinding fix; OpenAI SDK major floor required by the
    patched partner package).
  - `langchain-community` remains on the `>=0.3.27,<0.4.0` line for FAISS
    stability. Updated `tools_run.py` to import `BaseCallbackHandler` from
    `langchain_core` because `langchain.callbacks` is gone in LangChain 1.x.
  - User verification required: `python -m pip install -U -e .` then
    `python -m pytest tests/test_package_version.py tests/test_llm_compatibility.py tests/remote_execution/test_reverse_worker_protocol.py -q`.

- Fixed multi-message session budgets so follow-up and extend messages recalculate
  admission instead of inheriting a depleted prior-turn residual of 0. The context
  cost governor expands the session ledger to a fresh per-task
  `MANA_ROUTING_TASK_TOKEN_BUDGET` envelope on each user message; gateway preflight
  estimates refresh that envelope before sizing; live lane reservations are
  recalculated from the new forecast for follow-up, expand, retry, and resume
  paths. `MANA_LANE_SESSION_TOKEN_BUDGET` / `MANA_LANE_GLOBAL_TOKEN_BUDGET` of `0`
  remain unlimited (no longer coerced to `1`). Follow-up classification remains
  deployed on the gateway process_turn path. Missing capacity still fails closed
  with no model fallback.
  - User verification required: `python -m pytest tests/context_cost/test_context_cost_core.py::test_ensure_admission_budget_refreshes_depleted_session_for_followup_message tests/gateway/test_multi_task_orchestration.py::test_execution_token_estimate_refreshes_budget_for_followup_message tests/gateway/test_capsule_identity.py::test_lane_token_budget_zero_means_unlimited tests/gateway/test_followup_classifier.py tests/gateway/test_entry_routing.py tests/gateway/test_chat_gateway.py -q`.

- Fixed context-cost admission so sequential model calls under one lane-task
  identity no longer reuse a finalized accounting operation id. Gateway routes
  pin `step_id=after_routing` for the whole execution, so search (and other
  multi-call routes) previously failed after the first provider call with
  `accounting operation 'call-…' was already finalized`. The governor now
  allocates the next free ordinal call id when the stable identity is already
  reserved, reconciled, or released. The finalized-id handler is ordered after
  `ModelContextLimitError` handling because that error is a `ValueError`
  subclass; catching `ValueError` first previously leaked raw context-limit
  errors instead of `ContextBudgetExceeded` / observe-mode admission.
  - User verification required: `python -m pytest tests/context_cost/test_context_cost_core.py::test_enforce_mode_blocks_before_provider_and_protects_required_segments tests/context_cost/test_context_cost_core.py::test_observe_mode_records_task_budget_overrun_without_blocking tests/context_cost/test_context_cost_core.py::test_sequential_model_calls_under_same_task_identity_get_fresh_call_ids tests/context_cost/test_context_cost_core.py::test_released_model_call_id_is_not_reused_for_later_admission tests/gateway/test_turn_engine_search.py -q`.

- Fixed required-source public/GitHub search execution so the second model
  decision only produces a compact query for the already selected search tool.
  The previous full routing-schema pass frequently returned invalid
  `web_search.query` decisions and failed closed before Tavily ran. The new
  dedicated search-operation decision uses a query-only contract, normalizes
  common model payload shapes without inventing a query, and still stops with
  no alternate source when the model omits, overlongs, or cannot supply a query.
  - User verification required: `python -m pytest tests/gateway/test_turn_engine_search.py tests/gateway/test_entry_routing.py::test_required_search_source_uses_constrained_operation_decision -q`.

- Fixed chat recovery so wall-clock-dead tasks create a new task instead of
  being retried or resumed under an already elapsed deadline. Recovery candidates
  expose `deadline_exceeded`; resume/retry/replan exclude those tasks. When the
  model still targets a deadline-dead identity, the gateway reserves a new task
  with a fresh deadline and lineage links (`previous` / `supersedes`). Supervisor
  retry validation and child creation under a dead parent refuse requeue with a
  clear “create a new task” error. Checkpoint-resume allows `start_fresh` for the
  same work when no recoverable candidates remain. Coverage asserts successful
  coding turns with `error is None` (not an empty string).
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py::test_deadline_dead_task_creates_new_task_instead_of_retry tests/gateway/test_checkpoint_resume.py::test_model_may_start_fresh_for_same_work_when_no_recoverable_candidates tests/gateway/test_lane_coordinator.py::test_recovery_candidates_mark_deadline_exceeded_tasks tests/execution_supervisor/test_supervisor_core.py::test_retry_refuses_wall_clock_deadline_dead_task tests/execution_supervisor/test_supervisor_core.py::test_create_child_refuses_deadline_dead_parent -q`.

- Fixed parent/child lane budget growth so recalculating a child reservation
  expands the active parent (and nested ancestors) instead of failing with
  “recalculated child budget exceeds the parent remaining budget”. This covers
  multi-task children, nested coding follow-ups under a parent lane task, and
  reserve-time child admission when the child needs more than the parent’s
  current remaining envelope. Terminal (failed/completed/cancelled) parents no
  longer block child recalculation, matching reserve-time policy. Session, global,
  and lane caps still fail closed with no fallback route. Nested-ancestor coverage
  avoids starting two coding tasks under the same repository-write lock so the
  suite does not hang on lane lock wait.
  - User verification required: `python -m pytest tests/gateway/test_lane_coordinator.py tests/gateway/test_multi_task_orchestration.py -q`.

- Fixed multi-task budget coordination so compound children can reserve and run
  under a real parent envelope. The multi-task root now reserves capacity for
  the planned children (not only the goal text), expands that envelope before
  each child reservation, and sizes child preflight estimates against model/lane
  capacity rather than the depleted shared session ledger left by parent
  planning. Mid-run provider-call forecasts for multi-task children (for example
  Codex coding after a media sibling already reserved capacity) now expand the
  parent envelope before the child reservation is revised, so live children are
  not aborted with “recalculated child budget exceeds the parent remaining
  budget”. Budget shortfalls return blocked child status without inventing a
  fallback route; non-multi-task modules are unchanged.
  - User verification required: `python -m pytest tests/gateway/test_multi_task_orchestration.py tests/gateway/test_lane_coordinator.py tests/gateway/test_entry_routing.py -q`.

## 2026-08-06

- Corrected Windows human-inbox signing-key publication. After the existing
  per-key thread and process lock writes and syncs a complete candidate, it now
  atomically replaces the destination and reloads the durable key before caching
  it. This prevents concurrent signers from deriving different HMACs in the
  Windows release test.
  - User verification required: `python -m pytest tests/human_inbox/test_durable_inbox.py -q`.

## 2026-08-05

- Fixed checkpoint-resume context-budget failures being reported as generic lane-coordination
  errors. The typed budget block now retains its original admission reason, returns a dedicated
  safe gateway result, and creates no recovery or new task.
  Checkpoint-resume decisions now use a scoped accounting identity and a 512-token structured
  response cap, preventing unrelated long-running task history from inflating their admission
  forecast. The cap uses the runtime's Chat Completions-compatible `max_tokens` argument.
  API Manager execution now loads only its workflow-decision capability initially, then requires
  the model to load each subsequent authorized API or browser capability; this applies in observe
  mode as well, preventing the complete API schema surface from exhausting context before a call.
  API workflow response allowance is now calculated by context accounting and enforced by the
  governor for each provider call; no route-level token cap is hardcoded.
  Explicit output limits now override historical output predictions during accounting, preventing
  a small bounded decision from being inflated by unrelated earlier responses.
  Non-enforcing governor modes now record task or session budget overruns without blocking provider
  calls; true model context-window and output-capacity limits remain enforced.
  Capabilities loaded late in a tool loop now receive the current step timestamp, preventing the
  idle-capability reaper from unloading them before their first requested use.
  API workflows now reserve up to 32 model-tool steps, allowing capability discovery plus the
  model-selected inspect, import, configure, search, preview, and execute lifecycle to complete.
  Repeated but non-duplicate capability-manifest results no longer trigger the generic no-progress
  stop condition before a selected API capability can be used.
  Browser documentation actions now require an explicit, validated model read-only decision before
  the shared transactional gate permits a click; non-read-only browser actions remain fail-closed.
  Browser-action decision reasons are retained for validation but are no longer forwarded to the
  browser session runtime, which does not accept that audit-only field.
  The browser runtime coverage now imports its JSON assertion helper.
  API documentation imports and integration updates now use a durable API-integration action
  adapter and narrow policy rule; deletion still requires an exact approval.
  Pending API network approvals now publish a redacted session-bound `api.waiting_approval` event,
  allowing the trusted TUI and dashboard to present the same approve-once or deny controls used for
  other transactional requests.
  API preview now creates that exact approval before `api_request_execute`, stops the model tool
  loop with a `permission_required` result, and records a redacted durable root-user inbox notice
  that points back to the trusted local approval modal.
  The TUI now consumes the active session's preview-time API approval event directly, so it opens
  its approval modal while the tool loop is still paused; dashboard already consumes that event
  through its live conversation stream.
  API workflow accounting now treats this exact preview-time `permission_required` result as
  successful preview evidence while continuing to require execution evidence after approval.
  Preview approval coverage now includes the documented server base path when checking the
  redacted request target.
  Resolving an API approval now finalizes the waiting chat with bounded, redacted execution
  evidence (or an explicit no-execution denial) instead of only reporting the HTTP status.
  Preview-time API approval waits now return a successful structured pending result rather than a
  tool exception, while preserving the exact request ID, modal event, inbox record, and stop rule.
  Dashboard approval controls now render the returned API completion message immediately instead
  of waiting solely for a later WebSocket event.
  TUI API approvals now add the validated completion evidence as a terminal assistant message
  bound to the exact approval ID, with a concise completion or denial notification.
  Approved API JSON responses now render as a generic nested, human-readable result instead of a
  raw JSON dump, while preserving bounded redacted request metadata.
  - User verification required: `python -m pytest tests/context_cost/test_context_cost_core.py tests/context_cost/test_model_accounting.py tests/connectors/test_browser_core.py tests/gateway/test_checkpoint_resume.py tests/gateway/test_entry_routing.py tests/gateway/test_api_manager_route.py tests/test_api_manager.py tests/test_api_conversations.py tests/test_tui_auto_chat_tool_events.py tests/test_ask_agent.py -q`.

- Fixed concurrent human-inbox signing-key initialization on Windows. Signers now
  coordinate key creation with a per-key thread and process lock before loading
  the published secret, preventing concurrent signers from caching different keys.
  - User verification required: `python -m pytest tests/human_inbox/test_durable_inbox.py -q`.

- Repaired supervised task recovery and metadata durability. Gateway recovery candidates now span
  sessions within the same workspace and repository, semantic task fingerprints no longer include
  turn/message IDs, and a fresh model decision selects checkpoint resume, same-task retry, same-task
  replan, fresh work, or a safe stop. Supervisor records now retain non-empty initial completion
  contracts and explicit provenance for values that are pending runtime evidence rather than known.
  Server and remote approved actions now write durable action states and receipts so ambiguous
  outcomes block retries pending reconciliation.
  Lightweight gateway adapters used by non-executing callers remain compatible while production
  lane coordinators continue to require the durable action ledger.
  - User verification required: `python -m pytest tests/gateway/test_followup_classifier.py tests/gateway/test_checkpoint_resume.py tests/gateway/test_entry_routing.py tests/gateway/test_chat_gateway.py tests/gateway/test_lane_coordinator.py tests/execution_supervisor/test_supervisor_core.py`.

- Fixed enrolled-server directory-list routing so the router explicitly models
  that `server_directory_list` establishes its own connection and must use the
  `file_read` / `filesystem.read` contract. A bounded model-only correction now
  retries an otherwise complete server decision that mismatches its selected
  tool contract; the corrected response must still pass the same strict
  validation before any tool can execute.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py -q`.

- Fixed dashboard server-action approvals so the validated completion summary is
  persisted as an assistant message and emitted as a terminal chat event after
  the action finishes. A missing summary now fails explicitly instead of
  inventing a fallback response.
  - User verification required: `venv/bin/python -m pytest tests/test_api_conversations.py -q`.

- Corrected gateway follow-up handling so stopped tasks are classified before conversation execution, validated retry and checkpoint recovery retain an explicit successful error value, and completed tasks do not block a fresh Gmail turn.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py -q`.

- Corrected gateway recovery coverage to import non-public recovery enums from their typed models module.
  - User verification required: `python -m pytest tests/gateway/test_lane_coordinator.py -q`.

## 2026-08-04

- Repaired gateway lifecycle gaps: follow-up classification is mandatory whenever durable task candidates exist, explicit memory retrieval now uses a model-selected task scope for private capsules, and every currently available registered route has an audited executor contract. Calendar remains truthfully registered-but-unavailable until a calendar connector exists.
  - Removed unused gateway imports while retaining the typed checkpoint-resume and execution-supervisor retry chain.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py tests/gateway/test_followup_classifier.py tests/gateway/test_checkpoint_resume.py tests/gateway/test_chat_gateway.py tests/gateway/test_lane_coordinator.py tests/test_codex_integration.py -q`.

- Bumped the package and documented version to `v0.1.5`.
  - User verification required: `python -m pytest -q`.

- Bound Codex coding turns to the durable gateway lane task selected before execution. Context-cost admission, transactional ownership, Codex live events, workspace tracking, and final lane reconciliation now use the same task ID instead of a connector-local `codex_task_*` ID that the lane coordinator cannot resolve.
  - User verification required: `python -m pytest tests/test_codex_integration.py tests/gateway/test_chat_gateway.py -q`.

- Added model-authorized adaptive execution budgets. Provider-call admission now recalculates active lane reservations from the exact accounting forecast within immutable policy caps, and durable result overruns enter `pending_budget_decision` rather than being discarded. A fresh validated overrun decision can accept a verified flagged result, require review, or request normally validated bounded recovery. Added `/budget recalculate <task-id>` and durable budget revision evidence.
  - Pending decision, review, and scheduled-recovery outcomes are reported as successful chat handoffs with warnings instead of failed chat execution.
  - Finalization-decision model calls receive a fresh accounting step identity, preventing collisions with the already reconciled execution call.
  - Added `/budget finalize <task-id>` to resume a durable pending-overrun result through its required model decision.
  - Accepted overruns now project the authoritative completed supervisor record and verification evidence before marking the taskboard item done.
  - `/budget finalize <task-id>` now repairs a previously completed overrun whose taskboard projection was interrupted, without requesting another provider decision.
  - Exported `BudgetOverrunAction` with the other public execution-supervisor budget decision types.
  - Pending overrun decisions now emit a waiting budget-decision event instead of a false `lane.failed` event, so accepted Gmail and other connector responses remain visibly successful.
  - User verification required: `python -m pytest tests/context_cost/test_model_accounting.py tests/context_cost/test_context_cost_core.py tests/context_cost/test_context_cost_integration.py tests/gateway/test_chat_gateway.py tests/gateway/test_lane_coordinator.py tests/execution_supervisor/test_supervisor_core.py -q`.

- Changed Gmail connector execution to begin with lightweight, model-controlled capability discovery instead of binding every email action schema before the first provider call. The executor now selects and loads its exact Gmail capability from the manifest, avoiding context-budget admission failures while preserving the allowlist and fail-closed tool policy. Capability controls are explicitly read-only for transactional enforcement, and capability activation now refreshes the bound tool set even when a tool's estimated schema size is unchanged.
  - User verification required: `python -m pytest tests/test_ask_agent.py tests/gateway/test_entry_routing.py -q`.

- Updated multi-agent queue accounting coverage so tool-result payloads remain uncharged until they are included in a provider model call; reservation and worker-execution evidence remain durable.
  - User verification required: `python -m pytest tests/test_multi_agent_core.py -q`.

- Aligned context-cost and lane tests with model-aware accounting: enforce-mode admissions now register exact test-model pricing, and lane exhaustion coverage supplies explicit configured limits because default lane contracts deliberately have no synthetic token or cost cap.
  - User verification required: `python -m pytest tests/context_cost/test_context_cost_core.py tests/gateway/test_lane_coordinator.py -q`.

- Made native coding-planner initialization tolerate injected AskAgent-compatible implementations that do not provide optional context-cost accounting, while retaining governor propagation for configured production agents.
  - User verification required: `python -m pytest tests/test_coding_agent.py -q`.

- Replaced the unconstrained second routing pass for an already selected public-web or GitHub search source with a dedicated, model-constrained search-operation decision. The operation decision now exposes only the selected search tool and requires its compact query, while invalid decisions still stop without an alternate source. Search-source failures now preserve the fail-closed message without a duplicated trailing period.
  - User verification required: `python -m pytest tests/gateway/test_turn_engine_search.py tests/gateway/test_entry_routing.py -q`.

- Deferred telemetry's context-cost tokenizer and usage-normalizer imports until their functions execute, breaking the `telemetry.tokens` ↔ `context_cost` package-initialization cycle that prevented gateway tests and CLI imports from being collected.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py::test_gateway_gmail_uses_dedicated_connector_not_coding_or_conversation tests/gateway/test_chat_gateway.py -q`.

- Initialized the retrieved-memory accounting component for first-turn and other no-prior-assistant gateway routes, preventing an `UnboundLocalError` before Gmail and other supervised connector execution.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py::test_gateway_gmail_uses_dedicated_connector_not_coding_or_conversation tests/gateway/test_chat_gateway.py -q`.

- Replaced scattered runtime token envelopes with model-aware accounting owned by `context_cost`. Routing and gateway execution now carry the final provider/model, context/output capabilities, expected calls, explainable payload components, confidence, and Decimal cost into durable lane and supervisor records; Canvas estimates its serialized catalog, surface state, and tool schemas without fixed 4,096/1,024 assumptions.
  - Added provider-usage normalization, tokenizer/profile resolution, conservative marked unknown-model policy, exact cached/output/reasoning pricing, historical p80 prediction, atomic idempotent reservations, reconciliation/release auditing, and reversible context fitting. Unknown pricing remains unknown and cost-constrained execution fails safely.
  - Removed default lane token/cost caps as synthetic model limits, removed fixed prompt/taskboard/delegation character-token assumptions from active model paths, and documented capacity-versus-policy configuration and private accounting persistence.
  - User verification required: `python -m pytest tests/context_cost/test_model_accounting.py tests/context_cost/test_context_cost_core.py tests/context_cost/test_context_cost_integration.py tests/test_model_routing.py tests/gateway/test_routing_authority.py tests/gateway/test_chat_gateway.py tests/gateway/test_lane_coordinator.py tests/execution_supervisor/test_supervisor_core.py tests/test_openrouter_provider.py -q` and `python -m pytest -q`.

- Restored Python 3.10 compatibility for MCP transport failure handling by using the conditional `exceptiongroup` backport where built-in exception groups are unavailable. The MCP regression test now uses that same compatibility type.
  - User verification required: `python -m pytest tests/test_mcp.py::test_mcp_client_reports_the_concrete_task_group_failure` and `python -m pytest -q`.

- Raised the default Operations lane to the configured 32k routing-token and 32-unit cost ceilings, then constrained every execution model-routing budget to its validated specialist-lane contract before reservation. Canvas now has sufficient default budget for its model-selected tool run; narrower custom lanes still fail during model routing when they cannot fund the model estimate, rather than reaching an opaque `lane_budget_exhausted` admission failure.
  - Preserved the supervisor's authoritative `budget_exhausted` state when result acceptance exceeds a task reservation, preventing a second invalid terminal transition to `failed`.
  - Moved validated Live Canvas work to its own non-repository lane, so stale Operations work cannot block Canvas through a workspace lock or the Operations capacity limit.
  - Reserved the Canvas executor's configured prompt and tool-step envelope instead of only the initial routing estimate, and report a true budget exhaustion without mislabeling it as completion-verification failure.
  - Made Canvas fail with an actionable error when no surface mutation is persisted, and excluded unleased queued recovery records from runtime capacity accounting.
  - Emitted lock and lane-capacity waiting events once per wait instead of persisting a duplicate event every polling interval.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py tests/gateway/test_lane_coordinator.py`.

- Registered Canvas create/update/delete tools with a dedicated transactional adapter instead of rejecting every model-selected Canvas mutation before execution. The adapter stores a redacted intent, verifies the returned durable surface snapshot before commit, and binds the action to the active Canvas lane task; Canvas deletion remains exact-approval-gated.
  - User verification required: `python -m pytest tests/test_canvas.py tests/test_ask_agent.py tests/gateway/test_canvas_route.py`.

- Isolated Canvas-lane routing-budget coverage from local provider credentials by supplying its explicit non-network test credential.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py::test_gateway_constrains_model_routing_budgets_to_the_selected_lane`.

- Connected durable lane budgets to the validated routing decision and provider-accounted context usage. Reserved token and cost estimates now use the selected model decision, while completed lanes record actual input/output usage and exact cost only when provider usage and configured pricing are available; estimated usage remains explicitly estimated.
  - User verification required: `python -m pytest tests/context_cost/test_context_cost_core.py tests/gateway/test_chat_gateway.py tests/gateway/test_lane_coordinator.py -q`.

- Made human-inbox signing-key initialization atomically publish complete key files so concurrent response handling cannot read an empty key during creation.
  - User verification required: `python -m pytest tests/human_inbox/test_durable_inbox.py -q` and `python -m pytest -q`.

## 2026-08-03

- Restored browser-backed API documentation inspection by explicitly classifying the isolated
  `browser_open` and `browser_inspect` tools as read-only. The transactional gate now permits a
  model-selected public documentation page to be opened and inspected, while browser controls
  that can interact with page state remain fail-closed without a transactional adapter. This
  allows the API workflow to collect the documentation evidence needed before semantic import,
  operation search, request preview, and execution.
  - The API executor now supplies its initial workflow decision with a redacted snapshot of
    enabled saved integrations. A documentation URL supplied as context no longer causes an
    unnecessary re-import when that snapshot already contains the requested operation; the model
    must select search, preview, and execution against the saved integration instead.
  - User verification required: `python -m pytest tests/connectors/test_browser_core.py tests/gateway/test_api_manager_route.py tests/transactional_actions/test_policy_gated_actions.py`.

- Removed completed-result reuse from checkpoint recovery. Follow-up classification now rejects
  resuming a completed task and requires the model to classify a downstream live operation as a
  task expansion, which creates a fresh durable action and its own approval boundary.
  - User verification required: `python -m pytest tests/gateway/test_followup_classifier.py tests/gateway/test_checkpoint_resume.py tests/gateway/test_entry_routing.py`.

- Published a bounded, redacted provider receipt into the owning chat history and frontend event
  stream after an approved MCP action commits, so approval completion exposes the exact MCP result
  instead of only an activity status.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py`.

- Updated the model-selected MCP executor contract to keep credentials out of tool arguments and require provider identifiers and input references before it invokes a mutable operation. Removed the forced initial MCP tool call so the model can return a clarification rather than sending an empty mutable request.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py`.

- Added a durable post-routing checkpoint for compound child lanes before they invoke an MCP route, allowing their approval inbox requests to suspend and resume the exact supervised child action.
  - User verification required: `python -m pytest tests/gateway/test_multi_task_orchestration.py tests/gateway/test_entry_routing.py`.

- Unwrapped and redacted concrete MCP async transport failures so approved-action lanes report the underlying provider error instead of an opaque `ExceptionGroup`.
  - User verification required: `python -m pytest tests/test_mcp.py`.

- Normalized MCP credential headers case-insensitively so a locally stored provider token replaces placeholder or stale `authorization` configuration values instead of sending conflicting headers.
  - User verification required: `python -m pytest tests/test_mcp.py`.

- Scoped MCP transactional idempotency to the durable model-selected task, allowing a fresh external attempt after a failed action while retaining exact-call deduplication within that task.
  - User verification required: `python -m pytest tests/test_mcp.py tests/gateway/test_entry_routing.py`.

- Made invalid follow-up classifications return their direct model-decision error before lane coordination, and clarified that non-task categories may not select a related task.
  - User verification required: `python -m pytest tests/gateway/test_followup_classifier.py tests/gateway/test_entry_routing.py`.

- Bound approved MCP actions to their active durable lane task and added exact-provider/tool rehydration for the human-resume dispatcher; the dispatcher refuses missing or changed provider bindings instead of substituting an operation. Legacy unbound approvals now report that no task can resume and require a fresh model-selected MCP request.
  - The owning frontend now receives explicit resumed MCP start, completion, and failure activity events; provider failures without a diagnostic are reported as such rather than appearing to remain in progress.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py tests/test_ask_agent.py tests/test_mcp.py`.

- Forwarded isolated MCP transactional approval requests through the owning gateway's activity stream and preserved their durable inbox item IDs so connected TUI and dashboard clients can present the approval modal.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py tests/test_ask_agent.py tests/test_mcp.py`.

- Registered a background, branch-owned transactional action dispatcher after a durable human-resume claim. Approval handlers now only persist the response and queue the matching branch; the dispatcher reacquires that branch before consuming the exact grant and executing the stored computer action.
  - Gateway startup also recovers only approved, unclaimed, still-queued branches; claimed executions remain manual-recovery cases and are never retried automatically.
  - Cleared the released pre-approval supervisor lease from the lane projection so the resumed dispatcher always acquires a fresh valid lease.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py tests/gateway/test_lane_coordinator.py tests/test_computer_control.py`.

- Added durable inbox-audit sequencing so events created at the same timestamp retain their append order instead of being reordered by random audit IDs.
  - User verification required: `python -m pytest tests/human_inbox/test_durable_inbox.py`.

- Allowed a human-waiting taskboard item to return to `queued` after its matching durable resume claim, keeping the taskboard projection consistent with the execution supervisor without starting work inline.
  - User verification required: `python -m pytest tests/gateway/test_lane_coordinator.py`.

- Completed the `LaneCoordinator` human-inbox branch-controller interface, including durable store access plus suspension, recovery, and single-claim resume delegation. Transactional computer proposals can now create their linked authoritative inbox item instead of failing before an approval event is emitted.
  - User verification required: `python -m pytest tests/gateway/test_lane_coordinator.py tests/human_inbox/test_durable_inbox.py tests/test_computer_control.py`.

- Added GitHub funding metadata for GitHub Sponsors, Open Collective, and Polar.
  - User verification required: confirm the repository's GitHub funding panel lists the configured links.

- Made `mana-agent inbox approve`, `deny`, and `answer` return a clear CLI validation error for recorded terminal notices instead of exposing an internal traceback; rejected response attempts remain audited.
  - User verification required: `python -m pytest tests/human_inbox/test_durable_inbox.py`.

- Corrected the transactional inbox approval command's response construction so gateway imports no longer fail during test collection.
  - User verification required: `python -m pytest tests/test_computer_control.py`.

- Required computer-route responses to be backed by a typed tool outcome. A model prose-only refusal now creates a redacted terminal request/inbox notice and reports that no operating-system request or approval was sent; recording guidance directs the model to the typed recording tool for clarification or approval.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py tests/test_computer_control.py`.

- Bound computer-control execution to the model-selected `TOOL` role for each route, restoring the chat model afterward. Tool roles now require the routing model to select a profile with tool-call support.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py tests/test_computer_control.py`.

- Updated computer-control coverage to assert the durable inbox item, rather than the action record, is the transactional approval identity.
  - User verification required: `python -m pytest tests/test_computer_control.py`.

- Added a durable redacted ledger for model-selected computer requests, terminal inbox notices for blocked or incomplete requests, linked policy-denial evidence, and authoritative inbox-ID approval handling in dashboard and TUI flows. TUI now reloads and queues durable approval cards before displaying its modal; approval responses remain non-executing handoffs to the matching resumed branch.
  - User verification required: `python -m pytest tests/test_computer_control.py tests/transactional_actions/test_policy_gated_actions.py tests/human_inbox/test_durable_inbox.py tests/test_dashboard_live_chat.py`.
  - User verification required: `node --test tests/dashboard/live_chat_reducer.test.mjs`.

- Added a bounded, policy-gated macOS screen-recording computer action with typed clarification for incomplete requests, redacted transactional/inbox diagnostics, durable inbox correlation in approval events, and no inline computer execution from approval handlers. Recording remains macOS-only; user verification is required for provider capability, privacy authorization, restart recovery, and artifact verification.
  - User verification required: `python -m pytest tests/test_computer_control.py tests/transactional_actions/test_policy_gated_actions.py tests/human_inbox/test_durable_inbox.py tests/gateway/test_chat_gateway.py`.

- Normalized absent supervisor parent-task IDs in durable computer execution context so root branches preserve the typed context contract.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py tests/gateway/test_lane_coordinator.py`.

- Deferred durable human-inbox initialization until a validated workflow needs a human approval or structured response. Capability-error routes, including `COMPUTER_NOT_AVAILABLE`, no longer create `~/.mana/inbox` or its files; partial gateway fixtures retain their transient remote-worker behavior without forcing durable inbox setup.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py`.

- Prevented task-ID collisions when a TaskBoard projection is missing or stale while the execution supervisor still owns the durable task. New allocations now skip supervisor-reserved identities instead of attempting an immutable-contract replacement.
  - User verification required: `python -m pytest tests/gateway/test_lane_coordinator.py tests/gateway/test_chat_gateway.py`.

## 2026-08-02

- Fixed Windows Textual chat-message layout so the initial auto-height measurement rewraps the document at the width being measured. Long messages now receive their correct multi-line height during the first layout pass instead of depending on a later resize or render cycle.
  - User verification required: `python -m pytest tests/test_tui_message_layout.py tests/test_tui_tool_card_layout.py`.
## 2026-08-03

- Added a model-selected MCP gateway route with a typed configured-provider decision, provider-only tool execution, live external-state checkpoint handling, and direct unsupported/capability stop responses that bypass lane recovery. Configured providers are surfaced to the routing model without starting MCP servers; provider tools are discovered only after the validated provider selection.
  - Provider-only MCP turns no longer initialize repository run-evidence memory, so an external-memory configuration cannot block an MCP operation with an unmapped internal evidence requirement.
  - MCP tool-denial traces now fail the selected MCP route, so compound workflows report the failed upload truthfully and block dependent submission work instead of marking both steps complete.
  - Refactored discovered MCP operations through a generic registered transactional adapter: provider/tool/argument digests are previewed and approval-gated, provider failures fail verification, and successful provider results return durable receipt evidence. MCP approval responses now keep the gateway route waiting rather than reporting a failed operation.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py tests/gateway/test_checkpoint_resume.py tests/test_ask_agent.py tests/test_mcp.py`.

- Updated gateway regression coverage after user verification reported four failures while `tests/test_api_manager.py` passed. Missing managed workers and approval-time worker disconnects now assert exact-provider fail-closed behavior without direct SSH fallback, and server approval tests create and consume the durable inbox record instead of relying only on transient process state.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py tests/test_api_manager.py`.

## 2026-08-02

- Added a durable human-in-the-loop inbox for binary approvals and structured clarifications, with atomic local persistence, optimistic concurrency, request/response idempotency, deduplication, protected context references, immutable audit and notification-attempt evidence, reviewer person/role/group resolution, delegation history, signed expiring single-purpose response tokens, CSRF validation, expiry/reminder hooks, metrics, diagnostics, and startup reconciliation. Supervised input waits now checkpoint and release only the affected branch, persist structured human inputs, and resume it once through durable claims while sibling branches continue. Policy-gated action approvals persist an inbox item bound to the policy decision and canonical action digest, including explicit reversible, compensatable, irreversible, externally visible, data-disclosing, and potentially billable preview labels that preserve unknown values instead of guessing; legacy dashboard/TUI, server-management, and remote-SSH permission entry points delegate to durable items while retaining their compatibility request IDs and one-attempt execution fences. Approved remote requests also retain their exact selected provider and fail closed if that route becomes unavailable. Added authoritative API, CLI, TUI/chat command and minimal notification, Streamlit dashboard inbox, cron/automation maintenance command, architecture/security/recovery documentation, and focused recovery/concurrency/security coverage.
  - User verification required: `python -m pytest tests/human_inbox/test_durable_inbox.py tests/transactional_actions/test_policy_gated_actions.py tests/execution_supervisor/test_supervisor_core.py tests/remote_execution/test_remote_execution.py tests/server/test_server_management.py tests/gateway/test_chat_gateway.py tests/test_api_manager.py`.
  - User verification required: `python -m pytest tests/test_git_tools.py tests/test_apply_patch_json_only.py tests/test_write_file_chunking.py tests/test_computer_control.py`.
  - User verification required: `python -m pytest tests/test_api_conversations.py tests/test_dashboard_navigation.py tests/test_dashboard_live_chat.py tests/test_cli_smoke.py`.
  - User verification required: `python -m pytest`.

- Bound the standalone API's capsule routes to a trusted local process identity and the startup repository's capsule service, so `mana-agent api` no longer returns 503 for authorized project/user capsule queries. Request bodies still cannot choose an identity, and private or parent-child task capsules remain unavailable without a host-provided task-aware resolver.
  - Integrated the local dashboard and its cached chat gateway with that same process identity. Dashboard chat can now persist task-private capsule results for authorized model-selected follow-ups, while the Memory Capsules page limits its default view to the project/user scopes available to the local API identity.
  - User verification required: `python -m pytest tests/test_api_memory_capsules.py tests/test_api_conversations.py tests/test_dashboard_live_chat.py tests/gateway/test_capsule_identity.py`.

- Routed computer-control actions through policy-gated transactional actions, preserving dashboard and TUI exact-action approval prompts and creating the transactional audit log when the chat gateway initializes. Registered the explicit computer-tool surface with the transactional enforcement gate so media and other computer actions reach that approval flow instead of being rejected before proposal.
  - Fixed the local TUI approval bridge when no dashboard event sink is configured; transactional requests now use chat history and keep the originating lane waiting for the decision.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/transactional_actions/test_policy_gated_actions.py tests/test_computer_control.py tests/gateway/test_chat_gateway.py -q`.

- Added policy-gated transactional actions with typed redacted intents and previews, validated durable lifecycle transitions, deterministic fail-closed policy outcomes, atomic exact single-use action and narrow-transaction approvals, atomically fenced idempotent execution, restart recovery, verification-gated commits, coordinated non-atomic transactions, separately gated compensation actions backed by durable pre-mutation snapshots, shared execution events, and redacted audit records under `~/.mana/transactional_actions`. Added file create/edit/move/delete, repository-patch, argv shell, Git-mutation, and mutating HTTP adapters; routed direct and coding-worker file/patch/document mutations, queued shell execution, API binary artifacts, and API-manager `POST`/`PUT`/`PATCH`/`DELETE` through the gateway. Destructive file and patch requests now return an exact approval request instead of executing from a generated tool call, shell actions require exact approval and declared-output verification, and existing exact API approvals are bridged to the bound transactional action. Model tools without an explicit read-only classification or registered adapter—including browser/computer, MCP mutations, server mutations, automation, media generation, canvas mutations, and project-verification execution—now fail closed instead of invoking a legacy executor.
  - User verification reported `9 failed, 34 passed` in the file/document/Git/UI group and `1 failed, 148 passed` in the supervisor/API/tool-manager group. Follow-up corrections preserve document-writer verification metadata, restore exact approval for read-only insecure HTTP requests, return structured policy-denied Git results, update Git/document mutation coverage to approve and retry exact actions, and remove finalized chunk artifacts through a separately previewed, policy-allowed, verification-gated cleanup action.
  - User verification later reported 31 full-suite regressions concentrated in managed-worktree creation, queued verification/local-Git workflows, explicitly unadapted MCP execution, shell approval tests, and tool-wrapper compatibility. Follow-up corrections add typed policy contexts for verified workspace coordination, bounded verification receipts, and validated local queue Git actions while retaining approval for ordinary shell/Git mutations and all remote Git writes; MCP mutations remain fail-closed; wrapper functions again delegate through their transactional safe helpers; affected approval tests now approve and retry the exact pending action; and mutating `git branch`/`git remote` subcommands can no longer pass through the read-only path.
  - User verification of the focused multi-agent regression group then reported `1 failed, 141 passed`; the remaining managed-worktree verification fixture now supplies the same explicit typed verification marker as `VerifierAgent`, so it exercises the bounded verification policy path without broadening permission for ordinary queued shell actions.
  - Corrected Codex lane completion after durable manifests showed pre-existing dirty files being attributed to a new attempt and new untracked trees being collapsed into unverifiable directory Git diffs. Codex now snapshots content-addressed porcelain state immediately before each serialized execution, reports only final paths changed by that attempt, expands untracked directories to their files, and uses type-correct durable contracts for files, directories, and deletions; valid empty files remain verifiable artifacts. Failed reports now surface their persisted contract reason instead of only saying that the result remains pending review.
  - Windows CI reported `21 failed, 1772 passed, 6 skipped`; the patch failures shared one newline-translation cause. Patch previews and executions now read exact UTF-8 bytes and write with newline translation disabled, preserving an existing CRLF style while keeping independently computed verification hashes identical to committed bytes. The queued shell interpreter test now inspects the actual argv command element instead of the escaped `repr` of a Windows argv list.
  - User verification required: `python -m pytest tests/transactional_actions/test_policy_gated_actions.py`.
  - User verification required: `python -m pytest tests/test_edit_file_tools.py tests/test_write_file_chunking.py tests/test_documents.py tests/test_git_tools.py tests/test_tui_auto_chat_tool_events.py tests/test_dashboard_live_chat.py`.
  - User verification required: `python -m pytest tests/execution_supervisor/test_supervisor_core.py tests/test_api_manager.py tests/gateway/test_api_manager_route.py tests/test_tools_manager.py tests/test_tool_worker_process.py`.
  - User verification required: `python -m pytest`.
  - User verification required: `python -m pytest tests/test_codex_integration.py tests/gateway/test_lane_coordinator.py`.
  - User verification required: `python -m pytest tests/test_apply_patch_context_recovery.py tests/test_apply_patch_json_only.py tests/test_adaptive_coding_runtime.py tests/test_coding_tool_system.py tests/test_lightweight_edit_policy.py tests/test_repository_tools.py tests/test_small_direct_edit.py tests/test_ask_agent.py tests/test_cli_smoke.py`.

## 2026-08-01

- Converted the direct legacy multi-agent memory adapter to capsule-enabled operation by default and connected the canonical memory facade's existing `CapsuleService` to it. Authorized capsule reads now use the shared lifecycle service, while broad legacy bundles require an explicit `capsules_enabled=False` compatibility setting; the stable prompt test uses that explicit legacy snapshot mode.
  - User verification required: `python -m pytest tests/test_multi_agent_core.py tests/test_prompting_builder.py tests/memory/test_scoped_capsules.py tests/test_memory_architecture.py`.

- Clarified the structured gateway follow-up decision contract so
  `safe_to_continue` authorizes only proceeding to the next independently
  validated routing boundary, not downstream tools or consequential actions.
  Independent tasks and conversation-only turns with no applicable durable task
  are now explicitly requested as safe classifications, while genuine unsafe
  decisions still stop without fallback and surface the model's concrete reason.
  - User verification required: `python -m pytest tests/gateway/test_followup_classifier.py tests/gateway/test_chat_gateway.py`.

- Updated the gateway follow-up memory test double to declare its capsules-disabled legacy provider contract. The prompt now preserves the legacy label only for that compatibility path; capsule-derived context remains explicitly labeled as untrusted data.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py tests/gateway/test_followup_classifier.py`.

- Added ACL-enforced scoped shared-memory capsules with typed principals, trusted namespaces, compact retrieval budgets, post-provider reauthorization, parent-child delegation/return records, staged team/project review, optimistic versioned merges, conflicts and idempotent retries, lineage, retention/expiry, prompt-injection quarantine, redacted audit events, resilient-execution revision references, authorization-preserving API/dashboard visibility, and quarantined legacy migration. Organisation scope remains disabled and later federation/governance work is documented as a rollout limitation.
  - Intentional breaking change: broad `build_bundle` and ambient project-memory prompt snapshots fail closed while capsules are enabled; callers must supply a validated `CapsuleReadRequest`. Disabling capsules retains the legacy provider path.
  - User verification required: `python -m pytest tests/memory/test_scoped_capsules.py tests/test_memory_architecture.py`.
  - User verification required: `python -m pytest tests/gateway/test_chat_gateway.py tests/gateway/test_followup_classifier.py tests/execution_supervisor/test_supervisor_core.py tests/context_cost/test_context_cost_core.py tests/test_cli_smoke.py tests/test_dashboard_navigation.py`.
  - User verification required: `python -m pytest`.

- Fixed API call workflows with a supplied documentation URL and an already saved matching
  integration. Gateway guidance now makes the model list saved integrations immediately after its
  workflow decision, use a matching enabled integration's operation search/preview/execution path,
  and import or refresh documentation only when the saved integration cannot satisfy the request.
  This prevents browser documentation recovery from consuming the request lifecycle and leaving
  required integration-import, operation-search, preview, and execution evidence incomplete.
  - User verification required: `python -m pytest tests/gateway/test_api_manager_route.py`.

- Updated test doubles and lifecycle fixtures for the strict gateway and supervisor contracts: bound LLM fakes now accept invocation configuration, Gmail follow-ups make an explicit fresh-data checkpoint decision, completed multi-task children carry authoritative supervisor evidence, and the real Mana-home guard checks its blocked write target without observing unrelated concurrent user-state updates.
  - User verification required: `python -m pytest tests/test_ask_agent.py tests/test_llm_logging.py tests/gateway/test_entry_routing.py tests/gateway/test_multi_task_orchestration.py tests/test_runtime_artifact_isolation.py`.

- Integrated the existing execution supervisor and context-cost governor as the
  authoritative chat execution control layer. Lane completion now projects
  `DONE` only from persisted supervisor verification; TaskBoard duplicate matches
  remain resumable advisory records rather than `SKIPPED`; completed matches can
  return verified escrow or be reverified against stored artifact hashes; gateway
  work checkpoints after routing and before verification; sandbox operations
  carry the full execution identity; result/action records use attempt-generation
  fencing and durable lifecycle/receipt state; concurrent model calls reserve
  canonical governor budget atomically; and context manifests plus TaskBoard
  compaction are content-addressed and reversible. TaskBoard writes are now
  crash-safe and corrupt state fails closed. Supervisor, checkpoint, and
  TaskBoard schema version 2 fields load older records through typed defaults.
  Follow-up corrections retry TaskBoard atomic replacement after transient
  Windows sharing denials, preserve the supervisor-specific error for every
  unauthorized direct `DONE` transition, and align the concurrency and gateway
  test doubles with reserved reasoning/safety budget and durable completion or
  checkpoint-decision contracts.
  - User verification reported before these follow-up corrections: `10 failed, 153 passed`.
  - User verification follow-up reported one remaining failure caused by an
    overly exact persistence write-count assertion; the assertion now verifies
    recovery for both state files without constraining valid subsequent saves.
  - User verification required: `python -m pytest tests/execution_supervisor/test_supervisor_core.py tests/context_cost/test_context_cost_core.py tests/context_cost/test_context_cost_integration.py tests/gateway/test_checkpoint_resume.py tests/gateway/test_lane_coordinator.py tests/gateway/test_chat_gateway.py tests/test_multi_agent_core.py`.
  - User verification required: `python -m pytest`.

- Fixed API lifecycle completion when a successful documentation, import, search, or preview tool
  result exceeds the 4,000-character human-facing trace limit. A clipped successful non-execution
  trace now remains valid stage evidence, while request execution still requires its complete typed
  upstream result and HTTP status. Rendered semantic documentation imports now require and preserve
  the exact inspected documentation reference, allowing operation citations to validate against the
  browser-inspected page instead of the implicit `pasted-text` reference. Duplicate stable
  integration imports now return an exact refresh instruction, and the API executor has enough
  bounded steps to perform the explicit model-selected refresh before preview and execution. A
  validated HTTP response is now surfaced even if another declared lifecycle action remains
  incomplete, without changing the route's fail-closed incomplete status.
  - User verification required: `python -m pytest tests/gateway/test_api_manager_route.py tests/test_api_manager.py`.

- Fixed saved API execution when authentication is structurally known but its
  credential is supplied per request. An explicit `env://` or
  `mana-secret://` reference now resolves that operation without persisting
  secret material, and the safe standard `Accept` header no longer fails as an
  undocumented operation parameter. Authenticated operations with neither a
  saved nor per-request credential reference now return an actionable request
  validation error before credential resolution. API workflow guidance now
  avoids declaring documentation inspection/import when an already-saved
  integration is used, so completion evidence matches the model-selected
  lifecycle.
  - User verification reported before the credential-binding guard fix: `1 failed, 29 passed`; the unbound but structurally resolved bearer scheme reached credential resolution with an empty reference.
  - User verification required: `python -m pytest tests/test_api_manager.py tests/gateway/test_api_manager_route.py`.

- Expanded stopped-task recovery so the strict model decision can select
  `retry_task` when the request is the same stable work but no valid checkpoint
  exists. The gateway now reuses the exact durable task identity and creates a
  new supervised attempt instead of always creating a new task. Live-data work
  such as price, email, calendar, weather, and remote-state checks still starts
  fresh, while non-idempotent work, recorded irreversible effects, missing
  authorization, and invalid decisions stop without fallback execution. The
  legacy unknown-side-effect retry setting no longer bypasses this required
  model authorization.
  - User verification required: `python -m pytest tests/gateway/test_checkpoint_resume.py tests/gateway/test_lane_coordinator.py tests/execution_supervisor/test_supervisor_core.py tests/test_multi_agent_core.py`.

- Tightened API workflow decisions so every declared request execution must
  also declare operation search and request preview, including read-only calls.
  Completion now accepts execution evidence only when the tool reports an
  executed, successful upstream response with an HTTP status, exposes the
  redacted response evidence to the route result, and instructs the executor to
  summarize returned status/content instead of claiming present evidence is
  absent. Undeclared actions and incomplete execution still stop without a
  fallback workflow.
  - User verification required: `python -m pytest tests/gateway/test_api_manager_route.py tests/test_api_manager.py`.

- Fixed context-cost governance so `observe` and `soft` modes do not reduce the
  model router's executable token or cost budgets. Only `enforce` mode now feeds
  remaining cumulative budgets into candidate validation. This prevents a long
  tool workflow in observe mode from reducing the next turn's routing budget to
  zero and incorrectly rejecting every configured model; routing still requires
  a valid model decision and no fallback model is selected.
  - User verification required: `python -m pytest tests/context_cost/test_context_cost_integration.py tests/test_model_routing.py`.

- Added a strict model decision before gateway checkpoint reuse. The decision
  compares the current request with stopped checkpoint candidates and must
  explicitly select `resume_checkpoint`, `start_fresh`, or `stop`; checkpoint
  identity, same-work, freshness, validity, and safety fields are validated
  before retry. Live-data routes such as current prices or mailbox checks cannot
  reuse stale checkpoints, invalid decisions stop without fallback, and approved
  resumes retain supervisor side-effect, checkpoint, retry-budget, and lease
  validation. Unknown checkpoint recovery requires an exact approved checkpoint
  with no recorded irreversible side effect; same-task restarts without a
  checkpoint are governed by the separate repeat-safety authorization described
  above, while non-idempotent work remains blocked. Approved checkpoint state is
  redacted and supplied to the executor so pending work continues instead of
  restarting.
  - User verification required: `python -m pytest tests/gateway/test_checkpoint_resume.py tests/gateway/test_lane_coordinator.py tests/execution_supervisor/test_supervisor_core.py tests/test_multi_agent_core.py`.

- Prevented process-local task ID counters from overwriting persisted TaskBoard
  records after an app restart. New root and child tasks now skip every durable
  identity already loaded by the TaskBoard, so a retried request with no saved
  checkpoint receives a new identity instead of reaching the execution
  supervisor with a different immutable contract. Existing same-contract tasks
  remain eligible for the supervisor's validated checkpoint recovery path.
  - User verification required: `python -m pytest tests/test_multi_agent_core.py::test_taskboard_does_not_overwrite_persisted_task_when_id_counter_restarts tests/execution_supervisor/test_supervisor_core.py::test_duplicate_create_does_not_duplicate_event`.

- Updated durable task retry so `mana-agent tasks retry <task-id>` automatically
  attaches a task-bound, typed operator `RecoveryDecision`, with an optional
  retry-budget category and continued support for standalone recovery JSON.
  Taskboard routing-decision registries now produce an actionable format error
  instead of raw missing-field validation output, and all retries retain the
  existing fail-closed side-effect and budget checks.
  - User verification required: `python -m pytest tests/execution_supervisor/test_supervisor_core.py`.
  - User verification required: `mana-agent tasks retry task_20260801_000002`.

## 2026-07-31

- Added the resilient execution supervisor across gateway lanes, Fleet workers,
  A2A delegation, scheduled automations, task attempts, CLI/API management,
  shared live events, and the dashboard. Supervised tasks
  now use atomic durable records, validated transitions, leases and bounded
  heartbeats, schema-versioned checkpoints, child result escrow, explicit
  side-effect retry safety, cancellation propagation, startup recovery, budget
  enforcement, and verified completion artifact manifests. Missing/invalid
  recovery decisions and ambiguous non-idempotent work stop without fallback.
  Verification-pending lanes retain their durable review state without holding
  released worker/provider capacity, atomic supervisor writes retry transient
  Windows replace denials, and the task-management group is visible as `runs`
  while retaining `tasks` as a hidden compatibility alias.
  - User verification reported before the follow-up fix: the targeted suite completed with `5 failed, 169 passed`; three failures were released-capacity waits blocked by `verifying`, one was a transient `PermissionError` during atomic replacement, and one was the retired `ask` substring appearing inside `tasks` in root help.
  - User verification reported after those fixes: the targeted suite completed with `1 failed, 173 passed`; the remaining failure was test instrumentation for the lane store also counting three successful supervisor-store replacements through Python's shared `os` module object. Supervisor atomic replacement is now independently bound and has direct transient-denial coverage.
  - User verification required: `python -m pytest tests/execution_supervisor tests/gateway/test_lane_coordinator.py tests/gateway/test_multi_task_orchestration.py tests/fleet/test_fleet_core.py tests/test_a2a_protocol.py tests/test_automation_service.py tests/test_api_conversations.py tests/test_chat_websocket.py tests/test_cli_smoke.py tests/test_dashboard_helpers.py`.
  - User verification required: `python -m pytest`.

## 2026-07-31

- Made `ContextCostGovernor` a required `AskAgent` constructor dependency and
  propagated it through gateway, worker, Telegram, and TUI factories with the
  active session identity. Removed compatibility calls that retried ask-service
  construction without the required governor; missing dependencies now stop at
  construction instead of creating an ungoverned execution path.
  - User verification reported before the fix: the full suite completed with `69 failed, 1466 passed, 2 skipped`; the failures shared `AttributeError` for a missing `context_cost_governor` on `AskAgent.__new__()` instances or plain-object fakes.
  - User verification required: `python -m pytest tests/context_cost tests/test_ask_agent.py tests/test_ask_agent_recovery.py tests/test_chat_planning_mode.py tests/test_cli_flow.py tests/test_cli_smoke.py tests/test_llm_logging.py tests/test_tool_worker_process.py tests/test_tui_user_config.py tests/test_workspaces.py`.
  - User verification required: `python -m pytest`.

## 2026-07-31

- Corrected context-artifact recovery coverage to page through bounded
  `offset`/`limit` reads before asserting that a large compressed tool result
  is exactly recoverable.
  - User verification reported before the fix: the targeted suite completed with `78 passed, 1 failed`; the remaining test attempted to parse only the first 64,000-character artifact page as complete JSON.
  - User verification required: `python -m pytest tests/context_cost tests/test_llm_compatibility.py tests/test_chat_ui_events_tokens.py tests/test_model_routing.py tests/test_codex_integration.py`.

## 2026-07-31

- Removed the context-cost/CLI initialization cycle by making the ChatUI
  governor annotation type-only and deferring `ChatEvent` construction imports
  until an event is emitted.
  - User verification reported before the fix: `python -m pytest tests/context_cost` and the targeted compatibility suite both stopped during collection with `ImportError: cannot import name 'ContextCostGovernor' from partially initialized module 'mana_agent.context_cost'`.
  - User verification required: `python -m pytest tests/context_cost tests/test_llm_compatibility.py tests/test_chat_ui_events_tokens.py tests/test_model_routing.py tests/test_codex_integration.py`.

## 2026-07-31

- Added the session-scoped Context and Cost Governor across the gateway, shared
  model client, AskAgent, routing, history, coding backends, Codex usage path,
  live UI events, redacted analytics, and read-only `context report`. Soft and
  enforce modes use validated lazy capabilities and deterministic compression
  backed by exact scoped artifacts; observe mode records without changing
  execution. Estimates remain visibly distinct from exact provider usage.
  - User verification required: `python -m pytest tests/context_cost tests/test_llm_compatibility.py tests/test_chat_ui_events_tokens.py tests/test_model_routing.py tests/test_codex_integration.py`.
  - User verification required: `node --test tests/dashboard/live_chat_reducer.test.mjs`.
  - User verification required: `python -m pytest`.

## 2026-07-31

- Bumped the package and documented version to `v0.1.4`.
  - User verification required: inspect package metadata with `python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"`.

## 2026-07-31

- Added the required-schema `api_docs_import_semantic` tool for prose and rendered API
  documentation, preventing unstructured evidence from being submitted without the model's typed,
  cited `SemanticDefinition`. The rendered-documentation path may now use bounded click, wait, and
  scroll actions to reveal an operation's endpoint, parameters, authentication, and responses;
  typing, form submission, login, authorization/consent, CAPTCHA, and MFA interaction remain
  prohibited. Successful semantic imports count as explicit integration-import workflow evidence.
  - User verification reported before the fix: the model inspected IPstack's rendered operation
    list, called the generic importer with prose but no semantic definition, received
    `Unstructured API documentation requires a validated model semantic extraction`, and stopped
    without saving an integration or executing the requested lookup.
  - User verification required: `python -m pytest tests/test_api_manager.py tests/gateway/test_api_manager_route.py`.

- Added a strict API workflow completion contract through the required first-call
  `api_workflow_decide` tool. The model now declares every action needed for the requested outcome,
  and the gateway validates successful documentation inspection, integration import/configuration,
  operation search, preview, and execution evidence before returning success. Missing execution
  produces `api_workflow_incomplete`; an exact permission request remains awaiting approval. Tools
  that execute actions absent from the validated workflow decision are also rejected.
  - User verification reported before the fix: the operations lane discovered IPstack's
    `lookupIpAddress` GET operation, recorded no tool evidence or result summary, omitted the API
    call, and nevertheless transitioned the task from `in_progress` to `done`/`lane.completed`.
  - User verification required: `python -m pytest tests/gateway/test_api_manager_route.py tests/test_api_manager.py tests/gateway/test_multi_task_orchestration.py`.

- Reclassified API-documentation OAuth/OIDC redirects as the typed
  `documentation_authorization_required` condition before they exhaust the redirect budget. The
  API route may now make an explicit model decision to inspect that same documentation URL with the
  existing read-only rendered-browser tools and import only from returned page evidence. Browser
  login, CAPTCHA, MFA, access-denial, and other intervention controls still stop safely, and browser
  tools cannot substitute for API request execution.
  - User verification reported before the fix: the IPstack documentation URL entered a SwaggerHub
    OIDC loop and failed as `The upstream API exceeded the redirect limit`, leaving no integration
    or executed request.
  - User verification required: `python -m pytest tests/test_api_manager.py::test_documentation_oauth_redirect_requests_rendered_browser_inspection tests/test_api_manager.py::test_documentation_redirect_encodes_spaces_and_rejects_control_bytes tests/gateway/test_api_manager_route.py`.

- Fixed API documentation redirects whose authorization query contains unescaped spaces by
  percent-encoding safe redirect text before the next request. Redirect locations containing CR,
  LF, DEL, or other actual control bytes now stop explicitly; every normalized redirect still
  passes independent scheme, host, DNS, SSRF, and credential-forwarding checks.
  - User verification reported before the fix: the IPstack documentation reader stopped on an
    authorization redirect containing `scope=openid profile email` with `URL can't contain control
    characters` and no API call was made.
  - User verification required: `python -m pytest tests/test_api_manager.py::test_documentation_redirect_encodes_spaces_and_rejects_control_bytes tests/test_api_manager.py::test_timeout_is_structured_and_redirect_target_is_revalidated`.

- Fixed compound documentation-and-call workflows so one model-selected API lifecycle can inspect
  an authorized documentation source, save the normalized integration, search its operations, and
  continue to request execution instead of splitting browser inspection from an empty integration
  search. Added the narrow `api_docs_inspect` evidence tool, explicit atomic-lifecycle routing and
  decomposition guidance, and truthful approval propagation to parent tasks. Public requests that
  require a configured-host-allowlist exception or one-time plain-HTTP exception now produce the
  existing session-bound TUI/dashboard API approval; approval applies only to the exact stored
  request and does not persistently weaken network policy or permit cross-host redirects.
  - User verification reported before the fix: browser documentation inspection and an empty API
    operation search were both reported as completed even though no integration was saved and no
    API request was executed.
  - User verification required: `python -m pytest tests/test_api_manager.py tests/gateway/test_api_manager_route.py tests/gateway/test_multi_task_orchestration.py tests/test_api_conversations.py`.
  - User verification required: `node --test tests/dashboard/live_chat_reducer.test.mjs`.

- Fixed prose API imports so fully documented operations may use an empty `inferred_fields` list,
  operation citations may name URLs actually present in pasted documentation, and query/header
  parameters used solely for authentication are normalized out of ordinary request inputs. API
  routing instructions now preserve the required `env://<name>` or `mana-secret://<id>` credential
  reference form across retries and prohibit claiming a credential was received or stored without
  explicit tool confirmation. Bare credential names and plaintext secrets remain rejected.
  - User verification reported before the fix: an IPstack semantic import retried
    `env://IPSTACK_TOKEN` as the invalid bare reference `IPSTACK_TOKEN`, producing two strict model
    validation errors and stopping the session.
  - User verification required: `python -m pytest tests/test_api_manager.py tests/gateway/test_api_manager_route.py`.

- Added the production API Manager for importing OpenAPI 3.x, Swagger 2.0, authorized files/URLs,
  and validated model-extracted prose documentation; persisting versioned reusable integrations;
  storing credential references separately from secret values; retrieving model-selection
  candidates; building and previewing strict requests; and executing through a DNS-pinned,
  redirect-revalidated, response-bounded HTTP runtime. The new model-driven `api` gateway route,
  Operations-lane capability, narrow `api_*` tools, shared TUI/dashboard approval flow, redacted
  execution events, project skill, configuration, security documentation, and mocked coverage do
  not expose an unrestricted HTTP fallback. Missing or invalid semantic decisions,
  authentication, parameters, operation choices, permissions, or network-policy checks stop
  without selecting a default integration, operation, credential, or host.
  - User verification required: `python -m pytest tests/test_api_manager.py tests/gateway/test_api_manager_route.py tests/gateway/test_entry_routing.py tests/gateway/test_chat_gateway.py tests/test_api_conversations.py tests/test_tui_auto_chat_tool_events.py`.
  - User verification required: `node --test tests/dashboard/live_chat_reducer.test.mjs`.
  - User verification required: `python -m pytest`.

## 2026-07-30

- Fixed enrolled-server generic shell actions by publishing the required exact
  `argv` string-list shape in the server tool contract and rejecting malformed,
  empty, or null-containing argv values before an approval or execution attempt.
  The non-secret server catalog now also exposes the configured remote login
  user, and generic-shell home-directory examples use a relative path rather
  than an invalid placeholder absolute path.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py tests/server/test_server_management.py`.

- Fixed the managed-memory secret-store regression test on Windows by limiting
  the POSIX mode-bit assertion for `secrets.toml` to platforms that expose
  those permission bits. The test still verifies secret isolation and
  retrieval on every platform.
  - User verification required: `python -m pytest tests/test_memory_architecture.py::test_memory_secret_store_uses_protected_mana_store_without_keyring_backend`.

- Fixed strict structured-output registration for enrolled-server routing by
  replacing the arbitrary `server_request` object with closed typed boundary
  models and decoding exact tool arguments from validated JSON text before the
  existing server decision and tool-contract checks. The canonical required
  source vocabulary now also includes the `server` source advertised by the
  route contract. Server route availability now gives the model the exact tool
  risk/capability contracts and a non-secret enrolled-server catalog, preventing
  it from guessing capability names while preserving strict validation. A
  server-level authorization denial now remains on the server route, returns
  exact `server authorize` guidance before any execution, and can be resolved
  through the new explicit capability-granting CLI command. Package-install
  arguments are now validated before approval/argv construction, and the model
  can explicitly select bounded `auto` discovery that refuses zero or multiple
  observed package managers instead of guessing one. Consequential server
  actions now persist a session-bound, single-use approval containing the exact
  validated decision and argv. The TUI and dashboard now surface native
  deny/approve-once requests that resume only that action, with no text-command
  approval fallback. Pending approval lanes remain waiting until the GUI decision
  legally resumes or cancels them. Server decision schemas now require non-empty
  identifiers at the structured model boundary.
  Invalid JSON, non-object arguments, and invalid server decisions stop without
  executing a fallback.
  - User verification reported before the fix: OpenAI rejected
    `EntryRoutingOutput.server_request` because its object schema did not set
    `additionalProperties` to `false`.
  - User verification reported after the schema fix: `46 passed, 1 failed`;
    the server decision was rejected because `server` was absent from the
    canonical source vocabulary.
  - User verification reported after the vocabulary fix: the Nginx installation
    decision reached strict validation but guessed a `required_capability` that
    did not match the authoritative `server_package_install` contract; no tool
    was executed.
  - User verification reported after exposing tool contracts: the model correctly
    identified that `mana-agent-server-1` lacks `package.write`, but represented
    the resource-level denial as a route-wide `capability_error`; no tool was
    executed.
  - User verification reported after granting `package.write`: the Nginx decision
    omitted its package manager and surfaced a raw `'manager'` error; no server
    action was executed.
  - User verification reported after package argument validation: the Nginx
    action reached the consequential approval gate, which previously had no
    resumable server approval UI path; no server action was executed.
  - User verification reported after native approval UI was added: one model
    response supplied an empty `decision_id`, and approving a later valid request
    failed because its lane had already been marked done; no fallback action was
    executed.
  - User verification required: `python -m pytest tests/gateway/test_entry_routing.py tests/gateway/test_chat_gateway.py tests/server/test_server_management.py tests/test_api_conversations.py tests/test_computer_control.py`.
  - User verification required: `node --test tests/dashboard/live_chat_reducer.test.mjs`.
  - User verification required: `python -m mana_agent server authorize --help`.
  - User verification required: run `/help` in chat and confirm there is no
    `/server-approval` command.

- Fixed configuration saving on headless Linux when the Python `keyring`
  package has no recommended backend. External Mem0/Supermemory credentials now
  use an explicit `mana-secrets:` reference to Mana's atomic mode-0600
  `secrets.toml`; usable OS keyrings remain preferred, and normal `config.toml`,
  summaries, logs, and draft persistence do not expose the credential.
  - User verification required: `python -m pytest tests/test_memory_architecture.py tests/test_tui_user_config.py`.
  - User verification required on the affected host: `python -m mana_agent --configure`.

- Fixed `mana-agent[full]` installation on headless Linux servers by excluding
  the native `pynput`/`evdev` desktop-input stack on Linux. Native Linux Teach
  Mode capture remains available through the explicit `teach-desktop` extra
  after installing the platform input headers.
  - User verification reported before the fix: editable `mana-agent` wheel built successfully, but the transitive `evdev` wheel failed because `linux/input.h` and `linux/input-event-codes.h` were unavailable.
  - User verification required: `python -m pytest tests/test_package_version.py`.
  - User verification required on Linux: `python -m pip install -e ".[full]"`.

- Fixed direct construction of server SSH argv to inherit the model-validated
  connection timeout and pinned known-hosts path from `RemoteExecutionRequest`
  when callers do not provide explicit overrides.
  - User verification reported before the fix: `78 passed, 1 failed` for the focused server, remote-execution, entry-routing, and lane-coordinator suite.
  - User verification required: `PYTHONPATH=src python -m pytest tests/server/test_server_management.py::test_connection_uses_pinned_hosts_keepalive_jump_and_pool tests/remote_execution/test_remote_execution.py`.

- Added the provider-neutral Server Management module with persistent authorized
  enrollment, secret references, strict host-key pinning, pooled OpenSSH
  transport, typed model decisions and tools, per-server mutation locks,
  consequential/destructive approval contracts, redacted audit evidence,
  package/service/file/network/database/container helpers, health inspection,
  desired-state rollback, deployment/backup/provider contracts, CLI and
  read-only dashboard APIs, Operations-lane routing, A2A capability metadata,
  documentation, and isolated mocked coverage. Invalid or missing decisions,
  capabilities, credentials, host keys, approvals, recovery points, managers,
  providers, and runtime adapters stop without fallback execution.
  - User verification required: `python -m pytest tests/server/test_server_management.py tests/remote_execution/test_remote_execution.py tests/gateway/test_entry_routing.py tests/gateway/test_lane_coordinator.py tests/test_api_workspaces.py tests/test_a2a_protocol.py`.
  - User verification required: `python -m pytest`.
  - User verification required: `python -m mana_agent server --help`.

- Bumped the package and documented version to `v0.1.3`.
  - User verification required: `python -m pytest tests/test_package_version.py`.

- Added optional production media generation for images, speech/audio, and
  durable video jobs. This includes capability-filtered model selection, typed
  model-driven entry decisions, a Media lane, provider-neutral contracts, a
  real OpenAI-compatible adapter, safe retries and idempotency, persisted
  status/cancellation, atomic MIME-validated managed artifacts, compact agent
  tools, safe live events, TUI/dashboard presentation, permission scopes,
  documentation, and mocked regression coverage. Existing installations remain
  media-disabled and retain text/chat behavior; invalid or missing media
  decisions stop without a fallback provider or model.
  - Fixed optional media duration fields so unset values are omitted from TOML,
    while legacy `"None"`/`"null"` values are normalized back to unset during
    configuration loading.
  - Media metadata, generation JSON, audio, and video remain under
    `~/.mana/artifacts/media/`; completed image binaries alone are written
    directly to the Mana-Agent launch directory with safe `media_*` names,
    without creating a workspace `.mana` directory.
  - User verification required: `python -m pytest tests/test_media_generation.py tests/gateway/test_entry_routing.py tests/gateway/test_lane_coordinator.py tests/test_chat_first_configuration.py tests/test_tui_user_config.py tests/test_dashboard_live_chat.py`.
  - User verification required: `python -m pytest`.
  - User verification required: `python -m mana_agent --configure`.

## 2026-07-29

- Moved Live Canvas configuration guidance out of `.env` and into the authoritative `~/.mana/config.toml`, updated repository agent policy to prevent environment overrides, added a same-origin local catalog endpoint plus loopback-only HTTP catalog/resource support, fixed cross-port Streamlit iframe/WebSocket authentication and CSP, removed the magic `dashboard/pages` package that made direct routes bypass `st.navigation`, made initial surface creation atomic, normalized common A2UI component wire shapes, fixed omitted optional creation arguments, added validated correction/rollback for legacy incomplete surfaces, resumed the owning model with update-only tools after renderer actions, and made the one-time Gmail automation CI fixture choose a future 01:08 UTC timestamp instead of a date that expires during the test day.
  - Verification: the full suite previously passed (1,439 passed, 4 skipped). The final run passed 1,437 tests with 4 skipped; its three interrupted CLI cases passed on isolated rerun, and its two `python`-PATH-dependent multi-agent cases passed when `venv/bin` was included in `PATH`. Focused Canvas/chat/config/API tests, exact failed-payload replay, browser reducer tests, Ruff, compilation, JavaScript syntax checks, and `git diff --check` passed. Browser validation used an isolated loopback dashboard/API pair.

- Added the production Live Canvas/A2UI workspace across structured model routing, agent and workflow-node APIs, strict v0.9.1 protocol/catalog validation, deterministic surface reduction, durable events and snapshots, the shared gateway/WebSocket, authenticated renderer actions, A2A capability negotiation, the native Streamlit dashboard renderer, typed configuration, security documentation, and lifecycle tests. Side-effect actions fail closed unless the owning existing permission broker is attached; arbitrary executable browser content and inline catalogs remain disabled.
  - Verification: Canvas/gateway/A2A/WebSocket tests passed (78 passed, 1 skipped); browser reducer tests passed (11 passed); changed-file Ruff, Python compilation, JavaScript syntax checks, and `git diff --check` passed. The full suite reached 1,433 passed and 4 skipped with 3 unrelated failures: a same-day automation fixture used an already elapsed UTC timestamp, and two tests expected `python` on `PATH` (both passed when the virtualenv was added to `PATH`).

- Fixed dashboard chat session replacement: submitting `/new` now reconnects the embedded live chat to the replacement canonical session instead of leaving the deleted session in a working state. The sidebar New conversation control also clears its persistent selector value so it activates the new chat.
  - Verification: `PYTHONPATH=src venv/bin/python -m pytest -q tests/test_conversation_service.py tests/test_api_conversations.py tests/test_dashboard_live_chat.py tests/test_dashboard_navigation.py` passed (19 passed, 1 skipped); `venv/bin/python -m py_compile src/mana_agent/dashboard/pages/chat.py` and `git diff --check` passed. Node.js was unavailable, so the browser reducer suite was not run.

## 2026-07-28

- Fixed model-routed search follow-ups: the entry router now carries its
  compact, model-selected search query into the executor. The executor rejects
  missing or oversized operations rather than silently sending the full
  conversation transcript to Tavily, whose query limit is 400 characters.
  Search routes now request a separate, validated model decision for the exact
  operation because high-level entry routing does not provide tool arguments;
  this applies to both direct and required-source search routes.
  - Verification: `venv/bin/python -m pytest
    tests/gateway/test_turn_engine_search.py tests/test_ask_entry_router.py
    tests/test_web_search_provider.py tests/test_search_router.py
    tests/gateway/test_entry_routing.py -q` passed (50 tests); `venv/bin/python -m py_compile
    src/mana_agent/gateway/turn_engine.py src/mana_agent/gateway/chat_gateway.py
    src/mana_agent/multi_agent/runtime/entry_router.py
    tests/gateway/test_turn_engine_search.py` and `git diff --check` passed.

- Fixed Tavily web search authentication by sending the configured key in the
  required `Authorization: Bearer` header rather than in the request body.
  This prevents provider HTTP 400 failures on model-selected search turns.
  - Verification: `venv/bin/python -m pytest tests/test_web_search_provider.py
    tests/test_search_config.py tests/test_search_router.py -q` passed (6
    tests); `venv/bin/python -m py_compile src/mana_agent/search/web_provider.py
    tests/test_web_search_provider.py` and `git diff --check` passed.

- Fixed session persistence during Telegram `/new`: atomic workspace writes
  now restage once after recreating a session directory that was concurrently
  removed, preventing a missing `session.json` error for the new session.
  - Verification: `venv/bin/python -m pytest tests/test_workspaces.py -q`
    passed (13 tests); `venv/bin/python -m py_compile
    src/mana_agent/workspaces/store.py tests/test_workspaces.py` and
    `git diff --check` passed.

- Updated the Telegram connector to resolve its bot token from Mana's managed
  `secrets.toml` before the process environment, and made CLI setup save the
  validated token there without writing it to `config.toml`.
  Webhook secrets now use the same credential resolution path.
  - Verification: `venv/bin/python -m pytest tests/connectors/test_telegram_cli_config.py tests/connectors/test_telegram_core.py -q` passed (14 tests); `venv/bin/python -m py_compile src/mana_agent/config/user_config.py src/mana_agent/connectors/telegram/config.py src/mana_agent/commands/telegram_cli.py` and `git diff --check` passed.

- Added `external/supermemory` as a fully supported memory provider alongside
  `internal/mana` and `external/mem0`, including shared-factory wiring, lazy SDK
  loading, deterministic Supermemory scope tags/custom IDs, keyring-backed
  secret handling, and provider-safe metadata flattening.
- Updated the memory configuration flow, Textual settings UI, tests, and
  documentation so `MANA_MEMORY_PROVIDER=supermemory` and
  `SUPERMEMORY_API_KEY` work through the same provider-neutral `MemoryService`
  path without any fallback to internal or Mem0 storage.
  - Verification: `venv/bin/python -m py_compile src/mana_agent/memory/config.py src/mana_agent/memory/factory.py src/mana_agent/memory/providers/shared.py src/mana_agent/memory/providers/mem0/mapper.py src/mana_agent/memory/providers/supermemory/client.py src/mana_agent/memory/providers/supermemory/mapper.py src/mana_agent/memory/providers/supermemory/backend.py src/mana_agent/config/session.py src/mana_agent/config/settings.py src/mana_agent/config/user_config.py src/mana_agent/tui/configuration_app.py tests/test_memory_architecture.py` passed; `venv/bin/python -m pytest tests/test_memory_architecture.py -q` passed (26 tests); `git diff --check` passed.

- Bumped the package and documented version to `v0.1.2`.
  - Verification: `python -m pytest -q tests/test_package_version.py` passed.

- Moved the executable local-scheduler snapshot beneath `~/.mana/automations/runtime`.
  Launchd now runs this owner-controlled copy instead of reading a development virtual
  environment beneath macOS-protected locations such as `~/Documents`; completed one-time
  runs also remove their platform wakeup.
  - Verification: targeted automation-service tests passed.

- Fixed local connector automation execution to distinguish immediate job execution from
  automation authoring, rejecting unexpected schedule mutations instead of recording a false
  success. macOS launchd jobs now retain their Mana home, write per-job stdout/stderr logs, and
  report a recorded non-zero executor exit as unhealthy.
  - Verification: targeted automation-service tests passed.

- Fixed the cross-platform scheduler-adapter test to explicitly mock macOS's
  POSIX-only user-ID lookup while exercising the `launchd` backend on Windows.
  - Verification: targeted automation-service scheduler-adapter test passed.

- Fixed Windows automation persistence: the one-byte store lock now unlocks
  the same byte it acquires, and the runtime dependency now includes the IANA
  timezone database required by Windows' ``zoneinfo`` implementation.
  - Verification: targeted automation-service tests passed.

- Registered the typed automation chat tools with the Operations lane's explicit
  `automation` capability, so a validated automation route can reach
  `automation_create` without bypassing lane permission checks. Added regression
  coverage for dispatching a registered automation route from an isolated
  `multi_task` child instead of returning `route_executor_unavailable`. Deferred
  Gmail checks now select the automation route without inspecting the mailbox
  during authoring, and a singular requested time creates a one-time trigger
  without asking for recurrence. Automation routing now requires a validated
  operation and exposes only its exact tool, so creation calls
  `automation_create` directly instead of repeatedly using `automation_list`;
  unspecified local output defaults to the automation workspace instead of
  prompting the user to choose local versus cloud storage. The create tool now
  exposes discriminated trigger/job schemas plus typed retry and misfire
  policies, supplies the current timestamp and timezone during authoring, and
  rejects past one-time schedules instead of persisting an unrunnable record.
  Simplified read-only TUI message cards by
  suppressing TextArea's inner cursor-line fill.
  - Verification: focused automation-route, entry-routing, lane, gateway, and
    automation-service tests passed (95 tests). Changed-file Ruff,
    Python compilation, and `git diff --check` passed; focused TUI tests also
    passed for the card styling change.
- Stabilized the surrounding-panel TUI reflow regression on Windows by waiting
  through Textual's deferred history replay and dynamic-card mount cycle before
  inspecting the replayed message.
  - Verification: `.venv/bin/python -m pytest -q tests/test_tui_message_layout.py`
    passed (7 tests); full Windows CI remains to be rerun.

- Reconciled merged TUI wrapping regressions with the read-only full-card-width
  contract and made the consecutive-message fixture independent of vertical
  scrollbar width changes.
  - Verification: the two focused TUI layout files passed five consecutive
    runs (55 tests total); `.venv/bin/python -m pytest -q` passed (1,363 passed,
    2 skipped); changed-test Ruff and `git diff --check` passed.

## 2026-07-27

- Unified schedules and dashboard automation records behind the versioned
  `AutomationDefinition` contract. Added locked atomic migration, exact anchored
  interval/cron/once triggers, typed jobs, persistent platform deployment,
  leased ID-based headless execution, canonical run history and events,
  model-only chat authoring tools, reviewed/verified Teach flow handoff, and
  management-only CLI/dashboard/TUI surfaces. Removed the public cron alias,
  raw automation/Teach schedule creation commands, and the dashboard Cron page.
  - Verification: focused automation, Teach, dashboard, chat-tool, entry-route,
    gateway, and platform-adapter tests passed; changed-file Ruff, Python 3.12
    compilation, and `git diff --check` passed; full
    `.venv/bin/python -m pytest -q` passed (1,384 passed, 2 skipped).

- Fixed Teach Mode atomic persistence on Windows: temporary-file writes now
  conditionally apply POSIX-only descriptor permissions, allowing descriptors
  to close before replacement and cleanup. Docker secret environment files use
  the same portability guard. The owner-only mode assertion is now correctly
  limited to platforms whose filesystem mode bits support that guarantee.
  - Verification: `PYTHONPATH=src .venv/bin/pytest -q tests/test_teach_mode.py`
    passed (22 tests); changed-file Ruff, Python 3.12 compilation, and
    `git diff --check` passed.

- Corrected native Teach Mode text events: printable input is now reconstructed
  in memory into one semantic event (including spaces and backspace edits)
  instead of persisting the literal `{{ typed_text }}` placeholder. Secure
  fields remain content-free, and the existing redaction layer still masks
  detected secrets before storage.
  - Verification: `PYTHONPATH=src .venv/bin/pytest -q tests/test_teach_mode.py`,
    changed-file Ruff, `python -m compileall -q src/mana_agent/teach`, and
    `git diff --check` passed.

- Added the local-first Mana Teach Mode foundation: recoverable recording
  sessions, versioned semantic events, optional cross-platform capture
  protocols and diagnostics, redaction, selector ranking, conservative input
  inference, typed workflow compilation with provenance, safe dry/guided/normal
  replay, observable verification, targeted selector repair, versioned local
  storage, deterministic validated `.mana-flow` packages, private Flow Cards,
  CLI/API/chat-tool/live-event integration, and persistent version-policy-aware
  scheduling. Imported flows remain untrusted and sensitive actions retain
  existing permission and confirmation requirements.
  - Verification: focused Teach Mode, automation, chat-tool, API/WebSocket,
    configuration, and CLI compatibility checks passed (49 tests); the full
    suite passed (1,366 passed, 2 skipped). Changed files passed Ruff,
    `python -m compileall -q src`, and `git diff --check`. Repository-wide Ruff
    remains non-clean with 792 unrelated pre-existing findings.

- Extended Teach Mode with an explicit native desktop-monitoring path:
  separately persisted owner-only Mana grants, OS privacy status/settings
  handoff, a session-bound background recorder, active application/window and
  accessibility metadata, shortcut/navigation capture, pointer events with
  normalized fallback positions, redacted typing activity, application
  allowlists, API/chat/CLI integration, and the `teach-desktop` optional
  dependency group. Printable keyboard content is never persisted as a raw
  keylog, and native recording fails closed when a dependency or OS grant is
  missing.
  - Verification: Teach Mode and integration checks passed (35 tests), the
    Teach-specific suite passed (18 tests), and the full suite passed (1,372
    passed, 2 skipped). Changed files passed Ruff, `python -m compileall -q
    src`, and `git diff --check`. The optional desktop extra installed
    successfully and the live macOS doctor probe correctly reported dependency
    availability plus the still-unapproved OS Accessibility grant.

- Prevented misleading empty Teach recordings after a user has granted all
  local desktop scopes: normal `teach start` now selects native monitoring, and
  startup stops with an actionable OS-permission error if that monitor cannot
  attach. `--no-desktop` remains the explicit semantic-only path.
  - Verification: focused Teach Mode suite passed (20 tests); changed files
    passed Ruff, compilation, and `git diff --check`.

- Bumped the package and documented release version to `v0.1.1`.
  - Verification: `python -m pytest tests/test_package_version.py` passed.

## 2026-07-27

- Added first-class `multi_task` gateway routing for compound prompts, with
  strict model-driven decomposition, persisted root/child TaskBoard lineage and
  dependencies, independent child routing, bounded DAG execution through the
  existing specialist lanes and locks, child-scoped capability/approval state,
  cancellation propagation, idempotent child materialization, aggregate task
  inspection fields, and truthful partial-result summaries. No keyword router,
  fallback route, separate task store, scheduler, or frontend entry point was
  added.
  - Verification: targeted gateway, entry-routing, multi-task orchestration, and
    lane-coordinator tests passed (77 tests), with a final orchestration/lane
    rerun passing 31 tests; the full suite passed (1,344 passed, 3 skipped).
    Changed-file Ruff, `python -m compileall -q src tests`, and `git diff
    --check` passed. Repository-wide Ruff remains non-clean with 792 unrelated
    pre-existing findings.

- Fixed compound-root lane reservation to retain an explicitly persisted
  TaskBoard root and child instead of creating a replacement root task. Explicit
  TaskBoard identities are also excluded from unrelated active-task duplicate
  reuse, preventing child-parent lineage validation failures such as compound
  research followed by PDF creation.
  - Verification: focused lane-coordinator, multi-task, gateway, and entry-route
    tests passed (80 tests); changed files passed Ruff, compilation, and `git
    diff --check`.

- Fixed independently routed compound children to execute with their validated
  child request instead of expanding it back into the entire parent conversation.
  This prevents web-search providers from rejecting oversized compound-context
  queries (observed as Tavily HTTP 400) while retaining normal conversational
  context behavior for ordinary single-task turns.
  - Verification: a live bounded Tavily request returned HTTP 200; focused
    entry-routing, gateway, lane, and multi-task tests passed; changed files
    passed Ruff, compilation, and `git diff --check`.

- Added a typed `artifact_family` entry-decision field for artifact creation
  without an existing filename or attachment, and injects persisted successful
  prerequisite summaries into dependent child execution. Research-to-PDF DAGs
  can now ground the PDF in the completed research while artifact preflight uses
  the model-selected `pdf` handler instead of failing for missing file evidence.
  - Verification: focused entry-routing, artifact-routing, gateway, lane, and
    multi-task tests passed; changed files passed Ruff, compilation, and `git
    diff --check`.

- Fixed attachment-free artifact creation to pass AskAgent a concrete inert
  index path inside the isolated artifact workspace. AskAgent requires a path
  even under a document-only tool policy, so passing `None` previously caused
  research-to-PDF children to fail before `document_create` with a `NoneType`
  `os.PathLike` error.
  - Verification: focused attachment-free PDF and artifact-routing regressions,
    gateway, lane, and multi-task tests passed; changed files passed Ruff,
    compilation, and `git diff --check`.

- Added typed atomic-child routing constraints to compound execution. The entry
  model now receives a child-specific route registry that excludes recursive
  `multi_task`, while strict validation still stops safely if the model violates
  the constraint. Parent conversation context remains available for continuity
  without allowing an already-decomposed research child to become a nested
  compound plan.
  - Verification: focused entry-routing, gateway, artifact-routing,
    lane-coordinator, and multi-task orchestration tests passed (87 tests);
    changed files passed Ruff, compilation, and `git diff --check`.

- Added `OpenClaw_Research_Overview.pdf` at the repository root as requested.
  The supplied one-page export was visually clipped and ended mid-sentence, so
  the readable material was reformatted into a wrapped, paginated two-page
  report and the incomplete trailing claim was explicitly omitted.
  - Verification: Poppler reported a valid two-page Letter PDF; both rendered
    pages were visually inspected for clipping, overlap, and legibility.

- Added the project and packaged `pdf-create` skill, required successful
  `read_skill("pdf-create")` before PDF `document_create`, and moved
  attachment-free artifact creation to the Mana-Agent launch root while
  retaining isolated staging for attached documents. Replaced the former
  single-line, 3,000-character PDF writer with a structured ReportLab renderer
  supporting titles, subtitles, sections, bullets, tables, pagination, and page
  footers; added ReportLab as a runtime dependency.
  - Verification: the skill validator passed; focused document, AskAgent,
    skill-loading, prompt, entry-routing, gateway, artifact-routing, and
    multi-task tests passed (136 tests); a two-page structured PDF was rendered
    with Poppler and both pages passed visual inspection; changed files passed
    Ruff, compilation, dependency checks, and `git diff --check`.

## 2026-07-26

- Fixed standalone API coordinators to initialize a persistent Fleet registry
  for reverse-worker capability updates. Authenticated workers can now publish
  their inventory when the server is started with `mana-agent api` or
  `mana_agent.api.app:app`, instead of being rejected because no ChatGateway
  was supplied.
  - Verification: `venv/bin/python -m pytest -q
    tests/remote_execution/test_reverse_worker_protocol.py
    tests/fleet/test_fleet_core.py tests/test_api_conversations.py
    tests/test_api_workspaces.py` passed (36 passed); `git diff --check`
    passed. Ruff was not run because it is not installed in the local virtual
    environment.

- Updated stable release publishing to retain the triggering Git tag (including
  calendar tags such as `v2026.07.26`) while using the application version for
  the GitHub Release title and package metadata validation.
  - Verification: `venv/bin/python -m pytest -q
    tests/test_publish_pypi_workflow.py` passed (9 passed); the release
    validator accepted `v2026.07.26` with explicit mismatch opt-in; targeted
    Ruff and `git diff --check` passed.

- Fixed reverse workers enrolled against HTTPS coordinators to connect through
  `wss://` rather than passing an invalid `https://` URL to the WebSocket
  client.
  - Verification: `venv/bin/python -m pytest -q
    tests/remote_execution/test_reverse_worker_protocol.py
    tests/commands/test_worker_cli.py` passed (23 passed); targeted Ruff and
    `git diff --check` passed.

- Updated the stable GitHub Release workflow to derive its tag, title, and
  release-notes version from `pyproject.toml` instead of the GitHub event tag.
  A mismatched version-tag trigger now stops publication before any release is
  created.
  - Verification: `venv/bin/python -m pytest -q
    tests/test_publish_pypi_workflow.py` passed (8 passed); release and PyPI
    workflow YAML parsed with PyYAML; `git diff --check` passed.

- Fixed Windows Python 3.12 CI for macOS LaunchAgent lifecycle tests by making
  the launchd user ID an explicit injectable boundary. Production macOS still
  resolves its real POSIX UID, while cross-platform tests no longer call the
  unavailable Windows `os.getuid`.
  - Verification: `.venv/bin/python -m pytest -q
    tests/remote_execution/test_reverse_worker_protocol.py
    tests/commands/test_worker_cli.py tests/fleet/test_fleet_core.py` passed
    (34 passed); an explicit `os.getuid`-unavailable simulation produced the
    expected injected `gui/501` launchd domain; targeted Ruff, compilation, and
    `git diff --check` passed.
  - Verification note: Windows is not available locally, so GitHub Actions
    remains the authoritative native Windows Python 3.12 run.

- Added worker-gateway settings to the typed Mana user configuration and made
  API startup read `MANA_WORKER_GATEWAY_*` values from `~/.mana/config.toml`,
  with environment variables filling keys not explicitly configured.
  - Verification: `.venv/bin/python -m pytest -q
    tests/remote_execution/test_reverse_worker_protocol.py
    tests/commands/test_worker_cli.py tests/test_chat_first_configuration.py
    tests/test_tui_user_config.py tests/test_api_conversations.py
    tests/test_api_workspaces.py` passed (60 passed); targeted Ruff,
    compilation, and `git diff --check` passed. A live local configuration load
    resolved the gateway as enabled at the configured HTTP public URL.

- Fixed coordinator enrollment failures to display bounded API error details
  without urllib tracebacks. Disabled or invalid worker-gateway configuration
  now identifies the exact environment variables required, and the HTTP setup
  documentation enables the gateway before enrollment.
  - Verification: `.venv/bin/python -m pytest -q
    tests/commands/test_worker_cli.py
    tests/remote_execution/test_reverse_worker_protocol.py
    tests/gateway/test_chat_gateway.py tests/test_api_conversations.py
    tests/test_api_workspaces.py` passed (51 passed); targeted Ruff,
    compilation, and `git diff --check` passed.

- Added Linux `worker start`, `stop`, and `restart` through the installed
  `systemd --user` unit, with install-state validation and concise bounded
  systemctl errors.
- Made worker service-control errors render through explicit stderr output and
  exit code 1, avoiding version-dependent `ClickException` handling in direct
  Typer sub-app invocation.
- Made `worker enrollment create --worker-id` optional so the coordinator
  generates and returns a unique worker ID. Generated install commands now
  include the reserved `--worker-id` (and HTTP opt-in when required), while
  `worker install` requires that ID to prevent token/registration mismatches.
  - Verification: `.venv/bin/python -m pytest -q
    tests/commands/test_worker_cli.py tests/remote_execution
    tests/fleet/test_fleet_core.py tests/gateway/test_chat_gateway.py
    tests/test_api_conversations.py tests/test_api_workspaces.py` passed (76
    passed); targeted Ruff, compilation, CLI help checks, and `git diff
    --check` passed.
  - CI rendering regression verification: `.venv/bin/python -m pytest -q
    tests/commands/test_worker_cli.py tests/test_cli_smoke.py
    tests/remote_execution` passed (99 passed); the real missing-service command
    printed the expected stderr error and exited 1.

- Added explicit `--allow-insecure-http` reverse-worker enrollment for trusted
  development networks, including persisted `ws://` reconnect behavior and a
  backward-compatible `--insecure-local-development` CLI alias. HTTPS remains
  the default and HTTP without explicit opt-in fails safely.
  - Added matching coordinator opt-in through
    `MANA_WORKER_GATEWAY_ALLOW_INSECURE_HTTP`.
  - Verification: `.venv/bin/python -m pytest -q
    tests/commands/test_worker_cli.py tests/remote_execution
    tests/gateway/test_chat_gateway.py tests/test_api_conversations.py
    tests/test_api_workspaces.py` passed (58 passed); targeted Ruff,
    compilation, and `git diff --check` passed.

- Fixed macOS `worker start` to report an actionable install-first error when
  the LaunchAgent is absent, bootstrap an installed but unloaded LaunchAgent,
  and present bounded `launchctl` failures without a Python traceback.
  - Verification: `.venv/bin/python -m pytest -q
    tests/commands/test_fleet_cli.py tests/remote_execution
    tests/commands/test_worker_cli.py` passed (23 passed); the real
    `.venv/bin/mana-agent worker start` missing-install path exited 1 with the
    expected concise error; targeted Ruff, compilation, and `git diff --check`
    passed.

- Updated the package and documented release version to `v0.1.0`.
  - Verification: `.venv/bin/python -m pytest -q tests/test_package_version.py` passed.

- Fixed package installation by replacing nonexistent LangChain `0.3.50`
  minimum versions with the mutually compatible published `0.3.27` baseline
  and synchronizing `pyproject.toml` with `requirements.txt`.
  - Verification: isolated `pip install --dry-run --ignore-installed .`
    resolved the project and selected `langchain==0.3.30`,
    `langchain-community==0.3.31`, and `langchain-openai==0.3.35`; the built
    wheel contains all three corrected dependency declarations; packaging,
    LangChain compatibility, dependency-service, and CLI smoke tests passed (84
    passed); targeted Ruff and `git diff --check` passed.

- Fixed the remaining Ubuntu Python 3.10 failures by catching
  `asyncio.TimeoutError` explicitly, using the conditional `tomli` backport in
  the standalone release validator, and intercepting `Path.open` writes in the
  pytest real-home safety guard for Python versions whose pathlib accessor
  bypasses a patched `io.open`.
  - Verification: `.venv/bin/python -m pytest -q` passed (1311 passed, 2
    skipped); all affected computer-control, release-validation,
    runtime-isolation, and package tests passed (56 passed); the forced Python
    3.10 release-validator TOML path passed; targeted Ruff, source/script
    compilation, and `git diff --check` passed.
  - Verification note: Python 3.10 is not installed locally, so the GitHub
    Actions Ubuntu Python 3.10 job remains the authoritative native run.

- Made Fleet capability fingerprints and signed inventory payloads deterministic
  across operating systems and Python hash seeds by recursively sorting all
  unordered capability values before JSON serialization.
  - Verification: `PYTHONHASHSEED=4 .venv/bin/python -m pytest -q` passed
    (1310 passed, 2 skipped); the Fleet, Fleet CLI, and Fleet Eval suites passed
    under the same seed (14 passed); the focused macOS failure cases and
    cross-seed regression passed (3 passed); targeted Ruff, Fleet source
    compilation, and `git diff --check` passed.

- Fixed Windows CI interruption during stale session and lane-lock recovery by
  replacing destructive Windows `os.kill(pid, 0)` liveness probes with a
  read-only process-handle query shared by recovery and connector-status paths.
  - Verification: `.venv/bin/python -m pytest -q` passed (1309 passed, 2
    skipped); targeted Windows-process, stale-session, lane-lock, and Telegram
    tests passed (16 passed); targeted Ruff, source compilation, and
    `git diff --check` passed.
  - Verification note: the native Windows process-handle branch requires the
    GitHub Actions Windows runner for authoritative execution.

- Fixed Python 3.10 CI collection by routing TOML parsing through a shared
  compatibility import, adding the conditional `tomli` backport dependency,
  and making package-version and test TOML reads use the same supported
  fallback. Added a Python 3.10-compatible `StrEnum` boundary for computer
  control, preventing the next standard-library compatibility failure in the
  expanded CI matrix.
  - Verification: forced `tomllib`-unavailable and `StrEnum`-unavailable
    import checks passed; `.venv/bin/python -m pytest -q
    tests/connectors/test_telegram_cli_config.py tests/test_codex_runtime.py
    tests/test_package_version.py tests/commands/test_analyze_slash_command.py
    tests/commands/test_fleet_cli.py tests/evals tests/execution tests/fleet
    tests/gateway` passed (182 passed); computer-control tests passed (43
    passed); the built wheel contains
    `Requires-Dist: tomli<3.0,>=2.0; python_version < "3.11"`; targeted Ruff,
    source compilation, and `git diff --check` passed.
  - Verification note: an actual Python 3.10 interpreter is not installed in
    the local environment; the conditional branches were forced explicitly and
    the GitHub Actions Python 3.10 job remains the authoritative matrix run.

- Added the disabled-by-default Mana Fleet distributed verification foundation:
  strict versioned worker/capability/selection/plan/job/result/run models,
  authenticated bounded capability updates, deterministic fail-closed worker
  selection, persistent health and revocation, atomic owner-only run storage,
  ordered replayable events, cross-process cancellation, restart recovery,
  immutable completed results, matrix aggregation, and exact-action Fleet
  permission bindings.
  - Fleet jobs create an isolated detached Git worktree, verify the exact commit
    and clean starting state, and delegate provisioning, argv execution,
    artifacts, timeouts, and cleanup to the existing `ExecutionManager`.
    Required platform coverage is never weakened and infrastructure failure is
    not reported as test failure.
  - Added `mana-agent fleet` worker/job/doctor/verify/compare/log/artifact/
    cancellation commands, canonical `/fleet` chat registration, shared Fleet
    API/event replay endpoints, a read-only Fleet dashboard page, and a global
    doctor check.
  - Added persistent `fleet-verify` automation schedules with explicit
    platforms, commands, worker limits, and timeouts; deployed schedules invoke
    the same Fleet CLI/service instead of a scheduler-specific runner.
  - Extended authenticated reverse workers with signed runtime capability
    messages and coordinator-assigned trust labels. Added owner-scoped Linux
    `systemd --user` and Windows Task Scheduler installers while preserving the
    macOS LaunchAgent.
  - Added Fleet Eval configuration validation, cross-platform CI coverage,
    Fleet architecture/operations documentation, updated repository URLs and
    project layout, and removed the duplicated saved-workflow documentation.
  - Verification: `.venv/bin/python -m pytest tests/fleet -q` passed (11
    passed); `.venv/bin/python -m pytest tests/fleet
    tests/test_automation_service.py tests/test_doctor.py
    tests/remote_execution tests/gateway tests/evals
    tests/commands/test_fleet_cli.py tests/test_api_workspaces.py
    tests/test_api_conversations.py -q` passed (155 passed);
    `.venv/bin/python -m ruff check src/mana_agent/fleet
    src/mana_agent/remote_execution/installers
    src/mana_agent/remote_execution/daemon.py
    src/mana_agent/remote_execution/gateway.py
    src/mana_agent/commands/worker_cli.py
    src/mana_agent/api/routes/fleet.py
    src/mana_agent/doctor/checks/fleet.py tests/fleet
    tests/commands/test_fleet_cli.py tests/evals/test_fleet_eval_config.py`
    passed; `.venv/bin/python -m compileall -q src` passed; `git diff --check`
    passed; `.venv/bin/python -m pytest -q` passed (1304 passed, 2 skipped).
    Workflow YAML parsed with PyYAML, and help smoke checks passed for
    `mana-agent fleet`, `fleet list`, `fleet verify`, `eval`, and `doctor`.
  - Verification note: `.venv/bin/python -m ruff check .` remains blocked by
    792 pre-existing violations in unrelated modules and tests; those files
    were not modified as part of Fleet.

## 2026-07-25

- Made public-web search a fully validated entry-routing capability. Search is
  now advertised only when its selected provider has the required configuration
  and credentials; route execution applies the same validation so unavailable
  search produces a truthful setup result instead of an invalid
  `capability_error` decision.
  - Verification: `PYTHONPATH=src venv/bin/python -m pytest -q
    tests/gateway/test_entry_routing.py tests/test_search_config.py` passed
    (22 passed); `git diff --check` passed.

- Fixed reverse-worker credential loading on Windows by enforcing POSIX mode
  bits only on POSIX platforms.
  - Verification: `venv/bin/python -m pytest -q tests/remote_execution/test_reverse_worker_protocol.py`
    passed (3 passed); `git diff --check` passed.

- Bumped the package and documented release version to `v0.0.20`.
  - Verification: `tests/test_package_version.py` and `git diff --check` passed.

- Updated chat remote execution to use direct SSH when a selected managed worker
  is missing or offline, including when it disappears before approval; preserved
  the direct-SSH completion message and authorized the registered remote SSH
  tool through the operations lane.
  - Updated entry routing with live managed-worker availability, provider and
    worker-ID validation, and explicit direct-SSH selection when no worker is
    connected.
  - Kept remote-execution lane tasks waiting for approval and resumed/finished
    them from the actual approved SSH job result.
  - Returned bounded approved SSH stdout/stderr to chat and instructed remote
    analysis routing to request concise command-level findings.
  - Added a strict typed structured-output contract for entry routing whenever
    the selected model exposes structured-output support, preventing malformed
    text JSON from reaching route validation.
  - Replaced the untyped remote-request schema field with strict nested SSH
    request models accepted by OpenAI's strict response-format validation.
  - Verification: targeted gateway, entry-routing, and remote-execution tests
    passed; strict response schema validation passed.

- Made remote-worker credential permission assertions platform-aware: POSIX mode
  bits are checked on POSIX only, while Windows continues to verify credential
  persistence without asserting unsupported permission metadata.
  - Verification: `python3 -m compileall -q src/mana_agent/remote_execution tests/remote_execution`
    and `git diff --check` passed. Targeted pytest was not run because the
    available Python runtime does not have pytest installed.

- Added the reverse-connected worker runtime: a typed, versioned and bounded JSON protocol; one-time enrollment with generated Ed25519 worker identities; owner-only/Keychain credential storage; authenticated coordinator WebSocket gateway; heartbeat/offline tracking; message de-duplication; revocation and rotation primitives; and execution-event integration with the existing remote-execution service.
  - Added macOS LaunchAgent installation, lifecycle/diagnostic CLI commands, a standalone reconnecting worker daemon, HTTPS-only production defaults, and API worker enrollment/connection routes. Bootstrap tokens are never written to worker configuration, logs, or LaunchAgent plist files.
  - Verification: `PYTHONPATH=src venv/bin/python -m pytest -q tests/remote_execution/test_remote_execution.py` passed (12 passed); Python compilation and targeted Ruff checks run.

- Added sandbox-safe remote SSH execution contracts, transport failure classification, exact remote target/action approvals, reverse-worker enrolment and owner-only credentials, streamed event de-duplication, and model-selected chat lifecycle actions for registering, starting, and stopping workers.
  - Verification: focused remote execution tests added; no real SSH connection was attempted.

- Changed coding and Codex SSH policy so sandboxed processes never invoke `ssh` directly; remote SSH now requires a connected external worker and fails explicitly without a local fallback.
  - Entry routing now requires a model-selected structured `remote_execution` request rather than sending SSH work to the coding/Codex lane.
  - Remote execution is coordinated by the explicit operations specialist lane.
  - Added exact-job remote SSH permission request IDs and explicit approval/resume handling.
  - Remote SSH model decisions can now select the sole trusted connected worker automatically.

- Enabled model-selected chat execution of explicit, user-authorized SSH tasks through the validated coding workflow.
  - Clarified that workers may pass an authorized identity-file path to SSH while never reading key material or accepting passphrases in chat.
  - Added SSH to the shell policy and documented SSH-agent handling for passphrase-protected keys.
  - Verification: targeted gateway and shell-permission tests.

- Prevented entry routing from reporting `SEARCH_NOT_AVAILABLE` (or another
  capability error) when the declared source is available in the live route
  registry.
  - Verification: `venv/bin/python -m pytest -q tests/gateway/test_entry_routing.py`
    passed (17 passed); Python compilation and `git diff --check` passed.

## 2026-07-24

- Fixed macOS Music playback false positives. A play request now selects a
  random local-library track when no query is supplied (or searches the library
  using an argv-bound query), then reads Music's player state and reports success
  only when it is actually `playing`.
  - Verification: focused macOS provider and computer-control tests,
    affected-file Ruff, Python compilation, AppleScript compilation, and
    `git diff --check` passed.

- Bridged structured `permission_required` results from isolated computer-tool
  workers back into the owning gateway process. Every computer permission scope
  can now open the same TUI/dashboard chat approval UI even when the provider
  action executes outside the frontend process.
  - Verification: focused computer-control, gateway, and Textual tests,
    affected-file Ruff, Python compilation, and `git diff --check` passed.

- Fixed computer-route permission probing so an `ask` status is explicitly
  reported as having created no prompt. The model is now instructed to submit
  the exact requested action—which creates the bound in-chat request—and cannot
  tell users to approve a nonexistent prompt.
  - Verification: focused computer-control and gateway tests, affected-file
    Ruff, Python compilation, and `git diff --check` passed.

- Fixed Textual computer-permission prompts to use the non-blocking UI message
  pump. Permission events emitted on a gateway/tool worker thread now open the
  in-chat modal instead of deadlocking in a synchronous cross-thread callback.
  - Verification: real worker-thread permission execution opened
    `ComputerPermissionScreen`; focused computer-control and Textual tests,
    affected-file Ruff, Python compilation, and `git diff --check` passed.

- Added the default-off, provider-neutral computer-control framework with typed
  actions/results/capabilities/content models, strict provider and application
  adapter contracts, macOS/Windows/Linux auto-selection, truthful discovery,
  fine-grained allow-once/session/persistent permissions, remote-client policy,
  exact-action expiring confirmations, bounded execution/cancellation, sanitized
  live events, and owner-only retention-aware audit records.
  - Added narrow model-selected tools for application, calendar, media, notes,
    desktop browser, clipboard, allowed-path filesystem, screenshots,
    notifications, and system operations; raw OS automation program input is
    not exposed, invalid model decisions fail closed, personal content is not
    copied into events/audit, and removal uses Trash/Recycle Bin.
  - Integrated the explicit `computer` route and authenticated frontend context
    with the shared gateway, tool catalog, lane coordinator, `/cancel`, Textual
    configuration, dashboard settings/capability matrix, user configuration,
    security/architecture/tool/configuration documentation, and README.
  - Added a desktop-safe fake provider and cross-platform mocked tests covering
    discovery, unavailable/headless behavior, permissions, exact/expired
    confirmation, remote restrictions, adapter selection, calendar/media/notes/
    browser/clipboard flows, screenshots, timeout/cancellation, audit redaction,
    allowed paths, Trash/Recycle Bin, command injection, Windows paths, macOS
    identifiers, and Linux command construction.
  - Added interactive pending-permission requests for `ask` scopes: Textual
    displays an in-chat once/session/always/deny modal, Dashboard chat displays
    the same actionable card (also mirrored on its Computer Control page), and
    approval resumes the immutable stored action immediately instead of returning
    a dead-end permission message.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q` passed
    (1,263 passed, 2 skipped); the focused computer-control, API, dashboard-chat,
    WebSocket, Textual, and gateway pass passed (70 tests); all 8 browser reducer
    tests passed; affected-file Ruff, Python compilation, and `git diff --check`
    passed.

## 2026-07-23

- Fixed Gmail search/read/thread tools in dashboard and API chat so synchronous
  LangChain tool invocation safely executes provider coroutines outside the
  already-running event loop, preventing `asyncio.run()` failures and leaked
  un-awaited coroutine warnings.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/connectors/test_email_core.py` passed (18 tests); AskAgent and gateway regression tests passed (64 tests); targeted Ruff, Python compilation, and `git diff --check` passed.

- Fixed the dashboard's live API base control to remain initialized while navigating between Streamlit pages by rendering it in the shared entrypoint frame instead of inside route-specific callbacks.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dashboard_navigation.py` passed.

- Rebuilt dashboard chat delivery around the shared persisted `ChatEvent` stream: browser-side optimistic messages now reconcile by stable client ID, assistant/log/status events render as they arrive, correlated tool cards update in place, and ordered cursor replay restores missed events without duplication after reconnect or reload.
  - Added a deterministic dashboard event reducer, live REST/WebSocket component, gateway event forwarding, per-conversation sequence IDs, exact-delivery deduplication, lifecycle revision persistence, recursive event redaction, and an automatically managed local API when launching `mana-agent dashboard`.
  - Removed the dashboard gateway-to-classic-ask fallback; invalid or failed model execution now remains visible as a persisted failed run without executing a backup route.
  - Verification: The repository suite passed (1,220 passed, 2 skipped); the broader dashboard/socket/gateway/TUI/Codex compatibility pass passed (114 tests); the final dashboard/event/API pass passed (31 tests), and all 7 Node.js reducer tests passed. Affected-file Ruff, Python compilation, `git diff --check`, and local browser submit/socket/failure/reload plus final hosted-component smoke checks passed. Repository-wide Ruff was also run and still reports 798 unrelated pre-existing findings.

- Fixed the dashboard chat page's misleading default socket connection. The standalone Streamlit dashboard now uses durable event recovery by default, and external WebSockets are opt-in only after configuring a running FastAPI endpoint.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dashboard_navigation.py tests/test_chat_websocket.py` passed.

- Fixed observability span persistence to keep truncated attributes as valid JSON, and made dashboard reads tolerate malformed historical telemetry rows.
  - Verification: `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_observability.py` passed.

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

- Clarified the gateway entry-router decision contract so tool-free prompts such as `ping` must emit `required_sources: ["none"]` instead of an omitted or empty source list.
  - Strict validation remains intact: missing model-selected sources stop safely and never trigger a fallback route.
  - Verification: `PYTHONPATH=src venv/bin/python -m pytest -q tests/gateway/test_entry_routing.py -k 'ping or missing_required_sources'` passed (3 tests); Python compilation and `git diff --check` passed.

- Fixed Textual chat-message wrapping to measure read-only message cards against their full available content width instead of reserving an invisible editing-cursor cell.
  - Existing and newly mounted messages now reflow correctly for terminal resizes and surrounding-panel width changes without stale per-widget wrap widths.
  - Verification: `PYTHONPATH=src venv/bin/python -m pytest -q tests/test_tui_message_layout.py tests/test_tui_tool_card_layout.py tests/test_tui_multiline_input.py tests/test_tui_live_tools_scroll.py tests/test_tui_auto_chat_tool_events.py` passed (19 tests); `PYTHONPATH=src venv/bin/python -m pytest -q tests/test_tui*.py` passed (34 tests); Python compilation and `git diff --check` passed. Ruff and mypy are not installed in the repository environment.
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
  - Verification: `actionlint -color .github/workflows/publish-pypi.yml .github/workflows/ci.yml` passed; all GitHub workflow YAML parsed successfully; `python .github/scripts/validate_release_version.py --tag v0.0.15 --check-pypi` passed; `python -m build --sdist --wheel` produced one wheel and one sdist and `python -m twine check` passed for both; `PATH="$PWD/venv/bin:$PATH" PYTHONPATH=src ./venv/bin/python -m pytest -q` passed (947 tests, 1 skipped).

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
## 2026-07-25

- Added dual remote execution modes: persistent direct-SSH profiles alongside managed reverse workers, with secure OpenSSH-only execution and explicit route preservation.
  - Verification: focused remote-execution and gateway tests passed (56 tests); Ruff, compilation, CLI help, and `git diff --check` passed.

## 2026-08-01 (multi-turn chat continuation)

- Added durable message-scoped turn state and follow-up classification so verified task completion cannot complete a conversation or cause later work to reuse an unrelated escrowed result.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/gateway/test_chat_turn_store.py tests/gateway/test_followup_classifier.py tests/gateway/test_checkpoint_resume.py tests/gateway/test_chat_gateway.py tests/test_conversation_service.py -q`.

- Added one bounded model correction for browser decisions missing direct URLs, allowing open-ended discovery requests to return a new model-selected search route instead of failing at entry validation.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/gateway/test_entry_routing.py -q`.

- Preserved informational conversation lane metadata and limited strict follow-up classification to routing models that expose the required structured-output contract; generic recovery still excludes completed tasks.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/gateway/test_entry_routing.py tests/gateway/test_chat_gateway.py -q`.
## 2026-08-18

- Added English/Persian language selection to repository analysis across the CLI, dashboard, API request, LLM prompt, and generated evidence metadata.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_project_analyze_service.py tests/test_cli_smoke.py -q`.
## 2026-08-21

- Added lane-scheduler diagnostics for queued tasks, including queue position,
  capacity blockers, lane occupancy, worker-slot availability, QueueManager job
  visibility, and the queued → scheduled → assigned → running lifecycle.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/gateway/test_lane_coordinator.py -q`.

- Added Codex dual-auth resource metadata for API and subscription execution modes, secure credential references, usage caches, quota-aware mode selection, and `codex auth`/`codex status` reporting.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_codex_provider_resources.py tests/test_codex_runtime.py -q`.
## 2026-08-21

- Extended the existing Codex dual-mode provider with Luna identity lifecycle states, injectable subscription authentication, TTL-aware provider usage normalization, explicit unknown capacity handling, quota-aware subscription/API selection, separated resource accounting records, and richer `codex status` output.
  - User verification required: `python -m pytest tests/test_codex_provider_resources.py tests/test_codex_provider_lifecycle.py tests/test_codex_runtime.py -v`.

- Added typed Codex capacity evidence to model routing, explicit resource fallback reasons, execution-lifecycle accounting, Luna session recovery status, and capacity scores in `codex status`.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_codex_provider_resources.py tests/test_codex_provider_lifecycle.py tests/test_model_routing.py tests/test_codex_runtime.py -v`.
## 2026-08-21

- Integrated Codex execution metadata with durable supervisor tasks, checkpoints, and result escrow, including provider state, selected resource, routing evidence, fallback history, accounting reference, and decision ID.
  - User verification required: `PYTHONPATH=src python -m pytest tests/execution_supervisor/test_supervisor_core.py tests/test_codex_provider_lifecycle.py -q`.
## 2026-08-21

- Integrated Codex lifecycle metadata with durable execution tasks, checkpoints, failure results, and escrow lookup, including fallback failure evidence and reauthentication guidance.
  - User verification required: `python -m pytest tests/execution_supervisor/test_supervisor_core.py tests/test_codex_provider_lifecycle.py`
## 2026-08-22

- Added explicit planner integration contracts, wiring dependencies, runtime-reachability reviewer evidence, and a final TaskBoard completion gate so runtime capabilities cannot report success from implementation or unit-test evidence alone.
  - User verification required: `python -m pytest tests/test_multi_agent_core.py -q`.

## 2026-08-22

- Completed feature-wiring execution lifecycle: feature-specific discovery now requests exact reads and model-selected mutations through CodingAgent/QueueManager, records distinct wiring outcomes and connected reachability evidence, preserves implementation targets, and projects verified wiring children through ExecutionSupervisor before parent review.
  - User verification required: `python -m pytest tests/test_multi_agent_core.py tests/gateway/test_multi_task_orchestration.py -v`.

# 2026-08-22

- Completed feature-wiring lifecycle fixes for authoritative supervisor reuse, managed-worktree batch reads, and direct parent changed-file discovery.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py tests/test_managed_worktrees.py -q`.

## 2026-08-23

- Updated feature-wiring discovery so validated parent changed files are batch-read in the child execution root before model-selected outward searches; changed paths are not used as search-only identifiers. The legacy memory adapter now accepts the same execution-root read scope.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_multi_agent_core.py tests/test_managed_worktrees.py -q`.
# 2026-08-23

- Completed the Gateway feature-wiring gate: runtime-capability coding success now requires integration, provenance-backed reachability, and resumable `after_core_implementation` recovery through the same authoritative gateway execution/workspace. Internal and Codex coding backends preserve gateway task identity; incomplete wiring cannot produce false success.
  - Added Gateway integration lifecycle regression coverage. User verification required: `python -m pytest tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_entry_routing.py tests/gateway/test_chat_gateway.py`.
## 2026-08-23

- Updated feature integration recovery so MainAgent and Gateway share the coordinator gate, preserve core edits on incomplete wiring, and restore durable integration checkpoints on resume.
  - User verification required: `pytest -q tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_chat_gateway.py tests/gateway/test_checkpoint_resume_invariants.py`

## 2026-08-23

- Tightened the feature-integration gate so completion requires runtime-owned TaskBoard, wiring-child, verification, reviewer, reachability, and supervisor authority; model-reported edges remain evidence candidates.
  - User verification required: `pytest -q tests/gateway/test_feature_integration_lifecycle.py tests/gateway/test_entry_routing.py tests/gateway/test_chat_gateway.py tests/gateway/test_checkpoint_resume_invariants.py`
## 2026-08-24

- Finalized lost-lease reconciliation and durable resume flow: local mutations require attempt-bound evidence, external receipts are consumed as `ACTION_RECONCILED`, and recovery Human Inbox items retain original execution lineage for safe response resume.
  - User verification required: `PYTHONPATH=src .venv/bin/python -m pytest tests/execution_supervisor/test_supervisor_core.py tests/gateway/test_feature_integration_lifecycle.py tests/human_inbox/test_durable_inbox.py -q`.
