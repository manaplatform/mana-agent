# Evidence-based model routing

Mana-Agent selects inference models through the existing `mana_agent.model_routing.ModelRouter`. `GatewayRoutingAuthority` owns that router for CLI, TUI, API, dashboard, protocol, automation, and Codex execution. It adds task/session/workspace identity, persists every request/decision pair to `~/.mana/routing/decisions.jsonl`, and emits routing lifecycle events. Logical model levels and explicitly configured role models remain profile hints only. A missing, invalid, or unpersistable decision stops execution; there is no default-model bypass.

Spirit is resolved before this router runs and is bound to the selected
provider/model afterward. Curious, bold, and calm are identity, not routing
signals: they are not added to scoring, cost, risk, latency, or capability
evaluation. Model changes therefore do not change Mana identity. Durable
routing records store `spirit_id` and `spirit_version` only.

## Request and profiles

A `RoutingRequest` carries task/parent/session/workspace/repository IDs, lane/role, task description and category, estimated complexity and risk, cached `RepositoryMetadata`, required tools/capabilities/context/output, latency, task-tree depth, health/evidence signals, budgets, and the coordinating model's explicit decomposition or competition request. Repository inspection uses a Git-index/config fingerprint and reuses its bounded inventory until relevant metadata changes. Languages, frameworks, build systems, repository/test size, changed-file scope, and sensitive changed areas contribute to routing demand.

Each `ModelProfile` identifies the provider/model and records supported roles, tools, reasoning settings, context and output limits, tokenizer, monetary or logical cost, reliability, language preferences, category benchmarks, availability, and patch/structured-output/tool/verification capabilities. `MANA_MODEL_PROFILES` accepts explicit profiles. Existing `MODEL_LEVEL_*` settings identify configured candidates. Capability and limit evidence comes from the native registry. The final candidate is estimated through the shared [model-aware token accounting](model-token-accounting.md) service.

## Main model selection

Deterministic validation filters the configured candidates by role, tool and
capability evidence, context, availability, circuit state, latency, and spend.
The explicitly configured Main model receives this set with the task constraints
and chooses a candidate, mode, and any verifier. The decision is schema validated
and checked against the offered IDs, authorization, isolation, and total spend.
A malformed result, an unavailable candidate, or expired evidence stops the
operation. The router records the Main model's confidence and selection reason.

Routing modes are `single`, `single_with_verification`, `multi_agent`,
`parallel_candidates`, and `multi_agent_with_parallel_candidates`. The request
lists the modes admitted by policy and available candidates. Role assignment at
startup uses a schema with the selected model ID, reason, and confidence. When a parallel
mode is available, the schema reserves that field for a parallel selection and
requires an empty list for the other modes. A verifier must be independent of the
selected implementation candidates. A provider circuit
breaker still prevents repeated calls to an unhealthy provider. The selected
configuration and budgets are recorded with its request, decision, and task IDs.

## Budgets

The router enforces per-task token/cost limits, remaining session cost, competition cost, verification cost, retry cost, and a verification reserve ratio. Implementation candidates are compared against the spend remaining after the reserve. If none fit, routing stops with the rejected constraints. A controlled override is possible only when the validated task policy explicitly enables it.

The context-cost governor contributes cumulative remaining token and cost limits
to routing only in `enforce` mode. `observe` records usage without changing
execution, while `soft` may compact context but does not reject model candidates
by replacing their configured routing budgets with cumulative remaining values.

## Candidate competition

Competition is permitted only when the main coordinating model requests it and gateway policy confirms at least two compatible models, an independent verifier, isolated execution, no ownership conflict, concurrency, latency, and total candidate-plus-verification budget. Difficulty alone does not enable competition. `CandidateCompetition` requires distinct isolated worktrees or patch roots and rejects the active checkout, duplicate roots, candidates without a diff and executed checks, and incomplete verifier judgments.

The verifier receives normalized diffs, check outcomes, diagnostics, changed files, patch size, cost, and latency. It must score correctness, test results, regression risk, security, scope discipline, maintainability, repository conventions, patch size, verification completeness, and cost/latency. The same exact author configuration is avoided when a qualified independent verifier fits the context and verification budget. Only the winner is promoted; losing workspaces are cleaned. Any execution or judgment failure cleans every created candidate and promotes nothing.

## Outcome evidence and diagnostics

`RoutingHistory` is an interface with in-memory and append-only JSONL implementations. Records contain model/configuration metadata, task/repository categories, score/reason, usage/cost/latency, failures/retries, verification and acceptance, and competition outcome—never prompts, source code, credentials, or raw secrets. Evidence retention is configurable.

Run `mana-agent doctor --only routing/models --json` to inspect enforcement, candidate validity and availability, static/bypass diagnostics, circuit state, decision/evidence persistence, budgets, task/concurrency limits, independent-verifier availability, and managed-worktree competition support.

## Live task control

The existing gateway lane coordinator is the authoritative task-control service. Each record carries its routing decision, provider/model, parent/children, ownership/locks, progress, tool activity, verification, budgets, evidence, cancellation, errors, and result. Validated states cover creation, routing, queuing, running, waiting, blocking, pausing, cancellation, verification, winner selection, application, and terminal outcomes. Restart recovery never automatically repeats completed work; interrupted write tasks require revalidation.

CLI and TUI share the gateway commands `/route`, `/route explain`, `/tasks`, `/task <id>`, `/task cancel|pause|resume|retry|replan [id]`, `/budget`, `/candidates`, and `/models health`. These commands mutate task state only through validated gateway operations. Omitting the task id auto-selects only when exactly one recoverable or active candidate exists; reserved verbs such as `create` or `Execute` are never treated as task ids. Normal chat turns auto-select resume/retry/replan or create a new durable task without requiring `/tasks`.

## Configuration

Safe defaults enforce gateway routing and keep both multi-agent and parallel execution disabled. `MANA_ROUTING_MULTI_AGENT_ENABLED` and `MANA_ROUTING_PARALLEL_ENABLED` opt into those capabilities; they do not force their use. `MANA_ROUTING_MAX_CANDIDATES`, `MANA_ROUTING_MAX_TASK_TREE_DEPTH`, `MANA_ROUTING_MAX_CONCURRENT_TASKS`, task/stall/cancellation timeouts, retention, and detail level bound orchestration. Existing installations therefore remain single-model unless explicitly configured and approved per task.
