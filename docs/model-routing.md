# Evidence-based model routing

Mana-Agent selects inference models through the existing `mana_agent.model_routing.ModelRouter`. `GatewayRoutingAuthority` owns that router for CLI, TUI, API, dashboard, protocol, automation, and Codex execution. It adds task/session/workspace identity, persists every request/decision pair to `~/.mana/routing/decisions.jsonl`, and emits routing lifecycle events. Logical model levels and explicitly configured role models remain profile hints only. A missing, invalid, or unpersistable decision stops execution; there is no default-model bypass.

Spirit is resolved before this router runs and is bound to the selected
provider/model afterward. Curious, bold, and calm are identity, not routing
signals: they are not added to scoring, cost, risk, latency, or capability
evaluation. Model changes therefore do not change Mana identity. Durable
routing records store `spirit_id` and `spirit_version` only.

## Request and profiles

A `RoutingRequest` carries task/parent/session/workspace/repository IDs, lane/role, task description and category, estimated complexity and risk, cached `RepositoryMetadata`, required tools/capabilities/context/output, latency, task-tree depth, health/evidence signals, budgets, and the coordinating model's explicit decomposition or competition request. Repository inspection uses a Git-index/config fingerprint and reuses its bounded inventory until relevant metadata changes. Languages, frameworks, build systems, repository/test size, changed-file scope, and sensitive changed areas contribute to routing demand.

Each `ModelProfile` identifies the provider/model and records supported roles, tools, reasoning settings, context and output limits, tokenizer, monetary or logical cost, reliability, language preferences, category benchmarks, availability, and patch/structured-output/tool/verification capabilities. `MANA_MODEL_PROFILES` accepts explicit profiles. Existing `MODEL_LEVEL_*` values are migrated into profiles with cost, reliability, latency, and benchmark hints; their former role mapping is not a final selection rule. The final candidate is estimated through the shared [model-aware token accounting](model-token-accounting.md) service.

## Deterministic score

Candidates are rejected before scoring when they cannot meet role, tool, capability, context, availability, circuit-breaker, latency, token, cost, or verification-reserve constraints. For remaining candidates, the score combines:

```text
capability fit
+ demand-adjusted profile/benchmark quality
+ demand-adjusted historical success, verification, tool, and structured-output reliability
+ repository-language fit
+ inverse estimated cost (weighted more heavily for low-demand work)
+ latency fit
- bounded, exponentially decaying recent-failure penalty
```

The configured weights are stable, candidates are sorted by score, then cost, then provider/model ID, so identical requests and evidence produce identical decisions. Token estimates combine the request estimate with historical usage for the same task category. Provider/model errors, authentication/rate-limit failures, bad tool calls, unsupported parameters, malformed output, verification failures, and timeouts lower the score. Repeated failures inside the configured window open a provider circuit breaker.

The returned `RoutingDecision` includes request/decision/task identity, selected provider/model/configuration, model context/output capability, expected call count, routing mode, score, estimate confidence/components/assumptions, token/cost/latency estimates, verification level, reasons, rejected candidates, agent/candidate permissions, limits/deadline, verifier selection, and effective budgets. Modes are `single`, `single_with_verification`, `multi_agent`, `parallel_candidates`, and `multi_agent_with_parallel_candidates`. `single` is the default. `decision.concise()` is safe for normal diagnostics; credentials are excluded from persistence.

## Budgets

The router enforces per-task token/cost limits, remaining session cost, competition cost, verification cost, retry cost, and a verification reserve ratio. Implementation candidates are compared against the spend remaining after the reserve. If none fit, routing stops with the rejected constraints. A controlled override is possible only when the validated task policy explicitly enables it.

The context-cost governor contributes cumulative remaining token and cost limits
to routing only in `enforce` mode. `observe` records usage without changing
execution, while `soft` may compact context but does not reject model candidates
by replacing their configured routing budgets with cumulative remaining values.

## Candidate competition

Competition is permitted only when the main coordinating model requests it and gateway policy confirms sufficient evidence, at least two compatible models, an independent verifier, isolated execution, no ownership conflict, concurrency, latency, and total candidate-plus-verification budget. Difficulty alone does not enable competition. The evidence score includes demand, initial-model uncertainty, similar failures, historical output variance and competition benefit, candidate diversity, and multiple plausible strategies. `CandidateCompetition` requires distinct isolated worktrees or patch roots and rejects the active checkout, duplicate roots, candidates without a diff and executed checks, and incomplete verifier judgments.

The verifier receives normalized diffs, check outcomes, diagnostics, changed files, patch size, cost, and latency. It must score correctness, test results, regression risk, security, scope discipline, maintainability, repository conventions, patch size, verification completeness, and cost/latency. The same exact author configuration is avoided when a qualified independent verifier fits the context and verification budget. Only the winner is promoted; losing workspaces are cleaned. Any execution or judgment failure cleans every created candidate and promotes nothing.

## Outcome evidence and diagnostics

`RoutingHistory` is an interface with in-memory and append-only JSONL implementations. Records contain model/configuration metadata, task/repository categories, score/reason, usage/cost/latency, failures/retries, verification and acceptance, and competition outcome—never prompts, source code, credentials, or raw secrets. Evidence retention is configurable.

Run `mana-agent doctor --only routing/models --json` to inspect enforcement, candidate validity and availability, static/bypass diagnostics, circuit state, decision/evidence persistence, budgets, task/concurrency limits, independent-verifier availability, and managed-worktree competition support.

## Live task control

The existing gateway lane coordinator is the authoritative task-control service. Each record carries its routing decision, provider/model, parent/children, ownership/locks, progress, tool activity, verification, budgets, evidence, cancellation, errors, and result. Validated states cover creation, routing, queuing, running, waiting, blocking, pausing, cancellation, verification, winner selection, application, and terminal outcomes. Restart recovery never automatically repeats completed work; interrupted write tasks require revalidation.

CLI and TUI share the gateway commands `/route`, `/route explain`, `/tasks`, `/task <id>`, `/task cancel|pause|resume|retry|replan [id]`, `/budget`, `/candidates`, and `/models health`. These commands mutate task state only through validated gateway operations. Omitting the task id auto-selects only when exactly one recoverable or active candidate exists; reserved verbs such as `create` or `Execute` are never treated as task ids. Normal chat turns auto-select resume/retry/replan or create a new durable task without requiring `/tasks`.

## Configuration

Safe defaults enforce gateway routing and keep both multi-agent and parallel execution disabled. `MANA_ROUTING_MULTI_AGENT_ENABLED` and `MANA_ROUTING_PARALLEL_ENABLED` opt into those capabilities; they do not force their use. `MANA_ROUTING_MIN_PARALLEL_EVIDENCE`, `MANA_ROUTING_MAX_CANDIDATES`, `MANA_ROUTING_MAX_TASK_TREE_DEPTH`, `MANA_ROUTING_MAX_CONCURRENT_TASKS`, task/stall/cancellation timeouts, retention, and detail level bound orchestration. Existing installations therefore remain single-model unless explicitly configured and approved per task.
