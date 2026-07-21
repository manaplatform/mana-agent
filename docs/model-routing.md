# Evidence-based model routing

Mana-Agent selects inference models through `mana_agent.model_routing.ModelRouter`. Gateway construction, CLI service builders, agents, subagents, tool workers, planners, reviewers, verifiers, and retry callers reach it through `route_model` or the compatibility-preserving `resolve_model_for_role` entry point. A missing or invalid decision raises `RoutingFailure`; there is no default-model bypass.

## Request and profiles

A `RoutingRequest` carries the lane/role, task description and category, estimated complexity and risk, cached `RepositoryMetadata`, required tools/capabilities, context and response estimates, latency class, budgets, prior verification state, and candidate-competition permission. Repository inspection uses a Git-index/config fingerprint and reuses its bounded inventory until relevant metadata changes. Languages, frameworks, build systems, repository/test size, changed-file scope, and sensitive changed areas contribute to routing demand.

Each `ModelProfile` identifies the provider/model and records supported roles, tools, reasoning settings, context limit, latency class, monetary or logical cost, reliability, language preferences, category benchmarks, availability, and patch/structured-output/tool/verification capabilities. `MANA_MODEL_PROFILES` accepts explicit profiles. Existing `MODEL_LEVEL_*` values are migrated into profiles with cost, reliability, latency, and benchmark hints; their former role mapping is not a final selection rule.

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

The returned `RoutingDecision` includes the selected provider/model/configuration, role, score, confidence, token/cost/latency estimates, reasons, rejected candidates, competition state, verifier selection and independence, and effective budgets. `decision.concise()` is safe for verbose diagnostics; credentials are excluded from persisted outcome configuration.

## Budgets

The router enforces per-task token/cost limits, remaining session cost, competition cost, verification cost, retry cost, and a verification reserve ratio. Implementation candidates are compared against the spend remaining after the reserve. If none fit, routing stops with the rejected constraints. A controlled override is possible only when the validated task policy explicitly enables it.

## Candidate competition

Competition is permitted only when the routing request allows it, at least two compatible models exist, complexity/risk or previous failure crosses policy thresholds, latency permits it, and the competition budget fits. `CandidateCompetition` requires a provider executor to create distinct isolated worktrees or patch roots. It rejects the active checkout, duplicate roots, candidates without a diff and executed checks, and incomplete verifier judgments.

The verifier receives normalized diffs, check outcomes, diagnostics, changed files, patch size, cost, and latency. It must score correctness, test results, regression risk, security, scope discipline, maintainability, repository conventions, patch size, verification completeness, and cost/latency. The same exact author configuration is avoided when a qualified independent verifier fits the context and verification budget. Only the winner is promoted; losing workspaces are cleaned. Any execution or judgment failure cleans every created candidate and promotes nothing.

## Outcome evidence and diagnostics

`RoutingHistory` is an interface with in-memory and append-only JSONL implementations. Records contain model/configuration metadata, task/repository categories, score/reason, usage/cost/latency, failures/retries, verification and acceptance, and competition outcome—never prompts, source code, credentials, or raw secrets. Evidence retention is configurable.

Run `mana-agent doctor --only routing/models --json` to inspect candidate validity and availability, missing monetary pricing, logical cost hints, circuit state, evidence-store health, budgets, independent-verifier availability, and managed-worktree competition support.
