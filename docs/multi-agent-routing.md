# Multi-Agent Routing

Mana Agent routes every public command and LLM-facing request through
`mana_agent.multi_agent.MainAgent`. The old command names remain public, but the internal record starts with a
TaskBoard item, a route decision, agent assignments, and a final SummarizerAgent
summary.

## Hierarchy

```text
MainAgent
  └── HeadDecisionAgent
        ├── PlannerAgent
        ├── ResearchAgent
        ├── CodingAgent
        │     └── CodingSubAgent(s)
        ├── ToolAgent
        ├── VerifierAgent
        ├── ReviewerAgent
        └── SummarizerAgent
```

## TaskBoard

TaskBoard state is persisted in `.mana/taskboard/state.json`; append-only events
are written to `.mana/taskboard/history.jsonl`. Tasks store status, risk,
assigned agents, required capabilities, files, queue jobs, plan, evidence,
assumptions, blockers, discussions, decisions, and verification results. Compound
roots additionally persist child IDs, decomposition ID mappings, and aggregate
progress; children persist explicit TaskBoard dependencies, entry route, lane,
acceptance criteria, routing evidence, verification, artifacts, approvals, and
result summary. Older TaskBoard JSON remains loadable because these fields have
empty backward-compatible defaults.

## Communication And Decisions

Agents exchange concise structured messages through `MessageBus`. Complex,
mutation, ambiguous, or higher-risk requests open a `DecisionRoom`, where
HeadDecisionAgent records the selected route, rationale summary, risks,
assumptions, rejected options, assigned agents, and verification needs.

## Queue And Tools

CodingAgent never executes tools directly. It creates QueueManager jobs.
QueueManager schedules jobs FIFO with priority ordering, serializes write jobs
with locks, and delegates execution to ToolsManager. ToolsManager wraps the
existing repository-safe commands and blocks dangerous shell operations such as
`rm -rf /`, `.env` reads, `printenv`, `git reset --hard`, and `git clean -fd`.

## Gateway specialist lanes

The production `AgentChatGateway` is the outer resource owner for CLI, TUI, dashboard, API, connector, and automation turns. It coordinates the `artifact`, `media`, `canvas`, `coding`, `research`, `review`, `verify`, `release`, and `operations` lanes before dispatching into the existing agent/taskboard/queue runtime. There is no second orchestration entry point.

Lane contracts define ownership, handoffs, tool capabilities, model restrictions, per-lane concurrency, subagent limits, token/cost budgets, priority, repository/write requirements, lock policy, timeout, and retry policy. The coordinator also applies global, provider/model, repository-mutation, and per-session limits. Capacity-constrained work remains queued and interactive priority precedes background priority without changing task identity.

Routing budgets are consumption-aware. Before each decision, the gateway feeds
the shared ledger's remaining task-token and session-cost values into
`RoutingBudgets`. Subagents, workers, competitions, reviewers, verifiers, and
retries receive bounded child ledgers; allocations cannot exceed the parent's
remaining reservation unless the existing routing decision explicitly permits
a controlled override. The verification reserve remains governed by
`MANA_ROUTING_VERIFICATION_RESERVE_RATIO`.

Every shared model-client call accounts protected system/safety instructions
and the current request separately from history, retrieval, schemas, and tool
results. Degradation removes idle schemas and exact duplicates before
reversible result compression and old-history eviction. It never slices the
final prompt. Enforce-mode failures stop before a provider call rather than
selecting a fallback model or tool.

Active-task fingerprints include normalized intent, repository, workspace, session, target files, lane, and parent relationship. Equivalent active work attaches to the existing task. Review and verification remain distinct lane stages in the same lineage and therefore are not collapsed into their coding stage.

## Compound request orchestration

`AgentChatGateway` exposes `multi_task` as a first-class, tool-free entry route.
Its structured planner decomposes a compound goal, persists one root and its
children in the existing TaskBoard, and schedules the validated DAG with bounded
concurrency. Every child returns to the same typed entry-routing authority and
then enters its selected specialist lane; the parent never claims child tools or
capabilities. QueueManager, lane capacity, workspace/repository/file locks,
permission approvals, cancellation, budgets, and verification remain in force.

Independent read-only children may overlap. Dependencies wait for successful
prerequisites, mutations remain lock-serialized, and capability failures or
approval waits remain local to the owning child. The root aggregates structured
child status and never reports full success after partial completion. `/tasks`,
`/task`, dashboard/TUI events, WebSocket events, and traces use the same persisted
root/child identities. No second task store, scheduler, frontend path, or public
orchestration entry point is introduced.

Multi-task budget coordination is parent-envelope based: after decomposition the
root reserves capacity for orchestration plus every planned child, then expands
that envelope before each child lane reservation so siblings do not starve under
a goal-only parent budget. When a live multi-task child revises its reservation
from a real provider-call forecast (coding/Codex, media, etc.), the same parent
envelope is expanded first so mid-run growth is not rejected by parent-remaining
checks. Child preflight sizing uses model and optional lane capacity only;
depleted shared session remaining from parent planning does not hard-fail child
admission. Actual provider calls still pass through the session context
governor. Missing or insufficient parent envelope capacity fails closed as a
blocked multi-task budget decision with no fallback route.

## Verification

VerifierAgent records verification requirements for every mutation route and
stores `VerificationResult` rows on the TaskBoard. Existing command paths still
run their concrete tests or analyze flows after the mandatory multi-agent route
has been recorded.

## CLI Behavior

- Bare `mana-agent` records a MainAgent route and opens chat directly; there is
  no root application-mode menu. Explicit legacy mode flags still dispatch
  through the same route boundary.
- `mana-agent chat` records command start and each substantive user turn through MainAgent.
- `/analyze` inside chat records an analyze route before running the analyzer.
- `/plan` inside chat records a planning route before generating a plan answer.
- `mana-agent analyze` records an analyze route before generating artifacts.
- `mana-agent plan` records a planning route before rendering/saving the plan.
- `mana-agent continue` records a continuation route before resuming a run.
- `mana-agent skills init/list/show` record skill-command routes before reading
  or writing skill files.
- Coding/edit turns record a coding route with PlannerAgent, CodingAgent,
  QueueManager, ToolAgent, VerifierAgent, ReviewerAgent, and SummarizerAgent.

The live runtime now lives under `mana_agent.multi_agent.runtime`; the previous
top-level LLM runtime package path is retired.

There is no `--no-multi-agent` flag, `MANA_MULTI_AGENT=0` bypass, or config key
that disables multi-agent routing.
