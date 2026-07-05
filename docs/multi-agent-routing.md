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
assumptions, blockers, discussions, decisions, and verification results.

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

## Verification

VerifierAgent records verification requirements for every mutation route and
stores `VerificationResult` rows on the TaskBoard. Existing command paths still
run their concrete tests or analyze flows after the mandatory multi-agent route
has been recorded.

## CLI Behavior

- Root mode flags and menu selections record a MainAgent route before dispatching
  to the selected command.
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
