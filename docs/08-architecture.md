# Architecture

`mana-agent` is a Python CLI + service stack that performs repository analysis and
LLM-driven “agentic” workflows. The architecture is organized around three
axes:

1. **Evidence gathering** (search/read/indexing)
2. **Decision + orchestration** (flow + work queue + tool managers/workers)
3. **Artifacts + presentation** (generated files, reports, HTML/console UI)

The most recent architecture centers the *live* coding-work execution loop:
the planner produces a sequence of gated tool jobs, and a queue runner executes
them while a sniffer steers follow-up reads/edits/verification.

## Major Components

### CLI / command surface

- **`src/mana_agent/commands/`**: command entry points and “chat command” helpers.
  For example, `/analyze` format parsing and the numbered menu are centralized in
  `src/mana_agent/commands/analyze_formats.py` as the single source of truth for
  artifact filenames under `.mana/`.
  See: `src/mana_agent/commands/analyze_formats.py:1-174`.

### Prompting and flow context

- **`src/mana_agent/agent/flow.py`** builds an `AgentFlow` (goal + phase + verification
  plan) that drives which parts of the workflow run.
  See: `src/mana_agent/agent/flow.py:1-109`.

- **`src/mana_agent/prompting/builder.py`** assembles the stable prompt (rules,
  tool policy, skill index hints, verification rules) and the ephemeral prompt
  (per-call task context, retrieved snippets, recent summaries).
  See: `src/mana_agent/prompting/builder.py:1-353`.

### Coding orchestration (work queue + decision lifecycle)

- **`src/mana_agent/multi_agent/runtime/agent_work_queue.py`** is the orchestration spine.
  It defines:
  - `WorkItem`: a unit of work (tool call + gate + priority + dependencies +
    fingerprint)
  - `AgentWorkQueue`: a dependency-aware, fingerprint-deduplicated queue
  - `EventBus` + `TaskBoard`: status transitions and a live renderable progress view
  - `WorkQueueRunner`: `claim -> execute -> classify -> broadcast -> sniff`
  - `JobSniffer`: a hook for the coding-agent to emit follow-up jobs
  See: `src/mana_agent/multi_agent/runtime/agent_work_queue.py:1-1969`.

Key lifecycle details from the implementation:

1. **Queue submission**: jobs are enqueued via `AgentWorkQueue.submit()`.
   Fingerprints prevent redundant work for idempotent kinds (discover/search/read).
2. **Claiming**: the runner claims the highest-priority job whose dependencies are
   satisfied.
3. **Execution + classification**: execution is done through an injected
   `execute(item) -> WorkResult`. Runner classifies results into `done/failed/retry`.
4. **Sniffed follow-ups**: when a job finishes successfully, the sniffer may enqueue
   additional jobs.
5. **Mutation phase + verification gate**: edit jobs compile/execute registered
   mutation tools, then verification is summarized from the execution trace.
   If required deliverables are missing or verification fails, the run is blocked
   without fabricating content.

### Tool worker and tool manager processes

This architecture assumes tool execution is delegated to a “worker” client.
The queue runner itself stays deterministic by injecting the worker executor:

- `QueueManager` owns a `worker_client` and builds a worker executor using
  `make_worker_executor()` (imported in `QueueManager.run`).
  See: `src/mana_agent/multi_agent/runtime/agent_work_queue.py` around `QueueManager.run`.

### Mutation commands (the contract for edits)

Mutation tool execution uses a typed command contract:

- The queue executor calls `execute_registered_mutation_command(repo_root, command)`.
  This validates the `MutationCommand` and routes to safe repository mutation tools
  (`write_file`, `create_file`, `delete_file`, `apply_patch`, `apply_patch_batch`).
  See: `src/mana_agent/multi_agent/runtime/agent_work_queue.py` `execute_registered_mutation_command`.

### Repository access and mutation tools

- **Patch application**: `src/mana_agent/tools/apply_patch.py` implements a Codex-style
  patch format with strict path validation and optional read-before-patch safety.
  It also writes patch history under the configured logs directory.
  See: `src/mana_agent/tools/apply_patch.py:1-447`.

### Analysis / ask service (indexed Q&A)

- **`src/mana_agent/services/ask_service.py`** is the central orchestration layer for
  answering questions over indexed code context.
  It supports:
  - “classic ask” using FAISS semantic search when the semantic index exists
  - a fallback to direct project search when the FAISS index is missing/empty
  - an agent/tool path (`ask_with_tools` / `ask_with_tools_dir_mode`)
  See: `src/mana_agent/services/ask_service.py:1-448`.

Important behaviors:

- Semantic index missing triggers a direct project search fallback with explicit
  warnings. See: `src/mana_agent/services/ask_service.py:1-448` (constants and fallback paths).

### Indexing and embeddings (FAISS vector store)

- **`src/mana_agent/vector_store/embeddings.py`** constructs an embeddings client
  compatible with the configured `base_url`.
  In particular, it supports NVIDIA endpoints by:
  - disabling client-side tokenization (`check_embedding_ctx_length=False`)
  - setting `extra_body["input_type"]` to `"query"` vs `"passage"`
  See: `src/mana_agent/vector_store/embeddings.py:1-88`.

- The ask service uses the FAISS store (`FaissStore`) and falls back when the on-disk
  index under `.mana/` or the requested `index_dir` is missing.
  See: `src/mana_agent/services/ask_service.py:1-448`.

### Skills and progressive skill loading

- **`src/mana_agent/skills/manager.py`** loads skills from:
  - project-local `skills/`
  - global user skills under `~/.mana/skills/`
  - built-in skills bundled in the package

It also supports building a skill index and reading individual skills.
See: `src/mana_agent/skills/manager.py:1-441`.

### UI and rendering layer

- **`src/mana_agent/ui/banner.py`** renders the CLI banner and compact mode headers.
  See: `src/mana_agent/ui/banner.py:1-56`.

- **`src/mana_agent/renderers/html_report.py`** renders HTML for analyze/describe/report
  flows using helper primitives (sections, badges, tables, details blocks, copy buttons).
  See: `src/mana_agent/renderers/html_report.py:1-613`.

## Data Flow (end-to-end)

### A. Analyze / describe / report artifacts
1. A CLI/command handler selects an analyze/describe/report mode.
2. Services run analysis and generate artifact payloads.
3. Renderers convert payloads into Markdown/HTML artifacts.
4. Output format selection is shared via `commands/analyze_formats.py`.

### B. Ask (indexed Q&A)
1. `AskService.ask()` chooses semantic index search when present.
2. If FAISS semantic index is missing, it falls back to direct `project_search()`.
3. `QnAChain` produces the final answer from retrieved snippets.
See: `src/mana_agent/services/ask_service.py:1-448`.

### C. Coding / mutation workflows (agentic loop)
1. **Flow building**: `build_agent_flow()` computes goal/phase/verification plan.
   See: `src/mana_agent/agent/flow.py:1-109`.
2. **Prompt assembly**: `prompting/builder.py` composes stable + ephemeral prompt
   context for the coding agent.
   See: `src/mana_agent/prompting/builder.py:1-353`.
3. **Work queue planning/execution**:
   - queue runner executes gated tool jobs
   - sniffer emits additional read/edit/verify jobs on successful job completion
   See: `src/mana_agent/multi_agent/runtime/agent_work_queue.py:1-1969`.
4. **Mutation execution**:
   - mutation commands compile and execute against safe repository mutation tools
   - verification is derived from the tool execution trace
   - missing deliverables or failed checks block the final result
   See: `src/mana_agent/multi_agent/runtime/agent_work_queue.py` and `src/mana_agent/tools/apply_patch.py`.

## Repository Layout

```text
src/mana_agent/
  agent/                 # flow + phase selection + verification planning
  analysis/              # static analysis + chunk helpers
  commands/              # CLI/chat command surface and output format contracts
  config/                # settings, runtime config
  dependencies/          # dependency graph support
  describe/              # repository description flow
  multi_agent/runtime/                   # prompt chains, agents, tool managers/workers, queue
  parsers/               # source parsing entry points
  prompting/             # stable/ephemeral prompt assembly and memory snapshots
  renderers/            # HTML rendering and export helpers
  services/             # ask/analyze/report orchestration services
  skills/               # skill loading and skill index matching
  tools/                # repository access + safe mutation tools
  ui/                   # console UI helpers
  utils/                # guards, IO, discovery, helper glue
  vector_store/         # FAISS store and embedding construction
```

## Related Docs

- [Overview](./01-overview.md)
- [Project Diagram](./07-diagram.md)
- [README](../README.md)
