# Adaptive Coding Runtime

Coding turns begin with one validated `ExecutionScopeDecision`. The model owns
semantic choices: intent, scope level, mutation targets, read-only related
files, evidence needs, tool families, budgets, mutation strategy, verification,
delegation, and stop conditions. Runtime code canonicalizes paths and enforces
that contract. A missing, unsafe, or invalid decision stops before tools run.

## Scope ladder

- Level 0: directly named, isolated, low-risk work; no search or delegation.
- Level 1: one bounded lookup around selected files or symbols.
- Level 2: dependency and impact investigation.
- Level 3: repository-wide architecture or cross-cutting work.

Escalation advances one level at a time through a typed child request containing
the missing fact and existing evidence references. A parent agent must approve
the new budgets; workers cannot widen their own scope.

## Root causes removed

The former queue path combined several independent expansion mechanisms. A
keyword classifier could select discovery, `QueueManager` seeded a search,
`CodingAgentSniffer` followed generic candidates/imports and appended edit plus
verification, document evidence helpers enumerated representative source and
configuration files, mutation synthesis attempted repair and legacy passes, and
a forced mutation pass could run after evidence was already sufficient. Reads
went through model workers, while path fingerprints did not share a canonical,
case-safe file identity or range coverage.

The adaptive path instead resolves model-selected paths directly, executes exact
reads through a run-scoped `EvidenceLedger`, batches known files, reuses full
reads for covered chunks, invalidates only changed files, generates one focused
mutation, retries only a failed patch hunk once, and performs the verification
strategy selected for the changed-file risk. The sniffer stops adding work once
the deliverable, mutation, focused verification, and escalation conditions are
satisfied.

Delegated work uses a task-specific `DelegationRequest` with canonical targets,
evidence references, tool/token budgets, boundaries, identities, expected
result, and stop conditions. Typed, bounded agent messages carry evidence and
requested actions without copying raw file contents.

Each run records routing/delegation calls, searches, unique reads, cache hits,
deduplicated jobs, mutations and retries, verification commands, messages,
escalations, queue/model/tool latency, and total elapsed time. Trace rows record
tool selection, requester, evidence references, cache status, expansion reason,
and the final stop reason without hidden reasoning.

## Internal Coding Engine: Parallel Tool Execution and Batch File Operations

Mana-Agent's **Internal Coding Engine** provides high-throughput, native software engineering capabilities operating inside the universal `model_directed` Autonomous Loop.

### Parallel Execution Architecture

When a model response returns multiple tool calls within a single turn, the engine uses an intelligent, dependency-aware scheduler (`DependencyGraph` and `ParallelToolExecutor`) to analyze and schedule operations:

1. **Independent Reads & Searches**: Executed concurrently under a bounded semaphore (default `max_concurrency=10`).
2. **Non-Conflicting Writes**: Target paths are inspected; mutations touching completely disjoint files execute concurrently without race conditions.
3. **Conflicting Writes**: Two or more writes touching the same file path are strictly serialized in turn order.
4. **Read-after-Write & Write-after-Read**: Dependencies on the same path are preserved to prevent stale reads and out-of-order mutations.
5. **Verification & Tests**: Test scripts, shell commands, and verification checks (`verify_project`, `run_command`) run sequentially after all preceding write operations finish.
6. **In-Flight Deduplication & Caching**: Concurrent identical reads are coalesced via single-flight deduplication. Read results are cached with TTL and automatically invalidated when any write touches the corresponding path.
7. **Partial Failure Isolation**: An error in one independent branch does not cancel successful sibling operations. Only downstream dependent children are marked as skipped.

### Batch Filesystem Layer

High-throughput batch tools allow models to perform multiple operations in a single tool call:
- `batch_read_files`: Concurrently reads multiple repository files with per-file SHA-256 hashes, line range slicing, and size caps.
- `batch_write_files`: Staged atomic writes into temporary files with optimistic concurrency check (`expected_sha256`) and transactional rollback on partial failure.
- `batch_apply_patches`: Multi-file Codex patch application with dry-run validation pass and rollback support.
- `batch_search_files`: Concurrent regex and literal searches across repository text.
- `batch_list_files`: Evaluates multiple glob patterns in a single filesystem pass.
- `batch_move_files`: Relocates files safely with boundary checks and rollback.
- `batch_delete_files`: Safe multi-file deletion preventing directory removal and path escapes.

### Configuration

Configured via `InternalCodingEngineConfig`:
```python
InternalCodingEngineConfig(
    enabled=True,
    max_concurrency=10,
    batch_operations_enabled=True,
    max_batch_size=50,
    max_batch_bytes=2_000_000,
    parallel_scheduling_enabled=True,
    cache_enabled=True,
    cache_ttl_seconds=300,
    rollback_on_batch_failure=True,
)
```

### Activity Box and TUI Streaming

All operations stream structured events into a single persistent `CodingActivityBox` in the TUI:
- Active operation groups (parallel groups, batch counts).
- Per-tool duration and status icons (`✓`, `◌`, `✗`, `■`).
- Active concurrency meter.
- Current phase (`Exploration`, `Implementation`, `Verification`, `Review`).
- Automatic error signature recording into `GlobalMemory`.
