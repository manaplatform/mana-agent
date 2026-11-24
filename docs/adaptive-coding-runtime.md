# Adaptive Coding Runtime

## Supervised AUTO and DEBUG

After the gateway validates an AUTO or DEBUG decision, realtime Internal
Coding receives one Coding-lane assignment. The native loop chooses its tools
from the authority of that assignment. It can read, edit, run a diagnostic,
or verify immediately. TaskBoard retains progress and tool receipts, and the
Execution Supervisor fences the attempt and confirms completion. Explicit PLAN
retains its exact-revision approval boundary. External Codex, explicit Batch,
and existing staged tasks keep their staged execution contracts.

`verify_coding_change` runs selected arguments in the assigned checkout for
realtime write tasks. Read-only verification uses the isolated Docker verifier.
A failed check returns its exit code and bounded output to Coding, which can
repair the implementation in the same attempt. Passing checks must match the
current versions of files associated with the assignment's tool receipts and
any observed source drift.
The native terminal decision can report completion, identify a blocker, or
ask for material clarification. Production capability changes additionally
require source-backed runtime reachability and existing integration verification.

For a realtime Internal write task, `read_file`, `repo_search`, and the
top-level `ls` are ordinary tool choices alongside edits and checks. Each source request supplies
`evidence_need` with the missing fact and the edit or check it could affect.
Searches name specific paths or a targeted glob. Use `ls` when the top-level
location is unknown. An edit or useful diagnostic can be the first action, and
a failing check can lead to another scoped read.

The loop shows the remaining tool calls and model turns before each decision.
It leaves capacity for an edit and a focused check before the first change and
after a failed check. An unverified edit needs only the check reserve, and a
passing check releases it. One model turn stays available for the typed
result. A rejected group dispatches nothing and identifies the invalid call
when a specific call caused the rejection. Each distinct error can get one
correction within the remaining model budget and the configured no-progress
limit. See the [search correction regression](regressions/unified-adaptive-coding.md#distinct-tool-errors-shared-one-correction).

After a source-inspection shell rejection, Coding pauses the generic shell
tools while the model corrects its selection. The model can use offered
repository tools, edits, or `verify_coding_change` with direct arguments.
For a top-level `ls`, supply only `evidence_need`. Provider argument errors
refer to the offered schema for the affected tool. The pending correction
survives a saved continuation. See the [shell inspection incident](regressions/unified-adaptive-coding.md#shell-inspection-repeated-during-correction).

When the assignment approaches its tool or model-turn limit, the model can
request `renew_coding_budget` with the progress made and the edit or check it
still needs. Renewal uses the active accounting reservation, the configured
increments, and any explicitly configured renewal count. Further renewals
remain available while work advances and the reservation has capacity.
Granted allowances and consumed calls survive resume. Failed build results
on new source versions can show progress toward the next renewal. Coding
Activity alerts the user when actions repeat without advancing the task.

Source inspection uses the repository tools. The adaptive loop validates
Codex patch syntax and isolated check inputs before charging a tool call.
Use a direct command argv and `cwd` of `.` or a repository-relative directory.
Realtime Internal write assignments run verification through the shell action
policy in the assigned checkout. Use `setup_argv` for configuration or build
prerequisites. Setup and the final check share the checkout and its build
outputs. A failed prerequisite stops that sequence and returns its output for
the next repair. A required build must pass against the current source versions
before the assignment can report success.

ToolRunner fingerprints observed files before and after mutating commands.
In a Git checkout this covers tracked files and non-ignored candidates. Tools
with explicit file targets still observe those paths. The capture budget limits
retained diff data. Hashing streams the file contents under the operation's
deadline and cancellation checks.

Each model turn must select an offered coding tool or submit a terminal result
with one direct `submit_coding_result` call. If the model replies with prose
alone, Coding requests one correction from the same model within the remaining
turn budget. The model retains completed operations and their tool results.
A second reply without a tool call stops the attempt. See the
[missing tool call regression](regressions/unified-adaptive-coding.md#missing-tool-call-in-a-native-coding-response).

When an assignment changes production runtime behavior, completion requires a
passing check and one source-backed wiring chain in execution order. Every edge
target must match the next edge's source. If a report breaks the chain, Coding
returns the disconnected edge pairs to the model for one correction when a turn
remains. The correction uses the existing edit and check receipts.

Native file reads default to the whole source within the smaller of the
remaining model context allowance and the 48,000-byte tool-content ceiling.
Batch reads share that allowance. The result distinguishes full and scoped
reads from budget-truncated ranges, returns the actual line boundaries and
file hash, and provides the next line when needed. Files beyond the
2,000,000-byte source-provenance limit require a narrower artifact or search.

The Repository Evidence Board saves source-backed observations as tools return
and checks their hashes before reuse. Duplicate observations reference existing
results, and invalid references return a correction diagnostic. On recovery,
the supervisor resumes the saved Coding assignment when action receipts reconcile.
If a previously observed source file changes between tools or during a model
request, Coding retains the completed operations, marks stale reads, and asks
the model to select its next action from the current workspace. It discards a
pending model response based on superseded source and requires fresh verification
for the current versions. A version that diverged from its tool receipt stays
out of the task's reported edits until a new tool receipt accounts for it.
Command outputs covered by the checkout's current Git ignore rules are omitted
from this comparison. Tracked files and explicit file edits remain in scope.
Coding Activity labels reads, searches, edits, diagnostics, checks, and review
in one stream. See the [workspace drift regression](regressions/unified-adaptive-coding.md#workspace-changes-after-a-tool-receipt).

## Unresolved bugs and external evidence in staged workflows

The Main Model can select research at coding turn/yield boundaries after evaluating a typed debugging report, local evidence, attempted fixes and installed versions. Both engines return a report as pending work. Research uses the existing autonomous controller and search providers, then returns an untrusted, sourced recommendation to the same task/backend. A fixed number of failures never triggers a search.

Default task limits are 12 search requests in four rounds, with at most three queries and three source inspections per round. Queries are sanitized; task-scoped results, consumed counters and delivery receipts survive recovery. Missing permissions or provider availability are reported to the Main Model without fallback searches. The existing integration and independent completion gates still apply. Configuration, recovery boundaries, limitations and user-owned verification are documented in [the regression note](regressions/coding-bug-research.md).

## Queue-based coding turns

Queue-based coding turns begin with one validated `ExecutionScopeDecision`. The model owns
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
- Current activity, such as reading, editing, debugging, or verifying.
- Automatic error signature recording into `GlobalMemory`.
