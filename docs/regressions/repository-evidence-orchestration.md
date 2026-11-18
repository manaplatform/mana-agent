# Repository Evidence Board and independently scheduled investigation

Status: implementation on the feature branch with uncommitted follow-up changes.
Tests, linters, type checks, builds, provider calls, and deployment were not run
by the agent.

## Symptoms and demonstrated mechanisms

An editable install failed at CLI startup with `SyntaxError: keyword argument
repeated: continuation_context`. The Codex parser supplied that keyword once
for the existing continuation ID and again for the evidence handoff. Build one
continuation payload and preserve the ID for all results. Include the provider
thread, turn, and observed changed paths when the engine yields for evidence.
Do not add a second continuation result field or discard the existing ID.

The next live attempt reached coding assignment dispatch and failed with
`CodexSettings.worktree_isolation: Instance is frozen`. A shallow copy is
still a frozen Pydantic model. The task facade now uses `model_copy(update=...)`
to start in the checkout already assigned by the lane. Preserve the parent
settings when constructing task workers. The facade fixture covers both engine
selections, the child checkout, and cancellation by the parent.

In the cancelled NetPeek investigation, the first OpenRouter request responded
in 2.7 seconds and the next request showed no completion or error for 2 minutes
46 seconds before cancellation. The native coding loop and inference engine
both published a start event for the same provider request. The inference engine
now owns that lifecycle, reports elapsed time and timeout, and the coding
worker cancels its in-flight await when the user stops the task. A shallow
layout assignment offers only the permitted layout and evidence tools.

A later investigation returned an enclosing JSON fence and the coding engine
rejected the response before checking the findings. Both engines now remove
one optional enclosing fence at their typed result boundary. Rejected values
stay out of user-facing validation errors. Existing child attempts preserve
their evidence references and are described to the model for same-task recovery.
Do not add a second activity entry for a native request, publish rejected provider
input in a diagnosis, or retry an investigation under a new scope.

### Native investigation report format regression

On 2026-11-18 at 22:47, investigation `task_20261118_000032` completed
`batch_read_files`, `repo_search`, and `read_file`. The provider returned
successfully and Mana rejected the final report with
`Invalid InvestigationResult (decision: json invalid)`. The journal omits
the final text, so the exact syntax error is unknown.

The native coding loop treated text without tool calls as a completion and
passed it to the JSON parser. Describing the schema in the prompt did not
constrain that text. The earlier JSON fence fixtures tested report decoding
without exercising the native completion path.

The native tool loop now accepts completion through one direct
`report_investigation` call. Its argument schema comes from the admitted
assignment. The terminal call uses the existing model budget, leaves no
pending tool action, and goes through the typed report and publication
checks. The loop rejects a report mixed with other calls or submitted by a
hosted program. Codex and existing text-based adapters keep their JSON
boundary. Malformed JSON reports explain the required format, and schema
errors omit rejected provider data from the diagnostic traceback.

Do not treat free-form text or successful repository reads as a native
investigation report. Preserve the selected backend, scope, and validation
when completing the report.

```sh
venv/bin/python -m pytest -q tests/coding/test_investigation_engines.py tests/llm/test_engine.py -k 'native_investigation_completes or investigation_result or rejects_unregistered_and_invalid_tool_calls'
```

The agent did not run the tests or replay the failed live task.

### Evidence orchestration changes

Repository inspection found that AUTO/DEBUG saved inspection dictionaries on a
parent workflow and PLAN ran a native investigation session on that parent.
`ModePhase.UNDERSTANDING` and `INVESTIGATING` coupled knowledge acquisition to
workflow transitions. Neither form had independently schedulable task identity.
The run-scoped EvidenceLedger and EvidenceMemory could reuse reads, but they did
not represent durable repository assertions, supporting relationships, or
contradictory findings shared by different tasks.

The specialist queue selected jobs by priority without checking `depends_on`.
Its pending jobs were in memory. TaskBoard saved every cached task row, allowing
a stale board instance to overwrite unrelated progress. MainAgent's autonomous
step runner could mark a child complete without executing it. TaskBoard's
completion helpers caught transition failures and forced DONE; a queue worker's
completion message also did not establish independent goal verification.
These were source findings, not reproduced live incidents.

Native coding also had an unavailable-model branch that read allowed files and
returned a completed message. Both native calling paths now stop before selecting
any tool when their model is unavailable. Reading allowed files cannot substitute
for a missing decision or for implementing the objective.

The completion path also rebuilt a result after finding an already verified
escrow receipt. Repeated completion now reuses that receipt. Relative changed
paths are resolved in the admitted checkout before building artifact contracts.

## Ownership and corrected contracts

- The Autonomous Loop advances decisions and determines goal satisfaction.
  MainAgent coordinates through typed decisions and existing TaskBoard operations.
- TaskBoard owns objectives, assignments, dependencies, requirements, blockers,
  progress, and verification. Investigation and understanding are task kinds.
- RepositoryEvidenceBoard owns knowledge and its provenance. It has no task
  status, scheduling loop, worker claims, or goal-completion authority.
- Execution Supervisor owns attempts, leases, fencing, checkpoints, result
  escrow, recovery, and completion enforcement.
- QueueManager admits native tool work; AgentWorkQueue remains the native work
  adapter. Registered operations and queue jobs have durable correlation and
  atomic claims. Neither queue independently creates TaskBoard tasks.
- ToolRunner executes native tools. MainAgent does not directly mutate files.
  External Codex retains its own inner tool loop and provider provenance.

An existing Autonomous Loop can project its already admitted task into a
coordination lane using the same supervisor attempt and lease. This creates no
second root execution. Synchronous children reuse the existing capacity-lending
contract, and the projection follows the supervisor when the controller returns.

## Knowledge model and retrieval

Workspace SQLite migration 2 adds evidence revisions, source and dependency
indexes, relationships, task links, and durable operation/job records. Existing
task rows gain a compare-and-set revision. Only locally changed TaskBoard rows
are saved. A stale writer receives an explicit conflict instead of replacing
newer state. Queue jobs also use version-checked publication.

Evidence records carry repository/workspace and checkout identity, sources and
symbols/ranges, content hashes, category, structured observations and
relationships, limitations/errors, discovery time, producing agent/backend,
task/attempt/generation, decision/invocation, and result provenance.

The category is extensible. Standard uses include repository structure, code
behavior, dependencies, entrypoints, integration wiring, API/configuration
contracts, errors, root causes, test results, security constraints, and existing
implementation decisions. No repository-specific filenames are encoded.

Observed, inferred, and hypothetical are separate classifications. Supported,
contradicted, rejected, and pending validation are separate from freshness.
Model confidence never promotes a claim into a verified observation. Only host
ingestion records actual observations. Model findings need supporting evidence;
unverified hypotheses cannot satisfy a required fact.

Queries filter by repository, task, paths, symbols, category, dependency, and
one-hop relationships. They default to 20 records and a conservative 4,000-byte
model-context budget, with pagination and explicit detail retrieval. Optional
memory providers do not own these records or become startup dependencies.

Pagination advances only past consumed rows. Oversized entries return an ID for
paged detail retrieval. Required tools are checked against actual observations,
including bounded supporting evidence. Equivalent concurrent finding submissions
share an identity while each producing task retains its publication receipt.

File stamps come from the bytes read. Bounded immediate-directory manifests can
support layout observations without recursive source scans. Searches and errors
without a complete source version remain observations with unknown freshness;
they cannot silently become reusable confirmed repository facts.

Known mutations invalidate affected findings in that source view and derived
findings. Query/dispatch/completion also recheck sources to detect external
edits, deletions, renames, and changed checkout contents. Other worktrees can
reuse a finding only if its declared source/dependency versions match. Unknown
or oversized source coverage remains an explicit limitation.

## Dynamic tasks, engines, and missing evidence

An investigation/understanding task has its own objective, owner, bounded scope,
evidence requirements, acceptance criteria, dependencies, and output references.
Equivalent requests reuse a task; stale completed findings require new evidence.
Understanding may synthesize current evidence without another source search.
Reading a certain number of files is never a completion criterion.

The selected coding engine executes these tasks in read-only mode. Internal
execution uses registered scoped read tools. External Codex receives the typed
assignment, read-only sandbox, disabled network access, and a structured result
schema. An unavailable or incompatible selected engine fails explicitly; it does
not switch to a native investigator or another provider. Task-local facades keep
concurrent continuation and working-directory state separate.

Each assignment has a task-specific Codex continuation namespace and uses the
checkout already admitted by its lane. Parent cancellation reaches these workers.
Native operations reject assignments whose objective, scope, requirements, or
dependencies changed during execution; external turns are checked at their outer
boundary, preserving Codex's ownership of its inner loop.

Findings can cite existing current observations. New citations are validated
through exact registered reads selected by the submitted citation, and retain
their source revision. The finding remains an inference. Provider operation
details are not relabeled as native ToolRunner receipts.

### Finding classification regression

On 2026-11-18 at 22:15, internal investigation `task_20261118_000023`
completed `batch_read_files` and received its final model response. The shared
parser reported literal errors on the first two findings' `classification`
fields. The child checkpointed before publishing findings. The logs preserve
the affected fields and omit the submitted labels, so the exact inputs are
unknown.

The model saw two classification vocabularies in the assignment. A finding
can be `inferred` or `hypothetical`, while an evidence requirement can also
accept host-created `observed` records. The old prompt described the
distinction in prose, the parser required exact enum spelling, and the error
only said `literal error`. Existing decoder fixtures used exact valid labels.

The prompt now states the exact finding labels and explains how to reference
host observations. The shared investigation decoder handles whitespace and
capitalization and interprets `observed` on a model finding as `inferred`.
Normal citation and evidence checks still apply. Only registered host reads
can produce observations that satisfy an observed-only requirement. Other
invalid labels fail with the allowed choices and without the submitted values.
Codex also surfaces this diagnostic when its investigation result is invalid.
No data migration is needed.

Do not admit model-supplied observations to the Evidence Board or classify
unknown labels as inferences.

User verification uses the following command. The agent did not run it or
resume the affected live task.

```sh
venv/bin/python -m pytest -q tests/coding/test_investigation_engines.py tests/test_codex_integration.py -k 'investigation_classification or observed_label'
```

### Architecture relationship output regression

On 2026-11-18 at 21:55, investigation `task_20261118_000018` completed
`batch_read_files` and received a final model response. The result parser then
rejected entries 0 and 1 in the eighth finding's `architecture_relationships`
with string-type errors. The child checkpointed and blocked before publishing
findings. The redacted journal preserves the field locations, so the exact
rejected values are unknown.

The earlier schema advertised each link as an open string map. The prompt
supplied no field names or example for representing multiple targets. Runtime
validation required string values and rejected the report when an entry used
another type. Earlier parser fixtures omitted architecture links.

The shared investigation prompt and schema now request one link per object with
string fields `source`, `relation`, and `target`. The local parser still accepts
earlier flat string maps and rejects nested values. Both shapes serialize to
the existing evidence format, so existing records need no migration. The
fixtures cover schema shape, round-trip parsing, publication, and redaction.

Do not reintroduce an open map in the model-facing architecture relationship
schema. Send multiple targets as separate links and keep invalid relationship
values out of public errors.

User verification uses the following command. The agent did not run it or
resume the affected live task.

```sh
venv/bin/python -m pytest -q tests/coding/test_investigation_engines.py -k 'investigation_architecture_relationships or uses_selected_engine'
```

### Finding requirement binding regression

On 2026-11-18 at 22:08, investigation `task_20261118_000021` completed a
`repo_search` and received a structured report. The session journal recorded
five saved findings before the child checkpointed and blocked with
`Investigation requirement references an unknown finding`. The journal does
not retain the rejected index.

The earlier response contract used a separate map of requirement IDs to
zero-based finding positions. An out-of-range position passed type validation,
and the publisher checked the bounds only after saving the findings. The
selected-engine fixtures contained valid positions and missed the partial
publication path.

The shared schema now places `requirement_ids` on each finding. The publisher
validates these IDs against the assigned requirements before resolving citations
or saving findings, then applies the existing classification, category, and
evidence checks. The IDs belong to the report and stay out of reusable evidence
identity and storage. Existing evidence can still be bound by its board
reference. Older reports with valid positions remain usable. Invalid positions
are rejected before any report findings are published. No migration is needed.

Do not expose finding positions as references in the model-facing schema or
defer binding validation until after publishing the report.

User verification uses the following command. The agent did not run it or
resume the affected live task.

```sh
venv/bin/python -m pytest -q tests/coding/test_investigation_engines.py -k 'uses_selected_engine or binds_requirements_on_each_finding'
```

### Unknown evidence requirement regression

The 2026-11-18 session at 22:22 reported
`Investigation report referenced an unknown evidence requirement`.
TaskBoard's admitted assignments for children `task_20261118_000025` and
`task_20261118_000026` each contained only `repository_finding`. The response
passed structural validation and failed the assignment membership check. The
session does not retain the rejected ID or the binding field that supplied it.

The response schema allowed arbitrary strings in `findings[].requirement_ids`
and arbitrary keys in `evidence_bindings`. The prompt also showed
`entrypoints` as a sample ID even when the assignment had no such
requirement. That example could have been copied. Earlier fixtures used known
IDs for successful responses and did not check the schema against the
assignment.

The prompt now refers to the assignment directly. The shared schema builder
lists the admitted IDs as the enum for each finding's requirement list and
as the only keys in `evidence_bindings`. Codex receives the same schema through
its output schema parameter. Native text responses still pass through the
publisher's exact membership check before citation reads or finding
publication. Valid legacy positional reports remain supported. Errors
identify the binding field and omit the rejected ID. No migration is needed.

Do not put sample requirement IDs into the prompt or accept arbitrary
binding IDs. Keep requirement selection tied to the admitted assignment.

User verification uses the following command. The agent did not run it or
replay the affected session.

```sh
venv/bin/python -m pytest -q tests/coding/test_investigation_engines.py tests/test_codex_integration.py -k 'uses_selected_engine or investigation_schema'
```

### Source citation regression

NetPeek investigation `task_20261118_000011` completed its model turn and several
source reads, then failed during directory citation validation. The SHA-256
submitted for `src` matched the retained `repo_search` output artifact. The
registered listing of `src` produced a different digest. The old result schema
required a digest on every citation and exposed artifact references alongside
actual source stamps. Existing engine fixtures provided the correct digest in
advance and missed this distinction.

Both engines can return exact source locations without a SHA-256 or an
excerpt. The host reuses a matching current observation or runs the precise,
scope-checked read selected by the citation. New file citations specify
1-based inclusive start and end lines. The stored source version and bounded
excerpt come from the read. Earlier reports can still submit a digest or
excerpt, which the host validates against the selected source. No database
migration is needed.

Follow-up child attempts `task_20261118_000013`, `task_20261118_000015`, and
`task_20261118_000016` also failed citation validation after successful model
and tool calls. Their journals establish which selected reads finished. They
do not retain the rejected excerpt needed to identify each mismatch.

The advertised result schema leaves source hashes and revisions to the host.
Native batches attach per-file references to the first 20 result rows when
they contain complete, current, bounded reads. Each reference identifies the
file observation and exact line coverage from the registered batch receipt.
This allows a finding to reuse the same observation without reproducing file
text or a digest. New directory citations use the location and directory kind.
Matching legacy file excerpts allows indentation and blank-line differences
while preserving spaces within lines. Legacy directory excerpts can be
manifest text or a list of observed entry names, with every name required to
match.

#### Host-captured excerpts

At 22:30 on 2026-11-18, investigation `task_20261118_000028` stopped after a
successful validation reread of `scanner.c` lines 846–867. The saved initial
read covered lines 840–875. Both reads have the same file digest, and the
overlapping content is identical. The session retained the selected line range
but omitted the submitted excerpt and digest. The exact rejected value is
therefore unknown.

The result schema had still advertised model-written excerpts, and its
validator required an excerpt for hash-free file citations even when the
model supplied a complete range. This made exact source selection depend on
reproducing source text. The schema now leaves excerpt capture to the host.
The publisher saves the text from the selected lines and reports the failing
finding, source index, and safe reason on a mismatch. Earlier tests populated
the excerpt when testing hash-free citations. The updated fixtures cover a
range-only read, reuse of a broader observation, and rejection of a bad range
or invalid legacy quote or digest.

Do not require a model-written quote to resolve an exact source location.
Validate the path, source kind, line coverage, and any supplied legacy proof.

```sh
venv/bin/python -m pytest -q tests/coding/test_investigation_engines.py -k 'uses_selected_engine or matches_multiline_code_citation or source_versions_host_owned or file_citation_requires'
```

The agent did not run this command or resume the failed live task.

#### Shallow layout citation regression

At 22:39 on 2026-11-18, investigation `task_20261118_000030` completed a
root listing, then stopped after the model returned its report. The transcript
and the parent execution journal recorded the failure as `'read_file'`.
The saved child assignment allowed only shallow root discovery. There was
no external file-read call or registered citation read in the journal.

ToolRunner correctly excluded `read_file` from shallow authority. The
report schema still advertised file citations, and the source validator
indexed the missing tool schema when resolving one. That lookup failed before
a ToolRunner request could enforce the scope. Earlier citation fixtures used
file-capable assignments.

The assignment now carries the shallow-layout restriction derived from the
same TaskBoard predicate as ToolRunner. The shared report schema and prompt
permit a direct citation only to the immediate root listing for that scope.
Report validation rejects file and nested directory citations before any
source validation read or finding publication. Citation resolution reports
a scope error if a required tool is unavailable. Explicit audit and scoped
file assignments retain their existing capabilities. No data migration is
needed.

Do not advertise file citations in a shallow-only assignment or register
a file-read capability to recover from an invalid citation.

```sh
venv/bin/python -m pytest -q tests/coding/test_investigation_engines.py -k 'shallow or audit_file_citation or unavailable_read_tool'
```

The agent did not run the tests or replay the live task.

Mode, compound, and MainAgent investigation dispatch all register the same
supervisor verifier for durable evidence requirements. Saved investigations keep
their attempt identity and scope when recovered, while different parents can
reuse evidence through Board references.

Tests cover hash-free file and directory citations with both engines, reuse of a
current observation, a search artifact hash in place of the source hash,
wrong excerpts and out-of-range citations, and recovery with a changed scope.
The original NetPeek task was cancelled after it saved its checkpoint. The
later DEBUG parent `task_20261118_000014` is waiting with a saved checkpoint.
A successful live run of the latest citation path remains unverified.

MissingEvidenceRequest identifies the existing coding task, exact requirement,
reason, and continuation reference. The existing controller can create focused
work, retain completed patches/checkpoints, and resume the same authorized
assignment through a validated recovery decision. Only dependent work waits.
MultiTaskOrchestrator reads model-authorized additions to the live TaskBoard
graph while independent children run. A worker's completed message is rejected
without TaskBoard and supervisor verification.

## Modes, compatibility, and recovery

AUTO and DEBUG choose tasks dynamically. DEBUG retains reported evidence and its
causal repair requirement. Explicit PLAN can write task/evidence state but cannot
mutate repository files without the human inbox approval for the exact revision.
New plan revisions cite evidence. Older approved plan payloads retain their digest
when the new optional reference field is absent. Evidence updates do not expand
approved file operations or replace the user objective.

Old understanding/investigating labels decode to the active lifecycle when the
workflow persists again; new work does not emit those phases. Legacy narrative
evidence is not imported as confirmed knowledge without provenance and matching
source versions. Existing run caches remain optimizations.

Checkpoints retain assignment snapshots, evidence references, requirements, and
existing receipt/recovery context. Stale attempts cannot publish board or tool
results. An interrupted operation is reconciled from its durable record rather
than automatically replayed. Secret-bearing queued work must use credential
references, rather than persist inline secrets. Stop old runtime processes before
upgrading; do not run mixed binaries against the migrated workspace database.

The existing Activity and Coding Activity surfaces receive correlated updates.
This change adds no standalone board dashboard, scheduler, external service,
dependency, release, or deployment.

## Do not reintroduce

- Never implement `INVESTIGATE → UNDERSTAND → PLAN → CODE` as required stages.
- Never require unrelated investigation tasks to finish before coding can run.
- Never create investigation work where selected current evidence is sufficient.
- Never treat a model assertion, confidence, source filename, planned check, or
  checked-off task as independent execution/verification evidence.
- Never force DONE after a rejected transition or manufacture executed subtasks.
- Never complete the Autonomous Loop from a pending escrow record or a missing
  subtask executor; project the verified supervisor receipt into TaskBoard.
- Never let queue jobs create duplicate tasks or disregard declared dependencies.
- Never overwrite another task from a stale whole-board cache.
- Never turn an evidence update into permission to broaden an approved objective.
- Keep artifact digests separate from source digests. Resolve source versions
  from registered observations and validate any explicitly supplied digest.
- Never insert Mana decisions into external Codex's inner tool loop or silently
  change the selected engine after failure.

## User-owned verification

Start with the new contracts and integration fixtures:

```sh
venv/bin/python -m pytest -q tests/evidence/test_repository_evidence_board.py tests/coding/test_investigation_engines.py tests/gateway/test_evidence_driven_orchestration.py
venv/bin/python -m pytest -q tests/llm/test_engine.py -k timeout_cancels_selected_request
```

Then run the affected compatibility surfaces:

```sh
venv/bin/python -m pytest -q tests/persistence/test_workspace_sqlite_repository.py tests/test_multi_agent_core.py tests/test_agent_work_queue.py tests/execution/test_autonomous_loop_taskboard.py tests/execution/test_coding_research_recovery.py tests/execution/test_unified_autonomous_loop.py tests/gateway/test_multi_task_orchestration.py tests/gateway/test_debug_plan_mode_contract.py tests/gateway/test_lane_coordinator.py tests/coding/test_tool_runner_execution.py tests/test_chat_planning_mode.py tests/execution_supervisor/test_supervisor_core.py
node --test tests/dashboard/live_chat_reducer.test.mjs
```

The fixtures cover empty/reused evidence, distinct observations/inferences,
unsupported hypotheses, stale sources and dependent findings, contradictory
records, task/repository isolation, bounded retrieval, source security,
task/job version conflicts, selected-engine assignments, missing-evidence
handoffs, task-kind identity/deduplication, and dependency isolation.

For live acceptance, use disposable repositories with each configured engine:
start with an empty board, repeat a related request, introduce a missing evidence
requirement during coding, allow independent work to progress, change a cited
source, interrupt an attempt, resume it, and review actual task/tool traces and
verification receipts. Confirm PLAN remains read-only before approval and that
runtime feature integration cannot complete without its existing evidence gates.
Provider behavior, interactive UI, performance, migration from real persisted
sessions, and broad compatibility remain unverified until these checks run.
Several legacy mode fixtures still script the retired embedded investigator and
expect sequential `evidence-N` IDs. Those fixtures need migration to typed
selected-engine responses and revisioned board references.
