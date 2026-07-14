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
