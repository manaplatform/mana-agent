# Project Diagram

This page contains a few Mermaid diagrams that summarize how `mana-agent`
coordinates analysis, retrieval, and coding workflows.

## High-level flow

```mermaid
flowchart TD
    A[Local repository] --> B[Discover & index files]
    B --> C[Analyze]
    C --> D[Generate artifacts under .mana/]
    B --> E[Ask (search + evidence)]
    E --> F[LLM answers grounded in evidence]
    B --> G[Chat (REPL)]
    G --> H[Plan]
    H --> I[Inspect files & retrieve evidence]
    I --> J[Patch / write files through tools]
    J --> K[Run verification]
    K --> L[Summarize changes]
```

## Artifact outputs

```mermaid
flowchart LR
    A[Requested format(s)] --> B[analyze.json]
    A --> C[analyze.md]
    A --> D[analyze.html]
    A --> E[analyze.dot / graphml]
    A --> F[diagram.mmd]
    B --> G[Automation & CI]
    C --> H[Human review]
    D --> I[Browseable report]
    E --> J[Graph visualization]
    F --> K[Embeddable diagram]
```

## Coding-agent tool lifecycle

```mermaid
sequenceDiagram
    participant U as User
    participant R as QueueManager / planner
    participant W as Worker (tool runner)
    participant T as Repository Tools
    U->>R: coding request (e.g., patch docs)
    R->>T: repo_search / read_file (evidence)
    T-->>R: evidence sources
    R->>W: planned mutation intent
    W->>T: read/search + write/apply_patch/delete
    T-->>W: changed file list
    W->>T: run verification (when available)
    T-->>W: verification results
    W-->>R: final changed files & summary
    R-->>U: answer + changed files
```
