# Remote execution fabric

## Existing architecture mapping

Mana previously had two related, local-only execution layers:

| Existing component | Previous responsibility | Fabric mapping |
| --- | --- | --- |
| `AgentChatGateway` / `gateway.stack` | Construct the shared chat, coding, and tool runtime | Constructs and exposes one `ExecutionManager`; callers do not select providers |
| `WorkspaceManager` | Create, lock, resume, review, and retain isolated Git worktrees | Remains the repository lifecycle owner; its selected checkout is the portable `SandboxSpec.repository_source` |
| `QueueManager` | Own tool jobs, worktree locks, cancellation, and task identity | Passes task/session/workspace identity and the execution manager to `ToolsManager` |
| `ToolsManager` | Resolve a job checkout and execute shell commands on the host | Uses a `SandboxExecutionContext`; shell/test/lint commands execute through `ExecutionManager` |
| `AskAgent.run_command` | Execute model-selected shell commands directly with `subprocess` | Gateway injects `ExecutionManager`; direct subprocess remains only as an explicit construction seam for isolated legacy unit tests |
| `ToolWorkerClient` | Isolate the model-facing tool worker process | Remains a control-plane worker; task commands it requests use the fabric |
| `ExecutionEventHub` and existing event sinks | Normalize and persist events | Receive sanitized `sandbox.*` lifecycle events |
| `~/.mana/repositories/...` stores | Persist sessions, workspaces, and worktrees | `~/.mana/execution/sandboxes` persists provider-neutral handles and leases |

The worktree and sandbox lifecycles are deliberately separate. A worktree owns
Git mutation safety and merge policy. A sandbox owns compute, process, network,
resource, artifact, snapshot, lease, and cleanup state. Provisioning references
an existing checkout; it never creates a duplicate repository or worktree.

Host-control subprocesses (configuration UI launchers, protocol server launch,
Git worktree plumbing, package evaluation harness setup) are not task execution
and remain local. Agent- or tool-selected commands are the migration boundary.

## Decision and execution contract

The model decision layer produces a validated `RoutingRequest` containing trust,
risk, required capabilities, enforcement, resources, and any explicit provider.
`ExecutionRouter` applies organization configuration only to that structured
decision. Missing or invalid decisions stop execution; there is no keyword or
provider fallback. A failed isolated provider is never retried through a weaker
provider.

The lifecycle is:

```text
requested -> provisioning -> ready -> running
  -> suspending -> suspended -> resuming -> ready
  -> terminating -> terminated -> cleaning -> cleaned
```

Any active state may enter `failed`; termination and cleanup are idempotent.
Handles persist external runtime identity, associations, heartbeat/lease data,
snapshot references, cleanup status, and sanitized failures for restart recovery.

## Provider boundary

All providers implement `SandboxProvider`. Provider-native configuration and
clients stay inside `execution.providers`. Optional SDKs are imported lazily.
Capabilities report both feature presence and enforcement strength; the router
rejects a provider that cannot meet the requested guarantee.

`local-process` accurately reports best-effort resource controls and no network
isolation. Docker, SSH, Kubernetes, Modal, and custom HTTP providers require
explicit validated configuration. Remote integration tests are opt-in and are
not represented as verified by unit tests using fake clients.

## Recovery, secrets, and artifacts

The manager persists every transition atomically and uses heartbeat leases.
`CleanupController` recovers expired or interrupted handles after restart and
keeps original and cleanup failures separately. Secret values are resolved only
at the provider boundary, never serialized in a spec or event, and known values
are redacted from command output. Artifact paths are allowlisted, confined to
the workspace, symlink-safe, size-limited, and checksummed.
