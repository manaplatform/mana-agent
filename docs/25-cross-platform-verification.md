# Cross-platform verification

A Fleet run is an immutable matrix of platform, architecture, runtime, worker,
provider, command, result, artifact, and cleanup records.

```text
Model selection request
  → strict capability and coverage validation
  → FleetVerificationPlan
  → isolated Git worktree per job
  → ExecutionManager provider lifecycle
  → bounded logs and declared artifacts
  → cross-platform comparison
```

## Run a matrix

```bash
MANA_FLEET_ENABLED=true mana-agent fleet verify \
  --root-dir . \
  --platform linux \
  --platform windows \
  --platform macos \
  --python 3.12 \
  --tool git \
  --tool pytest \
  --label trusted \
  --command "python -m pytest -q"
```

The repository commit is resolved before selection. Every job gets a detached
managed worktree, verifies `git rev-parse HEAD`, and rejects a dirty starting
workspace. Commands are argv arrays. Artifact paths are relative, confined to
the workspace, checksum-addressed, and subject to per-run size limits.

Required coverage is never weakened. If Windows was required but no compatible
Windows worker ran, the outcome is `infrastructure_incomplete`, never
`fully_verified`.

## Outcomes and comparison

Job failures distinguish test/setup failures from provider failures,
disconnects, permission denials, capability mismatches, timeouts, repository
transfer failures, artifact failures, cleanup failures, and model/routing
failures. Cleanup failure is retained independently from the original test
result.

Run outcomes are:

- `fully_verified`
- `partially_verified`
- `failed_verification`
- `infrastructure_incomplete`
- `cancelled`

Inspect a run with:

```bash
mana-agent fleet jobs
mana-agent fleet job <job-id>
mana-agent fleet logs <job-id>
mana-agent fleet artifacts <job-id>
mana-agent fleet compare <fleet-run-id>
```

## Eval Lab

Eval suite contracts accept `local-worktree`, `execution-fabric`, and `fleet`
backend names. Fleet suites must declare platforms and minimum coverage:

```yaml
defaults:
  workspace_backend: fleet
  required_platforms: [linux, windows, macos]
  minimum_platform_coverage: 3
  runtime:
    python: ["3.12"]
  fleet:
    labels:
      include: [trusted]
      exclude: [experimental]
    maximum_workers: 6
    per_worker_concurrency: 1
```

The suite is rejected before execution when its typed Fleet defaults are
incoherent. Remote Eval execution requires an injected, validated Fleet service
decision; the workspace factory does not substitute a local backend.

## Recovery

On restart, non-terminal read-only jobs are truthfully marked failed instead of
being re-executed. Interrupted mutation jobs become
`revalidation_required`. Completed job results remain immutable. Retained
workspaces require the `fleet.workspace.retain` permission and remain subject to
maximum lifetime policy.

Automatic repair is disabled by default. Fleet failure evidence may create a
managed coding candidate only after a separate structured diagnosis and repair
decision. Verification workers are never edited, the primary checkout is not
patched, and promotion remains an explicit merge candidate.
