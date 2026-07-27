# Mana Fleet

Mana Fleet is the disabled-by-default orchestration layer for trusted,
cross-platform repository verification. It does not replace the execution
fabric, reverse-worker gateway, Eval Lab, taskboard, permission broker, or event
stream. A Fleet plan selects a worker and an existing execution provider; the
`ExecutionManager` remains responsible for provisioning, execution, artifact
collection, and cleanup.

## Trust and capabilities

Workers generate a bounded inventory from an allowlist of runtime probes. The
inventory includes OS, architecture, Python and Node versions, verification
tools, Docker/GPU availability, workspace modes, concurrency, and capacity. It
never includes environment values, usernames, keys, tokens, or filesystem
listings. Reverse workers sign the capability message with their enrolled
Ed25519 identity. Coordinator enrollment policy supplies trust labels; a worker
cannot self-assert `trusted`.

Inventories are capped at 64 KiB, have a five-minute default TTL, and are
rejected when malformed, stale, future-dated, oversized, unsigned, or bound to
another worker. The latest accepted inventory and SHA-256 fingerprint are
stored under `~/.mana/fleet/workers/`.

## Selection and failure policy

The gateway must supply a typed `FleetSelectionRequest`. Hard requirements are
checked before deterministic scoring. Required platforms, architectures,
runtimes, tools, provider capabilities, labels, trust, health, and concurrency
must all match. Offline, stale, busy, draining, and revoked workers are not
dispatchable. A missing or invalid decision stops execution:

```text
No compatible worker → actionable FleetSelectionError → no local fallback
```

Revoked workers cannot be enabled again; they must complete fresh enrollment.
Mutation jobs interrupted by restart require revalidation, and persisted
completed results are never removed from a run during an update.

## Worker installation

The same commands work on supported platforms:

```bash
mana-agent worker install --coordinator https://agent.example.com --token '<one-use-token>'
mana-agent worker status
mana-agent worker logs
mana-agent worker doctor
mana-agent worker reconnect
mana-agent worker uninstall --yes
```

- macOS uses an owner LaunchAgent.
- Linux uses `systemd --user`, restarts on failure with a bounded delay, and
  reads logs from the user journal.
- Windows uses an explicit least-privilege Task Scheduler installation. State
  is stored below `LOCALAPPDATA` and restricted with Windows ACLs.

No service definition contains the enrollment token, worker credential, or
private identity key. Production coordinators require HTTPS/WSS.

## CLI and events

Use `mana-agent fleet --help` for worker, job, comparison, log, artifact,
cancellation, and verification commands. `fleet verify` requires explicit
platforms and argv commands; it never exposes a generic interactive shell.

Persistent schedules use the same command/service:

```bash
mana-agent automation create \
  --name "Nightly cross-platform tests" \
  --action fleet-verify \
  --cron "0 2 * * *" \
  --target local \
  --root-dir . \
  --platform linux \
  --platform windows \
  --platform macos \
  --verify-command "python -m pytest -q"
```

Events are ordered and persisted with a sequence cursor. Event data carries
run, job, task, session, workspace, repository, worker, and provider identities.
Logs are bounded before persistence. Dashboard and API clients replay with
`after_sequence` to avoid duplicates.

Exact permission scopes are:

```text
fleet.workers.read        fleet.workers.manage
fleet.verify.read         fleet.verify.execute
fleet.verify.cancel       fleet.artifacts.read
fleet.workspace.retain    fleet.worker.revoke
fleet.remote.write
```

Permission grants bind the repository, commit, workers, and argv commands. A
changed action has a different action key and needs a new grant.

## Configuration

```text
MANA_FLEET_ENABLED=false
MANA_FLEET_MAX_WORKERS_PER_RUN=4
MANA_FLEET_MAX_CONCURRENT_JOBS=4
MANA_FLEET_CAPABILITY_TTL_SECONDS=300
MANA_FLEET_HEARTBEAT_TIMEOUT_SECONDS=90
MANA_FLEET_JOB_TIMEOUT_SECONDS=1800
MANA_FLEET_WORKSPACE_MAX_LIFETIME_SECONDS=3600
MANA_FLEET_MAX_LOG_BYTES=1048576
MANA_FLEET_MAX_ARTIFACT_BYTES=104857600
MANA_FLEET_RETAIN_DAYS=30
MANA_FLEET_AUTO_REPAIR_ENABLED=false
MANA_FLEET_REQUIRE_TRUSTED_LABEL=true
```

Fleet state uses atomic owner-only files under `~/.mana/fleet/`. Existing
execution, SSH, Eval, worker, CLI, and chat behavior is unchanged while Fleet
is disabled.

## Security limitations

Fleet workers are trusted code-execution environments, not security sandboxes.
Provider isolation and network claims are accepted only when the selected
execution provider reports and enforces them. Do not enroll personal machines
into a shared coordinator without reviewing repository and artifact policy.
Direct SSH continues to require strict host-key checking and is never silently
changed into reverse-worker execution.
