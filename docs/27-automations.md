# Unified automations

Mana-Agent exposes one authoring concept: an Automation. Users create, update,
enable, disable, delete, or run one through the model-driven `automation` chat
route. The public CLI and dashboard only list, inspect, and delete records.

## Canonical contract

`AutomationDefinition` is the versioned source of truth. It contains ownership,
source (`chat`, `teach`, or `migration`), an explicit IANA timezone, one typed
trigger, one typed job, retry/misfire policy, permission references, deployment
state, next/last run timestamps, and a bounded recent execution summary.

Triggers are `cron` for calendar schedules, `interval` for exact elapsed
seconds from an anchor, or `once` for an absolute instant. Jobs are
`agent_prompt`, `connector_action`, `tool_action`, `teach_flow`, or an explicitly
requested argv-based `command`. Definitions reject embedded secret values.

## Persistence and migration

The store lives under
`~/.mana/repositories/<repository-id>/automations/config.json`. Writes use an
inter-process lock, fsync, and atomic replacement. Version-2 `schedules`,
dashboard `automations`, and `runs` migrate into canonical records
idempotently. Valid IDs and run history are retained. Invalid legacy records
remain visible under `migration_errors`; they are never discarded silently.

## Deployment and execution

Platform adapters install a hidden persistent wakeup (cron, launchd, Windows
Task Scheduler, or an explicitly selected repository-safe GitHub workflow).
Unavailable backends produce a persisted `blocked` state; there is no
in-process scheduler fallback.

Every wakeup invokes the hidden ID entrypoint:

```text
mana-agent automation execute --automation-id <id>
```

The executor atomically claims a lease, reloads the full record, creates a run,
reconstructs a fresh headless runtime, executes the typed job, records bounded
redacted output, advances the next run, emits automation lifecycle events, and
releases the lease. Duplicate invocations cannot claim the same run.

## Teach Mode

Teach Mode exposes a secret-free automation handoff only after the exact flow
version has been reviewed and successfully verified. Automations pin that
`flow_id` and `flow_version`; later flow edits do not change production jobs.
