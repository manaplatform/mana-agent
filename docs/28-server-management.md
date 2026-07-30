# Server Management

Mana-Agent manages explicitly enrolled Linux servers through the existing
OpenSSH execution fabric. Server operations are model-selected, schema
validated, capability checked, approval gated, host-key pinned, serialized per
server for mutations, streamed as execution events, and written to a redacted
audit log.

## Security model

Server state is stored under `~/.mana/servers/` (or the configured `MANA_HOME`):

```text
servers/registry.json   non-secret enrollment metadata
servers/audit.jsonl     append-only redacted action evidence
servers/ssh-control/    bounded OpenSSH connection-control sockets
```

Private-key contents, passwords, tokens, database connection strings, and
secret-bearing environment values must not be placed in registry records,
plans, decisions, logs, or chat messages. Key authentication stores only an
authorized filesystem path behind a `secret://server/<id>` reference. Password
and token SSH authentication require an installed secret-provider adapter and
otherwise fail without trying another credential.

Mana-Agent always uses `StrictHostKeyChecking=yes`. Enrollment requires a
pinned fingerprint and known-hosts file. An unexpected key change is rejected;
review it as a possible interception or server-rebuild event before explicitly
trusting a replacement key.

## Enrollment

Create and verify an SSH profile, review the displayed fingerprint, and enroll
it:

```bash
mana-agent ssh add production-api --host 203.0.113.10 --user deploy --identity ~/.ssh/id_ed25519
mana-agent ssh trust-host production-api
mana-agent ssh test production-api
mana-agent server enroll production-api --server-id api-1 --mode inspect_only --capability inspect
mana-agent server list
mana-agent server status api-1
mana-agent server authorize api-1 --capability package.write
```

Use `--use-agent` instead of `--identity` to keep authentication entirely in
the system SSH agent. Agent forwarding is never enabled implicitly. Jump hosts,
keepalive, ports, connection concurrency, and provider identity are fields on
the enrolled `ServerDefinition` and are resolved before transport execution.

For administrative use, enroll with an explicit mode and the smallest required
capability set. Examples include `package.write`, `service.write`,
`filesystem.read`, `filesystem.write`, `network.read`, `firewall.write`,
`database.backup`, `database.restore`, `container.read`, `container.write`,
`deployment`, `power`, and `shell`.

Add capabilities to an existing enrollment only through the explicit
`server authorize` command. It validates capability names against the installed
typed tool catalog and asks for confirmation unless `--yes` is supplied.

Consequential chat actions return a session-bound one-time approval request
with the exact command preview. The TUI opens an approval modal and the dashboard
renders an inline approval card; both offer only **Deny** and **Approve once**.
There is no text-command approval fallback. Approval resumes only that exact
decision and cannot be reused or submitted from another session.

## Operating modes

- `inspect_only` permits only decisions whose registered tool contract is
  read-only.
- `managed_admin` permits typed administrative tools with exact approval for
  consequential actions.
- `trusted_admin` additionally permits `server_shell_execute` and interactive
  login shells. Authentication, host-key verification, approvals, redaction,
  cancellation, locks, and audit recording remain active.

The entry model must return a complete `ServerActionDecision`. The gateway then
validates the selected tool, action type, capability, read/write claim,
consequential/destructive classification, affected resources, recovery plan,
verification commands, enrolled identity, and operating mode. Missing, invalid,
unsafe, unauthorized, or contradictory decisions stop with no fallback tool,
server, command, agent, or credential.

## Inspection and audit

```bash
mana-agent server inspect api-1
mana-agent server logs api-1 --limit 200
mana-agent server remove api-1
```

The CLI `inspect` command shows safe enrollment metadata. Ask in chat for live
health collection; the `server_inspect` tool records exact evidence from the OS,
load, memory, disks, failed systemd services, and listening sockets. Registry and
audit data are also available to the dashboard through `/api/v1/servers` and
`/api/v1/servers/{server_id}/audit` without exposing credential references or
host-key paths.

Audit events contain the resolved target, decision and tool identity, exact
command, exit code, changed resources, approval ID, verification evidence,
rollback metadata, and timestamps. Output and structured payloads pass through
the shared redactor before persistence.

## Administration tools

The server tool catalog includes typed tools for connection and inspection,
shell execution and sessions, files and directories, processes, services,
packages, users and SSH keys, networking and firewall, logs and disks,
database backup/restore, containers, deployment, and power control. Each tool
has one action/capability/risk contract. The generic `server_shell_execute` tool
remains available only in `trusted_admin`, with its exact argv visible in audit
evidence.

Package helpers support explicit `apt`, `dnf`, `yum`, `pacman`, `apk`, `zypper`,
and Linux Homebrew decisions. If remote evidence finds no manager or multiple
plausible managers, package work stops for a model decision. Service helpers
support systemd, OpenRC, and SysV evidence without selecting a default.

File writes use a same-directory temporary file, permissions before rename, an
optional adjacent backup, and atomic replacement. Firewall plans must retain
the current enrolled SSH port and require verification through a second
connection. SSH daemon changes must similarly validate configuration and retain
the active connection before a restart is approved.

## Desired state, deployment, backup, and recovery

`ServerPlan` contains model-selected typed steps, affected resources, verification
commands, and rollback actions. The runner inspects drift, skips converged steps,
applies only drifted steps, verifies each step, and rolls back changed steps in
reverse order on failure. A drifted step without rollback metadata is rejected.

Deployment contracts require an exact repository revision, absolute release
root, secret references for environment values, atomic current-release symlink,
and an HTTP(S) health check. A release is not successful until that check passes;
the previous link is retained for rollback.

Backup records are valid only after SHA-256 verification. Database commands do
not accept credentials in argv; credentials are supplied by server-side secure
configuration or a non-output secret-provider adapter. Destructive restores,
shutdowns, and infrastructure deletion require exact identity, affected
resources, a recovery plan, explicit approval, and a verified recovery point.

## Cloud providers

The `ServerProvider` protocol is provider-neutral and covers region, size and
image discovery plus create, inspect, start, stop, reboot, resize, snapshot, and
delete operations. Hetzner, DigitalOcean, AWS EC2, and custom HTTP adapters use
injected authenticated transports so provider tokens never enter provider
models or logs.

Creation requires a `cost_approval_id` after showing provider, region, size,
image, and any available cost estimate. Deletion requires the exact provider
server ID and a snapshot name; no provider or lifecycle action is selected as a
fallback when configuration or approval is absent.

## Automations and protocols

Server jobs authored through chat reuse the existing automation scheduler. The
persisted job must contain credential references and a validated server decision,
and execution reuses the registry, permissions, locks, audit log, events, and
verification path. Do not create a parallel cron or scheduler.

The gateway advertises a first-class `server` route and tool catalog. Operations
run in the Operations lane, and A2A advertises `server-management` only when the
deployment enables that skill. Telegram, TUI, dashboard, Live Canvas, and other
gateway clients consume the same normalized lifecycle events.

## Troubleshooting

- `not enrolled`: enroll the exact profile/server ID; no host fallback occurs.
- `host-key fingerprint`: run `ssh trust-host`, independently verify the shown
  fingerprint, then enroll or explicitly update the enrollment.
- `capability is not authorized`: update the enrollment intentionally; changing
  prompt wording cannot bypass capabilities.
- `approval required`: review the target, exact operation, affected resources,
  recovery plan, and exact action key before approving.
- `runtime adapter unavailable`: install/configure the selected provider or tool
  adapter; Mana-Agent does not substitute a raw shell action.
- `multiple package managers`: have the model select one from current server
  evidence rather than relying on OS-name heuristics.
