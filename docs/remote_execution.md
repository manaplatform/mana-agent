# Dual remote execution modes

Mana-Agent supports two independent remote routes:

- `remote-ssh` is direct OpenSSH execution from the local Mana-Agent process. It needs no coordinator, worker daemon, WebSocket, or Mana-Agent installation on the target.
- `reverse-worker` sends work to an enrolled managed worker over its authenticated reverse connection.

Use `mana-agent ssh add <name> --host <host> --user <user> --identity <path>` (or `--use-agent`) to save non-secret SSH metadata for key-authenticated hosts. Private key contents are never read or stored.

For password-authenticated hosts, password mode is configured explicitly without exposing plaintext credentials in CLI arguments:
```bash
# Add a password-authenticated host with secure interactive prompt
mana-agent ssh add staging --host 192.0.2.10 --user ubuntu --password

# Or configure with an explicit secret reference (env://VAR or mana-secret://KEY)
mana-agent ssh add staging --host 192.0.2.10 --user ubuntu --auth-mode password --password-ref env://STAGING_SSH_PASS

# Or set/update the password securely anytime into ~/.mana/secrets.toml
mana-agent ssh set-password staging
```

`mana-agent ssh trust-host <name>` displays a scanned fingerprint and requires explicit approval before changing known-hosts. Direct SSH uses `StrictHostKeyChecking=yes` across both key and password authentication, and never silently switches authentication modes or becomes a worker route. Use `--ssh-only` to record that a target must not be bootstrapped as a worker.

The direct-SSH CLI supports `list`, `show`, `edit`, `remove`, `set-password`, `test`, `run`, `logs`, `doctor`, `upload`, and `download`. Chat routing uses the same `remote-ssh` contract and recognizes an explicit profile or explicitly supplied host, user, and authorized key/agent/password details. Tool actions are bound to `computer.ssh.connect`, `computer.ssh.execute`, `computer.ssh.read`, `computer.ssh.transfer`, profile mutations, and host trust permissions.

Managed workers remain the route for persistent reverse connectivity. They use an authenticated WebSocket or HTTPS long-poll connection to the coordinator; the coordinator never calls into the worker. Worker credentials are owner-readable only, and worker jobs remain distinct from SSH-only targets. Interactive shells, forwarding, transfers, writes, and privileged actions require distinct permissions and exact action approval.

Remote permission prompts are durable inbox items. The existing
`remote_permission_*` ID remains a compatibility handle, while the full typed SSH
request is stored only in owner-protected context and the notification discloses
only risk/category counts. A coordinator restart restores unresolved or approved
jobs from that record; concurrent or repeated reviewer responses still use the
inbox's single terminal transition. The selected provider is part of the exact
request: direct `local_ssh`/`remote-ssh` requests never switch to a worker, and
worker requests never switch to direct SSH when availability changes.
