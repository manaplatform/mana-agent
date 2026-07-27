# Dual remote execution modes

Mana-Agent supports two independent remote routes:

- `remote-ssh` is direct OpenSSH execution from the local Mana-Agent process. It needs no coordinator, worker daemon, WebSocket, or Mana-Agent installation on the target.
- `reverse-worker` sends work to an enrolled managed worker over its authenticated reverse connection.

Use `mana ssh add <name> --host <host> --user <user> --identity <path>` (or `--use-agent`) to save non-secret SSH metadata. Private key contents are never read or stored. `mana ssh trust-host <name>` displays a scanned fingerprint and requires explicit approval before changing known-hosts. Direct SSH uses `StrictHostKeyChecking=yes` and never silently becomes a worker route. Use `--ssh-only` to record that a target must not be bootstrapped as a worker.

The direct-SSH CLI supports `list`, `show`, `edit`, `remove`, `test`, `run`, `logs`, `doctor`, `upload`, and `download`. Chat routing uses the same `remote-ssh` contract and recognizes an explicit profile or explicitly supplied host, user, and authorized key/agent details. Tool actions are bound to `computer.ssh.connect`, `computer.ssh.execute`, `computer.ssh.read`, `computer.ssh.transfer`, profile mutations, and host trust permissions.

Managed workers remain the route for persistent reverse connectivity. They use an authenticated WebSocket or HTTPS long-poll connection to the coordinator; the coordinator never calls into the worker. Worker credentials are owner-readable only, and worker jobs remain distinct from SSH-only targets. Interactive shells, forwarding, transfers, writes, and privileged actions require distinct permissions and exact action approval.
