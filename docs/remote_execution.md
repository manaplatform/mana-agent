# Sandbox-safe remote SSH execution

When Mana-Agent's runtime sandbox cannot open outbound TCP connections, a reverse-connected worker can execute SSH. The worker opens an authenticated WebSocket or HTTPS long-poll connection to the coordinator; the coordinator never calls into the worker.

`key_path` is only a local reference resolved by the worker. Mana-Agent never reads, uploads, returns, or logs private-key contents; `agent` authentication uses the worker's SSH agent. Register with a one-time coordinator enrolment token, then store only the opaque credential:

Ask Mana-Agent to register, start, or stop a worker in chat. The entry-routing
model selects the typed `remote-worker` lifecycle action; no dedicated shell
subcommand performs worker control.

Worker credentials are owner-readable only. Revoke a worker at the coordinator to invalidate its credential. Targets default to `prompt_each_time`; approvals bind worker, host, port, remote user, key identity, and exact command. Unknown or changed host keys require explicit handling; strict host-key checking remains enabled.

Interactive shells, forwarding, transfers, writes, and privileged actions require distinct permissions. Full access is explicitly scoped and never bypasses the permission model. Authentication failures never fail over; only classified sandbox transport restrictions can choose a worker. Disconnected workers transition active jobs to `worker_disconnected`.
