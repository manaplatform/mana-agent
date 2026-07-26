# Reverse workers

Mana workers connect **outbound** to the coordinator over HTTPS/WSS. They do not open SSH, HTTP, or other inbound listener ports. The coordinator enrolls a host with a short-lived, one-use token; the host generates its own Ed25519 key locally and exchanges the token for an opaque worker credential stored in macOS Keychain when available (otherwise an owner-only state file).

## Coordinator

Enable the worker gateway with a public HTTPS URL. A reverse proxy must forward both `/api/v1/workers/enroll` and WebSocket upgrades on `/api/v1/workers/connect`. Do not terminate TLS with certificate verification disabled.

For Nginx, enable HTTP/1.1 upgrades and forward `Upgrade` and `Connection` headers for the connect path. For Caddy, a normal `reverse_proxy` handles WebSocket upgrades automatically.

For an explicitly insecure HTTP development coordinator, set
`MANA_WORKER_GATEWAY_PUBLIC_URL` to its `http://` URL and set
`MANA_WORKER_GATEWAY_ALLOW_INSECURE_HTTP=true`. The worker must independently
opt in with `--allow-insecure-http`.

## Installation

Create a short-lived enrollment token on the coordinator, then run on the target Mac:

```bash
mana-agent worker install --coordinator https://agent.example.com --token '<short-lived-token>' --name office-mac
```

For a trusted development network without TLS, HTTP can be enabled explicitly:

```bash
mana-agent worker install --coordinator http://192.168.1.10:8000 \
  --token '<short-lived-token>' --allow-insecure-http
```

This also enables the worker's unencrypted `ws://` connection. Enrollment
tokens, worker credentials, job data, and results can be observed or modified
in transit, so do not use this option across an untrusted network.

The installer enrolls first and stores the identity separately. On macOS it
writes non-secret configuration under
`~/Library/Application Support/ManaAgent/` and installs
`net.manaplatform.mana-agent.worker` as a user LaunchAgent. Linux installs
`mana-agent-worker.service` with `systemd --user`. Windows installs an explicit
least-privilege Task Scheduler task and owner-restricted state under
`LOCALAPPDATA`. Service definitions contain neither token nor private
credential.

Use `mana-agent worker status`, `logs`, `doctor`, `reconnect`, and `uninstall
--yes` for lifecycle management. `doctor --repair` only applies the registered
platform service repair. Revocation and identity rotation should be initiated
by the coordinator; a revoked host must enroll again.

On macOS, `worker start` can reload an installed but unloaded LaunchAgent. If
the LaunchAgent has not been installed, it exits with an instruction to run
`worker install` instead of attempting enrollment or service creation.

After authentication, workers send a signed, bounded Fleet capability inventory.
Runtime facts are probed locally; coordinator-assigned trust labels are not
self-asserted. See [Mana Fleet](24-fleet.md).

HTTP/WS is refused unless `--allow-insecure-http` is explicitly selected.
Production deployments require HTTPS/WSS, certificate validation, a reverse
proxy that preserves WebSocket upgrades, and short enrollment TTLs.
