# Reverse workers

Mana workers connect **outbound** to the coordinator over HTTPS/WSS. They do not open SSH, HTTP, or other inbound listener ports. The coordinator enrolls a host with a short-lived, one-use token; the host generates its own Ed25519 key locally and exchanges the token for an opaque worker credential stored in macOS Keychain when available (otherwise an owner-only state file).

## Coordinator

Enable the worker gateway with a public HTTPS URL. A reverse proxy must forward both `/api/v1/workers/enroll` and WebSocket upgrades on `/api/v1/workers/connect`. Do not terminate TLS with certificate verification disabled.

For Nginx, enable HTTP/1.1 upgrades and forward `Upgrade` and `Connection` headers for the connect path. For Caddy, a normal `reverse_proxy` handles WebSocket upgrades automatically.

## Installation

Create a short-lived enrollment token on the coordinator, then run on the target Mac:

```bash
mana-agent worker install --coordinator https://agent.example.com --token '<short-lived-token>' --name office-mac
```

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

After authentication, workers send a signed, bounded Fleet capability inventory.
Runtime facts are probed locally; coordinator-assigned trust labels are not
self-asserted. See [Mana Fleet](24-fleet.md).

HTTP/WS is refused except when explicitly enabling local-development mode for `localhost`. Production deployments require HTTPS/WSS, certificate validation, a reverse proxy that preserves WebSocket upgrades, and short enrollment TTLs.
