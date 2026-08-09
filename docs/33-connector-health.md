# Connector Health and Self-Healing

Mana-Agent treats connector health as the health of the full provider path, not
the liveness of a local process or gateway.

## States

| State | Meaning |
| --- | --- |
| `unknown` | Registered; not yet verified after startup/restart |
| `healthy` | Auth, transport, and required path probes succeeded |
| `degraded` | Partial path failure (e.g. ingress or subscription) |
| `recovering` | Automatic recovery in progress |
| `offline` | Path unavailable after failure threshold / circuit open |
| `auth_required` | Credentials expired/revoked; human reauth needed |
| `rate_limited` | Provider rate limit active; probes back off |
| `disabled` | Connector intentionally disabled |

A running process alone never yields `healthy`.

## Path signals

Health derives from independent signals:

- `runtime_alive`
- `transport_connected`
- `authenticated`
- `ingress_operational`
- `egress_operational`
- `subscription_operational`
- `acknowledgements_operational`

## CLI

```bash
mana-agent connectors status
mana-agent connectors status gmail
mana-agent connectors health telegram
mana-agent connectors incidents
mana-agent connectors recover gmail
mana-agent doctor --only connectors/health,connectors/credentials
```

## Configuration

Global defaults in `~/.mana/config.toml`:

```toml
[connector_health]
enabled = true
probe_interval_seconds = 60
failure_threshold = 3
recovery_enabled = true
max_recovery_attempts = 8
incident_retention_days = 30
synthetic_probe_mode = "passive"

[connectors.gmail.health]
probe_interval_seconds = 120

[connectors.telegram.health]
probe_interval_seconds = 30
synthetic_probe_mode = "passive"
```

## Persistence

Durable state lives under `~/.mana/connectors/`:

- `health/` — last snapshot per connector
- `incidents/` — open and recovered incident timelines
- `probes/` — bounded probe result logs
- `receipts/` — delivery receipts when reliability requires them
- `events.jsonl` — structured health events

Secrets are never persisted in health events.

## Integrations

- **Resilient Execution Supervisor** — connector-dependent branches checkpoint
  and wait (`waiting_for_connector`) instead of retrying into a known outage;
  resume exactly once when the connector is healthy again.
- **Durable HITL Inbox** — permanent auth failures create a minimal, redacted
  intervention item (reconnect / reauthorize).
- **Policy-gated transactional actions** — webhook re-registration and
  subscription mutations require policy authorization; local reconnects do not.
- **Context and Cost Governor** — routine probes are deterministic and do not
  call the model.

## Instrumented connectors

- **Gmail** — profile auth, safe metadata ingress probe, delivery-receipt egress/ack
- **Telegram** — `getMe`, poller/webhook ingress, webhook subscription probe

Synthetic user-visible messages are never sent by default.
