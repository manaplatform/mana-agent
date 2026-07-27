# Custom HTTP runtime contract v1

Clients send `Mana-Runtime-Version: 1`, `Content-Type: application/json`, an
optional bearer token, `Idempotency-Key` for provisioning/snapshot/deletion, and
an optional `X-Mana-Signature-SHA256` HMAC over the exact request body. Servers
must return `Mana-Runtime-Version: 1` and typed JSON objects.

| Method | Endpoint | Response member |
| --- | --- | --- |
| POST | `/v1/sandboxes` | `sandbox: SandboxHandle` |
| POST | `/v1/sandboxes/{id}/start` | `sandbox: SandboxHandle` |
| POST | `/v1/sandboxes/{id}/execute` | `result: ExecutionResult`, or status plus `poll_path` |
| POST | `/v1/sandboxes/{id}/suspend` | `sandbox: SandboxHandle` |
| POST | `/v1/sandboxes/{id}/resume` | `sandbox: SandboxHandle` |
| POST | `/v1/sandboxes/{id}/snapshots` | `snapshot: SnapshotRef` |
| POST | `/v1/snapshots/{id}/restore` | `sandbox: SandboxHandle` |
| POST | `/v1/sandboxes/{id}/artifacts` | `artifacts: ArtifactResult[]` |
| GET | `/v1/sandboxes/{id}` | `sandbox: SandboxHandle` |
| DELETE | `/v1/sandboxes/{id}` | empty object or deletion status |
| GET | `/v1/health` | availability, message, and `SandboxCapabilities` |

`SandboxSpec` omits secret values and sends `secret_references` separately.
Servers must treat idempotency keys as stable operation identities. Execution may
be polled; terminal partial failures must retain command and cleanup diagnostics
as separate sanitized fields. Artifact payloads are references/metadata, not
unbounded inline files. HTTP errors are remote failures and must never trigger
local execution.
