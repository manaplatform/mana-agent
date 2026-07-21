# Execution provider configuration and operations

The execution fabric is local-only by default. Existing users need no remote
configuration: trusted tool commands are explicitly routed to `local-process`.
Provider configuration contains credential references, never credential values.

```toml
MANA_EXECUTION_DEFAULT_PROVIDER = "local-process"
MANA_EXECUTION_ALLOWED_PROVIDERS = ["local-process", "local-docker", "remote-ssh", "kubernetes", "modal", "custom-http-runtime"]
MANA_EXECUTION_CLEANUP_ON_EXIT = true
MANA_EXECUTION_IDLE_TIMEOUT_SECONDS = 900
MANA_EXECUTION_MAX_LIFETIME_SECONDS = 7200

[MANA_EXECUTION_ROUTING]
high_risk_provider = "local-docker"
expensive_provider = "kubernetes"
gpu_provider = "modal"
deny_silent_fallback = true

[MANA_EXECUTION_PROVIDERS.local-process]
enabled = true

[MANA_EXECUTION_PROVIDERS.local-docker]
enabled = true
default_image = "python:3.12"
concurrency_limit = 4
```

The same nested objects may be supplied as JSON through their environment
variables. Run `mana-agent doctor` after changing provider configuration.

## Provider setup

- **Local process:** no setup. It has no network isolation and only best-effort
  resource controls, and therefore rejects isolation policies it cannot enforce.
- **Docker:** install a working Docker daemon. Deny-all maps to Docker's `none`
  network. Restricted egress and allowlists require a managed Docker network and
  are rejected by the built-in provider until one is configured. Disk quotas are
  storage-driver-specific and also fail closed.
- **SSH:** configure a `hosts` array with `hostname`, optional `user`, `port`,
  `known_hosts_file`, `identity_file`, `workspace_root`, and approved
  `resource_wrapper`/`network_wrapper` argv. Host-key checking is always on.
- **Kubernetes:** install the optional official `kubernetes` client and configure
  a runtime adapter that owns namespace, Pod/Job, SecurityContext, service
  account, Secret/ConfigMap, volume, NetworkPolicy, logs, snapshot, TTL, and
  owner-reference operations. Missing SDK or adapter is an actionable error.
- **Modal:** install the optional current `modal` SDK and configure the Modal
  runtime adapter. The adapter owns Image, Secret, Volume, GPU, timeout, parallel
  call, persisted-state, and cleanup operations.
- **Custom HTTP:** configure `base_url`, `credential_ref`, and optionally
  `signing_secret_ref`. Credential values are resolved at request time.

## Selecting a provider

Provider selection is part of the model-produced `RoutingRequest`; it is not
inferred from command keywords. An explicit Docker request sets
`explicit_provider="local-docker"`. GPU count, trust/risk, parallelism, network
enforcement, snapshots, suspension, and organization policy are structured
requirements. If the selected provider is unavailable or cannot enforce them,
the request fails; it is not re-run locally.

## Artifacts, snapshots, suspend, and restore

Artifacts must be declared in `SandboxSpec.artifact_paths`, then requested with
relative allowlisted paths. Absolute paths, traversal, and symlinks are rejected.
Results include SHA-256, size, MIME type, and provider-independent references.

Call `ExecutionManager.snapshot`, `suspend`, and `resume` with the persisted
`SandboxExecutionContext`. A restore validates schema/provider compatibility and
checksums. Emulated providers preserve the logical sandbox ID while recreating
their underlying runtime. Retained sandboxes are still lease-bound and recovered
by `CleanupController` after expiration.

## Troubleshooting

- `CapabilityMismatchError`: inspect rejected-provider reasons; enable a provider
  that supplies the requested enforcement rather than weakening the request.
- `ProviderConfigurationError`: install the optional SDK or complete its provider
  configuration. Private SSH key data must never be placed in configuration.
- `CleanupError`: the original task failure remains on the handle and the cleanup
  failure is stored separately. Retry the reaper after fixing provider access.
- A local network-policy failure is expected: `local-process` does not claim
  isolation. Use Docker or an approved remote runtime.
