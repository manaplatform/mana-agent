# API Manager

Mana-Agent's API Manager imports arbitrary external API documentation into normalized, reusable
integrations and calls saved operations through a controlled HTTP runtime. It does not expose a raw
unrestricted HTTP tool and does not add provider-specific routing.

## Architecture

```text
Authorized documentation
  → deterministic OpenAPI/Swagger parser or validated model semantic extraction
  → normalized integration registry under ~/.mana/api_manager/integrations
  → model-selected operation candidates
  → strict request builder and redacted preview
  → permission/approval gate
  → DNS-pinned HTTP executor
  → structured, redacted result and shared execution events
```

The implementation lives in `src/mana_agent/api_manager/`. The gateway registers an `api` entry
route and only these narrow tools:

```text
api_docs_import
api_integrations_list
api_integration_get
api_integration_update
api_integration_delete
api_operations_search
api_request_preview
api_request_execute
```

Every tool call requires the exact model decision ID and session ID. Invalid or missing decisions,
ambiguous operation choices, unresolved authentication, missing required inputs, and failed policy
checks stop without a fallback tool, host, credential, or operation.

## Supported documentation

- OpenAPI 3.x JSON and YAML
- Swagger 2.0 JSON and YAML
- Authorized local files within the active workspace
- Policy-controlled HTTP(S) documentation URLs
- Markdown, HTML, and plain or pasted text with a strict model-produced semantic definition

OpenAPI operations, parameters, request bodies, responses, servers, local `$ref` values, and
security schemes are normalized deterministically. External `$ref` values are rejected. Relative
servers are accepted only when an HTTP(S) documentation source gives them a safe base.

Unstructured documentation never uses heuristic endpoint extraction. The model must return the
typed semantic definition, cite every operation using the imported source or a URL present in that
source, identify only fields that were inferred (an empty list is valid for fully documented
operations), and leave undocumented required values and authentication unresolved. Parameters used
solely to carry authentication are normalized into authentication metadata instead of being exposed
as ordinary request inputs. Documentation content is treated as untrusted data; scripts and embedded
code are never executed.

## Importing and saving

From chat:

```text
Read this OpenAPI file and save it as “Acme CRM”.
Read https://docs.example.com/openapi.json and save the integration.
Import this pasted API documentation as an ephemeral one-time integration.
```

Formal specifications are imported directly. For prose, Mana-Agent first produces and validates a
semantic definition. Durable records use stable integration and operation identifiers and remain
available after CLI, gateway, or dashboard restart. Refreshing documentation appends a version
record; it does not silently replace the integration identity.

## Credentials and authentication

Supported authentication metadata:

- no authentication
- API key in a header
- API key in a query parameter
- bearer token
- basic authentication
- configured custom headers
- OAuth 2 flow/scopes and bearer credential references

Integration records contain only a `credential_reference`. The resolver accepts explicit
`env://<name>` or `mana-secret://<id>` references. For example, configure an integration's bearer
scheme with credential reference `env://ACME_CRM_TOKEN`, then provide that secret to the Mana process without
placing it in prompts or integration JSON:

```bash
export ACME_CRM_TOKEN='...'
```

Raw API keys, passwords, access/refresh tokens, and client secrets are rejected from normal
integration records. Resolved values are removed from previews, events, results, and exceptions.
OAuth browser authorization is represented but is not automatically performed.

Credential references must preserve their URI form on every import or update. `IPSTACK_TOKEN`, for
example, is rejected; use `env://IPSTACK_TOKEN` and provide the value to the Mana process separately.

## Calling an operation

```text
Get contact 123 from Acme CRM.
Update contact 123’s email to new@example.com.
```

For natural-language calls, the API route retrieves candidates from enabled integrations. A
structured model decision must select one of those candidates with sufficient confidence.
Mana-Agent asks only for genuinely missing required values, then validates path/query/header/cookie
parameters and the request body. Unknown fields are rejected unless the imported operation
explicitly permits them. The base URL always comes from the saved operation.

Read-only calls may execute after validation. Create, update, delete, and unknown/high-risk calls
first return a redacted preview and a session-bound approval request. The trusted TUI or dashboard
can approve that exact request once; the stored method, URL, headers, and body fingerprint must
still match before execution.

## Security and network policy

The executor:

- allows only HTTP(S), with HTTPS required by default
- blocks URL credentials
- blocks localhost, loopback, private, link-local, multicast, reserved, metadata-service, and other
  non-global addresses by default
- resolves and validates every redirect target
- pins the socket connection to the validated DNS result while preserving TLS hostname validation
- limits redirects and response size
- bounds connect/read time
- retries only documented retry statuses and transport failures, and never retries non-idempotent
  mutations unless their policy explicitly permits it
- supports cancellation and structured JSON, text, binary metadata, and managed file results

Trusted internal APIs are an administrator configuration, never something documentation can
expand. Configure exact hosts or CIDR networks in `~/.mana/config.toml`:

```toml
MANA_API_MANAGER_TRUSTED_INTERNAL_HOSTS = "crm.internal.example"
MANA_API_MANAGER_TRUSTED_INTERNAL_NETWORKS = "10.40.0.0/16"
MANA_API_MANAGER_ALLOWED_HOSTS = "api.example.com,crm.internal.example"
MANA_API_MANAGER_ALLOW_HTTP = false
MANA_API_MANAGER_MAX_REDIRECTS = 3
MANA_API_MANAGER_MAX_RESPONSE_BYTES = 10485760
```

Use the smallest possible allowlist. A trusted host exception permits non-global DNS results for
that exact hostname; a trusted network exception permits only the listed CIDR.

## Observability

Imports, integration lifecycle changes, operation selection, validation, approval requests, calls,
retries, rate limits, completion, and failures publish structured events to the shared
CLI/API/dashboard event stream. Safe events include integration ID, operation ID, method, redacted
host/path, status, latency, and routing evidence where available. They never include credential
values or raw sensitive bodies.

## Current limitations

- OAuth 2 browser authorization and refresh-token exchange are metadata-only.
- Remote external OpenAPI `$ref` resolution is intentionally unsupported; bundle references into a
  single authorized document.
- Multipart input supports explicit in-memory fields/file content. It does not accept arbitrary
  local paths from API documentation.
- Streaming responses are consumed incrementally under the response-size limit; model-visible
  results remain bounded rather than exposing an open byte stream.
