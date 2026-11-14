# Competition fields in startup routing

Starting `mana-agent` reaches role assignment before interactive chat opens. The
reported CLI error was `Competition candidates were supplied for a non-competition
mode`. That error requires a response that selects a non-parallel mode and also
contains a nonempty competition candidate list. Startup role assignment should
offer the model a single-execution contract it can satisfy without supplying
competition data.

After the competition field was removed, `mana-agent --full-access` still stopped
with `Main model selection failed (ValueError)`. The local routing journal showed
successful Main and Head Decision assignments, but the error wrapper did not
preserve the reason for the next failure. The rejected provider output was not
available for inspection.

The router used the same broad response schema for every decision. OpenAI strict
encoding marks every declared field required, so the provider schema required a
competition array even for a startup assignment. The prompt supplied candidate
IDs and broad feature flags without explaining when that array should be empty.
Runtime validation correctly rejected the contradictory response, which the CLI
reported as a parameter error.

Role initialization now asks the Main model for an eligible model ID, reason, and
confidence. The CLI reuses the gateway's tool-worker assignment and its selected
provider. Task routing still offers admitted execution modes according to policy,
request authority, isolation, concurrency, and eligible candidates. The competition
schema requires an empty array for non-parallel selections. The router rejects
extra fields, unadmitted modes, and contradictory decisions in one request.

OpenRouter Chat Completions sets `provider.require_parameters` on schema-constrained
requests so an endpoint that lacks the requested parameter is ineligible. This
addresses a documented way for a provider to return a 200 response without
enforcing the requested schema. It does not establish which failure occurred in
the reported call. JSON syntax and schema mismatches now produce safe diagnostics
that identify the role and omit raw provider output.
[OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection#requiring-providers-to-support-all-parameters)

**Do not reintroduce** execution-mode fields into role-assignment schemas or
reroute a tool worker that already has a gateway assignment. Leave candidate
selection to the Main model and reject outputs outside the offered authority.

User-owned verification is `python -m pytest -q tests/test_model_routing.py tests/llm/test_providers.py`,
followed by `mana-agent --full-access` from the updated checkout. The new cases
inspect the strict provider schema and cover invalid fields, diagnostic redaction,
and OpenRouter request mapping. The isolated competition case still covers an
admitted parallel decision. Live startup remains unverified.
