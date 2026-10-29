# Model-aware token accounting

Mana-Agent uses `mana_agent.context_cost` as the single accounting boundary for
model-backed execution. Routing first selects the final provider and model. The
accounting service then resolves that exact identity, tokenizes the serialized
payload, checks model capacity, reserves task policy, and records the estimate
before the provider call. It never substitutes another model.

## Capacity, policy, and estimates

These values have different meanings and are kept separate:

- `context_window` and `max_output_tokens` are factual capabilities of the
  selected model.
- task, session, monetary, and optional lane caps are product or user policy.
- a run estimate describes the actual messages, system instructions, tool
  schemas, retrieved context, attachments, expected follow-up calls, output
  allowance, and named safety margin.
- actual usage is provider-reported when available and tokenizer-estimated when
  it is not.

The effective run limit is the most restrictive applicable model or policy
limit. Output is also checked separately against the model output limit. A
request that does not fit reports the deficit; the governor may remove only
reversible context selected for fitting and recalculates after each change.
System instructions, current user intent, required tool schemas, approval state,
and transactional safety context are protected.

## Model metadata and unknown models

Provider catalog metadata is preferred, followed by maintained OpenAI-family
token limits for common `gpt-4.1*`, `gpt-4o*`, `gpt-5*`, and `o3`/`o4` models
when the catalog omits windows, then configured model profiles and explicit
custom-deployment metadata. A model profile includes provider, model ID,
context window, maximum output, tokenizer, usage-reporting support, and
Decimal input/cached-input/output/reasoning prices.

Explicit `MANA_MODEL_PROFILES` entries must provide both `context_window` and
`max_output_tokens`. Custom providers should publish the same fields through
their model catalog or profile, plus `tokenizer` and pricing when known.

`MANA_CONTEXT_UNKNOWN_MODEL_POLICY=require_metadata` rejects a selected model
whose capability metadata is absent. `conservative` permits the explicitly
configured unknown-model context and output limits, labels the profile and
token estimate low-confidence, and leaves monetary cost unknown when pricing is
absent. Unknown cost is never treated as free.

Eval suite runs pin variant models via `ChatGatewayConfig.pin_models` so
operator `MODEL_LEVEL_*` / `MANA_MODEL_*` preferences cannot rewrite the
measured models or their token profiles.

## Reservation and reconciliation

### CLI context admission inspector

The CLI's opt-in `/context` or `/context last` displays the most recent
governor admission, `/context history` lists retained invocation identifiers,
and `/context show <invocation-id>` selects one. These commands work in the
classic chat CLI and local Textual chat, not remote API, dashboard, or Telegram
frontends. The display is quiet until requested. Recording occurs in memory
while the governor is enabled; the classic CLI clears it on a session switch.
The local Textual command only reads snapshots matching the active session,
workspace and repository; its shared governor does not currently re-scope on
session switches, so snapshots may be unavailable after switching. It keeps at
most 50 snapshots and 100 displayed segments per snapshot. Category totals
still include components beyond the display limit.

The Textual chat uses the shared command registry, unlike the classic CLI's
direct-command path. Previously `/context` was implemented only in the latter,
so the Textual dispatcher returned `Unsupported command` and omitted it from
`/help`. The registry now exposes the same scoped renderer locally. **Do not
reintroduce:** advertise or implement this local command in only one dispatch
path, or expose its process-local snapshots to remote frontends without an
authenticated authorization design. Regression coverage exercises the registry,
Textual input path, and scope and privacy boundaries.

The version 1 snapshot contains opaque, random invocation and segment IDs,
ordered categories, admission/removal and call lifecycle status, fixed safe
explanations, segment estimates, category totals, model context window,
accounting input estimate and allowance, and provider-reported input when
available. Model, turn, task and attempt correlation references are keyed,
session-local hashes, not the original identifiers; the key rotates on reset.
Snapshot access checks the exact nonempty session, workspace and
repository scope before returning data. This is a process-local diagnostic,
not an authenticated remote access API or role-authorization system. It stores
no prompt text, source identifiers, paths, user-supplied reason strings, or
tool payloads, and does not provide raw-content viewing or export. The existing
governor's separately persisted manifests and accounting records retain their
existing storage and retention behavior; this feature does not migrate them.

The inspector is an **admission view**, not an execution trace or proof of the
final request: downstream callers may submit a different payload. Marked
removals are governor admission compaction/fitting decisions, not evidence of
provider-side truncation. Upstream selection reasons are not currently exposed.
The compatible-chat and AskAgent adapters supply admission candidates; the
Codex boundary exposes only Mana's outer inputs. Codex internals, hidden
provider instructions, provider framing and unavailable multimodal costs are
`NOT_OBSERVABLE`. Calls that bypass this governor have no snapshot. Do not
infer missing segments or instructions from an absent snapshot.

In particular, AskAgent's fallback admission supplies tool names as schema
proxies rather than the complete bound schemas. Codex's admission includes a
contract placeholder, goal and outer prompt, which may overlap and are not a
breakdown of Codex's real model request. Neither boundary's category totals
should be interpreted as exact tool-schema or provider-input costs.

Token counts reuse the existing accounting tokenizer's component estimates;
when no estimate is available, caller estimates are labeled as such. The
accounting input estimate includes policy safety margin and possibly historical
allowances, shown separately from segment totals. Framing and multimodal costs
may be incomplete. Provider-reported aggregate input is not attributed to
segments, and its difference from the estimate is not a measurement of a
specific component. Unknown model/tokenizer behavior follows the existing
accounting fallback; none of these numbers is presented as exact. No extra model
calls or extra tokenization are performed.

This initial feature does not implement persisted snapshots, diff, exports,
content expansion, privilege levels, or future-context exclusion. It does not
change context selection, approval, routing, or provider request content. A
future implementation of those controls needs separate authorization, stable
source identity, dependency validation and retention design.

**Regression invariant:** Never label an admitted candidate as submitted, or
serialize raw segment content/source metadata into the inspector. The observed
gap was that the existing manifest recorded admitted sources, which do not prove
what a caller ultimately sent; the metadata-only admission view keeps that
distinction explicit. This invariant remains unverified until the user runs the
focused tests.

Each model call follows this lifecycle:

```text
final provider/model decision
→ explainable estimate
→ model and policy validation
→ durable reservation
→ provider execution
→ reported or estimated actual usage
→ idempotent reconciliation or release
```

## Adaptive task reservations and overrun finalization

The gateway recalculates an active lane reservation from the exact provider-call
forecast before that call is made. A revision is accepted only when the revised
token and cost forecast fits the existing task, lane, parent, session, global,
and monetary limits. Recalculation never chooses another model, tool, workflow,
or policy cap.

If a provider has already returned a durable result that exceeds an immutable
limit, Mana stores the result and moves the task to
`pending_budget_decision`. A fresh validated
`BudgetOverrunFinalizationDecision` must match the task, attempt, result, and
evidence hash. It may accept a verified result with an explicit overrun flag,
require human review, or request normal bounded retry/replan recovery. Missing,
invalid, stale, or unsafe decisions leave the result pending and block further
provider or tool execution. Use `/budget recalculate <task-id>` to inspect the
current forecast and durable revision history without invoking a provider. Use
`/budget finalize <task-id>` to request the fresh model finalization decision
for an existing durable overrun result. Pending decisions are reported to chat
surfaces as waiting work, never as a failed execution; a later accepted decision
projects the same durable result as completed.

Retries, fallback models, parallel candidates, agents, verifier calls, and tool
continuations receive distinct operation/attempt identities. Changing the model
therefore creates a fresh estimate and reservation. Failed and cancelled calls
release unused capacity but retain an audit status.

Records are atomically stored below `~/.mana/accounting/reservations` and contain
only identifiers, provider/model metadata, token component counts, costs,
confidence, assumptions, and lifecycle state. Prompt text, source content, tool
arguments, credentials, and raw provider payloads are not persisted. Matching
historical records may refine future estimates with a bounded percentile; they
are not required for correctness.

## Configuration

```bash
MANA_CONTEXT_ESTIMATION_SAFETY_MARGIN_RATIO=0.05
MANA_CONTEXT_DEFAULT_OUTPUT_RATIO=0.20
MANA_CONTEXT_HISTORICAL_PREDICTION_ENABLED=true
MANA_CONTEXT_UNKNOWN_MODEL_POLICY=conservative
MANA_CONTEXT_UNKNOWN_MODEL_CONTEXT_WINDOW=16384
MANA_CONTEXT_UNKNOWN_MODEL_MAX_OUTPUT_TOKENS=4096
```

The two unknown-model limits are an explicit compatibility policy, not claimed
capabilities for known models. Existing routing token budgets remain spending
limits. Lane `token_budget` and `cost_budget` fields are optional policy caps;
omitting them does not impose a synthetic model context window.
