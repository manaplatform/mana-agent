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

Provider catalog metadata is preferred, followed by configured model profiles
and explicit custom-deployment metadata. A model profile includes provider,
model ID, context window, maximum output, tokenizer, usage-reporting support,
and Decimal input/cached-input/output/reasoning prices.

Explicit `MANA_MODEL_PROFILES` entries must provide both `context_window` and
`max_output_tokens`. Custom providers should publish the same fields through
their model catalog or profile, plus `tokenizer` and pricing when known.

`MANA_CONTEXT_UNKNOWN_MODEL_POLICY=require_metadata` rejects a selected model
whose capability metadata is absent. `conservative` permits the explicitly
configured unknown-model context and output limits, labels the profile and
token estimate low-confidence, and leaves monetary cost unknown when pricing is
absent. Unknown cost is never treated as free.

## Reservation and reconciliation

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
current forecast and durable revision history without invoking a provider.

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
