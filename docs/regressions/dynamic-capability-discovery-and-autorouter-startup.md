# Dynamic Capability Discovery and AutoRouter Startup Fix

## 1. Symptom and Trigger

When starting `mana-agent` with third-party or unlisted models (such as `openrouter/~deepseek/deepseek-flash-latest`), chat startup immediately aborted with the following failure:

```text
Chat startup failed.
No configured model satisfies the routing capability,
reliability, latency, and budget constraints.

Rejected candidates:
openrouter/~deepseek/deepseek-flash-latest

Reasons:
- Text or token-limit evidence is absent or expired.
- Required capabilities are unsupported: structured_output.
- No fallback action was executed.
```

### Scope and Reproduction Conditions
- Triggered when configuring a primary chat or routing model that does not have pre-cached descriptor facts in the local model cache or whose provider does not report native `structured_output` support.
- Triggered because `resolve_model_for_role` mandated `structured_output` for `AgentRole.MAIN` and `AgentRole.HEAD_DECISION`, and the router required token limit facts even for unconstrained tasks (`assignment_only`).

### Expected Behavior
- Missing or stale capability metadata should trigger dynamic provider discovery and/or authorized minimal probes.
- Ordinary chat startup and AutoRouter should function using any model capable of text generation, utilizing multi-strategy decision output (native structured output, tool-calling output, or schema-validated text decisions).
- Rejections must be informative and actionable, detailing exact capability states (`SUPPORTED`, `UNSUPPORTED`, `UNKNOWN`, `STALE`), probe authorization status, and recovery commands.

### Actual Behavior
- Models without explicit pre-cached facts were treated as permanently unsupported.
- Ordinary chat failed to open because native `structured_output` was an invariant requirement for the main role.
- OpenRouter and other candidate providers were skipped during provider metadata refresh.

---

## 2. Root Cause Analysis

1. **Mandatory Native Structured Output on Main Role:**
   `resolve_model_for_role` in `src/mana_agent/multi_agent/runtime/model_levels.py` unconditionally added `structured_output` to `required_capabilities` for `AgentRole.MAIN` and `AgentRole.HEAD_DECISION`. Similarly, `_route_entry_turn` and multi-task planning in `src/mana_agent/gateway/chat_gateway.py` enforced `required_capabilities=frozenset({"structured_output"})`.
2. **Binary Missing-as-Unsupported Assumption:**
   In `ModelProfile.fact(cap)` and `_winner()`, absent or expired facts collapsed into `UNKNOWN` with no lifecycle differentiation, and router candidate evaluation rejected models lacking pre-verified context limits even when no token limits were requested (`assignment_only` tasks).
3. **Incomplete Provider Discovery in Gateway Routing:**
   `_refresh_configured_metadata` in `GatewayRoutingAuthority` only checked `mana_ai_provider` (which defaults to `"openai"`), ignoring third-party providers configured via `mana_model_profiles` (e.g. `openrouter`).
4. **Hardcoded Fallbacks in Profile Helpers:**
   `profiles_for_pinned_models` and `profiles_from_legacy_configuration` in `src/mana_agent/model_routing/profiles.py` guessed static defaults (`16_384` context, `4_096` max output tokens, `can_tool_call=True`, `can_structured_output=True`) when descriptors were missing, masking real discovery gaps while failing when descriptors were inspected.
5. **Single Output Strategy in Native Engine:**
   `NativeLLMEngine.decide()` strictly called OpenAI Responses or structured outputs API, lacking fallback strategies for models with only tool-calling or text-generation capabilities.

---

## 3. Permanent Fix

1. **4-State Evidence Lifecycle (`SUPPORTED`, `UNSUPPORTED`, `UNKNOWN`, `STALE`):**
   - Added `CapabilityState.STALE = "STALE"` to `src/mana_agent/llm/capabilities.py`.
   - Fact expiration transitions facts to `STALE` with timestamp and policy tracking.
   - Helper methods `is_stale`, `is_unknown`, `is_unsupported` distinguish proven absence from missing information.
2. **Dynamic Provider Discovery & Candidate Refresh:**
   - Updated `GatewayRoutingAuthority._refresh_configured_metadata` to collect all providers from pinned profiles, default providers, and qualified model IDs (`provider/...`).
   - Added `ModelDiscoveryService.refresh_model` (async and sync) in `src/mana_agent/llm/discovery.py` to refresh model catalog records and extract catalog pricing.
   - Added `_ensure_candidate_evidence` in `src/mana_agent/model_routing/router.py` to automatically refresh stale/unknown candidate metadata and run authorized runtime probes before evaluating eligibility.
3. **Multi-Strategy Decision Output in Native Engine:**
   - Refactored `decide()` and `decide_sync()` in `src/mana_agent/llm/engine.py`:
     1. **Native Structured Output:** Used when `native_structured_output` is supported or strictly required (`require_native_structured_output=True`).
     2. **Tool-Calling Output:** Emits a synthetic `record_decision` function call and unpacks arguments when `tool_calling` is supported.
     3. **Validated Text Decision:** Appends strict JSON schema instructions to prompt, strips markdown code fences, validates against schema, and performs a bounded 1-attempt repair when text generation is supported.
4. **Decoupled Chat Gateway and Main Role:**
   - Removed mandatory `structured_output` from `AgentRole.MAIN` and `AgentRole.HEAD_DECISION` in `src/mana_agent/multi_agent/runtime/model_levels.py`.
   - Updated chat gateway route entry turns to accept models with verified text generation or tool calling.
5. **Bounded, Permitted Capability Probes:**
   - In `src/mana_agent/llm/probes.py`, added per-(provider, model, capability) probe locks (`_get_probe_lock`) to prevent probe storms across concurrent requests.
   - Added support for `input_text` and `output_text` minimal probes.
   - Separated feature incompatibility (HTTP 400/422 → `UNSUPPORTED`) from transient or auth failures (HTTP 401/403/429/500 → `UNKNOWN`).
6. **Actionable Rejection Diagnostics:**
   - Updated `CandidateRejection` and `RoutingFailure` in `src/mana_agent/model_routing/models.py` with structured diagnostics: required capabilities, available capabilities, unknown capabilities, stale capabilities, unsupported capabilities, verification permission, and next recovery action.
   - Updated CLI startup error formatting in `src/mana_agent/commands/chat_cli.py`.

---

## 4. "Do Not Reintroduce" Rules

- **Do not mandate `structured_output` for `AgentRole.MAIN` or general chat routing.** Models with text generation or tool calling must remain eligible via validated text or tool-call decisions.
- **Do not treat missing capability evidence as confirmed `UNSUPPORTED`.** Missing metadata is `UNKNOWN` and expired evidence is `STALE`; both must permit discovery or authorized verification before rejection.
- **Do not guess context window limits or token limits.** Never restore static defaults (`16_384` / `4_096`) for unknown models. Models with unknown limits remain eligible for tasks with no token constraints (`assignment_only`), but must fail closed when explicit budget limits are enforced.
- **Do not silently swap models or downgrade strict native structured output requests.** If an operation explicitly sets `require_native_structured_output=True`, it must fail closed if native output is unsupported.
- **Do not run paid probes without authorization.** Check `mana_capability_probe_permissions` and concurrency locks before dispatching network verification requests.

---

## 5. Verification Commands (User-Owned)

Under repository Rule 40, test execution is user-owned. To verify this fix, execute:

```bash
python -m pytest tests/test_dynamic_capability_discovery.py -v
```

Additional targeted suites:
```bash
python -m pytest tests/test_model_routing.py tests/llm/test_contracts.py tests/test_model_capabilities.py -q
```

## Entry routing reason regression

On 2026-11-16, OpenRouter completed an entry routing call for `deepseek/deepseek-v4.1-flash`. Mana then stopped the turn with `routing reason is required`. The saved activity establishes the validation failure. It does not contain the returned field value.

`EntryRoutingOutput` declared `reason` as a required string and accepted empty or whitespace-only values. The gateway trimmed the field and rejected the result after decoding, and its correction guidance omitted this validation error. A complete routing decision requires a brief top-level reason on every route.

The [gateway router](../../src/mana_agent/gateway/entry_routing.py) now advertises a nonempty string and strips whitespace before native decision validation. The native engine can ask for one correction using the selected model and response format. An unstructured router adapter can use the gateway's existing correction to supply a missing or blank reason. A decision with a persistently invalid reason stops the turn before dispatch. The change affects model output validation and requires no migration of saved decisions.

**Do not reintroduce** an unconstrained routing reason or a synthesized explanation that lets an incomplete model decision dispatch.

Run `venv/bin/python -m pytest -q tests/gateway/test_entry_routing.py tests/llm/test_engine.py -k entry_route_reason` to check missing and blank reasons, correction through each native response strategy, and terminal failure after one correction. Then restart Mana and resend the original request with the same routing model. The agent did not run the tests or replay the live request.

## Expired capability evidence before entry routing

On 2026-11-17, a chat turn stopped at `entry_route` with `openrouter/deepseek/deepseek-v4.1-flash: output_text=STALE`. The local capability records for the model had expired. A recent model picker cache still listed the model with text output and token limits, but that list was not bound to the inference connection and could not renew its capability evidence.

`NativeLLMEngine.decide()` chose the response format and checked capability evidence before calling `generate()`. With expired evidence, that check could reject the turn before generation reached its built-in catalog refresh. The refresh also used budget accessors that allow stale values, so it could skip renewal when only the token limits had expired.

The [native inference engine](../../src/mana_agent/llm/engine.py) now runs its existing metadata refresh before a decision chooses a response format. The refresh checks for current text and token limit evidence and keeps the existing catalog cache and connection scope. Inference still requires current evidence. Refresh only reads model metadata; it does not run capability probes or substitute another model. There is no configuration or cache migration.

**Do not reintroduce** decision capability checks ahead of metadata refresh, or treat a cached token limit as current solely because it has a numerical value.

Run `venv/bin/python -m pytest -q tests/llm/test_engine.py -k entry_route_refreshes_capability_evidence` to check same-model recovery, expired token limits, and failure before inference when text output is still unverified. Restart Mana and resend the original request to verify the live provider path. To recover on an older build, open `mana-agent --configure`, select OpenRouter, then use **Model capabilities → Refresh metadata** for `deepseek/deepseek-v4.1-flash`. The agent did not run the tests or replay the live request.

## Stale tool calling with a fresh catalog

At 22:18 Tehran time on 2026-11-19, entry routing stopped before inference
with `openrouter/anthropic/claude-haiku-5.5: tool_calling=STALE`. The model
catalog already advertised tools and matched the configured connection. The
registry merged those fresh facts first, then imported equivalent expired
facts from its saved connection profile. Deduplication kept the last copy
merged. The refresh skipped fetching another catalog because the cache was
recent. In addition, the preflight considered only text and token limits,
so a separately expired tool capability could skip refresh as well.

The registry now keeps the newest observation for equivalent facts. The
native decision preflight includes the required response capability, such
as tool calling for a composite decision on OpenRouter Messages. A fresh
cache can recover the same model without another catalog request. If
current metadata still does not confirm a required capability, routing
stops before inference. No cache migration is needed.

Do not let import order replace a newer capability fact. Include the
capability required by the decision format in preflight freshness checks.

```sh
venv/bin/python -m pytest -q tests/llm/test_registry.py tests/llm/test_engine.py -k 'older_equivalent_observations or refreshes_required_tool_calling or entry_route_refreshes_capability_evidence'
```

The saved catalog reproduced the failure before the fix. After the fix,
the same cache reported current support and entry routing succeeded with a
mock inference response on the configured model. Live inference remains
unverified.
