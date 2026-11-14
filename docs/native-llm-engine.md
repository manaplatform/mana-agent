# Native inference and capability evidence

Status: local implementation; acceptance is unverified. No tests, builds, linters,
compatibility installations, benchmarks, or paid API requests were run by the agent.
The migration inventory records baseline usages and replacements in
[native-llm-engine-migration.md](native-llm-engine-migration.md).

## Ownership and transports

`mana_agent.llm` contains JSON/Pydantic messages, content parts, tool calls, usage,
requests, responses, continuation receipts, streaming events and callable bindings.
OpenAI uses native Responses, Anthropic uses Messages, and OpenRouter, NVIDIA,
xAI and configured custom endpoints use Chat Completions. Credentials, base URLs,
headers, query parameters and explicit provider request settings stay isolated.
SDK retries are disabled. Only bounded connection-establishment retries may repeat
the same request; malformed decisions and interrupted streams never change format,
provider, model or coding backend.

The engine performs inference only. Controllers and lanes select actions; the
Autonomous Loop executes model decisions, TaskBoard records task progress, and
Supervisor owns attempts, leases, fencing, receipts and recovery. Tool bindings
contain schemas and handlers; model requests contain only specifications. Existing
ToolRunner authorization, approvals, path confinement and locks remain execution
boundaries. External Codex retains its entire inner coding loop.

Native streams assemble text and tool arguments in order and retain call IDs.
Tool execution consumes complete, validated calls, never argument fragments.
Activity and Coding Activity reuse their existing consolidated event surfaces.
No private reasoning text enters visible messages or durable checkpoints.

OpenAI stateless continuation retains complete public protocol items and encrypted
reasoning receipts separately from visible history. Hosted programmatic code stays
in the provider's isolated runtime. Mana executes only returned registered function
calls, through existing gates, and returns their original call IDs and program
caller IDs. A program output without a final assistant message is not completion.

Anthropic thinking with tools requires signed provider thinking blocks. These are
kept only in bounded volatile process memory (32 receipts, ten minute expiry),
while checkpoints contain an opaque receipt ID. Process restart or expiry requires
Supervisor reconciliation and an explicit fresh continuation decision; private
thinking is never written to disk. This limitation must be included in runtime
acceptance. Unsupported provider output or modality blocks fail explicitly.

Strict OpenAI response schemas encode unconstrained JSON-map/Any fields as declared
JSON strings on the wire, decode them once, then validate the native schema. This
keeps arbitrary tool-input dictionaries representable without a plain-text retry.
Missing or invalid native decisions stop the affected operation.

## Configuration workflow

1. Run `mana-agent configure`. Select a provider and save its credentials and URL.
   Test/refresh its catalog, then select Main, Coding and Tool models. No repository
   embedding selection is needed. The selector shows discovered IDs with missing
   capability evidence so they can be configured explicitly. A metadata refresh
   keeps saved selections and requires a deliberate first Main-model selection.
2. Open Capabilities. Enter the exact upstream model ID, including its namespace.
   Inspect evidence and refresh metadata. Discovery follows bounded pagination and
   registered detail endpoints. models.dev enriches exact provider/model matches;
   it never establishes account access or changes a configured endpoint.
3. Supply explicit capability overrides in the draft, then save configuration.
   Unknown capabilities block the requiring operation and point back to settings.
4. To test a feature, choose one probe and explicitly consent to its bounded cost.
   Test performs one request, with no retries, and supports cancellation. Tool tests
   expose only a synthetic function which is never executed. The spending governor
   requires evidence-backed token limits and applicable pricing before authorizing
   a request. Chat never runs probes.

Example override (persisted in `~/.mana/config.toml` through settings):

```json
{
  "provider": "custom",
  "model_id": "organization/exact-model-id",
  "endpoint": "https://your-provider.example/v1",
  "capabilities": {
    "input_text": true,
    "context_window": 32768,
    "max_output_tokens": 4096,
    "tool_calling": true,
    "tool_choice": ["auto", "required", "function"],
    "native_structured_output": true
  }
}
```

These numbers are illustrative operator settings, not defaults or verified model
limits. Override key: `MANA_MODEL_CAPABILITY_OVERRIDES`. Settings-created overrides
also carry their exact configuration scope. A bare model ID cannot grant features.
Reasoning support does not establish effort levels. Tool calling does not establish
choice modes, parallel calls or argument streaming. Text streaming does not establish
tool streaming. A probe for one rejected effort/mode does not reject all others.

Each capability has SUPPORTED, UNSUPPORTED or UNKNOWN state, source, method,
observation/expiry time, confidence and scope. Precedence is explicit override,
applicable conclusive probe, provider metadata, models.dev, then UNKNOWN. Conflicting
observations remain inspectable. Metadata expires after 24 hours and probes after
seven days. Expired evidence remains visible but cannot grant features. Probe scopes
hash provider, exact model, endpoint/protocol, API version, configuration, header/query
revisions, credential revision, deployment context and probe version. Secrets are not
stored in scope records. A key or configuration change makes old probes inapplicable.

Authentication, throttling, timeout, cancellation, malformed output and transient
failures are diagnostics, not evidence that a feature is unsupported. Reasoning
probes inspect public accepted-configuration metadata, never request hidden thoughts.
A provider without observable confirmation leaves that test UNKNOWN.

The Main model bootstraps selection directly from its explicit configuration. It
receives eligible candidates and evidence without recursive routing. Deterministic
validation enforces candidate identity, capability requirements, combined budgets,
independent verification, parallel permissions, isolation and conflicts. It does not
choose a cheaper or stronger model through a static scoring table.

## Retrieval and state migration

Repository search reads live files through existing text/read/symbol tools, without
indexes or credentials. Workspace search defaults to `text`; `semantic` is retired
and returns an instruction to use text/file/symbol search. `index` commands retain
chunk, symbol and manifest analysis metadata without vector generation. Independent
memory-provider search and media integrations retain their separate contracts.
Directory-mode chat searches discovered subproject roots even when analysis metadata
is absent. The existing `--max-indexes` flag caps the number of repository scopes.
Image encoding uses explicit `image_request_format` evidence (`images`,
`legacy_square`, or `legacy_single`) rather than guessing from model names.
`image_flexible_dimensions`, `image_quality_levels`, and `image_input_fidelity`
record optional image parameter support. Configure these per exact model/endpoint
when provider metadata omits them; existing image codecs and transports are retained.

Configuration schema 3 removes old embedding keys. Native JSON history upgrades
legacy `human`/`ai` role records without loading a framework object. Orphaned tool
results, unsupported serialized objects, unknown continuation versions and changed
continuation identities/configurations fail explicitly. Coding continuation records
native history, pending call IDs and protocol receipts. Pending side effects require
Supervisor receipt reconciliation before another model turn; they are never blindly
replayed.

The explicit cleanup action removed ten recognized files (five obsolete FAISS index
sets) under Mana-owned registered repository indexes. No pickle was deserialized.
The cleanup tool recognizes only retired vector files and the owned embeddings cache,
rejects symlink escapes, and preserves unrelated files and analysis records. To clean
additional authorized project roots, the user may run:

```bash
python scripts/purge_repository_vectors.py --root /absolute/path/to/project
```

## User-owned acceptance

Start with:

```bash
python -m pytest -q tests/llm tests/test_model_capabilities.py tests/test_model_routing.py tests/test_repository_tools.py tests/test_project_search.py
python -m pytest -q
python -m pip check
```

Then inspect hierarchy, approvals, recovery and consolidated activity:

```bash
python -m pytest -q tests/gateway/test_execution_hierarchy.py tests/coding/test_tool_runner_execution.py tests/execution_supervisor tests/execution/test_live_steering_integration.py tests/test_full_access_mode.py tests/test_configuration_session.py tests/test_index_incremental.py tests/test_chat_background_index.py tests/test_openai_batch_tools.py
python scripts/audit_native_dependencies.py --extras dev,full --output /tmp/mana-native-dependency-audit.json
```

Clean wheel installation, installed CLI smoke, resolved dependency audit, tests,
and optional sdist/twine checks are prepared as a user-run tool. Each extra and
Python interpreter gets a fresh environment. Missing interpreters and failures are
recorded as blocked/failed, never accepted. The installed CLI smoke runs outside
the source checkout, without PYTHONPATH. Tests then explicitly use the local source.

```bash
python scripts/verify_native_llm.py --python python3.10 --python python3.11 --python python3.12 --python python3.13 --python python3.14 --python python3.15 --all-extras --full-suite --build --output-dir /tmp/mana-native-acceptance
```

Run the corresponding available interpreters on Windows and macOS as well as Linux.
Use `--include-linux-desktop` only on a machine provisioned for that extra's native
input headers and display dependencies. Other platform markers remain in package
metadata. Python 3.15 is added to CI without triggering it. `requires-python` is
unchanged; actual compatibility requires user-supplied results. No transitive graph,
optional integration, packaged artifact or platform is claimed verified here.

## Repeatable benchmarks

Baseline commit: `6de4ad79be7745eea50e5f6727726b8dd5b69a17`. Use a separate baseline
checkout and its own clean environment; never install historical dependencies into
the native environment. The user creates that checkout/environment and runs the
current benchmark script against both sources:

```bash
python scripts/benchmark_native_llm.py --python /absolute/baseline/venv/bin/python --source-root /absolute/baseline/checkout --implementation baseline --label 6de4ad79 --runs 20 --output /tmp/mana-before.json
python scripts/benchmark_native_llm.py --python /absolute/native/venv/bin/python --source-root /absolute/native/checkout --implementation native --label native --runs 20 --compare /tmp/mana-before.json --output /tmp/mana-after.json
```

The harness records subprocess startup, CLI import time, traced Python peak memory,
repository dispatch, request construction, synthetic stream event assembly and JSON
context serialization. Use identical machines and Python versions. Traced allocations
are not process RSS; synthetic stream assembly is not live first-token latency.
Network streaming latency remains explicitly unmeasured and requires separately
authorized controlled/live transport measurements. No timing or improvement is claimed.

## Regression record: do not reintroduce

Baseline source inspection established overlapping capability tables, permissive
unknown profiles, name-based feature guesses, deterministic routing winners, malformed
structured/plain retries, framework serialization patches that removed task `status`,
and pickle-based repository vectors. The failed invariant was that framework/provider
convenience metadata could stand in for explicit model evidence and application state.
The replacement uses native schemas, scoped provenance, one validated Main selection,
lossless JSON worker state and direct repository search. Controlled-response coverage
lives in `tests/llm`; existing hierarchy/lifecycle tests remain acceptance gates.

Do not reintroduce a framework execution path, provider/model/backend fallback,
unknown-as-supported defaults, implicit paid probes, schema/text retry, global SDK/HTTP
monkey patches, recursive `status` stripping, vector pickles, or embedding-required
configuration. Do not treat a successful inference/tool result as verified task success.
These fixes remain unverified until the user's audit and test results establish them.

Repository text search now excludes symlinked files and known credential
filenames, and redacts matches before they enter model context. This closes a
path that became exposed to ordinary repository Q&A when live search replaced
vector retrieval. Keep that confinement on every repository search path.
The regression is covered by
`test_repository_search_ignores_external_symlinks_and_local_credential_files`.
