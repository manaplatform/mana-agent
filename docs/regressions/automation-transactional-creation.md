# Automation creation rejected by the transactional gate

On 2026-11-26, the automation chat route selected `automation_create` for a scheduling request, but AskAgent rejected the call with `no registered transactional action adapter`. No automation was created. The route's earlier tests invoked its tool directly and therefore did not exercise the model-tool enforcement gate.

The creation tool was classified as a side effect without an adapter, even though the automation route advertised it. Creation now has a dedicated action adapter: the model-selected arguments and task identify the action, the gateway evaluates policy before invoking the tool, and a fresh read of the durable definition must match the returned snapshot before commit. GitHub deployment requires exact approval; the preview includes redacted arguments. A successful action records the scheduler's actual deployment status, which may still be blocked or pending. Other automation mutations remain fail-closed until they have their own adapters and verification contracts. A persisted definition alone does not prove the scheduled job ran.


A subsequent user-run AskAgent regression failed after it persisted the definition: ToolRunner removes `prompt` fields from model-facing results, so hashing that projection could not match the full persisted record. The adapter now fingerprints a fresh authoritative read and checks a second read against it and the selected typed request. The regression covers an omitted prompt and a different requested job. The user supplied the failure; the correction has not been rerun.

**Do not reintroduce:** Do not allow automation mutations through the model-tool gate by classification alone or test only direct tool calls. Preserve the action gateway, policy, deduplication and persisted-state verification on the real AskAgent path. Do not compare a filtered model-facing result with an unfiltered stored definition.

The change is unverified until the user runs:

```sh
python -m pytest -q tests/gateway/test_automation_transactional.py tests/gateway/test_automation_route.py
python -m pytest -q tests/test_ask_agent.py -k automation_create_through_transactional_adapter
```
