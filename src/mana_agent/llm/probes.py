"""Explicit settings-only capability probes. No chat path imports this service."""
from __future__ import annotations

import asyncio
import uuid
from typing import Literal
from pydantic import BaseModel, ConfigDict
from mana_agent.llm.capabilities import CapabilityState, observation
from mana_agent.llm.messages import UserMessage
from mana_agent.llm.registry import GLOBAL_CAPABILITY_REGISTRY
from mana_agent.llm.types import InferenceRequest


class ProbePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Literal["mana-capability-ok"]


class ProbeResult(BaseModel):
    capability: str
    state: CapabilityState = CapabilityState.UNKNOWN
    diagnostic: str = ""
    request_id: str


async def run_capability_probe(engine, capability: str, *, user_initiated: bool,
                               consent: bool, effort: str = "", tool_choice: str = "required",
                               cancel: asyncio.Event | None = None) -> ProbeResult:
    """One minimal inference, explicitly authorized by the model-settings user."""
    if not user_initiated or not consent:
        raise PermissionError("Capability probes require user initiation and explicit settings consent")
    if capability not in {"tool_calling", "tool_choice", "native_structured_output", "tool_call_streaming", "reasoning_effort"}:
        raise ValueError("Unknown capability probe")
    if capability == "tool_choice" and tool_choice not in {"auto", "required", "none", "function"}:
        raise ValueError("Select a supported probe mode from auto, required, none, or function")
    from mana_agent.approval.models import ApprovalRequest, ApprovalDecision
    from mana_agent.approval.policy import get_approval_policy
    from mana_agent.approval.audit import record_approval_audit
    identity = f"probe_{uuid.uuid4().hex}"
    approval = ApprovalRequest(action_type="api", action_id=identity, tool_name="model_capability_probe",
        intent={"provider": engine.provider, "model": engine.model, "capability": capability, "max_requests": 1, "max_output_tokens": 256},
        metadata={"user_initiated": True, "harmless_synthetic_tool": True})
    policy = get_approval_policy()
    decision = policy.evaluate(approval)
    if not decision.approved:
        if not decision.interactive or decision.source not in {"requires_approval", "default_approval_required"}:
            raise PermissionError("The approval policy denied this capability probe")
        # The settings consent switch + explicit Test button is the human response.
        decision = ApprovalDecision(approved=True, mode=policy.mode, source="human", interactive=True,
            approval_id=identity, reason="User explicitly initiated this bounded model-settings probe")
        record_approval_audit(decision, approval, result_status="human_approved")
    request = InferenceRequest(provider=engine.provider, model=engine.model,
        messages=[UserMessage("For this capability test, return value mana-capability-ok. If a function is offered, call only mana_capability_echo with that value.")],
        parameters={**engine.parameters, "max_output_tokens": 256, "n": 1}, request_id=identity)
    if getattr(engine, "endpoint", "") == "responses":
        request.parameters.pop("n", None)
    elif getattr(engine, "endpoint", "") == "anthropic_messages":
        request.parameters.pop("n", None)
    if capability in {"tool_calling", "tool_choice", "tool_call_streaming"}:
        request.tools = [{"name": "mana_capability_echo", "description": "Harmless synthetic schema; this function is never executed.", "parameters": ProbePayload.model_json_schema()}]
    if capability == "tool_choice":
        request.tool_choice = {"type": "function", "name": "mana_capability_echo"} if tool_choice == "function" else tool_choice
    if capability == "native_structured_output":
        request.response_schema = ProbePayload.model_json_schema()
    if capability == "reasoning_effort":
        if not effort:
            raise ValueError("Select an explicit provider reasoning effort value before testing")
        request.parameters["reasoning_effort"] = effort
    for key in ("max_tokens", "max_completion_tokens"):
        request.parameters.pop(key, None)
    # The synthetic single-choice setting only bounds the probe. All other
    # provider settings still participate in the evidence scope.
    scoped_parameters = dict(request.parameters)
    scoped_parameters.pop("n", None)
    if "n" in engine.parameters:
        scoped_parameters["n"] = engine.parameters["n"]
    profile = engine.profile(scoped_parameters)
    request.configuration_fingerprint = profile.configuration_fingerprint
    if cancel is not None and cancel.is_set():
        raise asyncio.CancelledError()
    if engine.context_cost_governor is None or not engine.context_cost_governor.enabled:
        raise PermissionError("Capability tests require the existing context/spending governor")
    reservation = engine._reserve(request)
    adapter = engine._adapter()
    result = ProbeResult(capability=capability, request_id=identity)
    async def execute():
        if capability == "tool_call_streaming":
            argument_delta = False
            final = None
            async for event in adapter.stream(request):
                argument_delta |= event.type == "tool.arguments.delta" and bool(event.delta)
                if event.type == "completed":
                    final = event.response
            if not argument_delta or final is None:
                raise ValueError("Provider did not demonstrate incremental tool arguments and a completed response")
            return final
        return await adapter.generate(request)
    task = asyncio.create_task(execute())
    cancel_task = asyncio.create_task(cancel.wait()) if cancel is not None else None
    try:
        waiters = [task] + ([cancel_task] if cancel_task is not None else [])
        done, _ = await asyncio.wait(waiters, timeout=min(engine.timeout, 30), return_when=asyncio.FIRST_COMPLETED)
        if cancel_task in done:
            raise asyncio.CancelledError()
        if task not in done:
            raise asyncio.TimeoutError("Capability probe timed out")
        response = await task
        if reservation:
            engine.context_cost_governor.record_model_call(reservation, usage=response.usage.model_dump(),
                provider=engine.provider, model=engine.model, estimated_input=[m.content for m in request.messages])
            reservation = ""
        if capability == "tool_choice" and tool_choice == "none":
            if response.tool_calls or not isinstance(response.content, str) or response.content.strip() != "mana-capability-ok":
                raise ValueError("Provider did not return the expected text without a tool call")
        elif capability in {"tool_calling", "tool_choice", "tool_call_streaming"}:
            if len(response.tool_calls) != 1 or response.tool_calls[0].name != "mana_capability_echo":
                raise ValueError("Provider did not return the exact synthetic tool call")
            ProbePayload.model_validate(response.tool_calls[0].args, strict=True)
        elif capability == "native_structured_output":
            ProbePayload.model_validate_json(response.content, strict=True)
        elif response.configuration_observation.get("reasoning_effort") != effort:
            raise ValueError("The provider did not observably confirm the requested reasoning configuration")
        result.state = CapabilityState.SUPPORTED
        result.diagnostic = "Capability demonstrated by one bounded provider request"
    except asyncio.CancelledError:
        if reservation:
            engine.context_cost_governor.release_reservation(reservation, reason="Capability probe cancelled; no automatic replay")
        raise
    except Exception as exc:
        body = getattr(exc, "body", {}) or {}
        error = body.get("error", body) if isinstance(body, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
        parameter = error.get("param") if isinstance(error, dict) else None
        expected = {"tool_calling": {"tools"}, "tool_choice": {"tool_choice"},
            "native_structured_output": {"response_format", "text.format", "output_config.format"},
            "reasoning_effort": {"reasoning_effort", "reasoning.effort", "output_config.effort"},
            "tool_call_streaming": {"stream"}}[capability]
        if getattr(exc, "status_code", None) in {400, 422} and code in {"unsupported_parameter", "unsupported_feature"} and parameter in expected:
            result.state = CapabilityState.UNSUPPORTED
        result.diagnostic = f"{type(exc).__name__}; HTTP {getattr(exc, 'status_code', 'unavailable')}; capability remains {result.state.value}"
        if reservation:
            engine.context_cost_governor.release_reservation(reservation, reason=result.diagnostic)
    finally:
        for pending in (task, cancel_task):
            if pending is not None and not pending.done():
                pending.cancel()
        await asyncio.gather(*(pending for pending in (task, cancel_task) if pending is not None), return_exceptions=True)
        if engine.adapter is None:
            await adapter.close()
    if result.state is not CapabilityState.UNKNOWN:
        if profile.observations:
            profile = profile.model_copy(deep=True)
            for name, observations in profile.observations.items():
                profile.observations[name] = [fact for fact in observations if fact.method != "user_configuration"]
                if profile.facts.get(name) is not None and profile.facts[name].method == "user_configuration":
                    profile.facts.pop(name)
        # A rejection of one level or choice mode says nothing about other levels.
        if capability in {"reasoning_effort", "tool_choice"}:
            scoped_capability = f"{capability}.{effort if capability == 'reasoning_effort' else tool_choice}"
        else:
            scoped_capability = capability
        value = [effort] if capability == "reasoning_effort" else [tool_choice] if capability == "tool_choice" else True
        fact = observation(value if result.state is CapabilityState.SUPPORTED else False,
            source="probe", scope=profile.configuration_fingerprint, method="schema_constrained_api" if capability == "native_structured_output" else "synthetic_tool_api" if request.tools else "observable_provider_configuration", confidence=1)
        facts = {scoped_capability: fact}
        if result.state is CapabilityState.SUPPORTED:
            if scoped_capability != capability:
                facts[capability] = fact.model_copy(update={"value": value})
            demonstrated = "tool_calling" if response.tool_calls else "reasoning" if capability == "reasoning_effort" else None
            if demonstrated and demonstrated != scoped_capability:
                facts[demonstrated] = observation(True, source="probe", scope=profile.configuration_fingerprint,
                    method=fact.method, confidence=1)
        GLOBAL_CAPABILITY_REGISTRY.merge(profile, facts, persist=True)
    return result
