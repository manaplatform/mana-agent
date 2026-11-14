"""Native callable bindings. Inference receives specifications, never handlers."""
from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, get_type_hints
from pydantic import BaseModel, ConfigDict, create_model
from mana_agent.multi_agent.tools.operation_policy import validate_arguments

if TYPE_CHECKING:
    from mana_agent.tools.contracts import ToolContract


@dataclass
class ToolBinding:
    name: str
    description: str
    handler: Callable[..., Any] | None = field(repr=False)
    input_model: type[BaseModel] | dict[str, Any]
    async_handler: Callable[..., Any] | None = field(default=None, repr=False)
    output_model: type[BaseModel] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    contract: ToolContract | None = None
    return_direct: bool = False

    @property
    def input_schema(self) -> dict[str, Any]:
        return self.input_model if isinstance(self.input_model, dict) else self.input_model.model_json_schema()

    @classmethod
    def from_callable(cls, func: Callable[..., Any] | None = None, *, name: str | None = None,
                      description: str | None = None, input_model: Any = None,
                      coroutine: Callable[..., Any] | None = None, metadata: dict[str, Any] | None = None,
                      return_direct: bool = False, output_model: type[BaseModel] | None = None,
                      contract: ToolContract | None = None) -> ToolBinding:
        function = func or coroutine
        if function is None:
            raise ValueError("A native tool requires a registered implementation")
        tool_name = name or function.__name__
        if input_model is None:
            signature = inspect.signature(function)
            hints = get_type_hints(function)
            fields = {}
            for key, parameter in signature.parameters.items():
                if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
                    raise ValueError(f"{tool_name}: variadic tools require an explicit input schema")
                fields[key] = (hints.get(key, Any), ... if parameter.default is parameter.empty else parameter.default)
            input_model = create_model(f"{tool_name}Input", __config__=ConfigDict(extra="forbid"), **fields)
        return cls(tool_name, description or inspect.getdoc(function) or tool_name, func, input_model,
                   coroutine, output_model, dict(metadata or {}), contract, return_direct)

    def specification(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.input_schema}

    def _arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validate_arguments(arguments, self.input_schema)
        if isinstance(self.input_model, dict):
            return dict(arguments)
        return self.input_model.model_validate_json(json.dumps(arguments), strict=True).model_dump()

    def _output(self, value: Any) -> Any:
        if self.output_model is not None:
            value = self.output_model.model_validate(value, strict=True).model_dump(mode="json")
        if self.contract is not None and self.contract.output_schema:
            rendered = json.loads(value) if isinstance(value, str) else value
            validate_arguments(rendered, self.contract.output_schema, "result")
        return value

    def call(self, arguments: dict[str, Any], *, config: dict[str, Any] | None = None) -> Any:
        values = self._arguments(arguments)
        run_id = str((config or {}).get("tool_call_id") or uuid.uuid4().hex)
        observers = list((config or {}).get("callbacks") or [])
        for observer in observers:
            observer.tool_started({"name": self.name}, json.dumps(values, ensure_ascii=False), run_id=run_id)
        try:
            if self.handler is None:
                raise RuntimeError(f"{self.name} is async-only; use its async call entry point")
            result = self._output(self.handler(**values))
        except BaseException as exc:
            for observer in observers:
                observer.tool_failed(exc, run_id=run_id)
            raise
        for observer in observers:
            observer.tool_finished(result, run_id=run_id)
        return result

    async def call_async(self, arguments: dict[str, Any], *, config: dict[str, Any] | None = None) -> Any:
        if self.async_handler is None:
            return await asyncio.to_thread(self.call, arguments, config=config)
        values = self._arguments(arguments)
        run_id = str((config or {}).get("tool_call_id") or uuid.uuid4().hex)
        observers = list((config or {}).get("callbacks") or [])
        for observer in observers:
            observer.tool_started({"name": self.name}, json.dumps(values, ensure_ascii=False), run_id=run_id)
        try:
            result = self._output(await self.async_handler(**values))
        except BaseException as exc:
            for observer in observers:
                observer.tool_failed(exc, run_id=run_id)
            raise
        for observer in observers:
            observer.tool_finished(result, run_id=run_id)
        return result


def tool_specification(tool: Any) -> dict[str, Any]:
    if isinstance(tool, ToolBinding):
        return tool.specification()
    from mana_agent.tools.contracts import ToolContract
    if isinstance(tool, ToolContract):
        return {"name": tool.name, "description": tool.description, "parameters": tool.input_schema}
    if isinstance(tool, dict):
        if tool.get("type") and tool["type"] != "function":
            return dict(tool)
        function = tool.get("function", tool)
        if not function.get("name") or not isinstance(function.get("parameters"), dict):
            raise ValueError("Tool specifications require a registered name and JSON Schema")
        return {key: value for key, value in function.items() if key != "type"}
    raise TypeError("Inference accepts only native registered tool specifications")
