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
    metadata: dict[str, Any] | None = None
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
            has_var_keyword = False
            for key, parameter in signature.parameters.items():
                if parameter.kind == parameter.VAR_POSITIONAL:
                    raise ValueError(f"{tool_name}: variadic tools require an explicit input schema")
                if parameter.kind == parameter.VAR_KEYWORD:
                    has_var_keyword = True
                    continue
                fields[key] = (hints.get(key, Any), ... if parameter.default is parameter.empty else parameter.default)
            extra_policy = "allow" if has_var_keyword else "forbid"
            input_model = create_model(f"{tool_name}Input", __config__=ConfigDict(extra=extra_policy, strict=True), **fields)
        meta = dict(metadata) if metadata is not None else None
        return cls(tool_name, description or inspect.getdoc(function) or tool_name, func, input_model,
                   coroutine, output_model, meta, contract, return_direct)

    def specification(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.input_schema}

    def _arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {"input": arguments}
        elif not isinstance(arguments, dict):
            arguments = {"input": arguments}
        if isinstance(self.input_model, type) and issubclass(self.input_model, BaseModel):
            validated = self.input_model.model_validate(arguments)
            return {k: getattr(validated, k) for k in type(validated).model_fields}
        validate_arguments(arguments, self.input_schema)
        return dict(arguments)

    def _output(self, value: Any) -> Any:
        if self.output_model is not None:
            value = self.output_model.model_validate(value, strict=True).model_dump(mode="json")
        if self.contract is not None and self.contract.output_schema:
            rendered = json.loads(value) if isinstance(value, str) else value
            validate_arguments(rendered, self.contract.output_schema, "result")
        return value

    def call(self, arguments: Any, *, config: dict[str, Any] | None = None) -> Any:
        values = self._arguments(arguments)
        run_id = str((config or {}).get("tool_call_id") or uuid.uuid4().hex)
        observers = list((config or {}).get("callbacks") or [])
        for observer in observers:
            observer.tool_started({"name": self.name}, json.dumps(values, ensure_ascii=False), run_id=run_id)
        try:
            if self.handler is None:
                if self.async_handler is not None:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            result = self._output(pool.submit(asyncio.run, self.async_handler(**values)).result())
                    else:
                        result = self._output(asyncio.run(self.async_handler(**values)))
                else:
                    raise RuntimeError(f"{self.name} is async-only; use its async call entry point")
            else:
                result = self._output(self.handler(**values))
        except BaseException as exc:
            for observer in observers:
                observer.tool_failed(exc, run_id=run_id)
            raise
        for observer in observers:
            observer.tool_finished(result, run_id=run_id)
        return result

    async def call_async(self, arguments: Any, *, config: dict[str, Any] | None = None) -> Any:
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

    def invoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        return self.call(input, config=config)

    async def ainvoke(self, input: Any, config: dict[str, Any] | None = None) -> Any:
        return await self.call_async(input, config=config)


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
