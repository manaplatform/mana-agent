"""Permission-bounded lazy capability manifest and active-set lifecycle."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from mana_agent.context_cost.estimator import estimate_tool_schema_tokens
from mana_agent.context_cost.models import ActiveCapabilitySet, CapabilityManifestEntry

CORE_CAPABILITIES = frozenset({
    "capability_search", "capability_load", "capability_unload", "context_read_artifact",
})

_MUTATION_NAMES = frozenset({
    "edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file",
    "create_file", "delete_file", "document_create", "document_update", "document_delete",
})


class CapabilityRegistry:
    def __init__(
        self,
        tools: Iterable[Any],
        *,
        allowed_names: Iterable[str] = (),
        core_tools: Iterable[Any] = (),
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        supplied = list(tools)
        self._tools = {str(tool.name): tool for tool in supplied}
        self._core_tools = {str(tool.name): tool for tool in core_tools}
        self._tools.update(self._core_tools)
        requested = {str(name) for name in allowed_names if str(name)}
        self._allowed = requested if requested else set(self._tools)
        self._allowed.update(name for name in CORE_CAPABILITIES if name in self._tools)
        self._event_callback = event_callback
        self.manifest = {name: self._entry(tool) for name, tool in self._tools.items()}
        self.active = ActiveCapabilitySet()
        self.pinned: set[str] = set()

    def _entry(self, tool: Any) -> CapabilityManifestEntry:
        name = str(tool.name)
        mutation = name in _MUTATION_NAMES or any(token in name for token in ("delete", "write", "update", "create", "apply"))
        description = " ".join(str(getattr(tool, "description", "") or "").split())[:240]
        category = name.split("_", 1)[0] if "_" in name else "general"
        return CapabilityManifestEntry(
            name=name,
            category=category,
            description=description,
            risk_class="mutation" if mutation else "read",
            permission_requirements=("existing_mutation_policy",) if mutation else ("existing_tool_policy",),
            factory_key=name,
            estimated_schema_tokens=estimate_tool_schema_tokens(tool),
            aliases=tuple(sorted({name.replace("_", " "), category})),
        )

    def initial(self, names: Iterable[str], *, include_core: bool = True) -> list[Any]:
        selected = {str(name) for name in names if str(name)} & self._allowed
        if include_core:
            selected.update(CORE_CAPABILITIES & self._allowed)
        self._set_loaded(selected, event_type="context.capabilities_loaded", reason="initial_model_decision")
        return self.bound_tools()

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        words = {part.casefold() for part in str(query).split() if part}
        scored: list[tuple[int, CapabilityManifestEntry]] = []
        for name, entry in self.manifest.items():
            if name not in self._allowed:
                continue
            haystack = " ".join((entry.name, entry.category, entry.description, *entry.aliases)).casefold()
            score = sum(1 for word in words if word in haystack)
            if score or not words:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [
            {
                "name": entry.name, "category": entry.category, "description": entry.description,
                "risk_class": entry.risk_class, "permission_requirements": list(entry.permission_requirements),
                "estimated_schema_tokens": entry.estimated_schema_tokens,
            }
            for _, entry in scored[: max(1, min(int(limit), 50))]
        ]

    def load(self, names: Iterable[str], *, step: int = 0) -> dict[str, Any]:
        requested = {str(name) for name in names if str(name)}
        unknown = sorted(requested - set(self.manifest))
        denied = sorted((requested & set(self.manifest)) - self._allowed)
        accepted = sorted(requested & self._allowed)
        before = self.active.schema_tokens
        previous_loaded = set(self.active.loaded)
        self.active.loaded.update(accepted)
        for name in accepted:
            self.active.last_used_step[name] = int(step)
        self._refresh_revision(before, previous_loaded)
        payload = {
            "loaded": accepted, "denied": denied, "unknown": unknown,
            "schema_token_delta": self.active.schema_tokens - before,
            "schema_tokens": self.active.schema_tokens,
            "schema_tokens_avoided": self.avoided_schema_tokens,
            "active_capabilities": sorted(self.active.loaded),
            "active_revision": self.active.revision,
        }
        self._emit("context.capabilities_loaded", payload)
        return payload

    def pin(self, names: Iterable[str], *, step: int = 0) -> None:
        valid = {str(name) for name in names if str(name)} & self._allowed
        if not valid:
            return
        self.pinned.update(valid)
        before = self.active.schema_tokens
        previous_loaded = set(self.active.loaded)
        self.active.loaded.update(valid)
        for name in valid:
            self.active.last_used_step[name] = int(step)
        self._refresh_revision(before, previous_loaded)

    def unpin(self, names: Iterable[str]) -> None:
        self.pinned.difference_update(str(name) for name in names if str(name))

    def unload(self, names: Iterable[str], *, reason: str = "model_requested") -> dict[str, Any]:
        removable = ({str(name) for name in names if str(name)} - CORE_CAPABILITIES) - self.pinned
        before = self.active.schema_tokens
        previous_loaded = set(self.active.loaded)
        removed = sorted(removable & self.active.loaded)
        self.active.loaded.difference_update(removed)
        for name in removed:
            self.active.last_used_step.pop(name, None)
        self._refresh_revision(before, previous_loaded)
        payload = {"unloaded": removed, "schema_token_delta": self.active.schema_tokens - before, "schema_tokens": self.active.schema_tokens, "schema_tokens_avoided": self.avoided_schema_tokens, "active_capabilities": sorted(self.active.loaded), "reason": reason, "active_revision": self.active.revision}
        self._emit("context.capabilities_unloaded", payload)
        return payload

    def unload_idle(self, *, step: int, idle_steps: int) -> dict[str, Any]:
        expired = [name for name, last in self.active.last_used_step.items() if name not in CORE_CAPABILITIES and name not in self.pinned and int(step) - last >= max(1, int(idle_steps))]
        return self.unload(expired, reason="idle")

    def mark_used(self, name: str, step: int) -> None:
        if name in self.active.loaded:
            self.active.last_used_step[name] = int(step)

    def bound_tools(self) -> list[Any]:
        return [self._tools[name] for name in sorted(self.active.loaded) if name in self._tools]

    def schema_tokens_for(self, names: Iterable[str]) -> int:
        return sum(self.manifest[name].estimated_schema_tokens for name in set(names) if name in self.manifest)

    @property
    def avoided_schema_tokens(self) -> int:
        return max(0, self.schema_tokens_for(self._allowed) - self.active.schema_tokens)

    def _set_loaded(self, names: set[str], *, event_type: str, reason: str) -> None:
        before = self.active.schema_tokens
        previous_loaded = set(self.active.loaded)
        self.active.loaded = names & set(self._tools)
        self.active.last_used_step = {name: 0 for name in self.active.loaded}
        self._refresh_revision(before, previous_loaded)
        self._emit(event_type, {"loaded": sorted(self.active.loaded), "schema_token_delta": self.active.schema_tokens - before, "schema_tokens": self.active.schema_tokens, "schema_tokens_avoided": self.avoided_schema_tokens, "active_capabilities": sorted(self.active.loaded), "reason": reason, "active_revision": self.active.revision})

    def _refresh_revision(
        self, previous_schema_tokens: int, previous_loaded: set[str]
    ) -> None:
        new_tokens = self.schema_tokens_for(self.active.loaded)
        if (
            new_tokens != previous_schema_tokens
            or self.active.loaded != previous_loaded
        ):
            self.active.revision += 1
        self.active.schema_tokens = new_tokens

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_callback is not None:
            self._event_callback(event_type, payload)


def build_core_tools(
    registry_provider: Callable[[], CapabilityRegistry],
    artifact_reader: Callable[..., Any],
) -> list[Any]:
    """Create four lightweight controls without binding optional capabilities."""
    from langchain_core.tools import StructuredTool
    read_only_metadata = {"read_only": True, "side_effecting": False}

    def capability_search(query: str, limit: int = 20) -> str:
        return json.dumps(registry_provider().search(query, limit=limit), ensure_ascii=False, sort_keys=True)

    def capability_load(names: list[str]) -> str:
        return json.dumps(registry_provider().load(names), ensure_ascii=False, sort_keys=True)

    def capability_unload(names: list[str]) -> str:
        return json.dumps(registry_provider().unload(names), ensure_ascii=False, sort_keys=True)

    def context_read_artifact(
        artifact_ref: str,
        offset: int = 0,
        limit: int = 16000,
        line_start: int | None = None,
        line_end: int | None = None,
        json_path: str | None = None,
        section: str | None = None,
        record_start: int | None = None,
        record_count: int | None = None,
        search: str | None = None,
        query: str | None = None,
    ) -> str:
        value = artifact_reader(
            artifact_ref, offset=offset, limit=limit, line_start=line_start, line_end=line_end,
            json_path=json_path, section=section, record_start=record_start, record_count=record_count,
            search=search, query=query,
        )
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    return [
        StructuredTool.from_function(capability_search, name="capability_search", description="Search the authorized lightweight capability manifest. Does not load a tool.", metadata=read_only_metadata),
        StructuredTool.from_function(capability_load, name="capability_load", description="Load named authorized capabilities for the next model step. Never widens permissions.", metadata=read_only_metadata),
        StructuredTool.from_function(capability_unload, name="capability_unload", description="Unload named non-core capabilities to reclaim context.", metadata=read_only_metadata),
        StructuredTool.from_function(context_read_artifact, name="context_read_artifact", description="Read an exact scoped tool-result artifact by offset, line range, JSON path, section, records, or bounded search.", metadata=read_only_metadata),
    ]


__all__ = ["CORE_CAPABILITIES", "CapabilityRegistry", "build_core_tools"]
