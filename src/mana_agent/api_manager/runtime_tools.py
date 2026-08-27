"""Narrow structured API tools exposed to the chat/gateway runtime."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mana_agent.api_manager.documentation import SemanticDefinition
from mana_agent.api_manager.discovery import ApiRouteDecision
from mana_agent.api_manager.models import AuthenticationConfig
from mana_agent.api_manager.service import ApiManagerService, safe_result
from mana_agent.api_manager.events import api_event_scope


_SERVICES: dict[str, ApiManagerService] = {}
_SERVICES_LOCK = threading.RLock()


class ApiToolExecutionContext(BaseModel):
    model_config = ConfigDict(extra="ignore")
    session_id: str = "default-session"
    conversation_id: str = ""
    turn_id: str = ""
    execution_id: str = ""
    lane_task_id: str = ""
    checkpoint_id: str = ""
    source_decision_id: str = ""


class _Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_decision_id: str = Field(default="", min_length=0)
    session_id: str = Field(default="", min_length=0)


class _WorkflowDecision(_Decision):
    task_intent: str = Field(min_length=1)
    required_actions: tuple[
        Literal[
            "documentation_inspection",
            "integration_import",
            "integration_configuration",
            "operation_search",
            "request_preview",
            "request_execution",
        ],
        ...,
    ] = Field(min_length=1)
    reason: str = Field(min_length=1)
    safe_to_continue: bool

    @field_validator("required_actions", mode="before")
    @classmethod
    def _normalize_required_actions(cls, v: Any) -> Any:
        if isinstance(v, str):
            return (v,)
        if isinstance(v, (list, set)):
            return tuple(v)
        return v

    @model_validator(mode="after")
    def validate_action_dependencies(self) -> "_WorkflowDecision":
        actions = self.required_actions
        if len(set(actions)) != len(actions):
            raise ValueError("required_actions must not contain duplicates")
        canonical_order = {
            "documentation_inspection": 0,
            "integration_import": 1,
            "integration_configuration": 2,
            "operation_search": 3,
            "request_preview": 4,
            "request_execution": 5,
        }
        if list(actions) != sorted(actions, key=canonical_order.__getitem__):
            raise ValueError("required_actions must follow the declared API lifecycle order")
        if "request_execution" in actions:
            missing = [
                action
                for action in ("operation_search", "request_preview")
                if action not in actions
            ]
            if missing:
                raise ValueError(
                    "request_execution requires declared actions: " + ", ".join(missing)
                )
        if "request_preview" in actions and "operation_search" not in actions:
            raise ValueError("request_preview requires a declared operation_search action")
        if "integration_import" in actions and "documentation_inspection" not in actions:
            raise ValueError(
                "integration_import requires a declared documentation_inspection action"
            )
        return self


class _WorkflowTerminal(_Decision):
    outcome: Literal["unsupported_documentation"]
    documentation_ref: str = Field(
        min_length=71,
        max_length=71,
        pattern=r"^sha256:[a-f0-9]{64}$",
    )
    reason: str = Field(min_length=1)
class _Import(_Decision):
    name: str = Field(min_length=1, max_length=160)
    text: str = Field(default="", max_length=10 * 1024 * 1024)
    path: str = ""
    url: str = ""
    documentation_ref: str = ""
    semantic_definition: SemanticDefinition | None = None
    save: bool = True
    ephemeral: bool = False
    refresh_integration_id: str = Field(
        default="", pattern=r"^(|api_[a-f0-9]{24})$"
    )

    @model_validator(mode="after")
    def exactly_one_source(self) -> "_Import":
        if sum(bool(item) for item in (self.text, self.path, self.url, self.documentation_ref)) != 1:
            raise ValueError("Select exactly one of text, path, url, or documentation_ref.")
        return self


class _SemanticImport(_Decision):
    name: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=10 * 1024 * 1024)
    documentation_reference: str = Field(min_length=1, max_length=2048)
    semantic_definition: SemanticDefinition
    save: bool = True
    ephemeral: bool = False
    refresh_integration_id: str = Field(
        default="", pattern=r"^(|api_[a-f0-9]{24})$"
    )


class _Inspect(_Decision):
    text: str = Field(default="", max_length=10 * 1024 * 1024)
    path: str = ""
    url: str = ""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=2000, ge=1, le=16_000)
    @model_validator(mode="after")
    
    def exactly_one_source(self) -> "_Inspect":
        if sum(bool(item) for item in (self.text, self.path, self.url)) != 1:
            raise ValueError("Select exactly one of text, path, or url.")
        return self


class _List(_Decision):
    include_disabled: bool = True


class _IntegrationId(_Decision):
    integration_id: str = Field(pattern=r"^api_[a-f0-9]{24}$")


class _Update(_IntegrationId):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = None
    enabled: bool | None = None
    authentication: list[AuthenticationConfig] | None = None


class _Delete(_IntegrationId):
    explicit: bool


class _Search(_Decision):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=10, ge=1, le=50)


class _Request(_Decision):
    integration_id: str = Field(pattern=r"^api_[a-f0-9]{24}$")
    operation_id: str = Field(min_length=1)
    path_parameters: dict[str, Any] = Field(default_factory=dict)
    query_parameters: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)
    body: Any = None
    credential_reference: str = ""
    content_type: str = ""
    routing_decision: ApiRouteDecision

    def request_kwargs(self) -> dict[str, Any]:
        return self.model_dump(
            exclude={"source_decision_id", "session_id", "routing_decision"},
        )


class _Execute(_Request):
    approval_reference: str = ""

    def request_kwargs(self) -> dict[str, Any]:
        return self.model_dump(
            exclude={
                "source_decision_id",
                "session_id",
                "approval_reference",
                "routing_decision",
            },
        )


def _json(
    operation: Any,
    *,
    root: str | Path,
    session_id: str,
    source_decision_id: str,
) -> str:
    with api_event_scope(
        session_id=session_id,
        execution_id=source_decision_id,
        root=root,
    ):
        return json.dumps(safe_result(operation), ensure_ascii=False, default=str)





def build_api_manager_langchain_tools(
    root: str | Path,
    *,
    service: ApiManagerService | None = None,
    context: ApiToolExecutionContext | None = None,
) -> list[Any]:
    resolved_root = str(Path(root).expanduser().resolve())
    manager = service or _service_for_root(resolved_root)
    bound_context = (
        context.model_copy(deep=True)
        if context is not None
        else ApiToolExecutionContext()
    )

    def _resolve_identities(values: dict[str, Any]) -> tuple[str, str]:
        raw_session_id = str(values.get("session_id") or "").strip()
        raw_decision_id = str(values.get("source_decision_id") or "").strip()

        authoritative_session_id = bound_context.session_id or "default-session"
        if raw_session_id and raw_session_id != authoritative_session_id:
            raise PermissionError(
                f"Model-provided session_id {raw_session_id!r} does not match host-bound session {authoritative_session_id!r}."
            )

        authoritative_decision_id = (
            bound_context.source_decision_id
            or bound_context.turn_id
            or bound_context.execution_id
            or raw_decision_id
            or "api-turn-decision"
        )
        if raw_decision_id and bound_context.source_decision_id:
            is_valid_suffix = (
                raw_decision_id == bound_context.source_decision_id
                or raw_decision_id.startswith(f"{bound_context.source_decision_id}:")
                or raw_decision_id.startswith(f"{bound_context.source_decision_id}/")
                or raw_decision_id.startswith(f"{bound_context.source_decision_id}-")
                or raw_decision_id.startswith(f"{bound_context.source_decision_id}_")
            )
            if not is_valid_suffix:
                raise PermissionError(
                    f"Model-provided source_decision_id {raw_decision_id!r} does not match host-bound source decision {bound_context.source_decision_id!r}."
                )

        values["session_id"] = authoritative_session_id
        values["source_decision_id"] = authoritative_decision_id

        if "routing_decision" in values and isinstance(values["routing_decision"], dict):
            rd_raw_id = str(values["routing_decision"].get("source_decision_id") or "").strip()
            if rd_raw_id and bound_context.source_decision_id:
                is_valid_rd_suffix = (
                    rd_raw_id == bound_context.source_decision_id
                    or rd_raw_id.startswith(f"{bound_context.source_decision_id}:")
                    or rd_raw_id.startswith(f"{bound_context.source_decision_id}/")
                    or rd_raw_id.startswith(f"{bound_context.source_decision_id}-")
                    or rd_raw_id.startswith(f"{bound_context.source_decision_id}_")
                )
                if not is_valid_rd_suffix:
                    raise PermissionError(
                        f"Model-provided routing_decision.source_decision_id {rd_raw_id!r} does not match host-bound source decision {bound_context.source_decision_id!r}."
                    )
            values["routing_decision"]["source_decision_id"] = authoritative_decision_id

        return authoritative_session_id, authoritative_decision_id

    def encode(operation: Any, *, session_id: str, source_decision_id: str) -> str:
        return _json(
            operation,
            root=resolved_root,
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def import_docs(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)

        last_inspection = manager.get_last_inspection(session_id)
        if last_inspection is not None:
            canonical_spec = last_inspection.get("canonical_spec_url")
            if canonical_spec:
                values["url"] = str(canonical_spec)
                values["text"] = ""
                values["path"] = ""
                values["documentation_ref"] = ""
            elif last_inspection.get("url") or (
                last_inspection.get("reference")
                and urlsplit(str(last_inspection.get("reference") or "")).scheme in {"http", "https"}
            ):
                values["url"] = str(last_inspection.get("url") or last_inspection.get("reference"))
                values["text"] = ""
                values["path"] = ""
                values["documentation_ref"] = ""
            elif last_inspection.get("path") or (
                last_inspection.get("reference")
                and Path(str(last_inspection.get("reference") or "")).is_file()
            ):
                values["path"] = str(last_inspection.get("path") or last_inspection.get("reference"))
                values["text"] = ""
                values["url"] = ""
                values["documentation_ref"] = ""
            elif last_inspection.get("documentation_ref"):
                values["documentation_ref"] = str(last_inspection["documentation_ref"])
                values["text"] = ""
                values["path"] = ""
                values["url"] = ""

        request = _Import(**values)
        return _json(
            lambda: manager.import_documentation(
                name=request.name,
                source_decision_id=source_decision_id,
                text=request.text,
                path=request.path,
                url=request.url,
                documentation_ref=request.documentation_ref,
                session_id=session_id,
                semantic_definition=request.semantic_definition,
                save=request.save,
                ephemeral=request.ephemeral,
                refresh_integration_id=request.refresh_integration_id,
            ),
            root=resolved_root,
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def import_semantic_docs(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        request = _SemanticImport(**values)
        return _json(
            lambda: manager.import_documentation(
                name=request.name,
                source_decision_id=source_decision_id,
                text=request.text,
                text_reference=request.documentation_reference,
                session_id=session_id,
                semantic_definition=request.semantic_definition,
                save=request.save,
                ephemeral=request.ephemeral,
                refresh_integration_id=request.refresh_integration_id,
            ),
            root=resolved_root,
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def preview(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        request = _Request(**values)
        if (
            request.routing_decision.source_decision_id
            and request.routing_decision.source_decision_id != source_decision_id
            and not bound_context.source_decision_id
        ):
            raise ValueError("Routing decision ID does not match the tool decision ID.")
        return _json(
            lambda: manager.preview_request(
                session_id=session_id,
                source_decision_id=source_decision_id,
                routing_decision=request.routing_decision,
                context=bound_context,
                **request.request_kwargs(),
            ),
            root=resolved_root,
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def execute(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        request = _Execute(**values)
        if (
            request.routing_decision.source_decision_id
            and request.routing_decision.source_decision_id != source_decision_id
            and not bound_context.source_decision_id
        ):
            raise ValueError("Routing decision ID does not match the tool decision ID.")
        return _json(
            lambda: manager.execute_request(
                approval_reference=request.approval_reference,
                session_id=session_id,
                routing_decision=request.routing_decision,
                context=bound_context,
                **request.request_kwargs(),
            ),
            root=resolved_root,
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def workflow_decide(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        values["session_id"] = session_id
        values["source_decision_id"] = source_decision_id
        decision_obj = _WorkflowDecision(**values)
        return encode(
            lambda: decision_obj.model_dump(mode="json"),
            session_id=session_id,
            source_decision_id=source_decision_id,
        )
        
    def workflow_terminal(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        values["session_id"] = session_id
        values["source_decision_id"] = source_decision_id

        terminal = _WorkflowTerminal(**values)

        return encode(
            lambda: terminal.model_dump(mode="json"),
            session_id=session_id,
            source_decision_id=source_decision_id,
        )
        
    def inspect_docs(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        values["session_id"] = session_id
        values["source_decision_id"] = source_decision_id
        req = _Inspect(**values)
        return encode(
            lambda: manager.inspect_documentation(
                text=req.text,
                path=req.path,
                url=req.url,
                session_id=session_id,
                offset=req.offset,
                limit=req.limit,
            ),
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def list_integrations(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        values["session_id"] = session_id
        values["source_decision_id"] = source_decision_id
        req = _List(**values)
        return encode(
            lambda: manager.list_integrations(include_disabled=req.include_disabled),
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def get_integration(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        values["session_id"] = session_id
        values["source_decision_id"] = source_decision_id
        req = _IntegrationId(**values)
        return encode(
            lambda: manager.get_integration(req.integration_id),
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def update_integration(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        values["session_id"] = session_id
        values["source_decision_id"] = source_decision_id
        req = _Update(**values)
        return encode(
            lambda: manager.update_integration(
                req.integration_id,
                name=req.name,
                description=req.description,
                enabled=req.enabled,
                authentication=req.authentication,
            ),
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def delete_integration(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        values["session_id"] = session_id
        values["source_decision_id"] = source_decision_id
        req = _Delete(**values)
        return encode(
            lambda: manager.registry.delete(req.integration_id, explicit=req.explicit),
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    def search_operations(**values: Any) -> str:
        session_id, source_decision_id = _resolve_identities(values)
        values["session_id"] = session_id
        values["source_decision_id"] = source_decision_id
        req = _Search(**values)
        return encode(
            lambda: [
                item.model_dump(mode="json")
                for item in manager.discovery.search(req.query, limit=req.limit)
            ],
            session_id=session_id,
            source_decision_id=source_decision_id,
        )

    return [
        StructuredTool.from_function(
            name="api_workflow_decide",
            description=(
                "Record the model's strict ordered API workflow decision. This must be the first "
                "API-route tool call; completion is validated against its required_actions."
            ),
            args_schema=_WorkflowDecision,
            func=workflow_decide,
        ),
        StructuredTool.from_function(
            name="api_workflow_terminal",
            description=(
                "Record an evidence-backed terminal API workflow outcome when complete "
                "documentation inspection proves that no usable API definition can be "
                "safely imported or executed. Use only after api_docs_inspect has fully "
                "consumed the inspected documentation and truncated=false. "
                "Supply the exact documentation_ref returned by that inspection. "
                "This tool never imports, previews, or executes an API request."
            ),
            args_schema=_WorkflowTerminal,
            func=workflow_terminal,
        ),
        StructuredTool.from_function(
            name="api_docs_inspect",
            description=(
                "Read one authorized API documentation URL, workspace file, or pasted text source "
                "through the API network/file policy. Returns source evidence and documentation_ref "
                "and never infers, imports, or executes an operation. "
                "If truncated=true or more_available=true, the inspection is incomplete. "
                "Continue reading the same source using next_offset as offset before concluding "
                "that an API specification, endpoint, operation, parameter, or authentication "
                "detail is absent."
            ),
            args_schema=_Inspect,
            func=inspect_docs,
        ),
        StructuredTool.from_function(
            name="api_docs_import",
            description=(
                "Import authorized OpenAPI/Swagger documentation deterministically. Unstructured "
                "documentation must use api_docs_import_semantic so its typed semantic definition "
                "cannot be omitted. Never executes documentation content."
            ),
            args_schema=_Import,
            func=import_docs,
            metadata={"transactional_adapter": "api_integration"},
        ),
        StructuredTool.from_function(
            name="api_docs_import_semantic",
            description=(
                "Validate and import unstructured documentation using a required, cited, strict "
                "SemanticDefinition extracted by the model only from the supplied text evidence. "
                "Pass the exact inspected source reference in documentation_reference and cite "
                "that reference from every operation. The semantic_definition argument is "
                "mandatory; no heuristic extraction runs. If the integration already exists, "
                "retry this import with its exact refresh_integration_id."
            ),
            args_schema=_SemanticImport,
            func=import_semantic_docs,
            metadata={"transactional_adapter": "api_integration"},
        ),
        StructuredTool.from_function(
            name="api_integrations_list",
            description="List saved API integration metadata. Does not expose credentials.",
            args_schema=_List,
            func=list_integrations,
        ),
        StructuredTool.from_function(
            name="api_integration_get",
            description="Inspect one saved normalized API integration without credential values.",
            args_schema=_IntegrationId,
            func=get_integration,
        ),
        StructuredTool.from_function(
            name="api_integration_update",
            description="Update only explicit mutable integration metadata fields.",
            args_schema=_Update,
            func=update_integration,
            metadata={"transactional_adapter": "api_integration"},
        ),
        StructuredTool.from_function(
            name="api_integration_delete",
            description="Delete one saved integration only with explicit delete intent.",
            args_schema=_Delete,
            func=delete_integration,
            metadata={"transactional_adapter": "api_integration"},
        ),
        StructuredTool.from_function(
            name="api_operations_search",
            description=(
                "Retrieve and rank operations from enabled integrations. This returns candidates only; "
                "a structured model decision must select an operation before preview or execution."
            ),
            args_schema=_Search,
            func=search_operations,
        ),
        StructuredTool.from_function(
            name="api_request_preview",
            description=(
                "Build and validate a saved API operation and return a redacted preview. Use before "
                "every execution. Supply an explicit env:// or mana-secret:// credential_reference "
                "when the selected operation identifies its authentication scheme but the saved "
                "integration has not bound a credential. Network-policy exceptions create the exact "
                "trusted-local approval during preview and return permission_required; stop there "
                "until the TUI or dashboard resolves it. Arbitrary base URL overrides are not accepted."
            ),
            args_schema=_Request,
            func=preview,
        ),
        StructuredTool.from_function(
            name="api_request_execute",
            description=(
                "Execute one validated saved API operation through DNS-pinned SSRF protections. "
                "An explicit credential_reference may resolve a structurally known authentication "
                "requirement for this request without storing secret material. "
                "Mutations and network-policy exceptions fail closed unless the trusted approval "
                "flow supplies an approval reference."
            ),
            args_schema=_Execute,
            func=execute,
        ),
    ]


API_MANAGER_TOOL_NAMES = (
    "api_workflow_decide",
    "api_workflow_terminal",
    "api_docs_inspect",
    "api_docs_import",
    "api_docs_import_semantic",
    "api_integrations_list",
    "api_integration_get",
    "api_integration_update",
    "api_integration_delete",
    "api_operations_search",
    "api_request_preview",
    "api_request_execute",
)


def api_manager_service(root: str | Path) -> ApiManagerService:
    resolved_root = str(Path(root).expanduser().resolve())
    return _service_for_root(resolved_root)


def _service_for_root(resolved_root: str) -> ApiManagerService:
    with _SERVICES_LOCK:
        existing = _SERVICES.get(resolved_root)
        if existing is not None:
            return existing
        created = ApiManagerService(resolved_root)
        _SERVICES[resolved_root] = created
        return created
